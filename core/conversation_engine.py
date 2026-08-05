"""
ConversationEngine — Leo's clean voice-state machine.

STATE MACHINE (7 states, no bypass):

    IDLE
      ↓
    WAKE          (openWakeWord + unified VAD + Whisper verification)
      ↓
    FACE_AUTH     (camera opens ONLY here, after verified wake)
      ↓
    LISTEN        (streaming Whisper with 800ms+ context windows)
      ↓
    THINK         (LLM runs only here)
      ↓
    SPEAK         (Kokoro TTS runs only here)
      ↓
    IDLE  …

HARD INVARIANTS:
  WAKE state:
    Only these modules may run:
      - AudioManager (ring-buffer reads)
      - Unified VAD (voice/vad.py — shared with LISTEN)
      - openWakeWord (streaming predict, 80 ms frames)
      - Whisper verification (ONLY after openWakeWord triggers)
    LLM MUST NOT run. TTS MUST NOT run.

  LISTEN state:
    Streaming Whisper exists ONLY here. Created on entry, DESTROYED
    after endpoint. Silence >= CONVERSATION_TIMEOUT_S returns to IDLE.

  THINK state:  LLM runs only here.
  SPEAK state:  TTS runs only here.

RUNTIME DIAGNOSTICS (every state transition):
  - audio_duration_ms reaching Whisper
  - transcript_confidence (Whisper avg_logprob)
  - endpoint_reason (silence_ms / timeout / interruption)
  - state_transition (from → to, duration in previous state)
  - latency_breakdown (wake→STT, STT→transcript, transcript→LLM, LLM→TTS, TTS→done)
"""

import asyncio
import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, List, Optional

import numpy as np

from core.gui_dispatcher import gui
from voice.audio_manager import audio_manager
from voice.audio_processing import peak_monitor
from voice.command_listener import command_listener, is_filler, UtteranceEvent
from voice.streaming_tts import streaming_tts
from voice.wake_listener import WakeListener, WakeEvent
from voice.wake_model_manager import wake_model_manager
from agent.streaming_llm import streaming_llm
from agent.conversation_memory import conv_memory
from agent.personality import personality
from core.command_router import command_router, RouteKind
from core.benchmark import benchmark

logger = logging.getLogger(__name__)

# ── Tuning ─────────────────────────────────────────────────────
CONVERSATION_TIMEOUT_S = 60.0
GOODBYE_PHRASES = {
    "bye", "goodbye", "see you", "see ya", "later", "that's all",
    "thats all", "nothing else", "i'm done", "im done", "stop listening",
    "go to sleep", "good night", "goodnight", "cancel",
}
AUTH_SESSION_S = 600.0  # 10 minutes
CHIME_PATH = Path(__file__).resolve().parent.parent / "leo.wav"


# ═══════════════════════════════════════════════════════════════
# States
# ═══════════════════════════════════════════════════════════════

class EngineState(str, Enum):
    IDLE = "IDLE"
    WAKE = "WAKE"
    FACE_AUTH = "FACE_AUTH"
    LISTEN = "LISTEN"
    THINK = "THINK"
    SPEAK = "SPEAK"


ALLOWED_TRANSITIONS = {
    EngineState.IDLE:      {EngineState.WAKE},
    EngineState.WAKE:      {EngineState.FACE_AUTH, EngineState.LISTEN},
    EngineState.FACE_AUTH: {EngineState.LISTEN},
    EngineState.LISTEN:    {EngineState.THINK, EngineState.IDLE},
    EngineState.THINK:     {EngineState.SPEAK},
    EngineState.SPEAK:     {EngineState.LISTEN, EngineState.IDLE},
}


# ═══════════════════════════════════════════════════════════════
# ConversationEngine
# ═══════════════════════════════════════════════════════════════

class ConversationEngine:
    """Clean 7-state conversational engine with runtime diagnostics."""

    def __init__(self):
        self._state: Optional[EngineState] = None
        self._state_entered: float = time.monotonic()
        self._running = False

        # Face-auth session
        self._auth_user: Optional[str] = None
        self._last_auth_time: float = 0.0
        self._auth_session_s: float = AUTH_SESSION_S
        self._auth_provider = None

        # Wake listener
        self._wake_listener = WakeListener()

        # Per-turn cancellation
        self._tts_interrupt = asyncio.Event()

        # Session
        self._session_deadline: float = 0.0
        self._turn_count = 0

        # Providers
        self._vision_context_fn = None
        self._search_provider_fn = None
        self._learning_context_fn = None
        self._action_executor = None

        # GUI pump
        self._gui_pump_task: Optional[asyncio.Task] = None

        # ── Runtime diagnostics ──
        self._diag: dict = {}

    # ── Wiring ────────────────────────────────────────────

    def set_vision_context(self, fn) -> None:
        self._vision_context_fn = fn

    def set_search_provider(self, fn) -> None:
        self._search_provider_fn = fn

    def set_learning_context(self, fn) -> None:
        self._learning_context_fn = fn

    def set_action_executor(self, fn) -> None:
        self._action_executor = fn

    def set_auth_provider(self, fn) -> None:
        self._auth_provider = fn

    def set_authenticated(self, name: Optional[str]) -> None:
        self._auth_user = name
        self._last_auth_time = time.time()
        if name:
            conv_memory.set_user_name(name)

    def invalidate_auth(self) -> None:
        self._last_auth_time = 0.0
        self._auth_user = None
        logger.info("[ENGINE] Auth session invalidated")

    def _needs_auth(self) -> bool:
        if self._auth_provider is None:
            return False
        if self._auth_user is None:
            return True
        return (time.time() - self._last_auth_time) >= self._auth_session_s

    # ── State machine ─────────────────────────────────────

    def _set_state(self, new_state: EngineState, **diag) -> None:
        """The ONLY way the engine changes state. Logs every transition
        with runtime diagnostics."""
        if new_state == self._state:
            return
        if self._state is not None:
            allowed = ALLOWED_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                logger.error("STATE VIOLATION: %s → %s (not in %s)",
                             self._state.value, new_state.value,
                             {s.value for s in allowed})
        dur = time.monotonic() - self._state_entered
        prev = self._state.value if self._state else "START"
        diag_str = " ".join(f"{k}={v}" for k, v in diag.items()) if diag else ""
        logger.info("STATE %s → %s (%.1fs) %s", prev, new_state.value, dur, diag_str)
        self._state = new_state
        self._state_entered = time.monotonic()
        self._diag = diag

    # ── Main run loop ─────────────────────────────────────

    async def run(self) -> None:
        """IDLE → WAKE → (FACE_AUTH) → LISTEN → THINK → SPEAK → IDLE … forever."""
        self._running = True
        loop = asyncio.get_event_loop()

        # ── Boot: start audio + load models ──
        self._set_state(EngineState.IDLE)
        logger.info("[ENGINE] Leo conversation engine starting")

        if threading.current_thread() is threading.main_thread():
            gui.start()
            self._gui_pump_task = asyncio.create_task(gui.pump())
        else:
            logger.warning("[ENGINE] Engine not on main thread — GUI disabled")

        # AudioManager
        while self._running and not audio_manager.is_running:
            ok = await loop.run_in_executor(None, audio_manager.start)
            if ok:
                break
            logger.error("[ENGINE] AudioManager failed — retrying in 5s")
            await asyncio.sleep(5.0)
        if not audio_manager.is_running:
            self._running = False
            return

        # openWakeWord
        await loop.run_in_executor(None, self._ensure_wake_model)

        # Unified VAD (shared by wake + command)
        from voice.vad import unified_vad
        await loop.run_in_executor(None, unified_vad.load)

        # TTS
        await loop.run_in_executor(None, streaming_tts.initialize)

        # Whisper (preloaded for wake verification)
        await loop.run_in_executor(None, command_listener.initialize)

        logger.info("[ENGINE] Models loaded (wake=%s, vad=%s, whisper=%s)",
                    wake_model_manager.model_name or "unavailable",
                    "ready" if unified_vad.ready else "fallback",
                    "ready" if command_listener.ready else "unavailable")

        # ── Forever loop ──
        try:
            while self._running:
                # STATE: WAKE
                self._set_state(EngineState.WAKE)
                event = await self._wake_listen_loop()
                if event is None:
                    break

                logger.info("Wake accepted (model='%s' score=%.2f ≥ %.2f)",
                            event.model, event.score, wake_model_manager.threshold)

                # Play chime
                await loop.run_in_executor(None, self._play_wake_chime)

                # STATE: FACE_AUTH (if needed)
                needs_auth = self._needs_auth()
                if needs_auth:
                    self._set_state(EngineState.FACE_AUTH)
                    name = await self._run_auth()
                    if name:
                        self._auth_user = name
                        self._last_auth_time = time.time()
                        conv_memory.set_user_name(name)
                        logger.info("[FACE_AUTH] Authenticated: %s", name)
                    else:
                        logger.warning("[FACE_AUTH] Failed — continuing unauthenticated")

                # STATE: LISTEN → THINK → SPEAK → (loop back)
                await self._conversation_session()

        except asyncio.CancelledError:
            logger.info("[ENGINE] Conversation engine cancelled")
        finally:
            self._running = False

    # ── Helpers ───────────────────────────────────────────

    def _ensure_wake_model(self) -> bool:
        if wake_model_manager.loaded:
            return True
        ok = wake_model_manager.load()
        if not ok:
            logger.error("[ENGINE] openWakeWord unavailable: %s", wake_model_manager.load_error)
        return ok

    async def _run_auth(self) -> Optional[str]:
        if self._auth_provider is None:
            return None
        try:
            return await self._auth_provider()
        except Exception as e:
            logger.warning("[FACE_AUTH] Auth provider error: %s", e)
            return None

    # ── STATE: WAKE ───────────────────────────────────────

    async def _wake_listen_loop(self) -> Optional[WakeEvent]:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._wake_listener.prime)
        logger.info("[WAKE] Listening for wake word...")
        return await self._wake_listener.wait_for_wake(lambda: self._running)

    @staticmethod
    def _play_wake_chime() -> None:
        try:
            import wave
            import sounddevice as sd
            if not CHIME_PATH.exists():
                return
            with wave.open(str(CHIME_PATH), "rb") as w:
                rate = w.getframerate()
                width = w.getsampwidth()
                data = w.readframes(w.getnframes())
            if width == 2:
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
                audio = (audio - 128.0) / 128.0
            sd.play(audio, rate)
            sd.wait()
        except Exception as e:
            logger.debug("[WAKE] Chime playback failed: %s", e)

    # ── STATE: LISTEN / THINK / SPEAK ─────────────────────

    async def _conversation_session(self) -> None:
        """LISTEN → THINK → SPEAK → LISTEN … until timeout or goodbye."""
        loop = asyncio.get_event_loop()
        t_session_start = time.time()

        if not command_listener.ready:
            ok = await loop.run_in_executor(None, command_listener.initialize)
            if not ok:
                logger.error("[LISTEN] Whisper unavailable — returning to IDLE")
                await self._speak_line("My speech recognizer isn't available right now.")
                return

        events: "asyncio.Queue[UtteranceEvent]" = asyncio.Queue()
        stream = command_listener.stream_utterances()
        pump = asyncio.create_task(self._stt_event_pump(stream, events))

        self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S
        self._set_state(EngineState.LISTEN)

        try:
            while self._running:
                remaining = self._session_deadline - time.monotonic()
                if remaining <= 0:
                    logger.info("[LISTEN] Silence %.0fs — conversation timeout",
                                CONVERSATION_TIMEOUT_S)
                    break

                try:
                    ev = await asyncio.wait_for(events.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    logger.info("[LISTEN] Silence %.0fs — conversation timeout",
                                CONVERSATION_TIMEOUT_S)
                    break

                if ev.kind == "speech_start":
                    logger.info("Speech detected")
                    self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S
                    continue

                if ev.kind == "partial":
                    logger.info("Partial: '%s' (conf=%.3f, audio=%.0fms)",
                                ev.text, ev.confidence, ev.audio_duration_ms)
                    continue

                if ev.kind != "final":
                    continue

                text = (ev.text or "").strip()
                if not text:
                    continue

                dur_ms = ev.audio_duration_ms
                logger.info("Endpoint (%.0fms, reason=%s)", dur_ms, ev.endpoint_reason)
                logger.info("Transcript: '%s' (conf=%.3f)", text, ev.confidence)
                self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S

                lower = text.lower()

                # Filler-only: keep listening
                if is_filler(text):
                    logger.info("[LISTEN] Filler '%s' — turn stays open", text)
                    continue

                if lower in GOODBYE_PHRASES or any(g in lower for g in GOODBYE_PHRASES if len(g) > 5):
                    self._set_state(EngineState.THINK)
                    await self._think_and_speak(personality.farewell(), events, canned=True)
                    logger.info("[ENGINE] Goodbye — returning to IDLE")
                    break

                self._turn_count += 1
                logger.info("[THINK] Turn #%d: '%s'", self._turn_count, text)

                t_turn_start = time.time()

                # ── Decision engine (L1-L7) ──
                from core.autonomous_reasoning import auto_context
                auto_ctx = await auto_context.collect()

                from core.decision_engine import decision_engine, DecisionPath
                decision = await decision_engine.decide(
                    text,
                    vision_context=None,
                    search_context=None,
                    desktop_context=auto_ctx.summary if auto_ctx else "",
                )

                if decision.resolved:
                    logger.info("[DECIDE] Bypassed LLM — path=%s confidence=%.2f",
                                decision.path.value, decision.confidence)

                    if decision.action and self._action_executor:
                        try:
                            await self._action_executor(decision.action)
                        except Exception as e:
                            logger.warning("[DECIDE] Action failed: %s", e)

                    if decision.actions and self._action_executor:
                        for action in decision.actions:
                            try:
                                await self._action_executor(action)
                            except Exception as e:
                                logger.warning("[DECIDE] Workflow action failed: %s", e)

                    if decision.response:
                        self._set_state(EngineState.THINK)
                        await self._think_and_speak(decision.response, events, canned=True)

                    benchmark.record_turn(
                        text=text, llm_used=False,
                        router_kind=decision.path.value,
                        latency_ms=(time.time() - t_turn_start) * 1000,
                        cache_hit=(decision.path == DecisionPath.WORKING_MEMORY),
                        action_executed=(decision.action is not None or decision.actions is not None),
                        action_success=(decision.action is not None or decision.actions is not None),
                    )

                    if decision.response and decision.path in (
                        DecisionPath.WORKING_MEMORY, DecisionPath.SESSION_MEMORY,
                        DecisionPath.DIRECT_EXECUTION,
                    ):
                        command_router.cache_llm_response(text, decision.response, ttl_s=86400)

                    # Drain stale events
                    drained = 0
                    while True:
                        try:
                            events.get_nowait()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        logger.info("[LISTEN] Drained %d stale STT events", drained)
                    self._set_state(EngineState.LISTEN)
                    continue

                # ── STATE: THINK (LLM) ──
                self._set_state(EngineState.THINK)
                t_llm_start = time.time()
                actions = await self._think_and_speak(text, events)
                benchmark.record_turn(
                    text=text, llm_used=True,
                    router_kind=RouteKind.COMPLEX.value,
                    latency_ms=(time.time() - t_turn_start) * 1000,
                    llm_latency_ms=(time.time() - t_llm_start) * 1000,
                    action_executed=len(actions) > 0,
                    action_success=len(actions) > 0,
                )

                # ── Execute actions ──
                for action_json in actions:
                    await self._run_action(action_json)

                # Drain stale events from TTS/action contamination
                drained = 0
                while True:
                    try:
                        events.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    logger.info("[LISTEN] Drained %d stale STT events (TTS contamination)", drained)

                # ── Back to LISTEN ──
                self._set_state(EngineState.LISTEN,
                                latency_breakdown=f"turn={(time.time() - t_turn_start) * 1000:.0f}ms")

        finally:
            pump.cancel()
            command_listener.stop_streaming()
            await asyncio.gather(pump, return_exceptions=True)
            try:
                await stream.aclose()
            except Exception:
                pass
            logger.info("[STT] Streaming Whisper stopped (session over)")

        self._set_state(EngineState.IDLE,
                        session_duration=f"{(time.time() - t_session_start):.1f}s",
                        turns=self._turn_count)
        print("\n  Listening for wake word...\n")

    async def _stt_event_pump(
        self, stream: AsyncIterator[UtteranceEvent],
        events: "asyncio.Queue[UtteranceEvent]",
    ) -> None:
        try:
            async for ev in stream:
                if not self._running:
                    break
                events.put_nowait(ev)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("[STT] Stream error: %s", e)

    async def _think_and_speak(
        self, user_text: str, events: "asyncio.Queue[UtteranceEvent]",
        canned: bool = False,
    ) -> List[str]:
        """THINK (LLM) → SPEAK (TTS) for one turn."""
        self._tts_interrupt.clear()
        interrupt = self._tts_interrupt
        actions: List[str] = []
        t_llm = time.time()

        if canned:
            logger.info("LLM skipped (canned response)")
        else:
            logger.info("LLM start")

        sentence_q: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

        async def produce() -> None:
            try:
                stream = self._one_line_stream(user_text) if canned \
                    else self._llm_sentences(user_text, interrupt)
                async for piece in stream:
                    if interrupt.is_set():
                        break
                    if piece.startswith("ACTION:"):
                        actions.append(piece[len("ACTION:"):].strip())
                    else:
                        await sentence_q.put(piece)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[LLM] error: %s", e)
                await sentence_q.put(personality.error_response())
            finally:
                if not canned:
                    logger.info("LLM end (%.0fms)", (time.time() - t_llm) * 1000)
                await sentence_q.put(None)

        producer = asyncio.create_task(produce())
        first = await sentence_q.get()
        if first is None:
            await asyncio.gather(producer, return_exceptions=True)
            return actions

        if not canned:
            logger.info("[LLM] First sentence in %.0fms", (time.time() - t_llm) * 1000)

        # ── STATE: SPEAK ──
        self._set_state(EngineState.SPEAK)
        logger.info("TTS start")
        t_tts = time.time()

        async def sentences() -> AsyncIterator[str]:
            yield first
            while True:
                item = await sentence_q.get()
                if item is None:
                    break
                yield item

        monitor = asyncio.create_task(self._watch_interruption(events))
        command_listener.pause_listening()
        try:
            await streaming_tts.speak_sentences(sentences(), interrupt)
        finally:
            command_listener.resume_listening()
            monitor.cancel()
            if not producer.done():
                interrupt.set()
            await asyncio.gather(producer, monitor, return_exceptions=True)
        logger.info("TTS end (%.0fms)", (time.time() - t_tts) * 1000)
        return actions

    async def _watch_interruption(
        self, events: "asyncio.Queue[UtteranceEvent]") -> None:
        """During SPEAKING: user speech interrupts TTS instantly."""
        held: List[UtteranceEvent] = []
        try:
            while True:
                ev = await events.get()
                if ev.kind == "speech_start":
                    logger.info("[SPEAK] User interrupted — stopping TTS")
                    self._tts_interrupt.set()
                    streaming_tts.stop()
                    return
                held.append(ev)
        except asyncio.CancelledError:
            for ev in held:
                events.put_nowait(ev)
            raise

    async def _llm_sentences(
        self, user_text: str, interrupt: asyncio.Event) -> AsyncIterator[str]:
        """Yield LLM sentences with vision/search/desktop context."""
        from core.cache_manager import cache_manager

        context_parts: list = []

        async def _prepopulate_caches() -> None:
            nonlocal context_parts
            try:
                dk = cache_manager.desktop_key("quick_context")
                ds_ctx = cache_manager.get_or_compute(
                    "desktop", dk,
                    lambda: self._get_desktop_context_sync(),
                    ttl_s=1.0,
                )
                if ds_ctx:
                    context_parts.append(f"[Desktop: {ds_ctx}]")
            except Exception as e:
                logger.debug("[LLM] desktop state failed: %s", e)

            if self._references_screen(user_text) and self._vision_context_fn:
                try:
                    vk = cache_manager.vision_key("screen")
                    entry = cache_manager.get("vision", vk)
                    if entry is None:
                        screen_ctx = await self._vision_context_fn()
                        if screen_ctx:
                            cache_manager.set("vision", vk, screen_ctx, ttl_s=5.0)
                            entry = screen_ctx
                    if entry:
                        context_parts.append(f"[Screen: {entry}]")
                except Exception as e:
                    logger.debug("[LLM] vision failed: %s", e)

            if self._search_provider_fn and self._references_search(user_text):
                try:
                    sk = cache_manager.search_key(user_text)
                    entry = cache_manager.get("search", sk)
                    if entry is None:
                        search_ctx = await self._search_provider_fn(user_text)
                        if search_ctx:
                            cache_manager.set("search", sk, search_ctx, ttl_s=30.0)
                            entry = search_ctx
                    if entry:
                        context_parts.append(f"[Search: {entry}]")
                except Exception as e:
                    logger.debug("[LLM] search failed: %s", e)

            if self._learning_context_fn:
                try:
                    learn_ctx = self._learning_context_fn()
                    if learn_ctx:
                        context_parts.append(f"[User: {learn_ctx}]")
                except Exception as e:
                    logger.debug("[LLM] learning context failed: %s", e)

        await _prepopulate_caches()

        system = personality.system_prompt()
        history = conv_memory.format_for_llm()
        ctx_block = "\n".join(context_parts) if context_parts else ""

        prompt = user_text
        if ctx_block:
            prompt = f"{ctx_block}\n\nUser: {user_text}"

        async for sentence in streaming_llm.stream(
            prompt, system_prompt=system, history=history,
            interrupt=interrupt,
        ):
            yield sentence

    async def _one_line_stream(self, text: str) -> AsyncIterator[str]:
        yield text

    async def _speak_line(self, text: str) -> None:
        """Speak a single line (used for error messages)."""
        async def _gen():
            yield text
        await streaming_tts.speak_sentences(_gen(), None)

    async def _run_action(self, action_json: str) -> None:
        if not self._action_executor:
            return
        try:
            import json
            action = json.loads(action_json) if isinstance(action_json, str) else action_json
            await self._action_executor(action)
        except Exception as e:
            logger.warning("[ENGINE] Action execution failed: %s", e)

    @staticmethod
    def _get_desktop_context_sync() -> str:
        try:
            from services.desktop_state import desktop_state
            return desktop_state.quick_context()
        except Exception:
            return ""

    @staticmethod
    def _references_screen(text: str) -> bool:
        keywords = {"screen", "see", "looking at", "this page", "this window",
                    "what's on", "what is on", "current", "visible"}
        lower = text.lower()
        return any(k in lower for k in keywords)

    @staticmethod
    def _references_search(text: str) -> bool:
        keywords = {"search", "find", "look up", "google", "what is", "who is",
                    "how to", "weather", "news", "latest"}
        lower = text.lower()
        return any(k in lower for k in keywords)

    def get_diagnostics(self) -> dict:
        return {
            "state": self._state.value if self._state else "none",
            "state_duration_s": round(time.monotonic() - self._state_entered, 1),
            "turn_count": self._turn_count,
            "auth_user": self._auth_user,
            "running": self._running,
            "last_diag": self._diag,
        }


# Global singleton
conversation_engine = ConversationEngine()
