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
from typing import Any, AsyncIterator, List, Optional

import numpy as np

from core.gui_dispatcher import gui
from voice.audio_manager import audio_manager
from voice.audio_processing import peak_monitor
from voice.command_listener import command_listener, is_filler, is_garbage, UtteranceEvent
from voice.streaming_tts import streaming_tts
from voice.wake_listener import WakeListener, WakeEvent
from voice.wake_model_manager import wake_model_manager
from agent.conversation_memory import conv_memory
from agent.personality import personality
from core.benchmark import benchmark
from core.manual_session_recorder import session_recorder
from core.response_guarantee import response_guarantee

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

        # Providers (context for Brain's LLM pipeline)
        self._vision_context_fn = None
        self._search_provider_fn = None
        self._learning_context_fn = None

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

                # ── Record wake metrics ──
                session_recorder.new_turn()
                session_recorder.record_wake(
                    latency_ms=event.correlation * 1000 if event.correlation else 0,
                    confidence=event.score,
                    model=event.model,
                    transcript=event.transcript,
                )

                # Play chime
                await loop.run_in_executor(None, self._play_wake_chime)

                # STATE: FACE_AUTH (if needed)
                needs_auth = self._needs_auth()
                if needs_auth:
                    self._set_state(EngineState.FACE_AUTH)
                    t_auth_start = time.time()
                    name = await self._run_auth()
                    auth_latency = (time.time() - t_auth_start) * 1000
                    if name:
                        self._auth_user = name
                        self._last_auth_time = time.time()
                        conv_memory.set_user_name(name)
                        logger.info("[FACE_AUTH] Authenticated: %s", name)
                        session_recorder.record_face_auth(auth_latency, True, name)
                    else:
                        logger.warning("[FACE_AUTH] Failed — continuing unauthenticated")
                        session_recorder.record_face_auth(auth_latency, False)

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

                # ── Record utterance metrics ──
                session_recorder.record_utterance(
                    transcript=text,
                    confidence=ev.confidence,
                    speech_duration_ms=dur_ms,
                    endpoint_reason=ev.endpoint_reason,
                    whisper_latency_ms=ev.whisper_latency_ms,
                )

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

                # ── STATE: THINK — Brain orchestrates the full pipeline ──
                # Brain.process_command() runs:
                #   perceive → decide → plan → dispatch → verify → learn → respond
                # The engine ONLY speaks the response. No bypass is possible.
                self._set_state(EngineState.THINK)

                from agent.brain import agent_brain

                # ── RESPONSE GUARANTEE: never silent ──
                # Wrap the full turn (process → speak) so that every
                # completed utterance gets a spoken response. If the
                # Brain fails, returns an empty response, or TTS fails,
                # the guarantee layer speaks a recovery/generic fallback.
                result_holder: dict = {}

                async def _process() -> Any:
                    r = await agent_brain.process_command(text)
                    result_holder["result"] = r
                    return r

                async def _speak(response: str) -> bool:
                    spoke = await self._think_and_speak(response, events, canned=True)
                    # If the action spoke immediately, speak the followup
                    # confirmation after verification completes.
                    r = result_holder.get("result")
                    if spoke and r is not None and getattr(r, "speak_immediately", False):
                        followup = getattr(r, "followup_response", "") or ""
                        if followup and followup.strip():
                            spoke2 = await self._think_and_speak(followup, events, canned=True)
                            spoke = spoke or spoke2
                    return spoke

                await response_guarantee.run_turn(
                    transcript=text,
                    process_fn=_process,
                    speak_fn=_speak,
                )

                result = result_holder.get("result")

                # ── Record decision ──
                if result is not None:
                    session_recorder.record_decision(
                        classification=result.path or "BRAIN",
                        confidence=1.0,
                        latency_us=0.0,
                        llm_used=result.used_llm,
                        action=None,
                        actions=None,
                    )

                    benchmark.record_turn(
                        text=text, llm_used=result.used_llm,
                        router_kind=result.path,
                        latency_ms=result.latency_ms,
                        action_executed=result.actions_executed > 0,
                        action_success=result.actions_failed == 0,
                    )

                # ── Record turn end ──
                session_recorder.record_turn_end(
                    total_latency_ms=(time.time() - t_turn_start) * 1000,
                )

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
    ) -> bool:
        """
        SPEAK (TTS) for one turn.

        RESPONSIBILITY: The ConversationEngine ONLY speaks. The Brain
        generates all responses and executes all actions. This method
        receives a complete response string and speaks it via TTS.

        Returns:
            True if at least one audio chunk was queued for playback,
            False if nothing was spoken (TTS unavailable or failed).
        """
        self._tts_interrupt.clear()
        interrupt = self._tts_interrupt
        t_start = time.time()

        sentence_q: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

        async def produce() -> None:
            try:
                # Only one-line stream — the Brain already generated
                # the full response. No LLM, no ACTION parsing here.
                stream = self._one_line_stream(user_text)
                async for piece in stream:
                    if interrupt.is_set():
                        break
                    await sentence_q.put(piece)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[SPEAK] error: %s", e)
                await sentence_q.put(personality.error_response())
            finally:
                await sentence_q.put(None)

        producer = asyncio.create_task(produce())
        first = await sentence_q.get()
        if first is None:
            await asyncio.gather(producer, return_exceptions=True)
            return False

        # ── STATE: SPEAK ──
        self._set_state(EngineState.SPEAK)
        logger.info("TTS start")

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
            played = await streaming_tts.speak_sentences(sentences(), interrupt)
        finally:
            command_listener.resume_listening()
            monitor.cancel()
            if not producer.done():
                interrupt.set()
            await asyncio.gather(producer, monitor, return_exceptions=True)
        logger.info("TTS end (%.0fms)", (time.time() - t_start) * 1000)
        return played

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

    async def _one_line_stream(self, text: str) -> AsyncIterator[str]:
        """Split a response into natural sentences with pauses.

        CRITICAL FIX (Priority 2 — Speech Behaviour):
        - Split on sentence boundaries so TTS can synthesize/play each
          sentence with a natural pause between them.
        - Short responses ("Done.", "Opening Firefox.") are yielded whole
          so they play instantly with no artificial delay.
        - Longer responses get natural sentence-level pauses.
        """
        import re as _re
        text = (text or "").strip()
        if not text:
            return

        # Short responses — yield whole for instant playback
        if len(text) < 40:
            yield text
            return

        # Split into sentences on . ! ? followed by space/end
        sentences = _re.split(r'(?<=[.!?])\s+', text)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            yield sentence
            # Natural pause between sentences (not after the last one)
            if i < len(sentences) - 1:
                await asyncio.sleep(0.15)

    async def _speak_line(self, text: str) -> bool:
        """Speak a single line (used for error messages).

        Returns True if audio was queued for playback.
        """
        async def _gen():
            yield text
        return await streaming_tts.speak_sentences(_gen(), None)

    def get_diagnostics(self) -> dict:
        return {
            "state": self._state.value if self._state else "none",
            "state_duration_s": round(time.monotonic() - self._state_entered, 1),
            "turn_count": self._turn_count,
            "auth_user": self._auth_user,
            "running": self._running,
            "last_diag": self._diag,
            "response_guarantee": response_guarantee.get_diagnostics(),
        }


# Global singleton
conversation_engine = ConversationEngine()
