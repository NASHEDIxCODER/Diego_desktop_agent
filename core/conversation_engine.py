"""
ConversationEngine — Leo's STRICT voice-state machine.

This replaces the old "always-streaming Whisper" orchestrator. The previous
design started streaming STT at boot and detected the wake word from Whisper
transcripts, so Whisper ran forever and openWakeWord never gated anything.
That architecture is gone.

STATE MACHINE (no state may bypass another):

    BOOT
      ↓
    IDLE
      ↓
    WAKE_LISTEN
      ↓
    WAKE_DETECTED
      ↓
    FACE_AUTH        (mandatory — the camera opens ONLY here,
                      after a verified wake, NEVER at boot)
      ↓
    GREETING
      ↓
    COMMAND_LISTEN
      ↓
    THINKING
      ↓
    SPEAKING
      ↓
    FOLLOWUP
      ↓
    WAKE_LISTEN  …

HARD INVARIANTS:

  WAKE_LISTEN
      Only these modules may run:
        - AudioManager (ring-buffer reads)
        - openWakeWord (streaming predict, 80 ms frames)
        - optional Silero VAD (speech gate)
        - noise suppression (audio_preprocessor)
      Whisper MUST NOT run.
      LLM     MUST NOT run.
      TTS     MUST NOT run.

  WAKE_DETECTED
      openWakeWord score >= threshold → play wake chime, interrupt
      current buffers, THEN Whisper is allowed to start.

  COMMAND_LISTEN
      Streaming Whisper exists ONLY here. It is created on entry and
      DESTROYED (stream cancelled) immediately after endpoint detection
      ends the session. Silence >= CONVERSATION_TIMEOUT_S (60 s) returns
      the machine to WAKE_LISTEN.

  THINKING   — LLM runs only here.
  SPEAKING   — TTS runs only here (and GREETING / FACE_AUTH denial lines).
  FOLLOWUP   — decides: another COMMAND_LISTEN turn, or back to WAKE_LISTEN.

LOGGING CONTRACT (visible in the runtime logs):

    STATE WAKE_LISTEN
    Wake score 0.12
    Wake score 0.21
    Wake score 0.93
    Wake accepted
    STATE COMMAND_LISTEN
    Speech detected
    Endpoint
    Transcript
    LLM start
    LLM end
    TTS start
    TTS end
    STATE WAKE_LISTEN

The idle runtime behaves like Siri / Gemini Live / ChatGPT Voice: it idles
forever on the wake-word detector alone, consuming almost no CPU.
"""

import asyncio
import json
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
from voice.streaming_stt import streaming_stt, is_filler, UtteranceEvent
from voice.streaming_tts import streaming_tts
from voice.wake_listener import WakeListener, WakeEvent
from voice.wake_model_manager import wake_model_manager
from agent.streaming_llm import streaming_llm
from agent.conversation_memory import conv_memory
from agent.personality import personality
from core.command_router import command_router, RouteKind
from core.benchmark import benchmark

logger = logging.getLogger(__name__)

# ── Conversation lifecycle tuning ────────────────────────────────
CONVERSATION_TIMEOUT_S = 60.0   # silence this long → back to WAKE_LISTEN
GOODBYE_PHRASES = {
    "bye", "goodbye", "see you", "see ya", "later", "that's all",
    "thats all", "nothing else", "i'm done", "im done", "stop listening",
    "go to sleep", "good night", "goodnight", "cancel",
}
WAKE_VARIANTS = ("leo", "hey leo", "hello leo", "hi leo", "okay leo", "ok leo")

# Re-auth suppression: after a successful face auth, don't ask again for
# this many seconds (default 10 minutes) unless explicitly invalidated.
AUTH_SESSION_S = 600.0

CHIME_PATH = Path(__file__).resolve().parent.parent / "leo.wav"


# ═══════════════════════════════════════════════════════════════
# States + legal transitions
# ═══════════════════════════════════════════════════════════════

class EngineState(str, Enum):
    BOOT = "BOOT"
    FACE_AUTH = "FACE_AUTH"
    IDLE = "IDLE"
    WAKE_LISTEN = "WAKE_LISTEN"
    WAKE_DETECTED = "WAKE_DETECTED"
    GREETING = "GREETING"
    COMMAND_LISTEN = "COMMAND_LISTEN"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    FOLLOWUP = "FOLLOWUP"


# The complete transition table. Any transition not listed here is a
# state-machine violation and is logged as ILLEGAL (but never crashes Leo).
ALLOWED_TRANSITIONS = {
    EngineState.BOOT:           {EngineState.IDLE},
    EngineState.FACE_AUTH:      {EngineState.GREETING, EngineState.WAKE_LISTEN},
    EngineState.IDLE:           {EngineState.WAKE_LISTEN},
    EngineState.WAKE_LISTEN:    {EngineState.WAKE_DETECTED},
    EngineState.WAKE_DETECTED:  {EngineState.FACE_AUTH, EngineState.GREETING},
    EngineState.GREETING:       {EngineState.COMMAND_LISTEN, EngineState.WAKE_LISTEN},

    EngineState.COMMAND_LISTEN: {EngineState.THINKING, EngineState.WAKE_LISTEN},
    EngineState.THINKING:       {EngineState.SPEAKING},
    EngineState.SPEAKING:       {EngineState.FOLLOWUP},
    EngineState.FOLLOWUP:       {EngineState.COMMAND_LISTEN, EngineState.WAKE_LISTEN},
}


# ═══════════════════════════════════════════════════════════════
# ConversationEngine
# ═══════════════════════════════════════════════════════════════

class ConversationEngine:
    """
    Strict state-machine conversational engine.

    Owns the BOOT → … → WAKE_LISTEN cycle and coordinates
    openWakeWord → streaming Whisper → streaming LLM → streaming TTS,
    with hard module-activation rules per state.
    """

    def __init__(self):
        self._state: Optional[EngineState] = None
        self._state_entered: float = time.monotonic()
        self._running = False

        # Face-auth session state
        self._auth_user: Optional[str] = None
        self._last_auth_time: float = 0.0
        self._auth_session_s: float = AUTH_SESSION_S
        self._auth_provider = None       # async callable() -> Optional[str]

        # WAKE_LISTEN module — THE single wake pipeline (VAD → openWakeWord
        # → Whisper verification). See voice/wake_listener.py.
        self._wake_listener = WakeListener()

        # Per-turn cancellation (user interrupts Leo while speaking)
        self._tts_interrupt = asyncio.Event()

        # Conversation activity
        self._session_deadline: float = 0.0
        self._turn_count = 0

        # Vision context provider (set by main/leo)
        self._vision_context_fn = None

        # Search provider (set by main/leo) — callable(query) -> str
        self._search_provider_fn = None

        # Learning context provider (set by main/leo) — callable() -> str
        self._learning_context_fn = None

        # Action executor (set by main/leo) — maps ACTION dicts to desktop ops
        self._action_executor = None

        # GUI pump task (created in BOOT when running on the main thread)
        self._gui_pump_task: Optional[asyncio.Task] = None


    # ── Wiring ────────────────────────────────────────────

    def set_vision_context(self, fn) -> None:
        """Provide an async callable() -> str that returns current screen context."""
        self._vision_context_fn = fn

    def set_search_provider(self, fn) -> None:
        """Provide an async callable(query: str) -> str that returns search results."""
        self._search_provider_fn = fn

    def set_learning_context(self, fn) -> None:
        """Provide a callable() -> str that returns learned user profile context."""
        self._learning_context_fn = fn

    def set_action_executor(self, fn) -> None:
        """Provide an async callable(action_dict) -> str that runs desktop actions."""
        self._action_executor = fn

    def set_auth_provider(self, fn) -> None:
        """Provide an async callable() -> Optional[str] that runs face auth
        and returns the verified name (or None)."""
        self._auth_provider = fn

    def set_authenticated(self, name: Optional[str]) -> None:
        """Mark an externally-verified session (e.g. --no-auth dev mode)."""
        self._auth_user = name
        self._last_auth_time = time.time()
        if name:
            conv_memory.set_user_name(name)

    def invalidate_auth(self) -> None:
        """Force re-authentication on the next wake (manual logout / security)."""
        self._last_auth_time = 0.0
        self._auth_user = None
        logger.info("[ENGINE] Auth session invalidated — will re-authenticate on next wake")

    def _needs_auth(self) -> bool:
        """True if we must (re-)authenticate before entering conversation."""
        if self._auth_provider is None:
            return False  # no auth configured (--no-auth dev mode)
        if self._auth_user is None:
            return True
        return (time.time() - self._last_auth_time) >= self._auth_session_s

    # ── State machine core ────────────────────────────────

    def _set_state(self, new_state: EngineState) -> None:
        """The ONLY way the engine changes state. Logs every transition."""
        if new_state == self._state:
            return
        if self._state is not None:
            allowed = ALLOWED_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                logger.error(
                    "STATE VIOLATION: %s → %s is not a legal transition",
                    self._state.value, new_state.value)
        dur = time.monotonic() - self._state_entered
        if self._state is None:
            logger.info("STATE %s", new_state.value)
        else:
            logger.info("STATE %s → %s (%.1fs)",
                        self._state.value, new_state.value, dur)
        self._state = new_state
        self._state_entered = time.monotonic()

    # ── Main run loop ─────────────────────────────────────

    async def run(self) -> None:
        """
        BOOT → IDLE → (WAKE_LISTEN → … conversation …) forever.

        Face authentication NEVER runs at boot: the camera opens ONLY after
        a verified wake (WAKE_DETECTED → FACE_AUTH).
        """
        self._running = True
        loop = asyncio.get_event_loop()

        # ── STATE: BOOT ───────────────────────────────────
        self._set_state(EngineState.BOOT)
        logger.info("[BOOT] Leo conversation engine starting")

        # GUI dispatcher: the hidden Tk root is created ON THE MAIN THREAD
        # (this coroutine runs on it) and every later GUI operation is
        # marshalled here. The pump task keeps Tk events flowing from the
        # main event loop — mainloop() is NEVER called anywhere.
        if threading.current_thread() is threading.main_thread():
            gui.start()
            self._gui_pump_task = asyncio.create_task(gui.pump())
        else:
            logger.warning("[BOOT] Engine not on main thread — GUI disabled")

        # Audio capture (shared ring buffer). AudioManager owns the mic;
        # it is the ONLY audio source for every state. Leo NEVER exits:
        # if the microphone cannot be opened, BOOT retries forever.
        while self._running and not audio_manager.is_running:
            ok = await loop.run_in_executor(None, audio_manager.start)
            if ok:
                break
            logger.error("[BOOT] AudioManager failed to start — retrying "
                         "in 5s (Leo never exits)")
            await asyncio.sleep(5.0)
        if not audio_manager.is_running:
            self._running = False
            return

        # openWakeWord — the ONLY model that stays active forever.
        await loop.run_in_executor(None, self._ensure_wake_model)

        # Silero VAD speech gate for WAKE_LISTEN (owned by the WakeListener).
        await loop.run_in_executor(None, self._wake_listener.load)

        # TTS engine selection (no synthesis happens here — playback worker
        # idles until SPEAKING/GREETING actually enqueue audio).
        await loop.run_in_executor(None, streaming_tts.initialize)

        # Whisper is PRELOADED here: the wake-verification path
        # (WakeListener.verify_with_whisper) needs it hot — a lazy 3 s load
        # on the first trigger loses the wake-phrase window (proven live).
        # The LLM is still NOT contacted until THINKING.
        await loop.run_in_executor(None, streaming_stt.initialize)

        logger.info("[BOOT] Models loaded (wake=%s, vad_gate=%s, whisper=%s) — "
                    "LLM deferred to THINKING",
                    wake_model_manager.model_name or "unavailable",
                    "on" if self._wake_listener.vad.ready else "off",
                    "ready" if streaming_stt.ready else "unavailable")


        # ── STATE: IDLE ───────────────────────────────────
        # No boot authentication: BOOT → IDLE → WAKE_LISTEN. The camera
        # stays closed until a verified wake word requests it.
        self._set_state(EngineState.IDLE)
        logger.info("[IDLE] Only the wake-word detector is active — "
                    "Whisper inactive, LLM inactive, TTS inactive")

        # ── Forever: WAKE_LISTEN → conversation → WAKE_LISTEN ──
        try:
            while self._running:
                if self._state is not EngineState.WAKE_LISTEN:
                    self._set_state(EngineState.WAKE_LISTEN)
                event = await self._wake_listen_loop()
                if event is None:
                    break  # engine stopped
                self._set_state(EngineState.WAKE_DETECTED)
                logger.info("Wake accepted (model='%s' score=%.2f ≥ %.2f)",
                            event.model, event.score,
                            wake_model_manager.threshold)
                await self._handle_wake_detected()
        except asyncio.CancelledError:
            logger.info("[ENGINE] Conversation engine cancelled")
        finally:
            self._running = False

    # ── BOOT helpers ──────────────────────────────────────

    def _ensure_wake_model(self) -> bool:
        """Load openWakeWord if it isn't already (idempotent)."""
        if wake_model_manager.loaded:
            return True
        ok = wake_model_manager.load()
        if not ok:
            logger.error("[BOOT] openWakeWord unavailable: %s — "
                         "wake detection disabled", wake_model_manager.load_error)
        return ok

    async def _run_auth(self) -> Optional[str]:
        """Call the auth provider safely. NEVER raises."""
        if self._auth_provider is None:
            return None
        try:
            return await self._auth_provider()
        except Exception as e:
            logger.warning("[FACE_AUTH] Auth provider error: %s", e)
            return None

    # ── STATE: WAKE_LISTEN ────────────────────────────────

    async def _wake_listen_loop(self) -> Optional[WakeEvent]:
        """
        WAKE_LISTEN: idle forever on THE single wake pipeline
        (voice/wake_listener.py — AudioManager → Silero VAD → openWakeWord
        → Whisper verification).

        Whisper / LLM / TTS are NOT running in this state. Returns a
        WakeEvent on a VERIFIED wake, or None if the engine is shutting
        down. The detector never exits on its own.
        """
        # Thread inventory BEFORE the detector starts (any surviving GUI
        # thread is called out with its id).
        self._log_active_threads("WAKE_LISTEN")

        # Deterministic clean entry: hard model reset + ring-buffer drain
        # + refractory window — stale audio can never re-trigger a wake.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._wake_listener.prime)
        logger.info("[WAKE_LISTEN] Listening for wake word...")

        return await self._wake_listener.wait_for_wake(lambda: self._running)

    @staticmethod
    def _log_active_threads(context: str) -> None:
        """Print EVERY active thread before entering `context`.

        Any surviving GUI thread (popup/Tk) is called out with its thread id
        — after correct teardown there must be NONE.
        """
        threads = [t for t in threading.enumerate() if t.is_alive()]
        logger.info("[THREADS] %d active before %s:", len(threads), context)
        for t in threads:
            logger.info("[THREADS]   name=%r id=%s daemon=%s",
                        t.name, t.ident, t.daemon)
        gui_threads = [
            t for t in threads
            if t is not threading.main_thread()
            and any(k in t.name.lower() for k in ("popup", "tk", "gui"))
        ]
        for t in gui_threads:
            logger.warning(
                "[THREADS] GUI thread STILL ALIVE before %s: name=%r id=%s "
                "daemon=%s — teardown incomplete!",
                context, t.name, t.ident, t.daemon)
        if not gui_threads:
            logger.info("[THREADS] No GUI threads alive — teardown clean")


    # ── STATE: WAKE_DETECTED ──────────────────────────────

    async def _handle_wake_detected(self) -> None:
        """
        WAKE_DETECTED: play the wake chime, interrupt current buffers,
        then (re-)auth if the session expired → GREETING → conversation.

        ISSUE-1 FIX: STT streaming starts IMMEDIATELY after wake chime.
        Face authentication and greeting playback run ASYNCHRONOUSLY in
        the background. The user can speak at any time — if speech is
        detected during the greeting, TTS is interrupted instantly and
        the utterance is processed. Never delay COMMAND_LISTEN because
        of greeting playback.
        """
        loop = asyncio.get_event_loop()
        t_wake = time.time()

        # Interrupt current buffers: drop everything captured up to now so
        # neither the wake phrase nor the chime leaks into Whisper.
        audio_manager.get_recent_audio(0.05)  # touch to keep stream hot
        wake_model_manager.reset_stream()     # clear prediction streaks

        # Play wake chime.
        await loop.run_in_executor(None, self._play_wake_chime)

        # ── ISSUE-1: Launch face auth in background, do NOT block STT ──
        needs_auth = self._needs_auth()
        face_auth_task: Optional[asyncio.Task] = None
        face_auth_ok = asyncio.Event()
        face_auth_name: Optional[str] = None

        if needs_auth:
            self._set_state(EngineState.FACE_AUTH)
            logger.info("[FACE_AUTH] Starting background verification — "
                        "STT streaming begins immediately in parallel")

            async def _background_auth():
                nonlocal face_auth_name
                try:
                    face_auth_name = await self._run_auth()
                except Exception as e:
                    logger.warning("[FACE_AUTH] Background auth error: %s", e)
                finally:
                    face_auth_ok.set()
                    if face_auth_name:
                        logger.info("[FACE_AUTH] Background verification SUCCEEDED: %s",
                                    face_auth_name)
                    else:
                        logger.warning("[FACE_AUTH] Background verification FAILED")

            face_auth_task = asyncio.create_task(_background_auth())

        # ── ISSUE-1: Start STT streaming IMMEDIATELY (before greeting) ──
        # The user can speak at any time. The greeting plays asynchronously
        # and is interrupted if the user begins speaking.
        t_stt_start = time.time()
        logger.info("[PERF] wake→STT latency: %.0fms", (t_stt_start - t_wake) * 1000)

        # ── STATE: GREETING (plays asynchronously, does NOT block STT) ──
        self._set_state(EngineState.GREETING)
        name = conv_memory.user_name or self._auth_user
        if needs_auth:
            greeting = personality.greeting(returning=conv_memory.turn_count > 0)
        elif name and conv_memory.turn_count == 0:
            greeting = personality.greeting(returning=False).rstrip(".") + f", {name}."
        else:
            greeting = personality.greeting(returning=conv_memory.turn_count > 0)
        logger.info("[GREETING] '%s'", greeting)

        # Play greeting asynchronously — do NOT await it. The conversation
        # session starts immediately and the greeting is interrupted if the
        # user speaks.
        greeting_task = asyncio.create_task(self._speak_line(greeting))

        # ── STATE: COMMAND_LISTEN … (full session) ───────
        # Pass the pending face auth task so action execution can gate on it.
        t_cmd_start = time.time()
        logger.info("[PERF] wake→COMMAND_LISTEN latency: %.0fms",
                    (t_cmd_start - t_wake) * 1000)
        await self._conversation_session(
            face_auth_task=face_auth_task,
            face_auth_ok=face_auth_ok,
            greeting_task=greeting_task,
        )

        # ── Resolve face auth result after the session ──
        if face_auth_task is not None:
            if not face_auth_task.done():
                await face_auth_ok.wait()
            if face_auth_name:
                self._auth_user = face_auth_name
                self._last_auth_time = time.time()
                conv_memory.set_user_name(face_auth_name)
                logger.info("[FACE_AUTH] Session authenticated: %s", face_auth_name)
            else:
                logger.warning("[FACE_AUTH] Session unauthenticated — "
                               "actions were blocked this turn")

        # Ensure greeting task is cleaned up
        if not greeting_task.done():
            greeting_task.cancel()
            try:
                await greeting_task
            except (asyncio.CancelledError, Exception):
                pass

    @staticmethod
    def _play_wake_chime() -> None:
        """Play leo.wav synchronously (runs in an executor thread)."""

        try:
            import wave
            import sounddevice as sd
            if not CHIME_PATH.exists():
                logger.debug("[WAKE] Chime file missing: %s", CHIME_PATH)
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
            logger.info("[WAKE] Chime played")
        except Exception as e:
            logger.debug("[WAKE] Chime playback failed: %s", e)

    # ── STATE: COMMAND_LISTEN / THINKING / SPEAKING / FOLLOWUP ──

    async def _conversation_session(
        self,
        face_auth_task: Optional[asyncio.Task] = None,
        face_auth_ok: Optional[asyncio.Event] = None,
        greeting_task: Optional[asyncio.Task] = None,
    ) -> None:
        """
        Run one conversation session:

            COMMAND_LISTEN → THINKING → SPEAKING → FOLLOWUP → COMMAND_LISTEN …

        until 60 s of silence (→ WAKE_LISTEN) or a goodbye (→ WAKE_LISTEN).

        Streaming Whisper is created here and DESTROYED when the session
        ends — it never exists outside COMMAND_LISTEN-family states.

        ISSUE-1: face_auth_task runs in parallel with STT. Desktop actions
        are gated on face_auth_ok — if auth fails, actions are cancelled.
        greeting_task is the async greeting TTS — it is interrupted the
        moment the user begins speaking.
        """
        loop = asyncio.get_event_loop()

        # Whisper activates NOW — the first time a session actually starts.
        if not streaming_stt.ready:
            logger.info("[COMMAND_LISTEN] Loading Whisper (first activation)...")
            ok = await loop.run_in_executor(None, streaming_stt.initialize)
            if not ok:
                logger.error("[COMMAND_LISTEN] Whisper unavailable — "
                             "returning to WAKE_LISTEN")
                await self._speak_line("My speech recognizer isn't available right now.")
                self._set_state(EngineState.WAKE_LISTEN)
                return

        events: "asyncio.Queue[UtteranceEvent]" = asyncio.Queue()
        stream = streaming_stt.stream_utterances()
        pump = asyncio.create_task(self._stt_event_pump(stream, events))

        self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S
        self._set_state(EngineState.COMMAND_LISTEN)
        exit_state = EngineState.WAKE_LISTEN

        # ISSUE-1: Track whether the greeting has been interrupted
        _greeting_interrupted = False

        try:
            while self._running:
                remaining = self._session_deadline - time.monotonic()
                if remaining <= 0:
                    logger.info("[COMMAND_LISTEN] Silence %.0fs — conversation timeout",
                                CONVERSATION_TIMEOUT_S)
                    break
                try:
                    ev = await asyncio.wait_for(events.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    logger.info("[COMMAND_LISTEN] Silence %.0fs — conversation timeout",
                                CONVERSATION_TIMEOUT_S)
                    break

                if ev.kind == "speech_start":
                    logger.info("Speech detected")
                    # ISSUE-1: Interrupt greeting TTS immediately on first speech
                    if not _greeting_interrupted and greeting_task is not None and not greeting_task.done():
                        _greeting_interrupted = True
                        logger.info("[GREETING] User spoke — interrupting greeting TTS")
                        streaming_tts.stop()
                        greeting_task.cancel()
                        try:
                            await greeting_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S
                    continue

                if ev.kind == "partial":
                    # ISSUE-1: Also interrupt greeting on first partial transcript
                    if not _greeting_interrupted and greeting_task is not None and not greeting_task.done():
                        _greeting_interrupted = True
                        logger.info("[GREETING] Partial transcript received — interrupting greeting TTS")
                        streaming_tts.stop()
                        greeting_task.cancel()
                        try:
                            await greeting_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    # TASK 8: partial transcripts are VISIBLE while the user
                    # speaks (streaming Whisper, not batch-after-recording).
                    logger.info("Partial: '%s'", ev.text)
                    continue


                if ev.kind != "final":
                    continue

                text = (ev.text or "").strip()
                if not text:
                    continue

                dur_ms = max(0.0, (ev.ended_at - ev.started_at) * 1000.0)
                logger.info("Endpoint (%.0fms)", dur_ms)
                logger.info("Transcript: '%s'", text)
                self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S

                lower = text.lower()

                # User re-said the wake word mid-conversation → strip it.
                if self._contains_wake(lower):
                    stripped = self._strip_wake(lower)
                    if not stripped:
                        continue  # "leo" alone — already awake
                    text, lower = stripped, stripped

                # Filler-only: keep listening, don't respond.
                if is_filler(text):
                    logger.info("[COMMAND_LISTEN] Filler '%s' — turn stays open", text)
                    continue

                if self._is_goodbye(lower):
                    # Canned farewell: THINKING → SPEAKING → FOLLOWUP → WAKE_LISTEN
                    self._set_state(EngineState.THINKING)
                    await self._think_and_speak(personality.farewell(), events, canned=True)
                    self._set_state(EngineState.FOLLOWUP)
                    exit_state = EngineState.WAKE_LISTEN
                    logger.info("[FOLLOWUP] Goodbye — returning to WAKE_LISTEN")
                    break

                self._turn_count += 1
                logger.info("[THINKING] Turn #%d: '%s'", self._turn_count, text)

                # ── HIERARCHICAL DECISION ENGINE: think before LLM ──
                t_turn_start = time.time()

                # Collect autonomous context for "fix this" / "help" type queries
                from core.autonomous_reasoning import auto_context
                auto_ctx = await auto_context.collect()

                # Run the hierarchical decision engine (L1→L7)
                from core.decision_engine import decision_engine, DecisionPath
                decision = await decision_engine.decide(
                    text,
                    vision_context=None,  # Will be fetched by LLM path if needed
                    search_context=None,   # Will be fetched by LLM path if needed
                    desktop_context=auto_ctx.summary if auto_ctx else "",
                )

                if decision.resolved:
                    # Decision engine handled it — no LLM needed
                    logger.info("[DECIDE] Bypassed LLM — path=%s confidence=%.2f latency=%.0fµs",
                                decision.path.value, decision.confidence, decision.latency_us)

                    # Execute action if present
                    if decision.action and self._action_executor:
                        try:
                            await self._action_executor(decision.action)
                        except Exception as e:
                            logger.warning("[DECIDE] Action execution failed: %s", e)

                    # Execute multi-step actions if present
                    if decision.actions and self._action_executor:
                        for action in decision.actions:
                            try:
                                await self._action_executor(action)
                            except Exception as e:
                                logger.warning("[DECIDE] Workflow action failed: %s", e)

                    # Speak response if present
                    if decision.response:
                        self._set_state(EngineState.THINKING)
                        await self._think_and_speak(decision.response, events, canned=True)

                    self._set_state(EngineState.FOLLOWUP)

                    # Record benchmark: LLM bypassed
                    benchmark.record_turn(
                        text=text,
                        llm_used=False,
                        router_kind=decision.path.value,
                        latency_ms=(time.time() - t_turn_start) * 1000,
                        cache_hit=(decision.path == DecisionPath.WORKING_MEMORY),
                        action_executed=(decision.action is not None or decision.actions is not None),
                        action_success=(decision.action is not None or decision.actions is not None),
                    )

                    # Cache the response for future use
                    if decision.response and decision.path in (
                        DecisionPath.WORKING_MEMORY,
                        DecisionPath.SESSION_MEMORY,
                        DecisionPath.DIRECT_EXECUTION,
                    ):
                        command_router.cache_llm_response(text, decision.response, ttl_s=86400)

                    # Drain stale events and continue
                    drained = 0
                    while True:
                        try:
                            events.get_nowait()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        logger.info("[FOLLOWUP] Drained %d stale STT events", drained)
                    self._set_state(EngineState.COMMAND_LISTEN)
                    continue

                # ── STATE: THINKING (LLM path) ─────────────
                self._set_state(EngineState.THINKING)
                t_llm_start = time.time()
                actions = await self._think_and_speak(text, events)
                # Record benchmark: LLM used
                benchmark.record_turn(
                    text=text,
                    llm_used=True,
                    router_kind=RouteKind.COMPLEX.value,
                    latency_ms=(time.time() - t_turn_start) * 1000,
                    llm_latency_ms=(time.time() - t_llm_start) * 1000,
                    action_executed=len(actions) > 0,
                    action_success=len(actions) > 0,
                )

                # ── STATE: FOLLOWUP ───────────────────────
                self._set_state(EngineState.FOLLOWUP)
                for action_json in actions:
                    await self._run_action(action_json)
                logger.info("[FOLLOWUP] Listening for follow-up (%.0fs window)",
                            CONVERSATION_TIMEOUT_S)
                # ── DRAIN stale events accumulated during TTS/ACTION ──
                # The STT pump runs continuously during SPEAKING and
                # FOLLOWUP. It captures Leo's OWN voice (TTS audio played
                # through speakers) as phantom speech_start/final events.
                # These MUST be drained before re-entering COMMAND_LISTEN,
                # otherwise the first user command is always lost.
                drained = 0
                while True:
                    try:
                        events.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    logger.info("[FOLLOWUP] Drained %d stale STT events "
                                "(TTS/action contamination) — queue clean "
                                "for next turn", drained)
                self._set_state(EngineState.COMMAND_LISTEN)
        finally:
            # DESTROY streaming Whisper the instant the session ends.
            pump.cancel()
            streaming_stt.stop_streaming()
            await asyncio.gather(pump, return_exceptions=True)
            try:
                await stream.aclose()
            except Exception:
                pass
            logger.info("[STT] Streaming Whisper stopped (session over)")


        self._set_state(exit_state)
        print("\n  Listening for wake word...\n")

    async def _stt_event_pump(
        self,
        stream: AsyncIterator[UtteranceEvent],
        events: "asyncio.Queue[UtteranceEvent]",
    ) -> None:
        """Pump UtteranceEvents out of streaming Whisper into the session queue."""
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
        self,
        user_text: str,
        events: "asyncio.Queue[UtteranceEvent]",
        canned: bool = False,
    ) -> List[str]:
        """
        THINKING (LLM) → SPEAKING (TTS) for one user turn.

        The LLM streams sentences into a queue; SPEAKING begins as soon as
        the first sentence exists, so response latency stays minimal while
        the state ordering THINKING → SPEAKING is never violated.

        Returns the ACTION lines collected from the LLM.
        """
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
                await sentence_q.put(None)  # sentinel: stream exhausted

        producer = asyncio.create_task(produce())
        first = await sentence_q.get()
        if first is None:
            # Nothing to say (e.g. pure ACTION response).
            await asyncio.gather(producer, return_exceptions=True)
            return actions

        if not canned:
            logger.info("[LLM] First sentence in %.0fms", (time.time() - t_llm) * 1000)

        # ── STATE: SPEAKING ───────────────────────────────
        self._set_state(EngineState.SPEAKING)
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
        try:
            await streaming_tts.speak_sentences(sentences(), interrupt)
        finally:
            monitor.cancel()
            if not producer.done():
                interrupt.set()
            await asyncio.gather(producer, monitor, return_exceptions=True)
        logger.info("TTS end (%.0fms)", (time.time() - t_tts) * 1000)
        return actions

    async def _watch_interruption(
        self, events: "asyncio.Queue[UtteranceEvent]") -> None:
        """
        Runs ONLY during SPEAKING: confident user speech interrupts TTS
        instantly. Non-interrupt events are HELD (not re-queued, which would
        spin the consumer) and returned to the queue if SPEAKING ends
        normally; on interruption they are discarded because the utterance
        that caused them will re-finalize after we return to COMMAND_LISTEN.
        """
        held: List[UtteranceEvent] = []
        try:
            while True:
                ev = await events.get()
                if ev.kind == "speech_start":
                    logger.info("[SPEAKING] User interrupted — stopping TTS")
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
        """Yield LLM sentences, adding vision/search/desktop/learn/music context."""
        from core.cache_manager import cache_manager

        # Build context parts list instead of mutating user_text in-place
        # (prevents cascading prompt corruption where later stages see
        #  earlier stages' injected text)
        context_parts: list = []

        # ── Pre-populate async caches before LLM call ─────
        # These must run as coroutines so they don't block the event loop.
        # We run them in parallel where possible.
        async def _prepopulate_caches() -> None:
            nonlocal context_parts

            # Desktop state (sync, fast)
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

            # Screen context (async via vision_context_fn)
            if self._references_screen(user_text) and self._vision_context_fn:
                try:
                    vk = cache_manager.vision_key("screen")
                    entry = cache_manager.get("vision", vk)
                    if entry is None:
                        ctx = await self._vision_context_fn()
                        cache_manager.set("vision", vk, ctx, ttl_s=2.0)
                    else:
                        ctx = entry.value
                    if ctx:
                        context_parts.append(f"[Screen context: {ctx}]")
                except Exception as e:
                    logger.debug("[LLM] vision context failed: %s", e)

            # Search context (async via search_provider_fn)
            if self._references_search(user_text) and self._search_provider_fn:
                try:
                    sk = cache_manager.search_key(user_text)
                    entry = cache_manager.get("search", sk)
                    if entry is None:
                        search_ctx = await self._search_provider_fn(user_text)
                        cache_manager.set("search", sk, search_ctx, ttl_s=300.0)
                    else:
                        search_ctx = entry.value
                    if search_ctx:
                        context_parts.append(f"[Web search results: {search_ctx}]")
                except Exception as e:
                    logger.debug("[LLM] search context failed: %s", e)

            # Music status (async via music_agent.status())
            try:
                from services.music_agent import music_agent
                if music_agent.is_playing:
                    mk = cache_manager.make_key("music", "status")
                    entry = cache_manager.get("music", mk)
                    if entry is None:
                        status = await music_agent.status()
                        cache_manager.set("music", mk, status, ttl_s=5.0)
                    else:
                        status = entry.value
                    if status and "Nothing" not in status:
                        context_parts.append(f"[Music: {status}]")
            except Exception as e:
                logger.debug("[LLM] music agent failed: %s", e)

            # Context Composer (sync, fast)
            try:
                mk = cache_manager.memory_key(user_text)
                composed = cache_manager.get_or_compute(
                    "memory", mk,
                    lambda: self._get_composed_context(user_text),
                    ttl_s=10.0,
                )
                if composed:
                    context_parts.append(f"[{composed}]")
            except Exception as e:
                logger.debug("[LLM] context composer failed: %s", e)

            # ── ProjectMode auto-detection ────────────────
            try:
                from agent.project_mode import project_mode
                proj_ctx = project_mode.context_for_llm()
                if proj_ctx:
                    context_parts.append(f"[Project: {proj_ctx}]")
            except Exception as e:
                logger.debug("[LLM] project mode failed: %s", e)

            # ── DesktopObserver rich context ───────────────
            try:
                from services.desktop_state import desktop_state
                snap = desktop_state.snapshot()
                observer_parts = []
                if snap.focused_window and snap.focused_window.title:
                    observer_parts.append(f"Focused: {snap.focused_window.title}")
                if snap.terminal and snap.terminal.cwd:
                    observer_parts.append(f"Terminal: {snap.terminal.cwd}")
                if snap.terminal and snap.terminal.git_branch:
                    observer_parts.append(f"Git: {snap.terminal.git_branch}")
                if snap.browser and snap.browser.current_url:
                    observer_parts.append(f"Browser: {snap.browser.current_url}")
                if observer_parts:
                    context_parts.append("[Observer: " + " | ".join(observer_parts) + "]")
            except Exception as e:
                logger.debug("[LLM] desktop observer failed: %s", e)

        # Populate all caches before starting LLM generation
        await _prepopulate_caches()

        # Build final prompt: user text + all context parts joined
        if context_parts:
            user_text = f"{user_text}\n" + "\n".join(context_parts)

        async for sentence in streaming_llm.generate(user_text, interrupt):
            yield sentence

    # ── Cache helper methods ──────────────────────────────

    @staticmethod
    def _get_desktop_context_sync() -> str:
        """Synchronous wrapper for desktop state context."""
        try:
            from services.desktop_state import desktop_state
            return desktop_state.quick_context()
        except Exception:
            return ""

    def _get_vision_context_sync(self) -> str:
        """Synchronous wrapper for vision context."""
        if not self._vision_context_fn:
            return ""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._vision_context_fn(), loop)
                return future.result(timeout=3)
            return ""
        except Exception:
            return ""

    def _get_search_context_sync(self, user_text: str) -> str:
        """Synchronous wrapper for search context."""
        if not self._search_provider_fn:
            return ""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._search_provider_fn(user_text), loop)
                return future.result(timeout=5)
            return ""
        except Exception:
            return ""

    @staticmethod
    def _get_composed_context(user_text: str) -> str:
        """Synchronous wrapper for context composer."""
        try:
            from agent.context_composer import context_composer
            return context_composer.compose(user_text, max_tokens=800)
        except Exception:
            return ""

    @staticmethod
    def _get_music_status_sync() -> str:
        """Synchronous wrapper for music status."""
        try:
            from services.music_agent import music_agent
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    music_agent.status(), loop)
                return future.result(timeout=2)
            return ""
        except Exception:
            return ""

    @staticmethod
    async def _one_line_stream(text: str) -> AsyncIterator[str]:
        yield text

    async def _speak_line(self, text: str) -> None:
        """Speak a single line (GREETING / FACE_AUTH denial)."""
        logger.info("TTS start")
        t0 = time.time()
        self._tts_interrupt.clear()
        try:
            await streaming_tts.speak_sentences(self._one_line_stream(text),
                                                self._tts_interrupt)
        finally:
            logger.info("TTS end (%.0fms)", (time.time() - t0) * 1000)

    # ── Actions ───────────────────────────────────────────

    async def _run_action(self, action_json: str) -> None:
        """Parse and execute an ACTION line from the LLM."""
        if self._action_executor is None:
            return
        try:
            start = action_json.find("{")
            end = action_json.rfind("}")
            if start < 0 or end <= start:
                return
            action = json.loads(action_json[start:end + 1])
            logger.info("[ACTION] Executing: %s", action)
            result = await self._action_executor(action)
            if result:
                logger.info("[ACTION] Result: %s", result)
        except Exception as e:
            logger.warning("[ACTION] Execution failed: %s", e)

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _contains_wake(lower_text: str) -> bool:
        return any(v in lower_text for v in WAKE_VARIANTS)

    @staticmethod
    def _strip_wake(lower_text: str) -> str:
        """Remove the wake phrase and return the remaining command text."""
        for variant in sorted(WAKE_VARIANTS, key=len, reverse=True):
            idx = lower_text.find(variant)
            if idx >= 0:
                rest = (lower_text[:idx] + " " + lower_text[idx + len(variant):])
                rest = rest.strip(" ,.\t")
                rest = " ".join(rest.split())
                return rest
        return ""

    @staticmethod
    def _is_goodbye(lower_text: str) -> bool:
        return any(p in lower_text for p in GOODBYE_PHRASES)

    @staticmethod
    def _references_screen(text: str) -> bool:
        t = text.lower()
        keys = [
            "screen", "looking at", "this page", "this window", "what am i",
            "read this", "what does this say", "on my screen", "this button",
            "click the", "click this", "what's open", "whats open",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _references_search(text: str) -> bool:
        """Return True if the user query likely needs web search."""
        t = text.lower()
        keys = [
            "search", "look up", "find", "google", "what is", "who is",
            "how to", "how do i", "latest", "news", "weather",
            "definition", "meaning of", "what are", "what does",
            "tell me about", "information on", "learn about",
            "what's happening", "whats happening", "trending",
            "today", "this week", "current", "recent",
        ]
        return any(k in t for k in keys)

    # ── Conversation timeout watchdog (legacy API) ────────

    async def timeout_watchdog(self) -> None:
        """
        DEPRECATED no-op kept for API compatibility with leo.py.

        The 60-second conversation timeout is enforced inside
        COMMAND_LISTEN (session deadline), not by an external watchdog.
        """
        while self._running:
            await asyncio.sleep(1.0)

    # ── Shutdown ──────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully stop the engine."""
        logger.info("[ENGINE] Shutting down conversation engine")
        self._running = False
        streaming_stt.stop_streaming()
        streaming_tts.stop()
        # Stop the GUI pump and destroy the Tk root on the main thread.
        if self._gui_pump_task is not None:
            self._gui_pump_task.cancel()
            try:
                await self._gui_pump_task
            except (asyncio.CancelledError, Exception):
                pass
            self._gui_pump_task = None
        try:
            gui.stop()
        except Exception:
            pass
        report = peak_monitor.report()
        if report:
            logger.info("[ENGINE] Audio stage report: %s", report)


    @property
    def mode(self) -> str:
        """Current state name (e.g. 'WAKE_LISTEN')."""
        return self._state.value if self._state else "BOOT"

    @property
    def state(self) -> Optional[EngineState]:
        return self._state

    @property
    def running(self) -> bool:
        return self._running


# Global singleton
conversation_engine = ConversationEngine()
