"""
ConversationEngine — Leo's STRICT voice-state machine.

This replaces the old "always-streaming Whisper" orchestrator. The previous
design started streaming STT at boot and detected the wake word from Whisper
transcripts, so Whisper ran forever and openWakeWord never gated anything.
That architecture is gone.

STATE MACHINE (no state may bypass another):

    BOOT
      ↓
    FACE_AUTH
      ↓
    IDLE
      ↓
    WAKE_LISTEN
      ↓
    WAKE_DETECTED
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
      ends the session. Silence >= CONVERSATION_TIMEOUT_S (8 s) returns
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
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

from core.gui_dispatcher import gui
from voice.audio_manager import audio_manager

from voice.audio_processing import audio_preprocessor, peak_monitor
from voice.streaming_stt import streaming_stt, is_filler, UtteranceEvent
from voice.streaming_tts import streaming_tts
from voice.wake_model_manager import wake_model_manager, WARMUP_FRAME_SAMPLES
from agent.streaming_llm import streaming_llm
from agent.conversation_memory import conv_memory
from agent.personality import personality

logger = logging.getLogger(__name__)

# ── Conversation lifecycle tuning ────────────────────────────────
CONVERSATION_TIMEOUT_S = 8.0    # silence this long → back to WAKE_LISTEN
GOODBYE_PHRASES = {
    "bye", "goodbye", "see you", "see ya", "later", "that's all",
    "thats all", "nothing else", "i'm done", "im done", "stop listening",
    "go to sleep", "good night", "goodnight",
}
WAKE_VARIANTS = ("leo", "hey leo", "hello leo", "hi leo", "okay leo", "ok leo")

# Re-auth suppression: after a successful face auth, don't ask again for
# this many seconds (default 10 minutes) unless explicitly invalidated.
AUTH_SESSION_S = 600.0

# Wake-loop instrumentation
WAKE_SCORE_LOG_INTERVAL_S = 0.5   # "Wake score X.XX" cadence while idle
WAKE_VAD_HANGOVER_S = 0.6         # speech gate stays open this long after VAD drops
# If the VAD gate is closed, a score this far above threshold still wakes
# (protects against a flaky VAD rejecting a real wake).
WAKE_VAD_OVERRIDE_MARGIN = 0.30

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
    EngineState.BOOT:           {EngineState.FACE_AUTH},
    EngineState.FACE_AUTH:      {EngineState.IDLE, EngineState.GREETING,
                                 EngineState.WAKE_LISTEN},
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
# Optional Silero VAD gate for WAKE_LISTEN
# ═══════════════════════════════════════════════════════════════

class _WakeGateVAD:
    """
    Optional Silero VAD used ONLY inside WAKE_LISTEN.

    Its sole job is to keep the speech gate closed while the room is silent
    so openWakeWord scores produced by fan noise / keyboard clicks are never
    accepted. If Silero is unavailable the gate is permanently open and
    openWakeWord alone decides (exactly like before).
    """

    # silero-vad 6.x accepts ONLY 256/512/768-sample windows at 16 kHz.
    # 480 samples (30 ms) raises "Input audio chunk is too short" — which
    # the old code swallowed, silently forcing the gate open forever.
    FRAME = 512  # 32 ms @ 16 kHz (valid silero-vad 6.x window)


    def __init__(self):
        self._model = None
        self._ready = False

    def load(self) -> bool:
        if self._ready:
            return True
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            self._ready = True
            logger.info("[WAKE] Silero VAD gate loaded (WAKE_LISTEN speech gate)")
            return True
        except Exception as e:
            logger.info("[WAKE] Silero VAD gate unavailable (%s) — "
                        "openWakeWord runs ungated", e)
            self._ready = False
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    def max_speech_prob(self, audio: np.ndarray) -> float:
        """Highest speech probability across the chunk (0..1).

        Accepts float32 [-1, 1] (used AS-IS — NO renormalization, per the
        single-normalization rule) or legacy int16 PCM (decoded once via
        /32768 at this model boundary).
        """
        if not self._ready or len(audio) < self.FRAME:
            return 1.0  # gate open when VAD missing
        try:
            import torch
            best = 0.0
            for i in range(0, len(audio) - self.FRAME + 1, self.FRAME):
                frame = audio[i:i + self.FRAME]
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32) / 32768.0
                with torch.no_grad():
                    prob = self._model(torch.from_numpy(frame), 16000).item()
                if prob > best:
                    best = float(prob)
            return best
        except Exception:
            return 1.0


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

        # WAKE_LISTEN-only modules
        self._wake_vad = _WakeGateVAD()

        # Per-turn cancellation (user interrupts Leo while speaking)
        self._tts_interrupt = asyncio.Event()

        # Conversation activity
        self._session_deadline: float = 0.0
        self._turn_count = 0

        # Vision context provider (set by main/leo)
        self._vision_context_fn = None

        # Action executor (set by main/leo) — maps ACTION dicts to desktop ops
        self._action_executor = None

        # GUI pump task (created in BOOT when running on the main thread)
        self._gui_pump_task: Optional[asyncio.Task] = None


    # ── Wiring ────────────────────────────────────────────

    def set_vision_context(self, fn) -> None:
        """Provide a callable() -> str that returns current screen context."""
        self._vision_context_fn = fn

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
        BOOT → FACE_AUTH → IDLE → (WAKE_LISTEN → … conversation …) forever.
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
        # it is the ONLY audio source for every state.

        if not audio_manager.is_running:
            ok = await loop.run_in_executor(None, audio_manager.start)
            if not ok:
                logger.error("[BOOT] AudioManager failed to start — engine cannot run")
                self._running = False
                return

        # openWakeWord — the ONLY model that stays active forever.
        await loop.run_in_executor(None, self._ensure_wake_model)

        # Optional Silero VAD gate for WAKE_LISTEN (optional by design).
        await loop.run_in_executor(None, self._wake_vad.load)

        # TTS engine selection (no synthesis happens here — playback worker
        # idles until SPEAKING/GREETING actually enqueue audio).
        await loop.run_in_executor(None, streaming_tts.initialize)

        # Whisper is PRELOADED here: the wake-confirmation path
        # (_confirm_wake_by_transcript) needs it hot — a lazy 3 s load on
        # the first trigger loses the wake-phrase window (proven live).
        # The LLM is still NOT contacted until THINKING.
        await loop.run_in_executor(None, streaming_stt.initialize)

        logger.info("[BOOT] Models loaded (wake=%s, vad_gate=%s, whisper=%s) — "
                    "LLM deferred to THINKING",
                    wake_model_manager.model_name or "unavailable",
                    "on" if self._wake_vad.ready else "off",
                    "ready" if streaming_stt.ready else "unavailable")


        # ── STATE: FACE_AUTH ──────────────────────────────
        self._set_state(EngineState.FACE_AUTH)
        await self._boot_auth()

        # ── STATE: IDLE ───────────────────────────────────
        self._set_state(EngineState.IDLE)
        logger.info("[IDLE] Only the wake-word detector is active — "
                    "Whisper inactive, LLM inactive, TTS inactive")

        # ── Forever: WAKE_LISTEN → conversation → WAKE_LISTEN ──
        try:
            while self._running:
                if self._state is not EngineState.WAKE_LISTEN:
                    self._set_state(EngineState.WAKE_LISTEN)
                detection = await self._wake_listen_loop()
                if detection is None:
                    break  # engine stopped
                score_name, score = detection
                self._set_state(EngineState.WAKE_DETECTED)
                logger.info("Wake accepted (model='%s' score=%.2f ≥ %.2f)",
                            score_name, score, wake_model_manager.threshold)
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

    async def _boot_auth(self) -> None:
        """FACE_AUTH at boot: authenticate once if a provider is configured.

        --no-auth mode has no provider → instant pass-through to IDLE.
        Failure does NOT kill Leo: the machine still goes IDLE and
        re-authenticates after the next wake.
        """
        if self._auth_provider is None:
            logger.info("[FACE_AUTH] No auth provider — skipping (dev mode)")
            return
        name = await self._run_auth()
        if name:
            self._auth_user = name
            self._last_auth_time = time.time()
            conv_memory.set_user_name(name)
            logger.info("[FACE_AUTH] Authenticated: %s", name)
        else:
            logger.warning("[FACE_AUTH] Boot authentication failed — "
                           "will retry after the next wake")

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

    async def _wake_listen_loop(self) -> Optional[Tuple[str, float]]:
        """
        WAKE_LISTEN: idle forever on AudioManager + openWakeWord (+ VAD gate).

        Whisper / LLM / TTS are NOT running in this state. Returns
        (model_name, score) when the wake score crosses the threshold,
        or None if the engine is shutting down.
        """
        loop = asyncio.get_event_loop()
        # Resync to "now": any backlog in the ring buffer is dropped so the
        # detector scores fresh audio only.
        last_total = audio_manager.total_samples
        last_score_log = 0.0
        speech_gate_open_until = 0.0

        # ── Thread inventory BEFORE the detector starts ──
        # (spec: print every active thread before entering WAKE_LISTEN;
        # any surviving GUI thread must have its id printed).
        self._log_active_threads("WAKE_LISTEN")

        logger.info("[WAKE_LISTEN] Wake detector started — model='%s' "
                    "phrase='%s' threshold=%.2f vad_gate=%s",
                    wake_model_manager.model_name or "none",
                    wake_model_manager.wake_phrase,
                    wake_model_manager.threshold,
                    "on" if self._wake_vad.ready else "off")
        logger.info("[WAKE_LISTEN] Listening for wake word...")

        # Stall diagnostics: if openWakeWord produces NO inference for >2s,
        # dump exactly why the inference loop is not running.
        last_inference_time = time.monotonic()
        last_stall_dump = 0.0
        empty_reads = 0
        last_model_retry = 0.0
        WAKE_MODEL_RETRY_S = 5.0
        vad_prob = 1.0
        # Rate-limit Whisper wake-confirmations (each costs ~0.5–1 s CPU).
        last_confirm = 0.0
        CONFIRM_MIN_INTERVAL_S = 1.2


        while self._running:
            now = time.monotonic()

            # ── TASK 7: the wake detector MUST always be active ──
            # If the model is missing/unloaded, KEEP RETRYING forever —
            # never exit WAKE_LISTEN, never run deaf. Audio keeps flowing
            # into the ring buffer between retries so nothing is lost.
            if not wake_model_manager.loaded:
                if now - last_model_retry >= WAKE_MODEL_RETRY_S:
                    last_model_retry = now
                    logger.warning(
                        "[WAKE] Wake model NOT loaded (%s) — retrying "
                        "every %.0fs; detector stays in WAKE_LISTEN",
                        wake_model_manager.load_error or "no model",
                        WAKE_MODEL_RETRY_S)
                    ok = await loop.run_in_executor(None, wake_model_manager.load)
                    if ok:
                        logger.info(
                            "[WAKE] Wake model loaded: '%s' phrase='%s' "
                            "threshold=%.2f — detector ACTIVE",
                            wake_model_manager.model_name,
                            wake_model_manager.wake_phrase,
                            wake_model_manager.threshold)
                await asyncio.sleep(0.25)
                continue

            # ── TASK 7: no exception may EVER kill the wake loop ──
            try:
                new_audio, last_total = audio_manager.read_since(last_total)
                if len(new_audio) == 0:
                    empty_reads += 1
                    await asyncio.sleep(0.02)
                else:
                    empty_reads = 0

                # ── No-inference watchdog: dump why the loop is stalling ──
                if (now - last_inference_time > 2.0
                        and now - last_stall_dump > 2.0):
                    last_stall_dump = now
                    logger.warning(
                        "[WAKE] NO INFERENCE for %.1fs — diagnostics: "
                        "model_loaded=%s running=%s audio_empty_streak=%d "
                        "total_samples=%d ring_buffer_alive=%s",
                        now - last_inference_time,
                        wake_model_manager.loaded, self._running,
                        empty_reads, last_total, audio_manager.is_running)

                if len(new_audio) == 0:
                    continue

                # Noise suppression (spectral gating). NO AGC, NO normalization —
                # the signal is never amplified beyond unity.
                processed = await loop.run_in_executor(
                    None, audio_preprocessor.process, new_audio)
                # Stage trace: print peak/RMS of audio entering the wake detector.
                peak_monitor.log("wake_detector", processed)

                # Optional Silero VAD speech gate
                if self._wake_vad.ready:
                    vad_prob = await loop.run_in_executor(
                        None, self._wake_vad.max_speech_prob, processed)
                    if vad_prob > 0.5:
                        speech_gate_open_until = now + WAKE_VAD_HANGOVER_S
                else:
                    vad_prob = 1.0
                    speech_gate_open_until = now + WAKE_VAD_HANGOVER_S  # gate open

                # Feed openWakeWord in non-overlapping 80 ms streaming frames.
                for frame in self._iter_wake_frames(processed):
                    t_pred = time.perf_counter()
                    preds = await loop.run_in_executor(
                        None, wake_model_manager.predict_stream, frame)
                    pred_ms = (time.perf_counter() - t_pred) * 1000.0
                    if not preds:
                        continue
                    last_inference_time = time.monotonic()
                    score = max(float(s) for s in preds.values())

                    # ── TASK 4: EVERY prediction is printed with the full
                    # decision context — score, model, threshold, VAD,
                    # latency. If these lines stop, the detector is dead.
                    if score >= 0.01 or now - last_score_log >= WAKE_SCORE_LOG_INTERVAL_S:
                        logger.info(
                            "Wake score=%.3f model=%s threshold=%.2f "
                            "vad=%.2f latency=%.1fms",
                            score, wake_model_manager.model_name or "?",
                            wake_model_manager.threshold, vad_prob, pred_ms)
                        last_score_log = now

                    # ── PRODUCTION WAKE DECISION ─────────────────────────
                    # The small-data verifier CANNOT be trusted on mic-domain
                    # audio (proven: 0.001 on the real phrase through this
                    # mic, 0.79 on TV audio). So the wake decision is:
                    #   TRIGGER  — openWakeWord score ≥ threshold, OR
                    #              sustained VAD speech (the phrase the SVM
                    #              missed still reaches Whisper), rate-limited.
                    #   AUTHORITY — the transcript must contain the wake
                    #              phrase (strict verify_wake_transcript).
                    threshold = wake_model_manager.threshold
                    speech_now = vad_prob > 0.5
                    confirm_due = (now - last_confirm) >= CONFIRM_MIN_INTERVAL_S
                    triggered = (
                        score >= threshold
                        or (speech_now and confirm_due
                            and now <= speech_gate_open_until)
                    )
                    if not triggered:
                        continue

                    gate_open = now <= speech_gate_open_until
                    if not gate_open and score < threshold + WAKE_VAD_OVERRIDE_MARGIN:
                        logger.info(
                            "[WAKE] Score %.2f ≥ %.2f but VAD gate closed "
                            "(no speech) — rejected", score, threshold)
                        continue

                    last_confirm = now
                    verified, transcript = await self._confirm_wake_by_transcript()
                    if not verified:
                        if score >= threshold:
                            logger.info(
                                "[WAKE] score=%.2f ≥ trigger %.2f but transcript "
                                "rejected (%r) — still listening",
                                score, threshold, transcript or "<no speech>")
                        continue
                    logger.info("WAKE ACCEPTED score=%.3f model=%s vad=%.2f "
                                "transcript='%s'",
                                score, wake_model_manager.model_name, vad_prob,
                                transcript)
                    return wake_model_manager.highest_score()[0] or "wake", score


            except asyncio.CancelledError:
                raise
            except Exception:
                # TASK 7: the wake detector must NEVER die. Log the full
                # traceback (no hidden exceptions) and keep listening.
                logger.exception("[WAKE] Wake-loop iteration failed — "
                                 "recovering, detector stays active")
                await asyncio.sleep(0.1)

        return None


    @staticmethod
    def _iter_wake_frames(audio: np.ndarray):
        """Yield openWakeWord-sized frames (1280 samples = 80 ms)."""
        n = len(audio)
        step = WARMUP_FRAME_SAMPLES
        for i in range(0, n, step):
            frame = audio[i:i + step]
            if len(frame) < step:
                frame = np.pad(frame, (0, step - len(frame)))
            yield frame

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
        """
        loop = asyncio.get_event_loop()

        # Interrupt current buffers: drop everything captured up to now so
        # neither the wake phrase nor the chime leaks into Whisper.
        audio_manager.get_recent_audio(0.05)  # touch to keep stream hot
        wake_model_manager.reset_stream()     # clear prediction streaks

        # Play wake chime.
        await loop.run_in_executor(None, self._play_wake_chime)

        # Face auth — only when the session expired (never blocks a valid one).
        fresh_auth = False
        if self._needs_auth():
            self._set_state(EngineState.FACE_AUTH)
            name = await self._run_auth()
            if not name:
                logger.warning("[FACE_AUTH] Denied — returning to WAKE_LISTEN")
                await self._speak_line(
                    "I couldn't verify your identity. Please try again.")
                self._set_state(EngineState.WAKE_LISTEN)
                return
            self._auth_user = name
            self._last_auth_time = time.time()
            conv_memory.set_user_name(name)
            fresh_auth = True
            logger.info("[FACE_AUTH] Authenticated: %s", name)

        # ── STATE: GREETING ───────────────────────────────
        self._set_state(EngineState.GREETING)
        name = conv_memory.user_name or self._auth_user
        if fresh_auth and name:
            greeting = f"Welcome back, {name}."
        else:
            greeting = personality.greeting(returning=conv_memory.turn_count > 0)
            if name and conv_memory.turn_count == 0:
                greeting = greeting.rstrip(".") + f", {name}."
        logger.info("[GREETING] '%s'", greeting)
        await self._speak_line(greeting)

        # ── STATE: COMMAND_LISTEN … (full session) ───────
        await self._conversation_session()

    async def _confirm_wake_by_transcript(self) -> Tuple[bool, str]:
        """Whisper-based wake confirmation — the production wake AUTHORITY.

        openWakeWord's small-data verifier is only a trigger: it cannot
        generalize across room conditions (loud TV, TTS echo, fan noise).
        This method transcribes the most recent ~2.5 s of ring-buffer
        audio with the shared streaming-Whisper transcriber and applies
        the strict transcript verifier (must contain a real wake phrase —
        'leo'/'lio' as a distinctive word, never a lookalike).

        Returns (verified, transcript). NEVER raises: any failure returns
        (False, '') and WAKE_LISTEN simply continues.
        """
        try:
            from voice.wake_word import verify_wake_transcript
            from voice.audio_processing import float32_to_int16
            loop = asyncio.get_event_loop()
            # Shared transcriber, loaded lazily once (also used by
            # COMMAND_LISTEN — no duplicate model in memory).
            if not streaming_stt.ready:
                ok = await loop.run_in_executor(None, streaming_stt.initialize)
                if not ok:
                    return False, ""
            audio = audio_manager.get_recent_audio(2.5)
            if len(audio) < 8000:  # <0.5 s — nothing to transcribe yet
                return False, ""
            pcm = float32_to_int16(audio).tobytes()
            # vad_filter OFF: Silero already gated this audio as speech;
            # faster-whisper's internal VAD drops quiet-but-real phrases.
            text = await loop.run_in_executor(
                None, streaming_stt._whisper.transcribe, pcm, 16000, False)
            ok = verify_wake_transcript(text)
            return bool(ok), (text or "")

        except Exception as e:
            logger.debug("[WAKE] transcript confirmation error: %s", e)
            return False, ""

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

    async def _conversation_session(self) -> None:
        """
        Run one conversation session:

            COMMAND_LISTEN → THINKING → SPEAKING → FOLLOWUP → COMMAND_LISTEN …

        until 8 s of silence (→ WAKE_LISTEN) or a goodbye (→ WAKE_LISTEN).

        Streaming Whisper is created here and DESTROYED when the session
        ends — it never exists outside COMMAND_LISTEN-family states.
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
                    self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S
                    continue

                if ev.kind == "partial":
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

                # ── STATE: THINKING ───────────────────────
                self._set_state(EngineState.THINKING)

                if self._is_goodbye(lower):
                    # Canned farewell: THINKING → SPEAKING → FOLLOWUP → WAKE_LISTEN
                    await self._think_and_speak(personality.farewell(), events, canned=True)
                    self._set_state(EngineState.FOLLOWUP)
                    exit_state = EngineState.WAKE_LISTEN
                    logger.info("[FOLLOWUP] Goodbye — returning to WAKE_LISTEN")
                    break

                self._turn_count += 1
                logger.info("[THINKING] Turn #%d: '%s'", self._turn_count, text)

                actions = await self._think_and_speak(text, events)

                # ── STATE: FOLLOWUP ───────────────────────
                self._set_state(EngineState.FOLLOWUP)
                for action_json in actions:
                    await self._run_action(action_json)
                logger.info("[FOLLOWUP] Listening for follow-up (%.0fs window)",
                            CONVERSATION_TIMEOUT_S)
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
        """Yield LLM sentences, adding vision context when useful."""
        if self._references_screen(user_text) and self._vision_context_fn:
            try:
                loop = asyncio.get_event_loop()
                ctx = await loop.run_in_executor(None, self._vision_context_fn)
                if ctx:
                    user_text = f"{user_text}\n[Screen context: {ctx}]"
            except Exception as e:
                logger.debug("[LLM] vision context failed: %s", e)

        async for sentence in streaming_llm.generate(user_text, interrupt):
            yield sentence

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

    # ── Conversation timeout watchdog (legacy API) ────────

    async def timeout_watchdog(self) -> None:
        """
        DEPRECATED no-op kept for API compatibility with leo.py.

        The 8-second conversation timeout is enforced inside
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
