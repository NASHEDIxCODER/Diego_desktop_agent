"""
ConversationEngine — Diego's clean voice-state machine.

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
    after endpoint. The conversation session is ENDLESS: silence NEVER
    returns to IDLE — only an explicit sleep command ("go to sleep",
    "stop listening", …) closes the session and returns to wake mode.

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
import inspect
import logging
import re
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
# Silence budget: how long the system stays in LISTEN after the last speech
# before the session loop refreshes its wait. In the ENDLESS conversation
# session (2026-09-21) silence NEVER returns to IDLE (wake mode) — the
# deadline is refreshed and Diego keeps listening. Only an explicit SLEEP
# command closes the session. This value is kept as a re-arm cadence and
# for the between-turn deadline bookkeeping.
CONVERSATION_TIMEOUT_S = 60.0
# Per-state watchdog ceilings (Phase 3). Any state held longer than its
# ceiling is reported as a structured STATE TIMEOUT record and the turn
# recovers safely. WAKE and FACE_AUTH are intentionally unbounded (Diego
# waits forever for the wake word / camera popup), so they are NOT listed.
# LISTEN is also exempt while the endless conversation session is active
# (the session only ends on an explicit sleep command).

STATE_WATCHDOG_INTERVAL_S = 1.0
STATE_TIMEOUTS_S = {
    "LISTEN": CONVERSATION_TIMEOUT_S + 15.0,
    "THINK": 60.0,
    "SPEAK": 120.0,
}
# ── SLEEP commands (endless session exit, 2026-09-21) ─────────
# Explicit user commands that close the conversation session and return
# Diego to wake mode. Silence NEVER closes the session anymore.
SLEEP_PHRASES = {
    "go to sleep", "sleep", "that's all", "thats all", "nothing else",
    "i'm done", "im done", "stop listening", "stop the session",
    "end session", "end the session", "good night", "goodnight", "cancel",
}
# Deprecated alias — older scripts/tests (debug/voice_pipeline_validation.py)
# reference GOODBYE_PHRASES. Same set; sleep commands are the canonical name.
GOODBYE_PHRASES = SLEEP_PHRASES
AUTH_SESSION_S = 600.0  # 10 minutes
CHIME_PATH = Path(__file__).resolve().parent.parent / "Diego.wav"

# ── ENDLESS SESSION lifecycle (2026-09-21) ────────────────────
# Why the conversation session closed. ONLY an explicit sleep command is
# a normal close that returns Diego to wake-word mode; every other close
# is an internal failure the endless session must RECOVER from (it
# re-opens itself — the user never has to repeat the wake word).
SESSION_CLOSE_SLEEP = "sleep_command"
SESSION_CLOSE_ERROR = "session_error"
SESSION_CLOSE_STT_UNAVAILABLE = "stt_unavailable"
SESSION_CLOSE_SHUTDOWN = "shutdown"
# How often the session loop wakes up to check pump health / shutdown,
# even while the silence deadline is still far away. A dead STT pump
# must never leave a deaf, immortal session running.
SESSION_POLL_S = 1.0
# Bounded recovery: after this many consecutive session attempts that
# failed for INTERNAL reasons, Diego gives up and returns to wake mode
# (with one spoken explanation) instead of spinning forever.
SESSION_MAX_STREAM_RETRIES = 3
SESSION_RETRY_DELAY_S = 0.5


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
    # IDLE → WAKE is the normal wake-word path.
    # IDLE → FACE_AUTH is the wake-bypass path (--no-wake flag, or wake
    # model DEGRADED/unavailable) while face authentication is still
    # enabled — wake and auth are INDEPENDENT controls.
    # IDLE → LISTEN only when face auth is disabled (--no-auth) or the
    # auth session is still valid.
    EngineState.IDLE:      {EngineState.WAKE, EngineState.FACE_AUTH,
                            EngineState.LISTEN},
    EngineState.WAKE:      {EngineState.FACE_AUTH, EngineState.LISTEN},
    EngineState.FACE_AUTH: {EngineState.LISTEN},
    EngineState.LISTEN:    {EngineState.THINK, EngineState.IDLE},
    EngineState.THINK:     {EngineState.SPEAK},
    EngineState.SPEAK:     {EngineState.LISTEN, EngineState.IDLE},
}


# ═══════════════════════════════════════════════════════════════
# ENDLESS conversation-session state machine (2026-09-21)
# ═══════════════════════════════════════════════════════════════

class SessionState(str, Enum):
    """Lifecycle of the persistent post-auth conversation session.

        IDLE ──(wake + face auth)──▶ ACTIVE ──(explicit sleep)──▶ CLOSED
                                     │  ▲
                                     └──┘ internal errors re-open it

      IDLE     — wake mode: no conversation session exists.
      ACTIVE   — the endless LISTEN → THINK → SPEAK → LISTEN loop. Silence,
                 completed commands and TTS NEVER leave this state.
      CLOSING  — a close was requested (normally an explicit sleep command);
                 the farewell response is still being spoken.
      CLOSED   — the stream is torn down; `close_reason` records why.
    """
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"



# ═══════════════════════════════════════════════════════════════
# ConversationEngine
# ═══════════════════════════════════════════════════════════════

class ConversationEngine:
    """Clean 7-state conversational engine with runtime diagnostics."""

    def __init__(self):
        self._state: Optional[EngineState] = None
        self._state_entered: float = time.monotonic()
        self._running = False
        self._no_wake: bool = False
        # Wake availability: False when the wake model failed to load at
        # boot — the engine then follows the documented wake fallback
        # (always-LISTEN instead of spinning forever in the WAKE state
        # with repeated model retries). Wake and face authentication are
        # INDEPENDENT: a degraded wake model NEVER disables the auth
        # provider — only an explicit --no-auth does that.
        self._wake_active: bool = True

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
        # ── ENDLESS CONVERSATION SESSION STATE MACHINE (2026-09-21) ──
        # After wake + face authentication the engine runs a PERSISTENT
        # session: LISTEN → THINK → SPEAK → LISTEN continuously. Silence,
        # completed commands and TTS NEVER return Diego to wake-word mode;
        # only an explicit SLEEP command does (then IDLE → WAKE). The
        # session lifecycle is tracked explicitly (state + id + turn count
        # + close reason) so tests and diagnostics can prove the invariant.
        self._session_state: SessionState = SessionState.IDLE
        self._session_id: int = 0
        self._session_started_at: float = 0.0
        self._session_turns: int = 0
        self._session_close_reason: Optional[str] = None
        self._turn_count = 0

    # ── Endless-session state machine ─────────────────────

    @property
    def _endless_session(self) -> bool:
        """Back-compat flag: True while an endless conversation session is
        active or closing. While True the LISTEN watchdog ceiling is
        exempt — silence must never tear the session down."""
        return self._session_state in (SessionState.ACTIVE, SessionState.CLOSING)

    def _begin_session(self) -> None:
        """Open a fresh endless conversation session."""
        self._session_id += 1
        self._session_state = SessionState.ACTIVE
        self._session_started_at = time.monotonic()
        self._session_turns = 0
        self._session_close_reason = None
        logger.info("[SESSION] #%d opened — endless LISTEN ↔ THINK ↔ SPEAK "
                    "(exit only on an explicit sleep command)", self._session_id)

    def _request_session_close(self, reason: str) -> None:
        """Record WHY the session is closing.

        Only an explicit sleep command ('go to sleep', 'stop listening', …)
        is a normal close; every other reason is an internal failure the
        endless session recovers from by re-opening itself.
        """
        self._session_close_reason = reason
        if self._session_state == SessionState.ACTIVE:
            self._session_state = SessionState.CLOSING
        logger.info("[SESSION] #%d close requested (%s)",
                    self._session_id, reason)

    def _end_session(self) -> None:
        """Close the session; the close reason is preserved for the run
        loop (wake mode on sleep, self-recovery on internal errors)."""
        self._session_state = SessionState.CLOSED
        logger.info("[SESSION] #%d closed (%s) turns=%d duration=%.1fs",
                    self._session_id, self._session_close_reason or "unknown",
                    self._session_turns,
                    time.monotonic() - self._session_started_at
                    if self._session_started_at else 0.0)

    @staticmethod
    def _is_sleep_command(text: str) -> bool:
        """True ONLY for an explicit sleep command (endless-session exit).

        Matching rules:
          * the normalized utterance IS one of SLEEP_PHRASES, or
          * a SLEEP_PHRASES phrase appears as WHOLE WORDS in the
            utterance ("hey diego, go to sleep now" closes the session;
            the old raw-substring check let "cancel" inside
            "cancellation" or "cancelled" close it too), or
          * the utterance ENDS with the bare word "sleep" ("diego sleep",
            "please sleep") — but "sleep" INSIDE another command ("sleep
            music", "sleep paralysis") must never close the session.
        """
        lower = (text or "").strip().lower()
        lower = re.sub(r"[^\w\s']", " ", lower)
        lower = " ".join(lower.split())
        if not lower:
            return False
        if lower in SLEEP_PHRASES:
            return True
        words = lower.split()
        # A trailing bare "sleep" is an explicit sleep command even when
        # the ASR prepends fragments ("diego sleep", "please sleep") — but
        # "sleep" INSIDE another command ("sleep music", "sleep paralysis")
        # must never close the session.
        if words[-1] == "sleep":
            return True
        for phrase in SLEEP_PHRASES:
            # Short single-word phrases (≤5 chars, i.e. "sleep") are
            # handled by the exact match + trailing-word rule above only —
            # a whole-word search here would end the session on any
            # sentence containing the word "sleep".
            if len(phrase) <= 5 and " " not in phrase:
                continue
            if re.search(r"\b" + re.escape(phrase) + r"\b", lower):
                return True
        return False

    def _finish_turn_rearm(
            self, events: "asyncio.Queue[UtteranceEvent]") -> None:
        """Between-turns re-arm for the ENDLESS session.

        Runs after EVERY turn (Brain turn, identity response, failure
        response) before returning to LISTEN, so NO stale drain/gate/VAD
        state survives the THINK/SPEAK boundary:

          1. rearm_between_turns(): gate OPEN + drain request (consumed by
             the live streaming loop) + eager VAD reset,
          2. refresh the between-turns silence deadline.

        The shared event queue is intentionally not drained here. It is fed
        by the asynchronous STT pump, and a valid follow-up utterance may
        arrive while this turn is being completed. The listener's drain
        request is responsible for clearing stale events at the source.
        """
        try:
            command_listener.rearm_between_turns()
        except Exception as e:
            logger.warning("[LISTEN] rearm_between_turns failed: %s", e)

        self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S

        # GUI pump
        self._gui_pump_task: Optional[asyncio.Task] = None

        # ── Runtime diagnostics ──
        self._diag: dict = {}

    # ── Wiring ────────────────────────────────────────────

    def set_auth_provider(self, fn) -> None:
        self._auth_provider = fn

    def set_auth_disabled(self) -> None:
        """Disable face authentication entirely (--no-auth dev mode).

        Clears the auth provider so `_needs_auth()` always returns False.
        This is the ONLY correct way to bypass auth — `set_authenticated(None)`
        leaves the provider set, so `_needs_auth()` would still return True
        (because `_auth_user is None`) and the camera would still open.
        """
        self._auth_provider = None
        self._auth_user = None
        self._last_auth_time = time.time()
        logger.info("[ENGINE] Face authentication disabled (--no-auth)")

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

    async def run(self, no_wake: bool = False) -> None:
        """IDLE → WAKE → (FACE_AUTH) → LISTEN → THINK → SPEAK → IDLE … forever.

        Wake detection and face authentication are INDEPENDENT controls:

        * `no_wake=True` (explicit --no-wake) bypasses ONLY wake detection.
          Face authentication still runs (IDLE → FACE_AUTH → LISTEN) unless
          --no-auth was also supplied and disabled the auth provider.
        * If the wake model is UNAVAILABLE at boot (normal mode, no flags),
          wake is marked DEGRADED and the documented wake fallback applies:
          always-LISTEN mode. Face authentication remains ACTIVE — a wake
          failure is NEVER treated as --no-auth.
        * If face authentication fails or is unavailable, the failure is
          reported honestly; the auth provider stays active and wake
          detection is NEVER disabled by it.

        VAD, STT, routing, LLM, search, tools, verification and TTS all
        remain fully active on every path.
        """
        self._no_wake = no_wake
        self._running = True
        loop = asyncio.get_event_loop()

        # ── Boot: start audio + load models ──
        self._set_state(EngineState.IDLE)
        logger.info("[ENGINE] Diego conversation engine starting")

        # ── Local Knowledge subsystem (2026-09-02) ──
        # Background indexing + PC snapshot. start() NEVER blocks: it
        # only schedules daemon threads. Diego keeps booting regardless.
        try:
            from knowledge.service import knowledge_service
            knowledge_service.start()
        except Exception as e:
            logger.warning("[ENGINE] Knowledge subsystem not started: %s", e)

        self._setup_gui()

        # AudioManager — bounded retries, never infinite. A permanently
        # missing microphone (e.g. Docker without PulseAudio socket) must
        # NOT spin forever. After max retries, mark audio DEGRADED and
        # continue without voice input.
        if audio_manager.is_running:
            # Audio already active (e.g. pre-initialized by a caller or
            # test stub). Skip initialization entirely.
            logger.info("[ENGINE] AudioManager already running — skipping init")
            self._mic_available = True
        else:
            AUDIO_MAX_RETRIES = 3
            AUDIO_RETRY_DELAY_S = 5.0
            for _attempt in range(1, AUDIO_MAX_RETRIES + 1):
                if not self._running:
                    break
                ok = await loop.run_in_executor(None, audio_manager.start)
                if ok:
                    break
                logger.error("[ENGINE] AudioManager failed (attempt %d/%d) — %s",
                             _attempt, AUDIO_MAX_RETRIES,
                             f"retrying in {AUDIO_RETRY_DELAY_S}s"
                             if _attempt < AUDIO_MAX_RETRIES
                             else "no more retries — audio UNAVAILABLE")
                if _attempt < AUDIO_MAX_RETRIES:
                    await asyncio.sleep(AUDIO_RETRY_DELAY_S)
            if not audio_manager.is_running:
                logger.error(
                    "[ENGINE] AudioManager did not start after %d attempts — "
                    "audio input is UNAVAILABLE. Diego continues without "
                    "voice (text input, wake detection, and TTS disabled). "
                    "Mount the host PulseAudio socket to enable audio: "
                    "-v /run/user/$UID/pulse:/run/user/$UID/pulse",
                    AUDIO_MAX_RETRIES)
                # Do NOT set self._running = False — Diego keeps running
                # in text-only mode. The health check correctly reports
                # microphone FAILED and the wake listener idles.
                self._wake_active = False
                self._mic_available = False
            else:
                self._mic_available = True

        # openWakeWord — a boot-time load failure must NOT cause an
        # infinite WAKE-state retry loop. One attempt; on failure wake is
        # marked DEGRADED and the documented wake fallback applies
        # (always-LISTEN). This NEVER touches face authentication: the
        # configured auth provider stays active (only --no-auth disables
        # it).
        wake_ok = await loop.run_in_executor(None, self._ensure_wake_model)
        self._wake_active = wake_ok
        if not wake_ok and not no_wake:
            logger.error(
                "[ENGINE] Wake model UNAVAILABLE (%s) — wake detection is "
                "DEGRADED for this session: following the documented "
                "always-LISTEN wake fallback. Face authentication remains "
                "ACTIVE (it is disabled only by an explicit --no-auth).",
                wake_model_manager.load_error or "model failed to load")

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

        # ── State watchdog (Phase 3) ──
        watchdog = asyncio.create_task(self._state_watchdog())

        # ── Forever loop ──
        try:
            while self._running:
                if self._no_wake or not self._wake_active:
                    # ── Wake is bypassed (--no-wake flag) or DEGRADED
                    # (model unavailable). Face authentication is an
                    # INDEPENDENT control: the auth gate below still runs
                    # unless --no-auth disabled the provider. ──
                    if self._no_wake:
                        if self._auth_provider is None:
                            logger.info(
                                "[ENGINE] --no-wake --no-auth: bypassing "
                                "wake detection and face auth (dev mode)")
                        else:
                            logger.info(
                                "[ENGINE] --no-wake: bypassing wake "
                                "detection only — face auth remains active")
                    else:
                        logger.warning(
                            "[ENGINE] Wake DEGRADED: wake detection "
                            "unavailable this session (documented "
                            "always-LISTEN fallback) — face auth remains "
                            "active")
                    # STATE: FACE_AUTH (if needed) — identical policy to
                    # the normal post-wake path. Skipped only when auth is
                    # disabled (--no-auth) or the session is still valid.
                    await self._face_auth_gate(
                        trigger="no-wake" if self._no_wake else "wake-degraded")
                    # Documented wake fallback entry: always-LISTEN mode.
                    self._set_state(EngineState.LISTEN)
                    await self._conversation_session()
                    continue

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

                # STATE: FACE_AUTH (if needed) — independent of wake
                await self._face_auth_gate(trigger="wake")

                # STATE: LISTEN → THINK → SPEAK → (loop back)
                await self._conversation_session()

        except asyncio.CancelledError:
            logger.info("[ENGINE] Conversation engine cancelled")
        finally:
            self._running = False
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)

    # ── State watchdog (Phase 3) ─────────────────────────────────

    async def _state_watchdog(self) -> None:
        """Log a structured STATE TIMEOUT whenever a bounded state exceeds
        its ceiling, and drive a safe recovery. Runs until cancelled."""
        prev_state: Optional[str] = None
        try:
            while self._running:
                await asyncio.sleep(STATE_WATCHDOG_INTERVAL_S)
                if self._state is None:
                    continue
                name = self._state.value
                ceiling = STATE_TIMEOUTS_S.get(name)
                if ceiling is None:
                    prev_state = name
                    continue

                # ── ENDLESS SESSION (2026-09-21) ──
                # While the endless conversation session is active, LISTEN
                # may idle for an arbitrarily long time between turns — the
                # session is only closed by an explicit SLEEP command.
                # Silence must not trip the watchdog or tear down the stream.
                if name == "LISTEN" and self._endless_session:
                    prev_state = name
                    continue

                elapsed = time.monotonic() - self._state_entered
                if elapsed <= ceiling:
                    prev_state = name
                    continue

                logger.error(
                    "STATE TIMEOUT state=%s previous=%s elapsed=%.1fs "
                    "ceiling=%.1fs thread=%s audio_running=%s active_threads=%s",
                    name, prev_state or "START", elapsed, ceiling,
                    threading.current_thread().name, audio_manager.is_running,
                    threading.active_count())

                # Safe recovery: break a stalled LISTEN/THINK/SPEAK by
                # cancelling the current conversation session's stream pump.
                try:
                    command_listener.stop_streaming()
                except Exception:
                    pass

                # Prevent a spurious repeated log for the same stall.
                self._state_entered = time.monotonic()
                prev_state = name
        except asyncio.CancelledError:
            pass

    # ── Helpers ───────────────────────────────────────────

    def _setup_gui(self) -> None:
        """GUI-dispatcher bootstrap for EVERY host architecture.

        Threading contract (production):
          * Qt UI mode  — the Qt application/event loop owns the MAIN
            thread and hosts the GUI dispatcher (gui.start() + a Qt pump
            timer in ui/__main__.py). The ConversationEngine runs on the
            background DiegoPipeline thread and simply USES the
            dispatcher via thread-safe gui.submit() marshalling.
          * Headless CLI mode — the engine itself runs on the main
            thread, so it starts and pumps the dispatcher here.
          * No GUI host and a background engine thread — headless: the
            dispatcher stays unavailable and callers fall back safely.

        Long-running STT/LLM/TTS work NEVER moves onto the GUI thread in
        any of these modes.
        """
        if gui.available:
            # Host-owned dispatcher (Qt main thread) — never start or pump
            # it from the engine thread.
            logger.info("[ENGINE] GUI dispatcher hosted by the UI main "
                        "thread (id=%s) — engine marshals GUI work via "
                        "gui.submit()", gui.main_thread_ident)
            return
        if threading.current_thread() is threading.main_thread():
            if gui.start():
                self._gui_pump_task = asyncio.create_task(gui.pump())
            else:
                logger.info("[ENGINE] No GUI toolkit available — running "
                            "headless (GUI disabled)")
        else:
            logger.info("[ENGINE] Engine on background thread without a "
                        "GUI host — running headless (GUI disabled)")

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

    async def _face_auth_gate(self, trigger: str = "wake") -> None:
        """STATE: FACE_AUTH — the single face-auth gate for EVERY path.

        Used by the normal post-wake path AND the wake-bypass/degraded
        path so authentication policy is identical everywhere. Wake and
        auth are independent: this gate runs whenever `_needs_auth()` is
        True, regardless of how (or whether) wake was reached.

        A failure/denial is reported honestly and NEVER disables the auth
        provider — an auth failure is NOT equivalent to --no-auth.
        """
        if not self._needs_auth():
            return
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
            # Greet the user by name after successful auth.
            # Uses the guarded path so the greeting is NEVER
            # transcribed as a user command (laptop speakers).
            greeting = f"Welcome back, {name}!"
            logger.info("[FACE_AUTH] Greeting: '%s'", greeting)
            await self._speak_guarded(greeting)
        else:
            # Honest failure reporting: the provider stays ACTIVE so the
            # next gate retries authentication. This is NEVER silently
            # treated as --no-auth.
            logger.error(
                "[FACE_AUTH] Face authentication FAILED or UNAVAILABLE "
                "(trigger=%s) — reported honestly; auth provider remains "
                "ACTIVE (NOT equivalent to --no-auth). Continuing this "
                "session unauthenticated.", trigger)
            session_recorder.record_face_auth(auth_latency, False)

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
            # Play through the SAME resolved output device as TTS so the
            # chime is audible on the user's selected/validated speaker.
            sd.play(audio, rate, device=streaming_tts.output_device)
            sd.wait()
        except Exception as e:
            logger.debug("[WAKE] Chime playback failed: %s", e)

    # ── STATE: LISTEN / THINK / SPEAK ─────────────────────

    async def _conversation_session(self) -> None:
        """ENDLESS conversation session: LISTEN → THINK → SPEAK → LISTEN …

        After wake + face auth the session NEVER ends on its own:

          * silence refreshes the listen deadline (never returns to wake),
          * completed commands return to LISTEN (never to wake),
          * TTS is re-armed between turns (never to wake),
          * internal failures (dead STT pump, stream errors, Brain
            exceptions) RE-OPEN the session — the user never has to
            repeat the wake word.

        The session closes ONLY on an explicit SLEEP command (see
        `_is_sleep_command` / SLEEP_PHRASES), then the engine returns to
        wake mode (IDLE → WAKE).
        """
        loop = asyncio.get_event_loop()
        t_session_start = time.time()
        stream_failures = 0

        # ── ENDLESS SESSION outer loop ────────────────────────
        # One iteration = one complete stream lifecycle. The loop re-opens
        # the session after internal errors; it exits ONLY on an explicit
        # sleep command, recognizer unavailability, or shutdown.
        while self._running:
            if not command_listener.ready:
                ok = await loop.run_in_executor(
                    None, command_listener.initialize)
                if not ok:
                    logger.error(
                        "[LISTEN] Whisper unavailable — returning to IDLE")
                    await self._speak_guarded(
                        "My speech recognizer isn't available right now.")
                    self._request_session_close(SESSION_CLOSE_STT_UNAVAILABLE)
                    break

            # ── POST-WAKE READINESS FIX (2026-09-20) ──
            # The post-wake path runs FACE_AUTH (which greets via
            # _speak_guarded → resume_listening → _drain_requested=True)
            # BEFORE this LISTEN session. That drain request lingers because
            # no command stream consumed it, and stream_utterances()'s first
            # loop iteration would then execute it AFTER establishing its
            # command_session_start boundary — discarding the first 1-2s of
            # the spoken command. Prepare the listener NOW (clear stale
            # drain, force gate open, reset VAD, anchor the boundary) so the
            # command session boundary is honored from the very first sample
            # after face auth.
            try:
                command_listener.prepare_command_session()
            except Exception as e:
                logger.warning("[LISTEN] prepare_command_session failed: %s", e)

            events: "asyncio.Queue[UtteranceEvent]" = asyncio.Queue()
            # CRITICAL FIX (2026-08-29): stream_utterances() can raise (e.g.
            # _frames_to_bytes ValueError on empty frames). Wrap it so the
            # engine never crashes — an endless session re-opens itself.
            try:
                stream = command_listener.stream_utterances()
            except Exception as e:
                stream_failures += 1
                logger.error(
                    "[LISTEN] stream_utterances failed (%d/%d): %s",
                    stream_failures, SESSION_MAX_STREAM_RETRIES, e)
                self._request_session_close(SESSION_CLOSE_ERROR)
                if stream_failures >= SESSION_MAX_STREAM_RETRIES:
                    try:
                        await self._speak_guarded(
                            "I'm having trouble listening right now. "
                            "Say 'hello Diego' to try again.")
                    except Exception:
                        pass
                    break
                await asyncio.sleep(SESSION_RETRY_DELAY_S)
                continue
            pump = asyncio.create_task(self._stt_event_pump(stream, events))

            # ── ENDLESS SESSION (2026-09-21) ──
            # Mark the session active ONLY here (after the setup-error
            # paths, inside the try/finally reach): the finally block ends
            # the session state, so the LISTEN watchdog exemption can never
            # leak past a failed session attempt.
            self._begin_session()
            self._session_deadline = time.monotonic() + CONVERSATION_TIMEOUT_S
            self._set_state(EngineState.LISTEN)

            try:
                while self._running:
                    # ── ENDLESS SESSION robustness ──
                    # If the STT pump/stream dies, the session would idle
                    # deaf forever — it could not even hear a sleep command.
                    # Detect it and re-open the session instead.
                    if pump.done():
                        try:
                            exc = pump.exception()
                        except Exception:
                            exc = None  # cancelled pump — treat as clean exit
                        logger.error(
                            "[LISTEN] STT pump stopped (%s) — re-opening "
                            "the endless session",
                            repr(exc) if exc else "clean exit")
                        self._request_session_close(SESSION_CLOSE_ERROR)
                        break

                    remaining = self._session_deadline - time.monotonic()
                    if remaining <= 0:
                        if self._endless_session:
                            # ── ENDLESS SESSION (2026-09-21) ──
                            # Silence NEVER ends the session. Refresh the
                            # deadline and keep listening until an explicit
                            # SLEEP command.
                            logger.info(
                                "[LISTEN] Silence %.0fs — endless session "
                                "stays open (say a sleep command to end)",
                                CONVERSATION_TIMEOUT_S)
                            self._session_deadline = (
                                time.monotonic() + CONVERSATION_TIMEOUT_S)
                            continue
                        logger.info(
                            "[LISTEN] Silence %.0fs — conversation timeout",
                            CONVERSATION_TIMEOUT_S)
                        break

                    # Short poll so pump death / shutdown is noticed even
                    # while the silence deadline is still far away.
                    try:
                        ev = await asyncio.wait_for(
                            events.get(),
                            timeout=min(remaining, SESSION_POLL_S))
                    except asyncio.TimeoutError:
                        if time.monotonic() < self._session_deadline:
                            continue  # poll tick — re-check pump/shutdown
                        if self._endless_session:
                            logger.info(
                                "[LISTEN] Silence %.0fs — endless session "
                                "stays open (say a sleep command to end)",
                                CONVERSATION_TIMEOUT_S)
                            self._session_deadline = (
                                time.monotonic() + CONVERSATION_TIMEOUT_S)
                            continue
                        logger.info(
                            "[LISTEN] Silence %.0fs — conversation timeout",
                            CONVERSATION_TIMEOUT_S)
                        break

                    if ev.kind == "speech_start":
                        logger.info("Speech detected")
                        self._session_deadline = (
                            time.monotonic() + CONVERSATION_TIMEOUT_S)
                        continue

                    if ev.kind == "partial":
                        logger.info("Partial: '%s' (conf=%.3f, audio=%.0fms)",
                                    ev.text, ev.confidence, ev.audio_duration_ms)
                        continue

                    # ── TASK 6: explicit failure responses ──
                    # Diego must NEVER silently return to wake mode after a
                    # detected speech attempt. A "failure" event carries an
                    # explicit reason and must be spoken.
                    if ev.kind == "failure":
                        reason = (getattr(ev, "failure_reason", "")
                                  or "MISUNDERSTOOD")
                        logger.info("[LISTEN] Failure event (reason=%s, "
                                    "audio=%.0fms)", reason,
                                    ev.audio_duration_ms)
                        await self._speak_failure_response(reason)
                        # Failure turns must leave the listener re-armed too
                        # (endless session): gate open, stale audio drained.
                        self._finish_turn_rearm(events)
                        continue

                    if ev.kind != "final":
                        continue

                    text = (ev.text or "").strip()
                    if not text:
                        continue

                    dur_ms = ev.audio_duration_ms
                    logger.info("Endpoint (%.0fms, reason=%s)",
                                dur_ms, ev.endpoint_reason)
                    logger.info("Transcript: '%s' (conf=%.3f)",
                                text, ev.confidence)

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
                        logger.info("[LISTEN] Filler '%s' — turn stays open",
                                    text)
                        continue

                    # ── Self-introduction ──
                    if self._is_identity_question(lower):
                        self._set_state(EngineState.THINK)
                        intro = (
                            "I'm Diego, your desktop assistant. "
                            "I can control your computer, open apps, manage files, "
                            "play music, search the web, and help with coding. "
                            "Just say 'hello Diego' to wake me up, then tell me what you need."
                        )
                        # FIX (2026-09-21): the response leaves the engine in
                        # SPEAK state. The old code `continue`d without
                        # returning to LISTEN, so the SPEAK watchdog (120s)
                        # eventually force-stopped the LIVE command stream
                        # mid-session. Identity turns are just turns: speak,
                        # re-arm, back to LISTEN — the endless session stays.
                        try:
                            await self._think_and_speak(
                                intro, events, canned=True)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.error("[LISTEN] Identity turn failed: %s", e)
                        self._finish_turn_rearm(events)
                        self._set_state(EngineState.LISTEN,
                                        latency_breakdown="turn=identity")
                        continue

                    # ── SLEEP COMMAND (endless session exit, 2026-09-21) ──
                    # ONLY an explicit sleep command closes the session and
                    # returns Diego to wake mode. Silence, completed commands
                    # and TTS never do. Matching is whole-word based so
                    # "cancellation" or "sleepy" can never end the session.
                    if self._is_sleep_command(text):
                        self._request_session_close(SESSION_CLOSE_SLEEP)
                        self._set_state(EngineState.THINK)
                        await self._think_and_speak(
                            personality.farewell(), events, canned=True)
                        logger.info("[ENGINE] Sleep command '%s' — session "
                                    "over, returning to wake mode", text)
                        break

                    self._turn_count += 1
                    self._session_turns += 1
                    logger.info("[THINK] Turn #%d: '%s'",
                                self._turn_count, text)

                    t_turn_start = time.time()

                    # ── STATE: THINK — Brain orchestrates the full pipeline ──
                    # Brain.process_command() runs:
                    #   perceive → decide → plan → dispatch → verify → learn → respond
                    # The engine ONLY speaks the response. No bypass is possible.
                    self._set_state(EngineState.THINK)

                    # Pause listening during THINK so the command listener does
                    # not keep streaming and queue "speech_start" events from the
                    # user's continued speech (or TTS echo). Those stale events
                    # would otherwise be seen by the interruption monitor when
                    # TTS starts, causing an immediate false interrupt.
                    command_listener.pause_listening()

                    # ── ENDLESS SESSION (2026-09-21) ──
                    # The whole turn is exception-contained: a Brain crash,
                    # dispatcher failure or TTS explosion must NEVER end the
                    # session — Diego speaks a recovery response and keeps
                    # listening. Only an explicit sleep command may close it.
                    try:
                        from agent.brain import agent_brain

                        # ── RESPONSE GUARANTEE: never silent ──
                        # Wrap the full turn (process → speak) so that every
                        # completed utterance gets a spoken response. If the
                        # Brain fails, returns an empty response, or TTS fails,
                        # the guarantee layer speaks a recovery/generic fallback.
                        result_holder: dict = {}

                        async def _process() -> Any:
                            # BLOCKER 1 FIX (2026-08-30): forward the STT
                            # confidence + utterance audio duration so the Brain's
                            # intent sanity gate can combine transcript quality,
                            # speech evidence, and intent confidence before any
                            # tool execution.
                            r = await agent_brain.process_command(
                                text,
                                stt_confidence=getattr(ev, "confidence", None),
                                audio_duration_ms=getattr(
                                    ev, "audio_duration_ms", None),
                            )
                            result_holder["result"] = r
                            return r

                        async def _speak(response: str) -> bool:
                            spoke = await self._think_and_speak(
                                response, events, canned=True)
                            # If the action spoke immediately, speak the followup
                            # confirmation after verification completes.
                            r = result_holder.get("result")
                            if (spoke and r is not None
                                    and getattr(r, "speak_immediately", False)):
                                followup = (getattr(r, "followup_response", "")
                                            or "")
                                if followup and followup.strip():
                                    spoke2 = await self._think_and_speak(
                                        followup, events, canned=True)
                                    spoke = spoke or spoke2
                            return spoke

                        await response_guarantee.run_turn(
                            transcript=text,
                            process_fn=_process,
                            speak_fn=_speak,
                        )

                        result = result_holder.get("result")
                        # The turn completed cleanly — reset the bounded
                        # recovery counter of the endless session.
                        stream_failures = 0

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
                            total_latency_ms=(
                                time.time() - t_turn_start) * 1000,
                        )

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        # ── ENDLESS SESSION (2026-09-21) ──
                        # A turn error must NEVER end the session: speak a
                        # short recovery response and keep listening. Only
                        # an explicit sleep command may close the session.
                        logger.exception(
                            "[LISTEN] Turn failed — endless session stays "
                            "open: %s", e)
                        try:
                            # Recovery speech is a SPEAK phase.  The normal
                            # turn path reaches it through THINK → SPEAK, so
                            # do the same before guarded failure speech;
                            # otherwise the final SPEAK → LISTEN recovery is
                            # an invalid THINK → LISTEN transition.
                            self._set_state(EngineState.SPEAK)
                            await self._speak_failure_response(
                                "MISUNDERSTOOD")
                        except Exception as speak_err:
                            logger.error(
                                "[LISTEN] Failure response also failed: %s",
                                speak_err)
                    finally:
                        # ── LISTENING GUARD FIX (2026-08-30) ──
                        # The TTS guard pauses the command listener for the
                        # whole Brain turn. If ANY path left the listener
                        # paused (e.g. the Brain turn completed without TTS,
                        # or TTS failed), the microphone would never accept
                        # the NEXT real command.
                        # ── ENDLESS SESSION (2026-09-21) ──
                        # _finish_turn_rearm() combines the resume with an
                        # eager drain + VAD reset + stale-event purge so NO
                        # stale drain/gate/VAD state survives the THINK/SPEAK
                        # boundary between turns (the stream keeps running
                        # across turns, so the drain it requests is consumed
                        # by the live streaming loop). Runs on EVERY exit
                        # path — success, failure, or cancellation.
                        self._finish_turn_rearm(events)

                        # ── Back to LISTEN ──
                        self._set_state(
                            EngineState.LISTEN,
                            latency_breakdown=(
                                f"turn={(time.time() - t_turn_start) * 1000:.0f}ms"))

                # ── ENDLESS SESSION: stream teardown + bounded recovery ──
                # Reaching here means the inner loop broke: an explicit
                # sleep command, shutdown, or an internal failure.
                if self._session_close_reason == SESSION_CLOSE_SLEEP:
                    # Explicit sleep → wake mode (the ONLY normal close).
                    break
                if not self._running:
                    break
                if self._session_close_reason == SESSION_CLOSE_STT_UNAVAILABLE:
                    break

                # Internal failure → re-open the session (the user must
                # NEVER have to repeat the wake word). Bounded: after
                # SESSION_MAX_STREAM_RETRIES consecutive failures, give up
                # and return to wake mode with one spoken explanation.
                stream_failures += 1
                logger.error(
                    "[LISTEN] Session failed internally (reason=%s, %d/%d) "
                    "— %s", self._session_close_reason or "unknown",
                    stream_failures, SESSION_MAX_STREAM_RETRIES,
                    "re-opening the endless session"
                    if stream_failures < SESSION_MAX_STREAM_RETRIES
                    else "giving up — returning to wake mode")
                if stream_failures >= SESSION_MAX_STREAM_RETRIES:
                    try:
                        await self._speak_guarded(
                            "I'm having trouble listening right now. "
                            "Say 'hello Diego' to try again.")
                    except Exception:
                        pass
                    break
                await asyncio.sleep(SESSION_RETRY_DELAY_S)

            finally:
                # Endless session stream teardown. ALWAYS executed: sleep
                # close, shutdown, internal errors, and cancellations.
                pump.cancel()
                command_listener.stop_streaming()
                await asyncio.gather(pump, return_exceptions=True)
                try:
                    await stream.aclose()
                except Exception:
                    pass
                # Session closed — end the session state so the LISTEN
                # watchdog ceiling is enforced again outside a session and
                # silence semantics return to normal.
                self._end_session()
                logger.info(
                    "[STT] Streaming Whisper stopped (session over: %s)",
                    self._session_close_reason or "unknown")

        # ── ENDLESS SESSION post-loop bookkeeping ─────────────
        # Reset the close reason and return the session machine to IDLE.
        # The run() loop routes back to wake mode (IDLE → WAKE) because
        # this method only ever returns after a sleep close, shutdown, a
        # recognizer failure, or an unrecoverable stream failure — never
        # after mere silence or a completed command.
        logger.info("[SESSION] Endless conversation session ended (%s) — "
                    "returning to wake mode",
                    self._session_close_reason or "shutdown")
        self._session_close_reason = None
        self._set_state(EngineState.IDLE,
                        session_duration=f"{(time.time() - t_session_start):.1f}s",
                        turns=self._turn_count)
        print("\n  Listening for wake word...\n")

    async def _stt_event_pump(
        self, stream: AsyncIterator[UtteranceEvent] | Any,
        events: "asyncio.Queue[UtteranceEvent]",
    ) -> None:
        try:
            # Some command-listener implementations expose streaming as an
            # async generator, while older/test implementations return a
            # single event from an awaitable. Normalize both at this boundary
            # so the session loop has one event-stream contract.
            source = await stream if inspect.isawaitable(stream) else stream
            if hasattr(source, "__aiter__"):
                async for ev in source:
                    if not self._running:
                        break
                    events.put_nowait(ev)
            elif isinstance(source, UtteranceEvent):
                if self._running:
                    events.put_nowait(source)
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
        # ── DUPLICATE TTS GUARD (2026-09-03) ──
        # The runtime log showed "TTS end" immediately followed by another
        # "TTS start" for the same turn. This happens when the immediate
        # response and the followup confirmation are the same text (e.g.
        # "Opening Firefox." spoken twice). Skip speaking the same text
        # twice in a row.
        text_key = (user_text or "").strip().lower()
        if (hasattr(self, "_last_spoken_text")
                and self._last_spoken_text == text_key
                and text_key):
            logger.info("[SPEAK] Duplicate TTS text skipped: '%s'", user_text[:50])
            return True  # Already spoken — treat as success

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

        # Drain stale events before TTS so the interruption monitor never
        # mistakes a previous turn's speech (or TTS echo) for a new
        # interruption. This breaks the self-listening loop where Leo
        # transcribes its own voice and responds forever.
        while True:
            try:
                events.get_nowait()
            except asyncio.QueueEmpty:
                break

        monitor = asyncio.create_task(self._watch_interruption(events))
        command_listener.pause_listening()
        try:
            # CRITICAL FIX (2026-08-29): TTS can hang (e.g. Kokoro model
            # loading). Add a hard timeout so the SPEAK state watchdog can
            # recover. 120s matches the SPEAK state ceiling.
            played = await asyncio.wait_for(
                streaming_tts.speak_sentences(sentences(), interrupt),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            logger.error("[SPEAK] TTS timed out after 120s — recovering")
            streaming_tts.stop()
            played = False
        finally:
            # Let the TTS echo decay before resuming, so Diego never
            # transcribes its own voice as a user command. 0.5s covers
            # laptop-speaker reverberation in a normal room.
            await asyncio.sleep(0.5)
            command_listener.resume_listening()
            monitor.cancel()
            if not producer.done():
                interrupt.set()
            await asyncio.gather(producer, monitor, return_exceptions=True)
        logger.info("TTS end (%.0fms)", (time.time() - t_start) * 1000)
        # Record the last spoken text for duplicate-TTS prevention.
        self._last_spoken_text = text_key
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

        ABBREVIATION-AWARE SPLITTING (NEW):
          A naive `re.split(r'(?<=[.!?])\\\\s+')` splits mid-word after
          abbreviations ("vs.", "Mr.", "Dr.", "e.g.", "U.S.", "etc.")
          producing choppy, unnatural speech. This version only treats
          `.` as a sentence end when it is NOT followed by a lowercase
          letter — the classic heuristic for "is this an abbreviation
          or a sentence terminator?".
        """
        import re as _re
        text = (text or "").strip()
        if not text:
            return

        # Short responses — yield whole for instant playback
        if len(text) < 40:
            yield text
            return

        # Abbreviation-aware split:
        #   (?<=[.!?])  → lookbehind for sentence ender
        #   \s+         → whitespace after it
        #   (?!        ) → negative lookahead: don't split if the next
        #                 char is a lowercase letter/digit (abbreviation:
        #                 "Mr. Smith", "e.g. this", "v2.5", "U.S.")
        sentences = _re.split(
            r'(?<=[.!?])\s+(?![a-z0-9])', text)
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

    async def _speak_guarded(self, text: str) -> bool:
        """Speak while muting STT so Diego NEVER transcribes its own voice.

        Pause the command listener, speak, let the speaker echo decay, then
        drain the ring buffer and resume. Every TTS path that runs outside
        the normal _think_and_speak flow (greeting, failure responses) MUST
        use this so the microphone never picks up Diego's own output through
        the laptop speakers (no headphones required).
        """
        command_listener.pause_listening()
        try:
            spoke = await self._speak_line(text)
        finally:
            # Laptop-speaker echo decays quickly; give it time before we
            # re-arm the mic, then discard everything captured while muted.
            await asyncio.sleep(0.4)
            command_listener.resume_listening()
            audio_manager.read_since(audio_manager.total_samples)
        return spoke

    # ── TASK 6: explicit failure responses ─────────────────
    # Diego must NEVER silently return to wake mode after a detected speech
    # attempt. Each failure reason maps to a short spoken response.

    FAILURE_RESPONSES = {
        "MISUNDERSTOOD": "I didn't catch that. Could you say that again?",
        "LOW_CONFIDENCE": "I'm not sure I heard you. Could you say that again?",
        "TRANSCRIPTION_FAILED": "Sorry, I missed that. Could you say that again?",
        "TIMEOUT": "I didn't hear anything. Could you say that again?",
        "GARBAGE": "I didn't catch that. Could you say that again?",
    }

    async def _speak_failure_response(self, reason: str) -> bool:
        """Speak a short response for a failed speech attempt.

        Uses the guarded path so the failure response is never transcribed
        as a new command (it would otherwise echo forever).

        Returns True if audio was queued for playback.
        """
        text = self.FAILURE_RESPONSES.get(
            reason, self.FAILURE_RESPONSES["MISUNDERSTOOD"])
        logger.info("[LISTEN] Speaking failure response: '%s' (reason=%s)",
                    text, reason)
        return await self._speak_guarded(text)

    @staticmethod
    def _is_identity_question(text: str) -> bool:
        """Detect questions about Diego's identity."""
        identity_patterns = [
            "who are you", "what are you", "what is your name",
            "who is diego", "what is diego", "tell me about yourself",
            "introduce yourself", "what do you do", "what can you do",
            "who am i talking to", "what's your name", "whats your name",
        ]
        return any(p in text for p in identity_patterns)

    def get_diagnostics(self) -> dict:
        return {
            "state": self._state.value if self._state else "none",
            "state_duration_s": round(time.monotonic() - self._state_entered, 1),
            "turn_count": self._turn_count,
            "auth_user": self._auth_user,
            # Independent wake/auth status markers:
            #   wake: BYPASSED (--no-wake), DEGRADED (model unavailable),
            #         or READY
            #   auth: DISABLED only via explicit --no-auth, else ACTIVE
            "wake": ("BYPASSED" if self._no_wake
                     else "READY" if self._wake_active
                     else "DEGRADED"),
            "auth": "DISABLED" if self._auth_provider is None else "ACTIVE",
            "running": self._running,
            "last_diag": self._diag,
            "response_guarantee": response_guarantee.get_diagnostics(),
            # ── ENDLESS SESSION (2026-09-21) ──
            # Session lifecycle for diagnostics/UI/tests: the session must
            # be ACTIVE (or CLOSING) for the whole wake→auth→LISTEN phase
            # and only IDLE again in wake mode.
            "session": self.get_session_diagnostics(),
        }

    def get_session_diagnostics(self) -> dict:
        """ENDLESS conversation-session lifecycle snapshot (2026-09-21).

        Exposed for tests, the UI and logs so the invariant
        "silence/completed commands never close the session; only an
        explicit sleep command does" is observable at runtime.
        """
        return {
            "state": self._session_state.value,
            "id": self._session_id,
            "turns": self._session_turns,
            "started_at": self._session_started_at,
            "duration_s": round(
                time.monotonic() - self._session_started_at, 1)
            if self._session_state == SessionState.ACTIVE else 0.0,
            "close_reason": self._session_close_reason,
        }


# Global singleton
conversation_engine = ConversationEngine()