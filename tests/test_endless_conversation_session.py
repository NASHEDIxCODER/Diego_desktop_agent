"""
ENDLESS CONVERSATION SESSION tests (2026-09-21).

The invariant under test:

    After wake + face authentication, Diego stays in
    LISTEN → THINK → SPEAK → LISTEN continuously. The session is NOT
    closed by silence, timeouts, completed commands, TTS, identity
    questions, failure events, or Brain crashes. It closes ONLY on an
    explicit sleep command ("sleep", "go to sleep", "stop listening",
    "good night", …), after which Diego cleanly returns to wake mode
    (IDLE → WAKE).

Also covered:
  * the session-state machine (SessionState + close reasons),
  * between-turns re-arm (no stale drain/gate/VAD state),
  * self-recovery: a dead STT pump or stream error re-opens the session
    without another wake word,
  * the LISTEN watchdog exemption while the session is active,
  * the preserved post-wake prepare_command_session() fix.

These tests use fakes; no microphone, camera or GPU is required.
"""

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import numpy as np
import pytest

import core.conversation_engine as CE
from core.conversation_engine import (
    ALLOWED_TRANSITIONS,
    EngineState,
    SLEEP_PHRASES,
    SESSION_CLOSE_ERROR,
    SESSION_CLOSE_SLEEP,
    SessionState,
    UtteranceEvent,
)
import agent.brain as brain_module
import voice.command_listener as CL


# =================================================================
# Fakes (kept local to avoid coupling to other test modules)
# =================================================================

class FakeTTS:
    """Records everything Diego 'speaks'."""

    def __init__(self):
        self.spoken = []
        self.calls = 0
        self.output_device = None

    async def speak_sentences(self, sentences, interrupt=None):
        self.calls += 1
        chunks = []
        async for piece in sentences:
            if interrupt is not None and interrupt.is_set():
                break
            chunks.append(piece)
        text = " ".join(chunks)
        self.spoken.append(text)
        return bool(text)

    def stop(self):
        pass


class FakeBrainResult:
    def __init__(self, response="Done."):
        self.response = response
        self.path = "BRAIN"
        self.used_llm = False
        self.latency_ms = 1.0
        self.actions_executed = 0
        self.actions_failed = 0
        self.verified = True
        self.speak_immediately = False
        self.followup_response = ""


class FakeBrain:
    def __init__(self):
        self.commands = []

    async def process_command(self, text, stt_confidence=None,
                              audio_duration_ms=None):
        self.commands.append(text)
        await asyncio.sleep(0)
        return FakeBrainResult(response=f"Okay: {text}")


class FakeCommandListener:
    """Queue-backed fake of the streaming command listener.

    The engine talks to the module-level `command_listener` object; the
    fixture patches `CE.command_listener` with this fake. The stream
    yields whatever is pushed into `inbox` (None = end of stream).
    """

    def __init__(self, ready=True):
        self._ready = ready
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._listen_enabled = asyncio.Event()
        self._listen_enabled.set()
        self._drain_requested = False
        self._gate_closed_at = None
        self.prepare_calls = 0
        self.rearm_calls = 0
        self.stop_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.stream_open = False

    @property
    def ready(self):
        return self._ready

    async def initialize(self):
        self._ready = True
        return True

    def prepare_command_session(self):
        self.prepare_calls += 1
        self._drain_requested = False
        self._gate_closed_at = None
        self._listen_enabled.set()

    def stream_utterances(self):
        return self._gen()

    async def _gen(self):
        self.stream_open = True
        try:
            while True:
                ev = await self.inbox.get()
                if ev is None:
                    return
                yield ev
        finally:
            self.stream_open = False

    def pause_listening(self):
        self.pause_calls += 1
        self._listen_enabled.clear()
        if self._gate_closed_at is None:
            self._gate_closed_at = time.monotonic()

    def resume_listening(self):
        self.resume_calls += 1
        self._gate_closed_at = None
        self._drain_requested = True
        self._listen_enabled.set()

    def rearm_between_turns(self):
        self.rearm_calls += 1
        self._gate_closed_at = None
        self._drain_requested = True
        self._listen_enabled.set()

    def stop_streaming(self):
        self.stop_calls += 1

    def push_command(self, text, confidence=-0.1):
        self.inbox.put_nowait(UtteranceEvent(
            kind="final", text=text, is_final=True,
            confidence=confidence, audio_duration_ms=1200.0,
            endpoint_reason="silence"))

    def push_failure(self, reason="MISUNDERSTOOD"):
        self.inbox.put_nowait(UtteranceEvent(
            kind="failure", is_final=True, failure_reason=reason,
            audio_duration_ms=1500.0, endpoint_reason="silence"))

    def end_stream(self):
        """Make the STT pump exit (simulates a dead stream)."""
        self.inbox.put_nowait(None)


class FakeAudioManager:
    def __init__(self):
        self._total = 0
        self.is_running = True

    @property
    def total_samples(self):
        return self._total

    def read_since(self, last_total):
        return np.zeros(0, dtype=np.float32), self._total


class FakeVAD:
    """Minimal unified_vad stand-in for the real-listener unit tests."""

    def __init__(self):
        self.reset_calls = 0

    def reset_state(self):
        self.reset_calls += 1


async def wait_for(pred, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(interval)
    return False


async def say(listener, brain, engine, text, timeout=3.0):
    """Push one command and wait until the turn is fully done."""
    listener.push_command(text)
    assert await wait_for(lambda: text in brain.commands, timeout), \
        f"command '{text}' was never processed by the Brain"
    assert await wait_for(
        lambda: engine._state == EngineState.LISTEN, timeout), \
        f"engine did not return to LISTEN after '{text}'"


@pytest.fixture
def endless_engine(monkeypatch):
    """A ConversationEngine wired to fakes with fast session timeouts."""
    listener = FakeCommandListener()
    tts = FakeTTS()
    monkeypatch.setattr(CE, "command_listener", listener)
    monkeypatch.setattr(CE, "streaming_tts", tts)
    monkeypatch.setattr(CE, "audio_manager", FakeAudioManager())
    # Fast silence budget + fast poll so the tests exercise the deadline
    # refresh path quickly.
    monkeypatch.setattr(CE, "CONVERSATION_TIMEOUT_S", 0.25)
    monkeypatch.setattr(CE, "SESSION_POLL_S", 0.05)
    brain = FakeBrain()
    monkeypatch.setattr(brain_module, "agent_brain", brain)

    engine = CE.ConversationEngine()
    engine._running = True
    engine._set_state(EngineState.IDLE)
    engine._auth_provider = None
    return engine, listener, tts, brain


def start_session(engine):
    """Start the endless session exactly like run() does after wake."""
    engine._set_state(EngineState.WAKE)
    engine._set_state(EngineState.LISTEN)
    return asyncio.create_task(engine._conversation_session())


async def close_session_with_sleep(listener, engine, task,
                                   phrase="go to sleep"):
    listener.push_command(phrase)
    assert await wait_for(
        lambda: engine._session_close_reason == SESSION_CLOSE_SLEEP), \
        "session was not closed by the explicit sleep command"
    await asyncio.wait_for(task, timeout=5.0)


# =================================================================
# 1. The real desktop sequence: wake → auth → 5+ commands →
#    silence periods → sleep → wake mode
# =================================================================

def test_desktop_sequence_wake_auth_five_commands_silence_sleep_wake(
        endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        # ── WAKE ──
        engine._set_state(EngineState.IDLE)
        engine._set_state(EngineState.WAKE)
        assert engine._state == EngineState.WAKE

        # ── FACE AUTH (fake provider; same gate code path as the camera) ──
        auth_calls = []

        async def fake_provider():
            auth_calls.append(True)
            return "Nashedi"

        engine._auth_provider = fake_provider
        engine._auth_user = None
        engine._last_auth_time = 0.0
        await engine._face_auth_gate(trigger="wake")
        assert auth_calls, "face auth gate did not run the provider"
        assert engine._auth_user == "Nashedi"
        assert "Welcome back, Nashedi!" in tts.spoken

        # ── ENDLESS SESSION: 5 commands with silence gaps between them ──
        task = start_session(engine)
        commands = ["open firefox", "what time is it", "play some music",
                    "search for pandas", "tell me a joke"]
        for i, cmd in enumerate(commands):
            await say(listener, brain, engine, cmd)
            # Silence period LONGER than the (patched) conversation
            # timeout — this must NEVER end the session. Read the budget off
            # the ENGINE MODULE (not this module's import binding) so the
            # fixture's fast timeout applies: the bare imported name sticks
            # at the real 60 s and turned this test into a 10-minute sleep.
            await asyncio.sleep(CE.CONVERSATION_TIMEOUT_S * 2)
            assert engine._session_state == SessionState.ACTIVE, \
                f"silence ended the session after command {i + 1}!"
            assert engine._state == EngineState.LISTEN
        assert len(brain.commands) == 5

        # Every command ran inside ONE session (no wake re-entry).
        assert listener.prepare_calls == 1

        # ── SLEEP: the ONLY thing that may close the session ──
        assert engine._turn_count == 5
        await close_session_with_sleep(listener, engine, task)

        assert engine._session_close_reason is None  # consumed on exit
        assert engine._session_state == SessionState.CLOSED
        assert engine._state == EngineState.IDLE
        # The sleep command must NOT reach the Brain.
        assert len(brain.commands) == 5
        # A farewell was spoken.
        assert tts.spoken
        assert tts.spoken[-1]

        # ── WAKE MODE again: run() would now go IDLE → WAKE ──
        assert EngineState.WAKE in ALLOWED_TRANSITIONS[EngineState.IDLE]
        engine._set_state(EngineState.WAKE)
        assert engine._state == EngineState.WAKE

        # A second wake+session cycle works identically.
        engine._set_state(EngineState.IDLE)
        task2 = start_session(engine)
        await say(listener, brain, engine, "open notepad")
        assert engine._session_id == 2  # fresh session after sleep
        await close_session_with_sleep(listener, engine, task2, "sleep")
        assert engine._state == EngineState.IDLE

    asyncio.run(seq())


# =================================================================
# 2. Multiple commands without another wake word
# =================================================================

def test_eight_commands_without_another_wake_word(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        for cmd in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            await say(listener, brain, engine, cmd)
            assert engine._session_state == SessionState.ACTIVE
        assert len(brain.commands) == 8
        # ONE session entry, ONE stream — the wake listener was never
        # re-entered between commands.
        assert listener.prepare_calls == 1
        assert listener.stop_calls == 0
        # Between-turns re-arm ran after every turn.
        assert listener.rearm_calls >= 8
        await close_session_with_sleep(listener, engine, task)

    asyncio.run(seq())


# =================================================================
# 3. Silence never ends the session
# =================================================================

def test_silence_periods_keep_session_open(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        deadline_before = engine._session_deadline
        # Sit through 5 consecutive silence timeouts (> 5x budget). Budget is
        # read from the engine module so the fixture's fast timeout applies.
        await asyncio.sleep(CE.CONVERSATION_TIMEOUT_S * 5)
        assert engine._session_state == SessionState.ACTIVE
        assert engine._state == EngineState.LISTEN
        assert engine._session_deadline > deadline_before, \
            "silence deadline was never refreshed"
        # The user can still be heard after all that silence.
        await say(listener, brain, engine, "are you still there")
        assert engine._session_state == SessionState.ACTIVE
        await close_session_with_sleep(listener, engine, task)
        assert engine._state == EngineState.IDLE

    asyncio.run(seq())


# =================================================================
# 4. Failure events / identity questions / Brain crashes
# =================================================================

def test_failure_event_does_not_end_session(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        listener.push_failure("MISUNDERSTOOD")
        assert await wait_for(lambda: tts.spoken), "failure response not spoken"
        assert await wait_for(lambda: engine._state == EngineState.LISTEN)
        assert engine._session_state == SessionState.ACTIVE
        await say(listener, brain, engine, "open firefox")
        assert engine._session_state == SessionState.ACTIVE
        await close_session_with_sleep(listener, engine, task)

    asyncio.run(seq())


def test_identity_question_returns_to_listen_and_stays_open(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        listener.push_command("who are you")
        assert await wait_for(
            lambda: any("I'm Diego" in s for s in tts.spoken))
        # REGRESSION GUARD: the identity turn must leave the engine back
        # in LISTEN (it used to stay in SPEAK, which let the SPEAK
        # watchdog force-stop the live command stream mid-session).
        assert await wait_for(lambda: engine._state == EngineState.LISTEN)
        assert engine._session_state == SessionState.ACTIVE
        # The next command still works.
        await say(listener, brain, engine, "open firefox")
        await close_session_with_sleep(listener, engine, task)

    asyncio.run(seq())


def test_turn_exception_does_not_end_session(endless_engine, monkeypatch):
    engine, listener, tts, brain = endless_engine

    class ExplodingGuarantee:
        async def run_turn(self, transcript, process_fn, speak_fn):
            raise RuntimeError("pipeline exploded")
        def get_diagnostics(self):
            return {}

    monkeypatch.setattr(CE, "response_guarantee", ExplodingGuarantee())

    async def seq():
        task = start_session(engine)
        listener.push_command("open firefox")
        # The engine speaks its own recovery response and KEEPS listening.
        assert await wait_for(lambda: tts.spoken)
        assert await wait_for(lambda: engine._state == EngineState.LISTEN)
        assert engine._session_state == SessionState.ACTIVE
        await say(listener, brain, engine, "open chrome")
        assert engine._session_state == SessionState.ACTIVE
        await close_session_with_sleep(listener, engine, task)

    asyncio.run(seq())


# =================================================================
# 5. Self-recovery: dead pump / stream errors re-open the session
# =================================================================

def test_dead_pump_reopens_session_without_wake_word(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        listener.end_stream()  # STT stream dies
        # The session re-opens itself (2nd attempt) — no wake word needed.
        assert await wait_for(lambda: listener.prepare_calls >= 2), \
            "session did not re-open after the STT pump died"
        assert await wait_for(lambda: engine._session_id >= 2)
        assert engine._state == EngineState.LISTEN
        # The next command is captured by the NEW stream.
        await say(listener, brain, engine, "open firefox")
        assert engine._session_state == SessionState.ACTIVE
        await close_session_with_sleep(listener, engine, task)
        assert engine._state == EngineState.IDLE

    asyncio.run(seq())


def test_persistent_stream_failures_return_to_wake_mode(endless_engine):
    engine, listener, tts, brain = endless_engine

    attempts = {"n": 0}

    async def broken_stream():
        attempts["n"] += 1
        raise ValueError("empty frames")

    listener.stream_utterances = broken_stream

    async def seq():
        task = start_session(engine)
        # Bounded: SESSION_MAX_STREAM_RETRIES attempts, then wake mode.
        await asyncio.wait_for(task, timeout=10.0)
        assert attempts["n"] == CE.SESSION_MAX_STREAM_RETRIES
        assert engine._state == EngineState.IDLE
        assert any("trouble listening" in s for s in tts.spoken), \
            "user was never told about the persistent listener failure"

    asyncio.run(seq())


def test_shutdown_during_session_closes_cleanly(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        await asyncio.sleep(0.1)
        engine._running = False  # Ctrl+C / stop()
        await asyncio.wait_for(task, timeout=5.0)
        assert engine._state == EngineState.IDLE
        assert engine._session_state == SessionState.CLOSED
        assert listener.stop_calls >= 1, "stream was never torn down"

    asyncio.run(seq())


# =================================================================
# 6. Only an explicit sleep command ends the session
# =================================================================

@pytest.mark.parametrize("phrase", [
    "sleep", "diego sleep", "go to sleep", "go to sleep now",
    "stop listening", "good night", "goodnight", "end the session",
    "that's all", "im done", "cancel", "stop the session",
])
def test_sleep_matcher_closes_on_explicit_phrases(phrase):
    from core.conversation_engine import ConversationEngine
    assert ConversationEngine._is_sleep_command(phrase)


@pytest.mark.parametrize("phrase", [
    "sleep music", "sleep paralysis", "asleep at the wheel", "sleepy",
    "cancellation of my order", "open the sleep settings",
    "how do I sleep better", "open firefox", "what time is it",
])
def test_sleep_matcher_does_not_close_on_lookalikes(phrase):
    from core.conversation_engine import ConversationEngine
    assert not ConversationEngine._is_sleep_command(phrase)


def test_lookalike_phrases_do_not_close_a_live_session(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        for phrase in ["sleep music", "cancellation of my order",
                       "am I sleepy", "open the sleep settings"]:
            await say(listener, brain, engine, phrase)
            assert engine._session_state == SessionState.ACTIVE, \
                f"'{phrase}' wrongly ended the session!"
        # Now the explicit sleep command closes it.
        await close_session_with_sleep(listener, engine, task,
                                       phrase="stop listening")
        assert engine._state == EngineState.IDLE
        # "stop listening" is a sleep command → never reached the Brain.
        assert "stop listening" not in brain.commands
        assert all(p not in brain.commands for p in
                   ["sleep music", "cancellation of my order",
                    "am I sleepy", "open the sleep settings"])

    asyncio.run(seq())


# =================================================================
# 7. Between-turns re-arm: no stale drain/gate state
# =================================================================

def test_engine_rearms_between_turns(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        for _ in range(3):
            await say(listener, brain, engine, "next command")
            # Gate must be OPEN and no stale gate-hold timestamp survives.
            assert listener._listen_enabled.is_set()
            assert listener._gate_closed_at is None
            # Stale events pushed mid-turn are drained, not replayed.
            listener.inbox.put_nowait(UtteranceEvent(
                kind="speech_start", is_final=False))
        await close_session_with_sleep(listener, engine, task)

    asyncio.run(seq())


def test_command_listener_rearm_clears_stale_gate_and_drain(monkeypatch):
    """Unit test on the REAL CommandListener (mirrors the prepare test in
    test_post_wake_command_capture.py, but for the between-turns path)."""
    vad = FakeVAD()
    monkeypatch.setattr(CL, "unified_vad", vad)

    cl = CL.CommandListener()
    # Emulate a completed turn: the TTS guard paused the listener and the
    # Brain turn left a gate-hold timestamp behind.
    cl.pause_listening()
    assert cl._listen_enabled.is_set() is False
    assert cl._gate_closed_at is not None
    assert vad.reset_calls == 0

    # The engine calls this after EVERY turn of the endless session.
    cl.rearm_between_turns()

    assert cl._gate_closed_at is None
    assert cl._drain_requested is True          # consumed by the live loop
    assert cl._listen_enabled.is_set() is True  # gate OPEN
    assert vad.reset_calls == 1                 # VAD reset eagerly


# =================================================================
# 8. Watchdog: LISTEN exempt while the session is active
# =================================================================

def test_watchdog_exempts_listen_during_endless_session(endless_engine,
                                                        monkeypatch):
    engine, listener, tts, brain = endless_engine
    monkeypatch.setattr(CE, "STATE_WATCHDOG_INTERVAL_S", 0.02)
    monkeypatch.setattr(CE, "STATE_TIMEOUTS_S",
                        {"LISTEN": 0.05, "THINK": 0.05, "SPEAK": 0.05})

    async def seq():
        task = start_session(engine)
        engine._state_entered = time.monotonic() - 999  # far past ceiling

        watchdog = asyncio.create_task(engine._state_watchdog())
        await asyncio.sleep(0.2)
        # Session ACTIVE: the long-idle LISTEN state must NOT be kicked.
        assert engine._session_state == SessionState.ACTIVE
        assert listener.stop_calls == 0, \
            "watchdog force-stopped the live stream during the session"

        # Session CLOSED: the same stale LISTEN state IS recovered.
        engine._session_state = SessionState.CLOSED
        assert await wait_for(lambda: listener.stop_calls >= 1, timeout=2.0)
        engine._running = False
        await asyncio.gather(watchdog, return_exceptions=True)
        await close_session_with_sleep(listener, engine, task)

    asyncio.run(seq())


# =================================================================
# 9. Session state machine + diagnostics + preserved fixes
# =================================================================

def test_session_state_machine_lifecycle(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        assert engine._session_state == SessionState.IDLE
        assert engine._endless_session is False
        task = start_session(engine)
        await say(listener, brain, engine, "open firefox")
        assert engine._session_state == SessionState.ACTIVE
        assert engine._endless_session is True
        listener.push_command("go to sleep")
        # CLOSING while the farewell is being spoken.
        assert await wait_for(
            lambda: engine._session_state == SessionState.CLOSING
            or engine._session_state == SessionState.CLOSED)
        await asyncio.wait_for(task, timeout=5.0)
        assert engine._session_state == SessionState.CLOSED
        assert engine._endless_session is False

    asyncio.run(seq())


def test_session_diagnostics_exposed(endless_engine):
    engine, listener, tts, brain = endless_engine

    async def seq():
        task = start_session(engine)
        await say(listener, brain, engine, "open firefox")
        d = engine.get_diagnostics()
        s = d["session"]
        assert s["state"] == "ACTIVE"
        assert s["id"] == 1
        assert s["turns"] == 1
        assert s["close_reason"] is None
        sd = engine.get_session_diagnostics()
        assert sd["state"] == "ACTIVE"
        await close_session_with_sleep(listener, engine, task)
        d = engine.get_diagnostics()
        assert d["session"]["state"] == "CLOSED"
        assert engine._session_close_reason is None  # consumed on exit

    asyncio.run(seq())


def test_post_wake_prepare_command_session_fix_preserved():
    """Static contract: LISTEN entry still prepares the command listener
    (the 2026-09-20 post-wake capture fix must survive the endless
    session rewrite)."""
    import inspect
    src = inspect.getsource(CE.ConversationEngine._conversation_session)
    assert "prepare_command_session()" in src, \
        "LISTEN entry must prepare the command listener"
    assert "rearm_between_turns" in \
        inspect.getsource(CE.ConversationEngine._finish_turn_rearm), \
        "between-turns re-arm must release the TTS guard state"
    # SLEEP_PHRASES remain the canonical close set (alias preserved).
    assert "go to sleep" in SLEEP_PHRASES
    assert "stop listening" in SLEEP_PHRASES
    assert "good night" in SLEEP_PHRASES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
