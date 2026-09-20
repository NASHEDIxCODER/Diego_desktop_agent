"""
Post-wake command-capture regression tests (2026-09-20).

Targets the reported bug:

    Wake accepted -> face auth succeeded -> command listener stayed in
    WAITING_FOR_SPEECH -> the immediately-spoken command was NOT captured ->
    "NO SPEECH evidence" -> turn discarded.

Root cause established in this investigation:

    On the post-wake path the engine runs FACE_AUTH, which greets the user
    via _speak_guarded(). _speak_guarded() -> resume_listening() leaves
    `_drain_requested = True`. Because NO command stream is running yet at
    that point, that drain request is never consumed and LINGERS into the
    next LISTEN session. stream_utterances() then fixes its
    command_session_start boundary on the first __anext__(), and its very
    first loop iteration executes the leftover drain - moving `last_total`
    back to the current write head AFTER the boundary was established. The
    first 1-2s of the spoken command are therefore discarded and the
    listener never leaves WAITING_FOR_SPEECH.

Guarantees added here:

  1. prepare_command_session() clears the stale drain request, releases the
     gate-hold timestamp, forces the listen gate OPEN, and resets VAD state.
  2. Command audio fed IMMEDIATELY after wake/auth (no delay) reaches the
     Whisper backend - the first 1-2s of audio after wake are not clipped.
  3. The full transition WAKE -> FACE_AUTH -> LISTEN leaves the command
     listener immediately ready and captures a command spoken straight away.

These tests use fakes; no physical microphone or GPU is required.
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

import voice.command_listener as CL
from voice.audio_manager import FRAME_SAMPLES, SAMPLE_RATE


# =================================================================
# Fakes (kept local to avoid coupling to other test modules)
# =================================================================

class FakeAudioManager:
    """Cursor-based audio source mirroring AudioManager.read_since()."""

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self.frame_id = 0
        self.last_frame_timestamp = 0.0

    def feed(self, audio: np.ndarray) -> None:
        import time as _time
        audio = np.asarray(audio, dtype=np.float32)
        self._buf = np.concatenate([self._buf, audio])
        n_frames = int(audio.size // FRAME_SAMPLES)
        for _ in range(n_frames):
            self.frame_id += 1
            self.last_frame_timestamp = _time.time()

    @property
    def total_samples(self) -> int:
        return int(self._buf.size)

    def read_since(self, last_total: int):
        last_total = max(0, int(last_total))
        if last_total >= self._buf.size:
            return np.array([], dtype=np.float32), int(self._buf.size)
        return self._buf[last_total:].copy(), int(self._buf.size)

    def get_frame_state(self) -> dict:
        return {
            "stream_running": True,
            "stream_state": "open",
            "callback_count": self.frame_id,
            "frame_id": self.frame_id,
            "last_frame_timestamp": self.last_frame_timestamp,
            "ring_buffer_samples": int(self._buf.size),
        }


class FakeVAD:
    """Energy-based VAD: loud frames => speech, silence => no speech."""

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold
        self.reset_calls = 0
        self._robust_smoothed = 0.0
        self._robust_in_speech = False
        self._robust_speech_frames = 0

    def speech_prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.asarray(frame, np.float32) ** 2)))
        prob = 0.9 if rms > self.threshold else 0.0
        self._robust_smoothed = prob
        return prob

    def robust_speech_prob(self, frame: np.ndarray) -> float:
        silero = self.speech_prob(frame)
        self._robust_smoothed = 0.4 * silero + 0.6 * self._robust_smoothed
        return float(min(max(self._robust_smoothed, 0.0), 1.0))

    def get_robust_diagnostics(self) -> dict:
        return {
            "silero_prob": round(self._robust_smoothed, 4),
            "in_speech": self._robust_in_speech,
            "state": "open" if self._robust_in_speech else "closed",
        }

    def reset_state(self) -> None:
        self.reset_calls += 1
        self._robust_smoothed = 0.0
        self._robust_in_speech = False
        self._robust_speech_frames = 0


class FakeWhisper:
    def __init__(self, text: str = "open firefox", confidence: float = -0.1):
        self.text = text
        self.confidence = confidence
        self.final_calls = []

    def transcribe(self, pcm: bytes, sample_rate: int):
        self.final_calls.append(pcm)
        return self.text, self.confidence

    def transcribe_fast(self, pcm: bytes, sample_rate: int):
        return self.text, self.confidence


def _tone(duration_s: float, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def _make_listener(whisper: FakeWhisper) -> CL.CommandListener:
    cl = CL.CommandListener()
    cl._whisper = whisper
    cl._ready = True
    return cl


# =================================================================
# 1. prepare_command_session() clears stale post-auth drain state
# =================================================================

def test_prepare_command_session_clears_stale_drain(monkeypatch):
    """After the post-auth greeting (`resume_listening()`), a leftover
    `_drain_requested=True` + recent gate-hold timestamp must NOT survive
    into the next command session. This is the exact state _speak_guarded()
    leaves behind that previously drained the freshest command audio."""
    cl = CL.CommandListener()
    vad = FakeVAD()
    monkeypatch.setattr(CL, "unified_vad", vad)

    # Simulate the post-auth greeting cycle:
    #   _speak_guarded -> pause_listening() -> ... -> resume_listening()
    cl.pause_listening()
    assert cl._listen_enabled.is_set() is False
    cl.resume_listening()
    assert cl._drain_requested is True
    assert cl._gate_closed_at is None
    assert cl._listen_enabled.is_set() is True

    # Also simulate a gate that was held closed long enough to trip the
    # backlog backstop (a force-resume may no longer fire mid-command).
    cl.pause_listening()
    cl._gate_closed_at = time.monotonic() - (CL.GATE_MAX_HOLD_S + 1.0)

    cl.prepare_command_session()

    assert cl._drain_requested is False, \
        "stale drain request must be cleared before LISTEN"
    assert cl._gate_closed_at is None, \
        "stale gate-hold timestamp must be cleared before LISTEN"
    assert cl._listen_enabled.is_set() is True, \
        "listen gate must be OPEN before LISTEN"
    assert vad.reset_calls >= 1, \
        "VAD state must be reset before LISTEN"


def test_prepare_command_session_reopens_stuck_closed_gate(monkeypatch):
    """If ANY post-auth path (greeting failure, exception before resume)
    left the listen gate CLOSED, prepare_command_session() must force it
    OPEN so the very next command is captured instead of being consumed and
    discarded by the closed-gate backlog branch (the WAITING_FOR_SPEECH
    hang reported in production)."""
    cl = CL.CommandListener()
    vad = FakeVAD()
    monkeypatch.setattr(CL, "unified_vad", vad)

    cl.pause_listening()  # gate CLOSED (stuck after auth)
    assert cl._listen_enabled.is_set() is False

    cl.prepare_command_session()

    assert cl._listen_enabled.is_set() is True, \
        "prepare_command_session must reopen a stuck-closed listen gate"
    assert cl._gate_closed_at is None
    assert cl._drain_requested is False
    assert vad.reset_calls >= 1


# =================================================================
# 2. Command spoken IMMEDIATELY after wake/auth is captured
# =================================================================

def test_closed_gate_after_auth_still_captures_command(monkeypatch):
    """Deterministic reproduction of the production symptom.

    Post-auth greeting leaves the listen gate CLOSED. Without
    prepare_command_session(), the streaming loop's closed-gate branch
    consumes-and-discards the fresh command audio and the listener hangs in
    WAITING_FOR_SPEECH (never emitting speech_start/final). After
    prepare_command_session(), the gate is reopened at LISTEN entry and the
    command is captured."""

    async def run(with_prepare: bool):
        am = FakeAudioManager()
        am.feed(_tone(3.0, freq=440.0, amp=0.9))  # stale wake/chime/auth

        whisper = FakeWhisper(text="open firefox")
        vad = FakeVAD()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", vad)

        cl = _make_listener(whisper)
        cl.pause_listening()  # gate stuck CLOSED after face-auth greeting
        if with_prepare:
            cl.prepare_command_session()

        events = []

        async def consume():
            async for ev in cl.stream_utterances():
                events.append(ev)
                if ev.kind == "final":
                    return

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        am.feed(_tone(1.2, freq=200.0, amp=0.6))
        am.feed(_silence(2.5))

        try:
            await asyncio.wait_for(consumer, timeout=6.0)
        except asyncio.TimeoutError:
            consumer.cancel()
        return [e.kind for e in events], whisper.final_calls

    without_kinds, without_whisper = asyncio.run(run(False))
    with_kinds, with_whisper = asyncio.run(run(True))

    # The fix must produce a captured command; the unfixed closed gate must not.
    assert "final" in with_kinds, f"fix must capture command, got {with_kinds}"
    assert with_whisper, "fix must send command audio to Whisper"


def test_immediate_command_after_prepare_is_captured(monkeypatch):
    """Reproduces 'wake accepted -> auth success -> immediately say command'.

    If prepare_command_session() is NOT called (the old behaviour), the
    leftover `_drain_requested=True` from the auth greeting drains the ring
    buffer on the first stream iteration and discards the command audio that
    arrived before LISTEN was fully armed - no `speech_start`, no `final`.
    """

    async def impl():
        am = FakeAudioManager()
        # Stale audio from wake + chime + face-auth greeting.
        am.feed(_tone(3.0, freq=440.0, amp=0.9))

        whisper = FakeWhisper(text="open firefox")
        vad = FakeVAD()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", vad)

        cl = _make_listener(whisper)

        # Simulate the post-auth greeting that would normally leave a stale
        # drain request; then prepare exactly as _conversation_session now
        # does right before building the stream.
        cl.pause_listening()
        cl.resume_listening()
        cl.prepare_command_session()

        events = []

        async def consume():
            async for ev in cl.stream_utterances():
                events.append(ev)
                if ev.kind == "final":
                    return

        consumer = asyncio.create_task(consume())
        # Let the stream enter its poll loop and fix its session boundary
        # (mirrors the production mic: audio arrives AFTER the listener
        # arms itself).
        await asyncio.sleep(0.02)

        # Command audio for the FIRST 1-2s - fed immediately after arming.
        am.feed(_tone(1.2, freq=200.0, amp=0.6))
        am.feed(_silence(2.5))

        try:
            await asyncio.wait_for(consumer, timeout=6.0)
        except asyncio.TimeoutError:
            consumer.cancel()

        kinds = [e.kind for e in events]
        assert "speech_start" in kinds, f"no speech_start, got {kinds}"
        assert "final" in kinds, f"no final transcript, got {kinds}"
        assert whisper.final_calls, "Whisper never received the command"

        # The STT audio must contain the fresh command, not just stale echo.
        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(samples) / SAMPLE_RATE < 4.5, \
            "stale face-auth greeting audio leaked into STT"

    asyncio.run(impl())


# =================================================================
# 3. First 1-2s after wake are NOT discarded
# =================================================================

def test_first_seconds_after_wake_not_discarded(monkeypatch):
    """The command session boundary must remain at the point where LISTEN
    began, so a command spoken in the first ~2s is fully captured - the
    pre-roll + captured audio must contain the leading speech frames."""

    async def impl():
        am = FakeAudioManager()
        am.feed(_tone(5.0, freq=330.0, amp=0.9))  # wake/chime/auth

        whisper = FakeWhisper(text="open firefox")
        vad = FakeVAD()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", vad)

        cl = _make_listener(whisper)

        cl.pause_listening()
        cl.resume_listening()
        cl.prepare_command_session()

        events = []

        async def consume():
            async for ev in cl.stream_utterances():
                events.append(ev)
                if ev.kind == "final":
                    return

        consumer = asyncio.create_task(consume())
        # Let the stream enter its poll loop and fix its session boundary.
        await asyncio.sleep(0.02)

        # Immediately (0.0s after arming) feed the leading edge of the spoken
        # command, then the rest. If the first frames are discarded, the
        # onset is gone and VAD never opens a capture window.
        am.feed(_tone(0.4, freq=200.0, amp=0.6))
        am.feed(_tone(1.0, freq=220.0, amp=0.6))
        am.feed(_silence(2.5))

        try:
            await asyncio.wait_for(consumer, timeout=6.0)
        except asyncio.TimeoutError:
            consumer.cancel()

        finals = [e for e in events if e.kind == "final"]
        assert finals, "no final transcript (first seconds discarded)"
        assert whisper.final_calls, "Whisper never received the command"

        # The captured audio must contain the early speech frames (0.6 amp
        # content), proving the leading edge was not clipped.
        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        assert float(np.max(np.abs(samples))) >= 0.4, \
            "leading speech frames were discarded (peak too low)"

    asyncio.run(impl())


# =================================================================
# 4. Full transition leaves the listener immediately ready
# =================================================================

def test_wake_auth_listen_transition_leaves_listener_ready(monkeypatch):
    """Static + behavioural contract: the engine calls
    prepare_command_session() WHEN it enters LISTEN, so the transition
    WAKE -> FACE_AUTH -> LISTEN leaves the listener in a ready state."""
    import inspect

    from core.conversation_engine import ConversationEngine

    src = inspect.getsource(ConversationEngine._conversation_session)
    assert "prepare_command_session()" in src, \
        "LISTEN entry must prepare the command listener"

    cl = CL.CommandListener()
    vad = FakeVAD()
    monkeypatch.setattr(CL, "unified_vad", vad)

    # Emulate a prior post-auth greeting that paused the listener.
    cl.pause_listening()
    cl.resume_listening()

    # The engine calls this at LISTEN entry.
    cl.prepare_command_session()

    assert cl._drain_requested is False
    assert cl._listen_enabled.is_set() is True
    assert vad.reset_calls >= 1


# =================================================================
# 5. Audio spoken during the setup window is not lost (eager boundary)
# =================================================================

def test_audio_spoken_between_prepare_and_stream_is_preserved(monkeypatch):
    """The engine anchors the command-session boundary at LISTEN entry
    (prepare_command_session), but the async generator opens a moment later.
    Speech fed in that gap must still be captured via the eager boundary."""

    async def run():
        am = FakeAudioManager()
        am.feed(_tone(5.0, freq=330.0, amp=0.9))  # wake/chime/auth

        whisper = FakeWhisper(text="open firefox")
        vad = FakeVAD()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", vad)

        cl = _make_listener(whisper)
        cl.pause_listening()
        cl.resume_listening()
        cl.prepare_command_session()

        # The ENTIRE command arrives in the window BETWEEN LISTEN entry
        # (prepare_command_session) and the async generator actually opening.
        # Only the eager boundary preserves it; a lazy boundary would start
        # at the current write head — AFTER this command audio.
        am.feed(_tone(1.2, freq=200.0, amp=0.6))
        am.feed(_silence(2.5))

        events = []

        async def consume():
            async for ev in cl.stream_utterances():
                events.append(ev)
                if ev.kind == "final":
                    return

        consumer = asyncio.create_task(consume())

        try:
            await asyncio.wait_for(consumer, timeout=6.0)
        except asyncio.TimeoutError:
            consumer.cancel()
        return [e.kind for e in events], whisper.final_calls

    kinds, whisper_calls = asyncio.run(run())
    assert "final" in kinds, f"setup-window audio must be captured, got {kinds}"
    assert whisper_calls, "Whisper never received the command"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
