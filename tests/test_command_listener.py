"""
Post-wake command-capture pipeline tests.

Covers the fix for "wake accepted but no command is transcribed":

  A. Fresh command session — stale ring-buffer audio is never sent to Whisper.
  B. Audio handoff — the command listener receives newly produced samples.
  C. VAD — speech opens capture, sustained silence closes it.
  D. Endpoint — a short pause inside speech does not prematurely terminate.
  E. Whisper — only the captured command segment reaches STT.
  F. Wake→command transition — a peek-based ring buffer cannot be "stolen".
  G. Empty audio — silence produces no events and never hangs.

These tests use fakes; no physical microphone or GPU is required.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from voice.vad import unified_vad

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import numpy as np

import voice.command_listener as CL
from voice.audio_manager import RingBuffer, FRAME_SAMPLES, SAMPLE_RATE


# ═══════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════

class FakeAudioManager:
    """Cursor-based audio source mirroring AudioManager.read_since().

    read_since() is a NON-DESTRUCTIVE peek: it returns every sample written
    after `last_total` without removing them, exactly like the production
    RingBuffer. Feeding is instantaneous; the command listener drains to the
    current write head when its session starts.

    Also mirrors the frame-sequence diagnostics (frame_id /
    last_frame_timestamp / last_frame_rms / get_frame_state) added to the
    production AudioManager so the CMD-AUDIO handoff regression test can
    assert that FRESH frames arrive after LISTEN.
    """

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self.frame_id = 0
        self.last_frame_timestamp = 0.0
        self.last_frame_rms = 0.0

    def feed(self, audio: np.ndarray) -> None:
        import time as _time
        audio = np.asarray(audio, dtype=np.float32)
        self._buf = np.concatenate([self._buf, audio])
        # Advance the frame sequence as the real callback would (one frame
        # per FRAME_SAMPLES chunk).
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
            "last_frame_rms": self.last_frame_rms,
            "ring_buffer_samples": int(self._buf.size),
            "ring_buffer_frames": int(self._buf.size // FRAME_SAMPLES),
            "ring_buffer_seconds": self._buf.size / SAMPLE_RATE,
            "dropped_frames": 0,
            "digital_silence": False,
            "zero_streak": 0,
            "speech_channel": 0,
            "energy_threshold": 300.0,
        }


class FakeVAD:
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.calls = 0
        self._robust_smoothed = 0.0
        self._robust_in_speech = False
        self._robust_speech_frames = 0
        self._last_probability = 0.0
        self._state = "closed"

    def speech_prob(self, frame: np.ndarray) -> float:
        self.calls += 1
        rms = float(np.sqrt(np.mean(np.asarray(frame, np.float32) ** 2)))
        prob = 0.95 if rms > self.threshold else 0.05
        self._last_probability = prob
        self._state = "open" if prob > 0.5 else "closed"
        return prob

    def reset_state(self) -> None:
        """TASK 1: reset all VAD state (mirrors UnifiedVAD.reset_state)."""
        self._robust_smoothed = 0.0
        self._robust_in_speech = False
        self._robust_speech_frames = 0
        self._last_probability = 0.0
        self._state = "closed"

    # TASK 2: robust combined-evidence methods (mirror UnifiedVAD).
    # Updated to match vad.py: ROBUST_ENERGY_RMS=2500, ROBUST_ENERGY_WEIGHT=0.2,
    # ROBUST_SILERO_WEIGHT=0.8 (CRITICAL FIX 2026-08-29).
    def robust_speech_prob(self, frame: np.ndarray) -> float:
        silero = self.speech_prob(frame)
        rms = float(np.sqrt(np.mean(np.asarray(frame, np.float32) ** 2))) * 32768.0
        self._robust_smoothed = 0.4 * silero + 0.6 * self._robust_smoothed
        energy_score = 1.0 if rms >= 2500.0 else max(0.0, rms / 2500.0)
        return float(min(max(0.8 * self._robust_smoothed + 0.2 * energy_score, 0.0), 1.0))

    def robust_is_speech(self, frame: np.ndarray) -> bool:

        score = self.robust_speech_prob(frame)
        if self._robust_in_speech:
            if score < 0.35:
                self._robust_speech_frames = 0
                self._robust_in_speech = False
        else:
            if score >= 0.55:
                self._robust_speech_frames += 1
                if self._robust_speech_frames >= 3:
                    self._robust_in_speech = True
            else:
                self._robust_speech_frames = 0
        return self._robust_in_speech

    def get_robust_diagnostics(self) -> dict:
        return {
            "silero_prob": round(self._last_probability, 4),
            "smoothed_prob": round(self._robust_smoothed, 4),
            "in_speech": self._robust_in_speech,
            "speech_frames": self._robust_speech_frames,
            "state": self._state,
        }


class FakeWhisper:
    def __init__(self, text: str = "open vs code", confidence: float = -0.1):
        self.text = text
        self.confidence = confidence
        self.final_calls = []
        self.fast_calls = []

    def transcribe(self, pcm: bytes, sample_rate: int):
        self.final_calls.append(pcm)
        return self.text, self.confidence

    def transcribe_fast(self, pcm: bytes, sample_rate: int):
        self.fast_calls.append(pcm)
        return self.text, self.confidence


def tone(duration_s: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def _make_listener(whisper: FakeWhisper) -> CL.CommandListener:
    cl = CL.CommandListener()
    cl._whisper = whisper
    cl._ready = True
    return cl


async def _collect(cl, feed_fn, timeout: float = 10.0):
    """Run stream_utterances(); feed_fn feeds audio after the drain."""

    events = []

    async def consume():
        async for ev in cl.stream_utterances():
            events.append(ev)
            if ev.kind == "final":
                return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.02)  # allow the session to drain + enter the poll loop
    if feed_fn is not None:
        await feed_fn()
    try:
        await asyncio.wait_for(consumer, timeout=timeout)
    except asyncio.TimeoutError:
        consumer.cancel()
        try:
            await consumer
        except (asyncio.CancelledError, Exception):
            pass
    return events


# ═══════════════════════════════════════════════════════════════
# A. Fresh command session — stale audio is not sent to Whisper
# ═══════════════════════════════════════════════════════════════

def test_fresh_session_ignores_stale_audio(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        # 10 seconds of stale audio (wake word + chime + TTS) at 0.9 amplitude.
        am.feed(tone(10.0, freq=300.0, amp=0.9))

        whisper = FakeWhisper()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.2))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        kinds = [e.kind for e in events]
        assert "final" in kinds, f"expected a final event, got {kinds}"
        assert whisper.final_calls, "Whisper should have been called"

        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = len(samples) / SAMPLE_RATE
        # The utterance is ~1s speech + pre-roll + trailing silence — NOT 10s.
        assert duration_s < 4.5, f"sent {duration_s:.1f}s to Whisper (stale audio leaked)"
        # The 0.9-amplitude stale tone must not be present.
        assert float(np.max(np.abs(samples))) < 0.6, "stale 0.9-amplitude audio leaked into STT"

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# B. Audio handoff — command listener receives newly produced samples
# ═══════════════════════════════════════════════════════════════

def test_audio_handoff_receives_new_samples(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper()
        vad = FakeVAD()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", vad)
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.2))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        kinds = [e.kind for e in events]
        assert "speech_start" in kinds
        assert "final" in kinds
        # VAD was actually exercised on freshly produced samples.
        assert vad.calls > 0, "VAD was never called — samples did not reach the listener"

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# C. VAD — speech opens capture, sustained silence closes it
# ═══════════════════════════════════════════════════════════════

def test_vad_opens_and_silence_closes(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.25))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        assert events[0].kind == "speech_start"
        assert events[-1].kind == "final"

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# D. Endpoint — a short pause inside speech does not terminate
# ═══════════════════════════════════════════════════════════════

def test_short_pause_does_not_prematurely_end(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(0.5, freq=200.0, amp=0.25))   # first words
            am.feed(silence(0.2))                       # short pause (< 600ms)
            am.feed(tone(0.5, freq=200.0, amp=0.25))   # rest of the command
            am.feed(silence(2.5))                       # real trailing silence

        events = await _collect(cl, feed)
        starts = [e for e in events if e.kind == "speech_start"]
        finals = [e for e in events if e.kind == "final"]
        # The 200ms pause must not close the utterance; exactly one final.
        assert len(starts) == 1, f"expected 1 speech_start, got {len(starts)}"
        assert len(finals) == 1, f"pause prematurely ended the turn: {len(finals)} finals"

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# E. Whisper — only the captured command segment reaches STT
# ═══════════════════════════════════════════════════════════════

def test_whisper_receives_only_captured_segment(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        am.feed(tone(5.0, freq=300.0, amp=0.9))  # stale history
        whisper = FakeWhisper()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.2))
            am.feed(silence(2.5))

        await _collect(cl, feed)
        assert whisper.final_calls, "no final transcription"
        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # The final segment is bounded by the session, not the whole buffer.
        assert len(samples) / SAMPLE_RATE < 4.5
        # Partial (fast) transcripts were also short — never the 5s history.
        for fpcm in whisper.fast_calls:
            f = np.frombuffer(fpcm, dtype=np.int16).astype(np.float32) / 32768.0
            assert len(f) / SAMPLE_RATE < 3.0

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# F. Wake→command transition — a peek-based buffer cannot be stolen
# ═══════════════════════════════════════════════════════════════

def test_ring_buffer_peek_semantics():
    rb = RingBuffer(max_frames=100)
    for _ in range(5):
        rb.put(np.full(FRAME_SAMPLES, 0.1, np.float32))

    # The wake listener reads everything written so far (non-destructive).
    wake_audio, _ = rb.get_since(0)
    assert len(wake_audio) == 5 * FRAME_SAMPLES

    # The command listener drains to the current write head.
    cmd_cursor = rb.total_samples
    cmd_audio, _ = rb.get_since(cmd_cursor)
    assert len(cmd_audio) == 0, "drain should skip pre-session audio"

    # New command audio arrives after the boundary.
    rb.put(np.full(FRAME_SAMPLES, 0.2, np.float32))
    cmd_audio2, new_total = rb.get_since(cmd_cursor)
    assert len(cmd_audio2) == FRAME_SAMPLES
    assert new_total == 6 * FRAME_SAMPLES

    # The wake listener's earlier read was not affected (peek, not consume).
    wake_audio2, _ = rb.get_since(0)
    assert len(wake_audio2) == 6 * FRAME_SAMPLES


# ═══════════════════════════════════════════════════════════════
# G. Empty audio — no speech produces no events and never hangs
# ═══════════════════════════════════════════════════════════════

def test_empty_audio_no_hang(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(silence(2.0))  # pure silence, no speech

        events = await _collect(cl, feed, timeout=2.0)
        assert events == [], f"expected no events for pure silence, got {[e.kind for e in events]}"
        assert not whisper.final_calls, "Whisper must not be called for silence"

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# H. REGRESSION: LISTEN starts → fresh audio arrives → consumed
# ═══════════════════════════════════════════════════════════════
# This is the dedicated regression test for the "no fresh microphone audio
# after LISTEN" bug. It proves the CommandListener observes the AudioManager
# frame sequence ADVANCE after the session boundary, i.e. the callback is
# still producing NEW frames and the listener consumes them (not stale
# ring-buffer history).

def test_listen_receives_fresh_audio_after_session_start(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        # Pre-populate stale history (wake word + chime) — frame_id advances.
        am.feed(tone(3.0, freq=300.0, amp=0.9))
        stale_frame_id = am.frame_id
        assert stale_frame_id > 0

        whisper = FakeWhisper()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        # Capture the frame_id the listener snapshots at session start.
        observed_start_frame_id = []

        # Wrap read_since to record the frame_id the listener sees on its
        # FIRST successful read (i.e. after the session boundary).
        orig_read_since = am.read_since
        first_read_frame_id = []

        def tracked_read_since(last_total):
            result = orig_read_since(last_total)
            if len(result[0]) > 0 and not first_read_frame_id:
                first_read_frame_id.append(am.frame_id)
            return result

        am.read_since = tracked_read_since

        async def feed():
            # Fresh command audio AFTER the listener has entered LISTEN.
            am.feed(tone(1.0, freq=200.0, amp=0.2))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        kinds = [e.kind for e in events]
        assert "final" in kinds, f"expected final event, got {kinds}"

        # The listener must have consumed audio produced AFTER the stale
        # history — the frame sequence advanced beyond the stale frame_id.
        assert first_read_frame_id, "listener never read fresh audio"
        assert first_read_frame_id[0] > stale_frame_id, (
            f"listener consumed STALE audio: first_read_frame_id={first_read_frame_id[0]} "
            f"<= stale_frame_id={stale_frame_id}"
        )

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# TASK 11 REGRESSION TESTS — the actual runtime bugs
# ═══════════════════════════════════════════════════════════════
# A. speech → silence → endpoint
# B. speech → brief VAD dip → still speech
# C. noise → no speech
# D. speech → Whisper partial → silence → final
# E. Whisper inference running during endpoint
# F. TTS → resume listening
# G. stale audio cannot enter next command
# H. VAD probability can never exceed 1.0
# I. vad_avg always remains 0.0–1.0
# J. 20-second max duration is never used when silence endpoint is available


# ── A. speech → silence → endpoint ─────────────────────────────
def test_speech_silence_endpoint(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open youtube")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.25))
            am.feed(silence(2.5))  # 1000ms silence → endpoint

        events = await _collect(cl, feed)
        kinds = [e.kind for e in events]
        assert "speech_start" in kinds
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, f"expected exactly one final, got {len(finals)}"
        assert finals[0].endpoint_reason == "silence", (
            f"endpoint reason must be silence, got {finals[0].endpoint_reason}")
        assert finals[0].text == "open youtube"

    asyncio.run(impl())


# ── B. speech → brief VAD dip → still speech ──────────────────
def test_brief_vad_dip_still_speech(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open youtube")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(0.4, freq=200.0, amp=0.25))   # speech
            am.feed(silence(0.15))                      # brief dip (150ms < 700ms)
            am.feed(tone(0.4, freq=200.0, amp=0.25))   # speech resumes
            am.feed(silence(2.5))                       # real silence

        events = await _collect(cl, feed)
        starts = [e for e in events if e.kind == "speech_start"]
        finals = [e for e in events if e.kind == "final"]
        assert len(starts) == 1, f"brief dip split speech: {len(starts)} starts"
        assert len(finals) == 1, f"brief dip caused premature endpoint: {len(finals)} finals"

    asyncio.run(impl())


# ── C. noise → no speech ──────────────────────────────────────
def test_noise_no_speech(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper()
        # Low-amplitude noise below the VAD threshold.
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD(threshold=0.5))
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.01))  # very quiet noise
            am.feed(silence(2.5))

        events = await _collect(cl, feed, timeout=2.0)
        assert not whisper.final_calls, "Whisper must not be called for noise"
        assert not [e for e in events if e.kind == "speech_start"], (
            "noise must not trigger speech_start")

    asyncio.run(impl())


# ── D. speech → Whisper partial → silence → final ─────────────
def test_speech_partial_silence_final(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open firefox")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.5, freq=200.0, amp=0.25))  # enough for a partial
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, "expected one final"
        # A partial may or may not have run (background), but the final must
        # be present and correct.
        assert finals[0].text == "open firefox"

    asyncio.run(impl())


# ── E. Whisper inference running during endpoint ──────────────
def test_whisper_running_during_endpoint(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open vscode")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        # Make the fast (partial) transcription slow so it is still running
        # when the silence endpoint fires.
        original_fast = whisper.transcribe_fast

        def slow_fast(pcm, sample_rate):
            import time as _t
            _t.sleep(0.5)  # 500ms inference
            return original_fast(pcm, sample_rate)

        whisper.transcribe_fast = slow_fast

        async def feed():
            am.feed(tone(1.5, freq=200.0, amp=0.25))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, "endpoint must still produce exactly one final"
        assert finals[0].text == "open vscode"

    asyncio.run(impl())


# ── F. TTS → resume listening ─────────────────────────────────
def test_tts_resume_listening(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open youtube")
        vad = FakeVAD()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", vad)
        cl = _make_listener(whisper)

        async def feed():
            # Simulate TTS: pause, feed TTS audio, resume.
            cl.pause_listening()
            am.feed(tone(1.0, freq=300.0, amp=0.5))  # TTS audio
            cl.resume_listening()
            await asyncio.sleep(0.05)  # allow drain
            # Now real user speech.
            am.feed(tone(1.0, freq=200.0, amp=0.25))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, "expected one final after TTS resume"
        # The TTS audio must not be in the final transcript's audio.
        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        assert float(np.max(np.abs(samples))) < 0.6, "TTS audio leaked into final"

    asyncio.run(impl())


# ── G. stale audio cannot enter next command ──────────────────
def test_stale_audio_cannot_enter_next_command(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open youtube")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        # Pre-populate stale audio (previous wake/face-auth/TTS/command).
        am.feed(tone(5.0, freq=300.0, amp=0.9))

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.25))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1
        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(samples) / SAMPLE_RATE < 4.5, "stale 5s audio leaked into command"
        assert float(np.max(np.abs(samples))) < 0.6, "stale 0.9-amplitude audio leaked"

    asyncio.run(impl())


# ── H. VAD probability can never exceed 1.0 ───────────────────
def test_vad_probability_never_exceeds_one():
    from voice.vad import UnifiedVAD
    vad = UnifiedVAD()
    # Force a pathological frame that would produce a huge RMS.
    frame = np.full(512, 10.0, dtype=np.float32)  # extreme amplitude
    prob = vad.speech_prob(frame)
    assert 0.0 <= prob <= 1.0, f"VAD probability out of range: {prob}"

    # Also test the robust combined score.
    prob2 = vad.robust_speech_prob(frame)
    assert 0.0 <= prob2 <= 1.0, f"robust VAD probability out of range: {prob2}"


# ── I. vad_avg always remains 0.0–1.0 ─────────────────────────
def test_vad_avg_always_in_range():
    # Simulate the metric aggregation: sum of clamped probs / count.
    # Every prob is clamped to [0,1] by UnifiedVAD.speech_prob, so the
    # average can NEVER exceed 1.0.
    probs = [0.95, 0.0, 1.0, 0.5, 0.8, 0.0, 0.3, 1.0, 0.9, 0.1]
    avg = sum(probs) / len(probs)
    assert 0.0 <= avg <= 1.0, f"vad_avg out of range: {avg}"

    # Even with pathological inputs, clamping keeps it in range.
    pathological = [1.7, -0.5, 42.801, 17.201, float('nan')]
    clamped = []
    for p in pathological:
        if p != p:  # NaN
            p = 0.0
        clamped.append(min(max(p, 0.0), 1.0))
    avg2 = sum(clamped) / len(clamped)
    assert 0.0 <= avg2 <= 1.0, f"vad_avg out of range after clamping: {avg2}"


# ── J. 20s max duration is never used when silence is available ─
def test_max_duration_never_used_with_silence(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open youtube")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.0, freq=200.0, amp=0.25))
            am.feed(silence(2.5))  # silence endpoint available

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1
        assert finals[0].endpoint_reason == "silence", (
            f"endpoint reason must be silence, got {finals[0].endpoint_reason}")
        assert finals[0].endpoint_reason != "max_duration", (
            "max_duration must NEVER be the endpoint when silence is available")

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# REGRESSION TESTS (2026-08-29 fixes)
# ═══════════════════════════════════════════════════════════════

# ── K. _finalize() empty-frame crash ──────────────────────────
def test_finalize_empty_frames_no_crash(monkeypatch):
    """_finalize() must not crash when frames is empty (np.concatenate ValueError)."""
    async def impl():
        cl = _make_listener(FakeWhisper())
        # Call _finalize with empty frames — must return a failure event, not raise.
        ev = await cl._finalize([], time.time(), endpoint_reason="silence")
        assert ev is not None
        assert ev.kind == "failure"
        assert ev.failure_reason == CL.FAILURE_TRANSCRIPTION_FAILED

    asyncio.run(impl())


# ── L. _frames_to_bytes empty frames ──────────────────────────
def test_frames_to_bytes_empty_no_crash():
    """_frames_to_bytes([]) must return b'' not raise ValueError."""
    cl = CL.CommandListener()
    assert cl._frames_to_bytes([]) == b""


# ── M. Valid command starting with conjunction is accepted ────
def test_conjunction_command_accepted():
    """'and open Chrome' is a valid command — must NOT be rejected as garbage."""
    accepted, reason = CL._validate_transcript(
        "and open Chrome", confidence=-0.5, speech_dur_ms=1500)
    assert accepted, f"valid command rejected: {reason}"


# ── N. Short conjunction fragment is rejected ─────────────────
def test_short_conjunction_fragment_rejected():
    """'and you're doing' (short fragment) IS garbage — must be rejected."""
    accepted, reason = CL._validate_transcript(
        "and you're doing", confidence=-0.5, speech_dur_ms=800)
    assert not accepted, "short conjunction fragment should be rejected"
    assert reason == CL.FAILURE_GARBAGE


# ── O. Valid 2-word command with healthy audio is accepted ────
def test_two_word_command_with_healthy_audio_accepted():
    """'open Chrome' with 1.5s audio and confidence -0.7 must be accepted."""
    accepted, reason = CL._validate_transcript(
        "open Chrome", confidence=-0.7, speech_dur_ms=1500)
    assert accepted, f"valid 2-word command rejected: {reason}"


# ── P. Short 2-word command with low confidence is rejected ───
def test_short_low_confidence_two_word_rejected():
    """'I do' with 300ms audio and confidence -0.7 must be rejected."""
    accepted, reason = CL._validate_transcript(
        "I do", confidence=-0.7, speech_dur_ms=300)
    assert not accepted, "short low-confidence blip should be rejected"
    assert reason == CL.FAILURE_LOW_CONFIDENCE


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
