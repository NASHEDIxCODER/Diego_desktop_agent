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
from pathlib import Path

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
    """

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)

    def feed(self, audio: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, np.asarray(audio, dtype=np.float32)])

    @property
    def total_samples(self) -> int:
        return int(self._buf.size)

    def read_since(self, last_total: int):
        last_total = max(0, int(last_total))
        if last_total >= self._buf.size:
            return np.array([], dtype=np.float32), int(self._buf.size)
        return self._buf[last_total:].copy(), int(self._buf.size)


class FakeVAD:
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.calls = 0

    def speech_prob(self, frame: np.ndarray) -> float:
        self.calls += 1
        rms = float(np.sqrt(np.mean(np.asarray(frame, np.float32) ** 2)))
        return 0.95 if rms > self.threshold else 0.05


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


async def _collect(cl, feed_fn, timeout: float = 6.0):
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
            am.feed(silence(1.5))

        events = await _collect(cl, feed)
        kinds = [e.kind for e in events]
        assert "final" in kinds, f"expected a final event, got {kinds}"
        assert whisper.final_calls, "Whisper should have been called"

        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = len(samples) / SAMPLE_RATE
        # The utterance is ~1s speech + pre-roll + trailing silence — NOT 10s.
        assert duration_s < 3.0, f"sent {duration_s:.1f}s to Whisper (stale audio leaked)"
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
            am.feed(silence(1.5))

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
            am.feed(silence(1.5))

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
            am.feed(silence(1.5))                       # real trailing silence

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
            am.feed(silence(1.5))

        await _collect(cl, feed)
        assert whisper.final_calls, "no final transcription"
        pcm = whisper.final_calls[0]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # The final segment is bounded by the session, not the whole buffer.
        assert len(samples) / SAMPLE_RATE < 3.0
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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
