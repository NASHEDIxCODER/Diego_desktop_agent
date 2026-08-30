"""
Regression tests: SILENCE must NOT produce a "say again" response.

Distinguishes (2026-08-30 UX fix):
  A. TRUE SILENCE         -> keep listening silently, NO TTS, NO failure.
  B. BACKGROUND NOISE     -> no spoken error (noise is not speech).
  C. SPEECH + STT FAILURE -> recovery response IS appropriate.
  D. VALID SPEECH         -> normal processing (final event).

The listener accumulates per-frame speech evidence (Silero voiced frames
+ strong-energy frames); _finalize() discards audio with no speech
evidence SILENTLY (returns None) instead of emitting a failure event
that the ConversationEngine would speak.

Uses fakes; no physical microphone or GPU required.
"""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import voice.command_listener as CL
from tests.test_command_listener import (
    FakeAudioManager, FakeVAD, FakeWhisper,
    _make_listener, _collect, tone, silence,
)


# A. Pure silence -> no spoken error
def test_pure_silence_no_spoken_error(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(silence(4.0))  # user says nothing

        events = await _collect(cl, feed, timeout=3.0)
        kinds = [e.kind for e in events]
        assert "failure" not in kinds, f"silence must NOT fail: {kinds}"
        assert "final" not in kinds
        assert not whisper.final_calls, "Whisper must not run for silence"

    asyncio.run(impl())


# B. Background noise without speech -> no spoken error
def test_background_noise_no_spoken_error(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        # FakeVAD(threshold=0.9): Silero always low (0.05) for the quiet
        # noise — mirrors the real mic. amp=0.06 tone -> int16 RMS ~1390:
        # above the 900 energy-fallback floor, below strong_speech_rms
        # (2500) -> NO speech evidence.
        whisper = FakeWhisper(text="")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD(threshold=0.9))
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.2, freq=200.0, amp=0.06))  # background noise
            am.feed(silence(2.5))                      # endpoint

        events = await _collect(cl, feed, timeout=6.0)
        kinds = [e.kind for e in events]
        assert "failure" not in kinds, f"noise must NOT fail: {kinds}"
        assert "final" not in kinds
        # The NO-SPEECH gate runs BEFORE transcription.
        assert not whisper.final_calls, "Whisper must not run for noise"

    asyncio.run(impl())


# C. Real speech + STT failure -> recovery response
def test_speech_with_stt_failure_produces_recovery_response(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        # amp=0.25 -> int16 RMS ~5800 (real speech level, > 2500).
        whisper = FakeWhisper(text="", confidence=-0.5)  # STT failure
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.2, freq=200.0, amp=0.25))  # real speech
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        failures = [e for e in events if e.kind == "failure"]
        assert len(failures) == 1, f"expected 1 failure, got {[e.kind for e in events]}"
        assert failures[0].failure_reason == CL.FAILURE_TRANSCRIPTION_FAILED

    asyncio.run(impl())


# C2. Whisper hallucination on noise -> silent discard
def test_hallucinated_transcript_on_noise_discarded_silently(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="I'm sorry", confidence=-0.9)
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD(threshold=0.9))
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.2, freq=200.0, amp=0.06))  # noise, not speech
            am.feed(silence(2.5))

        events = await _collect(cl, feed, timeout=6.0)
        kinds = [e.kind for e in events]
        assert "failure" not in kinds, f"hallucination over noise must be silent: {kinds}"

    asyncio.run(impl())


# D. Valid speech -> normal processing
def test_valid_speech_normal_processing(monkeypatch):
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open firefox")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        async def feed():
            am.feed(tone(1.2, freq=200.0, amp=0.25))  # real speech
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, f"expected 1 final, got {[e.kind for e in events]}"
        assert finals[0].text == "open firefox"
        assert not [e for e in events if e.kind == "failure"]

    asyncio.run(impl())


# Unit tests for the evidence helpers
def test_evidence_helper_silero_voiced():
    assert CL.CommandListener._has_speech_evidence(
        {"silero_voiced_ms": 300.0, "strong_ms": 0.0}) is True


def test_evidence_helper_strong_energy():
    assert CL.CommandListener._has_speech_evidence(
        {"silero_voiced_ms": 0.0, "strong_ms": 300.0}) is True


def test_evidence_helper_noise_only():
    assert CL.CommandListener._has_speech_evidence(
        {"silero_voiced_ms": 0.0, "strong_ms": 0.0,
         "loud_ms": 2000.0, "peak_rms": 1500.0}) is False


def test_evidence_helper_empty():
    assert CL.CommandListener._has_speech_evidence({}) is False


def test_measure_speech_evidence_energy_levels():
    # amp 0.25 -> int16 RMS ~5800 (strong); amp 0.06 -> ~1390 (loud only).
    frames = [tone(0.032, amp=0.25), tone(0.032, amp=0.06)]
    ev = CL.CommandListener._measure_speech_evidence(frames)
    assert ev["strong_ms"] >= 32.0, f"strong frame not counted: {ev}"
    assert ev["loud_ms"] >= 64.0, f"loud frames not counted: {ev}"
    assert ev["peak_rms"] > 2500.0


def test_measure_speech_evidence_silence():
    ev = CL.CommandListener._measure_speech_evidence([silence(0.1)])
    assert ev["strong_ms"] == 0.0
    assert ev["loud_ms"] == 0.0
    assert ev["peak_rms"] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))