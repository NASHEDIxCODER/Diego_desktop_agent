"""
Regression tests for the 2026-08-30 sustained-speech hardening.

Covers:
  A. Weak/transient VAD spike does NOT create an utterance
     (no speech_start, no Whisper decode, no failure response).
  B. The STT guard (_has_speech_evidence) rejects a transient spike
     captured into a long buffer BEFORE Whisper runs.
  C. Invalid / low-quality transcripts ("you", "too") are rejected
     cheaply and never reach the Brain's expensive pipeline.
  D. Non-vision commands do NOT invoke OCR; vision commands DO.
  E. The next command works after TTS (listen guard resume).

These tests use fakes; no physical microphone, GPU, or OCR model needed.
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
from voice.audio_manager import FRAME_SAMPLES, SAMPLE_RATE

from tests.test_command_listener import (
    FakeAudioManager,
    FakeVAD,
    FakeWhisper,
    tone,
    silence,
    _make_listener,
    _collect,
)


# ═══════════════════════════════════════════════════════════════
# A. Weak VAD spike does not create an utterance
# ═══════════════════════════════════════════════════════════════

def test_weak_vad_spike_does_not_create_utterance(monkeypatch):
    """A transient spike (< 160ms confirmation window) must not open a
    capture window: no speech_start, no Whisper call, no failure event."""
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="you")
        # VAD threshold high enough that only the loud spike crosses it.
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD(threshold=0.5))
        cl = _make_listener(whisper)

        async def feed():
            # 96ms loud spike (3 frames < speech_start_confirm_frames=5)
            am.feed(tone(0.096, freq=200.0, amp=0.25))
            # then sustained silence → nothing should ever finalize
            am.feed(silence(2.5))

        events = await _collect(cl, feed, timeout=3.0)
        assert not whisper.final_calls, (
            "Whisper must not be called for a transient VAD spike")
        assert not [e for e in events if e.kind == "speech_start"], (
            "transient spike must not trigger speech_start")
        assert not [e for e in events if e.kind == "failure"], (
            "transient spike must not produce a failure response")

    asyncio.run(impl())


def test_background_noise_does_not_reach_whisper(monkeypatch):
    """Background noise (RMS in the 1200-2300 band, below the 2500
    energy-fallback floor) must not open a capture window even when it
    lasts a while."""
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="too")
        # FakeVAD threshold set so noise (amp 0.05 → int16 RMS ~1638)
        # crosses the Silero prob gate — mimicking a noisy mic where
        # Silero is unreliable and the energy fallback is the only guard.
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD(threshold=0.04))
        cl = _make_listener(whisper)

        async def feed():
            # 1s of moderate noise (RMS ~1638 < energy_fallback_rms 2500)
            am.feed(tone(1.0, freq=150.0, amp=0.05))
            am.feed(silence(2.5))

        events = await _collect(cl, feed, timeout=3.0)
        assert not whisper.final_calls, (
            "background noise must not trigger a Whisper decode")

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# B. STT guard: transient spike inside a long buffer → silent discard
# ═══════════════════════════════════════════════════════════════

def test_stt_guard_discards_transient_spike_before_whisper(monkeypatch):
    """A ~2s buffer containing only a brief loud spike has a voiced ratio
    far below min_voiced_ratio — _finalize must return None (silent
    discard) WITHOUT calling Whisper."""
    async def impl():
        whisper = FakeWhisper(text="you")
        cl = _make_listener(whisper)

        # 2s buffer: 100ms loud spike + 1.9s of silence
        frames = []
        spike = tone(0.1, freq=200.0, amp=0.25)
        n_spike_frames = len(spike) // FRAME_SAMPLES
        for i in range(n_spike_frames):
            frames.append(spike[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].copy())
        quiet = silence(1.9)
        for i in range(len(quiet) // FRAME_SAMPLES):
            frames.append(quiet[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].copy())

        final = await cl._finalize(frames, start=0.0, endpoint_reason="silence")
        assert final is None, (
            "transient spike must be discarded silently (no event at all)")
        assert not whisper.final_calls, (
            "Whisper must not run for a transient spike buffer")

    asyncio.run(impl())


def test_stt_guard_accepts_real_speech(monkeypatch):
    """A buffer of sustained voiced audio passes the STT guard and
    reaches Whisper."""
    async def impl():
        whisper = FakeWhisper(text="open firefox")
        cl = _make_listener(whisper)

        speech = tone(1.5, freq=200.0, amp=0.25)
        frames = []
        for i in range(len(speech) // FRAME_SAMPLES):
            frames.append(speech[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].copy())

        final = await cl._finalize(frames, start=0.0, endpoint_reason="silence")
        assert final is not None
        assert final.kind == "final"
        assert final.text == "open firefox"
        assert len(whisper.final_calls) == 1

    asyncio.run(impl())


# ═══════════════════════════════════════════════════════════════
# C. Invalid / low-quality transcript never reaches the Brain
# ═══════════════════════════════════════════════════════════════

def test_hallucinated_fragments_rejected_by_listener_validation():
    """'you' (conf -1.35) and 'too' (conf -0.657) from weak audio must be
    rejected by _validate_transcript."""
    # "you" — garbage pattern + single non-standalone word, low conf
    accepted, reason = CL._validate_transcript("you", -1.35, 2000.0)
    assert not accepted, "'you' must be rejected"
    assert reason in (CL.FAILURE_GARBAGE, CL.FAILURE_LOW_CONFIDENCE)

    # "too" — single non-standalone word, low conf
    accepted, reason = CL._validate_transcript("too", -0.657, 2000.0)
    assert not accepted, "'too' must be rejected"

    # Even with decent confidence, a non-standalone single word with
    # short audio is rejected.
    accepted, _ = CL._validate_transcript("too", -0.2, 400.0)
    assert not accepted


def test_valid_single_word_commands_still_accepted():
    """Real single-word commands/conversation must NOT be rejected."""
    for text, conf in (("stop", -0.5), ("yes", -0.7), ("open", -0.8),
                       ("firefox", -0.6), ("how are you", -0.737)):
        accepted, reason = CL._validate_transcript(text, conf, 1500.0)
        assert accepted, f"'{text}' (conf={conf}) must be accepted, got {reason}"


def test_is_low_quality_transcript_guard():
    """The cheap Brain-side guard rejects hallucinations and accepts
    real commands."""
    assert CL.is_low_quality_transcript("you")
    assert CL.is_low_quality_transcript("too")
    assert CL.is_low_quality_transcript("um")
    assert CL.is_low_quality_transcript("")
    assert CL.is_low_quality_transcript("I'm sorry")
    # Real commands / conversation are never rejected
    assert not CL.is_low_quality_transcript("open firefox")
    assert not CL.is_low_quality_transcript("how are you")
    assert not CL.is_low_quality_transcript("what is on my screen")
    assert not CL.is_low_quality_transcript("stop")
    assert not CL.is_low_quality_transcript("play some music")


def test_low_quality_transcript_never_reaches_brain_pipeline(monkeypatch):
    """agent_brain.process_command must return the REJECTED_TRANSCRIPT
    path for a hallucinated fragment — no perception, no planner, no LLM,
    no actions."""
    from agent.brain import AgentBrain

    brain = AgentBrain()
    brain._initialized = True  # skip subsystem wiring

    # Any call into perception/planner/LLM would fail the test.
    async def _fail_perceive(*a, **k):
        raise AssertionError("perception must not run for a rejected transcript")

    async def _fail_plan(*a, **k):
        raise AssertionError("planner must not run for a rejected transcript")

    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)

    result = asyncio.run(brain.process_command("you"))
    assert result.path == "REJECTED_TRANSCRIPT"
    assert result.used_llm is False
    assert result.actions_executed == 0
    assert result.response  # a gentle recovery response is spoken


# ═══════════════════════════════════════════════════════════════
# D. Demand-driven OCR
# ═══════════════════════════════════════════════════════════════

def test_non_vision_command_does_not_invoke_ocr():
    """'open firefox' / 'how are you?' must not require OCR."""
    from agent.brain import AgentBrain
    assert not AgentBrain._ocr_required("open firefox")
    assert not AgentBrain._ocr_required("how are you?")
    assert not AgentBrain._ocr_required("open chrome")
    assert not AgentBrain._ocr_required("tell me a joke")
    assert not AgentBrain._ocr_required("volume up")


def test_vision_command_requires_ocr():
    """'what is on my screen?' / 'read this error' must require OCR."""
    from agent.brain import AgentBrain
    assert AgentBrain._ocr_required("what is on my screen?")
    assert AgentBrain._ocr_required("read this error")
    assert AgentBrain._ocr_required("what's on my screen")
    assert AgentBrain._ocr_required("look at my screen")


def test_perceive_passes_ocr_flag_to_pipeline(monkeypatch):
    """_perceive must forward include_ocr=False for non-vision commands
    and include_ocr=True for vision commands."""
    from agent.brain import AgentBrain

    brain = AgentBrain()

    captured = {}

    class FakePerception:
        async def perceive(self, include_ocr=True, **kwargs):
            captured["include_ocr"] = include_ocr

            class Ctx:
                window_title = "Test"
                a11y_available = False
                ocr_used = False
                compact_summary = "Window: Test"
            return Ctx()

    brain._perception = FakePerception()

    asyncio.run(brain._perceive("open firefox"))
    assert captured["include_ocr"] is False, (
        "non-vision command must not invoke OCR")

    asyncio.run(brain._perceive("what is on my screen?"))
    assert captured["include_ocr"] is True, (
        "vision command must invoke perception with OCR")


# ═══════════════════════════════════════════════════════════════
# E. Next command works after TTS (listen guard)
# ═══════════════════════════════════════════════════════════════

def test_next_command_works_after_tts(monkeypatch):
    """After pause_listening() (TTS guard) + resume_listening(), a new
    spoken command must still be captured and finalized."""
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open youtube")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        # Simulate the TTS guard cycle.
        cl.pause_listening()
        assert cl._listen_enabled.is_set() is False
        cl.resume_listening()
        assert cl._listen_enabled.is_set() is True, (
            "listen gate must reopen after TTS")
        assert cl._drain_requested is True

        # Give the streaming loop a moment to process the drain request,
        # then feed the next command.
        async def feed():
            await asyncio.sleep(0.05)
            am.feed(tone(1.0, freq=200.0, amp=0.25))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, (
            "next command after TTS must be captured exactly once")
        assert finals[0].text == "open youtube"

    asyncio.run(impl())


def test_resume_listening_is_idempotent():
    """Calling resume_listening() repeatedly (engine + TTS paths) must
    never wedge the gate closed."""
    cl = CL.CommandListener()
    cl.pause_listening()
    cl.resume_listening()
    cl.resume_listening()
    cl.resume_listening()
    assert cl._listen_enabled.is_set() is True
    assert cl._drain_requested is True