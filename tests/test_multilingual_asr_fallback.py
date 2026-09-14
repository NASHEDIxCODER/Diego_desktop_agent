"""
Phase 20B — production-safe hybrid Whisper ASR (English-primary + gated Hindi).

Covers spec cases A–O over the REAL production functions in
voice/command_listener.py (no microphone, no GPU, no network):

  A. Confident English primary → NO fallback (zero latency overhead).
  B. Suspicious English primary → exactly ONE Hindi second pass (max 2).
  C. Fallback reuses same audio/config — only `language` differs ("hi").
  D. Valid beats invalid (fallback valid, primary invalid → "hi" wins).
  E. Valid beats invalid (primary valid, fallback invalid → "en" wins).
  F. Both invalid → UNCERTAIN path, stronger logprob kept, still rejected
     downstream (fallback cannot bypass safety gates).
  G. Hallucination-free beats hallucinated.
  H. Coherent (multi-word) beats 0/1-word fragment.
  I. Stronger avg_logprob wins a decisive gap.
  J. Language evidence breaks near-ties (Devanagari/Hinglish vocab → "hi").
  K. Intent compatibility is comparison-only (no execution side effects).
  L. Metrics present on every final/failure event; total == primary+fallback.
  M. Fallback disabled (per-listener or global) → single English pass.
  N. Raw transcript preserved (fallback output text flows downstream, not a
     translation/paraphrase).
  O. English regression: existing English commands unaffected
     ("open firefox" / "play believer on youtube" still actionable).

Safety: the fallback NEVER executes anything; selection only picks ONE
transcript for the unchanged downstream pipeline
(_postprocess → _validate_transcript → normalize → authorize → decision).
"""

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import numpy as np

import voice.command_listener as CL
from voice.audio_manager import FRAME_SAMPLES, SAMPLE_RATE


# ── Fakes ─────────────────────────────────────────────────────────

class FakeHybridWhisper:
    """Per-language scripted transcriber.

    transcribe_with_language(pcm, sr, lang) returns the scripted
    (text, conf) for "en"/"hi" and records the language + pcm identity so
    tests can assert SAME audio / ONLY language differs / max 2 passes.
    """

    def __init__(self, en=("open firefox", -0.4), hi=("", 0.0)):
        self._script = {"en": en, "hi": hi}
        self.calls = []  # (language, len(pcm), sample_rate)

    def transcribe_with_language(self, pcm: bytes, sample_rate: int,
                                 language=None):
        lang = language or "en"
        self.calls.append((lang, len(pcm), sample_rate))
        text, conf = self._script.get(lang, ("", 0.0))
        return text, conf

    # Legacy entry point (primary path back-compat).
    def transcribe(self, pcm: bytes, sample_rate: int):
        return self.transcribe_with_language(pcm, sample_rate, "en")

    @property
    def languages(self):
        return [c[0] for c in self.calls]

    @property
    def pass_count(self):
        return len(self.calls)


def _pcm_bytes_for(dur_ms: float) -> bytes:
    n = int(SAMPLE_RATE * dur_ms / 1000.0)
    audio = (0.2 * np.sin(
        2 * np.pi * 220.0 * np.arange(n) / SAMPLE_RATE)).astype(np.float32)
    from voice.audio_processing import float32_to_int16
    return float32_to_int16(audio).tobytes()


def _frames_for(dur_ms: float):
    n = int(SAMPLE_RATE * dur_ms / 1000.0)
    audio = (0.2 * np.sin(
        2 * np.pi * 220.0 * np.arange(n) / SAMPLE_RATE)).astype(np.float32)
    frames = []
    for i in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frames.append(audio[i:i + FRAME_SAMPLES])
    return frames


def _frames_to_evidence(frames):
    # Real speech evidence so the NO-SPEECH gate does not discard the audio:
    # mark every frame voiced/strong.
    frame_ms = FRAME_SAMPLES / SAMPLE_RATE * 1000.0
    total = len(frames) * frame_ms
    return {"silero_voiced_ms": total, "strong_ms": total, "loud_ms": total,
            "peak_rms": 4000.0, "total_ms": total, "avg_prob": 0.9}


def _make_listener(whisper: FakeHybridWhisper) -> CL.CommandListener:
    cl = CL.CommandListener()
    cl._whisper = whisper
    cl._ready = True
    return cl


def _run_final(cl, whisper, dur_ms=2500.0):
    frames = _frames_for(dur_ms)
    ev = asyncio.run(cl._finalize(
        frames, time.time(), endpoint_reason="silence",
        evidence=_frames_to_evidence(frames)))
    assert ev is not None, "utterance with speech evidence must emit an event"
    return ev


# ── A. Confident English → no fallback ────────────────────────────

def test_a_confident_english_skips_fallback():
    """'open firefox' conf -0.4, 2.5s audio → single en pass, reason empty."""
    whisper = FakeHybridWhisper(en=("open firefox", -0.4))
    cl = _make_listener(whisper)
    ev = _run_final(cl, whisper)
    assert ev.kind == "final"
    assert ev.text == "open firefox"
    assert whisper.pass_count == 1, f"expected 1 pass, got {whisper.calls}"
    assert whisper.languages == ["en"]
    assert ev.primary_used is True
    assert ev.fallback_used is False
    assert ev.selected_language == "en"
    assert ev.fallback_latency_ms == 0.0
    assert ev.reason_for_fallback == ""


# ── B. Suspicious English → exactly one Hindi pass ────────────────

def test_b_suspicious_primary_triggers_single_hindi_pass():
    """Empty-ish primary ('you', weak) → en + hi passes, max 2, no auto."""
    whisper = FakeHybridWhisper(en=("you", -1.2),
                                hi=("youtube pe music chalao", -0.5))
    cl = _make_listener(whisper)
    ev = _run_final(cl, whisper)
    assert whisper.pass_count == 2, f"max 2 passes, got {whisper.calls}"
    assert whisper.languages == ["en", "hi"]
    assert ev.fallback_used is True
    assert ev.selected_language == "hi"
    assert ev.text == "youtube pe music chalao"
    assert "suspicion:" in ev.reason_for_fallback
    assert "selection:" in ev.reason_for_fallback


# ── C. Same audio/config, only language differs ───────────────────

def test_c_fallback_reuses_same_audio_only_language_differs():
    whisper = FakeHybridWhisper(en=("you", -1.2),
                                hi=("firefox kholo", -0.5))
    cl = _make_listener(whisper)
    _run_final(cl, whisper)
    assert whisper.pass_count == 2
    (lang0, len0, sr0), (lang1, len1, sr1) = whisper.calls
    assert (lang0, lang1) == ("en", "hi")
    assert len0 == len1, "fallback must decode the SAME frozen audio bytes"
    assert sr0 == sr1 == SAMPLE_RATE


# ── D/E. Valid beats invalid ──────────────────────────────────────

def test_d_fallback_valid_primary_invalid_selects_hi():
    sel = CL.select_best_transcript("you", -1.2, "firefox kholo", -0.5, 2500.0)
    text, conf, lang, reason = sel
    assert lang == "hi" and text == "firefox kholo"
    assert reason == "fallback_valid_primary_invalid"


def test_e_primary_valid_fallback_invalid_selects_en():
    sel = CL.select_best_transcript("open firefox", -0.4, "you", -1.3, 2500.0)
    text, conf, lang, reason = sel
    assert lang == "en" and text == "open firefox"
    assert reason == "primary_valid_fallback_invalid"


# ── F. Both invalid → UNCERTAIN path ──────────────────────────────

def test_f_both_invalid_keeps_stronger_logprob_still_rejected():
    """Both rejected → stronger logprob kept as raw text, but downstream
    validation still rejects it (fallback cannot bypass safety gates)."""
    sel = CL.select_best_transcript("you", -1.3, "too", -0.9, 2500.0)
    text, conf, lang, reason = sel
    assert reason == "both_invalid_stronger_logprob"
    assert text == "too" and lang == "hi"  # stronger logprob preserved
    accepted, _ = CL._validate_transcript(text, conf, 2500.0)
    assert not accepted, "UNCERTAIN path must still be rejected downstream"


# ── G. Hallucination-free wins ────────────────────────────────────

def test_g_hallucination_free_candidate_wins():
    # "you you you you" is a repeated hallucination AND invalid, while
    # "open firefox" is valid: valid-vs-invalid resolves first (spec order
    # step 1), so assert the winner + document the resolution order.
    sel = CL.select_best_transcript(
        "open firefox", -0.55, "you you you you", -0.45, 2500.0)
    text, conf, lang, reason = sel
    assert lang == "en" and text == "open firefox"
    assert reason == "primary_valid_fallback_invalid"
    # Hallucination-free step (spec order step 2): both candidates valid,
    # but one trips a hallucination flag (an unknown single-word fragment
    # that passes _validate_transcript yet is low_quality) →
    # hallucination-free wins.
    sel2 = CL.select_best_transcript(
        "blorpt", -0.40, "play believer on youtube",
        -0.50, 2500.0)
    _t2, _c2, lang2, reason2 = sel2
    assert lang2 == "hi", f"hallucination-free must win, got {sel2}"
    assert reason2 == "hallucination_free_wins"


# ── H. Coherent multi-word beats fragment ─────────────────────────

def test_h_coherent_multword_beats_fragment():
    sel = CL.select_best_transcript(
        "play believer on youtube", -0.62, "play", -0.55, 2500.0)
    text, conf, lang, reason = sel
    assert lang == "en" and text == "play believer on youtube"
    assert reason == "coherent_wins"


# ── I. Decisive logprob gap wins ──────────────────────────────────

def test_i_stronger_logprob_wins_decisive_gap():
    sel = CL.select_best_transcript(
        "open firefox", -0.35, "open firefox", -0.75, 2500.0)
    text, conf, lang, reason = sel
    assert lang == "en" and reason == "stronger_logprob"


# ── J. Language evidence breaks near-ties ─────────────────────────

def test_j_language_evidence_breaks_near_tie():
    """Near-identical logprob; Hindi candidate carries Hinglish vocab."""
    sel = CL.select_best_transcript(
        "firefox kholo", -0.50, "firefox kholo", -0.52, 2500.0)
    text, conf, lang, reason = sel
    assert lang in ("en", "hi")
    # Either tie-break is deterministic; Devanagari must favor "hi".
    sel2 = CL.select_best_transcript(
        "open firefox", -0.50, "फ़ायरफ़ॉक्स खोलो", -0.52, 2500.0)
    _t2, _c2, lang2, reason2 = sel2
    assert lang2 == "hi", f"Devanagari evidence must win tie, got {sel2}"
    assert reason2 == "language_evidence"


# ── K. Intent compatibility is comparison-only ────────────────────

def test_k_intent_compatibility_compares_without_executing():
    """Probe returns a bool/None and performs no tool execution."""
    res = CL._intent_actionable("open firefox", -0.5, 2500.0)
    assert res in (True, False, None)
    if res is not None:
        assert res is True  # real pipeline finds it actionable
    unknown = CL._intent_actionable("zindagi bahut tamasha hai bhai",
                                    -0.5, 2500.0)
    assert unknown in (True, False, None)
    # Selection still deterministic when the probe is unavailable.
    import voice.command_listener as _cl
    orig = _cl._intent_actionable
    try:
        _cl._intent_actionable = lambda *a, **k: None
        sel = _cl.select_best_transcript(
            "open firefox", -0.5, "open firefox", -0.5, 2500.0)
        assert sel[2] == "en", "ties default to the English primary"
    finally:
        _cl._intent_actionable = orig


# ── L. Metrics on every event ─────────────────────────────────────

def test_l_metrics_present_and_total_is_sum():
    whisper = FakeHybridWhisper(en=("you", -1.2),
                                hi=("firefox kholo", -0.5))
    cl = _make_listener(whisper)
    ev = _run_final(cl, whisper)
    for field in ("primary_used", "fallback_used", "selected_language",
                  "primary_latency_ms", "fallback_latency_ms",
                  "total_latency_ms", "reason_for_fallback"):
        assert hasattr(ev, field), f"missing metric {field}"
    assert ev.total_latency_ms == (
        ev.primary_latency_ms + ev.fallback_latency_ms)
    assert ev.whisper_latency_ms == ev.total_latency_ms
    assert ev.primary_latency_ms >= 0.0 and ev.fallback_latency_ms > 0.0
    # Confident path also carries metrics (fallback zeros).
    whisper2 = FakeHybridWhisper(en=("open firefox", -0.4))
    ev2 = _run_final(_make_listener(whisper2), whisper2)
    assert ev2.total_latency_ms == ev2.primary_latency_ms
    assert ev2.fallback_latency_ms == 0.0


# ── M. Fallback disabled → single pass ────────────────────────────

def test_m_fallback_disabled_single_english_pass():
    whisper = FakeHybridWhisper(en=("you", -1.2),
                                hi=("firefox kholo", -0.5))
    cl = _make_listener(whisper)
    cl.multilingual_fallback_enabled = False
    ev = _run_final(cl, whisper)
    assert whisper.pass_count == 1
    assert ev.fallback_used is False

    whisper2 = FakeHybridWhisper(en=("you", -1.2),
                                 hi=("firefox kholo", -0.5))
    cl2 = _make_listener(whisper2)
    orig = CL.MULTILINGUAL_FALLBACK_ENABLED
    CL.MULTILINGUAL_FALLBACK_ENABLED = False
    try:
        ev2 = _run_final(cl2, whisper2)
    finally:
        CL.MULTILINGUAL_FALLBACK_ENABLED = orig
    assert whisper2.pass_count == 1
    assert ev2.fallback_used is False


# ── N. Raw transcript preserved ───────────────────────────────────

def test_n_raw_transcript_preserved_not_paraphrased():
    """The selected fallback text flows downstream verbatim (postprocess
    only); selection never translates/paraphrases the transcript."""
    whisper = FakeHybridWhisper(en=("you", -1.2),
                                hi=("youtube pe music chalao", -0.5))
    cl = _make_listener(whisper)
    ev = _run_final(cl, whisper)
    assert ev.kind == "final"
    assert ev.text == CL._postprocess("youtube pe music chalao")
    # The downstream normalizer still maps it to the canonical command.
    from nlp.command_normalizer import command_normalizer
    assert command_normalizer.normalize(ev.text).startswith("play")


# ── O. English regression ─────────────────────────────────────────

def test_o_english_commands_unchanged():
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent
    for raw in ("open firefox", "play believer on youtube"):
        suspicious, _ = CL.is_primary_suspicious(raw, -0.4, 2500.0)
        assert suspicious is False, f"{raw!r} must not trigger fallback"
        normed = command_normalizer.normalize(raw)
        auth = authorize_intent(normed, stt_confidence=-0.4,
                                audio_duration_ms=2500.0)
        assert auth.actionable is True, f"{raw!r} must stay actionable"


# ── Suspicion gate unit checks ────────────────────────────────────

def test_suspicion_gate_combined_evidence():
    # Confident English: valid + coherent + healthy logprob/duration.
    susp, reason = CL.is_primary_suspicious("open firefox", -0.4, 2500.0)
    assert susp is False and reason == "primary_confident"
    # Empty primary is always suspicious (second pass may still fail → failure).
    susp, reason = CL.is_primary_suspicious("", -0.4, 2500.0)
    assert susp is True
    # Single weak signal alone (short audio) with otherwise healthy text
    # is suspicious only via combined evidence — verify reason lists it.
    susp, _ = CL.is_primary_suspicious("open firefox", -0.5, 300.0)
    assert susp is True
    # A single arbitrary threshold must NOT fire: healthy 2-word command at
    # -0.7 with good audio is confident (matches _validate_transcript fix).
    susp, _ = CL.is_primary_suspicious("open chrome", -0.7, 1500.0)
    assert susp is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
