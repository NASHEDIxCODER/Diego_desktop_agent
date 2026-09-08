"""
Phase 19B - regression tests for the transcript hallucination guard.

Live-audit evidence (debug/asr_accent_audit_results.json) showed two real
Whisper hallucination loops that PASSED _validate_transcript and reached the
Brain:

  #9  "Up and high, up and high, ..." x56 (alternating phrase loop)
  #12 "Play, play, play on YouTube Play, play on youtube" (comma-separated)

Root cause: _is_repeated_hallucination only matched WHITESPACE-separated
identical words, so punctuation broke the match and alternating phrase
loops were never detected. The fix normalizes punctuation and detects
repeated 2-4 word n-grams (3+ occurrences).

Synthetic transcripts only - no audio, no models, no network.
"""

import compat  # noqa: F401

from voice.command_listener import (
    _is_repeated_hallucination,
    _validate_transcript,
    is_low_quality_transcript,
)


# ── Hallucination loops observed LIVE in the Phase 19B audit ──────

def test_comma_separated_repeat_rejected():
    """Record #12: comma-separated word loop must be a hallucination."""
    t = "Play, play, play on YouTube Play, play on youtube"
    assert _is_repeated_hallucination(t) is True
    accepted, reason = _validate_transcript(t, -0.804, 6816.0)
    assert accepted is False
    assert reason == "GARBAGE"


def test_alternating_phrase_loop_rejected():
    """Record #9: alternating n-gram loop must be a hallucination."""
    t = ", ".join(["Up and high"] * 56)
    assert _is_repeated_hallucination(t) is True
    accepted, reason = _validate_transcript(t, -0.127, 5376.0)
    assert accepted is False
    assert reason == "GARBAGE"


def test_consecutive_word_repeat_still_rejected():
    """Original behaviour preserved: 'you you you you'."""
    assert _is_repeated_hallucination("you you you you") is True
    assert _is_repeated_hallucination("I'm sorry, sorry, sorry") is True


def test_single_word_loop_still_rejected():
    assert _is_repeated_hallucination("okay okay okay") is True


def test_period_separated_loop_rejected():
    assert _is_repeated_hallucination(
        "It's 6 o'clock. It's 6 o'clock. It's 6 o'clock.") is True


# ── Control: legitimate speech must NOT be flagged ────────────────

import pytest  # noqa: E402


@pytest.mark.parametrize("text", [
    "open firefox",
    "play believer on youtube",
    "what's my ram?",
    "what time is it?",
    "open the file manager",
    "can you open the terminal and check the system information for me?",
    "hey Diego, how are you today?",
])
def test_normal_speech_not_flagged(text):
    assert _is_repeated_hallucination(text) is False


@pytest.mark.parametrize("text", [
    "open firefox",
    "play believer on youtube",
    "what's my ram?",
])
def test_normal_speech_accepted_with_typical_confidence(text):
    """Real commands at this system's typical confidence (-0.4..-0.9)
    with healthy audio duration must still be accepted."""
    accepted, reason = _validate_transcript(text, -0.7, 2500.0)
    assert accepted is True
    assert reason == ""


def test_low_quality_guard_unchanged_for_normal_speech():
    """Multi-word normal speech is never rejected by the cheap guard."""
    assert is_low_quality_transcript("open firefox") is False
    assert is_low_quality_transcript("play believer on youtube") is False
