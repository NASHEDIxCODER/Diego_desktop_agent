"""Tests for the transcript quality gate."""

from __future__ import annotations

import pytest

from nlp.transcript_quality import (
    TranscriptVerdict,
    evaluate_transcript_quality,
    is_transcript_coherent,
)


class TestMalformedTranscriptRejected:
    def test_broken_transcript_low_confidence(self):
        quality = evaluate_transcript_quality(
            "but do know about that web",
            stt_confidence=-0.299,
            audio_duration_ms=1500.0,
        )
        assert not quality.can_reach_rag
        assert not quality.can_reach_memory
        assert not quality.can_reach_web
        assert not quality.can_reach_llm
        assert not quality.can_reach_execution

    def test_broken_transcript_medium_confidence(self):
        quality = evaluate_transcript_quality(
            "can do it to yourself",
            stt_confidence=-0.890,
            audio_duration_ms=1500.0,
        )
        assert not quality.can_reach_rag
        assert not quality.can_reach_memory
        assert not quality.can_reach_web

    def test_deep_hallucination_rejected(self):
        quality = evaluate_transcript_quality(
            "Ending your soul.",
            stt_confidence=-1.423,
            audio_duration_ms=1000.0,
        )
        assert quality.verdict == TranscriptVerdict.REJECTED
        assert not quality.can_reach_rag

    def test_empty_transcript_rejected(self):
        quality = evaluate_transcript_quality("", stt_confidence=0.0)
        assert quality.verdict == TranscriptVerdict.REJECTED

    def test_single_char_rejected(self):
        quality = evaluate_transcript_quality("a", stt_confidence=0.0)
        assert quality.verdict == TranscriptVerdict.REJECTED

    def test_filler_heavy_rejected(self):
        quality = evaluate_transcript_quality(
            "but so and the a to of in on",
            stt_confidence=-0.5,
            audio_duration_ms=2000.0,
        )
        assert not quality.can_reach_rag
        assert not quality.can_reach_execution

    def test_suspicious_pattern_rejected(self):
        quality = evaluate_transcript_quality(
            "but but do do",
            stt_confidence=-0.6,
            audio_duration_ms=1500.0,
        )
        assert not quality.can_reach_rag


class TestValidConversationAccepted:
    def test_can_you_improve_yourself(self):
        quality = evaluate_transcript_quality(
            "Can you improve yourself?",
            stt_confidence=-0.4,
            audio_duration_ms=1500.0,
        )
        assert quality.can_reach_llm
        assert quality.can_reach_rag

    def test_what_do_you_think_about_my_project(self):
        quality = evaluate_transcript_quality(
            "What do you think about my project?",
            stt_confidence=-0.3,
            audio_duration_ms=2000.0,
        )
        assert quality.can_reach_llm
        assert quality.can_reach_rag

    def test_tell_me_about_my_architecture(self):
        quality = evaluate_transcript_quality(
            "Tell me about my architecture.",
            stt_confidence=-0.35,
            audio_duration_ms=1800.0,
        )
        assert quality.can_reach_llm

    def test_how_does_this_work(self):
        quality = evaluate_transcript_quality(
            "How does this work?",
            stt_confidence=-0.25,
            audio_duration_ms=1200.0,
        )
        assert quality.can_reach_llm

    def test_why_is_this_happening(self):
        quality = evaluate_transcript_quality(
            "Why is this happening?",
            stt_confidence=-0.3,
            audio_duration_ms=1500.0,
        )
        assert quality.can_reach_llm


class TestValidCommandAccepted:
    def test_open_firefox(self):
        quality = evaluate_transcript_quality(
            "Open Firefox",
            stt_confidence=-0.2,
            audio_duration_ms=1000.0,
        )
        assert quality.can_reach_execution
        assert quality.verdict == TranscriptVerdict.COMMAND_ACCEPTED

    def test_search_for_python(self):
        quality = evaluate_transcript_quality(
            "Search for Python",
            stt_confidence=-0.3,
            audio_duration_ms=1200.0,
        )
        assert quality.can_reach_web

    def test_what_is_on_my_screen(self):
        quality = evaluate_transcript_quality(
            "What is on my screen?",
            stt_confidence=-0.25,
            audio_duration_ms=1500.0,
        )
        assert quality.can_reach_rag


class TestFollowUpProtection:
    def test_short_follow_up_with_active_task(self):
        quality = evaluate_transcript_quality(
            "do it",
            stt_confidence=-0.5,
            audio_duration_ms=500.0,
            has_active_task=True,
        )
        assert quality.can_reach_execution
        assert quality.verdict == TranscriptVerdict.COMMAND_ACCEPTED

    def test_continue_with_active_task(self):
        quality = evaluate_transcript_quality(
            "continue",
            stt_confidence=-0.4,
            audio_duration_ms=400.0,
            has_active_task=True,
        )
        assert quality.can_reach_execution

    def test_yes_with_active_task(self):
        quality = evaluate_transcript_quality(
            "yes",
            stt_confidence=-0.3,
            audio_duration_ms=300.0,
            has_active_task=True,
        )
        assert quality.can_reach_execution

    def test_short_phrase_without_active_task(self):
        quality = evaluate_transcript_quality(
            "do it",
            stt_confidence=-0.5,
            audio_duration_ms=500.0,
            has_active_task=False,
        )
        assert not quality.can_reach_execution


class TestConfidenceRejection:
    def test_very_low_confidence_rejected(self):
        quality = evaluate_transcript_quality(
            "hello world",
            stt_confidence=-1.5,
            audio_duration_ms=1000.0,
        )
        assert quality.verdict == TranscriptVerdict.REJECTED

    def test_medium_confidence_coherent_accepted(self):
        quality = evaluate_transcript_quality(
            "open firefox",
            stt_confidence=-0.6,
            audio_duration_ms=1000.0,
        )
        assert quality.can_reach_execution

    def test_high_confidence_broken_rejected(self):
        quality = evaluate_transcript_quality(
            "but do know about that web",
            stt_confidence=-0.299,
            audio_duration_ms=1500.0,
        )
        assert not quality.can_reach_rag
        assert not quality.can_reach_execution


class TestConvenienceFunction:
    def test_valid_returns_true(self):
        coherent, reason = is_transcript_coherent(
            "Open Firefox",
            stt_confidence=-0.2,
            audio_duration_ms=1000.0,
        )
        assert coherent

    def test_invalid_returns_false(self):
        coherent, reason = is_transcript_coherent(
            "but do know about that web",
            stt_confidence=-0.299,
            audio_duration_ms=1500.0,
        )
        assert not coherent


class TestEdgeCases:
    def test_no_confidence_no_duration(self):
        quality = evaluate_transcript_quality("Open Firefox")
        assert quality.can_reach_execution

    def test_no_confidence_broken_text(self):
        quality = evaluate_transcript_quality("but so and the")
        assert not quality.can_reach_execution

    def test_long_valid_question(self):
        quality = evaluate_transcript_quality(
            "Can you tell me about the architecture of my Diego project?",
            stt_confidence=-0.4,
            audio_duration_ms=4000.0,
        )
        assert quality.can_reach_rag
        assert quality.can_reach_llm
