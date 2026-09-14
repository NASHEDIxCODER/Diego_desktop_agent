"""
Transcript Quality Gate - deterministic ASR transcript validation BEFORE routing.

ROOT CAUSE (2026-09-14 runtime evidence):
    ASR transcripts like "but do know about that web" (conf=-0.299) and
    "can do it to yourself" (conf=-0.890) were accepted and entered the
    command pipeline, reaching RAG, session memory, web search, and LLM.

This module provides a TWO-STAGE acceptance model:

    ASR_ACCEPTED  - acoustically decoded, passed basic VAD/STT thresholds
    COMMAND_ACCEPTED - semantically coherent enough to drive routing

A transcript may be ASR_ACCEPTED but still fail COMMAND_ACCEPTED.
Such transcripts get a clarification response and NEVER reach:
    - semantic RAG
    - session memory retrieval
    - web search
    - deterministic execution
    - browser control
    - LLM factual answer

Design: cheap deterministic/heuristic checks first. No LLM.
Dependency-light: no voice/audio imports (safe for unit tests).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TranscriptVerdict(str, Enum):
    """How the quality gate classified a transcript."""
    COMMAND_ACCEPTED = "COMMAND_ACCEPTED"
    ASR_ACCEPTED = "ASR_ACCEPTED"
    REJECTED = "REJECTED"


CONF_HARD_REJECT = -1.2
CONF_SUSPICIOUS = -0.8
MIN_AUDIO_MS_PER_WORD = 80.0
MAX_WORDS_PER_100MS = 1.5

_HALLUCINATION_WORDS = frozenset({
    "but", "so", "and", "the", "a", "an", "to", "of", "in", "on",
    "at", "for", "with", "by", "from", "is", "it", "that", "this",
    "was", "are", "be", "do", "did", "can", "could", "would", "should",
})

_HALLUCINATION_PATTERNS = [
    re.compile(r"\b(but|so|and|then)\s+(but|so|and|then)\b"),
    re.compile(r"\b(do|did|can|could|would|should)\s+(do|did|can|could|would|should)\b"),
    re.compile(r"\b(to|for|with|from)\s+(to|for|with|from)\b"),
]

_CONTENT_VERBS = frozenset({
    "open", "close", "play", "pause", "stop", "start", "run", "launch",
    "search", "find", "look", "set", "change", "switch", "click", "type",
    "write", "press", "send", "create", "delete", "remove", "install",
    "mute", "lock", "shutdown", "restart", "reboot", "volume", "brightness",
    "minimize", "maximize", "summarize", "translate", "calculate", "show",
    "tell", "give", "make", "call", "message", "email", "read", "list",
    "navigate", "go", "visit", "browse", "scroll", "select", "copy",
    "paste", "save", "load", "import", "export", "download", "upload",
    "connect", "disconnect", "enable", "disable", "configure", "update",
    "improve", "fix", "check", "get", "put", "add", "move", "rename",
})

_CONTENT_NOUNS = frozenset({
    "file", "folder", "document", "image", "video", "audio", "music",
    "screen", "window", "tab", "page", "browser", "firefox", "chrome",
    "app", "application", "program", "process", "system", "computer",
    "project", "code", "python", "architecture", "design", "component",
    "camera", "microphone", "speaker", "network", "internet", "web",
    "site", "url", "link", "weather", "time", "date", "calendar",
    "email", "note", "reminder", "task", "goal", "plan", "idea",
    "question", "answer", "problem", "feature", "function", "method",
    "class", "module", "library", "database", "server", "client",
    "user", "account", "profile", "settings", "configuration",
})


def _content_word_ratio(text: str) -> float:
    """Fraction of words that are content words (not just filler)."""
    words = [w.lower().strip(".,!?;:'\"()") for w in (text or "").split()]
    if not words:
        return 0.0
    content_count = sum(1 for w in words if w in _CONTENT_VERBS or w in _CONTENT_NOUNS)
    return content_count / len(words)


def _has_suspicious_pattern(text: str) -> bool:
    """Check for known ASR hallucination patterns."""
    t = " ".join((text or "").lower().split())
    for pattern in _HALLUCINATION_PATTERNS:
        if pattern.search(t):
            return True
    return False


def _filler_heavy(text: str) -> bool:
    """True when the transcript is mostly filler words."""
    words = [w.lower().strip(".,!?;:'\"()") for w in (text or "").split() if len(w) > 1]
    if len(words) <= 2:
        return False
    filler_count = sum(1 for w in words if w in _HALLUCINATION_WORDS)
    return (filler_count / len(words)) > 0.7


def _starts_with_filler(text: str) -> bool:
    """True when the transcript starts with a filler word (but, so, and).
    
    Real commands and questions rarely start with filler words.
    ASR hallucinations often do: "but do know about that web".
    
    Note: "do it", "can you", "would you" are valid phrases where
    the first word is a verb/auxiliary, not a filler.
    """
    words = (text or "").strip().lower().split()
    if not words:
        return False
    first = words[0].strip(".,!?;:'\"()")
    # These are only fillers when NOT followed by a pronoun/object
    filler_only = {"but", "so", "and", "then", "or"}
    if first in filler_only:
        return True
    # "do", "did", "can", "could", "would" are fillers only when followed
    # by another filler word (e.g., "do know", "can do")
    weak_fillers = {"do", "did", "can", "could", "would"}
    if first in weak_fillers and len(words) >= 2:
        second = words[1].strip(".,!?;:'\"()")
        if second in _HALLUCINATION_WORDS and second not in {"it", "you", "me", "us", "that", "this"}:
            return True
    return False


def _has_ambiguous_pronoun(text: str) -> bool:
    """True when the transcript contains ambiguous pronouns without context.
    
    Phrases like 'do it', 'open that', 'close it' are ambiguous without
    an active task providing context for what 'it'/'that' refers to.
    """
    words = [w.lower().strip(".,!?;:'\"()") for w in (text or "").split()]
    if len(words) > 3:
        return False  # Longer phrases have enough context
    ambiguous = {"it", "that", "this", "them", "those", "these"}
    return any(w in ambiguous for w in words)


def _has_valid_structure(text: str) -> bool:
    """Check if the transcript has a valid grammatical structure.
    
    Valid structures:
    - Imperative: verb + object ("open firefox", "search for python")
    - Question: question word + verb + object ("how does this work")
    - Statement: subject + verb ("I want to know")
    
    Invalid structures:
    - Filler + verb + filler + noun ("but do know about that web")
    - Repeated function words without clear structure
    """
    words = [w.lower().strip(".,!?;:'\"()") for w in (text or "").split() if len(w) > 1]
    if len(words) < 2:
        return True  # Short phrases are ambiguous, not necessarily invalid
    
    # Check for imperative structure: starts with a verb
    if words[0] in _CONTENT_VERBS:
        return True
    
    # Check for question structure: starts with question word
    question_starters = {"what", "whats", "what's", "who", "when", "where", "why", "how", "which", "is", "are", "can", "could", "does", "do", "did", "will", "would", "should"}
    if words[0] in question_starters:
        return True
    
    # Check for "tell me" pattern
    if len(words) >= 2 and words[0] == "tell" and words[1] == "me":
        return True
    
    # Check for "I want/need/like" pattern
    if words[0] in {"i", "we", "my"}:
        return True
    
    # If starts with filler and has no clear verb, likely broken
    if _starts_with_filler(text):
        # Check if there is a real verb after the filler
        has_verb = any(w in _CONTENT_VERBS for w in words[1:])
        if not has_verb:
            return False
        # Even with a verb, starting with filler is suspicious for short texts
        if len(words) <= 4:
            return False
    
    return True


@dataclass
class TranscriptQuality:
    """Result of the transcript quality gate."""
    verdict: TranscriptVerdict
    is_coherent: bool
    confidence: Optional[float] = None
    reason: str = ""
    coherence_score: float = 0.0

    @property
    def can_reach_rag(self) -> bool:
        return self.verdict == TranscriptVerdict.COMMAND_ACCEPTED

    @property
    def can_reach_memory(self) -> bool:
        return self.verdict == TranscriptVerdict.COMMAND_ACCEPTED

    @property
    def can_reach_web(self) -> bool:
        return self.verdict == TranscriptVerdict.COMMAND_ACCEPTED

    @property
    def can_reach_llm(self) -> bool:
        return self.verdict == TranscriptVerdict.COMMAND_ACCEPTED

    @property
    def can_reach_execution(self) -> bool:
        return self.verdict == TranscriptVerdict.COMMAND_ACCEPTED


def evaluate_transcript_quality(
    text: str,
    stt_confidence: Optional[float] = None,
    audio_duration_ms: Optional[float] = None,
    has_active_task: bool = False,
) -> TranscriptQuality:
    """Evaluate transcript coherence BEFORE routing to expensive layers."""
    t = (text or "").strip()
    if not t:
        return TranscriptQuality(
            verdict=TranscriptVerdict.REJECTED,
            is_coherent=False,
            confidence=stt_confidence,
            reason="empty transcript",
            coherence_score=0.0,
        )

    words = t.split()
    word_count = len(words)

    if len(t) <= 1:
        return TranscriptQuality(
            verdict=TranscriptVerdict.REJECTED,
            is_coherent=False,
            confidence=stt_confidence,
            reason="too short",
            coherence_score=0.0,
        )

    if stt_confidence is not None and stt_confidence < CONF_HARD_REJECT:
        return TranscriptQuality(
            verdict=TranscriptVerdict.REJECTED,
            is_coherent=False,
            confidence=stt_confidence,
            reason=f"confidence {stt_confidence:.3f} < {CONF_HARD_REJECT}",
            coherence_score=0.0,
        )

    coherence_signals = 0.0
    total_signals = 0.0

    total_signals += 1.0
    content_ratio = _content_word_ratio(t)
    if content_ratio >= 0.3:
        coherence_signals += 1.0
    elif content_ratio >= 0.15:
        coherence_signals += 0.5

    total_signals += 1.0
    if not _has_suspicious_pattern(t):
        coherence_signals += 1.0

    total_signals += 1.0
    if not _filler_heavy(t):
        coherence_signals += 1.0

    total_signals += 1.5
    if _has_valid_structure(t):
        coherence_signals += 1.5
    elif _starts_with_filler(t):
        coherence_signals -= 0.75  # Strong penalty for broken structure

    if audio_duration_ms is not None and audio_duration_ms > 0:
        total_signals += 1.0
        expected_min_ms = word_count * MIN_AUDIO_MS_PER_WORD
        if audio_duration_ms >= expected_min_ms * 0.7:
            coherence_signals += 1.0
        words_per_100ms = (word_count / audio_duration_ms) * 100
        if words_per_100ms <= MAX_WORDS_PER_100MS:
            coherence_signals += 0.5
            total_signals += 0.5

    if stt_confidence is not None:
        total_signals += 1.0
        if stt_confidence >= -0.5:
            coherence_signals += 1.0
        elif stt_confidence >= -0.8:
            coherence_signals += 0.5

    coherence_score = coherence_signals / total_signals if total_signals > 0 else 0.5

    # Hard rule: broken structure with filler start is always suspicious
    if not _has_valid_structure(t) and _starts_with_filler(t) and not has_active_task:
        coherence_score = min(coherence_score, 0.4)

    # Hard rule: ambiguous pronouns without active task context
    if _has_ambiguous_pronoun(t) and not has_active_task and word_count <= 3:
        coherence_score = min(coherence_score, 0.4)

    # Active task context override
    if has_active_task and word_count <= 3:
        return TranscriptQuality(
            verdict=TranscriptVerdict.COMMAND_ACCEPTED,
            is_coherent=True,
            confidence=stt_confidence,
            reason="short follow-up with active task",
            coherence_score=0.8,
        )

    if coherence_score >= 0.6:
        return TranscriptQuality(
            verdict=TranscriptVerdict.COMMAND_ACCEPTED,
            is_coherent=True,
            confidence=stt_confidence,
            reason=f"coherent ({coherence_score:.2f})",
            coherence_score=coherence_score,
        )

    if coherence_score >= 0.35:
        return TranscriptQuality(
            verdict=TranscriptVerdict.ASR_ACCEPTED,
            is_coherent=False,
            confidence=stt_confidence,
            reason=f"weak coherence ({coherence_score:.2f})",
            coherence_score=coherence_score,
        )

    return TranscriptQuality(
        verdict=TranscriptVerdict.REJECTED,
        is_coherent=False,
        confidence=stt_confidence,
        reason=f"low coherence ({coherence_score:.2f})",
        coherence_score=coherence_score,
    )


def is_transcript_coherent(
    text: str,
    stt_confidence: Optional[float] = None,
    audio_duration_ms: Optional[float] = None,
    has_active_task: bool = False,
) -> Tuple[bool, str]:
    """Convenience: (is_coherent, reason)."""
    quality = evaluate_transcript_quality(
        text, stt_confidence=stt_confidence,
        audio_duration_ms=audio_duration_ms,
        has_active_task=has_active_task,
    )
    return quality.is_coherent, quality.reason
