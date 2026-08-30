"""
IntentGate - transcript/intent sanity gate before planner/tool execution.

ROOT CAUSE (2026-08-30 runtime log):
    Whisper -> "Hello dear" (confidence=-1.168) -> transcript accepted
    -> THINK -> planner -> get_time + type_text("Hello dear") -> 29s turn.

A low-quality or implausible transcript must NEVER become arbitrary
desktop actions. It may only become:
    - a conversational response (greeting / small talk), or
    - a clarification ("Could you say that again?"), or
    - a silent discard.

Evidence sources combined (no blind threshold changes):
  1. TRANSCRIPT QUALITY  - greeting structure, command structure.
  2. SPEECH EVIDENCE     - audio duration of the captured utterance.
  3. INTENT CONFIDENCE   - Whisper avg_logprob (real speech -0.2..-0.9,
     hallucinations -0.8..-1.3 on this system).

Verdicts: "command" | "conversational" | "uncertain".
Dependency-light: no voice/audio imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Confidence bands (Whisper avg_logprob on this system)
CONF_HARD_REJECT = -1.0
CONF_SOFT_REJECT = -0.6
SHORT_AUDIO_MS = 600.0

_GREETING_WORDS = frozenset({
    "hello", "hi", "hey", "yo", "sup", "hiya", "hola", "greetings",
})
_GREETING_PREFIXES = (
    "good morning", "good afternoon", "good evening", "good day",
)
_SMALLTALK_PHRASES = (
    "what's up", "whats up", "what is up", "how are you",
    "how's it going", "hows it going", "how are things",
    "nice to meet you", "long time no see",
)

_ACTION_VERBS = frozenset({
    "open", "close", "play", "pause", "resume", "stop", "start", "run",
    "launch", "kill", "search", "find", "look", "set", "change", "switch",
    "scroll", "click", "type", "write", "press", "send", "create",
    "delete", "remove", "install", "uninstall", "mute", "unmute", "lock",
    "shutdown", "restart", "reboot", "volume", "brightness", "next",
    "previous", "skip", "shuffle", "repeat", "read", "show", "list",
    "tell", "give", "make", "call", "message", "email",
    "minimize", "maximize", "summarize", "translate", "calculate",
})
_QUESTION_CUES = frozenset({
    "what", "whats", "what's", "who", "when", "where", "why", "how",
    "which", "is", "are", "can", "could", "does", "do", "did", "will",
    "would", "should", "wheres", "hows",
})


@dataclass
class IntentVerdict:
    """Result of the transcript/intent sanity gate."""

    mode: str                       # "command" | "conversational" | "uncertain"
    tool_execution_allowed: bool
    reason: str = ""

    @property
    def is_conversational(self) -> bool:
        return self.mode == "conversational"

    @property
    def is_uncertain(self) -> bool:
        return self.mode == "uncertain"


def has_command_structure(text: str) -> bool:
    """True when the transcript looks like a command or a real question.

    A transcript with an action verb or an interrogative is plausible
    enough to route through the normal decision engine.
    """
    t = " ".join((text or "").lower().split())
    if not t:
        return False
    for w in t.split():
        w = w.strip(".,!?;:'\"")
        if w in _ACTION_VERBS or w in _QUESTION_CUES:
            return True
    return False


def is_greeting(text: str) -> bool:
    """True for short greeting/small-talk phrases ("hello dear",
    "hey there", "good morning", "hi diego").

    A transcript containing an action verb or a question is NEVER a
    greeting ("hey open firefox" is a command, not a greeting).
    """
    t = " ".join((text or "").lower().strip().strip(".,!?").split())
    if not t:
        return False
    if has_command_structure(t):
        return False
    words = t.split()
    if not words:
        return False
    if words[0].strip(".,!?") in _GREETING_WORDS and len(words) <= 3:
        return True
    if any(t.startswith(p) for p in _GREETING_PREFIXES) and len(words) <= 4:
        return True
    if t in _SMALLTALK_PHRASES:
        return True
    return False


def evaluate_intent(
    text: str,
    stt_confidence: Optional[float] = None,
    audio_duration_ms: Optional[float] = None,
) -> IntentVerdict:
    """Evaluate whether a transcript may drive tool execution.

    Combines transcript quality, speech evidence, and STT confidence.
    Never blindly raises/lowers thresholds - each rejection requires
    converging evidence (low confidence AND no command structure, or
    implausibly short audio for the transcript length).
    """
    t = (text or "").strip()
    if not t:
        return IntentVerdict("uncertain", False, "empty transcript")

    # Greetings / small talk -> conversational response, never tools.
    if is_greeting(t):
        return IntentVerdict(
            mode="conversational",
            tool_execution_allowed=False,
            reason="greeting/small-talk",
        )

    words = t.split()
    commandish = has_command_structure(t)

    # No STT confidence evidence (typed input, internal callers):
    # trust the text path - downstream routing still applies.
    if stt_confidence is None:
        return IntentVerdict("command", True, "no stt evidence (trusted path)")

    # HARD REJECT: Whisper hallucination band. Real commands essentially
    # never score below -1.0 on this system. Greetings are exempt (a
    # greeting response is harmless even for a mis-heard "hello dear").
    if stt_confidence < CONF_HARD_REJECT:
        return IntentVerdict(
            "uncertain", False,
            "confidence %.3f < %.1f (hallucination band)"
            % (stt_confidence, CONF_HARD_REJECT))

    # Soft reject: low confidence AND implausibly short audio AND a tiny
    # transcript AND no command structure.
    if (stt_confidence < CONF_SOFT_REJECT
            and audio_duration_ms is not None
            and audio_duration_ms < SHORT_AUDIO_MS
            and len(words) <= 2
            and not commandish):
        return IntentVerdict(
            "uncertain", False,
            "confidence %.3f with only %.0fms audio and no command structure"
            % (stt_confidence, audio_duration_ms))

    return IntentVerdict("command", True, "plausible transcript")


def transcript_allows_tool_execution(
    text: str,
    stt_confidence: Optional[float] = None,
    audio_duration_ms: Optional[float] = None,
) -> bool:
    """Convenience: True only when the transcript may drive tools."""
    return evaluate_intent(
        text, stt_confidence=stt_confidence,
        audio_duration_ms=audio_duration_ms,
    ).tool_execution_allowed
