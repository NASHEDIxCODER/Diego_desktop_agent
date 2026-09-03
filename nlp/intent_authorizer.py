"""
IntentAuthorizer — FINAL transcript → intent → action authorization boundary.

ROOT CAUSE (2026-08-30 runtime logs):
    - "Hello dear" (conf=-1.168) reached the planner and executed tools.
    - "Can you see my screen?" was normalized to "see my" and executed
      unrelated actions.
    - "Diego opened the tomb" reached the planner and caused close_app.

This module is the SINGLE authorization boundary between STT/normalization
and the dispatcher/planner. A transcript is classified into exactly one of:

    DETERMINISTIC_COMMAND  - imperative desktop command ("open firefox")
    VISION_COMMAND         - screen-dependent request ("what's on my screen?")
    SEARCH_REQUEST         - explicit web search ("search github for X")
    CONVERSATIONAL         - greeting / small talk / statement ("how are you")
    KNOWLEDGE_QUESTION     - factual question ("tell me about animal DNA")
    FOLLOW_UP              - context-dependent reply ("continue", "yes")
    MULTI_STEP_TASK        - compound request ("open firefox and search X")
    UNCERTAIN              - garbage / ambiguous / low-evidence transcript

ONLY actionable categories may reach the dispatcher or planner.
CONVERSATIONAL / KNOWLEDGE_QUESTION may reach the LLM for a spoken answer
but NEVER desktop actions. UNCERTAIN gets a clarification and pays for
NOTHING (no perception, no search, no planner, no LLM, no tools).

Evidence fusion (never a blind confidence threshold):
    - transcript confidence   (Whisper avg_logprob)
    - speech duration         (captured utterance length)
    - VAD evidence            (optional, from the command listener)
    - transcript coherence    (word count, structure, garbage patterns)
    - deterministic pattern   (command/vision/search pattern match)
    - intent confidence       (fused 0..1 score)

Dependency-light: no voice/audio imports (safe for unit tests).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """The eight authorized intent categories."""

    DETERMINISTIC_COMMAND = "DETERMINISTIC_COMMAND"
    VISION_COMMAND = "VISION_COMMAND"
    SEARCH_REQUEST = "SEARCH_REQUEST"
    CONVERSATIONAL = "CONVERSATIONAL"
    KNOWLEDGE_QUESTION = "KNOWLEDGE_QUESTION"
    FOLLOW_UP = "FOLLOW_UP"
    MULTI_STEP_TASK = "MULTI_STEP_TASK"
    LOCAL_KNOWLEDGE = "LOCAL_KNOWLEDGE"
    UNCERTAIN = "UNCERTAIN"


# Categories that may reach the dispatcher / planner.
ACTIONABLE_CATEGORIES = frozenset({
    IntentCategory.DETERMINISTIC_COMMAND,
    IntentCategory.VISION_COMMAND,
    IntentCategory.SEARCH_REQUEST,
    IntentCategory.FOLLOW_UP,
    IntentCategory.MULTI_STEP_TASK,
    IntentCategory.LOCAL_KNOWLEDGE,
})

# Categories that may reach the LLM for a spoken answer (never tools).
LLM_CATEGORIES = frozenset({
    IntentCategory.CONVERSATIONAL,
    IntentCategory.KNOWLEDGE_QUESTION,
    IntentCategory.MULTI_STEP_TASK,
    IntentCategory.DETERMINISTIC_COMMAND,
    IntentCategory.VISION_COMMAND,
    IntentCategory.SEARCH_REQUEST,
    IntentCategory.FOLLOW_UP,
    IntentCategory.LOCAL_KNOWLEDGE,
})


# ── Confidence bands (Whisper avg_logprob on this system) ──────
# Real speech scores -0.2..-0.9; hallucinations -0.8..-1.3.
CONF_HARD_REJECT = -1.0     # hallucination band for non-greetings
CONF_SOFT_REJECT = -0.6
SHORT_AUDIO_MS = 600.0
CONF_DEEP_HALLUCINATION = -1.3   # even greetings are suspect below this


# ── Greeting / small-talk detection ────────────────────────────
_GREETING_WORDS = frozenset({
    "hello", "hi", "hey", "yo", "sup", "hiya", "hola", "greetings",
})
_GREETING_PREFIXES = (
    "good morning", "good afternoon", "good evening", "good day",
)
_SMALLTALK_PHRASES = (
    "what's up", "whats up", "what is up", "how are you",
    "how's it going", "hows it going", "how are things",
    "nice to meet you", "long time no see", "how do you do",
    "who are you", "what is your name", "whats your name",
    "what can you do", "i am bored", "i'm bored", "tell me a joke",
    "tell me about yourself",
)

# ── Deterministic command verbs (imperative first word) ────────
_COMMAND_VERBS = frozenset({
    "open", "close", "play", "pause", "resume", "stop", "start",
    "run", "launch", "kill", "search", "find", "look", "set",
    "change", "switch", "scroll", "click", "type", "write",
    "press", "send", "create", "delete", "remove", "install",
    "uninstall", "mute", "unmute", "lock", "shutdown", "restart",
    "reboot", "volume", "brightness", "next", "previous", "skip",
    "shuffle", "repeat", "minimize", "maximize", "screenshot",
    "wifi", "wi-fi", "bluetooth", "google", "browse", "navigate",
    "summarize", "translate", "calculate", "read", "show", "list",
    "tell", "give", "make", "call", "message", "email",
})

# ── Deterministic live queries (routed by the decision engine) ──
_DETERMINISTIC_QUERIES: Tuple[str, ...] = (
    "what time is it", "what's the time", "whats the time",
    "what is the time", "tell me the time", "current time",
    "what's the date", "whats the date", "what is the date",
    "what day is it", "what's today", "whats today",
    "what apps are running", "which apps are running",
    "what windows are open", "which windows are open",
    "what programs are running", "what is running",
    "what's running", "whats running",
    "what's open", "whats open", "what is open",
    "what am i doing", "what was i doing", "what am i working on",
    "active window", "focused window", "current app",
    "what did i copy", "what's on my clipboard", "whats on my clipboard",
    "what did i run", "last command", "previous command",
    "what was that command", "battery", "battery level",
)

# ── Follow-up phrases (context-dependent, actionable) ──────────
_FOLLOW_UP_PHRASES = frozenset({
    "continue", "go on", "keep going", "carry on", "proceed",
    "go back", "back", "again", "do it again", "repeat",
    "say again", "cancel", "never mind", "nevermind", "forget it",
    "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "do it",
    "no", "nope", "nah", "stop", "done",
})

# ── Vision / screen patterns ───────────────────────────────────
# Screen-dependent requests. Checked BEFORE search and question cues
# so "can you see my screen" is never misrouted.
_VISION_PATTERNS: Tuple[str, ...] = (
    r"\bwhat(?:'s| is)?\s+(?:on|being\s+shown|displayed)\s+(?:my\s+|the\s+)?"
    r"(?:screen|display|monitor|window|page|tab|browser|editor|terminal)",
    r"\btell\s+me\s+what(?:'s| is)?\s+on\s+(?:my\s+|the\s+)?(?:screen|display|monitor)",
    r"^read\s+(?:the\s+|my\s+|this\s+)?(?:screen|display|monitor|window|page|tab|error|message|dialog)",
    r"^(?:can|could|will|would)\s+you\s+see\s+(?:my\s+|the\s+|this\s+)?(?:screen|display|monitor)",
    r"^are\s+you\s+able\s+to\s+see\s+(?:my\s+|the\s+|this\s+)?(?:screen|display|monitor)",
    r"^see\s+(?:my\s+|the\s+|this\s+)?(?:screen|display|monitor)",
    r"^what\s+(?:can|do)\s+you\s+see",
    r"^what\s+are\s+you\s+looking\s+at",
    r"^what\s+am\s+i\s+looking\s+at",
    r"^describe\s+(?:my\s+|the\s+|this\s+)?(?:screen|display|monitor)",
    r"^look\s+at\s+(?:my\s+|the\s+|this\s+)?(?:screen|display|monitor)",
    r"^what\s+do\s+you\s+see\s+on\s+(?:my\s+|the\s+)?(?:screen|display|monitor)",
    r"^(?:take|capture)\s+(?:a\s+)?screenshot",
    r"^screenshot\b",
    r"\bwhat(?:'s| is)?\s+this\s+(?:error|message|dialog|page|window)\b",
    r"\bwhat\s+(?:does\s+)?(?:the\s+)?error\s+say\b",
    r"\bwhat\s+button\s+(?:should|can)\s+i\b",
    r"\bwhich\s+button\s+(?:should|can)\s+i\b",
    r"\bwhat\s+should\s+i\s+click\b",
    r"\bwhere\s+is\s+the\s+button\b",
)

# ── Explicit search-request patterns ───────────────────────────
# ONLY explicit search commands. Factual questions ("what is X",
# "tell me about X") are KNOWLEDGE_QUESTION — they reach the LLM
# (with optional web grounding downstream), never a forced search.
_SEARCH_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"^search\s+the\s+web\s+(?:for\s+)?(.+)$", "query"),
    (r"^search\s+(?:for\s+)?(.+)$", "query"),
    (r"^look\s+up\s+(.+)$", "query"),
    (r"^google\s+(.+)$", "query"),
    (r"^find\s+(.+)$", "query"),
    (r"^weather\s+(?:in\s+|for\s+)?(.+)$", "query"),
    (r"^weather$", "news"),
    (r"^news\s+about\s+(.+)$", "query"),
    (r"^news$", "news"),
    (r"^how\s+to\s+(.+)$", "query"),
)

# ── LOCAL KNOWLEDGE / PROJECT / FILESYSTEM patterns (2026-09-03) ──
# Requests referring to the user's local projects, files, code, or
# PC knowledge must be routed to LOCAL knowledge — NEVER web search.
# These are checked BEFORE the generic search patterns so "find my
# project" is never classified as a web search.
_LOCAL_KNOWLEDGE_PATTERNS: Tuple[str, ...] = (
    r"\bmy\s+project\b",
    r"\bmy\s+projects\b",
    r"\bproject\s+i\s+worked\s+on\b",
    r"\brecently\s+worked\s+on\b",
    r"\bmy\s+recent\s+project\b",
    r"\bwhat\s+project\s+was\s+i\s+working\s+on\b",
    r"\bfind\s+my\s+code\b",
    r"\bfind\s+my\s+project\b",
    r"\bmy\s+local\s+files\b",
    r"\bmy\s+local\s+projects\b",
    r"\bmy\s+files\b",
    r"\bmy\s+code\b",
    r"\bmy\s+documents\b",
    r"\bmy\s+workspace\b",
    r"\bmy\s+workspaces\b",
    r"\bmy\s+repos?\b",
    r"\bmy\s+repositories?\b",
    r"\bmy\s+folders?\b",
    r"\bmy\s+directories?\b",
    r"\bmy\s+desktop\b",
    r"\bmy\s+downloads\b",
    r"\bmy\s+home\s+folder\b",
    r"\bmy\s+home\s+directory\b",
    r"\bwhat\s+am\s+i\s+working\s+on\b",
    r"\bwhat\s+was\s+i\s+working\s+on\b",
    r"\bwhat\s+did\s+i\s+work\s+on\b",
    r"\bwhat\s+have\s+i\s+been\s+working\s+on\b",
    r"\bwhat\s+projects?\s+(?:am|was|have)\s+i\b",
    r"\bwhere\s+is\s+my\s+project\b",
    r"\bwhere\s+are\s+my\s+projects\b",
    r"\bshow\s+me\s+my\s+projects?\b",
    r"\blist\s+my\s+projects?\b",
    r"\bopen\s+my\s+project\b",
    r"\bopen\s+my\s+projects?\b",
    r"\bmy\s+recent\s+work\b",
    r"\bmy\s+recent\s+files\b",
    r"\bmy\s+recent\s+projects?\b",
    r"\bmy\s+recent\s+code\b",
    r"\bmy\s+recent\s+documents\b",
    r"\bmy\s+recent\s+workspace\b",
    r"\bmy\s+recent\s+workspaces\b",
    r"\bmy\s+recent\s+repos?\b",
    r"\bmy\s+recent\s+repositories?\b",
    r"\bmy\s+recent\s+folders?\b",
    r"\bmy\s+recent\s+directories?\b",
    r"\bmy\s+recent\s+downloads\b",
    r"\bmy\s+recent\s+desktop\b",
    r"\bmy\s+recent\s+home\s+folder\b",
    r"\bmy\s+recent\s+home\s+directory\b",
)

# ── Question cues (knowledge questions) ────────────────────────
_QUESTION_CUES = frozenset({
    "what", "whats", "what's", "who", "whos", "who's", "when", "whens",
    "when's", "where", "wheres", "where's", "why", "how", "hows",
    "how's", "which", "is", "are", "does", "do", "did", "will",
    "would", "should", "can", "could", "define", "explain",
})
_KNOWLEDGE_PREFIXES = (
    "tell me about", "tell me why", "tell me how", "tell me a",
    "tell me", "explain", "what do you know about", "do you know",
)

# ── Coherence / garbage patterns ───────────────────────────────
_GARBAGE_PATTERNS = [
    r"^i'?m? (sorry|gonna|going to|not sure|afraid|just|so)",
    r"^i don'?t (know|think|have|understand|see)",
    r"^i (can'?t|cannot)",
    r"^oh[,.\s]?$",
    r"^uh[,.\s]?$",
    r"^um[,.\s]?$",
    r"^hmm[,.\s]?$",
    r"^(i|i'm|im|you|it|that|this)[.!?]?$",
    r"^(and|but|or|so|because|then)\s+(you|i|he|she|they|we)\b.*$",
]
_REPEATED_WORD_RE = re.compile(r"\b(\w+)\b(?:\s+\1\b){2,}", re.IGNORECASE)


def _is_garbage(text: str) -> bool:
    t = text.strip().lower()
    if not t or len(t) < 2:
        return True
    for pattern in _GARBAGE_PATTERNS:
        if re.match(pattern, t):
            return True
    if _REPEATED_WORD_RE.search(t):
        return True
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    return False


def _is_greeting(text: str) -> bool:
    t = " ".join(text.lower().strip().strip(".,!?").split())
    if not t:
        return False
    words = t.split()
    if not words:
        return False
    if words[0].strip(".,!?") in _GREETING_WORDS and len(words) <= 4:
        return True
    if any(t.startswith(p) for p in _GREETING_PREFIXES) and len(words) <= 5:
        return True
    if t in _SMALLTALK_PHRASES:
        return True
    return False


def _match_vision(text: str) -> bool:
    t = " ".join(text.lower().strip().strip(".,!?").split())
    if not t:
        return False
    for pattern in _VISION_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


def _match_search(text: str) -> Optional[str]:
    t = " ".join(text.lower().strip().strip(".,!?").split())
    if not t:
        return None
    for pattern, kind in _SEARCH_PATTERNS:
        m = re.match(pattern, t)
        if m:
            try:
                return m.group(1).strip() if kind == "query" else kind
            except (IndexError, AttributeError):
                return t
    return None


def _match_local_knowledge(text: str) -> bool:
    """True when the request refers to local projects/files/code.

    These must be routed to LOCAL knowledge — never web search.
    """
    t = " ".join(text.lower().strip().strip(".,!?").split())
    if not t:
        return False
    for pattern in _LOCAL_KNOWLEDGE_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


def _match_deterministic_query(text: str) -> bool:
    t = " ".join(text.lower().strip().strip(".,!?").split())
    if not t:
        return False
    for q in _DETERMINISTIC_QUERIES:
        if t == q or t.startswith(q) or q in t:
            return True
    return False


def _first_word(text: str) -> str:
    words = text.lower().strip().strip(".,!?").split()
    return words[0].strip(".,!?;:'\"") if words else ""


def _is_follow_up(text: str) -> bool:
    t = " ".join(text.lower().strip().strip(".,!?").split())
    return t in _FOLLOW_UP_PHRASES


def _is_multi_step(text: str) -> bool:
    """Compound request: two imperatives joined by and/then."""
    t = " ".join(text.lower().strip().strip(".,!?").split())
    if not t:
        return False
    parts = re.split(r"\b(?:and\s+then|then|and also|after that|and)\b", t)
    if len(parts) < 2:
        return False
    imperatives = 0
    for part in parts:
        part = part.strip()
        if part and _first_word(part) in _COMMAND_VERBS:
            imperatives += 1
    return imperatives >= 2


def _is_coherent_sentence(text: str) -> bool:
    """A plausible natural-language sentence (statement or question).

    Used to allow coherent conversational/factual utterances to reach
    the LLM even when Whisper confidence is negative.
    """
    t = text.strip()
    if not t:
        return False
    words = t.split()
    if len(words) < 2:
        return False
    if _is_garbage(t):
        return False
    return len(words) >= 2 and len(t) >= 6


@dataclass
class IntentAuthorization:
    """The result of the final intent authorization boundary."""

    category: IntentCategory
    actionable: bool                 # may reach dispatcher/planner
    llm_allowed: bool                # may reach the LLM for an answer
    confidence: float                # fused intent confidence 0..1
    reason: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    stt_confidence: Optional[float] = None
    audio_duration_ms: Optional[float] = None

    @property
    def route(self) -> str:
        """Human-readable route label for logging."""
        if self.category == IntentCategory.UNCERTAIN:
            return "clarification"
        if self.category == IntentCategory.CONVERSATIONAL:
            return "conversation"
        if self.category == IntentCategory.KNOWLEDGE_QUESTION:
            return "llm"
        if self.category == IntentCategory.VISION_COMMAND:
            return "vision"
        if self.category == IntentCategory.SEARCH_REQUEST:
            return "search"
        if self.category == IntentCategory.FOLLOW_UP:
            return "follow_up"
        if self.category == IntentCategory.MULTI_STEP_TASK:
            return "planner"
        if self.category == IntentCategory.LOCAL_KNOWLEDGE:
            return "local_knowledge"
        return "dispatcher"

    @property
    def is_actionable(self) -> bool:
        return self.actionable

    @property
    def is_uncertain(self) -> bool:
        return self.category == IntentCategory.UNCERTAIN


# ── Classification + evidence fusion ───────────────────────────

def _base_confidence(category: IntentCategory, text: str) -> float:
    """Prior confidence from the deterministic pattern match itself."""
    words = len(text.split())
    base = {
        IntentCategory.DETERMINISTIC_COMMAND: 0.90,
        IntentCategory.VISION_COMMAND: 0.90,
        IntentCategory.SEARCH_REQUEST: 0.85,
        IntentCategory.FOLLOW_UP: 0.80,
        IntentCategory.MULTI_STEP_TASK: 0.80,
        IntentCategory.KNOWLEDGE_QUESTION: 0.70,
        IntentCategory.CONVERSATIONAL: 0.65,
        IntentCategory.LOCAL_KNOWLEDGE: 0.85,
        IntentCategory.UNCERTAIN: 0.20,
    }[category]
    # Very short actionable fragments are weaker evidence.
    if category in ACTIONABLE_CATEGORIES and words <= 1:
        base -= 0.15
    return max(0.0, min(1.0, base))


def _stt_penalty(stt_confidence: Optional[float]) -> float:
    """Penalty from Whisper avg_logprob (never a hard gate by itself)."""
    if stt_confidence is None:
        return 0.0
    if stt_confidence >= -0.4:
        return 0.0
    if stt_confidence >= -0.9:
        return 0.10
    if stt_confidence >= CONF_HARD_REJECT:
        return 0.25
    return 0.45


def _classify(text: str) -> Tuple[IntentCategory, str, Dict[str, Any]]:
    """Pure structural classification (no confidence evidence).

    Returns (category, reason, entities).
    """
    t = (text or "").strip()
    if not t:
        return IntentCategory.UNCERTAIN, "empty transcript", {}

    # 1. Greetings / small talk → conversational (never tools).
    if _is_greeting(t):
        return IntentCategory.CONVERSATIONAL, "greeting/small-talk", {}

    # 2. Follow-up replies (context-dependent, actionable).
    if _is_follow_up(t):
        return IntentCategory.FOLLOW_UP, "follow-up phrase", {}

    # 3. Vision / screen requests (BEFORE search and question cues).
    if _match_vision(t):
        return IntentCategory.VISION_COMMAND, "explicit screen request", {}

    # 4. Deterministic live queries (time, running apps, clipboard...).
    if _match_deterministic_query(t):
        return (IntentCategory.DETERMINISTIC_COMMAND,
                "deterministic live query", {})

    # 4b. LOCAL KNOWLEDGE / PROJECT / FILESYSTEM requests.
    # CRITICAL FIX (2026-09-03): "find my project which I have worked on
    # recently" was classified as SEARCH_REQUEST and routed to web search.
    # Local project/file/code references must be routed to LOCAL knowledge
    # — checked BEFORE the generic search patterns.
    if _match_local_knowledge(t):
        return (IntentCategory.LOCAL_KNOWLEDGE,
                "local project/filesystem reference", {})

    # 5. Explicit search requests.
    query = _match_search(t)
    if query is not None:
        return (IntentCategory.SEARCH_REQUEST, "explicit search request",
                {"query": query})

    # 6. Multi-step compound requests.
    if _is_multi_step(t):
        return IntentCategory.MULTI_STEP_TASK, "compound imperative", {}

    # 7. Knowledge questions (factual — LLM, never desktop actions).
    # Checked BEFORE the imperative-verb check so "tell me about animal
    # DNA" is a knowledge question, not a "tell" command.
    first = _first_word(t)
    if first in _QUESTION_CUES or any(
            t.lower().startswith(p) for p in _KNOWLEDGE_PREFIXES):
        return (IntentCategory.KNOWLEDGE_QUESTION,
                "factual question", {})

    # 8. Imperative command (first word is a command verb).
    if first in _COMMAND_VERBS:
        return (IntentCategory.DETERMINISTIC_COMMAND,
                "imperative command verb '%s'" % first,
                {"verb": first})

    # 9. Coherent declarative sentence → conversational (LLM only).
    if _is_coherent_sentence(t):
        return IntentCategory.CONVERSATIONAL, "coherent statement", {}

    # 10. Anything else → uncertain.
    return IntentCategory.UNCERTAIN, "no recognizable intent structure", {}


def authorize_intent(
    text: str,
    stt_confidence: Optional[float] = None,
    audio_duration_ms: Optional[float] = None,
    vad_evidence: Optional[Dict[str, Any]] = None,
) -> IntentAuthorization:
    """FINAL authorization boundary: transcript → intent → route.

    Combines structural classification with all available evidence.
    Never blindly raises/lowers a confidence threshold — each
    downgrade requires converging evidence.

    Rules:
      - A coherent conversational/factual sentence reaches the LLM even
        with negative Whisper confidence.
      - An actionable-but-ambiguous low-confidence transcript becomes
        UNCERTAIN (clarification) instead of executing.
      - Greetings are exempt from the hard confidence band (a greeting
        response is harmless even for a mis-heard "hello dear").
    """
    category, reason, entities = _classify(text)

    # Garbage transcripts are uncertain regardless of confidence.
    if category != IntentCategory.CONVERSATIONAL and _is_garbage(text):
        return IntentAuthorization(
            category=IntentCategory.UNCERTAIN,
            actionable=False,
            llm_allowed=False,
            confidence=0.05,
            reason="garbage/hallucinated transcript pattern",
            stt_confidence=stt_confidence,
            audio_duration_ms=audio_duration_ms,
        )

    confidence = _base_confidence(category, text) - _stt_penalty(stt_confidence)
    confidence = max(0.0, min(1.0, confidence))

    # ── Evidence-fused downgrades (actionable → uncertain) ──────
    words = len((text or "").split())
    is_greeting = category == IntentCategory.CONVERSATIONAL and _is_greeting(text)

    if category in ACTIONABLE_CATEGORIES and stt_confidence is not None:
        # HARD: hallucination band — an actionable command in this band
        # is ambiguous → clarification, never execution.
        if stt_confidence < CONF_HARD_REJECT:
            return IntentAuthorization(
                category=IntentCategory.UNCERTAIN,
                actionable=False,
                llm_allowed=False,
                confidence=confidence,
                reason=("actionable intent but confidence %.3f < %.1f "
                        "(hallucination band) — asking for clarification"
                        % (stt_confidence, CONF_HARD_REJECT)),
                entities=entities,
                stt_confidence=stt_confidence,
                audio_duration_ms=audio_duration_ms,
            )
        # SOFT: low confidence AND implausibly short audio AND a tiny
        # transcript → ambiguous → clarification.
        if (stt_confidence < CONF_SOFT_REJECT
                and audio_duration_ms is not None
                and audio_duration_ms < SHORT_AUDIO_MS
                and words <= 2):
            return IntentAuthorization(
                category=IntentCategory.UNCERTAIN,
                actionable=False,
                llm_allowed=False,
                confidence=confidence,
                reason=("actionable intent but confidence %.3f with only "
                        "%.0fms audio and %d words — asking for "
                        "clarification" % (stt_confidence,
                                           audio_duration_ms, words)),
                entities=entities,
                stt_confidence=stt_confidence,
                audio_duration_ms=audio_duration_ms,
            )

    # Non-greeting categories in the hallucination band are noise.
    # Real speech scores -0.2..-0.9; hallucinations -0.8..-1.3. A
    # coherent sentence with mildly negative confidence still reaches
    # the LLM, but below -1.0 nothing is trusted (matches the intent
    # gate's hard band). Greetings are exempt down to -1.3 (a greeting
    # response is harmless even for a mis-heard "hello dear").
    _hard_band = (CONF_DEEP_HALLUCINATION if is_greeting
                  else CONF_HARD_REJECT)
    if (stt_confidence is not None and stt_confidence < _hard_band):
        return IntentAuthorization(
            category=IntentCategory.UNCERTAIN,
            actionable=False,
            llm_allowed=False,
            confidence=confidence,
            reason=("confidence %.3f < %.1f (deep hallucination band)"
                    % (stt_confidence, CONF_DEEP_HALLUCINATION)),
            entities=entities,
            stt_confidence=stt_confidence,
            audio_duration_ms=audio_duration_ms,
        )

    # VAD evidence (optional): when the listener reports NO real speech,
    # nothing actionable may execute.
    if (vad_evidence is not None
            and vad_evidence.get("has_speech") is False
            and category in ACTIONABLE_CATEGORIES):
        return IntentAuthorization(
            category=IntentCategory.UNCERTAIN,
            actionable=False,
            llm_allowed=False,
            confidence=confidence,
            reason="no VAD speech evidence for an actionable intent",
            entities=entities,
            stt_confidence=stt_confidence,
            audio_duration_ms=audio_duration_ms,
        )

    actionable = category in ACTIONABLE_CATEGORIES
    llm_allowed = category in LLM_CATEGORIES

    return IntentAuthorization(
        category=category,
        actionable=actionable,
        llm_allowed=llm_allowed,
        confidence=confidence,
        reason=reason,
        entities=entities,
        stt_confidence=stt_confidence,
        audio_duration_ms=audio_duration_ms,
    )


def route_is_expensive_free(auth: IntentAuthorization) -> bool:
    """True when this authorization must NOT pay for perception, search,
    planner, LLM, or tools."""
    return auth.category == IntentCategory.UNCERTAIN
