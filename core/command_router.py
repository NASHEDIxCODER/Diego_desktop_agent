"""
CommandRouter — Smart routing layer that avoids LLM calls for simple commands.

Architecture:
    User utterance
        ↓
    CommandRouter.classify(text)
        ↓
    ├── SIMPLE_DESKTOP → ActionDispatcher.execute() directly (NO LLM)
    ├── KNOWN_WORKFLOW → execute multi-step workflow directly (NO LLM)
    ├── CACHED_RESPONSE → return from response cache (NO LLM)
    ├── CONVERSATION → light personality response (NO LLM)
    └── COMPLEX → invoke LLM (only this path uses the LLM)

Goal: Reduce LLM usage by 80%+ for a typical desktop assistant workflow.

Usage:
    from core.command_router import command_router

    result = await command_router.route("open firefox")
    # result.kind == "SIMPLE_DESKTOP" → executed directly, no LLM

    result = await command_router.route("what's the weather in Tokyo")
    # result.kind == "COMPLEX" → needs LLM
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RouteKind(str, Enum):
    """Classification of a user utterance."""
    SIMPLE_DESKTOP = "SIMPLE_DESKTOP"      # Open app, volume, brightness, etc.
    KNOWN_WORKFLOW = "KNOWN_WORKFLOW"       # Multi-step workflow from ExperienceDB
    CACHED_RESPONSE = "CACHED_RESPONSE"     # Previously answered question
    CONVERSATION = "CONVERSATION"           # Greetings, small talk, acknowledgments
    COMPLEX = "COMPLEX"                     # Needs LLM reasoning


@dataclass
class RouteResult:
    """Result of command routing."""
    kind: RouteKind
    action: Optional[Dict[str, Any]] = None          # Action dict for ActionDispatcher
    actions: Optional[List[Dict[str, Any]]] = None   # Multi-step workflow
    response: Optional[str] = None                    # Pre-canned response
    should_speak: bool = True                         # Whether TTS should respond
    confidence: float = 1.0                           # How confident the router is


# ── Simple desktop command patterns (NO LLM needed) ────────────────

_SIMPLE_COMMANDS = [
    # App launching
    (r"^open\s+(?:the\s+)?(vs\s*code|vscode)$", "desktop_open", {"app": "code"}),
    (r"^open\s+(?:the\s+)?(firefox|browser)$", "desktop_open", {"app": "firefox"}),
    (r"^open\s+(?:the\s+)?(chrome|google\s*chrome)$", "desktop_open", {"app": "google-chrome"}),
    (r"^open\s+(?:the\s+)?terminal$", "desktop_open", {"app": "gnome-terminal"}),
    (r"^open\s+(?:the\s+)?(spotify)$", "desktop_open", {"app": "spotify"}),
    (r"^open\s+(?:the\s+)?(slack)$", "desktop_open", {"app": "slack"}),
    (r"^open\s+(?:the\s+)?(discord)$", "desktop_open", {"app": "discord"}),
    (r"^open\s+(?:the\s+)?(telegram)$", "desktop_open", {"app": "telegram-desktop"}),
    (r"^open\s+(?:the\s+)?(notion)$", "desktop_open", {"app": "notion-app"}),
    (r"^open\s+(?:the\s+)?(calculator|calc)$", "desktop_open", {"app": "gnome-calculator"}),
    (r"^open\s+(?:the\s+)?(settings|preferences)$", "desktop_open", {"app": "gnome-control-center"}),
    (r"^open\s+(?:the\s+)?(files|file\s*manager|nautilus)$", "desktop_open", {"app": "nautilus"}),

    # Volume
    (r"^volume\s*(up|increase|louder)$", "volume_up", {}),
    (r"^(?:turn\s+)?(?:the\s+)?volume\s*(up|increase|louder)$", "volume_up", {}),
    (r"^volume\s*(down|decrease|lower|quieter)$", "volume_down", {}),
    (r"^(?:turn\s+)?(?:the\s+)?volume\s*(down|decrease|lower|quieter)$", "volume_down", {}),
    (r"^mute$", "volume_mute", {}),
    (r"^(?:un)?mute$", "volume_mute", {}),
    (r"^volume\s*(?:set\s+)?(?:to\s+)?(\d+)(?:\s*%| percent)?$", "volume_set", {}),
    (r"^(?:set\s+)?(?:the\s+)?volume\s*(?:to\s+)?(\d+)(?:\s*%| percent)?$", "volume_set", {}),

    # Brightness
    (r"^brightness\s*(up|increase|brighter)$", "brightness_up", {}),
    (r"^brightness\s*(down|decrease|lower|dimmer)$", "brightness_down", {}),
    (r"^brightness\s*(?:set\s+)?(?:to\s+)?(\d+)(?:\s*%| percent)?$", "brightness_set", {}),

    # Music control
    (r"^(?:pause|stop)\s*(?:the\s+)?(?:music|song|track|playback)$", "music_pause", {}),
    (r"^(?:resume|play|unpause)\s*(?:the\s+)?(?:music|song|track|playback)$", "music_resume", {}),
    (r"^next\s*(?:song|track|one)?$", "music_next", {}),
    (r"^(?:go\s+)?next$", "music_next", {}),
    (r"^(?:previous|prev)\s*(?:song|track|one)?$", "music_previous", {}),
    (r"^(?:go\s+)?(?:previous|back)$", "music_previous", {}),
    (r"^skip$", "music_next", {}),
    (r"^shuffle$", "music_shuffle", {}),
    (r"^(?:toggle\s+)?shuffle$", "music_shuffle", {}),
    (r"^repeat$", "music_repeat", {}),
    (r"^(?:toggle\s+)?repeat$", "music_repeat", {}),
    (r"^what(?:'s| is|)(?: currently)? playing$", "music_status", {}),
    (r"^(?:what\s+)?(?:song|track|music)\s*(?:is\s+)?(?:this|playing|on)$", "music_status", {}),

    # Screen control
    (r"^(?:what(?:'s| is|) on\s+)?(?:my\s+)?screen$", "read_screen", {}),
    (r"^read\s+(?:the\s+)?screen$", "read_screen", {}),
    (r"^lock\s*(?:the\s+)?screen$", "lock_screen", {}),
    (r"^lock\s*(?:my\s+)?(?:computer|pc|laptop|desktop)$", "lock_screen", {}),
    (r"^shutdown$", "shutdown", {}),
    (r"^(?:shut\s+down|power\s+off)$", "shutdown", {}),
    (r"^restart$", "restart", {}),
    (r"^(?:reboot|restart\s+the\s+computer)$", "restart", {}),

    # Scroll
    (r"^scroll\s*(down|up)$", "scroll", {}),
    (r"^(?:scroll\s+)?(up)$", "scroll", {"direction": "up"}),
    (r"^(?:scroll\s+)?(down)$", "scroll", {"direction": "down"}),

    # Music play (simple patterns)
    (r"^play\s+(?:some\s+)?(.+)$", "play_media", {}),
]

# Compile patterns
_COMPILED_SIMPLE = [(re.compile(p, re.IGNORECASE), action, params)
                    for p, action, params in _SIMPLE_COMMANDS]

# ── Conversational patterns (small talk, no LLM needed) ────────────

_CONVERSATION_PATTERNS = {
    re.compile(r"^hey\s*$|^hi\s*$|^hello\s*$|^hey\s+leo\s*$|^hi\s+leo\s*$|^hello\s+leo\s*$", re.IGNORECASE):
        ["Hey.", "Hi there.", "Hello.", "Hey! What's up?"],

    re.compile(r"^how\s+are\s+you\??$", re.IGNORECASE):
        ["I'm good, thanks for asking.", "Doing well.", "All good on my end."],

    re.compile(r"^(?:thanks|thank\s+you|thx|ty)$", re.IGNORECASE):
        ["No problem.", "Anytime.", "Sure thing."],

    re.compile(r"^(?:good\s+(?:morning|evening|afternoon|night))$", re.IGNORECASE):
        ["Morning.", "Good evening.", "Afternoon.", "Night."],

    re.compile(r"^goodbye$|^bye$|^see\s+you$|^later$|^good\s+night$|^goodnight$", re.IGNORECASE):
        ["See you.", "Later.", "Goodbye."],

    re.compile(r"^(?:what\s+(?:can\s+)?you\s+(?:do|help\s+with)|what\s+are\s+you\s+capable\s+(?:of|doing)\??)$", re.IGNORECASE):
        ["I can open apps, control music, adjust volume and brightness, search the web, read your screen, and help with your projects. Just ask."],

    re.compile(r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name)\??$", re.IGNORECASE):
        ["I'm Leo, your desktop assistant."],

    re.compile(r"^(?:who\s+(?:made|created|built)\s+you)\??$", re.IGNORECASE):
        ["I was created by Yeshu."],
}

# ── Known workflows (multi-step, from ExperienceDB) ────────────────

_KNOWN_WORKFLOWS = {
    "start coding": [
        {"action": "desktop_open", "params": {"app": "code"}},
        {"action": "desktop_open", "params": {"app": "gnome-terminal"}},
    ],
    "start work": [
        {"action": "desktop_open", "params": {"app": "code"}},
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "desktop_open", "params": {"app": "gnome-terminal"}},
    ],
    "start music": [
        {"action": "play_media", "params": {"query": "lofi hip hop coding"}},
    ],
}


class CommandRouter:
    """
    Routes user utterances to the appropriate handler.

    The routing pipeline:
        1. Normalize text
        2. Check conversation cache for exact/partial match
        3. Match simple desktop patterns → execute directly
        4. Match known workflows → execute multi-step
        5. Match conversation patterns → canned response
        6. Check response cache → return cached LLM answer
        7. Fall through → COMPLEX (needs LLM)
    """

    def __init__(self):
        self._action_dispatcher = None
        self._conversation_engine = None

        # Response cache: hash(text) → (response, ttl_timestamp)
        self._response_cache: Dict[str, Tuple[str, float]] = {}
        self._response_ttl_s = 3600.0  # 1 hour for general responses

        # Stats
        self._total: int = 0
        self._bypassed_llm: int = 0
        self._used_llm: int = 0
        self._cache_hits: int = 0
        self._latency_ms_total: float = 0.0

    # ── Wiring ────────────────────────────────────────────────────

    def wire(self, action_dispatcher=None, conversation_engine=None) -> None:
        """Wire in the action dispatcher and conversation engine.

        NOTE: The router does NOT execute actions directly. The Brain
        is the single orchestrator. The router only CLASSIFIES and
        returns the action for the Brain to dispatch.
        """
        self._action_dispatcher = action_dispatcher
        self._conversation_engine = conversation_engine

    # ── Main routing entry point ──────────────────────────────────

    async def route(self, text: str) -> RouteResult:
        """
        Classify and optionally execute a user utterance.

        Returns a RouteResult indicating what happened and what (if
        anything) the caller should do next.
        """
        t0 = time.time()
        self._total += 1
        original = text
        text = text.strip()
        if not text:
            return RouteResult(kind=RouteKind.COMPLEX)

        normalized = self._normalize(text)

        # ── Layer 1: Conversation cache (exact/semantic match) ──
        cached = self._check_conversation_cache(normalized)
        if cached is not None:
            self._cache_hits += 1
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.CACHED_RESPONSE,
                response=cached,
                confidence=0.9,
            )

        # ── Layer 2: Simple desktop commands ───────────────────
        # NOTE: The router ONLY classifies. The Brain dispatches the
        # action through Permission → Dispatcher → Verifier → Learning.
        result = self._match_simple(normalized)
        if result is not None:
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.SIMPLE_DESKTOP,
                action=result[0],
                response=result[1] if len(result) > 1 else None,
                confidence=0.95,
            )

        # ── Layer 3: Known workflows ──────────────────────────
        # NOTE: The router ONLY returns the workflow. The Brain
        # dispatches each step through the pipeline.
        workflow = self._match_workflow(normalized)
        if workflow is not None:
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.KNOWN_WORKFLOW,
                actions=workflow,
                response="On it.",
                confidence=0.90,
            )

        # ── Layer 4: Conversation/small talk ──────────────────
        conv = self._match_conversation(normalized)
        if conv is not None:
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.CONVERSATION,
                response=conv,
                confidence=0.90,
            )

        # ── Layer 5: Response cache (prior LLM answers) ──────
        cached_llm = self._check_llm_cache(normalized)
        if cached_llm is not None:
            self._cache_hits += 1
            self._bypassed_llm += 1
            self._latency_ms_total += (time.time() - t0) * 1000
            return RouteResult(
                kind=RouteKind.CACHED_RESPONSE,
                response=cached_llm,
                confidence=0.7,
            )

        # ── Layer 6: Needs LLM ───────────────────────────────
        self._used_llm += 1
        self._latency_ms_total += (time.time() - t0) * 1000
        return RouteResult(kind=RouteKind.COMPLEX)

    # ── Matching helpers ──────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for matching."""
        t = text.lower().strip()
        t = re.sub(r"[.,!?;:]$", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    @staticmethod
    def _match_simple(text: str) -> Optional[Tuple[Dict, Optional[str]]]:
        """Try to match a simple desktop command."""
        for pattern, action_name, base_params in _COMPILED_SIMPLE:
            m = pattern.match(text)
            if not m:
                continue
            # Build params
            params = dict(base_params)
            groups = m.groups()
            if groups:
                for i, g in enumerate(groups):
                    if g is not None:
                        if action_name == "volume_set":
                            params["percent"] = int(groups[0])
                        elif action_name == "brightness_set":
                            params["percent"] = int(groups[0])
                        elif action_name == "volume_up" and g not in ("up", "increase", "louder"):
                            continue
                        elif action_name == "volume_down" and g not in ("down", "decrease", "lower", "quieter"):
                            continue
                        elif action_name == "scroll":
                            params["direction"] = g
                        elif action_name == "play_media":
                            params["query"] = g
                        elif action_name == "desktop_open":
                            # Already set in base_params
                            pass
            action = {"action": action_name, "params": params}

            # Natural confirmations
            confirmations = {
                "volume_up": "Got it.",
                "volume_down": "Sure.",
                "volume_mute": "Done.",
                "volume_set": "Set.",
                "brightness_up": "Got it.",
                "brightness_down": "Sure.",
                "brightness_set": "Adjusted.",
                "music_pause": "",
                "music_resume": "",
                "music_next": "",
                "music_previous": "",
                "music_shuffle": "",
                "music_repeat": "",
                "music_status": "",
                "read_screen": "",
                "lock_screen": "Locked.",
                "shutdown": "",
                "restart": "",
                "scroll": "",
                "desktop_open": "Opening.",
                "play_media": "Playing.",
            }
            conf = confirmations.get(action_name, "")
            return (action, conf)
        return None

    @staticmethod
    def _match_workflow(text: str) -> Optional[List[Dict]]:
        """Try to match a known multi-step workflow."""
        for phrase, actions in _KNOWN_WORKFLOWS.items():
            if phrase in text:
                return actions
        return None

    @staticmethod
    def _match_conversation(text: str) -> Optional[str]:
        """Try to match conversational small talk."""
        import random
        for pattern, responses in _CONVERSATION_PATTERNS.items():
            if pattern.match(text):
                return random.choice(responses)
        return None

    # ── Caching ────────────────────────────────────────────────────

    def _check_conversation_cache(self, text: str) -> Optional[str]:
        """Check short-term conversation context cache.

        Handles short contextual replies like "yes", "no", "continue",
        "open it", "that one" by resolving against recent conversation
        context without calling the LLM.
        """
        try:
            from agent.conversation_memory import conv_memory

            text_lower = text.lower().strip()

            # ── Affirmation ("yes", "yeah", "sure", "ok") ──
            if text_lower in ("yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "do it"):
                # Check if Leo recently asked a question
                recent = conv_memory.get_recent_turns(3)
                for turn in reversed(recent):
                    if turn.role == "assistant" and "?" in turn.text:
                        # Leo asked something — user is affirming
                        # Execute the last suggested action
                        if conv_memory._last_goal:
                            return f"Continuing with {conv_memory._last_goal}."
                        return "Got it."
                # Generic affirmation
                return None  # Let LLM handle ambiguous affirmations

            # ── Negation ("no", "nope", "nah") ──
            if text_lower in ("no", "nope", "nah", "not really", "never mind", "cancel"):
                recent = conv_memory.get_recent_turns(3)
                for turn in reversed(recent):
                    if turn.role == "assistant" and "?" in turn.text:
                        return "Alright, never mind then."
                return None

            # ── Continuation ("continue", "go on", "keep going") ──
            if text_lower in ("continue", "go on", "keep going", "carry on", "resume", "proceed"):
                if conv_memory._last_goal:
                    return f"Continuing with {conv_memory._last_goal}."
                return "What should I continue with?"

            # ── Pronoun resolution ("open it", "that one", "close it") ──
            resolved = conv_memory._resolve_pronouns(text)
            if resolved != text:
                # Pronoun was resolved — re-route the resolved text
                logger.info("[ROUTER] Pronoun resolved: '%s' → '%s'", text, resolved)
                # Don't return here — let the resolved text go through normal routing
                return None  # The resolved text will be re-routed in the next turn

            # ── "what was that" / "say again" ──
            if text_lower in ("what", "what was that", "say again", "repeat", "come again", "pardon"):
                recent = conv_memory.get_recent_turns(2)
                for turn in reversed(recent):
                    if turn.role == "assistant" and turn.text:
                        return f"I said: {turn.text}"
                return "I didn't say anything recently."

        except Exception:
            pass
        return None

    def _check_llm_cache(self, text: str) -> Optional[str]:
        """Check if we have a cached LLM response for this query."""
        cache_key = self._cache_key(text)
        if cache_key in self._response_cache:
            response, expires = self._response_cache[cache_key]
            if time.time() < expires:
                return response
            del self._response_cache[cache_key]
        return None

    def cache_llm_response(self, query: str, response: str, ttl_s: float = 3600.0) -> None:
        """Cache an LLM response for future use."""
        key = self._cache_key(query)
        # Only cache factual responses, not conversational ones
        if len(response) > 20 and not self._is_conversational(response):
            self._response_cache[key] = (response, time.time() + ttl_s)
            # Prune old entries
            if len(self._response_cache) > 200:
                now = time.time()
                expired = [k for k, (_, exp) in self._response_cache.items() if now >= exp]
                for k in expired:
                    del self._response_cache[k]

    @staticmethod
    def _cache_key(text: str) -> str:
        """Create a normalized cache key."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _is_conversational(text: str) -> bool:
        """Check if a response is conversational (shouldn't be cached long)."""
        short = len(text.split()) < 5
        greetings = any(g in text.lower() for g in ("hey", "hi ", "hello", "bye", "thanks", "thank"))
        return short or greetings

    # ── Stats ──────────────────────────────────────────────────────

    @property
    def llm_bypass_rate(self) -> float:
        """Fraction of utterances that bypassed the LLM."""
        if self._total == 0:
            return 0.0
        return self._bypassed_llm / self._total

    @property
    def avg_router_latency_ms(self) -> float:
        """Average router latency in ms."""
        if self._total == 0:
            return 0.0
        return self._latency_ms_total / self._total

    def report(self) -> Dict[str, Any]:
        """Return routing statistics."""
        return {
            "total_utterances": self._total,
            "bypassed_llm": self._bypassed_llm,
            "used_llm": self._used_llm,
            "llm_bypass_rate": f"{self.llm_bypass_rate:.1%}",
            "cache_hits": self._cache_hits,
            "avg_router_latency_ms": f"{self.avg_router_latency_ms:.1f}",
            "cached_responses": len(self._response_cache),
        }


# Global singleton
command_router = CommandRouter()