"""
DecisionEngine — Hierarchical routing that makes LLM the LAST option.

Architecture:
    User utterance
        ↓
    L0: Deterministic intent cache
        ↓
    L1: Can I answer from working memory? (conv_memory)
        ↓
    L2: Can I answer from session memory? (unified_memory recent)
        ↓
    Explicit vision intent (screen-dependent requests)
        ↓
    L3: Can I execute directly? (command_router simple commands)
        ↓
    L4: Can I reuse an existing successful plan?
        ↓
    L5: Do I already have vision context?
        ↓
    L6: Do I need web search?
        ↓
    L7: Do I actually need the LLM? (last resort)

This is the THINKING layer that Diego uses before calling the LLM.

Important routing rule:
    Explicit screen/vision requests are deterministic.
    They must NOT be handed to the generic planner simply because
    an old reusable plan or generic LLM interpretation exists.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionPath(str, Enum):
    """Which layer resolved the request."""

    WORKING_MEMORY = "WORKING_MEMORY"
    SESSION_MEMORY = "SESSION_MEMORY"
    DIRECT_EXECUTION = "DIRECT_EXECUTION"
    REUSED_PLAN = "REUSED_PLAN"
    VISION = "VISION"
    SEARCH = "SEARCH"
    LLM = "LLM"


@dataclass
class Decision:
    """The result of the decision engine's hierarchical routing."""

    path: DecisionPath
    needs_llm: bool

    # Pre-determined response
    response: Optional[str] = None

    # Single direct action
    action: Optional[Dict[str, Any]] = None

    # Multi-step plan
    actions: Optional[List[Dict[str, Any]]] = None

    confidence: float = 1.0
    latency_us: float = 0.0

    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        """True if this decision fully resolves the request."""
        return not self.needs_llm

    def __repr__(self) -> str:
        return (
            f"Decision("
            f"path={self.path.value}, "
            f"needs_llm={self.needs_llm}, "
            f"conf={self.confidence:.2f}, "
            f"latency={self.latency_us:.0f}µs)"
        )


class DecisionEngine:
    """
    Hierarchical decision engine.

    The first layer that can confidently resolve a request wins.
    Expensive LLM reasoning remains the final fallback.

    Target:
        <1ms routing for the majority of deterministic requests.
    """

    def __init__(self):
        self._command_router = None
        self._conv_memory = None
        self._unified_memory = None
        self._experience_db = None
        self._wired = False

        # ── Deterministic intent cache ─────────────────────────
        self._intent_cache: Dict[str, Decision] = {}
        self._intent_cache_max = 200

        # ── Stats ──────────────────────────────────────────────
        self._total_requests: int = 0
        self._path_counts: Dict[DecisionPath, int] = {
            p: 0 for p in DecisionPath
        }
        self._total_latency_us: float = 0.0
        self._llm_avoided: int = 0
        self._cache_hits: int = 0

    # ──────────────────────────────────────────────────────────
    # Wiring
    # ──────────────────────────────────────────────────────────

    def _ensure_wired(self) -> None:
        """Lazy-import and wire all subsystems."""

        if self._wired:
            return

        try:
            from core.command_router import command_router

            self._command_router = command_router

        except Exception as e:
            logger.debug(
                "[DECIDE] command_router unavailable: %s",
                e,
            )

        try:
            from agent.conversation_memory import conv_memory

            self._conv_memory = conv_memory

        except Exception as e:
            logger.debug(
                "[DECIDE] conv_memory unavailable: %s",
                e,
            )

        try:
            from memory.unified_memory import unified_memory

            self._unified_memory = unified_memory

        except Exception as e:
            logger.debug(
                "[DECIDE] unified_memory unavailable: %s",
                e,
            )

        try:
            from learning.experience_db import experience_db

            self._experience_db = experience_db

        except Exception as e:
            logger.debug(
                "[DECIDE] experience_db unavailable: %s",
                e,
            )

        self._wired = True

    # ──────────────────────────────────────────────────────────
    # Main decision entry point
    # ──────────────────────────────────────────────────────────

    async def decide(
        self,
        text: str,
        vision_context: Optional[str] = None,
        search_context: Optional[str] = None,
        desktop_context: Optional[str] = None,
    ) -> Decision:
        """
        Hierarchically decide how to handle a user request.

        Args:
            text:
                User's utterance.

            vision_context:
                Optional already-prepared screen context.

            search_context:
                Optional pre-fetched search context.

            desktop_context:
                Optional desktop state.

        Returns:
            Decision describing the selected execution path.
        """

        t_start = time.perf_counter_ns()

        self._ensure_wired()
        self._total_requests += 1

        normalized = text.strip()

        if not normalized:
            return Decision(
                path=DecisionPath.LLM,
                needs_llm=True,
                latency_us=(
                    time.perf_counter_ns() - t_start
                ) / 1000,
            )

        # ──────────────────────────────────────────────────────
        # L0 — Deterministic intent cache
        # ──────────────────────────────────────────────────────

        cache_key = self._cache_key(normalized)

        cached = self._intent_cache.get(cache_key)

        if cached is not None:
            self._cache_hits += 1
            self._llm_avoided += 1

            logger.debug(
                "[DECIDE:L0] Intent cache hit: '%s' → %s",
                normalized[:80],
                cached.path.value,
            )

            return cached

        # ──────────────────────────────────────────────────────
        # L1 — Working memory
        # ──────────────────────────────────────────────────────

        result = await self._check_working_memory(normalized)

        if result is not None:
            self._record(
                DecisionPath.WORKING_MEMORY,
                t_start,
            )

            self._cache_decision(
                cache_key,
                result,
            )

            return result

        # ──────────────────────────────────────────────────────
        # L2 — Session memory
        # ──────────────────────────────────────────────────────

        result = await self._check_session_memory(
            normalized
        )

        if result is not None:
            self._record(
                DecisionPath.SESSION_MEMORY,
                t_start,
            )

            self._cache_decision(
                cache_key,
                result,
            )

            return result



        # ──────────────────────────────────────────────────────
        # Explicit vision intent
        #
        # Screen-dependent requests must win BEFORE command_router.
        # Otherwise a generic direct route such as read_screen can
        # swallow the original question and drop the user context.
        #
        # The returned action deliberately carries the original
        # question so ActionDispatcher can pass it to the vision model.
        # ──────────────────────────────────────────────────────
        if self._needs_vision(normalized):
            self._record(
                DecisionPath.VISION,
                t_start,
            )

            logger.info(
                "[DECIDE:VISION] Explicit vision request: %s",
                normalized[:120],
            )

            decision = Decision(
                path=DecisionPath.VISION,
                needs_llm=False,
                action={
                    "action": "read_screen",
                    "params": {
                        "question": normalized,
                    },
                },
                confidence=0.98,
                latency_us=(
                    time.perf_counter_ns() - t_start
                ) / 1000,
                debug={
                    "explicit_vision": True,
                    "vision_context_available": bool(
                        vision_context
                    ),
                },
            )

            self._cache_decision(
                cache_key,
                decision,
            )

            return decision

        # ──────────────────────────────────────────────────────
        # L3 — Direct execution
        # ──────────────────────────────────────────────────────

        result = await self._check_direct_execution(
            normalized
        )

        if result is not None:
            self._record(
                DecisionPath.DIRECT_EXECUTION,
                t_start,
            )

            self._cache_decision(
                cache_key,
                result,
            )

            return result

        # ──────────────────────────────────────────────────────
        # L4 — Reuse an existing successful plan
        # ──────────────────────────────────────────────────────

        result = self._check_reusable_plan(
            normalized
        )

        if result is not None:
            self._record(
                DecisionPath.REUSED_PLAN,
                t_start,
            )

            self._cache_decision(
                cache_key,
                result,
            )

            return result

        # ──────────────────────────────────────────────────────
        # L6 — Existing vision context
        #
        # This handles requests where another part of the system
        # has already prepared visual context.
        # ──────────────────────────────────────────────────────

        if (
            vision_context
            and self._needs_vision(normalized)
        ):
            self._record(
                DecisionPath.VISION,
                t_start,
            )

            return Decision(
                path=DecisionPath.VISION,
                needs_llm=True,
                confidence=0.60,
                latency_us=(
                    time.perf_counter_ns() - t_start
                ) / 1000,
                debug={
                    "has_vision_context": True,
                },
            )

        # ──────────────────────────────────────────────────────
        # L7 — Search
        # ──────────────────────────────────────────────────────

        if (
            self._needs_search(normalized)
            and search_context
        ):
            self._record(
                DecisionPath.SEARCH,
                t_start,
            )

            return Decision(
                path=DecisionPath.SEARCH,
                needs_llm=True,
                confidence=0.50,
                latency_us=(
                    time.perf_counter_ns() - t_start
                ) / 1000,
                debug={
                    "has_search_context": True,
                },
            )

        # ──────────────────────────────────────────────────────
        # L8 — LLM LAST RESORT
        # ──────────────────────────────────────────────────────

        self._record(
            DecisionPath.LLM,
            t_start,
        )

        return Decision(
            path=DecisionPath.LLM,
            needs_llm=True,
            confidence=0.30,
            latency_us=(
                time.perf_counter_ns() - t_start
            ) / 1000,
        )

    # ──────────────────────────────────────────────────────────
    # Intent cache
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        """Create a normalized cache key."""

        import hashlib

        normalized = " ".join(
            text.lower().split()
        )

        return hashlib.md5(
            normalized.encode()
        ).hexdigest()[:16]

    def _cache_decision(
        self,
        key: str,
        decision: Decision,
    ) -> None:
        """
        Cache only deterministic decisions.

        The actual screen action still performs fresh capture.
        We are caching routing, not screen contents.
        """

        if decision.needs_llm:
            return

        if len(self._intent_cache) >= self._intent_cache_max:
            self._intent_cache.pop(
                next(iter(self._intent_cache))
            )

        self._intent_cache[key] = decision

    # ──────────────────────────────────────────────────────────
    # L1 — Working memory
    # ──────────────────────────────────────────────────────────

    async def _check_working_memory(
        self,
        text: str,
    ) -> Optional[Decision]:
        """
        Check whether conversation memory can resolve the request.

        Handles:
            - pronouns
            - short replies
            - memory facts
            - conversation cache
        """

        if self._conv_memory is None:
            return None

        text_lower = text.lower().strip()

        # Pronoun resolution
        resolved = self._conv_memory._resolve_pronouns(
            text
        )

        if resolved != text:
            logger.info(
                "[DECIDE:L1] Pronoun resolved: '%s' → '%s'",
                text,
                resolved,
            )

            if self._command_router is not None:
                try:
                    routed = await self._check_direct_execution(
                        resolved
                    )

                    if routed is not None:
                        routed.debug.update({
                            "resolved_from": text,
                        })

                        return routed

                except Exception as e:
                    logger.debug(
                        "[DECIDE:L1] Resolved re-route failed: %s",
                        e,
                    )

            # Do not pretend to have executed something that
            # could not actually be routed.
            return Decision(
                path=DecisionPath.LLM,
                needs_llm=True,
                confidence=0.50,
                debug={
                    "resolved_text": resolved,
                    "original": text,
                    "reason": (
                        "pronoun_resolved_but_no_action"
                    ),
                },
            )

        # Memory fact queries
        fact = self._conv_memory.query_facts(text)

        if fact:
            logger.info(
                "[DECIDE:L1] Answered from memory: %s",
                fact[:100],
            )

            return Decision(
                path=DecisionPath.WORKING_MEMORY,
                needs_llm=False,
                response=fact,
                confidence=0.80,
                debug={
                    "source": "conv_memory.facts",
                },
            )

        # Conversation cache
        if self._command_router:
            cached = (
                self._command_router
                ._check_conversation_cache(text)
            )

            if cached is not None:
                logger.info(
                    "[DECIDE:L1] Conversation cache hit: %s",
                    cached[:80],
                )

                return Decision(
                    path=DecisionPath.WORKING_MEMORY,
                    needs_llm=False,
                    response=cached,
                    confidence=0.90,
                    debug={
                        "source": "conversation_cache",
                    },
                )

        return None

    # ──────────────────────────────────────────────────────────
    # L2 — Session memory
    # ──────────────────────────────────────────────────────────

    async def _check_session_memory(
        self,
        text: str,
    ) -> Optional[Decision]:

        text_lower = text.lower().strip()

        # ── Desktop state queries ─────────────────────────────

        if any(
            phrase in text_lower
            for phrase in (
                "what's open",
                "whats open",
                "what is open",
                "what am i doing",
                "what was i doing",
                "what am i working on",
                "active window",
                "focused window",
                "current app",
            )
        ):
            try:
                from services.desktop_state import (
                    desktop_state,
                )

                snap = desktop_state.snapshot()

                parts: List[str] = []

                if (
                    snap.focused_window
                    and snap.focused_window.title
                ):
                    parts.append(
                        "Active window: "
                        + snap.focused_window.title
                    )

                if (
                    snap.terminal
                    and snap.terminal.cwd
                ):
                    parts.append(
                        "Terminal: "
                        + snap.terminal.cwd
                    )

                if (
                    snap.browser
                    and snap.browser.current_url
                ):
                    parts.append(
                        "Browser: "
                        + snap.browser.current_url
                    )

                if parts:
                    response = " | ".join(parts)

                    logger.info(
                        "[DECIDE:L2] Desktop state: %s",
                        response[:100],
                    )

                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=response,
                        confidence=0.95,
                        debug={
                            "source": "desktop_state",
                        },
                    )

            except Exception as e:
                logger.debug(
                    "[DECIDE:L2] desktop_state failed: %s",
                    e,
                )

        # ── Clipboard queries ────────────────────────────────

        if (
            self._unified_memory
            and any(
                phrase in text_lower
                for phrase in (
                    "what did i copy",
                    "what did i just copy",
                    "clipboard",
                    "what's on my clipboard",
                    "whats on my clipboard",
                )
            )
        ):
            try:
                recent = await self._unified_memory.recent(
                    "clipboard",
                    limit=3,
                )

                if recent:
                    clips = [
                        e.value[:100]
                        for e in recent
                    ]

                    response = (
                        "Recent clipboard: "
                        + " | ".join(clips)
                    )

                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=response,
                        confidence=0.90,
                        debug={
                            "source": (
                                "unified_memory.clipboard"
                            ),
                        },
                    )

            except Exception as e:
                logger.debug(
                    "[DECIDE:L2] clipboard query failed: %s",
                    e,
                )

        # ── Command history ──────────────────────────────────

        if (
            self._unified_memory
            and any(
                phrase in text_lower
                for phrase in (
                    "what did i run",
                    "last command",
                    "previous command",
                    "what was that command",
                )
            )
        ):
            try:
                recent = await self._unified_memory.recent(
                    "command",
                    limit=5,
                )

                if recent:
                    cmds = [
                        e.value[:80]
                        for e in recent
                    ]

                    response = (
                        "Recent commands: "
                        + " | ".join(cmds)
                    )

                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=response,
                        confidence=0.90,
                        debug={
                            "source": (
                                "unified_memory.command"
                            ),
                        },
                    )

            except Exception as e:
                logger.debug(
                    "[DECIDE:L2] command history failed: %s",
                    e,
                )

        # ── Knowledge base search ────────────────────────────

        try:
            from core.background_learning import (
                background_learner,
            )

            kb_results = background_learner.search(
                text,
                max_results=2,
            )

            if kb_results:
                summary = (
                    kb_results[0].summary[:200]
                )

                if (
                    len(summary) > 50
                    and self._keyword_match(
                        text_lower,
                        summary,
                    )
                ):
                    logger.info(
                        "[DECIDE:L2] Knowledge base match: %s",
                        summary[:80],
                    )

                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=(
                            "From what I've learned: "
                            + summary
                        ),
                        confidence=0.65,
                        debug={
                            "source": "knowledge_base",
                        },
                    )

        except Exception as e:
            logger.debug(
                "[DECIDE:L2] knowledge base query failed: %s",
                e,
            )

        return None

    # ──────────────────────────────────────────────────────────
    # L3 — Direct execution
    # ──────────────────────────────────────────────────────────

    async def _check_direct_execution(
        self,
        text: str,
    ) -> Optional[Decision]:

        if self._command_router is None:
            return None

        route = await self._command_router.route(
            text
        )

        from core.command_router import RouteKind

        if route.kind == RouteKind.SIMPLE_DESKTOP:
            logger.info(
                "[DECIDE:L3] Direct execution: %s",
                route.action,
            )

            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                action=route.action,
                response=route.response,
                confidence=route.confidence,
                debug={
                    "action": route.action,
                },
            )

        if route.kind == RouteKind.KNOWN_WORKFLOW:
            logger.info(
                "[DECIDE:L3] Known workflow: %d steps",
                len(route.actions or []),
            )

            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                actions=route.actions,
                response=route.response,
                confidence=route.confidence,
                debug={
                    "workflow_steps": len(
                        route.actions or []
                    ),
                },
            )

        if route.kind == RouteKind.CONVERSATION:
            logger.info(
                "[DECIDE:L3] Conversation response: %s",
                route.response,
            )

            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                response=route.response,
                confidence=route.confidence,
            )

        if route.kind == RouteKind.CACHED_RESPONSE:
            logger.info(
                "[DECIDE:L3] Cached response: %s",
                (route.response or "")[:80],
            )

            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                response=route.response,
                confidence=route.confidence,
                debug={
                    "cached": True,
                },
            )

        return None

    # ──────────────────────────────────────────────────────────
    # L5 — Reusable plans
    # ──────────────────────────────────────────────────────────

    def _check_reusable_plan(
        self,
        text: str,
    ) -> Optional[Decision]:
        """
        Reuse a previously successful plan only when highly confident.
        """

        if self._experience_db is None:
            return None

        try:
            best = self._experience_db.best_approach(
                text,
                top_n=1,
            )

            if best and best[0].get("success"):
                exp = best[0]

                plan_steps = exp.get(
                    "plan_steps",
                    [],
                )

                confidence = exp.get(
                    "confidence",
                    0.0,
                )

                if (
                    confidence >= 0.85
                    and plan_steps
                ):
                    logger.info(
                        "[DECIDE:L5] Reusing successful plan: "
                        "%s → %s (conf=%.2f)",
                        text[:50],
                        " → ".join(
                            plan_steps[:3]
                        ),
                        confidence,
                    )

                    return Decision(
                        path=DecisionPath.REUSED_PLAN,
                        needs_llm=False,
                        actions=[
                            {
                                "action": step,
                                "params": {},
                            }
                            for step in plan_steps
                        ],
                        response=(
                            "I know how to do this. "
                            "Running the plan."
                        ),
                        confidence=confidence,
                        debug={
                            "plan_source": (
                                "experience_db"
                            ),
                            "plan_steps": plan_steps,
                            "original_result": (
                                exp.get("result", "")
                            ),
                        },
                    )

                avoid = (
                    self._experience_db
                    .avoid_actions(text)
                )

                if avoid:
                    logger.info(
                        "[DECIDE:L5] "
                        "Will avoid these actions: %s",
                        avoid,
                    )

        except Exception as e:
            logger.debug(
                "[DECIDE:L5] Experience query failed: %s",
                e,
            )

        return None

    # ──────────────────────────────────────────────────────────
    # Vision heuristic
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _needs_vision(text: str) -> bool:
        """
        Return True only for explicit visual/screen requests.

        This intentionally avoids broad keywords such as:
            "fix this"
            "what is this"
            "click the"
            "what's open"

        because those can often be resolved without a screenshot.
        """

        t = " ".join(
            text.lower().split()
        )

        explicit_phrases = (
            "read my screen",
            "read the screen",
            "read my display",
            "read the display",
            "read this screen",
            "read this page",
            "read this window",
            "what is on my screen",
            "what's on my screen",
            "what is on screen",
            "what's on screen",
            "what is on the screen",
            "what's on the screen",
            "tell me what is on my screen",
            "tell me what's on my screen",
            "tell me what's on screen",
            "look at my screen",
            "look at the screen",
            "look at my display",
            "look at the display",
            "what do you see on my screen",
            "what do you see on the screen",
            "what do you see on screen",
            "describe my screen",
            "describe the screen",
            "describe what is on my screen",
            "what am i looking at",
            "what am i looking at on screen",
            "what is this on my screen",
            "what's this on my screen",
            "what error is on my screen",
            "what error is shown",
            "what error do you see",
        )

        return any(
            phrase in t
            for phrase in explicit_phrases
        )

    # ──────────────────────────────────────────────────────────
    # Search heuristic
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _needs_search(text: str) -> bool:
        """
        Determine whether the request likely needs web search.

        Explicit vision requests always take precedence.
        """

        if DecisionEngine._needs_vision(text):
            return False

        t = text.lower().strip()

        keys = (
            "search",
            "look up",
            "find",
            "google",
            "what is",
            "who is",
            "how to",
            "how do i",
            "latest",
            "news",
            "weather",
            "definition",
            "meaning of",
            "what are",
            "what does",
            "tell me about",
            "information on",
            "learn about",
            "what's happening",
            "whats happening",
            "trending",
            "today",
            "this week",
            "current",
            "recent",
        )

        return any(
            k in t
            for k in keys
        )

    # ──────────────────────────────────────────────────────────
    # Keyword matching
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _keyword_match(
        query: str,
        text: str,
    ) -> bool:
        """Check whether meaningful keywords overlap."""

        query_words = (
            set(query.lower().split())
            - {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "what",
                "how",
                "do",
                "does",
                "can",
                "you",
                "i",
                "me",
                "my",
            }
        )

        if not query_words:
            return False

        text_words = set(
            text.lower().split()
        )

        overlap = query_words & text_words

        return len(overlap) >= 2

    # ──────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────

    def _record(
        self,
        path: DecisionPath,
        t_start_ns: float,
    ) -> None:
        """Record a routing decision."""

        self._path_counts[path] += 1

        if path != DecisionPath.LLM:
            self._llm_avoided += 1

        elapsed_us = (
            time.perf_counter_ns()
            - t_start_ns
        ) / 1000

        self._total_latency_us += elapsed_us

    @property
    def llm_avoidance_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0

        return (
            self._llm_avoided
            / self._total_requests
        )

    @property
    def avg_latency_us(self) -> float:
        if self._total_requests == 0:
            return 0.0

        return (
            self._total_latency_us
            / self._total_requests
        )

    def report(self) -> Dict[str, Any]:
        """Return decision engine statistics."""

        return {
            "total_requests": self._total_requests,
            "llm_avoided": self._llm_avoided,
            "llm_avoidance_rate": (
                f"{self.llm_avoidance_rate:.1%}"
            ),
            "avg_latency_us": (
                f"{self.avg_latency_us:.0f}"
            ),
            "avg_latency_ms": (
                f"{self.avg_latency_us / 1000:.3f}"
            ),
            "cache_hits": self._cache_hits,
            "intent_cache_size": len(
                self._intent_cache
            ),
            "path_distribution": {
                p.value: self._path_counts[p]
                for p in DecisionPath
            },
        }


# ──────────────────────────────────────────────────────────────
# Global singleton
# ──────────────────────────────────────────────────────────────

decision_engine = DecisionEngine()