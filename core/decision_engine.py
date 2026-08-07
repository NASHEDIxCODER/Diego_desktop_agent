"""
DecisionEngine — Hierarchical routing that makes LLM the LAST option.

Architecture:
    User utterance
        ↓
    L1: Can I answer from working memory? (conv_memory)
    ↓
    L2: Can I answer from session memory? (unified_memory recent)
    ↓
    L3: Can I execute directly? (command_router simple commands)
    ↓
    L4: Can I reuse an existing plan? (experience_db successful plans)
    ↓
    L5: Do I need vision? (screen context)
    ↓
    L6: Do I need web search? (search provider)
    ↓
    L7: Do I actually need the LLM? (last resort)

This is the THINKING layer that Leo uses before calling the LLM.
It wraps and extends the existing CommandRouter with memory/plan checks.

Usage:
    from core.decision_engine import decision_engine

    result = await decision_engine.decide(user_text)
    if result.needs_llm:
        response = await llm.generate(...)
    else:
        response = result.response
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DecisionPath(str, Enum):
    """Which layer resolved the request."""
    WORKING_MEMORY = "WORKING_MEMORY"     # L1: answered from conversation memory
    SESSION_MEMORY = "SESSION_MEMORY"     # L2: answered from session/desktop memory
    DIRECT_EXECUTION = "DIRECT_EXECUTION" # L3: simple desktop command
    REUSED_PLAN = "REUSED_PLAN"           # L4: reused a successful plan
    VISION = "VISION"                     # L5: needed screen context
    SEARCH = "SEARCH"                     # L6: needed web search
    LLM = "LLM"                           # L7: needed full LLM reasoning


@dataclass
class Decision:
    """The result of the decision engine's hierarchical routing."""
    path: DecisionPath
    needs_llm: bool
    response: Optional[str] = None           # Pre-determined response
    action: Optional[Dict[str, Any]] = None  # Action to execute directly
    actions: Optional[List[Dict[str, Any]]] = None  # Multi-step plan
    confidence: float = 1.0
    latency_us: float = 0.0                  # Microseconds to decide
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        """True if this decision fully resolves the request (no LLM needed)."""
        return not self.needs_llm

    def __repr__(self) -> str:
        return (f"Decision(path={self.path.value}, needs_llm={self.needs_llm}, "
                f"conf={self.confidence:.2f}, latency={self.latency_us:.0f}µs)")


class DecisionEngine:
    """
    Hierarchical decision engine.

    Check layers in order of speed/cost. The first layer that can
    confidently handle the request wins. LLM is always last.

    Target: <1ms routing for 90%+ of requests.
    """

    def __init__(self):
        self._command_router = None      # Lazy import
        self._conv_memory = None         # Lazy import
        self._unified_memory = None      # Lazy import
        self._experience_db = None       # Lazy import
        self._wired = False

        # ── Deterministic intent cache (Priority 4) ──────────
        # Maps normalized command → (action, response) so repeated
        # commands resolve in <1ms without re-running the router.
        self._intent_cache: Dict[str, Decision] = {}
        self._intent_cache_max = 200

        # Stats
        self._total_requests: int = 0
        self._path_counts: Dict[DecisionPath, int] = {
            p: 0 for p in DecisionPath
        }
        self._total_latency_us: float = 0.0
        self._llm_avoided: int = 0
        self._cache_hits: int = 0

    # ── Wiring ──────────────────────────────────────────────

    def _ensure_wired(self) -> None:
        """Lazy-import and wire all subsystems."""
        if self._wired:
            return

        try:
            from core.command_router import command_router
            self._command_router = command_router
        except Exception as e:
            logger.debug("[DECIDE] command_router unavailable: %s", e)

        try:
            from agent.conversation_memory import conv_memory
            self._conv_memory = conv_memory
        except Exception as e:
            logger.debug("[DECIDE] conv_memory unavailable: %s", e)

        try:
            from memory.unified_memory import unified_memory
            self._unified_memory = unified_memory
        except Exception as e:
            logger.debug("[DECIDE] unified_memory unavailable: %s", e)

        try:
            from learning.experience_db import experience_db
            self._experience_db = experience_db
        except Exception as e:
            logger.debug("[DECIDE] experience_db unavailable: %s", e)

        self._wired = True

    # ── Main decision entry point ───────────────────────────

    async def decide(self, text: str, 
                     vision_context: Optional[str] = None,
                     search_context: Optional[str] = None,
                     desktop_context: Optional[str] = None) -> Decision:
        """
        Hierarchically decide how to handle a user request.

        Returns a Decision that indicates whether the LLM is needed
        and if not, what the response/action should be.

        Args:
            text: User's utterance
            vision_context: Pre-fetched screen context (optional)
            search_context: Pre-fetched search results (optional)
            desktop_context: Pre-fetched desktop state (optional)

        Returns:
            Decision object with routing result.
        """
        t_start = time.perf_counter_ns()
        self._ensure_wired()
        self._total_requests += 1

        normalized = text.strip()
        if not normalized:
            return Decision(
                path=DecisionPath.LLM, needs_llm=True,
                latency_us=(time.perf_counter_ns() - t_start) / 1000,
            )

        # ── L0: Deterministic intent cache (NEW — Priority 4) ──
        # Fastest path: if we've seen this exact command before and it
        # resolved deterministically, return the cached decision.
        # This makes repeated commands ("open firefox" twice) resolve
        # in <1ms without re-running the router or LLM.
        cache_key = self._cache_key(normalized)
        cached = self._intent_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            self._llm_avoided += 1
            logger.debug("[DECIDE:L0] Intent cache hit: '%s' → %s",
                         normalized[:50], cached.path.value)
            return cached

        # ── L1: Working Memory (conversation context) ──────
        result = await self._check_working_memory(normalized)
        if result is not None:
            self._record(DecisionPath.WORKING_MEMORY, t_start)
            self._cache_decision(cache_key, result)
            return result

        # ── L2: Session Memory (unified memory, desktop) ───
        result = await self._check_session_memory(normalized)
        if result is not None:
            self._record(DecisionPath.SESSION_MEMORY, t_start)
            return result

        # ── L3: Direct Execution (command router) ──────────
        result = await self._check_direct_execution(normalized)
        if result is not None:
            self._record(DecisionPath.DIRECT_EXECUTION, t_start)
            self._cache_decision(cache_key, result)
            return result

        # ── L4: Reuse Existing Plan (experience DB) ────────
        result = self._check_reusable_plan(normalized)
        if result is not None:
            self._record(DecisionPath.REUSED_PLAN, t_start)
            self._cache_decision(cache_key, result)
            return result

        # ── L5: Vision needed? ─────────────────────────────
        if self._needs_vision(normalized) and vision_context:
            self._record(DecisionPath.VISION, t_start)
            return Decision(
                path=DecisionPath.VISION,
                needs_llm=True,  # Still need LLM but with vision context
                confidence=0.6,
                latency_us=(time.perf_counter_ns() - t_start) / 1000,
                debug={"has_vision_context": True},
            )

        # ── L6: Search needed? ─────────────────────────────
        if self._needs_search(normalized) and search_context:
            self._record(DecisionPath.SEARCH, t_start)
            return Decision(
                path=DecisionPath.SEARCH,
                needs_llm=True,  # Still need LLM but with search context
                confidence=0.5,
                latency_us=(time.perf_counter_ns() - t_start) / 1000,
                debug={"has_search_context": True},
            )

        # ── L7: Needs LLM ──────────────────────────────────
        self._record(DecisionPath.LLM, t_start)
        return Decision(
            path=DecisionPath.LLM,
            needs_llm=True,
            confidence=0.3,
            latency_us=(time.perf_counter_ns() - t_start) / 1000,
        )

    # ── Intent cache helpers ────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        """Create a normalized cache key."""
        import hashlib
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _cache_decision(self, key: str, decision: Decision) -> None:
        """Cache a deterministic decision for future reuse."""
        if decision.needs_llm:
            return  # Only cache deterministic resolutions
        if len(self._intent_cache) >= self._intent_cache_max:
            # Simple LRU: remove oldest (dict preserves insertion order)
            self._intent_cache.pop(next(iter(self._intent_cache)))
        self._intent_cache[key] = decision

    # ── Layer checks ────────────────────────────────────────

    async def _check_working_memory(self, text: str) -> Optional[Decision]:
        """
        L1: Check if this can be answered from working (conversation) memory.

        Handles:
          - Pronoun resolution ("open it", "close that")
          - Short replies ("yes", "no", "continue", "cancel")
          - Memory queries ("what was my project called?")
          - "what was that" / "say again"
        """
        if self._conv_memory is None:
            return None

        text_lower = text.lower().strip()

        # ── Exact short-command matches (no LLM) ───────────
        # These are handled by command_router's conversation cache
        # but we check here first for lower latency.
        
        # Pronoun resolution — MUST re-route the resolved command, not
        # just reply "I'll open firefox." with no action dispatched.
        resolved = self._conv_memory._resolve_pronouns(text)
        if resolved != text:
            logger.info("[DECIDE:L1] Pronoun resolved: '%s' → '%s'", text, resolved)
            # Re-route the resolved text so the actual action dispatches.
            # Example: "open it" when last entity is "firefox" → "open firefox"
            # which the CommandRouter matches as SIMPLE_DESKTOP and returns an action.
            if self._command_router is not None:
                try:
                    routed = await self._check_direct_execution(resolved)
                    if routed is not None:
                        routed.debug.update({"resolved_from": text})
                        return routed
                except Exception as e:
                    logger.debug("[DECIDE:L1] Resolved re-route failed: %s", e)
            # CRITICAL FIX: If the resolved text could not be re-routed to
            # an action, do NOT return a response-only decision. That would
            # make Leo say "I'll open firefox." but never actually open it.
            # Instead, fall through to the LLM path so the Brain can plan
            # and dispatch the resolved command properly.
            return Decision(
                path=DecisionPath.LLM,
                needs_llm=True,
                confidence=0.5,
                debug={"resolved_text": resolved, "original": text,
                       "reason": "pronoun_resolved_but_no_action"},
            )

        # Memory fact queries
        fact = self._conv_memory.query_facts(text)
        if fact:
            logger.info("[DECIDE:L1] Answered from memory: %s", fact[:80])
            return Decision(
                path=DecisionPath.WORKING_MEMORY,
                needs_llm=False,
                response=fact,
                confidence=0.80,
                debug={"source": "conv_memory.facts"},
            )

        # ── Command router's conversation cache (yes/no/continue etc) ──
        if self._command_router:
            cached = self._command_router._check_conversation_cache(text)
            if cached is not None:
                logger.info("[DECIDE:L1] Conversation cache hit: '%s'", cached[:60])
                return Decision(
                    path=DecisionPath.WORKING_MEMORY,
                    needs_llm=False,
                    response=cached,
                    confidence=0.90,
                    debug={"source": "conversation_cache"},
                )

        return None

    async def _check_session_memory(self, text: str) -> Optional[Decision]:
        """
        L2: Check session memory (unified_memory, desktop state, knowledge base).

        Handles:
          - "what's open right now?"
          - "what was I working on?"
          - "what did I just copy?"
          - "what files did I change?"
          - "what was that command?"
        """
        text_lower = text.lower().strip()

        # ── Desktop state queries ───────────────────────────
        if any(phrase in text_lower for phrase in (
            "what's open", "whats open", "what is open",
            "what am i doing", "what was i doing", "what am i working on",
            "active window", "focused window", "current app",
        )):
            try:
                from services.desktop_state import desktop_state
                snap = desktop_state.snapshot()
                parts = []
                if snap.focused_window and snap.focused_window.title:
                    parts.append(f"Active window: {snap.focused_window.title}")
                if snap.terminal and snap.terminal.cwd:
                    parts.append(f"Terminal: {snap.terminal.cwd}")
                if snap.browser and snap.browser.current_url:
                    parts.append(f"Browser: {snap.browser.current_url}")
                if parts:
                    response = " | ".join(parts)
                    logger.info("[DECIDE:L2] Desktop state: %s", response[:80])
                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=response,
                        confidence=0.95,
                        debug={"source": "desktop_state"},
                    )
            except Exception as e:
                logger.debug("[DECIDE:L2] desktop_state failed: %s", e)

        # ── Clipboard queries ───────────────────────────────
        if self._unified_memory and any(phrase in text_lower for phrase in (
            "what did i copy", "what did i just copy", "clipboard",
            "what's on my clipboard", "whats on my clipboard",
        )):
            try:
                recent = await self._unified_memory.recent("clipboard", limit=3)
                if recent:
                    clips = [e.value[:100] for e in recent]
                    response = "Recent clipboard: " + " | ".join(clips)
                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=response,
                        confidence=0.90,
                        debug={"source": "unified_memory.clipboard"},
                    )
            except Exception as e:
                logger.debug("[DECIDE:L2] clipboard query failed: %s", e)

        # ── Command history ─────────────────────────────────
        if self._unified_memory and any(phrase in text_lower for phrase in (
            "what did i run", "last command", "previous command",
            "what was that command",
        )):
            try:
                recent = await self._unified_memory.recent("command", limit=5)
                if recent:
                    cmds = [e.value[:80] for e in recent]
                    response = "Recent commands: " + " | ".join(cmds)
                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=response,
                        confidence=0.90,
                        debug={"source": "unified_memory.command"},
                    )
            except Exception as e:
                logger.debug("[DECIDE:L2] command query failed: %s", e)

        # ── Knowledge base search ───────────────────────────
        try:
            from core.background_learning import background_learner
            kb_results = background_learner.search(text, max_results=2)
            if kb_results:
                summary = kb_results[0].summary[:200]
                if len(summary) > 50 and self._keyword_match(text_lower, summary):
                    logger.info("[DECIDE:L2] Knowledge base match: %s", summary[:80])
                    return Decision(
                        path=DecisionPath.SESSION_MEMORY,
                        needs_llm=False,
                        response=f"From what I've learned: {summary}",
                        confidence=0.65,
                        debug={"source": "knowledge_base"},
                    )
        except Exception as e:
            logger.debug("[DECIDE:L2] knowledge base query failed: %s", e)

        return None

    async def _check_direct_execution(self, text: str) -> Optional[Decision]:
        """
        L3: Check if this is a simple desktop command that can be executed directly.

        Delegates to CommandRouter for pattern matching.
        """
        if self._command_router is None:
            return None

        route = await self._command_router.route(text)

        from core.command_router import RouteKind

        if route.kind == RouteKind.SIMPLE_DESKTOP:
            logger.info("[DECIDE:L3] Direct execution: %s", route.action)
            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                action=route.action,
                response=route.response,
                confidence=route.confidence,
                debug={"action": route.action},
            )

        if route.kind == RouteKind.KNOWN_WORKFLOW:
            logger.info("[DECIDE:L3] Known workflow: %d steps", len(route.actions or []))
            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                actions=route.actions,
                response=route.response,
                confidence=route.confidence,
                debug={"workflow_steps": len(route.actions or [])},
            )

        if route.kind == RouteKind.CONVERSATION:
            logger.info("[DECIDE:L3] Conversation response: %s", route.response)
            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                response=route.response,
                confidence=route.confidence,
            )

        if route.kind == RouteKind.CACHED_RESPONSE:
            logger.info("[DECIDE:L3] Cached response: %s", (route.response or "")[:60])
            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                response=route.response,
                confidence=route.confidence,
                debug={"cached": True},
            )

        return None

    def _check_reusable_plan(self, text: str) -> Optional[Decision]:
        """
        L4: Check if there's a previously successful plan for this request.

        Looks in ExperienceDB for plans that succeeded in the past
        with high confidence.
        """
        if self._experience_db is None:
            return None

        try:
            best = self._experience_db.best_approach(text, top_n=1)
            if best and best[0].get("success"):
                exp = best[0]
                plan_steps = exp.get("plan_steps", [])
                confidence = exp.get("confidence", 0.0)

                # Only reuse if highly confident
                if confidence >= 0.85 and plan_steps:
                    logger.info(
                        "[DECIDE:L4] Reusing successful plan: %s → %s (conf=%.2f)",
                        text[:50], " → ".join(plan_steps[:3]), confidence,
                    )
                    return Decision(
                        path=DecisionPath.REUSED_PLAN,
                        needs_llm=False,
                        actions=[
                            {"action": s, "params": {}}
                            for s in plan_steps
                        ],
                        response=f"I know how to do this. Running the plan.",
                        confidence=confidence,
                        debug={
                            "plan_source": "experience_db",
                            "plan_steps": plan_steps,
                            "original_result": exp.get("result", ""),
                        },
                    )

            # Check for actions to avoid
            avoid = self._experience_db.avoid_actions(text)
            if avoid:
                logger.info("[DECIDE:L4] Will avoid these actions: %s", avoid)

        except Exception as e:
            logger.debug("[DECIDE:L4] Experience query failed: %s", e)

        return None

    # ── Heuristics ──────────────────────────────────────────

    @staticmethod
    def _needs_vision(text: str) -> bool:
        """Check if the request likely needs screen context."""
        t = text.lower()
        keys = [
            "screen", "looking at", "this page", "this window",
            "what am i", "read this", "what does this say",
            "on my screen", "this button", "click the", "click this",
            "what's open", "whats open", "fix this", "what is this",
            "this error", "this code", "this file",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _needs_search(text: str) -> bool:
        """Check if the request likely needs web search."""
        t = text.lower()
        keys = [
            "search", "look up", "find", "google", "what is", "who is",
            "how to", "how do i", "latest", "news", "weather",
            "definition", "meaning of", "what are", "what does",
            "tell me about", "information on", "learn about",
            "what's happening", "whats happening", "trending",
            "today", "this week", "current", "recent",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _keyword_match(query: str, text: str) -> bool:
        """Check if query keywords meaningfully overlap with text."""
        query_words = set(query.lower().split()) - {
            "the", "a", "an", "is", "are", "was", "were", "what",
            "how", "do", "does", "can", "you", "i", "me", "my",
        }
        if not query_words:
            return False
        text_words = set(text.lower().split())
        overlap = query_words & text_words
        return len(overlap) >= 2

    # ── Stats ───────────────────────────────────────────────

    def _record(self, path: DecisionPath, t_start_ns: float) -> None:
        """Record a decision for stats."""
        self._path_counts[path] += 1
        if path != DecisionPath.LLM:
            self._llm_avoided += 1
        elapsed_us = (time.perf_counter_ns() - t_start_ns) / 1000
        self._total_latency_us += elapsed_us

    @property
    def llm_avoidance_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._llm_avoided / self._total_requests

    @property
    def avg_latency_us(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._total_latency_us / self._total_requests

    def report(self) -> Dict[str, Any]:
        """Return decision engine statistics."""
        return {
            "total_requests": self._total_requests,
            "llm_avoided": self._llm_avoided,
            "llm_avoidance_rate": f"{self.llm_avoidance_rate:.1%}",
            "avg_latency_us": f"{self.avg_latency_us:.0f}",
            "avg_latency_ms": f"{self.avg_latency_us / 1000:.3f}",
            "cache_hits": self._cache_hits,
            "intent_cache_size": len(self._intent_cache),
            "path_distribution": {
                p.value: self._path_counts[p]
                for p in DecisionPath
            },
        }


# Global singleton
decision_engine = DecisionEngine()