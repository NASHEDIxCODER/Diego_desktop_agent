"""
ContextMonitor — runtime context-window measurement + guard for Diego.

Measures actual LLM context usage where the provider reports it, and falls
back to an explicit, clearly-labelled ESTIMATE when it does not. It never
confuses estimated tokens with actual (provider-reported) tokens.

Responsibilities:
  1. Record per-request token accounting (input / output / total / remaining).
  2. Log a compact per-request metric line WITHOUT logging prompt contents:
       CONTEXT model=qwen... input=18234 output=921 total=19155 \
       limit=131072 remaining=111917
  3. Context guard: before a request, reserve the output budget and prevent
     a prompt that would exceed the configured model context window.
  4. Priority-based context trimming: drop the LOWEST-priority context first,
     never blindly truncating the current user request or task state.

Priority of context (highest → lowest) for trimming:
    current user request
    → current task state
    → live state
    → relevant local knowledge
    → relevant history
    → older / low-value context
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Context priority (for trimming — higher number = keep longer)
# ═══════════════════════════════════════════════════════════════

class ContextPriority(IntEnum):
    """Trimming priority. The LOWEST priority is dropped first.

    The current user request and current task state must never be blindly
    truncated — they carry the highest priority.
    """
    OLDER_CONTEXT = 10        # older / low-value context (dropped first)
    RELEVANT_HISTORY = 30     # relevant conversation history
    TASK_LESSONS = 45         # reusable task lessons (Phase 21A P6)
    LOCAL_KNOWLEDGE = 50      # relevant local knowledge
    EVIDENCE = 60             # verified evidence / results (Phase 21A P3)
    LIVE_STATE = 70           # live desktop / screen state
    TASK_STATE = 90           # current task state
    USER_REQUEST = 100        # current user request (never dropped first)


# ═══════════════════════════════════════════════════════════════
# Usage record
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContextUsage:
    """Token accounting for a single LLM request."""
    model: str = ""
    context_limit: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    remaining: int = 0
    max_output_requested: int = 0
    # True when the counts are a local ESTIMATE, False when the provider/API
    # actually reported them. Never conflate the two.
    estimated: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "context_limit": self.context_limit,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "remaining": self.remaining,
            "max_output_requested": self.max_output_requested,
            "estimated": self.estimated,
            "timestamp": self.timestamp,
        }

    def compact_log(self) -> str:
        """Compact metric line (no prompt contents)."""
        tag = "estimated" if self.estimated else "reported"
        return (
            f"CONTEXT model={self.model} "
            f"input={self.input_tokens} output={self.output_tokens} "
            f"total={self.total_tokens} limit={self.context_limit} "
            f"remaining={self.remaining} ({tag})"
        )


# ═══════════════════════════════════════════════════════════════
# The monitor
# ═══════════════════════════════════════════════════════════════

# Default context window used when the model's real limit is unknown. This is
# a conservative, clearly-labelled fallback — not a provider-reported value.
DEFAULT_CONTEXT_LIMIT = 131072
# Default reserved output budget (tokens) when the caller does not specify one.
DEFAULT_OUTPUT_RESERVE = 512


class ContextMonitor:
    """Process-level context-window instrumentation + guard."""

    def __init__(self):
        self._model: str = ""
        self._context_limit: int = DEFAULT_CONTEXT_LIMIT
        self._default_output_reserve: int = DEFAULT_OUTPUT_RESERVE
        self._last: Optional[ContextUsage] = None
        self._history: List[ContextUsage] = []
        self._history_max: int = 64
        self._requests: int = 0
        # Bounded trimming/summarization event log (Phase 21A): makes
        # context trimming OBSERVABLE instead of silent.
        self._trim_events: List[Dict[str, Any]] = []
        self._trim_events_max: int = 32
        # Guard status of the most recent pre_request_check (Phase 21A).
        self._last_guard: Dict[str, Any] = {}

    # ── Configuration ───────────────────────────────────────────

    def configure(self, model: str, context_limit: Optional[int] = None,
                  default_output_reserve: Optional[int] = None) -> None:
        """Set the active model + its context limit (when known)."""
        if model:
            self._model = model
        if context_limit and context_limit > 0:
            self._context_limit = int(context_limit)
        if default_output_reserve and default_output_reserve > 0:
            self._default_output_reserve = int(default_output_reserve)
        logger.info(
            "[CONTEXT] configured model=%s limit=%d output_reserve=%d",
            self._model, self._context_limit, self._default_output_reserve,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_limit(self) -> int:
        return self._context_limit

    # ── Trimming / guard observability (Phase 21A) ─────────────

    def record_trim_event(self, event_type: str, detail: str = "",
                          tokens: int = 0) -> None:
        """Record one trimming/summarization event (bounded log).

        `event_type` is one of: "trimmed" (a context block was dropped),
        "summarized" (older context was compressed to a summary),
        "guard_refused" (the context guard rejected a request).
        Content is metadata only — never prompt text.
        """
        self._trim_events.append({
            "type": str(event_type)[:40],
            "detail": str(detail or "")[:120],
            "tokens": int(tokens),
            "timestamp": time.time(),
        })
        if len(self._trim_events) > self._trim_events_max:
            self._trim_events = self._trim_events[-self._trim_events_max:]

    @property
    def trim_events(self) -> List[Dict[str, Any]]:
        return list(self._trim_events)

    # ── Token estimation ────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for a string.

        Uses a real tokenizer when one is available; otherwise a chars/4
        heuristic. The result is ALWAYS an estimate and must be labelled as
        such by callers (estimated=True).
        """
        if not text:
            return 0
        # Prefer a real tokenizer if the project exposes one.
        try:
            from nlp.tokenizer import tokenize
            tokens = tokenize(text)
            if tokens:
                # Word-piece models expand words; add a small uplift.
                return max(1, int(len(tokens) * 1.3))
        except Exception:
            pass
        # Fallback heuristic (~4 chars per token for English text).
        return max(1, int(round(len(text) / 4.0)))

    # ── Recording ───────────────────────────────────────────────

    def record_request(
        self,
        *,
        model: Optional[str] = None,
        prompt: str = "",
        max_output: Optional[int] = None,
        provider_usage: Optional[Dict[str, Any]] = None,
        response_text: str = "",
    ) -> ContextUsage:
        """Record token accounting for one request.

        provider_usage: provider/API-reported counts when available, e.g.
            Ollama's {"prompt_eval_count": N, "eval_count": M}. When present
            these are AUTHORITATIVE (estimated=False). When absent, tokens are
            estimated from the prompt/response text (estimated=True).
        """
        model = model or self._model
        max_output = int(max_output or self._default_output_reserve)

        estimated = True
        input_tokens = 0
        output_tokens = 0

        if provider_usage:
            # Provider-reported counts are authoritative.
            input_tokens = int(
                provider_usage.get("prompt_eval_count")
                or provider_usage.get("prompt_tokens")
                or provider_usage.get("input_tokens")
                or 0)
            output_tokens = int(
                provider_usage.get("eval_count")
                or provider_usage.get("completion_tokens")
                or provider_usage.get("output_tokens")
                or 0)
            if input_tokens or output_tokens:
                estimated = False

        if estimated:
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(response_text)

        total_tokens = input_tokens + output_tokens
        remaining = max(0, self._context_limit - total_tokens)

        usage = ContextUsage(
            model=model,
            context_limit=self._context_limit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            remaining=remaining,
            max_output_requested=max_output,
            estimated=estimated,
            timestamp=time.time(),
        )
        self._last = usage
        self._requests += 1
        self._history.append(usage)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

        # Compact per-request metric line — NEVER logs prompt contents.
        logger.info(usage.compact_log())
        return usage

    # ── Context guard ───────────────────────────────────────────

    def pre_request_check(
        self,
        prompt: str,
        reserve_output: Optional[int] = None,
    ) -> Tuple[bool, str, int]:
        """Guard: will this prompt + reserved output fit the context window?

        Returns (ok, reason, prompt_tokens_estimate). The prompt token count
        is an ESTIMATE unless the caller already has provider counts.
        """
        reserve = int(reserve_output or self._default_output_reserve)
        self._last_guard = {
            "ok": None, "reserve_output": reserve, "reason": "",
        }
        prompt_tokens = self.estimate_tokens(prompt)
        budget = self._context_limit - reserve
        if budget <= 0:
            reason = (f"output reserve {reserve} >= context limit "
                      f"{self._context_limit}")
            self._last_guard = {"ok": False, "reserve_output": reserve,
                                "reason": reason}
            self.record_trim_event("guard_refused", reason, prompt_tokens)
            return (False, reason, prompt_tokens)
        if prompt_tokens > budget:
            reason = (f"prompt (~{prompt_tokens} tokens) + output reserve "
                      f"{reserve} exceeds context limit {self._context_limit}")
            self._last_guard = {"ok": False, "reserve_output": reserve,
                                "reason": reason}
            self.record_trim_event("guard_refused", reason, prompt_tokens)
            return (False, reason, prompt_tokens)
        self._last_guard = {"ok": True, "reserve_output": reserve, "reason": ""}
        return True, "", prompt_tokens

    def trim_to_fit(
        self,
        items: List[Tuple[int, str]],
        reserve_output: Optional[int] = None,
    ) -> List[str]:
        """Priority-based context trimming.

        Args:
            items: list of (priority, text). Use ContextPriority values.
                Higher priority = kept longer.
            reserve_output: tokens to reserve for the model's output.

        Returns:
            The list of text blocks that fit, ordered by original priority
            (highest first). The LOWEST-priority items are dropped first.
            The current user request / task state (highest priority) are
            never blindly truncated.
        """
        reserve = int(reserve_output or self._default_output_reserve)
        budget = max(0, self._context_limit - reserve)

        # Highest priority first so they are guaranteed to be considered.
        ordered = sorted(items, key=lambda it: it[0], reverse=True)

        kept: List[Tuple[int, str]] = []
        used = 0
        for priority, text in ordered:
            if not text:
                continue
            tokens = self.estimate_tokens(text)
            if used + tokens <= budget:
                kept.append((priority, text))
                used += tokens
            else:
                # Drop lower-priority context first. Highest-priority items
                # (user request / task state) are kept even if they push the
                # budget, so they are never blindly truncated.
                if priority >= int(ContextPriority.TASK_STATE):
                    kept.append((priority, text))
                    used += tokens
                else:
                    self.record_trim_event(
                        "trimmed", f"priority={priority}", tokens)
                    logger.info(
                        "[CONTEXT] trimmed %d tokens of priority=%d context",
                        tokens, priority,
                    )
        # Return text ordered by priority (highest first).
        return [text for _p, text in kept]

    # ── Diagnostics ─────────────────────────────────────────────

    @property
    def last_usage(self) -> Optional[ContextUsage]:
        return self._last

    def context_status(self) -> Dict[str, Any]:
        """Snapshot for the developer `context-status` diagnostic command."""
        last = self._last
        return {
            "model": self._model,
            "context_limit": self._context_limit,
            "default_output_reserve": self._default_output_reserve,
            "requests_recorded": self._requests,
            "last": last.to_dict() if last else None,
            "recent": [u.to_dict() for u in self._history[-5:]],
            # Phase 21A: trimming/summarization events + guard status.
            "trim_events": list(self._trim_events[-8:]),
            "last_guard": dict(self._last_guard),
        }


# Global singleton.
context_monitor = ContextMonitor()