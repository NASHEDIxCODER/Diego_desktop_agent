"""
ReasoningState — bounded, structured reasoning artifacts for Diego (Phase 21A).

The reasoning layer stores STRUCTURED REASONING ARTIFACTS ONLY:

    goal, user constraints, assumptions, context references, current
    objective, plan, completed steps, verified evidence, observations,
    failed attempts (+ structured failure diagnosis), alternatives
    considered, next action, confidence/uncertainty, replan counts,
    tool-reliability information, user decisions/confirmations.

It deliberately does NOT store raw hidden chain-of-thought: every field is
a bounded, inspectable, serializable artifact. The system can therefore
continue reasoning across replans / continuations without ever exposing
(or persisting) a private reasoning transcript.

Bounds: every list is capped and every text field is clipped, so the state
stays small even for very long tasks (large working context is handled by
agent/reasoning_context.py, which COMPRESSES older material against the
priority layers — it never inflates this state).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Bounds (structured artifacts stay compact) ─────────────────
MAX_LIST_ITEMS = 24
MAX_TEXT_CHARS = 400
MAX_OBSERVATIONS = 12


def _clip(text: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """Clip a text field to its bound."""
    s = str(text or "")
    return s if len(s) <= limit else s[:limit] + "…"


def _bound(items: Optional[List[Any]], n: int = MAX_LIST_ITEMS) -> List[Any]:
    """Keep the most recent `n` items of a list (bounded state)."""
    return list((items or [])[-n:])


class ReasoningPhase(str, Enum):
    """Reasoning loop phase (GOAL → … → CONTINUE/RECOVER/REPLAN/ASK)."""
    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE = "observe"
    VERIFY = "verify"
    REFLECT = "reflect"
    CONTINUE = "continue"
    RECOVER = "recover"
    REPLAN = "replan"
    ASK_USER = "ask_user"
    DONE = "done"


# Next-strategy vocabulary produced by failure reasoning (Task 6). The
# deterministic runtime — never the model — decides what actually runs.
NEXT_STRATEGIES = frozenset({
    "retry",              # transient failure → retry
    "repair_params",      # wrong parameters → repair parameters
    "observe_and_adapt",  # changed state → observe + adapt
    "alternative_tool",   # tool failure → alternative tool if safe
    "replan",             # plan invalid → bounded replan
    "ask_user",           # missing information → ask user
    "stop",               # unrecoverable / limits exhausted
})


@dataclass
class FailureAnalysis:
    """Structured failure context for one failed step (Task 6)."""
    step_index: int = 0
    action: str = ""
    params_summary: str = ""          # clipped param summary, not raw params
    observed_result: str = ""         # what ACTUALLY happened (evidence)
    failure_kind: str = "unknown"     # FailureKind value from task_state
    probable_cause: str = ""
    retry_suitable: bool = False
    alternative: str = ""
    next_strategy: str = "stop"       # one of NEXT_STRATEGIES
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "params_summary": self.params_summary,
            "observed_result": self.observed_result,
            "failure_kind": self.failure_kind,
            "probable_cause": self.probable_cause,
            "retry_suitable": self.retry_suitable,
            "alternative": self.alternative,
            "next_strategy": self.next_strategy,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureAnalysis":
        return cls(
            step_index=int(data.get("step_index", 0)),
            action=str(data.get("action", "")),
            params_summary=str(data.get("params_summary", "")),
            observed_result=str(data.get("observed_result", "")),
            failure_kind=str(data.get("failure_kind", "unknown")),
            probable_cause=str(data.get("probable_cause", "")),
            retry_suitable=bool(data.get("retry_suitable", False)),
            alternative=str(data.get("alternative", "")),
            next_strategy=str(data.get("next_strategy", "stop")),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass
class TaskReflection:
    """Bounded post-task reflection (Task 11). Compact and structured —
    NEVER a large reasoning transcript."""
    goal_achieved: bool = False
    successful_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    success_evidence: List[str] = field(default_factory=list)
    strategy_that_worked: str = ""
    strategy_that_failed: str = ""
    replanning_required: bool = False
    lesson: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_achieved": self.goal_achieved,
            "successful_steps": self.successful_steps[:MAX_LIST_ITEMS],
            "failed_steps": self.failed_steps[:MAX_LIST_ITEMS],
            "success_evidence": [
                _clip(e) for e in self.success_evidence[:MAX_LIST_ITEMS]],
            "strategy_that_worked": _clip(self.strategy_that_worked),
            "strategy_that_failed": _clip(self.strategy_that_failed),
            "replanning_required": self.replanning_required,
            "lesson": self.lesson,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ReasoningState:
    """Bounded internal reasoning/task state (Task 2).

    Structured artifacts only — no hidden chain-of-thought. This state is
    what the reasoning layer reads and updates between loop phases, and
    what survives replans / continuations so reasoning is never lost.
    """
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ── Goal & understanding ──────────────────────────────────
    goal: str = ""
    user_constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    context_refs: List[str] = field(default_factory=list)  # references only
    current_objective: str = ""

    # ── Plan & progress ───────────────────────────────────────
    plan: List[Dict[str, Any]] = field(default_factory=list)
    plan_version: int = 0
    completed_step_summaries: List[str] = field(default_factory=list)

    # ── Evidence & observations (live > everything else) ──────
    verified_evidence: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)

    # ── Failure reasoning (Task 6) ────────────────────────────
    failed_attempts: List[FailureAnalysis] = field(default_factory=list)
    alternatives_considered: List[str] = field(default_factory=list)

    # ── Reasoning trajectory ──────────────────────────────────
    next_action: str = ""
    confidence: float = 0.5
    uncertainty: List[str] = field(default_factory=list)
    phase: ReasoningPhase = ReasoningPhase.UNDERSTAND

    # ── Bounded counters (loop safety) ────────────────────────
    replan_count: int = 0
    adaptive_revision_count: int = 0

    # ── Tool reliability + user decisions ─────────────────────
    tool_reliability: Dict[str, float] = field(default_factory=dict)
    user_decisions: List[str] = field(default_factory=list)

    # ── Compressed history (Task 13) ──────────────────────────
    summarized_history: List[str] = field(default_factory=list)

    # ── Mutators (all bounded) ─────────────────────────────────

    def note_observation(self, text: str) -> None:
        if text:
            self.observations = _bound(
                self.observations + [_clip(text, 200)], MAX_OBSERVATIONS)

    def note_evidence(self, text: str) -> None:
        if text:
            self.verified_evidence = _bound(
                self.verified_evidence + [_clip(text)], MAX_LIST_ITEMS)

    def note_constraint(self, text: str) -> None:
        if text and text not in self.user_constraints:
            self.user_constraints = _bound(
                self.user_constraints + [_clip(text, 200)], MAX_LIST_ITEMS)

    def note_assumption(self, text: str) -> None:
        if text and text not in self.assumptions:
            self.assumptions = _bound(
                self.assumptions + [_clip(text, 200)], MAX_LIST_ITEMS)

    def note_alternative(self, text: str) -> None:
        if text and text not in self.alternatives_considered:
            self.alternatives_considered = _bound(
                self.alternatives_considered + [_clip(text, 200)],
                MAX_LIST_ITEMS)

    def note_user_decision(self, text: str) -> None:
        if text:
            self.user_decisions = _bound(
                self.user_decisions + [_clip(text, 200)], MAX_LIST_ITEMS)

    def note_failure(self, analysis: FailureAnalysis) -> None:
        self.failed_attempts = _bound(
            self.failed_attempts + [analysis], MAX_LIST_ITEMS)

    def note_completed_step(self, summary: str) -> None:
        if summary:
            self.completed_step_summaries = _bound(
                self.completed_step_summaries + [_clip(summary, 200)],
                MAX_LIST_ITEMS)

    def set_tool_reliability(self, stats: Dict[str, float]) -> None:
        """Bounded tool-reliability snapshot (name → success rate 0–1)."""
        self.tool_reliability = {
            str(k)[:60]: float(v)
            for k, v in list((stats or {}).items())[:MAX_LIST_ITEMS]
            if isinstance(v, (int, float))
        }

    def last_failure(self) -> Optional[FailureAnalysis]:
        return self.failed_attempts[-1] if self.failed_attempts else None

    # ── Compact rendering for the reasoning context (P1/P3) ───

    def goal_line(self) -> str:
        return f"GOAL: {self.goal}" if self.goal else ""

    def constraints_lines(self) -> List[str]:
        return [f"CONSTRAINT: {c}" for c in self.user_constraints]

    def evidence_lines(self) -> List[str]:
        return [f"VERIFIED: {e}" for e in self.verified_evidence]

    def state_lines(self) -> List[str]:
        """P1 — active task state (never silently truncated)."""
        lines = [self.goal_line()] if self.goal else []
        lines += self.constraints_lines()
        if self.current_objective:
            lines.append(f"CURRENT OBJECTIVE: {self.current_objective}")
        if self.next_action:
            lines.append(f"NEXT ACTION: {self.next_action}")
        lines.append(
            f"PROGRESS: {len(self.completed_step_summaries)} step(s) done, "
            f"{len(self.failed_attempts)} failed, "
            f"replans={self.replan_count}+{self.adaptive_revision_count}, "
            f"confidence={self.confidence:.2f}, phase={self.phase.value}")
        for summary in self.summarized_history:
            lines.append(f"EARLIER: {summary}")
        for s in self.completed_step_summaries[-8:]:
            lines.append(f"DONE: {s}")
        for fa in self.failed_attempts[-4:]:
            lines.append(
                f"FAILED: {fa.action} ({fa.failure_kind}) — "
                f"{fa.observed_result[:120]} strategy={fa.next_strategy}")
        for d in self.user_decisions[-4:]:
            lines.append(f"USER DECISION: {d}")
        if self.uncertainty:
            lines.append(
                "UNCERTAIN: " + "; ".join(self.uncertainty[-3:]))
        return lines

    # ── Serialization (bounded on both directions) ─────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": _clip(self.goal),
            "user_constraints": self.user_constraints[:MAX_LIST_ITEMS],
            "assumptions": self.assumptions[:MAX_LIST_ITEMS],
            "context_refs": self.context_refs[:MAX_LIST_ITEMS],
            "current_objective": _clip(self.current_objective),
            "plan": self.plan[:MAX_LIST_ITEMS],
            "plan_version": self.plan_version,
            "completed_step_summaries":
                self.completed_step_summaries[:MAX_LIST_ITEMS],
            "verified_evidence": self.verified_evidence[:MAX_LIST_ITEMS],
            "observations": self.observations[:MAX_OBSERVATIONS],
            "failed_attempts": [fa.to_dict() for fa in self.failed_attempts],
            "alternatives_considered":
                self.alternatives_considered[:MAX_LIST_ITEMS],
            "next_action": _clip(self.next_action),
            "confidence": round(self.confidence, 3),
            "uncertainty": self.uncertainty[:MAX_LIST_ITEMS],
            "phase": self.phase.value,
            "replan_count": self.replan_count,
            "adaptive_revision_count": self.adaptive_revision_count,
            "tool_reliability": self.tool_reliability,
            "user_decisions": self.user_decisions[:MAX_LIST_ITEMS],
            "summarized_history": self.summarized_history[:MAX_LIST_ITEMS],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningState":
        rs = cls(
            task_id=str(data.get("task_id", "")) or uuid.uuid4().hex[:12],
            goal=str(data.get("goal", "")),
            current_objective=str(data.get("current_objective", "")),
            plan=list(data.get("plan") or [])[:MAX_LIST_ITEMS],
            plan_version=int(data.get("plan_version", 0)),
            next_action=str(data.get("next_action", "")),
            confidence=float(data.get("confidence", 0.5)),
            replan_count=int(data.get("replan_count", 0)),
            adaptive_revision_count=int(
                data.get("adaptive_revision_count", 0)),
        )
        rs.user_constraints = [str(c) for c in
                               (data.get("user_constraints") or [])]
        rs.assumptions = [str(a) for a in (data.get("assumptions") or [])]
        rs.context_refs = [str(r) for r in (data.get("context_refs") or [])]
        rs.completed_step_summaries = [
            str(s) for s in (data.get("completed_step_summaries") or [])]
        rs.verified_evidence = [
            str(e) for e in (data.get("verified_evidence") or [])]
        rs.observations = [str(o) for o in (data.get("observations") or [])]
        rs.failed_attempts = [
            FailureAnalysis.from_dict(d)
            for d in (data.get("failed_attempts") or [])]
        rs.alternatives_considered = [
            str(a) for a in (data.get("alternatives_considered") or [])]
        rs.uncertainty = [str(u) for u in (data.get("uncertainty") or [])]
        try:
            rs.phase = ReasoningPhase(str(data.get("phase", "understand")))
        except ValueError:
            rs.phase = ReasoningPhase.UNDERSTAND
        rs.tool_reliability = {
            str(k): float(v)
            for k, v in (data.get("tool_reliability") or {}).items()
            if isinstance(v, (int, float))}
        rs.user_decisions = [str(d) for d in (data.get("user_decisions") or [])]
        rs.summarized_history = [
            str(s) for s in (data.get("summarized_history") or [])]
        return rs


# ═══════════════════════════════════════════════════════════════
# Deterministic constraint extraction (no model required)
# ═══════════════════════════════════════════════════════════════

_CONSTRAINT_MARKERS = (
    "must", "without", "don't", "do not", "never", "only ", "except",
    "instead of", "but not", "make sure", "be careful", "keep ",
)


def extract_constraints(goal: str) -> List[str]:
    """Extract explicit user constraints from the goal text.

    Deterministic heuristics only (no model): clauses carrying obligation,
    prohibition, or scope-limiting markers become structured constraints.
    These are the constraints that context trimming must NEVER drop.
    """
    constraints: List[str] = []
    text = (goal or "").strip()
    if not text:
        return constraints
    clauses = [c.strip() for c in re.split(r"[.!?;\n]|,\s", text) if c.strip()]
    for clause in clauses:
        low = clause.lower()
        if any(m in low for m in _CONSTRAINT_MARKERS):
            constraints.append(_clip(clause, 200))
    # Quoted segments are almost always constraints/exact values.
    for quoted in re.findall(r"[\"']([^\"']{2,120})[\"']", text):
        item = f"use exactly: {quoted}"
        if item not in constraints:
            constraints.append(_clip(item, 200))
    return constraints[:MAX_LIST_ITEMS]
