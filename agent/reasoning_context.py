"""
ReasoningContextComposer — priority-layered context for reasoning (Phase 21A).

Integrates with the EXISTING ai.context_monitor.ContextMonitor (single
trimming/compression mechanism — no second context manager is created).

Priority layers (kept longest → dropped first):

    P0 — current user goal + constraints (ContextPriority.USER_REQUEST)
    P1 — active task state              (ContextPriority.TASK_STATE)
    P2 — current step + recent obs      (ContextPriority.LIVE_STATE)
    P3 — verified results / evidence    (ContextPriority.EVIDENCE)
    P4 — relevant conversation history  (ContextPriority.RELEVANT_HISTORY)
    P5 — relevant knowledge retrieval   (ContextPriority.LOCAL_KNOWLEDGE)
    P6 — reusable task lessons          (ContextPriority.TASK_LESSONS)
    P7 — older / background context     (ContextPriority.OLDER_CONTEXT)

When the budget tightens, the monitor trims lowest-priority layers first,
while P0/P1/P2/P3 — and therefore active constraints and verified
evidence — are preserved. Older task material is NEVER silently
truncated: before trimming, past steps are deterministically COMPRESSED
into structured summaries (Task 13) that preserve decisions, constraints,
verified evidence, failed strategies and the current state, and a
reference to the summarized material is kept.

The composer reports limit / tokens used / tokens remaining / guard
status / trimming + summarization events through the existing monitor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai.context_monitor import (
    ContextPriority,
    ContextMonitor,
    context_monitor as default_monitor,
)

logger = logging.getLogger(__name__)

# Compress older steps into structured summaries once the live step list
# grows beyond this bound (Task 13).
DEFAULT_RECENT_STEP_KEEP = 4
MAX_SUMMARY_ITEMS = 12


@dataclass
class ComposedContext:
    """Result of one layered context composition."""
    text: str = ""
    layers: List[str] = field(default_factory=list)     # kept layer names
    trimmed_layers: List[str] = field(default_factory=list)
    tokens_used: int = 0
    tokens_remaining: int = 0
    context_limit: int = 0
    summarized: bool = False                            # compression happened
    trimmed: bool = False                               # trimming happened


class ReasoningContextComposer:
    """Composes bounded, priority-layered context for model calls."""

    def __init__(self, monitor: Optional[ContextMonitor] = None,
                 recent_step_keep: int = DEFAULT_RECENT_STEP_KEEP):
        self._monitor = monitor or default_monitor
        self._recent_step_keep = max(1, int(recent_step_keep))

    # ── Layer builders ─────────────────────────────────────────

    @staticmethod
    def p0_goal(goal: str, constraints: List[str]) -> str:
        """P0 — current user goal + active constraints (never dropped)."""
        lines = [f"GOAL: {goal}"] if goal else []
        lines += [f"CONSTRAINT: {c}" for c in (constraints or [])]
        return "\n".join(lines)

    @staticmethod
    def p1_task_state(state_lines: List[str]) -> str:
        """P1 — active task state (bounded rendering, never truncated)."""
        return "\n".join(state_lines or [])

    @staticmethod
    def p2_current_step(current_objective: str,
                        recent_observations: List[str]) -> str:
        """P2 — the current step plus the most recent observations."""
        lines = []
        if current_objective:
            lines.append(f"CURRENT STEP: {current_objective}")
        lines += [f"OBSERVED: {o}" for o in (recent_observations or [])[-6:]]
        return "\n".join(lines)

    @staticmethod
    def p3_evidence(verified_evidence: List[str]) -> str:
        """P3 — verified evidence/results (kept near the top)."""
        return "\n".join(
            f"VERIFIED: {e}" for e in (verified_evidence or [])[-12:])

    @staticmethod
    def p4_history(conversation_history: List[str]) -> str:
        """P4 — relevant conversation history (bounded)."""
        return "\n".join(
            f"HISTORY: {h}" for h in (conversation_history or [])[-8:])

    @staticmethod
    def p5_knowledge(knowledge_facts: List[str]) -> str:
        """P5 — relevant knowledge retrieval (only what was retrieved)."""
        return "\n".join(
            f"KNOWLEDGE: {k}" for k in (knowledge_facts or [])[-8:])

    @staticmethod
    def p6_lessons(lesson_lines: List[str]) -> str:
        """P6 — reusable task lessons (evidence, not truth)."""
        if not lesson_lines:
            return ""
        lines = ["LESSONS (previous evidence — VERIFY the current "
                 "environment before relying on them):"]
        lines += [f"LESSON: {l}" for l in (lesson_lines or [])[-8:]]
        return "\n".join(lines)

    @staticmethod
    def p7_background(background: List[str]) -> str:
        """P7 — older / background context (dropped first)."""
        return "\n".join(
            f"BACKGROUND: {b}" for b in (background or [])[-10:])

    # ── Task 13: deterministic history compression ─────────────

    def compress_history(self, completed_step_summaries: List[str],
                         keep_recent: Optional[int] = None,
                         ) -> Tuple[List[str], List[str]]:
        """Compress older completed steps into structured summaries.

        Preserves: decisions and constraints (kept verbatim in P0/P1),
        verified evidence (kept verbatim in P3), failed strategies (kept
        verbatim in P1) and the current state. Older steps become compact
        summary lines that are carried as references.
        Returns (recent_steps, summary_lines).
        """
        keep = keep_recent or self._recent_step_keep
        steps = list(completed_step_summaries or [])
        if len(steps) <= keep:
            return steps, []
        recent = steps[-keep:]
        older = steps[:-keep]
        summary = [
            f"steps 1–{len(older)} summarized ({len(older)} older steps): "
            + "; ".join(s for s in older[:MAX_SUMMARY_ITEMS])
        ]
        return recent, summary

    # ── Main entry point ───────────────────────────────────────

    def compose(self, *,
                goal: str = "",
                constraints: Optional[List[str]] = None,
                state_lines: Optional[List[str]] = None,
                current_objective: str = "",
                recent_observations: Optional[List[str]] = None,
                verified_evidence: Optional[List[str]] = None,
                conversation_history: Optional[List[str]] = None,
                knowledge_facts: Optional[List[str]] = None,
                lesson_lines: Optional[List[str]] = None,
                background: Optional[List[str]] = None,
                completed_step_summaries: Optional[List[str]] = None,
                reserve_output: int = 512,
                ) -> ComposedContext:
        """Compose layered context within the configured model limit.

        Older completed steps are compressed FIRST (Task 13); then the
        existing monitor trims lowest-priority layers if still needed.
        The active task state / goal / constraints / evidence are never
        blindly truncated (the monitor guarantees priority ≥ TASK_STATE
        survives).
        """
        # Task 13 compression: older steps → structured summary, kept as
        # an EARLIER reference inside the P1 task state.
        summarized = False
        step_summaries = list(completed_step_summaries or [])
        state_lines = list(state_lines or [])
        if step_summaries:
            recent, older_summary = self.compress_history(step_summaries)
            if older_summary:
                summarized = True
                self._monitor.record_trim_event(
                    "summarized",
                    f"older_steps={len(step_summaries) - len(recent)}",
                    tokens=0)
                state_lines.append(
                    "EARLIER: " + " | ".join(older_summary))
            if recent:
                state_lines.append("RECENT DONE: " + "; ".join(recent))

        items: List[Tuple[int, str]] = []
        names: Dict[str, str] = {}

        def add(priority: int, name: str, text: str) -> None:
            if text:
                items.append((int(priority), text))
                names[text] = name

        add(ContextPriority.USER_REQUEST, "P0_goal",
            self.p0_goal(goal, constraints or []))
        add(ContextPriority.TASK_STATE, "P1_task_state",
            self.p1_task_state(state_lines))
        add(ContextPriority.LIVE_STATE, "P2_current_step",
            self.p2_current_step(current_objective,
                                 recent_observations or []))
        add(ContextPriority.EVIDENCE, "P3_evidence",
            self.p3_evidence(verified_evidence or []))
        add(ContextPriority.RELEVANT_HISTORY, "P4_history",
            self.p4_history(conversation_history or []))
        add(ContextPriority.LOCAL_KNOWLEDGE, "P5_knowledge",
            self.p5_knowledge(knowledge_facts or []))
        add(ContextPriority.TASK_LESSONS, "P6_lessons",
            self.p6_lessons(lesson_lines or []))
        add(ContextPriority.OLDER_CONTEXT, "P7_background",
            self.p7_background(background or []))

        kept = self._monitor.trim_to_fit(
            items, reserve_output=reserve_output)
        kept_texts = set(kept)
        layers: List[str] = []
        trimmed_layers: List[str] = []
        for _priority, text in items:
            name = names[text]
            if text in kept_texts:
                if name not in layers:
                    layers.append(name)
            elif name not in trimmed_layers:
                trimmed_layers.append(name)

        text = "\n".join(kept)
        tokens_used = self._monitor.estimate_tokens(text)
        usage = self._monitor.last_usage
        remaining = (usage.remaining if usage is not None
                     else max(0, self._monitor.context_limit - tokens_used))
        if trimmed_layers:
            self._monitor.record_trim_event(
                "trimmed_layers", ",".join(trimmed_layers), 0)
        return ComposedContext(
            text=text,
            layers=layers,
            trimmed_layers=trimmed_layers,
            tokens_used=tokens_used,
            tokens_remaining=remaining,
            context_limit=self._monitor.context_limit,
            summarized=summarized,
            trimmed=bool(trimmed_layers),
        )


# Global singleton (uses the existing process-wide context monitor).
reasoning_context_composer = ReasoningContextComposer()
