"""
AgentTrace — structured, thread-safe store + fan-out for the Phase 22
computer-use control loop.

This is the single source of truth for what the agent is doing:

    GOAL → PLAN → OBSERVE → ACT → OBSERVE → VERIFY → RECOVER → COMPLETE

Design constraints (Phase 22):
  - Events are STRUCTURED OBJECTS (agent.trace_event.TraceEvent), never only
    log strings. Everything is JSON-serializable.
  - The store is USABLE WITHOUT THE UI: no Qt import here, ever. The PySide6
    panel subscribes via `subscribe()`; the EventBus fan-out is best-effort.
  - Bounded memory: a fixed-size ring of events + one active workflow.
  - Thread-safe: actions can be dispatched from worker threads (the existing
    dispatcher runs sync actions in a thread pool), so `record()` locks.
  - Additive: nothing here executes or verifies actions. The deterministic
    runtime remains authoritative; the trace only describes what it did.

Usage:
    from agent.trace import agent_trace

    agent_trace.goal("open firefox and search for python")
    agent_trace.plan(["open firefox", "search"], task_id=..., step_id=...)
    agent_trace.action_started("click", target="Search", method="accessibility")
    agent_trace.action_completed(action_result)
    agent_trace.verification(result="PASS", evidence={"url": "..."})
    agent_trace.completed("Opened Firefox.")

Logging contract: [TRACE]
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from agent.trace_event import (
    TERMINAL_EVENTS,
    TraceEvent,
    TraceEventType,
    coerce_event_type,
)

logger = logging.getLogger(__name__)

# How many events to retain for post-mortem + UI backfill.
_MAX_EVENTS = 500

# Task status values surfaced by WorkflowSnapshot (mirrors the panel states).
STATUS_IDLE = "IDLE"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_WAITING_CONFIRMATION = "WAITING_CONFIRMATION"

# Phase 23: coarse OPERATIONAL phases shown by the workflow panel. These are
# concise, user-facing states — never hidden model reasoning.
PHASE_IDLE = "IDLE"
PHASE_THINKING = "THINKING"
PHASE_PLANNING = "PLANNING"
PHASE_OBSERVING = "OBSERVING"
PHASE_ACTING = "ACTING"
PHASE_WAITING = "WAITING"
PHASE_VERIFYING = "VERIFYING"
PHASE_RECOVERING = "RECOVERING"
PHASE_ASKING_USER = "ASKING_USER"
PHASE_COMPLETED = "COMPLETED"
PHASE_FAILED = "FAILED"

PHASES = (
    PHASE_IDLE, PHASE_THINKING, PHASE_PLANNING, PHASE_OBSERVING, PHASE_ACTING,
    PHASE_WAITING, PHASE_VERIFYING, PHASE_RECOVERING, PHASE_ASKING_USER,
    PHASE_COMPLETED, PHASE_FAILED,
)


@dataclass
class WorkflowSnapshot:
    """Everything the Agent Workflow panel renders, derived from events.

    Kept deliberately flat: the UI reads fields, it never parses logs.
    """

    task_id: str = ""
    status: str = STATUS_IDLE
    goal: str = ""
    plan: List[str] = field(default_factory=list)
    current_step: str = ""
    current_step_index: int = 0
    total_steps: int = 0
    action: str = ""
    target: str = ""
    method: str = ""
    expected_effect: str = ""
    observation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    verification: str = ""
    verification_detail: str = ""
    recovery: str = ""
    confirmation: str = ""
    confirmation_required: bool = False
    final_result: str = ""
    last_error: str = ""
    retry_count: int = 0
    replan_count: int = 0
    updated_at: float = 0.0
    step_history: List[Dict[str, Any]] = field(default_factory=list)
    # Phase 23: coarse operational phase (THINKING…FAILED) + its short detail.
    phase: str = PHASE_IDLE
    phase_detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["plan"] = list(self.plan)
        d["evidence"] = dict(self.evidence)
        d["step_history"] = list(self.step_history)
        return d

    def is_active(self) -> bool:
        return self.status == STATUS_RUNNING


class AgentTrace:
    """Bounded, thread-safe record of the agent's computer-use workflow."""

    def __init__(self, max_events: int = _MAX_EVENTS) -> None:
        self._lock = threading.RLock()
        self._events: Deque[TraceEvent] = deque(maxlen=max_events)
        self._seq = 0
        self._subscribers: List[Callable[[TraceEvent], None]] = []
        self._workflow = WorkflowSnapshot()
        self._event_bus = None
        self._bus_wired = False

    # ── Subscriptions (UI side) ───────────────────────────────

    def subscribe(self, callback: Callable[[TraceEvent], None]) -> Callable[[], None]:
        """Register a listener; returns an unsubscribe callable.

        Listeners are called on the RECORDING thread — a Qt listener must
        marshal to the GUI thread itself (see ui/agent_workflow_panel.py,
        which uses the existing EventBridge queue for exactly that reason).
        Listener exceptions are swallowed so the agent loop never breaks.
        """
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return _unsubscribe

    def clear_subscribers(self) -> None:
        with self._lock:
            self._subscribers.clear()

    # ── Recording ─────────────────────────────────────────────

    def record(self, event: TraceEvent) -> TraceEvent:
        """Store an event, update the workflow snapshot, fan it out."""
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            if not event.timestamp:
                event.timestamp = time.time()
            self._events.append(event)
            self._apply(event)
            subscribers = list(self._subscribers)

        # Log outside the lock (handlers can be slow).
        try:
            logger.info(event.log_line())
        except Exception:  # pragma: no cover - logging must never raise
            pass

        for cb in subscribers:
            try:
                cb(event)
            except Exception as e:
                logger.debug("[TRACE] subscriber failed: %s", e)

        self._emit_to_bus(event)
        return event

    def emit(self, event_type: Any, *, task_id: str = "", **kwargs: Any) -> TraceEvent:
        """Record an event by type (enum or raw string)."""
        etype = coerce_event_type(event_type) if not isinstance(
            event_type, TraceEventType) else event_type
        if etype is None:
            logger.debug("[TRACE] unknown event type '%s' ignored", event_type)
            raise ValueError(f"unknown trace event type: {event_type!r}")
        with self._lock:
            if not task_id:
                task_id = self._workflow.task_id
        return self.record(TraceEvent(event_type=etype, task_id=task_id, **kwargs))

    def _apply(self, event: TraceEvent) -> None:
        """Fold one event into the workflow snapshot (never raises)."""
        w = self._workflow
        try:
            et = event.event_type
            if et == TraceEventType.GOAL_RECEIVED:
                return  # goal() already reset the snapshot
            if et == TraceEventType.PLAN_CREATED:
                w.plan = [str(s) for s in (event.evidence or {}).get("steps")
                          or []]
                w.total_steps = len(w.plan)
            elif et == TraceEventType.STEP_STARTED:
                w.current_step = event.detail[:120]
                idx = int((event.evidence or {}).get("index", 0) or 0)
                w.current_step_index = idx
                w.step_history.append({"step": event.detail[:120],
                                       "index": idx})
            elif et == TraceEventType.OBSERVATION:
                w.observation = (event.observation or event.detail)[:400]
            elif et == TraceEventType.ACTION_STARTED:
                w.action, w.target, w.method = (event.action, event.target,
                                                event.method)
                w.expected_effect = event.expected_effect
            elif et == TraceEventType.ACTION_COMPLETED:
                w.evidence = dict(event.evidence or {})
                w.verification_detail = event.detail[:200]
                if event.verification_result:
                    w.verification = event.verification_result
            elif et == TraceEventType.VERIFICATION_STARTED:
                w.verification = "PENDING"
            elif et == TraceEventType.VERIFICATION_COMPLETED:
                w.verification = event.verification_result
                w.observation = (event.observation or w.observation)[:400]
                w.evidence = dict(event.evidence or w.evidence)
            elif et == TraceEventType.RECOVERY_STARTED:
                w.recovery = event.detail[:160]
            elif et == TraceEventType.REPLAN_STARTED:
                w.replan_count += 1
                steps = [str(s) for s in
                         (event.evidence or {}).get("steps") or []]
                if steps:
                    w.plan = steps
            elif et == TraceEventType.CONFIRMATION_REQUIRED:
                w.confirmation_required = True
                w.confirmation = event.detail[:200]
                w.status = STATUS_WAITING_CONFIRMATION
                w.phase = PHASE_ASKING_USER
            elif et == TraceEventType.CONFIRMATION_GRANTED:
                w.confirmation_required = False
                w.status = STATUS_RUNNING
                w.confirmation = event.detail[:200] or w.confirmation
            elif et == TraceEventType.CONFIRMATION_DENIED:
                w.confirmation_required = False
                w.status = STATUS_FAILED
                w.final_result = "cancelled: confirmation denied"
            elif et == TraceEventType.TASK_COMPLETED:
                w.status = STATUS_COMPLETED
                w.final_result = event.detail[:300]
                w.current_step = ""
                w.phase = PHASE_COMPLETED
            elif et == TraceEventType.TASK_FAILED:
                w.status = STATUS_FAILED
                w.final_result = event.detail[:300]
                w.last_error = event.detail[:300]
                w.current_step = ""
                w.phase = PHASE_FAILED
            elif et == TraceEventType.PHASE_CHANGED:
                if event.detail:
                    w.phase = str(event.detail)[:32]
                w.phase_detail = str(event.observation or "")[:200]
            if event.retry_count:
                w.retry_count = max(w.retry_count, event.retry_count)
            w.updated_at = event.timestamp or time.time()
        except Exception as e:
            logger.debug("[TRACE] snapshot apply failed: %s", e)

    def _emit_to_bus(self, event: TraceEvent) -> None:
        """Best-effort EventBus fan-out; the store works without it."""
        try:
            if self._event_bus is None:
                from core.event_bus import event_bus
                self._event_bus = event_bus

            import asyncio

            coro = self._event_bus.emit("agent.trace", data=event.to_dict(),
                                        source="trace")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, loop)
            else:
                loop.run_until_complete(coro)
        except Exception as e:
            self._bus_wired = False
            logger.debug("[TRACE] bus fan-out skipped: %s", e)

    # ── Convenience emitters (used by computer/ + brain + task_state) ──

    def goal(self, description: str, *, task_id: str = "",
             **kwargs: Any) -> TraceEvent:
        """Start a new workflow. Resets the snapshot for this task."""
        with self._lock:
            if task_id and task_id == self._workflow.task_id:
                pass
            else:
                self._workflow = WorkflowSnapshot(
                    task_id=task_id, status=STATUS_RUNNING, goal=description,
                    phase=PHASE_THINKING,
                    phase_detail="understanding the request")
        return self.emit(TraceEventType.GOAL_RECEIVED, task_id=task_id,
                         detail=description, **kwargs)

    def phase(self, phase: str, *, detail: str = "", task_id: str = "",
              **kwargs: Any) -> TraceEvent:
        """Set the coarse operational phase (THINKING…FAILED).

        `detail` is a SHORT user-facing operational line ("Search box found"),
        never hidden model reasoning.
        """
        return self.emit(TraceEventType.PHASE_CHANGED, task_id=task_id,
                         detail=str(phase)[:32], observation=str(detail)[:200],
                         **kwargs)

    def plan(self, steps: List[str], *, task_id: str = "",
             **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.PLAN_CREATED, task_id=task_id,
                         detail=" → ".join(str(s) for s in steps),
                         evidence={"steps": [str(s) for s in steps]}, **kwargs)

    def step_started(self, description: str, *, index: int = 0,
                     total: int = 0, step_id: str = "", task_id: str = "",
                     **kwargs: Any) -> TraceEvent:
        return self.emit(
            TraceEventType.STEP_STARTED, task_id=task_id,
            step_id=step_id or (f"step-{index}" if index else ""),
            detail=description,
            evidence={"index": index, "total": total}, **kwargs)

    def observation(self, text: str, *, task_id: str = "",
                    step_id: str = "", method: str = "",
                    evidence: Optional[Dict[str, Any]] = None,
                    **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.OBSERVATION, task_id=task_id,
                         step_id=step_id, method=method, observation=text,
                         evidence=dict(evidence or {}), **kwargs)

    def action_started(self, action: str, *, target: str = "", method: str = "",
                       expected_effect: str = "", risk: str = "",
                       task_id: str = "", step_id: str = "",
                       **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.ACTION_STARTED, task_id=task_id,
                         step_id=step_id, action=action, target=target,
                         method=method, expected_effect=expected_effect,
                         risk=risk, **kwargs)

    def action_completed(self, action: str, *, success: bool = False,
                         target: str = "", method: str = "",
                         observed_effect: str = "", error: str = "",
                         evidence: Optional[Dict[str, Any]] = None,
                         retry_count: int = 0, risk: str = "",
                         latency_ms: float = 0.0, task_id: str = "",
                         step_id: str = "", **kwargs: Any) -> TraceEvent:
        kwargs.pop("verification_result", None)  # avoid duplicate-kwarg crash
        return self.emit(
            TraceEventType.ACTION_COMPLETED, task_id=task_id, step_id=step_id,
            action=action, target=target, method=method, risk=risk,
            observation=observed_effect, evidence=dict(evidence or {}),
            retry_count=retry_count, detail=error,
            verification_result="", **kwargs)

    def verification(self, result: str, *, action: str = "", target: str = "",
                     method: str = "", expected_effect: str = "",
                     observed: str = "", evidence: Optional[Dict[str, Any]] = None,
                     detail: str = "", retry_count: int = 0,
                     task_id: str = "", step_id: str = "", **kwargs: Any) -> TraceEvent:
        return self.emit(
            TraceEventType.VERIFICATION_COMPLETED, task_id=task_id,
            step_id=step_id, action=action, target=target, method=method,
            expected_effect=expected_effect, observation=observed,
            evidence=dict(evidence or {}), verification_result=str(result),
            retry_count=retry_count, detail=detail, **kwargs)

    def verification_started(self, *, action: str = "", target: str = "",
                             expected_effect: str = "", task_id: str = "",
                             step_id: str = "", **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.VERIFICATION_STARTED, task_id=task_id,
                         step_id=step_id, action=action, target=target,
                         expected_effect=expected_effect, **kwargs)

    def recovery(self, strategy: str, *, action: str = "", retry_count: int = 0,
                 task_id: str = "", step_id: str = "", **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.RECOVERY_STARTED, task_id=task_id,
                         step_id=step_id, action=action, detail=strategy,
                         retry_count=retry_count, **kwargs)

    def replan(self, reason: str, steps: Optional[List[str]] = None,
               *, task_id: str = "", **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.REPLAN_STARTED, task_id=task_id,
                         detail=reason,
                         evidence={"steps": [str(s) for s in (steps or [])]},
                         **kwargs)

    def confirmation_required(self, reason: str, *, action: str = "",
                              target: str = "", risk: str = "",
                              task_id: str = "", **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.CONFIRMATION_REQUIRED, task_id=task_id,
                         action=action, target=target, risk=risk,
                         detail=reason, verification_result="AMBIGUOUS",
                         **kwargs)

    def confirmation_resolved(self, granted: bool, *, action: str = "",
                              target: str = "", detail: str = "",
                              task_id: str = "", **kwargs: Any) -> TraceEvent:
        etype = (TraceEventType.CONFIRMATION_GRANTED if granted
                 else TraceEventType.CONFIRMATION_DENIED)
        return self.emit(etype, task_id=task_id, action=action, target=target,
                         detail=detail, **kwargs)

    def completed(self, summary: str = "", *, task_id: str = "",
                  evidence: Optional[Dict[str, Any]] = None,
                  **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.TASK_COMPLETED, task_id=task_id,
                         detail=summary, evidence=dict(evidence or {}), **kwargs)

    def failed(self, reason: str, *, action: str = "", target: str = "",
               task_id: str = "", evidence: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> TraceEvent:
        return self.emit(TraceEventType.TASK_FAILED, task_id=task_id,
                         action=action, target=target, detail=reason,
                         evidence=dict(evidence or {}), **kwargs)

    # ── Read access (UI / tests) ──────────────────────────────

    def snapshot(self) -> WorkflowSnapshot:
        """Current workflow state (cheap; UI polls or subscribes)."""
        with self._lock:
            return self._workflow

    def events(self, limit: int = 100) -> List[TraceEvent]:
        """Most recent events, oldest first (for UI backfill)."""
        with self._lock:
            return list(self._events)[-limit:]


# Global singleton
agent_trace = AgentTrace()
