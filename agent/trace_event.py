"""
TraceEvent — structured single step of the computer-use control loop (Phase 22).

The agent trace is a DATA structure, not a log string. Every stage of

    GOAL → PLAN → OBSERVE → ACT → OBSERVE → VERIFY → RECOVER → COMPLETE

is recorded as one TraceEvent carrying the same fields, so the PySide6
workflow panel, the console logs and post-mortem debugging all read the
SAME truth (agent/trace.py owns storage + fan-out).

Additive module: nothing here executes actions or verifies them; it only
describes what the deterministic runtime already did.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TraceEventType(str, Enum):
    """Every lifecycle stage the workflow panel can render."""

    GOAL_RECEIVED = "GOAL_RECEIVED"
    PLAN_CREATED = "PLAN_CREATED"
    STEP_STARTED = "STEP_STARTED"
    OBSERVATION = "OBSERVATION"
    ACTION_STARTED = "ACTION_STARTED"
    ACTION_COMPLETED = "ACTION_COMPLETED"
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    REPLAN_STARTED = "REPLAN_STARTED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    CONFIRMATION_GRANTED = "CONFIRMATION_GRANTED"
    CONFIRMATION_DENIED = "CONFIRMATION_DENIED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    # Phase 23: the coarse OPERATIONAL phase of the workflow (what the
    # workflow panel shows instead of private chain-of-thought).
    PHASE_CHANGED = "PHASE_CHANGED"


# Terminal events — a task that emits one of these is finished.
TERMINAL_EVENTS = frozenset({
    TraceEventType.TASK_COMPLETED, TraceEventType.TASK_FAILED,
})


@dataclass
class TraceEvent:
    """One structured trace record. Safe to serialize and to render raw."""

    event_type: TraceEventType
    task_id: str = ""
    step_id: str = ""
    action: str = ""
    target: str = ""
    method: str = ""                 # perception/execution method used
    expected_effect: str = ""
    observation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    verification_result: str = ""    # PASS | FAIL | AMBIGUOUS | NO_EVIDENCE | ""
    retry_count: int = 0
    risk: str = ""                   # ActionRisk value when relevant
    detail: str = ""
    timestamp: float = field(default_factory=time.time)
    seq: int = 0                     # monotonic id assigned by the store

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able form (used by the UI panel + EventBus payloads)."""
        return {
            "seq": self.seq,
            "event_type": self.event_type.value,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "action": self.action,
            "target": self.target,
            "method": self.method,
            "expected_effect": self.expected_effect,
            "observation": self.observation,
            "evidence": dict(self.evidence),
            "verification_result": self.verification_result,
            "retry_count": self.retry_count,
            "risk": self.risk,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEvent":
        return cls(
            event_type=TraceEventType(str(data.get("event_type"))),
            task_id=str(data.get("task_id", "")),
            step_id=str(data.get("step_id", "")),
            action=str(data.get("action", "")),
            target=str(data.get("target", "")),
            method=str(data.get("method", "")),
            expected_effect=str(data.get("expected_effect", "")),
            observation=str(data.get("observation", "")),
            evidence=dict(data.get("evidence") or {}),
            verification_result=str(data.get("verification_result", "")),
            retry_count=int(data.get("retry_count", 0) or 0),
            risk=str(data.get("risk", "")),
            detail=str(data.get("detail", "")),
            timestamp=float(data.get("timestamp", time.time())),
            seq=int(data.get("seq", 0) or 0),
        )

    def log_line(self) -> str:
        """One-line structured log entry (never contains raw screenshots)."""
        bits = [f"[TRACE] {self.event_type.value}"]
        if self.task_id:
            bits.append(f"task={self.task_id}")
        if self.step_id:
            bits.append(f"step={self.step_id}")
        if self.action:
            bits.append(f"action={self.action}")
        if self.target:
            bits.append(f"target='{str(self.target)[:60]}'")
        if self.method:
            bits.append(f"method={self.method}")
        if self.risk:
            bits.append(f"risk={self.risk}")
        if self.verification_result:
            bits.append(f"verify={self.verification_result}")
        if self.retry_count:
            bits.append(f"retry={self.retry_count}")
        if self.expected_effect:
            bits.append(f"expected='{str(self.expected_effect)[:80]}'")
        if self.observation:
            bits.append(f"observed='{str(self.observation)[:120]}'")
        if self.detail:
            bits.append(f"detail='{str(self.detail)[:160]}'")
        return " ".join(bits)


def make_event(event_type: TraceEventType, *, task_id: str = "",
               **kwargs: Any) -> TraceEvent:
    """Convenience constructor that accepts either enum or raw string type."""
    if not isinstance(event_type, TraceEventType):
        event_type = TraceEventType(str(event_type))
    kwargs.pop("event_type", None)
    return TraceEvent(event_type=event_type, task_id=task_id, **kwargs)


def coerce_event_type(value: Optional[str]) -> Optional[TraceEventType]:
    if not value:
        return None
    try:
        return TraceEventType(str(value))
    except ValueError:
        return None
