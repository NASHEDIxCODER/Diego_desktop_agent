"""
GoalRuntime core contracts (pure dataclasses + enums — no I/O).

Logging: none (pure). Logging happens in runtime/skills layers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Permission classes
# ═══════════════════════════════════════════════════════════════════

class PermissionClass(str, Enum):
    """The five permission classes every runtime action declares.

    READ_ONLY            — observe only; never changes any state → auto-run
    REVERSIBLE_LOCAL     — local change that can be undone locally → auto-run
    EXTERNAL_SIDE_EFFECT — affects the outside world (send/publish/purchase)
    DESTRUCTIVE          — deletes/overwrites/irreversibly destroys state
    SECURITY_TESTING     — probes/scans; requires explicit target+scope
    """

    READ_ONLY = "read_only"
    REVERSIBLE_LOCAL = "reversible_local"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    DESTRUCTIVE = "destructive"
    SECURITY_TESTING = "security_testing"

    @property
    def auto_run(self) -> bool:
        return self in (PermissionClass.READ_ONLY,
                        PermissionClass.REVERSIBLE_LOCAL)

    @property
    def needs_confirmation(self) -> bool:
        return self in (PermissionClass.EXTERNAL_SIDE_EFFECT,
                        PermissionClass.DESTRUCTIVE,
                        PermissionClass.SECURITY_TESTING)


# ═══════════════════════════════════════════════════════════════════
# Runtime statuses
# ═══════════════════════════════════════════════════════════════════

class GoalStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    VERIFYING = "verifying"
    REPLANNING = "replanning"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (GoalStatus.COMPLETED, GoalStatus.FAILED,
                        GoalStatus.CANCELLED)


# ═══════════════════════════════════════════════════════════════════
# Artifacts — context passed between subgoals
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Artifact:
    """A piece of produced context one subgoal hands to later subgoals."""

    name: str
    kind: str = "text"            # text | list | path | screenshot | json
    value: Any = None
    producer: str = ""            # producing subgoal id
    created_at: float = field(default_factory=time.time)

    def describe(self) -> str:
        v = str(self.value)
        if len(v) > 80:
            v = v[:77] + "..."
        return f"{self.name}({self.kind})={v!r} from={self.producer or '?'}"


# ═══════════════════════════════════════════════════════════════════
# Subgoals
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Subgoal:
    """One capability-scoped step of a decomposed goal."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    description: str = ""
    skill_id: str = ""            # e.g. "browser", "telegram", "filesystem"
    action: str = ""              # skill action, e.g. "search_contact"
    params: Dict[str, Any] = field(default_factory=dict)
    consumes: List[str] = field(default_factory=list)   # artifact names
    produces: List[str] = field(default_factory=list)   # artifact names
    permission_class: PermissionClass = PermissionClass.READ_ONLY
    status: GoalStatus = GoalStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    required: bool = True         # advisory subgoals may fail without
                                  # failing the goal
    error: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"[{self.id}] {self.skill_id}.{self.action} {self.description}"

    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts


# ═══════════════════════════════════════════════════════════════════
# Observations / results
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """What the runtime saw after an action (or before the next one)."""

    success: bool
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    evidence_level: str = ""      # accessibility | dom | native | ocr | vision | coords
    at: float = field(default_factory=time.time)


@dataclass
class SkillResult:
    """Structured result of one skill action execution."""

    skill_id: str
    action: str
    success: bool
    observation: Observation = field(default_factory=lambda: Observation(success=False))
    expected_effect: str = ""
    verification: str = ""        # human-readable verification verdict
    verified: bool = False
    permission_class: PermissionClass = PermissionClass.READ_ONLY
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    artifacts: List[Artifact] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill_id, "action": self.action,
            "success": self.success, "verified": self.verified,
            "expected_effect": self.expected_effect,
            "verification": self.verification,
            "permission": self.permission_class.value,
            "evidence": self.evidence, "error": self.error,
            "artifacts": [a.name for a in self.artifacts],
            "observation": {
                "text": self.observation.text,
                "evidence_level": self.observation.evidence_level,
                "data": self.observation.data,
            },
        }


__all__ = [
    "PermissionClass", "GoalStatus", "Artifact", "Subgoal",
    "Observation", "SkillResult",
]
