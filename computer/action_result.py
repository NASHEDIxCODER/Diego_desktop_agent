"""Phase 22: structured action results (additive)."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ActionOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"            # target unclear — must ASK, never guess
    CONFIRMATION_REQUIRED = "confirmation_required"
    LOGIN_REQUIRED = "login_required"  # never bypass auth/CAPTCHA
    UNAVAILABLE = "unavailable"        # capability/backend missing
    BLOCKED = "blocked"                # policy blocked


@dataclass
class ActionResult:
    """Structured result for every computer action (never a bare string)."""
    action: str
    target: str = ""
    method: str = ""                   # e.g. app_resolver, accessibility, dom, ocr, vision, coordinate
    success: bool = False
    outcome: ActionOutcome = ActionOutcome.FAILED
    evidence: Dict[str, Any] = field(default_factory=dict)
    verification: str = ""             # target-specific verification detail
    error: str = ""
    expected_effect: str = ""
    observed_effect: str = ""
    retry_count: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action, "target": self.target, "method": self.method,
            "success": self.success, "outcome": self.outcome.value,
            "evidence": dict(self.evidence), "verification": self.verification,
            "error": self.error, "expected_effect": self.expected_effect,
            "observed_effect": self.observed_effect, "retry_count": self.retry_count,
            "latency_ms": self.latency_ms,
        }
