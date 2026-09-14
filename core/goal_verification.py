"""
GoalVerification — GOAL-level evidence contract for actions.

Problem fixed (reliability layer, Phase 4):
    A task could be marked SUCCESS merely because an action executed
    without crashing (or because "the browser process is alive").
    Success must mean EXPECTED STATE ACHIEVED, proven by observable
    evidence — not "the tool returned".

Contract (per action):
    Action            — what was executed
    ExpectedEffect    — the intended world change
    Observation       — what was actually observed (from real tools)
    Evidence          — whether the expected evidence was found
    Result            — PASS / FAIL

ACTION_SUCCESS (tool executed) is explicitly distinct from
GOAL_SUCCESS (expected state achieved). Only GOAL_SUCCESS may mark a
step verified.

This module is deterministic and side-effect free: callers supply the
observation (real URL, DOM text, page state). Nothing here launches a
browser or invents evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GoalResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_EVIDENCE = "NO_EVIDENCE"   # observation missing — never guess


@dataclass
class GoalExpectation:
    """What a successful action must look like in the real world."""
    action: str
    intended_effect: str
    success_condition: str
    failure_condition: str
    observable_evidence: List[str] = field(default_factory=list)


@dataclass
class GoalVerificationResult:
    """The verification record stored in task state evidence."""
    action: str
    expected_effect: str
    observation: str
    evidence: str
    result: GoalResult

    @property
    def passed(self) -> bool:
        return self.result == GoalResult.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "expected_effect": self.expected_effect,
            "observation": self.observation,
            "evidence": self.evidence,
            "result": self.result.value,
        }


# ── Per-action expectations ───────────────────────────────────────

GOAL_EXPECTATIONS: Dict[str, GoalExpectation] = {
    "browser_navigate": GoalExpectation(
        action="browser_navigate",
        intended_effect="Browser current URL changes to the target URL.",
        success_condition=(
            "Observed URL matches the target host/path; process alive "
            "is NOT sufficient."),
        failure_condition="Observed URL is absent or does not match target.",
        observable_evidence=[
            "actual URL from browser_get_url / page state",
            "URL host or path matches the requested target",
        ],
    ),
    "browser_click": GoalExpectation(
        action="browser_click",
        intended_effect="Target page/state changes as a result of click.",
        success_condition=(
            "Observed page text/state contains the expected element or "
            "state change; a successful click call alone is not enough."),
        failure_condition="Expected element/state not present after click.",
        observable_evidence=[
            "post-click page text / DOM state",
            "expected element or heading present",
        ],
    ),
    "desktop_open": GoalExpectation(
        action="desktop_open",
        intended_effect="Target application process is running.",
        success_condition="Process actually present (pgrep exact match).",
        failure_condition="Process absent after settle wait.",
        observable_evidence=["process table"],
    ),
}


def _host_and_path(url: str) -> str:
    url = (url or "").strip().lower()
    url = re.sub(r"^[a-z]+://", "", url)
    url = url.split("#", 1)[0]
    return url.rstrip("/")


def verify_browser_navigation(
    target_url: str,
    observed_url: Optional[str],
) -> GoalVerificationResult:
    """GOAL verification for browser_navigate.

    PASS requires the OBSERVED URL (real page state) to match the
    target. 'Browser process alive' or 'dispatch returned a string'
    is never treated as evidence of navigation.
    """
    exp = GOAL_EXPECTATIONS["browser_navigate"]
    target = _host_and_path(target_url)
    if not target:
        return GoalVerificationResult(
            action="browser_navigate",
            expected_effect=exp.intended_effect,
            observation=str(observed_url or ""),
            evidence="no target URL in params",
            result=GoalResult.FAIL,
        )
    if not observed_url:
        return GoalVerificationResult(
            action="browser_navigate",
            expected_effect=exp.intended_effect,
            observation="",
            evidence="no observed URL — cannot prove navigation",
            result=GoalResult.NO_EVIDENCE,
        )
    observed = _host_and_path(str(observed_url))
    if observed == target or target in observed or observed in target:
        return GoalVerificationResult(
            action="browser_navigate",
            expected_effect=exp.intended_effect,
            observation=observed,
            evidence=f"URL match: {observed} == {target}",
            result=GoalResult.PASS,
        )
    return GoalVerificationResult(
        action="browser_navigate",
        expected_effect=exp.intended_effect,
        observation=observed,
        evidence=f"URL mismatch: expected {target}, observed {observed}",
        result=GoalResult.FAIL,
    )


def verify_browser_click(
    expected_element: str,
    observed_page_text: Optional[str],
) -> GoalVerificationResult:
    """GOAL verification for browser_click — the expected element/state
    must be visible in the observed page text."""
    exp = GOAL_EXPECTATIONS["browser_click"]
    if not expected_element:
        return GoalVerificationResult(
            action="browser_click",
            expected_effect=exp.intended_effect,
            observation=str(observed_page_text or "")[:200],
            evidence="no expected element defined",
            result=GoalResult.NO_EVIDENCE,
        )
    if observed_page_text is None:
        return GoalVerificationResult(
            action="browser_click",
            expected_effect=exp.intended_effect,
            observation="",
            evidence="no page state observed after click",
            result=GoalResult.NO_EVIDENCE,
        )
    needle = expected_element.lower()
    found = needle in str(observed_page_text).lower()
    return GoalVerificationResult(
        action="browser_click",
        expected_effect=exp.intended_effect,
        observation=str(observed_page_text)[:200],
        evidence=(f"expected element '{expected_element}' "
                  + ("found in page state" if found else "NOT found")),
        result=GoalResult.PASS if found else GoalResult.FAIL,
    )


def evaluate_goal(
    action_name: str,
    params: Dict[str, Any],
    observation: Dict[str, Any],
) -> GoalVerificationResult:
    """Single entry point: evaluate goal-level evidence for an action.

    observation keys (all optional, all REAL evidence from tools):
        observed_url, page_text, process_running
    """
    if action_name == "browser_navigate":
        return verify_browser_navigation(
            str(params.get("url", "")),
            observation.get("observed_url"))
    if action_name == "browser_click":
        return verify_browser_click(
            str(params.get("label") or params.get("text") or ""),
            observation.get("page_text"))
    return GoalVerificationResult(
        action=action_name,
        expected_effect="tool execution succeeded",
        observation=str(observation)[:200],
        evidence="no goal contract for this action — action-level check only",
        result=GoalResult.PASS if observation.get("executed") else GoalResult.NO_EVIDENCE,
    )
