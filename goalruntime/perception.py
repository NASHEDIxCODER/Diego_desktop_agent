"""
GoalRuntime perception recovery — the evidence hierarchy.

Ladder (strictly ordered, never skipped):
    accessibility/DOM/native UI  ->  OCR  ->  vision  ->  justified coordinates

Hard rules:
  * Focus validation BEFORE acting: the expected app/window must actually be
    focused; if not, the attempt fails fast with a recoverable error.
  * After a failed action the NEXT attempt MUST use different evidence or a
    different strategy — never blindly repeat the identical failed action.
  * Coordinates are only ever produced WITH justification (which element,
    from which evidence tier, and why the higher tiers failed).

Logging: [GOAL-PERCEPT]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Evidence tiers
# ═══════════════════════════════════════════════════════════════════

class EvidenceTier(str, Enum):
    ACCESSIBILITY = "accessibility"
    DOM = "dom"
    NATIVE = "native"
    OCR = "ocr"
    VISION = "vision"
    COORDS = "coords"

    @classmethod
    def ladder(cls) -> Tuple["EvidenceTier", ...]:
        return (cls.ACCESSIBILITY, cls.DOM, cls.NATIVE,
                cls.OCR, cls.VISION, cls.COORDS)


# ═══════════════════════════════════════════════════════════════════
# Attempt memory — the anti-repeat rule
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Attempt:
    tier: str
    strategy: str
    target: str
    succeeded: bool
    error: str = ""
    at: float = field(default_factory=time.time)

    def signature(self) -> Tuple[str, str, str]:
        return (self.tier, self.strategy, self.target)


class AttemptMemory:
    """Remembers (tier, strategy, target) triples that already failed.

    The runtime consults this before every action: an identical failed
    (tier, strategy, target) combination is never retried as-is.
    """

    def __init__(self) -> None:
        self._attempts: List[Attempt] = []

    def record(self, tier: str, strategy: str, target: str,
               succeeded: bool, error: str = "") -> None:
        self._attempts.append(Attempt(tier, strategy, target,
                                      succeeded, error))

    def failed(self, tier: str, strategy: str, target: str) -> bool:
        return any(a.signature() == (tier, strategy, target)
                   and not a.succeeded for a in self._attempts)

    def untried(self, tier: str, strategy: str, target: str) -> bool:
        return not self.failed(tier, strategy, target)

    def last_failure(self) -> Optional[Attempt]:
        for a in reversed(self._attempts):
            if not a.succeeded:
                return a
        return None

    def clear(self) -> None:
        self._attempts.clear()


# ═══════════════════════════════════════════════════════════════════
# Focus validation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FocusCheck:
    ok: bool
    expected: str = ""
    observed: str = ""
    window_class: str = ""
    reason: str = ""

    def describe(self) -> str:
        return (f"ok={self.ok} expected={self.expected!r} "
                f"observed={self.observed!r} ({self.reason})")


def validate_focus(backend: Any, expected_app: str) -> FocusCheck:
    """The expected app/window must be focused before acting on it."""
    expected = (expected_app or "").strip().lower()
    if not expected:
        return FocusCheck(ok=True, reason="no app identity to validate")
    try:
        win = backend.active_window()
    except Exception as e:
        return FocusCheck(ok=False, expected=expected, reason=str(e))
    if not win.get("success"):
        return FocusCheck(ok=False, expected=expected,
                          reason="no focused window")
    observed = f"{win.get('title', '')} {win.get('window_class', '')}".lower()
    if expected in observed or observed in expected:
        return FocusCheck(ok=True, expected=expected, observed=observed[:60],
                          reason="focus validated")
    return FocusCheck(
        ok=False, expected=expected,
        observed=str(win.get("title", ""))[:60],
        window_class=str(win.get("window_class", "")),
        reason=f"expected {expected!r} focused, saw something else")


def _windows_containing(backend: Any, needle: str) -> List[Dict[str, Any]]:
    needle = (needle or "").lower()
    try:
        wins = backend.list_windows() or []
    except Exception:
        return []
    return [w for w in wins
            if needle in f"{w.get('title', '')} "
                         f"{w.get('window_class', '')}".lower()]


def recover_focus(backend: Any, expected_app: str) -> FocusCheck:
    """If the wrong window is focused, try to focus the right one (once)."""
    check = validate_focus(backend, expected_app)
    if check.ok:
        return check
    if not _windows_containing(backend, expected_app):
        return check
    r = backend.focus_window(expected_app)
    if r.get("success"):
        return validate_focus(backend, expected_app)
    return check


# ═══════════════════════════════════════════════════════════════════
# Element location — the perception ladder
# ═══════════════════════════════════════════════════════════════════

def locate_element(backend: Any, target: str, *,
                   attempts: AttemptMemory,
                   expected_app: str = "") -> Dict[str, Any]:
    """Walk the perception ladder for ``target``.

    Returns {"found": bool, "tier": str, "evidence": {...}, "reason": str}.
    Tier order: accessibility → DOM → native → OCR → vision → coords
    (coordinates only ever WITH justification).
    """
    target = (target or "").strip()
    out: Dict[str, Any] = {"found": False, "tier": "", "evidence": {},
                           "reason": ""}
    if not target:
        out["reason"] = "empty target"
        return out

    # Focus must be validated/recovered first.
    if expected_app:
        fc = recover_focus(backend, expected_app)
        out["focus"] = fc.describe()
        if not fc.ok:
            out["reason"] = f"focus validation failed: {fc.reason}"
            return out

    # Tiers 1–3: accessibility / DOM / native UI (existing element_finder
    # path behind the backend's search_ui).
    for tier_name in ("accessibility", "dom", "native"):
        if not attempts.untried(tier_name, "find", target):
            continue  # already failed identically — use a DIFFERENT tier
        r = backend.search_ui(target)
        if r.get("success") and r.get("evidence"):
            out.update(found=True, tier=tier_name,
                       evidence=r.get("evidence") or {})
            attempts.record(tier_name, "find", target, True)
            return out
        attempts.record(tier_name, "find", target, False,
                        str(r.get("error") or "no candidates"))

    # Tier 4: OCR
    if attempts.untried("ocr", "find", target):
        r = backend.ocr()
        text = str(r.get("text") or "")
        if r.get("success") and target.lower() in text.lower():
            out.update(found=True, tier="ocr",
                       evidence={"ocr_text": text[:400]})
            attempts.record("ocr", "find", target, True)
            return out
        attempts.record("ocr", "find", target, False,
                        "target not found in OCR text")

    # Tier 5: vision model
    if attempts.untried("vision", "find", target):
        r = backend.vision_query(
            f"Is there a '{target}' element visible on screen? Answer yes/no.")
        answer = str(r.get("text") or "").strip().lower()
        if r.get("success") and answer.startswith("yes"):
            out.update(found=True, tier="vision",
                       evidence={"vision_answer": answer[:200]})
            attempts.record("vision", "find", target, True)
            return out
        attempts.record("vision", "find", target, False,
                        "vision model did not confirm the element")

    # Tier 6: justified coordinates — ONLY with a recorded justification.
    if attempts.untried("coords", "justified", target):
        justification = {
            "justification": (f"all higher evidence tiers failed for "
                              f"'{target}'; coordinates require manual "
                              f"review of the recorded evidence"),
            "target": target,
            "source_tier": "none",
        }
        out.update(found=False, tier="coords", evidence=justification,
                   reason="coordinates require justification; no tier "
                          "produced usable geometry")
        attempts.record("coords", "justified", target, False,
                        "no geometry evidence available")
    return out


# ═══════════════════════════════════════════════════════════════════
# Failure strategy rotation — the next attempt is always DIFFERENT
# ═══════════════════════════════════════════════════════════════════

_FAILURE_STRATEGIES = (
    "refocus_window",
    "refresh_state",
    "alternate_element_source",
    "scroll_and_retry",
    "keyboard_navigation",
    "ocr_reperceive",
    "vision_reperceive",
)


def next_strategy(prev_failed: str) -> str:
    """Return the next DIFFERENT strategy after a failure."""
    for i, s in enumerate(_FAILURE_STRATEGIES):
        if s == prev_failed:
            return (_FAILURE_STRATEGIES[i + 1]
                    if i + 1 < len(_FAILURE_STRATEGIES)
                    else _FAILURE_STRATEGIES[0])
    return _FAILURE_STRATEGIES[0]


__all__ = [
    "EvidenceTier", "Attempt", "AttemptMemory", "FocusCheck",
    "validate_focus", "recover_focus", "locate_element", "next_strategy",
]
