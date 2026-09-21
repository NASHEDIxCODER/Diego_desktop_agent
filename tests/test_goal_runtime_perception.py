"""GoalRuntime perception — evidence ladder, focus validation, anti-repeat."""

from __future__ import annotations

import pytest

from goalruntime.perception import (
    AttemptMemory, EvidenceTier, FocusCheck, locate_element, next_strategy,
    recover_focus, validate_focus,
)
from tests.fakes_goal_backend import SimulatedBackend


@pytest.fixture
def backend(tmp_path):
    return SimulatedBackend(tmp_path)


# ═══════════════════════════════════════════════════════════════════
# Evidence tier ladder
# ═══════════════════════════════════════════════════════════════════

def test_ladder_order():
    tiers = EvidenceTier.ladder()
    assert tiers == (EvidenceTier.ACCESSIBILITY, EvidenceTier.DOM,
                     EvidenceTier.NATIVE, EvidenceTier.OCR,
                     EvidenceTier.VISION, EvidenceTier.COORDS)


def test_accessibility_tier_wins_when_available(backend):
    backend.open_app("telegram")
    backend.ui_search_hits = ["Send button"]
    attempts = AttemptMemory()
    out = locate_element(backend, "Send button", attempts=attempts,
                         expected_app="telegram")
    assert out["found"] and out["tier"] == "accessibility"


def test_ocr_tier_used_when_ui_search_fails(backend):
    backend.open_app("telegram")
    backend.ui_search_fail_until = 10          # accessibility/DOM/native fail
    backend.ocr_text = "the Send button is visible on screen"
    attempts = AttemptMemory()
    out = locate_element(backend, "Send", attempts=attempts,
                         expected_app="telegram")
    assert out["found"] and out["tier"] == "ocr"


def test_vision_tier_used_when_ocr_fails(backend):
    backend.open_app("telegram")
    backend.ui_search_fail_until = 10
    backend.ocr_text = ""                       # OCR sees nothing
    backend.vision_answer = "yes, the Send button is visible"
    attempts = AttemptMemory()
    out = locate_element(backend, "Send", attempts=attempts,
                         expected_app="telegram")
    assert out["found"] and out["tier"] == "vision"


def test_coords_require_justification(backend):
    backend.open_app("telegram")
    backend.ui_search_fail_until = 10
    backend.ocr_text = ""
    backend.vision_answer = "no"
    attempts = AttemptMemory()
    out = locate_element(backend, "Send", attempts=attempts,
                         expected_app="telegram")
    # NOT found blindly: coordinates are only produced WITH justification.
    assert out["tier"] in ("coords", "") 
    if out["tier"] == "coords":
        assert "justification" in out["evidence"]
        assert out["found"] is False or "justification" in out["evidence"]


# ═══════════════════════════════════════════════════════════════════
# Anti-repeat rule
# ═══════════════════════════════════════════════════════════════════

def test_failed_combination_is_never_retried_identically(backend):
    backend.open_app("telegram")
    backend.ui_search_fail_until = 1            # first find call fails
    attempts = AttemptMemory()
    out1 = locate_element(backend, "Send", attempts=attempts,
                          expected_app="telegram")
    # The failed (tier, strategy, target) is recorded.
    assert attempts.failed("accessibility", "find", "Send")
    # A second identical attempt must SKIP the recorded-failed tier.
    out2 = locate_element(backend, "Send", attempts=attempts,
                          expected_app="telegram")
    assert out2["tier"] != "accessibility"      # DIFFERENT evidence tier


def test_untried_allows_retry_after_recovery(backend):
    attempts = AttemptMemory()
    assert attempts.untried("dom", "find", "Send")
    attempts.record("dom", "find", "Send", False, "gone")
    assert not attempts.untried("dom", "find", "Send")
    attempts.clear()
    assert attempts.untried("dom", "find", "Send")   # new context resets


def test_strategy_rotation_never_repeats():
    from goalruntime.perception import _FAILURE_STRATEGIES
    first = _FAILURE_STRATEGIES[0]
    seen = [first]
    cur = first
    for _ in range(len(_FAILURE_STRATEGIES)):
        cur = next_strategy(cur)
        assert cur != seen[-1]           # never repeat the failed strategy
        seen.append(cur)
    assert set(seen) == set(_FAILURE_STRATEGIES)   # full cycle covers all


# ═══════════════════════════════════════════════════════════════════
# Focus validation
# ═══════════════════════════════════════════════════════════════════

def test_focus_validation_blocks_wrong_app(backend):
    backend.windows = {"chrome": "Google Chrome"}
    backend._focused = "chrome"
    check = validate_focus(backend, "telegram")
    assert not check.ok


def test_focus_recovery_refocuses(backend):
    backend.windows = {"chrome": "Google Chrome",
                       "telegram": "Telegram Desktop"}
    backend._focused = "chrome"
    check = recover_focus(backend, "telegram")
    assert check.ok
    assert backend._focused == "telegram"


def test_focus_validation_passes_when_correct(backend):
    backend.open_app("telegram")
    assert validate_focus(backend, "telegram").ok


def test_empty_expectation_is_neutral(backend):
    assert validate_focus(backend, "").ok


# ═══════════════════════════════════════════════════════════════════
# FocusCheck contract
# ═══════════════════════════════════════════════════════════════════

def test_focus_check_describes_itself():
    fc = FocusCheck(ok=False, expected="telegram", observed="Chrome",
                    reason="wrong window")
    d = fc.describe()
    assert "telegram" in d and "Chrome" in d and "wrong window" in d
