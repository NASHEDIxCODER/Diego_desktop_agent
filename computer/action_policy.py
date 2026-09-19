"""Phase 22: computer action risk policy (additive, mirrors task_state safety)."""
from __future__ import annotations
import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class ActionRisk(str, Enum):
    READ_ONLY = "read_only"                  # screenshot, observe, inspect, read
    LOW_RISK = "low_risk"                    # open app, navigate, search, focus
    STATE_CHANGE = "state_change"            # type, edit, settings, scroll/click that edits
    EXTERNAL_SIDE_EFFECT = "external_side_effect"  # send/submit/publish/delete/purchase


_READ_ONLY = frozenset({
    "screenshot", "observe", "get_page_state", "get_window_state",
    "find_element", "find_candidates", "read_page", "extract_text",
    "extract_links", "wait_for_page", "read_screen", "list_windows",
    "browser_get_url", "get_time", "get_date", "inspect", "wait",
})
_LOW_RISK = frozenset({
    "open_app", "desktop_open", "close_app", "focus_window", "open_url",
    "browser_navigate", "browser_search", "navigate", "back", "forward",
    "refresh", "search", "scroll", "open_browser",
})
_STATE_CHANGE = frozenset({
    "click", "double_click", "type_text", "clear_text", "press_key",
    "hotkey", "drag", "type", "click_text", "key_press",
})
_EXTERNAL = frozenset({
    "send_message", "send_email", "post_message", "submit_form", "submit_order",
    "make_payment", "transfer_money", "purchase", "publish", "tweet",
    "delete_file", "delete_folder", "delete_data", "send", "submit", "delete",
    "shutdown", "restart", "format_disk",
})


def assess_risk(action: str, params: dict | None = None) -> Tuple[ActionRisk, str]:
    """Return (risk, reason). Explicit + visible; params inspected for side effects."""
    a = (action or "").lower()
    if a in _READ_ONLY:
        return ActionRisk.READ_ONLY, "read-only observation"
    if a in _EXTERNAL:
        return ActionRisk.EXTERNAL_SIDE_EFFECT, f"'{action}' has external/destructive effect"
    # Heuristic: composing a message/send/submit/publish/delete/purchase intent
    # via generic primitives (type_text/click) is still EXTERNAL when the
    # caller declares that intent in params.
    p = str(params or {}).lower()
    for kw in ("send", "submit", "publish", "post ", "purchase", "payment",
               "delete", "rm -rf", "tweet"):
        if kw in p and ("message" in p or "form" in p or "order" in p or
                        "post" in p or "payment" in p or "delete" in p or
                        "send" in p or "submit" in p or "tweet" in p):
            return ActionRisk.EXTERNAL_SIDE_EFFECT, f"params declare external effect ({kw.strip()})"
    if a in _LOW_RISK:
        return ActionRisk.LOW_RISK, "reversible local navigation/launch"
    if a in _STATE_CHANGE:
        return ActionRisk.STATE_CHANGE, "local state change"
    # Unknown actions default to STATE_CHANGE (safe side: never silent external).
    logger.debug("[ACTION-POLICY] unknown action '%s' defaulted to STATE_CHANGE", action)
    return ActionRisk.STATE_CHANGE, "unknown action — treated as state change"


def confirmation_required(action: str, params: dict | None = None) -> Tuple[bool, str]:
    """True only for EXTERNAL_SIDE_EFFECT. Decision is explicit + logged."""
    risk, reason = assess_risk(action, params)
    if risk == ActionRisk.EXTERNAL_SIDE_EFFECT:
        logger.info("[ACTION-POLICY] action=%s risk=%s confirmation=REQUIRED (%s)",
                    action, risk.value, reason)
        return True, f"Confirmation required: {reason}"
    return False, ""
