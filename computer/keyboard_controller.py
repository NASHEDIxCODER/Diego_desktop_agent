"""
Phase 22: keyboard primitives.

Thin, honest wrapper over agent.executor's native keyboard input. Includes
`clear_text` (select-all + delete) and `type_into` (click a located element,
then type) which is the safe way to fill inputs — never blind typing.

Logging: [KEYBOARD]
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

from computer.action_result import ActionOutcome, ActionResult

logger = logging.getLogger(__name__)


def _executor():
    try:
        import agent.executor as _m
        for name in ("system_executor", "executor"):
            obj = getattr(_m, name, None)
            if obj is not None and hasattr(obj, "keyboard_type"):
                return obj
    except Exception as e:
        logger.debug("[KEYBOARD] executor unavailable: %s", e)
    return None


def available() -> bool:
    return _executor() is not None


def _result(action: str, target: str, t0: float, ok: bool,
            msg: str) -> ActionResult:
    return ActionResult(
        action=action, target=target, method="native_input",
        success=bool(ok),
        outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
        evidence={"executor_msg": str(msg)[:200]},
        error="" if ok else str(msg)[:200],
        latency_ms=round((time.time() - t0) * 1000, 1))


def _unpack(raw: Any) -> Tuple[bool, str]:
    if isinstance(raw, tuple):
        ok = bool(raw[0]) if raw else False
        msg = str(raw[1]) if len(raw) > 1 else ""
        return ok, msg
    return bool(raw), ""


def type_text(text: str, *, target: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="type_text", target=target,
                            method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    return _result("type_text", target, t0, *_unpack(ex.keyboard_type(text)))


def press_key(key: str, *, target: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="press_key", target=target,
                            method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    return _result("press_key", target, t0, *_unpack(ex.keyboard_press(key)))


def hotkey(*keys: str, target: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="hotkey", target=target,
                            method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    return _result("hotkey", target or "+".join(keys), t0,
                   *_unpack(ex.keyboard_hotkey(*keys)))


def clear_text(*, target: str = "") -> ActionResult:
    """Select all, then delete (generic; no site/app assumptions)."""
    t0 = time.time()
    sel = hotkey("ctrl", "a", target=target)
    if not sel.success:
        sel.action = "clear_text"
        return sel
    dele = press_key("Delete", target=target)
    ok = dele.success
    return ActionResult(
        action="clear_text", target=target, method="native_input", success=ok,
        outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
        evidence={"select_all": sel.evidence, "delete": dele.evidence},
        error=dele.error,
        latency_ms=round((time.time() - t0) * 1000, 1))


def type_into(point: Tuple[int, int], text: str, *,
              target: str = "", label: str = "") -> ActionResult:
    """Click into a located input, then type — the safe input-fill path."""
    from computer import mouse_controller as mouse

    t0 = time.time()
    c = mouse.click(int(point[0]), int(point[1]), target=target, label=label)
    if not c.success:
        c.action = "type_into"
        return c
    typed = type_text(text, target=target)
    ok = typed.success
    return ActionResult(
        action="type_into", target=target, method="native_input", success=ok,
        outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
        evidence={"clicked": c.evidence, "typed": typed.evidence,
                  "chars": len(text)},
        error=typed.error,
        latency_ms=round((time.time() - t0) * 1000, 1))
