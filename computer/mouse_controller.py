"""
Phase 22: mouse primitives.

Thin, honest wrapper over agent.executor's native input — no new input
backend is introduced. Every call returns a structured ActionResult; `ok`
and the executor message are carried in evidence, never swallowed.

Logging: [MOUSE]
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

from computer.action_result import ActionOutcome, ActionResult

logger = logging.getLogger(__name__)


def _executor():
    """Duck-load the existing SystemExecutor singleton."""
    try:
        import agent.executor as _m
        for name in ("system_executor", "executor"):
            obj = getattr(_m, name, None)
            if obj is not None and hasattr(obj, "mouse_click"):
                return obj
    except Exception as e:
        logger.debug("[MOUSE] executor unavailable: %s", e)
    return None


def available() -> bool:
    return _executor() is not None


def _result(action: str, target: str, t0: float, ok: bool,
            msg: str, method: str = "native_input") -> ActionResult:
    return ActionResult(
        action=action, target=target, method=method, success=bool(ok),
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


def move(x: int, y: int, *, target: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="move", target=target, method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    return _result("move", target, t0, *_unpack(ex.mouse_move(int(x), int(y))))


def click(x: Optional[int] = None, y: Optional[int] = None, *,
          target: str = "", label: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="click", target=target, method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    raw = ex.mouse_click(x, y)
    res = _result("click", target or label, t0, *_unpack(raw))
    if x is not None and y is not None:
        res.evidence["point"] = [int(x), int(y)]
    if label:
        res.evidence["label"] = label
    return res


def double_click(x: Optional[int] = None, y: Optional[int] = None, *,
                 target: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="double_click", target=target,
                            method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    return _result("double_click", target, t0,
                   *_unpack(ex.mouse_double_click(x, y)))


def drag(start: Tuple[int, int], end: Tuple[int, int], *,
         target: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="drag", target=target, method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    return _result("drag", target, t0,
                   *_unpack(ex.mouse_drag(int(start[0]), int(start[1]),
                                          int(end[0]), int(end[1]))))


def scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None, *,
           target: str = "") -> ActionResult:
    ex, t0 = _executor(), time.time()
    if ex is None:
        return ActionResult(action="scroll", target=target,
                            method="native_input",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="executor unavailable")
    return _result("scroll", target, t0,
                   *_unpack(ex.scroll(int(clicks), x, y)))
