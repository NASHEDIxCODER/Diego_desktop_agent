"""
Phase 22: window management (native tier 3 of the perception hierarchy).

Generic X11 window operations via xdotool (already a Diego dependency —
verified available on this host). NEVER site- or app-specific.

Logging: [PERCEPTION]
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from computer.action_result import ActionOutcome, ActionResult

logger = logging.getLogger(__name__)

_TIMEOUT = 2.0


@dataclass
class WindowInfo:
    window_id: str = ""
    title: str = ""
    window_class: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.window_id, "title": self.title,
                "class": self.window_class}


def has_xdotool() -> bool:
    return shutil.which("xdotool") is not None


def _run(args: List[str]) -> str:
    try:
        r = subprocess.run(["xdotool", *args], capture_output=True, text=True,
                           timeout=_TIMEOUT)
        return r.stdout.strip()
    except Exception as e:
        logger.debug("[PERCEPTION] xdotool %s failed: %s", args[0], e)
        return ""


def list_windows(limit: int = 30) -> List[WindowInfo]:
    """All visible windows (bounded — never dumps an unbounded list)."""
    if not has_xdotool():
        return []
    ids = _run(["search", "--onlyvisible", "--name", ""]).split("\n")
    out: List[WindowInfo] = []
    for wid in [w for w in ids if w.strip()][:limit]:
        title = _run(["getwindowname", wid])
        wclass = _run(["getwindowclassname", wid])
        out.append(WindowInfo(window_id=wid, title=title, window_class=wclass))
    return out


def active_window() -> Optional[WindowInfo]:
    if not has_xdotool():
        return None
    wid = _run(["getactivewindow"])
    if not wid:
        return None
    return WindowInfo(
        window_id=wid,
        title=_run(["getwindowname", wid]),
        window_class=_run(["getwindowclassname", wid]),
    )


def search(query: str) -> List[WindowInfo]:
    """Windows whose title or class contains `query` (case-insensitive)."""
    q = str(query or "").strip().lower()
    if not q:
        return []
    hits = [w for w in list_windows()
            if q in w.title.lower() or q in w.window_class.lower()]
    return hits


def focus_window(query: str, *, action: str = "focus_window",
                 target: str = "") -> ActionResult:
    """Activate the first matching window and verify focus actually moved."""
    t0 = time.time()
    if not has_xdotool():
        return ActionResult(action=action, target=target,
                            method="native_window",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="xdotool unavailable", latency_ms=_ms(t0))
    matches = search(target or query)
    if not matches:
        return ActionResult(
            action=action, target=target or query, method="native_window",
            outcome=ActionOutcome.AMBIGUOUS,
            error=f"no window matching '{(target or query)[:60]}'",
            latency_ms=_ms(t0))
    win = matches[0]
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", win.window_id],
                       capture_output=True, timeout=_TIMEOUT)
    except Exception as e:
        return ActionResult(action=action, target=win.title[:80],
                            method="native_window",
                            outcome=ActionOutcome.FAILED, error=str(e),
                            latency_ms=_ms(t0))
    after = active_window()
    focused = bool(after and after.window_id == win.window_id)
    return ActionResult(
        action=action, target=win.title[:80], method="native_window",
        success=focused,
        outcome=(ActionOutcome.SUCCESS if focused else ActionOutcome.FAILED),
        evidence={"window_id": win.window_id, "class": win.window_class,
                  "focused_after": (after.title if after else "")},
        verification=("focused" if focused else "focus did not move"),
        latency_ms=_ms(t0))


def window_state() -> Dict[str, Any]:
    """Native window state of the focused window (bounded fields only)."""
    win = active_window()
    if win is None:
        return {"focused": False}
    d = win.to_dict()
    d["focused"] = True
    return d


def _ms(t0: float) -> float:
    return round((time.time() - t0) * 1000, 1)
