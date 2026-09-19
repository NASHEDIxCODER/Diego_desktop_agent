"""
Phase 22: element location through the full perception hierarchy.

Implements the Phase 22 perception ORDER (computer/perception.py):

    1. browser/DOM            (highest confidence)
    2. accessibility (AT-SPI)
    3. native window state
    4. OCR / text detection
    5. visual element detection
    6. coordinate fallback    (ONLY with explicit justification)

The returned ElementMatch always records WHICH tier answered and with what
evidence, so verification and the agent trace can show the method used.
A coordinate answer is refused unless `justification` is provided
(computer.vision_fallback enforces this).

Nothing here executes an action; it only locates targets.

Logging: [PERCEPTION]
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from computer.perception import PerceptionMethod

logger = logging.getLogger(__name__)


@dataclass
class ElementMatch:
    """One located target, tagged with the perception tier that found it."""

    label: str = ""
    method: str = ""
    confidence: float = 0.0
    point: Optional[Tuple[int, int]] = None
    bounds: Optional[Tuple[int, int, int, int]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.method)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "method": self.method,
            "confidence": round(self.confidence, 3),
            "point": (list(self.point) if self.point else None),
            "bounds": (list(self.bounds) if self.bounds else None),
            "evidence": dict(self.evidence),
        }


# ── Tier 1: browser/DOM ──────────────────────────────────────────

_DOM_SCRIPT = """
() => {
  const q = QUERY;
  const els = [...document.querySelectorAll(
    'a, button, input, [role=button], [role=link], [role=tab], label, summary')];
  const el = els.find(e => {
    const t = (e.innerText || e.value || e.placeholder || '').toLowerCase();
    return t.includes(q);
  });
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const x = window.screenX + r.x + r.width / 2;
  const y = window.screenY + (window.outerHeight - window.innerHeight)
            + r.y + r.height / 2;
  return JSON.stringify({x: Math.round(x), y: Math.round(y),
    label: (el.innerText || el.value || el.placeholder || q)
           .trim().slice(0, 80)});
}
"""


def _find_in_dom(query: str) -> Optional[ElementMatch]:
    try:
        from computer import browser_controller as bctl
        if not bctl.attached():
            return None
        script = _DOM_SCRIPT.replace("QUERY", repr(query.lower()))
        # `bctl.evaluate` answers for the CDP-attached browser AND for a bound
        # page, so the DOM tier is tier 1 in both cases (no direct backend use).
        raw = bctl.evaluate(script)
        if not raw:
            return None
        import json
        if isinstance(raw, dict):
            data = raw
        else:
            data = json.loads(raw)
        return ElementMatch(
            label=str(data.get("label") or query),
            method=PerceptionMethod.BROWSER_DOM.value,
            confidence=0.8,
            point=(int(data.get("x")), int(data.get("y"))),
            evidence={"source": "cdp_dom",
                      "url": bctl.get_page_state().get("url", "")},
        )
    except Exception as e:
        logger.debug("[PERCEPTION] tier=dom failed: %s", e)
        return None


# ── Tier 2: accessibility ────────────────────────────────────────

def _find_in_a11y(query: str) -> Optional[ElementMatch]:
    try:
        from computer import accessibility as a11y
        hits = a11y.find(query)
        for el in hits:
            x, y, w, h = el.bounds
            if el.enabled and w > 0 and h > 0:
                return ElementMatch(
                    label=el.name[:80], method=el.method,
                    confidence=el.confidence, point=el.center,
                    bounds=el.bounds,
                    evidence={"role": el.role, "clickable": el.clickable,
                              "text_input": el.text_input})
        return None
    except Exception as e:
        logger.debug("[PERCEPTION] tier=a11y failed: %s", e)
        return None


# ── Tier 3: native window ────────────────────────────────────────

def _find_in_native(query: str) -> Optional[ElementMatch]:
    try:
        from computer import window_manager as wm
        win = wm.active_window()
        if win is None:
            return None
        q = query.lower()
        if q in win.title.lower() or q in win.window_class.lower():
            return ElementMatch(
                label=win.title[:80],
                method=PerceptionMethod.NATIVE_WINDOW.value,
                confidence=0.7,
                point=None,  # window-level target: no click point
                evidence={"window_id": win.window_id,
                          "class": win.window_class},
            )
        return None
    except Exception as e:
        logger.debug("[PERCEPTION] tier=native failed: %s", e)
        return None


# ── Tier 4: OCR ──────────────────────────────────────────────────

def _find_by_ocr(query: str) -> Optional[ElementMatch]:
    try:
        from services.vision_service import vision_service

        async def _scan():
            ctx = await vision_service.force_analyze()
            return getattr(ctx, "raw_ocr_boxes", None) or []

        boxes = _run_async(_scan())
        q = query.lower()
        for box in boxes:
            if q in str(getattr(box, "text", "")).lower():
                center = getattr(box, "center", None)
                if center:
                    return ElementMatch(
                        label=str(getattr(box, "text", ""))[:80],
                        method=PerceptionMethod.OCR.value,
                        confidence=float(
                            getattr(box, "confidence", 0.5) or 0.5),
                        point=(int(center[0]), int(center[1])),
                        evidence={"source": "ocr"})
        return None
    except Exception as e:
        logger.debug("[PERCEPTION] tier=ocr failed: %s", e)
        return None


def _run_async(coro):
    """Run a coroutine from sync context (mirrors dispatcher's pattern)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result(timeout=8)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Tier 5: visual ───────────────────────────────────────────────

def _find_visually(query: str) -> Optional[ElementMatch]:
    try:
        from computer import vision_fallback as vf
        q = query.lower()
        for cand in vf.visual_candidates():
            if q and q in str(cand.label).lower():
                return ElementMatch(
                    label=cand.label[:80] or query,
                    method=cand.method, confidence=cand.confidence,
                    point=cand.center, bounds=cand.bounds,
                    evidence=dict(cand.metadata))
        return None
    except Exception as e:
        logger.debug("[PERCEPTION] tier=visual failed: %s", e)
        return None


# ── Public entry point ───────────────────────────────────────────

def find_element(query: str, *, allow_coordinate: bool = False,
                 justification: str = "") -> ElementMatch:
    """Locate `query` through the perception hierarchy (in order).

    Returns an ElementMatch with `.found` False (and method "") when NO tier
    could locate the target — callers must treat that as AMBIGUOUS, never
    guess coordinates. A coordinate answer requires an explicit justification.
    """
    t0 = time.time()
    q = str(query or "").strip()
    if not q:
        return ElementMatch()

    match: Optional[ElementMatch] = None
    for tier, fn in (("browser_dom", _find_in_dom),
                     ("accessibility", _find_in_a11y),
                     ("native_window", _find_in_native),
                     ("ocr", _find_by_ocr),
                     ("visual", _find_visually)):
        match = fn(q)
        if match is not None and match.found:
            break
        match = None

    if match is None and allow_coordinate:
        try:
            from computer import vision_fallback as vf
            x, y = int(q.split(",")[0]), int(q.split(",")[1])
            cand = vf.coordinate_fallback(x, y, justification=justification)
            match = ElementMatch(
                label=cand.label, method=cand.method,
                confidence=cand.confidence, point=cand.center,
                evidence=dict(cand.metadata))
        except Exception as e:
            logger.warning("[PERCEPTION] tier=coordinate refused: %s", e)

    if match is not None:
        match.evidence["search_ms"] = round((time.time() - t0) * 1000, 1)
        logger.info("[PERCEPTION] find '%s' -> method=%s conf=%.2f",
                    q[:60], match.method, match.confidence)
    else:
        logger.info("[PERCEPTION] find '%s' -> NOT FOUND (all tiers, %.0fms)",
                    q[:60], (time.time() - t0) * 1000)
    return match or ElementMatch()
