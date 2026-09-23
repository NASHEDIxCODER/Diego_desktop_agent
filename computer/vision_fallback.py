"""
Phase 22: perception fallback hierarchy (lowest-confidence tiers).

This module owns the LAST TWO ressort of the hierarchy described in
computer/perception.py:

    5. VISUAL  — visual element detection (layout analysis / frame differencing)
    6. COORDINATE — raw coordinates, used ONLY when explicitly justified

It deliberately does NOT reimplement OCR or accessibility: those live in the
existing vision/ and services/ subsystems and are called through
computer/element_finder.py. Nothing here executes an action; it only answers
"where is this thing, and how confident am I?".

Hard rule (Phase 22 acceptance): the LLM must never blindly guess coordinates.
`coordinate_fallback()` therefore refuses unless the caller passes an explicit
justification, and the returned match is always tagged
`PerceptionMethod.COORDINATE` so verification and the trace can show it.

Logging: [PERCEPTION]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from computer.perception import PerceptionMethod

logger = logging.getLogger(__name__)


@dataclass
class VisualCandidate:
    """One visually-detected region (never a raw screenshot blob)."""

    label: str = ""
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h
    confidence: float = 0.0
    method: str = PerceptionMethod.VISUAL.value
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "bounds": list(self.bounds),
            "center": list(self.center),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "metadata": dict(self.metadata),
        }


# ═══════════════════════════════════════════════════════════════
# Tier 5 — visual element detection
# ═══════════════════════════════════════════════════════════════

def visual_available() -> bool:
    """True when a visual backend (layout analysis / capture) can be used."""
    try:
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def visual_candidates(limit: int = 25) -> List[VisualCandidate]:
    """Detect interactive-looking regions via the existing vision subsystem.

    Uses vision/layout_analyzer (already part of Diego) rather than adding a
    new detector. Returns [] when unavailable — callers then degrade honestly
    instead of inventing coordinates.
    """
    t0 = time.time()
    out: List[VisualCandidate] = []
    try:
        from vision.layout_analyzer import layout_analyzer  # type: ignore
    except Exception as e:
        logger.info("[PERCEPTION] tier=visual unavailable (%s)", e)
        return out

    frame = None
    try:
        cap = getattr(layout_analyzer, "analyze", None)
        if callable(cap):
            result = cap()
            elements = getattr(result, "elements", None) or []
            for el in elements[:limit]:
                bounds = tuple(getattr(el, "bounds", (0, 0, 0, 0)) or (0, 0, 0, 0))
                out.append(VisualCandidate(
                    label=str(getattr(el, "label", "") or ""),
                    bounds=bounds,  # type: ignore[arg-type]
                    confidence=float(getattr(el, "confidence", 0.4) or 0.4),
                    metadata={"source": "layout_analyzer"},
                ))
    except Exception as e:
        logger.debug("[PERCEPTION] tier=visual failed: %s", e)

    logger.info("[PERCEPTION] tier=visual candidates=%d (%.0fms)",
                len(out), (time.time() - t0) * 1000)
    return out


# ═══════════════════════════════════════════════════════════════
# Tier 6 — coordinate fallback (justified only)
# ═══════════════════════════════════════════════════════════════

class CoordinateFallbackRefused(Exception):
    """Raised when a coordinate action is requested without justification."""


def coordinate_fallback(x: int, y: int, *, justification: str = "",
                        label: str = "") -> VisualCandidate:
    """Return an explicit-coordinate target — ONLY with a justification.

    Raises CoordinateFallbackRefused otherwise. This is the enforcement point
    for "Do NOT make the LLM blindly guess coordinates": a coordinate may only
    be used when every higher tier failed AND the reason is recorded.
    """
    if not justification or not justification.strip():
        logger.warning("[PERCEPTION] tier=coordinate REFUSED (no justification)")
        raise CoordinateFallbackRefused(
            "Coordinate fallback requires an explicit justification — "
            "use a higher-confidence perception tier instead."
        )
    logger.warning("[PERCEPTION] tier=coordinate used (%s) at (%d,%d) label='%s'",
                   justification.strip()[:120], x, y, label)
    return VisualCandidate(
        label=label or f"coordinate:{x},{y}",
        bounds=(int(x), int(y), 1, 1),
        confidence=0.3,
        method=PerceptionMethod.COORDINATE.value,
        metadata={"justification": justification.strip()[:200]},
    )


def coordinate_allowed(justification: str = "") -> bool:
    """Cheap guard callers can use before attempting a coordinate action."""
    return bool(justification and justification.strip())
