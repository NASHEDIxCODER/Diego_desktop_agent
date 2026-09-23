"""
Phase 24: structured DesktopContext — the desktop half of ComputerState.

The agent reasons about an OBSERVED desktop application state, never about a
screenshot. This module turns the existing desktop perception tiers into one
normalized, JSON-able record:

    active_application, focused_window, window_title, window_class, process,
    visible_text, interactive_elements, focused_element, screen_hash,
    application_state, observation_method, timestamp

Design rules:
  * Tier 1 (accessibility) and tier 2 (native window state) share with the
    existing computer/ stack; OCR/vision are the labelled fallback. The
    ``observation_method`` field records WHICH tier answered.
  * Nothing is invented: an unavailable desktop yields an EMPTY context with
    ``application_state == "unavailable"`` and ``observation_method == ""``.
  * No application knowledge: only generic roles/labels/text are recorded.

Logging: [PERCEPTION]
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InteractiveElement:
    """One generic interactive element observed in an application."""

    label: str = ""
    kind: str = "element"          # button | input | textbox | menu_item | ...
    role: str = ""
    value: str = ""
    enabled: bool = True
    clickable: bool = False
    text_input: bool = False
    confidence: float = 0.9
    method: str = "accessibility"

    @property
    def is_input(self) -> bool:
        return self.kind in ("input", "textarea", "textbox", "text") \
            or self.text_input

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "kind": self.kind, "role": self.role,
            "value": self.value[:400], "enabled": self.enabled,
            "clickable": self.clickable, "text_input": self.text_input,
            "confidence": self.confidence, "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractiveElement":
        return cls(
            label=str(data.get("label") or ""),
            kind=str(data.get("kind") or "element"),
            role=str(data.get("role") or ""),
            value=str(data.get("value") or ""),
            enabled=bool(data.get("enabled", True)),
            clickable=bool(data.get("clickable", False)),
            text_input=bool(data.get("text_input", False)),
            confidence=float(data.get("confidence", 0.9) or 0.9),
            method=str(data.get("method") or "accessibility"),
        )


@dataclass
class DesktopContext:
    """Structured, JSON-able observed desktop state (never invented)."""

    active_application: str = ""       # process/executable name, e.g. "telegram"
    focused_window: str = ""           # window id (native tier)
    window_title: str = ""
    window_class: str = ""
    process: str = ""
    visible_text: str = ""             # bounded excerpt (never a full dump)
    interactive_elements: List[InteractiveElement] = field(default_factory=list)
    focused_element: Optional[InteractiveElement] = None
    screen_hash: str = ""
    application_state: str = "unavailable"   # unavailable | present | empty
    observation_method: str = ""       # accessibility | native_window | ocr | ""
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        parts = []
        if self.active_application:
            parts.append(f"app={self.active_application}")
        if self.window_title:
            parts.append(f"window='{self.window_title[:80]}'")
        parts.append(f"elements={len(self.interactive_elements)}")
        parts.append(f"method={self.observation_method or 'none'}")
        if self.screen_hash:
            parts.append(f"hash={self.screen_hash[:12]}")
        return " ".join(parts) or "empty desktop state"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_application": self.active_application,
            "focused_window": self.focused_window,
            "window_title": self.window_title,
            "window_class": self.window_class,
            "process": self.process,
            "visible_text": self.visible_text[:4000],
            "interactive_elements": [e.to_dict()
                                     for e in self.interactive_elements[:80]],
            "focused_element": (self.focused_element.to_dict()
                                if self.focused_element else None),
            "screen_hash": self.screen_hash,
            "application_state": self.application_state,
            "observation_method": self.observation_method,
            "timestamp": self.timestamp,
        }

    # ── generic helpers (shared by skill + engine + verification) ──

    def inputs(self) -> List[InteractiveElement]:
        return [e for e in self.interactive_elements if e.is_input]

    def buttons(self) -> List[InteractiveElement]:
        return [e for e in self.interactive_elements
                if e.clickable and not e.text_input]

    def elements_containing(self, text: str) -> List[InteractiveElement]:
        needle = text.lower().strip()
        if not needle:
            return []
        return [e for e in self.interactive_elements
                if needle in (e.label or "").lower()]

    def contains(self, text: str) -> bool:
        needle = text.lower().strip()
        if not needle:
            return False
        blob = self.visible_text.lower()
        if needle in blob:
            return True
        return any(needle in (e.label or "").lower()
                   for e in self.interactive_elements)


def _page_hash(app: str, title: str, text: str) -> str:
    base = f"{app}|{title}|{len(text or '')}"
    return hashlib.md5(base.encode(errors="replace")).hexdigest()[:16]


def _screen_hash() -> str:
    """Live capture hash, else window identity (mirrors computer/ logic)."""
    try:
        import services.screen_capture as sc
        for name in ("screen_capture", "screen_capture_service", "_instance"):
            obj = getattr(sc, name, None)
            if obj is not None and hasattr(obj, "last_capture"):
                lc = obj.last_capture
                lc = lc() if callable(lc) else lc
                h = str(getattr(lc, "hash", "") or
                        getattr(lc, "fast_hash", "") or
                        getattr(lc, "frame_hash", "") or "")
                if h:
                    return h
    except Exception:
        pass
    return ""


class DesktopObserver:
    """Observes the desktop through the existing perception tiers.

    Tier order (never invented state):
      1. accessibility (computer.accessibility)  — richest structured data
      2. native window state (computer.window_manager)
      3. OCR / visible text                               — labelled fallback

    Returns an EMPTY DesktopContext when nothing can be observed.
    """

    def __init__(self, *, max_elements: int = 80, max_text: int = 4000) -> None:
        self._max_elements = max_elements
        self._max_text = max_text
        self._last: Optional[DesktopContext] = None

    def observe(self, note: str = "") -> DesktopContext:
        ctx = self._observe_accessibility()
        if ctx is None or (not ctx.interactive_elements
                           and not ctx.active_application
                           and not ctx.window_title):
            ctx = self._observe_native()
        if ctx is None or (not ctx.visible_text
                           and not ctx.interactive_elements):
            ctx = self._observe_ocr(ctx)
        if ctx is None:
            ctx = DesktopContext(application_state="unavailable")
        if self._last is not None and not ctx.observation_method:
            ctx.observation_method = self._last.observation_method
        self._last = ctx
        return ctx

    def _observe_accessibility(self) -> Optional[DesktopContext]:
        elements: List[InteractiveElement] = []
        method = ""
        try:
            from computer import accessibility as a11y
            for el in a11y.clickable(limit=self._max_elements):
                data = el.to_dict()
                elements.append(InteractiveElement(
                    label=str(data.get("name") or ""),
                    kind=("input" if data.get("text_input") else "button"),
                    role=str(data.get("role") or ""),
                    enabled=bool(data.get("enabled", True)),
                    clickable=bool(data.get("clickable", False)),
                    text_input=bool(data.get("text_input", False)),
                    confidence=float(data.get("confidence", 0.9) or 0.9),
                    method=str(data.get("method") or "accessibility"),
                ))
            method = "accessibility"
        except Exception as e:
            logger.debug("[PERCEPTION] desktop a11y observe failed: %s", e)
        win = self._native_window()
        if not elements and not (win or {}).get("title"):
            return None
        return DesktopContext(
            active_application=str((win or {}).get("window_class") or ""),
            focused_window=str((win or {}).get("id") or ""),
            window_title=str((win or {}).get("title") or ""),
            window_class=str((win or {}).get("class") or ""),
            interactive_elements=elements,
            observation_method=method,
            screen_hash=_screen_hash(),
            application_state="present",
        )

    def _observe_native(self) -> Optional[DesktopContext]:
        try:
            from computer import window_manager as wm
            win = wm.active_window()
            if win is None:
                return None
            return DesktopContext(
                active_application=win.window_class,
                focused_window=win.window_id,
                window_title=win.title,
                window_class=win.window_class,
                observation_method="native_window",
                screen_hash=_screen_hash(),
                application_state="present",
            )
        except Exception as e:
            logger.debug("[PERCEPTION] desktop native observe failed: %s", e)
            return None

    def _observe_ocr(self, ctx: Optional[DesktopContext]) -> Optional[DesktopContext]:
        text = self._ocr_text()
        if not text:
            return ctx
        c = ctx or DesktopContext(application_state="present",
                                  observation_method="ocr")
        c.visible_text = text[:self._max_text]
        if not c.observation_method:
            c.observation_method = "ocr"
        c.screen_hash = _screen_hash() or c.screen_hash
        return c

    @staticmethod
    def _ocr_text() -> str:
        """Best-effort OCR of the visible screen (never raises)."""
        try:
            from services.vision_service import vision_service
            import asyncio
            method = getattr(vision_service, "ocr_only", None)
            if not callable(method):
                return ""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(method(), loop)
                    return str(fut.result(timeout=8) or "")
                return str(loop.run_until_complete(method()) or "")
            except RuntimeError:
                return str(asyncio.run(method()) or "")
        except Exception as e:
            logger.debug("[PERCEPTION] desktop ocr unavailable: %s", e)
            return ""

    @staticmethod
    def _native_window() -> Dict[str, Any]:
        try:
            from computer import window_manager as wm
            win = wm.active_window()
            return win.to_dict() if win else {}
        except Exception:
            return {}


def observer() -> DesktopObserver:
    """A fresh observer bound to the live desktop perception hierarchy."""
    return DesktopObserver()


__all__ = [
    "DesktopContext", "DesktopObserver", "InteractiveElement", "observer",
]
