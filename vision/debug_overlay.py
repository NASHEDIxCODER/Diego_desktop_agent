"""
DebugOverlay — Live visual overlay for vision pipeline debugging.

Toggleable semi-transparent overlay that shows:
- Green boxes around detected UI elements
- Button/label names
- OCR text at each box
- Window name
- Mouse position
- Planner target
- Chosen click point
- Confidence per element
- OCR boxes that didn't become UI elements (red)

Usage:
    from vision.debug_overlay import debug_overlay
    debug_overlay.toggle()        # on/off
    debug_overlay.update(ctx)     # feed latest VisionContext
    debug_overlay.close()         # destroy

Commands:
    python -m vision.debug_overlay   # standalone test
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HAS_CV2 = False
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]


@dataclass
class OverlayElement:
    """A single element rendered on the overlay."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    text: str = ""
    element_type: str = ""
    confidence: float = 0.0
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR
    is_target: bool = False           # planner chose this element
    click_point: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebugOverlay:
    """
    Toggleable semi-transparent debug overlay for real-time vision pipeline inspection.

    Shows a live OpenCV window with all vision pipeline artifacts overlaid:
      - Green boxes: detected UI buttons/interactables
      - Blue boxes: text elements
      - Yellow boxes: input fields
      - Red boxes: OCR boxes that were discarded
      - White boxes: layout region boundaries
      - Magenta circle: planner's chosen click target
      - Cyan crosshair: current mouse position
      - Window name in top-left corner
      - OCR confidence per box
      - Click success/failure annotation
    """

    WINDOW_NAME = "Leo Vision Debug Overlay"
    OVERLAY_ALPHA = 0.4  # transparency of overlay

    # Element type → BGR color
    COLOR_MAP = {
        "button": (0, 255, 0),       # green
        "menu_item": (0, 215, 0),    # light green
        "tab": (255, 200, 0),        # amber
        "link": (255, 128, 0),       # orange
        "input": (255, 255, 0),      # yellow
        "textbox": (255, 255, 0),    # yellow
        "text": (255, 100, 0),       # blue-orange
        "label": (255, 150, 50),     # lighter orange
        "check": (0, 255, 255),      # cyan
        "dialog": (255, 0, 255),     # magenta
        "notification": (0, 128, 255),  # orange-blue
        "toolbar": (128, 128, 128),  # gray
        "sidebar": (64, 64, 64),     # dark gray
        "editor": (80, 80, 80),      # dark gray
        "status_bar": (128, 128, 0), # olive
        "window": (200, 200, 200),   # light gray
        "discarded_ocr": (0, 0, 255),  # red
        "planner_target": (255, 0, 255),  # magenta
        "mouse": (255, 255, 0),      # cyan
    }

    def __init__(self):
        self._enabled: bool = False
        self._lock = threading.Lock()
        self._elements: List[OverlayElement] = []
        self._layout_regions: List[OverlayElement] = []
        self._ocr_discarded: List[OverlayElement] = []
        self._planner_target: Optional[OverlayElement] = None
        self._click_point: Optional[Tuple[int, int]] = None
        self._mouse_pos: Optional[Tuple[int, int]] = None
        self._window_title: str = ""
        self._window_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._capture_resolution: str = ""
        self._frame_hash: str = ""
        self._pipeline_ms: float = 0.0
        self._status_text: List[str] = []
        self._closed: bool = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._frame_count: int = 0
        self._last_update: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────

    def toggle(self) -> bool:
        """Toggle overlay on/off. Returns new state."""
        self._enabled = not self._enabled
        if self._enabled:
            logger.info("[OVERLAY] Debug overlay ENABLED")
            if _HAS_CV2:
                cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.WINDOW_NAME, 960, 540)
        else:
            logger.info("[OVERLAY] Debug overlay DISABLED")
            if _HAS_CV2:
                try:
                    cv2.destroyWindow(self.WINDOW_NAME)
                except Exception:
                    pass
        return self._enabled

    def close(self) -> None:
        """Permanently destroy the overlay window."""
        self._enabled = False
        self._closed = True
        if _HAS_CV2:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except Exception:
                pass
        logger.info("[OVERLAY] Debug overlay closed")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Update from VisionContext ──────────────────────────

    def update_from_context(self, ctx: Any, planner_target: str = "",
                            click_point: Optional[Tuple[int, int]] = None,
                            mouse_pos: Optional[Tuple[int, int]] = None,
                            status: str = "") -> None:
        """
        Feed the latest VisionContext to the overlay.

        Args:
            ctx: VisionContext from vision_service.analyze()
            planner_target: The label the planner is looking for (e.g. "Run")
            click_point: (x, y) where the planner clicked
            mouse_pos: (x, y) current mouse position
            status: Status string (e.g. "Clicked Run at (450, 120)")
        """
        if not self._enabled:
            return

        with self._lock:
            self._frame_count += 1
            self._frame_hash = ctx.frame_hash[:8] if ctx.frame_hash else ""
            self._pipeline_ms = ctx.total_time_ms
            self._capture_resolution = f"{ctx.capture.width}x{ctx.capture.height}" if ctx.capture else "?"
            self._window_title = ctx.active_window_title
            self._planner_target_text = planner_target
            self._click_point = click_point
            self._mouse_pos = mouse_pos

            if status:
                self._status_text = [status]
                if len(self._status_text) > 10:
                    self._status_text = self._status_text[-10:]

            # Build element list from OCR boxes + UI tree
            self._elements.clear()
            self._layout_regions.clear()
            self._ocr_discarded.clear()

            # UI tree elements from desktop
            if ctx.desktop:
                for window in ctx.desktop.windows:
                    for child in window.walk():
                        if child.bounding_box and child.label:
                            oe = OverlayElement(
                                bbox=child.bounding_box,
                                text=child.label,
                                element_type=child.element_type.value,
                                confidence=child.confidence,
                                color=self.COLOR_MAP.get(
                                    child.element_type.value, (0, 255, 0)),
                                is_target=(planner_target.lower() in child.label.lower()
                                           if planner_target and child.label else False),
                            )
                            self._elements.append(oe)

            # Raw OCR boxes that didn't make it into the UI tree
            if ctx.desktop and ctx.raw_ocr_boxes:
                tree_labels = {e.text.lower() for e in self._elements}
                for box in ctx.raw_ocr_boxes:
                    if box.text.strip().lower() not in tree_labels:
                        self._ocr_discarded.append(OverlayElement(
                            bbox=box.bbox,
                            text=box.text.strip(),
                            element_type="discarded_ocr",
                            confidence=box.confidence,
                            color=self.COLOR_MAP["discarded_ocr"],
                        ))

            # Layout regions
            if ctx.layout:
                for region in ctx.layout.regions:
                    self._layout_regions.append(OverlayElement(
                        bbox=region.bounds,
                        text=region.label,
                        element_type=region.region_type.value,
                        confidence=region.confidence,
                        color=self.COLOR_MAP.get(
                            region.region_type.value, (128, 128, 128)),
                    ))

            # Planner target
            self._planner_target = None
            if planner_target:
                for el in self._elements:
                    if el.is_target:
                        self._planner_target = OverlayElement(
                            bbox=el.bbox,
                            text=f"TARGET: {el.text}",
                            element_type="planner_target",
                            confidence=el.confidence,
                            color=self.COLOR_MAP["planner_target"],
                            click_point=el.bbox and (
                                el.bbox[0] + el.bbox[2] // 2,
                                el.bbox[1] + el.bbox[3] // 2,
                            ),
                        )
                        break

        self._last_update = time.monotonic()

        # Render if CV2 available
        if _HAS_CV2 and self._enabled:
            self._render()

    def add_status(self, text: str) -> None:
        """Add a status line (shown in overlay)."""
        with self._lock:
            self._status_text.append(text)
            if len(self._status_text) > 10:
                self._status_text = self._status_text[-10:]

    # ── Rendering ──────────────────────────────────────────

    def _render(self) -> None:
        """Render the overlay window."""
        if not _HAS_CV2 or self._closed:
            return

        with self._lock:
            canvas_w = 1920
            canvas_h = 1080
            if self._window_bounds[2] > 0:
                canvas_w, canvas_h = self._window_bounds[2], self._window_bounds[3]

            # Create dark canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # Draw layout regions (faint)
            for region in self._layout_regions:
                x, y, w, h = region.bbox
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                if x + w > canvas_w or y + h > canvas_h:
                    continue
                cv2.rectangle(canvas, (x, y), (x + w, y + h), region.color, 1)
                cv2.putText(canvas, region.text[:20], (x + 4, y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, region.color, 1)

            # Draw UI elements
            for el in self._elements:
                x, y, w, h = el.bbox
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                if x + w > canvas_w or y + h > canvas_h:
                    continue

                thickness = 2 if el.is_target else 1
                cv2.rectangle(canvas, (x, y), (x + w, y + h), el.color, thickness)

                # Label
                lbl = el.text[:25]
                if el.confidence > 0:
                    lbl += f" [{el.confidence:.0%}]"
                cv2.putText(canvas, lbl, (x + 2, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, el.color, 1)

            # Draw discarded OCR boxes (red, faint)
            for box in self._ocr_discarded:
                x, y, w, h = box.bbox
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                if x + w > canvas_w or y + h > canvas_h:
                    continue
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 180), 1)
                cv2.putText(canvas, f"DISCARD: {box.text[:20]}",
                            (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 0, 180), 1)

            # Draw planner target
            if self._planner_target:
                x, y, w, h = self._planner_target.bbox
                cv2.rectangle(canvas, (x - 3, y - 3), (x + w + 3, y + h + 3),
                              self.COLOR_MAP["planner_target"], 3)
                cv2.putText(canvas, self._planner_target.text[:40],
                            (x - 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, self.COLOR_MAP["planner_target"], 2)
                # Crosshair at click point
                if self._planner_target.click_point:
                    cx, cy = self._planner_target.click_point
                    cv2.drawMarker(canvas, (cx, cy),
                                   self.COLOR_MAP["planner_target"],
                                   cv2.MARKER_CROSS, 20, 2)

            # Draw click point
            if self._click_point:
                cx, cy = self._click_point
                if 0 <= cx < canvas_w and 0 <= cy < canvas_h:
                    cv2.circle(canvas, (cx, cy), 8, (0, 0, 255), 2)
                    cv2.drawMarker(canvas, (cx, cy), (0, 0, 255),
                                   cv2.MARKER_DIAMOND, 15, 1)
                    cv2.putText(canvas, f"CLICK ({cx},{cy})",
                                (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 0, 255), 1)

            # Draw mouse position
            if self._mouse_pos:
                mx, my = self._mouse_pos
                if 0 <= mx < canvas_w and 0 <= my < canvas_h:
                    cv2.drawMarker(canvas, (mx, my), (0, 255, 255),
                                   cv2.MARKER_CROSS, 10, 1)
                    cv2.putText(canvas, f"({mx},{my})",
                                (mx + 10, my), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (0, 255, 255), 1)

            # ── Info panel (top-left) ──────────────────────
            y_offset = 18
            cv2.putText(canvas, f"Leo Vision Debug | Frame #{self._frame_count}",
                        (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(canvas, f"Window: {self._window_title[:80]}",
                        (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1)
            y_offset += 16
            cv2.putText(canvas,
                        f"Resolution: {self._capture_resolution} | "
                        f"Hash: {self._frame_hash} | "
                        f"Pipeline: {self._pipeline_ms:.1f}ms",
                        (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1)
            y_offset += 16
            cv2.putText(canvas,
                        f"UI Elements: {len(self._elements)} | "
                        f"OCR Discarded: {len(self._ocr_discarded)} | "
                        f"Regions: {len(self._layout_regions)}",
                        (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1)

            # Planner target summary
            y_offset += 20
            if self._planner_target:
                cv2.putText(canvas,
                            f"PLANNER TARGET: {self._planner_target_text}",
                            (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, self.COLOR_MAP["planner_target"], 2)
            else:
                cv2.putText(canvas,
                            f"PLANNER TARGET: {self._planner_target_text or 'none'}",
                            (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 100, 100), 1)

            # Status lines
            y_offset += 22
            for st in self._status_text[-6:]:
                cv2.putText(canvas, st[:120], (8, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                y_offset += 14

            # ── Legend (bottom-left) ───────────────────────
            legend_y = canvas_h - 100
            legend_items = [
                ("Green", (0, 255, 0), "Buttons/Clickable"),
                ("Blue", (255, 100, 0), "Text"),
                ("Yellow", (255, 255, 0), "Inputs"),
                ("Red", (0, 0, 255), "Discarded OCR"),
                ("Magenta", (255, 0, 255), "Planner Target"),
                ("Cyan", (255, 255, 0), "Mouse"),
            ]
            for i, (name, color, desc) in enumerate(legend_items):
                ly = legend_y + i * 14
                cv2.rectangle(canvas, (8, ly - 8), (28, ly), color, -1)
                cv2.putText(canvas, f"{name}: {desc}", (34, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Show the window
            try:
                cv2.imshow(self.WINDOW_NAME, canvas)
                cv2.waitKey(1)  # non-blocking refresh
            except Exception as e:
                logger.debug("[OVERLAY] Render error: %s", e)


# Global singleton
debug_overlay = DebugOverlay()


# ── Standalone test ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Leo Vision Debug Overlay — standalone test")
    parser.add_argument("--toggle", action="store_true",
                        help="Toggle overlay on/off")
    args = parser.parse_args()

    if args.toggle:
        state = debug_overlay.toggle()
        print(f"Debug overlay: {'ON' if state else 'OFF'}")
        if state:
            print("Press Ctrl+C to exit")
            try:
                while True:
                    time.sleep(0.1)
                    if cv2 and cv2.getWindowProperty(
                            DebugOverlay.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
            except KeyboardInterrupt:
                pass
            finally:
                debug_overlay.close()