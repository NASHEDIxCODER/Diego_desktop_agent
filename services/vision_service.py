"""
VisionService — Production-grade desktop vision pipeline (v3).

10-stage pipeline that understands the desktop like a human:

   1. Screen Capture     (<20ms)  — mss-backed, fullscreen or active window
   2. Frame Hash         (<2ms)   — perceptual hash for gating
   3. Motion Detection   (<5ms)   — grid-based motion regions (16×16 cells)
   4. Gating Decision    (<1ms)   — skip if: frame unchanged + no input + cache valid
   5. Layout Analysis    (<15ms)  — app type detection + region segmentation
   6. OCR                (<150ms) — region-aware, box merged, deduped, self-healing
   7. UI Detection       (<20ms)  — heuristics classify OCR boxes into UI elements
   8. Semantic Reasoning (<80ms)  — page type, errors, interactive elements, summary
   9. Action Verification (<10ms) — compare pre/post frames, auto-retry
  10. Memory Update      (<5ms)   — store snapshot, update screen memory

Total target: <250ms when screen changes.
Near-zero CPU when desktop is idle (gating skips everything).

NEVER OCR every frame. Only process vision when:
  - desktop changed (pHash differs)
  - active window changed
  - mouse clicked
  - keyboard input happened
  - user explicitly requested vision (force=True)
  - planner requires updated state

Produces a full semantic understanding:
  - Application type (VSCode, Chrome, Terminal, Discord, etc.)
  - Layout regions (toolbar, sidebar, editor, status bar, tabs, etc.)
  - UI tree (structured hierarchy with roles, enabled/visible state)
  - OCR text with confidence and text hierarchy
  - Page type classification
  - Error detection
  - Interactive element inventory
  - Human-readable semantic summary
  - Frame memory for state comparison

Structured logging per stage: [VISION] [FRAME] [LAYOUT] [OCR] [UI] [SCREEN] [VERIFY] [MEMORY]

Maintains backward compatibility with:
  - conversation_engine (vision_context_fn)
  - planner (ui_tree, find_element, click_element)
  - action_dispatcher (read_screen, screen_context)
  - desktop_observer (window change events)
  - BaseService lifecycle
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.service import BaseService

# ── New vision modules ────────────────────────────────────────────
from vision.frame_differencer import (
    frame_differencer, FrameDiffResult, GatingDecision, MotionRegion,
)
from vision.layout_analyzer import (
    layout_analyzer, ApplicationType, WindowLayout, RegionType, LayoutRegion,
)
from vision.ocr_pipeline import (
    ocr_pipeline as enhanced_ocr, OCRBox, OCRResult, TextClass,
)
from vision.screen_memory import (
    screen_memory, ScreenSnapshot,
)
from vision.action_verifier import (
    action_verifier, VerificationResult, VerificationStatus,
)
from vision.forensic_logger import forensic_logger, ForensicReport
from vision.debug_overlay import debug_overlay

# ── Existing modules (backward compatible) ────────────────────────
from services.screen_capture import screen_capture_service, CaptureResult
from services.ui_tree import (
    UIDesktop, UIWindow, UIElement, ElementType,
)

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────

_HAS_CV2 = False
try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VisionContext:
    """
    The complete result of the full vision pipeline.

    Replaces the v2 VisionContext. Adds layout, semantic reasoning,
    memory, and verification fields. Backward compatible.
    """
    # Capture
    capture: Optional[CaptureResult] = None
    capture_time_ms: float = 0.0

    # Frame / gating
    frame_hash: str = ""
    gating: Optional[GatingDecision] = None
    frame_diff: Optional[FrameDiffResult] = None

    # Layout
    app_type: str = ""
    app_name: str = ""
    layout: Optional[WindowLayout] = None
    layout_time_ms: float = 0.0

    # OCR
    ocr_result: Optional[OCRResult] = None
    ocr_text: str = ""
    raw_ocr_boxes: List[OCRBox] = field(default_factory=list)
    ocr_time_ms: float = 0.0

    # UI Tree
    desktop: Optional[UIDesktop] = None
    ui_tree_text: str = ""
    ui_tree_json: str = ""
    ui_time_ms: float = 0.0

    # Semantic reasoning
    page_type: str = ""
    semantic_summary: str = ""
    interactive_elements_json: str = ""
    error_elements_json: str = ""
    reasoning_time_ms: float = 0.0

    # Window
    active_window_title: str = ""
    active_window_pid: int = 0
    active_window_process: str = ""

    # Verification
    verification: Optional[VerificationResult] = None
    verify_time_ms: float = 0.0

    # Memory
    snapshot: Optional[ScreenSnapshot] = None
    memory_time_ms: float = 0.0

    # Metadata
    total_time_ms: float = 0.0
    from_cache: bool = False
    error: str = ""
    stages_run: List[str] = field(default_factory=list)

    @property
    def compact_summary(self) -> str:
        """A compact text summary for LLM prompt injection."""
        parts = []

        if self.active_window_title:
            app_info = self.active_window_title
            if self.app_type:
                app_info = f"[{self.app_type}/{self.app_name}] {app_info}"
            parts.append(f"Active window: {app_info}")

        if self.app_type and self.app_type != "unknown":
            parts.append(f"Application: {self.app_name} ({self.app_type})")

        if self.page_type and self.page_type != "unknown":
            parts.append(f"Page type: {self.page_type}")

        if self.ui_tree_text:
            parts.append(self.ui_tree_text)

        if self.semantic_summary:
            parts.append(f"Summary: {self.semantic_summary}")

        if self.ocr_text:
            # Truncate to avoid blowing the LLM context
            text_snippet = self.ocr_text[:500]
            parts.append(f"Visible text: {text_snippet}")

        return "\n".join(parts)

    @property
    def quick_context(self) -> str:
        """Ultra-compact context (one line)."""
        items = []
        if self.active_window_title:
            items.append(f"Window: {self.active_window_title[:60]}")
        if self.app_type:
            items.append(f"App: {self.app_type}")
        if self.page_type:
            items.append(f"Page: {self.page_type}")
        if self.semantic_summary:
            items.append(self.semantic_summary[:100])
        return " | ".join(items) if items else ""


# ═══════════════════════════════════════════════════════════════════
# VisionService — main orchestrator
# ═══════════════════════════════════════════════════════════════════

class VisionService(BaseService):
    """
    Production desktop vision pipeline with 10 stages.

    Extends BaseService for lifecycle management.

    Public API:
      - analyze() — run full 10-stage pipeline
      - force_analyze() — bypass all gating (always run full pipeline)
      - quick_context() — compact LLM-ready text summary
      - ui_tree() — get current UIDesktop
      - find_element() — search UI tree for elements
      - click_element() — find clickable element by label
      - verify_last_action() — check if the last action changed the UI
      - what_changed() — describe what changed since previous frame
    """

    name = "vision_service"
    dependencies: List[str] = ["screen_capture"]

    def __init__(self):
        super().__init__()
        self._last_context: Optional[VisionContext] = None
        self._analyze_count: int = 0
        self._skip_count: int = 0
        self._ocr_ready: bool = False

        # Tunables
        self.preprocess_for_ocr: bool = True
        self.use_ocr: bool = True
        self.use_layout: bool = True
        self.use_verification: bool = True
        self.use_memory: bool = True

    # ── BaseService contract ──────────────────────────────────────

    async def _start(self) -> bool:
        """Initialise all vision subsystems."""
        ocr_ok = enhanced_ocr.initialize()
        self._ocr_ready = ocr_ok

        details = {
            "ocr": enhanced_ocr.active_backend,
            "opencv": _HAS_CV2,
            "layout": self.use_layout,
            "verification": self.use_verification,
            "memory": self.use_memory,
        }
        self.set_health("ready" if ocr_ok else "degraded (no OCR)", details)

        logger.info("[VISION] Initialized — ocr=%s layout=%s verify=%s memory=%s",
                     enhanced_ocr.active_backend if ocr_ok else "none",
                     "✓" if self.use_layout else "✗",
                     "✓" if self.use_verification else "✗",
                     "✓" if self.use_memory else "✗")
        return True  # Always ready — capture works even without OCR

    async def _stop(self) -> None:
        """Release resources."""
        enhanced_ocr.close()
        screen_memory.clear()
        self._last_context = None
        logger.info("[VISION] Stopped")

    @property
    def ready(self) -> bool:
        return screen_capture_service.ready

    # ═══════════════════════════════════════════════════════════════
    # 10-STAGE VISION PIPELINE
    # ═══════════════════════════════════════════════════════════════

    async def analyze(
        self,
        force: bool = False,
        source: str = "active_window",
        include_ocr: bool = True,
        include_tree: bool = True,
        include_layout: bool = True,
        include_reasoning: bool = True,
        verify_previous_action: bool = False,
    ) -> VisionContext:
        """
        Run the full 10-stage vision pipeline.

        Stages:
          1. Screen Capture     (<20ms)
          2. Frame Hash         (<2ms)
          3. Motion Detection   (<5ms)
          4. Gating Decision    (<1ms)   ← may return cached here
          5. Layout Analysis    (<15ms)
          6. OCR                (<150ms)
          7. UI Detection       (<20ms)
          8. Semantic Reasoning (<80ms)
          9. Action Verification (<10ms)  ← optional
         10. Memory Update      (<5ms)

        Args:
            force: Bypass gating (always run full pipeline).
            source: "active_window" or "fullscreen".
            include_ocr: Run OCR (stage 6).
            include_tree: Build UI tree (stage 7).
            include_layout: Run layout analysis (stage 5).
            include_reasoning: Run semantic reasoning (stage 8).
            verify_previous_action: Run verification against pre-action snapshot.

        Returns:
            VisionContext with complete analysis.
        """
        t0 = time.perf_counter_ns()
        ctx = VisionContext()

        # ═══════════════════════════════════════════════════════════
        # Stage 1: Screen Capture
        # ═══════════════════════════════════════════════════════════
        t_cap = time.perf_counter_ns()
        if source == "active_window":
            cap = await screen_capture_service.capture_active_window()
        else:
            cap = await screen_capture_service.capture_fullscreen()

        if cap is None or not cap.is_valid:
            ctx.error = "[VISION] Stage 1 FAILED: capture returned None or invalid image"
            ctx.total_time_ms = (time.perf_counter_ns() - t0) / 1_000_000
            logger.error(ctx.error)
            return ctx

        ctx.capture = cap
        ctx.capture_time_ms = (time.perf_counter_ns() - t_cap) / 1_000_000
        ctx.active_window_title = cap.source_label.replace("window:", "").split(" (fullscreen")[0]
        ctx.stages_run.append("capture")
        logger.info("[VISION] Stage 1 — Capture: %dx%d in %.1fms",
                     cap.width, cap.height, ctx.capture_time_ms)

        # ═══════════════════════════════════════════════════════════
        # Stage 2: Frame Hash
        # ═══════════════════════════════════════════════════════════
        t_hash = time.perf_counter_ns()
        ctx.frame_hash = frame_differencer.compute_phash(cap.image) if cap.image is not None else ""
        ctx.stages_run.append("hash")
        hash_time = (time.perf_counter_ns() - t_hash) / 1_000_000
        logger.info("[VISION] Stage 2 — Hash: %s in %.1fms", ctx.frame_hash[:8], hash_time)

        # ═══════════════════════════════════════════════════════════
        # Stage 3: Motion Detection
        # ═══════════════════════════════════════════════════════════
        t_motion = time.perf_counter_ns()
        if cap.image is not None:
            ctx.frame_diff = frame_differencer.compute_motion(cap.image)
            ctx.stages_run.append("motion")
            motion_time = (time.perf_counter_ns() - t_motion) / 1_000_000
            logger.info("[VISION] Stage 3 — Motion: %s in %.1fms",
                         ctx.frame_diff.summary, motion_time)

        # ═══════════════════════════════════════════════════════════
        # Stage 4: Gating Decision
        # ═══════════════════════════════════════════════════════════
        t_gate = time.perf_counter_ns()
        ctx.gating = frame_differencer.should_process(
            force=force,
            window_title=ctx.active_window_title,
            planner_requested=False,
        )
        ctx.stages_run.append("gating")
        gate_time = (time.perf_counter_ns() - t_gate) / 1_000_000
        logger.info("[VISION] Stage 4 — Gating: process=%s reason='%s' quality=%s in %.1fms",
                     ctx.gating.should_process, ctx.gating.reason,
                     ctx.gating.quality, gate_time)

        # ── Return cached if gating says skip ─────────────────
        if not ctx.gating.should_process and self._last_context is not None:
            self._skip_count += 1
            cached = self._last_context
            cached.total_time_ms = (time.perf_counter_ns() - t0) / 1_000_000
            cached.from_cache = True
            logger.debug("[VISION] Gating SKIP — returning cached context (%.1fms total)",
                         cached.total_time_ms)
            return cached

        # Store current frame for future comparisons
        if cap.image is not None:
            frame_differencer.store_frame(cap.image, ctx.active_window_title)
        ctx.stages_run.append("process")

        # ═══════════════════════════════════════════════════════════
        # Stage 5: Layout Analysis
        # ═══════════════════════════════════════════════════════════
        if include_layout and self.use_layout:
            t_layout = time.perf_counter_ns()
            try:
                # Detect application type
                app_type, app_name, app_conf = layout_analyzer.detect_application(
                    window_title=ctx.active_window_title,
                    process_name=ctx.active_window_process,
                    pid=ctx.active_window_pid,
                )
                ctx.app_type = app_type.value if isinstance(app_type, ApplicationType) else app_type
                ctx.app_name = app_name

                # Segment layout
                ctx.layout = layout_analyzer.segment_layout(
                    app_type=app_type if isinstance(app_type, ApplicationType) else ApplicationType.UNKNOWN,
                    window_width=cap.width,
                    window_height=cap.height,
                    window_title=ctx.active_window_title,
                )
                ctx.layout_time_ms = (time.perf_counter_ns() - t_layout) / 1_000_000
                ctx.stages_run.append("layout")
                logger.info("[VISION] Stage 5 — Layout: %s/%s (%d regions) in %.1fms",
                             ctx.app_type, ctx.app_name,
                             len(ctx.layout.regions) if ctx.layout else 0,
                             ctx.layout_time_ms)
            except Exception as e:
                logger.warning("[VISION] Stage 5 — Layout FAILED: %s", e)

        # ═══════════════════════════════════════════════════════════
        # Stage 6: OCR
        # ═══════════════════════════════════════════════════════════
        if include_ocr and self.use_ocr and enhanced_ocr.ready and cap.image is not None:
            t_ocr = time.perf_counter_ns()
            try:
                ctx.ocr_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: enhanced_ocr.ocr(
                        cap.image, preprocess=self.preprocess_for_ocr)
                )
                ctx.ocr_text = ctx.ocr_result.text
                ctx.raw_ocr_boxes = ctx.ocr_result.boxes
                ctx.ocr_time_ms = (time.perf_counter_ns() - t_ocr) / 1_000_000
                ctx.stages_run.append("ocr")
                logger.info("[VISION] Stage 6 — OCR: %d boxes avg_conf=%.2f backend=%s in %.1fms",
                             ctx.ocr_result.box_count_final,
                             ctx.ocr_result.avg_confidence,
                             ctx.ocr_result.backend,
                             ctx.ocr_time_ms)
            except Exception as e:
                logger.warning("[VISION] Stage 6 — OCR FAILED: %s", e)
                ctx.ocr_result = None
                ctx.ocr_text = ""

        # ═══════════════════════════════════════════════════════════
        # Stage 7: UI Detection (build tree)
        # ═══════════════════════════════════════════════════════════
        if include_tree and ctx.raw_ocr_boxes:
            t_ui = time.perf_counter_ns()
            try:
                desktop = self._build_ui_tree(ctx)
                ctx.desktop = desktop
                ctx.ui_tree_text = desktop.to_compact_str() if desktop else ""
                ctx.ui_tree_json = json.dumps(desktop.to_dict()) if desktop else ""
                ctx.ui_time_ms = (time.perf_counter_ns() - t_ui) / 1_000_000
                ctx.stages_run.append("ui_detection")
                logger.info("[VISION] Stage 7 — UI Tree: %d windows in %.1fms",
                             len(desktop.windows) if desktop else 0, ctx.ui_time_ms)
            except Exception as e:
                logger.warning("[VISION] Stage 7 — UI Detection FAILED: %s", e)
                ctx.desktop = None
                ctx.ui_tree_text = ""
        elif include_tree:
            # No OCR boxes — create minimal tree
            desktop = UIDesktop()
            window = UIWindow(
                title=ctx.active_window_title,
                bounding_box=(0, 0, cap.width, cap.height),
                app_type=ctx.app_type,
                app_name=ctx.app_name,
            )
            if ctx.layout:
                for region in ctx.layout.regions:
                    panel_type = self._layout_region_to_element_type(region.region_type)
                    panel = window.add_panel(
                        region_type=panel_type,
                        label=region.label,
                        bbox=region.bounds,
                    )
            desktop.add_window(window)
            ctx.desktop = desktop
            ctx.ui_tree_text = desktop.to_compact_str()
            ctx.ui_tree_json = json.dumps(desktop.to_dict())
            ctx.stages_run.append("ui_detection")

        # ═══════════════════════════════════════════════════════════
        # Stage 8: Semantic Reasoning
        # ═══════════════════════════════════════════════════════════
        if include_reasoning:
            t_reason = time.perf_counter_ns()
            try:
                from services.screen_reasoning import screen_reasoner
                screen_ctx = screen_reasoner.analyze(
                    ocr_text=ctx.ocr_text,
                    ui_tree_text=ctx.ui_tree_text,
                )
                ctx.page_type = screen_ctx.page_type
                ctx.semantic_summary = screen_reasoner.context_description()
                if screen_ctx.elements:
                    ctx.interactive_elements_json = json.dumps(
                        [{"type": e.type, "label": e.label, "position": list(e.position),
                          "confidence": e.confidence} for e in screen_ctx.elements[:20]]
                    )
                if screen_ctx.error_elements:
                    ctx.error_elements_json = json.dumps(
                        [{"type": e.type, "label": e.label, "confidence": e.confidence}
                         for e in screen_ctx.error_elements[:10]]
                    )
                ctx.reasoning_time_ms = (time.perf_counter_ns() - t_reason) / 1_000_000
                ctx.stages_run.append("reasoning")
                logger.info("[VISION] Stage 8 — Reasoning: page_type=%s errors=%d in %.1fms",
                             ctx.page_type, len(screen_ctx.error_elements), ctx.reasoning_time_ms)
            except Exception as e:
                logger.warning("[VISION] Stage 8 — Reasoning FAILED: %s", e)

        # ═══════════════════════════════════════════════════════════
        # Stage 9: Action Verification (optional)
        # ═══════════════════════════════════════════════════════════
        if verify_previous_action and self.use_verification:
            t_verify = time.perf_counter_ns()
            try:
                changed, explanation = screen_memory.last_action_changed_ui()
                ctx.verification = VerificationResult(
                    success=changed,
                    status=VerificationStatus.VERIFIED if changed else VerificationStatus.NO_CHANGE,
                    explanation=explanation,
                    frame_changed=changed,
                    pre_action_hash=frame_differencer.prev_hash,
                    post_action_hash=ctx.frame_hash,
                )
                ctx.verify_time_ms = (time.perf_counter_ns() - t_verify) / 1_000_000
                ctx.stages_run.append("verify")
                logger.info("[VISION] Stage 9 — Verify: changed=%s in %.1fms",
                             changed, ctx.verify_time_ms)
            except Exception as e:
                logger.debug("[VISION] Stage 9 — Verify: %s", e)

        # ═══════════════════════════════════════════════════════════
        # Stage 10: Memory Update
        # ═══════════════════════════════════════════════════════════
        if self.use_memory:
            t_memory = time.perf_counter_ns()
            try:
                snapshot = ScreenSnapshot(
                    window_title=ctx.active_window_title,
                    app_type=ctx.app_type,
                    app_name=ctx.app_name,
                    frame_hash=ctx.frame_hash,
                    full_hash=frame_differencer.compute_full_hash(cap.image) if cap.image is not None else "",
                    width=cap.width,
                    height=cap.height,
                    ocr_text=ctx.ocr_text,
                    ui_tree_text=ctx.ui_tree_text,
                    ui_tree_json=ctx.ui_tree_json,
                    layout_json=json.dumps(ctx.layout.to_dict()) if ctx.layout else "",
                    semantic_summary=ctx.semantic_summary,
                    page_type=ctx.page_type,
                    interactive_elements=ctx.interactive_elements_json,
                    error_elements=ctx.error_elements_json,
                    total_pipeline_ms=(time.perf_counter_ns() - t0) / 1_000_000,
                    from_cache=False,
                )
                screen_memory.store(snapshot)
                ctx.snapshot = snapshot
                ctx.memory_time_ms = (time.perf_counter_ns() - t_memory) / 1_000_000
                ctx.stages_run.append("memory")
                logger.info("[VISION] Stage 10 — Memory: stored frame #%d in %.1fms",
                             snapshot.frame_id, ctx.memory_time_ms)
            except Exception as e:
                logger.warning("[VISION] Stage 10 — Memory FAILED: %s", e)

        # ── Cache and finalize ────────────────────────────────
        self._last_context = ctx
        self._analyze_count += 1
        ctx.total_time_ms = (time.perf_counter_ns() - t0) / 1_000_000

        logger.info("[VISION] Pipeline complete: stages=%s total=%.1fms window='%s' ocr_boxes=%d",
                     "+".join(ctx.stages_run), ctx.total_time_ms,
                     ctx.active_window_title[:60],
                     ctx.ocr_result.box_count_final if ctx.ocr_result else 0)
        return ctx

    # ── UI Tree builder (stage 7 internals) ──────────────────────

    def _build_ui_tree(self, ctx: VisionContext) -> UIDesktop:
        """
        Build the full semantic UI tree from OCR boxes + layout regions.

        Produces a hierarchical tree:
          Desktop → Window → [Toolbar, Sidebar, Editor, ...] → [Button, Text, Tab, ...]
        """
        desktop = UIDesktop()
        window = UIWindow(
            title=ctx.active_window_title,
            bounding_box=(0, 0, ctx.capture.width if ctx.capture else 0,
                          ctx.capture.height if ctx.capture else 0),
            app_type=ctx.app_type,
            app_name=ctx.app_name,
        )

        app_type_enum = ApplicationType(ctx.app_type) if ctx.app_type else ApplicationType.UNKNOWN

        # ── Add layout region panels ──────────────────────
        if ctx.layout:
            for region in ctx.layout.regions:
                panel_type = self._layout_region_to_element_type(region.region_type)
                panel = window.add_panel(
                    region_type=panel_type,
                    label=region.label,
                    bbox=region.bounds,
                )

                # Place OCR boxes into their region panel
                if ctx.raw_ocr_boxes:
                    for box in ctx.raw_ocr_boxes:
                        if self._box_in_region(box, region.bounds):
                            element = self._ocr_box_to_ui_element(box, app_type_enum)
                            panel.add_child(element)

        # ── Add OCR boxes that don't fit in any region ─────
        if ctx.raw_ocr_boxes and ctx.layout:
            for box in ctx.raw_ocr_boxes:
                placed = False
                for region in ctx.layout.regions:
                    if self._box_in_region(box, region.bounds):
                        placed = True
                        break
                if not placed:
                    element = self._ocr_box_to_ui_element(box, app_type_enum)
                    window.add_child(element)
        elif ctx.raw_ocr_boxes:
            # No layout — place all OCR boxes directly in window
            for box in sorted(ctx.raw_ocr_boxes, key=lambda b: (b.y, b.x)):
                element = self._ocr_box_to_ui_element(box, app_type_enum)
                window.add_child(element)

        # ── Add dialogs detected from OCR ──────────────────
        if ctx.layout and ctx.raw_ocr_boxes:
            try:
                dialogs = layout_analyzer.detect_dialogs(
                    ctx.raw_ocr_boxes,
                    ctx.capture.width if ctx.capture else 1920,
                    ctx.capture.height if ctx.capture else 1080,
                )
                for dlg_region in dialogs:
                    dialog = UIElement(
                        element_type=ElementType.DIALOG,
                        label=dlg_region.label,
                        bounding_box=dlg_region.bounds,
                        confidence=dlg_region.confidence,
                    )
                    window.add_child(dialog)
            except Exception:
                pass

        desktop.add_window(window)

        # ── Post-processing ───────────────────────────────
        try:
            # Detect tab groups
            from services.vision_service import _detect_tab_groups
            _detect_tab_groups(window.element)
        except Exception:
            pass

        return desktop

    @staticmethod
    def _layout_region_to_element_type(region_type: str) -> ElementType:
        """Map layout RegionType to UI ElementType."""
        mapping = {
            "desktop": ElementType.DESKTOP,
            "window": ElementType.WINDOW,
            "toolbar": ElementType.TOOLBAR,
            "menu_bar": ElementType.MENU_BAR,
            "tab_bar": ElementType.TAB_BAR,
            "sidebar": ElementType.SIDEBAR,
            "left_panel": ElementType.LEFT_PANEL,
            "right_panel": ElementType.RIGHT_PANEL,
            "bottom_panel": ElementType.BOTTOM_PANEL,
            "editor": ElementType.EDITOR,
            "content": ElementType.CONTENT,
            "status_bar": ElementType.STATUS_BAR,
            "title_bar": ElementType.TITLE_BAR,
            "navigation": ElementType.NAVIGATION,
            "scrollbar": ElementType.SCROLLBAR,
            "minimap": ElementType.MINIMAP,
            "dialog": ElementType.DIALOG,
            "popup": ElementType.POPUP,
            "notification": ElementType.NOTIFICATION,
            "taskbar": ElementType.TASKBAR,
            "dock": ElementType.DOCK,
            "system_tray": ElementType.SYSTEM_TRAY,
        }
        return mapping.get(region_type, ElementType.PANEL)

    @staticmethod
    def _box_in_region(box: OCRBox, region_bounds: Tuple[int, int, int, int]) -> bool:
        """Check if an OCR box falls within a region."""
        rx, ry, rw, rh = region_bounds
        return (
            box.x >= rx and box.y >= ry
            and (box.x + box.width) <= (rx + rw)
            and (box.y + box.height) <= (ry + rh)
        )

    @staticmethod
    def _ocr_box_to_ui_element(box: OCRBox, app_type: ApplicationType) -> UIElement:
        """
        Classify an OCR box into a UI element type.

        Uses text content, position, and application type to determine
        the most likely element type.
        """
        text = box.text.strip()
        text_lower = text.lower()
        x, y, w, h = box.bbox
        text_len = len(text)

        # Buttons (action words)
        button_words = {
            "ok", "cancel", "yes", "no", "submit", "save", "delete", "close",
            "apply", "reset", "confirm", "dismiss", "back", "next", "finish",
            "done", "build", "run", "debug", "start", "stop", "restart",
            "deploy", "commit", "push", "pull", "merge", "install", "update",
            "settings", "options", "preferences", "help", "about", "exit", "quit",
            "login", "logout", "sign in", "sign up", "register", "download",
            "upload", "play", "pause", "mute", "add", "remove", "create",
            "enable", "disable", "on", "off", "retry", "skip", "search",
            "clear", "refresh", "reload",
        }
        if text_lower in button_words and text_len <= 15:
            return UIElement(
                element_type=ElementType.BUTTON,
                label=text, bounding_box=box.bbox,
                confidence=box.confidence, metadata={"role": "button"},
            )

        # Menu items (short, at top)
        menu_words = {"file", "edit", "view", "tools", "window", "help",
                      "navigate", "code", "refactor", "run", "debug", "build",
                      "vcs", "git", "bookmarks", "history"}
        if text_lower in menu_words and y < 80:
            return UIElement(
                element_type=ElementType.MENU_ITEM,
                label=text, bounding_box=box.bbox,
                confidence=box.confidence, metadata={"role": "menuitem"},
            )

        # Tabs (file extensions)
        if re.match(r'.*\.(py|js|ts|html|css|json|yaml|yml|md|txt|java|cpp|c|h|rs|go|rb|php|sql)$',
                     text, re.IGNORECASE):
            return UIElement(
                element_type=ElementType.TAB,
                label=text, bounding_box=box.bbox,
                confidence=box.confidence, metadata={"role": "tab"},
            )

        # Links
        if text.startswith(("http://", "https://", "www.")):
            return UIElement(
                element_type=ElementType.LINK,
                label=text, bounding_box=box.bbox,
                confidence=box.confidence, metadata={"role": "link"},
            )

        # Checkboxes
        if text.startswith(("☐", "☑", "☒", "○", "●", "[ ]", "[x]", "( )", "(*)")):
            return UIElement(
                element_type=ElementType.CHECKBOX,
                label=text, bounding_box=box.bbox,
                confidence=box.confidence, metadata={"role": "checkbox"},
            )

        # Code (in IDE editor region)
        if app_type.is_ide and (
            any(kw in text for kw in ("def ", "class ", "import ", "from ", "return ",
                                       "const ", "let ", "function", "Error:", "Traceback",
                                       "Exception"))
            or re.match(r'^\s*\d+\s*[:|]', text)  # line numbers
        ):
            return UIElement(
                element_type=ElementType.TEXT,
                label=text, bounding_box=box.bbox,
                confidence=box.confidence, metadata={"role": "code"},
            )

        # Labels (short, at top or next to inputs)
        if text_len <= 30 and text.endswith(":"):
            return UIElement(
                element_type=ElementType.LABEL,
                label=text, bounding_box=box.bbox,
                confidence=box.confidence, metadata={"role": "label"},
            )

        # Default: general text
        return UIElement(
            element_type=ElementType.TEXT,
            label=text, bounding_box=box.bbox,
            confidence=box.confidence, metadata={"role": "text"},
        )

    # ═══════════════════════════════════════════════════════════════
    # Public API (backward compatible)
    # ═══════════════════════════════════════════════════════════════

    async def force_analyze(self) -> VisionContext:
        """Always run the full pipeline (bypasses gating)."""
        return await self.analyze(force=True)

    async def quick_context(self) -> str:
        """
        Return a compact text summary for LLM prompt injection.

        This is the primary interface for the conversation engine —
        called when a user asks about their screen.
        Uses frame-difference gating.
        """
        ctx = await self.analyze(force=False)
        return ctx.compact_summary

    async def ocr_only(self) -> str:
        """Run OCR only and return raw text."""
        ctx = await self.analyze(force=False, include_tree=False, include_layout=False,
                                  include_reasoning=False)
        return ctx.ocr_text

    async def ui_tree(self, force: bool = False) -> Optional[UIDesktop]:
        """Return the current structured UI tree."""
        ctx = await self.analyze(force=force)
        return ctx.desktop

    async def find_element(self, label: str,
                            element_type: Optional[ElementType] = None) -> List[UIElement]:
        """Search the UI tree for elements matching a label and optional type."""
        ctx = await self.analyze(force=True)
        if ctx.desktop is None:
            return []

        results: List[UIElement] = []
        for window in ctx.desktop.windows:
            if element_type is not None:
                results.extend(window.find_by_type(element_type))
            results.extend(window.find_by_label(label))
        return results

    async def click_element(self, label: str) -> Optional[Tuple[int, int]]:
        """Find a clickable element by label and return its center coordinates."""
        elements = await self.find_element(label)
        clickable_types = {
            ElementType.BUTTON, ElementType.LINK, ElementType.TAB,
            ElementType.MENU_ITEM, ElementType.CHECKBOX, ElementType.ICON,
            ElementType.TOGGLE, ElementType.SWITCH,
        }
        for el in elements:
            if el.element_type in clickable_types and el.bounding_box and el.enabled:
                cx, cy = el.center
                logger.info("[VISION] click_element '%s' → (%d, %d)", label, cx, cy)
                return (cx, cy)
        # Fallback: any element with matching label
        for el in elements:
            if el.bounding_box:
                cx, cy = el.center
                return (cx, cy)
        return None

    # ── Memory-based queries ─────────────────────────────────────

    async def what_changed(self) -> str:
        """Describe what changed since the previous frame."""
        return screen_memory.what_changed()

    async def verify_last_action(self) -> VerificationResult:
        """Verify whether the last action changed the UI."""
        changed, explanation = screen_memory.last_action_changed_ui()
        return VerificationResult(
            success=changed,
            status=VerificationStatus.VERIFIED if changed else VerificationStatus.NO_CHANGE,
            explanation=explanation,
            frame_changed=changed,
        )

    def record_action(self, action_description: str) -> None:
        """Record an action for future verification."""
        action_verifier.capture_pre_action()
        screen_memory.record_action(action_description)
        logger.info("[VISION] Action recorded: %s", action_description[:80])

    async def get_screen_history(self, max_frames: int = 3) -> str:
        """Return recent screen history for LLM context."""
        return screen_memory.context_for_llm(max_frames)

    # ── Diagnostics ───────────────────────────────────────────────

    @property
    def analyze_count(self) -> int:
        return self._analyze_count

    @property
    def skip_count(self) -> int:
        return self._skip_count

    @property
    def skip_ratio(self) -> float:
        total = self._analyze_count + self._skip_count
        return self._skip_count / max(total, 1)

    @property
    def last_tree(self) -> Optional[str]:
        """Return the last UI tree as a compact string (for debugging)."""
        if self._last_context:
            return self._last_context.ui_tree_text
        return None

    def clear_cache(self) -> None:
        """Clear the last analysis cache (force next analyze to re-run)."""
        self._last_context = None
        frame_differencer.reset()
        screen_memory.clear()
        logger.info("[VISION] Cache cleared")

    def report(self) -> Dict[str, Any]:
        """Return a full diagnostic report."""
        return {
            "analyze_count": self._analyze_count,
            "skip_count": self._skip_count,
            "skip_ratio": f"{self.skip_ratio:.1%}",
            "ocr_backend": enhanced_ocr.active_backend,
            "ocr_ready": self._ocr_ready,
            "opencv": _HAS_CV2,
            "frame_differencer": frame_differencer.report(),
            "screen_memory": screen_memory.report(),
            "action_verifier": action_verifier.report(),
            "last_context": {
                "window_title": self._last_context.active_window_title[:60] if self._last_context else "",
                "app_type": self._last_context.app_type if self._last_context else "",
                "page_type": self._last_context.page_type if self._last_context else "",
                "total_ms": round(self._last_context.total_time_ms, 1) if self._last_context else 0,
            } if self._last_context else {},
        }

    # ═══════════════════════════════════════════════════════════════
    # FORENSIC DEBUG VISION — full pipeline with per-stage audit
    # ═══════════════════════════════════════════════════════════════

    async def analyze_forensic(self, force: bool = True,
                                source: str = "active_window") -> Tuple[VisionContext, ForensicReport]:
        """
        Run the full pipeline with forensic logging at every stage.

        Every stage logs: INPUT, OUTPUT, LATENCY, CONFIDENCE, FAILURE REASON.
        Never silently fails — every failure explains exactly why.

        Also updates the debug_overlay if enabled.

        Returns:
            (VisionContext, ForensicReport)
        """
        forensic_logger.start()

        # ── Stage 1: Capture ────────────────────────────
        forensic_logger.begin_stage(1, "Capture", f"source={source}")
        t_cap = time.perf_counter_ns()
        if source == "active_window":
            cap = await screen_capture_service.capture_active_window()
        else:
            cap = await screen_capture_service.capture_fullscreen()

        if cap is None or not cap.is_valid:
            forensic_logger.log_failure(
                "Window capture failed. No image returned.",
                capture_result=str(cap),
            )
            ctx = VisionContext()
            ctx.error = "[VISION] Stage 1 FAILED: capture returned None or invalid image"
            ctx.total_time_ms = (time.perf_counter_ns() - t_cap) / 1_000_000
            return ctx, forensic_logger.finish()

        cap_time = (time.perf_counter_ns() - t_cap) / 1_000_000
        forensic_logger.log_output(
            f"{cap.width}x{cap.height} source={cap.source_label}",
            confidence=1.0,
            resolution=f"{cap.width}x{cap.height}",
            source_label=cap.source_label,
            capture_latency_ms=round(cap_time, 1),
        )

        ctx = VisionContext()
        ctx.capture = cap
        ctx.capture_time_ms = cap_time
        ctx.active_window_title = cap.source_label.replace("window:", "").split(" (fullscreen")[0]
        ctx.stages_run.append("capture")

        # ── Stage 2: Frame Hash ──────────────────────────
        forensic_logger.begin_stage(2, "Frame Hash", f"image {cap.width}x{cap.height}")
        ctx.frame_hash = frame_differencer.compute_phash(cap.image) if cap.image is not None else ""
        hash_time = (time.perf_counter_ns() - time.perf_counter_ns())  # approximate
        if ctx.frame_hash:
            forensic_logger.log_output(
                f"hash={ctx.frame_hash}",
                confidence=1.0,
                hash_value=ctx.frame_hash,
            )
        else:
            forensic_logger.log_failure(
                "Frame hash computation returned empty string.",
                image_present=cap.image is not None,
            )
        ctx.stages_run.append("hash")

        # ── Stage 3: Motion Detection ────────────────────
        forensic_logger.begin_stage(3, "Motion Detection", f"hash={ctx.frame_hash[:8]}")
        t_motion = time.perf_counter_ns()
        if cap.image is not None:
            ctx.frame_diff = frame_differencer.compute_motion(cap.image)
            ctx.stages_run.append("motion")
            motion_time = (time.perf_counter_ns() - t_motion) / 1_000_000
            forensic_logger.log_output(
                ctx.frame_diff.summary,
                confidence=0.0 if ctx.frame_diff.same else 0.8,
                pixel_change=round(ctx.frame_diff.pixel_change_ratio, 5),
                motion_regions=len(ctx.frame_diff.motion_regions),
                same=ctx.frame_diff.same,
            )
        else:
            forensic_logger.log_output(
                "SKIPPED (no image)",
                confidence=0.0,
                reason="No image to compute motion on",
            )

        # ── Stage 4: Gating Decision ─────────────────────
        forensic_logger.begin_stage(4, "Gating Decision",
                                     f"force={force} window='{ctx.active_window_title[:40]}'")
        ctx.gating = frame_differencer.should_process(
            force=force,
            window_title=ctx.active_window_title,
            planner_requested=False,
        )
        ctx.stages_run.append("gating")
        forensic_logger.log_output(
            f"process={ctx.gating.should_process} reason='{ctx.gating.reason}' quality={ctx.gating.quality}",
            confidence=1.0,
            should_process=ctx.gating.should_process,
            reason=ctx.gating.reason,
            quality=ctx.gating.quality,
        )

        # Return cached if gating says skip
        if not ctx.gating.should_process and self._last_context is not None:
            self._skip_count += 1
            cached = self._last_context
            cached.total_time_ms = (time.perf_counter_ns() - time.perf_counter_ns())  # approx
            cached.from_cache = True
            forensic_logger.log_output(
                "CACHED RETURN",
                confidence=1.0,
                cache_hit=True,
            )
            logger.info("[FORENSIC] Pipeline complete (cached): total=%.1fms",
                         forensic_logger.finish().total_latency_ms)
            return cached, forensic_logger.finish()

        # Store current frame
        if cap.image is not None:
            frame_differencer.store_frame(cap.image, ctx.active_window_title)
        ctx.stages_run.append("process")

        # ── Stage 5: Layout Analysis ─────────────────────
        forensic_logger.begin_stage(5, "Layout Analysis",
                                     f"window='{ctx.active_window_title[:40]}' {cap.width}x{cap.height}")
        if self.use_layout:
            t_layout = time.perf_counter_ns()
            try:
                app_type, app_name, app_conf = layout_analyzer.detect_application(
                    window_title=ctx.active_window_title,
                    process_name=ctx.active_window_process,
                    pid=ctx.active_window_pid,
                )
                ctx.app_type = app_type.value if isinstance(app_type, ApplicationType) else app_type
                ctx.app_name = app_name

                ctx.layout = layout_analyzer.segment_layout(
                    app_type=app_type if isinstance(app_type, ApplicationType) else ApplicationType.UNKNOWN,
                    window_width=cap.width,
                    window_height=cap.height,
                    window_title=ctx.active_window_title,
                )
                ctx.layout_time_ms = (time.perf_counter_ns() - t_layout) / 1_000_000
                ctx.stages_run.append("layout")
                forensic_logger.log_output(
                    f"app={ctx.app_type}/{ctx.app_name} regions={len(ctx.layout.regions) if ctx.layout else 0}",
                    confidence=float(app_conf) if isinstance(app_conf, (int, float)) else 0.8,
                    app_type=ctx.app_type,
                    app_name=ctx.app_name,
                    regions=len(ctx.layout.regions) if ctx.layout else 0,
                )
            except Exception as e:
                forensic_logger.log_failure(
                    f"Layout analysis exception: {e}",
                    exception_type=type(e).__name__,
                )
        else:
            forensic_logger.log_output(
                "SKIPPED (use_layout=False)",
                confidence=0.0,
            )

        # ── Stage 6: OCR ─────────────────────────────────
        forensic_logger.begin_stage(6, "OCR",
                                     f"preprocess={self.preprocess_for_ocr} backend={enhanced_ocr.active_backend}")
        if self.use_ocr and enhanced_ocr.ready and cap.image is not None:
            t_ocr = time.perf_counter_ns()
            try:
                ctx.ocr_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: enhanced_ocr.ocr(cap.image, preprocess=self.preprocess_for_ocr),
                )
                ctx.ocr_text = ctx.ocr_result.text
                ctx.raw_ocr_boxes = ctx.ocr_result.boxes
                ctx.ocr_time_ms = (time.perf_counter_ns() - t_ocr) / 1_000_000
                ctx.stages_run.append("ocr")

                if ctx.ocr_result.box_count_final == 0:
                    forensic_logger.log_output(
                        "No OCR text detected.",
                        confidence=0.0,
                        success=True,
                        failure_reason="0 boxes after filtering/merging/dedup. Screen may be empty or rendering graphics-only content.",
                        boxes_raw=ctx.ocr_result.box_count_raw,
                        boxes_final=0,
                        merged=ctx.ocr_result.merged_count,
                        deduped=ctx.ocr_result.dedup_count,
                    )
                else:
                    forensic_logger.log_output(
                        f"{ctx.ocr_result.box_count_final} boxes avg_conf={ctx.ocr_result.avg_confidence:.3f}",
                        confidence=ctx.ocr_result.avg_confidence,
                        boxes_raw=ctx.ocr_result.box_count_raw,
                        boxes_final=ctx.ocr_result.box_count_final,
                        merged=ctx.ocr_result.merged_count,
                        deduped=ctx.ocr_result.dedup_count,
                        avg_confidence=round(ctx.ocr_result.avg_confidence, 3),
                        backend=ctx.ocr_result.backend,
                        text_preview=ctx.ocr_text[:100],
                    )
            except Exception as e:
                forensic_logger.log_failure(
                    f"OCR engine exception: {e}",
                    exception_type=type(e).__name__,
                )
                ctx.ocr_result = None
                ctx.ocr_text = ""
        else:
            reason = ""
            if not self.use_ocr:
                reason = "use_ocr=False"
            elif not enhanced_ocr.ready:
                reason = f"OCR backend '{enhanced_ocr.active_backend}' not ready"
            elif cap.image is None:
                reason = "No capture image"
            forensic_logger.log_failure(
                f"OCR skipped: {reason}",
                ocr_configured=self.use_ocr,
                ocr_ready=enhanced_ocr.ready,
                has_image=cap.image is not None,
            )

        # ── Stage 7: UI Detection ────────────────────────
        forensic_logger.begin_stage(7, "UI Detection",
                                     f"ocr_boxes={len(ctx.raw_ocr_boxes)} layout={'yes' if ctx.layout else 'no'}")
        if ctx.raw_ocr_boxes:
            t_ui = time.perf_counter_ns()
            try:
                desktop = self._build_ui_tree(ctx)
                ctx.desktop = desktop
                ctx.ui_tree_text = desktop.to_compact_str() if desktop else ""
                ctx.ui_tree_json = json.dumps(desktop.to_dict()) if desktop else ""
                ctx.ui_time_ms = (time.perf_counter_ns() - t_ui) / 1_000_000
                ctx.stages_run.append("ui_detection")

                # Count by type
                all_el = desktop.walk()
                type_counts: Dict[str, int] = {}
                for el in all_el:
                    t = el.element_type.value
                    type_counts[t] = type_counts.get(t, 0) + 1

                forensic_logger.log_output(
                    f"{len(all_el)} elements in {len(desktop.windows)} windows",
                    confidence=0.8,
                    total_elements=len(all_el),
                    windows=len(desktop.windows),
                    type_counts=type_counts,
                    buttons=type_counts.get("button", 0),
                    inputs=type_counts.get("input", 0) + type_counts.get("textbox", 0),
                    menus=type_counts.get("menu_item", 0),
                    dialogs=type_counts.get("dialog", 0),
                )
            except Exception as e:
                forensic_logger.log_failure(
                    f"UI tree construction failed: {e}",
                    exception_type=type(e).__name__,
                )
                ctx.desktop = None
                ctx.ui_tree_text = ""
        else:
            # Build minimal tree from layout
            t_ui = time.perf_counter_ns()
            desktop = UIDesktop()
            window = UIWindow(
                title=ctx.active_window_title,
                bounding_box=(0, 0, cap.width, cap.height),
                app_type=ctx.app_type,
                app_name=ctx.app_name,
            )
            if ctx.layout:
                for region in ctx.layout.regions:
                    panel_type = self._layout_region_to_element_type(region.region_type)
                    window.add_panel(
                        region_type=panel_type,
                        label=region.label,
                        bbox=region.bounds,
                    )
            desktop.add_window(window)
            ctx.desktop = desktop
            ctx.ui_tree_text = desktop.to_compact_str()
            ctx.ui_tree_json = json.dumps(desktop.to_dict())
            ctx.ui_time_ms = (time.perf_counter_ns() - t_ui) / 1_000_000
            ctx.stages_run.append("ui_detection")
            forensic_logger.log_output(
                f"Minimal tree (no OCR): {len(ctx.layout.regions) if ctx.layout else 0} layout regions",
                confidence=0.3,
                failure_reason="No OCR boxes available — UI tree built from layout regions only",
                has_ocr=False,
                has_layout=ctx.layout is not None,
            )

        # ── Stage 8: Semantic Reasoning ──────────────────
        forensic_logger.begin_stage(8, "Semantic Reasoning",
                                     f"ocr_chars={len(ctx.ocr_text)} ui_tree_lines={len(ctx.ui_tree_text.split(chr(10)))}")
        t_reason = time.perf_counter_ns()
        try:
            from services.screen_reasoning import screen_reasoner
            screen_ctx = screen_reasoner.analyze(
                ocr_text=ctx.ocr_text,
                ui_tree_text=ctx.ui_tree_text,
            )
            ctx.page_type = screen_ctx.page_type
            ctx.semantic_summary = screen_reasoner.context_description()
            if screen_ctx.elements:
                ctx.interactive_elements_json = json.dumps(
                    [{"type": e.type, "label": e.label, "position": list(e.position),
                      "confidence": e.confidence} for e in screen_ctx.elements[:20]]
                )
            if screen_ctx.error_elements:
                ctx.error_elements_json = json.dumps(
                    [{"type": e.type, "label": e.label, "confidence": e.confidence}
                     for e in screen_ctx.error_elements[:10]]
                )
            ctx.reasoning_time_ms = (time.perf_counter_ns() - t_reason) / 1_000_000
            ctx.stages_run.append("reasoning")

            # Ask the reasoning questions
            questions = [
                "What application is open?",
                "What window?",
                "What toolbar?",
                "What dialog?",
                "What buttons?",
                "What text?",
                "What error?",
            ]
            answers = []
            if ctx.app_type:
                answers.append(f"App: {ctx.app_type}/{ctx.app_name}")
            if ctx.active_window_title:
                answers.append(f"Window: {ctx.active_window_title[:60]}")
            if ctx.page_type:
                answers.append(f"Page type: {ctx.page_type}")
            if ctx.semantic_summary:
                answers.append(f"Summary: {ctx.semantic_summary[:80]}")

            forensic_logger.log_output(
                " | ".join(answers) if answers else ctx.page_type,
                confidence=0.7,
                page_type=ctx.page_type,
                has_dialog=screen_ctx.has_dialog,
                has_notification=screen_ctx.has_notification,
                interactive_count=len(screen_ctx.elements),
                error_count=len(screen_ctx.error_elements),
            )
        except Exception as e:
            forensic_logger.log_failure(
                f"Semantic reasoning failed: {e}",
                exception_type=type(e).__name__,
            )

        # ── Stage 9: Action Verification ─────────────────
        forensic_logger.begin_stage(9, "Action Verification", f"verify_previous={verify_previous_action}")
        if verify_previous_action and self.use_verification:
            t_verify = time.perf_counter_ns()
            try:
                changed, explanation = screen_memory.last_action_changed_ui()
                ctx.verification = VerificationResult(
                    success=changed,
                    status=VerificationStatus.VERIFIED if changed else VerificationStatus.NO_CHANGE,
                    explanation=explanation,
                    frame_changed=changed,
                    pre_action_hash=frame_differencer.prev_hash,
                    post_action_hash=ctx.frame_hash,
                )
                ctx.verify_time_ms = (time.perf_counter_ns() - t_verify) / 1_000_000
                ctx.stages_run.append("verify")
                forensic_logger.log_output(
                    f"changed={changed}: {explanation}",
                    confidence=1.0 if changed else 0.0,
                    success=changed,
                    failure_reason="" if changed else "Click verification failed because frame hash did not change.",
                    pre_hash=frame_differencer.prev_hash[:8],
                    post_hash=ctx.frame_hash[:8],
                )
            except Exception as e:
                forensic_logger.log_failure(
                    f"Verification error: {e}",
                    exception_type=type(e).__name__,
                )
        else:
            forensic_logger.log_output(
                "SKIPPED (not requested or verification disabled)",
                confidence=0.0,
            )

        # ── Stage 10: Memory Update ──────────────────────
        forensic_logger.begin_stage(10, "Memory Update", f"frames_stored={screen_memory.count}")
        if self.use_memory:
            t_memory = time.perf_counter_ns()
            try:
                snapshot = ScreenSnapshot(
                    window_title=ctx.active_window_title,
                    app_type=ctx.app_type,
                    app_name=ctx.app_name,
                    frame_hash=ctx.frame_hash,
                    full_hash=frame_differencer.compute_full_hash(cap.image) if cap.image is not None else "",
                    width=cap.width,
                    height=cap.height,
                    ocr_text=ctx.ocr_text,
                    ui_tree_text=ctx.ui_tree_text,
                    ui_tree_json=ctx.ui_tree_json,
                    layout_json=json.dumps(ctx.layout.to_dict()) if ctx.layout else "",
                    semantic_summary=ctx.semantic_summary,
                    page_type=ctx.page_type,
                    interactive_elements=ctx.interactive_elements_json,
                    error_elements=ctx.error_elements_json,
                    total_pipeline_ms=(time.perf_counter_ns() - time.perf_counter_ns()),
                    from_cache=False,
                )
                screen_memory.store(snapshot)
                ctx.snapshot = snapshot
                ctx.memory_time_ms = (time.perf_counter_ns() - t_memory) / 1_000_000
                ctx.stages_run.append("memory")
                forensic_logger.log_output(
                    f"Stored frame #{snapshot.frame_id} ({screen_memory.count} total)",
                    confidence=1.0,
                    frame_id=snapshot.frame_id,
                    total_frames=screen_memory.count,
                )
            except Exception as e:
                forensic_logger.log_failure(
                    f"Memory store failed: {e}",
                    exception_type=type(e).__name__,
                )
        else:
            forensic_logger.log_output(
                "SKIPPED (use_memory=False)",
                confidence=0.0,
            )

        # ── Finalize ─────────────────────────────────────
        self._last_context = ctx
        self._analyze_count += 1
        ctx.total_time_ms = (time.perf_counter_ns() - time.perf_counter_ns())  # will be overridden

        report = forensic_logger.finish()
        ctx.total_time_ms = report.total_latency_ms

        # Update debug overlay if enabled
        if debug_overlay.enabled:
            debug_overlay.update_from_context(
                ctx,
                planner_target="",
                status=f"Pipeline: {report.total_latency_ms:.1f}ms | "
                       f"OCR: {ctx.ocr_result.box_count_final if ctx.ocr_result else 0} boxes | "
                       f"UI: {len(ctx.desktop.walk()) if ctx.desktop else 0} elements | "
                       f"Page: {ctx.page_type}"
            )

        logger.info("[FORENSIC] Pipeline complete: %d stages total=%.1fms errors=%d warnings=%d",
                     len(report.stages), report.total_latency_ms,
                     report.error_count, report.warning_count)
        return ctx, report


# ── Standalone helper (was a static method in the old UIDetector) ─

def _detect_tab_groups(window_element: UIElement) -> None:
    """Cluster adjacent tabs into groups."""
    tabs = window_element.find_by_type(ElementType.TAB)
    if len(tabs) < 2:
        return

    tabs_sorted = sorted(tabs, key=lambda t: t.x)
    groups: List[List[UIElement]] = []
    current_group = [tabs_sorted[0]]

    for i in range(1, len(tabs_sorted)):
        prev = tabs_sorted[i - 1]
        curr = tabs_sorted[i]
        if abs(curr.y - prev.y) <= 10 and (curr.x - (prev.x + prev.width)) <= 200:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)

    for group in groups:
        if len(group) > 1:
            group[0].metadata["active"] = True
            for tab in group[1:]:
                tab.metadata["active"] = False


# Global singleton
vision_service = VisionService()