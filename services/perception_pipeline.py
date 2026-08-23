
"""
Unified Perception Pipeline — Diego's single source of desktop truth.

Every request follows this exact flow:

    Screen Capture
        ↓
    Active Window (xdotool)
        ↓
    Window Title
        ↓
    Accessibility Tree (AT-SPI → Browser CDP → X11)
        ↓
    UI Tree (structured hierarchy)
        ↓
    OCR (FALLBACK ONLY — skipped if accessibility data is sufficient)
        ↓
    Desktop State (clipboard, git, terminal, browser, system)
        ↓
    Focused Element
        ↓
    Screen Reasoning (page type, errors, interactive elements)
        ↓
    Planner Context (LLM-ready compact summary)

Key principles:
  1. NEVER run OCR if accessibility data is sufficient
  2. Cache perception results — only rerun when screen/window changes
  3. Every action must be verified
  4. Prefer structured data over pixel data

Usage:
    from services.perception_pipeline import perception_pipeline

    # Full perception cycle
    ctx = await perception_pipeline.perceive()

    # Quick context for LLM
    summary = await perception_pipeline.quick_perceive()

    # Verify an action
    result = await perception_pipeline.verify_action("click", {"label": "Run"})

Logging: [PERCEPTION]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

@dataclass
class PerceptionContext:
    """
    Complete perception result — everything Diego knows about the desktop
    at a single point in time.
    """
    # ── Identity ────────────────────────────────────────────
    perception_id: str = ""             # unique hash of this perception
    timestamp: float = 0.0
    total_latency_ms: float = 0.0

    # ── Screen ──────────────────────────────────────────────
    screen_hash: str = ""               # perceptual hash of screen
    screen_width: int = 0
    screen_height: int = 0
    screen_changed: bool = False        # did screen change since last perception?

    # ── Window ──────────────────────────────────────────────
    window_id: str = ""                 # X11 window ID
    window_title: str = ""
    window_process: str = ""            # process name
    window_pid: int = 0
    window_class: str = ""              # WM_CLASS
    window_changed: bool = False

    # ── Accessibility ───────────────────────────────────────
    a11y_available: bool = False
    a11y_backend: str = ""              # "at-spi", "browser_cdp", "x11_window", "none"
    a11y_node_count: int = 0
    a11y_clickable_count: int = 0
    a11y_tree_json: str = ""            # JSON serialized tree
    a11y_tree_text: str = ""            # compact text representation

    # ── UI Tree ─────────────────────────────────────────────
    ui_tree_text: str = ""
    ui_tree_json: str = ""
    ui_element_count: int = 0

    # ── OCR ─────────────────────────────────────────────────
    ocr_used: bool = False              # was OCR actually run?
    ocr_skipped_reason: str = ""        # why OCR was skipped (or "" if run)
    ocr_text: str = ""
    ocr_box_count: int = 0
    ocr_confidence: float = 0.0
    ocr_latency_ms: float = 0.0

    # ── Desktop State ───────────────────────────────────────
    clipboard_text: str = ""
    git_branch: str = ""
    terminal_cwd: str = ""
    browser_tab: str = ""
    browser_url: str = ""
    current_file: str = ""              # file being edited in IDE
    current_ide: str = ""               # IDE/editor name

    # ── Focused Element ─────────────────────────────────────
    focused_element_type: str = ""      # "button", "text_box", "tab", etc.
    focused_element_label: str = ""
    focused_element_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # ── Reasoning ───────────────────────────────────────────
    page_type: str = ""                 # "editor", "browser", "terminal", "settings", etc.
    semantic_summary: str = ""          # human-readable description
    error_elements_json: str = ""       # detected errors
    interactive_elements_json: str = "" # clickable elements
    has_dialog: bool = False
    has_notification: bool = False

    # ── Application ─────────────────────────────────────────
    app_type: str = ""                  # from layout analyzer
    app_name: str = ""

    # ── Cache ───────────────────────────────────────────────
    from_cache: bool = False
    stages_run: List[str] = field(default_factory=list)

    # ── Verification ────────────────────────────────────────
    last_action_verified: bool = False
    last_action_explanation: str = ""

    @property
    def compact_summary(self) -> str:
        """Ultra-compact summary for LLM prompt injection."""
        parts: List[str] = []

        if self.window_title:
            app_info = self.window_title
            if self.app_type:
                app_info = f"[{self.app_type}] {app_info}"
            parts.append(f"Window: {app_info}")

        if self.page_type and self.page_type != "unknown":
            parts.append(f"Page: {self.page_type}")

        if self.a11y_available:
            parts.append(f"A11y: {self.a11y_clickable_count} clickable, {self.a11y_node_count} nodes ({self.a11y_backend})")

        if self.ui_tree_text:
            parts.append(self.ui_tree_text[:500])

        if self.semantic_summary:
            parts.append(f"Summary: {self.semantic_summary}")

        if self.ocr_used and self.ocr_text:
            parts.append(f"OCR: {self.ocr_text[:300]}")

        if self.focused_element_label:
            parts.append(f"Focused: {self.focused_element_type} '{self.focused_element_label}'")

        if self.clipboard_text:
            parts.append(f"Clipboard: {self.clipboard_text[:80]}")

        if self.git_branch:
            parts.append(f"Git: {self.git_branch}")

        if self.browser_tab:
            parts.append(f"Browser: {self.browser_tab}")

        if self.error_elements_json:
            try:
                errors = json.loads(self.error_elements_json)
                if errors:
                    parts.append(f"Errors: {len(errors)} detected")
            except Exception:
                pass

        return "\n".join(parts)

    @property
    def quick_context(self) -> str:
        """One-line context for fast injection."""
        items = []
        if self.window_title:
            items.append(f"Window: {self.window_title[:60]}")
        if self.app_type:
            items.append(f"App: {self.app_type}")
        if self.page_type:
            items.append(f"Page: {self.page_type}")
        if self.semantic_summary:
            items.append(self.semantic_summary[:100])
        if self.a11y_available:
            items.append(f"A11y: {self.a11y_backend}")
        elif self.ocr_used:
            items.append("OCR: used")
        return " | ".join(items) if items else ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/metrics."""
        return {
            "perception_id": self.perception_id,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "window_title": self.window_title[:100],
            "window_process": self.window_process,
            "app_type": self.app_type,
            "page_type": self.page_type,
            "a11y_available": self.a11y_available,
            "a11y_backend": self.a11y_backend,
            "a11y_node_count": self.a11y_node_count,
            "ocr_used": self.ocr_used,
            "ocr_skipped_reason": self.ocr_skipped_reason,
            "from_cache": self.from_cache,
            "stages_run": self.stages_run,
            "screen_changed": self.screen_changed,
            "window_changed": self.window_changed,
        }


# ═══════════════════════════════════════════════════════════════
# PerceptionPipeline — the unified orchestrator
# ═══════════════════════════════════════════════════════════════

class PerceptionPipeline:
    """
    Unified perception pipeline.

    Every perception request follows the same flow:
      Screen → Window → Title → A11y → UI Tree → OCR (fallback) →
      Desktop State → Clipboard → Focused Element → Reasoning → Planner

    Caches results and only reruns expensive stages (OCR) when
    the screen or window actually changed.
    """

    def __init__(self):
        self._initialized: bool = False
        self._last_ctx: Optional[PerceptionContext] = None
        self._last_screen_hash: str = ""
        self._last_window_id: str = ""
        self._perception_count: int = 0
        self._cache_hit_count: int = 0
        self._ocr_run_count: int = 0
        self._ocr_skip_count: int = 0
        self._a11y_success_count: int = 0
        self._a11y_fail_count: int = 0

        # Subsystems (lazy init)
        self._a11y = None
        self._vision = None
        self._desktop_state = None
        self._screen_reasoner = None
        self._layout_analyzer = None
        self._action_verifier = None
        self._screen_capture = None

    # ── Lazy subsystem access ───────────────────────────────

    @property
    def a11y(self):
        if self._a11y is None:
            from services.accessibility import accessibility_tree
            self._a11y = accessibility_tree
        return self._a11y

    @property
    def vision(self):
        if self._vision is None:
            from services.vision_service import vision_service
            self._vision = vision_service
        return self._vision

    @property
    def desktop_state(self):
        if self._desktop_state is None:
            from services.desktop_state import desktop_state
            self._desktop_state = desktop_state
        return self._desktop_state

    @property
    def screen_reasoner(self):
        if self._screen_reasoner is None:
            from services.screen_reasoning import screen_reasoner
            self._screen_reasoner = screen_reasoner
        return self._screen_reasoner

    @property
    def layout_analyzer(self):
        if self._layout_analyzer is None:
            from vision.layout_analyzer import layout_analyzer
            self._layout_analyzer = layout_analyzer
        return self._layout_analyzer

    @property
    def action_verifier(self):
        if self._action_verifier is None:
            from vision.action_verifier import action_verifier
            self._action_verifier = action_verifier
        return self._action_verifier

    @property
    def screen_capture(self):
        if self._screen_capture is None:
            from services.screen_capture import screen_capture_service
            self._screen_capture = screen_capture_service
        return self._screen_capture

    # ── Initialization ──────────────────────────────────────

    async def initialize(self) -> bool:
        """Initialize all perception subsystems."""
        logger.info("[PERCEPTION] Initializing perception pipeline...")

        # Initialize accessibility
        a11y_ok = self.a11y.initialize()
        if a11y_ok:
            logger.info("[PERCEPTION] Accessibility: %s", self.a11y.available_backends)
        else:
            logger.warning("[PERCEPTION] No accessibility backends — OCR will be primary")

        # Initialize screen capture service
        capture_ok = False
        try:
            capture_ok = await self.screen_capture._start()
            if capture_ok:
                logger.info("[PERCEPTION] Screen capture: %s", self.screen_capture._backend)
            else:
                logger.warning("[PERCEPTION] Screen capture: no backend available")
        except Exception as e:
            logger.warning("[PERCEPTION] Screen capture init failed: %s", e)

        # Initialize vision (OCR, layout, etc.)
        try:
            await self.vision._start()
            logger.info("[PERCEPTION] Vision service: ready")
        except Exception as e:
            logger.warning("[PERCEPTION] Vision service init failed: %s", e)

        self._initialized = True
        logger.info("[PERCEPTION] Pipeline initialized — a11y=%s capture=%s vision=%s",
                     "✓" if a11y_ok else "✗",
                     "✓" if capture_ok else "✗",
                     "✓" if self.vision.ready else "✗")
        return True

    # ═══════════════════════════════════════════════════════════
    # MAIN API: perceive()
    # ═══════════════════════════════════════════════════════════

    async def perceive(
        self,
        force: bool = False,
        force_ocr: bool = False,
        include_ocr: bool = True,
        include_reasoning: bool = True,
        include_desktop_state: bool = True,
    ) -> PerceptionContext:
        """
        Run the full unified perception pipeline.

        Flow:
          1. Screen Capture
          2. Active Window Detection
          3. Window Title
          4. Accessibility Tree (AT-SPI → Browser CDP → X11)
          5. UI Tree Construction
          6. OCR (FALLBACK — skipped if a11y is sufficient)
          7. Desktop State (clipboard, git, terminal, browser)
          8. Focused Element
          9. Screen Reasoning
          10. Build compact summary for planner

        Args:
            force: Bypass all caching — run full pipeline.
            force_ocr: Force OCR even if accessibility data is sufficient.
            include_ocr: Allow OCR (set False to never run OCR).
            include_reasoning: Run semantic reasoning stage.
            include_desktop_state: Collect desktop state (clipboard, git, etc.).

        Returns:
            PerceptionContext with complete desktop understanding.
        """
        t0 = time.perf_counter_ns()
        ctx = PerceptionContext()
        ctx.timestamp = time.time()
        self._perception_count += 1

        # ── Lazy initialization: make sure screen capture backend is up.
        # Without this, a missed initialize() call leaves the backend "none"
        # and every capture silently fails.
        if self.screen_capture._backend == "none":
            try:
                await self.screen_capture._start()
            except Exception as e:
                logger.warning("[PERCEPTION] Lazy screen-capture init failed: %s", e)

        # ═══════════════════════════════════════════════════════
        # Stage 1: Screen Capture
        # ═══════════════════════════════════════════════════════
        t_cap = time.perf_counter_ns()
        try:
            cap = await self.screen_capture.capture_fullscreen()
            if cap and cap.is_valid:
                ctx.screen_width = cap.width
                ctx.screen_height = cap.height
                ctx.screen_hash = self._compute_phash(cap.image) if cap.image is not None else ""
                ctx.stages_run.append("screen_capture")
                logger.debug("[PERCEPTION] Stage 1 — Screen: %dx%d hash=%s",
                             cap.width, cap.height, ctx.screen_hash[:8])
            else:
                logger.warning("[PERCEPTION] Stage 1 — Screen capture failed")
        except Exception as e:
            logger.warning("[PERCEPTION] Stage 1 — Screen capture error: %s", e)

        # Check if screen changed
        if ctx.screen_hash and ctx.screen_hash != self._last_screen_hash:
            ctx.screen_changed = True
            self._last_screen_hash = ctx.screen_hash

        # ═══════════════════════════════════════════════════════
        # Stage 2: Active Window Detection
        # ═══════════════════════════════════════════════════════
        t_win = time.perf_counter_ns()
        try:
            window_info = self.desktop_state._get_focused_window()
            ctx.window_title = window_info.title
            ctx.window_pid = window_info.pid
            ctx.window_process = window_info.application
            ctx.stages_run.append("window_detect")

            # Get window ID for cache key
            ctx.window_id = self._get_window_id()
            if ctx.window_id and ctx.window_id != self._last_window_id:
                ctx.window_changed = True
                self._last_window_id = ctx.window_id

            logger.debug("[PERCEPTION] Stage 2 — Window: '%s' pid=%d",
                         ctx.window_title[:60], ctx.window_pid)
        except Exception as e:
            logger.warning("[PERCEPTION] Stage 2 — Window detection error: %s", e)

        # ═══════════════════════════════════════════════════════
        # CACHE CHECK: Return cached if nothing changed
        # ═══════════════════════════════════════════════════════
        if not force and not ctx.screen_changed and not ctx.window_changed and self._last_ctx is not None:
            self._cache_hit_count += 1
            cached = self._last_ctx
            cached.from_cache = True
            cached.total_latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
            logger.debug("[PERCEPTION] CACHE HIT — returning cached perception (%.1fms)",
                         cached.total_latency_ms)
            return cached

        # ═══════════════════════════════════════════════════════
        # Stage 3: Application Type Detection
        # ═══════════════════════════════════════════════════════
        try:
            app_type, app_name, app_conf = self.layout_analyzer.detect_application(
                window_title=ctx.window_title,
                process_name=ctx.window_process,
                pid=ctx.window_pid,
            )
            from vision.layout_analyzer import ApplicationType
            ctx.app_type = app_type.value if isinstance(app_type, ApplicationType) else str(app_type)
            ctx.app_name = app_name
            ctx.stages_run.append("app_detect")
            logger.debug("[PERCEPTION] Stage 3 — App: %s/%s (conf=%.2f)",
                         ctx.app_type, ctx.app_name, app_conf)
        except Exception as e:
            logger.debug("[PERCEPTION] Stage 3 — App detection error: %s", e)

        # ═══════════════════════════════════════════════════════
        # Stage 4: Accessibility Tree (PRIMARY data source)
        # ═══════════════════════════════════════════════════════
        t_a11y = time.perf_counter_ns()
        try:
            a11y_tree = self.a11y.get_tree(force_refresh=force or ctx.window_changed)
            if a11y_tree is not None and a11y_tree.has_meaningful_content():
                ctx.a11y_available = True
                ctx.a11y_backend = a11y_tree.backend.value
                ctx.a11y_node_count = len(a11y_tree.walk())
                ctx.a11y_clickable_count = len(a11y_tree.find_clickable())
                ctx.a11y_tree_text = a11y_tree.to_compact_str()
                ctx.a11y_tree_json = json.dumps(a11y_tree.to_dict())
                self._a11y_success_count += 1
                ctx.stages_run.append("accessibility")
                logger.info("[PERCEPTION] Stage 4 — A11y: %d nodes, %d clickable (%s) in %.1fms",
                             ctx.a11y_node_count, ctx.a11y_clickable_count,
                             ctx.a11y_backend,
                             (time.perf_counter_ns() - t_a11y) / 1_000_000)
            else:
                self._a11y_fail_count += 1
                logger.debug("[PERCEPTION] Stage 4 — A11y: no meaningful data")
        except Exception as e:
            self._a11y_fail_count += 1
            logger.debug("[PERCEPTION] Stage 4 — A11y error: %s", e)

        # ═══════════════════════════════════════════════════════
        # Stage 5: UI Tree Construction
        # ═══════════════════════════════════════════════════════
        t_ui = time.perf_counter_ns()
        try:
            if ctx.a11y_available:
                # Build UI tree from accessibility data (preferred)
                ui_tree = self._build_ui_tree_from_a11y(ctx)
            else:
                # Build minimal UI tree from window info
                ui_tree = self._build_minimal_ui_tree(ctx)

            if ui_tree:
                ctx.ui_tree_text = ui_tree
                ctx.ui_tree_json = "{}"  # Will be populated if OCR runs
                ctx.stages_run.append("ui_tree")
                logger.debug("[PERCEPTION] Stage 5 — UI Tree built in %.1fms",
                             (time.perf_counter_ns() - t_ui) / 1_000_000)
        except Exception as e:
            logger.debug("[PERCEPTION] Stage 5 — UI Tree error: %s", e)

        # ═══════════════════════════════════════════════════════
        # Stage 6: OCR (FALLBACK ONLY)
        # ═══════════════════════════════════════════════════════
        if include_ocr:
            t_ocr = time.perf_counter_ns()

            # DECISION: Should we run OCR?
            # OCR is the FALLBACK — skip only when a11y data is sufficient
            # or when screen is unchanged and we have cached OCR.
            should_ocr = True
            ocr_skip_reason = ""

            # CRITICAL FIX: only skip OCR when a11y actually contains
            # meaningful UI content. A bare window shell (1 node, 0
            # clickable) tells us almost nothing — e.g. a browser page
            # exposed via x11_window has no page text. In that case OCR
            # must run so Diego can actually "see" the screen contents.
            if ctx.a11y_available and ctx.a11y_node_count > 1:
                should_ocr = False
                ocr_skip_reason = f"Accessibility data sufficient ({ctx.a11y_node_count} nodes from {ctx.a11y_backend})"
            elif not ctx.screen_changed and self._last_ctx is not None and self._last_ctx.ocr_text:
                should_ocr = False
                ocr_skip_reason = "Screen unchanged — using cached OCR"
                ctx.ocr_text = self._last_ctx.ocr_text
                ctx.ocr_box_count = self._last_ctx.ocr_box_count
                ctx.ocr_confidence = self._last_ctx.ocr_confidence

            if should_ocr or force_ocr:
                try:
                    ocr_ctx = await self.vision.analyze(
                        force=True,
                        source="fullscreen",
                        include_tree=False,
                        include_layout=False,
                        include_reasoning=False,
                    )
                    ctx.ocr_text = ocr_ctx.ocr_text
                    ctx.ocr_box_count = ocr_ctx.ocr_result.box_count_final if ocr_ctx.ocr_result else 0
                    ctx.ocr_confidence = ocr_ctx.ocr_result.avg_confidence if ocr_ctx.ocr_result else 0.0
                    ctx.ocr_used = True
                    ctx.ocr_latency_ms = ocr_ctx.ocr_time_ms
                    self._ocr_run_count += 1
                    ctx.stages_run.append("ocr")
                    logger.info("[PERCEPTION] Stage 6 — OCR: %d boxes, conf=%.2f in %.1fms",
                                 ctx.ocr_box_count, ctx.ocr_confidence, ctx.ocr_latency_ms)
                except Exception as e:
                    logger.warning("[PERCEPTION] Stage 6 — OCR error: %s", e)
                    ocr_skip_reason = f"OCR failed: {e}"
            else:
                ctx.ocr_skipped_reason = ocr_skip_reason
                self._ocr_skip_count += 1
                logger.info("[PERCEPTION] Stage 6 — OCR SKIPPED: %s", ocr_skip_reason)

        # ═══════════════════════════════════════════════════════
        # Stage 7: Desktop State
        # ═══════════════════════════════════════════════════════
        if include_desktop_state:
            t_ds = time.perf_counter_ns()
            try:
                snap = self.desktop_state.snapshot()
                ctx.clipboard_text = snap.clipboard_text
                ctx.git_branch = snap.terminal.git_branch
                ctx.terminal_cwd = snap.terminal.cwd
                ctx.browser_tab = snap.browser_tab
                ctx.browser_url = snap.browser_url
                ctx.current_file = self.desktop_state.current_file()
                ctx.current_ide = self.desktop_state.current_ide()
                ctx.stages_run.append("desktop_state")
                logger.debug("[PERCEPTION] Stage 7 — Desktop State in %.1fms",
                             (time.perf_counter_ns() - t_ds) / 1_000_000)
            except Exception as e:
                logger.debug("[PERCEPTION] Stage 7 — Desktop State error: %s", e)

        # ═══════════════════════════════════════════════════════
        # Stage 8: Focused Element
        # ═══════════════════════════════════════════════════════
        t_focus = time.perf_counter_ns()
        try:
            if ctx.a11y_available:
                focused = self.a11y.get_focused_element()
                if focused:
                    ctx.focused_element_type = focused.role.value
                    ctx.focused_element_label = focused.name
                    ctx.focused_element_bounds = focused.bounds
                    ctx.stages_run.append("focused_element")
                    logger.debug("[PERCEPTION] Stage 8 — Focused: %s '%s'",
                                 ctx.focused_element_type, ctx.focused_element_label[:40])
        except Exception as e:
            logger.debug("[PERCEPTION] Stage 8 — Focused element error: %s", e)

        # ═══════════════════════════════════════════════════════
        # Stage 9: Screen Reasoning
        # ═══════════════════════════════════════════════════════
        if include_reasoning:
            t_reason = time.perf_counter_ns()
            try:
                # Combine all text sources for reasoning
                combined_text = ctx.ocr_text
                if ctx.a11y_tree_text:
                    combined_text = ctx.a11y_tree_text + "\n" + combined_text
                if ctx.ui_tree_text:
                    combined_text = ctx.ui_tree_text + "\n" + combined_text

                screen_ctx = self.screen_reasoner.analyze(
                    ocr_text=combined_text,
                    ui_tree_text=ctx.ui_tree_text,
                    app_type=ctx.app_type,
                )
                ctx.page_type = screen_ctx.page_type
                ctx.semantic_summary = self.screen_reasoner.context_description()
                ctx.has_dialog = screen_ctx.has_dialog
                ctx.has_notification = screen_ctx.has_notification

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

                ctx.stages_run.append("reasoning")
                logger.info("[PERCEPTION] Stage 9 — Reasoning: page=%s errors=%d in %.1fms",
                             ctx.page_type,
                             len(screen_ctx.error_elements),
                             (time.perf_counter_ns() - t_reason) / 1_000_000)
            except Exception as e:
                logger.warning("[PERCEPTION] Stage 9 — Reasoning error: %s", e)

        # ═══════════════════════════════════════════════════════
        # Finalize
        # ═══════════════════════════════════════════════════════
        ctx.perception_id = self._compute_perception_id(ctx)
        ctx.total_latency_ms = (time.perf_counter_ns() - t0) / 1_000_000

        self._last_ctx = ctx

        logger.info("[PERCEPTION] Pipeline complete: stages=%s total=%.1fms window='%s' a11y=%s ocr=%s",
                     "+".join(ctx.stages_run), ctx.total_latency_ms,
                     ctx.window_title[:60],
                     "✓" if ctx.a11y_available else "✗",
                     "used" if ctx.ocr_used else f"skipped({ctx.ocr_skipped_reason[:40]})")
        return ctx

    # ── Convenience methods ─────────────────────────────────

    async def quick_perceive(self) -> str:
        """Fast perception — return compact summary for LLM injection."""
        ctx = await self.perceive()
        return ctx.compact_summary

    async def perceive_for_planner(self) -> str:
        """Perception optimized for the planner — one-line context."""
        ctx = await self.perceive()
        return ctx.quick_context

    async def force_perceive(self) -> PerceptionContext:
        """Force a full perception cycle (bypasses all caching)."""
        return await self.perceive(force=True, force_ocr=False)

    async def perceive_with_ocr(self) -> PerceptionContext:
        """Force OCR even if accessibility data is available."""
        return await self.perceive(force_ocr=True)

    # ── Action Verification ─────────────────────────────────

    async def verify_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        expected_outcome: str = "",
        expected_element: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify that a desktop action had the expected effect.

        Full verification chain:
          1. Check process exists (for open_app)
          2. Check focused window changed
          3. Check window title
          4. Check screen hash changed
          5. Run perception to confirm

        Returns:
            Dict with success, explanation, retry_needed, recovery_action
        """
        from vision.action_verifier import VerificationStatus

        # Capture pre-action state
        self.action_verifier.capture_pre_action()

        # Wait for UI to stabilize
        await asyncio.sleep(0.3)

        # Run fresh perception
        ctx = await self.perceive(force=True)

        # Build verification result
        result = {
            "success": False,
            "explanation": "",
            "retry_needed": False,
            "recovery_action": "",
            "details": {},
        }

        # ── Action-specific verification ─────────────────
        if action_type == "open_app":
            result = await self._verify_open_app(action_params, ctx)
        elif action_type == "click":
            result = await self._verify_click(action_params, ctx, expected_element)
        elif action_type == "type":
            result = await self._verify_type(action_params, ctx)
        elif action_type == "navigate":
            result = await self._verify_navigate(action_params, ctx)
        elif action_type == "scroll":
            result = await self._verify_scroll(ctx)
        else:
            # Generic verification: did anything change?
            if ctx.screen_changed or ctx.window_changed:
                result["success"] = True
                result["explanation"] = f"Screen or window changed after {action_type}"
            else:
                result["success"] = False
                result["explanation"] = f"No visible change after {action_type}"
                result["retry_needed"] = True
                result["recovery_action"] = f"Retry {action_type} with adjusted parameters"

        ctx.last_action_verified = result["success"]
        ctx.last_action_explanation = result["explanation"]

        logger.info("[PERCEPTION] Verify %s: success=%s — %s",
                     action_type, result["success"], result["explanation"])
        return result

    async def _verify_open_app(self, params: Dict[str, Any], ctx: PerceptionContext) -> Dict[str, Any]:
        """Verify an app was opened successfully."""
        app_name = params.get("app_name", params.get("label", ""))
        result = {
            "success": False,
            "explanation": "",
            "retry_needed": False,
            "recovery_action": "",
            "details": {},
        }

        checks = []

        # Check 1: Process exists
        process_exists = False
        if app_name:
            import shutil
            if shutil.which("pgrep"):
                try:
                    import subprocess
                    proc = subprocess.run(
                        ["pgrep", "-x", app_name],
                        capture_output=True, text=True, timeout=2
                    )
                    process_exists = proc.returncode == 0
                    checks.append(f"Process '{app_name}': {'found' if process_exists else 'not found'}")
                except Exception:
                    checks.append(f"Process check failed")
            result["details"]["process_exists"] = process_exists

        # Check 2: Window title contains app name
        window_match = app_name.lower() in ctx.window_title.lower() if app_name else False
        checks.append(f"Window title: {'matches' if window_match else 'no match'}")
        result["details"]["window_match"] = window_match

        # Check 3: Screen changed
        checks.append(f"Screen: {'changed' if ctx.screen_changed else 'unchanged'}")
        result["details"]["screen_changed"] = ctx.screen_changed

        # Determine success
        if process_exists and (window_match or ctx.screen_changed):
            result["success"] = True
            result["explanation"] = f"App '{app_name}' opened successfully. " + "; ".join(checks)
        elif process_exists:
            result["success"] = True
            result["explanation"] = f"Process '{app_name}' is running but window not detected. " + "; ".join(checks)
        else:
            result["success"] = False
            result["explanation"] = f"Failed to open '{app_name}'. " + "; ".join(checks)
            result["retry_needed"] = True
            result["recovery_action"] = f"Try opening {app_name} with different method"

        return result

    async def _verify_click(
        self, params: Dict[str, Any], ctx: PerceptionContext, expected_element: Optional[str]
    ) -> Dict[str, Any]:
        """Verify a click action."""
        label = params.get("label", "")
        result = {
            "success": False,
            "explanation": "",
            "retry_needed": False,
            "recovery_action": "",
            "details": {},
        }

        checks = []

        # Check 1: Screen changed
        checks.append(f"Screen: {'changed' if ctx.screen_changed else 'unchanged'}")
        result["details"]["screen_changed"] = ctx.screen_changed

        # Check 2: Window changed
        checks.append(f"Window: {'changed' if ctx.window_changed else 'unchanged'}")
        result["details"]["window_changed"] = ctx.window_changed

        # Check 3: Expected element appeared/disappeared
        if expected_element:
            element_found = expected_element.lower() in (ctx.ocr_text + ctx.a11y_tree_text).lower()
            checks.append(f"Element '{expected_element}': {'found' if element_found else 'not found'}")
            result["details"]["expected_element_found"] = element_found

        # Check 4: Dialog appeared
        if ctx.has_dialog:
            checks.append("Dialog appeared")
            result["details"]["dialog_appeared"] = True

        if ctx.screen_changed or ctx.window_changed or ctx.has_dialog:
            result["success"] = True
            result["explanation"] = f"Click '{label}' had effect. " + "; ".join(checks)
        else:
            result["success"] = False
            result["explanation"] = f"Click '{label}' had no visible effect. " + "; ".join(checks)
            result["retry_needed"] = True
            result["recovery_action"] = f"Retry clicking '{label}' or try nearby element"

        return result

    async def _verify_type(self, params: Dict[str, Any], ctx: PerceptionContext) -> Dict[str, Any]:
        """Verify a type action."""
        text = params.get("text", "")
        result = {
            "success": False,
            "explanation": "",
            "retry_needed": False,
            "recovery_action": "",
            "details": {},
        }

        # Check if typed text appears in OCR or a11y data
        combined = (ctx.ocr_text + " " + ctx.a11y_tree_text).lower()
        text_found = text.lower() in combined if text else False

        if text_found or ctx.screen_changed:
            result["success"] = True
            result["explanation"] = f"Text '{text[:30]}' typed successfully" if text_found else "Screen changed after typing"
        else:
            result["success"] = False
            result["explanation"] = f"Typed text '{text[:30]}' not found on screen"
            result["retry_needed"] = True
            result["recovery_action"] = "Click the target field first, then retry typing"

        result["details"]["text_found"] = text_found
        return result

    async def _verify_navigate(self, params: Dict[str, Any], ctx: PerceptionContext) -> Dict[str, Any]:
        """Verify a navigation action."""
        url = params.get("url", "")
        result = {
            "success": False,
            "explanation": "",
            "retry_needed": False,
            "recovery_action": "",
            "details": {},
        }

        if ctx.screen_changed or ctx.window_changed:
            result["success"] = True
            result["explanation"] = f"Navigation to '{url[:50]}' changed the screen"
        else:
            result["success"] = False
            result["explanation"] = f"Navigation to '{url[:50]}' had no visible effect"
            result["retry_needed"] = True
            result["recovery_action"] = "Retry navigation"

        return result

    async def _verify_scroll(self, ctx: PerceptionContext) -> Dict[str, Any]:
        """Verify a scroll action."""
        result = {
            "success": ctx.screen_changed,
            "explanation": "Screen changed after scrolling" if ctx.screen_changed else "No change after scrolling",
            "retry_needed": not ctx.screen_changed,
            "recovery_action": "Retry scroll" if not ctx.screen_changed else "",
            "details": {"screen_changed": ctx.screen_changed},
        }
        return result

    # ── UI Tree builders ────────────────────────────────────

    def _build_ui_tree_from_a11y(self, ctx: PerceptionContext) -> str:
        """Build UI tree text from accessibility data."""
        if not ctx.a11y_tree_text:
            return ""

        # The a11y tree text is already a compact representation
        lines = [f"Window: {ctx.window_title}"]
        if ctx.app_type:
            lines.append(f"Application: {ctx.app_name} ({ctx.app_type})")
        lines.append(f"Accessibility: {ctx.a11y_node_count} nodes, {ctx.a11y_clickable_count} clickable")
        lines.append(ctx.a11y_tree_text)
        return "\n".join(lines)

    def _build_minimal_ui_tree(self, ctx: PerceptionContext) -> str:
        """Build a minimal UI tree from window info only."""
        lines = [f"Window: {ctx.window_title}"]
        if ctx.app_type:
            lines.append(f"Application: {ctx.app_name} ({ctx.app_type})")
        if ctx.window_process:
            lines.append(f"Process: {ctx.window_process} (PID {ctx.window_pid})")
        return "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_phash(image) -> str:
        """Compute a simple perceptual hash of an image."""
        if image is None:
            return ""
        try:
            import hashlib
            import numpy as np
            if isinstance(image, np.ndarray):
                # Downsample to 8x8 grayscale
                if len(image.shape) == 3:
                    gray = np.mean(image, axis=2)
                else:
                    gray = image
                # Resize to 8x8
                h, w = gray.shape
                cell_h, cell_w = h // 8, w // 8
                if cell_h == 0 or cell_w == 0:
                    return hashlib.md5(image.tobytes()).hexdigest()[:16]
                small = np.zeros((8, 8))
                for i in range(8):
                    for j in range(8):
                        small[i, j] = np.mean(gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w])
                avg = np.mean(small)
                bits = (small > avg).flatten()
                hash_bytes = bytes(int(''.join(str(int(b)) for b in bits[i:i+8]), 2) for i in range(0, 64, 8))
                return hashlib.md5(hash_bytes).hexdigest()[:16]
            else:
                return hashlib.md5(image.tobytes()).hexdigest()[:16]
        except Exception:
            return ""

    @staticmethod
    def _get_window_id() -> str:
        """Get the X11 window ID of the active window."""
        import shutil
        if shutil.which("xdotool"):
            try:
                import subprocess
                result = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True, text=True, timeout=1
                )
                return result.stdout.strip()
            except Exception:
                pass
        return ""

    @staticmethod
    def _compute_perception_id(ctx: PerceptionContext) -> str:
        """Generate a unique ID for this perception."""
        import hashlib
        data = f"{ctx.window_id}:{ctx.screen_hash}:{ctx.timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    # ── Cache management ────────────────────────────────────

    def invalidate_cache(self) -> None:
        """Force the next perceive() to run the full pipeline."""
        self._last_ctx = None
        self._last_screen_hash = ""
        self._last_window_id = ""
        self.a11y.invalidate_cache()
        logger.info("[PERCEPTION] Cache invalidated")

    # ── Diagnostics ─────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Return comprehensive diagnostic report."""
        return {
            "perception_count": self._perception_count,
            "cache_hit_count": self._cache_hit_count,
            "cache_hit_ratio": f"{self._cache_hit_count / max(self._perception_count, 1):.1%}",
            "ocr_run_count": self._ocr_run_count,
            "ocr_skip_count": self._ocr_skip_count,
            "ocr_usage_rate": f"{self._ocr_run_count / max(self._ocr_run_count + self._ocr_skip_count, 1):.1%}",
            "a11y_success_count": self._a11y_success_count,
            "a11y_fail_count": self._a11y_fail_count,
            "a11y_usage_rate": f"{self._a11y_success_count / max(self._a11y_success_count + self._a11y_fail_count, 1):.1%}",
            "a11y_backends": self.a11y.available_backends if self.a11y._initialized else [],
            "last_window": self._last_ctx.window_title[:60] if self._last_ctx else "",
            "last_app_type": self._last_ctx.app_type if self._last_ctx else "",
            "last_page_type": self._last_ctx.page_type if self._last_ctx else "",
            "last_latency_ms": round(self._last_ctx.total_latency_ms, 1) if self._last_ctx else 0,
        }


# Global singleton
perception_pipeline = PerceptionPipeline()