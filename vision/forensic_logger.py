"""
ForensicLogger — Per-stage structured logging for the vision pipeline.

Every stage logs: INPUT, OUTPUT, LATENCY, CONFIDENCE, FAILURE REASON.
Never silently fails — every failure explains exactly why.

Used by vision_service.analyze() to produce a complete audit trail.

Also provides `inspect_screen()` — a comprehensive text report of the
current screen state for the `Diego inspect screen` command.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StageLog:
    """A single stage's forensic log entry."""
    stage: int = 0
    name: str = ""
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: float = 0.0
    confidence: float = 0.0
    success: bool = True
    failure_reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForensicReport:
    """Complete forensic audit of one vision pipeline run."""
    timestamp: float = 0.0
    total_latency_ms: float = 0.0
    stages: List[StageLog] = field(default_factory=list)
    overall_success: bool = True
    error_count: int = 0
    warning_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "overall_success": self.overall_success,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "stages": [
                {
                    "stage": s.stage,
                    "name": s.name,
                    "input": s.input_summary,
                    "output": s.output_summary,
                    "latency_ms": round(s.latency_ms, 1),
                    "confidence": round(s.confidence, 3),
                    "success": s.success,
                    "failure_reason": s.failure_reason,
                    "details": s.details,
                }
                for s in self.stages
            ],
        }

    def to_text(self) -> str:
        """Human-readable forensic report."""
        lines = [
            "=" * 72,
            "  DIEGO VISION FORENSIC AUDIT",
            "=" * 72,
            f"  Total latency: {self.total_latency_ms:.1f}ms",
            f"  Overall: {'PASS' if self.overall_success else 'FAIL'}",
            f"  Errors: {self.error_count}  Warnings: {self.warning_count}",
            "",
        ]
        for s in self.stages:
            status = "✓" if s.success else "✗ FAIL"
            lines.append(f"  Stage {s.stage}: {s.name}  [{status}]  {s.latency_ms:.1f}ms")
            lines.append(f"    INPUT:  {s.input_summary}")
            lines.append(f"    OUTPUT: {s.output_summary}")
            if s.confidence > 0:
                lines.append(f"    CONFIDENCE: {s.confidence:.3f}")
            if s.failure_reason:
                lines.append(f"    FAILURE REASON: {s.failure_reason}")
            if s.details:
                for k, v in s.details.items():
                    lines.append(f"    {k}: {v}")
            lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


class ForensicLogger:
    """
    Per-stage forensic logging for the vision pipeline.

    Usage:
        fl = ForensicLogger()
        fl.start()

        with fl.stage(1, "Capture"):
            # ... do capture ...
            fl.log_output("1920x1080 fullscreen", confidence=1.0)

        with fl.stage(2, "Frame Hash"):
            # ... compute hash ...
            fl.log_output("a3f2b1c0", confidence=1.0)

        report = fl.finish()
        print(report.to_text())
    """

    def __init__(self):
        self._report = ForensicReport()
        self._current_stage: Optional[StageLog] = None
        self._stage_start: float = 0.0

    def start(self) -> None:
        """Begin a new forensic audit."""
        self._report = ForensicReport(timestamp=time.time())
        self._current_stage = None

    def begin_stage(self, stage_num: int, name: str, input_summary: str = "") -> None:
        """Begin logging a new stage."""
        self._current_stage = StageLog(
            stage=stage_num,
            name=name,
            input_summary=input_summary,
        )
        self._stage_start = time.perf_counter_ns()

    def log_output(self, output_summary: str, confidence: float = 0.0,
                   success: bool = True, failure_reason: str = "",
                   **details: Any) -> None:
        """Log the output of the current stage."""
        if self._current_stage is None:
            return
        self._current_stage.output_summary = output_summary
        self._current_stage.confidence = confidence
        self._current_stage.success = success
        self._current_stage.failure_reason = failure_reason
        self._current_stage.latency_ms = (
            time.perf_counter_ns() - self._stage_start
        ) / 1_000_000
        self._current_stage.details = details

        if not success:
            self._report.error_count += 1
            logger.error("[FORENSIC] Stage %d %s FAILED: %s",
                         self._current_stage.stage,
                         self._current_stage.name,
                         failure_reason)
        elif failure_reason:
            self._report.warning_count += 1
            logger.warning("[FORENSIC] Stage %d %s WARNING: %s",
                           self._current_stage.stage,
                           self._current_stage.name,
                           failure_reason)

        self._report.stages.append(self._current_stage)
        self._current_stage = None

    def log_failure(self, failure_reason: str, **details: Any) -> None:
        """Log the current stage as failed."""
        self.log_output(
            output_summary="FAILED",
            confidence=0.0,
            success=False,
            failure_reason=failure_reason,
            **details,
        )

    def finish(self) -> ForensicReport:
        """Finalize and return the forensic report."""
        self._report.total_latency_ms = sum(
            s.latency_ms for s in self._report.stages
        )
        self._report.overall_success = all(
            s.success for s in self._report.stages
        )
        return self._report

    # ── Context manager support ────────────────────────────

    class StageContext:
        """Context manager for a single stage."""
        def __init__(self, logger: "ForensicLogger", stage_num: int,
                     name: str, input_summary: str = ""):
            self._logger = logger
            self._stage_num = stage_num
            self._name = name
            self._input = input_summary

        def __enter__(self):
            self._logger.begin_stage(self._stage_num, self._name, self._input)
            return self._logger

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self._logger.log_failure(
                    f"{exc_type.__name__}: {exc_val}",
                    exception=str(exc_val),
                )
            elif self._logger._current_stage is not None:
                # Stage didn't call log_output — auto-finalize
                self._logger.log_output(
                    output_summary="completed (no explicit output)",
                    confidence=0.5,
                )
            return False  # Don't suppress exceptions

    def stage(self, stage_num: int, name: str,
              input_summary: str = "") -> StageContext:
        """Return a context manager for a pipeline stage."""
        return self.StageContext(self, stage_num, name, input_summary)


# ── Screen inspection (for `Diego inspect screen` command) ──────

def inspect_screen(ctx: Any) -> str:
    """
    Produce a comprehensive text report of the current screen state.

    Args:
        ctx: VisionContext from vision_service.analyze(force=True)

    Returns:
        Multi-line human-readable inspection report.
    """
    lines = [
        "=" * 60,
        "  DIEGO SCREEN INSPECTION",
        "=" * 60,
        "",
    ]

    # Application
    lines.append("── Application ──")
    lines.append(f"  Window title:  {ctx.active_window_title or '(unknown)'}")
    lines.append(f"  App type:      {ctx.app_type or '(unknown)'}")
    lines.append(f"  App name:      {ctx.app_name or '(unknown)'}")
    lines.append(f"  Page type:     {ctx.page_type or '(unknown)'}")
    lines.append(f"  Resolution:    {ctx.capture.width}x{ctx.capture.height}" if ctx.capture else "  Resolution:    (no capture)")
    lines.append(f"  Frame hash:    {ctx.frame_hash[:16] if ctx.frame_hash else '(none)'}")
    lines.append("")

    # Window
    lines.append("── Window ──")
    if ctx.capture:
        lines.append(f"  Capture size:  {ctx.capture.width}x{ctx.capture.height}")
        lines.append(f"  Source:        {ctx.capture.source_label}")
        lines.append(f"  Timestamp:     {ctx.capture.timestamp:.3f}")
    else:
        lines.append("  (no capture data)")
    lines.append("")

    # Detected controls
    lines.append("── Detected Controls ──")
    if ctx.desktop:
        all_elements = ctx.desktop.walk()
        by_type: Dict[str, List[str]] = {}
        for el in all_elements:
            if el.label:
                by_type.setdefault(el.element_type.value, []).append(el.label)

        for etype in sorted(by_type.keys()):
            labels = by_type[etype][:15]
            lines.append(f"  {etype}: {len(by_type[etype])} total")
            for lbl in labels:
                lines.append(f"    - {lbl}")
        lines.append(f"  Total elements: {len(all_elements)}")
    else:
        lines.append("  (no UI tree)")
    lines.append("")

    # Buttons
    lines.append("── Buttons ──")
    if ctx.desktop:
        buttons = ctx.desktop.find_by_type(
            type(ctx.desktop.root).__annotations__.get("element_type", str)  # fallback
        )
        # Use walk + filter instead
        all_el = ctx.desktop.walk()
        buttons = [e for e in all_el if e.element_type.value == "button"]
        if buttons:
            for btn in buttons[:20]:
                bbox = btn.bounding_box
                pos = f"@({bbox[0]},{bbox[1]},{bbox[2]}x{bbox[3]})" if bbox else ""
                conf = f" [{btn.confidence:.0%}]" if btn.confidence > 0 else ""
                enabled = "" if btn.enabled else " [DISABLED]"
                lines.append(f"  {btn.label}{pos}{conf}{enabled}")
        else:
            lines.append("  (no buttons detected)")
    else:
        lines.append("  (no UI tree)")
    lines.append("")

    # Inputs
    lines.append("── Inputs ──")
    if ctx.desktop:
        all_el = ctx.desktop.walk()
        inputs = [e for e in all_el if e.element_type.value in ("input", "textbox")]
        if inputs:
            for inp in inputs[:10]:
                bbox = inp.bounding_box
                pos = f"@({bbox[0]},{bbox[1]},{bbox[2]}x{bbox[3]})" if bbox else ""
                lines.append(f"  {inp.label or '(unnamed)'}{pos}")
        else:
            lines.append("  (no inputs detected)")
    else:
        lines.append("  (no UI tree)")
    lines.append("")

    # Menus
    lines.append("── Menus ──")
    if ctx.desktop:
        all_el = ctx.desktop.walk()
        menus = [e for e in all_el if e.element_type.value in ("menu", "menu_item", "menu_bar")]
        if menus:
            for m in menus[:10]:
                lines.append(f"  {m.label}")
        else:
            lines.append("  (no menus detected)")
    else:
        lines.append("  (no UI tree)")
    lines.append("")

    # Dialogs
    lines.append("── Dialogs ──")
    if ctx.desktop:
        all_el = ctx.desktop.walk()
        dialogs = [e for e in all_el if e.element_type.value in ("dialog", "popup")]
        if dialogs:
            for d in dialogs:
                lines.append(f"  {d.label}")
        else:
            lines.append("  (no dialogs detected)")
    else:
        lines.append("  (no UI tree)")
    lines.append("")

    # Notifications
    lines.append("── Notifications ──")
    if ctx.desktop:
        all_el = ctx.desktop.walk()
        notifs = [e for e in all_el if e.element_type.value == "notification"]
        if notifs:
            for n in notifs:
                lines.append(f"  {n.label}")
        else:
            lines.append("  (no notifications detected)")
    else:
        lines.append("  (no UI tree)")
    lines.append("")

    # OCR
    lines.append("── OCR ──")
    if ctx.ocr_result:
        lines.append(f"  Backend:       {ctx.ocr_result.backend}")
        lines.append(f"  Boxes raw:     {ctx.ocr_result.box_count_raw}")
        lines.append(f"  Boxes final:   {ctx.ocr_result.box_count_final}")
        lines.append(f"  Merged:        {ctx.ocr_result.merged_count}")
        lines.append(f"  Deduped:       {ctx.ocr_result.dedup_count}")
        lines.append(f"  Avg confidence: {ctx.ocr_result.avg_confidence:.3f}")
        lines.append(f"  OCR time:      {ctx.ocr_result.elapsed_total_ms:.1f}ms")
        if ctx.ocr_result.error:
            lines.append(f"  ERROR:         {ctx.ocr_result.error}")
        lines.append(f"  Text preview:  {ctx.ocr_text[:300]}")
    else:
        lines.append("  (no OCR result)")
    lines.append("")

    # Planner target
    lines.append("── Planner Target ──")
    if ctx.semantic_summary:
        lines.append(f"  Summary: {ctx.semantic_summary}")
    else:
        lines.append("  (no semantic summary)")
    lines.append("")

    # Layout
    lines.append("── Layout Regions ──")
    if ctx.layout:
        for region in ctx.layout.regions:
            b = region.bounds
            lines.append(f"  {region.region_type.value}: {region.label} "
                         f"@({b[0]},{b[1]},{b[2]}x{b[3]}) "
                         f"[{region.confidence:.0%}]")
    else:
        lines.append("  (no layout)")
    lines.append("")

    # Pipeline stats
    lines.append("── Pipeline Stats ──")
    lines.append(f"  Total time:    {ctx.total_time_ms:.1f}ms")
    lines.append(f"  Stages run:    {', '.join(ctx.stages_run) if ctx.stages_run else 'none'}")
    lines.append(f"  From cache:    {ctx.from_cache}")
    if ctx.error:
        lines.append(f"  ERROR:         {ctx.error}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# Global singleton
forensic_logger = ForensicLogger()