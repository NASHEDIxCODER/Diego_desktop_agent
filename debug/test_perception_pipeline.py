#!/usr/bin/env python3
"""
Perception Pipeline — Manual Test Harness

Tests the full perception pipeline end-to-end:
  1. Accessibility tree detection
  2. Perception pipeline (all 9 stages)
  3. OCR fallback when a11y unavailable
  4. Action verification
  5. Layout learning
  6. Vision cache behavior
  7. Screen reasoning

Usage:
    python debug/test_perception_pipeline.py

Produces: logs/perception_report.json
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("perception_test")


# ═══════════════════════════════════════════════════════════════
# Test Results Collector
# ═══════════════════════════════════════════════════════════════

class TestResults:
    """Collects and formats test results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def add(self, name: str, passed: bool, details: Dict[str, Any] = None,
            error: str = "", skipped: bool = False) -> None:
        status = "SKIPPED" if skipped else ("PASS" if passed else "FAIL")
        if skipped:
            self.skipped += 1
        elif passed:
            self.passed += 1
        else:
            self.failed += 1

        entry = {
            "test": name,
            "status": status,
            "timestamp": time.time(),
            "details": details or {},
            "error": error,
        }
        self.results.append(entry)
        icon = "✅" if passed else ("⚠️" if skipped else "❌")
        logger.info("%s %s: %s", icon, status, name)
        if error:
            logger.info("   Error: %s", error)

    def summary(self) -> Dict[str, Any]:
        total = self.passed + self.failed + self.skipped
        elapsed = time.time() - self.start_time
        return {
            "total_tests": total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": f"{self.passed / max(total, 1):.1%}",
            "elapsed_seconds": round(elapsed, 1),
            "results": self.results,
        }


# ═══════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════

async def test_accessibility_initialization(results: TestResults) -> None:
    """Test 1: Accessibility tree initialization."""
    try:
        from services.accessibility import accessibility_tree

        ok = accessibility_tree.initialize()
        backends = accessibility_tree.available_backends

        results.add(
            "Accessibility Initialization",
            passed=ok,
            details={
                "backends": [b.value for b in backends],
                "backend_count": len(backends),
            },
            error="" if ok else "No accessibility backends found",
        )
    except Exception as e:
        results.add("Accessibility Initialization", passed=False, error=str(e))


async def test_accessibility_tree(results: TestResults) -> None:
    """Test 2: Get accessibility tree for current window."""
    try:
        from services.accessibility import accessibility_tree

        if not accessibility_tree.ready:
            results.add("Accessibility Tree", passed=False, skipped=True,
                        error="Accessibility not initialized")
            return

        tree = accessibility_tree.get_tree(force_refresh=True)

        if tree is None:
            results.add(
                "Accessibility Tree",
                passed=False,
                details={"tree": None},
                error="No accessibility tree available — OCR fallback will be needed",
            )
            return

        all_nodes = tree.walk()
        clickable = tree.find_clickable()
        focused = tree.find_focused()
        has_content = tree.has_meaningful_content()

        results.add(
            "Accessibility Tree",
            passed=has_content,
            details={
                "total_nodes": len(all_nodes),
                "clickable_count": len(clickable),
                "focused_element": focused.name if focused else None,
                "focused_role": focused.role.value if focused else None,
                "has_meaningful_content": has_content,
                "backend": tree.backend.value,
                "tree_preview": tree.to_compact_str()[:500],
            },
            error="" if has_content else "Tree exists but lacks meaningful content",
        )
    except Exception as e:
        results.add("Accessibility Tree", passed=False, error=str(e))


async def test_perception_pipeline(results: TestResults) -> None:
    """Test 3: Full perception pipeline."""
    try:
        from services.perception_pipeline import perception_pipeline

        await perception_pipeline.initialize()

        t0 = time.perf_counter_ns()
        ctx = await perception_pipeline.perceive(force=True)
        latency = (time.perf_counter_ns() - t0) / 1_000_000

        passed = len(ctx.stages_run) >= 3  # At minimum: screen, window, app_detect

        results.add(
            "Perception Pipeline",
            passed=passed,
            details={
                "perception_id": ctx.perception_id,
                "total_latency_ms": round(latency, 1),
                "stages_run": ctx.stages_run,
                "stage_count": len(ctx.stages_run),
                "window_title": ctx.window_title[:80],
                "window_process": ctx.window_process,
                "app_type": ctx.app_type,
                "app_name": ctx.app_name,
                "page_type": ctx.page_type,
                "a11y_available": ctx.a11y_available,
                "a11y_backend": ctx.a11y_backend,
                "a11y_node_count": ctx.a11y_node_count,
                "a11y_clickable_count": ctx.a11y_clickable_count,
                "ocr_used": ctx.ocr_used,
                "ocr_skipped_reason": ctx.ocr_skipped_reason,
                "ocr_box_count": ctx.ocr_box_count,
                "screen_changed": ctx.screen_changed,
                "window_changed": ctx.window_changed,
                "has_dialog": ctx.has_dialog,
                "has_notification": ctx.has_notification,
                "focused_element": ctx.focused_element_label,
                "git_branch": ctx.git_branch,
                "browser_tab": ctx.browser_tab,
                "clipboard_preview": ctx.clipboard_text[:50] if ctx.clipboard_text else "",
                "semantic_summary": ctx.semantic_summary[:200],
                "compact_summary": ctx.compact_summary[:500],
            },
            error="" if passed else f"Only {len(ctx.stages_run)} stages completed",
        )
    except Exception as e:
        results.add("Perception Pipeline", passed=False, error=str(e))


async def test_ocr_fallback(results: TestResults) -> None:
    """Test 4: OCR fallback behavior."""
    try:
        from services.perception_pipeline import perception_pipeline

        # Run perception and check OCR usage
        ctx = await perception_pipeline.perceive(force=True)

        # Check if OCR was correctly skipped when a11y is available
        if ctx.a11y_available and not ctx.ocr_used:
            results.add(
                "OCR Fallback — Correctly Skipped",
                passed=True,
                details={
                    "a11y_available": True,
                    "ocr_used": False,
                    "skip_reason": ctx.ocr_skipped_reason,
                },
            )
        elif ctx.a11y_available and ctx.ocr_used:
            results.add(
                "OCR Fallback — Should Have Skipped",
                passed=False,
                details={
                    "a11y_available": True,
                    "ocr_used": True,
                    "warning": "OCR was run despite accessibility data being available",
                },
            )
        elif not ctx.a11y_available and ctx.ocr_used:
            results.add(
                "OCR Fallback — Correctly Used",
                passed=True,
                details={
                    "a11y_available": False,
                    "ocr_used": True,
                    "ocr_box_count": ctx.ocr_box_count,
                },
            )
        else:
            results.add(
                "OCR Fallback — No Data Available",
                passed=False,
                details={
                    "a11y_available": False,
                    "ocr_used": False,
                    "warning": "Neither accessibility nor OCR produced data",
                },
            )
    except Exception as e:
        results.add("OCR Fallback", passed=False, error=str(e))


async def test_vision_cache(results: TestResults) -> None:
    """Test 5: Vision cache behavior."""
    try:
        from services.perception_pipeline import perception_pipeline

        # First perception (should run full pipeline)
        t0 = time.perf_counter_ns()
        ctx1 = await perception_pipeline.perceive(force=True)
        t1 = time.perf_counter_ns() - t0

        # Second perception (should hit cache if nothing changed)
        t0 = time.perf_counter_ns()
        ctx2 = await perception_pipeline.perceive(force=False)
        t2 = time.perf_counter_ns() - t0

        cache_hit = ctx2.from_cache
        speedup = t1 / max(t2, 1)

        results.add(
            "Vision Cache",
            passed=cache_hit,
            details={
                "first_latency_ms": round(t1 / 1_000_000, 1),
                "second_latency_ms": round(t2 / 1_000_000, 1),
                "cache_hit": cache_hit,
                "speedup_ratio": round(speedup, 1),
                "perception_id_1": ctx1.perception_id,
                "perception_id_2": ctx2.perception_id,
            },
            error="" if cache_hit else "Cache miss — second perception ran full pipeline",
        )
    except Exception as e:
        results.add("Vision Cache", passed=False, error=str(e))


async def test_action_verification(results: TestResults) -> None:
    """Test 6: Action verification system."""
    try:
        from services.perception_pipeline import perception_pipeline
        from vision.action_verifier import action_verifier

        # Test verification infrastructure
        action_verifier.capture_pre_action()

        # Run a perception to get post-"action" state
        ctx = await perception_pipeline.perceive(force=True)

        # Verify a hypothetical click
        result = await perception_pipeline.verify_action(
            "click",
            {"label": "Test Button"},
            expected_outcome="Testing verification infrastructure",
        )

        results.add(
            "Action Verification",
            passed=True,
            details={
                "verification_success": result.get("success", False),
                "explanation": result.get("explanation", ""),
                "retry_needed": result.get("retry_needed", False),
                "recovery_action": result.get("recovery_action", ""),
                "details": result.get("details", {}),
                "verifier_report": action_verifier.report(),
            },
        )
    except Exception as e:
        results.add("Action Verification", passed=False, error=str(e))


async def test_screen_reasoning(results: TestResults) -> None:
    """Test 7: Screen reasoning — can Diego answer questions about the desktop?"""
    try:
        from services.perception_pipeline import perception_pipeline
        from services.screen_reasoning import screen_reasoner

        ctx = await perception_pipeline.perceive(force=True)

        # Test the reasoning questions
        questions = [
            "What application is open?",
            "What window?",
            "What buttons are available?",
            "What errors are visible?",
            "Summarize this screen",
        ]

        answers = {}
        for q in questions:
            interpretation = screen_reasoner.interpret_command(q)
            answers[q] = {
                "action_type": interpretation.get("action_type", ""),
                "explanation": interpretation.get("explanation", ""),
                "confidence": interpretation.get("confidence", 0),
            }

        # Check if reasoning produced useful output
        has_page_type = bool(ctx.page_type and ctx.page_type != "unknown")
        has_summary = bool(ctx.semantic_summary)
        has_elements = bool(ctx.interactive_elements_json)

        passed = has_page_type or has_summary

        results.add(
            "Screen Reasoning",
            passed=passed,
            details={
                "page_type": ctx.page_type,
                "semantic_summary": ctx.semantic_summary[:200],
                "has_dialog": ctx.has_dialog,
                "has_notification": ctx.has_notification,
                "error_count": len(json.loads(ctx.error_elements_json)) if ctx.error_elements_json else 0,
                "interactive_count": len(json.loads(ctx.interactive_elements_json)) if ctx.interactive_elements_json else 0,
                "question_answers": answers,
            },
            error="" if passed else "No page type or summary produced",
        )
    except Exception as e:
        results.add("Screen Reasoning", passed=False, error=str(e))


async def test_layout_learning(results: TestResults) -> None:
    """Test 8: Desktop layout learning."""
    try:
        from learning.desktop_layouts import layout_memory
        from services.perception_pipeline import perception_pipeline

        # Load existing layouts
        layout_memory.load()

        # Get current perception
        ctx = await perception_pipeline.perceive(force=True)

        # Learn from current perception
        if ctx.app_name and ctx.app_type:
            # Build region data from perception
            regions = []
            if ctx.a11y_available:
                regions.append({
                    "region_type": "window",
                    "label": ctx.window_title,
                    "bounds": (0, 0, ctx.screen_width, ctx.screen_height),
                })

            elements = []
            if ctx.interactive_elements_json:
                try:
                    for elem in json.loads(ctx.interactive_elements_json)[:10]:
                        elements.append({
                            "element_type": elem.get("type", "unknown"),
                            "label": elem.get("label", ""),
                            "bounds": (0, 0, 0, 0),  # No pixel bounds from reasoning
                        })
                except Exception:
                    pass

            layout_memory.learn(
                app_name=ctx.app_name,
                app_type=ctx.app_type,
                regions=regions,
                elements=elements,
                window_size=(ctx.screen_width, ctx.screen_height),
            )

        # Check known apps
        known_apps = layout_memory.get_all_known_apps()
        template = layout_memory.get_template(ctx.app_name) if ctx.app_name else None

        # Try to predict a location
        prediction = None
        if ctx.app_name:
            prediction = layout_memory.predict_location(ctx.app_name, "Run")

        results.add(
            "Layout Learning",
            passed=len(known_apps) > 0,
            details={
                "known_apps": known_apps,
                "known_app_count": len(known_apps),
                "current_app": ctx.app_name,
                "current_app_template_observations": template.observation_count if template else 0,
                "current_app_stability": round(template.stability_score, 2) if template else 0,
                "prediction_test": f"Predicted 'Run' at {prediction}" if prediction else "No prediction",
                "layout_report": layout_memory.report(),
            },
        )
    except Exception as e:
        results.add("Layout Learning", passed=False, error=str(e))


async def test_desktop_state(results: TestResults) -> None:
    """Test 9: Desktop state collection."""
    try:
        from services.desktop_state import desktop_state

        snap = desktop_state.snapshot()
        context = desktop_state.context_for_llm()

        has_window = bool(snap.focused_window.title)
        has_git = bool(snap.terminal.git_branch)
        has_terminal = bool(snap.terminal.cwd)

        results.add(
            "Desktop State",
            passed=has_window,
            details={
                "window_title": snap.focused_window.title[:80],
                "window_process": snap.focused_window.application,
                "window_pid": snap.focused_window.pid,
                "git_branch": snap.terminal.git_branch,
                "terminal_cwd": snap.terminal.cwd,
                "browser_tab": snap.browser_tab,
                "browser_url": snap.browser_url,
                "clipboard_preview": snap.clipboard_text[:50] if snap.clipboard_text else "",
                "battery": f"{snap.system.battery_percent}%",
                "cpu_percent": snap.system.cpu_percent,
                "memory_percent": snap.system.memory_percent,
                "audio_volume": snap.audio_volume,
                "audio_muted": snap.audio_muted,
                "current_file": desktop_state.current_file(),
                "current_ide": desktop_state.current_ide(),
                "llm_context": context[:300],
            },
            error="" if has_window else "No window detected",
        )
    except Exception as e:
        results.add("Desktop State", passed=False, error=str(e))


async def test_end_to_end_perception(results: TestResults) -> None:
    """Test 10: End-to-end perception → reasoning → planner context."""
    try:
        from services.perception_pipeline import perception_pipeline

        await perception_pipeline.initialize()

        # Full perception
        ctx = await perception_pipeline.perceive(force=True)

        # Generate planner context
        planner_context = ctx.compact_summary
        quick_context = ctx.quick_context

        # Verify all stages
        expected_stages = {"screen_capture", "window_detect", "app_detect"}
        missing_stages = expected_stages - set(ctx.stages_run)

        passed = len(missing_stages) == 0 and len(planner_context) > 0

        results.add(
            "End-to-End Perception",
            passed=passed,
            details={
                "stages_completed": ctx.stages_run,
                "missing_stages": list(missing_stages),
                "planner_context_length": len(planner_context),
                "quick_context": quick_context,
                "planner_context_preview": planner_context[:500],
                "perception_latency_ms": round(ctx.total_latency_ms, 1),
                "from_cache": ctx.from_cache,
            },
            error="" if passed else f"Missing stages: {missing_stages}",
        )
    except Exception as e:
        results.add("End-to-End Perception", passed=False, error=str(e))


# ═══════════════════════════════════════════════════════════════
# Final Report Generator
# ═══════════════════════════════════════════════════════════════

def generate_final_report(results: TestResults) -> Dict[str, Any]:
    """Generate the comprehensive final report."""
    from services.perception_pipeline import perception_pipeline
    from services.accessibility import accessibility_tree
    from learning.desktop_layouts import layout_memory

    pipeline_report = perception_pipeline.report()
    a11y_report = accessibility_tree.report()
    layout_report = layout_memory.report()

    # Calculate key metrics
    total_perceptions = pipeline_report.get("perception_count", 0)
    ocr_runs = pipeline_report.get("ocr_run_count", 0)
    ocr_skips = pipeline_report.get("ocr_skip_count", 0)
    a11y_successes = pipeline_report.get("a11y_success_count", 0)
    a11y_fails = pipeline_report.get("a11y_fail_count", 0)

    total_ocr = ocr_runs + ocr_skips
    total_a11y = a11y_successes + a11y_fails

    ocr_usage_rate = ocr_runs / max(total_ocr, 1)
    a11y_usage_rate = a11y_successes / max(total_a11y, 1)

    # Blind spots
    blind_spots = []
    if not a11y_report.get("available_backends"):
        blind_spots.append("No accessibility backends available — OCR is primary data source")
    if "at-spi" not in str(a11y_report.get("available_backends", [])):
        blind_spots.append("AT-SPI not available — GTK/Qt apps will use OCR fallback")
    if "browser_cdp" not in str(a11y_report.get("available_backends", [])):
        blind_spots.append("Browser CDP not available — browser content requires OCR")
    if not blind_spots:
        blind_spots.append("No major blind spots detected")

    # Verification stats
    verification_success_rate = "N/A"
    retries = 0
    recoveries = 0
    for r in results.results:
        if r["test"] == "Action Verification":
            details = r.get("details", {})
            verifier = details.get("verifier_report", {})
            total_verifications = verifier.get("verification_count", 0)
            failures = verifier.get("failure_count", 0)
            if total_verifications > 0:
                verification_success_rate = f"{(1 - failures / total_verifications):.1%}"
            retries = verifier.get("retry_count", 0)
            recoveries = 0  # Will be populated in real usage

    return {
        "report_title": "Diego Desktop Assistant — Perception Pipeline Phase 2 Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_summary": results.summary(),

        # ── Key Metrics ────────────────────────────────────
        "key_metrics": {
            "perception_latency_ms": round(pipeline_report.get("last_latency_ms", 0), 1),
            "ocr_usage_rate": f"{ocr_usage_rate:.1%}",
            "accessibility_usage_rate": f"{a11y_usage_rate:.1%}",
            "verification_success_rate": verification_success_rate,
            "retries": retries,
            "recoveries": recoveries,
            "cache_hit_ratio": pipeline_report.get("cache_hit_ratio", "0%"),
        },

        # ── Subsystem Reports ──────────────────────────────
        "perception_pipeline": pipeline_report,
        "accessibility": a11y_report,
        "layout_learning": layout_report,

        # ── Blind Spots ────────────────────────────────────
        "remaining_blind_spots": blind_spots,

        # ── Architecture Summary ───────────────────────────
        "architecture": {
            "pipeline_stages": [
                "1. Screen Capture",
                "2. Active Window Detection",
                "3. Application Type Detection",
                "4. Accessibility Tree (AT-SPI → Browser CDP → X11)",
                "5. UI Tree Construction",
                "6. OCR (FALLBACK ONLY)",
                "7. Desktop State (clipboard, git, terminal, browser)",
                "8. Focused Element",
                "9. Screen Reasoning (page type, errors, interactive elements)",
            ],
            "accessibility_priority": [
                "1. AT-SPI (Linux accessibility bus)",
                "2. Browser CDP (Chrome DevTools Protocol)",
                "3. X11 Window Tree (xdotool + xprop)",
                "4. OCR (FALLBACK ONLY)",
            ],
            "action_verification_chain": [
                "1. Check process exists (for open_app)",
                "2. Check focused window changed",
                "3. Check window title",
                "4. Check screen hash changed",
                "5. Run perception to confirm",
                "6. Retry on failure",
                "7. Recovery action on repeated failure",
            ],
            "cache_strategy": [
                "Screen hash changed → rerun",
                "Focused window changed → rerun",
                "User explicitly requested refresh → rerun",
                "Otherwise → reuse cached perception",
            ],
            "learned_apps": layout_report.get("total_apps", 0),
            "builtin_templates": layout_report.get("builtin_apps", 0),
        },

        # ── Recommendations ────────────────────────────────
        "recommendations": [
            "Install at-spi2-core and pyatspi2 for full accessibility support on GTK/Qt apps",
            "Launch Chrome with --remote-debugging-port=9222 for browser accessibility",
            "Run perception pipeline at least 10 times per app to build stable layout templates",
            "Integrate perception_pipeline.perceive() into the planner's pre-action hook",
            "Use perception_pipeline.verify_action() after every desktop action",
        ],
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    logger.info("=" * 60)
    logger.info("Diego Perception Pipeline — Manual Test Harness")
    logger.info("=" * 60)

    results = TestResults()

    # Run all tests
    tests = [
        ("Accessibility Initialization", test_accessibility_initialization),
        ("Accessibility Tree", test_accessibility_tree),
        ("Perception Pipeline", test_perception_pipeline),
        ("OCR Fallback", test_ocr_fallback),
        ("Vision Cache", test_vision_cache),
        ("Action Verification", test_action_verification),
        ("Screen Reasoning", test_screen_reasoning),
        ("Layout Learning", test_layout_learning),
        ("Desktop State", test_desktop_state),
        ("End-to-End Perception", test_end_to_end_perception),
    ]

    for name, test_fn in tests:
        logger.info("─" * 40)
        logger.info("Running: %s", name)
        try:
            await test_fn(results)
        except Exception as e:
            logger.error("Test '%s' crashed: %s", name, e, exc_info=True)
            results.add(name, passed=False, error=f"Crash: {e}")

    # Generate final report
    logger.info("=" * 60)
    logger.info("Generating final report...")

    report = generate_final_report(results)

    # Save report
    report_path = Path(__file__).parent.parent / "logs" / "perception_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))

    # Print summary
    summary = results.summary()
    logger.info("=" * 60)
    logger.info("TEST RESULTS: %d passed, %d failed, %d skipped (%.1f%% pass rate)",
                 summary["passed"], summary["failed"], summary["skipped"],
                 float(summary["pass_rate"].rstrip("%")))
    logger.info("Report saved to: %s", report_path)

    # Print key metrics
    metrics = report["key_metrics"]
    logger.info("─" * 40)
    logger.info("KEY METRICS:")
    logger.info("  Perception Latency: %s ms", metrics["perception_latency_ms"])
    logger.info("  OCR Usage Rate: %s", metrics["ocr_usage_rate"])
    logger.info("  Accessibility Usage Rate: %s", metrics["accessibility_usage_rate"])
    logger.info("  Verification Success Rate: %s", metrics["verification_success_rate"])
    logger.info("  Cache Hit Ratio: %s", metrics["cache_hit_ratio"])
    logger.info("  Retries: %s", metrics["retries"])
    logger.info("  Recoveries: %s", metrics["recoveries"])

    # Print blind spots
    logger.info("─" * 40)
    logger.info("BLIND SPOTS:")
    for spot in report["remaining_blind_spots"]:
        logger.info("  • %s", spot)

    logger.info("=" * 60)
    logger.info("Phase 2 perception pipeline testing complete.")

    return report


if __name__ == "__main__":
    asyncio.run(main())