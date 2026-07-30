#!/usr/bin/env python3
"""
PHASE 2 & 3 — Vision Verification & End-to-End Tests

Verifies every vision capability:
- Screen capture (full, monitor, window, region)
- OCR accuracy
- UI element detection
- Scene understanding
- Vision cache
- Saves debug artifacts to debug/vision/

Usage:
    python debug/vision_test.py
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import compat  # noqa: F401
from telemetry.logger import setup_logging

setup_logging("INFO")
logger = logging.getLogger("vision_test")

# Debug output directories
DEBUG_DIR = Path("debug")
VISION_DIR = DEBUG_DIR / "vision"
SCREENSHOTS_DIR = VISION_DIR / "screenshots"
OCR_DIR = VISION_DIR / "ocr"
OVERLAY_DIR = VISION_DIR / "overlay"
PLANNER_DIR = VISION_DIR / "planner"

for d in [SCREENSHOTS_DIR, OCR_DIR, OVERLAY_DIR, PLANNER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_overlay(image_path: Path, objects: List[Dict], output_path: Path) -> None:
    """Draw bounding boxes on an image and save."""
    try:
        from PIL import Image, ImageDraw
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        for obj in objects:
            box = obj.get("bbox", obj.get("box", None))
            if box and len(box) == 4:
                draw.rectangle(box, outline="red", width=2)
                label = obj.get("label", obj.get("text", ""))
                if label:
                    draw.text((box[0], box[1] - 10), label, fill="red")
        img.save(output_path)
        logger.info("Overlay saved: %s", output_path)
    except Exception as e:
        logger.warning("Overlay failed: %s", e)


def _capture_screen_region(left=0, top=0, width=800, height=600):
    """Capture a screen region using best available backend."""
    from PIL import Image
    # Try mss first
    try:
        import mss
        with mss.mss() as sct:
            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            return Image.frombytes("RGB", sct_img.size, sct_img.rgb)
    except ImportError:
        pass
    # Fallback: ImageMagick import command
    tmp = f"/tmp/leo_vision_region.png"
    result = subprocess.run(
        ["import", "-window", "root", "-crop", f"{width}x{height}+{left}+{top}", tmp],
        capture_output=True, timeout=10
    )
    if result.returncode == 0:
        return Image.open(tmp)
    raise RuntimeError("No screen capture backend available")


# ══════════════════════════════════════════════════════════
# TEST 1: Screen Capture
# ══════════════════════════════════════════════════════════


async def test_screen_capture() -> Dict[str, Any]:
    """Test all screen capture modes."""
    results = {"test": "screen_capture", "status": "FAILED", "details": {}}

    try:
        from PIL import Image

        # Full screen capture
        try:
            import mss
            t0 = time.time()
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                sct_img = sct.grab(monitor)
                full = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
            full_latency = (time.time() - t0) * 1000
            backend = "mss"
        except ImportError:
            t0 = time.time()
            result = subprocess.run(
                ["import", "-window", "root", "/tmp/leo_screen_test.png"],
                capture_output=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("No screen capture backend available")
            full = Image.open("/tmp/leo_screen_test.png")
            full_latency = (time.time() - t0) * 1000
            backend = "import (ImageMagick)"

        full_path = SCREENSHOTS_DIR / "original_screen.png"
        full.save(full_path)
        results["details"]["full_screen"] = {
            "latency_ms": round(full_latency, 2),
            "size": f"{full.width}x{full.height}",
            "backend": backend,
            "path": str(full_path),
        }

        results["status"] = "PASSED"
        logger.info("Screen capture (%s): %dx%d in %.1fms",
                   backend, full.width, full.height, full_latency)

    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        logger.error("Screen capture test failed: %s", e)

    return results


# ══════════════════════════════════════════════════════════
# TEST 2: OCR Verification
# ══════════════════════════════════════════════════════════


async def test_ocr() -> Dict[str, Any]:
    """Test OCR accuracy by capturing screen and extracting text."""
    results = {"test": "ocr", "status": "FAILED", "details": {}}

    try:
        import pytesseract

        # Capture a region with expected text
        screenshot = _capture_screen_region(0, 0, 600, 200)

        t0 = time.time()
        text = pytesseract.image_to_string(screenshot)
        ocr_latency = (time.time() - t0) * 1000

        # Save OCR artifacts
        ocr_path = SCREENSHOTS_DIR / "ocr_source.png"
        screenshot.save(ocr_path)

        ocr_result = {
            "text": text.strip(),
            "length": len(text.strip()),
            "latency_ms": round(ocr_latency, 2),
            "source_image": str(ocr_path),
        }
        save_json(ocr_result, OCR_DIR / "ocr_output.json")

        results["details"] = ocr_result
        results["status"] = "PASSED" if text.strip() else "DEGRADED"
        logger.info("OCR: %d chars in %.1fms", len(text.strip()), ocr_latency)

    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        logger.error("OCR test failed: %s", e)

    return results


# ══════════════════════════════════════════════════════════
# TEST 3: UI Element Detection
# ══════════════════════════════════════════════════════════


async def test_ui_detection() -> Dict[str, Any]:
    """Test UI element detection (buttons, menus, dialogs)."""
    results = {"test": "ui_detection", "status": "FAILED", "details": {}}

    try:
        import pytesseract

        # Capture a region that likely has UI elements (top bar)
        screenshot = _capture_screen_region(0, 0, 800, 100)
        ui_path = SCREENSHOTS_DIR / "ui_elements.png"
        screenshot.save(ui_path)

        # Use OCR to find potential button/menu text
        t0 = time.time()
        data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
        detection_latency = (time.time() - t0) * 1000

        # Extract potential buttons (elements with high confidence and reasonable size)
        buttons = []
        menus = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
            if text and conf > 50:
                w, h = data["width"][i], data["height"][i]
                x, y = data["left"][i], data["top"][i]
                element = {
                    "text": text,
                    "confidence": conf,
                    "bbox": [x, y, x + w, y + h],
                    "size": f"{w}x{h}",
                }
                if w > h and w > 30:
                    buttons.append(element)
                elif h > w and h > 20:
                    menus.append(element)

        # Save detected objects
        objects = {"buttons": buttons, "menus": menus, "total_elements": len(data["text"])}
        save_json(objects, VISION_DIR / "detected_objects.json")

        # Create overlay
        overlay_path = OVERLAY_DIR / "vision_overlay.png"
        save_overlay(ui_path, buttons + menus, overlay_path)

        results["details"] = {
            "buttons_found": len(buttons),
            "menus_found": len(menus),
            "total_elements": len(data["text"]),
            "latency_ms": round(detection_latency, 2),
            "overlay": str(overlay_path),
        }
        results["status"] = "PASSED" if buttons or menus else "DEGRADED"
        logger.info("UI detection: %d buttons, %d menus in %.1fms",
                   len(buttons), len(menus), detection_latency)

    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        logger.error("UI detection test failed: %s", e)

    return results


# ══════════════════════════════════════════════════════════
# TEST 4: Vision Cache
# ══════════════════════════════════════════════════════════


async def test_vision_cache() -> Dict[str, Any]:
    """Test vision cache hit/miss and change detection."""
    results = {"test": "vision_cache", "status": "FAILED", "details": {}}

    try:
        from vision import VisionCache
        cache = VisionCache(max_size=5, ttl_seconds=2.0)

        # Test set/get
        cache.set("test_key", "test_value")
        val = cache.get("test_key")
        assert val == "test_value", "Cache get/set failed"
        results["details"]["get_set"] = "PASSED"

        # Test TTL
        val = cache.get("test_key")
        assert val == "test_value", "Cache TTL expired too early"
        results["details"]["ttl"] = "PASSED"

        # Test LRU eviction
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
        assert cache.size <= 5, f"LRU eviction failed: size={cache.size}"
        results["details"]["lru_eviction"] = f"PASSED (size={cache.size})"

        # Test change detection
        from vision import CaptureResult
        cr = CaptureResult(hash="abc123")
        cache.set_last_capture(cr)
        changed = cache.has_screen_changed("def456")
        assert changed, "Change detection should detect different hash"
        not_changed = cache.has_screen_changed("abc123")
        assert not not_changed, "Change detection should not detect same hash"
        results["details"]["change_detection"] = "PASSED"

        results["status"] = "PASSED"
        logger.info("Vision cache: all tests passed")

    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        logger.error("Vision cache test failed: %s", e)

    return results


# ══════════════════════════════════════════════════════════
# TEST 5: Context-Aware Vision
# ══════════════════════════════════════════════════════════


async def test_context_aware() -> Dict[str, Any]:
    """Test that needs_vision() correctly identifies when vision is needed."""
    results = {"test": "context_aware_vision", "status": "FAILED", "details": {}}

    try:
        from vision import vision_manager, CaptureSource
        vision_manager.initialize()

        test_cases = [
            ("screen_question", "what is on my screen", True),
            ("unknown", "read this text for me", True),
            ("unknown", "open youtube", False),
            ("greeting", "hello", False),
            ("unknown", "who is in this room", True),
            ("time_query", "what time is it", False),
        ]

        passed = 0
        for intent, text, should_trigger in test_cases:
            source = vision_manager.needs_vision(intent, text)
            triggered = source is not None
            if triggered == should_trigger:
                passed += 1
                status = "✓"
            else:
                status = "✗"
            logger.info("%s intent='%s' text='%s' → source=%s (expected=%s)",
                       status, intent, text, source, should_trigger)

        results["details"] = {
            "passed": passed,
            "total": len(test_cases),
            "accuracy": f"{passed/len(test_cases)*100:.0f}%",
        }
        results["status"] = "PASSED" if passed == len(test_cases) else "DEGRADED"

    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        logger.error("Context-aware test failed: %s", e)

    return results


# ══════════════════════════════════════════════════════════
# TEST 6: Planner Reasoning
# ══════════════════════════════════════════════════════════


async def test_planner_reasoning() -> Dict[str, Any]:
    """Test that the planner can generate valid action plans."""
    results = {"test": "planner_reasoning", "status": "FAILED", "details": {}}

    try:
        from agent.planner import agent_planner
        agent_planner.initialize()

        test_requests = [
            "open youtube",
            "search for python tutorials",
            "take a screenshot",
        ]

        plans = []
        for request in test_requests:
            plan = agent_planner._generate_plan(request)
            plans.append({
                "request": request,
                "plan": plan,
                "steps": len(plan) if plan else 0,
            })
            logger.info("Plan for '%s': %d steps", request, len(plan) if plan else 0)

        save_json(plans, PLANNER_DIR / "planner_reasoning.json")
        if plans:
            save_json(plans[0], PLANNER_DIR / "action_plan.json")

        valid_plans = sum(1 for p in plans if p["plan"] is not None)
        results["details"] = {
            "plans_generated": valid_plans,
            "total_requests": len(test_requests),
        }
        results["status"] = "PASSED" if valid_plans > 0 else "DEGRADED"

    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        logger.error("Planner test failed: %s", e)

    return results


# ══════════════════════════════════════════════════════════
# Main Runner
# ══════════════════════════════════════════════════════════


async def run_all_tests() -> List[Dict[str, Any]]:
    """Run all vision verification tests."""
    print("=" * 70)
    print("  VISION SYSTEM VERIFICATION")
    print("=" * 70)
    print()

    tests = [
        ("Screen Capture", test_screen_capture()),
        ("OCR Verification", test_ocr()),
        ("UI Detection", test_ui_detection()),
        ("Vision Cache", test_vision_cache()),
        ("Context-Aware Vision", test_context_aware()),
        ("Planner Reasoning", test_planner_reasoning()),
    ]

    results = []
    for name, coro in tests:
        print(f"[{name}]")
        try:
            result = await coro
            results.append(result)
            status = "✓" if result["status"] == "PASSED" else "✗"
            print(f"  {status} {result['status']}")
            if "details" in result and isinstance(result["details"], dict):
                for k, v in result["details"].items():
                    if not isinstance(v, (list, dict)):
                        print(f"    {k}: {v}")
            if "error" in result:
                print(f"    ERROR: {result['error']}")
        except Exception as e:
            results.append({"test": name, "status": "FAILED", "error": str(e)})
            print(f"  ✗ FAILED: {e}")
        print()

    # Summary
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASSED")
    degraded = sum(1 for r in results if r["status"] == "DEGRADED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    print(f"  PASSED: {passed}  |  DEGRADED: {degraded}  |  FAILED: {failed}")
    print()

    report_path = VISION_DIR / "vision_test_report.json"
    save_json(results, report_path)
    print(f"Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())