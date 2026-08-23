#!/usr/bin/env python3
"""
Standalone camera test for Diego Desktop Assistant.

Tests camera initialization, frame capture, and quality metrics.
Works independently of the main Diego codebase.

Usage:
    python debug/test_camera.py
"""

import os
import sys
import time
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

import cv2 as cv
import numpy as np

PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        print(f"  ✓ {label}")
        PASS += 1
    else:
        print(f"  ✗ {label}")
        if detail:
            print(f"    {detail}")
        FAIL += 1


def main():
    global PASS, FAIL
    print()
    print("  ═══════════════════════════════════════════")
    print("  CAMERA TEST")
    print("  ═══════════════════════════════════════════")
    print(f"  Python:  {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    print(f"  OpenCV:  {cv.__version__}")
    print()

    # ── 1. Camera initialization ───────────────────────
    print("  [1/5] Camera initialization")
    cam = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cam.isOpened():
        cam = cv.VideoCapture(0)
    check("Camera opened", cam.isOpened())

    if not cam.isOpened():
        print()
        print(f"  FAILED: Cannot open camera")
        sys.exit(1)

    # Configure
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv.CAP_PROP_FPS, 30)
    cam.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    w = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv.CAP_PROP_FPS)
    backend = cam.get(cv.CAP_PROP_BACKEND)
    fourcc = int(cam.get(cv.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    check("Camera configured", True, f"{w}x{h} @ {fps:.1f} FPS (backend={backend}, codec={codec})")

    # ── 2. Warmup ──────────────────────────────────────
    print("  [2/5] Warmup (5 frames)")
    for i in range(5):
        ret, _ = cam.read()
        if not ret:
            print(f"  ⚠ Warmup frame {i+1} failed")
    check("Warmup complete", True)

    # ── 3. Frame capture ───────────────────────────────
    print("  [3/5] Frame capture")
    ret, frame = cam.read()
    check("Frame captured", ret and frame is not None,
          f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "None")

    # ── 4. Quality metrics ─────────────────────────────
    print("  [4/5] Quality metrics")
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur = float(cv.Laplacian(gray, cv.CV_64F).var())
    check("Brightness", brightness > 10, f"{brightness:.1f} (threshold: 10)")
    check("Blur score", blur > 5, f"{blur:.2f} (threshold: 5)")

    if brightness < 90:
        print(f"  ⚠ Low brightness ({brightness:.1f} < 90) — preprocessing needed")

    # ── 5. FPS measurement ─────────────────────────────
    print("  [5/5] FPS measurement (10 frames)")
    frame_times = []
    for i in range(10):
        t0 = time.time()
        ret, _ = cam.read()
        frame_times.append(time.time() - t0)

    avg_frame_time = sum(frame_times) / len(frame_times)
    measured_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    check("FPS measurement", measured_fps > 5, f"{measured_fps:.1f} FPS (avg {avg_frame_time*1000:.0f}ms/frame)")

    # ── Save test frame ────────────────────────────────
    debug_dir = PROJECT_ROOT / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    filepath = debug_dir / "test_camera_frame.jpg"
    cv.imwrite(str(filepath), frame)
    print(f"\n  Saved test frame: {filepath}")

    # Cleanup
    cam.release()
    print(f"\n  Camera released")

    # Summary
    print()
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  ✓ ALL {total} TESTS PASSED")
    else:
        print(f"  ✗ {FAIL}/{total} TESTS FAILED")
    print()
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())