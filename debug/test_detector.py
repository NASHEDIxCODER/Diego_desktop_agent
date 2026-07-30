#!/usr/bin/env python3
"""
Standalone face detector test for Leo Desktop Assistant.

Tests the multi-backend face detector with live camera feed.
Shows raw frame, processed frame, detected faces, backend used, and latency.

Usage:
    python debug/test_detector.py
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

from auth.face_detector import (
    face_detector, FaceBox, FrameQuality,
    measure_brightness, measure_blur, measure_quality,
    preprocess_frame, apply_clahe, apply_gamma_correction, apply_auto_contrast,
    BRIGHTNESS_THRESHOLD, BLUR_THRESHOLD,
)

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
    print("  FACE DETECTOR TEST")
    print("  ═══════════════════════════════════════════")
    print(f"  Python:  {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    print(f"  OpenCV:  {cv.__version__}")
    print(f"  Backend: {face_detector.backend}")
    print(f"  Available: {face_detector.is_available}")
    print()

    if not face_detector.is_available:
        print("  ✗ No face detection backend available!")
        sys.exit(1)

    # ── 1. Open camera ─────────────────────────────────
    print("  [1/4] Open camera")
    cam = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cam.isOpened():
        cam = cv.VideoCapture(0)
    check("Camera opened", cam.isOpened())

    if not cam.isOpened():
        sys.exit(1)

    cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv.CAP_PROP_FPS, 30)
    cam.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    w = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    face_detector.set_input_size(w, h)
    check("Camera configured", True, f"{w}x{h}")

    # Warmup
    for i in range(5):
        cam.read()
    check("Warmup complete", True)

    # ── 2. Test on existing debug frame ────────────────
    print("  [2/4] Test on existing debug frame")
    debug_dir = PROJECT_ROOT / "debug"
    frames = sorted(debug_dir.glob("auth_capture_*.jpg"))
    if frames:
        latest = frames[-1]
        img = cv.imread(str(latest))
        quality = measure_quality(img)
        print(f"  Raw frame: {img.shape[1]}x{img.shape[0]}, brightness={quality.brightness:.1f}, blur={quality.blur:.2f}")

        # Detect on raw frame
        t0 = time.time()
        faces_raw = face_detector.detect(img)
        latency_raw = time.time() - t0
        check("Detection on raw frame", len(faces_raw) > 0,
              f"{len(faces_raw)} face(s) in {latency_raw*1000:.0f}ms" if faces_raw else "0 faces")

        # Preprocess and detect
        processed, proc_quality = preprocess_frame(img)
        t0 = time.time()
        faces_proc = face_detector.detect(processed)
        latency_proc = time.time() - t0
        check("Detection after preprocessing", len(faces_proc) > 0,
              f"{len(faces_proc)} face(s) in {latency_proc*1000:.0f}ms (brightness: {quality.brightness:.1f} → {proc_quality.brightness:.1f})" if faces_proc else "0 faces")

        # Save comparison
        if faces_raw:
            for f in faces_raw:
                cv.rectangle(img, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 255, 0), 2)
                cv.putText(img, f"{f.confidence:.2f} ({f.backend})", (f.x, f.y - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imwrite(str(debug_dir / "test_detector_raw.jpg"), img)

        if faces_proc:
            for f in faces_proc:
                cv.rectangle(processed, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 255, 0), 2)
                cv.putText(processed, f"{f.confidence:.2f} ({f.backend})", (f.x, f.y - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imwrite(str(debug_dir / "test_detector_processed.jpg"), processed)
        print(f"  Saved: test_detector_raw.jpg, test_detector_processed.jpg")
    else:
        print("  ⚠ No existing debug frames found")

    # ── 3. Live detection ──────────────────────────────
    print("  [3/4] Live detection (50 frames)")
    print("  Press 'q' to quit early, 'd' to toggle debug overlay")

    debug_overlay = True
    frame_count = 0
    face_count = 0
    latencies = []
    fps_start = time.time()
    fps_count = 0

    while frame_count < 50:
        ret, frame = cam.read()
        if not ret:
            continue

        frame_count += 1
        fps_count += 1

        # Measure quality
        quality = measure_quality(frame)

        # Detect faces
        t0 = time.time()
        faces = face_detector.detect(frame)
        latency = time.time() - t0
        latencies.append(latency)

        if faces:
            face_count += 1

        # FPS
        fps_elapsed = time.time() - fps_start
        if fps_elapsed >= 1.0:
            current_fps = fps_count / fps_elapsed
            fps_count = 0
            fps_start = time.time()
        else:
            current_fps = 0

        # Debug overlay
        if debug_overlay:
            display = frame.copy()

            # Semi-transparent background
            overlay_rect = display.copy()
            cv.rectangle(overlay_rect, (5, 5), (400, 180), (0, 0, 0), -1)
            cv.addWeighted(overlay_rect, 0.6, display, 0.4, 0, display)

            y = 25
            cv.putText(display, f"Frame: {frame_count}/50", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            cv.putText(display, f"FPS: {current_fps:.1f}" if current_fps > 0 else "FPS: ...", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            cv.putText(display, f"Brightness: {quality.brightness:.1f}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            cv.putText(display, f"Blur: {quality.blur:.2f}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            cv.putText(display, f"Backend: {face_detector.backend}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            cv.putText(display, f"Faces: {len(faces)}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            cv.putText(display, f"Latency: {latency*1000:.0f}ms", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw face boxes
            for face in faces:
                color = (0, 255, 0) if face.confidence > 0.7 else (0, 165, 255)
                cv.rectangle(display, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2)
                label = f"{face.confidence:.2f} ({face.backend})"
                cv.putText(display, label, (face.x, face.y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv.imshow("Leo Face Detector Test", display)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_overlay = not debug_overlay

    cv.destroyAllWindows()
    cam.release()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    check("Live detection completed", frame_count > 0, f"{frame_count} frames, {face_count} with faces")
    check("Average latency", avg_latency < 0.5, f"{avg_latency*1000:.0f}ms per frame")
    check("Face detection rate", face_count > 0, f"{face_count}/{frame_count} frames had faces")

    # ── 4. Backend summary ─────────────────────────────
    print("  [4/4] Backend summary")
    check("Backend available", face_detector.is_available, face_detector.backend)
    print(f"  Backend: {face_detector.backend}")
    print(f"  YuNet available: {face_detector._yunet_available}")
    print(f"  face_recognition available: {face_detector._face_recognition_available}")

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