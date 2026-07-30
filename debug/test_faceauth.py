#!/usr/bin/env python3
"""
Standalone face authentication test for Leo Desktop Assistant.

Tests the full face authentication pipeline:
  - Camera initialization
  - Continuous capture
  - Frame preprocessing
  - Face detection (multi-backend)
  - Face tracking (5 stable frames)
  - 128D embedding
  - Comparison against known encodings
  - Decision

Usage:
    python debug/test_faceauth.py
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

import logging

# Configure logging to show all stages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
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
    print("  FACE AUTHENTICATION TEST")
    print("  ═══════════════════════════════════════════")
    print(f"  Python:  {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    print()

    # ── 1. Import face auth module ─────────────────────
    print("  [1/4] Import face auth module")
    try:
        from auth.faceauth import recognize_faces, close, _load_encodings
        from auth.face_detector import face_detector
        check("Import faceauth", True)
        check("Detector backend", face_detector.is_available, face_detector.backend)
    except Exception as e:
        check("Import faceauth", False, str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 2. Load encodings ──────────────────────────────
    print("  [2/4] Load encodings")
    try:
        loaded = _load_encodings()
        check("Encodings loaded", loaded)
        if loaded:
            from auth.faceauth import _known_encodings, _known_names
            unique = len(set(_known_names))
            check("Encoding count", len(_known_encodings) > 0,
                  f"{len(_known_encodings)} samples, {unique} users: {sorted(set(_known_names))}")
    except Exception as e:
        check("Encodings loaded", False, str(e))
        sys.exit(1)

    # ── 3. Run face authentication ─────────────────────
    print("  [3/4] Run face authentication")
    print("  Stand in front of the camera...")
    print("  The test will run for up to 5 seconds.")
    print()

    try:
        t0 = time.time()
        result = recognize_faces()
        elapsed = time.time() - t0

        if result:
            check("Authentication", True, f"Authenticated as '{result}' in {elapsed:.1f}s")
        else:
            check("Authentication", False, f"No face recognized in {elapsed:.1f}s")
    except Exception as e:
        check("Authentication", False, f"Error: {e}")
        import traceback
        traceback.print_exc()

    # ── 4. Cleanup ─────────────────────────────────────
    print("  [4/4] Cleanup")
    try:
        close()
        check("Cleanup", True)
    except Exception as e:
        check("Cleanup", False, str(e))

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