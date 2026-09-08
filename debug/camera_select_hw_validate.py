#!/usr/bin/env python3
"""
Phase 18F-H -- real-hardware camera selection validation.

Selects ZEB LIVE PRO and verifies:
  - camera is /dev/video2
  - frames arrive
  - the existing YuNet face-detector pipeline works on those frames
  - only one camera stream is owned
  - selection is persisted
"""
import os, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("ALSA_CONFIG_PATH", ""); os.environ["ALSA_DEBUG"]="0"; os.environ["PULSE_LOG"]="0"
import compat  # noqa: F401
import cv2 as cv
import numpy as np

from auth.camera_selector import CameraSelector, get_selector, set_default_index
from auth.faceauth import set_default_index as _unused  # noqa: F401
from auth.face_detector import face_detector
import auth.faceauth as fa

def main():
    print("[HW] enumerating cameras...")
    sel = get_selector()
    for d in sel.enumerate_devices():
        print(f"  /dev/video{d.index} name={d.name!r} caps=0x{d.caps:x} capture_capable={d.capture_capable}")

    print("[HW] selecting ZEB LIVE PRO...")
    dev = sel.select_camera("ZEB LIVE PRO")
    assert dev is not None, "ZEB LIVE PRO not found"
    print(f"[HW] selected: {dev.label} -> {dev.path} (index={dev.index})")
    assert dev.path == "/dev/video2", f"expected /dev/video2, got {dev.path}"

    # Verify frames arrive via a short probe.
    print("[HW] probing for frames...")
    assert sel.probe_device(dev.index), "selected camera produced no frames"
    print("[HW] frames OK")

    # Hand to the existing face-auth pipeline.
    fa._camera = None; fa._camera_refcount = 0
    cam = fa._get_camera()
    assert cam is not None, "faceauth _get_camera returned None"
    print(f"[HW] pipeline opened backend cam index={dev.index}")

    # Read several frames and run YuNet detection.
    faces_found = 0
    for i in range(15):
        ret, frame = cam.read()
        assert ret and frame is not None, f"frame read failed at i={i}"
        faces = face_detector.detect(frame)
        if faces:
            faces_found += 1
        time.sleep(0.05)
    print(f"[HW] read 15 frames, face detected in {faces_found} of them")

    # Single-stream ownership: releasing once must fully close it.
    fa._release_camera()
    assert fa._camera is None, "camera not released"

    # Persistence check.
    stored = sel.load_selection()
    assert stored and "ZEB LIVE PRO" in stored.get("device_name", ""), "selection not persisted"
    print(f"[HW] persisted: {stored['device_path']}")

    print("[HW] ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
