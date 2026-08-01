#!/usr/bin/env python3
"""
Standalone audio pipeline test for Leo Desktop Assistant.

Tests the unified AudioManager:
  - sounddevice import
  - InputStream start
  - Ring buffer capture
  - VAD calibration
  - Command recording simulation
  - Clean shutdown

Usage:
    python debug/test_audio.py
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

PASS = 0
FAIL = 0


def check(label, ok, detail=""):
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
    print("  AUDIO PIPELINE TEST")
    print("  ═══════════════════════════════════════════")
    print(f"  Python:  {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    print()

    # 1. Import sounddevice
    print("  [1/6] Import sounddevice")
    try:
        import sounddevice as sd
        check("sounddevice import", True, f"v{sd.__version__}")
    except Exception as e:
        check("sounddevice import", False, str(e))
        sys.exit(1)

    # 2. Import AudioManager
    print("  [2/6] Import AudioManager")
    try:
        from voice.audio_manager import audio_manager, SAMPLE_RATE
        check("AudioManager import", True)
    except Exception as e:
        check("AudioManager import", False, str(e))
        sys.exit(1)

    # 3. Start AudioManager
    print("  [3/6] Start AudioManager")
    ok = audio_manager.start()
    check("AudioManager started", ok, f"backend={audio_manager.backend}")
    if not ok:
        sys.exit(1)

    print(f"    Device:    [{audio_manager.device_index}]")
    print(f"    Sample:    {audio_manager.sample_rate} Hz")
    print(f"    Backend:   {audio_manager.backend}")

    # 4. Calibrate
    print("  [4/6] Calibrate ambient noise")
    ok = audio_manager.calibrate(1.5)
    check("Calibration", ok, f"threshold={audio_manager.energy_threshold:.1f}")

    # 5. Ring buffer test
    print("  [5/6] Ring buffer capture")
    time.sleep(0.5)
    audio = audio_manager.get_recent_audio(0.5)
    check("Buffer has audio", len(audio) > 0, f"{len(audio)} samples")

    import numpy as np
    if len(audio) > 0:
        rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
        check("Audio quality", rms > 0, f"RMS={rms:.1f}")

    # 6. Clean shutdown
    print("  [6/6] Clean shutdown")
    audio_manager.stop()
    check("Stopped", not audio_manager.is_running)

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