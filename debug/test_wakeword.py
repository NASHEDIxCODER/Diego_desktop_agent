#!/usr/bin/env python3
"""
Standalone wake word test for Leo Desktop Assistant.

Tests the wake word detection pipeline in isolation.
Loads ONLY: microphone, wake detector, speech recognizer.
No planner, no plugins, no TTS.

Usage:
    python debug/test_wakeword.py

Output:
    Listening...
    Wake detected
    Transcript
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Python 3.14 compatibility: inject stubs for removed stdlib modules
# (aifc, audioop, imghdr) needed by speech_recognition
import compat  # noqa: F401

PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = ""):
    """Print a test result."""
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
    print("  WAKE WORD TEST")
    print("  ═══════════════════════════════════════════")
    print(f"  Python:  {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    print()

    # ── 1. Import sounddevice ──────────────────────────
    print("  [1/5] Import sounddevice")
    try:
        import sounddevice as sd
        check("sounddevice import", True, f"version {sd.__version__}")
        print(f"         Devices: {len(sd.query_devices())} found")
        print(f"         Default: {sd.default.device}")
    except Exception as e:
        check("sounddevice import", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)

    # ── 2. Import wake word engine ─────────────────────
    print("  [2/5] Import wake word engine")
    try:
        from voice.wake_word import WakeWordEngine, DEFAULT_WAKE_VARIANTS
        ww = WakeWordEngine()
        check("WakeWordEngine import", True, f"variants: {DEFAULT_WAKE_VARIANTS}")
        print(f"         Porcupine: {'available' if ww.has_offline_engine else 'not available'}")
    except Exception as e:
        check("WakeWordEngine import", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)

    # ── 3. Import microphone ───────────────────────────
    print("  [3/5] Import microphone")
    try:
        from voice.microphone import MicrophoneManager, SoundDeviceMic
        mic_mgr = MicrophoneManager()
        mic = mic_mgr.get_microphone()
        if mic is not None:
            check("Microphone init", True, f"backend={mic_mgr.backend}")
            if hasattr(mic, 'device_index') and mic.device_index is not None:
                print(f"         Device index: {mic.device_index}")
            if hasattr(mic, 'sample_rate') and mic.sample_rate is not None:
                print(f"         Sample rate:  {mic.sample_rate}")
        else:
            check("Microphone init", False, "No microphone available")
            print()
            print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
            print()
            sys.exit(1)
    except Exception as e:
        check("Microphone init", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)

    # ── 4. Import speech recognizer ────────────────────
    print("  [4/5] Import speech recognizer")
    try:
        from voice.stt import HAS_SOUNDDEVICE, AUDIO_BACKEND, listen_wake, listen
        check("STT module import", True, f"backend={AUDIO_BACKEND}")
        check("HAS_SOUNDDEVICE", HAS_SOUNDDEVICE, f"sounddevice: {AUDIO_BACKEND}")

        if not HAS_SOUNDDEVICE:
            check("Wake detection", False,
                  "sounddevice not available - wake detection disabled")
            print()
            print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
            print()
            sys.exit(1)
    except Exception as e:
        check("STT module import", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)

    # ── 5. Listen for wake word ────────────────────────
    print("  [5/5] Listen for wake word")
    print()
    print("  ═══════════════════════════════════════════")
    print("  SAY: 'hello leo' or 'hey leo'")
    print("  ═══════════════════════════════════════════")
    print()

    # Test 1: quick device check
    try:
        import sounddevice as sd
        import numpy as np

        # Quick test - record 1 second to verify audio pipeline
        print("  Testing audio pipeline (recording 1s)...")
        try:
            device_idx = sd.default.device[0]
            test_rec = sd.rec(int(1 * 16000), samplerate=16000, channels=1,
                            dtype='int16', device=device_idx, blocking=True)
            test_rms = float(np.sqrt(np.mean(test_rec.astype(float)**2)))
            check("Audio pipeline test", True, f"RMS={test_rms:.1f}")
            if test_rms < 5.0:
                print("  ⚠ WARNING: Low audio level - check microphone")
        except Exception as e:
            check("Audio pipeline test", False, f"{type(e).__name__}: {e}")
            print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
    except Exception as e:
        print(f"  ⚠ Audio pipeline test skipped: {e}")

    print()
    print("  Listening... (say 'hello leo' or press Ctrl+C to stop)")
    print()

    try:
        # Use the continuous wake listener (blocks until wake word detected)
        text = listen_wake(phrase_time_limit=5.0)

        if text:
            print()
            print("  ═══════════════════════════════════════════")
            print("  Wake detected!")
            print("  ═══════════════════════════════════════════")
            print(f"  Transcript: '{text}'")
            print()

            # Check if the wake word engine recognizes it
            if ww.detect(text):
                check("Wake word matched", True, f"'{text}'")
            else:
                check("Wake word matched", False,
                      f"'{text}' did not match any wake word variants")

            # Try to get a command
            print()
            print("  Now say a command (e.g., 'what time is it')...")
            print("  (timeout: 8 seconds)")
            print()
            command = listen(timeout=8.0, phrase_time_limit=7.0)
            if command:
                print()
                print("  ═══════════════════════════════════════════")
                print(f"  Command: '{command}'")
                print("  ═══════════════════════════════════════════")
                print()
                check("Command received", True, f"'{command}'")
            else:
                print("  No command received (timeout or silence)")
                check("Command received", False, "No speech detected")
        else:
            check("Wake word detection", False, "No wake word detected")
            print()
            print("  No wake word detected - check microphone and audio levels")
            print()

    except KeyboardInterrupt:
        print()
        print("  Interrupted by user")
        check("Wake word detection", False, "Interrupted")
    except Exception as e:
        check("Wake word detection", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")

    # ── Summary ─────────────────────────────────────────
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