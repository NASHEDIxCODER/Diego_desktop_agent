#!/usr/bin/env python3
"""
Standalone microphone test for Leo Desktop Assistant.

Tests sounddevice import, device enumeration, audio capture, and STT.
Works completely independently of the main Leo codebase.

Usage:
    python debug/test_microphone.py

Output:
    ✓ sounddevice import
    ✓ devices
    ✓ selected device
    ✓ records 5 seconds
    ✓ writes test.wav
    ✓ transcribes test.wav
"""

import os
import sys
import time
import wave
import traceback
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Python 3.14 compatibility: inject stubs for removed stdlib modules
# (aifc, audioop, imghdr) needed by speech_recognition
import compat  # noqa: F401

# Configuration
TEST_DURATION = 5.0  # seconds
SAMPLE_RATE = 16000
OUTPUT_WAV = PROJECT_ROOT / "debug" / "test_microphone_output.wav"

PASS = 0
FAIL = 0
ERRORS = []


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
    global PASS, FAIL, ERRORS
    print()
    print("  ═══════════════════════════════════════════")
    print("  MICROPHONE TEST")
    print("  ═══════════════════════════════════════════")
    print(f"  Python:      {sys.executable}")
    print(f"  Version:     {sys.version.split()[0]}")
    print(f"  Project:     {PROJECT_ROOT}")
    print(f"  Duration:    {TEST_DURATION}s")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print()

    # ── 1. sounddevice import ──────────────────────────
    print("  [1/6] sounddevice import")
    try:
        import sounddevice as sd
        check("sounddevice import", True, f"version {sd.__version__}")
    except ImportError as e:
        check("sounddevice import", False, f"ImportError: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)
    except ModuleNotFoundError as e:
        check("sounddevice import", False, f"ModuleNotFoundError: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)
    except OSError as e:
        check("sounddevice import", False, f"OSError (missing PortAudio?): {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)
    except Exception as e:
        check("sounddevice import", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)

    # ── 2. Device enumeration ──────────────────────────
    print("  [2/6] Device enumeration")
    try:
        devices = sd.query_devices()
        check("sd.query_devices()", True, f"{len(devices)} devices found")
        for d in devices:
            print(f"         [{d['index']}] {d['name']} "
                  f"(in={d['max_input_channels']}, out={d['max_output_channels']}, "
                  f"{d['default_samplerate']} Hz)")
    except Exception as e:
        check("sd.query_devices()", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        # Continue anyway

    # ── 3. Selected device ─────────────────────────────
    print("  [3/6] Selected device")
    try:
        try:
            default_idx = sd.default.device[0]
            check("sd.default.device", True, f"default input device: [{default_idx}]")
        except Exception as e:
            check("sd.default.device", False, str(e))
            default_idx = None

        # Find input devices
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            check("Input devices", False, "No input devices found")
            print()
            print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
            print()
            sys.exit(1)

        check("Input devices", True, f"{len(input_devices)} found")

        # Select the best device
        if default_idx is not None:
            selected = next((d for d in input_devices if d['index'] == default_idx), input_devices[0])
        else:
            selected = input_devices[0]

        selected_name = selected['name']
        selected_idx = selected['index']
        selected_sr = int(selected['default_samplerate']) if selected.get('default_samplerate') else SAMPLE_RATE
        selected_ch = selected['max_input_channels']

        check("Selected device", True,
              f"[{selected_idx}] {selected_name} (ch={selected_ch}, sr={selected_sr} Hz)")

    except Exception as e:
        check("Selected device", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)

    # ── 4. Record audio ────────────────────────────────
    print(f"  [4/6] Record {TEST_DURATION}s audio")
    try:
        print(f"         Recording for {TEST_DURATION}s...")
        recording = sd.rec(
            int(TEST_DURATION * selected_sr),
            samplerate=selected_sr,
            channels=1,
            dtype='int16',
            device=selected_idx,
            blocking=True,
        )
        print(f"         Recording complete!")
        audio_bytes = recording.tobytes()
        duration_actual = len(audio_bytes) / 2 / selected_sr

        # Check quality
        import numpy as np
        rms = float(np.sqrt(np.mean(recording.astype(float)**2)))
        peak = float(np.max(np.abs(recording)))
        check("Audio recorded", True,
              f"{len(audio_bytes)} bytes, {duration_actual:.1f}s, RMS={rms:.1f}, peak={peak}")

        if rms < 5.0:
            print("         ⚠ WARNING: Very low audio level (RMS < 5.0)")
            print("         Check microphone connection and volume")
    except Exception as e:
        check("Audio recorded", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")
        print()
        print(f"  FAILED: {FAIL}/{PASS + FAIL} tests passed")
        print()
        sys.exit(1)

    # ── 5. Write test.wav ──────────────────────────────
    print("  [5/6] Write test.wav")
    try:
        OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(OUTPUT_WAV), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(selected_sr)
            wav.writeframes(audio_bytes)
        file_size = OUTPUT_WAV.stat().st_size
        check("test.wav written", True, f"{OUTPUT_WAV} ({file_size} bytes, {selected_sr} Hz, 16-bit mono)")
    except Exception as e:
        check("test.wav written", False, f"{type(e).__name__}: {e}")
        print(f"\n  Full traceback:\n  {traceback.format_exc().strip()}")

    # ── 6. Transcribe test.wav ─────────────────────────
    print("  [6/6] Transcribe test.wav")
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(str(OUTPUT_WAV)) as source:
            audio = r.record(source)
        print("         Sending to Google STT...")
        text = r.recognize_google(audio, language="en-IN")
        if text and text.strip():
            check("Transcription", True, f"'{text.strip()}'")
        else:
            check("Transcription", False, "Empty transcript")
    except ImportError:
        check("Transcription", False, "speech_recognition not installed")
    except sr.UnknownValueError:
        check("Transcription", False, "Google could not understand audio (silence?)")
    except sr.RequestError as e:
        check("Transcription", False, f"Google STT error: {e}")
    except Exception as e:
        check("Transcription", False, f"{type(e).__name__}: {e}")
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