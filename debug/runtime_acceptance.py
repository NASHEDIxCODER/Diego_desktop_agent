#!/usr/bin/env python3
"""
Runtime Acceptance Test — Physical Hardware Required.

Run with:
    python debug/runtime_acceptance.py

This script runs the FULL acceptance suite:
  - 100 wake attempts
  - 50 authenticated sessions
  - 50 command sessions

Reports:
  - wake success rate
  - face auth success rate
  - command recognition success rate
  - false wakes
  - average latency
  - average command duration

Requirements:
  - Physical microphone connected
  - Physical camera connected
  - Registered face encodings in auth/Known_encodings.p
  - Wake model trained (run --train-wake if needed)

Usage:
  Stand in front of the camera and say "Hello Leo" when prompted.
  For command sessions, say a command like "what time is it" when prompted.
"""

import sys
import os
import time
import json
import asyncio
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compat  # noqa

from voice.audio_manager import audio_manager, shutdown_event, SAMPLE_RATE
from voice.audio_processing import audio_preprocessor
from voice.stt import _recognize_bytes, _init_whisper, listen_wake_continuous
from voice.wake_word import verify_wake_transcript, wake_word_engine
from voice.wake_model_manager import wake_model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("acceptance")


# ─── Results ──────────────────────────────────────────────────────────────

class AcceptanceResults:
    def __init__(self):
        self.wake_attempts = 0
        self.wake_successes = 0
        self.wake_false_positives = 0
        self.wake_latencies = []
        self.face_attempts = 0
        self.face_successes = 0
        self.face_latencies = []
        self.command_attempts = 0
        self.command_successes = 0
        self.command_durations = []
        self.command_latencies = []
        self.errors = []

    def summary(self) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wake": {
                "attempts": self.wake_attempts,
                "successes": self.wake_successes,
                "success_rate": (self.wake_successes / self.wake_attempts * 100
                                 if self.wake_attempts else 0),
                "false_wakes": self.wake_false_positives,
                "false_wake_rate": (self.wake_false_positives / self.wake_attempts * 100
                                    if self.wake_attempts else 0),
                "avg_latency_s": (sum(self.wake_latencies) / len(self.wake_latencies)
                                  if self.wake_latencies else 0),
            },
            "face_auth": {
                "attempts": self.face_attempts,
                "successes": self.face_successes,
                "success_rate": (self.face_successes / self.face_attempts * 100
                                 if self.face_attempts else 0),
                "avg_latency_s": (sum(self.face_latencies) / len(self.face_latencies)
                                  if self.face_latencies else 0),
            },
            "command": {
                "attempts": self.command_attempts,
                "successes": self.command_successes,
                "success_rate": (self.command_successes / self.command_attempts * 100
                                 if self.command_attempts else 0),
                "avg_duration_s": (sum(self.command_durations) / len(self.command_durations)
                                  if self.command_durations else 0),
                "avg_latency_s": (sum(self.command_latencies) / len(self.command_latencies)
                                  if self.command_latencies else 0),
            },
            "errors": self.errors,
        }

    def print_report(self):
        s = self.summary()
        print("\n" + "=" * 70)
        print("RUNTIME ACCEPTANCE REPORT")
        print("=" * 70)
        print(f"Timestamp: {s['timestamp']}")
        print()
        print("── Wake Word ──")
        w = s["wake"]
        print(f"  Attempts:        {w['attempts']}")
        print(f"  Successes:       {w['successes']}")
        print(f"  Success Rate:    {w['success_rate']:.1f}%")
        print(f"  False Wakes:     {w['false_wakes']}")
        print(f"  False Wake Rate: {w['false_wake_rate']:.1f}%")
        print(f"  Avg Latency:     {w['avg_latency_s']:.3f}s")
        print()
        print("── Face Authentication ──")
        f = s["face_auth"]
        print(f"  Attempts:        {f['attempts']}")
        print(f"  Successes:       {f['successes']}")
        print(f"  Success Rate:    {f['success_rate']:.1f}%")
        print(f"  Avg Latency:     {f['avg_latency_s']:.3f}s")
        print()
        print("── Command Recognition ──")
        c = s["command"]
        print(f"  Attempts:        {c['attempts']}")
        print(f"  Successes:       {c['successes']}")
        print(f"  Success Rate:    {c['success_rate']:.1f}%")
        print(f"  Avg Duration:    {c['avg_duration_s']:.3f}s")
        print(f"  Avg Latency:     {c['avg_latency_s']:.3f}s")
        print()
        if s["errors"]:
            print("── Errors ──")
            for err in s["errors"]:
                print(f"  {err}")
        print("=" * 70)

    def save(self, path="debug/acceptance_report.json"):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        logger.info("Report saved to %s", path)


# ─── Wake Test ────────────────────────────────────────────────────────────

def test_wake(results: AcceptanceResults, n_attempts: int = 100):
    """Run n wake attempts. User says 'Hello Leo' when prompted."""
    print(f"\n{'─' * 70}")
    print(f"WAKE TEST: {n_attempts} attempts")
    print(f"{'─' * 70}")
    print("Say 'Hello Leo' when prompted. Stay silent for false-wake checks.")
    print()

    if not audio_manager.start():
        results.errors.append("AudioManager failed to start")
        return

    time.sleep(0.5)  # Stabilize

    for i in range(n_attempts):
        if shutdown_event.is_set():
            break

        print(f"  [{i+1}/{n_attempts}] Say 'Hello Leo' now...", end="", flush=True)
        t_start = time.time()

        try:
            # Listen for wake word (timeout 10s)
            transcript = listen_wake_continuous(timeout=10.0)
            latency = time.time() - t_start

            results.wake_attempts += 1
            results.wake_latencies.append(latency)

            if transcript and verify_wake_transcript(transcript):
                results.wake_successes += 1
                print(f" ✓ ({latency:.2f}s) '{transcript}'")
            elif transcript:
                # Got a transcript but it didn't verify — could be false wake
                results.wake_false_positives += 1
                print(f" ✗ FALSE WAKE ({latency:.2f}s) '{transcript}'")
            else:
                print(f" ✗ timeout ({latency:.2f}s)")

        except Exception as e:
            results.errors.append(f"wake[{i}]: {e}")
            print(f" ERROR: {e}")

        time.sleep(0.5)  # Brief pause between attempts

    audio_manager.stop()


# ─── Face Auth Test ───────────────────────────────────────────────────────

def test_face_auth(results: AcceptanceResults, n_attempts: int = 50):
    """Run n face auth attempts. User stands in front of camera."""
    print(f"\n{'─' * 70}")
    print(f"FACE AUTH TEST: {n_attempts} attempts")
    print(f"{'─' * 70}")
    print("Stand in front of the camera when prompted.")
    print()

    try:
        from auth.faceauth import recognize_faces, close as close_faceauth
    except Exception as e:
        results.errors.append(f"faceauth import: {e}")
        print(f"  faceauth import failed: {e}")
        return

    for i in range(n_attempts):
        if shutdown_event.is_set():
            break

        print(f"  [{i+1}/{n_attempts}] Look at the camera...", end="", flush=True)
        t_start = time.time()

        try:
            # recognize_faces() returns the user name on success, None on failure
            user_name = recognize_faces()
            latency = time.time() - t_start

            results.face_attempts += 1
            results.face_latencies.append(latency)

            if user_name:
                results.face_successes += 1
                print(f" ✓ ({latency:.2f}s) '{user_name}'")
            else:
                print(f" ✗ ({latency:.2f}s)")

        except Exception as e:
            results.errors.append(f"face[{i}]: {e}")
            print(f" ERROR: {e}")

        time.sleep(0.5)

    # Release camera between test phases
    try:
        close_faceauth()
    except Exception:
        pass


# ─── Command Test ─────────────────────────────────────────────────────────

def test_command(results: AcceptanceResults, n_attempts: int = 50):
    """Run n command recording + recognition attempts."""
    print(f"\n{'─' * 70}")
    print(f"COMMAND TEST: {n_attempts} attempts")
    print(f"{'─' * 70}")
    print("Say a command (e.g. 'what time is it') when prompted.")
    print()

    # Initialize Whisper
    _init_whisper()

    if not audio_manager.start():
        results.errors.append("AudioManager failed to start")
        return

    time.sleep(0.5)

    for i in range(n_attempts):
        if shutdown_event.is_set():
            break

        print(f"  [{i+1}/{n_attempts}] Say a command now...", end="", flush=True)
        t_start = time.time()

        try:
            # Record command (timeout=8s, phrase_limit=7s)
            audio_bytes = audio_manager.record_command(timeout=8.0, phrase_limit=7.0)
            record_duration = time.time() - t_start

            results.command_attempts += 1
            results.command_durations.append(record_duration)

            if audio_bytes is None:
                print(f" ✗ no audio ({record_duration:.2f}s)")
                continue

            # Verify phrase_limit invariant
            audio_duration = len(audio_bytes) / SAMPLE_RATE / 2  # 16-bit mono
            if audio_duration > 7.0 + 0.1:
                results.errors.append(
                    f"command[{i}]: INVARIANT VIOLATION duration={audio_duration:.2f}s > phrase_limit=7.0s"
                )
                print(f" ✗ INVARIANT VIOLATION ({audio_duration:.2f}s > 7.0s)")
                continue

            # Preprocess + recognize (process() returns float32 [-1, 1];
            # convert to PCM16 once at the STT sink).
            import numpy as _np
            from voice.audio_processing import float32_to_int16
            samples = audio_preprocessor.process(
                _np.frombuffer(audio_bytes, dtype=_np.int16)
            )
            text = _recognize_bytes(float32_to_int16(samples).tobytes(), SAMPLE_RATE)
            latency = time.time() - t_start
            results.command_latencies.append(latency)

            if text and text.strip():
                results.command_successes += 1
                print(f" ✓ ({record_duration:.2f}s, {latency:.2f}s) '{text}'")
            else:
                print(f" ✗ no transcript ({record_duration:.2f}s)")

        except Exception as e:
            results.errors.append(f"command[{i}]: {e}")
            print(f" ERROR: {e}")

        time.sleep(0.5)

    audio_manager.stop()


# ─── Main ─────────────────────────────────────────────────────────────────

async def main():
    results = AcceptanceResults()

    print("=" * 70)
    print("LEO DESKTOP ASSISTANT — RUNTIME ACCEPTANCE TEST")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("This test requires:")
    print("  - Physical microphone")
    print("  - Physical camera")
    print("  - Registered face encodings")
    print("  - Trained wake model")
    print()
    print("Tests:")
    print("  1. Wake:    100 attempts (say 'Hello Leo')")
    print("  2. Face:     50 attempts (look at camera)")
    print("  3. Command:  50 attempts (say a command)")
    print()

    # Check prerequisites
    if not wake_model_manager.load():
        print("WARNING: Wake model not loaded. Wake tests will likely fail.")
        print("  Run: python main.py --train-wake")

    try:
        # Run tests
        test_wake(results, n_attempts=100)
        test_face_auth(results, n_attempts=50)
        test_command(results, n_attempts=50)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        results.errors.append(f"fatal: {e}")
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        # Ensure cleanup
        shutdown_event.set()
        audio_manager.stop()

    # Report
    results.print_report()
    results.save()

    # Exit code: 0 if all tests passed, 1 otherwise
    s = results.summary()
    all_passed = (
        s["wake"]["success_rate"] >= 80 and
        s["face_auth"]["success_rate"] >= 80 and
        s["command"]["success_rate"] >= 80 and
        s["wake"]["false_wake_rate"] <= 5
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)