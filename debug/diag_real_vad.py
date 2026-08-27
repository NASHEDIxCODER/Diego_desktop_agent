"""
Diagnostic: capture REAL microphone audio and run Silero VAD on it.
Determines whether Silero returns 0.0 on real audio (content/processing
issue) or works fine (concurrency/state issue).

Run (speak during the recording window):
    python debug/diag_real_vad.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.audio_manager import AudioManager, SAMPLE_RATE, FRAME_SAMPLES
from voice.vad import unified_vad, VAD_FRAME_SAMPLES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    print("=" * 70)
    print("REAL MIC VAD DIAGNOSTIC")
    print("=" * 70)
    print("Recording 5 seconds — SPEAK during this window...")
    print()

    am = AudioManager()
    if not am.start():
        print("FAILED to start audio (no working mic)")
        return 1

    unified_vad.load()
    print(f"Silero ready={unified_vad.ready} backend={unified_vad.get_diagnostics()['backend']}")
    print(f"Device: [{am.device_index}] {am.device_name} @ {am.sample_rate} Hz")
    print(f"Speech channel: {am.speech_channel}")
    print()

    # Record 5 seconds
    time.sleep(5.0)

    # Get the ring buffer (post AGC+resample+highpass)
    ring = am.get_recent_audio(5.0)
    if len(ring) == 0:
        print("EMPTY ring buffer!")
        am.stop()
        return 1

    a64 = ring.astype(np.float64)
    rms = float(np.sqrt(np.mean(a64 ** 2)))
    peak = float(np.max(np.abs(a64)))
    print(f"Ring buffer: {len(ring)} samples ({len(ring)/SAMPLE_RATE:.2f}s) "
          f"rms={rms:.6f} (int16={rms*32768:.1f}) peak={peak:.6f}")

    # Run Silero on each 512-sample frame
    print("\nSilero VAD per frame (512 samples @ 16kHz):")
    probs = []
    for i in range(0, len(ring) - 512 + 1, 512):
        p = unified_vad.speech_prob(ring[i:i + 512])
        probs.append(p)
    for i, p in enumerate(probs):
        bar = "#" * int(p * 40)
        print(f"  frame {i:2d}: {p:.3f} {bar}")
    if probs:
        print(f"  max={max(probs):.3f} avg={sum(probs)/len(probs):.3f} "
              f"above_0.5={sum(1 for p in probs if p >= 0.5)}/{len(probs)}")

    # Also test with a fresh model instance to rule out state corruption
    print("\n[Fresh model test]")
    try:
        from silero_vad import load_silero_vad
        fresh = load_silero_vad(onnx=True)
        import torch
        fresh_probs = []
        for i in range(0, len(ring) - 512 + 1, 512):
            frame = ring[i:i + 512]
            tensor = torch.from_numpy(np.asarray(frame, dtype=np.float32))
            with torch.no_grad():
                p = fresh(tensor, 16000).item()
            fresh_probs.append(float(p))
        print(f"  fresh model max={max(fresh_probs):.3f} "
              f"avg={sum(fresh_probs)/len(fresh_probs):.3f} "
              f"above_0.5={sum(1 for p in fresh_probs if p >= 0.5)}/{len(fresh_probs)}")
    except Exception as e:
        print(f"  fresh model test failed: {e}")

    # Dump raw input for offline analysis
    path = am.dump_raw_input(5.0)
    if path:
        print(f"\nRaw input saved: {path}")

    am.stop()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())