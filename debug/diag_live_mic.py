"""
Live microphone diagnostic — capture real audio through the full pipeline
and measure Silero's response at every stage.

Run (speak during the recording window):
    LEO_VAD_DIAG=1 python debug/diag_live_mic.py

Records ~4 seconds of live audio, then reports:
  1. raw mic (44.1kHz)
  2. post-AGC
  3. post-resample (16kHz)
  4. post-high-pass
  5. Silero VAD probabilities per 512-sample frame
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


def metrics(name, a, sr):
    a = np.asarray(a)
    if a.size == 0:
        print(f"  [{name}] EMPTY")
        return
    a64 = a.astype(np.float64)
    rms = float(np.sqrt(np.mean(a64 ** 2)))
    peak = float(np.max(np.abs(a64)))
    finite = bool(np.all(np.isfinite(a64)))
    zero_ratio = float(np.mean(a64 == 0.0))
    print(
        f"  [{name}] sr={sr} dtype={a.dtype} shape={a.shape} "
        f"min={a64.min():.6f} max={a64.max():.6f} "
        f"rms={rms:.6f} (int16={rms*32768:.1f}) peak={peak:.6f} "
        f"zero_ratio={zero_ratio:.4f} finite={finite}"
    )


def main():
    print("=" * 70)
    print("LIVE MICROPHONE DIAGNOSTIC")
    print("=" * 70)
    print("Recording 4 seconds — SPEAK during this window...")
    print()

    am = AudioManager()
    if not am.start():
        print("FAILED to start audio (no working mic)")
        return 1

    # Load Silero
    unified_vad.load()
    print(f"Silero ready={unified_vad.ready} backend={unified_vad.get_diagnostics()['backend']}")
    print(f"Device: [{am.device_index}] {am.device_name} @ {am.sample_rate} Hz")
    print(f"Speech channel: {am.speech_channel}")
    print()

    # Record 4 seconds
    time.sleep(4.0)

    # Get the ring buffer (post high-pass, 16kHz)
    ring = am.get_recent_audio(4.0)
    metrics("ring buffer (post AGC+resample+highpass)", ring, SAMPLE_RATE)

    # Run Silero on each 512-sample frame
    print()
    print("Silero VAD per frame (512 samples @ 16kHz):")
    probs = []
    for i in range(0, len(ring) - 512 + 1, 512):
        p = unified_vad.speech_prob(ring[i:i + 512])
        probs.append(p)
    for i, p in enumerate(probs):
        bar = "#" * int(p * 40)
        print(f"  frame {i:2d}: {p:.3f} {bar}")
    if probs:
        print(f"  max={max(probs):.3f} avg={sum(probs)/len(probs):.3f}")

    # Dump raw input for offline analysis
    path = am.dump_raw_input(4.0)
    if path:
        print(f"\nRaw input saved: {path}")

    am.stop()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())