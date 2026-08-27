"""
Offline diagnostic: run Silero VAD on REAL microphone WAV files
(verification_*.wav) captured during actual wake/command sessions.

This isolates whether Silero returns 0.0 on real audio CONTENT
(processing/format issue) or works fine (state/concurrency issue).

Usage:
    python debug/diag_offline_vad.py [wav_path ...]
    (defaults to the most recent verification_*.wav files)
"""

import logging
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.vad import unified_vad, VAD_FRAME_SAMPLES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def load_wav(path: Path) -> np.ndarray:
    """Load a WAV file as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch)[:, 0]
    return a


def main():
    # Default: the most recent verification WAVs
    dbg = Path(__file__).resolve().parent.parent / "debug"
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        wavs = sorted(dbg.glob("verification_*.wav"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        paths = wavs[:5]

    if not paths:
        print("No WAV files found")
        return 1

    unified_vad.load()
    print(f"Silero ready={unified_vad.ready} backend={unified_vad.get_diagnostics()['backend']}")
    print()

    for path in paths:
        if not path.exists():
            print(f"SKIP {path.name} (missing)")
            continue
        audio = load_wav(path)
        a64 = audio.astype(np.float64)
        rms = float(np.sqrt(np.mean(a64 ** 2)))
        peak = float(np.max(np.abs(a64)))
        dur = len(audio) / 16000.0

        # Run Silero on each 512-sample frame
        probs = []
        for i in range(0, len(audio) - 512 + 1, 512):
            p = unified_vad.speech_prob(audio[i:i + 512])
            probs.append(p)

        n_above = sum(1 for p in probs if p >= 0.5)
        print(f"{path.name}: dur={dur:.2f}s rms={rms:.6f} (int16={rms*32768:.1f}) "
              f"peak={peak:.6f} frames={len(probs)} "
              f"max={max(probs):.3f} avg={sum(probs)/len(probs):.3f} "
              f"above_0.5={n_above}/{len(probs)}")
        # Show first 20 frame probs
        print(f"  probs[:20]={[round(p,3) for p in probs[:20]]}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())