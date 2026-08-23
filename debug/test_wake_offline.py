#!/usr/bin/env python3
"""
Offline acceptance test of the PRODUCTION wake pipeline — no microphone.

Drives the real WakeListener + WakeModelManager (openWakeWord + custom
verifier) + faster-whisper verification through a simulated ring buffer:

  1. "hello Diego" (clean synthetic speech) → trigger → Whisper → ACCEPTED
  2. user's real wake recording (legacy, clipped at capture) → trigger fires
  3. silence → never triggers
  4. espeak "thank you very much" → if it triggers, Whisper REJECTS
  5. stale phrase audio in the ring at WAKE_LISTEN entry → prime() drains
     it → NO false re-trigger (no wake loops)

Run:  python debug/test_wake_offline.py
"""

import logging
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compat  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(message)s")
for n in ("voice.audio_manager", "voice.audio_processing",
          "onnxruntime", "faster_whisper"):
    logging.getLogger(n).setLevel(logging.ERROR)

import voice.wake_listener as wl
from voice.wake_listener import WakeListener
from voice.wake_model_manager import wake_model_manager

ROOT = Path(__file__).resolve().parent.parent


class SimAudio:
    """Ring-buffer double with the real drain semantics: samples are
    appended by feed_* (advancing total_samples, like the capture callback)
    and read_since(last) serves at most 480 samples written AFTER `last`
    (the real callback cadence). prime()'s resync therefore drains exactly
    like production."""

    is_running = True

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0      # read cursor
        self._total = 0    # samples ever written (== write cursor)

    @property
    def total_samples(self):
        return self._total

    def feed_wav(self, path):
        with wave.open(str(path), "rb") as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        self._buf = np.concatenate([self._buf, a.astype(np.float32) / 32768.0])
        self._total += len(a)

    def feed_silence(self, seconds):
        a = np.zeros(int(16000 * seconds), dtype=np.float32)
        self._buf = np.concatenate([self._buf, a])
        self._total += len(a)

    def read_since(self, last):
        # Real RingBuffer semantics: return EVERYTHING written after
        # `last` (the engine's loop polls fast enough that deltas stay
        # small). The returned cursor is the write total, exactly like
        # AudioManager.read_since.
        start = max(self._pos, min(int(last), len(self._buf)))
        chunk = self._buf[start:]
        self._pos = len(self._buf)
        return chunk, self._total

    def get_recent_processed(self, dur):
        n = int(dur * 16000)
        return self._buf[max(0, self._pos - n):self._pos].copy()

    def get_recent_audio(self, dur):
        return self.get_recent_processed(dur)


def espeak_wav(phrase) -> str:
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["espeak", "-w", tmp, phrase], capture_output=True, timeout=15)
    with wave.open(tmp, "rb") as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr != 16000:
        from scipy import signal as sc
        g = math.gcd(16000, sr)
        a = sc.resample_poly(a.astype(np.float64), 16000 // g, sr // g).astype(np.int16)
    with wave.open(tmp, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(a.tobytes())
    return tmp


def main() -> int:
    sim = SimAudio()
    wl.audio_manager = sim
    wl.wake_model_manager = wake_model_manager

    listener = WakeListener()
    listener.vad._ready = False  # gate open (Silero is exercised separately)
    assert wake_model_manager.load(), "wake model load failed"
    from voice.streaming_stt import streaming_stt
    assert streaming_stt.initialize(), "whisper load failed"

    def run(feed, max_chunks=800):
        """prime() (drains stale audio), THEN feed fresh audio, then drive
        process() until a verification runs.
        Returns ('ACCEPTED'|'REJECTED'|'no-trigger', (score, transcript))."""
        listener.prime()
        listener._refractory_until = 0.0  # offline fast-forward
        feed()
        for _ in range(max_chunks):
            chunk, listener._last_total = sim.read_since(listener._last_total)
            if len(chunk) == 0:
                # Room tone keeps the trigger-settle logic moving.
                chunk = np.zeros(480, dtype=np.float32)
            trigger = listener.process(chunk)
            if trigger is not None:
                name, score = trigger
                listener._last_verify = 0.0
                verified, transcript = listener.verify_with_whisper()
                return ("ACCEPTED" if verified else "REJECTED"), (score, transcript)
        return "no-trigger", None

    results = []

    print("\n[1] 'hello Diego' (clean synthetic speech — full path incl. Whisper) …")
    def feed_wake():
        sim.feed_wav(espeak_wav("hello Diego"))
        sim.feed_silence(1.0)   # room tone after the phrase (settles trigger)
    r, d = run(feed_wake)
    print(f"    → {r}  score={d[0]:.3f} transcript={d[1]!r}" if d
          else f"    → {r}")
    results.append(r == "ACCEPTED")

    print("[2] user's real wake recording (legacy clipped capture) …")
    def feed_real():
        sim.feed_wav(ROOT / "models" / "wake" / "wake_phrase.wav")
        sim.feed_silence(1.0)
    r, d = run(feed_real)
    print(f"    → {r}  score={d[0]:.3f} transcript={d[1]!r}" if d
          else f"    → {r}")
    # The acoustic trigger MUST fire; Whisper may not transcribe the
    # legacy clipped recording (capture artifact, not a pipeline bug).
    results.append(d is not None and d[0] >= wake_model_manager.threshold)

    print("[3] pure silence …")
    r, _ = run(lambda: sim.feed_silence(3.0))
    print(f"    → {r}  (want no-trigger)")
    results.append(r == "no-trigger")

    print("[4] espeak 'thank you very much' …")
    def feed_esp():
        sim.feed_wav(espeak_wav("thank you very much"))
        sim.feed_silence(1.0)
    r, d = run(feed_esp)
    print(f"    → {r} {d if d else ''} (want no-trigger or REJECTED)")
    results.append(r in ("no-trigger", "REJECTED"))

    print("[5] stale phrase audio in ring at WAKE_LISTEN entry …")
    # The phrase sits in the ring BEFORE prime() (e.g. the tail of the
    # previous interaction). prime() must drain it: no trigger may fire.
    sim.feed_wav(ROOT / "models" / "wake" / "wake_phrase.wav")
    listener.prime()
    listener._refractory_until = 0.0
    triggered = False
    for _ in range(200):
        chunk, listener._last_total = sim.read_since(listener._last_total)
        if len(chunk) == 0:
            chunk = np.zeros(480, dtype=np.float32)
        if listener.process(chunk) is not None:
            triggered = True
            break
    print(f"    → {'FALSE RE-TRIGGER (BAD)' if triggered else 'no re-trigger (GOOD)'}")
    results.append(not triggered)

    ok = all(results)
    print("\nRESULT:", "ALL PASS" if ok else f"FAIL {results}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
