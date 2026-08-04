#!/usr/bin/env python3
"""
Wake-detection TEST MODE — live diagnostics over the PRODUCTION pipeline.

    python debug/test_wake.py

Drives the SAME code as `python main.py` (AudioManager → WakeListener:
Silero VAD → openWakeWord → Whisper verification) and prints one line
per second:

    Mic RMS | Peak | VAD | Wake score | Transcript | Decision

No GUI. Ctrl+C stops. When a wake is accepted the listener re-primes and
keeps monitoring (deterministic re-entry, exactly like the engine).
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compat  # noqa: F401

# Quiet libraries; the per-second status line carries the information.
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("voice.audio_manager", "voice.audio_processing", "onnxruntime"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
# The wake pipeline's own transitions stay visible.
logging.getLogger("voice.wake_listener").setLevel(logging.INFO)
logging.getLogger("voice.wake_model_manager").setLevel(logging.INFO)


def main() -> int:
    from voice.audio_manager import audio_manager
    from voice.wake_listener import WakeListener
    from voice.wake_model_manager import wake_model_manager

    print("\n  ════════════════════════════════════════════════════════")
    print("  WAKE TEST MODE — say 'Hello Leo'. Ctrl+C to stop.")
    print("  ════════════════════════════════════════════════════════\n")

    if not audio_manager.start():
        print("  FAILED: no working microphone (see report above)")
        return 1

    listener = WakeListener()
    listener.load()                       # Silero VAD gate
    if not wake_model_manager.loaded:
        wake_model_manager.load()
    listener.prime()                      # deterministic clean entry

    # Whisper is preloaded so the first verification doesn't miss the
    # wake-phrase window (same as the engine's BOOT).
    from voice.streaming_stt import streaming_stt
    streaming_stt.initialize()

    print(f"  Model='{wake_model_manager.model_name}' "
          f"threshold={wake_model_manager.threshold:.2f} "
          f"VAD={'on' if listener.vad.ready else 'off'}\n")
    header = ("  %-8s %-8s %-8s %-10s %-24s %s"
              % ("MicRMS", "Peak", "VAD", "WakeScore", "Transcript", "Decision"))
    print(header)
    print("  " + "─" * (len(header) - 2))

    last_print = 0.0
    try:
        while True:
            new_audio, listener._last_total = audio_manager.read_since(
                listener._last_total)
            if len(new_audio) == 0:
                time.sleep(0.02)
                continue

            trigger = listener.process(new_audio)
            if trigger is not None:
                model_name, score = trigger
                listener._last_verify = time.monotonic()
                print(f"\n  ── trigger (score={score:.3f}) — verifying…")
                verified, transcript = listener.verify_with_whisper()
                if verified:
                    listener.last_decision = "WAKE ACCEPTED"
                    print(f"  ── WAKE ACCEPTED (transcript='{transcript}') — "
                          f"re-priming listener\n")
                    listener.prime()
                else:
                    listener.last_decision = "rejected"
                    print(f"  ── wake rejected (transcript='{transcript}')\n")
                    listener._cooldown_until = time.monotonic() + 2.0

            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                rms = float(np.sqrt(np.mean(new_audio.astype(np.float64) ** 2))) * 32768.0
                peak = float(np.max(np.abs(new_audio))) * 32768.0
                print("  %-8.0f %-8.0f %-8.2f %-10.3f %-24s %s"
                      % (rms, peak, listener.last_vad, listener.last_score,
                         (listener.last_transcript or "—")[:24],
                         listener.last_decision))
    except KeyboardInterrupt:
        print("\n\n  Stopped.")
    finally:
        audio_manager.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
