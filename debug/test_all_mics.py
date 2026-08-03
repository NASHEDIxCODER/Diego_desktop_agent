#!/usr/bin/env python3
"""
test_all_mics.py — STEP 7: test EVERY microphone on the system.

For every input device:
  - opens a stream
  - records 3 seconds
  - writes debug/device_<index>.wav
  - prints RMS, Peak, Speech %

Finally prints BEST DEVICE (highest speech energy).

Usage:
    python debug/test_all_mics.py
    python debug/test_all_mics.py --duration 5
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.audio_manager import (  # noqa: E402
    enumerate_input_devices,
    print_device_table,
    _signal_metrics,
    REJECT_KEYWORDS,
)

DEBUG_DIR = Path(__file__).resolve().parent
RECORD_DURATION = 3.0


def record_device(sd, target, samplerate: int, channels: int,
                  duration: float) -> np.ndarray:
    """Record `duration` seconds from a device. Returns int16 array (n, ch).

    `target` is the device index, or the exact device NAME for
    capture-by-name physical codecs (PipeWire hides them as in=0).
    """
    try:
        rec = sd.rec(int(duration * samplerate), samplerate=samplerate,
                     channels=channels, dtype="int16", device=target,
                     blocking=True)
        data = np.asarray(rec)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception:
        # Retry mono if the native channel count failed.
        if channels > 1:
            rec = sd.rec(int(duration * samplerate), samplerate=samplerate,
                         channels=1, dtype="int16", device=target,
                         blocking=True)
            data = np.asarray(rec)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data
        raise



def speech_percent(mono: np.ndarray, samplerate: int) -> float:
    """% of 30 ms frames whose RMS exceeds a speech floor (int16 scale)."""
    frame = int(0.03 * samplerate)
    if mono.size < frame:
        return 0.0
    n_frames = mono.size // frame
    frames = mono[: n_frames * frame].reshape(n_frames, frame).astype(np.float64)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Speech floor: well above digital silence, below normal speech.
    return float(np.mean(frame_rms > 150.0) * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test every microphone")
    parser.add_argument("--duration", type=float, default=RECORD_DURATION,
                        help="seconds to record per device (default 3)")
    args = parser.parse_args()
    duration = args.duration

    import sounddevice as sd

    print()
    print("  ══════════════════════════════════════════════════════════════════")
    print("  TEST ALL MICROPHONES")
    print("  ══════════════════════════════════════════════════════════════════")

    devices = enumerate_input_devices(sd)
    print_device_table(devices)

    if not devices:
        print("  No input devices found.")
        return 1

    results = []
    for dev in devices:
        idx = dev["index"]
        name = dev["name"]
        samplerate = int(dev["default_samplerate"]) or 16000
        by_name = bool(dev.get("capture_by_name"))
        # Capture-by-name codecs open by exact NAME and report in=0.
        target = name if by_name else idx
        channels = 2 if by_name else max(1, min(int(dev["max_input_channels"]), 4))
        virtual = any(k in name.lower() for k in REJECT_KEYWORDS)

        print(f"  ── [{idx}] {name}")
        print(f"     recording {duration:.0f}s @ {samplerate} Hz, {channels} ch"
              f"{'  (virtual bus)' if virtual else ''}"
              f"{'  (PHYSICAL, by name)' if by_name else ''} ...")

        try:
            data = record_device(sd, target, samplerate, channels, duration)

        except Exception as e:
            print(f"     ERROR: {e}")
            results.append({"index": idx, "name": name, "ok": False,
                            "rms": 0.0, "peak": 0.0, "speech_pct": 0.0,
                            "speech_energy": 0.0, "error": str(e)})
            continue

        # Analyze the strongest channel.
        per_ch = [_signal_metrics(data[:, c], samplerate) for c in range(data.shape[1])]
        best_c = int(np.argmax([m["rms"] for m in per_ch]))
        mono = data[:, best_c]
        metrics = per_ch[best_c]
        spct = speech_percent(mono, samplerate)

        # Write device_<index>.wav (best channel, native rate).
        wav_path = DEBUG_DIR / f"device_{idx}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(samplerate)
            w.writeframes(mono.astype(np.int16).tobytes())

        speech_energy = metrics["rms"] * (0.5 + 0.5 * metrics["voice_probability"])
        results.append({
            "index": idx, "name": name, "ok": True,
            "rms": metrics["rms"], "peak": metrics["peak"],
            "speech_pct": spct, "speech_energy": speech_energy,
            "voice_probability": metrics["voice_probability"],
            "speech_channel": best_c, "wav": str(wav_path),
        })
        print(f"     → {wav_path.name}  RMS={metrics['rms']:.1f}  "
              f"Peak={metrics['peak']:.0f}  Speech={spct:.1f}%  "
              f"(speech_channel={best_c})")

    # ── Summary ──
    print()
    print("  ══════════════════════════════════════════════════════════════════")
    print("  SUMMARY")
    print("  ══════════════════════════════════════════════════════════════════")
    print(f"  {'IDX':<4} {'DEVICE':<45} {'RMS':>9} {'PEAK':>8} {'SPEECH%':>8}")
    print("  " + "─" * 76)
    for r in results:
        if r["ok"]:
            print(f"  [{r['index']:<2}] {r['name']:<45} {r['rms']:>9.1f} "
                  f"{r['peak']:>8.0f} {r['speech_pct']:>7.1f}%")
        else:
            print(f"  [{r['index']:<2}] {r['name']:<45} {'ERROR':>9} "
                  f"{'—':>8} {'—':>8}")

    working = [r for r in results if r["ok"] and r["rms"] >= 1.0 and r["peak"] >= 8.0]
    print()
    if working:
        best = max(working, key=lambda r: r["speech_energy"])
        print(f"  BEST DEVICE: [{best['index']}] {best['name']}  "
              f"RMS={best['rms']:.1f}  Speech={best['speech_pct']:.1f}%")
        return 0
    print("  BEST DEVICE: NONE — No working microphone detected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
