#!/usr/bin/env python3
"""
Audio Diagnostics for Leo Desktop Assistant.

Measures real microphone characteristics:
- Device enumeration (all microphones)
- Noise floor, SNR, RMS, clipping
- Latency measurements
- CPU usage
- Ring buffer status

Usage:
    python debug/audio_diagnostics.py
"""

import os
import sys
import time
import platform
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"

import compat  # noqa: F401


def get_cpu_usage() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.5)
    except ImportError:
        return 0.0


def get_process_memory() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def measure_device(device_idx: int, name: str, duration: float = 3.0) -> dict:
    """Record from a device and measure its characteristics."""
    import sounddevice as sd

    result = {
        "device_index": device_idx,
        "name": name,
        "error": None,
    }

    try:
        dev_info = sd.query_devices(device_idx)
        sample_rate = int(dev_info['default_samplerate'])
        channels = dev_info['max_input_channels']

        # Record
        t0 = time.time()
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            device=device_idx,
            blocking=True,
        )
        latency = time.time() - t0

        audio = recording.flatten().astype(np.float64)
        result["latency_s"] = latency
        result["sample_rate"] = sample_rate
        result["channels"] = channels

        # Measurements
        peak = np.max(np.abs(audio))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        noise_floor = float(np.percentile(np.abs(audio), 5))  # 5th percentile
        speech_rms = float(np.percentile(np.abs(audio), 95))  # 95th percentile
        snr_db = 20 * np.log10((speech_rms + 1e-10) / (noise_floor + 1e-10)) if noise_floor > 0 else 0

        # Clipping percentage
        clipping = float(np.mean(np.abs(audio) >= 32760) * 100)

        # Silence percentage (below 1% of peak)
        silence_thresh = max(50, peak * 0.01)
        silence = float(np.mean(np.abs(audio) < silence_thresh) * 100)

        # Speech detection percentage (above noise floor * 3)
        speech_thresh = max(noise_floor * 3, 100)
        speech_pct = float(np.mean(np.abs(audio) >= speech_thresh) * 100)

        result["peak"] = peak
        result["rms"] = rms
        result["noise_floor"] = noise_floor
        result["speech_rms"] = speech_rms
        result["snr_db"] = snr_db
        result["clipping_pct"] = clipping
        result["silence_pct"] = silence
        result["speech_pct"] = speech_pct
        result["gain"] = 1.0  # No preamp on raw device

    except Exception as e:
        result["error"] = str(e)

    return result


def enumerate_microphones():
    """Enumerate all input devices."""
    import sounddevice as sd
    devices = []
    for d in sd.query_devices():
        if d['max_input_channels'] > 0:
            devices.append({
                "index": d['index'],
                "name": d['name'],
                "channels": d['max_input_channels'],
                "samplerate": d['default_samplerate'],
                "is_default": False,
            })
    try:
        default_idx = sd.default.device[0]
        for d in devices:
            d["is_default"] = d["index"] == default_idx
    except Exception:
        pass
    return devices


def main():
    print()
    print("=" * 70)
    print("  LEO AUDIO DIAGNOSTICS")
    print("=" * 70)
    print()

    # ── System info ─────────────────────────────────
    print("[SYSTEM]")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  Platform:   {platform.platform()}")
    print(f"  CPU usage:  {get_cpu_usage():.1f}%")
    print(f"  Memory:     {get_process_memory():.1f} MB")
    print()

    # ── Audio backend ───────────────────────────────
    print("[BACKEND]")
    try:
        import sounddevice as sd
        print(f"  sounddevice: {sd.__version__}")
        print(f"  PortAudio:   {sd.get_portaudio_version()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Check PipeWire
    try:
        result = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Server Name' in line or 'Server Version' in line:
                    print(f"  {line.strip()}")
    except Exception:
        pass
    print()

    # ── Microphone enumeration ──────────────────────
    print("[MICROPHONES]")
    mics = enumerate_microphones()
    if not mics:
        print("  NO INPUT DEVICES FOUND")
        sys.exit(1)

    for m in mics:
        default_mark = "  ← DEFAULT" if m["is_default"] else ""
        print(f"  [{m['index']}] {m['name']} "
              f"(ch={m['channels']}, sr={m['samplerate']}Hz){default_mark}")
    print()

    # ── Measure each microphone ─────────────────────
    print("[MEASUREMENTS]")
    print("  Recording 3 seconds from each device...")
    print()

    measurements = []
    for m in mics:
        print(f"  Recording from [{m['index']}] {m['name']}...")
        meas = measure_device(m["index"], m["name"])
        measurements.append(meas)
        if meas.get("error"):
            print(f"    ERROR: {meas['error']}")
        else:
            print(f"    Latency:       {meas.get('latency_s', 0)*1000:.1f} ms")
            print(f"    Sample rate:   {meas.get('sample_rate', 0)} Hz")
            print(f"    Peak:          {meas.get('peak', 0):.0f}")
            print(f"    RMS:           {meas.get('rms', 0):.1f}")
            print(f"    Noise floor:   {meas.get('noise_floor', 0):.1f}")
            print(f"    Speech RMS:    {meas.get('speech_rms', 0):.1f}")
            print(f"    SNR:           {meas.get('snr_db', 0):.1f} dB")
            print(f"    Clipping:      {meas.get('clipping_pct', 0):.1f}%")
            print(f"    Silence:       {meas.get('silence_pct', 0):.1f}%")
            print(f"    Speech:        {meas.get('speech_pct', 0):.1f}%")
        print()

    # ── AudioManager state ──────────────────────────
    print("[AUDIOMANAGER]")
    try:
        from voice.audio_manager import audio_manager
        if audio_manager.start():
            diag = audio_manager.get_diagnostics()
            print(f"  Backend:       {diag['backend']}")
            print(f"  Device:        [{diag['device_index']}]")
            print(f"  Sample rate:   {diag['sample_rate']} Hz")
            print(f"  Buffer:        {diag['buffer_seconds']:.1f}s")
            print(f"  Threshold:     {diag['energy_threshold']:.1f}")
            print(f"  VAD state:     {diag['vad_state']}")
            preproc = diag.get('preprocessor', {})
            print(f"  Preprocessing: {preproc.get('processed_count', 0)} frames processed")
            print(f"  Noise floor:   {preproc.get('noise_floor_db', 'N/A')} dB")
            audio_manager.stop()
        else:
            print("  FAILED to start AudioManager")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # ── Summary ─────────────────────────────────────
    print("[SUMMARY]")
    if measurements:
        # Find best microphone by SNR
        valid = [m for m in measurements if not m.get("error")]
        if valid:
            best = max(valid, key=lambda m: m.get("snr_db", 0))
            print(f"  Best microphone: [{best['device_index']}] {best['name']}")
            print(f"    SNR: {best.get('snr_db', 0):.1f} dB")
            print(f"    Noise floor: {best.get('noise_floor', 0):.1f}")
            print()
            if best.get('snr_db', 0) < 10:
                print("  ⚠  WARNING: Low SNR — the microphone environment is very noisy.")
                print("     Leo will apply noise suppression before wake detection.")
            if best.get('snr_db', 0) < 3:
                print("  ⚠  CRITICAL: Extremely low SNR — speech may not be detectable.")
                print("     Try using a different microphone or reducing background noise.")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()