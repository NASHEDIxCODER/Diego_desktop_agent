"""
Microphone Selector — Scores and selects the best microphone for Leo.

Never blindly uses the default device. Enumerates every input device,
measures noise/signal characteristics, and scores each microphone.

Scoring criteria (higher = better):
  - SNR (weight 40%)
  - Noise floor (lower = better, weight 20%)
  - Latency (lower = better, weight 10%)
  - Clipping (lower = better, weight 20%)
  - Voice activity level (weight 10%)

The best microphone is automatically selected at startup.

If multiple microphones exist, allow manual selection with:
    python main.py --select-mic

The selection is saved in data/mic_selection.json.
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Selection storage
SELECTION_PATH = Path(__file__).resolve().parent.parent / "data" / "mic_selection.json"

# Measurement duration
MEASURE_DURATION = 2.0

# Scoring weights
WEIGHTS = {
    "snr": 0.40,
    "noise_floor": 0.20,
    "clipping": 0.20,
    "latency": 0.10,
    "activity": 0.10,
}

# Prefer hardware devices over virtual ones
HARDWARE_KEYWORDS = ["hw:", "usb", "analog", "stereo", "mic", "microphone"]
VIRTUAL_KEYWORDS = ["pipewire", "pulse", "default", "monitor"]


def load_saved_selection() -> Optional[Dict[str, Any]]:
    """Load saved microphone selection."""
    try:
        if SELECTION_PATH.exists():
            with open(SELECTION_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.debug("Failed to load mic selection: %s", e)
    return None


def save_selection(selection: Dict[str, Any]) -> None:
    """Save microphone selection."""
    try:
        SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SELECTION_PATH, "w") as f:
            json.dump(selection, f, indent=2)
        logger.info("Microphone selection saved: %s", selection)
    except Exception as e:
        logger.warning("Failed to save mic selection: %s", e)


def enumerate_input_devices() -> List[Dict[str, Any]]:
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
                "is_hardware": any(kw.lower() in str(d['name']).lower() for kw in HARDWARE_KEYWORDS),
                "is_virtual": any(kw.lower() in str(d['name']).lower() for kw in VIRTUAL_KEYWORDS),
            })
    return devices


def measure_device(device: Dict[str, Any], duration: float = MEASURE_DURATION) -> Dict[str, Any]:
    """
    Record from a device and measure its characteristics.

    Returns:
        Device entry with metrics:
        - noise_floor: 5th percentile abs value
        - speech_rms: 95th percentile abs value
        - snr_db: speech/noise ratio
        - clipping_pct: % of samples at max level
        - latency_s: record blocking time (already duration-scaled)
        - rms: overall RMS
    """
    import sounddevice as sd

    result = dict(device)
    result["error"] = None

    try:
        sample_rate = int(device["samplerate"])
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device["index"],
            blocking=True,
        )

        audio = recording.flatten().astype(np.float64)

        # Basic metrics
        result["rms"] = float(np.sqrt(np.mean(audio ** 2)))
        result["peak"] = float(np.max(np.abs(audio)))

        # Noise floor (5th percentile) — low amplitude noise
        result["noise_floor"] = float(np.percentile(np.abs(audio), 5))

        # Speech RMS (95th percentile) — high amplitude speech
        result["speech_rms"] = float(np.percentile(np.abs(audio), 95))

        # SNR
        if result["noise_floor"] > 1.0:
            result["snr_db"] = 20 * np.log10(
                (result["speech_rms"] + 1e-10) / (result["noise_floor"] + 1e-10)
            )
        else:
            result["snr_db"] = 40.0  # Very quiet noise floor

        # Clipping percentage
        result["clipping_pct"] = float(np.mean(np.abs(audio) >= 32760) * 100)

        # Latency (blocking record duration, should be ~duration)
        result["latency_s"] = len(audio) / sample_rate

        # Activity: % of samples above noise floor * 3
        threshold = max(result["noise_floor"] * 3, 50)
        result["activity_pct"] = float(np.mean(np.abs(audio) >= threshold) * 100)

    except Exception as e:
        result["error"] = str(e)

    return result


def score_device(device: Dict[str, Any]) -> float:
    """
    Score a device from 0-1 with 1 being best.

    Weights:
      - SNR (40%): 60 dB = 1.0, 0 dB = 0.0
      - Noise floor (20%): <10 = 1.0, >1000  = 0.0 (inverse log scale)
      - Clipping (20%): 0% = 1.0, 10% = 0.0
      - Latency (10%): lower = better
      - Activity (10%): 30% = 1.0
    """
    if device.get("error"):
        return 0.0

    # SNR score (0-60+ dB mapped to 0-1)
    snr = max(0, min(60, device.get("snr_db", 0)))
    snr_score = snr / 60.0

    # Noise floor score (inverse log scale)
    nf = max(0.1, device.get("noise_floor", 0))
    nf_score = max(0, 1 - math.log10(nf) / 4)  # log10(10)=1, log10(10000)=4

    # Clipping score (0% = 1.0, 10% = 0.0)
    clip = min(10, device.get("clipping_pct", 0))
    clip_score = 1.0 - clip / 10.0

    # Latency score (1s = 1.0, 10s = 0.0)
    latency = device.get("latency_s", 0)
    latency_score = max(0, 1 - latency / 10.0)

    # Activity score (30% = 1.0)
    activity = min(30, device.get("activity_pct", 0))
    activity_score = activity / 30.0

    # Prefer hardware devices
    hardware_bonus = 0.05 if device.get("is_hardware") else 0.0

    total = (
        snr_score * WEIGHTS["snr"] +
        nf_score * WEIGHTS["noise_floor"] +
        clip_score * WEIGHTS["clipping"] +
        latency_score * WEIGHTS["latency"] +
        activity_score * WEIGHTS["activity"]
    ) + hardware_bonus

    return max(0, min(1.0, total))


def select_best_microphone(force_measure: bool = False) -> Dict[str, Any]:
    """
    Score all microphones and select the best one.

    CRITICAL PRIORITY:
      1. Hardware microphones (hw:, USB, analog) — ALWAYS preferred
      2. Virtual devices (pipewire, pulse, default) — ONLY if no hardware exists

    Never choose a virtual device when a real hardware microphone exists.

    Returns:
        Device dict with index, name, and metrics.
    """
    # Check saved selection first (but only if it's a hardware device)
    saved = load_saved_selection()
    if saved and not force_measure:
        saved_idx = saved.get("index")
        devices = enumerate_input_devices()
        for d in devices:
            if d["index"] == saved_idx:
                # Only use saved selection if it's hardware or no hardware exists
                hardware_exists = any(dd["is_hardware"] for dd in devices)
                if d["is_hardware"] or not hardware_exists:
                    logger.info("[MIC-SELECT] Using saved selection: [%d] %s",
                                saved_idx, saved.get("name", d["name"]))
                    return saved
                else:
                    logger.info("[MIC-SELECT] Saved selection [%d] is virtual, but hardware exists — re-selecting",
                                saved_idx)
                    break

    # Enumerate and measure all devices
    devices = enumerate_input_devices()
    if not devices:
        logger.error("[MIC-SELECT] No input devices found")
        return {}

    # CRITICAL: Separate hardware from virtual devices
    hardware_devices = [d for d in devices if d["is_hardware"]]
    virtual_devices = [d for d in devices if not d["is_hardware"]]

    # Only measure hardware devices if any exist
    devices_to_measure = hardware_devices if hardware_devices else virtual_devices
    if hardware_devices:
        logger.info("[MIC-SELECT] %d hardware microphone(s) found — ignoring virtual devices",
                    len(hardware_devices))
    else:
        logger.warning("[MIC-SELECT] No hardware microphones found — falling back to virtual devices")

    measured = []
    for d in devices_to_measure:
        m = measure_device(d)
        score = score_device(m)
        m["score"] = score
        measured.append(m)
        logger.debug("[MIC-SELECT] [%d] %s: SNR=%.1fdB NF=%.1f Clip=%.1f%% Lat=%.1fs Score=%.3f",
                     m["index"], m["name"],
                     m.get("snr_db", 0), m.get("noise_floor", 0),
                     m.get("clipping_pct", 0), m.get("latency_s", 0), score)

    # Score and pick best from the measured set
    measured_sorted = sorted(measured, key=lambda m: m.get("score", 0), reverse=True)
    best = measured_sorted[0] if measured_sorted else {}

    if best:
        logger.info("[MIC-SELECT] Best microphone: [%d] %s (score=%.3f, type=%s)",
                    best.get("index"), best.get("name"), best.get("score", 0),
                    "HARDWARE" if best.get("is_hardware") else "VIRTUAL")

        # Save selection
        selection = {
            "index": best.get("index"),
            "name": best.get("name"),
            "score": best.get("score"),
            "snr_db": best.get("snr_db"),
            "noise_floor": best.get("noise_floor"),
            "clipping_pct": best.get("clipping_pct"),
            "is_hardware": best.get("is_hardware", False),
            "measured_at": time.time(),
        }
        save_selection(selection)

    return best


def list_microphones_for_selection() -> None:
    """Print all microphones with metrics for user selection."""
    devices = enumerate_input_devices()
    if not devices:
        print("  No input devices found")
        return

    print()
    print("  Available Microphones:")
    print("  " + "-" * 70)
    print(f"  {'ID':<4} {'Name':<45} {'Type':<10}")
    print("  " + "-" * 70)

    for d in devices:
        dev_type = "HARDWARE" if d["is_hardware"] else ("VIRTUAL" if d["is_virtual"] else "OTHER")
        print(f"  [{d['index']:<2}] {d['name']:<45} {dev_type:<10}")
    print("  " + "-" * 70)
    print()


def select_microphone_interactive() -> Dict[str, Any]:
    """
    Interactively select a microphone.

    Prints:
      ID, Name, Backend, Channels, Noise, SNR, Latency

    Saves selection to data/mic_selection.json.
    """
    devices = enumerate_input_devices()
    if not devices:
        print("  No input devices found")
        return {}

    print()
    print("  ═══════════════════════════════════════════════")
    print("  MICROPHONE SELECTION")
    print("  ═══════════════════════════════════════════════")
    print()

    measured = []
    for d in devices:
        m = measure_device(d)
        score = score_device(m)
        m["score"] = score
        measured.append(m)

        status = "OK" if not m.get("error") else f"ERROR: {m['error']}"
        print(f"  [{m['index']}] {m['name']}")
        print(f"      Channels: {m.get('channels', 0)} | Noise: {m.get('noise_floor', 0):.1f} | "
              f"SNR: {m.get('snr_db', 0):.1f}dB | Latency: {m.get('latency_s', 0):.1f}s | "
              f"Clip: {m.get('clipping_pct', 0):.1f}% | Score: {score:.3f}")
        if m.get("error"):
            print(f"      ERROR: {m['error']}")
        print()

    # Sort by score
    measured_sorted = sorted(measured, key=lambda m: m.get("score", 0), reverse=True)
    best = measured_sorted[0] if measured_sorted else {}

    print(f"  Best: [{best.get('index')}] {best.get('name')} (score={best.get('score', 0):.3f})")
    print()

    while True:
        try:
            user_input = input("  Select device ID (or press Enter for best): ").strip()
            if not user_input:
                selected = best
                break
            selected_idx = int(user_input)
            selected = next((m for m in measured if m["index"] == selected_idx), None)
            if selected is None:
                print(f"  Invalid device ID: {selected_idx}")
                continue
            break
        except (ValueError, KeyboardInterrupt):
            print("  Invalid input")
            continue
        except EOFError:
            selected = best
            break

    if selected:
        selection = {
            "index": selected.get("index"),
            "name": selected.get("name"),
            "score": selected.get("score"),
            "snr_db": selected.get("snr_db"),
            "noise_floor": selected.get("noise_floor"),
            "clipping_pct": selected.get("clipping_pct"),
            "measured_at": time.time(),
        }
        save_selection(selection)
        print(f"\n  ✓ Selected microphone: [{selected.get('index')}] {selected.get('name')}")

    return selected


# Global singleton
mic_selector = None  # Lazily initialized