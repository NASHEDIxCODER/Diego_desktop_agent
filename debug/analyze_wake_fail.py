#!/usr/bin/env python3
"""
Wake Fail Audio Analyzer for Leo Desktop Assistant.

Analyzes every saved wake_fail.wav file and computes:
  - Maximum amplitude
  - Histogram
  - Dynamic range
  - Spectrogram
  - Crest factor
  - Signal-to-noise ratio
  - Voice activity percentage
  - Reason why STT failed

Usage:
    python debug/analyze_wake_fail.py
    python debug/analyze_wake_fail.py --file debug/wake_fail_20260730_142839_clipping.wav
"""

import argparse
import os
import sys
import wave
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


def analyze_wav(filepath: Path) -> dict:
    """Analyze a WAV file and compute all diagnostic metrics."""
    result = {
        "file": str(filepath),
        "error": None,
    }

    try:
        with wave.open(str(filepath), "rb") as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            raw = wav.readframes(n_frames)

        if sampwidth != 2:
            result["error"] = f"Unsupported sample width: {sampwidth} bytes"
            return result

        samples = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            samples = samples[::n_channels]  # Take first channel

        duration = len(samples) / framerate
        result["duration_s"] = round(duration, 3)
        result["sample_rate"] = framerate
        result["channels"] = n_channels
        result["num_samples"] = len(samples)

        # ── Maximum amplitude ─────────────────────────────
        peak = float(np.max(np.abs(samples)))
        result["peak"] = peak
        result["peak_db"] = round(20 * np.log10(peak / 32768.0 + 1e-10), 2)

        # ── RMS ───────────────────────────────────────────
        rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
        result["rms"] = round(rms, 2)
        result["rms_db"] = round(20 * np.log10(rms / 32768.0 + 1e-10), 2)

        # ── Histogram ─────────────────────────────────────
        hist, bin_edges = np.histogram(samples, bins=32, range=(-32768, 32768))
        result["histogram"] = {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist(),
        }

        # ── Dynamic range ─────────────────────────────────
        # Dynamic range = 20*log10(peak / noise_floor)
        # Noise floor = 5th percentile of absolute values
        noise_floor = float(np.percentile(np.abs(samples), 5))
        result["noise_floor"] = round(noise_floor, 2)
        if noise_floor > 0:
            result["dynamic_range_db"] = round(20 * np.log10(peak / noise_floor), 2)
        else:
            result["dynamic_range_db"] = None

        # ── Crest factor ──────────────────────────────────
        # Crest factor = peak / RMS
        if rms > 0:
            result["crest_factor"] = round(peak / rms, 2)
            result["crest_factor_db"] = round(20 * np.log10(peak / rms), 2)
        else:
            result["crest_factor"] = None
            result["crest_factor_db"] = None

        # ── Signal-to-noise ratio ─────────────────────────
        # SNR = 20*log10(speech_rms / noise_floor)
        # speech_rms = 95th percentile of absolute values
        speech_rms = float(np.percentile(np.abs(samples), 95))
        result["speech_rms"] = round(speech_rms, 2)
        if noise_floor > 0:
            result["snr_db"] = round(20 * np.log10(speech_rms / noise_floor), 2)
        else:
            result["snr_db"] = None

        # ── Voice activity percentage ─────────────────────
        # Voice = samples above noise_floor * 3
        voice_thresh = max(noise_floor * 3, 100)
        voice_pct = float(np.mean(np.abs(samples) >= voice_thresh) * 100)
        result["voice_activity_pct"] = round(voice_pct, 2)

        # ── Clipping ──────────────────────────────────────
        clipping_pct = float(np.mean(np.abs(samples) >= 32760) * 100)
        result["clipping_pct"] = round(clipping_pct, 4)

        # ── Spectrogram (summary stats) ───────────────────
        try:
            from scipy import signal as scipy_signal
            f, t, Sxx = scipy_signal.spectrogram(
                samples.astype(np.float64) / 32768.0,
                fs=framerate,
                nperseg=256,
                noverlap=128,
            )
            # Compute spectral centroid
            freqs = f
            power = Sxx
            total_power = np.sum(power, axis=0)
            centroid = np.sum(freqs[:, np.newaxis] * power, axis=0) / (total_power + 1e-10)
            result["spectral_centroid_hz"] = round(float(np.mean(centroid)), 2)
            result["spectral_bandwidth_hz"] = round(float(np.std(centroid)), 2)
            result["spectrogram_shape"] = list(Sxx.shape)
        except Exception as e:
            result["spectrogram_error"] = str(e)

        # ── Reason classification ─────────────────────────
        if clipping_pct > 0.0:
            result["reason"] = f"clipping_{clipping_pct:.1f}pct"
        elif rms < 300:
            result["reason"] = "too_quiet"
        elif result.get("snr_db") is not None and result["snr_db"] < 3:
            result["reason"] = "low_snr"
        else:
            result["reason"] = "stt_failed"

    except Exception as e:
        result["error"] = str(e)

    return result


def print_report(result: dict) -> None:
    """Print a formatted analysis report."""
    print()
    print("  ═══════════════════════════════════════════")
    print(f"  WAKE FAIL ANALYSIS: {Path(result['file']).name}")
    print("  ═══════════════════════════════════════════")

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return

    print(f"  Duration:        {result['duration_s']:.2f}s")
    print(f"  Sample rate:     {result['sample_rate']} Hz")
    print(f"  Channels:        {result['channels']}")
    print(f"  Samples:         {result['num_samples']}")
    print()
    print(f"  Peak:            {result['peak']:.0f} ({result['peak_db']:.1f} dBFS)")
    print(f"  RMS:             {result['rms']:.1f} ({result['rms_db']:.1f} dBFS)")
    print(f"  Noise floor:     {result['noise_floor']:.1f}")
    print(f"  Speech RMS:      {result['speech_rms']:.1f}")
    print(f"  Dynamic range:   {result['dynamic_range_db']} dB")
    print(f"  Crest factor:    {result['crest_factor']} ({result['crest_factor_db']} dB)")
    print(f"  SNR:             {result['snr_db']} dB")
    print(f"  Voice activity:  {result['voice_activity_pct']}%")
    print(f"  Clipping:        {result['clipping_pct']}%")
    print(f"  Spectral centroid: {result.get('spectral_centroid_hz', 'N/A')} Hz")
    print(f"  Spectral bandwidth: {result.get('spectral_bandwidth_hz', 'N/A')} Hz")
    print()
    print(f"  REASON:          {result['reason']}")
    print("  ═══════════════════════════════════════════")

    # Print histogram summary
    hist = result.get("histogram", {})
    if hist:
        bins = hist["bins"]
        counts = hist["counts"]
        print("\n  Histogram (amplitude distribution):")
        max_count = max(counts) if counts else 1
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            bar_len = int(counts[i] / max_count * 30)
            bar = "█" * bar_len
            print(f"    {lo:7.0f} to {hi:7.0f}: {bar} {counts[i]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze wake_fail.wav files")
    parser.add_argument("--file", type=str, help="Specific WAV file to analyze")
    parser.add_argument("--dir", type=str, default="debug",
                        help="Directory to scan for wake_fail*.wav files")
    args = parser.parse_args()

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)
        result = analyze_wav(filepath)
        print_report(result)
        return

    # Scan directory for wake_fail*.wav files
    debug_dir = Path(args.dir)
    if not debug_dir.exists():
        print(f"ERROR: Directory not found: {debug_dir}")
        sys.exit(1)

    wav_files = sorted(debug_dir.glob("wake_fail*.wav"))
    if not wav_files:
        print(f"No wake_fail*.wav files found in {debug_dir}")
        return

    print(f"Found {len(wav_files)} wake_fail*.wav files")
    for wav_file in wav_files:
        result = analyze_wav(wav_file)
        print_report(result)


if __name__ == "__main__":
    main()