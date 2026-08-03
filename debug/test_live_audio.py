#!/usr/bin/env python3
"""
Live Audio Test — Real-time microphone verification.

Displays real-time audio metrics 10 times per second:
- Current microphone
- Current backend
- Current sample rate
- Current RMS
- Current Peak
- Current SNR
- Current VAD state
- Current wake confidence
- Current speech confidence

When you speak, you MUST immediately see:
- Speech detected
- Current RMS
- Current VAD
- Current wake confidence

If not, the microphone is wrong.

Usage:
    python debug/test_live_audio.py
"""

import os
import sys
import time
import difflib
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

from voice.audio_manager import audio_manager
from voice.audio_processing import audio_preprocessor
from voice.wake_word import wake_word_engine
from voice.stt import _recognize_bytes


def compute_similarity(variant: str, text: str) -> float:
    """Compute fuzzy similarity between variant and text."""
    return difflib.SequenceMatcher(None, text.lower(), variant.lower()).ratio()


def bar(value, max_val, width=30):
    """Create a text bar."""
    pct = min(max(int(value / max_val * width), 0), width)
    return "█" * pct + "░" * (width - pct)


def main():
    print()
    print("=" * 70)
    print("  LIVE AUDIO TEST — REAL MICROPHONE")
    print("=" * 70)
    print("  Say 'hello leo' to test wake detection")
    print("  Press Ctrl+C to stop")
    print()

    # Start AudioManager (auto-selects best hardware mic)
    print("  Starting AudioManager...")
    if not audio_manager.start():
        print("  FAILED: AudioManager could not start")
        sys.exit(1)

    am_diag = audio_manager.get_diagnostics()
    print(f"  Device:        [{am_diag['device_index']}]")
    print(f"  Backend:       {am_diag['backend']}")
    print(f"  Sample rate:   {am_diag['sample_rate']} Hz")
    print(f"  Buffer:        {am_diag['buffer_seconds']:.1f}s")
    print(f"  Energy thresh: {am_diag['energy_threshold']:.1f}")
    print()

    # VAD state
    energy_threshold = audio_manager.energy_threshold
    silence_duration = 0.8
    min_speech_duration = 0.4
    max_speech_duration = 5.0
    samplerate = audio_manager.sample_rate

    speech_buffer: list = []
    is_speaking = False
    speech_start = 0.0
    last_voice = 0.0
    phrase_count = 0

    # Adaptive threshold tracking
    noise_floor_ema = None
    threshold_ema = energy_threshold

    print("  Listening... (speak naturally)")
    print()

    try:
        while True:
            # Get recent audio (100ms chunks)
            recent = audio_manager.get_recent_audio(0.1)
            if len(recent) == 0:
                time.sleep(0.02)
                continue

            # Compute metrics (ring-buffer audio is float32 [-1, 1])
            rms = float(np.sqrt(np.mean(recent.astype(float) ** 2))) * 32768.0
            peak = float(np.max(np.abs(recent))) * 32768.0
            norm_rms = rms / 32768.0
            now = time.time()

            # Adaptive threshold: track noise floor with EMA
            if noise_floor_ema is None:
                noise_floor_ema = rms
            else:
                noise_floor_ema = 0.95 * noise_floor_ema + 0.05 * rms
            # Threshold = noise floor * 1.5, min 300
            threshold_ema = max(300.0, noise_floor_ema * 1.5)

            # VAD check
            if rms > threshold_ema:
                if not is_speaking:
                    is_speaking = True
                    speech_start = now
                    speech_buffer = []
                last_voice = now
                speech_buffer.append(recent.copy())
            else:
                if is_speaking:
                    if now - last_voice > silence_duration:
                        duration = now - speech_start
                        if duration >= min_speech_duration:
                            phrase_count += 1

                            # Concatenate speech
                            audio = np.concatenate(speech_buffer)
                            if duration > max_speech_duration:
                                audio = audio[:int(max_speech_duration * samplerate)]

                            # Apply full noise suppression (NO AGC — unity gain).
                            # process() returns float32 [-1, 1]; PCM16 only at the STT sink.
                            from voice.audio_processing import float32_to_int16
                            processed = audio_preprocessor.process(audio)
                            processed16 = float32_to_int16(processed)
                            audio_bytes = processed16.tobytes()
                            speech_rms = float(np.sqrt(np.mean(processed16.astype(float) ** 2)))

                            # STT
                            t0 = time.time()
                            text = _recognize_bytes(audio_bytes, samplerate)
                            stt_time = time.time() - t0

                            print(f"\n  ═════ SPEECH #{phrase_count} ({duration:.1f}s, RMS={speech_rms:.1f}) ═════")

                            if text:
                                print(f"  STT ({(stt_time*1000):.0f}ms): '{text}'")

                                # Wake confidence
                                best_ratio = 0.0
                                best_variant = ""
                                for variant in wake_word_engine._variants:
                                    ratio = compute_similarity(variant, text)
                                    if ratio > best_ratio:
                                        best_ratio = ratio
                                        best_variant = variant

                                detected = wake_word_engine.detect(text)
                                print(f"  Wake: '{best_variant}' conf={best_ratio:.3f} → {'DETECTED ✓' if detected else 'NO'}")

                                # SNR
                                preproc_metrics = audio_preprocessor.get_metrics()
                                nf = preproc_metrics.get('noise_floor_raw', 0)
                                if nf and nf > 0:
                                    snr_db = 20 * np.log10((speech_rms / 32768.0 + 1e-10) / (nf + 1e-10))
                                else:
                                    snr_db = 40.0
                                print(f"  SNR: {snr_db:.1f} dB")
                            else:
                                print(f"  STT ({(stt_time*1000):.0f}ms): no speech recognized")
                                print(f"  Audio bytes: {len(audio_bytes)}")

                        # Reset
                        is_speaking = False
                        speech_buffer = []
                        print()
                    else:
                        speech_buffer.append(recent.copy())

            # Real-time display (10 Hz)
            preproc_metrics = audio_preprocessor.get_metrics()
            nf_raw = preproc_metrics.get('noise_floor_raw') or 0
            nf_db = 20 * np.log10(nf_raw / 32768.0) if nf_raw and nf_raw > 0 else -120.0
            gain = preproc_metrics.get('gain_applied', 1.0)

            # Wake confidence (based on current audio energy)
            wake_conf = min(1.0, rms / (threshold_ema * 3)) if threshold_ema > 0 else 0

            vad_state = "SPEECH" if is_speaking else "SILENCE"

            sys.stdout.write(f"\r")
            sys.stdout.write(
                f" In: {bar(rms, 32768)} {norm_rms*100:5.1f}% "
                f"| RMS={rms:6.0f} Peak={peak:6.0f} "
                f"| Thresh={threshold_ema:6.0f} "
                f"| Noise={nf_db:6.1f}dB "
                f"| Gain={gain:.2f}x "
                f"| VAD={vad_state} "
                f"| Wake={wake_conf:.2f}"
            )
            sys.stdout.flush()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n  Stopped by user")
    finally:
        audio_manager.stop()
        print("  AudioManager stopped")
        print()

    print(f"  {phrase_count} phrase(s) detected")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()