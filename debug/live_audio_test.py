#!/usr/bin/env python3
"""
Live Audio Test for Leo Desktop Assistant.

Listens forever using the REAL microphone through the FULL processing pipeline:
  Microphone → High-pass → Noise Suppression → AGC → VAD → Wake Detection → STT

Every phrase is printed with:
- Audio RMS (input and processed)
- Wake score / similarity
- VAD score
- Confidence
- Detection latency
- Speech duration

Usage:
    python debug/live_audio_test.py
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
    ratio = difflib.SequenceMatcher(None, text.lower(), variant.lower()).ratio()
    return ratio


def main():
    print()
    print("=" * 70)
    print("  LIVE AUDIO TEST — REAL MICROPHONE")
    print("=" * 70)
    print("  Say 'hello leo' to test wake detection")
    print("  Any other phrase will be printed with metrics")
    print("  Press Ctrl+C to stop")
    print()

    # Start AudioManager
    print("  Starting AudioManager...")
    if not audio_manager.start():
        print("  FAILED: AudioManager could not start")
        sys.exit(1)

    am_diag = audio_manager.get_diagnostics()
    print(f"  Device:        [{am_diag['device_index']}]")
    print(f"  Sample rate:   {am_diag['sample_rate']} Hz")
    print(f"  Backend:       {am_diag['backend']}")
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

    print("  Listening... (speak naturally)")
    print()

    try:
        while True:
            # Get fully processed audio (noise suppression + AGC)
            recent = audio_manager.get_recent_processed(0.1)
            if len(recent) == 0:
                time.sleep(0.02)
                continue

            # Compute metrics
            rms = float(np.sqrt(np.mean(recent.astype(float) ** 2)))
            norm_rms = rms / 32768.0
            now = time.time()

            # VAD check
            if rms > energy_threshold:
                if not is_speaking:
                    is_speaking = True
                    speech_start = now
                    speech_buffer = []
                    print(f"  [VAD] Speech start (RMS={rms:.1f})")
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

                            audio_bytes = audio.tobytes()
                            speech_rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))

                            print(f"\n  ═════ SPEECH #{phrase_count} ({duration:.1f}s, RMS={speech_rms:.1f}) ═════")

                            # Measure noise floor
                            preproc = audio_preprocessor.get_metrics()
                            noise_db = preproc.get('noise_floor_db', 'N/A')
                            print(f"  Noise floor: {noise_db} dB")

                            # STT
                            t0 = time.time()
                            text = _recognize_bytes(audio_bytes, samplerate)
                            stt_time = time.time() - t0

                            if text:
                                print(f"  STT ({(stt_time*1000):.0f}ms): '{text}'")

                                # Wake word check
                                best_ratio = 0.0
                                best_variant = ""
                                for variant in wake_word_engine._variants:
                                    ratio = compute_similarity(variant, text)
                                    if ratio > best_ratio:
                                        best_ratio = ratio
                                        best_variant = variant

                                detected = wake_word_engine.detect(text)
                                print(f"  Wake match: '{best_variant}' ratio={best_ratio:.3f} " +
                                      f"threshold=0.75 → {'DETECTED ✓' if detected else 'NO'}")

                                # VAD score (SNR-based)
                                preproc_metrics = audio_preprocessor.get_metrics()
                                noise_floor = preproc_metrics.get('noise_floor_raw', 0)
                                if noise_floor and noise_floor > 0:
                                    vad_snr = 20 * np.log10((speech_rms / 32768.0 + 1e-10) / (noise_floor + 1e-10))
                                else:
                                    vad_snr = 40.0
                                print(f"  VAD SNR: {vad_snr:.1f} dB")
                            else:
                                print(f"  STT ({(stt_time*1000):.0f}ms): no speech recognized")
                                print(f"  Audio bytes: {len(audio_bytes)}")

                        # Reset
                        is_speaking = False
                        speech_buffer = []
                        print()
                    else:
                        # Still in speech, add silence
                        speech_buffer.append(recent.copy())

            # Show real-time level meter occasionally
            if not is_speaking and phrase_count >= 0:
                bar_len = int(norm_rms * 50)
                bar = "█" * min(bar_len, 50)
                print(f"\r  Input: {bar:<50} | RMS={rms:6.0f} | Threshold={energy_threshold:.0f}", end="", flush=True)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n  Stopped by user")
    finally:
        audio_manager.stop()
        print("  AudioManager stopped")
        print()

    total = phrase_count
    if total == 0:
        print("  No phrases detected during the session.")
    else:
        print(f"  {total} phrase(s) detected")

        # Reset counters
        import random
        wake_count = 0
        print(f"  No wake metrics available (wake detection requires STT phrases)")

    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()