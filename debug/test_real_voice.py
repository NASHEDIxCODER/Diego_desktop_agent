#!/usr/bin/env python3
"""
Real Voice Regression Test for Leo Desktop Assistant.

Records YOUR REAL VOICE saying "Hello Leo" 20 times through the
physical microphone, runs all STT engines on each recording,
and reports the recognition rate.

NO mocked audio. NO injected transcripts. Only real microphone recordings.

Usage:
    python debug/test_real_voice.py
"""

import os
import sys
import time
import wave
import difflib
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

from voice.audio_manager import audio_manager
from voice.audio_processing import audio_preprocessor
from voice.wake_word import wake_word_engine

# Test parameters
NUM_TRIALS = 20
RECORD_TIMEOUT = 8.0
PHRASE_LIMIT = 5.0
OUTPUT_DIR = PROJECT_ROOT / "debug" / "real_voice_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_similarity(variant: str, text: str) -> float:
    """Compute fuzzy similarity between variant and text."""
    return difflib.SequenceMatcher(None, text.lower(), variant.lower()).ratio()


def analyze_wav(wav_path: Path) -> dict:
    """Analyze a WAV file with ffprobe and numpy."""
    result = {
        "path": str(wav_path),
        "duration": 0.0,
        "codec": "pcm_s16le",
        "channels": 1,
        "sample_rate": 0,
        "bit_depth": 16,
        "rms": 0.0,
        "peak": 0.0,
        "dc_offset": 0.0,
        "clipping_pct": 0.0,
    }

    # ffprobe
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(wav_path)],
            capture_output=True, text=True, timeout=5
        )
        if probe.returncode == 0:
            import json
            data = json.loads(probe.stdout)
            if data.get("streams"):
                s = data["streams"][0]
                result["codec"] = s.get("codec_name", "pcm_s16le")
                result["channels"] = s.get("channels", 1)
                result["sample_rate"] = int(s.get("sample_rate", 0))
                result["bit_depth"] = s.get("bits_per_sample", 16)
            if data.get("format"):
                result["duration"] = float(data["format"].get("duration", 0))
    except Exception:
        pass

    # numpy analysis
    try:
        with wave.open(str(wav_path), "rb") as wav:
            sr = wav.getframerate()
            n = wav.getnframes()
            audio = np.frombuffer(wav.readframes(n), dtype=np.int16).astype(np.float64)
        if len(audio) > 0:
            result["sample_rate"] = sr
            result["duration"] = len(audio) / sr
            result["rms"] = float(np.sqrt(np.mean(audio ** 2)))
            result["peak"] = float(np.max(np.abs(audio)))
            result["dc_offset"] = float(np.mean(audio))
            result["clipping_pct"] = float(np.mean(np.abs(audio) >= 32760) * 100)
    except Exception:
        pass

    return result


def try_google_stt(wav_path: Path) -> str:
    """Try Google Web Speech API."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(str(wav_path)) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language="en-IN")
        return text.strip() if text else ""
    except Exception as e:
        return f"[ERROR: {type(e).__name__}]"


def try_whisper_stt(wav_path: Path) -> str:
    """Try Whisper offline."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(wav_path))
        return result.get("text", "").strip()
    except ImportError:
        return "[NOT INSTALLED]"
    except Exception as e:
        return f"[ERROR: {type(e).__name__}]"


def try_vosk_stt(wav_path: Path) -> str:
    """Try Vosk offline."""
    try:
        import vosk
        import json
        model = vosk.Model("model")
        rec = vosk.KaldiRecognizer(model, 16000)
        with wave.open(str(wav_path), "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()
    except ImportError:
        return "[NOT INSTALLED]"
    except Exception as e:
        return f"[ERROR: {type(e).__name__}]"


def main():
    print()
    print("=" * 70)
    print("  REAL VOICE REGRESSION TEST")
    print("=" * 70)
    print(f"  Trials: {NUM_TRIALS}")
    print(f"  Phrase: 'Hello Leo'")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Start AudioManager
    print("  Starting AudioManager...")
    if not audio_manager.start():
        print("  FAILED: AudioManager could not start")
        sys.exit(1)

    am_diag = audio_manager.get_diagnostics()
    print(f"  Device:        [{am_diag['device_index']}]")
    print(f"  Backend:       {am_diag['backend']}")
    print(f"  Sample rate:   {am_diag['sample_rate']} Hz")
    print()

    # Pre-seed noise profile
    print("  Pre-seeding noise profile (2s)...")
    time.sleep(2)
    audio_manager.get_recent_processed(1.0)
    print()

    results = []
    success_count = 0

    for trial in range(1, NUM_TRIALS + 1):
        print(f"  [{trial}/{NUM_TRIALS}] Say 'Hello Leo'...")

        # Record speech
        audio_bytes = audio_manager.record_command(timeout=RECORD_TIMEOUT, phrase_limit=PHRASE_LIMIT)
        if audio_bytes is None:
            print("    ✗ No speech detected")
            results.append({"trial": trial, "success": False, "reason": "no_speech"})
            continue

        # Process with noise suppression
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        processed = audio_preprocessor.process(samples)
        proc_bytes = processed.tobytes()

        # Save WAV
        wav_path = OUTPUT_DIR / f"trial_{trial:02d}.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(audio_manager.sample_rate)
            wav.writeframes(proc_bytes)

        # Analyze WAV
        analysis = analyze_wav(wav_path)
        print(f"    WAV: {wav_path.name} ({analysis['duration']:.1f}s, "
              f"RMS={analysis['rms']:.0f}, Peak={analysis['peak']:.0f}, "
              f"Clip={analysis['clipping_pct']:.1f}%)")

        # Run all STT engines
        google_text = try_google_stt(wav_path)
        whisper_text = try_whisper_stt(wav_path)
        vosk_text = try_vosk_stt(wav_path)

        print(f"    Google:  '{google_text}'")
        print(f"    Whisper: '{whisper_text}'")
        print(f"    Vosk:    '{vosk_text}'")

        # Check wake word match
        best_ratio = 0.0
        best_engine = ""
        best_text = ""
        for engine, text in [("google", google_text), ("whisper", whisper_text), ("vosk", vosk_text)]:
            if text and not text.startswith("[") and not text.startswith("["):
                for variant in wake_word_engine._variants:
                    ratio = compute_similarity(variant, text)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_engine = engine
                        best_text = text

        detected = wake_word_engine.detect(best_text) if best_text else False
        success = detected and best_ratio >= 0.75

        if success:
            success_count += 1
            print(f"    ✓ WAKE DETECTED via {best_engine}: '{best_text}' (ratio={best_ratio:.3f})")
        else:
            print(f"    ✗ Wake NOT detected (best: {best_engine} '{best_text}' ratio={best_ratio:.3f})")

        results.append({
            "trial": trial,
            "success": success,
            "google": google_text,
            "whisper": whisper_text,
            "vosk": vosk_text,
            "best_engine": best_engine,
            "best_text": best_text,
            "best_ratio": best_ratio,
            "analysis": analysis,
        })
        print()

    audio_manager.stop()

    # Summary
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Success: {success_count}/{NUM_TRIALS}")
    print(f"  Rate:    {success_count/NUM_TRIALS*100:.1f}%")
    print()

    # Per-engine stats
    engine_stats = {"google": 0, "whisper": 0, "vosk": 0}
    for r in results:
        if r.get("best_engine") in engine_stats:
            engine_stats[r["best_engine"]] += 1
    for engine, count in engine_stats.items():
        print(f"  {engine}: {count}/{NUM_TRIALS} wake detections")

    print()
    if success_count >= 20:
        print("  ✓ PASSED: 20/20 real voice recordings recognized")
    else:
        print(f"  ✗ FAILED: Only {success_count}/20 recognized")
        print("  Check the saved WAV files in debug/real_voice_test/")
        print("  Run ffprobe on each file to inspect audio quality")
    print()
    print("=" * 70)
    print()

    return 0 if success_count >= 20 else 1


if __name__ == "__main__":
    sys.exit(main())