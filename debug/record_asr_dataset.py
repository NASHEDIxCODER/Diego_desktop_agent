"""
Record real microphone utterances for the ASR benchmark.

Records each command from debug/asr_dataset.py through Leo's verified
microphone (AudioManager) and saves 16 kHz mono WAV files to
data/asr_test_audio/ with a manifest capturing expected transcript,
language, and command category.

Usage:
    python debug/record_asr_dataset.py            # record all
    python debug/record_asr_dataset.py --start 5   # start at index 5
    python debug/record_asr_dataset.py --list      # list commands only
    python debug/record_asr_dataset.py --one 3     # record just index 3

Recording flow per command:
  1. Print the command to say.
  2. Wait for Enter.
  3. Record until silence (energy endpoint) or a max duration.
  4. Save WAV + manifest entry.

The audio is captured via Leo's unified AudioManager so it is the SAME
16 kHz mono normalized signal the production pipeline sees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from debug.asr_dataset import COMMANDS

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "asr_test_audio"
MANIFEST_PATH = OUT_DIR / "manifest.json"

MAX_RECORD_S = 8.0
SAMPLE_RATE = 16000


def record_one(audio_manager, max_s: float = MAX_RECORD_S):
    """Record one utterance using Leo's AudioManager ring buffer.

    Returns float32 [-1,1] mono audio, or None on failure.
    """
    start_total = audio_manager.total_samples
    frames = []
    silence_frames = 0
    speech_frames = 0
    deadline = time.time() + max_s
    last_total = start_total

    print("  (speak now…)")
    while time.time() < deadline:
        new_audio, last_total = audio_manager.read_since(last_total)
        if len(new_audio) > 0:
            frames.append(new_audio.astype(np.float32, copy=False))
            rms = float(np.sqrt(np.mean(new_audio.astype(np.float64) ** 2)))
            if rms > 0.01:
                speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1
            if speech_frames > 5 and silence_frames > 30:  # ~1s trailing silence
                break
        time.sleep(0.03)

    if not frames:
        return None
    audio = np.concatenate(frames)
    if len(audio) < SAMPLE_RATE * 0.2:
        return None
    return audio


def save_wav(audio: np.ndarray, path: Path) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Record real ASR benchmark utterances")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--one", type=int, default=None, help="Record only this index")
    parser.add_argument("--list", action="store_true", help="List commands and exit")
    args = parser.parse_args()

    if args.list:
        for i, (spoken, expected, lang, cat) in enumerate(COMMANDS):
            print(f"[{i:03d}] ({lang}/{cat}) '{spoken}' -> '{expected}'")
        return

    from voice.audio_manager import AudioManager

    am = AudioManager()
    if not am.start():
        print("No working microphone — aborting.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
        except Exception:
            manifest = []

    indices = [args.one] if args.one is not None else range(args.start, len(COMMANDS))

    try:
        for i in indices:
            spoken, expected, lang, cat = COMMANDS[i]
            print(f"\n[{i:03d}/{len(COMMANDS)-1}] SAY: {spoken!r}  (expected: {expected!r}, {lang}/{cat})")
            answer = input("  Press Enter when ready, or 's' to skip: ").strip()
            if answer.lower() == "s":
                continue
            audio = record_one(am)
            if audio is None:
                print("  (no speech captured — skipping)")
                continue
            wav_name = f"{i:03d}_{lang}_{cat}.wav"
            wav_path = OUT_DIR / wav_name
            save_wav(audio, wav_path)
            manifest.append({
                "index": i,
                "file": wav_name,
                "spoken": spoken,
                "expected": expected,
                "language": lang,
                "category": cat,
                "duration_s": round(len(audio) / SAMPLE_RATE, 3),
            })
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
            print(f"  saved {wav_path} ({len(audio)/SAMPLE_RATE:.2f}s)")
    finally:
        am.stop()

    print(f"\nRecorded {len(manifest)} utterances -> {OUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()