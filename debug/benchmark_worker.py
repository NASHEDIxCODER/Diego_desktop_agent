"""
Subprocess worker for the ASR benchmark.

Runs ONE provider in isolation so a native crash (e.g. a segfault in a
sherpa-onnx C++ model loader) cannot kill the whole benchmark. The worker
loads the provider, transcribes every command, and writes results to a JSON
file that the parent aggregates.

Usage (invoked by benchmark_asr_alternatives.py):
    python debug/benchmark_worker.py --provider <name> --out <json> \
        [--quick] [--audio-dir <dir>] [--synth-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.asr_provider import (
    word_error_rate, first_word_accuracy, normalize_transcript,
    get_gpu_stats, get_cpu_ram_stats,
)
from voice.providers.whisper_provider import WhisperProvider
from voice.providers.sherpa_providers import (
    Qwen3ASRProvider, ParakeetProvider, ZipformerStreamProvider,
    FireRedASRProvider, SenseVoiceProvider,
)
from debug.asr_dataset import COMMANDS


def build_provider(name: str):
    mapping = {
        "faster-whisper": WhisperProvider,
        "qwen3-asr-0.6b-int8": Qwen3ASRProvider,
        "parakeet-ctc-1.1b-int8": ParakeetProvider,
        "zipformer-streaming-en-int8": ZipformerStreamProvider,
        "firered-asr2-ctc-zh-en-int8": FireRedASRProvider,
        "sensevoice-int8": SenseVoiceProvider,
    }
    cls = mapping.get(name)
    if cls is None:
        return None
    return cls()


def synthesize_wav(text: str, out_path: Path, language: str) -> bool:
    try:
        voice = "hi" if language in ("hi", "hinglish") else "en-us"
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "150", "-w", str(out_path), text],
            check=True, capture_output=True, timeout=30,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def load_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def add_noise(audio: np.ndarray, kind: str, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = len(audio)
    if kind == "fan":
        t = np.arange(n) / sr
        noise = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * rng.standard_normal(n)
    elif kind == "keyboard":
        noise = np.zeros(n)
        for _ in range(int(n / sr * 8)):
            pos = rng.integers(0, n - 100)
            noise[pos:pos + 100] += rng.uniform(0.2, 0.6)
    elif kind == "music":
        t = np.arange(n) / sr
        noise = 0.25 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 660 * t)
    else:
        noise = 0.01 * rng.standard_normal(n)
    return np.clip(audio + noise.astype(np.float32), -1.0, 1.0).astype(np.float32)


def run_transcribe(provider, audio, expected, lang, cat):
    r = {
        "model": provider.name, "language": lang, "category": cat,
        "expected": expected, "transcript": "", "wer": 0.0,
        "command_accuracy": False, "first_word_accuracy": False,
        "exact_match": False, "hallucination": False,
        "time_to_first_ms": 0.0, "finalization_latency_ms": 0.0,
        "rtf": 0.0, "cpu_percent": 0.0, "gpu_vram_mb": 0.0,
        "gpu_util_percent": 0.0, "ram_mb": 0.0,
        "audio_duration_s": len(audio) / 16000.0, "ok": True, "error": "",
    }
    if not provider.health().get("ready", False):
        r["ok"] = False
        r["error"] = "not ready"
        return r
    gpu0 = get_gpu_stats()
    cpu0 = get_cpu_ram_stats()
    t0 = time.time()
    transcript = provider.transcribe(audio, 16000, None)
    t1 = time.time()
    r["transcript"] = transcript
    r["finalization_latency_ms"] = (t1 - t0) * 1000
    r["time_to_first_ms"] = r["finalization_latency_ms"]
    r["rtf"] = (t1 - t0) / r["audio_duration_s"] if r["audio_duration_s"] else 0.0
    gpu1 = get_gpu_stats()
    cpu1 = get_cpu_ram_stats()
    r["gpu_vram_mb"] = max(gpu1.get("gpu_vram_mb", 0.0), gpu0.get("gpu_vram_mb", 0.0))
    r["gpu_util_percent"] = max(gpu1.get("gpu_util_percent", 0.0), gpu0.get("gpu_util_percent", 0.0))
    r["cpu_percent"] = cpu1.get("cpu_percent", 0.0)
    r["ram_mb"] = cpu1.get("ram_mb", 0.0)
    r["wer"] = word_error_rate(expected, transcript)
    r["command_accuracy"] = normalize_transcript(transcript) == normalize_transcript(expected)
    r["first_word_accuracy"] = first_word_accuracy(expected, transcript)
    r["exact_match"] = r["command_accuracy"]
    exp_words = set(normalize_transcript(expected).split())
    hyp_words = set(normalize_transcript(transcript).split())
    r["hallucination"] = bool(transcript) and not (exp_words & hyp_words)
    r["ok"] = True
    return r


def run_streaming_test(provider, audio, expected):
    sr = 16000
    chunk = int(sr * 0.32)
    provider.reset()
    partials = []
    t_start = time.time()
    first_partial = 0.0
    first_useful = None
    for i in range(0, len(audio), chunk):
        seg = audio[i:i + chunk]
        if len(seg) < sr * 0.1:
            break
        text = provider.stream(seg, sr)
        if text:
            partials.append(text)
            if first_partial == 0.0:
                first_partial = (time.time() - t_start) * 1000
            if first_useful is None and normalize_transcript(text):
                first_useful = (time.time() - t_start) * 1000
    stability = 0.0
    if len(partials) >= 2:
        same = sum(1 for a, b in zip(partials, partials[1:]) if a == b)
        stability = same / (len(partials) - 1)
    return {
        "first_partial_ms": first_partial,
        "first_useful_word_ms": first_useful or first_partial,
        "partial_count": len(partials),
        "stability": stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--synth-dir", type=Path, default=None)
    args = parser.parse_args()

    provider = build_provider(args.provider)
    result = {
        "provider": args.provider,
        "ready": False,
        "load_time_s": 0.0,
        "load_error": "",
        "results": [],
        "streaming": [],
        "noise_results": [],
    }

    if provider is None:
        result["load_error"] = f"unknown provider {args.provider}"
        Path(args.out).write_text(json.dumps(result))
        return

    t0 = time.time()
    try:
        ok = provider.start()
    except Exception as e:
        ok = False
        result["load_error"] = f"{type(e).__name__}: {e}"
    result["load_time_s"] = time.time() - t0
    result["ready"] = ok
    if not ok:
        result["load_error"] = result["load_error"] or provider.health().get("load_error", "")
        Path(args.out).write_text(json.dumps(result))
        return

    commands = COMMANDS[:20] if args.quick else COMMANDS
    audio_cache: Dict[int, np.ndarray] = {}

    for idx, (spoken, expected, lang, cat) in enumerate(commands):
        if args.audio_dir and args.audio_dir.exists():
            wav = args.audio_dir / f"{idx:03d}_{lang}_{cat}.wav"
            if wav.exists():
                audio_cache[idx] = load_wav_mono16k(wav)
                continue
        if args.synth_dir:
            wav_path = args.synth_dir / f"{idx:03d}_{lang}.wav"
            if not wav_path.exists():
                synthesize_wav(spoken, wav_path, lang)
            if wav_path.exists():
                audio_cache[idx] = load_wav_mono16k(wav_path)

    for idx, (spoken, expected, lang, cat) in enumerate(commands):
        audio = audio_cache.get(idx)
        if audio is None:
            continue
        result["results"].append(run_transcribe(provider, audio, expected, lang, cat))

    stream_subset = commands[:15] if not args.quick else commands[:8]
    for idx, (spoken, expected, lang, cat) in enumerate(stream_subset):
        audio = audio_cache.get(idx)
        if audio is None:
            continue
        result["streaming"].append(run_streaming_test(provider, audio, expected))

    noise_cmds = [c for c in commands if c[3] == "noise"][:10]
    for spoken, expected, lang, cat in noise_cmds:
        idx = commands.index((spoken, expected, lang, cat))
        base = audio_cache.get(idx)
        if base is None:
            continue
        for kind in ["quiet", "fan", "keyboard", "music"]:
            noisy = add_noise(base, kind)
            result["noise_results"].append(
                run_transcribe(provider, noisy, expected, lang, f"noise_{kind}"))

    provider.stop()
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()