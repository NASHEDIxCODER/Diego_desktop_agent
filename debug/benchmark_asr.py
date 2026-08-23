"""
ASR Benchmark — Nemotron vs Whisper on real Diego commands.

Synthesizes the real Diego command set (English, natural speech, Hindi/Hinglish)
to 16 kHz mono WAV via espeak-ng, then runs each audio clip through every
available ASRProvider and records per-transcript metrics:

  model, language, transcript, expected, WER, command accuracy,
  first-word accuracy, time-to-first-token, finalization latency,
  real-time factor, CPU, GPU VRAM, GPU utilization, RAM, dropped chunks,
  hallucinations.

This benchmark does NOT modify Diego's voice architecture. It only measures
the providers behind the new ASRProvider interface.

Usage:
    python debug/benchmark_asr.py                 # full run
    python debug/benchmark_asr.py --quick         # 8-command smoke test
    python debug/benchmark_asr.py --report       # print last report
    python debug/benchmark_asr.py --no-tts        # skip synthesis (use cached)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark-asr")

from voice.asr_provider import (
    ASRMetrics, ASRProvider, word_error_rate, first_word_accuracy,
    normalize_transcript, get_gpu_stats, get_cpu_ram_stats,
)
from voice.providers.whisper_provider import WhisperProvider
from voice.providers.nemotron_provider import NemotronProvider

# ── Real Diego command set ────────────────────────────────────────
# (spoken, expected, language)
BENCHMARK_COMMANDS: List[Tuple[str, str, str]] = [
    # ── English commands ──
    ("open firefox", "open firefox", "en"),
    ("open youtube", "open youtube", "en"),
    ("open vscode", "open vscode", "en"),
    ("play music", "play music", "en"),
    ("pause music", "pause music", "en"),
    ("stop", "stop", "en"),
    ("continue", "continue", "en"),
    ("go back", "go back", "en"),
    ("search youtube", "search youtube", "en"),
    ("what time is it", "what time is it", "en"),
    ("open ghostline", "open ghostline", "en"),
    ("close firefox", "close firefox", "en"),
    # ── Natural speech ──
    ("hey Diego open youtube", "open youtube", "en"),
    ("Diego can you open firefox", "open firefox", "en"),
    ("open youtube and search for music", "open youtube and search for music", "en"),
    ("actually open vscode instead", "open vscode", "en"),
    ("no, cancel that", "cancel", "en"),
    ("continue", "continue", "en"),
    ("open it again", "open it again", "en"),
    # ── Hindi / Hinglish ──
    ("youtube kholo", "youtube kholo", "hi"),
    ("firefox kholo", "firefox kholo", "hi"),
    ("gaana chalao", "gaana chalao", "hi"),
    ("volume kam karo", "volume kam karo", "hi"),
    ("isko band karo", "isko band karo", "hi"),
    ("youtube par music chalao", "youtube par music chalao", "hi"),
]


@dataclass
class BenchmarkReport:
    """Aggregate ASR benchmark report."""
    hardware: dict = field(default_factory=dict)
    models_tested: List[str] = field(default_factory=list)
    total_commands: int = 0
    per_model: Dict[str, dict] = field(default_factory=dict)
    results: List[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


def synthesize_wav(text: str, out_path: Path, language: str) -> bool:
    """Synthesize speech to a 16 kHz mono WAV using espeak-ng."""
    try:
        voice = "hi" if language == "hi" else "en-us"
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "150", "-w", str(out_path), text],
            check=True, capture_output=True, timeout=30,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        logger.warning("TTS synthesis failed for %r: %s", text, e)
        return False


def load_wav_mono16k(path: Path) -> np.ndarray:
    """Load a WAV as float32 [-1,1] mono at 16 kHz."""
    import wave
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    if sr != 16000:
        # Simple linear resample (benchmark-only; production uses scipy)
        from scipy.signal import resample_poly
        import math
        g = math.gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def run_provider_transcribe(
    provider: ASRProvider,
    audio: np.ndarray,
    expected: str,
    language: str,
) -> ASRMetrics:
    """Run one transcription and measure all metrics."""
    m = ASRMetrics(model=provider.name, language=language, expected=expected)
    m.audio_duration_s = len(audio) / 16000.0

    if not provider.health().get("ready", False):
        m.ok = False
        m.error = "provider not ready"
        return m

    # Pre-read resource baselines
    gpu0 = get_gpu_stats()
    cpu0 = get_cpu_ram_stats()

    t0 = time.time()
    transcript = provider.transcribe(audio, 16000, language if language != "hi" else None)
    t1 = time.time()

    m.transcript = transcript
    m.finalization_latency_ms = (t1 - t0) * 1000
    m.time_to_first_token_ms = m.finalization_latency_ms  # non-streaming: same
    m.real_time_factor = (t1 - t0) / m.audio_duration_s if m.audio_duration_s else 0.0

    # Post-read resources
    gpu1 = get_gpu_stats()
    cpu1 = get_cpu_ram_stats()
    m.gpu_vram_mb = max(gpu1.get("gpu_vram_mb", 0.0), gpu0.get("gpu_vram_mb", 0.0))
    m.gpu_util_percent = max(gpu1.get("gpu_util_percent", 0.0), gpu0.get("gpu_util_percent", 0.0))
    m.cpu_percent = cpu1.get("cpu_percent", 0.0)
    m.ram_mb = cpu1.get("ram_mb", 0.0)

    # Accuracy
    m.wer = word_error_rate(expected, transcript)
    m.command_accuracy = normalize_transcript(transcript) == normalize_transcript(expected)
    m.first_word_accuracy = first_word_accuracy(expected, transcript)

    # Hallucination heuristic: transcript produced but WER > 1.0 and
    # no word overlap with expected.
    exp_words = set(expected.lower().split())
    hyp_words = set(transcript.lower().split())
    if transcript and not (exp_words & hyp_words):
        m.hallucinations = 1

    m.ok = True
    return m


async def run_benchmark(quick: bool, no_tts: bool, out_dir: Path) -> BenchmarkReport:
    commands = BENCHMARK_COMMANDS
    if quick:
        commands = commands[:8]

    report = BenchmarkReport()
    report.total_commands = len(commands)

    # Hardware
    import platform
    report.hardware = {
        "gpu": "NVIDIA GeForce RTX 3050",
        "vram_total_mb": 4096,
        "ram_total_mb": 16384,
        "os": platform.platform(),
        "cpu": platform.processor(),
    }

    # Providers
    providers: List[ASRProvider] = [WhisperProvider(), NemotronProvider()]
    for p in providers:
        logger.info("Starting provider: %s", p.name)
        ok = await asyncio.get_event_loop().run_in_executor(None, p.start)
        logger.info("  -> %s (ready=%s)", "OK" if ok else "FAILED", ok)
        report.models_tested.append({
            "name": p.name,
            "ready": ok,
            "health": p.health(),
        })

    # Synthesis cache dir
    synth_dir = out_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    for idx, (spoken, expected, lang) in enumerate(commands):
        logger.info("[%d/%d] '%s' (%s)", idx + 1, len(commands), spoken, lang)
        wav_path = synth_dir / f"{idx:03d}_{lang}.wav"

        if not no_tts or not wav_path.exists():
            if not synthesize_wav(spoken, wav_path, lang):
                logger.warning("Synthesis failed for %r — skipping", spoken)
                continue

        audio = load_wav_mono16k(wav_path)

        for provider in providers:
            m = await asyncio.get_event_loop().run_in_executor(
                None, run_provider_transcribe, provider, audio, expected, lang)
            report.results.append(asdict(m))
            logger.info("  [%s] '%s' -> '%s' WER=%.2f cmd_acc=%s first=%s lat=%.0fms",
                        provider.name, expected, m.transcript, m.wer,
                        m.command_accuracy, m.first_word_accuracy,
                        m.finalization_latency_ms)

    # Aggregate per model
    for p in providers:
        rows = [r for r in report.results if r["model"] == p.name and r["ok"]]
        if not rows:
            report.per_model[p.name] = {
                "ready": p.health().get("ready", False),
                "n": 0,
                "error": p.health().get("load_error", ""),
            }
            continue
        n = len(rows)
        wer = [r["wer"] for r in rows]
        lat = [r["finalization_latency_ms"] for r in rows]
        rtf = [r["real_time_factor"] for r in rows]
        report.per_model[p.name] = {
            "ready": True,
            "n": n,
            "avg_wer": statistics.mean(wer),
            "median_wer": statistics.median(wer),
            "command_accuracy": sum(1 for r in rows if r["command_accuracy"]) / n * 100,
            "first_word_accuracy": sum(1 for r in rows if r["first_word_accuracy"]) / n * 100,
            "avg_latency_ms": statistics.mean(lat),
            "p95_latency_ms": sorted(lat)[int(n * 0.95)] if n > 1 else lat[0],
            "avg_rtf": statistics.mean(rtf),
            "avg_gpu_vram_mb": statistics.mean([r["gpu_vram_mb"] for r in rows]),
            "avg_gpu_util": statistics.mean([r["gpu_util_percent"] for r in rows]),
            "avg_cpu": statistics.mean([r["cpu_percent"] for r in rows]),
            "avg_ram_mb": statistics.mean([r["ram_mb"] for r in rows]),
            "hallucinations": sum(r["hallucinations"] for r in rows),
            "dropped_chunks": sum(r["dropped_chunks"] for r in rows),
        }

    # Stop providers
    for p in providers:
        p.stop()

    return report


def format_report(report: BenchmarkReport) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  DIEGO ASR BENCHMARK — NEMOTRON VS WHISPER")
    lines.append("=" * 72)
    lines.append(f"  Hardware: {report.hardware.get('gpu','?')} "
                 f"({report.hardware.get('vram_total_mb','?')} MB VRAM)")
    lines.append(f"  Commands: {report.total_commands}")
    lines.append("")
    for name, stats in report.per_model.items():
        lines.append("-" * 72)
        lines.append(f"  MODEL: {name}")
        lines.append("-" * 72)
        if not stats.get("ready"):
            lines.append(f"  NOT READY — {stats.get('error','')}")
            continue
        lines.append(f"  N:                      {stats['n']}")
        lines.append(f"  Avg WER:                {stats['avg_wer']:.3f}")
        lines.append(f"  Median WER:             {stats['median_wer']:.3f}")
        lines.append(f"  Command accuracy:       {stats['command_accuracy']:.1f}%")
        lines.append(f"  First-word accuracy:    {stats['first_word_accuracy']:.1f}%")
        lines.append(f"  Avg latency:            {stats['avg_latency_ms']:.1f}ms")
        lines.append(f"  P95 latency:            {stats['p95_latency_ms']:.1f}ms")
        lines.append(f"  Avg RTF:                {stats['avg_rtf']:.3f}")
        lines.append(f"  Avg GPU VRAM:           {stats['avg_gpu_vram_mb']:.0f}MB")
        lines.append(f"  Avg GPU util:           {stats['avg_gpu_util']:.1f}%")
        lines.append(f"  Avg CPU:                {stats['avg_cpu']:.1f}%")
        lines.append(f"  Avg RAM:                {stats['avg_ram_mb']:.0f}MB")
        lines.append(f"  Hallucinations:         {stats['hallucinations']}")
        lines.append(f"  Dropped chunks:         {stats['dropped_chunks']}")
        lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "asr_benchmark.json"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Diego ASR benchmark (Nemotron vs Whisper)")
    parser.add_argument("--quick", action="store_true", help="8-command smoke test")
    parser.add_argument("--no-tts", action="store_true", help="Skip synthesis (use cached WAVs)")
    parser.add_argument("--report", action="store_true", help="Print last report only")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/Diego_asr_benchmark"))
    args = parser.parse_args()

    if args.report:
        if REPORT_PATH.exists():
            data = json.loads(REPORT_PATH.read_text())
            report = BenchmarkReport(**data)
            print(format_report(report))
        else:
            print("No benchmark report found.")
        return

    report = await run_benchmark(args.quick, args.no_tts, args.out_dir)
    print(format_report(report))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(asdict(report), indent=2))
    logger.info("Report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    asyncio.run(main())