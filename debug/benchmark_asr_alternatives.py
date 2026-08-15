"""
Leo ASR Alternatives Benchmark — objective comparison of local ASR engines.

Candidates:
  1. Qwen3-ASR 0.6B INT8 (sherpa-onnx)
  2. NVIDIA Parakeet CTC 1.1B INT8 (sherpa-onnx)
  3. sherpa-onnx streaming Zipformer EN INT8
  4. FireRedASR2 CTC zh_en INT8 (sherpa-onnx)
  5. SenseVoice INT8 (sherpa-onnx)
  6. faster-whisper (existing baseline)

Every model receives IDENTICAL 16 kHz mono normalized audio (synthesized
via espeak-ng, or real recordings from data/asr_test_audio/). No model
creates its own VAD / AudioManager / preprocessor.

Each provider runs in an ISOLATED subprocess (debug/benchmark_worker.py)
so a native crash in one model loader cannot kill the whole benchmark.

Measures per model:
  WER, command accuracy, first-word accuracy, exact command match,
  hallucination rate, partial stability, time-to-first-transcript,
  finalization latency, RTF, CPU, RAM, VRAM, startup time, model load time.

Also runs:
  - Streaming test (partial → update → endpoint → final)
  - Short-command test (stop/pause/resume/yes/no/cancel/back/again)
  - Noise test (hallucination / false-transcript rate)
  - Hindi / Hinglish accuracy (reported separately)

Ranking weights (per spec):
  40% command accuracy, 20% first-word accuracy, 15% streaming latency,
  10% hallucination rate, 10% resource usage, 5% Hindi/Hinglish.

Usage:
    python debug/benchmark_asr_alternatives.py --quick
    python debug/benchmark_asr_alternatives.py --models sensevoice-int8,faster-whisper
    python debug/benchmark_asr_alternatives.py --audio-dir data/asr_test_audio
    python debug/benchmark_asr_alternatives.py --report
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("benchmark-asr-alt")

from debug.asr_dataset import COMMANDS

REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "asr_alternatives_benchmark.json"
WORKER = Path(__file__).resolve().parent / "benchmark_worker.py"

ALL_PROVIDER_NAMES = [
    "faster-whisper",
    "qwen3-asr-0.6b-int8",
    "parakeet-ctc-1.1b-int8",
    "zipformer-streaming-en-int8",
    "firered-asr2-ctc-zh-en-int8",
    "sensevoice-int8",
]


def _get(r, key, default=0.0):
    if isinstance(r, dict):
        return r.get(key, default)
    return getattr(r, key, default)


def aggregate(results) -> dict:
    ok = [r for r in results if _get(r, "ok", True)]
    if not ok:
        return {"n": 0, "ready": False}
    n = len(ok)
    wer = [_get(r, "wer") for r in ok]
    lat = [_get(r, "finalization_latency_ms") for r in ok]
    rtf = [_get(r, "rtf") for r in ok]
    return {
        "ready": True,
        "n": n,
        "avg_wer": statistics.mean(wer),
        "median_wer": statistics.median(wer),
        "command_accuracy": sum(1 for r in ok if _get(r, "command_accuracy")) / n * 100,
        "first_word_accuracy": sum(1 for r in ok if _get(r, "first_word_accuracy")) / n * 100,
        "exact_match": sum(1 for r in ok if _get(r, "exact_match")) / n * 100,
        "hallucination_rate": sum(1 for r in ok if _get(r, "hallucination")) / n * 100,
        "avg_latency_ms": statistics.mean(lat),
        "p95_latency_ms": sorted(lat)[int(n * 0.95)] if n > 1 else lat[0],
        "avg_rtf": statistics.mean(rtf),
        "avg_gpu_vram_mb": statistics.mean([_get(r, "gpu_vram_mb") for r in ok]),
        "avg_gpu_util": statistics.mean([_get(r, "gpu_util_percent") for r in ok]),
        "avg_cpu": statistics.mean([_get(r, "cpu_percent") for r in ok]),
        "avg_ram_mb": statistics.mean([_get(r, "ram_mb") for r in ok]),
    }


def rank_models(per_model: Dict[str, dict], streaming: Dict[str, dict]) -> List[Tuple[str, float]]:
    scores = {}
    for name, m in per_model.items():
        if not m.get("ready") or m.get("n", 0) == 0:
            scores[name] = 0.0
            continue
        cmd_acc = m["command_accuracy"]
        first_acc = m["first_word_accuracy"]
        s = streaming.get(name, {})
        stream_lat = s.get("avg_first_useful_word_ms", m["avg_latency_ms"])
        lat_score = max(0.0, 100.0 - (stream_lat / 20.0))
        hall = m["hallucination_rate"]
        hall_score = max(0.0, 100.0 - hall)
        ram = m["avg_ram_mb"]
        vram = m["avg_gpu_vram_mb"]
        res_score = max(0.0, 100.0 - (ram / 100.0) - (vram / 50.0))
        hi = m.get("hi", {}).get("command_accuracy", 0.0)
        hing = m.get("hinglish", {}).get("command_accuracy", 0.0)
        hindi_score = (hi + hing) / 2.0

        score = (
            0.40 * cmd_acc +
            0.20 * first_acc +
            0.15 * lat_score +
            0.10 * hall_score +
            0.10 * res_score +
            0.05 * hindi_score
        )
        scores[name] = score
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def run_worker(provider_name: str, quick: bool, audio_dir: Optional[Path],
               synth_dir: Path, out_dir: Path) -> dict:
    out_file = out_dir / f"{provider_name}.json"
    cmd = [
        sys.executable, str(WORKER),
        "--provider", provider_name,
        "--out", str(out_file),
        "--synth-dir", str(synth_dir),
    ]
    if quick:
        cmd.append("--quick")
    if audio_dir:
        cmd.extend(["--audio-dir", str(audio_dir)])

    logger.info("Running worker for %s ...", provider_name)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    logger.info("Worker %s finished in %.1fs (rc=%d)", provider_name, elapsed, proc.returncode)

    if out_file.exists():
        try:
            return json.loads(out_file.read_text())
        except Exception as e:
            logger.warning("Failed to parse %s: %s", out_file, e)
    return {
        "provider": provider_name,
        "ready": False,
        "load_time_s": elapsed,
        "load_error": f"worker crashed (rc={proc.returncode}): {proc.stderr[-500:]}",
        "results": [],
        "streaming": [],
        "noise_results": [],
    }


def run_benchmark(quick: bool, only: Optional[List[str]],
                  audio_dir: Optional[Path], out_dir: Path) -> dict:
    names = only or ALL_PROVIDER_NAMES
    commands = COMMANDS[:20] if quick else COMMANDS

    report = {
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_total_mb": 4096,
            "ram_total_mb": 16384,
        },
        "total_commands": len(commands),
        "models": {},
        "results": [],
        "streaming": {},
        "noise_results": [],
        "per_model": {},
        "ranking": [],
        "timestamp": time.time(),
    }

    synth_dir = out_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    worker_results = {}
    for name in names:
        wr = run_worker(name, quick, audio_dir, synth_dir, out_dir)
        worker_results[name] = wr
        report["models"][name] = {
            "ready": wr["ready"],
            "load_time_s": wr["load_time_s"],
            "load_error": wr["load_error"],
        }
        report["results"].extend(wr["results"])
        report["noise_results"].extend(wr["noise_results"])
        if wr["streaming"]:
            s = wr["streaming"]
            report["streaming"][name] = {
                "avg_first_partial_ms": statistics.mean([x["first_partial_ms"] for x in s]),
                "avg_first_useful_word_ms": statistics.mean([x["first_useful_word_ms"] for x in s]),
                "avg_partial_count": statistics.mean([x["partial_count"] for x in s]),
                "avg_stability": statistics.mean([x["stability"] for x in s]),
            }

    per_model = {}
    for name in names:
        wr = worker_results[name]
        rows = [r for r in wr["results"] if r.get("ok")]
        per_model[name] = aggregate(rows)
        for lang in ("en", "hi", "hinglish"):
            lang_rows = [r for r in rows if r["language"] == lang]
            per_model[name][lang] = aggregate(lang_rows)
        short_rows = [r for r in rows if r["category"] == "short"]
        per_model[name]["short"] = aggregate(short_rows)
        noise_rows = [r for r in wr["noise_results"] if r.get("ok")]
        per_model[name]["noise"] = aggregate(noise_rows)

    report["per_model"] = per_model
    report["ranking"] = rank_models(per_model, report["streaming"])
    return report


def format_report(report: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("  LEO ASR ALTERNATIVES BENCHMARK")
    lines.append("=" * 78)
    lines.append(f"  Commands: {report['total_commands']}")
    lines.append("")
    for name, m in report["per_model"].items():
        lines.append("-" * 78)
        lines.append(f"  MODEL: {name}")
        lines.append("-" * 78)
        if not m.get("ready"):
            lines.append(f"  NOT READY — {report['models'][name].get('load_error','')}")
            continue
        lines.append(f"  N:                      {m['n']}")
        lines.append(f"  Command accuracy:       {m['command_accuracy']:.1f}%")
        lines.append(f"  First-word accuracy:    {m['first_word_accuracy']:.1f}%")
        lines.append(f"  Exact match:            {m['exact_match']:.1f}%")
        lines.append(f"  Avg WER:                {m['avg_wer']:.3f}")
        lines.append(f"  Hallucination rate:     {m['hallucination_rate']:.1f}%")
        lines.append(f"  Avg latency:            {m['avg_latency_ms']:.1f}ms")
        lines.append(f"  Avg RTF:                {m['avg_rtf']:.3f}")
        lines.append(f"  Avg RAM:                {m['avg_ram_mb']:.0f}MB")
        lines.append(f"  Avg GPU VRAM:           {m['avg_gpu_vram_mb']:.0f}MB")
        lines.append(f"  Load time:              {report['models'][name]['load_time_s']:.1f}s")
        for lang in ("en", "hi", "hinglish"):
            lm = m.get(lang, {})
            if lm.get("n"):
                lines.append(f"  {lang:<10} accuracy:      {lm['command_accuracy']:.1f}% (n={lm['n']})")
        sm = m.get("short", {})
        if sm.get("n"):
            lines.append(f"  short cmds accuracy:    {sm['command_accuracy']:.1f}% (n={sm['n']})")
        nm = m.get("noise", {})
        if nm.get("n"):
            lines.append(f"  noise hallucination:    {nm['hallucination_rate']:.1f}% (n={nm['n']})")
        s = report["streaming"].get(name, {})
        if s:
            lines.append(f"  first useful word:      {s['avg_first_useful_word_ms']:.0f}ms")
            lines.append(f"  partial stability:      {s['avg_stability']:.2f}")
        lines.append("")
    lines.append("=" * 78)
    lines.append("  RANKING (weighted)")
    lines.append("=" * 78)
    for i, (name, score) in enumerate(report["ranking"]):
        lines.append(f"  {i+1}. {name:<32} {score:.1f}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leo ASR alternatives benchmark")
    parser.add_argument("--quick", action="store_true", help="20-command smoke test")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated provider names to test")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="Use real recordings from this directory")
    parser.add_argument("--report", action="store_true", help="Print last report only")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/leo_asr_alt_benchmark"))
    args = parser.parse_args()

    if args.report:
        if REPORT_PATH.exists():
            report = json.loads(REPORT_PATH.read_text())
            print(format_report(report))
        else:
            print("No report found.")
        return

    only = args.models.split(",") if args.models else None
    report = run_benchmark(args.quick, only, args.audio_dir, args.out_dir)
    print(format_report(report))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    main()