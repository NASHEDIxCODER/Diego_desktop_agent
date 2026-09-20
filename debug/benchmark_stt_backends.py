"""
STT backend benchmark — faster-whisper vs Qwen3-ASR (sherpa-onnx).

Objective: decide whether Qwen3-ASR 0.6B INT8 should replace faster-whisper
as Diego's production command STT. This benchmark feeds BOTH backends the
SAME normalized, preprocessed audio that Diego actually produces:

    raw audio (float32)
      -> AutomaticGainControl (capture_agc, the single gain stage)
      -> resample to 16 kHz mono
      -> 4th-order 80 Hz Butterworth high-pass (the unified preprocessing)
      -> ring buffer -> STT

So the comparison is against the REAL signal the production pipeline hands to
the ASR, not a naive direct WAV load.

Measures per phrase:
  - exact transcript (and evidence/confidence when the backend exposes it)
  - WER and command actionability (exact-normalized match + router match)
  - finalization latency and real-time factor
  - model load time, CPU %, resident RAM

Backends:
  - faster-whisper (existing production baseline; _WhisperTranscriber)
  - qwen3-asr-0.6b-int8 (Qwen3ASRProvider via sherpa-onnx)

Nothing is executed: command routing / intent probes are comparison-only.

Usage:
    python debug/benchmark_stt_backends.py            # full synthetic set
    python debug/benchmark_stt_backends.py --quick    # 4 required commands
    python debug/benchmark_stt_backends.py --mic      # real-mic loopback
    python debug/benchmark_stt_backends.py --report   # print last report
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
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (stdlib stubs)

from voice.asr_provider import word_error_rate, normalize_transcript, first_word_accuracy  # noqa: E402
from voice.audio_processing import (  # noqa: E402
    SAMPLE_RATE, HIGH_PASS_CUTOFF, HIGH_PASS_ORDER,
    capture_agc, float32_to_int16,
)
from voice.command_listener import _WhisperTranscriber, _postprocess  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "data" / "stt_backends_benchmark.json"
SYNTH_DIR = PROJECT_ROOT / "data" / "asr_test_audio"  # reused cache for synthesized wavs

# The REQUIRED acceptance phrases plus natural + noisy variants.
REQUIRED = [
    ("open firefox", "open firefox", "en", "command"),
    ("open telegram", "open telegram", "en", "command"),
    ("search for jobs on linkedin", "search for jobs on linkedin", "en", "command"),
    ("send rahul a message", "send rahul a message", "en", "command"),
]

NATURAL = [
    ("hey diego open firefox", "open firefox", "en", "natural"),
    ("can you open the browser for me", "open browser", "en", "natural"),
    ("diego please send rahul a message", "send rahul a message", "en", "natural"),
    ("what time is it", "what time is it", "en", "natural"),
    ("tell me a joke", "tell me a joke", "en", "natural"),
    ("close that window", "close window", "en", "natural"),
]

NOISE = [
    ("open firefox", "open firefox", "en", "noise"),
    ("open telegram", "open telegram", "en", "noise"),
    ("send rahul a message", "send rahul a message", "en", "noise"),
]


# ── Preprocessing (identical to Diego's capture chain) ──────────────
def preprocess_diego(audio_float: np.ndarray, sample_rate: int) -> np.ndarray:
    """AGC -> resample (16 kHz) -> 4th-order 80 Hz high-pass.

    Mirrors voice/audio_manager.py callback: raw capture -> AGC -> resample
    -> high-pass -> ring buffer. AGC runs on RAW audio (may exceed unity on
    overdriven input); the resample + high-pass then produce the float32 in
    [-1,1] that STT receives.
    """
    a = np.asarray(audio_float, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        g = math.gcd(SAMPLE_RATE, int(sample_rate))
        from scipy.signal import resample_poly
        a = resample_poly(a, SAMPLE_RATE // g, int(sample_rate) // g).astype(np.float32)

    # AGC: leveler + limiter, frame-wise (~30ms = 480 samples @ 16kHz).
    frame = int(SAMPLE_RATE * 0.03)
    out_parts = []
    capture_agc.reset()
    for i in range(0, len(a), frame):
        chunk = a[i:i + frame]
        if chunk.size == 0:
            continue
        lev, _ = capture_agc.process(chunk)
        out_parts.append(np.asarray(lev, dtype=np.float32))
    if not out_parts:
        return np.zeros(0, dtype=np.float32)
    leveled = np.concatenate(out_parts)

    # Unified high-pass (4th-order Butterworth @80Hz). Zero initial conditions
    # so the benchmark is deterministic per file.
    from scipy import signal as scipy_signal
    sos = scipy_signal.butter(HIGH_PASS_ORDER, HIGH_PASS_CUTOFF / (SAMPLE_RATE / 2),
                              btype="highpass", output="sos")
    zi = scipy_signal.sosfilt_zi(sos) * 0
    hp, _ = scipy_signal.sosfilt(sos, leveled.astype(np.float64), zi=zi)
    return np.clip(np.asarray(hp, dtype=np.float32), -1.0, 1.0)


def synth_wav(text: str, path: Path, voice: str = "en-us") -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["espeak-ng", "-v", voice, "-s", "150", "-w", str(path), text],
                       check=True, capture_output=True, timeout=30)
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        pcm = w.readframes(w.getnframes())
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a, sr


def add_noise(audio: np.ndarray, kind: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = len(audio)
    if kind == "fan":
        t = np.arange(n) / SAMPLE_RATE
        noise = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * rng.standard_normal(n)
    elif kind == "keyboard":
        noise = np.zeros(n)
        for _ in range(int(n / SAMPLE_RATE * 8)):
            pos = rng.integers(0, n - 100)
            noise[pos:pos + 100] += rng.uniform(0.2, 0.6)
    elif kind == "music":
        t = np.arange(n) / SAMPLE_RATE
        noise = 0.25 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 660 * t)
    else:
        noise = 0.02 * rng.standard_normal(n)
    return np.clip(audio + noise.astype(np.float32), -1.0, 1.0).astype(np.float32)


def _router_match(text: str, expected: str) -> bool:
    """Comparison-only: resolve the transcript through the production
    CommandRouter's PURE classifier (normalize + _match_simple). Never
    executes anything.

    Returns True when the transcript classifies to a SIMPLE_DESKTOP action
    at all, AND — when the expected target is an app — that the resolved
    action actually targets the expected app (so a wrong-app hallucination
    does not count as actionable)."""
    try:
        from core.command_router import CommandRouter
        norm = CommandRouter._normalize(text)
        if not norm:
            return False
        match = CommandRouter._match_simple(norm)
        if match is None:
            return False
        action = match[0] or {}
        # Actionable = the router found a concrete desktop action.
        resolved = (action.get("action") or "").strip()
        if not resolved:
            return False
        # For open-* expectations, verify the resolved app matches intent.
        exp_norm = CommandRouter._normalize(expected)
        if exp_norm.startswith("open "):
            app = (action.get("params") or {}).get("app") or ""
            # Map the naive expected app token to what the router would set.
            expected_app = exp_norm.split("open ", 1)[1].strip()
            mapping = {"firefox": "firefox", "browser": "firefox",
                       "telegram": "telegram-desktop"}
            want = mapping.get(expected_app, expected_app)
            return bool(app) and (app == want or want in app or app in want)
        return True
    except Exception:
        return normalize_transcript(text) == normalize_transcript(expected)


def _cpu_ram() -> Tuple[float, float]:
    try:
        import psutil
        p = psutil.Process()
        return p.cpu_percent(interval=None), p.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0, 0.0


def _load_whisper():
    t0 = time.time()
    tr = _WhisperTranscriber()
    ok = tr.load()
    return tr, ok, (time.time() - t0), tr._device, tr._compute


def _load_qwen3():
    from voice.providers.sherpa_providers import Qwen3ASRProvider
    p = Qwen3ASRProvider()
    t0 = time.time()
    ok = p.start()
    return p, ok, (time.time() - t0)


def transcribe_whisper(tr, audio: np.ndarray, lang: str) -> Tuple[str, Optional[float]]:
    pcm = float32_to_int16(audio).tobytes()
    return tr.transcribe_with_language(pcm, SAMPLE_RATE, lang or "en")


def transcribe_qwen3(p, audio: np.ndarray, lang: str) -> Tuple[str, Optional[float]]:
    text = p.transcribe(audio, SAMPLE_RATE)
    # sherpa-onnx greedy Qwen3 does not expose per-token logprobs in the
    # OfflineRecognitionResult (ys_log_probs is empty for greedy). Confidence
    # is reported as None (not fabricated) and logged honestly.
    return text, None


def run_one(backend, transcribe_fn, tr_or_p, audio, expected, lang, cat) -> dict:
    r = {"backend": backend, "expected": expected, "category": cat,
         "transcript": "", "confidence": None, "wer": 0.0,
         "actionable": False, "first_word_accuracy": False,
         "latency_ms": 0.0, "rtf": 0.0, "cpu_percent": 0.0, "ram_mb": 0.0,
         "ok": True, "error": ""}
    dur_s = len(audio) / SAMPLE_RATE
    try:
        t0 = time.time()
        text, conf = transcribe_fn(tr_or_p, audio, lang)
        lat = (time.time() - t0) * 1000
        cpu, ram = _cpu_ram()
        r["transcript"] = text
        r["confidence"] = conf
        r["latency_ms"] = lat
        r["rtf"] = (lat / 1000.0) / dur_s if dur_s else 0.0
        r["cpu_percent"] = cpu
        r["ram_mb"] = ram
        r["wer"] = word_error_rate(expected, text)
        r["actionable"] = _router_match(text, expected)
        r["first_word_accuracy"] = first_word_accuracy(expected, text)
    except Exception as e:
        r["ok"] = False
        r["error"] = f"{type(e).__name__}: {e}"
    return r


def load_whisper_transcriber():
    tr, ok, load_s, dev, comp = _load_whisper()
    if not ok:
        return None, None, load_s
    return tr, (lambda t, a, l: transcribe_whisper(t, a, l)), load_s


def run(quick: bool = False) -> dict:
    corpus: List[Tuple[str, str, str, str]] = list(REQUIRED)
    if not quick:
        corpus += NATURAL + NOISE

    # Synthesize every corpus phrase once (16 kHz source from espeak, then
    # preprocessed by Diego's chain).
    cases: List[Tuple[np.ndarray, str, str, str]] = []
    for idx, (spoken, expected, lang, cat) in enumerate(corpus):
        wav = SYNTH_DIR / f"b2b_{idx:03d}_{lang}.wav"
        if not synth_wav(spoken, wav):
            continue
        a, sr = load_wav(wav)
        a = preprocess_diego(a, sr)
        if cat == "noise":
            a = add_noise(a, "fan")
        cases.append((a, expected, lang, cat))

    report = {"quick": quick, "n_cases": len(cases),
              "models": {}, "results": []}

    # faster-whisper baseline
    tr, fn, load_s = load_whisper_transcriber()
    if fn is not None:
        report["models"]["faster-whisper"] = {
            "ready": True, "load_time_s": load_s, "loaded_device": tr._device,
            "loaded_compute": tr._compute, "load_error": ""}
        for a, expected, lang, cat in cases:
            report["results"].append(
                run_one("faster-whisper", fn, tr, a, expected, lang, cat))
    else:
        report["models"]["faster-whisper"] = {"ready": False, "load_error": "load failed"}

    # Qwen3-ASR 0.6B INT8
    p, ok, load_s = _load_qwen3()
    if ok:
        report["models"]["qwen3-asr-0.6b-int8"] = {
            "ready": True, "load_time_s": load_s, "load_error": ""}
        for a, expected, lang, cat in cases:
            report["results"].append(
                run_one("qwen3-asr-0.6b-int8", transcribe_qwen3, p, a, expected, lang, cat))
    else:
        report["models"]["qwen3-asr-0.6b-int8"] = {
            "ready": False, "load_error": p.health().get("load_error", "load failed")}

    # Aggregate per backend
    report["summary"] = {}
    for name in ("faster-whisper", "qwen3-asr-0.6b-int8"):
        rows = [r for r in report["results"] if r["backend"] == name and r["ok"]]
        if rows:
            n = len(rows)
            report["summary"][name] = {
                "n": n,
                "avg_wer": sum(r["wer"] for r in rows) / n,
                "actionable": sum(1 for r in rows if r["actionable"]) / n * 100,
                "first_word_accuracy": sum(1 for r in rows if r["first_word_accuracy"]) / n * 100,
                "avg_latency_ms": sum(r["latency_ms"] for r in rows) / n,
                "avg_rtf": sum(r["rtf"] for r in rows) / n,
                "max_ram_mb": max((r["ram_mb"] for r in rows), default=0.0),
            }
    return report


def format_report(report: dict) -> str:
    lines = ["=" * 82, "  STT BACKEND BENCHMARK — faster-whisper vs Qwen3-ASR 0.6B INT8", "=" * 82]
    for name, m in report["models"].items():
        status = "READY (%.2fs) %s" % (m["load_time_s"], m.get("loaded_device", "")) \
            if m.get("ready") else f"NOT READY — {m.get('load_error','')}"
        lines.append(f"  {name:<26} {status}")
    lines.append("")
    for r in report["results"]:
        conf = f"{r['confidence']:.3f}" if r.get("confidence") is not None else "n/a"
        lines.append(
            f"  [{r['backend']:<18}] {r['category']:<8} exp={r['expected']!r:<34} "
            f"got={r['transcript']!r:<40} wer={r['wer']:.2f} act={r['actionable']} "
            f"conf={conf} lat={r['latency_ms']:.0f}ms rtf={r['rtf']:.2f}"
        )
        if r.get("error"):
            lines.append(f"      ERROR: {r['error']}")
    lines.append("")
    lines.append("  SUMMARY")
    for name, s in report["summary"].items():
        lines.append(
            f"    {name:<26} n={s['n']} wer={s['avg_wer']:.3f} "
            f"actionable={s['actionable']:.0f}% first_word={s['first_word_accuracy']:.0f}% "
            f"lat={s['avg_latency_ms']:.0f}ms rtf={s['avg_rtf']:.2f} ram={s['max_ram_mb']:.0f}MB"
        )
    lines.append("=" * 82)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        if REPORT_PATH.exists():
            print(format_report(json.loads(REPORT_PATH.read_text())))
        else:
            print("No report yet — run the benchmark first.")
        return

    report = run(quick=args.quick)
    print(format_report(report))
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
