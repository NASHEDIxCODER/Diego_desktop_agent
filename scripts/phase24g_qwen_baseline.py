#!/usr/bin/env python3
"""Phase 24.7Q - Qwen3-ASR 1.7B baseline on real Diego audio + 20E synthetic.

BENCHMARK ONLY. Inference only; never executes actions, never touches
production routing/config. Sarvam side intentionally ABSENT (no API key):
this run freezes the dataset, clips, and preprocessing byte-identical so the
Saaras side merges 1:1 later.

Rows (~35): real-diego (Diego_voice_backup.wav, Diego.wav) + real-wake
(11 mic "hello Diego" positives) + real-noise (2 non-speech controls) +
synth-en/hx/hi/mx/amb (14 phase-20E DATASET rows, espeak-ng, labels verbatim).

Preprocessing (shared by both future providers): wave int16 -> mono-mean ->
/32768 -> linear resample to 16 kHz float32. Qwen goes through the
production composite int16 path (build_stt_backend); Sarvam later gets the
same float32.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "phase20e_asr_model_benchmark",
    PROJECT_ROOT / "scripts" / "phase20e_asr_model_benchmark.py")
P20E = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P20E
_spec.loader.exec_module(P20E)

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "phase24g_qwen_baseline.json"
SAMPLE_RATE = 16000


def load_wav_to_16k_mono_f32(path: str | Path) -> tuple[np.ndarray, float, dict]:
    """Any PCM WAV -> 16 kHz mono float32 [-1,1]. Returns (audio, dur, meta)."""
    with wave.open(str(path), "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: only 16-bit PCM (sampwidth={sw})")
    a = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    a /= 32768.0
    if sr != SAMPLE_RATE:
        n = int(len(a) * SAMPLE_RATE / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n),
                      np.arange(len(a)), a).astype(np.float32)
    return a, len(a) / SAMPLE_RATE, {"src_sr": sr, "src_ch": nch,
                                     "file": str(path)}


def to_pcm16_bytes(audio_f32: np.ndarray) -> bytes:
    """float32 [-1,1] -> int16 PCM bytes (production composite input)."""
    return (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def build_rows() -> list[dict]:
    """Assemble the frozen 4-block row list (real + synthetic)."""
    rows: list[dict] = []
    for rel in ("Diego_voice_backup.wav", "Diego.wav"):
        p = PROJECT_ROOT / rel
        if p.exists():
            rows.append({"id": f"real-{p.stem}", "group": "real-diego",
                         "kind": "real", "path": str(p), "keywords": [],
                         "language": "en"})
    for p in sorted((PROJECT_ROOT / "models/wake/positives").glob("*.wav")):
        rows.append({"id": f"wake-{p.stem}", "group": "real-wake",
                     "kind": "real", "path": str(p),
                     "keywords": ["diego", "hello"], "language": "en"})
    for name in ("neg_esp_010", "neg_sil_066"):
        p = PROJECT_ROOT / "models/wake/negatives" / f"{name}.wav"
        if p.exists():
            rows.append({"id": f"noise-{name}", "group": "real-noise",
                         "kind": "real", "path": str(p), "keywords": [],
                         "language": "en"})
    for uid, group, voice, speak, kw in P20E.DATASET:
        rows.append({"id": uid, "group": f"synth-{group}", "kind": "synthetic",
                     "path": None, "voice": voice, "prompt": speak,
                     "keywords": list(kw),
                     "language": "hi" if voice == "hi" else "en"})
    return rows

def main() -> int:
    """Run the frozen baseline: check mode or full Qwen3-1.7B decode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="dataset + preprocessing only, no model")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()

    rows = build_rows()
    print(f"rows: {len(rows)} "
          f"({sum(1 for r in rows if r['kind'] == 'real')} real, "
          f"{sum(1 for r in rows if r['kind'] == 'synthetic')} synthetic)")

    clips: dict[str, dict] = {}
    for r in rows:
        if r["kind"] == "real":
            audio, dur, meta = load_wav_to_16k_mono_f32(r["path"])
        else:
            audio = P20E.synth_clip(r["prompt"], r["voice"])
            dur = len(audio) / SAMPLE_RATE
            meta = {"src": "espeak-ng", "voice": r["voice"],
                    "prompt": r["prompt"]}
        clips[r["id"]] = {"audio": audio, "dur": dur, "meta": meta}

    if args.check:
        for r in rows:
            c = clips[r["id"]]
            print(f"  [{r['id']}/{r['group']}] dur={c['dur']:.2f}s "
                  f"n={len(c['audio'])} meta={c['meta']}")
        print("CHECK OK - no model loaded, no config touched.")
        return 0

    from voice.stt_backend import build_stt_backend  # noqa: E402

    print("loading production composite (Qwen3-1.7B + whisper verify) ...")
    t0 = time.time()
    backend = build_stt_backend()
    print(f"  type={type(backend).__name__} ready={backend.ready} "
          f"load={time.time() - t0:.1f}s")
    assert type(backend).__name__ == "_CompositeTranscriber", \
        f"not Qwen3 composite: {type(backend).__name__}"

    results: dict = {
        "phase": "24.7Q",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": "qwen3-1.7b",
        "kind": "local",
        "note": "production default via build_stt_backend(). "
                "Sarvam side pending API key; clips frozen for 1:1 merge.",
        "preprocessing": "wave->mono-mean->/32768->linear-16k-f32; "
                         "Qwen via production int16 composite path",
        "utterances": [],
    }
    for r in rows:
        c = clips[r["id"]]
        pcm = to_pcm16_bytes(c["audio"])
        lat: list[float] = []
        text, conf = "", None
        for _ in range(max(1, args.reps)):
            t1 = time.time()
            text, conf = backend.transcribe_with_language(
                pcm, SAMPLE_RATE, r["language"])
            lat.append((time.time() - t1) * 1000.0)
        med = float(statistics.median(lat))
        entry = {"id": r["id"], "group": r["group"], "kind": r["kind"],
                 "dur_s": round(c["dur"], 3), "clip_meta": c["meta"],
                 "transcript": text, "confidence": conf,
                 "first_latency_ms": round(lat[0], 1),
                 "latencies_ms": [round(x, 1) for x in lat],
                 "median_latency_ms": round(med, 1),
                 "rtf": round(med / max(c["dur"] * 1000.0, 1.0), 3),
                 "latency_kind": "on-device"}
        if r["keywords"]:
            entry.update(P20E.score_row(text, 0.0, r["keywords"]))
        else:
            entry.update({"usable": bool((text or "").strip()),
                          "keyword_recall": None,
                          "note": "no ground truth; verbatim only"})
        results["utterances"].append(entry)
        print(f"  [{r['id']}] {text!r} med={med:.0f}ms rtf={entry['rtf']} "
              f"kw={entry.get('keyword_recall')}")

    for g in sorted({r["group"] for r in rows}):
        uu = [u for u in results["utterances"] if u["group"] == g]
        meds = [u["median_latency_ms"] for u in uu]
        kw = [u["keyword_recall"] for u in uu
              if u.get("keyword_recall") is not None]
        results[f"stats_{g}"] = {
            "n": len(uu),
            "median_ms": round(float(statistics.median(meds)), 1),
            "p95_ms": round(P20E.pct(meds, 95), 1),
            "keyword_recall_mean": (round(float(np.mean(kw)), 4)
                                    if kw else None),
            "nonempty_rate": round(sum(1 for u in uu
                                       if (u["transcript"] or "").strip())
                                   / len(uu), 4),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nSaved Qwen3 baseline ONLY to {out} (no production changes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
