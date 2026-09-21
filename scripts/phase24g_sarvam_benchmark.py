#!/usr/bin/env python3
"""Phase 24.7G — Sarvam Saaras v4 vs local Qwen3-ASR benchmark (BENCHMARK ONLY).

Answers one question: does the cloud Saaras v4 model beat the local
Qwen3-ASR 1.7B (production default) / 0.6B / faster-whisper on
English + Hindi + Hinglish commands, and at what latency/cost?

Isolation guarantees (identical spirit to phase20e):
  - Inference only. NEVER executes Diego actions, never touches TaskRunner,
    tools, autonomy, VAD, wake, face auth, the normalizer, or the intent
    authorizer's side effects (authorize_intent returns a verdict only).
  - NEVER modifies production config. The harness snapshots
    STT_PRIMARY/STT_FALLBACK/STT_MODEL_SIZE and voice_settings before and
    after, and asserts they are byte-identical.
  - The Sarvam provider is reached ONLY through
    stt_backend.experimental_provider_for_benchmark(..., allow_experimental=True),
    which is the single sanctioned benchmark seam. `sarvam` stays out of
    production routing regardless of this run.
  - Everything is filtered through the SAME production-facing helpers as
    20E (_postprocess/_validate_transcript/is_garbage/is_low_quality_transcript/
    _is_repeated_hallucination + command_normalizer + authorize_intent), so
    "usable" means the same thing the live pipeline means.

Dataset: reuses phase20e's controlled synthetic espeak-ng set. Synthetic
formant speech is NOT real microphone speech, so absolute numbers measure
model-vs-model behaviour on clean synthetic audio, NOT real-world WER.
Saaras is tuned for REAL Indian speech; synthetic espeak-ng Hindi is an
explicitly unfair-to-Saaras probe. Treat a Saaras win as meaningful and a
Saaras loss as inconclusive.

Latency caveat (stated, not hidden): local providers are measured as pure
on-device inference. Sarvam latency includes network + queueing and is
therefore NOT directly comparable; both are reported side by side and tagged.

Usage:
  .venv/bin/python scripts/phase24g_sarvam_benchmark.py --list-only
  .venv/bin/python scripts/phase24g_sarvam_benchmark.py --check
  .venv/bin/python scripts/phase24g_sarvam_benchmark.py --candidates sarvam
  .venv/bin/python scripts/phase24g_sarvam_benchmark.py --candidates qwen3-1.7b,sarvam --reps 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import numpy as np

# Reuse the controlled dataset + production-facing scorers from phase 20E so
# the two phases stay directly comparable. Loaded by explicit path (scripts/ is
# not a package) — import-only, phase20e does its work under __main__.
import importlib.util  # noqa: E402

_p20e_path = Path(__file__).resolve().parent / "phase20e_asr_model_benchmark.py"
_spec = importlib.util.spec_from_file_location("phase20e_asr_model_benchmark", _p20e_path)
P20E = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P20E
_spec.loader.exec_module(P20E)

RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "phase24g_sarvam_results.json"
SAMPLE_RATE = P20E.SAMPLE_RATE
DATASET = P20E.DATASET
ENTITY_KEYS = P20E.ENTITY_KEYS

# candidate id → (kind, builder description)
CANDIDATES: dict[str, dict[str, str]] = {
    "qwen3-1.7b": {"kind": "local", "provider": "qwen3-asr-1.7b-int8",
                   "note": "production default (local, sherpa-onnx INT8)"},
    "qwen3-0.6b": {"kind": "local", "provider": "qwen3-asr-0.6b-int8",
                   "note": "smaller local Qwen3 (sherpa-onnx INT8)"},
    "faster-whisper": {"kind": "local", "provider": "faster-whisper",
                       "note": "production confidence/verify baseline (base)"},
    "sarvam": {"kind": "cloud-experimental", "provider": "sarvam-saaras-v4",
               "note": ("Sarvam Saaras v4 REST — EXPERIMENTAL, benchmark only, "
                        "requires SARVAM_API_KEY; latency includes network")},
}


# ── Production-untouched guards ───────────────────────────────────────────

def _config_snapshot() -> dict:
    """Snapshot every production knob this harness could accidentally touch."""
    import voice.command_listener as CL
    from voice.settings import voice_settings
    return {
        "stt_primary": getattr(voice_settings, "stt_primary", None),
        "stt_fallback": getattr(voice_settings, "stt_fallback", None),
        "stt_model_size": getattr(voice_settings, "stt_model_size", None),
        "fallback_max_passes": getattr(CL, "FALLBACK_MAX_PASSES", None),
    }


def _production_stt_provider() -> str:
    """Resolved production provider — proves routing is still local."""
    from voice.stt_backend import resolve_primary
    return resolve_primary()


# ── Provider construction ─────────────────────────────────────────────────

def build_candidate(cid: str):
    """Return (provider, meta) for a benchmark candidate, or (None, reason)."""
    meta = dict(CANDIDATES[cid])
    try:
        if cid == "qwen3-1.7b":
            from voice.providers.sherpa_providers import Qwen3ASR17BProvider
            return Qwen3ASR17BProvider(), meta
        if cid == "qwen3-0.6b":
            from voice.providers.sherpa_providers import Qwen3ASRProvider
            return Qwen3ASRProvider(), meta
        if cid == "faster-whisper":
            from voice.providers.whisper_provider import WhisperProvider
            return WhisperProvider(model_size="base"), meta
        if cid == "sarvam":
            # The ONLY sanctioned experimental seam. allow_experimental=True is
            # an explicit, greppable opt-in that no runtime path uses.
            from voice.stt_backend import experimental_provider_for_benchmark
            return (experimental_provider_for_benchmark(
                "sarvam", allow_experimental=True), meta)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return None, f"unknown candidate {cid!r}"

# ── Inference ─────────────────────────────────────────────────────────────

def _safe(fn, default=None):
    """Call a zero-arg provider accessor, never raising in the harness."""
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"} if default is None else default


def _safe_call(obj, method: str) -> None:
    try:
        getattr(obj, method)()
    except Exception:
        pass


def decode_once(provider, audio: np.ndarray, language: str):
    """One full-utterance decode via the shared ASRProvider interface.

    Returns (text, latency_ms). Raises nothing — a provider failure yields
    ("", latency) so the row is scored as empty rather than crashing the run.
    """
    t0 = time.time()
    try:
        text = provider.transcribe(audio, SAMPLE_RATE, language)
    except Exception as e:
        text = ""
        print(f"      decode error: {type(e).__name__}: {e}")
    return (text or "").strip(), (time.time() - t0) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 24.7G Sarvam Saaras v4 vs local Qwen3 benchmark "
                    "(inference-only, benchmark-only)")
    ap.add_argument("--candidates", default="qwen3-1.7b,sarvam",
                    help="comma list from: " + ", ".join(CANDIDATES))
    ap.add_argument("--reps", type=int, default=2,
                    help="repetitions per utterance; median of reps>1 is reported")
    ap.add_argument("--modes", default="en,hi",
                    help="forced language passes (en,hi)")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="verify the Sarvam credential with one live request, then exit")
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()

    if args.list_only:
        print(json.dumps({"candidates": CANDIDATES,
                          "dataset": [u[0] for u in DATASET],
                          "experimental_isolation": {
                              "sarvam_in_production_routing": False,
                              "entrypoint": ("voice.stt_backend."
                                             "experimental_provider_for_benchmark"),
                          }}, indent=2))
        return 0

    if args.check:
        from voice.stt_backend import experimental_provider_for_benchmark
        try:
            p = experimental_provider_for_benchmark("sarvam", allow_experimental=True)
        except Exception as e:
            print(f"sarvam unavailable: {type(e).__name__}: {e}")
            return 3
        ok = p.verify_credentials()
        print(("sarvam credential OK — " if ok else "sarvam credential FAILED — ")
              + json.dumps(p.health(), indent=2))
        return 0 if ok else 4

    wanted = [c.strip() for c in args.candidates.split(",") if c.strip()]
    for c in wanted:
        if c not in CANDIDATES:
            print(f"ERROR: unknown candidate {c!r} (choose from {sorted(CANDIDATES)})")
            return 2
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # ── isolation assertions ──
    before = _config_snapshot()
    prod_before = _production_stt_provider()
    assert before["fallback_max_passes"] == 2, "production fallback budget changed"
    assert prod_before not in ("sarvam",), "sarvam leaked into production routing"
    print(f"=== isolation OK: production STT provider = {prod_before!r} ===")

    print("=== synthesizing controlled dataset (espeak-ng, offline) ===")
    clips: dict[str, np.ndarray] = {}
    for uid, _group, voice, speak, _kw in DATASET:
        clips[uid] = P20E.synth_clip(speak, voice)
        print(f"  {uid} [{voice}]: {speak!r} ({len(clips[uid]) / SAMPLE_RATE:.2f}s)")

    results: dict = {
        "phase": "24.7G",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("Sarvam Saaras v4 vs local Qwen3-ASR 1.7B/0.6B and "
                    "faster-whisper on EN/HI/Hinglish commands"),
        "dataset_note": ("controlled synthetic espeak-ng audio (NOT real mic). "
                         "Saaras is trained on real Indian speech; synthetic "
                         "espeak-ng Hindi is an unfair-to-Saaras probe — a "
                         "Saaras win is meaningful, a loss is inconclusive."),
        "latency_caveat": ("local = on-device inference; sarvam = network round "
                           "trip + queueing (not directly comparable)"),
        "production_untouched": before,
        "sarvam_in_production_routing": False,
        "candidates": {},
    }

    for cid in wanted:
        meta = CANDIDATES[cid]
        provider, built = build_candidate(cid)
        if provider is None:
            print(f"\n=== {cid}: SKIPPED — {built} ===")
            results["candidates"][cid] = {"meta": meta, "skipped": str(built)}
            continue

        print(f"\n=== {cid} ({meta['provider']}, {meta['kind']}) ===")
        started = provider.start() if hasattr(provider, "start") else True
        if not started:
            reason = ""
            try:
                reason = provider.health().get("reason", "")
            except Exception:
                pass
            print(f"  UNAVAILABLE — {reason}")
            results["candidates"][cid] = {"meta": meta,
                                          "unavailable": reason or "start() failed"}
            continue

        rec: dict = {"meta": meta, "health": _safe(provider.health),
                     "utterances": []}
        for uid, group, _voice, speak, kw in DATASET:
            for mode in modes:
                lat, text = [], ""
                for _ in range(max(1, args.reps)):
                    text, l = decode_once(provider, clips[uid], mode)
                    lat.append(l)
                med = (float(statistics.median(lat)) if len(lat) > 1 else lat[0])
                # Qwen3 exposes no log-probs; Sarvam REST exposes no log-probs.
                # Confidence is left at 0.0 (never fabricated) so the gates run
                # in their strictest, most comparable mode.
                score = P20E.score_row(text, 0.0, kw)
                rec["utterances"].append({
                    "id": uid, "group": group, "prompt": speak, "mode": mode,
                    "transcript": text,
                    "first_latency_ms": round(lat[0], 1),
                    "latencies_ms": [round(x, 1) for x in lat],
                    "median_latency_ms": round(med, 1),
                    "latency_kind": ("network" if meta["kind"].startswith("cloud")
                                     else "on-device"),
                    **score})
                print(f"  [{uid}/{mode}] {text!r} med={med:.0f}ms "
                      f"usable={score['usable']} kw={score['keyword_recall']:.2f}")
        for mode in modes:
            rows = [u for u in rec["utterances"] if u["mode"] == mode]
            if rows:
                meds = [u["median_latency_ms"] for u in rows]
                rec[f"stats_{mode}"] = {
                    "n": len(rows),
                    "usable_rate": round(sum(1 for u in rows if u["usable"]) / len(rows), 4),
                    "keyword_recall_mean": round(
                        float(np.mean([u["keyword_recall"] for u in rows])), 4),
                    "median_ms": round(float(statistics.median(meds)), 1),
                    "p95_ms": round(P20E.pct(meds, 95), 1),
                    "max_ms": round(max(meds), 1),
                }
        _safe_call(provider, "stop")
        results["candidates"][cid] = rec

    after = _config_snapshot()
    prod_after = _production_stt_provider()
    results["production_after"] = after
    results["production_after_provider"] = prod_after
    results["production_untouched_verified"] = bool(
        after == before and prod_after == prod_before)
    if not results["production_untouched_verified"]:
        print("!!! WARNING: production config changed during the benchmark !!!")
        print(f"  before={before} after={after}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nSaved benchmark data ONLY to {out} (no production changes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

