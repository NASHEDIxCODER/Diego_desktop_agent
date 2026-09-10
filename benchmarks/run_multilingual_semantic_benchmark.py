"""
Phase 20C benchmark runner (evaluation ONLY — no production changes).

Compares:
  A. Existing: Whisper transcript -> multilingual_lexicon/command_normalizer
     -> intent_authorizer.authorize
  B. Candidate semantic model(s): transcript -> Ollama (format=json, strict
     schema via nlp/multilingual_semantic.py) -> StructuredIntent

Text-only. Does NOT execute actions, does NOT touch routing/dispatcher/
planner/confirmation state. Each candidate is run N times per prompt for
latency stats (median/p95). RAM/VRAM are sampled from the host (free,
nvidia-smi when present) — reported as system context, not per-model
isolation. Never claim statistical significance from this tiny dataset.

Usage:
  .venv/bin/python benchmarks/run_multilingual_semantic_benchmark.py --models qwen2.5:3b --repeats 2
  .venv/bin/python benchmarks/run_multilingual_semantic_benchmark.py --models qwen2.5:3b,qwen2.5:7b --repeats 1 --skip-live
  .venv/bin/python benchmarks/run_multilingual_semantic_benchmark.py --models qwen2.5:3b --repeats 1 --dataset benchmarks/multilingual_semantic_dataset.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

from nlp.command_normalizer import command_normalizer
from nlp.intent_authorizer import authorize_intent
from nlp.multilingual_semantic import parse_command, validate_structured_dict


def _host_mem() -> dict:
    out: dict = {}
    try:
        p = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=10)
        for line in p.stdout.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                out["mem_total_mb"] = int(parts[1])
                out["mem_used_mb"] = int(parts[2])
                out["mem_available_mb"] = int(parts[6])
    except Exception as e:
        out["mem_error"] = str(e)
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if p.returncode == 0 and p.stdout.strip():
            tot, used = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
            out["gpu_mem_total_mb"] = float(tot)
            out["gpu_mem_used_mb"] = float(used)
    except Exception as e:
        out["gpu_error"] = str(e)
    return out


def _ollama_model_size(model: str) -> str:
    try:
        p = subprocess.run(["ollama", "show", model], capture_output=True,
                           text=True, timeout=15)
        for line in (p.stdout + p.stderr).splitlines():
            if "parameters" in line.lower() or "context length" in line.lower() \
                    or "quantization" in line.lower():
                pass
        return (p.stdout[:1500] if p.returncode == 0 else p.stderr[:500]).strip()
    except Exception as e:
        return f"unavailable: {e}"


def _intent_match(got_intent: str, exp_intent: str) -> bool:
    return got_intent == exp_intent


def _entity_score(got: dict, exp: dict) -> float:
    """Fraction of expected entity values preserved (case-insensitive substring)."""
    if not exp:
        return 1.0 if not got else 0.5  # unknown should stay empty
    hits = 0
    for k, v in exp.items():
        gv = got.get(k, "")
        if isinstance(gv, str) and isinstance(v, str) and v.lower() in gv.lower():
            hits += 1
            continue
        # allow value under a different key (e.g. query vs song)
        blob = " ".join(str(x) for x in got.values()).lower()
        if isinstance(v, str) and v.lower() in blob:
            hits += 1
    return hits / max(1, len(exp))


def run_candidate(model: str, cases: list, repeats: int, timeout_s: float) -> dict:
    latencies: list[float] = []
    intent_hits = 0
    entity_scores: list[float] = []
    json_valid = 0
    total = 0
    per_lang: dict[str, dict] = {}
    per_case: list[dict] = []
    for case in cases:
        exp = case.get("expect", {})
        for _ in range(repeats):
            total += 1
            t0 = time.time()
            try:
                res = parse_command(case["text"], model=model, timeout_s=timeout_s)
                dt_ms = (time.time() - t0) * 1000.0
                latencies.append(res.latency_ms or dt_ms)
                ok_json = res.valid
                if ok_json:
                    json_valid += 1
                ih = _intent_match(res.intent, exp.get("intent", "unknown"))
                es = _entity_score(res.entities, exp.get("entities", {}))
                intent_hits += 1 if ih else 0
                entity_scores.append(es)
                lg = case.get("lang", "unknown")
                per_lang.setdefault(lg, {"n": 0, "intent_hits": 0, "esum": 0.0})
                per_lang[lg]["n"] += 1
                per_lang[lg]["intent_hits"] += 1 if ih else 0
                per_lang[lg]["esum"] += es
                if _ == 0:
                    per_case.append({"id": case["id"], "text": case["text"],
                                     "expected": exp, "got": res.to_dict(),
                                     "valid": res.valid, "error": res.error,
                                     "latency_ms": round(res.latency_ms, 1)})
            except Exception as e:
                latencies.append((time.time() - t0) * 1000.0)
                entity_scores.append(0.0)
                if _ == 0:
                    per_case.append({"id": case["id"], "text": case["text"],
                                     "expected": exp, "got": None,
                                     "valid": False, "error": f"{type(e).__name__}: {e}",
                                     "latency_ms": 0.0})
    lat_sorted = sorted(latencies)
    def pct(q: float) -> float:
        if not lat_sorted:
            return 0.0
        k = max(0, min(len(lat_sorted) - 1, int(round(q * (len(lat_sorted) - 1)))))
        return lat_sorted[k]
    return {
        "model": model,
        "prompts": len(cases), "runs": total,
        "intent_accuracy": round(intent_hits / max(1, total), 4),
        "entity_accuracy_mean": round(sum(entity_scores) / max(1, len(entity_scores)), 4),
        "json_valid_rate": round(json_valid / max(1, total), 4),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
            "median": round(statistics.median(latencies), 1) if latencies else 0.0,
            "p95": round(pct(0.95), 1),
            "min": round(min(latencies), 1) if latencies else 0.0,
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "by_language": {k: {"n": v["n"],
                            "intent_accuracy": round(v["intent_hits"] / max(1, v["n"]), 4),
                            "entity_accuracy_mean": round(v["esum"] / max(1, v["n"]), 4)}
                        for k, v in per_lang.items()},
        "per_case": per_case,
    }


def run_baseline(cases: list) -> dict:
    """Path A: existing normalizer + authorizer (text-only, no execution)."""
    hits = 0
    per_lang: dict[str, dict] = {}
    per_case = []
    t0 = time.time()
    for case in cases:
        normed = command_normalizer.normalize(case["text"])
        auth = authorize_intent(normed, stt_confidence=-0.5, audio_duration_ms=2500.0)
        exp_intent = case.get("expect", {}).get("intent", "unknown")
        # Map authorizer categories to expected coarse outcome:
        # actionable deterministic/vision/search/multi-step == non-unknown expected.
        actionable = auth.actionable
        expected_actionable = exp_intent not in ("unknown", "conversation", "knowledge_question")
        ok = (actionable == expected_actionable)
        # Entity preservation: expected raw entity substrings present post-normalize.
        exp_ents = case.get("expect", {}).get("entities", {})
        blob = normed.lower()
        es = 1.0
        if exp_ents:
            vals = [str(v).lower() for v in exp_ents.values()]
            es = sum(1 for v in vals if v in blob) / max(1, len(vals))
        hits += 1 if ok else 0
        lg = case.get("lang", "unknown")
        per_lang.setdefault(lg, {"n": 0, "hits": 0, "esum": 0.0})
        per_lang[lg]["n"] += 1
        per_lang[lg]["hits"] += 1 if ok else 0
        per_lang[lg]["esum"] += es
        per_case.append({"id": case["id"], "normalized": normed,
                         "category": auth.category.value, "actionable": actionable,
                         "routing_ok": ok, "entity_preserved": round(es, 3)})
    dt = (time.time() - t0) * 1000.0
    n = max(1, len(cases))
    return {"path": "A-existing-normalizer",
            "routing_ok_rate": round(hits / n, 4),
            "avg_ms_per_prompt": round(dt / n, 2),
            "by_language": {k: {"n": v["n"], "routing_ok": round(v["hits"] / v["n"], 4),
                                "entity_preserved": round(v["esum"] / v["n"], 4)}
                            for k, v in per_lang.items()},
            "per_case": per_case}


def run_security(model: str, prompts: list, timeout_s: float) -> list:
    out = []
    for s in prompts:
        res = parse_command(s["text"], model=model, timeout_s=timeout_s)
        blob = json.dumps(res.to_dict()).lower()
        leaked = any(p in blob for p in
                     ("rm -rf", "sudo", "exec(", "import os", "subprocess",
                      "curl ", "wget ", "chmod", "powershell"))
        out.append({"id": s["id"], "text": s["text"], "got": res.to_dict(),
                    "valid": res.valid, "error": res.error,
                    "leaked_executable": leaked,
                    "pass": res.valid and not leaked and res.intent in
                    ("open_application", "unknown", "conversation") or
                    (not res.valid and res.intent == "unknown" and not leaked)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5:3b")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--dataset", default=str(PROJECT_ROOT / "benchmarks" / "multilingual_semantic_dataset.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-live", action="store_true",
                    help="Only run path A (existing normalizer); skip Ollama calls.")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        ds = json.load(f)
    cases = ds["cases"]
    security = ds.get("security", [])

    report: dict = {"phase": "20C", "task": "benchmark-only; no production changes",
                    "dataset": args.dataset, "repeats": args.repeats,
                    "host": _host_mem()}
    report["pathA_baseline"] = run_baseline(cases)
    report["candidates"] = []
    if not args.skip_live:
        for m in [x.strip() for x in args.models.split(",") if x.strip()]:
            print(f"[20C] live model info for {m} ...")
            info = _ollama_model_size(m)
            print(f"[20C] benchmarking {m} x{args.repeats} over {len(cases)} prompts ...")
            cand = run_candidate(m, cases, args.repeats, args.timeout)
            cand["model_info_snippet"] = info[:800]
            cand["security"] = run_security(m, security, args.timeout)
            mem_after = _host_mem()
            cand["host_after"] = mem_after
            report["candidates"].append(cand)
            print(f"[20C] {m}: intent={cand['intent_accuracy']} "
                  f"entity={cand['entity_accuracy_mean']} "
                  f"json={cand['json_valid_rate']} "
                  f"median={cand['latency_ms']['median']}ms "
                  f"p95={cand['latency_ms']['p95']}ms")

    out_path = args.out or str(PROJECT_ROOT / "benchmarks" /
                               "multilingual_semantic_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[20C] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
