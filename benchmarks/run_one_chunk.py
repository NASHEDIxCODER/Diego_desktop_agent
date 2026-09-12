#!/usr/bin/env python3
"""Phase 21E chunked runner: one benchmark slice per invocation (fits tool timeout).

Usage:
  .venv/bin/python benchmarks/run_one_chunk.py <model> <chunk> [state_suffix]

Chunks: planAB, planCD(safety diag/revise part1), planEFGH, planIJ, safety, diagnose, revise, reflect.
State accumulates in benchmarks/_21e_state_<safe_model><suffix>.json.
Finalize with: .venv/bin/python benchmarks/run_one_chunk.py <model> finalize [suffix]
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.context_monitor import context_monitor  # noqa: E402
from ai.reasoning_model import OllamaReasoningModel  # noqa: E402

import run_reasoning_model_benchmark as B  # noqa: E402

sys.path.insert(0, str(ROOT / "benchmarks"))


def state_path(model: str, suffix: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in ".-") else "_" for c in model)
    return ROOT / "benchmarks" / f"_21e_state_{safe}{suffix}.json"


def load_state(p: Path):
    if p.exists():
        return json.loads(p.read_text())
    return {"calls": {"plan": [], "revise": [], "diagnose": [], "reflect": []}, "reps_done": 0}


CHUNKS = {
    # incremental chunks each do a few model calls (each call ~2-10s)
    "p1": ("plan", ["A"]),
    "p2": ("plan", ["B"]),
    "p3": ("plan", ["C"]),
    "p4": ("plan", ["G"]),
    "p5": ("plan", ["H"]),
    "p6": ("plan", ["I"]),
    "p7": ("plan", ["J"]),
    "s1": ("plan", ["S1"]),
    "s2": ("plan", ["S2"]),
    "s3": ("plan", ["S3"]),
    "d1": ("diagnose", ["D"]),
    "e1": ("revise", ["E"]),
    "e2": ("revise", ["F"]),
    "r1": ("reflect", ["R"]),
}

CASES = {c["id"]: c for c in B.DATASET["cases"]}
CASES["R"] = B.DATASET["reflection_case"]


async def do_chunk(model: str, chunk: str) -> dict:
    role, ids = CHUNKS[chunk]
    m = OllamaReasoningModel(model=model)
    m._checked = True
    out = []
    for cid in ids:
        case = CASES[cid]
        t0 = time.perf_counter()
        try:
            if role == "plan":
                extra = (" " + case["lesson"]) if case.get("lesson") else ""
                res = await m.plan(case["goal"], context=B.ctx_for(case.get("context_level", "normal")) + extra, timeout_s=120)
            elif role == "diagnose":
                res = await m.diagnose(case["goal"], case["failed_action"], case["observed_result"], case["error"], case["attempt"], timeout_s=120)
            elif role == "revise":
                res = await m.revise_plan(case["goal"], case["observation"], case["completed"], case["remaining"], timeout_s=120)
            else:
                res = await m.reflect(case["goal"], case["outcome"], timeout_s=120)
            dt = time.perf_counter() - t0
            u = context_monitor.last_usage
            out.append({"case": cid, "status": res.status.value, "latency_s": round(dt, 3),
                        "model": res.model, "error": (res.error or "")[:120],
                        "input_tokens": u.input_tokens if u else None,
                        "output_tokens": u.output_tokens if u else None,
                        "data_keys": sorted((res.data or {}).keys()),
                        "raw": (res.raw_text or "")[:400], "data": res.data})
        except Exception as ex:  # noqa: BLE001
            out.append({"case": cid, "status": "harness_error",
                        "latency_s": round(time.perf_counter() - t0, 3),
                        "error": str(ex)[:120]})
    try:
        await m.close()
    except Exception:  # noqa: BLE001
        pass
    return {"role": role, "entries": out}


async def main() -> None:
    model = sys.argv[1]
    chunk = sys.argv[2]
    suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    sp = state_path(model, suffix)
    st = load_state(sp)
    if chunk == "finalize":
        summary = B.score_calls(st["calls"])
        quality: dict = {}
        for role, entries in st["calls"].items():
            flags: dict = {}
            for e in entries:
                for k, v in (e.get("scores") or {}).items():
                    if isinstance(v, bool):
                        flags.setdefault(k, []).append(v)
            quality[role] = {k: round(sum(v) / len(v), 3) for k, v in flags.items()}
        by_case: dict = {}
        for e in st["calls"]["plan"]:
            if e.get("status") == "ok":
                s = e.get("scores", {})
                by_case.setdefault(e["case"], []).append(bool(s.get("correct_plan") and s.get("passes_validator")))
        dcases = {c["id"]: c for c in B.DATASET["cases"]}
        ctx_eff = {}
        for level in ("short", "normal", "large"):
            vals = [v for cid, vs in by_case.items() for v in vs
                    if dcases.get(cid, {}).get("context_level") == level]
            ctx_eff[level] = round(sum(vals) / len(vals), 3) if vals else None
        try:
            lst = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001
            lst = ""
        result = {"model": model, "roles": summary, "quality": quality,
                  "context_efficiency_valid_rate": ctx_eff,
                  "prompt_est_tokens": {"short": context_monitor.estimate_tokens(B.SHORT_CTX),
                                        "normal": context_monitor.estimate_tokens(B.FULL_CTX),
                                        "large": context_monitor.estimate_tokens(B.LARGE_CTX)},
                  "resource": {"ollama_list": lst.strip()[:800]}, "calls": st["calls"]}
        safe = "".join(c if (c.isalnum() or c in ".-") else "_" for c in model)
        outp = ROOT / "benchmarks" / f"reasoning_21e_results_{safe}{suffix}.json"
        outp.write_text(json.dumps(result, indent=1))
        print(json.dumps({"roles": summary, "quality": quality, "ctx": ctx_eff}, indent=1))
        print(f"WROTE {outp}")
        return
    r = await do_chunk(model, chunk)
    st["calls"][r["role"]].extend(r["entries"])
    sp.write_text(json.dumps(st, indent=1))
    for e in r["entries"]:
        print(json.dumps({"case": e["case"], "status": e["status"],
                          "latency_s": e["latency_s"],
                          "in": e.get("input_tokens"), "out": e.get("output_tokens"),
                          "keys": e.get("data_keys"), "err": e.get("error")}))


if __name__ == "__main__":
    asyncio.run(main())
