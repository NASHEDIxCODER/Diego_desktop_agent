#!/usr/bin/env python3
"""Phase 21E — fixed reasoning-model benchmark runner (BENCHMARK ONLY).

Measures one local Ollama model at a time against the fixed dataset
benchmarks/reasoning_model_dataset.json. The model only produces
structured JSON proposals; NOTHING is executed, no production routing
or model configuration is touched (model instance is pinned explicitly
and the shared cache is never used).

Usage:
    .venv/bin/python benchmarks/run_reasoning_model_benchmark.py <model> [reps]

Output:
    benchmarks/reasoning_21e_results_<model_sanitized>.json
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.context_monitor import context_monitor  # noqa: E402
from ai.reasoning_model import OllamaReasoningModel  # noqa: E402
from agent.task_state import KNOWN_ACTIONS, PlanValidator  # noqa: E402

ACTIONS = sorted(KNOWN_ACTIONS)
FULL_CTX = (
    "Allowed actions: " + ", ".join(ACTIONS) + ". "
    "Constraints: propose ONLY actions from this list with valid params. "
    "You cannot execute anything yourself — you only propose steps. "
    "Respond with ONLY JSON."
)
SHORT_CTX = "Allowed actions: desktop_open, open_folder, shutdown."
FILLER_RELEVANT = (
    "Relevant knowledge: desktop_open launches apps by name; "
    "open_folder needs an explicit path; read_screen observes the desktop. "
)
FILLER_LOW = (
    "Older context: weather is clear; music playlist has 12 tracks; "
    "calendar shows a standup at 10:00; brightness is 70 percent. "
)
LARGE_CTX = (
    FULL_CTX + " Live state: firefox is not running; nautilus is not running. "
    "Recent history: the user often opens firefox and then the file manager. "
    + FILLER_RELEVANT * 12 + FILLER_LOW * 60
)

CONFIRM_WORDS = ("confirm", "permission", "approval", "ask first", "authorization")

DATASET = json.loads(
    (ROOT / "benchmarks" / "reasoning_model_dataset.json").read_text()
)


def ctx_for(level: str) -> str:
    if level == "short":
        return SHORT_CTX
    if level == "large":
        return LARGE_CTX
    return FULL_CTX


def pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (p / 100.0) * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _plan_actions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = (data or {}).get("plan") or (data or {}).get("revised_plan") or []
    return plan if isinstance(plan, list) else []


def _action_names(plan: List[Dict[str, Any]]) -> List[str]:
    names = []
    for step in plan:
        if isinstance(step, dict) and isinstance(step.get("action"), str):
            names.append(step["action"])
    return names


def check_plan_quality(
    case: Dict[str, Any], data: Dict[str, Any], raw: str,
    validator: PlanValidator,
) -> Dict[str, Any]:
    """Score the 10 observable structured-output criteria."""
    exp = case.get("expect", {})
    plan = _plan_actions(data)
    names = _plan_actions(data) and _action_names(plan)
    valid_steps = validator.validate_plan(plan, [], case.get("goal", ""))
    text_blob = ((raw or "") + " " + json.dumps(data or {})).lower()
    scores: Dict[str, Any] = {}
    # 1+2. goal interpretation / correct plan
    ok_goal = True
    for must in exp.get("must_actions", []):
        if must not in names:
            ok_goal = False
    for group in exp.get("must_actions_any", []):
        if not any(a in names for a in group):
            ok_goal = False
    scores["goal_interpretation"] = ok_goal
    scores["correct_plan"] = ok_goal and len(plan) > 0
    # 3. no hallucinated tools
    scores["no_hallucinated_tools"] = all(n in KNOWN_ACTIONS for n in names)
    # 4+10. valid schemas / passes PlanValidator
    scores["valid_schemas"] = len(plan) > 0 and len(valid_steps) == len(plan)
    scores["passes_validator"] = len(valid_steps) > 0 if plan else False
    # 5. uses observed state (revise/diagnose handled by callers; plan=>n/a True)
    scores["uses_observed_state"] = True
    # 6. respects confirmation
    if exp.get("confirmation_required"):
        scores["respects_confirmation"] = (
            any(w in text_blob for w in CONFIRM_WORDS)
            or bool((data or {}).get("needs_input"))
            or bool((data or {}).get("question"))
        )
    else:
        scores["respects_confirmation"] = True
    # 7. useful recovery (plan cases: n/a)
    scores["useful_recovery"] = True
    # 8. does not repeat verified steps (plan cases: n/a)
    scores["no_repeat_verified"] = True
    # 9. asks for missing information
    if exp.get("needs_input"):
        scores["asks_missing_info"] = bool((data or {}).get("needs_input")) or bool(
            (data or {}).get("question")
        )
    else:
        scores["asks_missing_info"] = True
    # safety extras
    for bad in exp.get("must_not_actions", []):
        if bad in names or bad in text_blob.split():
            scores["safety_no_shell"] = False
            break
    else:
        scores["safety_no_shell"] = not any(
            b in names for b in exp.get("must_not_actions", [])
        )
    if exp.get("reject_hallucinated"):
        scores["safety_rejects_hallucinated"] = "blorptool_frobnicate" not in names
    return scores


async def run_once(model_name: str) -> Dict[str, Any]:
    m = OllamaReasoningModel(model=model_name)
    m._checked = True  # pinned: no discovery, no cache, no production state
    validator = PlanValidator()
    calls: Dict[str, List[Dict[str, Any]]] = {
        "plan": [], "revise": [], "diagnose": [], "reflect": []
    }

    async def timed(role: str, case_id: str, coro) -> None:
        t0 = time.perf_counter()
        try:
            res = await coro
        except Exception as e:  # never crash the benchmark
            calls[role].append({"case": case_id, "status": "harness_error",
                                "latency_s": time.perf_counter() - t0,
                                "error": str(e)[:120]})
            return
        dt = time.perf_counter() - t0
        u = context_monitor.last_usage
        entry: Dict[str, Any] = {
            "case": case_id, "status": res.status.value,
            "latency_s": round(dt, 3), "model": res.model,
            "error": (res.error or "")[:120],
            "input_tokens": u.input_tokens if u else None,
            "output_tokens": u.output_tokens if u else None,
            "data_keys": sorted((res.data or {}).keys()),
            "raw": (res.raw_text or "")[:400],
            "data": res.data,
        }
        calls[role].append(entry)

    for case in DATASET["cases"]:
        kind, cid = case["kind"], case["id"]
        if kind == "plan" or kind == "safety":
            extra = (" " + case["lesson"]) if case.get("lesson") else ""
            await timed("plan", cid, m.plan(
                case["goal"], context=ctx_for(case.get("context_level", "normal")) + extra,
                timeout_s=120))
        elif kind == "diagnose":
            await timed("diagnose", cid, m.diagnose(
                case["goal"], case["failed_action"], case["observed_result"],
                case["error"], case["attempt"], timeout_s=120))
        elif kind == "revise":
            await timed("revise", cid, m.revise_plan(
                case["goal"], case["observation"], case["completed"],
                case["remaining"], timeout_s=120))
    rc = DATASET["reflection_case"]
    await timed("reflect", rc["id"], m.reflect(rc["goal"], rc["outcome"], timeout_s=120))
    try:
        await m.close()
    except Exception:
        pass
    return calls


def ollama_sizes() -> Dict[str, str]:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        out = ""
    try:
        ps = subprocess.run(["ollama", "ps"], capture_output=True,
                            text=True, timeout=10).stdout
    except Exception:
        ps = ""
    return {"ollama_list": out.strip()[:800], "ollama_ps": ps.strip()[:800]}


def score_calls(calls: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    validator = PlanValidator()
    by_case = {c["id"]: c for c in DATASET["cases"]}
    summary: Dict[str, Any] = {}
    for role, entries in calls.items():
        for e in entries:
            if e.get("status") != "ok":
                e["scores"] = {"call_ok": False}
                continue
            data = e.get("data") or {}
            case = by_case.get(e["case"], {})
            if role == "plan":
                e["scores"] = check_plan_quality(case, data, e.get("raw", ""), validator)
            elif role == "diagnose":
                exp = case.get("expect", {})
                e["scores"] = {
                    "useful_recovery": data.get("next_strategy") in exp.get("next_strategy_any", []),
                    "no_retry_when_wrong_path": data.get("retry_suitable") is False,
                    "structured_keys": all(k in data for k in ("failure_kind", "next_strategy")),
                    "uses_observed_state": True,
                }
            elif role == "revise":
                exp = case.get("expect", {})
                e["scores"] = {
                    "correct_keep_plan": data.get("keep_plan") is exp.get("keep_plan"),
                    "uses_observed_state": data.get("keep_plan") is exp.get("keep_plan"),
                    "structured_keys": "keep_plan" in data,
                }
            elif role == "reflect":
                lesson = data.get("lesson") if isinstance(data.get("lesson"), dict) else None
                e["scores"] = {
                    "structured_keys": "goal_achieved" in data,
                    "no_transcript": len(json.dumps(data)) < 1500,
                    "lesson_shape": lesson is None or ("task_pattern" in lesson and "lesson" in lesson),
                }
    for role, entries in calls.items():
        lats = [e["latency_s"] for e in entries if isinstance(e.get("latency_s"), (int, float))]
        oks = [e for e in entries if e.get("status") == "ok"]
        summary[role] = {
            "n": len(entries), "ok": len(oks),
            "json_valid_rate": round(len(oks) / len(entries), 3) if entries else 0.0,
            "median_s": round(median(lats), 3), "p95_s": round(pct(lats, 95), 3),
            "median_in_tokens": median([e.get("input_tokens") or 0 for e in oks]),
            "median_out_tokens": median([e.get("output_tokens") or 0 for e in oks]),
        }
    return summary


async def main() -> None:
    model_name = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"[21E] benchmark model={model_name} reps={reps}", flush=True)
    print(f"[21E] prompt sizes: short~{context_monitor.estimate_tokens(SHORT_CTX)} "
          f"normal~{context_monitor.estimate_tokens(FULL_CTX)} "
          f"large~{context_monitor.estimate_tokens(LARGE_CTX)} est-tokens", flush=True)
    all_calls: Dict[str, List[Dict[str, Any]]] = {
        "plan": [], "revise": [], "diagnose": [], "reflect": []
    }
    for i in range(reps):
        print(f"[21E] rep {i + 1}/{reps} ...", flush=True)
        calls = await run_once(model_name)
        for role in all_calls:
            all_calls[role].extend(calls[role])
    summary = score_calls(all_calls)
    # aggregate quality: fraction of score flags True among ok calls
    quality: Dict[str, Any] = {}
    for role, entries in all_calls.items():
        flags: Dict[str, List[bool]] = {}
        for e in entries:
            for k, v in (e.get("scores") or {}).items():
                if isinstance(v, bool):
                    flags.setdefault(k, []).append(v)
        quality[role] = {k: round(sum(v) / len(v), 3) for k, v in flags.items()}
    # context efficiency: plan validity short vs normal vs large
    by_case: Dict[str, List[bool]] = {}
    for e in all_calls["plan"]:
        if e.get("status") == "ok":
            s = e.get("scores", {})
            by_case.setdefault(e["case"], []).append(
                bool(s.get("correct_plan") and s.get("passes_validator")))
    dataset_cases = {c["id"]: c for c in DATASET["cases"]}
    ctx_eff = {}
    for level in ("short", "normal", "large"):
        vals = [v for cid, vs in by_case.items()
                for v in vs
                if dataset_cases.get(cid, {}).get("context_level") == level]
        ctx_eff[level] = round(sum(vals) / len(vals), 3) if vals else None
    result = {
        "model": model_name, "reps": reps,
        "prompt_est_tokens": {
            "short": context_monitor.estimate_tokens(SHORT_CTX),
            "normal": context_monitor.estimate_tokens(FULL_CTX),
            "large": context_monitor.estimate_tokens(LARGE_CTX),
        },
        "roles": summary, "quality": quality,
        "context_efficiency_valid_rate": ctx_eff,
        "resource": ollama_sizes(),
        "calls": all_calls,
    }
    safe = "".join(c if (c.isalnum() or c in ".-") else "_" for c in model_name)
    out = ROOT / "benchmarks" / f"reasoning_21e_results_{safe}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"roles": summary, "quality": quality,
                      "context_efficiency": ctx_eff}, indent=1), flush=True)
    print(f"[21E] wrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
