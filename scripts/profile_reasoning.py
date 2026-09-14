#!/usr/bin/env python3
"""
Phase 21C — reasoning performance profiler.

Runs the four canonical scenarios with a DELAYED mock model so per-phase
latency (model vs non-model) can be separated, and estimates median/p95.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.reasoning_model import ReasoningCallStatus, ReasoningResult
from agent.reasoning_agent import ReasoningAgent, Mode
from agent.task_state import TaskLimits
from agent.lessons import TaskLessonStore


class InstrumentedModel:
    """Counts + times every model call with an artificial delay so the
    model contribution to total latency is measurable and separable."""

    name = "instrumented"
    delay_s = 0.05  # per model call

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self._plan = []
        self._diagnose = []
        self._revise = []
        self._reflect = []
        self._seed()

    def _seed(self):
        self._plan = [
            ReasoningResult(status=ReasoningCallStatus.OK, data={
                "plan": [
                    {"action": "desktop_open", "params": {"app": "firefox"}},
                    {"action": "open_folder", "params": {"path": "/tmp"}},
                ], "assumptions": [], "constraints": []}),
            ReasoningResult(status=ReasoningCallStatus.OK, data={
                "plan": [
                    {"action": "find_file", "params": {"pattern": "context_monitor.py"}},
                    {"action": "file_info", "params": {"path": "ai/context_monitor.py"}},
                ], "assumptions": [], "constraints": []}),
            ReasoningResult(status=ReasoningCallStatus.OK, data={
                "plan": [
                    {"action": "find_file", "params": {"pattern": "event_bus.py"}},
                    {"action": "read_file_head", "params": {"path": "core/event_bus.py"}},
                ], "assumptions": [], "constraints": []}),
            ReasoningResult(status=ReasoningCallStatus.OK, data={
                "plan": [
                    {"action": "file_info", "params": {"path": "core/does_not_exist.py"}},
                ], "assumptions": [], "constraints": []}),
            ReasoningResult(status=ReasoningCallStatus.OK, data={
                "plan": [
                    {"action": "find_file", "params": {"pattern": "event_bus.py"}},
                    {"action": "file_info", "params": {"path": "core/event_bus.py"}},
                ], "assumptions": [], "constraints": []}),
        ]
        self._diagnose = [
            ReasoningResult(status=ReasoningCallStatus.OK, data={
                "failure_kind": "unavailable_capability",
                "probable_cause": "wrong path", "retry_suitable": False,
                "alternative": "find the real file first",
                "next_strategy": "replan"})
        ]
        self._revise = [
            ReasoningResult(status=ReasoningCallStatus.OK, data={"keep_plan": True})
        ]
        self._reflect = [
            ReasoningResult(status=ReasoningCallStatus.OK, data={
                "goal_achieved": True, "what_worked": "direct plan",
                "what_failed": ""}
        )]

    async def _call(self, kind):
        t0 = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.calls.append({"kind": kind, "ms": elapsed})
        bank = {"plan": self._plan, "diagnose": self._diagnose,
                "revise": self._revise, "reflect": self._reflect}[kind]
        if bank:
            return bank.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.OK, data={})

    async def plan(self, goal, context="", **kw):
        return await self._call("plan")
    async def diagnose(self, goal, action, observed, error, attempt, **kw):
        return await self._call("diagnose")
    async def revise_plan(self, goal, observation, completed, remaining, **kw):
        return await self._call("revise")
    async def reflect(self, goal, outcome, **kw):
        return await self._call("reflect")
    async def reason(self, prompt, context="", **kw):
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)


class FakeExec:
    def __init__(self, fail=None):
        self.fail = set(fail or [])
        self.calls = []
    async def __call__(self, action):
        self.calls.append(dict(action))
        n = action.get("action", "")
        if n in self.fail:
            return False, f"{n} failed"
        return True, f"{n} ok"


class FakeObs:
    async def __call__(self):
        return "fake observation"


def make_agent(model, fail=None):
    return ReasoningAgent(
        executor=FakeExec(fail=fail), observer=FakeObs(),
        planner=None, reasoning_model=model,
        limits=TaskLimits(max_task_steps=8, max_retries_per_step=2,
                          max_replans=2, max_total_execution_time=30),
        lesson_store=TaskLessonStore(path=":memory:"))


SCENARIOS = [
    ("A", "Open Firefox and then open the file manager", None),
    ("B", "Find a project file in my workspace and report its size", None),
    ("C", "Find a project file, inspect it, and explain what it does", None),
    ("D", "Report the size of the event bus file starting from a wrong path",
     ["file_info"]),
]


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (p / 100.0) * (len(s) - 1)
    f_, c_ = int(k), min(int(k) + 1, len(s) - 1)
    return s[f_] + (k - f_) * (s[c_] - s[f_])


async def run_once(name, goal, fail, delay_s):
    model = InstrumentedModel()
    model.delay_s = delay_s
    agent = make_agent(model, fail=fail)
    t0 = time.perf_counter()
    result = await agent.run(goal, mode=None)
    total_ms = (time.perf_counter() - t0) * 1000.0
    model_ms = sum(c["ms"] for c in model.calls)
    return {
        "name": name, "success": result.success,
        "total_ms": total_ms, "model_ms": model_ms,
        "non_model_ms": total_ms - model_ms,
        "model_calls": len(model.calls),
        "call_kinds": {k: sum(1 for c in model.calls if c["kind"]==k)
                       for k in ["plan", "diagnose", "revise", "reflect"]},
        "profile": result.profile,
        "replans": result.task_state.replan_count,
    }


async def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    delay_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    print(f"Profiling {iterations} iterations/scenario, model delay={delay_s}s\n")
    for name, goal, fail in SCENARIOS:
        rows = []
        for _ in range(iterations):
            rows.append(await run_once(name, goal, fail, delay_s))
        totals = [r["total_ms"] for r in rows]
        models = [r["model_ms"] for r in rows]
        nonmodels = [r["non_model_ms"] for r in rows]
        calls = [r["model_calls"] for r in rows]
        print(f"=== Scenario {name}: {goal[:50]} ===")
        print(f"  success rate: {sum(r['success'] for r in rows)}/{iterations}")
        print(f"  total   median={statistics.median(totals):.0f}ms  "
              f"p95={pct(totals,95):.0f}ms")
        print(f"  model   median={statistics.median(models):.0f}ms  "
              f"p95={pct(models,95):.0f}ms")
        print(f"  non-model median={statistics.median(nonmodels):.0f}ms  "
              f"p95={pct(nonmodels,95):.0f}ms")
        print(f"  model calls median={statistics.median(calls):.0f}")
        # Aggregate phase profile from last successful row
        for r in rows:
            if r["profile"]:
                p = r["profile"]
                print(f"  phases(ms): plan={p.get('plan',0):.1f} "
                      f"execute={p.get('execute',0):.1f} "
                      f"reflect={p.get('reflect',0):.1f} "
                      f"context={p.get('context',0):.1f} "
                      f"validate={p.get('validate',0):.1f} "
                      f"persist={p.get('persist',0):.1f} "
                      f"total(p)={p.get('total',0):.1f}")
                print(f"  model call kinds: {r['call_kinds']}")
                break
        print()


if __name__ == "__main__":
    asyncio.run(main())
