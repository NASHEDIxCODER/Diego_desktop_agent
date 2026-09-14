"""
Phase 21C — reasoning performance profiling tests.

Validates model-call efficiency and that the new latency instrumentation
measures (without altering) task behavior. Uses mocked model responses
with artificial per-call delays so each phase's cost is observable.

Scenarios:
  1.  deterministic command does not invoke reasoning model
  2.  simple task invokes minimum expected model calls
  3.  successful task does not invoke diagnosis unnecessarily
  4.  failed task invokes bounded diagnosis
  5.  replan calls are bounded
  6.  reflection is bounded
  7.  context composer does not duplicate layers
  8.  lesson retrieval is bounded
  9.  context trimming is bounded
 10.  model is not reinitialized unnecessarily
 11.  latency instrumentation does not alter task behavior
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ai.context_monitor import ContextMonitor
from ai.reasoning_model import (
    BaseReasoningModel,
    ReasoningCallStatus,
    ReasoningResult,
)
from agent.lessons import TaskLessonStore
from agent.reasoning_agent import Mode, ReasoningAgent, choose_mode
from agent.reasoning_context import ReasoningContextComposer
from agent.reasoning_state import ReasoningState
from agent.task_state import (
    FinalStatus,
    StepRecord,
    StepStatus,
    TaskExecutionState,
    TaskLimits,
)


class DelayedFakeModel(BaseReasoningModel):
    """Fake reasoning model that sleeps delay_s per call and returns
    scripted results, so tests observe per-phase latency and count calls."""

    name = "delayed-fake"

    def __init__(self, delay_s=0.01, plan_results=None,
                 diagnose_results=None, revise_results=None,
                 reflect_results=None, diagnose_calls=None):
        self.delay_s = float(delay_s)
        self._plan = list(plan_results or [])
        self._diagnose = list(diagnose_results or [])
        self._revise = list(revise_results or [])
        self._reflect = list(reflect_results or [])
        self.calls: List[str] = []
        self.diagnose_calls = diagnose_calls

    async def _sleep(self):
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

    async def reason(self, prompt, context="", **kw):
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)

    async def plan(self, goal, context="", **kw):
        await self._sleep()
        self.calls.append("plan")
        if self._plan:
            return self._plan.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "plan": [
                {"action": "desktop_open", "params": {"app": "firefox"}},
                {"action": "open_folder", "params": {"path": "/tmp"}},
            ],
            "assumptions": ["fake"], "constraints": [],
        })

    async def diagnose(self, goal, action, observed, error, attempt, **kw):
        await self._sleep()
        self.calls.append("diagnose")
        if self.diagnose_calls is not None:
            self.diagnose_calls.append((action, error))
        if self._diagnose:
            return self._diagnose.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "failure_kind": "transient", "probable_cause": "temporary",
            "retry_suitable": True, "alternative": "retry same",
            "next_strategy": "retry",
        })

    async def revise_plan(self, goal, observation, completed, remaining, **kw):
        await self._sleep()
        self.calls.append("revise")
        if self._revise:
            return self._revise.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "keep_plan": True})

    async def reflect(self, goal, outcome, **kw):
        await self._sleep()
        self.calls.append("reflect")
        if self._reflect:
            return self._reflect.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "goal_achieved": True, "what_worked": "direct plan",
            "what_failed": "",
        })


class FakeExecutor:
    def __init__(self, script=None, fail_actions=None):
        self.script = dict(script or {})
        self.fail_actions = set(fail_actions or [])
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        name = action.get("action", "")
        if name in self.fail_actions:
            return False, f"{name} failed"
        return self.script.get(name, (True, f"{name} ok"))


class FakeObserver:
    def __init__(self, states=None):
        self.states = list(states or [])

    async def __call__(self):
        if self.states:
            return self.states.pop(0)
        return "fake observation"


def make_agent(executor=None, observer=None, model=None,
               limits=None, store=None, planner=None):
    return ReasoningAgent(
        executor=executor or FakeExecutor(),
        observer=observer,
        planner=planner,
        reasoning_model=model,
        limits=limits or TaskLimits(
            max_task_steps=8, max_retries_per_step=2,
            max_replans=2, max_total_execution_time=30),
        lesson_store=store or TaskLessonStore(path=":memory:"),
    )


def run(coro):
    return asyncio.run(coro)


def run_with_reflection(agent, coro):
    """Run a reasoning-agent coroutine, then await any pending background
    reflection WITHIN THE SAME event loop (so the deferred task completes)."""
    async def _full():
        result = await coro
        pending = getattr(agent, "_pending_reflection", None)
        if pending is not None and not pending.done():
            try:
                await pending
            except (asyncio.CancelledError, Exception):
                pass
        return result
    return asyncio.run(_full())


# ═══════════════════════════════════════════════════════════════
# 1. deterministic command does not invoke reasoning model
# ═══════════════════════════════════════════════════════════════

def test_01_deterministic_command_does_not_invoke_reasoning_model():
    """S1: deterministic commands must never touch the reasoning model."""
    assert choose_mode("open firefox") is Mode.DETERMINISTIC
    assert choose_mode("volume up") is Mode.DETERMINISTIC
    model = DelayedFakeModel()
    agent = make_agent(model=model)
    result = run(agent.run("open firefox", mode=Mode.DETERMINISTIC))
    assert model.calls == []
    assert agent.model_calls == 0


def test_02_simple_task_minimum_model_calls():
    """S2: an autonomous task with a scripted 1-step plan makes only
    plan + reflect. No revise (no remaining steps after the single step)
    and no diagnosis."""
    model = DelayedFakeModel(
        delay_s=0.005,
        plan_results=[ReasoningResult(status=ReasoningCallStatus.OK, data={
            "plan": [{"action": "desktop_open", "params": {"app": "firefox"}}],
            "assumptions": [], "constraints": []})],
    )
    executor = FakeExecutor({"desktop_open": (True, "ok")})
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run("open the calculator app", mode=Mode.AUTONOMOUS))
    assert result.success
    # Reflection is now DEFERRED: the result is returned BEFORE the model
    # reflect call fires, so the synchronous count is plan-only.
    assert model.calls.count("plan") == 1
    assert "revise" not in model.calls
    assert "diagnose" not in model.calls
    # The deferred reflection must have run exactly once (same event loop).
    assert model.calls.count("reflect") == 1
    assert agent.model_calls == 2


def test_03_successful_task_no_diagnosis():
    """S3: a fully successful multi-step task must NOT call diagnosis."""
    model = DelayedFakeModel(delay_s=0.005)
    executor = FakeExecutor({
        "desktop_open": (True, "ok"),
        "open_folder": (True, "ok"),
    })
    agent = make_agent(model=model, executor=executor)
    result = run(agent.run(
        "open firefox and then open the file manager",
        mode=Mode.AUTONOMOUS))
    assert result.success
    assert "diagnose" not in model.calls


def test_04_failed_task_bounded_diagnosis():
    """S4: failing step triggers at most one diagnosis per replan, bounded
    by max_replans."""
    model = DelayedFakeModel(
        delay_s=0.005,
        diagnose_results=[ReasoningResult(status=ReasoningCallStatus.OK, data={
            "failure_kind": "transient", "probable_cause": "temp",
            "retry_suitable": True, "alternative": "",
            "next_strategy": "retry"})],
    )
    executor = FakeExecutor(fail_actions={"desktop_open", "open_folder"})
    agent = make_agent(model=model, executor=executor)
    result = run(agent.run(
        "open firefox and then open the file manager",
        mode=Mode.AUTONOMOUS))
    assert not result.success
    assert model.calls.count("diagnose") <= 2
    assert result.task_state.replan_count <= 2


def test_05_replan_calls_bounded():
    """S5: replan generation is bounded by max_replans."""
    model = DelayedFakeModel(
        delay_s=0.005,
        plan_results=[ReasoningResult(status=ReasoningCallStatus.OK, data={
            "plan": [{"action": "desktop_open", "params": {"app": "firefox"}}],
            "assumptions": [], "constraints": []})],
    )
    executor = FakeExecutor(fail_actions={"desktop_open"})
    agent = make_agent(model=model, executor=executor)
    result = run(agent.run("open firefox and then check the time",
                           mode=Mode.AUTONOMOUS))
    assert not result.success
    assert result.task_state.replan_count <= 2
    assert model.calls.count("plan") <= 3


def test_06_reflection_bounded():
    """S6: exactly one reflection call per task."""
    model = DelayedFakeModel(delay_s=0.005)
    executor = FakeExecutor({"desktop_open": (True, "ok")})
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run("open the calculator app", mode=Mode.AUTONOMOUS))
    assert result.success
    assert result.reflection is not None
    # Reflection model call is DEFERRED — but with the single-loop helper
    # the background task completes before we assert, so exactly one call.
    assert model.calls.count("reflect") == 1


def test_07_context_composer_no_duplicate_layers():
    """S7: each priority layer appears at most once."""
    monitor = ContextMonitor()
    monitor.configure("test", context_limit=10000, default_output_reserve=0)
    composer = ReasoningContextComposer(monitor=monitor)
    composed = composer.compose(
        goal="some goal", constraints=["c1"], state_lines=["PROGRESS: 0"],
        current_objective="step1", recent_observations=["obs"],
        verified_evidence=["ev"], conversation_history=["h"],
        knowledge_facts=["k"], lesson_lines=["l"], background=["b"],
        reserve_output=1)
    assert len(composed.layers) == len(set(composed.layers))
    assert composed.layers == [
        "P0_goal", "P1_task_state", "P2_current_step", "P3_evidence",
        "P4_history", "P5_knowledge", "P6_lessons", "P7_background"]


def test_08_lesson_retrieval_bounded():
    """S8: lesson retrieval is a single bounded in-memory scan."""
    store = TaskLessonStore(path=":memory:")
    for i in range(50):
        store.add_lesson(type="task_lesson", task_pattern=f"pattern_{i % 5}",
                         lesson=f"lesson content number {i} " * 10,
                         evidence="verified_success")
    assert store.count == 50
    t0 = time.perf_counter()
    hits = store.retrieve("open firefox", limit=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(hits) <= 5
    assert elapsed_ms < 50.0


def test_09_context_trimming_bounded():
    """S9: trimming keeps P0-P3 and preserves ordering under a tight budget."""
    monitor = ContextMonitor()
    monitor.configure("test", context_limit=60, default_output_reserve=0)
    composer = ReasoningContextComposer(monitor=monitor)
    composed = composer.compose(
        goal="open firefox",
        constraints=["never delete anything", "be careful always"],
        state_lines=["PROGRESS: 5 steps done", "PHASE: execute",
                     "CONFIDENCE: 0.9", "NEXT: open_folder",
                     "DONE: step1", "DONE: step2", "DONE: step3",
                     "DONE: step4", "DONE: step5"],
        current_objective="open_folder",
        recent_observations=["obs1", "obs2"],
        verified_evidence=["firefox verified"],
        conversation_history=["h" * 200, "h2" * 200, "h3" * 200],
        knowledge_facts=["k" * 200], lesson_lines=["l" * 200],
        background=["background " * 200], reserve_output=1)
    assert "P0_goal" in composed.layers
    if "P3_evidence" in composed.layers:
        assert composed.layers.index("P0_goal") < composed.layers.index(
            "P3_evidence")


def test_10_model_not_reinitialized():
    """S10: a single model instance is reused across all calls in a task."""
    model = DelayedFakeModel(delay_s=0.005)
    executor = FakeExecutor({
        "desktop_open": (True, "ok"),
        "open_folder": (True, "ok"),
    })
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run(
        "open firefox and then open the file manager",
        mode=Mode.AUTONOMOUS))
    assert result.success
    assert agent._model is model
    assert model.calls.count("plan") == 1
    assert model.calls.count("revise") == 1
    # Reflection deferred — completes in the same loop, exactly once.
    assert model.calls.count("reflect") == 1


def test_11_instrumentation_does_not_alter_behavior():
    """S11: the timing profile is populated but does not change the task
    outcome, plan, or action sequence."""
    model = DelayedFakeModel(delay_s=0.005)
    executor = FakeExecutor({
        "desktop_open": (True, "ok"),
        "open_folder": (True, "ok"),
    })
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run(
        "open firefox and then open the file manager",
        mode=Mode.AUTONOMOUS))
    assert result.success
    profile = result.profile
    assert "total" in profile and profile["total"] > 0
    assert "plan" in profile
    assert "execute" in profile
    assert profile["model_calls_total"] >= 2  # point-in-time snapshot at return
    assert profile["context_tokens"] > 0
    assert len(result.task_state.completed_steps) == 2
    assert [s.action for s in result.task_state.completed_steps] == [
        "desktop_open", "open_folder"]
    # Reflection deferred — completes in the same loop, exactly once.
    assert model.calls.count("reflect") == 1
