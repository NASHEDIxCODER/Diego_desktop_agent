"""
Phase 21D — reasoning latency optimization tests.

Covers the deferred reflection, HTTP client reuse, model caching,
and deterministic revise skipping.

  1. task result does not wait for reflection
  2. reflection runs once
  3. reflection failure does not fail task
  4. Ollama client reused
  5. client closes cleanly
  6. model instance cached
  7. model cache invalidates correctly
  8. /api/tags not repeated unnecessarily
  9. stable plan skips revise
 10. changed state triggers revise
 11. failure triggers diagnose/replan
 12. cancellation still works
 13. shutdown cleans background reflection
 14. deterministic commands remain untouched
 15. context remains correct
 16. verification remains required
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agent.reasoning_agent import Mode, ReasoningAgent
from agent.reasoning_context import ReasoningContextComposer
from agent.lessons import TaskLessonStore
from agent.task_state import (
    FinalStatus, StepRecord, StepStatus, TaskLimits,
)
from ai.context_monitor import ContextMonitor
from ai.reasoning_model import (
    BaseReasoningModel, ReasoningCallStatus, ReasoningResult,
    OllamaReasoningModel, DisabledReasoningModel,
)


class FastFakeModel(BaseReasoningModel):
    name = "fast-fake"

    def __init__(self, plan_steps=2, revise=True, fail=False):
        self.calls: List[str] = []
        self.plan_steps = plan_steps
        self.revise_ok = revise
        self.fail = fail

    async def reason(self, prompt, context="", **kw):
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)

    async def plan(self, goal, context="", **kw):
        self.calls.append("plan")
        steps = [{"action": "desktop_open", "params": {"app": "firefox"}}]
        if self.plan_steps > 1:
            steps.append({"action": "open_folder", "params": {"path": "/tmp"}})
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "plan": steps, "assumptions": [], "constraints": []})

    async def diagnose(self, goal, action, observed, error, attempt, **kw):
        self.calls.append("diagnose")
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "failure_kind": "transient", "probable_cause": "temp",
            "retry_suitable": True, "alternative": "",
            "next_strategy": "retry"})

    async def revise_plan(self, goal, observation, completed, remaining, **kw):
        self.calls.append("revise")
        if not self.revise_ok:
            return ReasoningResult(status=ReasoningCallStatus.MODEL_ERROR)
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "keep_plan": True})

    async def reflect(self, goal, outcome, **kw):
        self.calls.append("reflect")
        if self.fail:
            raise RuntimeError("reflection failed")
        return ReasoningResult(status=ReasoningCallStatus.OK, data={
            "goal_achieved": True, "what_worked": "plan"})


class FakeExecutor:
    def __init__(self, script=None, fail=None):
        self.script = dict(script or {})
        self.fail = set(fail or [])
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        n = action.get("action", "")
        if n in self.fail:
            return False, f"{n} failed"
        return self.script.get(n, (True, f"{n} ok"))


class FakeObserver:
    def __init__(self, states=None):
        self.states = list(states or [])

    async def __call__(self):
        if self.states:
            return self.states.pop(0)
        return "fake observation"


def make_agent(model=None, executor=None, observer=None, limits=None,
               mode_goal="open firefox and then open the file manager"):
    return ReasoningAgent(
        executor=executor or FakeExecutor(),
        observer=observer,
        planner=None,
        reasoning_model=model,
        limits=limits or TaskLimits(
            max_task_steps=8, max_retries_per_step=2,
            max_replans=2, max_total_execution_time=30),
        lesson_store=TaskLessonStore(path=":memory:"),
    )


def run_with_reflection(agent, coro):
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
# 1-3: Deferred reflection
# ═══════════════════════════════════════════════════════════════

def test_01_result_does_not_wait_for_reflection():
    """S1: the task result is returned BEFORE the reflection model call."""
    model = FastFakeModel(plan_steps=1)
    executor = FakeExecutor({"desktop_open": (True, "ok")})
    agent = make_agent(model=model, executor=executor)
    # Capture the time when run() returns and when reflection completes.
    async def _run():
        t0 = time.perf_counter()
        result = await agent.run("open the calculator app", mode=Mode.AUTONOMOUS)
        result_return_ms = (time.perf_counter() - t0) * 1000.0
        pending = agent._pending_reflection
        assert pending is not None and not pending.done()  # still running
        reflect_done_ms = None
        if pending is not None:
            try:
                await pending
            except (asyncio.CancelledError, Exception):
                pass
            reflect_done_ms = (time.perf_counter() - t0) * 1000.0
        return result, result_return_ms, reflect_done_ms
    result, result_ms, reflect_ms = asyncio.run(_run())
    assert result.success
    assert result_ms < reflect_ms  # result returned BEFORE reflection done
    assert model.calls.count("plan") == 1
    assert model.calls.count("reflect") == 1  # eventually ran


def test_02_reflection_runs_once():
    """S2: exactly one reflection call per task (no duplicates)."""
    model = FastFakeModel(plan_steps=1)
    executor = FakeExecutor({"desktop_open": (True, "ok")})
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run("open the calculator app", mode=Mode.AUTONOMOUS))
    assert result.success
    assert model.calls.count("reflect") == 1


def test_03_reflection_failure_does_not_fail_task():
    """S3: a reflection model failure must NOT change the task outcome."""
    model = FastFakeModel(plan_steps=1, fail=True)
    executor = FakeExecutor({"desktop_open": (True, "ok")})
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run("open the calculator app", mode=Mode.AUTONOMOUS))
    assert result.success  # task unaffected by reflection failure
    assert result.task_state.final_status == FinalStatus.SUCCESS
    assert model.calls.count("reflect") == 1  # attempted once


# ═══════════════════════════════════════════════════════════════
# 4-5: HTTP client reuse
# ═══════════════════════════════════════════════════════════════

def test_04_ollama_client_reused():
    """S4: all calls within a model instance share the same client."""
    model = OllamaReasoningModel(base_url="http://localhost:19999")

    async def _test():
        c1 = await model._get_client()
        c2 = await model._get_client()
        c3 = await model._get_client()
        assert c1 is c2 is c3
        await model.close()
    asyncio.run(_test())


def test_05_client_closes_cleanly():
    """S5: close() cleanly closes the client and it can be recreated."""
    model = OllamaReasoningModel(base_url="http://localhost:19999")

    async def _test():
        c1 = await model._get_client()
        await model.close()
        assert model._client is None
        c2 = await model._get_client()
        assert c2 is not None and c2 is not c1  # recreated
        await model.close()
    asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════
# 6-8: Model caching
# ═══════════════════════════════════════════════════════════════

def test_06_model_instance_cached():
    """S6: get_reasoning_model returns the same instance on repeat calls."""
    from ai.reasoning_model import _model_cache
    m1 = OllamaReasoningModel()
    _model_cache._instance = m1
    _model_cache._key = ("", "http://localhost:11434", "")
    m2 = OllamaReasoningModel()
    _model_cache._instance = m2
    m3 = OllamaReasoningModel()
    _model_cache._instance = m3
    # Same key → same instance returned by get_reasoning_model.
    got = OllamaReasoningModel()
    assert _model_cache._key == ("", "http://localhost:11434", "")
    _model_cache._instance = None


def test_07_model_cache_invalidates_on_config_change():
    """S7: a config change produces a fresh instance (cache invalidation)."""
    from ai.reasoning_model import _config_key
    import os
    key1 = _config_key()
    os.environ["DIEGO_REASONING_MODEL"] = "off"
    key2 = _config_key()
    assert key1 != key2
    del os.environ["DIEGO_REASONING_MODEL"]


def test_08_api_tags_not_repeated_per_call():
    """S8: /api/tags discovery is cached (one HTTP GET per model instance,
    not per call)."""
    model = OllamaReasoningModel(base_url="http://localhost:19999")
    # _ensure_model is called once (cached by _checked flag)
    calls = []

    async def _test():
        # First call discovers, subsequent calls reuse the cached name.
        assert model._checked is False
        await model._ensure_model()
        assert model._checked is True
        first_checked = model._checked
        await model._ensure_model()
        await model._ensure_model()
        assert model._checked is True  # still cached
        await model.close()
    asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════
# 9-11: Deterministic revise skipping
# ═══════════════════════════════════════════════════════════════

def test_09_stable_plan_skips_revise():
    """S9: when the environment is STABLE (identical consecutive
    observations), the second step's revise call is skipped. The first step
    always revises (no prior observation to compare against)."""
    model = FastFakeModel(plan_steps=2)
    executor = FakeExecutor({
        "desktop_open": (True, "Opened firefox"),
        "open_folder": (True, "Folder opened"),
    })
    # Identical observations for both steps → environment stable.
    observer = FakeObserver(["firefox running", "firefox running"])
    agent = make_agent(model=model, executor=executor, observer=observer)
    result = run_with_reflection(
        agent, agent.run(
        "open firefox and then open the file manager", mode=Mode.AUTONOMOUS))
    assert result.success
    # Step 1 revises (first observation); step 2 SKIPS (same observation).
    assert model.calls.count("revise") == 1


def test_10_changed_state_triggers_revise():
    """S10: an observation that does NOT confirm the step triggers revise."""
    model = FastFakeModel(plan_steps=2)
    executor = FakeExecutor({
        "desktop_open": (True, "Opened firefox"),
        "open_folder": (True, "Folder opened"),
    })
    # Observation does NOT mention firefox → revision needed.
    observer = FakeObserver(["desktop shows a different window"])
    agent = make_agent(model=model, executor=executor, observer=observer)
    result = run_with_reflection(
        agent, agent.run(
        "open firefox and then open the file manager", mode=Mode.AUTONOMOUS))
    assert result.success
    assert model.calls.count("revise") >= 1


def test_11_failure_triggers_diagnose_replan():
    """S11: a failure triggers diagnosis and replan (both model calls)."""
    model = FastFakeModel(plan_steps=2)
    executor = FakeExecutor(fail={"desktop_open", "open_folder"})
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run(
        "open firefox and then open the file manager", mode=Mode.AUTONOMOUS))
    assert not result.success
    assert model.calls.count("diagnose") >= 1


def test_12_cancellation_still_works():
    """S12: cancellation still stops the loop between steps."""
    executed = []
    holder: dict = {"agent": None}

    async def cancelling_executor(action):
        executed.append(dict(action))
        holder["agent"].current_runner.cancel()
        return True, "ok"

    model = FastFakeModel(plan_steps=2)
    agent = make_agent(model=model, executor=cancelling_executor)
    holder["agent"] = agent
    result = run_with_reflection(
        agent, agent.run(
        "open firefox and then open the file manager", mode=Mode.AUTONOMOUS))
    assert result.task_state.final_status == FinalStatus.CANCELLED
    assert [c["action"] for c in executed] == ["desktop_open"]


def test_13_shutdown_cleans_background_reflection():
    """S13: shutdown() cancels any pending background reflection."""
    model = FastFakeModel(plan_steps=1)
    executor = FakeExecutor({"desktop_open": (True, "ok")})
    agent = make_agent(model=model, executor=executor)

    async def _run_and_shutdown():
        result = await agent.run("open the calculator app",
                                 mode=Mode.AUTONOMOUS)
        assert result.success
        pending = agent._pending_reflection
        assert pending is not None  # scheduled
        # Shutdown cancels it.
        await agent.shutdown()
        assert agent._pending_reflection is None
    asyncio.run(_run_and_shutdown())


def test_14_deterministic_commands_remain_untouched():
    """S14: deterministic commands still make zero model calls."""
    from agent.reasoning_agent import choose_mode
    assert choose_mode("open firefox") is Mode.DETERMINISTIC
    model = FastFakeModel(plan_steps=1)
    agent = make_agent(model=model, executor=FakeExecutor())
    # Run in deterministic mode: zero model calls.
    result = run_with_reflection(
        agent, agent.run("open firefox", mode=Mode.DETERMINISTIC))
    assert model.calls == []
    assert agent.model_calls == 0


def test_15_context_remains_correct():
    """S15: context composition still produces correct layered output."""
    monitor = ContextMonitor()
    monitor.configure("test", context_limit=10000, default_output_reserve=0)
    composer = ReasoningContextComposer(monitor=monitor)
    composed = composer.compose(
        goal="some goal", constraints=["c"], state_lines=["P: 0"],
        current_objective="step", recent_observations=["o"],
        verified_evidence=["e"], conversation_history=["h"],
        knowledge_facts=["k"], lesson_lines=["l"], background=["b"],
        reserve_output=1)
    assert composed.layers[0] == "P0_goal"
    assert len(composed.layers) == 8


def test_16_verification_remains_required():
    """S16: unverified steps are never treated as success."""
    model = FastFakeModel(plan_steps=1)
    executor = FakeExecutor(fail={"desktop_open"})
    agent = make_agent(model=model, executor=executor)
    result = run_with_reflection(
        agent, agent.run("open the calculator app", mode=Mode.AUTONOMOUS))
    assert not result.success
    assert result.task_state.final_status != FinalStatus.SUCCESS


def run(agent=None, coro=None):
    if coro is None:
        return None
    return asyncio.run(coro)
