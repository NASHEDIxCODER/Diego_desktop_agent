"""Deterministic tests for Diego's CLOSED-LOOP TASK AGENT (agent/task_state.py)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agent.task_state import (
    FinalStatus,
    FollowUpResolver,
    LoopDetector,
    PlanValidator,
    StepRecord,
    StepStatus,
    TaskExecutionState,
    TaskLimits,
    TaskRunner,
    TaskStateStore,
    classify_failure,
    extract_artifacts,
    task_state_store,
)
import agent.task_state as task_state_module


class FakeExecutor:
    """Scripted executor: pops (ok, result) per call; records all calls."""

    def __init__(self, script: List[Tuple[bool, str]]):
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        self.calls.append(dict(action))
        if self.script:
            return self.script.pop(0)
        return False, "no scripted result"


class FakeObserver:
    """Scripted observer: pops observed-state strings per call."""

    def __init__(self, states: List[str]):
        self.states = list(states)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.states:
            return self.states.pop(0)
        return ""


class FakePlanner:
    """Scripted planner: pops plans per call; records (request, context)."""

    def __init__(self, plans: List[Optional[List[Dict[str, Any]]]]):
        self.plans = list(plans)
        self.contexts: List[Dict[str, Any]] = []

    async def __call__(self, request: str,
                       context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        self.contexts.append(dict(context))
        if self.plans:
            return self.plans.pop(0)
        return None


def make_runner(executor, observer=None, planner=None,
                limits: Optional[TaskLimits] = None,
                transcript: str = "open firefox and search github",
                validator: Optional[PlanValidator] = None) -> TaskRunner:
    return TaskRunner(
        executor=executor,
        observer=observer,
        planner=planner,
        validator=validator or PlanValidator(),
        limits=limits or TaskLimits(max_task_steps=12, max_retries_per_step=2,
                                    max_replans=3, max_total_execution_time=60),
        transcript=transcript,
    )


# ═══════════════════════════════════════════════════════════════
# Core loop tests
# ═══════════════════════════════════════════════════════════════

def test_single_step_task_success():
    """A one-step task completes with SUCCESS only after verification."""
    ex = FakeExecutor([(True, "Opened firefox")])
    obs = FakeObserver(["desktop: firefox window focused"])
    runner = make_runner(ex, obs)
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))
    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].verified is True
    assert state.completed_steps[0].status == StepStatus.COMPLETED
    assert ex.calls[0]["action"] == "desktop_open"
    # Logging contract
    assert any(l.startswith("[TASK] id=") for l in state.log)
    assert "[PLAN] v1 steps=1" in state.log
    assert "[STEP 1/1] desktop_open" in state.log
    assert "[VERIFY] success" in state.log
    assert "[TASK] COMPLETE" in state.log


def test_multi_step_task_success():
    """Multi-step tasks execute step-by-step, each verified before the next."""
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "Here's what I found for websocket: https://github.com/a/b"),
        (True, "Opened https://github.com/a/b"),
        (True, "On screen: WebSocket examples in Python"),
    ])
    obs = FakeObserver([
        "",                                   # baseline
        "firefox running",                    # after open
        "github results page",                # after search
        "repo page open",                     # after navigate
        "page content visible",               # after read
    ])
    runner = make_runner(ex, obs)
    plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "web_search",
         "params": {"query": "Python WebSocket examples site:github.com"}},
        {"action": "browser_navigate", "params": {"url": "https://github.com/a/b"}},
        {"action": "read_screen", "params": {}},
    ]
    state = asyncio.run(runner.run(
        "Open Firefox, search GitHub for Python WebSocket examples, "
        "open the best result, read it, and summarize it.", plan))
    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 4
    assert all(s.verified for s in state.completed_steps)
    assert [c["action"] for c in ex.calls] == [
        "desktop_open", "web_search", "browser_navigate", "read_screen"]
    # Real artifacts captured from dispatch results (never invented)
    assert state.artifacts.get("last_url") == "https://github.com/a/b"
    assert "WebSocket examples" in state.artifacts.get("answer", "")
    # The response must come from ACTUAL content, not "Done" after opening
    summary = state.summary()
    assert "WebSocket examples" in summary
    assert summary != "Done."


def test_failed_step_retry_succeeds():
    """A transient failure is retried (with adjusted params) and succeeds."""
    ex = FakeExecutor([
        (False, "Launch timed out"),          # first attempt fails
        (True, "Opened firefox"),             # retry succeeds
    ])
    obs = FakeObserver(["", "firefox running", "firefox running"])
    runner = make_runner(ex, obs)
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))
    assert state.final_status == FinalStatus.SUCCESS
    assert state.retry_count == 1
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].retries == 1
    assert len(ex.calls) == 2
    assert any("[RETRY]" in l for l in state.log)


def test_failed_step_replan_from_current_state():
    """When retry fails, the runner re-plans from the CURRENT state and
    the re-plan must NOT repeat the failed action."""
    ex = FakeExecutor([
        (False, "Launch timed out"),          # attempt 1 (transient)
        (False, "Couldn't find firefox-esr"), # adjusted retry fails
        (True, "Opened chromium-browser"),    # replanned alternate succeeds
    ])
    obs = FakeObserver(["", "", "", "chromium running"])
    planner = FakePlanner([
        [{"action": "desktop_open", "params": {"app": "chromium-browser"}}],
    ])
    runner = make_runner(ex, obs, planner)
    state = asyncio.run(runner.run(
        "open a browser",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))
    assert state.final_status == FinalStatus.SUCCESS
    assert state.retry_count == 1   # one safe retry before replanning
    assert state.replan_count == 1
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].action == "desktop_open"
    assert state.completed_steps[0].params["app"] == "chromium-browser"
    ctx = planner.contexts[0]
    assert ctx["failure_context"]
    assert "desktop_open" in " ".join(ctx["failed"])
    assert any("[REPLAN]" in l for l in state.log)
    assert "[PLAN] v2" in " ".join(state.log)


def test_changed_state_after_step_is_observed():
    """After each step the runner observes the NEW real state and uses it
    as the baseline for the next step (never assumes the old plan holds)."""
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "Here's what I found: https://github.com/x/y"),
    ])
    obs = FakeObserver([
        "empty desktop",
        "firefox window focused",
        "github search results shown",
    ])
    runner = make_runner(ex, obs)
    state = asyncio.run(runner.run(
        "open firefox and search github",
        [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "web_search", "params": {"query": "websocket examples"}},
        ]))
    assert state.final_status == FinalStatus.SUCCESS
    assert state.observed_state == "github search results shown"
    assert len(state.verification_results) == 2
    assert all(v["verified"] for v in state.verification_results)


def test_already_satisfied_step_is_idempotent(monkeypatch):
    """'open Firefox' when Firefox is already open → no duplicate launch;
    the step is marked ALREADY_SATISFIED from real OS evidence."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: True)
    ex = FakeExecutor([])  # must NEVER be called
    runner = make_runner(ex)
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))
    assert state.final_status == FinalStatus.SUCCESS
    assert ex.calls == []  # no duplicate launch
    assert state.completed_steps[0].status == StepStatus.ALREADY_SATISFIED
    assert "already open" in state.completed_steps[0].verification


def test_duplicate_action_prevented(monkeypatch):
    """A plan containing the same step twice must not execute it twice."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(True, "Opened firefox")])
    obs = FakeObserver(["", "firefox running"])
    runner = make_runner(ex, obs)
    plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "desktop_open", "params": {"app": "firefox"}},  # duplicate
    ]
    state = asyncio.run(runner.run("open firefox", plan))
    assert state.final_status == FinalStatus.SUCCESS
    assert len(ex.calls) == 1  # executed exactly once
    assert len(state.completed_steps) == 1


def test_loop_detection_repeated_identical_state(monkeypatch):
    """Repeated identical observed state (no progress) stops the task
    safely with an honest explanation — never an infinite loop."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(True, "Opened x")] * 5)
    obs = FakeObserver(["same screen"] * 10)
    runner = make_runner(ex, obs)
    plan = [{"action": "desktop_open", "params": {"app": "app%d" % i}}
            for i in range(5)]
    state = asyncio.run(runner.run("do five things", plan))
    assert state.final_status == FinalStatus.PARTIAL_FAILURE
    assert "loop" in state.blocker.lower() or "identical" in state.blocker.lower()
    assert "STOPPED" in " ".join(state.log)
    assert len(ex.calls) < 5  # stopped BEFORE executing all five steps


def test_loop_detector_repeated_identical_action():
    """Three identical successful actions in a row are detected as a loop
    (unit-level: identical actions inside one plan are duplicates and are
    skipped by plan validation, so the loop detector guards replans)."""
    ld = LoopDetector()
    sig = "desktop_open:[('app', 'x')]"
    assert ld.record_action(sig) is None
    assert ld.record_action(sig) is None
    assert ld.record_action(sig) is not None


def test_max_retry_limit(monkeypatch):
    """Retries are bounded by MAX_RETRIES_PER_STEP; after exhaustion the
    runner re-plans (or stops) — it never retries forever."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(False, "Launch timed out")] * 10)
    obs = FakeObserver([""] * 20)
    planner = FakePlanner([None])  # replanning yields nothing
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=2,
                        max_replans=1, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox", [{"action": "desktop_open", "params": {"app": "firefox"}}]))
    assert state.final_status == FinalStatus.FAILED
    assert state.retry_count == 2  # exactly MAX_RETRIES_PER_STEP
    assert len(ex.calls) == 3      # 1 initial + 2 retries
    assert state.replan_count == 1


def test_max_replan_limit(monkeypatch):
    """Re-planning is bounded by MAX_REPLANS; the planner producing failing
    plans repeatedly cannot loop the agent forever."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(False, "Couldn't find app")] * 20)
    obs = FakeObserver([""] * 40)
    planner = FakePlanner([
        [{"action": "desktop_open", "params": {"app": "ghostapp"}}]
    ] * 10)
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=3, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open ghostapp",
        [{"action": "desktop_open", "params": {"app": "ghostapp"}}]))
    assert state.replan_count <= limits.max_replans
    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)
    assert state.blocker  # honest explanation present


def test_cancellation():
    """A cancelled task stops safely and reports CANCELLED."""
    obs = FakeObserver(["", "firefox running"])
    holder = {}
    inner = FakeExecutor([(True, "Opened firefox"), (True, "ok")])

    class CancellingExecutor:
        def __init__(self, runner_holder, inner_exec):
            self.inner = inner_exec
            self.holder = runner_holder

        async def __call__(self, action):
            result = await self.inner(action)
            self.holder["runner"].cancel()  # cancel after step 1
            return result

    runner = make_runner(CancellingExecutor(holder, inner), obs)
    holder["runner"] = runner
    plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "web_search", "params": {"query": "test"}},
    ]
    state = asyncio.run(runner.run("open firefox and search", plan))
    assert state.final_status == FinalStatus.CANCELLED
    assert len(state.completed_steps) == 1  # step 1 done, step 2 never ran
    assert len(inner.calls) == 1


def test_partial_failure_reported_honestly(monkeypatch):
    """One verified success + one unrecoverable failure → PARTIAL_FAILURE
    with an honest summary (never 'Done')."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (False, "Couldn't load the page"),
    ])
    obs = FakeObserver(["", "firefox running", ""])
    planner = FakePlanner([None])  # no recovery plan possible
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=1, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox and open settings",
        [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "click_text", "params": {"text": "Settings"}},
        ]))
    assert state.final_status == FinalStatus.PARTIAL_FAILURE
    assert len(state.completed_steps) == 1
    assert len(state.failed_steps) == 1
    summary = state.summary()
    assert "1 step(s)" in summary
    assert "failed" in summary.lower()


def test_needs_input_on_unavailable_capability(monkeypatch):
    """An unavailable capability after partial progress → NEEDS_INPUT with
    an honest blocker and an offer of alternatives."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (False, "Couldn't find the settings panel"),
    ])
    obs = FakeObserver(["", "firefox running", ""])
    planner = FakePlanner([None])
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=1, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox and open settings",
        [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "click_text", "params": {"text": "Settings"}},
        ]))
    assert state.final_status == FinalStatus.NEEDS_INPUT
    assert "different approach" in state.summary()


def test_completion_criteria_not_met_after_failure(monkeypatch):
    """'Last action succeeded' ≠ 'whole task succeeded': a task with any
    unverified required step is never SUCCESS."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (False, "Couldn't load the page"),
    ])
    obs = FakeObserver(["", "firefox running", ""])
    planner = FakePlanner([None, None, None])
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=2, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox and open github",
        [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "browser_navigate", "params": {"url": "https://github.com"}},
        ]))
    assert state.final_status != FinalStatus.SUCCESS
    assert state.final_status in (FinalStatus.PARTIAL_FAILURE, FinalStatus.FAILED)


def test_planner_hallucinated_action_rejected():
    """A hallucinated action (not in the dispatcher's real action set) is
    rejected BEFORE dispatch — it never reaches the executor."""
    ex = FakeExecutor([])
    runner = make_runner(ex, transcript="do the thing")
    plan = [
        {"action": "self_destruct", "params": {}},          # hallucinated
        {"action": "desktop_open", "params": {}},           # missing 'app'
    ]
    state = asyncio.run(runner.run("do the thing", plan))
    assert state.final_status == FinalStatus.FAILED
    assert ex.calls == []  # nothing was dispatched
    assert "hallucinated" in " ".join(state.log)
    assert state.blocker


def test_planner_action_gate_rejection():
    """The Brain's intent-evidence gate blocks actions with no verb
    evidence in the transcript (e.g. close_app for 'open firefox')."""
    def gate(transcript: str, action: Dict[str, Any]) -> bool:
        return not action["action"].startswith("close")

    ex = FakeExecutor([])
    runner = make_runner(
        ex, validator=PlanValidator(action_gate=gate),
        transcript="open firefox")
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "close_app", "params": {"app": "firefox"}}]))
    assert state.final_status == FinalStatus.FAILED
    assert ex.calls == []
    assert "not authorized" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# Failure classification + loop detector unit tests
# ═══════════════════════════════════════════════════════════════

def test_failure_classification():
    from agent.task_state import FailureKind
    assert classify_failure("desktop_open", "Couldn't find firefox", "") \
        == FailureKind.UNAVAILABLE_CAPABILITY
    assert classify_failure("desktop_open", "Launch timed out", "") \
        == FailureKind.TRANSIENT
    assert classify_failure("click_text", "Couldn't click Settings", "") \
        == FailureKind.CHANGED_STATE
    assert classify_failure("desktop_open", "unknown action", "") \
        == FailureKind.WRONG_TOOL


def test_loop_detector_oscillation():
    ld = LoopDetector()
    assert ld.record_state("A") is None
    assert ld.record_state("B") is None
    assert ld.record_state("A") is None
    assert ld.record_state("B") is not None  # A,B,A,B oscillation


def test_loop_detector_repeated_plan():
    ld = LoopDetector()
    plan = [{"action": "desktop_open", "params": {"app": "x"}}]
    assert ld.record_plan(plan) is None
    assert ld.record_plan(plan) is not None


# ═══════════════════════════════════════════════════════════════
# Follow-up / task state preservation
# ═══════════════════════════════════════════════════════════════

def test_followup_open_first_result_uses_previous_task_state():
    """'open the first result' must resolve against the PREVIOUS task's
    real search results — not start from zero."""
    store = TaskStateStore()
    prev = TaskExecutionState(original_request="search github websocket")
    prev.final_status = FinalStatus.SUCCESS
    extract_artifacts(
        "web_search", {"query": "websocket"},
        "Here's what I found: https://github.com/a/b https://github.com/c/d",
        prev.artifacts)
    store.save(prev)

    cont = store.build_continuation("open the first result")
    assert cont is not None
    request, plan, inherited = cont
    assert inherited is prev
    assert plan[0]["action"] == "browser_navigate"
    assert plan[0]["params"]["url"] == "https://github.com/a/b"

    ex = FakeExecutor([(True, "Opened https://github.com/a/b")])
    obs = FakeObserver(["", "repo page open"])
    runner = make_runner(ex, obs, transcript=request)
    state = asyncio.run(runner.run(request, plan, inherited=inherited))
    assert state.final_status == FinalStatus.SUCCESS
    assert ex.calls[0]["params"]["url"] == "https://github.com/a/b"


def test_followup_continue_resumes_active_task(monkeypatch):
    """'continue' resumes the REMAINING steps of a partially-completed
    task (completed steps are not repeated)."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    store = TaskStateStore()
    prev = TaskExecutionState(
        original_request="open firefox and search and open result")
    prev.normalized_goal = "open firefox and search and open result"
    prev.final_status = FinalStatus.PARTIAL_FAILURE
    prev.current_plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "web_search", "params": {"query": "websocket"}},
        {"action": "browser_navigate", "params": {"url": "https://github.com/a/b"}},
    ]
    done = StepRecord(index=1, action="desktop_open",
                      params={"app": "firefox"}, status=StepStatus.COMPLETED,
                      verified=True)
    prev.completed_steps = [done]
    store.save(prev)
    assert store.active is prev

    cont = store.build_continuation("continue")
    assert cont is not None
    request, plan, inherited = cont
    # Only the REMAINING steps — the completed open is not repeated
    assert [s["action"] for s in plan] == ["web_search", "browser_navigate"]

    ex = FakeExecutor([
        (True, "Here's what I found: https://github.com/a/b"),
        (True, "Opened https://github.com/a/b"),
    ])
    obs = FakeObserver(["", "results shown", "page open"])
    runner = make_runner(ex, obs, transcript=request)
    state = asyncio.run(runner.run(request, plan, inherited=inherited))
    assert state.final_status == FinalStatus.SUCCESS
    assert [c["action"] for c in ex.calls] == ["web_search", "browser_navigate"]


def test_followup_repeat_for_and_close_last():
    store = TaskStateStore()
    prev = TaskExecutionState(original_request="open firefox")
    prev.final_status = FinalStatus.SUCCESS
    prev.completed_steps = [StepRecord(index=1, action="desktop_open",
                                       params={"app": "firefox"},
                                       status=StepStatus.COMPLETED,
                                       verified=True)]
    prev.artifacts["last_app"] = "firefox"
    store.save(prev)

    cont = store.build_continuation("do the same for chrome")
    assert cont is not None
    _req, plan, _inh = cont
    assert plan[0]["action"] == "desktop_open"
    assert plan[0]["params"]["app"] == "chrome"

    cont2 = store.build_continuation("close that")
    assert cont2 is not None
    _req2, plan2, _inh2 = cont2
    assert plan2[0]["action"] == "close_app"
    assert plan2[0]["params"]["app"] == "firefox"


def test_followup_cancel():
    store = TaskStateStore()
    prev = TaskExecutionState(original_request="long task")
    prev.final_status = FinalStatus.PARTIAL_FAILURE
    prev.current_plan = [{"action": "web_search", "params": {"query": "x"}}]
    store.save(prev)
    assert store.active is prev
    cont = store.build_continuation("cancel the task")
    assert cont is not None
    assert cont[1] == []  # cancel → empty plan
    assert store.active is None


def test_followup_resolver_patterns():
    assert FollowUpResolver.match("continue") == ("continue", None)
    assert FollowUpResolver.match("go on") == ("continue", None)
    assert FollowUpResolver.match("open the first result") == ("open_result", 0)
    assert FollowUpResolver.match("open the second result") == ("open_result", 1)
    assert FollowUpResolver.match("try another one") == ("retry_last", None)
    assert FollowUpResolver.match("do the same for chrome")[0] == "repeat_for"
    assert FollowUpResolver.match("close that") == ("close_last", None)
    assert FollowUpResolver.match("cancel") == ("cancel", None)
    # Non-follow-ups must NOT match
    assert FollowUpResolver.match("open firefox and search github") is None
    assert FollowUpResolver.match("what time is it") is None


# ═══════════════════════════════════════════════════════════════
# Integration-style scenarios (production plan shapes, faked execution)
# ═══════════════════════════════════════════════════════════════

def test_integration_open_firefox_search_github(monkeypatch):
    """'Open Firefox and search GitHub for Python WebSocket examples.'

    Proves: goal → plan → step → observation → verification → completion.
    Diego must NOT say 'Done' after merely opening Firefox — the search
    step must actually run and produce real content.
    """
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "Here's what I found on GitHub for Python WebSocket examples: "
               "websockets is a library for building WebSocket servers and "
               "clients in Python (from https://github.com/python-websockets/websockets)"),
    ])
    obs = FakeObserver([
        "empty desktop",
        "firefox window focused",
        "github search results visible",
    ])
    runner = make_runner(ex, obs,
                         transcript="open firefox and search github for "
                                    "python websocket examples")
    plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "web_search",
         "params": {"query": "Python WebSocket examples site:github.com"}},
    ]
    state = asyncio.run(runner.run(
        "Open Firefox and search GitHub for Python WebSocket examples.", plan))
    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 2
    # The answer comes from the ACTUAL search result
    assert "websockets" in state.summary().lower()
    assert state.artifacts["search_results"] == \
        ["https://github.com/python-websockets/websockets"]


def test_integration_search_open_best_read_summarize(monkeypatch):
    """'Search X, open the best result, read it and summarize it.'

    Proves the full chain: search → open → obtain real page content →
    answer from actual content. Never stops after opening the browser.
    """
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Here's what I found: https://github.com/python-websockets/websockets"),
        (True, "Opened https://github.com/python-websockets/websockets"),
        (True, "On screen: websockets — a library for building WebSocket "
               "servers and clients in Python with a focus on correctness "
               "and performance."),
    ])
    obs = FakeObserver([
        "browser closed",
        "search results page",
        "websockets repo page",
        "repo readme visible",
    ])
    runner = make_runner(ex, obs,
                         transcript="search websocket python open the best "
                                    "result read it and summarize it")
    plan = [
        {"action": "web_search", "params": {"query": "python websocket library"}},
        {"action": "browser_navigate",
         "params": {"url": "https://github.com/python-websockets/websockets"}},
        {"action": "read_screen", "params": {}},
    ]
    state = asyncio.run(runner.run(
        "Search X, open the best result, read it and summarize it.", plan))
    assert state.final_status == FinalStatus.SUCCESS
    # The summary is built from the ACTUAL page content captured by
    # read_screen — not from the planner's imagination.
    summary = state.summary()
    assert "WebSocket" in summary
    assert "servers and clients" in summary


def test_integration_vscode_inspect_and_continue(monkeypatch):
    """'Open VS Code, create/open X, inspect result, continue.'

    Proves: partial task state is preserved and 'continue' resumes the
    remaining steps using the previous task's context.
    """
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    # ── Turn 1: open VS Code + inspect ──
    ex1 = FakeExecutor([
        (True, "Opened code"),
        (True, "On screen: VS Code welcome tab, no folder open"),
    ])
    obs1 = FakeObserver(["", "vs code window focused", "welcome tab visible"])
    runner1 = make_runner(ex1, obs1, transcript="open vs code and inspect")
    plan1 = [
        {"action": "desktop_open", "params": {"app": "code"}},
        {"action": "read_screen", "params": {}},
    ]
    state1 = asyncio.run(runner1.run(
        "Open VS Code and inspect the result.", plan1))
    assert state1.final_status == FinalStatus.SUCCESS
    assert "VS Code welcome tab" in state1.summary()

    # ── Turn 2: 'continue' — open the project folder (remaining step) ──
    store = TaskStateStore()
    state1.current_plan = plan1 + [
        {"action": "open_folder", "params": {"path": "~/projects/Diego"}}]
    # The task is INCOMPLETE (a required step remains) — that is what
    # makes it resumable via 'continue'.
    state1.final_status = FinalStatus.PARTIAL_FAILURE
    store.save(state1)
    cont = store.build_continuation("continue")
    assert cont is not None
    _req, plan2, inherited = cont
    assert [s["action"] for s in plan2] == ["open_folder"]
    ex2 = FakeExecutor([(True, "Opened folder ~/projects/Diego")])
    obs2 = FakeObserver(["", "diego project open in explorer"])
    runner2 = make_runner(ex2, obs2, transcript="continue")
    state2 = asyncio.run(runner2.run(_req, plan2, inherited=inherited))
    assert state2.final_status == FinalStatus.SUCCESS
    assert state2.observed_state == "diego project open in explorer"


def test_integration_wifi_on_verified(monkeypatch):
    """'Turn Wi-Fi on, verify it is on.'

    Proves: real OS state is checked BEFORE acting (idempotency) and the
    step is only completed when the radio state is verified enabled.
    """
    # ── Case 1: Wi-Fi already on → idempotent skip, no duplicate action ──
    monkeypatch.setattr(task_state_module, "radio_enabled", lambda d: True)
    ex = FakeExecutor([])
    runner = make_runner(ex, transcript="turn wifi on")
    state = asyncio.run(runner.run(
        "turn Wi-Fi on", [{"action": "wifi_on", "params": {}}]))
    assert state.final_status == FinalStatus.SUCCESS
    assert ex.calls == []
    assert "already enabled" in state.summary()

    # ── Case 2: Wi-Fi off → enable, then verify it is on ──
    monkeypatch.setattr(task_state_module, "radio_enabled", lambda d: False)
    ex2 = FakeExecutor([(True, "Wi-Fi enabled")])
    obs2 = FakeObserver(["", "wifi enabled"])
    runner2 = make_runner(ex2, obs2, transcript="turn wifi on")
    state2 = asyncio.run(runner2.run(
        "turn Wi-Fi on", [{"action": "wifi_on", "params": {}}]))
    assert state2.final_status == FinalStatus.SUCCESS
    assert len(ex2.calls) == 1
    assert state2.completed_steps[0].verified is True


def test_task_state_store_singleton_exists():
    """The global store used by the Brain exists and round-trips state."""
    s = TaskExecutionState(original_request="x")
    s.final_status = FinalStatus.SUCCESS
    task_state_store.save(s)
    assert task_state_store.last is s
