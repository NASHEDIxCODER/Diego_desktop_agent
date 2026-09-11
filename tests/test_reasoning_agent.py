"""
Phase 21A — reasoning-capable autonomous core (tests/test_reasoning_agent.py).

30 scenarios covering:
  1  simple command stays deterministic
  2  complex goal creates a plan
  3  plan executes sequentially
  4  observation changes next step (adaptive planning)
  5  verification is required
  6  failed step triggers diagnosis
  7  retry uses corrected strategy
  8  repeated failure triggers replan
  9  replan uses current observed state
  10 verified step is never repeated
  11 cancellation stops execution
  12 confirmation pauses execution
  13 confirmation resumes the same task
  14 unrelated command does not destroy task state
  15 context trimming preserves active goal
  16 context trimming preserves constraints
  17 context trimming preserves verified evidence
  18 previous lesson is retrieved
  19 stale lesson is not blindly trusted
  20 successful task creates lesson
  21 failed task creates failure lesson
  22 unverified result creates no success lesson
  23 user preference does not bypass authorization
  24 model timeout fails safely
  25 malformed model JSON fails safely
  26 model cannot execute tools directly
  27 infinite planning loop is bounded
  28 infinite retry loop is bounded
  29 infinite replan loop is bounded
  30 final SUCCESS requires verification

All model responses are MOCKED; all tools are FAKE. The deterministic
runtime (TaskRunner) retains full authority over execution/verification.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest

import agent.task_state as task_state_module
from agent.task_state import (
    FinalStatus,
    StepRecord,
    StepStatus,
    TaskExecutionState,
    TaskLimits,
    task_state_store,
)
from agent.reasoning_agent import (
    Mode,
    ReasoningAgent,
    choose_mode,
    diagnose_failure_deterministic,
)
from agent.reasoning_context import ReasoningContextComposer
from agent.reasoning_state import ReasoningState
from agent.lessons import (
    TaskLessonStore,
    learn_user_preference,
    get_user_preference,
    is_safe_preference,
)
from ai.context_monitor import ContextMonitor
from ai.reasoning_model import (
    BaseReasoningModel,
    ReasoningCallStatus,
    ReasoningResult,
    extract_json,
)


# ═══════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════

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
    def __init__(self, states: List[str]):
        self.states = list(states)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.states:
            return self.states.pop(0)
        return ""


class FakePlanner:
    """Scripted planner: pops plans per call; records all contexts."""

    def __init__(self, plans: List[Optional[List[Dict[str, Any]]]],
                 default: Optional[List[Dict[str, Any]]] = None):
        self.plans = list(plans)
        self.default = default
        self.contexts: List[Dict[str, Any]] = []

    async def __call__(self, request: str, context: Dict[str, Any]
                       ) -> Optional[List[Dict[str, Any]]]:
        self.contexts.append(dict(context))
        if self.plans:
            return self.plans.pop(0)
        return self.default


class FakeModel(BaseReasoningModel):
    """Mocked reasoning model returning scripted ReasoningResults."""

    name = "fake"

    def __init__(self, plan_results=None, diagnose_results=None,
                 revise_results=None, reflect_results=None):
        self.plan_calls: List[Tuple[str, str]] = []
        self.diagnose_calls: List[Tuple[str, str]] = []
        self.revise_calls: List[Tuple[str, List[str]]] = []
        self.reflect_calls: List[Tuple[str, str]] = []
        self._plan_results = list(plan_results or [])
        self._diagnose_results = list(diagnose_results or [])
        self._revise_results = list(revise_results or [])
        self._reflect_results = list(reflect_results or [])

    @staticmethod
    def ok_data(data: Dict[str, Any]) -> ReasoningResult:
        return ReasoningResult(status=ReasoningCallStatus.OK, data=data)

    async def reason(self, prompt: str, context: str = "", **kw
                     ) -> ReasoningResult:
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)

    async def plan(self, goal: str, context: str = "", **kw
                   ) -> ReasoningResult:
        self.plan_calls.append((goal, context))
        if self._plan_results:
            return self._plan_results.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)

    async def diagnose(self, goal: str, action: str, observed_result: str,
                       error: str, attempt: int, **kw) -> ReasoningResult:
        self.diagnose_calls.append((action, error))
        if self._diagnose_results:
            return self._diagnose_results.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)

    async def revise_plan(self, goal: str, observation: str,
                          completed: List[str], remaining: List[Dict], **kw
                          ) -> ReasoningResult:
        self.revise_calls.append(
            (observation, [r.get("action") for r in remaining]))
        if self._revise_results:
            return self._revise_results.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)

    async def reflect(self, goal: str, outcome: str, **kw
                      ) -> ReasoningResult:
        self.reflect_calls.append((goal, outcome))
        if self._reflect_results:
            return self._reflect_results.pop(0)
        return ReasoningResult(status=ReasoningCallStatus.DISABLED)


class FakeConfirmation:
    def __init__(self, responses: List[bool]):
        self.responses = list(responses)
        self.requests: List[Tuple[str, str, Dict[str, Any]]] = []

    async def __call__(self, action: str, reason: str, params: Dict[str, Any]
                       ) -> bool:
        self.requests.append((action, reason, params))
        if self.responses:
            return self.responses.pop(0)
        return False


def fresh_store(tmp_path=None) -> TaskLessonStore:
    path = os.path.join(str(tmp_path or tempfile.mkdtemp()), "lessons.json")
    return TaskLessonStore(path=path)


def make_agent(executor, observer=None, planner=None, model=None,
               limits: Optional[TaskLimits] = None, store=None,
               confirmation=None, approved_actions=None, action_gate=None
               ) -> ReasoningAgent:
    return ReasoningAgent(
        executor=executor,
        observer=observer,
        planner=planner,
        reasoning_model=model,
        limits=limits or TaskLimits(max_task_steps=8,
                                    max_retries_per_step=2,
                                    max_replans=2,
                                    max_total_execution_time=30),
        confirmation_callback=confirmation,
        action_gate=action_gate,
        lesson_store=store or fresh_store(),
        transcript="",
        approved_actions=approved_actions,
    )


# ═══════════════════════════════════════════════════════════════
# Scenarios 1–10
# ═══════════════════════════════════════════════════════════════

def test_01_simple_command_stays_deterministic():
    """S1: simple known commands never enter expensive reasoning."""
    assert choose_mode("open firefox") is Mode.DETERMINISTIC
    assert choose_mode("play believer") is Mode.DETERMINISTIC
    assert choose_mode("volume up") is Mode.DETERMINISTIC
    assert choose_mode("what is the time") is Mode.DETERMINISTIC
    # Multi-step and ambiguous goals DO use reasoning.
    assert choose_mode("open firefox and then open the file manager"
                       ) is Mode.AUTONOMOUS
    assert choose_mode("find a file in my workspace and report its size"
                       ) is Mode.REASONING


def test_02_complex_goal_creates_plan(monkeypatch):
    """S2: a complex goal gets a plan from the reasoning model."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([(True, "Opened firefox"), (True, "On screen: ok")])
    model = FakeModel(plan_results=[FakeModel.ok_data({
        "plan": [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "read_screen", "params": {}},
        ],
        "assumptions": ["firefox is installed"],
    })])
    agent = make_agent(ex, model=model)
    result = asyncio.run(agent.run("summarize what is on my screen after "
                                   "opening the browser"))
    assert result.task_state.current_plan
    assert len(model.plan_calls) == 1
    assert result.success
    assert len(result.task_state.completed_steps) == 2
    assert "firefox is installed" in result.reasoning_state.assumptions


def test_03_plan_executes_sequentially(monkeypatch):
    """S3: plan steps run in order."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([(True, "Opened firefox"), (True, "Folder opened")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]])
    agent = make_agent(ex, planner=planner)
    result = asyncio.run(agent.run("open firefox and then open the file "
                                   "manager"))
    assert result.success
    actions = [c["action"] for c in ex.calls]
    assert actions == ["desktop_open", "open_folder"]


def test_04_observation_changes_next_step(monkeypatch):
    """S4: after a VERIFIED step, the observation can revise the remaining
    plan (adaptive planning), and the revision is validated + bounded."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([(True, "Opened firefox"), (True, "On screen: ready")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "browser_search", "params": {"query": "kittens"}},
    ]])
    model = FakeModel(revise_results=[FakeModel.ok_data({
        "keep_plan": False,
        "revised_plan": [{"action": "read_screen", "params": {}}],
        "reason": "browser search unnecessary — screen already shows it",
    })])
    agent = make_agent(ex, observer=FakeObserver(["firefox window open"]),
                       planner=planner, model=model)
    result = asyncio.run(agent.run("open firefox and then search for "
                                   "kittens"))
    assert result.success
    # The revised step replaced the original plan step.
    assert ex.calls[1]["action"] == "read_screen"
    assert agent.rs.adaptive_revision_count == 1
    assert len(model.revise_calls) == 1


def test_05_verification_is_required(monkeypatch):
    """S5: an unverified step is never treated as success."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([(False, "firefox is not installed")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
    ]])
    agent = make_agent(ex, planner=planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert not result.success
    assert result.task_state.final_status != FinalStatus.SUCCESS
    assert any(not vr.get("verified")
               for vr in result.task_state.verification_results)


def test_06_failed_step_triggers_diagnosis(monkeypatch):
    """S6: a failed step produces a STRUCTURED diagnosis (model-backed)."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([
        (False, "couldn't click the icon"),
        (True, "Opened firefox"),
    ])
    planner = FakePlanner([
        [{"action": "click_text", "params": {"text": "the icon"}}],
        [{"action": "desktop_open", "params": {"app": "firefox"}}],
    ])
    model = FakeModel(diagnose_results=[FakeModel.ok_data({
        "failure_kind": "wrong_tool",
        "probable_cause": "action does not exist in the registry",
        "retry_suitable": False,
        "alternative": "use desktop_open",
        "next_strategy": "alternative_tool",
    })])
    agent = make_agent(ex, planner=planner, model=model)
    result = asyncio.run(agent.run("open firefox and then click the icon"))
    assert model.diagnose_calls, "model diagnosis was not consulted"
    assert agent.last_diagnosis.failure_kind == "wrong_tool"
    assert agent.last_diagnosis.next_strategy == "alternative_tool"
    assert result.reasoning_state.failed_attempts
    assert result.success  # recovery via corrected plan


def test_07_retry_uses_corrected_strategy(monkeypatch):
    """S7: transient failure → strategic retry (bounded), then success."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([
        (False, "temporarily busy, try again"),
        (True, "Opened firefox"),
    ])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
    ]])
    agent = make_agent(ex, planner=planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert result.success
    assert result.task_state.retry_count == 1
    assert len(ex.calls) == 2  # original attempt + ONE retry


def test_08_repeated_failure_triggers_replan(monkeypatch):
    """S8: retry-exhausted/unsafe failure → re-plan from CURRENT state."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([
        (False, "couldn't find the app firefox"),
        (True, "Opened chromium"),
    ])
    planner = FakePlanner([
        [{"action": "desktop_open", "params": {"app": "firefox"}}],
        [{"action": "desktop_open", "params": {"app": "chromium"}}],
    ])
    agent = make_agent(ex, planner=planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert result.success
    assert result.task_state.replan_count == 1
    assert ex.calls[1]["action"] == "desktop_open"
    assert ex.calls[1]["params"]["app"] == "chromium"


def test_09_replan_uses_current_observed_state(monkeypatch):
    """S9: the replan context carries the CURRENT observation, completed
    and failed steps — context is never lost between replans."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([
        (False, "couldn't find the app firefox"),
        (True, "Opened chromium"),
    ])
    planner = FakePlanner([
        [{"action": "desktop_open", "params": {"app": "firefox"}}],
        [{"action": "desktop_open", "params": {"app": "chromium"}}],
    ])
    agent = make_agent(ex, observer=FakeObserver(["firefox window closed"]),
                       planner=planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert result.success
    replan_ctx = planner.contexts[-1]
    assert replan_ctx["observed_state"] == "firefox window closed"
    assert any("firefox" in f for f in replan_ctx["failed"])
    # Structured reasoning state is part of the replan context.
    assert replan_ctx["reasoning_state"]["goal"]
    assert replan_ctx["failure_diagnosis"]["failure_kind"]


def test_10_verified_step_never_repeated(monkeypatch):
    """S10: a step identical to an already-VERIFIED step is skipped, not
    re-executed (no repeated side effects)."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([(True, "Opened firefox")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "desktop_open", "params": {"app": "firefox"}},
    ]])
    agent = make_agent(ex, planner=planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert result.success
    open_calls = [c for c in ex.calls if c["action"] == "desktop_open"]
    assert len(open_calls) == 1


# ═══════════════════════════════════════════════════════════════
# Scenarios 11–19
# ═══════════════════════════════════════════════════════════════

def test_11_cancellation_stops_execution(monkeypatch):
    """S11: cancellation stops the loop between steps — never mid-action."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)

    executed = []
    holder: dict = {"agent": None}

    async def cancelling_executor(action):
        executed.append(dict(action))
        holder["agent"].current_runner.cancel()
        return True, "Opened firefox"

    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]])
    agent = ReasoningAgent(
        executor=cancelling_executor,
        planner=planner,
        limits=TaskLimits(max_task_steps=8, max_retries_per_step=2,
                          max_replans=2, max_total_execution_time=30),
        lesson_store=fresh_store(),
    )
    holder["agent"] = agent
    result = asyncio.run(agent.run("open firefox and then open the file "
                                   "manager"))
    assert result.task_state.final_status == FinalStatus.CANCELLED
    # Only the first step ran; the second never executed.
    assert [c["action"] for c in executed] == ["desktop_open"]


def test_12_confirmation_pauses_execution(monkeypatch):
    """S12: a sensitive action pauses the task awaiting confirmation."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "System shutting down"),
    ])
    confirm = FakeConfirmation([False])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "shutdown", "params": {}},
    ]])
    agent = make_agent(ex, planner=planner, confirmation=confirm)
    result = asyncio.run(agent.run(
        "open firefox and then shut down the system"))
    assert result.task_state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert result.task_state.pending_confirmation is not None
    assert result.task_state.pending_confirmation.action == "shutdown"
    assert not result.success
    # The shutdown step was NEVER executed.
    assert [c["action"] for c in ex.calls] == ["desktop_open"]
    assert "denied shutdown" in result.reasoning_state.user_decisions


def test_13_confirmation_resumes_same_task(monkeypatch):
    """S13: with the user's approval recorded, the SAME task resumes and
    completes without re-asking."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "System shutting down"),
    ])
    confirm = FakeConfirmation([])  # must NOT be called again
    sig = StepRecord(index=0, action="shutdown", params={}).signature()
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "shutdown", "params": {}},
    ]])
    agent = make_agent(ex, planner=planner, confirmation=confirm,
                       approved_actions=frozenset({sig}))
    result = asyncio.run(agent.run(
        "open firefox and then shut down the system"))
    assert result.success
    assert confirm.requests == []  # pre-approved — no re-confirmation
    assert [c["action"] for c in ex.calls] == ["desktop_open", "shutdown"]


def test_14_unrelated_command_does_not_destroy_task_state():
    """S14: task state (incl. paused confirmations) survives unrelated
    activity — persistence is keyed per task id."""
    state_a = TaskExecutionState(original_request="open firefox",
                                 normalized_goal="open firefox")
    state_a.current_plan = [{"action": "desktop_open",
                             "params": {"app": "firefox"}}]
    state_a.completed_steps = [StepRecord(
        index=1, action="desktop_open", params={"app": "firefox"},
        status=StepStatus.COMPLETED, verified=True, result="Opened firefox")]
    path = task_state_store.persist(state_a)
    assert path
    # An unrelated task runs and finishes afterwards.
    ex = FakeExecutor([(True, "Opened chromium")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "chromium"}},
    ]])
    agent = make_agent(ex, planner=planner)
    result_b = asyncio.run(agent.run("open chromium and then check the time"))
    assert result_b.success
    # Task A's persisted state is untouched and restorable.
    restored = task_state_store.load(state_a.task_id)
    assert restored is not None
    assert restored.original_request == "open firefox"
    assert restored.completed_steps


def test_15_context_trimming_preserves_active_goal():
    """S15: trimming drops background context first, keeps the goal."""
    monitor = ContextMonitor()
    monitor.configure("test-model", context_limit=120,
                      default_output_reserve=0)
    composer = ReasoningContextComposer(monitor=monitor)
    composed = composer.compose(
        goal="open firefox",
        background=["filler " * 100],
        reserve_output=1,
    )
    assert "GOAL: open firefox" in composed.text
    assert composed.trimmed and "P7_background" in composed.trimmed_layers
    assert monitor.trim_events  # trimming is observable, never silent


def test_16_context_trimming_preserves_constraints():
    """S16: user constraints survive trimming (P0 layer)."""
    monitor = ContextMonitor()
    monitor.configure("test-model", context_limit=120,
                      default_output_reserve=0)
    composer = ReasoningContextComposer(monitor=monitor)
    composed = composer.compose(
        goal="clean up my downloads folder",
        constraints=["do not delete any files"],
        background=["noise " * 100],
        conversation_history=["unrelated " * 50],
        reserve_output=1,
    )
    assert "CONSTRAINT: do not delete any files" in composed.text


def test_17_context_trimming_preserves_verified_evidence():
    """S17: verified evidence (P3) survives trimming and older steps are
    summarized (never silently truncated)."""
    monitor = ContextMonitor()
    monitor.configure("test-model", context_limit=220,
                      default_output_reserve=0)
    # (reserve_output=1 — the monitor maps reserve 0 to its default.)
    composer = ReasoningContextComposer(monitor=monitor, recent_step_keep=2)
    steps = [f"step {i} verified done" for i in range(8)]
    composed = composer.compose(
        goal="prepare the report",
        constraints=["never overwrite the original"],
        verified_evidence=["report file verified at /tmp/report.txt"],
        completed_step_summaries=steps,
        background=["history " * 100],
        reserve_output=1,
    )
    assert "VERIFIED: report file verified at /tmp/report.txt" in composed.text
    assert "CONSTRAINT: never overwrite the original" in composed.text
    # Older steps were COMPRESSED into a summary (with a reference kept).
    assert composed.summarized
    assert "EARLIER:" in composed.text and "summarized" in composed.text
    # The most recent steps are still visible.
    assert "step 7 verified done" in composed.text


def test_18_previous_lesson_is_retrieved():
    """S18: relevant prior lessons are retrieved before planning."""
    store = fresh_store()
    store.add_lesson(type="task_lesson", task_pattern="open_application",
                     lesson="Firefox launches reliably with desktop_open",
                     evidence="verified_success")
    ex = FakeExecutor([(True, "Opened firefox")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
    ]])
    agent = make_agent(ex, planner=planner, store=store)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert result.success
    assert agent._lessons_used, "lesson was not retrieved"
    lesson, score = agent._lessons_used[0]
    assert lesson.task_pattern == "open_application"
    assert score > 0
    # The lesson reached the composed context (P6 layer).
    composed = agent._compose_context("open firefox", agent.rs,
                                      agent._current_lesson_lines())
    assert "LESSON:" in composed.text


def test_19_stale_lesson_is_not_blindly_trusted():
    """S19: low-confidence lessons are flagged untrusted; a failed reuse
    lowers confidence further; pruning removes decayed lessons. Lessons
    are always framed as evidence, never truth."""
    store = fresh_store()
    lesson = store.add_lesson(
        type="task_lesson", task_pattern="open_application",
        lesson="Firefox opens via the legacy xdotool hack",
        evidence="verified_success", confidence=0.35)
    assert lesson is not None
    assert not lesson.trustworthy  # below the trust threshold
    assert store.retrieve("open firefox", trusted_only=True) == []
    # Framing: the composed lesson block instructs verification.
    from agent.reasoning_context import ReasoningContextComposer
    block = ReasoningContextComposer.p6_lessons([lesson.line()])
    assert "VERIFY the current environment" in block
    # A remembered strategy that fails loses confidence (Task 10).
    before = lesson.confidence
    store.record_reuse(lesson.id, success=False)
    assert lesson.confidence < before
    assert lesson.failures == 1
    store.record_reuse(lesson.id, success=False)
    assert not lesson.trustworthy
    # Prune removes it once it decays below the floor.
    lesson.confidence = 0.04
    store.prune()
    assert store.get(lesson.id) is None


# ═══════════════════════════════════════════════════════════════
# Scenarios 20–30
# ═══════════════════════════════════════════════════════════════

def test_20_successful_task_creates_lesson(monkeypatch):
    """S20: a VERIFIED success produces a reusable success lesson."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    store = fresh_store()
    ex = FakeExecutor([(True, "Opened firefox")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
    ]])
    agent = make_agent(ex, planner=planner, store=store)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert result.success
    assert result.lessons
    types = {l.type for l in result.lessons}
    assert "successful_strategy" in types
    for lesson in result.lessons:
        assert lesson.evidence == "verified_success"
        assert lesson.source_task_id == result.task_state.task_id
        assert lesson.confidence > 0


def test_21_failed_task_creates_failure_lesson(monkeypatch):
    """S21: a verified failure produces a failed-strategy lesson."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    store = fresh_store()
    ex = FakeExecutor([(False, "couldn't find the app firefox")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
    ]])
    agent = make_agent(ex, planner=planner, store=store)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert not result.success
    assert result.lessons
    lesson = result.lessons[0]
    assert lesson.type == "failed_strategy"
    assert lesson.evidence == "verified_failure"
    assert "firefox" in lesson.lesson


def test_22_unverified_result_creates_no_success_lesson():
    """S22: cancelled / needs-confirmation tasks NEVER yield success
    lessons (learning from unverified results is impossible)."""
    store = fresh_store()
    # Cancelled task WITH verified-looking steps → still no lesson.
    cancelled = TaskExecutionState(original_request="open firefox",
                                   normalized_goal="open firefox",
                                   final_status=FinalStatus.CANCELLED)
    cancelled.completed_steps = [StepRecord(
        index=1, action="desktop_open", params={"app": "firefox"},
        status=StepStatus.COMPLETED, verified=True, result="Opened firefox")]
    assert store.record_task_outcome(cancelled) == []
    # Paused for confirmation → no lesson.
    paused = TaskExecutionState(original_request="shutdown the system",
                                normalized_goal="shutdown the system",
                                final_status=FinalStatus.NEEDS_CONFIRMATION)
    paused.pending_confirmation = StepRecord(index=1, action="shutdown",
                                             params={})
    assert store.record_task_outcome(paused) == []
    # Needs input → no lesson.
    needs_input = TaskExecutionState(
        original_request="open the thing", normalized_goal="open the thing",
        final_status=FinalStatus.NEEDS_INPUT)
    assert store.record_task_outcome(needs_input) == []
    # Unverified evidence is refused at the store level too.
    assert store.add_lesson(type="task_lesson", task_pattern="x",
                            lesson="claim", evidence="unverified") is None
    assert store.count == 0


def test_23_user_preference_does_not_bypass_authorization(monkeypatch):
    """S23: safe preferences are learned; sensitive ones are refused; and
    a learned preference NEVER removes the confirmation requirement."""
    assert learn_user_preference("app", "browser", "firefox",
                                 confidence_boost=0.7) == "firefox"
    assert get_user_preference("app", "browser") == "firefox"
    # Sensitive categories are never learnable.
    assert learn_user_preference("credentials", "password", "hunter2"
                                 ) is None
    assert not is_safe_preference("permission", "sudo")
    # Preference present — confirmation STILL required for shutdown.
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([(True, "Opened firefox"), (True, "shutting down")])
    confirm = FakeConfirmation([False])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "shutdown", "params": {}},
    ]])
    agent = make_agent(ex, planner=planner, confirmation=confirm)
    result = asyncio.run(agent.run(
        "open firefox and then shut down the system"))
    assert result.task_state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert len(confirm.requests) == 1


def test_24_model_timeout_fails_safely(monkeypatch):
    """S24: a model timeout is a SAFE failure — the deterministic planner
    path still completes the task, and the uncertainty is recorded."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor([(True, "Opened firefox")])
    planner = FakePlanner([
        None,  # initial planning declines → model consulted (times out)
        [{"action": "desktop_open", "params": {"app": "firefox"}}],
    ])
    model = FakeModel(plan_results=[ReasoningResult(
        status=ReasoningCallStatus.TIMEOUT, error="timed out after 30s")])
    agent = make_agent(ex, planner=planner, model=model)
    result = asyncio.run(agent.run("organize my project files and report "
                                   "what changed"))
    assert result.success  # deterministic fallback worked
    assert any("model unavailable" in u
               for u in result.reasoning_state.uncertainty)


def test_25_malformed_model_json_fails_safely():
    """S25: malformed model output → explicit INVALID_JSON, data=None."""
    with pytest.raises(ValueError):
        extract_json("")
    with pytest.raises(ValueError):
        extract_json("no json at all, sorry")
    with pytest.raises(ValueError):
        extract_json('{"plan": [oops}')
    # Valid forms parse (fences, prose-wrapped).
    assert extract_json('{"plan": []}')["plan"] == []
    assert extract_json('```json\n{"plan": []}\n```')["plan"] == []
    assert extract_json('Sure! {"plan": [{"action": "x"}]} done'
                        )["plan"][0]["action"] == "x"
    # The safe-failure result carries no data.
    bad = ReasoningResult(status=ReasoningCallStatus.INVALID_JSON,
                          raw_text="garbage", error="no parsable JSON")
    assert not bad.ok and bad.data is None


def test_26_model_cannot_execute_tools_directly(monkeypatch):
    """S26: model-proposed unknown actions are REJECTED by the runtime —
    the model has no tool-execution path at all."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    monkeypatch.setattr(task_state_module, "registry_tool_available",
                        lambda name: False)
    ex = FakeExecutor([])
    model = FakeModel(plan_results=[FakeModel.ok_data({
        "plan": [{"action": "exec_python",
                  "params": {"code": "import os; os.system('touch /tmp/x')"}}],
    })])
    agent = make_agent(ex, model=model)
    result = asyncio.run(agent.run("run some python for me please"))
    assert ex.calls == []  # nothing executed
    assert not result.success
    # The model interface has no tool-execution surface.
    assert not hasattr(model, "execute")
    assert not hasattr(model, "dispatch")
    assert not hasattr(BaseReasoningModel, "execute")


def test_27_infinite_planning_loop_is_bounded(monkeypatch):
    """S27: a planner returning the SAME plan forever is stopped by the
    loop detector + replan budget."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    ex = FakeExecutor(default := [])  # noqa: F841

    always_fail = []

    async def failing_executor(action):
        always_fail.append(dict(action))
        return False, "couldn't find the app firefox"

    same_plan = [{"action": "desktop_open", "params": {"app": "firefox"}}]
    planner = FakePlanner([], default=same_plan)
    agent = make_agent(failing_executor, planner=planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    assert result.task_state.final_status is not None
    assert result.task_state.final_status != FinalStatus.SUCCESS
    assert result.task_state.replan_count <= 2  # bounded by max_replans


def test_28_infinite_retry_loop_is_bounded(monkeypatch):
    """S28: transient failures retry only up to max_retries_per_step."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)

    calls = []

    async def failing_executor(action):
        calls.append(dict(action))
        return False, "temporarily busy, try again"

    same_plan = [{"action": "desktop_open", "params": {"app": "firefox"}}]
    planner = FakePlanner([], default=same_plan)
    agent = make_agent(failing_executor, planner=planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    state = result.task_state
    assert state.final_status != FinalStatus.SUCCESS
    # Bounded per step: no single step retried more than the limit.
    assert max(s.retries for s in state.failed_steps) <= 2
    assert state.final_status is not None  # loop terminated


def test_29_infinite_replan_loop_is_bounded(monkeypatch):
    """S29: alternating (non-identical) replans are still bounded by the
    replan budget — the loop always terminates."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)

    calls = []

    async def failing_executor(action):
        calls.append(dict(action))
        return False, "couldn't find the app"

    plans = iter([
        [{"action": "desktop_open", "params": {"app": "firefox"}}],
        [{"action": "desktop_open", "params": {"app": "chrome"}}],
        [{"action": "desktop_open", "params": {"app": "chromium"}}],
        [{"action": "desktop_open", "params": {"app": "brave"}}],
    ])

    async def alternating_planner(request, context):
        try:
            return next(plans)
        except StopIteration:
            return None

    agent = make_agent(failing_executor, planner=alternating_planner)
    result = asyncio.run(agent.run("open firefox and then check the time"))
    state = result.task_state
    assert state.replan_count == 2  # exactly the replan budget
    assert state.final_status != FinalStatus.SUCCESS
    assert state.ended_at is not None  # terminated


def test_30_final_success_requires_verification(monkeypatch):
    """S30: SUCCESS only when every step completed WITH verification;
    tool invocation alone is never enough."""
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)
    # Verified path → SUCCESS.
    ex = FakeExecutor([(True, "Opened firefox"), (True, "Folder opened")])
    planner = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]])
    agent = make_agent(ex, planner=planner)
    result = asyncio.run(agent.run("open firefox and then open the file "
                                   "manager"))
    assert result.success
    assert all(vr.get("verified")
               for vr in result.task_state.verification_results)
    assert result.task_state.final_status == FinalStatus.SUCCESS

    # Unverified path → NEVER SUCCESS.
    ex2 = FakeExecutor([(True, "Opened firefox"),
                        (False, "folder is not accessible")])
    planner2 = FakePlanner([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]])
    agent2 = make_agent(ex2, planner=planner2)
    result2 = asyncio.run(agent2.run("open firefox and then open the file "
                                     "manager"))
    assert result2.task_state.final_status != FinalStatus.SUCCESS
    assert result2.task_state.blocker  # honest blocker reported
