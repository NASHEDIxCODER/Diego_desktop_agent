"""End-to-end tests for Diego's BOUNDED AUTONOMOUS TASK CONTROLLER.

Tests cover all required scenarios:
  - successful multi-step task
  - failed action + recovery
  - incorrect initial plan + replan
  - verification failure
  - stale knowledge overridden by live state
  - bounded retries
  - loop detection
  - timeout
  - partial observation
  - unknown state
  - refusal/confirmation for sensitive actions
  - truthful final reporting
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agent.task_state import (
    Evidence,
    EvidenceSource,
    FinalStatus,
    PlanValidator,
    StepRecord,
    StepStatus,
    TaskExecutionState,
    TaskLimits,
    TaskRunner,
    is_sensitive_action,
)
from agent.task_controller import (
    AutonomousTaskController,
    EvidenceFirstVerifier,
    WorldState,
    WorldStateGatherer,
    build_final_report,
)
import agent.task_state as task_state_module


# ═══════════════════════════════════════════════════════════════
# Test fakes
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


class FakeConfirmation:
    """Scripted confirmation callback."""

    def __init__(self, responses: List[bool]):
        self.responses = list(responses)
        self.requests: List[Tuple[str, str, Dict[str, Any]]] = []

    async def __call__(self, action: str, reason: str,
                       params: Dict[str, Any]) -> bool:
        self.requests.append((action, reason, params))
        if self.responses:
            return self.responses.pop(0)
        return False


def make_controller(executor, observer=None, planner=None,
                    limits: Optional[TaskLimits] = None,
                    confirmation=None,
                    action_gate=None) -> AutonomousTaskController:
    return AutonomousTaskController(
        executor=executor,
        observer=observer,
        planner=planner,
        limits=limits or TaskLimits(max_task_steps=12, max_retries_per_step=2,
                                    max_replans=3, max_total_execution_time=60),
        confirmation_callback=confirmation,
        action_gate=action_gate,
    )


def make_runner(executor, observer=None, planner=None,
                limits: Optional[TaskLimits] = None,
                confirmation=None,
                transcript: str = "open firefox and search github") -> TaskRunner:
    return TaskRunner(
        executor=executor,
        observer=observer,
        planner=planner,
        validator=PlanValidator(),
        limits=limits or TaskLimits(max_task_steps=12, max_retries_per_step=2,
                                    max_replans=3, max_total_execution_time=60),
        transcript=transcript,
        confirmation_callback=confirmation,
    )


# ═══════════════════════════════════════════════════════════════
# 1. Successful multi-step task
# ═══════════════════════════════════════════════════════════════

def test_successful_multi_step_task(monkeypatch):
    """A multi-step task completes with SUCCESS only after all steps verified.

    HERMETIC: the desktop observation boundary (agent.task_state.app_running,
    the real pgrep check used by IdempotencyChecker) is replaced by a
    deterministic STATEFUL fake — Firefox is ABSENT before the action and
    PRESENT after the executor succeeds. The test never depends on the
    real desktop session / DISPLAY / installed apps.
    """
    desktop = {"firefox": False}  # controlled fake desktop state

    def fake_app_running(app: str) -> bool:
        return desktop.get(str(app).lower(), False)

    monkeypatch.setattr(task_state_module, "app_running", fake_app_running)

    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "Here's what I found: https://github.com/example/repo"),
        (True, "Opened https://github.com/example/repo"),
    ])

    async def observing_executor(action):
        ok, msg = await ex(action)
        # A successful open CHANGES the fake desktop state (app appears).
        if ok and action.get("action") == "desktop_open":
            app = str(action.get("params", {}).get("app", "")).lower()
            desktop[app] = True
        return ok, msg

    obs = FakeObserver([
        "empty desktop",
        "firefox window focused",
        "search results shown",
        "repo page open",
    ])
    controller = make_controller(observing_executor, obs)
    plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "web_search", "params": {"query": "python websocket"}},
        {"action": "browser_navigate", "params": {"url": "https://github.com/example/repo"}},
    ]
    state = asyncio.run(controller.run(
        "Open Firefox, search for python websocket, open the result", plan))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 3
    assert all(s.verified for s in state.completed_steps)
    assert len(ex.calls) == 3
    # Evidence is recorded
    assert len(state.evidence_log) > 0


def test_idempotent_already_satisfied_skips_execution(monkeypatch):
    """HERMETIC REGRESSION A: the target state is ALREADY present → the
    step is marked ALREADY_SATISFIED with verified deterministic evidence
    and is NOT re-executed (no duplicate side effect). The observation is
    faked, so the real session is irrelevant."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: True)
    ex = FakeExecutor([])  # nothing may be executed
    obs = FakeObserver(["firefox already open"])
    controller = make_controller(ex, obs)
    state = asyncio.run(controller.run(
        "Open Firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert ex.calls == []  # already satisfied → no execution
    rec = state.completed_steps[0]
    assert rec.status == StepStatus.ALREADY_SATISFIED
    assert rec.verified is True
    assert rec.evidence_source == EvidenceSource.DETERMINISTIC_SYSTEM
    assert "already open" in rec.verification


def test_idempotent_absent_target_executes_and_verifies(monkeypatch):
    """HERMETIC REGRESSION B: the target is ABSENT → the action executes
    exactly once and verification marks it COMPLETED. The fake desktop
    state flips to "present" only after the successful action."""
    desktop = {"firefox": False}  # absent before the action

    def fake_app_running(app: str) -> bool:
        return desktop.get(str(app).lower(), False)

    monkeypatch.setattr(task_state_module, "app_running", fake_app_running)
    ex = FakeExecutor([(True, "Opened firefox")])

    async def observing_executor(action):
        ok, msg = await ex(action)
        if ok and action.get("action") == "desktop_open":
            app = str(action.get("params", {}).get("app", "")).lower()
            desktop[app] = True  # present AFTER a successful action
        return ok, msg

    obs = FakeObserver(["firefox window focused"])
    controller = make_controller(observing_executor, obs)
    state = asyncio.run(controller.run(
        "Open Firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert [c["action"] for c in ex.calls] == ["desktop_open"]  # executed once
    rec = state.completed_steps[0]
    assert rec.status == StepStatus.COMPLETED
    assert rec.verified is True


def test_successful_multi_step_with_artifacts():
    """Multi-step task captures artifacts from real results."""
    ex = FakeExecutor([
        (True, "Here's what I found: https://github.com/a/b https://github.com/c/d"),
        (True, "Opened https://github.com/a/b"),
        (True, "On screen: WebSocket library documentation"),
    ])
    obs = FakeObserver(["", "results page", "repo page", "docs visible"])
    controller = make_controller(ex, obs)
    plan = [
        {"action": "web_search", "params": {"query": "websocket library"}},
        {"action": "browser_navigate", "params": {"url": "https://github.com/a/b"}},
        {"action": "read_screen", "params": {}},
    ]
    state = asyncio.run(controller.run("search and read", plan))

    assert state.final_status == FinalStatus.SUCCESS
    assert state.artifacts.get("last_url") == "https://github.com/a/b"
    assert "WebSocket" in state.artifacts.get("answer", "")
    # Summary comes from actual content
    assert "WebSocket" in state.summary()


# ═══════════════════════════════════════════════════════════════
# 2. Failed action + recovery
# ═══════════════════════════════════════════════════════════════

def test_failed_action_retry_recovery(monkeypatch):
    """A transient failure is retried and recovers."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (False, "Launch timed out"),          # first attempt fails
        (True, "Opened firefox"),             # retry succeeds
    ])
    obs = FakeObserver(["", "", "firefox running"])
    runner = make_runner(ex, obs)
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert state.retry_count == 1
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].retries == 1
    assert len(ex.calls) == 2


def test_failed_action_replan_recovery(monkeypatch):
    """When retries fail, replan from current state recovers with alternative."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (False, "Launch timed out"),
        (False, "Couldn't find firefox-esr"),
        (True, "Opened chromium-browser"),
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
    assert state.replan_count == 1
    assert state.completed_steps[0].params["app"] == "chromium-browser"
    # Replan context includes failure information
    ctx = planner.contexts[0]
    assert "failure_context" in ctx
    assert ctx["failure_context"]


# ═══════════════════════════════════════════════════════════════
# 3. Incorrect initial plan + replan
# ═══════════════════════════════════════════════════════════════

def test_incorrect_plan_replan(monkeypatch):
    """An incorrect initial plan is detected and replanned."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (False, "Couldn't find nonexistent-app"),
        (True, "Opened firefox"),
    ])
    obs = FakeObserver(["", "", "firefox running"])
    planner = FakePlanner([
        [{"action": "desktop_open", "params": {"app": "firefox"}}],
    ])
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=3, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "nonexistent-app"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert state.replan_count == 1
    assert len(state.failed_steps) == 1  # original failed step recorded
    assert len(state.completed_steps) == 1  # replanned step succeeded


def test_hallucinated_plan_rejected():
    """A plan with hallucinated actions is rejected before execution."""
    ex = FakeExecutor([])
    runner = make_runner(ex, transcript="do something")
    state = asyncio.run(runner.run(
        "do something",
        [{"action": "teleport_to_moon", "params": {}}]))

    assert state.final_status == FinalStatus.FAILED
    assert ex.calls == []  # nothing executed
    assert "hallucinated" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# 4. Verification failure
# ═══════════════════════════════════════════════════════════════

def test_verification_failure_reported(monkeypatch):
    """A step that fails verification is not marked complete."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (False, "Couldn't load the page"),
    ])
    obs = FakeObserver(["", "firefox running", ""])
    planner = FakePlanner([None])
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=1, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox and navigate",
        [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "browser_navigate", "params": {"url": "https://example.com"}},
        ]))

    assert state.final_status == FinalStatus.PARTIAL_FAILURE
    assert len(state.completed_steps) == 1
    assert len(state.failed_steps) == 1
    assert state.failed_steps[0].verified is False


def test_dispatch_failure_markers_detected():
    """Dispatch results with failure markers are not verified as success."""
    verifier = EvidenceFirstVerifier()

    # "Couldn't" marker
    ok, reason, source = asyncio.run(verifier.verify(
        "desktop_open", {"app": "x"}, "Couldn't find app", True, "", ""))
    assert ok is False

    # "unavailable" marker
    ok, reason, source = asyncio.run(verifier.verify(
        "volume_set", {"percent": 50}, "Volume control unavailable", True, "", ""))
    assert ok is False


# ═══════════════════════════════════════════════════════════════
# 5. Stale knowledge overridden by live state
# ═══════════════════════════════════════════════════════════════

def test_live_state_overrides_stale_knowledge():
    """Live observation takes priority over stale local knowledge."""
    world = WorldState(
        live_observation="firefox window focused, github.com open",
        system_facts={"os": "Linux"},
        knowledge_facts=[
            Evidence(fact="user usually works on project X",
                     source=EvidenceSource.LOCAL_KNOWLEDGE),
        ],
    )

    # Live observation wins
    fact, source = world.get_fact("firefox")
    assert source == EvidenceSource.LIVE_OBSERVATION

    # System facts are deterministic
    fact, source = world.get_fact("os")
    assert source == EvidenceSource.DETERMINISTIC_SYSTEM
    assert fact == "Linux"

    # Knowledge is lower priority but available
    fact, source = world.get_fact("project X")
    assert source == EvidenceSource.LOCAL_KNOWLEDGE


def test_unknown_state_marked_explicitly():
    """Facts without evidence are marked UNKNOWN, not guessed."""
    world = WorldState(
        live_observation="",
        system_facts={},
        knowledge_facts=[],
    )

    fact, source = world.get_fact("nonexistent")
    assert fact is None
    assert source == EvidenceSource.UNKNOWN

    world.add_unknown("nonexistent")
    assert "nonexistent" in world.unknown_facts


# ═══════════════════════════════════════════════════════════════
# 6. Bounded retries
# ═══════════════════════════════════════════════════════════════

def test_bounded_retries(monkeypatch):
    """Retries are bounded by MAX_RETRIES_PER_STEP."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(False, "Launch timed out")] * 10)
    obs = FakeObserver([""] * 20)
    planner = FakePlanner([None])
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=2,
                        max_replans=1, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))

    assert state.final_status == FinalStatus.FAILED
    assert state.retry_count == 2  # exactly MAX_RETRIES_PER_STEP
    assert len(ex.calls) == 3      # 1 initial + 2 retries


def test_bounded_replans(monkeypatch):
    """Replans are bounded by MAX_REPLANS."""
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


def test_bounded_task_steps(monkeypatch):
    """Total task steps are bounded by MAX_TASK_STEPS."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(True, "Opened app")] * 20)
    obs = FakeObserver([f"state {i}" for i in range(30)])
    limits = TaskLimits(max_task_steps=3, max_retries_per_step=0,
                        max_replans=0, max_total_execution_time=60)
    runner = make_runner(ex, obs, limits=limits)
    plan = [{"action": "desktop_open", "params": {"app": f"app{i}"}}
            for i in range(10)]
    state = asyncio.run(runner.run("open many apps", plan))

    assert len(state.completed_steps) + len(state.failed_steps) <= 3
    assert "step limit" in state.blocker.lower()


# ═══════════════════════════════════════════════════════════════
# 7. Loop detection
# ═══════════════════════════════════════════════════════════════

def test_loop_detection_repeated_state(monkeypatch):
    """Repeated identical observed state stops the task safely."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(True, "Opened x")] * 5)
    obs = FakeObserver(["same screen"] * 10)
    runner = make_runner(ex, obs)
    plan = [{"action": "desktop_open", "params": {"app": f"app{i}"}}
            for i in range(5)]
    state = asyncio.run(runner.run("do five things", plan))

    assert state.final_status == FinalStatus.PARTIAL_FAILURE
    assert "loop" in state.blocker.lower() or "identical" in state.blocker.lower()
    assert len(ex.calls) < 5  # stopped before all steps


def test_loop_detection_repeated_failure(monkeypatch):
    """Repeated identical failures are detected and stop the task."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(False, "same error")] * 10)
    obs = FakeObserver([""] * 20)
    planner = FakePlanner([
        [{"action": "desktop_open", "params": {"app": "x"}}],
        [{"action": "desktop_open", "params": {"app": "y"}}],
        [{"action": "desktop_open", "params": {"app": "z"}}],
    ])
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=3, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open app", [{"action": "desktop_open", "params": {"app": "x"}}]))

    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)


# ═══════════════════════════════════════════════════════════════
# 8. Timeout
# ═══════════════════════════════════════════════════════════════

def test_timeout_stops_task(monkeypatch):
    """Task exceeding MAX_TASK_RUNTIME is stopped."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)

    class SlowExecutor:
        def __init__(self):
            self.calls = 0

        async def __call__(self, action):
            self.calls += 1
            await asyncio.sleep(0.1)  # simulate slow action
            return True, "ok"

    ex = SlowExecutor()
    obs = FakeObserver([""] * 100)
    limits = TaskLimits(max_task_steps=100, max_retries_per_step=0,
                        max_replans=0, max_total_execution_time=0.05)  # very short
    runner = make_runner(ex, obs, limits=limits)
    plan = [{"action": "desktop_open", "params": {"app": f"app{i}"}}
            for i in range(50)]
    state = asyncio.run(runner.run("open many apps", plan))

    assert "time" in state.blocker.lower() or "budget" in state.blocker.lower()
    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)


# ═══════════════════════════════════════════════════════════════
# 9. Partial observation
# ═══════════════════════════════════════════════════════════════

def test_partial_observation(monkeypatch):
    """Task continues with partial/missing observation."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "Searched"),
    ])
    # Observer returns empty for some calls
    obs = FakeObserver(["", "", ""])
    runner = make_runner(ex, obs)
    state = asyncio.run(runner.run(
        "open firefox and search",
        [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "web_search", "params": {"query": "test"}},
        ]))

    # Task completes even with missing observation
    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 2


def test_observation_failure_nonfatal(monkeypatch):
    """Observer failure does not crash the task."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)

    class FailingObserver:
        async def __call__(self):
            raise RuntimeError("observation failed")

    ex = FakeExecutor([(True, "Opened firefox")])
    runner = make_runner(ex, FailingObserver())
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))

    assert state.final_status == FinalStatus.SUCCESS


# ═══════════════════════════════════════════════════════════════
# 10. Unknown state
# ═══════════════════════════════════════════════════════════════

def test_unknown_state_handling():
    """Unknown facts are explicitly tracked, not guessed."""
    state = TaskExecutionState(original_request="test")

    # Add evidence with unknown source
    state.add_evidence("something happened", EvidenceSource.UNKNOWN, 0.5)

    unknown_evidence = state.get_evidence(EvidenceSource.UNKNOWN)
    assert len(unknown_evidence) == 1
    assert unknown_evidence[0].confidence == 0.5


def test_step_evidence_source_tracked(monkeypatch):
    """Step records track their evidence source."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: True)
    ex = FakeExecutor([])
    runner = make_runner(ex)
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))

    # Idempotent step has deterministic evidence
    assert state.completed_steps[0].evidence_source == EvidenceSource.DETERMINISTIC_SYSTEM
    assert state.completed_steps[0].status == StepStatus.ALREADY_SATISFIED


# ═══════════════════════════════════════════════════════════════
# 11. Refusal/confirmation for sensitive actions
# ═══════════════════════════════════════════════════════════════

def test_sensitive_action_detection():
    """Sensitive actions are correctly detected."""
    # Destructive
    ok, reason = is_sensitive_action("shutdown", {})
    assert ok is True

    ok, reason = is_sensitive_action("delete_file", {"path": "/tmp/x"})
    assert ok is True

    # Communication
    ok, reason = is_sensitive_action("send_email", {"to": "x@y.com"})
    assert ok is True

    # Read-only is allowed
    ok, reason = is_sensitive_action("read_screen", {})
    assert ok is False

    ok, reason = is_sensitive_action("web_search", {"query": "test"})
    assert ok is False


def test_sensitive_action_denied_without_callback(monkeypatch):
    """Sensitive actions are denied when no confirmation callback exists."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([])
    runner = make_runner(ex, confirmation=None)
    state = asyncio.run(runner.run(
        "shutdown the computer",
        [{"action": "shutdown", "params": {}}]))

    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert ex.calls == []  # action was never executed
    assert state.pending_confirmation is not None
    assert "confirmation" in state.blocker.lower()


def test_sensitive_action_approved_with_callback(monkeypatch):
    """Sensitive actions execute when user approves."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(True, "Shutting down")])
    obs = FakeObserver(["", "shutdown initiated"])
    confirm = FakeConfirmation([True])  # user approves
    runner = make_runner(ex, obs, confirmation=confirm)
    state = asyncio.run(runner.run(
        "shutdown the computer",
        [{"action": "shutdown", "params": {}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(ex.calls) == 1
    assert len(confirm.requests) == 1
    assert confirm.requests[0][0] == "shutdown"


def test_sensitive_action_denied_by_user(monkeypatch):
    """Sensitive actions are blocked when user denies."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([])
    confirm = FakeConfirmation([False])  # user denies
    runner = make_runner(ex, confirmation=confirm)
    state = asyncio.run(runner.run(
        "shutdown the computer",
        [{"action": "shutdown", "params": {}}]))

    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert ex.calls == []
    assert len(confirm.requests) == 1


def test_sensitive_action_in_params_detected(monkeypatch):
    """Actions with destructive keywords in params are detected."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([])
    runner = make_runner(ex, confirmation=None)
    # type_text with destructive command
    state = asyncio.run(runner.run(
        "run a command",
        [{"action": "type_text", "params": {"text": "rm -rf /"}}]))

    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert ex.calls == []


# ═══════════════════════════════════════════════════════════════
# 12. Truthful final reporting
# ═══════════════════════════════════════════════════════════════

def test_truthful_report_success():
    """Success report reflects actual completed work."""
    state = TaskExecutionState(original_request="test")
    state.final_status = FinalStatus.SUCCESS
    state.completed_steps = [
        StepRecord(index=1, action="desktop_open", params={"app": "firefox"},
                   status=StepStatus.COMPLETED, verified=True,
                   result="Opened firefox"),
    ]
    report = build_final_report(state)
    assert "Opened firefox" in report


def test_truthful_report_partial_failure():
    """Partial failure report is honest about what failed."""
    state = TaskExecutionState(original_request="test")
    state.final_status = FinalStatus.PARTIAL_FAILURE
    state.completed_steps = [
        StepRecord(index=1, action="desktop_open", status=StepStatus.COMPLETED,
                   verified=True),
    ]
    state.failed_steps = [
        StepRecord(index=2, action="browser_navigate", status=StepStatus.FAILED,
                   verified=False),
    ]
    state.blocker = "page could not be loaded"
    report = build_final_report(state)

    assert "1 step(s)" in report
    assert "failed" in report.lower()
    assert "page could not be loaded" in report


def test_truthful_report_needs_confirmation():
    """Needs-confirmation report explains what requires approval."""
    state = TaskExecutionState(original_request="test")
    state.final_status = FinalStatus.NEEDS_CONFIRMATION
    state.confirmation_reason = "'shutdown' is a sensitive action"
    report = build_final_report(state)

    assert "confirmation" in report.lower()
    assert "sensitive" in report.lower()


def test_no_false_success_claims(monkeypatch):
    """Never claim success for unverified actions."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (False, "Couldn't navigate"),
    ])
    obs = FakeObserver(["", "firefox running", ""])
    planner = FakePlanner([None])
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=0,
                        max_replans=1, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits)
    state = asyncio.run(runner.run(
        "open firefox and navigate",
        [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "browser_navigate", "params": {"url": "https://x.com"}},
        ]))

    # Must NOT be SUCCESS
    assert state.final_status != FinalStatus.SUCCESS
    report = build_final_report(state)
    assert "failed" in report.lower() or "couldn't" in report.lower()


# ═══════════════════════════════════════════════════════════════
# Evidence-first verification tests
# ═══════════════════════════════════════════════════════════════

def test_evidence_first_verifier_stateful_action():
    """Stateful actions require observable state change."""
    verifier = EvidenceFirstVerifier()

    # State changed — verified
    ok, reason, source = asyncio.run(verifier.verify(
        "desktop_open", {"app": "firefox"}, "Opened firefox", True,
        "empty desktop", "firefox window focused"))
    assert ok is True
    assert source == EvidenceSource.LIVE_OBSERVATION

    # No state change — not verified
    ok, reason, source = asyncio.run(verifier.verify(
        "desktop_open", {"app": "firefox"}, "Opened firefox", True,
        "same state", "same state"))
    assert ok is False
    assert source == EvidenceSource.UNKNOWN


def test_evidence_first_verifier_informational_action():
    """Informational actions use their result as evidence."""
    verifier = EvidenceFirstVerifier()

    ok, reason, source = asyncio.run(verifier.verify(
        "read_screen", {}, "On screen: Hello World", True, "", ""))
    assert ok is True
    assert source == EvidenceSource.TOOL_RESULT
    assert "Hello World" in reason

    # Empty result — not verified
    ok, reason, source = asyncio.run(verifier.verify(
        "read_screen", {}, "", True, "", ""))
    assert ok is False


# ═══════════════════════════════════════════════════════════════
# World state gatherer tests
# ═══════════════════════════════════════════════════════════════

def test_world_state_priority():
    """World state respects priority of truth."""
    world = WorldState(
        live_observation="current: chrome is open",
        system_facts={"cpu": "Intel i7"},
        knowledge_facts=[
            Evidence(fact="user prefers firefox", source=EvidenceSource.LOCAL_KNOWLEDGE),
        ],
    )

    # Live observation wins over knowledge
    fact, source = world.get_fact("chrome")
    assert source == EvidenceSource.LIVE_OBSERVATION

    # System facts are deterministic
    fact, source = world.get_fact("cpu")
    assert source == EvidenceSource.DETERMINISTIC_SYSTEM

    # Knowledge is available but lower priority
    fact, source = world.get_fact("firefox")
    assert source == EvidenceSource.LOCAL_KNOWLEDGE


# ═══════════════════════════════════════════════════════════════
# Integration: controller with all components
# ═══════════════════════════════════════════════════════════════

def test_controller_full_pipeline(monkeypatch):
    """Full pipeline: goal → world state → plan → execute → verify → report."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "Here's what I found: https://github.com/test"),
    ])
    obs = FakeObserver(["", "firefox open", "results shown"])
    controller = make_controller(ex, obs)
    plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "web_search", "params": {"query": "test query"}},
    ]
    state = asyncio.run(controller.run("open firefox and search", plan))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 2

    # Final report is truthful
    report = build_final_report(state)
    assert "found" in report.lower() or "steps completed" in report.lower()


def test_controller_with_confirmation_flow(monkeypatch):
    """Controller handles sensitive action confirmation flow."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([
        (True, "Opened firefox"),
        (True, "System shutting down"),
    ])
    obs = FakeObserver(["", "firefox open", "shutdown"])
    confirm = FakeConfirmation([True])
    controller = make_controller(ex, obs, confirmation=confirm)
    plan = [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "shutdown", "params": {}},
    ]
    state = asyncio.run(controller.run("open firefox then shutdown", plan))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(confirm.requests) == 1
    assert confirm.requests[0][0] == "shutdown"