"""Phase 15F regression tests — AUTONOMOUS TASK LIFECYCLE EVENTS.

Verifies that the TaskRunner publishes authoritative task lifecycle
transitions on the EXISTING event bus (core/event_bus.py) so the rest of
Diego can observe task start, progress, failure, retry, replan,
cancellation, and blocking — without a second event mechanism:

    task.started / task.step.started / task.step.completed
    task.step.failed / task.verification.failed
    task.retry / task.replan
    task.completed / task.failed / task.cancelled
    task.needs_confirmation / task.needs_input

Event payloads carry identifiers/counts/short reasons only — never action
params or sensitive command contents. Emission failures are non-fatal.

Mocks/fakes only — no LLM, network, OS, audio, camera, or real consumers.
"""

from __future__ import annotations

import asyncio

import pytest

from core.event_bus import EventBus
from agent.task_state import (
    FinalStatus,
    PlanValidator,
    StepStatus,
    TaskLimits,
    TaskRunner,
    task_state_store,
)
import agent.task_state as task_state_module


# ═══════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════

class EventRecorder:
    """A real EventBus with recording wildcard handlers (no UI)."""

    def __init__(self):
        self.bus = EventBus()
        self.events = []            # list of (type, data)

        async def recorder(event):
            self.events.append((event.type, dict(event.data or {})))

        self.bus.on("*", recorder)

    def types(self):
        return [t for t, _ in self.events]

    def payloads(self, event_type):
        return [d for t, d in self.events if t == event_type]


class ScriptedExecutor:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        if self.script:
            return self.script.pop(0)
        return False, "no scripted result"


class FailingBus:
    """A bus whose emit() always raises — consumer unavailable."""

    async def emit(self, *args, **kwargs):
        raise RuntimeError("event consumer unavailable")


class FakeObserver:

    def __init__(self, states):
        self.states = list(states)

    async def __call__(self):
        if self.states:
            return self.states.pop(0)
        return ""


class FakePlanner:

    def __init__(self, plans):
        self.plans = list(plans)

    async def __call__(self, request, context):
        if self.plans:
            return self.plans.pop(0)
        return None


def make_runner(executor, observer=None, planner=None, transcript="open firefox",
                **kwargs):
    return TaskRunner(
        executor=executor,
        observer=observer,
        planner=planner,
        validator=PlanValidator(),
        limits=TaskLimits(max_task_steps=12, max_retries_per_step=2,
                          max_replans=3, max_total_execution_time=60),
        transcript=transcript,
        **kwargs,
    )


PLAN = [{"action": "desktop_open", "params": {"app": "firefox"}}]


@pytest.fixture
def recorder(monkeypatch):
    """Patch the TaskRunner's bus import to a recorded EventBus."""
    rec = EventRecorder()
    monkeypatch.setattr("core.event_bus.bus", rec.bus, raising=False)
    return rec


@pytest.fixture(autouse=True)
def patch_app_running(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)


# ═══════════════════════════════════════════════════════════════
# 1. Task start emits the start event
# ═══════════════════════════════════════════════════════════════

def test_task_start_event(recorder):
    ex = ScriptedExecutor([(True, "Opened firefox")])
    runner = make_runner(ex, FakeObserver(["", ""]))
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.final_status == FinalStatus.SUCCESS
    assert "task.started" in recorder.types()
    payload = recorder.payloads("task.started")[0]
    assert payload["task_id"] == state.task_id
    assert payload["steps"] == 1
    assert payload["status"] == ""  # not optimistically SUCCESS
    # Privacy: the raw request text is NOT in the payload
    assert "open firefox" not in repr(payload)


# ═══════════════════════════════════════════════════════════════
# 2. Step start/completion lifecycle is observable
# ═══════════════════════════════════════════════════════════════

def test_step_lifecycle_observable(recorder):
    ex = ScriptedExecutor([(True, "Opened firefox")])
    runner = make_runner(ex, FakeObserver(["", ""]))
    state = asyncio.run(runner.run("open firefox", PLAN))

    types = recorder.types()
    assert "task.step.started" in types
    assert "task.step.completed" in types
    started = recorder.payloads("task.step.started")[0]
    completed = recorder.payloads("task.step.completed")[0]
    assert started["step"] == 1 and started["action"] == "desktop_open"
    assert completed["step"] == 1 and completed["action"] == "desktop_open"
    assert completed["evidence"] == "Opened firefox"
    # Ordering: start before completion
    assert types.index("task.step.started") < types.index("task.step.completed")


# ═══════════════════════════════════════════════════════════════
# 3. Verification failure produces the appropriate failure event
# ═══════════════════════════════════════════════════════════════

def test_verification_failure_event(recorder):
    ex = ScriptedExecutor([
        (False, "Verification failed: screen did not change"),
        (True, "Opened firefox"),
    ])
    runner = make_runner(ex, FakeObserver(["", ""]))
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.final_status == FinalStatus.SUCCESS  # recovered via retry
    failed = recorder.payloads("task.verification.failed")
    assert len(failed) == 1
    assert failed[0]["action"] == "desktop_open"
    assert failed[0]["step"] == 1
    assert "screen did not change" in failed[0]["reason"]
    assert "task.step.failed" in recorder.types()


# ═══════════════════════════════════════════════════════════════
# 4. Retry emits a retry event with bounded retry count
# ═══════════════════════════════════════════════════════════════

def test_retry_event_with_bounded_count(recorder):
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=2,
                        max_replans=3, max_total_execution_time=60)
    ex = ScriptedExecutor([
        (False, "Launch timed out"),
        (False, "Launch timed out"),
        (True, "Opened firefox"),
    ])
    runner = TaskRunner(
        executor=ex, observer=FakeObserver([""] * 10),
        validator=PlanValidator(), limits=limits,
        transcript="open firefox")
    state = asyncio.run(runner.run("open firefox", PLAN))

    retries = recorder.payloads("task.retry")
    assert [r["retry"] for r in retries] == [1, 2]   # bounded sequence
    assert all(r["max_retries"] == limits.max_retries_per_step
               for r in retries)
    assert all(r["action"] == "desktop_open" for r in retries)
    assert all(r["reason"] == "transient" for r in retries)
    assert state.retry_count == 2


# ═══════════════════════════════════════════════════════════════
# 5. Replan emits a replan event with bounded replan count
# ═══════════════════════════════════════════════════════════════

def test_replan_event_with_bounded_count(recorder):
    ex = ScriptedExecutor([(False, "Launch timed out")] * 50)
    planner = FakePlanner([PLAN] * 2)  # 2 successful replans, then stop
    runner = make_runner(ex, FakeObserver([""] * 100), planner)
    state = asyncio.run(runner.run("open firefox", PLAN))

    replans = recorder.payloads("task.replan")
    assert [r["replan"] for r in replans] == [1, 2]
    assert all(r["max_replans"] == 3 for r in replans)
    assert all("failed" in r["reason"] or r["reason"] for r in replans)
    assert state.replan_count == 2
    assert state.final_status != FinalStatus.SUCCESS


# ═══════════════════════════════════════════════════════════════
# 6. TASK_COMPLETED only after verified success
# ═══════════════════════════════════════════════════════════════

def test_completed_only_after_verified_success(recorder):
    ex = ScriptedExecutor([(True, "Opened firefox")])
    runner = make_runner(ex, FakeObserver(["", ""]))
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.final_status == FinalStatus.SUCCESS
    assert recorder.types().count("task.completed") == 1
    payload = recorder.payloads("task.completed")[0]
    assert payload["task_id"] == state.task_id
    assert payload["status"] == "SUCCESS"
    assert payload["completed_steps"] == 1
    # task.completed comes after the verified step completion
    types = recorder.types()
    assert types.index("task.step.completed") < types.index("task.completed")


# ═══════════════════════════════════════════════════════════════
# 7. Failed task emits TASK_FAILED and never TASK_COMPLETED
# ═══════════════════════════════════════════════════════════════

def test_failed_task_event(recorder):
    ex = ScriptedExecutor([(False, "Couldn't find ghostapp")] * 20)
    runner = make_runner(ex, FakeObserver([""] * 20), FakePlanner([]))
    state = asyncio.run(runner.run("open ghostapp", PLAN))

    assert state.final_status != FinalStatus.SUCCESS
    assert "task.failed" in recorder.types()
    assert "task.completed" not in recorder.types()
    payload = recorder.payloads("task.failed")[0]
    assert payload["task_id"] == state.task_id
    assert payload["blocker"]


# ═══════════════════════════════════════════════════════════════
# 8. Cancellation emits TASK_CANCELLED and never TASK_COMPLETED
# ═══════════════════════════════════════════════════════════════

def test_cancelled_task_event(recorder):
    ex = ScriptedExecutor([(True, "Opened firefox"),
                           (True, "Opened firefox")])
    runner = make_runner(ex, FakeObserver([""] * 10),
                         transcript="open firefox then again")

    # Wrap the executor to cancel after step 1 completes
    original = runner._executor

    async def cancel_after_first(action):
        result = await original(action)
        runner.cancel()
        return result

    runner._executor = cancel_after_first
    state = asyncio.run(runner.run(
        "open firefox then again",
        [PLAN[0], dict(PLAN[0], params={"app": "code"})]))

    assert state.final_status == FinalStatus.CANCELLED
    assert "task.cancelled" in recorder.types()
    assert "task.completed" not in recorder.types()
    assert recorder.payloads("task.cancelled")[0]["status"] == "CANCELLED"


# ═══════════════════════════════════════════════════════════════
# 9. NEEDS_CONFIRMATION emits/retains the correct state, never FAILED
# ═══════════════════════════════════════════════════════════════

def test_needs_confirmation_event_distinct(recorder):
    ex = ScriptedExecutor([(True, "Shutting down")])
    runner = make_runner(ex, FakeObserver([""]),
                         transcript="shut down the machine")

    async def deny(action, reason, params):
        return False

    runner._confirmation_callback = deny
    state = asyncio.run(runner.run(
        "shut down the machine",
        [{"action": "shutdown", "params": {}}]))

    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    types = recorder.types()
    assert "task.needs_confirmation" in types
    assert "task.failed" not in types
    assert "task.completed" not in types
    payload = recorder.payloads("task.needs_confirmation")[0]
    assert payload["status"] == "NEEDS_CONFIRMATION"
    assert payload["reason"]


# ═══════════════════════════════════════════════════════════════
# 10. NEEDS_INPUT remains distinct from FAILED
# ═══════════════════════════════════════════════════════════════

def test_needs_input_event_distinct(recorder):
    ex = ScriptedExecutor([(True, "Opened firefox"),
                           (False, "Couldn't find mpv — mpv is not installed")] * 20)
    runner = make_runner(ex, FakeObserver([""] * 30), FakePlanner([]))
    state = asyncio.run(runner.run(
        "open firefox and play music",
        [PLAN[0], {"action": "play_media", "params": {"query": "test"}}]))

    assert state.final_status == FinalStatus.NEEDS_INPUT
    types = recorder.types()
    assert "task.needs_input" in types
    assert "task.completed" not in types
    payload = recorder.payloads("task.needs_input")[0]
    assert payload["status"] == "NEEDS_INPUT"
    assert payload["blocker"]


# ═══════════════════════════════════════════════════════════════
# 11. Event consumer failure does not crash TaskRunner or change outcome
# ═══════════════════════════════════════════════════════════════

def test_event_consumer_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr("core.event_bus.bus", FailingBus(), raising=False)
    ex = ScriptedExecutor([
        (False, "Launch timed out"),
        (True, "Opened firefox"),
    ])
    runner = make_runner(ex, FakeObserver(["", ""]))
    state = asyncio.run(runner.run("open firefox", PLAN))

    # The task outcome is identical to the event-less case
    assert state.final_status == FinalStatus.SUCCESS
    assert state.completed_steps[0].verified is True
    assert state.retry_count == 1
    # Sensitive path also unaffected (confirmation pause reached normally)
    runner2 = make_runner(ScriptedExecutor([(True, "x")]),
                          FakeObserver([""]),
                          transcript="shut down")
    async def deny(action, reason, params):
        return False
    runner2._confirmation_callback = deny
    state2 = asyncio.run(runner2.run(
        "shut down", [{"action": "shutdown", "params": {}}]))
    assert state2.final_status == FinalStatus.NEEDS_CONFIRMATION


# ═══════════════════════════════════════════════════════════════
# 12. Payloads never expose action params / sensitive command contents
# ═══════════════════════════════════════════════════════════════

def test_events_do_not_expose_action_params(recorder):
    ex = ScriptedExecutor([(False, "Couldn't find ghostapp")] * 10)
    runner = make_runner(ex, FakeObserver([""] * 20), FakePlanner([]))
    # The sensitive content is in the plan params AND the request text
    asyncio.run(runner.run(
        "open ghostapp password=secret password123",
        [{"action": "desktop_open",
          "params": {"app": "ghostapp", "password": "secret password123"}}]))

    blob = repr(recorder.events)
    assert "secret password123" not in blob
    assert "password" not in blob.lower()