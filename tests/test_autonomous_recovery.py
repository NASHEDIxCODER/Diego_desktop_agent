"""Phase 15E regression tests — AUTONOMOUS FAILURE RECOVERY AND BOUNDED REPLAN.

Audits and locks in the EXISTING failure → retry → recovery → replan path
of the authoritative TaskRunner (agent/task_state.py):

    validate → execute → observe → verify → classify_failure
    → bounded retry (safe kinds only, adjusted params)
    → bounded replan (structured failure context)
    → LoopDetector (repeated failed strategy)
    → honest final states (SUCCESS only when verified)

Plus the Phase 15E fix: CANCELLED is authoritative inside the recovery
path — a cancellation requested during a failing step does NOT burn the
retry budget and never triggers a post-failure re-plan.

Mocks/fakes only — no LLM, network, OS, or real tools.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.task_state import (
    FailureKind,
    FinalStatus,
    PlanValidator,
    StepStatus,
    TaskLimits,
    TaskRunner,
    classify_failure,
    task_state_store,
)
import agent.task_state as task_state_module


# ═══════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════

class ScriptedExecutor:
    """Returns scripted (ok, result) tuples in order; records every call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        if self.script:
            return self.script.pop(0)
        return False, "no scripted result"


class CancelDuringFailureExecutor:
    """First call fails transiently and requests cancellation mid-recovery;
    subsequent calls would succeed (proving no retry/replan ran)."""

    def __init__(self, runner_ref):
        self.runner_ref = runner_ref
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        if len(self.calls) == 1:
            # Request cancellation DURING the failing step (before retry)
            self.runner_ref["runner"].cancel()
            return False, "Launch timed out"
        return True, "Opened firefox"


class FakeObserver:

    def __init__(self, states):
        self.states = list(states)

    async def __call__(self):
        if self.states:
            return self.states.pop(0)
        return ""


class FakePlanner:
    """Records every replan call's (request, context) for inspection."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = []          # list of (request, context)

    async def __call__(self, request, context):
        self.calls.append((request, dict(context or {})))
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


# ═══════════════════════════════════════════════════════════════
# 1. Transient execution failure → existing retry succeeds
# ═══════════════════════════════════════════════════════════════

def test_transient_failure_retry_succeeds(monkeypatch):
    """First attempt fails transiently, the existing retry path succeeds,
    and SUCCESS is reached only after a verified attempt."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = ScriptedExecutor([
        (False, "Launch timed out"),   # attempt 1 — transient failure
        (True, "Opened firefox"),      # attempt 2 — verified success
    ])
    runner = make_runner(ex, FakeObserver(["", ""]))
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(ex.calls) == 2
    assert state.retry_count == 1
    assert state.failed_steps == []          # the step ultimately completed
    assert state.completed_steps[0].retries == 1
    assert state.completed_steps[0].verified is True
    assert state.completed_steps[0].verification == "success"
    assert any("[RETRY]" in line for line in state.log)
    assert "[VERIFY] success (retry 1)" in " ".join(state.log)
    assert "[TASK] COMPLETE" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# 2. Verification failure → recovery occurs → SUCCESS only when verified
# ═══════════════════════════════════════════════════════════════

def test_verification_failure_recovers_then_success(monkeypatch):
    """Execution 'succeeds' but verification says NO (executor returns
    ok=False with a clean-looking result) → the existing retry path runs
    and the task is SUCCESS only after a verified attempt."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = ScriptedExecutor([
        (False, "Verification failed: screen did not change"),  # attempt 1
        (True, "Opened firefox"),                               # attempt 2
    ])
    runner = make_runner(ex, FakeObserver(["", ""]))
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(ex.calls) == 2
    assert state.completed_steps[0].retries == 1
    assert state.completed_steps[0].verified is True
    # The failed attempt is recorded as verification evidence, never as success
    assert state.verification_results[0]["verified"] is False
    assert state.verification_results[1]["verified"] is True


# ═══════════════════════════════════════════════════════════════
# 3. Retry bound — retries never exceed the configured max
# ═══════════════════════════════════════════════════════════════

def test_retry_bound_never_exceeded(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=2,
                        max_replans=3, max_total_execution_time=60)
    ex = ScriptedExecutor([(False, "Launch timed out")] * 50)
    runner = TaskRunner(
        executor=ex, observer=FakeObserver([""] * 100),
        validator=PlanValidator(), limits=limits,
        transcript="open firefox")
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.retry_count <= limits.max_retries_per_step
    assert all(s.retries <= limits.max_retries_per_step
               for s in state.failed_steps)
    assert len(ex.calls) == 1 + limits.max_retries_per_step
    assert state.final_status != FinalStatus.SUCCESS


# ═══════════════════════════════════════════════════════════════
# 4. Replan bound — replans never exceed the configured max
# ═══════════════════════════════════════════════════════════════

def test_replan_bound_never_exceeded(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=2,
                        max_replans=3, max_total_execution_time=60)
    ex = ScriptedExecutor([(False, "Launch timed out")] * 100)
    planner = FakePlanner([PLAN] * 20)  # always proposes the same plan
    runner = TaskRunner(
        executor=ex, observer=FakeObserver([""] * 200), planner=planner,
        validator=PlanValidator(), limits=limits,
        transcript="open firefox")
    state = asyncio.run(runner.run("open firefox", PLAN))

    # The replan budget is a hard ceiling; the LoopDetector may stop the
    # loop EARLIER (repeated identical plan) — both are honest bounds.
    assert state.replan_count <= limits.max_replans
    assert state.final_status != FinalStatus.SUCCESS
    assert state.blocker  # honest explanation (replan limit or loop detection)


# ═══════════════════════════════════════════════════════════════
# 5. Replan receives structured failure context
# ═══════════════════════════════════════════════════════════════

def test_replan_receives_failure_context(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = ScriptedExecutor([(False, "Launch timed out")] * 10)
    planner = FakePlanner([PLAN])  # one replan, then stop
    runner = make_runner(ex, FakeObserver([""] * 20), planner)
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert len(planner.calls) >= 1
    request, context = planner.calls[0]
    # The replan context carries the failed action, the failed step, and
    # the failure reason (existing structured context — reused, not rebuilt)
    assert context["goal"] == "open firefox"
    assert any("desktop_open" in f for f in context["failed"])
    assert any("timed out" in f for f in context["failed"])
    assert "desktop_open" in context["failure_context"]
    assert "failed" in context["failure_context"]
    assert context["replan"] == 1
    assert context["observed_state"] is not None
    # The planner is asked for the REMAINING steps from the CURRENT state
    assert any("desktop_open" in c for c in context["completed"]) or \
        context["completed"] == []


# ═══════════════════════════════════════════════════════════════
# 6. Repeated failed strategy — loop detection prevents infinite execution
# ═══════════════════════════════════════════════════════════════

def test_repeated_failed_strategy_is_bounded(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = ScriptedExecutor([(False, "Launch timed out")] * 100)
    # The planner keeps proposing the SAME failing action
    planner = FakePlanner([PLAN] * 20)
    runner = make_runner(ex, FakeObserver([""] * 200), planner)
    state = asyncio.run(runner.run("open firefox", PLAN))

    # Bounded by replan budget and/or loop detection — never infinite
    assert state.replan_count <= 3
    assert len(ex.calls) <= 12 * (1 + 2)  # hard ceiling: steps × (1+retries)
    assert state.final_status != FinalStatus.SUCCESS
    assert state.blocker  # honest explanation present
    # Loop detection or replan exhaustion stopped the loop
    assert any("loop detected" in line or "replan" in line.lower()
               for line in state.log)


# ═══════════════════════════════════════════════════════════════
# 7. Cancellation — no retry, no replan, CANCELLED is authoritative
# ═══════════════════════════════════════════════════════════════

def test_cancellation_during_failure_no_retry_no_replan(monkeypatch):
    """Phase 15E fix: cancel requested DURING a failing step must not
    burn the retry budget and must not trigger a post-failure replan."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    runner_ref = {}
    ex = CancelDuringFailureExecutor(runner_ref)
    planner = FakePlanner([PLAN] * 5)  # would be called if replan ran
    runner = make_runner(ex, FakeObserver([""] * 10), planner)
    runner_ref["runner"] = runner

    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.final_status == FinalStatus.CANCELLED
    assert state.final_status != FinalStatus.SUCCESS
    # Exactly ONE execution — the failing attempt. No retry, no replan.
    assert len(ex.calls) == 1
    assert state.retry_count == 0
    assert state.replan_count == 0
    assert planner.calls == []
    assert "[TASK] CANCELLED" in " ".join(state.log)


def test_cancellation_after_failure_no_replan(monkeypatch):
    """Cancel after a step failed (before recovery) → CANCELLED, no replan."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = ScriptedExecutor([(False, "Launch timed out")] * 10)
    planner = FakePlanner([PLAN] * 5)
    runner = make_runner(ex, FakeObserver([""] * 10), planner)
    # Cancel right after the first (failing) execution returns
    original = runner._executor

    async def cancel_after_first(action):
        result = await original(action)
        runner.cancel()
        return result

    runner._executor = cancel_after_first
    state = asyncio.run(runner.run("open firefox", PLAN))

    assert state.final_status == FinalStatus.CANCELLED
    assert len(ex.calls) == 1          # no retry
    assert state.replan_count == 0     # no replan
    assert planner.calls == []


# ═══════════════════════════════════════════════════════════════
# 8. NEEDS_CONFIRMATION — no retry/replan bypasses the gate
# ═══════════════════════════════════════════════════════════════

def test_needs_confirmation_no_bypass(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = ScriptedExecutor([(True, "Shutting down")])
    planner = FakePlanner([PLAN] * 5)
    runner = make_runner(ex, FakeObserver([""]),
                         transcript="shut down the machine")

    async def deny(action, reason, params):
        return False

    runner._confirmation_callback = deny
    state = asyncio.run(runner.run(
        "shut down the machine",
        [{"action": "shutdown", "params": {}}]))

    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.action == "shutdown"
    assert ex.calls == []              # never executed
    assert state.replan_count == 0     # no replan bypassed the gate
    assert planner.calls == []


# ═══════════════════════════════════════════════════════════════
# 9. Final failure honesty — exhausted recovery ends FAILED, never SUCCESS
# ═══════════════════════════════════════════════════════════════

def test_exhausted_recovery_is_honestly_failed(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = ScriptedExecutor([(False, "Couldn't find ghostapp")] * 50)
    planner = FakePlanner([])  # replanning produces nothing usable
    runner = make_runner(ex, FakeObserver([""] * 50), planner)
    state = asyncio.run(runner.run("open ghostapp", PLAN))

    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE,
                                  FinalStatus.NEEDS_INPUT)
    assert state.final_status != FinalStatus.SUCCESS
    assert state.blocker
    assert state.failed_steps
    assert all(s.verified is False for s in state.failed_steps)
    assert state.completed_steps == []


# ═══════════════════════════════════════════════════════════════
# 10. Failure classification is deterministic (existing mechanism reused)
# ═══════════════════════════════════════════════════════════════

def test_failure_classification_drives_recovery():
    assert classify_failure("desktop_open", "Launch timed out", "") \
        == FailureKind.TRANSIENT
    assert classify_failure("desktop_open", "Couldn't find ghostapp", "") \
        == FailureKind.UNAVAILABLE_CAPABILITY
    assert classify_failure("desktop_open", "Couldn't open firefox", "") \
        == FailureKind.WRONG_PARAMS
    assert classify_failure("click_text", "Couldn't click Save", "") \
        == FailureKind.CHANGED_STATE