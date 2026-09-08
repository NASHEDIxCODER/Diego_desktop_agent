"""Phase 15G regression tests — MULTI-STEP TASK CONTINUATION + STATE RECOVERY.

Verifies coherent continuation across interruption using the EXISTING
mechanisms only (TaskStateStore persist/load + TaskRunner inherited state):

  VERIFIED work is never re-executed; merely-executed work is re-verified;
  continuation resumes from authoritative state; replans preserve verified
  work; CANCELLED persists honestly; corrupt persistence fails safely;
  bounds and lifecycle events survive continuation.

Mocks/fakes + pytest tmp_path only — no OS, network, LLM, camera,
microphone, browser, or models.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from agent.task_state import (
    FinalStatus,
    PlanValidator,
    StepRecord,
    StepStatus,
    TaskExecutionState,
    TaskLimits,
    TaskRunner,
    TaskStateStore,
    task_state_store,
    _task_state_to_dict,
)
import agent.task_state as task_state_module


# ═══════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════

class ScriptedExecutor:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        if self.script:
            return self.script.pop(0)
        return False, "no scripted result"


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
        self.calls = []

    async def __call__(self, request, context):
        self.calls.append(dict(context or {}))
        if self.plans:
            return self.plans.pop(0)
        return None


def make_runner(executor, observer=None, planner=None, transcript="",
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


S1 = {"action": "desktop_open", "params": {"app": "firefox"}}
S2 = {"action": "desktop_open", "params": {"app": "code"}}
S3 = {"action": "desktop_open", "params": {"app": "nautilus"}}


def make_store(tmp_path):
    store = TaskStateStore()
    os.environ.pop("DIEGO_TASK_PERSIST", None)
    import agent.task_state as tsm
    store.TASK_DIR = str(tmp_path)
    return store


def verified_step(index, step, result):
    rec = StepRecord(index=index, action=step["action"],
                     params=dict(step["params"]), status=StepStatus.COMPLETED,
                     result=result, verification="success", verified=True)
    return rec


@pytest.fixture(autouse=True)
def patch_app_running(monkeypatch):
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)


# ═══════════════════════════════════════════════════════════════
# 1. Multi-step task: all steps verified → SUCCESS with all evidence
# ═══════════════════════════════════════════════════════════════

def test_multi_step_all_verified_success():
    ex = ScriptedExecutor([
        (True, "Opened firefox"), (True, "Opened code"), (True, "Opened files")])
    runner = make_runner(ex, FakeObserver(["", "", "", ""]))
    state = asyncio.run(runner.run(
        "open firefox then code then files", [S1, S2, S3]))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 3
    assert all(s.verified for s in state.completed_steps)
    assert len(state.verification_results) == 3
    assert all(v["verified"] for v in state.verification_results)


# ═══════════════════════════════════════════════════════════════
# 2. Completed-step continuation: verified step NOT executed again
# ═══════════════════════════════════════════════════════════════

def test_verified_step_not_reexecuted_on_resume(tmp_path):
    # First run: step 1 verified (step 2 not yet attempted)
    ex1 = ScriptedExecutor([(True, "Opened firefox")])
    first = asyncio.run(make_runner(ex1, FakeObserver(["", ""])).run(
        "open firefox and code", [S1]))
    assert first.final_status == FinalStatus.SUCCESS

    store = make_store(tmp_path)
    store.save(first)
    persisted = store.load(first.task_id)
    assert persisted is not None
    assert len(persisted.completed_steps) == 1

    # Resume with the FULL plan (planner regenerating everything) — the
    # verified step 1 must be skipped, only step 2 executes.
    ex2 = ScriptedExecutor([(True, "Opened code")])
    resumed = asyncio.run(make_runner(ex2, FakeObserver(["", ""])).run(
        "open firefox and code", [S1, S2], inherited=persisted))

    assert resumed.final_status == FinalStatus.SUCCESS
    assert len(ex2.calls) == 1                    # only step 2 ran
    assert ex2.calls[0]["action"] == "desktop_open"
    assert ex2.calls[0]["params"] == {"app": "code"}
    assert len(resumed.completed_steps) == 2      # inherited + new


# ═══════════════════════════════════════════════════════════════
# 3. Unverified step: NOT trusted as complete — executes again
# ═══════════════════════════════════════════════════════════════

def test_unverified_step_not_trusted_as_complete(tmp_path):
    # Persisted state where step 1 EXECUTED but was never verified.
    state = TaskExecutionState(
        task_id="unv1",
        original_request="open firefox and code",
        normalized_goal="open firefox and code",
        current_plan=[S1, S2],
        failed_steps=[StepRecord(
            index=1, action="desktop_open", params=dict(S1["params"]),
            status=StepStatus.FAILED, verified=False,
            result="executed but unverified", error="unverified")],
        final_status=FinalStatus.PARTIAL_FAILURE,
    )
    # Remaining-work calculation: unverified step is NOT complete → remains
    store = make_store(tmp_path)
    remaining = store._remaining_steps(state)
    assert remaining == [S1] or (len(remaining) == 2 and S1 in remaining)

    # Resume: step 1 must execute/verify again — never trusted.
    ex = ScriptedExecutor([(True, "Opened firefox"), (True, "Opened code")])
    resumed = asyncio.run(make_runner(ex, FakeObserver(["", "", ""])).run(
        "open firefox and code", [S1, S2], inherited=state))

    assert resumed.final_status == FinalStatus.SUCCESS
    # step 1 WAS executed again (not trusted as complete)
    assert any(c["params"] == {"app": "firefox"} for c in ex.calls)
    assert all(s.verified for s in resumed.completed_steps)


# ═══════════════════════════════════════════════════════════════
# 4. Partial continuation: only remaining work executes
# ═══════════════════════════════════════════════════════════════

def test_partial_continuation_executes_only_remaining(tmp_path):
    # First run completes steps 1-2 (step 3 not yet attempted)
    first = asyncio.run(make_runner(ScriptedExecutor(
        [(True, "Opened firefox"), (True, "Opened code")]),
        FakeObserver(["", "", ""])).run(
        "open firefox and code and files", [S1, S2]))
    assert first.final_status == FinalStatus.SUCCESS
    assert len(first.completed_steps) == 2
    # The FULL task plan includes step 3 (not yet attempted) — record it in
    # the plan before persisting, as the continuation expects.
    first.current_plan = [S1, S2, S3]

    store = make_store(tmp_path)
    store.save(first)
    persisted = store.load(first.task_id)
    remaining = store._remaining_steps(persisted)
    assert remaining == [S3]

    ex = ScriptedExecutor([(True, "Opened files")])
    resumed = asyncio.run(make_runner(ex, FakeObserver(["", ""])).run(
        "open firefox and code and files", remaining, inherited=persisted))

    assert resumed.final_status == FinalStatus.SUCCESS
    assert len(ex.calls) == 1                     # only step 3 executed
    assert ex.calls[0]["params"] == {"app": "nautilus"}
    assert len(resumed.completed_steps) == 3


# ═══════════════════════════════════════════════════════════════
# 5. Replan after partial completion: planner sees verified work
# ═══════════════════════════════════════════════════════════════

def test_replan_preserves_verified_work(tmp_path):
    first = asyncio.run(make_runner(ScriptedExecutor(
        [(True, "Opened firefox")]), FakeObserver(["", ""])).run(
        "open firefox and files", [S1]))
    assert first.final_status == FinalStatus.SUCCESS
    store = make_store(tmp_path)
    store.save(first)
    persisted = store.load(first.task_id)

    # Continuation: step 2 fails persistently → replan. The planner must
    # see the verified step 1 and replan only remaining work.
    ex = ScriptedExecutor([(False, "Couldn't find files"),
                           (True, "Opened files")])
    planner = FakePlanner([[S3]])  # alternative strategy for remaining work
    resumed = asyncio.run(make_runner(
        ex, FakeObserver([""] * 30), planner,
        transcript="open firefox and files").run(
        "open firefox and files", [S2], inherited=persisted))

    assert len(planner.calls) >= 1
    context = planner.calls[0]
    # Verified step 1 is in the replan context — preserved work
    assert any("firefox" in c for c in context["completed"])
    # The failed step + reason are in the context
    assert any("desktop_open" in f and "Couldn't find files" in f
               for f in context["failed"])
    assert context["failure_context"]
    # Final: the alternative step 3 executed and the task succeeded
    assert resumed.final_status == FinalStatus.SUCCESS
    assert any(c["params"] == {"app": "nautilus"} for c in ex.calls)


# ═══════════════════════════════════════════════════════════════
# 6. All work already verified → SUCCESS without side effects
# ═══════════════════════════════════════════════════════════════

def test_already_verified_task_finishes_success(tmp_path):
    first = asyncio.run(make_runner(ScriptedExecutor(
        [(True, "Opened firefox"), (True, "Opened code")]),
        FakeObserver(["", "", ""])).run(
        "open firefox and code", [S1, S2]))
    assert first.final_status == FinalStatus.SUCCESS
    store = make_store(tmp_path)
    store.save(first)
    persisted = store.load(first.task_id)
    assert persisted.final_status == FinalStatus.SUCCESS
    assert store._remaining_steps(persisted) == []   # nothing left

    # Resume against the FULL plan — everything is already verified →
    # honest SUCCESS with ZERO side effects.
    ex = ScriptedExecutor([])
    resumed = asyncio.run(make_runner(ex, FakeObserver([""])).run(
        "open firefox and code", [S1, S2], inherited=persisted))

    assert resumed.final_status == FinalStatus.SUCCESS
    assert ex.calls == []                          # nothing re-executed
    assert len(resumed.completed_steps) == 2


# ═══════════════════════════════════════════════════════════════
# 7. Cancellation: persisted CANCELLED never auto-resumes as SUCCESS
# ═══════════════════════════════════════════════════════════════

def test_cancelled_state_not_resumed(tmp_path):
    first = asyncio.run(make_runner(ScriptedExecutor(
        [(True, "Opened firefox")]), FakeObserver(["", ""])).run(
        "open firefox and code", [S1, S2]))
    # Simulate a user cancel of the active task
    store = make_store(tmp_path)
    store.save(first)
    assert store.active is first
    assert store.cancel_active() is True
    assert first.final_status == FinalStatus.CANCELLED
    # Persist and reload: CANCELLED is retained honestly
    store.save(first)
    persisted = store.load(first.task_id)
    assert persisted.final_status == FinalStatus.CANCELLED
    # No automatic continuation: "continue" returns None (active cleared)
    assert store.build_continuation("continue") is None
    # And even a direct inherited resume does NOT execute or fake SUCCESS
    ex = ScriptedExecutor([])
    resumed = asyncio.run(make_runner(ex, FakeObserver([""])).run(
        "open firefox and code", [S2], inherited=persisted))
    assert resumed.final_status == FinalStatus.CANCELLED
    assert ex.calls == []


# ═══════════════════════════════════════════════════════════════
# 8. NEEDS_CONFIRMATION stays blocked until explicitly confirmed
# ═══════════════════════════════════════════════════════════════

def test_needs_confirmation_persisted_still_blocked(tmp_path):
    ex1 = ScriptedExecutor([(True, "Shutting down")])
    runner1 = make_runner(ex1, FakeObserver([""]),
                          transcript="shut down the machine")

    async def deny(action, reason, params):
        return False

    runner1._confirmation_callback = deny
    first = asyncio.run(runner1.run(
        "shut down the machine",
        [{"action": "shutdown", "params": {}}]))
    assert first.final_status == FinalStatus.NEEDS_CONFIRMATION
    store = make_store(tmp_path)
    store.save(first)
    persisted = store.load(first.task_id)
    assert persisted.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert persisted.pending_confirmation is not None

    # Resume WITHOUT explicit confirmation → blocked again, never executed
    ex2 = ScriptedExecutor([])
    runner2 = make_runner(ex2, FakeObserver([""]),
                          transcript="shut down the machine")
    runner2._confirmation_callback = deny
    resumed = asyncio.run(runner2.run(
        "shut down the machine",
        [{"action": "shutdown", "params": {}}], inherited=persisted))
    assert resumed.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert ex2.calls == []            # no sensitive action executed


# ═══════════════════════════════════════════════════════════════
# 9. Corrupt/malformed persistence fails safely — never fabricated SUCCESS
# ═══════════════════════════════════════════════════════════════

def test_corrupt_persistence_fails_safely(tmp_path):
    store = make_store(tmp_path)
    # Garbage JSON
    bad = os.path.join(str(tmp_path), "task_corrupt_1.json")
    with open(bad, "w") as fh:
        fh.write("{not json at all")
    assert store.load("corrupt") is None
    # Valid JSON, malformed status
    ok_json = os.path.join(str(tmp_path), "task_corrupt_2.json")
    with open(ok_n := ok_json, "w") as fh:
        fh.write('{"task_id": "corrupt", "final_status": "NOT_A_STATUS"}')
    assert store.load("corrupt") is None
    # A valid snapshot that gets truncated after persist
    good = asyncio.run(make_runner(ScriptedExecutor(
        [(True, "Opened firefox")]), FakeObserver(["", ""])).run(
        "open firefox", [S1]))
    path = store.persist(good)
    assert path
    with open(path, "w") as fh:
        fh.write('{"task_id": "trunc", "completed_ste')
    assert store.load(good.task_id) is None
    # Corrupt load yields None — never a fabricated SUCCESS
    ex = ScriptedExecutor([(True, "Opened code")])
    # And a run WITHOUT inherited state proceeds normally (no fabrication)
    resumed = asyncio.run(make_runner(ex, FakeObserver(["", ""])).run(
        "open code", [S2]))
    assert resumed.final_status == FinalStatus.SUCCESS
    assert len(ex.calls) == 1


# ═══════════════════════════════════════════════════════════════
# 10. Retry/replan bounds remain authoritative after continuation
# ═══════════════════════════════════════════════════════════════

def test_bounds_survive_continuation(tmp_path):
    first = asyncio.run(make_runner(ScriptedExecutor(
        [(True, "Opened firefox")]), FakeObserver(["", ""])).run(
        "open firefox and code", [S1]))
    store = make_store(tmp_path)
    store.save(first)
    persisted = store.load(first.task_id)

    limits = TaskLimits(max_task_steps=12, max_retries_per_step=2,
                        max_replans=2, max_total_execution_time=60)
    ex = ScriptedExecutor([(False, "Launch timed out")] * 50)
    planner = FakePlanner([[S2]] * 10)
    runner = TaskRunner(
        executor=ex, observer=FakeObserver([""] * 100), planner=planner,
        validator=PlanValidator(), limits=limits, transcript="open code")
    resumed = asyncio.run(runner.run(
        "open firefox and code", [S2], inherited=persisted))

    # The retry bound is PER STEP (each failed attempt record ≤ max);
    # the replan bound is global for the resumed run.
    assert all(s.retries <= limits.max_retries_per_step
               for s in resumed.failed_steps)
    assert resumed.replan_count <= limits.max_replans
    assert resumed.final_status != FinalStatus.SUCCESS
    # Hard ceiling: each replanned step gets at most 1 + max_retries calls
    assert len(ex.calls) <= (1 + limits.max_retries_per_step) \
        * (limits.max_replans + 1)


# ═══════════════════════════════════════════════════════════════
# 11. Lifecycle events during continuation remain truthful
# ═══════════════════════════════════════════════════════════════

def test_lifecycle_events_truthful_during_continuation(tmp_path, monkeypatch):
    from core.event_bus import EventBus
    events = []

    class RecorderBus:
        async def emit(self, event_type, data=None, source=None):
            events.append((event_type, dict(data or {})))

    monkeypatch.setattr("core.event_bus.bus", RecorderBus(), raising=False)
    monkeypatch.setattr(task_state_module, "app_running", lambda a: False)

    first = asyncio.run(make_runner(ScriptedExecutor(
        [(True, "Opened firefox")]), FakeObserver(["", ""])).run(
        "open firefox and code", [S1]))
    assert first.final_status == FinalStatus.SUCCESS
    events.clear()  # observe only the continuation

    store = make_store(tmp_path)
    store.save(first)
    persisted = store.load(first.task_id)
    ex = ScriptedExecutor([(True, "Opened code")])
    resumed = asyncio.run(make_runner(ex, FakeObserver(["", ""])).run(
        "open firefox and code", [S2], inherited=persisted))
    assert resumed.final_status == FinalStatus.SUCCESS

    types = [t for t, _ in events]
    started = [d for t, d in events if t == "task.started"]
    assert len(started) == 1
    # Truthful: the started event reflects the ALREADY verified work
    assert started[0]["completed_steps"] == 1
    assert "task.completed" in types
    completed = [d for t, d in events if t == "task.completed"]
    assert completed[0]["status"] == "SUCCESS"
    assert completed[0]["completed_steps"] == 2
    # step.completed emitted only for the actually executed step
    step_completed = [d for t, d in events if t == "task.step.completed"]
    assert len(step_completed) == 1
    assert step_completed[0]["step"] == 1  # step index within the resumed run
    assert step_completed[0]["action"] == "desktop_open"
    # No task.failed — continuation succeeded honestly
    assert "task.failed" not in types