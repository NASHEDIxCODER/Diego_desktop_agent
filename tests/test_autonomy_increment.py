"""Regression tests for the autonomy increment (agent/task_state.py, agent/brain.py).

Covers:
  1. StepRecord serialization round-trip
  2. TaskExecutionState JSON round-trip
  3. Lightweight JSON persistence (data/tasks snapshots)
  4. Persistence disabled via DIEGO_TASK_PERSIST=0
  5. Single-step success attaches verification evidence
  6. In-flight cooperative cancellation via TaskStateStore
  7. Failure → retry → replan remains bounded

All tests use mocks/fakes only — no microphone, speaker, camera,
Ollama, network, real models, real browser, or real OS actions.
"""

from __future__ import annotations

import asyncio
import os

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
    TaskStateStore,
    task_state_store,
    _step_record_to_dict,
    _step_record_from_dict,
    _task_state_to_dict,
    _task_state_from_dict,
)
import agent.task_state as task_state_module


class FakeExecutor:

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
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.states:
            return self.states.pop(0)
        return ""


class FakePlanner:




    def __init__(self, plans):
        self.plans = list(plans)
        self.contexts = []

    async def __call__(self, request, context):
        self.contexts.append(dict(context))
        if self.plans:
            return self.plans.pop(0)
        return None


def make_runner(executor, observer=None, planner=None, limits=None,
                transcript="open firefoxand search github"):
    return TaskRunner(
        executor=executor,
        observer=observer,
        planner=planner,
        validator=PlanValidator(),
        limits=limits or TaskLimits(max_task_steps=12, max_retries_per_step=2,
                                    max_replans=3, max_total_execution_time=60),
        transcript=transcript,
    )


# ═══════════════════════════════════════════════════════════════
# 1. StepRecord round-trip
# ═══════════════════════════════════════════════════════════════

def test_step_record_round_trip():
    """A representative StepRecord survives the existing serializer round-trip."""
    rec = StepRecord(
        index=3,
        action="desktop_open",
        params={"app": "firefox"},
        description="Open Firefox",
        status=StepStatus.COMPLETED,
        result="Opened firefox",
        verification="success",
        verified=True,
        retries=1,
        error="",
        latency_ms=12.5,
        evidence_source=EvidenceSource.ACTION_VERIFICATION,
        sensitive_reason="",
    )
    data = _step_record_to_dict(rec)
    restored = _step_record_from_dict(data)
    assert restored.index == rec.index
    assert restored.action == rec.action
    assert restored.params == rec.params
    assert restored.description == rec.description
    assert restored.status == rec.status
    assert restored.result == rec.result
    assert restored.verification == rec.verification
    assert restored.verified is rec.verified
    assert restored.retries == rec.retries
    assert restored.error == rec.error
    assert restored.latency_ms == rec.latency_ms
    assert restored.evidence_source == rec.evidence_source
    assert restored.sensitive_reason == rec.sensitive_reason


# ═══════════════════════════════════════════════════════════════
# 2. TaskExecutionState JSON round-trip
# ═══════════════════════════════════════════════════════════════

def test_task_execution_state_json_round_trip():
    """A representative task state (steps/evidence/status) survives JSON."""
    state = TaskExecutionState(
        task_id="abc123",
        original_request="open firefoxand search github",
        normalized_goal="open firefoxand search github",
        current_plan=[{"action": "desktop_open", "params": {"app": "firefox"}}],
        completed_steps=[
            StepRecord(
                index=1,
                action="desktop_open",
                params={"app": "firefox"},
                status=StepStatus.COMPLETED,
                verified=True,
                result="Opened firefox",
                verification="success",
            ),
        ],
        current_step=None,
        failed_steps=[
            StepRecord(
                index=2,
                action="browser_navigate",
                params={"url": "https://x.com"},
                status=StepStatus.FAILED,
                verified=False,
                error="Couldn't load",
            ),
        ],
        observed_state="firefox window focused",
        verification_results=[
            {"step": 1, "action": "desktop_open", "verified": True, "evidence": "Opened firefox"},
        ],
        retry_count=1,
        replan_count=2,
        total_steps=3,
        final_status=FinalStatus.PARTIAL_FAILURE,
        plan_version=2,
        started_at=1000.0,
        ended_at=2000.0,
        artifacts={"last_app": "firefox", "urls": ["https://x.com"]},
        log=["[TASK] id=abc123 goal=\"open firefox\""],
        blocker="step failed",
        evidence_log=[
            Evidence(
                fact="firefox is open",
                source=EvidenceSource.LIVE_OBSERVATION,
                timestamp=1500.0,
                confidence=0.9,
            ),
        ],
        pending_confirmation=None,
        confirmation_reason="",
    )
    data = _task_state_to_dict(state)
    restored = _task_state_from_dict(data)
    assert restored.task_id == state.task_id
    assert restored.original_request == state.original_request
    assert restored.normalized_goal == state.normalized_goal
    assert restored.current_plan == state.current_plan
    assert len(restored.completed_steps) == 1
    assert restored.completed_steps[0].action == "desktop_open"
    assert restored.completed_steps[0].verified is True
    assert restored.current_step is None
    assert len(restored.failed_steps) == 1
    assert restored.failed_steps[0].error == "Couldn't load"
    assert restored.observed_state == state.observed_state
    assert restored.verification_results == state.verification_results
    assert restored.retry_count == state.retry_count
    assert restored.replan_count == state.replan_count
    assert restored.total_steps == state.total_steps
    assert restored.final_status == state.final_status
    assert restored.plan_version == state.plan_version
    assert restored.started_at == state.started_at
    assert restored.ended_at == state.ended_at
    assert restored.artifacts == state.artifacts
    assert restored.log == state.log
    assert restored.blocker == state.blocker
    assert len(restored.evidence_log) == 1
    assert restored.evidence_log[0].fact == "firefox is open"
    assert restored.evidence_log[0].source == EvidenceSource.LIVE_OBSERVATION
    assert restored.evidence_log[0].timestamp == 1500.0
    assert restored.evidence_log[0].confidence == 0.9
    assert restored.pending_confirmation is None
    assert restored.confirmation_reason == state.confirmation_reason


# ═══════════════════════════════════════════════════════════════
# 3. Persistence
# ═══════════════════════════════════════════════════════════════

def test_persistence_round_trip(tmp_path, monkeypatch):
    """A persisted task state loads back with the same content."""
    store = TaskStateStore()
    monkeypatch.setattr(store, "TASK_DIR", str(tmp_path))
    state = TaskExecutionState(
        task_id="persist1",
        original_request="open firefox",
        normalized_goal="open firefox",
    )
    state.final_status = FinalStatus.SUCCESS
    state.completed_steps = [
        StepRecord(
            index=1,
            action="desktop_open",
            params={"app": "firefox"},
            status=StepStatus.COMPLETED,
            verified=True,
            result="Opened firefox",
            verification="success",
        ),
    ]
    path = store.persist(state)
    assert path is not None
    assert os.path.exists(path)
    restored = store.load("persist1")
    assert restored is not None
    assert restored.task_id == "persist1"
    assert restored.original_request == "open firefox"
    assert restored.normalized_goal == "open firefox"
    assert restored.final_status == FinalStatus.SUCCESS
    assert len(restored.completed_steps) == 1
    assert restored.completed_steps[0].action == "desktop_open"
    assert restored.completed_steps[0].params == {"app": "firefox"}
    assert restored.completed_steps[0].verified is True
    assert restored.completed_steps[0].verification == "success"
    assert restored.completed_steps[0].result == "Opened firefox"


# ═══════════════════════════════════════════════════════════════
# 4. Persistence disabled
# ═══════════════════════════════════════════════════════════════

def test_persistence_disabled(tmp_path, monkeypatch):
    """DIEGO_TASK_PERSIST=0 disables snapshots (env restored afterward)."""
    store = TaskStateStore()
    monkeypatch.setattr(store, "TASK_DIR", str(tmp_path))
    monkeypatch.setenv("DIEGO_TASK_PERSIST", "0")  # restored automatically
    state = TaskExecutionState(task_id="nopersist", original_request="x")
    path = store.persist(state)
    assert path is None
    # No snapshot file was created (tmp_path itself is pre-created by pytest)
    assert not list(tmp_path.glob("*.json"))
    assert store.load("nopersist") is None


# ═══════════════════════════════════════════════════════════════
# 5. Single-step success + verification evidence
# ═══════════════════════════════════════════════════════════════

def test_single_step_success_attaches_verification_evidence(monkeypatch):
    """The smallest successful task path attaches verification evidence."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(True, "Opened firefox")])
    obs = FakeObserver(["", "firefox window focused"])
    runner = make_runner(ex, obs, transcript="open firefox")
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))
    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 1
    step = state.completed_steps[0]
    assert step.verified is True
    assert step.verification == "success"
    assert step.status == StepStatus.COMPLETED
    assert step.result == "Opened firefox"
    # Verification evidence is recorded in the task state
    assert len(state.verification_results) == 1
    assert state.verification_results[0]["verified"] is True
    assert state.verification_results[0]["action"] == "desktop_open"
    assert state.verification_results[0]["evidence"] == "Opened firefox"
    # Real artifacts are captured from the dispatch result
    assert state.artifacts.get("last_app") == "firefox"
    assert state.observed_state == "firefox window focused"
    assert "[VERIFY] success" in " ".join(state.log)
    assert "[TASK] COMPLETE" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# 6. In-flight cancellation
# ═══════════════════════════════════════════════════════════════

class BlockingExecutor:



    def __init__(self, entered, release):
        self.entered = entered
        self.release = release
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        self.entered.set()
        await self.release.wait()
        return True, "Opened firefox"


def test_in_flight_cancellation_via_store(monkeypatch):
    """A registered in-flight runner cancels cooperatively → CANCELLED."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    task_state_store.unregister_runner()  # clean slate
    entered = asyncio.Event()
    release = asyncio.Event()
    ex = BlockingExecutor(entered, release)
    obs = FakeObserver(["", "firefox running"])
    runner = make_runner(ex, obs, transcript="open firefoxand search")

    async def scenario():
        task = asyncio.create_task(runner.run(
            "open firefoxand search",
            [
                {"action": "desktop_open", "params": {"app": "firefox"}},
                {"action": "web_search", "params": {"query": "test"}},
            ],
        ))
        await entered.wait()  # step 1 is in-flight
        assert task_state_store.has_running_task() is True
        cancelled = task_state_store.cancel_active_runner()
        assert cancelled is True
        release.set()  # let step 1 finish cooperatively
        state = await task
        return state

    state = asyncio.run(scenario())
    assert state.final_status == FinalStatus.CANCELLED
    assert state.final_status != FinalStatus.SUCCESS
    assert len(state.completed_steps) == 1  # step 1 completed before cancel
    assert len(ex.calls) == 1      # step 2 never ran
    assert state.current_step is None
    assert task_state_store.has_running_task() is False  # unregistered afterward
    assert "[TASK] CANCELLED" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# 7. Failure → retry → replan bound
# ═══════════════════════════════════════════════════════════════

def test_failure_retry_replan_bounded(monkeypatch):
    """Retries (per step) and replans (global) remain bounded."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    ex = FakeExecutor([(False, "Launch timed out")] * 50)
    obs = FakeObserver([""] * 100)
    planner = FakePlanner([
        [{"action": "desktop_open", "params": {"app": "ghostapp"}}]
    ] * 20)
    limits = TaskLimits(max_task_steps=12, max_retries_per_step=2,
                        max_replans=3, max_total_execution_time=60)
    runner = make_runner(ex, obs, planner, limits=limits,
                         transcript="open ghostapp")
    state = asyncio.run(runner.run(
        "open ghostapp",
        [{"action": "desktop_open", "params": {"app": "ghostapp"}}]))
    # Retries are bounded per step
    assert all(s.retries <= limits.max_retries_per_step for s in state.failed_steps)
    # Replans are bounded globally
    assert state.replan_count <= limits.max_replans
    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)  # never SUCCESS
    assert state.blocker  # honest explanation present
    assert any("[RETRY]" in l for l in state.log)
    assert any("[REPLAN]" in l for l in state.log)
