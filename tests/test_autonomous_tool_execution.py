"""Phase 15B regression tests — GENERAL AUTONOMOUS TOOL EXECUTION.

Verifies that a registered general tool (core/tool_registry.py) can travel
the EXISTING autonomous path end-to-end:

    PLAN
    → PlanValidator (agent/task_state.py — accepts registry tools)
    → TaskRunner (agent/task_state.py — the authoritative loop)
    → executor = Brain._dispatch_and_verify contract
    → ActionDispatcher.execute (agent/action_dispatcher.py — registry consult)
    → ToolRegistry tool execution (core/tool_registry.py)
    → honest result conversion ("Couldn't ..." failure marker)
    → verification (AgentBrain._verify fail-fast / _trust_dispatch_result)
    → observation / verification evidence / state update

All tests use mocks/fakes only — no microphone, speaker, camera, Ollama,
network, real models, real browser, or real OS actions. The Tool/ToolResult
machinery exercised is the REAL core/tool_registry.py code; only tool
HANDLERS are fakes.
"""

from __future__ import annotations

import asyncio

import pytest

import core.tool_registry as tool_registry_module
from core.tool_registry import Tool, ToolResult
from agent.action_dispatcher import ActionDispatcher
from agent.brain import AgentBrain
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

class FakeRegistry:
    """Fake ToolRegistry singleton (patched over core.tool_registry.tool_registry).

    Uses the REAL Tool.execute_async machinery; only tool handlers are fakes.
    """

    def __init__(self):
        self.tools = {}
        self.install_calls = 0
        self.execute_records = []   # (name, params) of every registry execution

    def install_builtin_tools(self):
        self.install_calls += 1  # idempotent no-op for the fake

    def is_available(self, name):
        return name in self.tools

    def get(self, name):
        return self.tools.get(name)

    def register(self, tool):
        self.tools[tool.name] = tool

    async def execute(self, name, params):
        self.execute_records.append((name, dict(params)))
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return await tool.execute_async(params)


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
        self.contexts.append(dict(context or {}))
        if self.plans:
            return self.plans.pop(0)
        return None


class BrainLikeExecutor:
    """Executor mirroring the REAL Brain._dispatch_and_verify contract:

    dispatch through the REAL ActionDispatcher, then verify with the REAL
    Brain verification function that registry tools fall through to
    (AgentBrain._trust_dispatch_result — fails on "Couldn't ..." failure
    strings). No verification logic is re-implemented here.
    """

    def __init__(self, dispatcher, verification_override=None):
        self.dispatcher = dispatcher
        self.verification_override = verification_override
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        try:
            result = await self.dispatcher.execute(action)
        except Exception as e:
            return False, f"dispatch error: {e}"
        verified = AgentBrain._trust_dispatch_result(result)
        if self.verification_override is not None:
            verified = self.verification_override
        return bool(verified), result or ""


class BlockingExecutor:

    def __init__(self, entered, release):
        self.entered = entered
        self.release = release
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        self.entered.set()
        await self.release.wait()
        return True, "echo ran"


def make_runner(executor, observer=None, planner=None, transcript="run the echo tool",
                confirmation_callback=None):
    return TaskRunner(
        executor=executor,
        observer=observer,
        planner=planner,
        validator=PlanValidator(),
        limits=TaskLimits(max_task_steps=8, max_retries_per_step=2,
                          max_replans=2, max_total_execution_time=60),
        transcript=transcript,
        confirmation_callback=confirmation_callback,
    )


@pytest.fixture
def fake_registry(monkeypatch):
    """Patch the global tool_registry singleton with a FakeRegistry."""
    registry = FakeRegistry()
    monkeypatch.setattr(tool_registry_module, "tool_registry", registry)
    return registry


def register_echo_tool(registry, result):
    """Register a representative fake general tool through the REAL Tool class."""
    def handler(params):
        return result

    tool = Tool(name="echo_tool", description="fake general tool",
                handler=handler)
    registry.register(tool)
    return tool


# ═══════════════════════════════════════════════════════════════
# A. A registered tool travels the full autonomous path
# ═══════════════════════════════════════════════════════════════

def test_registered_tool_travels_full_autonomous_path(fake_registry, monkeypatch):
    """PLAN → validate → TaskRunner → dispatcher → ToolRegistry → verify → SUCCESS."""
    register_echo_tool(
        fake_registry,
        ToolResult(success=True, output="echo ran: hello"))

    dispatcher = ActionDispatcher()
    executor = BrainLikeExecutor(dispatcher)
    observer = FakeObserver(["", "echo output visible"])
    runner = make_runner(executor, observer, transcript="run the echo tool")

    state = asyncio.run(runner.run(
        "run the echo tool",
        [{"action": "echo_tool", "params": {"message": "hello"},
          "description": "Run the fake echo tool"}]))

    # Plan validation accepted the registry tool (step was NOT rejected)
    assert "hallucinated/unknown action" not in " ".join(state.log)
    assert "REJECTED" not in " ".join(state.log)
    # The step was selected and dispatched through the executor exactly once
    assert len(executor.calls) == 1
    assert executor.calls[0]["action"] == "echo_tool"
    assert executor.calls[0]["params"] == {"message": "hello"}
    # The registry actually executed the tool with the planned params
    assert fake_registry.execute_records == [
        ("echo_tool", {"message": "hello"})]
    # The dispatcher converted the ToolResult honestly
    step = state.completed_steps[0]
    assert step.result == "echo ran: hello"
    assert step.status == StepStatus.COMPLETED
    # Task completes with verified success
    assert state.final_status == FinalStatus.SUCCESS
    assert "[VERIFY] success" in " ".join(state.log)
    assert "[TASK] COMPLETE" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# B. Successful execution + successful verification → evidence
# ═══════════════════════════════════════════════════════════════

def test_successful_verification_attaches_evidence(fake_registry, monkeypatch):
    """A verified registry-tool step records verification evidence in state."""
    register_echo_tool(
        fake_registry,
        ToolResult(success=True, output="echo ran: hello"))

    dispatcher = ActionDispatcher()
    executor = BrainLikeExecutor(dispatcher)
    observer = FakeObserver(["", ""])
    runner = make_runner(executor, observer)

    state = asyncio.run(runner.run(
        "run the echo tool",
        [{"action": "echo_tool", "params": {"message": "hello"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert len(state.completed_steps) == 1
    step = state.completed_steps[0]
    assert step.verified is True
    assert step.verification == "success"
    # Verification evidence is attached to the task state
    assert len(state.verification_results) == 1
    vr = state.verification_results[0]
    assert vr["verified"] is True
    assert vr["action"] == "echo_tool"
    assert vr["evidence"] == "echo ran: hello"
    assert "[VERIFY] success" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# C. Successful execution + FAILED verification ≠ SUCCESS
# ═══════════════════════════════════════════════════════════════

def test_verification_failure_is_never_success(fake_registry):
    """Two honest ways execution can succeed while verification fails:

    1. The tool itself reports failure (ToolResult.success=False) — the
       dispatcher converts it to a "Couldn't ..." failure string and the
       real Brain verification fails fast on it.
    2. The dispatch returned cleanly but the verification stage says NO —
       the runner must still refuse SUCCESS.
    """
    # ── C1: tool reports failure through the real dispatcher conversion ──
    register_echo_tool(
        fake_registry,
        ToolResult(success=False, error="command exited 1"))

    dispatcher = ActionDispatcher()
    executor = BrainLikeExecutor(dispatcher)
    observer = FakeObserver(["", ""])
    runner = make_runner(executor, observer)
    state = asyncio.run(runner.run(
        "run the echo tool",
        [{"action": "echo_tool", "params": {"message": "hello"}}]))

    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)
    assert state.final_status != FinalStatus.SUCCESS
    assert len(state.completed_steps) == 0
    assert len(state.failed_steps) == 1
    assert state.failed_steps[0].verified is False
    # The dispatcher's honest failure marker made verification fail
    assert "Couldn't echo tool" in state.failed_steps[0].result
    vr = state.verification_results[-1]
    assert vr["verified"] is False

    # ── C2: clean execution, verification overridden to FAIL ──
    register_echo_tool(
        fake_registry,
        ToolResult(success=True, output="echo ran: hello"))
    executor2 = BrainLikeExecutor(ActionDispatcher(),
                                  verification_override=False)
    runner2 = make_runner(executor2, FakeObserver(["", ""]))
    state2 = asyncio.run(runner2.run(
        "run the echo tool",
        [{"action": "echo_tool", "params": {"message": "hello"}}]))

    assert state2.final_status != FinalStatus.SUCCESS
    assert len(state2.completed_steps) == 0
    assert all(vr["verified"] is False for vr in state2.verification_results)


# ═══════════════════════════════════════════════════════════════
# D. Tool/dispatch failure follows the existing failure/retry path
# ═══════════════════════════════════════════════════════════════

def test_tool_failure_follows_retry_path(fake_registry):
    """A failing tool is retried within the existing bounded retry budget
    and finally reported honestly (never SUCCESS)."""
    register_echo_tool(
        fake_registry,
        ToolResult(success=False, error="echo timed out"))

    dispatcher = ActionDispatcher()
    executor = BrainLikeExecutor(dispatcher)
    observer = FakeObserver([""] * 20)
    planner = FakePlanner([])  # re-planning produces nothing usable
    limits = TaskLimits(max_task_steps=8, max_retries_per_step=2,
                        max_replans=2, max_total_execution_time=60)
    runner = TaskRunner(
        executor=executor, observer=observer, planner=planner,
        validator=PlanValidator(), limits=limits,
        transcript="run the echo tool")

    state = asyncio.run(runner.run(
        "run the echo tool",
        [{"action": "echo_tool", "params": {"message": "hello"}}]))

    # Retries are bounded per step (existing retry path)
    assert 0 < len(executor.calls) <= 1 + limits.max_retries_per_step
    assert all(s.retries <= limits.max_retries_per_step
               for s in state.failed_steps)
    assert any("[RETRY]" in line for line in state.log)
    assert state.replan_count <= limits.max_replans
    # Honest final state — never SUCCESS
    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)
    assert state.final_status != FinalStatus.SUCCESS
    assert state.blocker


# ═══════════════════════════════════════════════════════════════
# E. Unknown/unregistered tool fails honestly — no silent execution
# ═══════════════════════════════════════════════════════════════

def test_unknown_tool_fails_honestly(fake_registry):
    """An unregistered tool name is rejected at plan validation and never
    silently executes arbitrary behavior."""
    # Even a "make_coffee" tool handler is registered nowhere; register a
    # sentinel to prove NO tool executes for an unknown name.
    sentinel_calls = []

    def sentinel_handler(params):
        sentinel_calls.append(dict(params))
        return ToolResult(success=True, output="coffee made")

    fake_registry.register(Tool(name="completely_different_tool",
                                description="sentinel", handler=sentinel_handler))

    # 1. Plan validation rejects the unknown action up front
    validator = PlanValidator()
    ok, reason = validator.validate_step(
        {"action": "make_coffee", "params": {}}, [], transcript="make coffee")
    assert ok is False
    assert "hallucinated/unknown action 'make_coffee'" in reason

    # 2. TaskRunner honestly reports the rejected plan; nothing executed
    dispatcher = ActionDispatcher()
    executor = BrainLikeExecutor(dispatcher)
    runner = make_runner(executor, FakeObserver([""]))
    state = asyncio.run(runner.run(
        "make coffee",
        [{"action": "make_coffee", "params": {}}]))
    assert state.final_status == FinalStatus.FAILED
    assert "hallucinated/unknown action" in state.blocker
    assert len(executor.calls) == 0
    assert fake_registry.execute_records == []
    assert sentinel_calls == []

    # 3. Defense in depth: even if a plan bypassed validation, the
    #    dispatcher/verification path fails honestly for an unknown tool.
    result = asyncio.run(dispatcher.execute(
        {"action": "make_coffee", "params": {}}))
    assert result is not None and "Couldn't" in result
    ok2, msg = asyncio.run(BrainLikeExecutor(dispatcher)(
        {"action": "make_coffee", "params": {}}))
    assert ok2 is False
    assert "Couldn't" in msg
    assert sentinel_calls == []


# ═══════════════════════════════════════════════════════════════
# F. Cooperative cancellation remains intact
# ═══════════════════════════════════════════════════════════════

def test_cancellation_remains_intact(fake_registry, monkeypatch):
    """A running autonomous task with registry-tool steps cancels
    cooperatively → authoritative CANCELLED (never SUCCESS)."""
    register_echo_tool(
        fake_registry,
        ToolResult(success=True, output="echo ran"))

    task_state_store.unregister_runner()  # clean slate
    entered = asyncio.Event()
    release = asyncio.Event()
    ex = BlockingExecutor(entered, release)
    observer = FakeObserver(["", ""])
    runner = make_runner(ex, observer, transcript="run the echo tool then again")

    async def scenario():
        task = asyncio.create_task(runner.run(
            "run the echo tool then again",
            [
                {"action": "echo_tool", "params": {"message": "a"}},
                {"action": "echo_tool", "params": {"message": "b"}},
            ],
        ))
        await entered.wait()  # step 1 is in-flight
        assert task_state_store.has_running_task() is True
        assert task_state_store.cancel_active_runner() is True
        release.set()  # step 1 finishes cooperatively
        return await task

    state = asyncio.run(scenario())
    assert state.final_status == FinalStatus.CANCELLED
    assert state.final_status != FinalStatus.SUCCESS
    assert len(ex.calls) == 1  # step 2 never ran
    assert "[TASK] CANCELLED" in " ".join(state.log)
    assert task_state_store.has_running_task() is False


# ═══════════════════════════════════════════════════════════════
# G. Safety preserved: sensitive registry-tool commands still gated
# ═══════════════════════════════════════════════════════════════

def test_sensitive_registry_tool_requires_confirmation(fake_registry):
    """The existing human-safety gate applies to general tools: a
    destructive command routed through a registry tool pauses for
    explicit confirmation instead of executing."""
    # A representative "terminal" general tool must exist for plan
    # validation; its handler must NEVER run because the sensitive
    # command is denied before execution.
    def terminal_handler(params):
        raise AssertionError("sensitive tool must never execute")

    fake_registry.register(Tool(name="terminal",
                                description="fake terminal tool",
                                handler=terminal_handler))

    async def deny(action, reason, params):
        return False

    dispatcher = ActionDispatcher()
    executor = BrainLikeExecutor(dispatcher)
    runner = make_runner(executor, FakeObserver([""]),
                         transcript="clean the disk",
                         confirmation_callback=deny)
    state = asyncio.run(runner.run(
        "clean the disk",
        [{"action": "terminal", "params": {"command": "rm -rf /tmp/x"}}]))

    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.action == "terminal"
    assert len(executor.calls) == 0          # nothing executed
    assert fake_registry.execute_records == []  # registry never invoked

