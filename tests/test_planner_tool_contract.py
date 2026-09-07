"""Phase 15C regression tests — PLANNER / VALIDATOR / DISPATCHER CONTRACT.

Verifies that the planner's ADVERTISED actions are exactly what the
existing autonomous loop can validate and dispatch:

    ToolRegistry (core/tool_registry.py)  ← canonical source
    → planner prompt section (agent/planner.py: registry_tools_prompt_section)
    → PlanValidator (agent/task_state.py)
    → ActionDispatcher registry path (agent/action_dispatcher.py)
    → Brain planner-action gate (agent/brain.py: _PLANNER_ACTION_SCHEMA)

No LLM and no network is tested (a fake LLM client is used where the
planner's prompt composition is exercised). No real tool handler runs —
real tools are only used for pure validation/structure checks, and the
one dispatch-path failure test uses `terminal` with an EMPTY command,
which returns an honest error WITHOUT spawning any shell.
"""

from __future__ import annotations

import asyncio
import re

import pytest

import core.tool_registry as tool_registry_module
from core.tool_registry import Tool, ToolResult
from agent.action_dispatcher import ActionDispatcher
from agent.brain import AgentBrain
from agent.planner import (
    AgentPlanner,
    PLANNER_SYSTEM_PROMPT,
    registry_tools_prompt_section,
)
from agent.task_state import (
    FinalStatus,
    KNOWN_ACTIONS,
    PlanValidator,
    TaskLimits,
    TaskRunner,
    is_sensitive_action,
    registry_tool_available,
)
import agent.task_state as task_state_module


# ═══════════════════════════════════════════════════════════════
# Helpers / fakes
# ═══════════════════════════════════════════════════════════════

_LINE_RE = re.compile(r"^- ([a-z_]+)\(")


def advertised_registry_tools():
    """(name, params_dict, description) for every registry tool the
    planner prompt advertises — derived from the CANONICAL registry."""
    section = registry_tools_prompt_section()
    assert section, "planner must advertise registry tools"
    names = [_LINE_RE.match(line).group(1)
             for line in section.splitlines()
             if _LINE_RE.match(line)]
    from core.tool_registry import tool_registry as real
    by_name = {t.name: t for t in real.all()}
    out = []
    for name in names:
        tool = by_name[name]
        sample = {p: "x" for p in (tool.parameters or {})}
        out.append((name, sample, tool.description))
    return section, out


class FakeRegistry:
    """Fake ToolRegistry singleton mirroring the REAL registry's
    names/parameters, with fake handlers (no OS effects)."""

    def __init__(self):
        self.tools = {}

    def install_builtin_tools(self):
        pass  # fake: pre-populated by the test

    def is_available(self, name):
        return name in self.tools

    def get(self, name):
        return self.tools.get(name)

    def register(self, tool):
        self.tools[tool.name] = tool

    async def execute(self, name, params):
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return await tool.execute_async(params)


class FakeLLM:
    """Fake LLM client returning a canned JSON plan (no network/LLM)."""

    def __init__(self, response):
        self.response = response
        self.last_prompt = ""

    async def chat(self, prompt):
        self.last_prompt = prompt
        return self.response


class BrainLikeExecutor:
    """Executor mirroring the REAL Brain._dispatch_and_verify contract
    (real dispatcher + real Brain verification fallback function)."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        try:
            result = await self.dispatcher.execute(action)
        except Exception as e:
            return False, f"dispatch error: {e}"
        return bool(AgentBrain._trust_dispatch_result(result)), result or ""


class FakeObserver:

    def __init__(self, states):
        self.states = list(states)

    async def __call__(self):
        if self.states:
            return self.states.pop(0)
        return ""


def make_runner(executor, observer=None, transcript="", **kwargs):
    return TaskRunner(
        executor=executor,
        observer=observer,
        validator=PlanValidator(),
        limits=TaskLimits(max_task_steps=8, max_retries_per_step=2,
                          max_replans=2, max_total_execution_time=60),
        transcript=transcript,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════
# A. Every planner-advertised registry tool is accepted by PlanValidator
# ═══════════════════════════════════════════════════════════════

def test_every_advertised_registry_tool_passes_validation():
    """The planner only advertises actions the validator accepts."""
    _section, tools = advertised_registry_tools()
    assert tools, "planner must advertise at least one registry tool"
    validator = PlanValidator()
    for name, sample_params, _desc in tools:
        ok, reason = validator.validate_step(
            {"action": name, "params": sample_params}, [], "")
        assert ok is True, f"planner-advertised tool '{name}' rejected: {reason}"
        # Also accepted with the exact declared parameter names
        ok2, reason2 = validator.validate_step(
            {"action": name, "params": dict(sample_params)}, [], "")
        assert ok2 is True, reason2


# ═══════════════════════════════════════════════════════════════
# B. Every advertised registry tool reaches ActionDispatcher cleanly
# ═══════════════════════════════════════════════════════════════

def test_every_advertised_registry_tool_reaches_dispatcher(monkeypatch):
    """Same canonical name + parameter names flow dispatcher → registry
    with no schema conversion failure (fake handlers, no OS effects)."""
    _section, tools = advertised_registry_tools()
    registry = FakeRegistry()
    outputs = {}
    for name, sample_params, _desc in tools:
        outputs[name] = f"{name} ok"

        def make_handler(expected):
            def handler(params):
                return ToolResult(success=True, output=expected)
            return handler

        registry.register(Tool(name=name, description="fake mirror",
                               handler=make_handler(outputs[name])))
    monkeypatch.setattr(tool_registry_module, "tool_registry", registry)

    dispatcher = ActionDispatcher()
    for name, sample_params, _desc in tools:
        result = asyncio.run(dispatcher.execute(
            {"action": name, "params": dict(sample_params)}))
        assert result == outputs[name], (
            f"dispatcher could not execute advertised tool '{name}': {result}")
        # The real Brain verification fallback accepts the honest result
        assert AgentBrain._trust_dispatch_result(result) is True
        ok, msg = asyncio.run(BrainLikeExecutor(dispatcher)(
            {"action": name, "params": dict(sample_params)}))
        assert ok is True and msg == outputs[name]


# ═══════════════════════════════════════════════════════════════
# C. Unknown actions are still rejected (guard preserved)
# ═══════════════════════════════════════════════════════════════

def test_unknown_action_still_rejected():
    validator = PlanValidator()
    ok, reason = validator.validate_step(
        {"action": "make_coffee", "params": {}}, [], "")
    assert ok is False
    assert "hallucinated/unknown action 'make_coffee'" in reason


# ═══════════════════════════════════════════════════════════════
# D. Registry tool with missing parameters fails honestly
# ═══════════════════════════════════════════════════════════════

def test_registry_tool_missing_params_fails_honestly():
    """`terminal` without a command: no PARAM_REQUIREMENTS entry exists,
    so the step passes validation and fails through the EXISTING honest
    dispatch/verification path (the real terminal tool returns an error
    for an empty command WITHOUT spawning any shell)."""
    dispatcher = ActionDispatcher()
    executor = BrainLikeExecutor(dispatcher)
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="run a shell command")
    state = asyncio.run(runner.run(
        "run a shell command",
        [{"action": "terminal", "params": {}}]))

    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)
    assert state.final_status != FinalStatus.SUCCESS
    assert len(state.completed_steps) == 0
    failed = state.failed_steps[0]
    assert failed.verified is False
    assert "Couldn't terminal" in failed.result
    assert "No command provided" in failed.result


# ═══════════════════════════════════════════════════════════════
# E. Existing KNOWN_ACTIONS remain accepted
# ═══════════════════════════════════════════════════════════════

def test_known_actions_remain_accepted():
    validator = PlanValidator()
    for step in (
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "browser_navigate", "params": {"url": "https://x.com"}},
        {"action": "get_time", "params": {}},
        {"action": "volume_set", "params": {"percent": 40}},
    ):
        ok, reason = validator.validate_step(step, [], "")
        assert ok is True, f"{step['action']} rejected: {reason}"
    # And the static planner prompt still advertises them
    assert "desktop_open(app)" in PLANNER_SYSTEM_PROMPT
    assert "browser_navigate(url)" in PLANNER_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════
# F. Destructive/sensitive registry actions keep the confirmation gate
# ═══════════════════════════════════════════════════════════════

def test_sensitive_registry_action_requires_confirmation(monkeypatch):
    # The existing sensitive-action detector flags the destructive command
    sensitive, reason = is_sensitive_action(
        "terminal", {"command": "rm -rf /tmp/x"})
    assert sensitive is True and reason

    registry = FakeRegistry()

    def terminal_handler(params):
        raise AssertionError("sensitive tool must never execute")

    registry.register(Tool(name="terminal", description="fake terminal",
                           handler=terminal_handler))
    monkeypatch.setattr(tool_registry_module, "tool_registry", registry)

    async def deny(action, reason_, params):
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
    assert executor.calls == []  # nothing executed


# ═══════════════════════════════════════════════════════════════
# G. No duplicate/conflicting planner tool definitions
# ═══════════════════════════════════════════════════════════════

def test_no_duplicate_or_conflicting_definitions():
    section, tools = advertised_registry_tools()
    names = [name for name, _p, _d in tools]

    # 1. No duplicates within the advertised registry section
    assert len(names) == len(set(names)), f"duplicate advertised tools: {names}"
    # 2. No advertised registry tool duplicates a KNOWN_ACTION definition
    overlap = set(names) & set(KNOWN_ACTIONS)
    assert not overlap, f"registry tools already in KNOWN_ACTIONS: {overlap}"
    # 3. Prompt descriptions come from the CANONICAL registry (no second list)
    for name, _p, description in tools:
        assert description and f"- {name}(" in section
        assert description in section
    # 4. Prompt ↔ Brain gate consistency: every advertised tool is gated,
    #    and every non-KNOWN_ACTIONS gate entry resolves in the registry.
    schema = AgentBrain._PLANNER_ACTION_SCHEMA
    for name in names:
        assert name in schema, (
            f"planner advertises '{name}' but Brain's action gate does not "
            f"know it — the step would be blocked after planning")
    for gate_name in schema:
        if gate_name not in KNOWN_ACTIONS:
            assert registry_tool_available(gate_name), (
                f"Brain gate allows '{gate_name}' but it is not a registered "
                f"tool nor a KNOWN_ACTION")


# ═══════════════════════════════════════════════════════════════
# H. The live planner prompt includes the registry section (fake LLM)
# ═══════════════════════════════════════════════════════════════

def test_planner_prompt_composition_includes_registry_tools():
    """_generate_plan composes PLANNER_SYSTEM_PROMPT + the registry section,
    and a planned registry-tool step survives plan parsing + validation."""
    plan_json = ('[{"action": "terminal", '
                 '"params": {"command": "echo hi"}, '
                 '"description": "Run echo in a shell"}]')
    fake_llm = FakeLLM(plan_json)
    planner = AgentPlanner()
    planner._llm_client = fake_llm  # no initialize() — no LLM/network

    plan = asyncio.run(planner._generate_plan("run a shell command"))
    assert plan and plan[0]["action"] == "terminal"
    # The composed prompt advertises the registry tools
    assert "General tools (tool registry)" in fake_llm.last_prompt
    assert "- terminal(" in fake_llm.last_prompt
    # And the planned step is accepted by the validator (same contract)
    ok, reason = PlanValidator().validate_step(plan[0], [], "")
    assert ok is True, reason