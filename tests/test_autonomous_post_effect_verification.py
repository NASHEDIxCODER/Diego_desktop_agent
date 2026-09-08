"""Phase 15D regression tests — AUTONOMOUS POST-EFFECT VERIFICATION.

Verifies that registry/general tools get deterministic post-effect
verification when their existing action + params + result expose a safe
observable effect, WITHOUT creating a second verification system:

    TaskRunner (existing loop, untouched)
    → ActionDispatcher._execute_via_registry (existing honest conversion)
    → core/tool_registry.verify_post_effect (NEW canonical post-effect check)
    → verified   → success output + "[post-effect verified: <what>]" evidence
    → check failed → honest "Couldn't ..." marker → existing fail-fast rejects
    → no safe verifier → EXECUTION_CONFIRMED_ONLY (unchanged contract)

Classification matrix (production):
  POST_EFFECT_VERIFIED   : filesystem write/create_dir/read/list
                           (+ deterministic absence check if a delete
                           action is ever exposed by the tool)
  EXECUTION_CONFIRMED_ONLY: terminal, python, git, git_status, docker,
                           clipboard_*, notify, mouse, keyboard, open_*,
                           search, browser, volume, brightness

Mocks/fakes + pytest tmp_path only. No real shell, network, camera,
microphone, models, or browser. The only real filesystem operations are
confined to pytest tmp_path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import core.tool_registry as tool_registry_module
from core.tool_registry import Tool, ToolResult, verify_post_effect
from agent.action_dispatcher import ActionDispatcher
from agent.brain import AgentBrain
from agent.task_state import (
    FinalStatus,
    PlanValidator,
    StepStatus,
    TaskLimits,
    TaskRunner,
    is_sensitive_action,
    task_state_store,
)
import agent.task_state as task_state_module


# ═══════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════

class FakeRegistry:

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


class FakeObserver:

    def __init__(self, states):
        self.states = list(states)

    async def __call__(self):
        if self.states:
            return self.states.pop(0)
        return ""


class BrainLikeExecutor:
    """Executor mirroring the REAL Brain._dispatch_and_verify contract:
    real dispatcher + the real Brain verification fallback function that
    registry tools flow through (fail-fast on honest failure markers)."""

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


class ScriptedExecutor:
    """Plain scripted executor for KNOWN_ACTIONS (existing path)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        if self.script:
            return self.script.pop(0)
        return False, "no scripted result"


class BlockingExecutor:

    def __init__(self, entered, release):
        self.entered = entered
        self.release = release
        self.calls = []

    async def __call__(self, action):
        self.calls.append(dict(action))
        self.entered.set()
        await self.release.wait()
        return True, "step ran"


def make_fake_filesystem_tool(calls=None):
    """Fake 'filesystem' tool mirroring the real tool's semantics, scoped
    to tmp_path. Supports an extra 'delete' action ONLY to exercise the
    deterministic absence verifier — production adds NO delete action."""

    def handler(params):
        if calls is not None:
            calls.append(dict(params))
        action = params.get("action", "list")
        path = Path(str(params.get("path", ""))).expanduser()
        if action == "write":
            text = params.get("text", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            return ToolResult(success=True,
                              output=f"Wrote {len(text)} chars to {path}")
        if action == "create_dir":
            path.mkdir(parents=True, exist_ok=True)
            return ToolResult(success=True,
                              output=f"Created directory {path}")
        if action == "read":
            if not path.is_file():
                return ToolResult(success=False, error=f"Not a file: {path}")
            return ToolResult(success=True, output=path.read_text()[:100])
        if action == "list":
            if not path.exists():
                return ToolResult(success=False, error=f"Path not found: {path}")
            return ToolResult(success=True,
                              output="\n".join(c.name for c in path.iterdir()))
        if action == "delete":
            path.unlink()
            return ToolResult(success=True, output=f"Deleted {path}")
        return ToolResult(success=False, error=f"Unknown filesystem action: {action}")

    return Tool(name="filesystem", description="fake filesystem tool",
                handler=handler)


def make_simple_tool(name, output):
    """A successful tool with NO deterministic post-effect signal."""
    calls = []

    def handler(params):
        calls.append(dict(params))
        return ToolResult(success=True, output=output)

    tool = Tool(name=name, description=f"fake {name} tool", handler=handler)
    return tool, calls


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


@pytest.fixture
def fake_registry(monkeypatch):
    registry = FakeRegistry()
    monkeypatch.setattr(tool_registry_module, "tool_registry", registry)
    return registry


# ═══════════════════════════════════════════════════════════════
# A. Filesystem write: deterministic post-effect verification
# ═══════════════════════════════════════════════════════════════

def test_filesystem_write_post_effect_verified(fake_registry, tmp_path):
    target = tmp_path / "notes" / "hello.txt"
    fake_registry.register(make_fake_filesystem_tool())

    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="write the file")
    state = asyncio.run(runner.run(
        "write the file",
        [{"action": "filesystem",
          "params": {"action": "write", "path": str(target),
                     "text": "hello world"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert target.is_file() and target.read_text() == "hello world"
    step = state.completed_steps[0]
    assert step.verified is True
    assert step.status == StepStatus.COMPLETED
    # Evidence states WHAT was actually checked (deterministic effect)
    assert "[post-effect verified: target file exists:" in step.result
    assert "[VERIFY] success" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# B. Filesystem write where verification fails → never SUCCESS
# ═══════════════════════════════════════════════════════════════

def test_filesystem_write_verification_failure_is_not_success(fake_registry,
                                                              tmp_path):
    """The tool REPORTS success but the deterministic effect check fails
    (e.g. the file never materialized) → honest failure via the existing
    "Couldn't ..." fail-fast marker."""
    target = tmp_path / "ghost.txt"

    def lying_handler(params):
        # Reports success WITHOUT producing the effect
        return ToolResult(success=True,
                          output=f"Wrote 5 chars to {target}")

    fake_registry.register(Tool(name="filesystem", description="fake",
                                handler=lying_handler))

    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="write the file")
    state = asyncio.run(runner.run(
        "write the file",
        [{"action": "filesystem",
          "params": {"action": "write", "path": str(target),
                     "text": "hello"}}]))

    assert state.final_status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE)
    assert state.final_status != FinalStatus.SUCCESS
    assert not target.exists()
    assert len(state.completed_steps) == 0
    failed = state.failed_steps[0]
    assert failed.verified is False
    assert "Couldn't filesystem" in failed.result
    assert "post-effect check failed" in failed.result
    assert state.verification_results[-1]["verified"] is False


# ═══════════════════════════════════════════════════════════════
# C. Filesystem delete: deterministic absence verification
# ═══════════════════════════════════════════════════════════════

def test_filesystem_delete_absence_verified_after_confirmation(fake_registry,
                                                               tmp_path):
    """If a delete action is ever exposed through the filesystem tool,
    the shared verifier proves deterministic ABSENCE (read-only check).
    The destructive action still goes through the existing confirmation
    gate first (approved here so the effect can be verified)."""
    target = tmp_path / "trash.txt"
    target.write_text("discard me")
    fake_registry.register(make_fake_filesystem_tool())

    async def approve(action, reason, params):
        return True

    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="delete the temp file",
                         confirmation_callback=approve)
    state = asyncio.run(runner.run(
        "delete the temp file",
        [{"action": "filesystem",
          "params": {"action": "delete", "path": str(target)}}]))

    assert state.final_status == FinalStatus.SUCCESS
    assert not target.exists()
    step = state.completed_steps[0]
    assert step.verified is True
    assert "[post-effect verified: target no longer present:" in step.result


def test_create_dir_post_effect_verified(fake_registry, tmp_path):
    target = tmp_path / "newdir"
    fake_registry.register(make_fake_filesystem_tool())
    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="create the folder")
    state = asyncio.run(runner.run(
        "create the folder",
        [{"action": "filesystem",
          "params": {"action": "create_dir", "path": str(target)}}]))
    assert state.final_status == FinalStatus.SUCCESS
    assert target.is_dir()
    assert "[post-effect verified: target directory exists:" \
        in state.completed_steps[0].result


# ═══════════════════════════════════════════════════════════════
# D. Tool with no safe post-effect verifier → EXECUTION_CONFIRMED_ONLY
# ═══════════════════════════════════════════════════════════════

def test_no_post_effect_verifier_stays_execution_confirmed(fake_registry):
    """A successful tool without a safe deterministic effect (e.g. git
    status) must NOT be upgraded to a post-effect claim."""
    tool, calls = make_simple_tool("git", "On branch main\nnothing to commit")
    fake_registry.register(tool)

    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="show the git status")
    state = asyncio.run(runner.run(
        "show the git status",
        [{"action": "git", "params": {"action": "status", "path": "."}}]))

    assert state.final_status == FinalStatus.SUCCESS  # execution-confirmed
    step = state.completed_steps[0]
    assert step.verified is True
    # Honest classification: execution evidence only — NO post-effect claim
    assert "[post-effect verified" not in step.result
    assert all("[post-effect verified" not in str(vr)
               for vr in state.verification_results)
    # The verifier did not execute the tool again for verification
    assert len(calls) == 1


# ═══════════════════════════════════════════════════════════════
# E. Terminal success does NOT automatically become post-effect verified
# ═══════════════════════════════════════════════════════════════

def test_terminal_success_is_not_post_effect_verified(fake_registry):
    tool, calls = make_simple_tool("terminal", "hello from the shell")
    fake_registry.register(tool)

    # 1. Verifier classification: terminal has no safe post-effect check
    verdict, check = verify_post_effect(
        "terminal", {"command": "echo hi"},
        ToolResult(success=True, output="hello from the shell"))
    assert verdict is None and check == ""

    # 2. Through the autonomous path: success WITHOUT a post-effect claim
    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="run the command")
    state = asyncio.run(runner.run(
        "run the command",
        [{"action": "terminal", "params": {"command": "echo hi"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    step = state.completed_steps[0]
    assert step.verified is True
    assert "[post-effect verified" not in step.result
    assert "[post-effect verified" not in " ".join(state.log)
    # Verification did not execute any additional code
    assert len(calls) == 1


# ═══════════════════════════════════════════════════════════════
# F. Existing KNOWN_ACTION verification remains intact
# ═══════════════════════════════════════════════════════════════

def test_known_action_verification_unchanged(monkeypatch):
    """KNOWN_ACTIONS keep their existing verification path: no registry
    post-effect suffix, same evidence format, same vision mapping."""
    monkeypatch.setattr(task_state_module, "app_running", lambda app: False)
    executor = ScriptedExecutor([(True, "Opened firefox")])
    runner = make_runner(executor, FakeObserver(["", "firefox focused"]),
                         transcript="open firefox")
    state = asyncio.run(runner.run(
        "open firefox",
        [{"action": "desktop_open", "params": {"app": "firefox"}}]))

    assert state.final_status == FinalStatus.SUCCESS
    step = state.completed_steps[0]
    assert step.verified is True
    assert step.result == "Opened firefox"  # unchanged result text
    assert "[post-effect verified" not in step.result
    vr = state.verification_results[0]
    assert vr["verified"] is True and vr["evidence"] == "Opened firefox"
    # Existing vision verifier mapping untouched
    assert AgentBrain._map_action_to_verify_type("desktop_open") == "open_app"
    assert AgentBrain._map_action_to_verify_type("browser_navigate") == "navigate"


# ═══════════════════════════════════════════════════════════════
# G. Cancellation remains authoritative CANCELLED
# ═══════════════════════════════════════════════════════════════

def test_cancellation_remains_authoritative(fake_registry):
    fake_registry.register(make_fake_filesystem_tool())
    task_state_store.unregister_runner()
    entered = asyncio.Event()
    release = asyncio.Event()
    ex = BlockingExecutor(entered, release)
    runner = make_runner(ex, FakeObserver(["", ""]),
                         transcript="write then delete")

    async def scenario():
        task = asyncio.create_task(runner.run(
            "write then delete",
            [
                {"action": "filesystem",
                 "params": {"action": "write", "path": "/tmp/x.txt",
                            "text": "a"}},
                {"action": "filesystem",
                 "params": {"action": "write", "path": "/tmp/y.txt",
                            "text": "b"}},
            ],
        ))
        await entered.wait()
        assert task_state_store.cancel_active_runner() is True
        release.set()
        return await task

    state = asyncio.run(scenario())
    assert state.final_status == FinalStatus.CANCELLED
    assert state.final_status != FinalStatus.SUCCESS
    assert len(ex.calls) == 1
    assert "[TASK] CANCELLED" in " ".join(state.log)


# ═══════════════════════════════════════════════════════════════
# H. Sensitive registry action still requires confirmation
# ═══════════════════════════════════════════════════════════════

def test_sensitive_delete_requires_confirmation(fake_registry, tmp_path):
    sensitive, reason = is_sensitive_action(
        "filesystem", {"action": "delete", "path": str(tmp_path)})
    assert sensitive is True and reason

    target = tmp_path / "keep.txt"
    target.write_text("keep")
    calls = []
    fake_registry.register(make_fake_filesystem_tool(calls=calls))

    async def deny(action, reason_, params):
        return False

    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver([""]),
                         transcript="delete the file",
                         confirmation_callback=deny)
    state = asyncio.run(runner.run(
        "delete the file",
        [{"action": "filesystem",
          "params": {"action": "delete", "path": str(target)}}]))

    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.action == "filesystem"
    assert executor.calls == []          # never dispatched
    assert calls == []                   # tool never executed
    assert target.exists()               # nothing destructive happened


# ═══════════════════════════════════════════════════════════════
# I. Verification evidence records what was actually checked
# ═══════════════════════════════════════════════════════════════

def test_verification_evidence_records_the_check(fake_registry, tmp_path):
    target = tmp_path / "evidence.txt"
    fake_registry.register(make_fake_filesystem_tool())
    executor = BrainLikeExecutor(ActionDispatcher())
    runner = make_runner(executor, FakeObserver(["", ""]),
                         transcript="write the file")
    state = asyncio.run(runner.run(
        "write the file",
        [{"action": "filesystem",
          "params": {"action": "write", "path": str(target),
                     "text": "data"}}]))

    vr = state.verification_results[-1]
    assert vr["verified"] is True
    assert vr["action"] == "filesystem"
    # The evidence string states the deterministic check AND its target
    assert "post-effect verified" in vr["evidence"]
    assert "target file exists" in vr["evidence"]
    assert str(target) in vr["evidence"]
    assert "[VERIFY] success" in " ".join(state.log)