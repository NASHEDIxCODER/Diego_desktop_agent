"""
Regression tests for continuous task confirmation + cross-turn resumption.

Covers:
  - YouTube search → confirmation (never plays before confirmation)
  - "yes" resumes the SAME task
  - "no" cancels the task cleanly
  - "yes" without a pending confirmation does nothing
  - unrelated command while a task is pending (pending stays alive)
  - pending task expires safely
  - playback only after confirmation
  - playback requires verification (never claims unverified playback)
  - task state survives conversation state transitions
  - general multi-step sensitive task pause → resume (TaskRunner)
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from agent.task_continuation import (
    PendingTaskManager,
    classify_confirmation,
    pending_task_manager,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _clear_pending():
    """Every test starts with NO pending confirmation."""
    pending_task_manager.clear()
    yield
    pending_task_manager.clear()


def _make_brain():
    """A bare AgentBrain with no subsystems wired (methods under test are
    monkeypatched individually)."""
    from agent.brain import AgentBrain
    return AgentBrain()


def _recording_dispatch(calls, result=(True, "ok")):
    """Build a fake _dispatch_and_verify that records every action."""
    async def _dispatch(action):
        calls.append(action)
        if callable(result):
            return result(action)
        return result
    return _dispatch


# ═══════════════════════════════════════════════════════════════
# classify_confirmation
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "yes", "yeah", "yep", "sure", "okay", "ok", "do it", "go ahead",
    "play it", "Yes!", "  yeah  ",
])
def test_classify_confirm(text):
    assert classify_confirmation(text) == "confirm"


@pytest.mark.parametrize("text", [
    "no", "nope", "cancel", "stop", "don't", "dont", "no thanks", "Cancel it.",
])
def test_classify_cancel(text):
    assert classify_confirmation(text) == "cancel"


@pytest.mark.parametrize("text", [
    "what's my cpu", "open firefox", "play believer", "hello", "",
])
def test_classify_unrelated(text):
    assert classify_confirmation(text) is None


# ═══════════════════════════════════════════════════════════════
# YouTube: search → confirmation (no playback yet)
# ═══════════════════════════════════════════════════════════════

async def test_youtube_search_then_confirmation():
    brain = _make_brain()
    calls = []
    brain._dispatch_and_verify = _recording_dispatch(
        calls, result=(True, "Opened YouTube results for believer"))

    decision = SimpleNamespace(action={
        "action": "play_media",
        "params": {"query": "believer", "youtube": True},
    })
    result = SimpleNamespace(
        path="", used_llm=True, actions_executed=0, actions_succeeded=0,
        actions_failed=0, verified=True, speak_immediately=True,
        response="", latency_ms=0.0,
    )
    conv = SimpleNamespace(add_assistant=lambda *_: None)

    handled = await brain._maybe_confirm_youtube_playback(
        decision, result, None, conv, time.time())

    assert handled is True
    # Only the SEARCH was dispatched — playback must NOT happen yet.
    assert len(calls) == 1
    assert calls[0]["action"] == "youtube_search"
    assert calls[0]["params"]["query"] == "believer"
    # A pending confirmation is stored.
    pending = pending_task_manager.get_pending()
    assert pending is not None
    assert pending.resume_step["action"] == "play_media"
    assert pending.resume_step["params"]["youtube"] is True
    # The response ASKS for confirmation and never claims playback.
    assert "should i play it" in result.response.lower()
    assert result.verified is False


async def test_youtube_no_confirmation_before_search():
    """The gate must not ask to play when the search itself failed."""
    brain = _make_brain()
    brain._dispatch_and_verify = _recording_dispatch(
        [], result=(False, "Couldn't open YouTube"))

    decision = SimpleNamespace(action={
        "action": "play_media",
        "params": {"query": "believer", "youtube": True},
    })
    result = SimpleNamespace(
        path="", used_llm=True, actions_executed=0, actions_succeeded=0,
        actions_failed=0, verified=True, speak_immediately=True,
        response="", latency_ms=0.0,
    )
    conv = SimpleNamespace(add_assistant=lambda *_: None)

    handled = await brain._maybe_confirm_youtube_playback(
        decision, result, None, conv, time.time())

    assert handled is True
    # No pending confirmation when search failed.
    assert pending_task_manager.get_pending() is None
    assert "couldn't" in result.response.lower()


async def test_non_youtube_play_not_intercepted():
    brain = _make_brain()
    decision = SimpleNamespace(action={
        "action": "play_media", "params": {"query": "some song"},
    })
    result = SimpleNamespace(response="")
    handled = await brain._maybe_confirm_youtube_playback(
        decision, result, None, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    assert handled is False


# ═══════════════════════════════════════════════════════════════
# "yes" resumes the SAME task → playback + verification
# ═══════════════════════════════════════════════════════════════

async def test_yes_resumes_same_task_and_plays():
    brain = _make_brain()
    calls = []
    brain._dispatch_and_verify = _recording_dispatch(
        calls, result=(True, "Playing believer on YouTube"))

    pending_task_manager.set_pending(
        goal="play believer on youtube",
        resume_step={"action": "play_media",
                     "params": {"query": "believer", "youtube": True}},
        confirmation_prompt="I found believer on YouTube. Should I play it?",
    )

    from agent.brain import CommandResult
    result = CommandResult()
    conv = SimpleNamespace(add_assistant=lambda *_: None)

    handled = await brain._handle_pending_confirmation("yes", result, conv,
                                                       time.time())
    assert handled is True
    # The SAME task's resume step was dispatched (playback).
    assert len(calls) == 1
    assert calls[0]["action"] == "play_media"
    assert calls[0]["params"]["youtube"] is True
    assert calls[0]["params"]["query"] == "believer"
    # Pending cleared after completion.
    assert pending_task_manager.get_pending() is None
    assert result.verified is True
    assert "believer" in result.response.lower()


@pytest.mark.parametrize("word", ["yeah", "play it", "do it", "go ahead", "sure"])
async def test_natural_confirmations_resume(word):
    brain = _make_brain()
    calls = []
    brain._dispatch_and_verify = _recording_dispatch(calls, result=(True, "ok"))
    pending_task_manager.set_pending(
        goal="play x on youtube",
        resume_step={"action": "play_media",
                     "params": {"query": "x", "youtube": True}},
    )
    from agent.brain import CommandResult
    result = CommandResult()
    handled = await brain._handle_pending_confirmation(
        word, result, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    assert handled is True
    assert len(calls) == 1


# ═══════════════════════════════════════════════════════════════
# Playback requires verification
# ═══════════════════════════════════════════════════════════════

async def test_playback_requires_verification():
    brain = _make_brain()
    brain._dispatch_and_verify = _recording_dispatch(
        [], result=(False, "Couldn't verify playback"))

    pending_task_manager.set_pending(
        goal="play believer on youtube",
        resume_step={"action": "play_media",
                     "params": {"query": "believer", "youtube": True}},
    )
    from agent.brain import CommandResult
    result = CommandResult()
    handled = await brain._handle_pending_confirmation(
        "yes", result, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    assert handled is True
    # Must NOT claim success when verification failed.
    assert result.verified is False
    assert result.actions_failed == 1
    assert "couldn't verify" in result.response.lower()


# ═══════════════════════════════════════════════════════════════
# "no" cancels cleanly
# ═══════════════════════════════════════════════════════════════

async def test_no_cancels_task():
    brain = _make_brain()
    calls = []
    brain._dispatch_and_verify = _recording_dispatch(calls)
    pending_task_manager.set_pending(
        goal="play believer on youtube",
        resume_step={"action": "play_media",
                     "params": {"query": "believer", "youtube": True}},
    )
    from agent.brain import CommandResult
    result = CommandResult()
    handled = await brain._handle_pending_confirmation(
        "no", result, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    assert handled is True
    assert calls == []                      # nothing executed
    assert pending_task_manager.get_pending() is None
    assert "cancel" in result.response.lower()


# ═══════════════════════════════════════════════════════════════
# "yes" with NO pending confirmation does nothing
# ═══════════════════════════════════════════════════════════════

async def test_yes_without_pending_does_nothing():
    brain = _make_brain()
    calls = []
    brain._dispatch_and_verify = _recording_dispatch(calls)
    from agent.brain import CommandResult
    result = CommandResult()
    handled = await brain._handle_pending_confirmation(
        "yes", result, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    # Not handled — falls through to normal processing, executes NOTHING.
    assert handled is False
    assert calls == []
    assert pending_task_manager.get_pending() is None


# ═══════════════════════════════════════════════════════════════
# Unrelated command while pending keeps the task alive
# ═══════════════════════════════════════════════════════════════

async def test_unrelated_command_keeps_pending_alive():
    brain = _make_brain()
    calls = []
    brain._dispatch_and_verify = _recording_dispatch(calls)
    pending_task_manager.set_pending(
        goal="play believer on youtube",
        resume_step={"action": "play_media",
                     "params": {"query": "believer", "youtube": True}},
    )
    from agent.brain import CommandResult
    result = CommandResult()
    handled = await brain._handle_pending_confirmation(
        "what's my cpu", result, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    # Unrelated — not consumed, pending task survives.
    assert handled is False
    assert calls == []
    assert pending_task_manager.get_pending() is not None

    # Now confirming still resumes the SAME task.
    brain._dispatch_and_verify = _recording_dispatch(calls, result=(True, "ok"))
    handled = await brain._handle_pending_confirmation(
        "yes", result, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    assert handled is True
    assert len(calls) == 1


# ═══════════════════════════════════════════════════════════════
# Pending task expires safely
# ═══════════════════════════════════════════════════════════════

async def test_pending_task_expires_safely():
    mgr = PendingTaskManager()
    pending = mgr.set_pending(goal="play x", resume_step={"action": "play_media"},
                              ttl_s=1.0)
    # Force expiry.
    pending.expires_at = time.time() - 1.0
    assert mgr.get_pending() is None

    # A late "yes" after expiry must do nothing.
    brain = _make_brain()
    calls = []
    brain._dispatch_and_verify = _recording_dispatch(calls)
    # Use the global manager (already cleared by fixture) — nothing pending.
    from agent.brain import CommandResult
    handled = await brain._handle_pending_confirmation(
        "yes", CommandResult(), SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    assert handled is False
    assert calls == []


# ═══════════════════════════════════════════════════════════════
# Task state survives conversation state transitions
# ═══════════════════════════════════════════════════════════════

async def test_pending_survives_across_turns():
    """Simulate LISTEN→THINK→SPEAK→LISTEN: the pending task set in one turn
    is still present and resumable in a later turn."""
    brain = _make_brain()
    calls = []

    # Turn 1: YouTube search → confirmation pending.
    brain._dispatch_and_verify = _recording_dispatch(
        calls, result=(True, "results"))
    decision = SimpleNamespace(action={
        "action": "play_media", "params": {"query": "believer", "youtube": True},
    })
    from agent.brain import CommandResult
    r1 = CommandResult()
    await brain._maybe_confirm_youtube_playback(
        decision, r1, None, SimpleNamespace(add_assistant=lambda *_: None),
        time.time())
    assert pending_task_manager.get_pending() is not None

    # Turn 2: unrelated command — pending survives.
    r2 = CommandResult()
    assert await brain._handle_pending_confirmation(
        "what time is it", r2, SimpleNamespace(add_assistant=lambda *_: None),
        time.time()) is False
    assert pending_task_manager.get_pending() is not None

    # Turn 3: confirmation — resumes the SAME task.
    brain._dispatch_and_verify = _recording_dispatch(calls, result=(True, "ok"))
    r3 = CommandResult()
    assert await brain._handle_pending_confirmation(
        "play it", r3, SimpleNamespace(add_assistant=lambda *_: None),
        time.time()) is True
    assert pending_task_manager.get_pending() is None
    # The last dispatched action is the playback.
    assert calls[-1]["action"] == "play_media"


# ═══════════════════════════════════════════════════════════════
# General multi-step task: sensitive action pause → resume
# ═══════════════════════════════════════════════════════════════

async def test_sensitive_task_pauses_then_resumes():
    """A multi-step task with a sensitive action pauses for confirmation and
    resumes the SAME task once approved (no re-ask)."""
    from agent.task_state import (
        TaskRunner, TaskLimits, FinalStatus, StepRecord,
    )

    executed = []

    async def executor(action):
        executed.append(action)
        return True, "done"

    plan = [{"action": "shutdown", "params": {}, "description": "Shut down"}]

    # Run 1: no confirmation callback → pauses with NEEDS_CONFIRMATION.
    runner = TaskRunner(executor=executor, limits=TaskLimits())
    state = await runner.run("shut down the computer", plan)
    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert state.pending_confirmation is not None
    assert executed == []  # nothing ran before confirmation

    # Resume with the pending step pre-approved.
    sig = StepRecord(index=0, action="shutdown", params={}).signature()
    runner2 = TaskRunner(executor=executor, limits=TaskLimits(),
                         approved_actions=frozenset({sig}))
    state2 = await runner2.run("shut down the computer", plan, inherited=state)
    assert state2.final_status == FinalStatus.SUCCESS
    assert len(executed) == 1
    assert executed[0]["action"] == "shutdown"


async def test_register_pending_confirmation_from_state():
    """The Brain registers a pending continuation from a NEEDS_CONFIRMATION
    task state so yes/no can resume/cancel it."""
    from agent.task_state import (
        TaskRunner, TaskLimits, FinalStatus, task_state_store,
    )

    async def executor(action):
        return True, "done"

    plan = [{"action": "shutdown", "params": {}, "description": "Shut down"}]
    runner = TaskRunner(executor=executor, limits=TaskLimits())
    state = await runner.run("shut down", plan)
    task_state_store.save(state)

    brain = _make_brain()
    brain._register_pending_confirmation(state)

    pending = pending_task_manager.get_pending()
    assert pending is not None
    assert pending.task_id == state.task_id
    assert pending.resume_step["action"] == "shutdown"
    assert pending.resume_plan  # remaining plan captured
    # NEEDS_CONFIRMATION is resumable in the task store.
    assert task_state_store.active is state