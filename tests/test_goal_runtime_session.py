"""GoalRuntime session — persistent ACTIVE_SESSION until explicit sleep."""

from __future__ import annotations

import pytest

from goalruntime.session import (
    AutonomousSession, SessionState, SLEEP_COMMANDS,
)


def test_wake_without_auth_goes_straight_active():
    s = AutonomousSession()
    assert s.wake() is SessionState.ACTIVE_SESSION
    assert s.is_active


def test_wake_with_auth_requires_auth_before_active():
    s = AutonomousSession(require_auth=True)
    assert s.wake() is SessionState.WAKE_PENDING_AUTH
    assert not s.is_active
    assert s.auth_result(True) is SessionState.ACTIVE_SESSION
    assert s.is_active


def test_failed_auth_does_not_activate():
    s = AutonomousSession(require_auth=True)
    s.wake()
    assert s.auth_result(False) is SessionState.WAKE_PENDING_AUTH
    assert not s.is_active
    # Successful retry activates.
    assert s.auth_result(True) is SessionState.ACTIVE_SESSION


# ═══════════════════════════════════════════════════════════════════
# The session NEVER leaves ACTIVE_SESSION except via sleep
# ═══════════════════════════════════════════════════════════════════

def test_silence_keeps_session_active():
    s = AutonomousSession()
    s.wake()
    for _ in range(5):
        s.handle_utterance("")          # silence
    assert s.is_active
    kinds = [e.kind for e in s.events]
    assert kinds.count("silence") == 5
    assert "sleep" not in kinds


def test_completed_goals_keep_session_active():
    s = AutonomousSession()
    s.wake()
    for i in range(3):
        s.accept_goal(f"goal {i}")
        s.goal_completed(f"goal {i}")
        s.record_tts("done")
    assert s.is_active                  # NEVER back to wake mode
    assert s.summary()["goals_completed"] == 3


def test_normal_failures_keep_session_active():
    s = AutonomousSession()
    s.wake()
    s.accept_goal("impossible goal")
    s.goal_failed("impossible goal")
    s.record_failure("app crashed")
    assert s.is_active


def test_tts_output_keeps_session_active():
    s = AutonomousSession()
    s.wake()
    s.record_tts("Here is your summary.")
    assert s.is_active


def test_task_continuation_stays_inside_session():
    s = AutonomousSession()
    s.wake()
    s.accept_goal("send a message")
    s.record_task_continuation("awaiting confirmation")
    s.record_tts("Should I send it?")
    assert s.is_active                  # the pause is INSIDE the session


# ═══════════════════════════════════════════════════════════════════
# Explicit sleep — the ONLY exit
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cmd", [
    "sleep", "go to sleep", "diego sleep", "sleep now", "stop listening",
    "that's all", "thats all", "bye diego", "good night",
])
def test_explicit_sleep_commands(cmd):
    s = AutonomousSession()
    s.wake()
    assert s.handle_utterance(cmd) == SessionState.SLEEPING.value
    assert s.is_sleeping


def test_goal_mentioning_sleep_is_NOT_a_sleep_command():
    s = AutonomousSession()
    s.wake()
    s.handle_utterance("open the sleep tracker website")
    assert s.is_active                  # long goal, not a sleep command


def test_rewake_after_sleep():
    s = AutonomousSession()
    s.wake()
    s.sleep_command()
    assert s.is_sleeping
    s.wake()
    assert s.is_active                  # re-enters the active session
    s.reset()
    assert s.state is SessionState.DORMANT


def test_utterances_before_wake_do_not_activate():
    s = AutonomousSession()
    assert s.handle_utterance("open instagram") == SessionState.DORMANT.value
    assert not s.is_active


def test_sleep_vocabulary_is_explicit_only():
    """Every sleep phrase is an explicit command — no implicit exits."""
    assert "sleep" in SLEEP_COMMANDS
    assert "stop listening" in SLEEP_COMMANDS
