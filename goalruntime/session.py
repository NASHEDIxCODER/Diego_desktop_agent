"""
GoalRuntime session layer — persistent multi-turn autonomous sessions.

Hard rules:
  * After wake + auth the session is ACTIVE_SESSION.
  * It stays ACTIVE_SESSION across silence, completed tasks, TTS output, and
    normal failures — those NEVER return the session to wake mode.
  * ONLY an explicit sleep command ("sleep", "go to sleep", "diego sleep",
    "stop listening", "that's all", "bye diego") transitions to SLEEPING.
  * Goals accepted while active are tracked and reported on request.

Logging: [GOAL-SESSION]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    DORMANT = "dormant"                    # pre-wake
    WAKE_PENDING_AUTH = "wake_pending_auth"
    ACTIVE_SESSION = "active_session"
    SLEEPING = "sleeping"


# Explicit sleep vocabulary — the ONLY way out of ACTIVE_SESSION.
SLEEP_COMMANDS = (
    "go to sleep", "goto sleep", "go sleep", "sleep now", "sleep",
    "diego sleep", "stop listening", "stop the session", "end session",
    "that's all", "thats all", "thats it", "that is all", "bye diego",
    "goodbye diego", "good night", "goodnight",
)
_WAKE_COMMANDS = ("diego", "wake up", "hello diego")


@dataclass
class SessionEvent:
    at: float
    kind: str          # wake | auth_ok | auth_fail | goal_accepted |
                       # goal_completed | goal_failed | silence | tts |
                       # sleep
    detail: str = ""


class AutonomousSession:
    """The persistent autonomous session state machine.

    Silence, task completion, TTS, and failures are recorded as events but
    NEVER change the state out of ACTIVE_SESSION. Only sleep commands do.
    """

    def __init__(self, *, require_auth: bool = False) -> None:
        self.require_auth = require_auth
        self.state = SessionState.DORMANT
        self.events: List[SessionEvent] = []
        self.goals_accepted: List[Dict[str, Any]] = []
        self.goals_completed: List[str] = []
        self.goals_failed: List[str] = []
        self._wake_at: Optional[float] = None

    # ── lifecycle ────────────────────────────────────────────────

    def wake(self) -> SessionState:
        if self.state is SessionState.SLEEPING:
            # Re-wake from sleep: same flow as the first wake.
            self.state = (SessionState.WAKE_PENDING_AUTH
                          if self.require_auth
                          else SessionState.ACTIVE_SESSION)
        elif self.state is SessionState.DORMANT:
            self.state = (SessionState.WAKE_PENDING_AUTH
                          if self.require_auth
                          else SessionState.ACTIVE_SESSION)
        self._wake_at = time.time()
        self._log("wake")
        return self.state

    def auth_result(self, authenticated: bool) -> SessionState:
        if self.state is not SessionState.WAKE_PENDING_AUTH:
            return self.state
        if authenticated:
            self.state = SessionState.ACTIVE_SESSION
            self._log("auth_ok")
        else:
            self._log("auth_fail")
        return self.state

    def sleep_command(self) -> SessionState:
        self.state = SessionState.SLEEPING
        self._log("sleep")
        return self.state

    def reset(self) -> None:
        """Full reset (used by tests and by a hard shutdown)."""
        self.state = SessionState.DORMANT
        self.events.clear()
        self.goals_accepted.clear()
        self.goals_completed.clear()
        self.goals_failed.clear()
        self._wake_at = None

    # ── utterance routing ────────────────────────────────────────

    def is_sleep_command(self, text: str) -> bool:
        t = " ".join((text or "").strip().lower().split()).strip(".,!?")
        if not t:
            return False
        for c in SLEEP_COMMANDS:
            if t == c or t.startswith(c + " ") or t.endswith(" " + c):
                return True
        return False

    def handle_utterance(self, text: str) -> str:
        """Route one utterance; returns the resulting state value.

        Sleep commands are the only state-changing utterances; everything
        else is accepted as a goal (or noted as silence).
        """
        text = (text or "").strip()
        if not text:
            self.record_silence()
            return self.state.value
        if self.state is not SessionState.ACTIVE_SESSION:
            return self.state.value
        if self.is_sleep_command(text):
            return self.sleep_command().value
        return self.state.value

    # ── activity events (state-NEUTRAL by design) ────────────────

    def accept_goal(self, goal_text: str) -> str:
        gid = f"g{len(self.goals_accepted) + 1}"
        self.goals_accepted.append({"id": gid, "text": goal_text,
                                    "at": time.time()})
        self._log("goal_accepted", goal_text[:80])
        return gid

    def goal_completed(self, goal_text: str) -> None:
        self.goals_completed.append(goal_text)
        self._log("goal_completed", goal_text[:80])

    def goal_failed(self, goal_text: str) -> None:
        self.goals_failed.append(goal_text)
        self._log("goal_failed", goal_text[:80])

    def record_silence(self) -> None:
        """Silence is recorded — it NEVER ends the active session."""
        self._log("silence")

    def record_tts(self, text: str = "") -> None:
        """TTS output is recorded — it NEVER ends the active session."""
        self._log("tts", text[:80])

    def record_failure(self, detail: str = "") -> None:
        """Normal failures are recorded — they NEVER end the session."""
        self._log("failure", detail[:120])

    def record_task_continuation(self, detail: str = "") -> None:
        """A paused/resumed task continues INSIDE the same session."""
        self._log("task_continuation", detail[:120])

    # ── inspection ───────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state is SessionState.ACTIVE_SESSION

    @property
    def is_sleeping(self) -> bool:
        return self.state is SessionState.SLEEPING

    def summary(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "goals_accepted": len(self.goals_accepted),
            "goals_completed": len(self.goals_completed),
            "goals_failed": len(self.goals_failed),
            "events": len(self.events),
        }

    # ── internals ────────────────────────────────────────────────

    def _log(self, kind: str, detail: str = "") -> None:
        self.events.append(SessionEvent(at=time.time(), kind=kind,
                                        detail=detail))
        logger.debug("[GOAL-SESSION] %s %s (state=%s)", kind, detail,
                     self.state.value)


def _short_sleepish(t: str) -> bool:
    """A sleep command is short; long sentences mentioning 'sleep' are
    goals (e.g. 'open the sleep tracker website')."""
    return len(t.split()) <= 5


__all__ = ["AutonomousSession", "SessionState", "SessionEvent",
           "SLEEP_COMMANDS"]
