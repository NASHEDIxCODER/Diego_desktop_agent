"""
TaskContinuation — Continuous task confirmation + cross-turn resumption.

Multi-step tasks can pause for user confirmation and resume the SAME task
without the user repeating the original request. The pending state survives
the LISTEN → THINK → SPEAK → LISTEN cycle.

This module deliberately adds ONLY the minimum state needed on top of the
existing TaskController / TaskExecutionState — it does NOT create another
task system:

    pending_confirmation   — the action/step awaiting approval
    confirmation_prompt    — what Diego asked the user
    resume_step            — the single action to run on confirmation
    resume_plan            — optional remaining multi-step plan
    task_id                — the owning TaskExecutionState id
    expires_at             — safety expiry so stale prompts never fire

Natural confirmations ("yes", "yeah", "do it", "play it", …) and
cancellations ("no", "cancel", "stop", …) are recognised ONLY while a
pending confirmation actually exists. Without one, "yes" does nothing.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Natural confirmation / cancellation vocabulary
# ═══════════════════════════════════════════════════════════════

# Concise affirmations. Only treated as confirmation when a pending
# confirmation exists.
CONFIRM_PHRASES = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "k",
    "do it", "go ahead", "play it", "go for it", "yes please",
    "yeah sure", "sure thing", "absolutely", "definitely", "confirm",
    "proceed", "continue",
})

# Concise cancellations. Only treated as cancellation when a pending
# confirmation exists.
CANCEL_PHRASES = frozenset({
    "no", "nope", "nah", "cancel", "cancel it", "stop", "stop it",
    "don't", "dont", "do not", "don't do it", "dont do it", "do not do it",
    "no thanks", "never mind", "nevermind",
    "forget it", "abort",
})

# Verbs that indicate "resume the pending playback/action" even when phrased
# as a short imperative ("play it", "do it", "go ahead").
_CONFIRM_RE = re.compile(
    r"^(?:"
    r"yes|yeah|yep|yup|sure|okay|ok|k|confirm|proceed|continue|"
    r"do\s+it|go\s+ahead|play\s+it|go\s+for\s+it|yes\s+please|"
    r"sure\s+thing|absolutely|definitely"
    r")[\s!.?]*$",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"^(?:"
    r"no|nope|nah|cancel|cancel\s+it|stop|stop\s+it|abort|forget\s+it|"
    r"don'?t|do\s+not|don'?t\s+do\s+it|do\s+not\s+do\s+it|"
    r"no\s+thanks|never\s*mind"
    r")[\s!.?]*$",
    re.IGNORECASE,
)


def classify_confirmation(text: str) -> Optional[str]:
    """Classify a short utterance as a confirmation / cancellation.

    Returns:
        "confirm"  — the user approved the pending action
        "cancel"   — the user rejected the pending action
        None       — not a confirmation/cancellation (unrelated command)

    NOTE: Callers must only ACT on "confirm"/"cancel" when a pending
    confirmation actually exists. Without one these words are just
    conversation and must not execute anything.
    """
    t = " ".join((text or "").lower().strip(" .!?,").split())
    if not t:
        return None
    if t in CONFIRM_PHRASES or _CONFIRM_RE.match(t):
        return "confirm"
    if t in CANCEL_PHRASES or _CANCEL_RE.match(t):
        return "cancel"
    return None


# ═══════════════════════════════════════════════════════════════
# Pending task record
# ═══════════════════════════════════════════════════════════════

# Default time-to-live for a pending confirmation (seconds). After this the
# pending task expires safely and a later "yes" does nothing.
DEFAULT_PENDING_TTL_S = 300.0


@dataclass
class PendingTask:
    """The minimum state required to pause + resume a task at a
    confirmation point."""

    # The owning task (TaskExecutionState.task_id) — lets us resume the SAME
    # task rather than starting a new one.
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # The user's original goal ("play believer on youtube").
    goal: str = ""

    # The single action to execute when the user confirms.
    # e.g. {"action": "play_media", "params": {"query": "believer", "youtube": True}}
    resume_step: Optional[Dict[str, Any]] = None

    # Optional remaining multi-step plan (for general multi-step tasks).
    # When present, resumption runs this plan through the closed-loop runner.
    resume_plan: List[Dict[str, Any]] = field(default_factory=list)

    # Action signatures the user has already approved (skip re-confirmation).
    approved_signatures: List[str] = field(default_factory=list)

    # What Diego actually asked the user (for re-prompting / diagnostics).
    confirmation_prompt: str = ""

    # Reference to the owning TaskExecutionState (for inheritance on resume).
    # Kept as an opaque object to avoid a hard import cycle.
    task_state: Optional[Any] = None

    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return self.expires_at > 0 and now >= self.expires_at


# ═══════════════════════════════════════════════════════════════
# Pending task manager (singleton)
# ═══════════════════════════════════════════════════════════════

class PendingTaskManager:
    """Holds at most ONE pending confirmation at a time.

    A new pending task replaces any previous one. The pending state survives
    across conversational turns (LISTEN → THINK → SPEAK → LISTEN) because it
    lives in this process-level singleton, not in a single turn's scope.
    """

    def __init__(self):
        self._pending: Optional[PendingTask] = None
        # One-shot record of the most recently EXPIRED pending task. Lets the
        # Brain answer a late "yes"/"no" with an honest explanation ("that
        # confirmation has expired") instead of silently dropping it.
        self._last_expired: Optional[PendingTask] = None

    # ── Mutation ────────────────────────────────────────────────

    def set_pending(
        self,
        goal: str,
        resume_step: Optional[Dict[str, Any]] = None,
        resume_plan: Optional[List[Dict[str, Any]]] = None,
        confirmation_prompt: str = "",
        task_state: Optional[Any] = None,
        task_id: Optional[str] = None,
        approved_signatures: Optional[List[str]] = None,
        ttl_s: float = DEFAULT_PENDING_TTL_S,
    ) -> PendingTask:
        """Register a pending confirmation. Replaces any existing one."""
        now = time.time()
        pending = PendingTask(
            task_id=task_id or uuid.uuid4().hex[:12],
            goal=goal,
            resume_step=resume_step,
            resume_plan=list(resume_plan or []),
            approved_signatures=list(approved_signatures or []),
            confirmation_prompt=confirmation_prompt,
            task_state=task_state,
            created_at=now,
            expires_at=now + ttl_s if ttl_s and ttl_s > 0 else 0.0,
        )
        self._pending = pending
        # A fresh pending task supersedes any previously expired one.
        self._last_expired = None
        logger.info(
            "[PENDING] set task_id=%s goal=%r resume_step=%s prompt=%r ttl=%.0fs",
            pending.task_id, goal[:60],
            (resume_step or {}).get("action"), confirmation_prompt[:60], ttl_s,
        )
        return pending

    def clear(self) -> None:
        """Drop the pending task (after completion/cancellation/expiry)."""
        if self._pending is not None:
            logger.info("[PENDING] cleared task_id=%s", self._pending.task_id)
        self._pending = None
        self._last_expired = None

    def cancel(self) -> Optional[PendingTask]:
        """Cancel and drop the pending task. Returns what was cancelled."""
        pending = self._pending
        self._pending = None
        self._last_expired = None
        if pending is not None:
            logger.info("[PENDING] cancelled task_id=%s", pending.task_id)
        return pending

    # ── Inspection ──────────────────────────────────────────────

    def get_pending(self, now: Optional[float] = None) -> Optional[PendingTask]:
        """Return the live pending task, or None if absent/expired.

        An expired pending task is cleared as a side effect so a stale
        prompt can never fire later. The expired record is kept (one-shot)
        so callers can explain the expiry honestly via pop_expired().
        """
        pending = self._pending
        if pending is None:
            return None
        if pending.is_expired(now):
            logger.info(
                "[PENDING] expired task_id=%s (age=%.0fs)",
                pending.task_id, (now or time.time()) - pending.created_at,
            )
            self._last_expired = pending
            self._pending = None
            return None
        return pending

    def pop_expired(self) -> Optional[PendingTask]:
        """Return (once) the pending task that expired since last checked.

        Returns None after the first call — a late confirmation is answered
        honestly exactly once; every later utterance is plain conversation.
        """
        expired = self._last_expired
        self._last_expired = None
        return expired

    @property
    def has_pending(self) -> bool:
        return self.get_pending() is not None


# Global singleton — the pending confirmation survives across turns.
pending_task_manager = PendingTaskManager()