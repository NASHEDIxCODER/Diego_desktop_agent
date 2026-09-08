"""
Phase 19A — REAL autonomous goal continuation tests.

Covers the production multi-turn continuation flow end-to-end (through the
REAL AgentBrain.process_command pipeline, wired to mocks/fakes only):

  TURN 1: "Play <song> on YouTube"
    → youtube_search dispatch+verify → pending confirmation stored
    → "I found X on YouTube. Should I play it?" (NOTHING plays yet)
  TURN 2: "yes" (only meaningful BECAUSE a pending confirmation exists)
    → resume the SAME task → play_media dispatch+verify → truthful result
  "no" / "cancel" / "stop"  → cancel cleanly, nothing executes
  unrelated command         → answered normally, pending stays alive
  expired confirmation      → honest explanation, nothing executes

Also covers the general multi-step flow ("Open Firefox and then open the
file manager"), sensitive-action gating (NEEDS_CONFIRMATION can never be
silently bypassed), retry/replan on resumed execution, truthful lifecycle
events, and truthful final responses.

NO real YouTube / browser / mic / camera / audio / network / Ollama —
every external boundary is a controlled fake; production interfaces
(AgentBrain.process_command, TaskRunner, PendingTaskManager,
TaskStateStore, CommandResult, FinalStatus) are used as-is.
"""

from __future__ import annotations

import asyncio
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

import compat  # noqa: F401

import agent.conversation_memory as _cm
import agent.personality as _pers
import agent.task_continuation as _tc
import agent.task_state as _tsm
import core.decision_engine as _dem
import core.event_bus as _eb
from agent.brain import AgentBrain, CommandResult  # noqa: F401
from agent.task_continuation import PendingTaskManager
from agent.task_state import TaskStateStore, FinalStatus
from core.decision_engine import Decision, DecisionPath


# ═══════════════════════════════════════════════════════════════
# Fakes (controlled; no real hardware / network / LLM)
# ═══════════════════════════════════════════════════════════════

class FakeDecisionEngine:
    """Deterministic routing:

      - explicit "play X on youtube" → play_media(youtube=True) decision
        (the production router does exactly this; process_command funnels
        it through the YouTube confirmation gate);
      - multi-step / task phrases → LLM decision (planner path);
      - anything else → a resolved conversational decision (never tools).
    """

    def __init__(self):
        self.decisions: List[str] = []

    @staticmethod
    def _youtube_query(t: str) -> Optional[str]:
        q = re.sub(r"^play\s+", "", t)
        q = re.sub(r"\s+(?:on|from|in)\s+youtube$", "", q)
        return q.strip() or None

    async def decide(self, text: str, vision_context=None, search_context=None,
                     desktop_context=None) -> Decision:
        t = " ".join((text or "").lower().split())
        self.decisions.append(t)
        if "youtube" in t and (t.startswith("play") or " play " in t):
            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                action={"action": "play_media",
                        "params": {"youtube": True,
                                   "query": self._youtube_query(t)}})
        if "ram" in t:
            return Decision(
                path=DecisionPath.CONVERSATION, needs_llm=False,
                response="Your RAM looks fine — nothing to worry about.")
        if ("firefox" in t or "shut down" in t or "file manager" in t
                or "files" in t or "open" in t):
            return Decision(path=DecisionPath.LLM, needs_llm=True)
        return Decision(path=DecisionPath.CONVERSATION, needs_llm=False,
                        response="acknowledged")


class FakePlanner:
    """Returns canned plans for a goal, and alternative plans on re-plan.

    `plans` is keyed by goal substring; `replans` by a failed-step action
    substring. Mirrors Planner.generate_plan_only(request, context).
    """

    def __init__(self, plans=None, replans=None):
        self.plans: Dict[str, List[Dict[str, Any]]] = dict(plans or {})
        self.replans: Dict[str, List[Dict[str, Any]]] = dict(replans or {})
        self.calls: List[tuple] = []

    def generate_plan_only(self, request: str,
                           context: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        context = context or {}
        self.calls.append((request, context))
        # TaskRunner passes failed steps as "action: error" strings.
        failed = [str(s) for s in context.get("failed", [])]
        if failed:
            for needle, plan in self.replans.items():
                if any(needle in f for f in failed):
                    return [dict(step) for step in plan]
        g = request.lower()
        for needle, plan in self.plans.items():
            if needle in g:
                return [dict(step) for step in plan]
        return None


class FakeDispatch:
    """Controlled (ok, result) per action name; records every call.

    A False entry means dispatch+verification FAILED (the production
    Truthful-Result contract: success is never assumed — the user is only
    told what was actually verified). An entry may also be a list of
    results (sequence) or an async callable for per-params behavior.
    """

    def __init__(self, script: Optional[Dict[str, Any]] = None):
        self.script: Dict[str, Any] = dict(script or {})
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, action: Dict[str, Any]) -> tuple:
        self.calls.append(dict(action))
        name = action.get("action", "")
        entry = self.script.get(name, (True, f"{name} executed"))
        if callable(entry):
            return await entry(action)
        if isinstance(entry, list):
            value = entry[0] if len(entry) == 1 else entry.pop(0)
        else:
            value = entry
        return value


class MemConv:
    """In-memory conversation memory stub."""

    def __init__(self):
        self.user: List[str] = []
        self.assistant: List[str] = []
        self.goals: List[str] = []

    def add_user(self, text: str) -> None:
        self.user.append(text)

    def track_goal(self, goal: str) -> None:
        self.goals.append(goal)

    def add_assistant(self, text: str) -> None:
        self.assistant.append(text)


class StubPersonality:
    def contextual_response(self, user_text: str) -> Optional[str]:
        return None

    def acknowledgment(self) -> str:
        return "Understood."

    def task_confirmation(self) -> str:
        return "Done."


class NoSearchEngine:
    """Prevents the Brain LLM path from fetching web context (no network)."""

    @staticmethod
    def _needs_search(text: str) -> bool:
        return False


class RecorderBus:
    def __init__(self):
        self.events: List[tuple] = []

    async def emit(self, event_type: str, data=None, source=None):
        self.events.append((event_type, dict(data or {})))


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    """Isolated Brain wiring: fresh pending store, task store, event bus,
    conversation memory, personality, and disabled network/OS checks."""
    pending = PendingTaskManager()
    monkeypatch.setattr(_tc, "pending_task_manager", pending)

    store = TaskStateStore()
    store.TASK_DIR = str(tmp_path / "tasks")
    monkeypatch.setattr(_tsm, "task_state_store", store)
    monkeypatch.setattr(_tsm, "app_running", lambda app: False)
    monkeypatch.setenv("DIEGO_TASK_PERSIST", "0")

    monkeypatch.setattr(_cm, "conv_memory", MemConv())
    monkeypatch.setattr(_pers, "personality", StubPersonality())
    monkeypatch.setattr(_dem, "decision_engine", NoSearchEngine())

    bus = RecorderBus()
    monkeypatch.setattr(_eb, "bus", bus, raising=False)

    return SimpleNamespace(
        pending=pending, store=store, bus=bus,
        monkeypatch=monkeypatch, tmp_path=tmp_path,
    )


def make_brain(harness, decision=None, planner=None, dispatch=None):
    """Build a REAL AgentBrain wired to the fakes (same shape as the
    production controller wiring in create_production_controller)."""
    b = AgentBrain()
    b._initialized = True
    b._planner = planner
    b._perception = None
    b._dispatcher = None
    b._verifier = None
    b._learning = None
    b._llm_client = None
    b._decision_engine = decision or FakeDecisionEngine()
    b._dispatch_and_verify = dispatch
    return b


def run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════
# YouTube scenario helpers (search → confirmation → play)
# ═══════════════════════════════════════════════════════════════

def youtube_harness(harness, script=None):
    """A Brain + fakes for the primary "Play X on YouTube" scenario."""
    planner = FakePlanner(plans={
        "youtube": [{"action": "youtube_search",
                     "params": {"query": "test song"}, "sensitive": False}]})
    dispatch = FakeDispatch(script=script or {
        "youtube_search": (True, "Opened YouTube results for test song"),
        "play_media": (True, "Playing test song on YouTube"),
    })
    b = make_brain(harness, decision=FakeDecisionEngine(),
                   planner=planner, dispatch=dispatch)
    return b, planner, dispatch
# REQUIRED TEST 1 — YouTube search → confirmation prompt
def test_youtube_search_asks_confirmation(harness):
    b, _, dispatch = youtube_harness(harness)
    res = run(b.process_command("Play Test Song on YouTube"))
    # Turn 1 must NOT play — it searches and asks.
    assert [c["action"] for c in dispatch.calls] == ["youtube_search"]
    assert "play_media" not in [c["action"] for c in dispatch.calls]
    low = (res.response or "").lower()
    assert "test song" in low
    assert "should i play it" in low
    assert res.verified is False          # playback not started/verified yet
    assert res.path == "YOUTUBE_CONFIRMATION"
    assert harness.pending.get_pending() is not None


# REQUIRED TEST 2 — pending confirmation stored correctly
def test_pending_confirmation_stored_correctly(harness):
    b, _, _ = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    pending = harness.pending.get_pending()
    assert pending is not None
    assert "youtube" in (pending.goal or "").lower()
    assert "test song" in (pending.goal or "").lower()
    assert pending.resume_step is not None
    assert pending.resume_step["action"] == "play_media"
    assert pending.resume_step["params"]["youtube"] is True
    assert "should i play it" in (pending.confirmation_prompt or "").lower()
    assert pending.expires_at > time.time()      # TTL enforced


# REQUIRED TEST 3 — "yes" resumes the SAME task
def test_yes_resumes_same_task(harness):
    b, _, dispatch = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    res = run(b.process_command("yes"))
    assert [c["action"] for c in dispatch.calls] == [
        "youtube_search", "play_media"]
    assert res.verified is True
    assert res.task_status == "SUCCESS"
    assert harness.pending.get_pending() is None   # pending consumed


# REQUIRED TEST 4 — resume does NOT repeat the already-verified search
def test_resume_does_not_repeat_verified_search(harness):
    b, planner, dispatch = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    searches_before = [c for c in dispatch.calls
                       if c["action"] == "youtube_search"]
    res = run(b.process_command("yes"))
    searches_after = [c for c in dispatch.calls
                      if c["action"] == "youtube_search"]
    assert len(searches_after) == len(searches_before) == 1  # not repeated
    assert res.verified is True
    # No re-planning of the goal either — the stored resume_step is used.
    assert not any("youtube" in str(g).lower() for g, _ in planner.calls)


# REQUIRED TEST 5 — playback executes only after confirmation
def test_playback_executes_after_confirmation(harness):
    b, _, dispatch = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    assert "play_media" not in [c["action"] for c in dispatch.calls]
    run(b.process_command("yes"))
    assert "play_media" in [c["action"] for c in dispatch.calls]


# REQUIRED TEST 6 — playback must be VERIFIED before SUCCESS
def test_playback_verified_before_success(harness):
    b, _, dispatch = youtube_harness(harness, script={
        "youtube_search": (True, "Opened YouTube results for test song"),
        "play_media": (False, "Couldn't verify playback"),
    })
    run(b.process_command("Play Test Song on YouTube"))
    res = run(b.process_command("yes"))
    # The play action executed but its verification failed → NOT success.
    assert [c["action"] for c in dispatch.calls] == [
        "youtube_search", "play_media"]
    assert res.task_status != "SUCCESS"
    assert res.verified is False
    low = (res.response or "").lower()
    assert ("couldn't" in low or "fail" in low or "not" in low)


# REQUIRED TEST 7 — verified playback is truthful success
def test_verified_playback_is_truthful_success(harness):
    b, _, _ = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    res = run(b.process_command("yes"))
    assert res.task_status == "SUCCESS"
    assert res.verified is True
    low = (res.response or "").lower()
    assert "playing" in low and "test song" in low


# REQUIRED TEST 8 — "no" cancels the pending task
def test_no_cancels_pending_task(harness):
    b, _, dispatch = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    res = run(b.process_command("no"))
    assert harness.pending.get_pending() is None
    assert "play_media" not in [c["action"] for c in dispatch.calls]
    assert "cancelled" in (res.response or "").lower()
    assert res.verified is False


# REQUIRED TEST 9 — "cancel"/"stop"/"don't do it" cancel, nothing executes
@pytest.mark.parametrize("word", ["cancel", "stop", "don't do it"])
def test_cancel_cancels_pending_task(harness, word):
    b, _, dispatch = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    res = run(b.process_command(word))
    assert harness.pending.get_pending() is None
    assert "play_media" not in [c["action"] for c in dispatch.calls]
    assert "cancelled" in (res.response or "").lower()


# REQUIRED TEST 10 — confirmation words with NO pending task do nothing
@pytest.mark.parametrize("word", ["yes", "yeah", "do it", "play it", "go ahead"])
def test_confirmation_without_pending_does_nothing(harness, word):
    b, _, dispatch = youtube_harness(harness)
    res = run(b.process_command(word))
    assert dispatch.calls == []                     # NOTHING executed
    assert res.actions_executed == 0
    assert harness.pending.get_pending() is None


# REQUIRED TEST 11 — unrelated command does NOT consume the pending task
def test_unrelated_command_keeps_pending(harness):
    b, _, dispatch = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    assert harness.pending.get_pending() is not None

    res = run(b.process_command("What's my RAM?"))
    assert "ram" in (res.response or "").lower()     # answered normally
    assert harness.pending.get_pending() is not None  # still alive
    assert "play_media" not in [c["action"] for c in dispatch.calls]

    # The pending task can STILL be resumed by the next "yes".
    res = run(b.process_command("yes"))
    assert res.task_status == "SUCCESS"
# ═══════════════════════════════════════════════════════════════
# General multi-step tasks (sensitive pause → resume → execute)
# ═══════════════════════════════════════════════════════════════

SENSITIVE_PLANS = {
    "firefox and then shut down": [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "shutdown", "params": {}},
    ],
}


def sensitive_harness(harness, script, replans=None):
    planner = FakePlanner(plans=SENSITIVE_PLANS, replans=replans or {})
    dispatch = FakeDispatch(script=script)
    b = make_brain(harness, decision=FakeDecisionEngine(),
                   planner=planner, dispatch=dispatch)
    return b, planner, dispatch


# REQUIRED TEST 12 — pending task expires
def test_pending_task_expires(harness):
    b, _, _ = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    pending = harness.pending.get_pending()
    assert pending is not None
    pending.expires_at = time.time() - 1.0          # force TTL expiry
    assert harness.pending.get_pending() is None   # gone
    assert not harness.pending.has_pending


# REQUIRED TEST 13 — an expired confirmation cannot execute + honest reason
def test_expired_confirmation_cannot_execute(harness):
    b, _, dispatch = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    pending = harness.pending.get_pending()
    pending.expires_at = time.time() - 1.0          # force TTL expiry

    res = run(b.process_command("yes"))
    # Nothing executes and the user gets an honest explanation.
    assert "play_media" not in [c["action"] for c in dispatch.calls]
    assert res.verified is False
    assert res.path == "TASK_CONFIRMATION_EXPIRED"
    assert "expired" in (res.response or "").lower()


# REQUIRED TEST 14 — resumed task can retry (transient failure recovers)
def test_resumed_task_can_retry(harness):
    b, _, dispatch = sensitive_harness(harness, script={
        "desktop_open": (True, "Opened Firefox"),
        "shutdown": [(False, "shutdown timed out, please try again"),
                     (True, "Shutdown initiated")],
    })
    res1 = run(b.process_command("Open Firefox and then shut down the computer"))
    assert "shutdown" not in [c["action"] for c in dispatch.calls]
    assert "confirmation" in (res1.response or "").lower()
    pending = harness.pending.get_pending()
    assert pending is not None
    assert pending.resume_plan                       # full remaining plan kept

    res2 = run(b.process_command("yes"))
    assert res2.task_status == "SUCCESS"
    assert res2.verified is True
    executed = [c["action"] for c in dispatch.calls]
    assert executed[0] == "desktop_open"
    assert executed[1] == "shutdown"
    assert res2.actions_succeeded == 2
    # Retry happened inside the resumed run (same task, not a restart).
    assert executed.count("desktop_open") == 1


# REQUIRED TEST 15 — resumed task can re-plan from the current state
def test_resumed_task_can_replan(harness):
    b, _, dispatch = sensitive_harness(harness, script={
        "desktop_open": (True, "Opened Firefox"),
        "shutdown": (False, "shutdown is not available on this system"),
        "close_app": (True, "Closed Firefox"),
    }, replans={
        "shutdown": [{"action": "close_app", "params": {"app": "firefox"}}],
    })
    run(b.process_command("Open Firefox and then shut down the computer"))
    pending = harness.pending.get_pending()
    assert pending is not None

    res = run(b.process_command("yes"))
    # The failed sensitive step was replaced by a verified alternative.
    assert res.task_status == "SUCCESS"
    assert res.verified is True
    actions = [c["action"] for c in dispatch.calls]
    assert "close_app" in actions
# REQUIRED TEST 16 — cancellation during resumed execution stays CANCELLED
def test_cancellation_remains_cancelled(harness):
    b, _, dispatch = sensitive_harness(harness, script={
        "desktop_open": (True, "Opened Firefox"),
        "shutdown": (True, "Shutdown initiated"),
    })
    run(b.process_command("Open Firefox and then shut down the computer"))
    pending = harness.pending.get_pending()
    assert pending is not None and pending.task_state is not None
    task_id = pending.task_state.task_id

    res = run(b.process_command("no"))
    assert "cancelled" in (res.response or "").lower()

    # The paused task is finalized as CANCELLED in the store — it can never
    # be silently resumed (not by "yes", not by a follow-up "continue").
    assert harness.store.active is None
    assert harness.store.last is not None
    assert harness.store.last.task_id == task_id
    assert harness.store.last.final_status == FinalStatus.CANCELLED

    # A later "yes" does NOT resurrect it and executes nothing.
    res2 = run(b.process_command("yes"))
    assert res2.actions_executed == 0
    assert "shutdown" not in [c["action"] for c in dispatch.calls]
    assert harness.pending.get_pending() is None


# REQUIRED TEST 17 — a sensitive action can never bypass confirmation
def test_sensitive_action_cannot_bypass_confirmation(harness):
    b, _, dispatch = sensitive_harness(harness, script={
        "desktop_open": (True, "Opened Firefox"),
        "shutdown": (True, "Shutdown initiated"),
    })
    res1 = run(b.process_command("Open Firefox and then shut down the computer"))
    # The sensitive step was NOT executed on turn 1 — the task paused.
    assert "shutdown" not in [c["action"] for c in dispatch.calls]
    assert harness.pending.get_pending() is not None
    assert "confirmation" in (res1.response or "").lower()

    # Only the user's explicit "yes" (resuming the SAME task) runs it.
    res2 = run(b.process_command("yes"))
    assert "shutdown" in [c["action"] for c in dispatch.calls]
    assert res2.task_status == "SUCCESS"
# REQUIRED TEST 18 — lifecycle events are truthful
def test_lifecycle_events_truthful(harness):
    # 18a — generic multi-step success emits truthful activity.
    planner = FakePlanner(plans={
        "firefox and then open the files": [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "desktop_open", "params": {"app": "files"}},
        ]})
    b = make_brain(harness, decision=FakeDecisionEngine(), planner=planner,
                   dispatch=FakeDispatch(script={
                       "desktop_open": (True, "Opened it")}))
    res = run(b.process_command("Open Firefox and then open the file manager"))
    assert res.task_status == "SUCCESS"
    types = [t for t, _ in harness.bus.events]
    assert "task.started" in types
    assert types.count("task.completed") == 1
    completed = [d for t, d in harness.bus.events if t == "task.completed"]
    assert completed[0]["status"] == "SUCCESS"
    assert completed[0]["completed_steps"] == 2
    assert "task.failed" not in types
    # No raw action params are exposed on the bus (privacy contract).
    for _, data in harness.bus.events:
        assert "params" not in data

    # 18b — sensitive pause emits task.needs_confirmation, never FAILED.
    harness.bus.events.clear()
    b2, _, _ = sensitive_harness(harness, script={
        "desktop_open": (True, "Opened Firefox"),
        "shutdown": (True, "Shutdown initiated"),
    })
    run(b2.process_command("Open Firefox and then shut down the computer"))
    types = [t for t, _ in harness.bus.events]
    assert "task.needs_confirmation" in types
    assert "task.failed" not in types
    assert "task.completed" not in types


# REQUIRED TEST 19 — final response is truthful (verified facts only)
def test_final_response_truthful(harness):
    # 19a — verified success says what actually happened.
    b, _, _ = youtube_harness(harness)
    run(b.process_command("Play Test Song on YouTube"))
    res = run(b.process_command("yes"))
    assert res.task_status == "SUCCESS"
    assert res.verified is True
    assert ("playing" in res.response.lower()
            and "test song" in res.response.lower())

    # 19b — failed verification NEVER claims success.
    b2, _, _ = youtube_harness(harness, script={
        "youtube_search": (True, "Opened YouTube results for test song"),
        "play_media": (False, "Couldn't verify playback"),
    })
    run(b2.process_command("Play Test Song on YouTube"))
    res2 = run(b2.process_command("yes"))
    assert res2.task_status == "FAILED"
    assert res2.verified is False
    assert "playing test song" not in (res2.response or "").lower()


# REQUIRED TEST 20 — the autonomy regression surface is intact
# (the FULL regression run is the outer command; here we re-assert the
# shared contracts those suites depend on).
def test_regression_surface_green(harness):
    from agent.task_continuation import classify_confirmation, PendingTask
    from agent.task_state import TaskRunner, TaskLimits, PlanValidator
    assert classify_confirmation("yes") == "confirm"
    assert classify_confirmation("cancel") == "cancel"
    assert classify_confirmation("what's my cpu") is None

    # PendingTaskManager API used by the broader suites is intact.
    mgr = PendingTaskManager()
    p = mgr.set_pending("goal", resume_step={"action": "x"}, ttl_s=5)
    assert isinstance(p, PendingTask)
    assert mgr.get_pending() is p
    mgr.clear()
    assert mgr.pop_expired() is None

    # TaskRunner constructor surface used across the autonomy suites.
    async def ex(action):
        return True, "ok"
    runner = TaskRunner(executor=ex, limits=TaskLimits(),
                        validator=PlanValidator())
    assert runner is not None
# ═══════════════════════════════════════════════════════════════
# GENERIC MULTI-STEP goal (non-sensitive) — REQUIRED extension
# ═══════════════════════════════════════════════════════════════

def test_multi_step_firefox_then_file_manager_success(harness):
    """'Open Firefox and then open the file manager' → plan with 2 steps
    → step 1 execute+verify → step 2 execute+verify → SUCCESS."""
    planner = FakePlanner(plans={
        "firefox and then open the files": [
            {"action": "desktop_open", "params": {"app": "firefox"}},
            {"action": "desktop_open", "params": {"app": "files"}},
        ]})
    dispatch = FakeDispatch(script={
        "desktop_open": (True, "Opened it"),
    })
    b = make_brain(harness, decision=FakeDecisionEngine(),
                   planner=planner, dispatch=dispatch)
    res = run(b.process_command("Open Firefox and then open the file manager"))

    assert res.task_status == "SUCCESS"
    assert res.verified is True
    params = [c["params"].get("app") for c in dispatch.calls]
    assert params == ["firefox", "files"]               # order preserved
    assert res.actions_succeeded == 2
    assert res.actions_failed == 0
    assert res.actions_executed == 2
    # The spoken summary reflects the verified count.
    assert "2" in (res.response or "")


async def _desktop_open_per_app(action):
    """desktop_open that fails ONLY for the 'files' app (not available)."""
    app = action.get("params", {}).get("app", "")
    if app == "files":
        return False, "app 'files' is not available on this system"
    return True, f"Opened {app}"


def test_multi_step_step2_failure_replans_to_alternative(harness):
    """step 1 verified → step 2 fails (capability unavailable) → recovery
    re-plans an alternative step → verified completion."""
    planner = FakePlanner(
        plans={
            "firefox and then open the files": [
                {"action": "desktop_open", "params": {"app": "firefox"}},
                {"action": "desktop_open", "params": {"app": "files"}},
            ]},
        replans={
            "desktop_open": [
                {"action": "desktop_open", "params": {"app": "thunar"}}],
        })
    dispatch = FakeDispatch(script={"desktop_open": _desktop_open_per_app})
    b = make_brain(harness, decision=FakeDecisionEngine(),
                   planner=planner, dispatch=dispatch)
    res = run(b.process_command("Open Firefox and then open the file manager"))

    assert res.task_status == "SUCCESS"
    assert res.verified is True
    apps = [c["params"].get("app") for c in dispatch.calls]
    assert apps[0] == "firefox"          # step 1 verified
    assert apps[1] == "files"            # original step 2 attempted
    assert apps[2] == "thunar"           # alternative step after replan
    assert res.actions_succeeded >= 2
    assert res.actions_failed >= 1        # the failed attempt is reported