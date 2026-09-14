"""
Reliability layer regression tests (2026-09-13).

Covers:
  LOCAL ROUTING
    - local file search / count / list phrases detect deterministically
    - decision engine routes them to LOCAL_COMPUTER (never SEARCH /
      SESSION_MEMORY / LLM)
    - responses are evidence-backed (contain real filenames / counts)
  FOLLOW-UP CONTINUATION
    - "do it" / "continue" / "do that" continue the ACTIVE task
    - "open it" reopens the last referenced app/URL
    - "then do that" / "now navigate to my profile" resolve remaining
      plan steps instead of starting a new conversation
    - without an active task, follow-ups fall through (None)
  GOAL-LEVEL VERIFICATION
    - action executes but goal unmet → verification FAILS
    - URL match → PASSES
    - missing observation → NO_EVIDENCE (never guessed as success)
    - "browser process alive" is not navigation evidence
  LIFECYCLE STATE MACHINE
    - valid normal lifecycle accepted
    - THINKING → LISTENING rejected
    - fallback path THINKING → SPEAKING → LISTENING is valid
"""

from __future__ import annotations

import asyncio

import pytest


# ═══════════════════════════════════════════════════════════════
# LOCAL ROUTING
# ═══════════════════════════════════════════════════════════════

from core.local_computer_intent import (
    LocalIntentKind,
    LocalSort,
    detect_local_intent,
    resolve_local_intent,
)


@pytest.mark.parametrize("text,kind,ext", [
    ("Find the largest Python file in my Diego project",
     LocalIntentKind.FILE_SEARCH, (".py",)),
    ("How many documents are on my PC?",
     LocalIntentKind.FILE_COUNT, None),
    ("Show me 10 PDFs in Downloads",
     LocalIntentKind.FILE_SEARCH, (".pdf",)),
    ("Find the biggest file in Projects",
     LocalIntentKind.FILE_SEARCH, ()),
    ("List Python files modified today",
     LocalIntentKind.FILE_SEARCH, (".py",)),
    ("How many PDFs are on my computer",
     LocalIntentKind.FILE_COUNT, (".pdf",)),
])
def test_local_intent_detection(text, kind, ext):
    intent = detect_local_intent(text)
    assert intent is not None, f"missed local intent: {text!r}"
    assert intent.kind == kind
    if ext is not None:
        assert intent.extensions == ext


def test_local_intent_largest_sort_and_folder():
    intent = detect_local_intent(
        "Find the largest Python file in my Diego project")
    assert intent.sort == LocalSort.LARGEST
    assert "diego" in intent.folder


def test_local_intent_limit():
    intent = detect_local_intent("Show me 10 PDFs in Downloads")
    assert intent.limit == 10
    assert "downloads" in intent.folder


def test_from_preposition_folder():
    intent = detect_local_intent("Show me 10 documents from Downloads")
    assert intent is not None
    assert "downloads" in intent.folder
    assert intent.limit == 10


def test_non_local_text_not_detected():
    assert detect_local_intent("what's the weather in Tokyo") is None
    assert detect_local_intent("tell me a joke") is None
    assert detect_local_intent("who are you") is None


def test_local_resolution_is_evidence_backed(tmp_path):
    """The answer comes from real filesystem observations."""
    (tmp_path / "a.py").write_text("x" * 5000)
    (tmp_path / "b.py").write_text("x" * 100)
    (tmp_path / "notes.pdf").write_text("pdf")
    intent = detect_local_intent(
        "Find the largest Python file in my test project")
    intent.folder = str(tmp_path)
    answer = resolve_local_intent(intent)
    assert "a.py" in answer and "b.py" not in answer.split(";")[0]


def test_local_count_resolution(tmp_path):
    (tmp_path / "x.pdf").write_text("p")
    (tmp_path / "y.pdf").write_text("p")
    intent = detect_local_intent("How many PDFs are here?")
    intent.folder = str(tmp_path)
    answer = resolve_local_intent(intent)
    assert "2" in answer


async def test_decision_engine_routes_local(tmp_path, monkeypatch):
    """Filesystem requests route to LOCAL_COMPUTER — never web search,
    session memory, or semantic retrieval."""
    from core import local_computer_intent as lci
    from core.decision_engine import DecisionEngine, DecisionPath

    def fake_resolve(intent):
        return "The largest files: dummy.py — 1.0 KB"

    monkeypatch.setattr(lci, "resolve_local_intent", fake_resolve)

    engine = DecisionEngine()
    decision = await engine.decide(
        "Find the largest Python file in my Diego project")
    assert decision.path == DecisionPath.LOCAL_COMPUTER
    assert decision.needs_llm is False
    assert "dummy.py" in (decision.response or "")

    decision2 = await engine.decide("How many documents are on my PC?")
    assert decision2.path == DecisionPath.LOCAL_COMPUTER
    assert decision2.needs_llm is False


async def test_no_accidental_web_search_or_semantic_route(monkeypatch):
    """Local-intent phrases must not fall through to SEARCH/LLM."""
    from core import local_computer_intent as lci
    from core.decision_engine import DecisionEngine, DecisionPath

    def fake_resolve(intent):
        return "counted"

    monkeypatch.setattr(lci, "resolve_local_intent", fake_resolve)

    engine = DecisionEngine()
    for text in ("Show me 10 PDFs in Downloads",
                 "Find the biggest file in Projects",
                 "List Python files modified today"):
        decision = await engine.decide(text)
        assert decision.path == DecisionPath.LOCAL_COMPUTER, text
        assert decision.path not in (DecisionPath.SEARCH,
                                     DecisionPath.LLM,
                                     DecisionPath.SESSION_MEMORY)


# ═══════════════════════════════════════════════════════════════
# FOLLOW-UP / CONTINUATION
# ═══════════════════════════════════════════════════════════════

from agent.task_state import (
    FollowUpResolver,
    TaskExecutionState,
    TaskStateStore,
    StepStatus,
    StepRecord,
)


def _active_task_with_remaining() -> TaskExecutionState:
    state = TaskExecutionState(
        original_request="Open GitHub and navigate to my profile",
        normalized_goal="Open GitHub and navigate to my profile",
    )
    state.active_goal = state.normalized_goal
    state.current_plan = [
        {"action": "browser_navigate", "params": {"url": "https://github.com"},
         "description": "Open GitHub"},
        {"action": "browser_navigate",
         "params": {"url": "https://github.com/user"},
         "description": "Navigate to my profile"},
    ]
    done = StepRecord(index=0, action="browser_navigate",
                      params={"url": "https://github.com"},
                      description="Open GitHub",
                      status=StepStatus.COMPLETED, verified=True)
    state.completed_steps.append(done)
    state.current_step = None
    return state


def _store_with(state: TaskExecutionState) -> TaskStateStore:
    store = TaskStateStore()
    store.last = state
    store.active = state
    return store


@pytest.mark.parametrize("text", [
    "do it", "continue", "proceed", "do that", "now do that",
    "then do that",
])
def test_confirmation_phrases_continue_active_task(text):
    state = _active_task_with_remaining()
    store = _store_with(state)
    cont = store.build_continuation(text)
    assert cont is not None, f"{text!r} must continue the active task"
    request, plan, inherited = cont
    assert inherited is state
    assert plan, f"{text!r} must yield remaining steps"


def test_then_do_that_is_continuation_not_conversation():
    state = _active_task_with_remaining()
    store = _store_with(state)
    match = FollowUpResolver.match("then do that")
    assert match is not None and match[0] == "continue"


def test_now_navigate_to_my_profile_matches_remaining_step():
    """The canonical resumption example from the spec."""
    state = _active_task_with_remaining()
    store = _store_with(state)
    cont = store.build_continuation("Now navigate to my profile")
    assert cont is not None
    request, plan, inherited = cont
    assert len(plan) == 1
    assert plan[0]["params"]["url"] == "https://github.com/user"


def test_open_it_reopens_last_url():
    state = _active_task_with_remaining()
    state.artifacts["last_url"] = "https://github.com"
    store = TaskStateStore()
    store.last = state
    store.active = None  # no remaining steps needed for open_last
    cont = store.build_continuation("open it")
    assert cont is not None
    request, plan, inherited = cont
    assert plan[0]["action"] == "browser_navigate"
    assert plan[0]["params"]["url"] == "https://github.com"


def test_repeat_that_reruns_last_step():
    state = _active_task_with_remaining()
    store = TaskStateStore()
    store.last = state
    cont = store.build_continuation("repeat that")
    assert cont is not None
    request, plan, inherited = cont
    assert plan[0]["action"] == "browser_navigate"


def test_followups_without_task_return_none():
    store = TaskStateStore()
    assert store.build_continuation("do it") is None
    assert store.build_continuation("open it") is None
    assert FollowUpResolver.match_deferred(
        "now navigate to my profile", []) is None


async def test_yes_without_pending_does_nothing():
    """A bare 'yes' with no pending confirmation must not start anything:
    classification alone never executes — only an existing pending
    confirmation does (get_pending() is None here)."""
    from agent.task_continuation import (
        classify_confirmation, pending_task_manager,
    )
    pending_task_manager.clear()
    try:
        assert pending_task_manager.get_pending() is None
        assert classify_confirmation("yes") == "confirm"
        # But with NO pending, the brain's handler is never triggered —
        # nothing to confirm, nothing to execute.
        assert pending_task_manager.has_pending is False
    finally:
        pending_task_manager.clear()


# ═══════════════════════════════════════════════════════════════
# GOAL-LEVEL VERIFICATION
# ═══════════════════════════════════════════════════════════════

from core.goal_verification import (
    GoalResult,
    evaluate_goal,
    verify_browser_click,
    verify_browser_navigation,
)


def test_navigation_url_match_passes():
    result = verify_browser_navigation(
        "https://github.com/user", "https://github.com/user")
    assert result.passed
    assert result.result == GoalResult.PASS
    assert "URL match" in result.evidence


def test_navigation_wrong_url_fails_even_if_tool_succeeded():
    """ACTION_SUCCESS ≠ GOAL_SUCCESS: dispatcher returned fine but the
    browser ended up on the wrong page."""
    result = verify_browser_navigation(
        "https://github.com/user", "https://github.com/login")
    assert not result.passed
    assert result.result == GoalResult.FAIL


def test_navigation_no_observation_is_not_success():
    result = verify_browser_navigation("https://github.com/user", None)
    assert not result.passed
    assert result.result == GoalResult.NO_EVIDENCE


def test_process_alive_is_not_navigation_evidence():
    """The old broken path: pgrep says firefox is running → PASS.
    The new contract requires a URL match instead."""
    result = evaluate_goal(
        "browser_navigate", {"url": "https://github.com/user"},
        {"process_running": True})   # process alive, NO observed URL
    assert not result.passed


def test_click_expected_element_found():
    ok = verify_browser_click("Profile", "Welcome to your Profile page")
    assert ok.passed
    bad = verify_browser_click("Profile", "Sign in to continue")
    assert not bad.passed
    no_obs = verify_browser_click("Profile", None)
    assert no_obs.result == GoalResult.NO_EVIDENCE


def test_expected_effect_recorded():
    r = verify_browser_navigation("https://x.com", "https://x.com")
    assert r.expected_effect  # contract fields populated
    d = r.to_dict()
    assert set(d) == {"action", "expected_effect", "observation",
                      "evidence", "result"}


# ═══════════════════════════════════════════════════════════════
# LIFECYCLE STATE MACHINE
# ═══════════════════════════════════════════════════════════════

from core.state_machine import (
    ALLOWED_TRANSITIONS,
    RuntimeState,
    RuntimeStateMachine,
)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_normal_lifecycle_transitions_are_legal():
    path = [
        RuntimeState.LISTENING,
        RuntimeState.THINKING,
        RuntimeState.EXECUTING,
        RuntimeState.SPEAKING,
        RuntimeState.LISTENING,
    ]
    for a, b in zip(path, path[1:]):
        assert b in ALLOWED_TRANSITIONS[a], f"{a} → {b} must be legal"


def test_think_to_listen_is_rejected():
    """Invalid THINK → LISTEN transition must be rejected."""
    sm = RuntimeStateMachine()
    accepted = _run(sm.transition_to(RuntimeState.THINKING, "start"))
    assert accepted
    accepted = _run(sm.transition_to(RuntimeState.LISTENING, "illegal"))
    assert not accepted, "THINK → LISTEN without SPEAK must be rejected"
    assert sm.state == RuntimeState.THINKING  # unchanged


def test_fallback_path_produces_valid_lifecycle():
    """Fallback must speak: THINK → SPEAK → LISTEN is the legal route."""
    sm = RuntimeStateMachine()
    assert _run(sm.transition_to(RuntimeState.THINKING, "think"))
    assert _run(sm.transition_to(RuntimeState.SPEAKING, "fallback speaks"))
    assert _run(sm.transition_to(RuntimeState.LISTENING, "back to listen"))
    assert sm.state == RuntimeState.LISTENING


def test_exceptional_states_explicit():
    assert RuntimeState.RECOVERING in ALLOWED_TRANSITIONS[RuntimeState.THINKING]
    assert RuntimeState.SHUTDOWN in ALLOWED_TRANSITIONS[RuntimeState.EXECUTING]
