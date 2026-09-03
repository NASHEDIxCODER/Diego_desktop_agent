"""
Regression tests — runtime bug-fix pass (2026-09-03).

Each test covers a bug REPRODUCED in a live runtime run (debug/
live_pipeline_run.py + debug/repro_pause_resume.py), not a hypothetical:

1. "find my Diego project" → normalized "search my project" was
   routed to web_search DIRECT_EXECUTION (Diego literally opened
   google.com/search?q=my+project). Local project references must
   never be web-searched.
2. "find the project I was working on" slipped past the local-knowledge
   pattern list (only "worked on" was covered, not "was working on").
3. LOCAL_KNOWLEDGE transcripts were still actionable, so the planner
   ran and generated a web_search action for them.
4. Standalone "resume" was hijacked by pronoun resolution into
   "continue <last goal>", sending a media command into the planner
   (9s of doomed replans ending in "planner produced the same plan
   repeatedly").
5. "pause" failed verification but Diego still said "Paused." — an
   unverified completion claim spoken to the user (false success).
6. "tell me a joke" was answered "Hey." (greeting fallback) and later
   from a spurious local-document match; small talk must reach the LLM.
"""

import asyncio

import pytest


# ── 1 + 2: local-knowledge classification ──────────────────────────

@pytest.mark.parametrize("spoken,normalized", [
    ("find my Diego project", "search my project"),
    ("Diego, find the project I was working on",
     "search the project i was working on"),
    ("find the project I was using",
     "search the project i was using"),
])
def test_local_project_phrases_classified_local_knowledge(spoken, normalized):
    """Local project references must classify as LOCAL_KNOWLEDGE and must
    NOT be actionable (they are answered from the local index + LLM)."""
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent, _match_local_knowledge

    norm = command_normalizer.normalize(spoken)
    assert norm == normalized
    assert _match_local_knowledge(norm), f"pattern miss: {norm!r}"

    auth = authorize_intent(norm)
    assert auth.category.value == "LOCAL_KNOWLEDGE"
    assert auth.actionable is False
    assert auth.llm_allowed is True


def test_explicit_web_search_still_classified_search():
    """Guard against over-blocking: genuine web searches still route to
    SEARCH_REQUEST."""
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent

    auth = authorize_intent(command_normalizer.normalize("search cats"))
    assert auth.category.value == "SEARCH_REQUEST"
    assert auth.actionable is True


# ── 3: decision engine must not direct-execute web_search for local refs ──

def test_decision_engine_blocks_web_search_for_local_knowledge():
    """L3 direct execution must not return a web_search action for a
    local-knowledge transcript (reproduced live: opened a Google search
    for "my project")."""
    from core.decision_engine import decision_engine

    decision_engine._ensure_wired()
    decision = asyncio.run(
        decision_engine._check_direct_execution("search my project"))
    assert decision is None, (
        "web_search direct execution must be blocked for local-knowledge "
        "requests")


def test_decision_engine_still_executes_plain_web_search():
    """Guard against over-blocking: plain "search firefox" must still be
    direct-executable."""
    from core.decision_engine import decision_engine

    decision_engine._ensure_wired()
    decision = asyncio.run(
        decision_engine._check_direct_execution("search firefox"))
    assert decision is not None
    assert decision.action is not None
    assert decision.action.get("action") == "web_search"


# ── 4: "resume" must not become a task continuation ────────────────

def test_resume_not_hijacked_into_task_continuation():
    """Standalone "resume" is a media command; pronoun resolution must
    return it unchanged even when a goal was tracked (reproduced live:
    'resume' → 'continue search my project' → planner)."""
    from agent.conversation_memory import conv_memory

    conv_memory.track_goal("search my project")
    conv_memory.track_goal("open firefox")
    try:
        resolved = conv_memory._resolve_pronouns("resume")
        assert resolved == "resume"
        # "continue" still resolves (the intended task-continuation cue).
        resolved_continue = conv_memory._resolve_pronouns("continue")
        assert resolved_continue.startswith("continue ")
    finally:
        conv_memory.clear()


# ── 5: false-success guard ──────────────────────────────────────────

def test_failed_single_action_never_reports_success():
    """A fully-failed single action must NOT produce a success claim
    ("Paused."/"Resumed."/"Mostly done") — reproduced live: 'pause'
    failed verification yet the response was 'Paused.'."""
    from agent.brain import AgentBrain, CommandResult

    result = CommandResult()
    result.actions_executed = 1
    result.actions_succeeded = 0
    result.actions_failed = 1
    response = AgentBrain._default_response(result)
    lowered = response.lower()
    assert "paused" not in lowered
    assert "resumed" not in lowered
    assert "mostly done" not in lowered
    assert "couldn't" in lowered or "could not" in lowered


def test_partial_failure_still_reports_issues():
    """Multi-step partial failure keeps an honest 'issues' phrasing."""
    from agent.brain import AgentBrain, CommandResult

    result = CommandResult()
    result.actions_executed = 3
    result.actions_succeeded = 2
    result.actions_failed = 1
    response = AgentBrain._default_response(result)
    assert "issues" in response.lower()


# ── 6: small talk reaches the LLM, never a canned non-answer ───────

def test_small_talk_is_conversational_not_actionable():
    """'tell me a joke' must classify CONVERSATIONAL (llm-only) so the
    brain falls through to the LLM instead of a greeting fallback or a
    spurious local-document match (both reproduced live)."""
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent

    auth = authorize_intent(command_normalizer.normalize("tell me a joke"))
    assert auth.category.value == "CONVERSATIONAL"
    assert auth.actionable is False
    assert auth.llm_allowed is True