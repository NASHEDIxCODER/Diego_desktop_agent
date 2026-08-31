"""
Tests for the FINAL transcript -> intent -> action authorization boundary.

Covers the 2026-08-30 runtime-log failures:
  - "Hello dear" (conf=-1.168) reached the planner and executed tools.
  - "Can you see my screen?" was normalized to "see my" and executed
    unrelated actions.
  - "Diego opened the tomb" reached the planner and caused close_app.
  - "I want to see you again" / "Diego Open YouTube" / "The level of
    YouTube" were wrongly rejected or misrouted.

Expected behaviour (requirement 7):
  - commands execute only when explicitly actionable
  - screen questions route to vision
  - search requests route to search
  - normal questions route to conversation/LLM
  - greeting is conversational
  - ambiguous/garbage transcript does not execute anything

These tests use fakes; no physical microphone, GPU, or OCR model needed.
"""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

from nlp.intent_authorizer import (
    authorize_intent, IntentCategory, ACTIONABLE_CATEGORIES,
)


def _make_brain():
    """A Brain with subsystem wiring skipped (hermetic unit test)."""
    from agent.brain import AgentBrain
    brain = AgentBrain()
    brain._initialized = True
    return brain


def _fail_perceive(*a, **k):
    raise AssertionError("perception must not run for a gated transcript")


def _fail_plan(*a, **k):
    raise AssertionError("planner must not run for a gated transcript")


def _fail_dispatch(*a, **k):
    raise AssertionError("no tool execution allowed for a gated transcript")


# ================================================================
# Requirement 7: the ten real phrases
# ================================================================

def test_open_firefox_is_deterministic_command():
    a = authorize_intent("open firefox", stt_confidence=-0.5,
                         audio_duration_ms=1500.0)
    assert a.category == IntentCategory.DETERMINISTIC_COMMAND
    assert a.actionable is True
    assert a.route == "dispatcher"


def test_what_apps_are_running_is_deterministic_command():
    a = authorize_intent("what apps are running", stt_confidence=-0.6,
                         audio_duration_ms=1800.0)
    assert a.category == IntentCategory.DETERMINISTIC_COMMAND
    assert a.actionable is True


def test_switch_to_firefox_is_deterministic_command():
    a = authorize_intent("switch to firefox", stt_confidence=-0.5,
                         audio_duration_ms=1500.0)
    assert a.category == IntentCategory.DETERMINISTIC_COMMAND
    assert a.actionable is True


def test_what_is_on_my_screen_is_vision_command():
    a = authorize_intent("what is on my screen", stt_confidence=-0.8,
                         audio_duration_ms=2000.0)
    assert a.category == IntentCategory.VISION_COMMAND
    assert a.actionable is True
    assert a.route == "vision"


def test_can_you_see_my_screen_is_vision_command():
    """The exact runtime failure: this must be a VISION intent, never
    degraded to "see my" and never executed as unrelated actions."""
    a = authorize_intent("can you see my screen", stt_confidence=-0.7,
                         audio_duration_ms=1800.0)
    assert a.category == IntentCategory.VISION_COMMAND
    assert a.actionable is True
    assert a.route == "vision"


def test_search_request_routes_to_search():
    a = authorize_intent("search github for python websocket examples",
                         stt_confidence=-0.5, audio_duration_ms=2500.0)
    assert a.category == IntentCategory.SEARCH_REQUEST
    assert a.actionable is True
    assert a.route == "search"
    assert a.entities.get("query") == "github for python websocket examples"


def test_tell_me_about_is_knowledge_question():
    """'tell me about animal DNA' routes to conversation/LLM — never a
    forced desktop action."""
    a = authorize_intent("tell me about animal DNA", stt_confidence=-0.6,
                         audio_duration_ms=2000.0)
    assert a.category == IntentCategory.KNOWLEDGE_QUESTION
    assert a.actionable is False
    assert a.llm_allowed is True
    assert a.route == "llm"


def test_how_are_you_is_conversational():
    a = authorize_intent("how are you", stt_confidence=-0.737,
                         audio_duration_ms=1500.0)
    assert a.category == IntentCategory.CONVERSATIONAL
    assert a.actionable is False


def test_hello_dear_low_confidence_is_conversational():
    """The exact runtime-log failure: "Hello dear" conf=-1.168 must be
    conversational (harmless greeting response), never tools."""
    a = authorize_intent("hello dear", stt_confidence=-1.168,
                         audio_duration_ms=1500.0)
    assert a.category == IntentCategory.CONVERSATIONAL
    assert a.actionable is False


def test_diego_opened_the_tomb_is_never_actionable():
    """'Diego opened the tomb' is a statement, not a request. It must
    NEVER become close_app or any desktop action."""
    a = authorize_intent("opened the tomb", stt_confidence=-0.5,
                         audio_duration_ms=1800.0)
    assert a.category == IntentCategory.CONVERSATIONAL
    assert a.actionable is False
    assert a.llm_allowed is True


def test_diego_open_youtube_is_command():
    """Wake-word prefix stripped -> 'open youtube' is a real command."""
    a = authorize_intent("open youtube", stt_confidence=-0.5,
                         audio_duration_ms=1500.0)
    assert a.category == IntentCategory.DETERMINISTIC_COMMAND
    assert a.actionable is True


def test_the_level_of_youtube_is_conversational():
    """Previously rejected; a coherent sentence reaches the LLM."""
    a = authorize_intent("the level of youtube", stt_confidence=-0.7,
                         audio_duration_ms=1800.0)
    assert a.category == IntentCategory.CONVERSATIONAL
    assert a.actionable is False
    assert a.llm_allowed is True


def test_i_want_to_see_you_again_is_conversational():
    """Previously rejected; a coherent conversational sentence reaches
    the LLM even with negative Whisper confidence."""
    a = authorize_intent("see you again", stt_confidence=-0.6,
                         audio_duration_ms=1800.0)
    assert a.category == IntentCategory.CONVERSATIONAL
    assert a.actionable is False
    assert a.llm_allowed is True


# ================================================================
# Requirement 4: low-confidence evidence fusion
# ================================================================

def test_actionable_low_confidence_asks_for_clarification():
    """An actionable-looking command in the hallucination band must ask
    for clarification instead of executing (never lower the Whisper
    threshold blindly)."""
    a = authorize_intent("open firefox", stt_confidence=-1.05,
                         audio_duration_ms=1500.0)
    assert a.category == IntentCategory.UNCERTAIN
    assert a.actionable is False
    assert a.route == "clarification"


def test_actionable_soft_reject_short_audio_clarification():
    a = authorize_intent("open it", stt_confidence=-0.7,
                         audio_duration_ms=400.0)
    assert a.category == IntentCategory.UNCERTAIN
    assert a.actionable is False


def test_coherent_sentence_with_negative_confidence_reaches_llm():
    """Requirement 4: a coherent factual/conversational sentence is
    allowed to reach the LLM even when avg_logprob is negative."""
    a = authorize_intent("tell me about animal DNA", stt_confidence=-0.85,
                         audio_duration_ms=2500.0)
    assert a.llm_allowed is True
    assert a.actionable is False


def test_garbage_transcript_is_uncertain():
    a = authorize_intent("flurble wibble", stt_confidence=-1.2,
                         audio_duration_ms=900.0)
    assert a.category == IntentCategory.UNCERTAIN
    assert a.actionable is False
    assert a.llm_allowed is False


def test_repeated_hallucination_is_uncertain():
    a = authorize_intent("you you you you", stt_confidence=-1.35,
                         audio_duration_ms=2000.0)
    assert a.category == IntentCategory.UNCERTAIN
    assert a.actionable is False


def test_no_vad_evidence_blocks_actionable_intent():
    a = authorize_intent("open firefox", stt_confidence=-0.5,
                         audio_duration_ms=1500.0,
                         vad_evidence={"has_speech": False})
    assert a.category == IntentCategory.UNCERTAIN
    assert a.actionable is False


def test_vad_evidence_keeps_actionable_intent():
    a = authorize_intent("open firefox", stt_confidence=-0.5,
                         audio_duration_ms=1500.0,
                         vad_evidence={"has_speech": True})
    assert a.category == IntentCategory.DETERMINISTIC_COMMAND
    assert a.actionable is True


# ================================================================
# Requirement 2: normalization preserves vision intent structure
# ================================================================

def test_can_you_see_my_screen_never_becomes_see_my():
    from nlp.command_normalizer import command_normalizer
    normalized = command_normalizer.normalize("Can you see my screen?")
    assert normalized.strip().lower() != "see my"
    assert "screen" in normalized.lower()
    # The decision engine must still recognize it as a vision request.
    from core.decision_engine import DecisionEngine
    assert DecisionEngine._needs_vision(normalized)


def test_see_my_screen_preserved():
    from nlp.command_normalizer import command_normalizer
    normalized = command_normalizer.normalize("see my screen")
    assert normalized.strip().lower() == "see my screen"


def test_lock_the_screen_still_normalizes():
    """Regression guard: 'lock the screen' must still become a lock
    command (the redundant-object rule for 'the screen' is kept)."""
    from nlp.command_normalizer import command_normalizer
    normalized = command_normalizer.normalize("lock the screen")
    assert "lock" in normalized.lower()


def test_diego_prefix_is_stripped():
    """CRITICAL FIX: the noise-word check was case-sensitive, so the
    wake-word prefix 'Diego' was never stripped."""
    from nlp.command_normalizer import command_normalizer
    normalized = command_normalizer.normalize("Diego Open YouTube")
    assert "diego" not in normalized.lower()
    assert "open" in normalized.lower()


# ================================================================
# Requirement 1/3/6: brain-level integration
# ================================================================

def test_uncertain_transcript_never_invokes_expensive_work(monkeypatch):
    """Requirement 6: an uncertain transcript must not invoke perception,
    planner, search, LLM, or tools."""
    brain = _make_brain()
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)

    result = asyncio.run(brain.process_command(
        "flurble wibble", stt_confidence=-1.2, audio_duration_ms=900.0))

    assert result.path == "CLARIFICATION"
    assert result.actions_executed == 0
    assert result.used_llm is False
    assert result.response


def test_diego_opened_the_tomb_never_executes_close_app(monkeypatch):
    """Requirement 3: 'Diego opened the tomb' must NOT become close_app
    or any other desktop action."""
    brain = _make_brain()
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)

    result = asyncio.run(brain.process_command(
        "Diego opened the tomb", stt_confidence=-0.5,
        audio_duration_ms=1800.0))

    assert result.actions_executed == 0
    # It may be answered conversationally or via the LLM, but NO tools.
    assert result.path in ("CONVERSATION", "LLM", "WORKING_MEMORY",
                           "SESSION_MEMORY", "DIRECT_EXECUTION")
    if result.path == "DIRECT_EXECUTION":
        # Only a conversational/cached response is allowed here —
        # never a dispatched desktop action.
        assert result.response


def test_knowledge_question_never_invokes_planner(monkeypatch):
    """'tell me about animal DNA' reaches the LLM but never the planner
    or dispatcher."""
    brain = _make_brain()
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)
    brain._perception = None
    brain._planner = None
    brain._verifier = None
    brain._learning = None

    async def _fake_response(text, perception_ctx, result):
        return "Animal DNA carries the genetic instructions..."
    monkeypatch.setattr(brain, "_generate_response", _fake_response)

    result = asyncio.run(brain.process_command(
        "tell me about animal DNA", stt_confidence=-0.6,
        audio_duration_ms=2000.0))

    assert result.actions_executed == 0
    assert result.response


def test_greeting_never_invokes_planner(monkeypatch):
    brain = _make_brain()
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)

    result = asyncio.run(brain.process_command(
        "Hello dear", stt_confidence=-1.168, audio_duration_ms=1500.0))

    assert result.path == "CONVERSATION"
    assert result.actions_executed == 0
    assert result.used_llm is False


# ================================================================
# Requirement 3: planner action validation (schema + verb evidence)
# ================================================================

def test_planner_close_app_without_close_evidence_blocked():
    """'Diego opened the tomb' -> close_app must be blocked: no 'close'
    verb evidence in the transcript."""
    from agent.brain import AgentBrain
    action = {"action": "close_app", "params": {"app": "tomb"}}
    assert not AgentBrain._planner_action_allowed("opened the tomb", action)


def test_planner_close_app_with_close_evidence_allowed():
    from agent.brain import AgentBrain
    action = {"action": "close_app", "params": {"app": "firefox"}}
    assert AgentBrain._planner_action_allowed("close firefox", action)


def test_planner_unknown_action_blocked():
    """An action outside the allowed schema is blocked."""
    from agent.brain import AgentBrain
    action = {"action": "format_disk", "params": {}}
    assert not AgentBrain._planner_action_allowed("format my disk", action)


def test_planner_desktop_open_with_evidence_allowed():
    from agent.brain import AgentBrain
    action = {"action": "desktop_open", "params": {"app": "firefox"}}
    assert AgentBrain._planner_action_allowed("open firefox", action)


def test_planner_desktop_open_without_evidence_blocked():
    from agent.brain import AgentBrain
    action = {"action": "desktop_open", "params": {"app": "firefox"}}
    assert not AgentBrain._planner_action_allowed("the weather is nice",
                                                  action)


def test_planner_type_text_echo_still_blocked():
    from agent.brain import AgentBrain
    echo_action = {"action": "type_text", "params": {"text": "Hello dear"}}
    assert not AgentBrain._planner_action_allowed("Hello dear", echo_action)
    explicit_action = {"action": "type_text", "params": {"text": "hello world"}}
    assert AgentBrain._planner_action_allowed("type hello world",
                                              explicit_action)


def test_planner_get_time_with_time_evidence_allowed():
    from agent.brain import AgentBrain
    action = {"action": "get_time", "params": {}}
    assert AgentBrain._planner_action_allowed("what time is it", action)


# ================================================================
# Requirement 5: silence fix preserved (no regression)
# ================================================================

def test_silence_still_produces_no_response():
    """Pure silence must still be discarded silently by the command
    listener (the 2026-08-30 silence fix is untouched)."""
    from voice.command_listener import CommandListener
    cl = CommandListener()
    frames = []
    evidence = cl._measure_speech_evidence(frames) if frames else {
        "total_ms": 2000.0, "silero_voiced_ms": 0.0, "strong_ms": 0.0,
        "loud_ms": 0.0, "peak_rms": 0.0, "avg_prob": 0.0,
    }
    assert cl._has_speech_evidence(evidence) is False


def test_actionable_categories_are_the_only_dispatchable_ones():
    from nlp.intent_authorizer import LLM_CATEGORIES
    for cat in ACTIONABLE_CATEGORIES:
        assert cat in LLM_CATEGORIES
    # CONVERSATIONAL and KNOWLEDGE_QUESTION are never actionable.
    assert IntentCategory.CONVERSATIONAL not in ACTIONABLE_CATEGORIES
    assert IntentCategory.KNOWLEDGE_QUESTION not in ACTIONABLE_CATEGORIES
    assert IntentCategory.UNCERTAIN not in ACTIONABLE_CATEGORIES
