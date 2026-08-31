"""
Regression tests for the 2026-08-30 turn-hardening fixes.

Covers the three runtime blockers:

  BLOCKER 1 - BAD STT TRANSCRIPT CAN REACH THE PLANNER
    A. "Hello dear" (conf -1.168) -> conversational response, NEVER
       planner / tool execution.
    B. Low-confidence nonsense -> clarification, no tool execution.
    C. Normal greeting -> conversational response.
    D. Planner type_text echo of the transcript is blocked.

  BLOCKER 2 - TTS/LISTEN GATE
    E. Next command after TTS works with NO stale backlog.
    F. TTS cannot permanently hold the listening gate (backstop).

  BLOCKER 3 - EXPENSIVE PERCEPTION
    G. "open firefox" -> direct tool path, no unnecessary perception.
    H. "what is on my screen?" -> perception path (OCR).
    I. Ordinary conversation on the LLM path skips perception.

These tests use fakes; no physical microphone, GPU, or OCR model needed.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import voice.command_listener as CL
from tests.test_command_listener import (
    FakeAudioManager,
    FakeVAD,
    FakeWhisper,
    tone,
    silence,
    _make_listener,
    _collect,
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
# BLOCKER 1: bad transcripts never reach the planner / tools
# ================================================================

def test_hello_dear_never_reaches_planner_or_tools(monkeypatch):
    """The exact runtime-log failure: Whisper -> "Hello dear"
    (confidence=-1.168) must become a CONVERSATIONAL response, never
    get_time + type_text desktop actions."""
    brain = _make_brain()
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)

    result = asyncio.run(brain.process_command(
        "Hello dear", stt_confidence=-1.168, audio_duration_ms=1500.0))

    assert result.path == "CONVERSATION", (
        "a greeting must be answered conversationally, not routed to tools")
    assert result.actions_executed == 0
    assert result.used_llm is False
    assert result.response  # a real spoken response exists


def test_hello_dear_gated_even_without_confidence(monkeypatch):
    """Defense in depth: even with NO confidence evidence, a greeting
    phrase must never reach the planner."""
    brain = _make_brain()
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)

    result = asyncio.run(brain.process_command("hello dear"))
    assert result.path == "CONVERSATION"
    assert result.actions_executed == 0


def test_low_confidence_nonsense_no_tool_execution(monkeypatch):
    """Low-confidence nonsense must become a clarification - never a
    desktop action."""
    brain = _make_brain()
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)

    result = asyncio.run(brain.process_command(
        "flurble wibble", stt_confidence=-1.2, audio_duration_ms=900.0))

    assert result.path in ("CLARIFICATION", "REJECTED_TRANSCRIPT")
    assert result.actions_executed == 0
    assert result.used_llm is False
    assert result.response


def test_normal_greeting_conversational_response(monkeypatch):
    """A normal greeting gets a conversational response."""
    brain = _make_brain()
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    monkeypatch.setattr(brain, "_dispatch_and_verify", _fail_dispatch)

    result = asyncio.run(brain.process_command("hello"))
    assert result.path == "CONVERSATION"
    assert result.actions_executed == 0
    assert result.used_llm is False
    assert result.response


def test_planner_type_text_echo_blocked():
    """A planner-generated type_text that merely echoes the transcript
    (planner hallucination) is blocked; an explicit type request is not."""
    from agent.brain import AgentBrain

    echo_action = {"action": "type_text", "params": {"text": "Hello dear"}}
    assert not AgentBrain._planner_action_allowed("Hello dear", echo_action)

    explicit_action = {"action": "type_text", "params": {"text": "hello world"}}
    assert AgentBrain._planner_action_allowed("type hello world", explicit_action)


# ================================================================
# BLOCKER 3: perception is demand-driven
# ================================================================

class _FakeDecisionEngine:
    """Deterministic fake decision engine for routing tests."""

    def __init__(self, direct_action=None, needs_llm=False):
        self._direct_action = direct_action
        self._needs_llm = needs_llm

    async def decide(self, text, **kwargs):
        from core.decision_engine import Decision, DecisionPath
        if self._direct_action is not None:
            return Decision(
                path=DecisionPath.DIRECT_EXECUTION,
                needs_llm=False,
                action=self._direct_action,
                response="Opening Firefox.",
            )
        if self._needs_llm:
            return Decision(path=DecisionPath.LLM, needs_llm=True)
        return Decision(path=DecisionPath.LLM, needs_llm=True)


class _FakeDispatcher:
    def __init__(self):
        self.executed = []

    async def execute(self, action):
        self.executed.append(action)
        return "ok"


class _FakePerception:
    def __init__(self):
        self.calls = []

    async def perceive(self, include_ocr=True, **kwargs):
        self.calls.append(include_ocr)

        class Ctx:
            window_title = "Test"
            a11y_available = False
            ocr_used = include_ocr
            compact_summary = "Window: Test"
        return Ctx()


def test_open_firefox_direct_tool_path_no_perception(monkeypatch):
    """'open firefox' must take the DIRECT tool path with NO perception
    (no OCR, no screen capture)."""
    brain = _make_brain()
    brain._decision_engine = _FakeDecisionEngine(
        direct_action={"action": "desktop_open", "params": {"app": "firefox"}})
    brain._dispatcher = _FakeDispatcher()
    brain._verifier = None
    brain._learning = None
    monkeypatch.setattr(brain, "_perceive", _fail_perceive)
    monkeypatch.setattr(brain, "_plan", _fail_plan)
    # Skip OS-level verification (pgrep settle-wait) for speed.
    async def _ok_verify(*a, **k):
        return True
    monkeypatch.setattr(brain, "_verify", _ok_verify)
    monkeypatch.setattr(brain, "_check_already_running", lambda action: "")

    result = asyncio.run(brain.process_command(
        "open firefox", stt_confidence=-0.5, audio_duration_ms=1500.0))

    assert result.path == "DIRECT_EXECUTION"
    assert result.actions_executed == 1
    assert brain._dispatcher.executed == [
        {"action": "desktop_open", "params": {"app": "firefox"}}]


def test_screen_question_perception_path(monkeypatch):
    """'what is on my screen?' must take the perception path with OCR."""
    brain = _make_brain()
    brain._decision_engine = _FakeDecisionEngine(needs_llm=True)
    perception = _FakePerception()
    brain._perception = perception
    brain._planner = None
    brain._verifier = None
    brain._learning = None

    async def _fake_response(text, perception_ctx, result):
        return "Here is what I see."
    monkeypatch.setattr(brain, "_generate_response", _fake_response)

    result = asyncio.run(brain.process_command(
        "what is on my screen?", stt_confidence=-0.8, audio_duration_ms=2000.0))

    assert perception.calls == [True], (
        "vision request must invoke perception WITH OCR")
    assert result.used_llm is True


def test_llm_path_conversation_skips_perception(monkeypatch):
    """Ordinary conversation on the LLM path ('tell me a joke') must NOT
    pay the perception cost."""
    brain = _make_brain()
    brain._decision_engine = _FakeDecisionEngine(needs_llm=True)
    perception = _FakePerception()
    brain._perception = perception
    brain._planner = None
    brain._verifier = None
    brain._learning = None

    async def _fake_response(text, perception_ctx, result):
        return "Here is one."
    monkeypatch.setattr(brain, "_generate_response", _fake_response)

    result = asyncio.run(brain.process_command(
        "tell me a joke", stt_confidence=-0.5, audio_duration_ms=1500.0))

    assert perception.calls == [], (
        "ordinary conversation must not invoke perception")
    # 2026-08-30: "tell me a joke" is now recognized as small talk by
    # the intent authorizer and answered conversationally (cheaper and
    # more natural than the LLM path). Either way it must NOT invoke
    # perception and must NOT execute tools.
    assert result.used_llm is True or result.path == "CONVERSATION"

def test_no_stale_backlog_after_long_tts_pause(monkeypatch):
    """After a long TTS/THINK pause (20s of audio while the gate is
    closed), the next real command must be captured EXACTLY ONCE with no
    stale-audio contamination and no 10-20s drain."""
    async def impl():
        am = FakeAudioManager()
        whisper = FakeWhisper(text="open youtube")
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(whisper)

        # Simulate the TTS guard cycle: pause, 20s of audio accumulates
        # (TTS + THINK), then resume.
        cl.pause_listening()
        assert cl._listen_enabled.is_set() is False
        am.feed(silence(20.0))
        cl.resume_listening()
        assert cl._listen_enabled.is_set() is True
        assert cl._drain_requested is True

        # Give the streaming loop a moment to process the drain request
        # (which must skip ~0 samples because the gate loop consumed the
        # stale audio), then feed the next real command.
        async def feed():
            await asyncio.sleep(0.05)
            am.feed(tone(1.0, freq=200.0, amp=0.25))
            am.feed(silence(2.5))

        events = await _collect(cl, feed)
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, (
            "next command after TTS must be captured exactly once "
            "(stale 20s backlog must never be transcribed)")
        assert finals[0].text == "open youtube"

    asyncio.run(impl())


def test_tts_cannot_permanently_hold_listen_gate(monkeypatch):
    """The gate backstop: if any path holds the listen gate closed longer
    than GATE_MAX_HOLD_S, the streaming loop force-resumes it."""
    async def impl():
        am = FakeAudioManager()
        monkeypatch.setattr(CL, "audio_manager", am)
        monkeypatch.setattr(CL, "unified_vad", FakeVAD())
        cl = _make_listener(FakeWhisper(text="hi"))

        cl.pause_listening()
        assert cl._listen_enabled.is_set() is False

        # Shrink the backstop so the test is fast.
        monkeypatch.setattr(CL, "GATE_MAX_HOLD_S", 0.5)

        async def feed():
            for _ in range(10):
                am.feed(silence(0.3))
                await asyncio.sleep(0.1)

        async def consume():
            async for _ev in cl.stream_utterances():
                pass

        consumer = asyncio.create_task(consume())
        feeder = asyncio.create_task(feed())
        # Wait for the backstop to trip (checked every 1s in the loop).
        for _ in range(40):
            await asyncio.sleep(0.1)
            if cl._listen_enabled.is_set():
                break
        feeder.cancel()
        consumer.cancel()
        await asyncio.gather(feeder, consumer, return_exceptions=True)

        assert cl._listen_enabled.is_set() is True, (
            "TTS must never permanently hold the listening gate")
        assert cl._gate_closed_at is None

    asyncio.run(impl())


def test_gate_hold_exceeded_helper():
    """_gate_hold_exceeded() reflects the tracked gate-closed time."""
    cl = CL.CommandListener()
    assert cl._gate_hold_exceeded() is False  # gate open
    cl.pause_listening()
    assert cl._gate_hold_exceeded() is False  # just closed
    cl._gate_closed_at = time.monotonic() - (CL.GATE_MAX_HOLD_S + 1.0)
    assert cl._gate_hold_exceeded() is True
    cl.resume_listening()
    assert cl._gate_hold_exceeded() is False


# ================================================================
# Intent gate unit tests (the gate itself)
# ================================================================

def test_intent_gate_verdicts():
    """The gate combines transcript quality + speech evidence + confidence."""
    from nlp.intent_gate import evaluate_intent

    # The exact runtime-log failure case.
    v = evaluate_intent("Hello dear", stt_confidence=-1.168,
                        audio_duration_ms=1500.0)
    assert v.mode == "conversational"
    assert v.tool_execution_allowed is False

    # Low-confidence nonsense -> uncertain (clarification, no tools).
    v = evaluate_intent("flurble wibble", stt_confidence=-1.2,
                        audio_duration_ms=900.0)
    assert v.mode == "uncertain"
    assert v.tool_execution_allowed is False

    # A real command with healthy confidence is allowed.
    v = evaluate_intent("open firefox", stt_confidence=-0.5,
                        audio_duration_ms=1500.0)
    assert v.mode == "command"
    assert v.tool_execution_allowed is True

    # A real command with short audio but clear structure is allowed.
    v = evaluate_intent("open chrome", stt_confidence=-0.7,
                        audio_duration_ms=400.0)
    assert v.tool_execution_allowed is True

    # Greeting with a command verb is NOT a greeting ("hey open firefox").
    v = evaluate_intent("hey open firefox", stt_confidence=-0.5,
                        audio_duration_ms=1500.0)
    assert v.mode == "command"
    assert v.tool_execution_allowed is True

    # Typed/internal input (no confidence evidence) is trusted.
    v = evaluate_intent("summarize this document")
    assert v.tool_execution_allowed is True


def test_intent_gate_never_blocks_real_commands():
    """Regression guard: the gate must not swallow valid commands."""
    from nlp.intent_gate import transcript_allows_tool_execution
    for cmd in ("open firefox", "what is on my screen?", "play some music",
                "volume up", "how are you", "what time is it",
                "close the browser", "search for python tutorials"):
        assert transcript_allows_tool_execution(cmd, -0.6, 1500.0), cmd
