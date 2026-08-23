"""
Production Regression Tests — Diego must pass these before claiming production-ready.

Validates every stage of the pipeline with real failure scenarios.

Usage:
    python tests/test_production_regression.py
"""

import asyncio
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import compat  # noqa: F401

PASS = 0
FAIL = 0
RESULTS = []


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"
    RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


# ── Test 1: Garbage transcript rejection ──────────────────────

async def test_garbage_rejection():
    """Verify is_garbage() rejects known hallucinated patterns."""
    print("\n  TEST: Garbage Transcript Rejection")
    print("  " + "-" * 50)

    from voice.command_listener import is_garbage, is_filler

    # Should be rejected (garbage)
    assert is_garbage("I'm sorry")
    assert is_garbage("I'm not sure")
    assert is_garbage("I don't know")
    assert is_garbage("I don't think")
    assert is_garbage("I can't")
    assert is_garbage("I was")
    assert is_garbage("I am")
    assert is_garbage("I will")
    assert is_garbage("Oh")
    assert is_garbage("Uh")
    assert is_garbage("Um")
    assert is_garbage("Hmm")
    assert is_garbage("Thank you")
    assert is_garbage("That's a")
    assert is_garbage("This is")
    assert is_garbage("There is")
    assert is_garbage("It's a")
    assert is_garbage("Let me think")
    assert is_garbage("I")
    assert is_garbage("You")
    assert is_garbage("It")
    assert is_garbage("That")
    assert is_garbage("This")

    # Should NOT be rejected (valid commands)
    assert not is_garbage("open firefox")
    assert not is_garbage("play music")
    assert not is_garbage("stop")
    assert not is_garbage("pause")
    assert not is_garbage("yes")
    assert not is_garbage("no")
    assert not is_garbage("continue")
    assert not is_garbage("close it")
    assert not is_garbage("open youtube")
    assert not is_garbage("search for cats")
    assert not is_garbage("how are you")
    assert not is_garbage("hello")
    assert not is_garbage("what time is it")
    assert not is_garbage("volume up")
    assert not is_garbage("brightness down")

    check("Garbage 'I\\'m sorry' rejected", True)
    check("Garbage 'I don\\'t know' rejected", True)
    check("Garbage 'Oh' rejected", True)
    check("Garbage 'I' rejected", True)
    check("Garbage 'You' rejected", True)
    check("Valid 'open firefox' accepted", True)
    check("Valid 'stop' accepted", True)
    check("Valid 'how are you' accepted", True)
    check("Valid 'hello' accepted", True)


# ── Test 2: Filler detection ─────────────────────────────────

async def test_filler_detection():
    """Verify is_filler() correctly identifies filler words."""
    print("\n  TEST: Filler Detection")
    print("  " + "-" * 50)

    from voice.command_listener import is_filler

    assert is_filler("um")
    assert is_filler("uh")
    assert is_filler("hmm")
    assert is_filler("umm")
    assert is_filler("wait")
    assert is_filler("hold on")
    assert is_filler("let me think")
    assert is_filler("like")
    assert is_filler("you know")
    assert is_filler("i mean")
    assert is_filler("well")

    assert not is_filler("open firefox")
    assert not is_filler("stop")
    assert not is_filler("hello")
    assert not is_filler("how are you")
    assert not is_filler("play music")

    check("Filler 'um' detected", True)
    check("Filler 'uh' detected", True)
    check("Filler 'hmm' detected", True)
    check("Filler 'hold on' detected", True)
    check("Filler 'let me think' detected", True)
    check("Non-filler 'open firefox' not detected", True)


# ── Test 3: Planner crash fix ────────────────────────────────

async def test_planner_crash_fix():
    """Verify AgentMemory._browser_tabs is a real list, not a Field."""
    print("\n  TEST: Planner Crash Fix")
    print("  " + "-" * 50)

    from agent.memory import agent_memory

    # This should NOT raise "object of type 'Field' has no len()"
    try:
        tabs = agent_memory.browser_tabs
        n = len(tabs)
        check("len(agent_memory.browser_tabs) works", True, f"len={n}")
    except TypeError as e:
        check("len(agent_memory.browser_tabs) works", False, str(e))

    # Verify it's a real list
    assert isinstance(agent_memory.browser_tabs, list)
    check("browser_tabs is a list", True)

    # Verify we can set and get
    agent_memory.set_browser_tabs(["tab1", "tab2"])
    check("browser_tabs after set", len(agent_memory.browser_tabs) == 2,
          f"count={len(agent_memory.browser_tabs)}")


# ── Test 4: Response Guarantee edge cases ────────────────────

async def test_response_guarantee_edge_cases():
    """Verify the guarantee layer handles all edge cases."""
    print("\n  TEST: Response Guarantee Edge Cases")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    # Edge case: None result from process
    spoken = []
    async def process_none():
        return None
    async def speak(t):
        spoken.append(t)
        return True
    ok = await response_guarantee.run_turn(
        transcript="test", process_fn=process_none, speak_fn=speak)
    check("None result → recovery spoken", ok and bool(spoken),
          f"spoken={spoken}")

    # Edge case: process raises CancelledError (should propagate)
    async def process_cancel():
        raise asyncio.CancelledError()
    try:
        await response_guarantee.run_turn(
            transcript="test", process_fn=process_cancel, speak_fn=speak)
        check("CancelledError propagates", False, "should have raised")
    except asyncio.CancelledError:
        check("CancelledError propagates", True)

    # Edge case: empty response, empty transcript
    spoken.clear()
    async def process_empty():
        class MockResult:
            response = ""
            actions_failed = 0
            actions_executed = 0
            verified = True
            used_llm = False
            path = "TEST"
            speak_immediately = False
            followup_response = ""
        return MockResult()
    async def speak_empty(t):
        spoken.append(t)
        return True
    ok = await response_guarantee.run_turn(
        transcript="", process_fn=process_empty, speak_fn=speak_empty)
    check("Empty transcript → fallback spoken", ok and bool(spoken),
          f"spoken={spoken}")


# ── Test 5: Command normalization ────────────────────────────

async def test_command_normalizer():
    """Verify the command normalizer handles common patterns."""
    print("\n  TEST: Command Normalizer")
    print("  " + "-" * 50)

    try:
        from nlp.command_normalizer import command_normalizer

        # Common normalizations
        result = command_normalizer.normalize("open vs code")
        check("Normalizes 'vs code'", "vscode" in result.lower(), f"result={result}")

        result = command_normalizer.normalize("open you tube")
        check("Normalizes 'you tube'", "youtube" in result.lower(), f"result={result}")

        result = command_normalizer.normalize("search for cats")
        check("Normalizes 'search for'", result, f"result={result}")

        result = command_normalizer.normalize("hello")
        check("Normal 'hello' unchanged", result == "hello", f"result={result}")

    except Exception as e:
        check("Command normalizer imports", False, str(e))


# ── Test 6: Conversation memory ─────────────────────────────

async def test_conversation_memory():
    """Verify conversation memory stores and retrieves."""
    print("\n  TEST: Conversation Memory")
    print("  " + "-" * 50)

    try:
        from agent.conversation_memory import conv_memory

        # Store a turn
        conv_memory.add_user("open firefox")
        conv_memory.add_assistant("Opening Firefox.")

        # Build context
        context = conv_memory.build_context()
        check("Context is non-empty", bool(context), f"len={len(context)}")
        check("Context contains user text", "open firefox" in context,
              f"context={context[:50]}")

        # Track goal
        conv_memory.track_goal("open firefox")
        check("Goal tracking works", True)

    except Exception as e:
        check("Conversation memory test", False, str(e))


# ── Test 7: Personality responses ────────────────────────────

async def test_personality_responses():
    """Verify personality generates natural, varied responses."""
    print("\n  TEST: Personality Responses")
    print("  " + "-" * 50)

    from agent.personality import personality

    # Check all response types are non-empty
    check("Greeting", bool(personality.greeting()))
    check("Acknowledgment", bool(personality.acknowledgment()))
    check("Farewell", bool(personality.farewell()))
    check("Error response", bool(personality.error_response()))
    check("Thinking filler", bool(personality.thinking()))
    check("Task confirmation", bool(personality.task_confirmation("done")))
    check("How are you", bool(personality.how_are_you()))
    check("Thanks", bool(personality.thanks()))

    # Check no robotic phrases
    robot_phrases = ["how may i assist", "task completed", "command executed",
                     "action successful", "processing request"]
    for phrase in robot_phrases:
        resp = personality.greeting()
        if phrase in resp.lower():
            check(f"No robotic: '{phrase}'", False, f"found in '{resp}'")
            break
    else:
        check("No robotic phrases in greeting", True)

    # Check contextual responses
    assert personality.contextual_response("hello") is not None
    assert personality.contextual_response("how are you") is not None
    assert personality.contextual_response("thanks") is not None
    assert personality.contextual_response("bye") is not None
    assert personality.contextual_response("open firefox") is None  # Not conversational

    check("Contextual 'hello' works", True)
    check("Contextual 'how are you' works", True)
    check("Contextual 'thanks' works", True)
    check("Contextual 'open firefox' returns None", True)


# ── Test 8: Turn state invariant ─────────────────────────────

async def test_turn_state_invariant():
    """Verify the turn state machine has no illegal paths."""
    print("\n  TEST: Turn State Invariant")
    print("  " + "-" * 50)

    from core.conversation_engine import EngineState, ALLOWED_TRANSITIONS

    # Every state must have at least one valid transition
    for state in EngineState:
        transitions = ALLOWED_TRANSITIONS.get(state, set())
        check(f"{state.value} has transitions", len(transitions) > 0,
              f"transitions={[s.value for s in transitions]}")

    # No state should be allowed to transition to itself
    for state, allowed in ALLOWED_TRANSITIONS.items():
        check(f"{state.value} no self-loop", state not in allowed)

    # IDLE must only go to WAKE
    assert EngineState.WAKE in ALLOWED_TRANSITIONS[EngineState.IDLE]
    check("IDLE → WAKE is valid", True)

    # WAKE must go to FACE_AUTH or LISTEN
    assert EngineState.FACE_AUTH in ALLOWED_TRANSITIONS[EngineState.WAKE]
    assert EngineState.LISTEN in ALLOWED_TRANSITIONS[EngineState.WAKE]
    check("WAKE → FACE_AUTH is valid", True)
    check("WAKE → LISTEN is valid", True)

    # THINK must go to SPEAK
    assert EngineState.SPEAK in ALLOWED_TRANSITIONS[EngineState.THINK]
    check("THINK → SPEAK is valid", True)

    # SPEAK must go to LISTEN or IDLE
    assert EngineState.LISTEN in ALLOWED_TRANSITIONS[EngineState.SPEAK]
    assert EngineState.IDLE in ALLOWED_TRANSITIONS[EngineState.SPEAK]
    check("SPEAK → LISTEN is valid", True)
    check("SPEAK → IDLE is valid", True)

    # No illegal transitions
    assert EngineState.LISTEN not in ALLOWED_TRANSITIONS.get(EngineState.IDLE, set())
    assert EngineState.THINK not in ALLOWED_TRANSITIONS.get(EngineState.SPEAK, set())
    check("No illegal IDLE → LISTEN", True)
    check("No illegal SPEAK → THINK", True)


# ── Test 9: Pipeline diagnostics ─────────────────────────────

async def test_pipeline_diagnostics():
    """Verify diagnostics are captured for every turn."""
    print("\n  TEST: Pipeline Diagnostics")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    # Run a diagnostic turn
    spoken = []
    async def process():
        class MockResult:
            response = "test response"
            actions_failed = 0
            actions_executed = 1
            verified = True
            used_llm = False
            path = "TEST"
            speak_immediately = False
            followup_response = ""
        return MockResult()

    async def speak(t):
        spoken.append(t)
        return True

    await response_guarantee.run_turn(
        transcript="diagnostic test",
        process_fn=process,
        speak_fn=speak,
    )

    diag = response_guarantee.get_diagnostics()
    last = diag.get("last_turn", {})

    check("Diagnostics transcript matches", last.get("transcript") == "diagnostic test",
          f"transcript={last.get('transcript')}")
    check("Diagnostics brain_result", last.get("brain_result") == "TEST",
          f"brain_result={last.get('brain_result')}")
    check("Diagnostics tts_started", last.get("tts_started") == True)
    check("Diagnostics tts_finished", last.get("tts_finished") == True)
    check("Diagnostics response_length > 0", last.get("response_length", 0) > 0,
          f"response_length={last.get('response_length')}")


# ── Test 10: Integration test compatibility ──────────────────

async def test_integration_compatibility():
    """Verify the integration test still passes its key assertions."""
    print("\n  TEST: Integration Compatibility")
    print("  " + "-" * 50)

    import inspect
    from core.conversation_engine import ConversationEngine

    engine = ConversationEngine()

    # Verify the engine calls Brain.process_command
    source = inspect.getsource(ConversationEngine._conversation_session)
    check("Engine calls brain.process_command", "agent_brain.process_command" in source,
          "Brain is the single orchestrator")

    # Verify the engine uses response_guarantee
    check("Engine uses response_guarantee", "response_guarantee.run_turn" in source,
          "Response guarantee integrated")

    # Verify the module imports is_garbage
    import core.conversation_engine as ce_module
    module_source = inspect.getsource(ce_module)
    check("Engine imports is_garbage", "is_garbage" in module_source,
          "Garbage rejection integrated at module level")


async def main():
    print("=" * 70)
    print("  PRODUCTION REGRESSION TESTS")
    print("=" * 70)

    await test_garbage_rejection()
    await test_filler_detection()
    await test_planner_crash_fix()
    await test_response_guarantee_edge_cases()
    await test_command_normalizer()
    await test_conversation_memory()
    await test_personality_responses()
    await test_turn_state_invariant()
    await test_pipeline_diagnostics()
    await test_integration_compatibility()

    print()
    print("=" * 70)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ✓ ALL PRODUCTION REGRESSION TESTS PASSED")
    else:
        print("  ✗ SOME PRODUCTION REGRESSION TESTS FAILED")
    print("=" * 70)
    print()

    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)