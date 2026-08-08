"""
Response Guarantee Tests — Leo must NEVER be silent.

Validates that every completed user utterance receives a spoken
response, no matter what fails in the pipeline.

Scenarios covered:
  1. Normal response — real response spoken
  2. Empty response from Brain — generic fallback spoken
  3. Brain raises exception — recovery response spoken
  4. TTS fails on real response — recovery fallback spoken
  5. TTS fails on everything — [FATAL] SILENT TURN logged
  6. Action failed — recovery response spoken
  7. Watchdog logging — diagnostics recorded for every turn

Usage:
    python tests/test_response_guarantee.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Python 3.14 compatibility
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


class MockResult:
    """Minimal CommandResult-like object."""
    def __init__(self, response="", actions_failed=0, actions_executed=0,
                 verified=True, used_llm=False, path="TEST",
                 speak_immediately=False, followup_response=""):
        self.response = response
        self.actions_failed = actions_failed
        self.actions_executed = actions_executed
        self.verified = verified
        self.used_llm = used_llm
        self.path = path
        self.speak_immediately = speak_immediately
        self.followup_response = followup_response


async def test_normal_response():
    """Scenario 1: Normal response — real response spoken."""
    print("\n  TEST: Normal Response")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    spoken = []

    async def process():
        return MockResult(response="Opening Firefox.", actions_executed=1, verified=True)

    async def speak(text):
        spoken.append(text)
        return True

    ok = await response_guarantee.run_turn(
        transcript="open firefox",
        process_fn=process,
        speak_fn=speak,
    )

    check("Turn completed", ok)
    check("Real response spoken", spoken and spoken[0] == "Opening Firefox.",
          f"spoken={spoken}")
    check("No silent turns", response_guarantee.silent_turns == 0,
          f"silent={response_guarantee.silent_turns}")


async def test_empty_response_fallback():
    """Scenario 2: Empty response from Brain — generic fallback spoken."""
    print("\n  TEST: Empty Response → Generic Fallback")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    spoken = []

    async def process():
        return MockResult(response="", actions_executed=0, verified=True)

    async def speak(text):
        spoken.append(text)
        return True

    ok = await response_guarantee.run_turn(
        transcript="something",
        process_fn=process,
        speak_fn=speak,
    )

    check("Turn completed", ok)
    check("Fallback spoken (non-empty)", bool(spoken) and bool(spoken[0].strip()),
          f"spoken={spoken}")
    check("Fallback is a generic response", spoken and spoken[0] in (
        "I didn't catch that.", "Could you repeat that?", "I'm not sure I understood.",
        "Something went wrong.", "Let me try another way.", "I'm still working on it.",
        "I lost the conversation context.",
    ), f"spoken={spoken}")


async def test_brain_exception_recovery():
    """Scenario 3: Brain raises exception — recovery response spoken."""
    print("\n  TEST: Brain Exception → Recovery Response")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    spoken = []

    async def process():
        raise RuntimeError("Brain crashed")

    async def speak(text):
        spoken.append(text)
        return True

    ok = await response_guarantee.run_turn(
        transcript="do something",
        process_fn=process,
        speak_fn=speak,
    )

    check("Turn completed", ok)
    check("Recovery response spoken", bool(spoken) and bool(spoken[0].strip()),
          f"spoken={spoken}")


async def test_tts_failure_retry():
    """Scenario 4: TTS fails on real response — recovery fallback spoken."""
    print("\n  TEST: TTS Failure → Recovery Fallback")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    spoken = []
    attempts = 0

    async def process():
        return MockResult(response="This is the real response.", actions_executed=1, verified=True)

    async def speak(text):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return False  # First attempt fails
        spoken.append(text)
        return True

    ok = await response_guarantee.run_turn(
        transcript="test",
        process_fn=process,
        speak_fn=speak,
    )

    check("Turn completed", ok)
    check("Retried with fallback", attempts >= 2, f"attempts={attempts}")
    check("Fallback spoken", bool(spoken), f"spoken={spoken}")


async def test_tts_total_failure_fatal():
    """Scenario 5: TTS fails on everything — [FATAL] SILENT TURN logged."""
    print("\n  TEST: TTS Total Failure → FATAL Logged")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    # Capture critical logs
    log_records = []
    handler = logging.Handler()
    handler.emit = lambda record: log_records.append(record)
    logger = logging.getLogger("core.response_guarantee")
    logger.addHandler(handler)
    logger.setLevel(logging.CRITICAL)

    async def process():
        return MockResult(response="Real response", actions_executed=1, verified=True)

    async def speak(text):
        return False  # Always fails

    ok = await response_guarantee.run_turn(
        transcript="test",
        process_fn=process,
        speak_fn=speak,
    )

    logger.removeHandler(handler)

    check("Turn reports failure", not ok)
    check("FATAL SILENT TURN logged", any(
        "SILENT TURN" in r.getMessage() for r in log_records
    ), f"records={[r.getMessage() for r in log_records]}")
    check("Silent turn counter incremented", response_guarantee.silent_turns >= 1,
          f"silent={response_guarantee.silent_turns}")


async def test_action_failed_recovery():
    """Scenario 6: Action failed — recovery response spoken."""
    print("\n  TEST: Action Failed → Recovery Response")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    spoken = []

    async def process():
        return MockResult(response="", actions_failed=1, actions_executed=1, verified=False)

    async def speak(text):
        spoken.append(text)
        return True

    ok = await response_guarantee.run_turn(
        transcript="open firefox",
        process_fn=process,
        speak_fn=speak,
    )

    check("Turn completed", ok)
    check("Recovery response spoken", bool(spoken) and bool(spoken[0].strip()),
          f"spoken={spoken}")


async def test_watchdog_diagnostics():
    """Scenario 7: Watchdog logging — diagnostics recorded for every turn."""
    print("\n  TEST: Watchdog Diagnostics")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    async def process():
        return MockResult(response="Hello there.", actions_executed=1, verified=True,
                          used_llm=True, path="LLM")

    async def speak(text):
        return True

    await response_guarantee.run_turn(
        transcript="say hello",
        process_fn=process,
        speak_fn=speak,
    )

    diag = response_guarantee.get_diagnostics()
    check("Diagnostics has total_turns", "total_turns" in diag)
    check("Diagnostics has silent_turns", "silent_turns" in diag)
    check("Diagnostics has last_turn", "last_turn" in diag)

    last = diag.get("last_turn", {})
    check("Last turn has transcript", last.get("transcript") == "say hello",
          f"transcript={last.get('transcript')}")
    check("Last turn has brain_result", "brain_result" in last)
    check("Last turn has tts_started", "tts_started" in last)
    check("Last turn has tts_finished", "tts_finished" in last)
    check("Last turn has response_length", "response_length" in last)
    check("Last turn has response_length > 0", last.get("response_length", 0) > 0,
          f"response_length={last.get('response_length')}")


async def test_200_turns_no_silent():
    """Scenario 8: 200 simulated turns — 0 silent turns."""
    print("\n  TEST: 200 Turns — 0 Silent")
    print("  " + "-" * 50)

    from core.response_guarantee import response_guarantee

    silent = 0
    for i in range(200):
        spoken = []

        async def process(i=i):
            if i % 5 == 0:
                raise RuntimeError(f"Simulated failure {i}")
            if i % 7 == 0:
                return MockResult(response="", actions_executed=0, verified=True)
            if i % 11 == 0:
                return MockResult(response="", actions_failed=1, actions_executed=1, verified=False)
            return MockResult(response=f"Response {i}", actions_executed=1, verified=True)

        async def speak(text):
            spoken.append(text)
            return True

        ok = await response_guarantee.run_turn(
            transcript=f"command {i}",
            process_fn=process,
            speak_fn=speak,
        )
        if not ok or not spoken:
            silent += 1

    check("0 silent turns in 200", silent == 0, f"silent={silent}")


async def main():
    print("=" * 70)
    print("  RESPONSE GUARANTEE TESTS — LEO NEVER SILENT")
    print("=" * 70)

    await test_normal_response()
    await test_empty_response_fallback()
    await test_brain_exception_recovery()
    await test_tts_failure_retry()
    await test_tts_total_failure_fatal()
    await test_action_failed_recovery()
    await test_watchdog_diagnostics()
    await test_200_turns_no_silent()

    print()
    print("=" * 70)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ✓ ALL RESPONSE GUARANTEE TESTS PASSED")
    else:
        print("  ✗ SOME RESPONSE GUARANTEE TESTS FAILED")
    print("=" * 70)
    print()

    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)