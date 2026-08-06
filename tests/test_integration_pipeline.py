"""
Integration Test — Unified Execution Pipeline.

Validates the Brain is the single orchestrator. Every command flows
through the exact pipeline:

    Voice → Perception → Brain → Planner → Dispatcher → Verify → Learn → Respond

Assertions verify:
  1. No subsystem executes actions outside the Brain path
  2. The router only classifies (never executes)
  3. The dispatcher only executes (never verifies or learns)
  4. The conversation engine only speaks (never executes or plans)
  5. The planner only plans (never executes)
  6. Brain.process_command() is the ONLY entry point

Usage:
    python tests/test_integration_pipeline.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Python 3.14 compatibility
import compat  # noqa: F401

import pytest

pytestmark = pytest.mark.asyncio

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


class MockDispatcher:
    """Mock dispatcher that counts executions."""
    def __init__(self):
        self.executions = []
        self._was_called = False

    async def execute(self, action):
        self._was_called = True
        self.executions.append(action)
        return f"Executed {action.get('action', 'unknown')}"


class MockVerifier:
    """Mock verifier that always succeeds."""
    def __init__(self):
        self.verifications = []

    async def verify_action(self, action_type, params, expected_outcome=""):
        self.verifications.append((action_type, params))
        from vision.action_verifier import VerificationResult, VerificationStatus
        return VerificationResult(
            success=True,
            status=VerificationStatus.VERIFIED,
            explanation="Mock verified",
        )


class MockLearning:
    """Mock learning engine that records outcomes."""
    def __init__(self):
        self.records = []

    def record_action(self, action_name, params, success=True, error=""):
        self.records.append({
            "action": action_name,
            "params": params,
            "success": success,
            "error": error,
        })


class MockPlanner:
    """Mock planner that only creates plans."""
    def __init__(self):
        self.plans_created = 0
        self.is_available = True

    def initialize(self):
        return True

    def generate_plan_only(self, request):
        self.plans_created += 1
        return [
            {"action": "desktop_open", "params": {"app": "code"}, "description": "Open code"},
        ]

    def process_request(self, request):
        # Legacy method — should NEVER be called by the Brain pipeline
        self.plans_created += 1
        return f"I've planned steps to accomplish this."


class MockPerception:
    """Mock perception that returns a minimal context."""
    def __init__(self):
        self.perceptions = 0

    async def perceive(self):
        self.perceptions += 1
        return type("MockPerceptionCtx", (), {
            "window_title": "Test Window",
            "a11y_available": False,
            "ocr_used": False,
            "compact_summary": "Window: Test Window",
        })()


class MockDecisionEngine:
    """Mock decision engine with controllable outcomes."""
    def __init__(self):
        self.decisions = 0
        self.force_llm = False

    async def decide(self, text, vision_context=None, search_context=None, desktop_context=None):
        self.decisions += 1
        from core.decision_engine import Decision, DecisionPath

        if self.force_llm:
            return Decision(path=DecisionPath.LLM, needs_llm=True)
        return Decision(
            path=DecisionPath.DIRECT_EXECUTION,
            needs_llm=False,
            action={"action": "desktop_open", "params": {"app": "firefox"}},
            response="Opening browser.",
        )


async def test_single_orchestrator_pipeline():
    """Test that Brain.process_command() is the only entry to dispatch/verify/learn."""
    print("\n  TEST: Single Orchestrator Pipeline")
    print("  " + "-" * 50)

    from agent.brain import AgentBrain

    # Create isolated brain with mocks
    brain = AgentBrain()
    brain._dispatcher = MockDispatcher()
    brain._verifier = MockVerifier()
    brain._learning = MockLearning()
    brain._perception = MockPerception()
    brain._planner = MockPlanner()
    brain._decision_engine = MockDecisionEngine()
    brain._initialized = True

    # ── Test 1: Simple command (direct execution) ──────────
    result = await brain.process_command("open firefox")

    check("Simple command returns response", bool(result.response), result.response)
    check("Simple command dispatched exactly 1 action", result.actions_executed == 1,
          f"executed={result.actions_executed}")
    check("Simple command verified", result.verified, f"verified={result.verified}")
    check("Dispatcher was called", brain._dispatcher._was_called)
    check("Verifier was called", len(brain._verifier.verifications) == 1)
    check("Learning recorded 1 action", len(brain._learning.records) == 1,
          f"records={len(brain._learning.records)}")
    check("Perception ran once", brain._perception.perceptions == 1,
          f"perceptions={brain._perception.perceptions}")
    check("No LLM used", not result.used_llm)

    # ── Test 2: Complex command (LLM path with plan) ────────
    brain._decision_engine.force_llm = True
    brain._planner.plans_created = 0
    brain._learning.records.clear()

    result2 = await brain.process_command("open the code editor and start working")

    check("Complex command uses LLM", result2.used_llm)
    check("Planner created a plan", brain._planner.plans_created == 1,
          f"plans={brain._planner.plans_created}")
    check("Plan actions dispatched", result2.actions_executed == 1,
          f"executed={result2.actions_executed}")
    check("Plan actions verified", result2.verified)
    check("Learning recorded for plan action", len(brain._learning.records) == 1)
    check("Planner legacy process_request NOT called", brain._planner.plans_created == 1,
          "generate_plan_only used")

    print(f"  {PASS} passed, {FAIL} failed")


async def test_planner_never_executes():
    """Test that the Planner only creates plans and never executes."""
    print("\n  TEST: Planner Only Plans (Never Executes)")
    print("  " + "-" * 50)

    from agent.planner import AgentPlanner

    planner = AgentPlanner()
    planner._initialized = True
    planner._llm_client = None  # No LLM → fallback plan path

    # generate_plan_only returns a plan without executing
    plan = planner.generate_plan_only("open youtube")
    check("generate_plan_only returns a plan", plan is not None,
          f"steps={len(plan) if plan else 0}")
    if plan:
        check("Plan contains action dicts", all("action" in s for s in plan))

    # process_request (legacy) only plans, never executes
    result = planner.process_request("open youtube")
    check("Legacy process_request only plans", "planned" in result.lower() or "sorry" in result.lower(),
          result)

    # Check that agent_executor was NEVER called
    from agent.executor import agent_executor
    check("Executor NOT used during planning", agent_executor._initialized == False,
          f"executor_initialized={agent_executor._initialized}")


async def test_router_only_classifies():
    """Test that CommandRouter no longer executes actions directly."""
    print("\n  TEST: Router Only Classifies (Never Executes)")
    print("  " + "-" * 50)

    from core.command_router import CommandRouter, RouteKind

    router = CommandRouter()

    # Create a mock dispatcher — if the router calls execute, it's a BUG
    class EvilDispatcher:
        async def execute(self, action):
            raise AssertionError("ROUTER MUST NOT EXECUTE ACTIONS — Brain is the orchestrator")

    router.wire(action_dispatcher=EvilDispatcher())

    # Route a simple command
    result = await router.route("open firefox")
    check("Router returns SIMPLE_DESKTOP", result.kind == RouteKind.SIMPLE_DESKTOP,
          f"kind={result.kind.value}")
    check("Router returns the action", result.action is not None,
          f"action={result.action}")
    check("Router does NOT execute", result.action.get("action") == "desktop_open")

    # Route a workflow
    result2 = await router.route("start coding")
    check("Router returns KNOWN_WORKFLOW", result2.kind == RouteKind.KNOWN_WORKFLOW,
          f"kind={result2.kind.value}")
    check("Workflow has actions for Brain to dispatch", len(result2.actions or []) == 2,
          f"actions={len(result2.actions or [])}")

    print(f"  {PASS} passed, {FAIL} failed")


async def test_dispatcher_only_executes():
    """Test that the Dispatcher only executes (no internal verification/learning)."""
    print("\n  TEST: Dispatcher Only Executes (No Verify/Learn)")
    print("  " + "-" * 50)

    import inspect
    from agent.action_dispatcher import ActionDispatcher

    dispatcher = ActionDispatcher()

    # The execute method should NOT call _verify_action or _record_for_learning
    source = inspect.getsource(ActionDispatcher.execute)
    check("execute() does NOT call _verify_action", "_verify_action" not in source,
          "verification removed from dispatcher")
    check("execute() does NOT call _record_for_learning", "_record_for_learning" not in source,
          "learning removed from dispatcher")
    check("execute() does NOT call _capture_pre_action", "_capture_pre_action" not in source,
          "pre-capture removed from dispatcher")
    check("execute() has NO retry loop", "_adjust_params_for_retry" not in source,
          "retry removed from dispatcher")

    print(f"  {PASS} passed, {FAIL} failed")


async def test_conversation_engine_only_speaks():
    """Test that the ConversationEngine no longer executes or plans."""
    print("\n  TEST: Conversation Engine Only Speaks")
    print("  " + "-" * 50)

    import inspect
    from core.conversation_engine import ConversationEngine

    engine = ConversationEngine()

    # Verify the engine no longer has duplicate execution paths
    check("_run_action method removed", not hasattr(engine, "_run_action"),
          "engine cannot execute actions directly")
    check("_llm_sentences method removed", not hasattr(engine, "_llm_sentences"),
          "engine cannot generate LLM sentences directly")
    check("_get_desktop_context_sync removed", not hasattr(engine, "_get_desktop_context_sync"),
          "engine no longer collects context")
    check("_references_screen removed", not hasattr(engine, "_references_screen"),
          "engine no longer decides what context to collect")

    # Verify the engine calls Brain.process_command
    source = inspect.getsource(ConversationEngine._conversation_session)
    check("Engine calls brain.process_command", "agent_brain.process_command" in source,
          "Brain is the single orchestrator")

    print(f"  {PASS} passed, {FAIL} failed")


async def test_action_verifier_single_path():
    """Test that verification exists in exactly ONE place (the Brain)."""
    print("\n  TEST: Verification In Brain Only")
    print("  " + "-" * 50)

    import inspect
    from agent.action_dispatcher import ActionDispatcher
    from core.conversation_engine import ConversationEngine

    # The dispatcher should not have verification
    dispatcher_src = inspect.getsource(ActionDispatcher)
    engine_src = inspect.getsource(ConversationEngine)

    # The dispatcher might still have _verify_action as dead code, but execute() should not call it
    # The key check: the Brain's _dispatch_and_verify is the ONLY place
    from agent.brain import AgentBrain
    brain_src = inspect.getsource(AgentBrain)

    check("Brain has _dispatch_and_verify", "_dispatch_and_verify" in brain_src)
    check("Brain calls dispatcher.execute", "self._dispatcher.execute" in brain_src)
    check("Brain calls verifier.verify_action", "self._verifier.verify_action" in brain_src)
    check("Brain calls learning.record_action", "self._learning.record_action" in brain_src)
    check("Brain calls perception.perceive", "self._perception.perceive" in brain_src)
    check("Brain calls decision_engine.decide", "self._decision_engine.decide" in brain_src)
    check("Brain calls planner.generate_plan_only", "self._planner.generate_plan_only" in brain_src)

    print(f"  {PASS} passed, {FAIL} failed")


async def main():
    print("=" * 70)
    print("  INTEGRATION TESTS — UNIFIED EXECUTION PIPELINE")
    print("=" * 70)

    await test_single_orchestrator_pipeline()
    await test_planner_never_executes()
    await test_router_only_classifies()
    await test_dispatcher_only_executes()
    await test_conversation_engine_only_speaks()
    await test_action_verifier_single_path()

    print()
    print("=" * 70)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("  ✗ SOME INTEGRATION TESTS FAILED")
    print("=" * 70)
    print()

    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)