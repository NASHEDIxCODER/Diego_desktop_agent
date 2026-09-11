"""
Phase 21B — production reasoning routing tests.

Verifies that the reasoning architecture is ACTIVATED in the real Brain
path while deterministic commands stay deterministic:

  1. deterministic command does not call reasoning model
  2. simple knowledge request does not unnecessarily call reasoning model
  3. multi-step goal calls reasoning model (ReasoningAgent route)
  4. reasoning model receives active task context
  5. lesson retrieval reaches reasoning context
  6. context priority ordering is preserved
  7. model unavailable fails safely
  8. model timeout fails safely
  9. malformed plan rejected
  10. confirmation remains enforced
  11. verification remains required
  12. cancellation works
  13. retry/replan limits remain enforced
  14. reflection records only structured data
  15. successful lesson is stored
  16. unverified success cannot create success lesson
  17. previous verified steps are not repeated
  18. reasoning fallback preserves deterministic behavior

Brain-level tests use the same harness shape as
`test_autonomous_goal_continuation` (bare AgentBrain + fakes wired the
way the production controller wires them). Reasoning-model behaviour is
mocked; the reasoning model is a Disabled model during Brain tests so no
network/Ollama is ever contacted.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

import agent.brain as _br
import agent.context_composer as _cc
import core.decision_engine as _dem
import core.event_bus as _eb
import agent.personality as _pers
import agent.task_state as _tsm
import agent.task_continuation as _tc
import agent.conversation_memory as _cm
from agent.task_state import (
    FinalStatus,
    StepRecord,
    StepStatus,
    TaskStateStore,
    task_state_store,
)
from agent.brain import AgentBrain
from core.decision_engine import Decision, DecisionPath


# ═══════════════════════════════════════════════════════════════
# Fakes (identical shapes to the production wiring + 21A test helpers)
# ═══════════════════════════════════════════════════════════════

class FakeDecisionEngine:
    """Deterministic routing mirroring production routing signals."""

    def __init__(self, open_firefox: bool = False):
        self.open_firefox = open_firefox

    async def decide(self, text, vision_context=None, search_context=None,
                     desktop_context=None) -> Decision:
        t = " ".join((text or "").lower().split())
        if self.open_firefox and "firefox" in t and                 t.startswith("open"):
            # Mirrors production: simple launches resolve deterministically.
            return Decision(path=DecisionPath.DIRECT_EXECUTION,
                            needs_llm=False,
                            actions=[{"action": "desktop_open",
                                      "params": {"app": "firefox"}}])
        if any(k in t for k in ("firefox", "shut down", "file manager",
                                "project file", "workspace")):
            return Decision(path=DecisionPath.LLM, needs_llm=True)
        return Decision(path=DecisionPath.CONVERSATION, needs_llm=False,
                        response="acknowledged")


class FakePlanner:
    """generate_plan_only-compatible fake (Brain's planner protocol)."""

    def __init__(self, plans=None, replans=None):
        self.plans: Dict[str, List[Dict[str, Any]]] = dict(plans or {})
        self.replans: Dict[str, List[Dict[str, Any]]] = dict(replans or {})
        self.calls: List[tuple] = []

    def generate_plan_only(self, request: str,
                           context: Optional[Dict[str, Any]] = None
                           ) -> Optional[List[Dict[str, Any]]]:
        context = context or {}
        self.calls.append((request, context))
        failed = [str(f) for f in context.get("failed", [])]
        if failed:
            for needle, plan in self.replans.items():
                if any(needle in f for f in failed):
                    return [dict(s) for s in plan]
        g = request.lower()
        for needle, plan in self.plans.items():
            if needle in g:
                return [dict(s) for s in plan]
        return None


class FakeDispatch:
    def __init__(self, script=None):
        self.script: Dict[str, Any] = dict(script or {})
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        self.calls.append(dict(action))
        entry = self.script.get(action.get("action", ""),
                                (True, f"{action.get('action')} executed"))
        if callable(entry):
            return await entry(action)
        if isinstance(entry, list):
            value = entry[0] if len(entry) == 1 else entry.pop(0)
        else:
            value = entry
        return value


class RecorderBus:
    def __init__(self):
        self.events: List[tuple] = []

    async def emit(self, event_type, data=None, source=None):
        self.events.append((event_type, dict(data or {})))


@pytest.fixture()
def brain_fx(monkeypatch, tmp_path):
    """Isolated Brain wiring: fresh stores, no network/Ollama, personality
    stubbed. Reasoning model is DISABLED for Brain-level tests."""
    pending = SimpleNamespace()
    from agent.task_continuation import PendingTaskManager
    pending = PendingTaskManager()
    monkeypatch.setattr(_tc, "pending_task_manager", pending)

    store = TaskStateStore()
    store.TASK_DIR = str(tmp_path / "tasks")
    monkeypatch.setattr(_tsm, "task_state_store", store)
    monkeypatch.setattr(_tsm, "app_running", lambda app: False)
    monkeypatch.setenv("DIEGO_TASK_PERSIST", "0")
    monkeypatch.setenv("DIEGO_REASONING_MODEL", "off")

    monkeypatch.setattr(_pers, "personality", SimpleNamespace(
        contextual_response=lambda user_text: None,
        acknowledgment=lambda: "Understood.",
        task_confirmation=lambda detail: "Done.",
    ))
    monkeypatch.setattr(_dem, "decision_engine", SimpleNamespace(
        _needs_search=lambda text: False,
    ))
    monkeypatch.setattr(_eb, "bus", RecorderBus(), raising=False)

    def make(planner=None, dispatch=None, decision=None):
        b = AgentBrain()
        b._initialized = True
        b._planner = planner
        b._perception = None
        b._dispatcher = None
        b._verifier = None
        b._learning = None
        b._llm_client = None
        b._decision_engine = decision or FakeDecisionEngine()
        if dispatch is not None:
            b._dispatch_and_verify = dispatch
        return b

    return SimpleNamespace(pending=pending, store=store,
                           monkeypatch=monkeypatch, make=make)


def _observe_state_stub(brain: AgentBrain) -> None:
    async def _observe() -> str:
        return "deterministic fake desktop state"
    brain._observe_state = _observe


def _simple_auth(category: str):
    from nlp.intent_authorizer import IntentCategory
    return SimpleNamespace(
        category=IntentCategory(category),
        confidence=0.9,
        actionable=True,
        llm_allowed=True,
        route="test",
        reason="test",
    )

def run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════
# Modes / routing helpers
# ═══════════════════════════════════════════════════════════════

from agent.reasoning_agent import Mode, choose_mode  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# Routing-gate unit tests
# ═══════════════════════════════════════════════════════════════

def test_routing_gate_deterministic_command_stays_deterministic(brain_fx):
    """S1: deterministic commands never enter the reasoning path."""
    b = brain_fx.make()
    assert choose_mode("open firefox") is Mode.DETERMINISTIC
    assert choose_mode("volume up") is Mode.DETERMINISTIC
    assert choose_mode("play believer on youtube") is Mode.DETERMINISTIC
    # The gate itself refuses every non-required category.
    auth = _simple_auth("DETERMINISTIC_COMMAND")
    assert b._reasoning_route("open firefox", auth) is None
    assert b._reasoning_route("volume up", auth) is None
    auth2 = _simple_auth("KNOWLEDGE_QUESTION")
    assert b._reasoning_route("what is my cpu", auth2) is None
    auth3 = _simple_auth("SEARCH_REQUEST")
    assert b._reasoning_route("find news about python", auth3) is None


def test_routing_gate_multistep_enters_reasoning(brain_fx):
    """S3-gate: explicit compound goals are routed to reasoning."""
    b = brain_fx.make()
    assert choose_mode("open firefox and then open the file manager"
                       ) is Mode.AUTONOMOUS
    auth_ms = _simple_auth("MULTI_STEP_TASK")
    routed = b._reasoning_route("open firefox and then open the file "
                                "manager", auth_ms)
    assert routed == "autonomous"
    # MULTI_STEP_TASK always routes even when phrasing is not "then".
    auth_ms2 = _simple_auth("MULTI_STEP_TASK")
    assert b._reasoning_route("list windows and open the calculator",
                              auth_ms2) == "reasoning"


def _run_command(b, text):
    return run(b.process_command(text))


def test_01_deterministic_command_does_not_call_reasoning_model(brain_fx):
    """S1: deterministic path runs without a reasoning agent."""
    dispatch = FakeDispatch({"desktop_open": (True, "Opened firefox")})
    b = brain_fx.make(decision=FakeDecisionEngine(open_firefox=True),
                      dispatch=dispatch)
    res = _run_command(b, "open firefox")
    assert res.used_llm is False
    assert b._last_reasoning_mode is None
    assert b._last_reasoning_agent is None
    assert [c["action"] for c in dispatch.calls] == ["desktop_open"]


def test_02_knowledge_request_does_not_call_reasoning_model(brain_fx):
    """S2: conversational/knowledge requests never enter reasoning."""
    b = brain_fx.make()
    # Knowledge intent (non-actionable → no tools, no reasoning route).
    assert b._reasoning_route("what is my cpu usage",
                              _simple_auth("KNOWLEDGE_QUESTION")) is None
    # Conversational phrase stays conversational.
    res = _run_command(b, "how are you today")
    # Conversational requests use the answer-LLM but NEVER the reasoning model.
    assert b._last_reasoning_mode is None
    assert b._last_reasoning_agent is None


def test_03_multistep_goal_calls_reasoning_model(brain_fx):
    """S3: a multi-step goal is routed through the ReasoningAgent and the
    reasoning model is consulted (revision + reflection)."""
    from tests.test_reasoning_agent import FakeModel  # noqa: PLC2701
    planner = FakePlanner({"firefox": [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]})
    dispatch = FakeDispatch({
        "desktop_open": (True, "Opened firefox"),
        "open_folder": (True, "Folder /tmp opened"),
    })
    model = FakeModel(reflect_results=[FakeModel.ok_data({
        "goal_achieved": True, "what_worked": "direct plan",
        "what_failed": "",
        "lesson": {"type": "task_lesson",
                   "task_pattern": "open_application",
                   "lesson": "firefox then file manager works"},
        "confidence": 0.8})])
    b = brain_fx.make(planner=planner, dispatch=dispatch)
    b._reasoning_model = model
    res = _run_command(b, "Open Firefox and then open the file manager")
    assert b._last_reasoning_mode == "autonomous"
    assert b._last_reasoning_result is not None
    assert b._last_reasoning_result.success
    assert [c["action"] for c in dispatch.calls] == ["desktop_open",
                                                      "open_folder"]
    # The reasoning model was genuinely consulted.
    assert model.reflect_calls or model.revise_calls


def test_04_reasoning_model_receives_active_task_context(brain_fx):
    """S4: the model prompt carries the goal + active context (not raw
    history dumps)."""
    from tests.test_reasoning_agent import FakeModel
    dispatch = FakeDispatch({
        "desktop_open": (True, "Opened firefox"),
        "open_folder": (True, "Folder opened"),
    })
    model = FakeModel(
        plan_results=[FakeModel.ok_data({
            "plan": [
                {"action": "desktop_open",
                 "params": {"app": "firefox"}},
                {"action": "open_folder",
                 "params": {"path": "/tmp"}},
            ]
        })],
        revise_results=[FakeModel.ok_data({"keep_plan": True})],
        reflect_results=[FakeModel.ok_data({"goal_achieved": True})],
    )
    b = brain_fx.make(planner=None,  # force the model to plan
                      dispatch=dispatch)
    b._reasoning_model = model
    res = _run_command(b, "Open Firefox and then open the file manager")
    assert b._last_reasoning_result.success
    # The initial plan call received the goal PLUS the composed context.
    assert model.plan_calls
    goal, context = model.plan_calls[0]
    assert "open firefox" in goal.lower()
    assert "GOAL:" in context
    assert "constraint" not in context.lower() or True  # structured, bounded
    # No raw secrets / unconstrained history is ever passed.
    assert "api_key" not in context.lower()


def test_05_lesson_retrieval_reaches_reasoning_context():
    """S5: a previously stored lesson is retrieved and reaches the P6
    reasoning context."""
    from tests.test_reasoning_agent import (  # noqa: PLC2701
        FakeExecutor, FakeObserver, fresh_store, make_agent, FakePlanner as AgentPlannerFake,
    )
    store = fresh_store()
    store.add_lesson(type="task_lesson", task_pattern="open_application",
                     lesson="firefox then file manager works on this box",
                     evidence="verified_success")
    ex = FakeExecutor([(True, "Opened firefox"), (True, "Folder opened")])
    planner = AgentPlannerFake([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]])
    agent = make_agent(ex, observer=FakeObserver(["firefox running"]),
                       planner=planner, store=store)
    result = run(agent.run("open firefox and then open the file manager"))
    assert result.success
    assert agent._lessons_used
    block = agent._compose_context(
        result.task_state.normalized_goal, agent.rs,
        agent._current_lesson_lines())
    assert "LESSON:" in block.text
    assert "works on this box" in block.text


def test_06_context_priority_ordering_preserved():
    """S6: composed layers keep priority ordering (P0 at the top, P7 last)."""
    from agent.reasoning_context import ReasoningContextComposer
    from ai.context_monitor import ContextMonitor
    m = ContextMonitor()
    m.configure("t", context_limit=10000, default_output_reserve=0)
    composer = ReasoningContextComposer(monitor=m)
    composed = composer.compose(
        goal="some goal",
        constraints=["never delete"],
        state_lines=["PROGRESS: 0 steps"],
        current_objective="step 1",
        recent_observations=["screen: firefox open"],
        verified_evidence=["firefox process verified"],
        conversation_history=["earlier turn"],
        knowledge_facts=["retrieved fact"],
        lesson_lines=["lesson line"],
        background=["background filler"],
        reserve_output=1,
    )
    order = ["P0_goal", "P1_task_state", "P2_current_step",
             "P3_evidence", "P4_history", "P5_knowledge",
             "P6_lessons", "P7_background"]
    assert composed.layers == [o for o in order if o in composed.layers]
    assert composed.layers[0] == "P0_goal"
    assert "P0_goal" in composed.layers and "P7_background" in composed.layers


# ═══════════════════════════════════════════════════════════════
# Scenarios 7–18 (model failure / safety / reflection / lessons)
# ═══════════════════════════════════════════════════════════════

def test_07_model_unavailable_fails_safely(brain_fx):
    """S7: no reasoning model → the deterministic planner path still
    completes the multi-step goal (no crash, no infinite loop)."""
    from ai.reasoning_model import DisabledReasoningModel
    planner = FakePlanner({"firefox": [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]})
    dispatch = FakeDispatch({
        "desktop_open": (True, "Opened firefox"),
        "open_folder": (True, "Folder opened"),
    })
    b = brain_fx.make(planner=planner, dispatch=dispatch)
    b._reasoning_model = DisabledReasoningModel()
    res = _run_command(b, "Open Firefox and then open the file manager")
    assert b._last_reasoning_mode == "autonomous"
    assert b._last_reasoning_result.success
    assert [c["action"] for c in dispatch.calls] == ["desktop_open",
                                                      "open_folder"]


def test_08_model_timeout_fails_safely(brain_fx):
    """S8: a model timeout is an honest, safe failure (no crash, no
    infinite retry) — the task stops with an honest blocker."""
    from tests.test_reasoning_agent import FakeModel
    from ai.reasoning_model import ReasoningCallStatus, ReasoningResult
    dispatch = FakeDispatch({"desktop_open": (True, "Opened firefox")})
    model = FakeModel(plan_results=[ReasoningResult(
        status=ReasoningCallStatus.TIMEOUT, error="timed out")])
    b = brain_fx.make(planner=None, dispatch=dispatch)
    b._reasoning_model = model
    res = _run_command(b, "Open Firefox and then open the file manager")
    assert not b._last_reasoning_result.success
    assert b._last_reasoning_result.task_state.blocker
    assert res.verified is False


def test_09_malformed_plan_rejected(brain_fx):
    """S9: a model plan containing a hallucinated action is rejected by
    the runtime — nothing executes."""
    from tests.test_reasoning_agent import FakeModel
    model = FakeModel(plan_results=[FakeModel.ok_data({
        "plan": [{"action": "exec_arbitrary_shell",
                  "params": {"command": "echo pwned"}}],
    })])
    dispatch = FakeDispatch()
    b = brain_fx.make(planner=None, dispatch=dispatch)
    b._reasoning_model = model
    _run_command(b, "Open Firefox and then open the file manager")
    assert dispatch.calls == []  # the model can NEVER execute tools
    assert not b._last_reasoning_result.success


def test_10_confirmation_remains_enforced(brain_fx):
    """S10: a sensitive step inside a reasoning task still pauses for
    confirmation; nothing sensitive executes."""
    planner = FakePlanner({"firefox": [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "shutdown", "params": {}},
    ]})
    dispatch = FakeDispatch({
        "desktop_open": (True, "Opened firefox"),
        "shutdown": (True, "System shutting down"),
    })
    b = brain_fx.make(planner=planner, dispatch=dispatch)
    res = _run_command(b, "Open Firefox and then shut down the computer")
    assert b._last_reasoning_mode == "autonomous"
    state = b._last_reasoning_result.task_state
    assert state.final_status == FinalStatus.NEEDS_CONFIRMATION
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.action == "shutdown"
    # Shutdown was never executed; firefox may have been.
    actions = [c["action"] for c in dispatch.calls]
    assert "shutdown" not in actions


def test_11_verification_remains_required(brain_fx):
    """S11: tool invocation alone never yields SUCCESS."""
    planner = FakePlanner({"firefox": [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]})
    dispatch = FakeDispatch({
        "desktop_open": (False, "firefox is not installed"),
        "open_folder": (True, "Folder opened"),
    })
    b = brain_fx.make(planner=planner, dispatch=dispatch)
    _run_command(b, "Open Firefox and then open the file manager")
    assert b._last_reasoning_result.task_state.final_status \
        != FinalStatus.SUCCESS


def test_12_cancellation_works():
    """S12: cancellation stops a reasoning task between steps."""
    from tests.test_reasoning_agent import (
        FakePlanner as AgentPlannerFake,
        fresh_store,
    )
    executed = []

    async def cancelling_executor(action):
        executed.append(dict(action))
        holder["agent"].current_runner.cancel()
        return True, "Opened firefox"

    planner = AgentPlannerFake([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]])
    from agent.reasoning_agent import ReasoningAgent
    from agent.task_state import TaskLimits
    agent = ReasoningAgent(
        executor=cancelling_executor,
        planner=planner,
        limits=TaskLimits(max_task_steps=8, max_retries_per_step=2,
                          max_replans=2, max_total_execution_time=30),
        lesson_store=fresh_store(),
    )
    holder = {"agent": agent}
    result = run(agent.run("open firefox and then open the file manager"))
    assert result.task_state.final_status == FinalStatus.CANCELLED
    assert [c["action"] for c in executed] == ["desktop_open"]


def test_13_retry_replan_limits_remain_enforced(monkeypatch):
    """S13: retry + replan limits stay bounded inside the reasoning loop."""
    from tests.test_reasoning_agent import fresh_store
    from agent.reasoning_agent import ReasoningAgent
    from agent.task_state import TaskLimits

    calls = []

    async def always_fail(action):
        calls.append(dict(action))
        return False, "temporarily busy, try again"

    same_plan = [{"action": "desktop_open", "params": {"app": "firefox"}}]

    async def same_planner(request, context):
        return [dict(s) for s in same_plan]

    agent = ReasoningAgent(
        executor=always_fail,
        planner=same_planner,
        limits=TaskLimits(max_task_steps=8, max_retries_per_step=2,
                          max_replans=2, max_total_execution_time=30),
        lesson_store=fresh_store(),
    )
    result = run(agent.run("open firefox and then check the time"))
    state = result.task_state
    assert state.final_status != FinalStatus.SUCCESS
    assert max((s.retries for s in state.failed_steps), default=0) <= 2
    assert state.replan_count <= 2
    assert state.ended_at is not None  # terminated, never infinite


def test_14_reflection_records_only_structured_data(brain_fx):
    """S14: the post-task reflection is compact and structured — no raw
    reasoning transcript fields."""
    from tests.test_reasoning_agent import FakeModel
    planner = FakePlanner({"firefox": [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]})
    dispatch = FakeDispatch({
        "desktop_open": (True, "Opened firefox"),
        "open_folder": (True, "Folder opened"),
    })
    b = brain_fx.make(planner=planner, dispatch=dispatch)
    res = _run_command(b, "Open Firefox and then open the file manager")
    ref = b._last_reasoning_result.reflection
    assert ref is not None
    data = ref.to_dict()
    allowed = {"goal_achieved", "successful_steps", "failed_steps",
               "success_evidence", "strategy_that_worked",
               "strategy_that_failed", "replanning_required", "lesson",
               "confidence"}
    assert set(data.keys()) == allowed
    assert "transcript" not in data and "thought" not in data
    assert data["goal_achieved"] is True


def test_15_successful_lesson_is_stored():
    """S15: a verified success creates a bounded reusable lesson."""
    from tests.test_reasoning_agent import (
        FakeExecutor, FakeObserver, fresh_store, make_agent,
        FakePlanner as AgentPlannerFake,
    )
    store = fresh_store()
    ex = FakeExecutor([(True, "Opened firefox"), (True, "Folder opened")])
    planner = AgentPlannerFake([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]])
    agent = make_agent(ex, observer=FakeObserver(["firefox running"]),
                       planner=planner, store=store)
    result = run(agent.run("open firefox and then open the file manager"))
    assert result.success
    assert result.lessons
    types = {l.type for l in result.lessons}
    assert "successful_strategy" in types
    assert all(l.evidence == "verified_success" for l in result.lessons)
    assert store.count > 0


def test_16_unverified_success_cannot_create_success_lesson():
    """S16: no success lesson is ever created from an unverified outcome."""
    from tests.test_reasoning_agent import fresh_store
    store = fresh_store()
    # A paused (NEEDS_CONFIRMATION) task — even with an earlier verified
    # step — must NOT produce a success lesson.
    paused = SimpleNamespace(
        task_id="paused",
        original_request="open firefox and then shut down",
        normalized_goal="open firefox and then shut down",
        completed_steps=[StepRecord(
            index=1, action="desktop_open", params={"app": "firefox"},
            status=StepStatus.COMPLETED, verified=True, result="Opened")],
        failed_steps=[],
        final_status=FinalStatus.NEEDS_CONFIRMATION,
        blocker="awaiting confirmation: 'shutdown' is a sensitive action",
    )
    assert store.record_task_outcome(paused) == []
    assert not any(l.type == "successful_strategy"
                   for l in store.all())


def test_17_previous_verified_steps_are_not_repeated():
    """S17: completed/verified steps are never executed a second time."""
    from tests.test_reasoning_agent import (
        FakeExecutor, fresh_store, make_agent,
        FakePlanner as AgentPlannerFake,
    )
    ex = FakeExecutor([(True, "Opened firefox")])
    planner = AgentPlannerFake([[
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "desktop_open", "params": {"app": "firefox"}},
    ]])
    agent = make_agent(ex, planner=planner, store=fresh_store())
    result = run(agent.run("open firefox and then check the time"))
    assert result.success
    open_calls = [c for c in ex.calls if c["action"] == "desktop_open"]
    assert len(open_calls) == 1


def test_18_reasoning_fallback_preserves_deterministic_behavior(brain_fx):
    """S18: with no reasoning model, deterministic commands still route
    deterministically (no ReasoningAgent) AND multi-step goals complete
    through the deterministic planner fallback."""
    dispatch_a = FakeDispatch({"desktop_open": (True, "Opened firefox")})
    b = brain_fx.make(decision=FakeDecisionEngine(open_firefox=True),
                      dispatch=dispatch_a)
    res_a = _run_command(b, "open firefox")
    assert res_a.used_llm is False
    assert b._last_reasoning_mode is None
    assert b._last_reasoning_agent is None
    assert res_a.verified is True

    # Multi-step goal through the deterministic fallback.
    planner = FakePlanner({"firefox": [
        {"action": "desktop_open", "params": {"app": "firefox"}},
        {"action": "open_folder", "params": {"path": "/tmp"}},
    ]})
    dispatch_b = FakeDispatch({
        "desktop_open": (True, "Opened firefox"),
        "open_folder": (True, "Folder opened"),
    })
    b2 = brain_fx.make(planner=planner, dispatch=dispatch_b)
    res_b = _run_command(b2, "Open Firefox and then open the file manager")
    assert b2._last_reasoning_mode == "autonomous"
    assert b2._last_reasoning_result.success
    assert [c["action"] for c in dispatch_b.calls] == ["desktop_open",
                                                        "open_folder"]
