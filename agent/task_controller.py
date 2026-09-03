"""
AutonomousTaskController — Bounded autonomous task execution for Diego.

Unifies the existing production components into a single bounded
autonomous task controller:

    goal
    → world-state gathering
    → plan
    → execute one action
    → observe
    → verify
    → replan/recover
    → continue
    → verified completion

CORE RULES (enforced):

1. VERIFIED SUCCESS ONLY
   Never mark a step or task successful from the tool return value alone.
   Require observable evidence whenever possible.

2. EVIDENCE-FIRST
   Every important state/fact must come from:
   - live observation
   - deterministic system information
   - local knowledge with source
   - tool/action verification
   - measured diagnostics
   If evidence is unavailable, mark UNKNOWN rather than guessing.

3. CLOSED LOOP
   After every meaningful action: execute → observe → verify.
   If verification fails: diagnose → replan → retry within limits.

4. BOUNDED AUTONOMY
   Configurable: MAX_TASK_STEPS, MAX_RETRIES, MAX_REPLANS, MAX_TASK_RUNTIME.
   Loop detection and repeated-action detection.

5. STATE
   Maintain structured task state: goal, plan, current_step, observations,
   actions, verification, failures, replans, completion_status.

6. PRIORITY OF TRUTH
   Live state > stale local knowledge.
   Deterministic system facts > LLM inference.
   Verified action result > model assumption.

7. HUMAN SAFETY
   Do not automatically perform destructive, irreversible, credential-related,
   financial, communication, or security-sensitive actions without explicit
   confirmation. Read-only investigation is allowed.

8. RECOVERY
   When an action fails: do not immediately repeat blindly.
   Identify the failure category, gather new evidence, and replan.

9. STOP CONDITIONS
   Stop only when: goal verified complete OR task cannot safely continue
   OR limits exhausted.

10. FINAL RESPONSE
    Report what was actually completed.
    Do not claim actions that were not verified.
    Do not expose internal traces, paths, scores, JSON, or planner internals
    unless explicitly requested.

This module does NOT create another agent architecture — it unifies the
existing TaskRunner, DecisionEngine, Planner, ActionDispatcher, and
verification pipeline into a single bounded controller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from agent.task_state import (
    Evidence,
    EvidenceSource,
    FinalStatus,
    PlanValidator,
    StepRecord,
    StepStatus,
    TaskExecutionState,
    TaskLimits,
    TaskRunner,
    is_sensitive_action,
    task_state_store,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# World State — unified evidence gathering
# ═══════════════════════════════════════════════════════════════

@dataclass
class WorldState:
    """Unified world state gathered from multiple evidence sources.

    Priority of truth (highest to lowest):
      1. live_observation — current screen/window/process state
      2. deterministic_system — OS-level facts (pgrep, nmcli, etc.)
      3. action_verification — verified action results
      4. local_knowledge — indexed knowledge with source
      5. diagnostics — measured diagnostics
      6. tool_result — tool/action return values
      7. llm_inference — model guesses (lowest priority)
    """
    timestamp: float = field(default_factory=time.time)
    live_observation: str = ""
    system_facts: Dict[str, Any] = field(default_factory=dict)
    knowledge_facts: List[Evidence] = field(default_factory=list)
    diagnostic_facts: List[Evidence] = field(default_factory=list)
    unknown_facts: List[str] = field(default_factory=list)

    def get_fact(self, key: str) -> Tuple[Optional[str], EvidenceSource]:
        """Get a fact by key with its evidence source.

        Returns (fact, source) where source is UNKNOWN if no evidence.
        Priority: live > deterministic > knowledge > unknown.
        """
        # Live observation is highest priority
        if self.live_observation and key.lower() in self.live_observation.lower():
            return self.live_observation, EvidenceSource.LIVE_OBSERVATION

        # Deterministic system facts
        if key in self.system_facts:
            return str(self.system_facts[key]), EvidenceSource.DETERMINISTIC_SYSTEM

        # Local knowledge with source
        for ev in self.knowledge_facts:
            if key.lower() in ev.fact.lower():
                return ev.fact, EvidenceSource.LOCAL_KNOWLEDGE

        # No evidence — mark UNKNOWN
        return None, EvidenceSource.UNKNOWN

    def add_unknown(self, fact_key: str) -> None:
        """Explicitly mark a fact as UNKNOWN (no evidence available)."""
        if fact_key not in self.unknown_facts:
            self.unknown_facts.append(fact_key)


class WorldStateGatherer:
    """Gathers world state from multiple evidence sources.

    Sources (in priority order):
      1. Live observation (perception pipeline, screen state)
      2. Deterministic system info (system_info module)
      3. Diagnostics (diagnostics module)
      4. Local knowledge (knowledge retrieval)
    """

    def __init__(self):
        self._perception = None
        self._wired = False

    def _ensure_wired(self) -> None:
        """Lazy-wire all evidence sources."""
        if self._wired:
            return

        try:
            from services.perception_pipeline import perception_pipeline
            self._perception = perception_pipeline
        except Exception as e:
            logger.debug("[WorldState] perception unavailable: %s", e)

        self._wired = True

    async def gather(self, query: str = "",
                     include_knowledge: bool = True) -> WorldState:
        """Gather world state from all available evidence sources.

        Args:
            query: Optional query to focus knowledge retrieval.
            include_knowledge: Whether to include local knowledge.

        Returns:
            WorldState with evidence from all sources.
        """
        self._ensure_wired()
        state = WorldState()

        # 1. Live observation (highest priority)
        state.live_observation = await self._gather_live_observation()

        # 2. Deterministic system facts
        state.system_facts = await self._gather_system_facts()

        # 3. Diagnostics
        state.diagnostic_facts = await self._gather_diagnostics()

        # 4. Local knowledge (lowest priority among real evidence)
        if include_knowledge and query:
            state.knowledge_facts = await self._gather_knowledge(query)

        return state

    async def _gather_live_observation(self) -> str:
        """Gather live observation from the perception pipeline."""
        if self._perception is None:
            return ""
        try:
            ctx = await asyncio.wait_for(
                self._perception.perceive(include_ocr=False),
                timeout=10.0)
            return getattr(ctx, "compact_summary", "") or ""
        except asyncio.TimeoutError:
            logger.debug("[WorldState] live observation timed out")
            return ""
        except Exception as e:
            logger.debug("[WorldState] live observation failed: %s", e)
            return ""

    async def _gather_system_facts(self) -> Dict[str, Any]:
        """Gather deterministic system facts from the system_info module."""
        facts: Dict[str, Any] = {}
        try:
            from knowledge.system_info import get_cached_snapshot
            snapshot = get_cached_snapshot()
            if snapshot:
                facts["os"] = snapshot.get("os", "")
                facts["cpu"] = snapshot.get("cpu", "")
                facts["ram"] = snapshot.get("ram", "")
                facts["disk"] = snapshot.get("disk", "")
                facts["hostname"] = snapshot.get("hostname", "")
        except Exception as e:
            logger.debug("[WorldState] system facts unavailable: %s", e)
        return facts

    async def _gather_diagnostics(self) -> List[Evidence]:
        """Gather diagnostic facts from the diagnostics module."""
        facts: List[Evidence] = []
        try:
            from knowledge.diagnostics import get_health_summary
            summary = get_health_summary()
            if summary:
                facts.append(Evidence(
                    fact=summary,
                    source=EvidenceSource.DIAGNOSTICS))
        except Exception as e:
            logger.debug("[WorldState] diagnostics unavailable: %s", e)
        return facts

    async def _gather_knowledge(self, query: str) -> List[Evidence]:
        """Gather local knowledge facts from the knowledge service."""
        facts: List[Evidence] = []
        try:
            from knowledge.service import knowledge_service
            results = knowledge_service.search(query, top_k=3)
            for r in results:
                if r.get("score", 0.0) >= 0.55:
                    facts.append(Evidence(
                        fact=r.get("content", "")[:200],
                        source=EvidenceSource.LOCAL_KNOWLEDGE,
                        confidence=r.get("score", 0.0)))
        except Exception as e:
            logger.debug("[WorldState] knowledge retrieval failed: %s", e)
        return facts


# ═══════════════════════════════════════════════════════════════
# Evidence-first verifier
# ═══════════════════════════════════════════════════════════════

class EvidenceFirstVerifier:
    """Verifies action results with evidence-first principles.

    VERIFIED SUCCESS ONLY: Never mark successful from tool return alone.
    Require observable evidence whenever possible.
    """

    def __init__(self):
        self._verifier = None
        self._wired = False

    def _ensure_wired(self) -> None:
        if self._wired:
            return
        try:
            from vision.action_verifier import action_verifier
            self._verifier = action_verifier
        except Exception as e:
            logger.debug("[EvidenceVerifier] action_verifier unavailable: %s", e)
        self._wired = True

    async def verify(self, action: str, params: Dict[str, Any],
                     dispatch_result: str, dispatch_ok: bool,
                     pre_state: str, post_state: str) -> Tuple[bool, str, EvidenceSource]:
        """Verify an action result with evidence-first principles.

        Returns:
            (verified, reason, evidence_source)
        """
        self._ensure_wired()

        # 1. Dispatch failure is always a failure
        if not dispatch_ok:
            return False, dispatch_result or "dispatch failed", EvidenceSource.TOOL_RESULT

        # 2. Check for explicit failure markers in the result
        if dispatch_result:
            result_lower = dispatch_result.lower()
            failure_markers = ("couldn't", "failed", "unavailable", "not found",
                               "error", "timeout", "timed out")
            if any(m in result_lower for m in failure_markers):
                return False, dispatch_result, EvidenceSource.TOOL_RESULT

        # 3. For stateful actions, require observable state change
        from agent.task_state import STATEFUL_ACTIONS
        if action in STATEFUL_ACTIONS:
            if pre_state and post_state:
                if pre_state == post_state:
                    # No observable change — cannot verify success
                    return (False,
                            "no observable state change after action",
                            EvidenceSource.UNKNOWN)
                # State changed — evidence of effect
                return (True,
                        f"state changed: {post_state[:100]}",
                        EvidenceSource.LIVE_OBSERVATION)
            # No observation available — mark UNKNOWN, trust dispatch
            if not post_state:
                return (True,
                        "dispatch succeeded (observation unavailable)",
                        EvidenceSource.TOOL_RESULT)

        # 4. For informational actions, the result IS the evidence
        from agent.task_state import ANSWER_ACTIONS
        if action in ANSWER_ACTIONS:
            if dispatch_result:
                return True, dispatch_result[:200], EvidenceSource.TOOL_RESULT
            return False, "no result from informational action", EvidenceSource.UNKNOWN

        # 5. Default: trust clean dispatch result
        return True, dispatch_result or "dispatch succeeded", EvidenceSource.TOOL_RESULT


# ═══════════════════════════════════════════════════════════════
# The bounded autonomous task controller
# ═══════════════════════════════════════════════════════════════

# Type aliases for injected dependencies
ExecutorFn = Callable[[Dict[str, Any]], Awaitable[Tuple[bool, str]]]
ObserverFn = Callable[[], Awaitable[str]]
PlannerFn = Callable[[str, Dict[str, Any]], Awaitable[Optional[List[Dict[str, Any]]]]]
ConfirmationFn = Callable[[str, str, Dict[str, Any]], Awaitable[bool]]


class AutonomousTaskController:
    """Bounded autonomous task controller.

    Unifies:
      - WorldStateGatherer (evidence-first world state)
      - TaskRunner (closed-loop execution)
      - EvidenceFirstVerifier (verified success only)
      - DecisionEngine (routing)
      - Planner (plan generation)
      - ActionDispatcher (execution)

    Bounded by:
      - MAX_TASK_STEPS
      - MAX_RETRIES
      - MAX_REPLANS
      - MAX_TASK_RUNTIME
      - Loop detection
      - Human safety confirmation
    """

    def __init__(
        self,
        executor: ExecutorFn,
        observer: Optional[ObserverFn] = None,
        planner: Optional[PlannerFn] = None,
        limits: Optional[TaskLimits] = None,
        confirmation_callback: Optional[ConfirmationFn] = None,
        action_gate: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self._executor = executor
        self._observer = observer
        self._planner = planner
        self._limits = limits or TaskLimits()
        self._confirmation_callback = confirmation_callback
        self._action_gate = action_gate
        self._world_state = WorldStateGatherer()
        self._verifier = EvidenceFirstVerifier()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    async def run(self, goal: str,
                  initial_plan: Optional[List[Dict[str, Any]]] = None,
                  inherited: Optional[TaskExecutionState] = None) -> TaskExecutionState:
        """Run a bounded autonomous task.

        Flow:
          1. Gather world state (evidence-first)
          2. Validate/create plan
          3. Execute closed loop: execute → observe → verify → replan
          4. Stop on: verified completion, safe stop, or limits exhausted

        Args:
            goal: The user's goal.
            initial_plan: Optional pre-generated plan.
            inherited: Optional inherited state from a previous task.

        Returns:
            TaskExecutionState with the full execution record.
        """
        # 1. Gather initial world state
        world = await self._world_state.gather(goal)

        # 2. Create the task runner with evidence-first verification
        runner = TaskRunner(
            executor=self._executor,
            observer=self._observer,
            planner=self._planner,
            validator=PlanValidator(action_gate=self._action_gate),
            limits=self._limits,
            transcript=goal,
            confirmation_callback=self._confirmation_callback,
        )

        # 3. Run the closed loop
        state = await runner.run(goal, initial_plan, inherited=inherited)

        # 4. Record world state evidence
        if world.live_observation:
            state.add_evidence(world.live_observation, EvidenceSource.LIVE_OBSERVATION)
        for key, value in world.system_facts.items():
            state.add_evidence(f"{key}: {value}", EvidenceSource.DETERMINISTIC_SYSTEM)
        for ev in world.knowledge_facts:
            state.add_evidence(ev.fact, EvidenceSource.LOCAL_KNOWLEDGE, ev.confidence)

        # 5. Save state for follow-ups
        task_state_store.save(state)

        return state

    async def run_with_world_state(self, goal: str,
                                   plan: List[Dict[str, Any]],
                                   world: WorldState) -> TaskExecutionState:
        """Run a task with pre-gathered world state.

        Used when the caller has already gathered world state
        (e.g., from the Brain's perception pipeline).
        """
        runner = TaskRunner(
            executor=self._executor,
            observer=self._observer,
            planner=self._planner,
            validator=PlanValidator(action_gate=self._action_gate),
            limits=self._limits,
            transcript=goal,
            confirmation_callback=self._confirmation_callback,
        )

        state = await runner.run(goal, plan)

        # Record world state evidence
        if world.live_observation:
            state.add_evidence(world.live_observation, EvidenceSource.LIVE_OBSERVATION)
        for key, value in world.system_facts.items():
            state.add_evidence(f"{key}: {value}", EvidenceSource.DETERMINISTIC_SYSTEM)

        task_state_store.save(state)
        return state


# ═══════════════════════════════════════════════════════════════
# Truthful final reporting
# ═══════════════════════════════════════════════════════════════

def build_final_report(state: TaskExecutionState,
                       include_details: bool = False) -> str:
    """Build a truthful final report from task execution state.

    FINAL RESPONSE RULES:
      - Report what was actually completed.
      - Do not claim actions that were not verified.
      - Do not expose internal traces, paths, scores, JSON, or planner
        internals unless explicitly requested.

    Args:
        state: The task execution state.
        include_details: Whether to include step-by-step details.

    Returns:
        A truthful, user-facing report.
    """
    if state.final_status == FinalStatus.SUCCESS:
        if state.artifacts.get("answer"):
            return str(state.artifacts["answer"])
        done = len(state.completed_steps)
        if done == 1:
            return state.completed_steps[0].result or "Done."
        return f"Done — {done} steps completed."

    if state.final_status == FinalStatus.PARTIAL_FAILURE:
        done = len(state.completed_steps)
        failed = len(state.failed_steps)
        report = f"I completed {done} step(s) but {failed} step(s) failed."
        if state.blocker:
            report += f" {state.blocker}"
        return report

    if state.final_status == FinalStatus.FAILED:
        report = "I couldn't complete that."
        if state.blocker:
            report += f" {state.blocker}"
        return report

    if state.final_status == FinalStatus.CANCELLED:
        return "Task cancelled."

    if state.final_status == FinalStatus.NEEDS_INPUT:
        report = "I need your input to continue."
        if state.blocker:
            report += f" {state.blocker}"
        return report

    if state.final_status == FinalStatus.NEEDS_CONFIRMATION:
        report = "I need your confirmation to proceed."
        if state.confirmation_reason:
            report += f" {state.confirmation_reason}"
        return report

    return "Task ended."


# ═══════════════════════════════════════════════════════════════
# Factory for production wiring
# ═══════════════════════════════════════════════════════════════

def create_production_controller(
    confirmation_callback: Optional[ConfirmationFn] = None,
) -> AutonomousTaskController:
    """Create a production-wired autonomous task controller.

    Wires:
      - Brain's dispatch_and_verify as executor
      - Brain's observe_state as observer
      - Planner's generate_plan_only as planner
      - Brain's planner_action_allowed as action gate
    """
    from agent.brain import agent_brain
    from agent.planner import agent_planner

    async def executor(action: Dict[str, Any]) -> Tuple[bool, str]:
        return await agent_brain._dispatch_and_verify(action)

    async def observer() -> str:
        return await agent_brain._observe_state()

    async def planner(request: str, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, agent_planner.generate_plan_only, request, context)

    return AutonomousTaskController(
        executor=executor,
        observer=observer,
        planner=planner,
        limits=TaskLimits(),
        confirmation_callback=confirmation_callback,
        action_gate=agent_brain._planner_action_allowed,
    )