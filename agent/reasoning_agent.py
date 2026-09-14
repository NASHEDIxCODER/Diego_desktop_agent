"""
ReasoningAgent — reasoning-capable autonomous task core (Phase 21A).

Upgrades the EXISTING bounded loop (agent.task_state.TaskRunner +
agent.task_controller.AutonomousTaskController) with a reasoning layer.
This is NOT a competing architecture: the deterministic runtime keeps full
authority over authorization, tool execution, confirmation, state
transitions, verification, retry limits, cancellation, timeouts and
persistence. The model only:

    understand goals → reason about options → construct plans →
    interpret observations → propose recovery → summarize outcomes →
    extract reusable lessons

Reasoning loop (bounded by the existing TaskLimits + LoopDetector):

    GOAL → UNDERSTAND → PLAN → EXECUTE → OBSERVE → VERIFY → REFLECT
         → CONTINUE / RECOVER / REPLAN / ASK

Reasoning modes (Task 14):
    DETERMINISTIC — simple known commands stay fast (no model calls)
    REASONING     — complex/ambiguous goals use the reasoning model
    AUTONOMOUS    — multi-step goals: planning + execution + verification
    RECOVERY      — failures trigger diagnosis + alternative strategy
    REFLECTION    — completed tasks produce reusable lessons

Trivial commands are NEVER sent through expensive reasoning.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from agent.reasoning_context import ReasoningContextComposer
from agent.reasoning_state import (
    FailureAnalysis,
    NEXT_STRATEGIES,
    ReasoningPhase,
    ReasoningState,
    TaskReflection,
    extract_constraints,
)
from agent.lessons import (
    TaskLesson,
    TaskLessonStore,
    task_lesson_store,
    task_pattern_from_goal,
)

logger = logging.getLogger(__name__)


from enum import Enum


class Mode(str, Enum):
    DETERMINISTIC = "deterministic"
    REASONING = "reasoning"
    AUTONOMOUS = "autonomous"
    RECOVERY = "recovery"
    REFLECTION = "reflection"


# Backwards-friendly alias (single canonical enum)
ReasoningMode = Mode  # noqa: F811

# ── Mode selection (Task 14) — deterministic heuristics only ──

_SIMPLE_PATTERNS = (
    r"^(open|launch|start)\s+[\w.\- ]+$",
    r"^(close|quit|kill)\s+[\w.\- ]+$",
    r"^(volume|brightness)\s+(up|down|mute)$",
    r"^(volume|brightness)\s+(to|set)\s*\d+%?$",
    r"^set\s+(volume|brightness)\s+(to\s*)?\d+%?$",
    r"^what('s| is)?\s+(the\s+)?(time|date)(\s+now|\s+today)?$",
    r"^(pause|resume|next|previous|stop)\s*(music|song|track)?$",
    r"^play\s+.+$",
    r"^(turn\s+)(on|off)\s+(wifi|bluetooth)$",
    r"^(mute|unmute)$",
)
_MULTI_STEP_MARKERS = re.compile(
    r"\b(then|after that|and then|afterwards|following that|" +
    r"first\b.*\bthen\b)\b", re.IGNORECASE)


def choose_mode(text: str) -> Mode:
    """Mode A/B/C selection. Recovery (D) and reflection (E) are loop
    outcomes rather than entry modes."""
    t = " ".join((text or "").lower().strip().split())
    if not t:
        return Mode.DETERMINISTIC
    # Multi-step goals are checked FIRST: a simple-command pattern can
    # otherwise swallow phrases like "open firefox and then nautilus".
    if _MULTI_STEP_MARKERS.search(t):
        return Mode.AUTONOMOUS
    for pattern in _SIMPLE_PATTERNS:
        if re.match(pattern, t):
            return Mode.DETERMINISTIC
    return Mode.REASONING


# ── Task 6: deterministic failure diagnosis (fallback path) ──

_KIND_TO_STRATEGY = {
    "transient": (True, "retry"),
    "wrong_params": (True, "repair_params"),
    "changed_state": (True, "observe_and_adapt"),
    "wrong_tool": (False, "alternative_tool"),
    "unavailable_capability": (False, "ask_user"),
    "impossible": (False, "stop"),
    "unknown": (False, "replan"),
}


def diagnose_failure_deterministic(action: str, error: str,
                                   attempt: int) -> FailureAnalysis:
    """Structured failure context WITHOUT a model (existing classifier)."""
    from agent.task_state import classify_failure
    kind = classify_failure(action, "", error)
    kind_value = kind.value if hasattr(kind, "value") else str(kind)
    retry_suitable, strategy = _KIND_TO_STRATEGY.get(
        kind_value, (False, "replan"))
    probable = {
        "transient": "temporary/system timing issue — a retry can succeed",
        "wrong_params": "parameters did not match reality",
        "changed_state": "the environment moved between plan and action",
        "wrong_tool": "the chosen action is not available/appropriate",
        "unavailable_capability": "the capability is not installed/available",
        "impossible": "the requested effect cannot be achieved",
        "unknown": "insufficient evidence for a specific cause",
    }.get(kind_value, "insufficient evidence")
    return FailureAnalysis(
        action=action,
        observed_result=(error or "")[:400],
        failure_kind=kind_value,
        probable_cause=probable,
        retry_suitable=retry_suitable,
        next_strategy=strategy if strategy in NEXT_STRATEGIES else "replan",
    )


@dataclass
class ReasoningRunResult:
    """Outcome of one reasoning-agent run (bounded, structured)."""
    task_state: Any = None                # TaskExecutionState (authoritative)
    reasoning_state: Optional[ReasoningState] = None
    reflection: Optional[TaskReflection] = None
    lessons: List[TaskLesson] = field(default_factory=list)
    mode: Mode = Mode.REASONING
    # Phase timing profile (ms) populated by _phase() / _model_call().
    # Keys: context, lessons, plan, plan_model, execute, revise, diagnose,
    # replan, replan_model, reflect, reflection_model, validate, persist,
    # total, plus model_calls (int) and context_tokens (int).
    profile: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        from agent.task_state import FinalStatus
        return (self.task_state is not None
                and self.task_state.final_status == FinalStatus.SUCCESS)


class ReasoningAgent:
    """Reasoning wrapper around the EXISTING bounded TaskRunner loop.

    The agent owns the structured reasoning state (agent.reasoning_state),
    the layered context (agent.reasoning_context over ai.context_monitor),
    lesson retrieval/reinforcement (agent.lessons) and model-backed
    diagnosis / adaptive plan revision (ai.reasoning_model). Execution,
    verification, confirmation, cancellation and every limit remain in
    the deterministic runtime.
    """

    def __init__(self,
                 executor: Callable[[Dict[str, Any]],
                                     Awaitable[Tuple[bool, str]]],
                 observer: Optional[Callable[[], Awaitable[str]]] = None,
                 planner: Optional[Callable[[str, Dict[str, Any]],
                                            Awaitable[Optional[
                                                List[Dict[str, Any]]]]]] = None,
                 reasoning_model: Optional[Any] = None,
                 limits: Optional[Any] = None,
                 confirmation_callback: Optional[Callable] = None,
                 action_gate: Optional[Callable] = None,
                 lesson_store: Optional[TaskLessonStore] = None,
                 composer: Optional[ReasoningContextComposer] = None,
                 transcript: str = "",
                 approved_actions: Optional[frozenset] = None,
                 ):
        from agent.task_state import TaskLimits
        self._executor = executor
        self._observer = observer
        self._planner = planner
        self._model = reasoning_model
        self._limits = limits or TaskLimits()
        self._confirmation_callback = confirmation_callback
        self._action_gate = action_gate
        self._lesson_store = lesson_store or task_lesson_store
        self._composer = composer or ReasoningContextComposer()
        self._transcript = transcript
        self._approved_actions = approved_actions or frozenset()
        self.current_runner: Optional[Any] = None
        self.rs: Optional[ReasoningState] = None
        # Instrumentation (Phase 21B): how many reasoning-model calls this
        # agent actually made. Deterministic modes must keep it 0.
        self.model_calls: int = 0
        # Phase 21C: per-phase timing profile (latency in ms) + model-call
        # counts + context token usage. Populated during run() and attached
        # to the result.profile. Zero-cost (perf_counter) when not read.
        self._profile: Dict[str, Any] = {}
        self._context_tokens: int = 0
        # Phase 21D: background reflection task (deferred so SUCCESS does not
        # wait for the expensive model-reflection call). None when idle.
        self._pending_reflection: Optional[asyncio.Task] = None
        # Phase 21D: tracks the last observation for the conservative
        # revision-skip check (environment-stability detection).
        self._last_observation: Optional[str] = None
        # Lessons consulted during THIS run (for reuse feedback, Task 10).
        self._lessons_used: List[Tuple[TaskLesson, float]] = []
        self.last_diagnosis: Optional[FailureAnalysis] = None

    # ── Control ────────────────────────────────────────────────

    def cancel(self) -> None:
        if self.current_runner is not None:
            self.current_runner.cancel()

    # ── Phase 21C: minimal latency instrumentation ─────────────

    @contextlib.contextmanager
    def _phase(self, name: str):
        """Time a (sync or async) phase, accumulating ms into _profile[name]."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self._profile[name] = self._profile.get(name, 0.0) + ms

    async def _model_call(self, name: str, coro) -> Any:
        """Time one reasoning-model call, record ms + increment model_calls."""
        t0 = time.perf_counter()
        try:
            return await coro
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self._profile[name] = self._profile.get(name, 0.0) + ms
            self.model_calls += 1

    # ── Phase 21D: deferred (background) reflection ───────────

    async def shutdown(self) -> None:
        """Cancel any pending background reflection (call on shutdown).
        Reflection is optional learning — never blocks shutdown."""
        pending = self._pending_reflection
        self._pending_reflection = None
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, Exception):
                pass

    def _schedule_reflection(self, goal: str, task_state, rs: ReasoningState,
                             reflection) -> None:
        """Schedule the expensive model-reflection call to run in the
        background AFTER the task result is returned. No-op for deterministic
        tasks, when no model is configured, or when one is already pending
        (prevents duplicate reflections)."""
        if self._mode is Mode.DETERMINISTIC:
            return
        if self._model is None:
            return
        if self._pending_reflection is not None \
                and not self._pending_reflection.done():
            return  # already reflecting — do not duplicate
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop; skip background reflection
        self._pending_reflection = loop.create_task(
            self._reflect_background(goal, task_state, rs, reflection))

    async def _reflect_background(self, goal: str, task_state, rs,
                                  reflection) -> None:
        """Background model-reflection. Runs after SUCCESS is returned.
        Any failure is swallowed — it must not affect the completed task."""
        try:
            outcome = (
                f"status={task_state.final_status.value if task_state.final_status else ''} "
                f"completed={len([s for s in (task_state.completed_steps or []) if s.verified])} "
                f"failed={len(task_state.failed_steps or [])} "
                f"replans={task_state.replan_count} "
                f"blocker={task_state.blocker[:100]}")
            result = await self._model_call(
                "reflect_model",
                self._model.reflect(goal, outcome))
            if result is not None and result.ok:
                data = result.data or {}
                # Model may only REFINE — never flip verified facts.
                if task_state.final_status == FinalStatus.SUCCESS \
                        and data.get("goal_achieved") is True:
                    pass
                reflection.strategy_that_worked = (
                    reflection.strategy_that_worked
                    or str(data.get("what_worked", ""))[:200])
                reflection.strategy_that_failed = (
                    reflection.strategy_that_failed
                    or str(data.get("what_failed", ""))[:200])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("[ReasoningAgent] background reflection failed "
                         "safely: %s", e)

    # ── Main entry (Task 4 loop) ───────────────────────────────

    async def run(self, goal: str,
                  initial_plan: Optional[List[Dict[str, Any]]] = None,
                  inherited: Optional[Any] = None,
                  mode: Optional[Mode] = None,
                  ) -> ReasoningRunResult:
        from agent.task_state import (
            FinalStatus, PlanValidator, TaskRunner,
        )
        mode = mode or choose_mode(goal)
        self._mode = mode  # Phase 21C: lets model-call sites respect mode
        self._last_observation = None  # reset per task (Phase 21D)

        # ── UNDERSTAND ─────────────────────────────────────────
        rs = ReasoningState(goal=goal)
        for constraint in extract_constraints(goal):
            rs.note_constraint(constraint)
        if inherited is not None:
            # Task continuation: inherit observed state + verified record
            # so reasoning is never lost between continuations.
            if inherited.observed_state:
                rs.note_observation(inherited.observed_state)
            for s in inherited.completed_steps or []:
                if s.verified:
                    rs.note_completed_step(
                        f"{s.action} {s.params} — {s.verification}")
            for s in inherited.failed_steps or []:
                rs.note_failure(diagnose_failure_deterministic(
                    s.action, s.error or s.result, s.retries))
            for k, v in (inherited.artifacts or {}).items():
                if k in ("urls", "last_url", "last_app", "last_query"):
                    rs.context_refs.append(f"artifact:{k}={v}")
        self.rs = rs

        # ── Memory reuse BEFORE planning (Task 10) ─────────────
        with self._phase("lessons"):
            self._lessons_used = []
            pattern = task_pattern_from_goal(goal)
            lesson_hits = self._lesson_store.retrieve(
                f"{pattern} {goal}", limit=4) if self._lesson_store else []
            self._lessons_used = list(lesson_hits)
            lesson_lines = [lesson.line() for lesson, _s in lesson_hits]
            for lesson, _score in lesson_hits:
                rs.context_refs.append(f"lesson:{lesson.id}")
                rs.note_assumption(
                    f"prior evidence suggests: {lesson.lesson[:80]}")

        # Tool reliability snapshot (bounded, best-effort).
        try:
            from core.tool_reliability import tool_reliability
            rs.set_tool_reliability(tool_reliability.all_confidences())
        except Exception as e:
            logger.debug("[ReasoningAgent] tool reliability unavailable: %s", e)

        # ── PLAN ───────────────────────────────────────────────
        rs.phase = ReasoningPhase.PLAN
        plan = initial_plan
        plan_assumptions: List[str] = []
        if not plan and mode is not Mode.DETERMINISTIC:
            with self._phase("plan"):
                plan, plan_assumptions = await self._initial_plan(
                    goal, rs, lesson_lines, pattern)
        with self._phase("validate"):
            plan = self._validated(plan)
        if plan:
            rs.plan = plan
            rs.plan_version = 1
            rs.current_objective = (plan[0].get("description")
                                    or str(plan[0].get("action", "")))
            rs.next_action = str(plan[0].get("action", ""))
        for a in plan_assumptions[:8]:
            rs.note_assumption(a)
        if not plan and not mode_changed_needed(mode):
            rs.uncertainty.append("no plan could be constructed")

        # ── EXECUTE (existing bounded closed loop) ─────────────
        rs.phase = ReasoningPhase.EXECUTE
        with self._phase("execute"):
            runner = TaskRunner(
                executor=self._executor,
                observer=self._observer,
                planner=self._replan_adapter,
                validator=PlanValidator(action_gate=self._action_gate),
                limits=self._limits,
                transcript=self._transcript or goal,
                confirmation_callback=self._wrap_confirmation(),
                approved_actions=self._approved_actions,
                step_adapter=(self._adaptive_step_hook
                              if mode is not Mode.DETERMINISTIC else None),
            )
            self.current_runner = runner
            task_state = await runner.run(goal, plan, inherited=inherited)
            self.current_runner = None

        # ── Sync reasoning state from the AUTHORITATIVE record ─
        self._sync_from_task_state(rs, task_state)

        # ── Lesson reuse feedback (Task 10) ────────────────────
        task_success = task_state.final_status == FinalStatus.SUCCESS
        for lesson, _score in self._lessons_used:
            try:
                self._lesson_store.record_reuse(lesson.id, task_success)
            except Exception:
                pass

        # ── REFLECT + learn (Tasks 11 / 9) ─────────────────────
        rs.phase = ReasoningPhase.REFLECT
        # Deterministic (CPU-only) reflection + lesson persistence happen
        # synchronously — they are fast and the lessons are valuable.
        with self._phase("reflect"):
            reflection = self._build_reflection(goal, task_state, rs)
        reflection.confidence = rs.confidence
        with self._phase("persist"):
            lessons = []
            try:
                lessons = self._lesson_store.record_task_outcome(task_state)
            except Exception as e:
                logger.debug("[ReasoningAgent] lesson extraction failed: %s", e)
        reflection.lesson = (lessons[0].to_dict() if lessons else None)
        rs.phase = ReasoningPhase.DONE
        # Attach the timing profile (Phase 21C). Includes per-phase ms,
        # total wall-clock, model-call count and context-token usage.
        self._profile["total"] = sum(
            v for v in self._profile.values() if isinstance(v, (int, float)))
        self._profile["model_calls_total"] = self.model_calls
        self._profile["context_tokens"] = self._context_tokens
        result = ReasoningRunResult(
            task_state=task_state,
            reasoning_state=rs,
            reflection=reflection,
            lessons=lessons,
            mode=mode,
            profile=dict(self._profile),
        )
        # Phase 21D: schedule the EXPENSIVE model-reflection call to run in
        # the background AFTER the result is returned. SUCCESS does not wait.
        self._schedule_reflection(goal, task_state, rs, reflection)
        return result

    # ── Planning helpers ───────────────────────────────────────

    def _compose_context(self, goal: str, rs: Optional[ReasoningState],
                         lesson_lines: List[str],
                         knowledge: Optional[List[str]] = None,
                         history: Optional[List[str]] = None,
                         ) -> Any:
        """Task 3: layered context via the EXISTING monitor."""
        rs = rs or self.rs or ReasoningState()
        return self._composer.compose(
            goal=goal,
            constraints=rs.user_constraints,
            state_lines=rs.state_lines(),
            current_objective=rs.current_objective,
            recent_observations=rs.observations,
            verified_evidence=rs.verified_evidence,
            conversation_history=history or [],
            knowledge_facts=knowledge or [],
            lesson_lines=lesson_lines or [],
            background=rs.summarized_history,
            completed_step_summaries=rs.completed_step_summaries,
        )

    def _validated(self, plan: Optional[List[Dict[str, Any]]]
                   ) -> List[Dict[str, Any]]:
        from agent.task_state import PlanValidator
        if not plan:
            return []
        validator = PlanValidator(action_gate=self._action_gate)
        return validator.validate_plan(plan, [], self._transcript or
                                       (self.rs.goal if self.rs else ""))

    async def _initial_plan(self, goal: str, rs: ReasoningState,
                            lesson_lines: List[str], pattern: str,
                            ) -> Tuple[Optional[List[Dict[str, Any]]],
                                       List[str]]:
        """PLAN phase: deterministic planner first, reasoning model second.
        Both proposals are VALIDATED by the runtime before execution."""
        with self._phase("context"):
            composed = self._compose_context(goal, rs, lesson_lines)
        self._context_tokens = composed.tokens_used
        planner_context = {
            "goal": goal,
            "observed_state": rs.observations[-1] if rs.observations else "",
            "completed": list(rs.completed_step_summaries),
            "failed": [f"{fa.action}: {fa.observed_result[:80]}"
                       for fa in rs.failed_attempts],
            "failure_context": "",
            "lessons": lesson_lines,
            "context_block": composed.text,
            "replan": 0,
        }
        if self._planner:
            with self._phase("plan_adapters"):
                try:
                    plan = await self._planner(goal, planner_context)
                    if plan:
                        return plan, []
                except Exception as e:
                    logger.warning("[ReasoningAgent] planner failed: %s", e)
        if self._model is not None:
            result = await self._model_call(
                "plan_model", self._model.plan(goal, composed.text))
            if result.ok:
                data = result.data or {}
                if data.get("needs_input"):
                    rs.phase = ReasoningPhase.ASK_USER
                    rs.uncertainty.append(
                        str(data.get("question", "missing information"))[:200])
                    return None, []
                steps = data.get("plan") or []
                if steps:
                    return steps, [str(a)[:200] for a in
                                   (data.get("assumptions") or [])[:8]]
            else:
                # Model failure is safe: no plan, deterministic path.
                logger.info("[ReasoningAgent] model plan unavailable (%s)",
                            result.status.value)
                rs.uncertainty.append(
                    f"model unavailable: {result.status.value}")
        return None, []

    # ── Task 6: failure diagnosis feeding bounded replans ─────

    async def _replan_adapter(self, request: str,
                              context: Dict[str, Any]
                              ) -> Optional[List[Dict[str, Any]]]:
        """Planner adapter for TaskRunner re-plans: builds a structured
        failure diagnosis (model with deterministic fallback), composes
        layered context, then re-plans from the CURRENT observed state."""
        rs = self.rs or ReasoningState()
        failure_context = str(context.get("failure_context", ""))
        failed_list = list(context.get("failed") or [])
        last_failed = failed_list[-1] if failed_list else ""
        action = last_failed.split(":", 1)[0].strip() if last_failed else ""
        error = failure_context or last_failed

        diagnosis: Optional[FailureAnalysis] = None
        if (self._model is not None
                and self._mode is not Mode.DETERMINISTIC):
            try:
                result = await self._model_call(
                    "diagnose",
                    self._model.diagnose(
                        request, action,
                        str(context.get("observed_state", "")), error,
                        attempt=len(failed_list) + 1))
                if result.ok:
                    d = result.data or {}
                    strategy = str(d.get("next_strategy", ""))
                    diagnosis = FailureAnalysis(
                        action=action,
                        observed_result=error[:400],
                        failure_kind=str(d.get("failure_kind", "unknown")),
                        probable_cause=str(d.get("probable_cause", ""))[:200],
                        retry_suitable=bool(d.get("retry_suitable", False)),
                        alternative=str(d.get("alternative", ""))[:200],
                        next_strategy=(strategy if strategy in NEXT_STRATEGIES
                                       else "replan"),
                    )
            except Exception as e:
                logger.debug("[ReasoningAgent] model diagnosis failed: %s", e)
        if diagnosis is None:
            diagnosis = diagnose_failure_deterministic(
                action, error, len(failed_list) + 1)
        self.last_diagnosis = diagnosis
        rs.phase = ReasoningPhase.RECOVER
        rs.note_failure(diagnosis)
        if diagnosis.alternative:
            rs.note_alternative(diagnosis.alternative)

        lesson_lines = self._current_lesson_lines()
        with self._phase("context_replan"):
            composed = self._compose_context(request, rs, lesson_lines)
        enriched = dict(context)
        enriched["failure_diagnosis"] = diagnosis.to_dict()
        enriched["lessons"] = lesson_lines
        enriched["reasoning_state"] = rs.to_dict()
        enriched["context_block"] = composed.text
        enriched["note"] = "prior lessons are evidence, not truth — " \
                           "verify the current environment"

        if self._planner:
            with self._phase("plan_adapters_replan"):
                try:
                    plan = await self._planner(request, enriched)
                    if plan:
                        return plan
                except Exception as e:
                    logger.warning("[ReasoningAgent] replan planner failed: %s", e)
        if (self._model is not None
                and self._mode is not Mode.DETERMINISTIC):
            diag_block = (f"FAILURE DIAGNOSIS: {diagnosis.to_dict()}\n"
                          f"CONTEXT:\n{composed.text}")
            try:
                result = await self._model_call(
                    "replan_model",
                    self._model.plan(request, diag_block))
            except Exception as e:
                logger.debug("[ReasoningAgent] model replan failed: %s", e)
                result = None
            if result is not None and result.ok:
                data = result.data or {}
                if data.get("needs_input"):
                    rs.phase = ReasoningPhase.ASK_USER
                    rs.uncertainty.append(
                        str(data.get("question", "missing information"))[:200])
                    return None
                steps = data.get("plan") or []
                if steps:
                    return steps
        return None

    # ── Phase 21D: deterministic revision necessity check ────

    @staticmethod
    def _step_target(rec) -> str:
        """Extract a concrete target token from a completed step to compare
        against the observation (e.g. an app name or path)."""
        params = rec.params or {}
        for key in ("app", "path", "url", "query", "pattern", "file"):
            val = str(params.get(key, "")).strip()
            if val:
                return val
        return str(rec.action or "").strip()

    def _observation_confirms_step(self, rec, observed: str) -> bool:
        """Cheap deterministic check: does the observed state mention the
        target of the just-completed step? If yes, the plan is on track."""
        target = self._step_target(rec)
        if not target or not observed:
            return False
        obs = observed.lower()
        # Match the full target or its last path/component token.
        if target.lower() in obs:
            return True
        token = target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return len(token) >= 2 and token.lower() in obs

    def _revision_needed(self, state, rec, remaining) -> bool:
        """Deterministic decision whether the remaining plan may be stale and
        therefore warrants an expensive model revision. Revision is SKIPPED
        only when the environment is COMPLETELY STABLE (observation identical
        to the previous step's) and the remaining plan has no unresolved
        dependencies — the conservative case where the plan cannot change."""
        # A prior failure/recovery means the plan may be stale → revise.
        if state.failed_steps:
            return True
        # Unverified step → state uncertain → revise.
        if not rec.verified:
            return True
        observed = (state.observed_state or rec.result or "").strip()
        # No observation: cannot assess, default to revise (conservative).
        if not observed:
            return True
        # Environment completely stable (observation identical to the
        # previous step's) AND remaining deps resolved → skip revision.
        if observed == (self._last_observation or None) or \
                (self._last_observation is not None and
                 observed == self._last_observation):
            if not self._remaining_has_unresolved_deps(remaining):
                return False
        # Any new/different observation → potential state change → revise.
        return True

    @staticmethod
    def _remaining_has_unresolved_deps(remaining) -> bool:
        """True if any remaining step has a placeholder/empty required
        parameter, i.e. a dependency the plan cannot yet resolve."""
        for step in remaining:
            params = step.get("params") or {}
            if not params:
                continue
            for v in params.values():
                s = str(v).strip()
                if not s or s.startswith("<") or s.startswith("?") or \
                   s.lower() in ("todo", "tbd", "unknown"):
                    return True
        return False

    # ── Task 7: adaptive plan revision after a VERIFIED step ──

    async def _adaptive_step_hook(self, state, rec,
                                  remaining: List[Dict[str, Any]]
                                  ) -> Optional[List[Dict[str, Any]]]:
        """Ask the reasoning model whether the observation invalidated the
        remaining plan. The RUNTIME still validates/bounds any revision.
        Phase 21D: skips the model call when deterministic checks show the
        remaining plan is still valid (saves a model call per stable step)."""
        if self._model is None or not remaining:
            return None
        observed = (state.observed_state or rec.result or "").strip()
        # Deterministic skip: only revise if the plan may actually be stale.
        revision_needed = self._revision_needed(state, rec, remaining)
        # Always track the observation for the next step's stability check
        # (must happen on EVERY step, not only when revising).
        self._last_observation = observed
        if not revision_needed:
            return None
        rs = self.rs
        if rs is not None:
            rs.phase = ReasoningPhase.OBSERVE
            rs.note_observation(observed)
            rs.note_completed_step(
                f"{rec.action} {rec.params} — verified")
        try:
            result = await self._model_call(
                "revise",
                self._model.revise_plan(
                    state.normalized_goal,
                    state.observed_state or rec.result or "",
                    [f"{s.action} {s.params}" for s in state.completed_steps],
                    remaining,
                ))
        except Exception as e:
            logger.debug("[ReasoningAgent] revise_plan failed: %s", e)
            return None
        if rs is not None:
            rs.phase = ReasoningPhase.VERIFY
        if not result.ok:
            return None
        data = result.data or {}
        if data.get("keep_plan", True):
            return None
        revised = data.get("revised_plan") or []
        if rs is not None and revised:
            rs.adaptive_revision_count += 1
            rs.note_alternative(str(data.get("reason", "plan revised"))[:200])
        return revised or None

    def _current_lesson_lines(self) -> List[str]:
        try:
            return [l.line() for l, _s in self._lessons_used]
        except Exception:
            return []

    # ── Confirmation wrapper: user decisions enter the state ──

    def _wrap_confirmation(self) -> Optional[Callable]:
        base = self._confirmation_callback
        if base is None:
            return None

        async def wrapped(action: str, reason: str,
                          params: Dict[str, Any]) -> bool:
            approved = bool(await base(action, reason, params))
            if self.rs is not None:
                self.rs.note_user_decision(
                    f"{'approved' if approved else 'denied'} {action}")
                self.rs.phase = ReasoningPhase.ASK_USER
            return approved

        return wrapped

    # ── Task 5/11: authoritative sync + bounded reflection ────

    def _sync_from_task_state(self, rs: ReasoningState,
                              task_state) -> None:
        """Sync the reasoning state from the AUTHORITATIVE task record.
        Evidence comes only from verified steps / real observations."""
        from agent.task_state import FinalStatus, StepStatus
        rs.task_id = task_state.task_id
        if task_state.observed_state:
            rs.note_observation(task_state.observed_state)
        for s in task_state.completed_steps or []:
            if s.verified and s.status in (
                    StepStatus.COMPLETED, StepStatus.ALREADY_SATISFIED):
                rs.note_completed_step(f"{s.action} {s.params} — verified")
                if s.result:
                    rs.note_evidence(f"{s.action}: {s.result}")
        for s in task_state.failed_steps or []:
            rs.note_failure(diagnose_failure_deterministic(
                s.action, s.error or s.result or "", s.retries))
        rs.replan_count = max(rs.replan_count, task_state.replan_count)
        rs.plan = list(task_state.current_plan or [])
        rs.plan_version = task_state.plan_version
        if task_state.final_status == FinalStatus.SUCCESS:
            rs.confidence = 0.9
            rs.phase = ReasoningPhase.VERIFY
        elif task_state.final_status in (
                FinalStatus.NEEDS_INPUT, FinalStatus.NEEDS_CONFIRMATION):
            rs.confidence = 0.4
            rs.phase = ReasoningPhase.ASK_USER
        elif task_state.final_status == FinalStatus.CANCELLED:
            rs.confidence = 0.3
        else:
            rs.confidence = 0.2
            rs.phase = ReasoningPhase.RECOVER
        if task_state.blocker:
            rs.uncertainty.append(task_state.blocker[:200])

    def _build_reflection(self, goal: str, task_state,
                          rs: ReasoningState) -> TaskReflection:
        """Task 11: bounded structured reflection (never a transcript). Phase 21D: now CPU-only (model call deferred to background)."""
        from agent.task_state import FinalStatus, StepStatus
        success = task_state.final_status == FinalStatus.SUCCESS
        completed = [s for s in task_state.completed_steps or []
                     if s.verified and s.status in (
                         StepStatus.COMPLETED, StepStatus.ALREADY_SATISFIED)]
        failed = list(task_state.failed_steps or [])
        reflection = TaskReflection(
            goal_achieved=success,
            successful_steps=[f"{s.action} {s.params}" for s in completed],
            failed_steps=[f"{s.action}: {(s.error or s.result or '')[:120]}"
                          for s in failed],
            success_evidence=[(s.result or s.verification)[:200]
                              for s in completed],
            strategy_that_worked=(" → ".join(
                s.action for s in completed) if success else ""),
            strategy_that_failed=(" → ".join(
                s.action for s in failed) if failed else ""),
            replanning_required=(task_state.replan_count > 0),
            confidence=rs.confidence,
        )
        # NOTE (Phase 21D): the expensive model-reflection call is no longer
        # made here. It runs in the background via _schedule_reflection()
        # after the task result is returned, so SUCCESS does not wait for it.
        # This method now builds the deterministic (CPU-only) reflection.
        return reflection

    def mode(self) -> Mode:
        return Mode.REASONING

    def error_result(self, goal: str, error: str) -> ReasoningRunResult:
        """Safe, honest failure result when the loop itself cannot start
        (must never be SUCCESS). Used by the Brain orchestrator only."""
        from agent.task_state import (
            FinalStatus, TaskExecutionState, task_state_store,
        )
        state = TaskExecutionState(original_request=goal,
                                   normalized_goal=goal)
        state.final_status = FinalStatus.FAILED
        state.blocker = str(error)[:200]
        state.ended_at = time.time()
        state.note("[TASK] Reasoning loop failed safely — honest failure")
        try:
            task_state_store.save(state)
        except Exception:
            pass
        return ReasoningRunResult(
            task_state=state,
            reasoning_state=self.rs,
            reflection=None,
            lessons=[],
            mode=Mode.REASONING,
        )


def mode_changed_needed(mode: Mode) -> bool:
    """True when the mode is allowed to construct a plan via the model
    (deterministic mode keeps the fast path)."""
    return mode is not Mode.DETERMINISTIC


# Global singleton factory helper.
def create_reasoning_agent(executor, **kwargs) -> ReasoningAgent:
    """Build a ReasoningAgent with the configured reasoning model."""
    from ai.reasoning_model import get_reasoning_model
    kwargs.setdefault("reasoning_model", get_reasoning_model())
    return ReasoningAgent(executor=executor, **kwargs)
