"""
Phase 24: DesktopGoalEngine — goal-directed desktop orchestration for Diego.

This layer makes Diego achieve desktop-application GOALS (the first real one
being Telegram Desktop messaging) instead of firing isolated desktop actions.
It orchestrates existing capabilities and duplicates none:

    goal parsing      → agent.desktop_goal (pure vocabulary + contracts)
    desktop state     → agent.desktop_context (structured DesktopContext)
    skills            → agent.desktop_skill_registry (semantic capability map)
    execution         → computer.computer_controller (Phase 22 facade)
    element location  → computer.element_finder (perception hierarchy)
    verification      → core.goal_verification + agent.desktop_goal effects
    workflow truth    → agent.trace (AgentTrace structured events)
    task identity     → the caller's task_id (Brain/TaskState own the task)

Control loop (bounded at every stage):

    GOAL → PLAN → (OBSERVE → ACT → OBSERVE → VERIFY → [RECOVER|REPLAN])* → COMPLETE

Hard rules enforced here:
  * COMPLETED is only reported when the message is observed INSIDE the
    intended conversation — never from a frame/screen/window change.
  * Ambiguous contacts → ASK_USER, never guess.
  * SEND_MESSAGE is an EXTERNAL_SIDE_EFFECT: the engine PAUSES for explicit
    user confirmation before submitting, then resumes the SAME run.
  * No Telegram API / token; no hard-coded coordinates; semantic perception.
  * Bounded budgets: no arbitrary long sleeps, no infinite recovery loop.

Logging: [DESKTOP-GOAL]
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.desktop_context import DesktopContext, DesktopObserver
from agent.desktop_goal import (
    COMPUTER_ACTION,
    ROLE_MESSAGE_INPUT,
    ROLE_SEARCH,
    DesktopAction,
    DesktopGoal,
    DesktopGoalKind,
    DesktopStatus,
    DesktopStep,
    EffectVerdict,
    classify_state,
    expected_effect_text,
    parse_desktop_goal,
    verify_expected_effect,
)
from agent.desktop_skill_registry import (
    DesktopSkill,
    DesktopSkillRegistry,
    desktop_skill_registry,
)
from agent.browser_goal import (
    Candidate,
    normalize_label,
    rank_candidates,
    resolve_answer_to_candidate,
)

logger = logging.getLogger(__name__)

TERMINAL = (DesktopStatus.COMPLETED, DesktopStatus.FAILED)


@dataclass
class DesktopLimits:
    """Bounded budgets — the engine never enters an infinite desktop loop."""

    max_steps: int = 24
    max_retries_per_step: int = 2
    max_replans: int = 2
    max_waits: int = 4
    wait_timeout_s: float = 8.0
    wait_interval_s: float = 0.35
    max_observations: int = 40
    max_actions: int = 30


@dataclass
class DesktopTaskRun:
    """Desktop-specific execution state (resumable, JSON-able)."""

    task_id: str = ""
    goal: DesktopGoal = field(default_factory=DesktopGoal)
    steps: List[DesktopStep] = field(default_factory=list)
    index: int = 0
    status: DesktopStatus = DesktopStatus.RUNNING
    phase: str = "THINKING"
    question: str = ""
    candidates: List[Candidate] = field(default_factory=list)
    pending_reason: str = ""
    resolved_contact: str = ""          # user-confirmed contact identity
    observations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    last_context: Optional[DesktopContext] = None
    app_open: bool = False
    contact_verified: bool = False
    conversation_verified: bool = False
    draft_verified: bool = False
    send_confirmed: bool = False
    sent_verified: bool = False
    actions: int = 0
    recoveries: int = 0
    replans: int = 0
    waits: int = 0
    observation_count: int = 0
    error: str = ""
    final_message: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    _trace_seq: int = 0
    _trace_by_seq: Dict[int, Any] = field(default_factory=dict)

    def trace_events(self) -> List[Any]:
        return [self._trace_by_seq[seq] for seq in sorted(self._trace_by_seq)
                if seq <= self._trace_seq]

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    @property
    def verified(self) -> bool:
        return self.status == DesktopStatus.COMPLETED

    def plan_lines(self) -> List[str]:
        return [s.description or s.action.value for s in self.steps]

    def active_step(self) -> Optional[DesktopStep]:
        try:
            return self.steps[self.index]
        except IndexError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal.describe(),
            "goal_kind": self.goal.kind.value,
            "status": self.status.value,
            "phase": self.phase,
            "steps": [s.to_dict() for s in self.steps],
            "index": self.index,
            "question": self.question,
            "candidates": [c.to_dict() for c in self.candidates[:6]],
            "resolved_contact": self.resolved_contact,
            "observations": list(self.observations[-20:]),
            "evidence": dict(self.evidence),
            "app_open": self.app_open,
            "contact_verified": self.contact_verified,
            "conversation_verified": self.conversation_verified,
            "draft_verified": self.draft_verified,
            "send_confirmed": self.send_confirmed,
            "sent_verified": self.sent_verified,
            "recoveries": self.recoveries, "replans": self.replans,
            "waits": self.waits, "actions": self.actions,
            "observation_count": self.observation_count,
            "error": self.error, "final_message": self.final_message,
        }

    def user_message(self) -> str:
        if self.status == DesktopStatus.ASKING_USER:
            lines = [self.question or "I need one clarification."]
            for i, c in enumerate(self.candidates[:6], 1):
                lines.append(f"  {i}. {c.label}"
                             + (f" ({c.kind})" if c.kind else ""))
            return "\n".join(lines)
        if self.status == DesktopStatus.WAITING_CONFIRMATION:
            return self.question or "I need your confirmation to send."
        if self.status == DesktopStatus.COMPLETED:
            return self.final_message or self.summary()
        if self.status == DesktopStatus.FAILED:
            base = f"I couldn't complete '{self.goal.describe()}'."
            return (base + (f" {self.error}" if self.error else "")
                    + (f" {self.final_message}" if self.final_message else ""))
        return self.summary()

    def summary(self) -> str:
        bits = [f"{self.goal.describe()}: {self.status.value}"]
        if self.last_context is not None:
            bits.append(self.last_context.summary())
        return " | ".join(bits)


class DesktopGoalEngine:
    """Orchestrates a desktop goal through the existing Diego subsystems."""

    def __init__(self, *, controller: Any = None, observer: Any = None,
                 trace: Any = None, limits: Optional[DesktopLimits] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 skills: Optional[DesktopSkillRegistry] = None) -> None:
        self._controller = controller
        self._observer = observer
        self._trace = trace
        self._limits = limits or DesktopLimits()
        self._sleep = sleep
        self._skills = skills or desktop_skill_registry
        self._pending_run: Optional[DesktopTaskRun] = None
        self._runs: Dict[str, DesktopTaskRun] = {}
        self._unsubs: Dict[str, Any] = {}

    # ── injected collaborators (lazy production defaults) ────────

    def controller(self) -> Any:
        if self._controller is None:
            from computer.computer_controller import computer_controller
            self._controller = computer_controller
        return self._controller

    def observer(self) -> Any:
        if self._observer is None:
            self._observer = DesktopObserver()
        return self._observer

    def trace(self) -> Any:
        if self._trace is None:
            from agent.trace import agent_trace
            self._trace = agent_trace
        return self._trace

    @property
    def limits(self) -> DesktopLimits:
        return self._limits

    def skill(self, app: str) -> DesktopSkill:
        return self._skills.skill_for(app)

    # ── trace / phase plumbing ───────────────────────────────────

    def _watch_run(self, run: DesktopTaskRun) -> None:
        if not run.task_id or run.task_id in self._unsubs:
            return
        try:
            unsubscribe = self.trace().subscribe(self._on_trace_event)
        except Exception as e:
            logger.debug("[DESKTOP-GOAL] trace subscribe failed: %s", e)
            return
        if callable(unsubscribe):
            self._unsubs[run.task_id] = unsubscribe

    def _unwatch_run(self, run: DesktopTaskRun) -> None:
        unsubscribe = self._unsubs.pop(run.task_id, None)
        if callable(unsubscribe):
            try:
                unsubscribe()
            except Exception:
                pass

    def _on_trace_event(self, event: Any) -> None:
        try:
            task_id = str(getattr(event, "task_id", "") or "")
            run = self._runs.get(task_id)
            if run is None:
                return
            seq = int(getattr(event, "seq", 0) or 0)
            if seq <= 0:
                return
            run._trace_by_seq[seq] = event
            if seq > run._trace_seq:
                run._trace_seq = seq
        except Exception:
            pass

    def _phase(self, run: DesktopTaskRun, phase: str, detail: str = "") -> None:
        run.phase = phase
        try:
            self.trace().phase(phase, detail=detail, task_id=run.task_id)
        except Exception as e:
            logger.debug("[DESKTOP-GOAL] trace phase failed: %s", e)

    def _observe_trace(self, run: DesktopTaskRun, text: str,
                       ctx: Optional[DesktopContext] = None) -> None:
        run.observations.append(text[:200])
        try:
            self.trace().observation(
                text[:200], task_id=run.task_id,
                method=(ctx.observation_method if ctx else ""),
                evidence=({
                    "app": ctx.active_application,
                    "title": ctx.window_title[:120],
                    "elements": len(ctx.interactive_elements),
                } if ctx else {}))
        except Exception as e:
            logger.debug("[DESKTOP-GOAL] trace observation failed: %s", e)

    def _trace_action(self, run: DesktopTaskRun, step: DesktopStep,
                      result: Any) -> None:
        try:
            self.trace().action_completed(
                step.action.value,
                success=bool(getattr(result, "success", False)),
                target=step.target[:80],
                method=str(getattr(result, "method", "")),
                observed_effect=str(getattr(result, "observed_effect", ""))[:120],
                error=str(getattr(result, "error", ""))[:200],
                evidence=dict(getattr(result, "evidence", {}) or {}),
                retry_count=step.attempts, task_id=run.task_id)
        except Exception:
            pass

    # ── planning ─────────────────────────────────────────────────

    def plan(self, goal: DesktopGoal) -> List[DesktopStep]:
        """Deterministic, generic plan for an interpreted desktop goal."""
        steps: List[DesktopStep] = []
        app = goal.app
        if goal.kind == DesktopGoalKind.SEND_MESSAGE:
            steps = self._send_plan(goal)
        elif goal.kind == DesktopGoalKind.READ_MESSAGES:
            steps = [
                DesktopStep(
                    action=DesktopAction.OPEN_APPLICATION,
                    description=f"Open {goal.app_display or app}",
                    params={"app": app},
                    expected_effect=expected_effect_text(
                        DesktopAction.OPEN_APPLICATION, {"app": app})),
                DesktopStep(
                    action=DesktopAction.READ_CONVERSATION,
                    description="Read the visible conversation",
                    params={},
                    expected_effect=expected_effect_text(
                        DesktopAction.READ_CONVERSATION)),
            ]
        else:
            steps.append(DesktopStep(
                action=DesktopAction.ASK_USER,
                description="Ask what to do in the application",
                params={"question": "What should I do in the application?"},
                expected_effect=expected_effect_text(DesktopAction.ASK_USER)))

        for i, step in enumerate(steps):
            step.index = i
            if not step.expected_effect:
                step.expected_effect = expected_effect_text(step.action,
                                                            step.params)
        return steps[:self._limits.max_steps]

    def _send_plan(self, goal: DesktopGoal) -> List[DesktopStep]:
        recipient = goal.recipient
        msg = goal.message
        steps: List[DesktopStep] = [
            DesktopStep(
                action=DesktopAction.OPEN_APPLICATION,
                description=f"Open {goal.app_display or goal.app}",
                params={"app": goal.app},
                expected_effect=expected_effect_text(
                    DesktopAction.OPEN_APPLICATION, {"app": goal.app})),
            DesktopStep(
                action=DesktopAction.FIND_CONTACT,
                description=f"Find the contact '{recipient}'",
                params={"target": recipient, "recipient": recipient},
                expected_effect=expected_effect_text(
                    DesktopAction.FIND_CONTACT, {"target": recipient})),
            DesktopStep(
                action=DesktopAction.VERIFY_CONTACT,
                description=f"Verify the contact '{recipient}' is unique",
                params={"target": recipient, "recipient": recipient},
                expected_effect=expected_effect_text(DesktopAction.VERIFY_CONTACT)),
            DesktopStep(
                action=DesktopAction.OPEN_CONVERSATION,
                description=f"Open the conversation with '{recipient}'",
                params={"target": recipient, "recipient": recipient},
                expected_effect=expected_effect_text(
                    DesktopAction.OPEN_CONVERSATION, {"recipient": recipient})),
            DesktopStep(
                action=DesktopAction.VERIFY_CONVERSATION,
                description=f"Verify the conversation with '{recipient}'",
                params={"recipient": recipient},
                expected_effect=expected_effect_text(
                    DesktopAction.VERIFY_CONVERSATION, {"recipient": recipient})),
        ]
        if not msg:
            # Phase 24.6 §4: the goal gave a recipient but no body — ASK the
            # user (a clarification pause, never a failure, never a guess).
            steps.append(DesktopStep(
                action=DesktopAction.ASK_USER,
                description=f"Ask what message to send to '{recipient}'",
                params={"question": (f"What message should I send to "
                                     f"'{recipient}'?"),
                        "kind": "compose_text"},
                expected_effect=expected_effect_text(DesktopAction.ASK_USER)))
        steps.extend([
            DesktopStep(
                action=DesktopAction.FIND_INPUT,
                description="Locate the message input",
                params={"target": ROLE_MESSAGE_INPUT},
                expected_effect=expected_effect_text(DesktopAction.FIND_INPUT)),
            DesktopStep(
                action=DesktopAction.CLEAR_INPUT,
                description="Clear the message input",
                params={"target": ROLE_MESSAGE_INPUT},
                expected_effect=expected_effect_text(DesktopAction.CLEAR_INPUT),
                optional=True),
            DesktopStep(
                action=DesktopAction.COMPOSE_MESSAGE,
                description=f"Type the message '{msg[:40]}'",
                params={"text": msg, "message": msg},
                expected_effect=expected_effect_text(
                    DesktopAction.COMPOSE_MESSAGE, {"text": msg})),
            DesktopStep(
                action=DesktopAction.VERIFY_DRAFT,
                description="Verify the drafted message",
                params={"text": msg, "message": msg},
                expected_effect=expected_effect_text(
                    DesktopAction.VERIFY_DRAFT, {"text": msg})),
            DesktopStep(
                action=DesktopAction.SEND_MESSAGE,
                description=f"Send the message to '{recipient}'",
                params={"recipient": recipient, "text": msg},
                expected_effect=expected_effect_text(
                    DesktopAction.SEND_MESSAGE, {"recipient": recipient})),
            DesktopStep(
                action=DesktopAction.VERIFY_SENT_MESSAGE,
                description="Verify the message was sent",
                params={"recipient": recipient, "text": msg},
                expected_effect=expected_effect_text(
                    DesktopAction.VERIFY_SENT_MESSAGE, {"recipient": recipient})),
        ])
        return steps

    # ── observation ──────────────────────────────────────────────

    def _budget_left(self, run: DesktopTaskRun) -> bool:
        return run.observation_count < self._limits.max_observations

    def _observe(self, run: DesktopTaskRun, note: str = "") -> DesktopContext:
        if not self._budget_left(run):
            ctx = run.last_context or DesktopContext(application_state="unavailable")
            run.last_context = ctx
            return ctx
        ctx = self.observer().observe(note=note)
        run.observation_count += 1
        run.last_context = ctx
        line = f"{classify_state(ctx)} — {ctx.summary()}"
        if note:
            line = f"{line} ({note})"
        self._observe_trace(run, line, ctx)
        return ctx

    def _wait_until(self, run: DesktopTaskRun, predicate: Callable[[Any], bool],
                    description: str, *, ctx: Optional[DesktopContext] = None
                    ) -> Tuple[DesktopContext, bool]:
        ctx = ctx or run.last_context or self._observe(run, note=description)
        if predicate(ctx):
            return ctx, True
        if run.waits >= self._limits.max_waits:
            self._observe_trace(run, "wait budget exhausted", ctx)
            return ctx, False
        run.waits += 1
        self._phase(run, "WAITING", description)
        max_iter = max(2, int(self._limits.wait_timeout_s
                              / max(self._limits.wait_interval_s, 0.01)))
        for _ in range(max_iter):
            if not self._budget_left(run):
                break
            self._sleep(self._limits.wait_interval_s)
            ctx = self._observe(run, note=description)
            if predicate(ctx):
                return ctx, True
        return ctx, False

    # ── ACT (through the existing ComputerController) ────────────

    def _prepare_params(self, step: DesktopStep,
                        ctx: DesktopContext) -> Optional[Dict[str, Any]]:
        """Resolve semantic roles/targets against the OBSERVED desktop."""
        params = dict(step.params)
        if step.action == DesktopAction.FIND_INPUT:
            skill = self.skill(ctx.active_application or params.get("app", ""))
            el = skill.message_input(ctx)
            if el is not None:
                params["target"] = el.label or ROLE_MESSAGE_INPUT
            else:
                params["target"] = ROLE_MESSAGE_INPUT
            return params
        if step.action == DesktopAction.SEND_MESSAGE:
            # Semantic submit: if the skill names a send control, click it;
            # otherwise the focused compose box submits on Enter. The Enter
            # "submit" is the same generic behaviour the browser tier already
            # documents for search boxes — never a guessed coordinate.
            skill = self.skill(ctx.active_application or "")
            send = skill.send_control(ctx)
            if send is not None:
                params["_send_target"] = send.label or "Send"
            else:
                params.setdefault("key", "enter")
            return params
        if step.action == DesktopAction.FIND_CONTACT:
            target = str(params.get("target") or params.get("recipient") or "")
            skill = self.skill(ctx.active_application or "")
            search = skill.contact_search_input(ctx)
            if search is not None and target:
                params["_search_input"] = search.label or ROLE_SEARCH
            return params
        if step.action == DesktopAction.CLEAR_INPUT:
            if params.get("target") == ROLE_MESSAGE_INPUT:
                skill = self.skill(ctx.active_application or "")
                el = skill.message_input(ctx)
                if el is not None:
                    params["target"] = el.label or ROLE_MESSAGE_INPUT
            return params
        return params

    def _act(self, run: DesktopTaskRun, step: DesktopStep,
             ctx: DesktopContext,
             params: Optional[Dict[str, Any]] = None
             ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        self._phase(run, "ACTING", step.description)
        if params is None:
            params = self._prepare_params(step, ctx)
        if params is None:
            return None, None
        comp_action = COMPUTER_ACTION.get(step.action, "")
        # Engine-internal steps (verification/read) perform no controller
        # action: they are satisfied purely by observation.
        if not comp_action:
            return None, params
        run.actions += 1
        try:
            self.trace().action_started(
                step.action.value, target=step.target[:80],
                method=ctx.observation_method,
                expected_effect=step.expected_effect, task_id=run.task_id)
        except Exception:
            pass
        # Copy params without engine-internal keys.
        exec_params = {k: v for k, v in params.items()
                       if not str(k).startswith("_")}
        # SEND_MESSAGE: click the observed send control when one was found,
        # otherwise press Enter to submit the focused compose box.
        if step.action == DesktopAction.SEND_MESSAGE and params.get("_send_target"):
            comp_action = "click"
            exec_params["target"] = params["_send_target"]
        try:
            result = self.controller().execute(comp_action, exec_params)
        except Exception as e:
            logger.warning("[DESKTOP-GOAL] controller failed (%s): %s",
                           comp_action, e)
            from computer.action_result import ActionOutcome, ActionResult
            result = ActionResult(action=comp_action,
                                  outcome=ActionOutcome.FAILED, error=str(e))
        self._trace_action(run, step, result)
        return result, params

    # ── VERIFY (declared expected effect vs observed state) ──────

    def _verify_step(self, run: DesktopTaskRun, step: DesktopStep,
                     before: DesktopContext, after: DesktopContext,
                     params: Optional[Dict[str, Any]]) -> EffectVerdict:
        self._phase(run, "VERIFYING", step.expected_effect or step.description)
        p = dict(params or step.params or {})
        p.pop("_search_input", None)
        verdict = verify_expected_effect(step.action, p, before, after)
        step.verdict = verdict.result.value
        step.observation = verdict.observed_effect
        step.status = "DONE" if verdict.passed else "FAILED"
        _mark_flags(run, step, verdict.passed)
        try:
            self.trace().verification(
                verdict.result.value, action=step.action.value,
                target=step.target[:80], method=after.observation_method,
                expected_effect=step.expected_effect,
                observed=verdict.observed_effect,
                evidence=dict(verdict.evidence), detail=verdict.detail,
                retry_count=step.attempts, task_id=run.task_id)
        except Exception:
            pass
        logger.info("[DESKTOP-GOAL] verify %s %s -> %s (%s)",
                    step.action.value, step.target[:60], verdict.result.value,
                    verdict.detail[:120])
        return verdict

    # ── one step: OBSERVE → ACT → OBSERVE → VERIFY (+recovery) ──

    def _run_step(self, run: DesktopTaskRun, step: DesktopStep) -> str:
        """Returns 'done' | 'retry' | 'paused' | 'failed'."""
        try:
            self.trace().step_started(
                step.description or step.action.value, index=step.index,
                total=len(run.steps), task_id=run.task_id)
        except Exception:
            pass
        if step.action == DesktopAction.ASK_USER:
            return self._ask_user(run, str(step.params.get("question")
                                           or "How should I proceed?"), None)
        if step.action == DesktopAction.SEND_MESSAGE:
            if not run.send_confirmed:
                return self._request_send_confirmation(run, step)
        if not self._budget_left(run):
            return self._fail(run, "observation budget exhausted")

        before = self._observe(run, note=f"before {step.action.value}")

        # Contact resolution has to resolve ambiguity BEFORE acting.
        if step.action == DesktopAction.FIND_CONTACT:
            decided = self._resolve_contact(run, step, before)
            if decided is not None:
                return decided

        params = self._prepare_params(step, before)
        result, params = self._act(run, step, before, params)
        if params is None:
            return self._recover_missing_target(run, step, before)

        if result is not None and not bool(getattr(result, "success", False)):
            handled = self._handle_problem(run, step, result, before)
            if handled in ("paused", "failed"):
                return handled

        after = self._observe(run, note=f"after {step.action.value}")
        verdict = self._verify_step(run, step, before, after, params)
        if verdict.passed:
            return "done"
        return self._recover_effect(run, step, before, after, verdict)

    def _resolve_contact(self, run: DesktopTaskRun, step: DesktopStep,
                         ctx: DesktopContext) -> Optional[str]:
        """Ambiguity handling for FIND_CONTACT: never guess."""
        target = str(step.params.get("target") or step.params.get("recipient")
                     or run.goal.recipient or "")
        if not target:
            return None
        raw = self._contact_candidates(ctx, target)
        ranked, ambiguous, reason = rank_candidates(target, raw)
        if ambiguous:
            return self._ask_user(
                run,
                f"I found {len(ranked)} possible contacts for '{target}'. "
                f"Which one should I use?",
                ctx, candidates=ranked[:6], role=target)
        if not ranked:
            # Nothing matched on the current screen: recover, don't guess.
            return None
        chosen = ranked[0]
        run.resolved_contact = chosen.label
        step.params["target"] = chosen.label
        step.params["resolved"] = chosen.label
        logger.info("[DESKTOP-GOAL] contact resolved '%s' -> '%s' (%s)",
                    target[:60], chosen.label[:60], reason)
        return None

    def _contact_candidates(self, ctx: DesktopContext,
                            target: str) -> List[Dict[str, Any]]:
        raw: List[Dict[str, Any]] = []
        for e in ctx.interactive_elements:
            if e.label:
                raw.append({"label": e.label, "kind": e.kind,
                            "method": e.method})
        # Also treat visible-text lines containing the target as candidates.
        for line in ctx.visible_text.splitlines():
            line = (line or "").strip()
            if line and normalize_label(target) in normalize_label(line):
                raw.append({"label": line, "kind": "text", "method": "ocr"})
        return raw

    # ── confirmation (EXTERNAL_SIDE_EFFECT send) ─────────────────

    def _request_send_confirmation(self, run: DesktopTaskRun,
                                   step: DesktopStep) -> str:
        """PAUSE for explicit user confirmation before sending."""
        run.status = DesktopStatus.WAITING_CONFIRMATION
        recipient = str(step.params.get("recipient") or run.resolved_contact
                        or run.goal.recipient or "")
        message = str(step.params.get("text") or step.params.get("message")
                      or run.goal.message or "")
        run.question = (f"CONFIRMATION REQUIRED\nRecipient: {recipient}\n"
                        f"Message: \"{message}\"")
        run.pending_reason = "send_message"
        run.evidence["pending_send"] = {"recipient": recipient,
                                        "message": message}
        self._phase(run, "ASKING_USER", "send confirmation required")
        try:
            self.trace().confirmation_required(
                f"Send a message to '{recipient}'", action="send_message",
                target=recipient[:80], risk="external_side_effect",
                task_id=run.task_id,
                evidence={"recipient": recipient, "message": message})
        except Exception:
            pass
        logger.info("[DESKTOP-GOAL] paused for send confirmation to '%s'",
                    recipient)
        return "paused"

    # ── recovery ─────────────────────────────────────────────────

    def _handle_problem(self, run: DesktopTaskRun, step: DesktopStep,
                        result: Any, ctx: DesktopContext) -> Optional[str]:
        outcome = str(getattr(getattr(result, "outcome", None), "value",
                              getattr(result, "outcome", "")) or "")
        error = str(getattr(result, "error", "") or "")
        if outcome == "confirmation_required":
            return self._refuse(
                run, step, error or "this action has an external side effect")
        if outcome == "unavailable":
            if step.action == DesktopAction.OPEN_APPLICATION:
                return self._fail(run, error or "application unavailable",
                                  detail=f"could not open '{step.target}'")
            return self._fail(run, error or "capability unavailable",
                              detail=f"{step.action.value} could not run")
        return None

    def _recovery_strategy(self, step: DesktopStep,
                           ctx: DesktopContext) -> str:
        """Phase 24.6 §6: a DIFFERENT strategy per attempt — never the same
        click with identical evidence.

            attempt 1 → semantic accessibility / native re-observe
            attempt 2 → inspect application state (accessibility + OCR)
            attempt 3 → refocus the target window / re-observe / alternate
        """
        if step.attempts == 1:
            if step.action in (DesktopAction.FIND_CONTACT,
                               DesktopAction.OPEN_CONVERSATION,
                               DesktopAction.FIND_INPUT):
                return "re-observe"
            return "wait for content"
        if step.attempts == 2:
            return "inspect application state (accessibility + OCR)"
        return "refocus the target window and re-observe"

    def _failure_signature(self, step: DesktopStep,
                           ctx: DesktopContext) -> str:
        """Identity of one failure: action + target + observed-state hash."""
        screen = str(getattr(ctx, "screen_hash", "") or "")
        return f"{step.action.value}:{step.target[:40]}:{screen[:32]}"

    def _escalate_if_identical(self, run: DesktopTaskRun, step: DesktopStep,
                               ctx: DesktopContext, strategy: str) -> str:
        """Phase 24.6 §10: repeated IDENTICAL failure must trigger a
        DIFFERENT recovery strategy, never the same action again."""
        sig = self._failure_signature(step, ctx)
        hist = run.evidence.setdefault("failure_signatures", [])
        same_as_last = bool(hist) and hist[-1] == sig
        hist.append(sig)
        if same_as_last and strategy != "refocus the target window and re-observe":
            escalated = "refocus the target window and re-observe"
            run.evidence["replan_diff_reason"] = (
                f"identical failure repeated ({sig[:60]}); escalated "
                f"'{strategy}' → '{escalated}'")
            return escalated
        return strategy

    def _recover_missing_target(self, run: DesktopTaskRun, step: DesktopStep,
                                ctx: DesktopContext) -> str:
        step.attempts += 1
        run.recoveries += 1
        if step.attempts > self._limits.max_retries_per_step:
            return self._fail(
                run, f"'{step.target[:60]}' is not present on the current "
                     f"screen",
                detail=f"classified NOT_FOUND ({ctx.summary()})")
        strategy = self._escalate_if_identical(
            run, step, ctx, self._recovery_strategy(step, ctx))
        self._phase(run, "RECOVERING", strategy)
        try:
            self.trace().recovery(strategy, action=step.action.value,
                                  retry_count=step.attempts,
                                  task_id=run.task_id,
                                  evidence={"target": step.target[:80],
                                            "why_different":
                                                run.evidence.get(
                                                    "replan_diff_reason", "")})
        except Exception:
            pass
        if strategy == "scroll to reveal the target":
            self.controller().execute("scroll", {"delta": 600})
        elif strategy == "inspect application state (accessibility + OCR)":
            # Phase 24.6 §6 attempt 2: look at what IS on the screen before
            # choosing the next action — record it as evidence, not as a click.
            fresh = self._observe(run, note="inspect state (a11y + OCR)")
            try:
                self.trace().observation(
                    f"inspect: {fresh.summary()[:160]}", task_id=run.task_id)
            except Exception:
                pass
            return "retry"
        elif strategy == "refocus the target window and re-observe":
            app = run.goal.app or step.params.get("app") or ""
            if app:
                try:
                    self.controller().execute("focus_window",
                                              {"app": app, "target": app})
                except Exception:
                    pass
            self._observe(run, note="re-observe after refocus (recovery)")
            return "retry"
        self._observe(run, note="recovery re-observe")
        return "retry"

    def _recover_effect(self, run: DesktopTaskRun, step: DesktopStep,
                        before: DesktopContext, after: DesktopContext,
                        verdict: EffectVerdict) -> str:
        step.attempts += 1
        run.recoveries += 1
        if step.attempts > self._limits.max_retries_per_step:
            if self._replan(run, f"{step.action.value} did not produce "
                                 f"{step.expected_effect[:60]}"):
                return "retry"
            return self._fail(
                run, f"expected effect not observed for {step.action.value}",
                detail=verdict.detail[:200])
        strategy = self._escalate_if_identical(
            run, step, after, self._recovery_strategy(step, after))
        self._phase(run, "RECOVERING", strategy)
        try:
            self.trace().recovery(strategy, action=step.action.value,
                                  retry_count=step.attempts,
                                  task_id=run.task_id,
                                  evidence={"why_different":
                                                run.evidence.get(
                                                    "replan_diff_reason", ""),
                                            **dict(verdict.evidence or {})})
        except Exception:
            pass
        if strategy == "wait for content":
            self._wait_until(run, lambda c: bool(c.visible_text),
                             "waiting for visible content", ctx=after)
        elif strategy == "scroll to reveal the target":
            self.controller().execute("scroll", {"delta": 600})
            self._observe(run, note="after scroll (recovery)")
        elif strategy == "inspect application state (accessibility + OCR)":
            fresh = self._observe(run, note="inspect state (a11y + OCR)")
            try:
                self.trace().observation(
                    f"inspect: {fresh.summary()[:160]}", task_id=run.task_id)
            except Exception:
                pass
        elif strategy == "refocus the target window and re-observe":
            app = run.goal.app or step.params.get("app") or ""
            if app:
                try:
                    self.controller().execute("focus_window",
                                              {"app": app, "target": app})
                except Exception:
                    pass
            self._observe(run, note="re-observe after refocus (recovery)")
        else:
            self._observe(run, note="re-observe (recovery)")
        return "retry"

    def _replan(self, run: DesktopTaskRun, reason: str) -> bool:
        if run.replans >= self._limits.max_replans:
            return False
        run.replans += 1
        try:
            self.trace().replan(reason, task_id=run.task_id)
        except Exception:
            pass
        logger.info("[DESKTOP-GOAL] replan #%d: %s", run.replans, reason[:80])
        return True

    # ── pause states: clarification ──────────────────────────────

    def _ask_user(self, run: DesktopTaskRun, question: str,
                  ctx: Optional[DesktopContext],
                  *, candidates: Optional[List[Candidate]] = None,
                  role: str = "", pending_reason: str = "") -> str:
        run.status = DesktopStatus.ASKING_USER
        run.question = question
        run.candidates = list(candidates or [])
        run.pending_reason = (pending_reason
                              or ("ambiguous_target" if run.candidates
                                  else "clarify"))
        self._phase(run, "ASKING_USER", question[:120])
        try:
            self.trace().confirmation_required(
                question[:200], action="clarification",
                target=role[:80], task_id=run.task_id)
        except Exception:
            pass
        return "paused"

    def _refuse(self, run: DesktopTaskRun, step: DesktopStep,
                reason: str) -> str:
        run.status = DesktopStatus.FAILED
        run.error = reason
        run.final_message = reason
        run.ended_at = time.time()
        self._phase(run, "FAILED", reason[:140])
        try:
            self.trace().failed(reason[:240], action=step.action.value,
                                target=step.target[:80], task_id=run.task_id)
        except Exception:
            pass
        self._unwatch_run(run)
        return "failed"

    def _fail(self, run: DesktopTaskRun, error: str, *,
              detail: str = "") -> str:
        run.status = DesktopStatus.FAILED
        run.error = error[:300]
        run.final_message = (detail or error)[:300]
        run.ended_at = time.time()
        self._phase(run, "FAILED", error[:140])
        try:
            self.trace().failed(f"{error[:200]} {detail[:120]}".strip(),
                                action="desktop_goal",
                                target=run.goal.describe()[:80],
                                task_id=run.task_id,
                                evidence={"actions": run.actions,
                                          "recoveries": run.recoveries,
                                          "replans": run.replans})
        except Exception:
            pass
        logger.info("[DESKTOP-GOAL] task %s FAILED: %s (%s)",
                    run.task_id, error[:120], detail[:120])
        self._unwatch_run(run)
        return "failed"

    def _complete(self, run: DesktopTaskRun) -> None:
        run.status = DesktopStatus.COMPLETED
        run.ended_at = time.time()
        run.final_message = (f"Message sent to '{run.goal.recipient}' — "
                             f"verified inside the conversation.")
        self._phase(run, "COMPLETED", run.goal.describe()[:140])
        try:
            self.trace().completed(
                run.final_message, task_id=run.task_id,
                evidence={"recipient": run.goal.recipient,
                          "sent_verified": run.sent_verified})
        except Exception:
            pass
        logger.info("[DESKTOP-GOAL] task %s COMPLETED: %s",
                    run.task_id, run.final_message[:160])
        self._unwatch_run(run)

    def _goal_outcome_verified(self, run: DesktopTaskRun) -> bool:
        if run.goal.kind == DesktopGoalKind.SEND_MESSAGE:
            return run.sent_verified
        if run.goal.kind == DesktopGoalKind.READ_MESSAGES:
            return bool(run.last_context and run.last_context.visible_text)
        return False

    def _finalize(self, run: DesktopTaskRun) -> None:
        if run.status == DesktopStatus.WAITING_CONFIRMATION:
            return  # paused for send confirmation — keep the task
        if run.status == DesktopStatus.ASKING_USER:
            return  # paused for clarification — keep the task
        if self._goal_outcome_verified(run):
            self._complete(run)
            return
        self._fail(
            run, "the goal's outcome was never verified",
            detail=("the steps ran but the sent message was not observed "
                    "inside the intended conversation — sending alone is "
                    "never completion"))

    def _drive(self, run: DesktopTaskRun) -> None:
        retries_without_progress = 0
        last_index = run.index
        while not run.finished:
            if run.actions > self._limits.max_actions:
                self._fail(run, "action budget exhausted")
                return
            if run.observation_count > self._limits.max_observations:
                self._fail(run, "observation budget exhausted")
                return
            step = run.active_step()
            if step is None:
                self._finalize(run)
                return
            if run.index != last_index:
                retries_without_progress = 0
                last_index = run.index
            else:
                retries_without_progress += 1
                if retries_without_progress > self._limits.max_steps * 2:
                    self._fail(run, "no progress — desktop loop stopped")
                    return
            outcome = self._run_step(run, step)
            if outcome == "done":
                run.index += 1
                retries_without_progress = 0
            elif outcome in ("paused", "failed"):
                return
            # "retry" → loop continues (bounded by step.attempts + budgets).
        if run.status == DesktopStatus.RUNNING:
            self._finalize(run)

    # ── start / resume / command entry ───────────────────────────

    def start(self, text: str, *, task_id: str = "") -> Optional[DesktopTaskRun]:
        """Interpret `text` as a desktop goal, plan it, and register the run."""
        goal = parse_desktop_goal(text)
        if goal is None:
            return None
        task_id = task_id or f"desktop-{uuid.uuid4().hex[:8]}"
        run = DesktopTaskRun(task_id=task_id, goal=goal)
        run.steps = self.plan(goal)
        self._pending_run = run
        self._runs[task_id] = run
        self._watch_run(run)
        self._phase(run, "THINKING", goal.describe()[:140])
        try:
            self.trace().goal(goal.describe(), task_id=task_id,
                              evidence={"kind": goal.kind.value,
                                        "requirement":
                                        goal.completion_requirement()})
            self.trace().plan(run.plan_lines(), task_id=task_id)
        except Exception:
            pass
        self._phase(run, "PLANNING", f"{len(run.steps)} step(s) planned")
        logger.info("[DESKTOP-GOAL] start %s: %s (%d steps)", task_id,
                    goal.describe()[:100], len(run.steps))
        return run

    def resume(self, answer: str, *, task_id: str = "") -> Optional[DesktopTaskRun]:
        """Continue a PAUSED run from its observed state (never a restart)."""
        run = (self._runs.get(task_id) if task_id else None) or self._pending_run
        if run is None or run.finished:
            return run
        text = str(answer or "").strip()
        low = text.lower()
        if any(w in low for w in ("cancel", "stop", "never mind", "nevermind")):
            try:
                self.trace().confirmation_resolved(
                    False, action="desktop_goal",
                    detail="user cancelled the paused desktop task",
                    task_id=run.task_id)
            except Exception:
                pass
            self._fail(run, "cancelled by the user")
            return run

        if run.status == DesktopStatus.ASKING_USER:
            chosen = resolve_answer_to_candidate(text, list(run.candidates))
            step = run.active_step()
            if step is not None:
                if chosen is not None:
                    run.resolved_contact = chosen.label
                    step.params["target"] = chosen.label
                    step.params["recipient"] = chosen.label
                    for later in run.steps[run.index + 1:]:
                        if str(later.params.get("recipient") or "") == \
                                run.goal.recipient:
                            later.params["recipient"] = chosen.label
                            later.params["target"] = chosen.label
                elif text:
                    run.resolved_contact = text
                    step.params["target"] = text
            run.candidates = []
            run.pending_reason = ""
            run.status = DesktopStatus.RUNNING
            self._watch_run(run)
            self._drive(run)
            return run

        if run.status == DesktopStatus.WAITING_CONFIRMATION:
            from agent.task_continuation import classify_confirmation
            verdict = classify_confirmation(text) if text else None
            if verdict == "confirm":
                run.send_confirmed = True
                run.status = DesktopStatus.RUNNING
                run.question = ""
                run.evidence.pop("pending_send", None)
                try:
                    self.trace().confirmation_resolved(
                        True, action="send_message",
                        target=run.goal.recipient[:80],
                        detail="user confirmed sending the message",
                        task_id=run.task_id)
                except Exception:
                    pass
                self._watch_run(run)
                self._drive(run)
            elif verdict == "cancel":
                try:
                    self.trace().confirmation_resolved(
                        False, action="send_message",
                        target=run.goal.recipient[:80],
                        detail="user cancelled the send",
                        task_id=run.task_id)
                except Exception:
                    pass
                self._fail(run, "cancelled by the user")
            return run
        return run

    def run_to_terminal(self, run: Optional[DesktopTaskRun]
                        ) -> Optional[DesktopTaskRun]:
        if run is None:
            return None
        if not run.finished and run.status == DesktopStatus.RUNNING:
            self._drive(run)
        if not run.finished and run.status == DesktopStatus.RUNNING:
            self._finalize(run)
        return run

    def handle_command(self, text: str) -> Optional[str]:
        """Entry point used by the Brain. Returns a user-facing message when
        this text was (or resumed) a desktop goal, else None."""
        run = self._pending_run
        if run is not None and run.status in (
                DesktopStatus.ASKING_USER, DesktopStatus.WAITING_CONFIRMATION):
            self.resume(text)
            return self._pending_run.user_message()
        goal = parse_desktop_goal(text)
        if goal is None:
            return None
        new_run = self.start(text)
        if new_run is None:
            return None
        self.run_to_terminal(new_run)
        return new_run.user_message()

    # ── accessors ────────────────────────────────────────────────

    def has_pending(self) -> bool:
        run = self._pending_run
        return (run is not None and not run.finished
                and run.status in (DesktopStatus.ASKING_USER,
                                   DesktopStatus.WAITING_CONFIRMATION))

    def pending_run(self) -> Optional[DesktopTaskRun]:
        return self._pending_run

    def run_for(self, task_id: str) -> Optional[DesktopTaskRun]:
        return self._runs.get(task_id)

    def runs(self) -> Dict[str, DesktopTaskRun]:
        return dict(self._runs)


def _mark_flags(run: DesktopTaskRun, step: DesktopStep, passed: bool) -> None:
    """Set goal-level flags ONLY from a PASSED verification (never guessed)."""
    if not passed:
        return
    if step.action == DesktopAction.OPEN_APPLICATION:
        run.app_open = True
    elif step.action == DesktopAction.VERIFY_CONTACT:
        run.contact_verified = True
    elif step.action == DesktopAction.VERIFY_CONVERSATION:
        run.conversation_verified = True
    elif step.action == DesktopAction.VERIFY_DRAFT:
        run.draft_verified = True
    elif step.action == DesktopAction.VERIFY_SENT_MESSAGE:
        run.sent_verified = True
        run.evidence["sent_message"] = dict(step.params)


# Module-level singleton (the Brain imports this).
desktop_goal_engine = DesktopGoalEngine()


__all__ = [
    "DesktopGoalEngine", "DesktopLimits", "DesktopTaskRun",
    "desktop_goal_engine",
]










