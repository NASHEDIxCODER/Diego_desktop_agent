"""
GoalRuntime — the PLAN → EXECUTE → OBSERVE → VERIFY → REPLAN loop.

This is the unified autonomous goal runtime. It composes (never duplicates):
    planning        → goalruntime.planner.DeterministicPlanner
    skills          → goalruntime.skills.SkillRegistry / SkillExecutor
    permissions     → goalruntime.permissions.ScopedPermissionManager
    perception      → goalruntime.perception (ladder + anti-repeat)
    models          → goalruntime.llm.ModelRouter (provider-agnostic)
    the world       → goalruntime.backends.RuntimeBackend

Loop (bounded at every stage):
    PLAN → (EXECUTE → OBSERVE → VERIFY → [CONFIRM | REPLAN])* → COMPLETED

Hard rules enforced here:
  * Read-only + reversible-local actions run automatically.
  * External/destructive/security actions pause for confirmation unless a
    scoped grant exists; the user's yes/no flows through the SAME
    confirmation path as the existing task-continuation flow.
  * A failed action's retry MUST use different evidence/strategy
    (goalruntime.perception.AttemptMemory is the enforcer).
  * Focus is validated before acting on an application.
  * REPLAN is bounded (RuntimeLimits.max_replans).
  * Artifacts flow between subgoals; evidence is recorded on every step.

Logging: [GOAL-RUNTIME]
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from goalruntime.models import (
    Artifact, GoalStatus, Observation, PermissionClass, SkillResult, Subgoal,
)
from goalruntime.permissions import (
    PermissionDecision, ScopedPermissionManager, classify_action,
)
from goalruntime.skills import SkillExecutor, SkillRegistry, terminal_permission
from goalruntime.planner import DeterministicPlanner, PlannerLimits
from goalruntime.perception import AttemptMemory, next_strategy
from goalruntime.llm import ModelRouter, ModelRole

logger = logging.getLogger(__name__)

TERMINAL = (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED)


@dataclass
class RuntimeLimits:
    max_subgoals: int = 8
    max_replans: int = 3
    max_total_actions: int = 40
    confirm_timeout_s: float = 300.0


@dataclass
class GoalRun:
    """The record of one goal being pursued."""

    goal_id: str
    text: str
    status: GoalStatus = GoalStatus.PENDING
    subgoals: List[Subgoal] = field(default_factory=list)
    results: List[SkillResult] = field(default_factory=list)
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    confirmations: List[Dict[str, Any]] = field(default_factory=list)
    replans: int = 0
    actions_taken: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def user_message(self) -> str:
        """User-facing summary (mirrors DesktopTaskRun.user_message)."""
        if self.status is GoalStatus.COMPLETED:
            done = sum(1 for s in self.subgoals
                       if s.status is GoalStatus.COMPLETED)
            return (f"Goal completed: {self.text} "
                    f"({done}/{len(self.subgoals)} steps)")
        if self.status is GoalStatus.FAILED:
            return f"Goal failed: {self.text} — {self.error}"
        if self.status is GoalStatus.CANCELLED:
            return f"Goal cancelled: {self.text}"
        if self.status is GoalStatus.AWAITING_CONFIRMATION:
            c = self.confirmations[-1] if self.confirmations else {}
            return str(c.get("prompt") or "Confirmation required.")
        return f"Goal in progress: {self.text}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id, "text": self.text,
            "status": self.status.value,
            "subgoals": [s.describe() for s in self.subgoals],
            "subgoal_status": [s.status.value for s in self.subgoals],
            "replans": self.replans, "actions": self.actions_taken,
            "artifacts": {k: v.describe() for k, v in self.artifacts.items()},
            "error": self.error,
        }


class GoalRuntime:
    """Unified autonomous desktop goal runtime."""

    def __init__(self, *,
                 backend: Any = None,
                 router: Optional[ModelRouter] = None,
                 permissions: Optional[ScopedPermissionManager] = None,
                 registry: Optional[SkillRegistry] = None,
                 planner: Optional[DeterministicPlanner] = None,
                 limits: Optional[RuntimeLimits] = None,
                 on_user_prompt: Optional[Callable[[str], str]] = None,
                 session: Any = None) -> None:
        self.backend = backend
        self.router = router or ModelRouter()
        self.permissions = permissions or ScopedPermissionManager()
        self.registry = registry or SkillRegistry()
        self.planner = planner or DeterministicPlanner(
            router=self.router,
            limits=PlannerLimits(max_subgoals=limits.max_subgoals
                                 if limits else 8))
        self.limits = limits or RuntimeLimits()
        self.on_user_prompt = on_user_prompt
        self.session = session
        self.attempts = AttemptMemory()
        self._runs: Dict[str, GoalRun] = {}
        self._pending: Optional[GoalRun] = None

    # ═══════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════

    def run_goal(self, goal_text: str) -> GoalRun:
        """Execute a goal through the full loop to a terminal state."""
        run = self.start(goal_text)
        if run is None:
            raise RuntimeError("goal could not be started")
        return self.run_to_terminal(run)

    def start(self, goal_text: str) -> Optional[GoalRun]:
        """PLAN phase: decompose the goal into subgoals."""
        goal_text = (goal_text or "").strip()
        if not goal_text:
            return None
        run = GoalRun(goal_id=uuid.uuid4().hex[:10], text=goal_text,
                      status=GoalStatus.PLANNING)
        run.subgoals = self.planner.plan(goal_text)
        if not run.subgoals:
            run.status = GoalStatus.FAILED
            run.error = "planner produced no subgoals"
            run.finished_at = time.time()
            self._runs[run.goal_id] = run
            return run
        run.status = GoalStatus.PENDING
        self._runs[run.goal_id] = run
        logger.info("[GOAL-RUNTIME] planned %d subgoal(s) for: %s",
                    len(run.subgoals), goal_text[:80])
        return run

    def run_to_terminal(self, run: GoalRun) -> GoalRun:
        """Drive the loop until a terminal state (bounded)."""
        guard = 0
        while run.status not in TERMINAL and guard < self.limits.max_total_actions * 2:
            guard += 1
            run.status = GoalStatus.EXECUTING
            pending_sub = self._next_pending(run)
            if pending_sub is None:
                self._finalize(run)
                break
            sub = pending_sub
            result = self._execute_subgoal(run, sub)
            # A confirmation pause is NOT a failure: the run stays parked in
            # AWAITING_CONFIRMATION until the user answers (resume()).
            if run.status is GoalStatus.AWAITING_CONFIRMATION \
                    or sub.status is GoalStatus.AWAITING_CONFIRMATION:
                return run
            run.results.append(result)
            run.actions_taken += 1
            if run.actions_taken > self.limits.max_total_actions:
                run.status = GoalStatus.FAILED
                run.error = "action budget exhausted"
                break

            # ── VERIFY ───────────────────────────────────────────
            if result.success and result.verified:
                sub.status = GoalStatus.COMPLETED
                for a in result.artifacts:
                    run.artifacts[a.name] = a
                if sub.produces and not result.artifacts:
                    # record explicit produces as artifacts of the evidence
                    for name in sub.produces:
                        run.artifacts[name] = Artifact(
                            name=name, value=result.evidence.get(name,
                                                                 result.evidence),
                            producer=sub.id)
                continue

            if not result.success:
                # ── REPLAN ───────────────────────────────────────
                sub.status = GoalStatus.FAILED
                sub.error = result.error
                if not sub.required:
                    # Advisory subgoal (e.g. perception escalation) — its
                    # failure never fails the goal; continue to the next.
                    continue
                if run.replans >= self.limits.max_replans:
                    run.status = GoalStatus.FAILED
                    run.error = (f"subgoal '{sub.description}' failed after "
                                 f"{sub.attempts} attempt(s): {result.error}")
                    break
                run.replans += 1
                run.status = GoalStatus.REPLANNING
                new_subs = self.planner.replan(run.text, sub,
                                               result.observation.text)
                if not new_subs:
                    run.status = GoalStatus.FAILED
                    run.error = result.error or "replan produced nothing"
                    break
                idx = run.subgoals.index(sub)
                run.subgoals[idx + 1:idx + 1] = new_subs
                logger.info("[GOAL-RUNTIME] replan #%d after %s failure: "
                            "+%d subgoal(s)", run.replans, sub.skill_id,
                            len(new_subs))
                continue

            # success but NOT verified → treat as failure (never fake success)
            if run.replans >= self.limits.max_replans:
                sub.status = GoalStatus.FAILED
                run.status = GoalStatus.FAILED
                run.error = f"verification failed: {result.verification}"
                break
            run.replans += 1
            run.status = GoalStatus.REPLANNING
            new_subs = self.planner.replan(run.text, sub,
                                           result.observation.text)
            idx = run.subgoals.index(sub)
            run.subgoals[idx + 1:idx + 1] = new_subs
            sub.status = GoalStatus.FAILED

        if run.status not in TERMINAL:
            self._finalize(run)
        if run.finished_at is None:
            run.finished_at = time.time()
        if self.session is not None:
            if run.status is GoalStatus.COMPLETED:
                self.session.goal_completed(run.text)
            elif run.status is GoalStatus.FAILED:
                self.session.goal_failed(run.text)
        logger.info("[GOAL-RUNTIME] goal %s → %s (replans=%d actions=%d)",
                    run.goal_id, run.status.value, run.replans,
                    run.actions_taken)
        return run

    # ── confirmation flow (mirrors task continuation) ────────────

    def has_pending_confirmation(self) -> bool:
        return self._pending is not None and \
            self._pending.status is GoalStatus.AWAITING_CONFIRMATION

    def pending_run(self) -> Optional[GoalRun]:
        return self._pending

    def resume(self, user_text: str) -> Optional[GoalRun]:
        """Resolve a pending confirmation with the user's utterance."""
        run = self._pending
        if run is None:
            return None
        decision = self.permissions.resolve_confirmation(user_text)
        run.confirmations.append({
            "user": user_text, "allowed": decision.allowed,
            "mode": decision.mode, "reason": decision.reason,
            "at": time.time(),
        })
        if not decision.allowed:
            if decision.mode == "deny":
                run.status = GoalStatus.CANCELLED
                run.error = "user denied the action"
                self._pending = None
                return run
            # still awaiting a real yes/no — keep pending
            return run
        # approved → execute the waiting subgoal now
        self._pending = None
        sub = self._next_pending(run)
        if sub is None:
            self._finalize(run)
            return run
        run.status = GoalStatus.EXECUTING
        result = self._execute_subgoal(run, sub)
        run.results.append(result)
        run.actions_taken += 1
        if result.success and result.verified:
            sub.status = GoalStatus.COMPLETED
            for a in result.artifacts:
                run.artifacts[a.name] = a
            self._finalize(run)
        else:
            sub.status = GoalStatus.FAILED
            if run.replans >= self.limits.max_replans:
                run.status = GoalStatus.FAILED
                run.error = result.error or result.verification
            else:
                run.replans += 1
                run.status = GoalStatus.REPLANNING
                new_subs = self.planner.replan(run.text, sub,
                                               result.observation.text)
                idx = run.subgoals.index(sub)
                run.subgoals[idx + 1:idx + 1] = new_subs
        if run.status not in TERMINAL:
            return self.run_to_terminal(run)
        return run

    # ── accessors ────────────────────────────────────────────────

    def run_for(self, goal_id: str) -> Optional[GoalRun]:
        return self._runs.get(goal_id)

    def runs(self) -> Dict[str, GoalRun]:
        return dict(self._runs)

    # ═══════════════════════════════════════════════════════════
    # Internals
    # ═══════════════════════════════════════════════════════════

    def _next_pending(self, run: GoalRun) -> Optional[Subgoal]:
        for s in run.subgoals:
            if s.status is GoalStatus.PENDING:
                return s
        return None

    def _execute_subgoal(self, run: GoalRun, sub: Subgoal, *,
                         approved: bool = False) -> SkillResult:
        """Execute one subgoal. ``approved=True`` skips the permission gate
        (used right after the user explicitly approved this exact action)."""
        sub.attempts += 1
        sub.status = GoalStatus.EXECUTING
        self._inject_artifacts(run, sub)

        # ── permission gate (deterministic BEFORE any model call) ────
        pclass = self._classify(run, sub)
        sub.permission_class = pclass
        if approved:
            decision = PermissionDecision(
                True, "confirmed_by_grant", "user approved this action",
                pclass, "")
        elif pclass is PermissionClass.SECURITY_TESTING:
            decision = self._security_decision(sub)
        else:
            decision = self.permissions.decide(
                pclass, sub.action, sub.params,       # bare action verb
                target=str(sub.params.get("target") or ""),
                operation=sub.action)
        if decision.mode == "confirmation":
            prompt = (f"{decision.reason} — allow "
                      f"{sub.skill_id}.{sub.action} "
                      f"({_params_summary(sub.params)})?")
            pending_c = self.permissions.request_confirmation(
                pclass, sub.action, sub.params)
            run.confirmations.append({
                "prompt": pending_c.prompt, "scope": decision.scope_key,
                "at": time.time(), "allowed": False, "mode": "confirmation",
            })
            sub.status = GoalStatus.AWAITING_CONFIRMATION
            run.status = GoalStatus.AWAITING_CONFIRMATION
            self._pending = run
            logger.info("[GOAL-RUNTIME] confirmation required: %s",
                        decision.scope_key)
            if self.on_user_prompt is not None:
                answer = self.on_user_prompt(pending_c.prompt)
                return self._resume_after_prompt(run, sub, answer)
            return SkillResult(
                skill_id=sub.skill_id, action=sub.action, success=False,
                permission_class=pclass,
                error=f"confirmation required: {decision.reason}",
                expected_effect="", verification="")
        if not decision.allowed:
            sub.status = GoalStatus.FAILED
            return SkillResult(
                skill_id=sub.skill_id, action=sub.action, success=False,
                permission_class=pclass,
                error=f"permission denied: {decision.reason}",
                expected_effect="", verification="")

        # ── focus validation BEFORE acting ───────────────────────────
        # Skipped for actions whose JOB is to open/focus the app.
        app = str(sub.params.get("app") or "")
        if app and self.backend is not None and sub.action not in (
                "open", "open_app", "focus_app", "focus_window"):
            from goalruntime.perception import validate_focus
            fc = validate_focus(self.backend, app)
            if not fc.ok:
                from goalruntime.perception import recover_focus
                fc = recover_focus(self.backend, app)
                if not fc.ok:
                    self.attempts.record("focus", sub.action, app, False,
                                         fc.reason)
                    return SkillResult(
                        skill_id=sub.skill_id, action=sub.action,
                        success=False, permission_class=pclass,
                        error=f"focus validation failed: {fc.reason}",
                        expected_effect="", verification="focus not validated")

        # ── anti-repeat: same tier/strategy/target must not repeat ───
        key = (sub.skill_id, sub.action,
               str(sub.params.get("target") or sub.params.get("contact")
                   or sub.params.get("path") or ""))
        if self.attempts.failed(key[0], key[1], key[2]):
            alt = next_strategy(sub.action)
            logger.info("[GOAL-RUNTIME] anti-repeat: rotating strategy "
                        "%s → %s", sub.action, alt)
            sub.params["_strategy"] = alt

        # ── EXECUTE via the skill layer ──────────────────────────────
        executor = SkillExecutor(self.registry)
        result = executor.execute(sub, self.backend,
                                  {"artifacts": run.artifacts})
        self.attempts.record(key[0], key[1], key[2],
                             result.success and result.verified,
                             result.error)
        sub.evidence = dict(result.evidence)
        if result.success and not result.verified:
            sub.error = result.verification or "effect not observed"
        elif not result.success:
            sub.error = result.error
        return result

    def _resume_after_prompt(self, run: GoalRun, sub: Subgoal,
                             answer: str) -> SkillResult:
        decision = self.permissions.resolve_confirmation(answer)
        run.confirmations.append({
            "user": answer, "allowed": decision.allowed,
            "mode": decision.mode, "at": time.time(),
        })
        if decision.allowed:
            run.status = GoalStatus.EXECUTING
            self._pending = None
            # The user approved THIS action: execute without re-gating.
            return self._execute_subgoal(run, sub, approved=True)
        if decision.mode == "deny":
            run.status = GoalStatus.CANCELLED
            run.error = "user denied the action"
            self._pending = None
        return SkillResult(
            skill_id=sub.skill_id, action=sub.action, success=False,
            permission_class=sub.permission_class,
            error="user denied" if decision.mode == "deny"
            else "awaiting explicit confirmation",
            expected_effect="", verification="")

    def _classify(self, run: GoalRun, sub: Subgoal) -> PermissionClass:
        """Deterministic permission classification (never an LLM call)."""
        if sub.permission_class and sub.permission_class not in (
                PermissionClass.READ_ONLY,):
            # planner already declared a stricter class — keep it
            return sub.permission_class
        if sub.skill_id == "terminal":
            return terminal_permission(str(sub.params.get("cmd") or ""))
        cls, _ = classify_action(f"{sub.skill_id}.{sub.action}", sub.params)
        return cls

    def _security_decision(self, sub: Subgoal) -> PermissionDecision:
        """Security actions: EVERY operation must be inside an authorized
        scope for the EXACT target. Scope is never expanded autonomously.
        """
        target = str(sub.params.get("target") or "")
        ops = list(sub.params.get("operations") or [sub.action])
        allowed, reason = True, ""
        for op in ops:
            d = self.permissions.decide(
                PermissionClass.SECURITY_TESTING,
                f"{sub.skill_id}.{sub.action}", sub.params,
                target=target, operation=op)
            if not d.allowed:
                allowed, reason = False, d.reason
                break
        return PermissionDecision(
            allowed,
            "confirmed_by_grant" if allowed else "deny",
            reason or (f"all {len(ops)} operation(s) within authorized "
                       f"scope on {target}"),
            PermissionClass.SECURITY_TESTING,
            self.permissions._scope_key(f"{sub.skill_id}.{sub.action}",
                                        sub.params))

    def _inject_artifacts(self, run: GoalRun, sub: Subgoal) -> None:
        """Artifact passing: named artifacts become params for consumers."""
        for name in sub.consumes:
            art = run.artifacts.get(name)
            if art is None:
                continue
            if name not in sub.params:
                sub.params[name] = art.value
            # cross-app artifact fields
            if name == "contact" and "contact" not in sub.params:
                sub.params["contact"] = str(art.value)
            if name == "search_results" and "query" not in sub.params:
                sub.params["query"] = str(art.value)

    def _finalize(self, run: GoalRun) -> None:
        required_subs = [s for s in run.subgoals if s.required]
        done = all(s.status in (GoalStatus.COMPLETED, GoalStatus.FAILED)
                   for s in run.subgoals)

        def superseded(s: Subgoal) -> bool:
            """A FAILED subgoal whose identical LATER retry completed is a
            superseded failure — the replan fixed it; it no longer fails
            the goal."""
            if s.status is not GoalStatus.FAILED:
                return False
            idx = run.subgoals.index(s)
            for later in run.subgoals[idx + 1:]:
                if (later.skill_id == s.skill_id
                        and later.action == s.action
                        and later.status is GoalStatus.COMPLETED):
                    return True
            return False

        any_required_failed = any(
            s.status is GoalStatus.FAILED and not superseded(s)
            for s in required_subs)
        if done and not any_required_failed:
            run.status = GoalStatus.COMPLETED
        elif done:
            run.status = GoalStatus.FAILED
            failed = [s for s in required_subs
                      if s.status is GoalStatus.FAILED and not superseded(s)]
            run.error = "; ".join(s.error or s.description
                                  for s in failed[:2])
        else:
            run.status = GoalStatus.FAILED
            pending = [s for s in run.subgoals
                       if s.status in (GoalStatus.PENDING,
                                       GoalStatus.EXECUTING)]
            run.error = ("action budget exhausted with pending subgoals: "
                         + "; ".join(s.description for s in pending[:2])
                         if pending else run.error or "incomplete run")
        run.finished_at = time.time()


def _params_summary(params: Dict[str, Any]) -> str:
    parts = []
    for k in ("contact", "text", "path", "url", "target", "cmd", "app",
              "query"):
        if k in params and params[k]:
            v = str(params[k])
            parts.append(f"{k}={v[:40]}")
    return ", ".join(parts) if parts else "no parameters"


__all__ = ["GoalRuntime", "GoalRun", "RuntimeLimits", "TERMINAL"]
