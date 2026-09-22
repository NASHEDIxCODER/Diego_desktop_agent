"""
Phase 23: BrowserGoalEngine — goal-directed browser orchestration for Diego.

This layer makes Diego achieve browser GOALS instead of firing isolated
browser actions. It orchestrates existing capabilities and duplicates none:

    goal parsing      → agent.browser_goal (pure vocabulary + contracts)
    page state        → agent.browser_context (structured BrowserContext)
    execution         → computer.computer_controller (Phase 22 facade)
    element location  → computer.element_finder (perception hierarchy)
    verification      → core.goal_verification + agent.browser_goal effects
    workflow truth    → agent.trace (AgentTrace structured events)
    task identity     → the caller's task_id (Brain/TaskState own the task)

Control loop (bounded at every stage):

    PLAN → (OBSERVE → ACT → OBSERVE → VERIFY → [RECOVER|REPLAN])* → COMPLETE

Hard rules enforced here:
  * COMPLETED is only reported when the goal's outcome was VERIFIED — mere
    navigation is never "task complete".
  * A generic screen change is never accepted as success (expected effects
    are declared per step and checked against observed state).
  * Ambiguous targets (two equally plausible matches) → ASK_USER, never guess.
  * Authentication is never bypassed: the task PAUSES and asks the user to
    sign in, then resumes from the observed state (no restart).
  * No arbitrary long sleeps: bounded observe → wait → observe.
  * No website-specific workflow: only generic roles/labels/URLs.
  * No model call per click: the engine is deterministic; the model decides at
    the goal level and the controller performs low-level actions.

Logging: [BROWSER-GOAL]
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.browser_context import (
    LOADED,
    BrowserContext,
    BrowserContextObserver,
    InteractiveElement,
)
from agent.browser_goal import (
    COMPUTER_ACTION,
    BrowserAction,
    BrowserExtraction,
    BrowserGoal,
    BrowserGoalKind,
    BrowserResult,
    BrowserStatus,
    BrowserStep,
    Candidate,
    EffectResult,
    EffectVerdict,
    PageState,
    classify_page,
    detect_authenticated,
    detect_login,
    expected_effect_text,
    guardrail_reason,
    guardrail_step_reason,
    normalize_label,
    parse_browser_goal,
    rank_candidates,
    resolve_answer_to_candidate,
    resolve_site_url,
    select_target,
    verify_expected_effect,
)

logger = logging.getLogger(__name__)

# Semantic role tokens resolved against the OBSERVED page at execution time
# (never a site-specific selector).
ROLE_SEARCH_INPUT = "@search_input"
ROLE_SEARCH_SUBMIT = "@search_submit"

# Search-interface / submit wording used for semantic identification. Generic
# words only ("search", "query", "submit") — no site knowledge.
_SEARCH_WORDS = ("search", "query", "keyword", "filter")
_SUBMIT_WORDS = ("search", "submit", "find")

# Actions whose "target" names an ELEMENT on the page (semantic resolution
# applies). Actions carrying a url/query/text instead are left untouched.
_ELEMENT_ACTIONS = frozenset({
    BrowserAction.FIND_ELEMENT, BrowserAction.CLICK_ELEMENT,
    BrowserAction.TYPE_INPUT, BrowserAction.CLEAR_INPUT,
    BrowserAction.SELECT_OPTION, BrowserAction.PAGINATE,
})

# Actions whose expected effect is RENDERED CONTENT: when the first
# observation is empty the engine waits a bounded interval and observes again
# (dynamic pages render after `readyState === complete`).
_CONTENT_ACTIONS = frozenset({
    BrowserAction.EXTRACT_RESULTS, BrowserAction.EXTRACT_TEXT,
    BrowserAction.EXTRACT_LINKS, BrowserAction.READ_PAGE,
})

TERMINAL = (BrowserStatus.COMPLETED, BrowserStatus.FAILED)


@dataclass
class BrowserLimits:
    """Bounded budgets — the engine never enters an infinite browser loop."""

    max_steps: int = 24
    max_retries_per_step: int = 2
    max_replans: int = 2
    max_waits: int = 4
    wait_timeout_s: float = 8.0
    wait_interval_s: float = 0.35
    max_observations: int = 40
    max_results: int = 12
    max_actions: int = 30


@dataclass
class BrowserTaskRun:
    """Browser-specific execution state (resumable, JSON-able)."""

    task_id: str = ""
    goal: BrowserGoal = field(default_factory=BrowserGoal)
    steps: List[BrowserStep] = field(default_factory=list)
    index: int = 0
    status: BrowserStatus = BrowserStatus.RUNNING
    phase: str = "THINKING"
    question: str = ""
    candidates: List[Candidate] = field(default_factory=list)
    pending_reason: str = ""
    pending_role: str = ""
    extraction: BrowserExtraction = field(default_factory=BrowserExtraction)
    observations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    last_context: Optional[BrowserContext] = None
    navigation_verified: bool = False
    target_verified: bool = False
    results_verified: bool = False
    info_verified: bool = False
    actions: int = 0
    recoveries: int = 0
    replans: int = 0
    waits: int = 0
    observation_count: int = 0
    error: str = ""
    final_message: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    # Phase 23 trace mirror: every AgentTrace event emitted for this run,
    # keyed by its (global) seq. The workflow panel and the tests therefore
    # read the SAME truth as the console logs and the agent trace.
    _trace_seq: int = 0
    _trace_by_seq: Dict[int, Any] = field(default_factory=dict)

    def trace_events(self) -> List[Any]:
        """This run's trace events, oldest first (ordered by trace seq)."""
        return [self._trace_by_seq[seq]
                for seq in sorted(self._trace_by_seq)
                if seq <= self._trace_seq]

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    @property
    def verified(self) -> bool:
        return self.status == BrowserStatus.COMPLETED

    def plan_lines(self) -> List[str]:
        return [s.description or s.action.value for s in self.steps]

    def active_step(self) -> Optional[BrowserStep]:
        if 0 <= self.index < len(self.steps):
            return self.steps[self.index]
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
            "extraction": self.extraction.to_dict(),
            "observations": list(self.observations[-20:]),
            "evidence": dict(self.evidence),
            "navigation_verified": self.navigation_verified,
            "target_verified": self.target_verified,
            "results_verified": self.results_verified,
            "info_verified": self.info_verified,
            "recoveries": self.recoveries, "replans": self.replans,
            "waits": self.waits, "actions": self.actions,
            "observation_count": self.observation_count,
            "error": self.error, "final_message": self.final_message,
        }

    def user_message(self) -> str:
        """Concise user-facing report (operational, evidence-backed)."""
        if self.status == BrowserStatus.ASKING_USER:
            lines = [self.question or "I need one clarification."]
            for i, c in enumerate(self.candidates[:6], 1):
                lines.append(f"  {i}. {c.label}"
                             + (f" ({c.kind})" if c.kind else ""))
            return "\n".join(lines)
        if self.status == BrowserStatus.WAITING_FOR_USER:
            return self.question or "I need you to finish something manually."
        if self.status == BrowserStatus.COMPLETED:
            return self.final_message or self.summary()
        if self.status == BrowserStatus.FAILED:
            base = f"I couldn't complete '{self.goal.describe()}'."
            return (base + (f" {self.error}" if self.error else "")
                    + (f" {self.final_message}" if self.final_message else ""))
        return self.summary()

    def summary(self) -> str:
        bits = [f"{self.goal.describe()}: {self.status.value}"]
        if self.extraction.count:
            bits.append(f"{self.extraction.count} result(s)")
        if self.last_context is not None:
            bits.append(self.last_context.summary())
        return " | ".join(bits)


def expected_effect_text_of(action: Any,
                            params: Optional[Dict[str, Any]] = None) -> str:
    """Small helper kept for readability inside the engine."""
    return expected_effect_text(action, params or {})


def expect_effect(name: str) -> str:
    """Short operational expectation string for a BrowserAction name."""
    from agent.browser_goal import BrowserAction as _BA
    try:
        return expected_effect_text(_BA(name))
    except ValueError:
        return "the action reports a verified effect"


class BrowserGoalEngine:
    """Orchestrates a browser goal through the existing Diego subsystems."""

    def __init__(self, *, controller: Any = None, observer: Any = None,
                 trace: Any = None, limits: Optional[BrowserLimits] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 target_site: Optional[str] = None, local_base: Optional[str] = None,
                 resolve_site_overrides: Optional[Dict[str, str]] = None) -> None:
        self._controller = controller
        self._observer = observer
        self._trace = trace
        self._limits = limits or BrowserLimits()
        self._sleep = sleep
        self._pending_run: Optional[BrowserTaskRun] = None
        self._runs: Dict[str, BrowserTaskRun] = {}
        # task_id -> AgentTrace unsubscribe callable (per-run trace mirror).
        self._unsubs: Dict[str, Any] = {}
        # task_id -> merged browser-session evidence (existing-Chrome attach
        # contract; emitted as BROWSER_SESSION trace events).
        self._session_evidence: Dict[str, Dict[str, Any]] = {}
        # Phase 23 (live/integration): optional local server overrides so the
        # engine can resolve known site names to a controlled HTTP base instead
        # of a public resolution (no external dependency required).
        self._target_site = (target_site or "").strip().lower()
        self._local_base = (local_base or "").strip().rstrip("/")
        self._resolve_site_overrides = {
            normalize_label(k): str(v).strip().rstrip("/")
            for k, v in (resolve_site_overrides or {}).items() if k and v}

    # ── injected collaborators (lazy production defaults) ────────

    def controller(self) -> Any:
        if self._controller is None:
            from computer.computer_controller import computer_controller
            self._controller = computer_controller
        return self._controller

    def observer(self) -> Any:
        if self._observer is None:
            self._observer = BrowserContextObserver()
        return self._observer

    def trace(self) -> Any:
        if self._trace is None:
            from agent.trace import agent_trace
            self._trace = agent_trace
        return self._trace

    @property
    def limits(self) -> BrowserLimits:
        return self._limits

    # ── trace / phase plumbing ───────────────────────────────────

    def _watch_run(self, run: BrowserTaskRun) -> None:
        """Mirror every trace event of this run onto the run itself.

        AgentTrace subscriptions are its documented extension point (the UI
        panel uses the same one), so the workflow panel, the console logs and
        the run all read ONE truth.
        """
        if not run.task_id or run.task_id in self._unsubs:
            return
        try:
            unsubscribe = self.trace().subscribe(self._on_trace_event)
        except Exception as e:
            logger.debug("[BROWSER-GOAL] trace subscribe failed: %s", e)
            return
        if callable(unsubscribe):
            self._unsubs[run.task_id] = unsubscribe

    def _unwatch_run(self, run: BrowserTaskRun) -> None:
        """Stop mirroring a finished run (never raises)."""
        unsubscribe = self._unsubs.pop(run.task_id, None)
        if callable(unsubscribe):
            try:
                unsubscribe()
            except Exception:
                pass

    def _on_trace_event(self, event: Any) -> None:
        """Store ONE trace event on its run (listener must never raise).

        Terminal events (TASK_COMPLETED / TASK_FAILED) are emitted AFTER the
        run's status changes, so this must not skip finished runs — the mirror
        is detached by :meth:`_unwatch_run` instead.
        """
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

    def _phase(self, run: BrowserTaskRun, phase: str, detail: str = "") -> None:
        """Record the coarse operational phase (never chain-of-thought)."""
        run.phase = phase
        try:
            self.trace().phase(phase, detail=detail, task_id=run.task_id)
        except Exception as e:  # tracing must never break the task
            logger.debug("[BROWSER-GOAL] trace phase failed: %s", e)

    def _observe_trace(self, run: BrowserTaskRun, text: str,
                       ctx: Optional[BrowserContext] = None) -> None:
        run.observations.append(text[:200])
        try:
            self.trace().observation(
                text[:200], task_id=run.task_id,
                method=(ctx.perception_method if ctx else ""),
                evidence=({"url": ctx.current_url[:200],
                           "domain": ctx.domain,
                           "title": ctx.page_title[:120],
                           "elements": len(ctx.interactive_elements)}
                          if ctx else {}))
        except Exception as e:
            logger.debug("[BROWSER-GOAL] trace observation failed: %s", e)

    def _trace_action(self, run: BrowserTaskRun, step: BrowserStep,
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

    def _resolve_start_url(self, goal: BrowserGoal) -> str:
        """Resolve the goal's start URL, honouring a pinned base override.

        The parser resolves a site name to a URL up-front, so the override is
        applied to the RESOLVED value's host (an explicit URL on the pinned
        host also maps to the base, preserving its path). Used by validation
        harnesses so real browsing needs no public network; production never
        configures an override and keeps the resolved URL unchanged.
        """
        target = str(goal.url or "").strip()
        if not target and goal.site:
            target = resolve_site_url(goal.site)
        return self._base_override(target) or target

    def _base_override(self, value: str) -> str:
        """Map a pinned host (``target_site``) to its controlled base URL."""
        raw = str(value or "").strip()
        if not raw:
            return ""
        m = re.match(r"https?://([^/\s?#]+)([/?#][^\s]*)?", raw, re.IGNORECASE)
        if m:
            host, path = m.group(1), (m.group(2) or "")
        else:
            host, _, rest = raw.partition("/")
            path = f"/{rest}" if rest else ""
        key = normalize_label(host)
        if not key:
            return ""
        base = str(self._resolve_site_overrides.get(key) or "")
        if not base and self._local_base and self._target_site \
                and key == normalize_label(self._target_site):
            base = self._local_base
        if not base:
            return ""
        base = base.rstrip("/")
        return f"{base}{path}" if path else base

    def _item_required(self, goal: BrowserGoal) -> bool:
        """True when the goal demands a specific ITEM, not just its site page.

        ``"open workhub.test"`` parses with ``target == site`` — navigation
        alone IS that goal's outcome. ``"open the people page on workhub.test"``
        carries a distinct target, so the item itself must be verified.
        """
        if goal.kind in (BrowserGoalKind.FIND_ITEM, BrowserGoalKind.OPEN_ITEM):
            return True
        if goal.kind == BrowserGoalKind.OPEN_SITE and goal.target:
            return (normalize_label(goal.target)
                    != normalize_label(goal.site or ""))
        return False

    def plan(self, goal: BrowserGoal) -> List[BrowserStep]:
        """Deterministic, generic plan for an interpreted browser goal."""
        steps: List[BrowserStep] = []
        site_url = self._resolve_start_url(goal)
        if site_url:
            steps.append(BrowserStep(
                action=BrowserAction.NAVIGATE,
                description=f"Navigate to {site_url}",
                params={"url": site_url, "target": site_url},
                expected_effect=expected_effect_text(
                    BrowserAction.NAVIGATE, {"url": site_url})))
        if goal.kind == BrowserGoalKind.SEARCH:
            steps.extend(self._search_steps(goal.query or goal.target))
        elif goal.kind in (BrowserGoalKind.FIND_ITEM, BrowserGoalKind.OPEN_ITEM):
            steps.extend(self._find_steps(goal))
        elif goal.kind == BrowserGoalKind.OPEN_SITE and self._item_required(goal):
            # "Open Rahul's profile on linkedin" — an ITEM goal with a scope:
            # navigation alone is never completion, the item must be found
            # (and ambiguity resolved, never guessed).
            item = replace(goal, kind=BrowserGoalKind.OPEN_ITEM)
            steps.extend(self._find_steps(item))
        elif goal.kind in (BrowserGoalKind.EXTRACT, BrowserGoalKind.READ):
            subject = goal.query or goal.target
            steps.append(BrowserStep(
                action=BrowserAction.READ_PAGE,
                description="Read the observed page",
                params={},
                expected_effect=expected_effect_text(BrowserAction.READ_PAGE)))
            steps.append(BrowserStep(
                action=BrowserAction.EXTRACT_TEXT,
                description=(f"Extract '{subject[:60]}' from the page" if subject
                             else "Extract the page text"),
                params={"query": subject},
                expected_effect=expected_effect_text(
                    BrowserAction.EXTRACT_TEXT, {"query": subject})))
        elif goal.kind == BrowserGoalKind.OPEN_SITE and not site_url:
            steps.append(BrowserStep(
                action=BrowserAction.ASK_USER,
                description="Ask which site to open",
                params={"question": "Which site should I open?"},
                expected_effect=expected_effect_text(BrowserAction.ASK_USER)))
        elif goal.kind == BrowserGoalKind.UNKNOWN:
            steps.append(BrowserStep(
                action=BrowserAction.ASK_USER,
                description="Ask what the browser should do",
                params={"question": ("I'm not sure what to do in the browser "
                                     "— what should I look for?")},
                expected_effect=expected_effect_text(BrowserAction.ASK_USER)))
        for i, step in enumerate(steps):
            step.index = i
            if not step.expected_effect:
                step.expected_effect = expected_effect_text(step.action,
                                                            step.params)
        return steps[:self._limits.max_steps]

    def _search_steps(self, query: str) -> List[BrowserStep]:
        """Generic search plan: locate interface → enter → submit → extract."""
        return [
            BrowserStep(
                action=BrowserAction.FIND_ELEMENT,
                description="Locate the search interface",
                params={"target": ROLE_SEARCH_INPUT,
                        "role": ROLE_SEARCH_INPUT},
                expected_effect=expected_effect_text(
                    BrowserAction.FIND_ELEMENT, {"target": ROLE_SEARCH_INPUT})),
            BrowserStep(
                action=BrowserAction.TYPE_INPUT,
                description=f"Enter '{query[:60]}' in the search input",
                params={"target": ROLE_SEARCH_INPUT, "text": query},
                expected_effect=expected_effect_text(
                    BrowserAction.TYPE_INPUT, {"text": query})),
            BrowserStep(
                action=BrowserAction.PRESS_KEY,
                description="Submit the search",
                params={"key": "enter", "expect_url_change": True},
                expected_effect=("search results become visible "
                                 "(navigation or result state change)")),
            BrowserStep(
                action=BrowserAction.EXTRACT_RESULTS,
                description=f"Extract results for '{query[:60]}'",
                params={"query": query},
                expected_effect=expected_effect_text(
                    BrowserAction.EXTRACT_RESULTS, {"query": query})),
        ]

    def _find_steps(self, goal: BrowserGoal) -> List[BrowserStep]:
        target = goal.target or goal.query
        steps = [BrowserStep(
            action=BrowserAction.FIND_ELEMENT,
            description=f"Verify '{target[:60]}' is present",
            params={"target": target},
            expected_effect=expected_effect_text(BrowserAction.FIND_ELEMENT,
                                                 {"target": target}))]
        if goal.kind == BrowserGoalKind.OPEN_ITEM:
            steps.append(BrowserStep(
                action=BrowserAction.CLICK_ELEMENT,
                description=f"Open '{target[:60]}'",
                params={"target": target, "expect_url_change": True},
                expected_effect=("the target interaction causes the declared "
                                 "page transition")))
        return steps

    # ── semantic target resolution (no site-specific selectors) ──

    @staticmethod
    def search_input(ctx: BrowserContext) -> Optional[InteractiveElement]:
        """Generic search-interface identification from observed elements."""
        inputs = [e for e in ctx.inputs()
                  if not e.is_password and e.type.lower() != "hidden"]
        if not inputs:
            return None
        for e in inputs:
            hay = " ".join([e.label, e.placeholder, e.name, e.type]).lower()
            if any(w in hay for w in _SEARCH_WORDS):
                return e
        for e in inputs:
            if e.type.lower() == "search":
                return e
        if len(inputs) == 1:
            return inputs[0]          # only one text field — unambiguous
        return None

    @staticmethod
    def search_submit(ctx: BrowserContext) -> Optional[InteractiveElement]:
        """Generic submit/search control identification."""
        for e in ctx.buttons():
            hay = " ".join([e.label, e.name, e.type]).lower()
            if any(w in hay for w in _SUBMIT_WORDS):
                return e
        return None

    def _resolve_role(self, ctx: BrowserContext, role: str) -> str:
        """Turn a semantic role token into an observed element label."""
        if role == ROLE_SEARCH_INPUT:
            el = self.search_input(ctx)
            if el is None:
                return ""
            return el.label or el.placeholder or el.name or role
        if role == ROLE_SEARCH_SUBMIT:
            el = self.search_submit(ctx)
            if el is None:
                return ""
            return el.label or el.name or role
        return role

    # ── evidence: are results visible? what was extracted? ───────

    def results_evidence(self, ctx: BrowserContext,
                         query: str) -> Dict[str, Any]:
        """Generic 'search results became visible' evidence.

        Counts observed links whose text shares meaningful tokens with the
        query, plus explicit result wording. Nothing is invented: the count is
        what the PAGE reports.
        """
        q_tokens = {t for t in normalize_label(query).split() if len(t) > 2}
        matching = []
        for link in ctx.links:
            text = str(link.get("text") or "")
            tokens = set(normalize_label(text).split())
            if q_tokens and (q_tokens & tokens):
                matching.append({"text": text[:120],
                                 "href": str(link.get("href") or "")[:200]})
        wording = any(w in ctx.text_blob(2200).lower()
                      for w in ("results", "result", "showing", "found"))
        result_count = len(matching)
        verified = bool(result_count >= 1 and (wording or result_count >= 3))
        return {"verified": verified, "result_count": result_count,
                "wording_present": wording, "matches": matching[:8],
                "url": ctx.current_url[:200]}

    def extract_results(self, ctx: BrowserContext, subject: str) -> BrowserExtraction:
        """Extract result-like items as OBSERVED facts (evidence only)."""
        q_tokens = {t for t in normalize_label(subject).split() if len(t) > 2}
        results: List[BrowserResult] = []
        seen = set()
        for link in ctx.links:
            text = str(link.get("text") or "").strip()
            href = str(link.get("href") or "")
            if not text or not href or href in seen:
                continue
            tokens = set(normalize_label(text).split())
            if q_tokens and not (q_tokens & tokens):
                continue
            seen.add(href)
            results.append(BrowserResult(
                title=text[:200], url=href, source=ctx.domain,
                visible_text="",
                metadata={"perception": ctx.perception_method,
                          "page_title": ctx.page_title[:120]},
                evidence={"matched_tokens": sorted(q_tokens & tokens),
                          "page_url": ctx.current_url[:200]},
                missing=["visible_text"],
            ))
            if len(results) >= self._limits.max_results:
                break
        if not results:
            for el in ctx.interactive_elements:
                if not el.is_link:
                    # Form controls / buttons are interface chrome, not
                    # results: counting them would be a false positive.
                    continue
                label = el.label.strip()
                tokens = set(normalize_label(label).split())
                if not label or (q_tokens and not (q_tokens & tokens)):
                    continue
                if label.lower() in seen:
                    continue
                seen.add(label.lower())
                results.append(BrowserResult(
                    title=label[:200], url=el.href, source=ctx.domain,
                    metadata={"perception": el.method, "kind": el.kind},
                    evidence={"matched_tokens": sorted(q_tokens & tokens)},
                    missing=["url"] if not el.href else [],
                ))
                if len(results) >= self._limits.max_results:
                    break
        return BrowserExtraction(
            subject=subject, results=results, count=len(results),
            page_title=ctx.page_title, page_url=ctx.current_url,
            evidence={"perception_method": ctx.perception_method,
                      "links_observed": len(ctx.links),
                      "elements_observed": len(ctx.interactive_elements),
                      "query_tokens": sorted(q_tokens)},
            missing=["visible_text"] if not ctx.visible_text else [])

    # ── OBSERVE (bounded) ────────────────────────────────────────

    def _budget_left(self, run: BrowserTaskRun) -> bool:
        return run.observation_count < self._limits.max_observations

    def _observe(self, run: BrowserTaskRun, note: str = "") -> BrowserContext:
        if not self._budget_left(run):
            # Budget exhausted: reuse the last observed state instead of adding
            # another observation. The bounded loop can therefore never push
            # `observation_count` past `max_observations`; the caller fails
            # honestly on the same iteration.
            ctx = run.last_context or BrowserContext(
                perception_method="unavailable", browser_attached=False,
                error="observation budget exhausted")
            run.last_context = ctx
            return ctx
        ctx = self.observer().observe(note=note)
        run.observation_count += 1
        run.last_context = ctx
        self._refresh_session_evidence(run, ctx)
        state = ctx.page_state
        line = f"{state.value} — {ctx.summary()}"
        if note:
            line = f"{line} ({note})"
        self._observe_trace(run, line, ctx)
        return ctx

    def _wait_until(self, run: BrowserTaskRun, predicate: Callable[[Any], bool],
                    description: str, *, ctx: Optional[BrowserContext] = None,
                    ) -> Tuple[BrowserContext, bool]:
        """Bounded observe → wait → observe (never one arbitrary long sleep)."""
        ctx = ctx or run.last_context or self._observe(run, note=description)
        if predicate(ctx):
            return ctx, True
        if run.waits >= self._limits.max_waits:
            self._observe_trace(run, "wait budget exhausted — classifying state")
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

    def _observed_candidates(self, ctx: BrowserContext
                             ) -> List[Dict[str, Any]]:
        """Raw observed elements/links for semantic scoring (generic).

        One observed element yields exactly ONE candidate: the same link
        appearing in both ``interactive_elements`` and ``links`` must not be
        counted twice, or a single unambiguous match would tie against its
        own duplicate and resolution would (wrongly) report no dominance.
        """
        raw: List[Dict[str, Any]] = []
        seen: set = set()
        for item in ([{"label": e.label, "kind": e.kind, "href": e.href,
                       "value": e.value, "method": e.method, "tag": e.tag}
                      for e in ctx.interactive_elements if e.label]
                     + [{"label": str(link.get("text") or ""), "kind": "link",
                         "href": str(link.get("href") or ""),
                         "value": "", "method": "browser_dom", "tag": "a"}
                        for link in ctx.links
                        if str(link.get("text") or "")]):
            key = (normalize_label(item["label"]), item["href"],
                   item["kind"])
            if key in seen:
                continue
            seen.add(key)
            raw.append(item)
        return raw

    def _prepare_params(self, step: BrowserStep,
                        ctx: BrowserContext) -> Optional[Dict[str, Any]]:
        """Resolve semantic roles/targets against the OBSERVED page.

        Returns the params to execute, or None when the target cannot be
        resolved at all. A two-way ambiguity is reported to the caller through
        ``params["_ambiguous"]`` so the engine can ASK the user (never guess).
        An unresolvable semantic target is passed through unchanged so the
        controller can still answer honestly.
        """
        params = dict(step.params)
        target = str(params.get("target") or "")
        if not target:
            return params
        if target.startswith("@"):
            resolved = self._resolve_role(ctx, target)
            if not resolved:
                return None
            params["target"] = resolved
            if step.action in _ELEMENT_ACTIONS:
                _, ranked, ambiguous, _ = select_target(
                    resolved, self._observed_candidates(ctx))
                if ambiguous:
                    params["_ambiguous"] = ranked
                elif ranked:
                    params["candidates"] = [c.to_dict() for c in ranked[:3]]
            return params
        if step.action in _ELEMENT_ACTIONS:
            label, ranked, ambiguous, reason = select_target(
                target, self._observed_candidates(ctx))
            if ambiguous:
                params["_ambiguous"] = ranked
                return params
            if label and normalize_label(label) != normalize_label(target):
                # Semantic identification: act on what the PAGE actually
                # shows, never on a phrase the page does not contain.
                params["target"] = label
                params["resolved_from"] = target
                params["resolution"] = reason
                logger.info("[BROWSER-GOAL] resolved '%s' -> '%s' (%s)",
                            target[:60], label[:60], reason)
            if ranked:
                params["candidates"] = [c.to_dict() for c in ranked[:3]]
        return params

    def _act(self, run: BrowserTaskRun, step: BrowserStep,
             ctx: BrowserContext,
             params: Optional[Dict[str, Any]] = None
             ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Execute one step via ComputerController. Returns (result, params)."""
        self._phase(run, "ACTING", step.description)
        if params is None:
            params = self._prepare_params(step, ctx)
        if params is None:
            return None, None
        params.pop("_ambiguous", None)   # handled by the caller (ask the user)
        run.actions += 1
        try:
            self.trace().action_started(
                step.action.value, target=step.target[:80],
                method=ctx.perception_method,
                expected_effect=step.expected_effect, task_id=run.task_id)
        except Exception:
            pass
        comp_action = COMPUTER_ACTION.get(step.action, "")
        if not comp_action:
            return None, params
        try:
            result = self.controller().execute(comp_action, params)
        except Exception as e:
            logger.warning("[BROWSER-GOAL] controller failed (%s): %s",
                           comp_action, e)
            from computer.action_result import ActionOutcome, ActionResult
            result = ActionResult(action=comp_action,
                                  outcome=ActionOutcome.FAILED, error=str(e))
        self._trace_action(run, step, result)
        return result, params

    # ── VERIFY (declared expected effect vs observed state) ──────

    def _verify_step(self, run: BrowserTaskRun, step: BrowserStep,
                     before: BrowserContext, after: BrowserContext,
                     params: Optional[Dict[str, Any]]) -> EffectVerdict:
        self._phase(run, "VERIFYING", step.expected_effect or step.description)
        p = dict(params or step.params or {})

        # Extraction steps compute their own evidence BEFORE verification.
        if step.action == BrowserAction.EXTRACT_RESULTS:
            subject = str(p.get("query") or run.goal.query or "")
            run.extraction = self.extract_results(after, subject)
            p["extracted"] = run.extraction.count
            ev = self.results_evidence(after, subject)
            run.evidence["results"] = ev
            if ev.get("verified") or run.extraction.count > 0:
                run.results_verified = True
        elif step.action == BrowserAction.READ_PAGE:
            p["extracted"] = 1 if after.visible_text.strip() else 0
            run.info_verified = run.info_verified or bool(
                after.visible_text.strip())
        elif step.action == BrowserAction.EXTRACT_TEXT:
            subject = str(p.get("query") or "")
            if subject:
                run.info_verified = run.info_verified or after.contains(subject)
            else:
                run.info_verified = run.info_verified or bool(
                    after.visible_text.strip())
        elif step.action == BrowserAction.EXTRACT_LINKS:
            run.info_verified = run.info_verified or bool(after.links)

        verdict = verify_expected_effect(step.action, p, before, after)
        step.verdict = verdict.result.value
        step.observation = verdict.observed_effect
        step.status = "DONE" if verdict.passed else "FAILED"

        # Goal-level flags (only ever set from a PASSED verification).
        _item_kind = self._item_required(run.goal)
        if verdict.passed:
            if step.action in (BrowserAction.NAVIGATE, BrowserAction.OPEN_URL):
                run.navigation_verified = True
                run.evidence["navigation"] = verdict.to_dict()
            if step.action == BrowserAction.FIND_ELEMENT:
                if _item_kind:
                    run.target_verified = True
            if step.action == BrowserAction.CLICK_ELEMENT:
                if bool((verdict.evidence.get("checks") or {}).get(
                        "url_changed")):
                    run.navigation_verified = True
                if _item_kind:
                    run.target_verified = True
        try:
            self.trace().verification(
                verdict.result.value, action=step.action.value,
                target=step.target[:80], method=after.perception_method,
                expected_effect=step.expected_effect,
                observed=verdict.observed_effect,
                evidence=dict(verdict.evidence), detail=verdict.detail,
                retry_count=step.attempts, task_id=run.task_id)
        except Exception:
            pass
        logger.info("[BROWSER-GOAL] verify %s %s -> %s (%s)",
                    step.action.value, step.target[:60], verdict.result.value,
                    verdict.detail[:120])
        return verdict

    # ── existing-Chrome session evidence ─────────────────────────

    def _browser_tier_source(self) -> str:
        """Which browser the session contract applies to for THIS engine.

        Deterministic harnesses inject controller/observer (or bind their
        own page) and own their browser entirely; production uses the real
        existing-Chrome adapter.
        """
        if self._controller is not None or self._observer is not None:
            return "injected_harness"
        try:
            from computer import browser_controller as _bctl
            if getattr(_bctl, "_external_page", None) is not None:
                return "bound_page"
        except Exception:
            pass
        return "production"

    def _ensure_browser_session(self, run: BrowserTaskRun) -> Optional[str]:
        """Attach to the user's CURRENT Chrome session; None = ready.

        Production: agent.browser → agent.chrome_session (discover the
        running Chrome, its real user-data dir + profile, validate against
        the process, attach over DevTools). On failure returns the
        BROWSER_SESSION_UNAVAILABLE reason so the task fails honestly.
        """
        source = self._browser_tier_source()
        if source != "production":
            self._record_session_evidence(run, {
                "browser_process": source,
                "user_data_dir": "(owned by the caller)",
                "profile_directory": "(owned by the caller)",
                "connection_method": f"{source}",
                "authenticated_state": "unknown",
            })
            return None
        try:
            from agent.browser import browser_controller as bc
        except Exception as e:
            return f"browser controller import failed: {e}"
        try:
            attached = bool(getattr(bc, "is_attached", False)) or \
                bool(bc.initialize())
        except Exception as e:
            return f"browser session attach raised: {e}"
        if attached:
            self._record_session_evidence(run, bc.session_evidence())
            return None
        evidence = bc.unavailable_evidence()
        self._record_session_evidence(run, evidence)
        return str(evidence.get("unavailable")
                   or "browser session unavailable")

    def _record_session_evidence(self, run: BrowserTaskRun,
                                 evidence: Dict[str, Any]) -> None:
        """Merge + emit the BROWSER_SESSION trace event (never raises)."""
        merged = dict(self._session_evidence.get(run.task_id) or {})
        merged.update({k: v for k, v in dict(evidence or {}).items()
                       if v not in (None, "")})
        self._session_evidence[run.task_id] = merged
        try:
            self.trace().browser_session(dict(merged), task_id=run.task_id)
        except Exception:
            pass
        logger.info("[BROWSER-GOAL] browser session: %s",
                    {k: str(v)[:60] for k, v in merged.items()})

    def _refresh_session_evidence(self, run: BrowserTaskRun,
                                  ctx: BrowserContext) -> None:
        """Keep authenticated_state / active_tab / current_url truthful.

        Updated from the OBSERVED page state — emitted only when something
        actually changed, so the trace stays readable.
        """
        evidence = self._session_evidence.get(run.task_id)
        if evidence is None or not evidence:
            return
        try:
            state_value = ctx.page_state.value
        except Exception:
            state_value = ""
        if ctx.authenticated:
            auth = "authenticated"
        elif state_value == PageState.LOGIN_REQUIRED.value:
            auth = "unauthenticated"
        else:
            auth = "unknown"
        active_tab = str(ctx.page_title or "")
        current_url = str(ctx.current_url or "")
        if (evidence.get("authenticated_state") == auth
                and evidence.get("active_tab") == active_tab
                and evidence.get("current_url") == current_url):
            return
        self._record_session_evidence(run, {
            "authenticated_state": auth,
            "active_tab": active_tab,
            "current_url": current_url,
        })

    # ── one step: OBSERVE → ACT → OBSERVE → VERIFY (+recovery) ──

    def _run_step(self, run: BrowserTaskRun, step: BrowserStep) -> str:
        """Returns 'done' | 'retry' | 'paused' | 'failed'."""
        if not self._budget_left(run):
            # The observation budget is enforced EXACTLY: no step may push the
            # count past the limit (a bounded loop, never an endless one).
            return self._fail(
                run, "observation budget exhausted",
                detail="too many page observations without progress")
        before = self._observe(run, note=f"before {step.action.value}")
        state = before.page_state

        if state == PageState.LOADING:
            before, ok = self._wait_until(
                run, lambda c: not c.is_loading,
                "waiting for the page to finish loading", ctx=before)
            if not ok:
                state = before.page_state

        if state == PageState.LOGIN_REQUIRED and not before.authenticated:
            if run.goal.kind in (BrowserGoalKind.SEARCH, BrowserGoalKind.FIND_ITEM,
                                 BrowserGoalKind.OPEN_ITEM,
                                 BrowserGoalKind.EXTRACT):
                return self._pause_for_login(run, before)
        if state == PageState.BLOCKED:
            return self._pause_for_blocked(run, before)

        if step.action == BrowserAction.ASK_USER:
            return self._ask_user(
                run, str(step.params.get("question") or "How should I proceed?"),
                before)

        params = self._prepare_params(step, before)
        if params is not None and params.get("_ambiguous"):
            # Two (or more) equally plausible matches: ask, never guess.
            ranked = params.pop("_ambiguous")
            return self._ask_user(
                run,
                (f"I found {len(ranked)} possible matches for "
                 f"'{step.target[:60]}'. Which one should I use?"),
                before, candidates=list(ranked)[:6],
                role=str(step.params.get("target") or ""))

        result, params = self._act(run, step, before, params)

        if params is None:
            return self._recover_missing_target(run, step, before)

        if step.action == BrowserAction.FIND_ELEMENT:
            decided = self._settle_candidates(run, step, before, result, params)
            if decided is not None:
                return decided

        if result is not None and not bool(getattr(result, "success", False)):
            handled = self._handle_problem(run, step, result, before)
            if handled in ("paused", "failed"):
                return handled

        after = self._observe(run, note=f"after {step.action.value}")
        verdict = self._verify_step(run, step, before, after, params)
        if verdict.passed:
            return "done"
        return self._recover_effect(run, step, before, after, verdict)

    def _settle_candidates(self, run: BrowserTaskRun, step: BrowserStep,
                           ctx: BrowserContext, result: Any,
                           params: Dict[str, Any]) -> Optional[str]:
        """Ambiguity handling for FIND_ELEMENT (never guess)."""
        target = str(params.get("target") or "")
        role = str(step.params.get("role") or "")
        raw = list((getattr(result, "evidence", {}) or {}).get("candidates") or [])
        if role == ROLE_SEARCH_INPUT:
            raw = [c for c in raw if str(c.get("kind")) in ("input", "select",
                                                            "element")
                   or str(c.get("tag")) in ("input", "textarea", "select")]
        elif role == ROLE_SEARCH_SUBMIT:
            raw = [c for c in raw if str(c.get("kind")) in ("button", "submit")
                   or str(c.get("tag")) == "button"]
        ranked, ambiguous, reason = rank_candidates(target, raw)
        if ambiguous:
            return self._ask_user(
                run,
                (f"I found {len(ranked)} possible matches for '{target}'. "
                 f"Which one should I use?"),
                ctx, candidates=ranked[:6], role=target)
        if raw and ranked:
            params["candidates"] = [c.to_dict() for c in ranked[:3]]
        elif role and target:
            # The semantic role resolved to an observed element: that IS the
            # evidence (the DOM had no textual candidate to enumerate).
            params["candidates"] = [{"label": target, "kind": "identified",
                                     "score": 0.9}]
        return None

    # ── controller-reported problems ─────────────────────────────

    def _handle_problem(self, run: BrowserTaskRun, step: BrowserStep,
                        result: Any, ctx: BrowserContext) -> Optional[str]:
        """React to AMBIGUOUS / LOGIN_REQUIRED / UNAVAILABLE / CONFIRMATION."""
        outcome = str(getattr(getattr(result, "outcome", None), "value",
                              getattr(result, "outcome", "")) or "")
        error = str(getattr(result, "error", "") or "")
        target = str((getattr(result, "target", "") or step.target) or "")

        if outcome == "confirmation_required":
            return self._refuse(
                run, step, error or "this action has an external side effect")
        if outcome == "login_required":
            return self._pause_for_login(run, ctx)
        if outcome == "ambiguous" and step.action in (
                BrowserAction.CLICK_ELEMENT, BrowserAction.TYPE_INPUT,
                BrowserAction.FIND_ELEMENT):
            candidates = ctx.candidates(target, limit=6)
            if not candidates:
                # Zero observed candidates is NOT ambiguity — nothing matched
                # at all. Classify NOT_FOUND and run the bounded recovery
                # (refind / re-read / scroll / replan); asking "which one?"
                # with an empty list would be a guess-shaped question.
                return self._recover_missing_target(run, step, ctx)
            return self._ask_user(
                run,
                f"I couldn't uniquely locate '{target[:60]}'. Which element "
                f"should I use?", ctx, candidates=candidates[:6], role=target)
        if outcome == "unavailable":
            # Only ONE bounded recovery attempt: make the browser available.
            if step.action != BrowserAction.OPEN_BROWSER:
                if self._ensure_browser(run):
                    return "retry"
            return self._fail(run, error or "browser capability unavailable",
                              detail=f"{step.action.value} could not run")
        return None

    def _ensure_browser(self, run: BrowserTaskRun) -> bool:
        """One bounded attempt to launch the browser (generic app resolution)."""
        self._phase(run, "RECOVERING", "starting the browser")
        try:
            self.trace().recovery("open browser (capability unavailable)",
                                  action="open_app", task_id=run.task_id)
        except Exception:
            pass
        for app in ("firefox", "google-chrome", "chromium"):
            try:
                res = self.controller().execute("open_app", {"app": app})
            except Exception as e:
                logger.debug("[BROWSER-GOAL] launch %s failed: %s", app, e)
                continue
            if bool(getattr(res, "success", False)):
                return True
        return False

    # ── recovery ─────────────────────────────────────────────────

    def _wait_for_content(self, run: BrowserTaskRun, step: BrowserStep,
                          ctx: BrowserContext) -> BrowserContext:
        """Bounded observe → wait → observe until rendered content appears.

        Never one arbitrary long sleep: the wait budget and observation budget
        bound the loop, and the timeout is reported honestly when it expires.
        """
        subject = str(step.params.get("query") or run.goal.query
                      or run.goal.target or "")
        if step.action == BrowserAction.EXTRACT_RESULTS:
            def predicate(c: BrowserContext) -> bool:
                return self.extract_results(c, subject).count > 0
        elif subject:
            def predicate(c: BrowserContext) -> bool:
                return c.contains(subject)
        else:
            def predicate(c: BrowserContext) -> bool:
                return bool(c.visible_text.strip())
        waited, ok = self._wait_until(
            run, predicate, "waiting for rendered content", ctx=ctx)
        if not ok:
            self._observe_trace(
                run, "content still not rendered after the bounded wait",
                waited)
        return waited

    def _recover_effect(self, run: BrowserTaskRun, step: BrowserStep,
                        before: BrowserContext, after: BrowserContext,
                        verdict: EffectVerdict) -> str:
        """Declared effect NOT observed → bounded recovery, then replan/fail."""
        step.attempts += 1
        run.recoveries += 1
        if step.attempts > self._limits.max_retries_per_step:
            if self._replan(run, f"{step.action.value} did not produce "
                                 f"{step.expected_effect[:60]}"):
                return "retry"
            return self._fail(
                run, f"expected effect not observed for {step.action.value}",
                detail=verdict.detail[:200])
        strategy = self._recovery_strategy(step, after)
        self._phase(run, "RECOVERING", strategy)
        try:
            self.trace().recovery(strategy, action=step.action.value,
                                  retry_count=step.attempts,
                                  task_id=run.task_id,
                                  evidence=dict(verdict.evidence or {}))
        except Exception:
            pass
        if strategy == "wait for loading":
            self._wait_until(run, lambda c: not c.is_loading,
                             "waiting for dynamic content")
            return "retry"
        if strategy == "wait for content":
            self._wait_for_content(run, step, after)
            return "retry"
        if strategy == "scroll to reveal the target":
            self.controller().execute("scroll", {"delta": 600})
            self._observe(run, note="after scroll (recovery)")
            return "retry"
        if strategy == "reopen the page":
            target_url = self._resolve_start_url(run.goal)
            if target_url:
                self.controller().execute("navigate", {"url": target_url,
                                                       "target": target_url})
                self._observe(run, note="after reopen (recovery)")
            return "retry"
        self._observe(run, note="re-read (recovery)")
        return "retry"

    def _recovery_strategy(self, step: BrowserStep,
                           ctx: BrowserContext) -> str:
        """Pick a bounded, generic recovery for the current attempt count."""
        if ctx.is_loading:
            return "wait for loading"
        if step.attempts == 1:
            if step.action in (BrowserAction.FIND_ELEMENT,
                               BrowserAction.CLICK_ELEMENT,
                               BrowserAction.TYPE_INPUT):
                return "refind the element"
            if step.action in _CONTENT_ACTIONS:
                # Dynamic content may still be rendering: observe → wait a
                # bounded interval → observe again (never a guessed sleep).
                return "wait for content"
            return "re-read the page"
        if step.attempts == 2:
            return ("scroll to reveal the target"
                    if step.action in (BrowserAction.FIND_ELEMENT,
                                       BrowserAction.CLICK_ELEMENT)
                    else "reopen the page")
        return "reopen the page"

    def _recover_missing_target(self, run: BrowserTaskRun, step: BrowserStep,
                                ctx: BrowserContext) -> str:
        """The semantic target could not be resolved on the observed page."""
        step.attempts += 1
        run.recoveries += 1
        if step.attempts > self._limits.max_retries_per_step:
            if self._replan(run, "target not found on the current page"):
                return "retry"
            return self._fail(
                run,
                f"'{step.target[:60]}' is not present on "
                f"{ctx.domain or 'the page'}",
                detail=f"classified NOT_FOUND ({ctx.summary()})")
        strategy = ("wait for loading" if ctx.is_loading else
                    ("re-read the page" if step.attempts == 1
                     else "scroll to reveal the target"))
        self._phase(run, "RECOVERING", strategy)
        try:
            self.trace().recovery(strategy, action=step.action.value,
                                  retry_count=step.attempts,
                                  task_id=run.task_id,
                                  evidence={"target": step.target[:80]})
        except Exception:
            pass
        if strategy == "wait for loading":
            self._wait_until(run, lambda c: not c.is_loading,
                             "waiting for dynamic content")
        elif strategy == "scroll to reveal the target":
            self.controller().execute("scroll", {"delta": 600})
            self._observe(run, note="after scroll (recovery)")
        else:
            self._observe(run, note="re-read (recovery)")
        return "retry"

    def _replan(self, run: BrowserTaskRun, reason: str) -> bool:
        """Deterministic plan repair from the OBSERVED page (bounded)."""
        if run.replans >= self._limits.max_replans:
            return False
        run.replans += 1
        ctx = run.last_context or self._observe(run, note="replan")
        steps: List[BrowserStep] = []
        kind = run.goal.kind
        if kind == BrowserGoalKind.SEARCH:
            query = run.goal.query or run.goal.target
            submit = self.search_submit(ctx)
            if self.search_input(ctx) is not None:
                steps = self._search_steps(query)
                if submit is not None:
                    # Prefer an explicit click on the OBSERVED submit control
                    # (semantic identification, never a fixed selector).
                    for s in steps:
                        if s.action == BrowserAction.PRESS_KEY:
                            s.action = BrowserAction.CLICK_ELEMENT
                            s.description = (
                                f"Click '{submit.label[:40]}' to search")
                            s.params = {"target": submit.label,
                                        "expect_url_change": True,
                                        "expect": query}
                            s.expected_effect = (
                                "clicking the submit control shows results")
            elif ctx.links:
                steps = [BrowserStep(
                    action=BrowserAction.FIND_ELEMENT,
                    description=f"Locate a link matching '{query[:50]}'",
                    params={"target": query},
                    expected_effect=expect_effect("FIND_ELEMENT"))]
        elif kind in (BrowserGoalKind.FIND_ITEM, BrowserGoalKind.OPEN_ITEM):
            target = run.goal.target or run.goal.query
            steps = [BrowserStep(
                action=BrowserAction.FIND_ELEMENT,
                description=f"Verify '{target[:60]}' is present",
                params={"target": target},
                expected_effect=expect_effect("FIND_ELEMENT"))]
        elif kind in (BrowserGoalKind.EXTRACT, BrowserGoalKind.READ):
            subject = run.goal.query or run.goal.target
            steps = [BrowserStep(
                action=BrowserAction.EXTRACT_TEXT,
                description=f"Extract '{subject[:50]}' from the page",
                params={"query": subject},
                expected_effect=expect_effect("EXTRACT_TEXT"))]
        if not steps:
            return False
        for i, step in enumerate(steps):
            step.index = run.index + i
            if not step.expected_effect:
                step.expected_effect = expected_effect_text(step.action,
                                                            step.params)
        run.steps = run.steps[:run.index] + steps
        try:
            self.trace().replan(reason, [s.description for s in steps],
                                task_id=run.task_id)
        except Exception:
            pass
        logger.info("[BROWSER-GOAL] replan #%d: %s -> %d step(s)",
                    run.replans, reason[:80], len(steps))
        return True
# ── pause states: login / blocked / clarification ───────────

    def _pause_for_login(self, run: BrowserTaskRun, ctx: BrowserContext) -> str:
        """Authentication is NEVER bypassed: pause and ask the user."""
        run.status = BrowserStatus.WAITING_FOR_USER
        target = ctx.domain or ctx.page_title or ctx.current_url or "this site"
        run.question = (
            f"Login required on {target}. Please sign in yourself in the "
            f"browser — I never touch credentials or bypass authentication. "
            f"Say 'continue' when you're signed in and I'll carry on from "
            f"where I stopped (nothing restarts).")
        self._phase(run, "ASKING_USER", "login required — waiting for the user")
        self._observe_trace(run, f"Login page detected on {target}", ctx)
        try:
            self.trace().confirmation_required(
                "login required — manual sign-in needed (no bypass)",
                action="login", target=target[:80], task_id=run.task_id)
        except Exception:
            pass
        logger.info("[BROWSER-GOAL] paused for manual login on %s", target)
        return "paused"

    def _pause_for_blocked(self, run: BrowserTaskRun,
                           ctx: BrowserContext) -> str:
        """CAPTCHA / anti-bot / access-control: stop, never circumvent."""
        run.status = BrowserStatus.WAITING_FOR_USER
        target = ctx.domain or ctx.current_url or "the page"
        run.question = (
            f"{target} is showing an access-control / anti-bot page "
            f"(CAPTCHA or similar). I won't try to get around it. Please "
            f"resolve it manually, then say 'continue'.")
        self._phase(run, "ASKING_USER", "blocked state — need the user")
        self._observe_trace(run, f"Blocked state detected on {target}", ctx)
        try:
            self.trace().confirmation_required(
                "blocked/anti-bot state — manual intervention needed",
                action="blocked", target=target[:80], task_id=run.task_id)
        except Exception:
            pass
        return "paused"

    def _ask_user(self, run: BrowserTaskRun, question: str,
                  ctx: Optional[BrowserContext] = None,
                  *, candidates: Optional[List[Candidate]] = None,
                  role: str = "") -> str:
        """Ask for clarification (ambiguity) and keep the task for resumption."""
        run.status = BrowserStatus.ASKING_USER
        run.question = question
        run.candidates = list(candidates or [])
        run.pending_reason = "ambiguous_target" if run.candidates else "clarify"
        run.pending_role = role
        self._phase(run, "ASKING_USER", question[:120])
        self._observe_trace(run, question[:200], ctx)
        try:
            self.trace().confirmation_required(
                question[:200], action="clarification",
                target=role[:80] or (ctx.domain if ctx else ""),
                task_id=run.task_id)
        except Exception:
            pass
        return "paused"

    def _refuse(self, run: BrowserTaskRun, step: BrowserStep,
                reason: str) -> str:
        """A guardrail stopped the step: report honestly, never execute it."""
        run.status = BrowserStatus.FAILED
        run.error = reason
        run.final_message = reason
        run.ended_at = time.time()
        self._phase(run, "FAILED", reason[:140])
        self._observe_trace(run, f"Refused: {reason[:160]}")
        try:
            self.trace().failed(reason[:240], action=step.action.value,
                                target=step.target[:80], task_id=run.task_id)
        except Exception:
            pass
        self._unwatch_run(run)
        return "failed"

    # ═════════════════════════════════════════════════════════════
    # Public task-level control loop
    # ═════════════════════════════════════════════════════════════

    def _goal_outcome_verified(self, run: BrowserTaskRun) -> bool:
        """COMPLETED requires the goal's OWN outcome verified — never just
        navigation. Each goal kind maps to exactly one verified flag, and
        those flags are only ever set from a PASSED expected-effect check."""
        kind = run.goal.kind
        if kind in (BrowserGoalKind.OPEN_SITE, BrowserGoalKind.OPEN_URL):
            if self._item_required(run.goal):
                # "open Rahul's profile on X": navigation AND the item itself.
                return run.navigation_verified and run.target_verified
            return run.navigation_verified
        if kind == BrowserGoalKind.SEARCH:
            return run.results_verified
        if kind in (BrowserGoalKind.FIND_ITEM, BrowserGoalKind.OPEN_ITEM):
            return run.target_verified
        if kind in (BrowserGoalKind.EXTRACT, BrowserGoalKind.READ):
            return run.info_verified
        return False

    def _fail(self, run: BrowserTaskRun, error: str, *,
              detail: str = "") -> str:
        """Honest terminal failure (never a guessed partial success)."""
        run.status = BrowserStatus.FAILED
        run.error = error[:300]
        run.final_message = (detail or error)[:300]
        run.ended_at = time.time()
        self._phase(run, "FAILED", error[:140])
        try:
            self.trace().failed(f"{error[:200]} {detail[:120]}".strip(),
                                action="browser_goal",
                                target=run.goal.describe()[:80],
                                task_id=run.task_id,
                                evidence={"actions": run.actions,
                                          "recoveries": run.recoveries,
                                          "replans": run.replans})
        except Exception:
            pass
        logger.info("[BROWSER-GOAL] task %s FAILED: %s (%s)",
                    run.task_id, error[:120], detail[:120])
        self._unwatch_run(run)
        return "failed"

    def _complete(self, run: BrowserTaskRun) -> None:
        """Terminal success — only reached with a verified goal outcome."""
        run.status = BrowserStatus.COMPLETED
        run.ended_at = time.time()
        bits = [run.goal.describe(), "completed"]
        if run.extraction.count:
            bits.append(f"{run.extraction.count} result(s) extracted")
        if run.last_context is not None:
            bits.append(f"verified on {run.last_context.domain or 'page'}")
        run.final_message = " | ".join(bits)
        self._phase(run, "COMPLETED", run.final_message[:140])
        try:
            self.trace().completed(
                run.final_message[:240], task_id=run.task_id,
                evidence={"goal": run.goal.describe(),
                          "requirement": run.goal.completion_requirement(),
                          "actions": run.actions,
                          "recoveries": run.recoveries,
                          "replans": run.replans,
                          "extraction": run.extraction.count,
                          **({"results": [r.to_dict() for r in
                                          run.extraction.results[:3]]}
                             if run.extraction.count else {})})
        except Exception:
            pass
        logger.info("[BROWSER-GOAL] task %s COMPLETED: %s",
                    run.task_id, run.final_message[:160])
        self._unwatch_run(run)

    def _finalize(self, run: BrowserTaskRun) -> None:
        """All steps done (or budget exhausted): decide honestly."""
        ctx = run.last_context
        if run.status == BrowserStatus.WAITING_FOR_USER:
            return  # paused for login/blocked — not finished, keep the task
        if ctx is not None and ctx.page_state == PageState.BLOCKED:
            self._pause_for_blocked(run, ctx)
            return
        if ctx is not None and ctx.page_state == PageState.LOGIN_REQUIRED \
                and not ctx.authenticated:
            self._pause_for_login(run, ctx)
            return
        if self._goal_outcome_verified(run):
            self._complete(run)
            return
        self._fail(
            run,
            f"the goal's outcome was never verified: "
            f"{run.goal.completion_requirement()}",
            detail=("the steps ran but the declared end state was not "
                    "observed — navigation alone is never completion"))

    def _drive(self, run: BrowserTaskRun) -> None:
        """The bounded PLAN → (OBSERVE→ACT→OBSERVE→VERIFY→RECOVER)* loop."""
        retries_without_progress = 0
        last_index = run.index
        while not run.finished:
            if run.actions > self._limits.max_actions:
                self._fail(run, "action budget exhausted",
                           detail="too many low-level browser actions")
                return
            if run.observation_count > self._limits.max_observations:
                self._fail(run, "observation budget exhausted",
                           detail="too many page observations without "
                                  "progress")
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
                    self._fail(run, "no progress — browser loop stopped",
                               detail="the same step kept failing")
                    return
            outcome = self._run_step(run, step)
            if outcome == "done":
                run.index += 1
                retries_without_progress = 0
            elif outcome == "paused":
                return  # resumable: login / blocked / clarification
            elif outcome == "failed":
                return
            # "retry" → loop continues on the same/replaced step (bounded by
            # step.attempts, replan budget and retries_without_progress).
        if run.status == BrowserStatus.RUNNING:
            self._finalize(run)

    # ── start / resume / command entry ───────────────────────────

    def start(self, text: str, *, task_id: str = "") -> \
            Optional[BrowserTaskRun]:
        """Interpret `text` as a browser goal, plan it, and register the run."""
        goal = parse_browser_goal(text)
        if goal is None:
            return None
        reason = guardrail_reason(text)
        if reason:
            task_id = task_id or f"browser-{uuid.uuid4().hex[:8]}"
            run = BrowserTaskRun(task_id=task_id, goal=goal,
                                 status=BrowserStatus.FAILED,
                                 error=reason, final_message=reason,
                                 ended_at=time.time())
            self._pending_run = run
            self._runs[task_id] = run
            self._phase(run, "FAILED", reason[:140])
            try:
                self.trace().failed(reason[:240], action="browser_goal",
                                    target=goal.describe()[:80],
                                    task_id=task_id)
            except Exception:
                pass
            logger.info("[BROWSER-GOAL] refused: %s", reason[:160])
            return run
        if not (goal.site or goal.url or goal.kind == BrowserGoalKind.READ):
            return None  # not site-scoped: the existing pipeline keeps it
        task_id = task_id or f"browser-{uuid.uuid4().hex[:8]}"
        run = BrowserTaskRun(task_id=task_id, goal=goal)
        run.steps = self.plan(goal)
        self._pending_run = run
        self._runs[task_id] = run
        self._watch_run(run)   # mirror the trace BEFORE the first event
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
        logger.info("[BROWSER-GOAL] start %s: %s (%d steps)", task_id,
                    goal.describe()[:100], len(run.steps))
        # The task runs on the USER'S EXISTING Chrome session. Attach (or
        # verify the attach) BEFORE any step; an unattachable running Chrome
        # fails the task HONESTLY with BROWSER_SESSION_UNAVAILABLE — it is
        # never silently swapped for a fresh/clean browser.
        session_error = self._ensure_browser_session(run)
        if session_error:
            self._fail(run, "BROWSER_SESSION_UNAVAILABLE",
                       detail=session_error[:280])
        return run

    def resume(self, answer: str, *, task_id: str = "") -> \
            Optional[BrowserTaskRun]:
        """Continue a PAUSED run from the observed state (never a restart)."""
        run = (self._runs.get(task_id) if task_id else None) \
            or self._pending_run
        if run is None or run.finished:
            return run
        text = str(answer or "").strip()
        low = text.lower()
        if any(w in low for w in ("cancel", "stop", "never mind",
                                  "nevermind", "forget it")):
            try:
                self.trace().confirmation_resolved(
                    False, action="browser_goal",
                    detail="user cancelled the paused browser task",
                    task_id=run.task_id)
            except Exception:
                pass
            self._fail(run, "cancelled by the user")
            return run
        # A paused task resumes on the SAME existing Chrome session. If that
        # session is gone (Chrome closed / debugging endpoint lost), fail
        # honestly instead of continuing against a dead or different browser.
        session_error = self._ensure_browser_session(run)
        if session_error:
            self._fail(run, "BROWSER_SESSION_UNAVAILABLE",
                       detail=session_error[:280])
            return run
        if run.status == BrowserStatus.ASKING_USER:
            chosen = resolve_answer_to_candidate(text, list(run.candidates))
            step = run.active_step()
            if step is not None:
                original = str(step.params.get("target") or "")
                if chosen is not None:
                    step.params["target"] = chosen.label
                elif text:
                    # The user answered with a free-form target — use it as
                    # the semantic target (still resolved against the
                    # OBSERVED page, never as a guess).
                    step.params["target"] = text
                    chosen = Candidate(label=text)
                # The clarification applies to the WHOLE remaining plan:
                # later steps that still refer to the same unresolved target
                # must use the user's choice too (never re-ask, never guess).
                if chosen is not None and original:
                    for later in run.steps[run.index + 1:]:
                        later_target = str(later.params.get("target") or "")
                        if (later_target and normalize_label(later_target)
                                == normalize_label(original)):
                            later.params["target"] = chosen.label
            if chosen is not None:
                try:
                    self.trace().confirmation_resolved(
                        True, action="clarification",
                        target=chosen.label[:80],
                        detail=f"using the user's choice "
                               f"'{chosen.label[:60]}'",
                        task_id=run.task_id)
                except Exception:
                    pass
            run.candidates = []
            run.pending_reason = ""
            run.status = BrowserStatus.RUNNING
            self._watch_run(run)   # keep the trace mirror attached
            self._phase(run, "ACTING", "continuing with your answer")
            self._drive(run)
            return run
        if run.status == BrowserStatus.WAITING_FOR_USER:
            # Re-observe: has the state the user was asked to produce
            # appeared? The task continues from the CURRENT page.
            ctx = self._observe(run, note="resume after manual intervention")
            state = ctx.page_state
            if state == PageState.BLOCKED:
                run.question = ("The page is still showing an access-control "
                                "check. Say 'continue' once it is resolved.")
                self._observe_trace(run, "still blocked — staying paused", ctx)
                return run
            if state == PageState.LOGIN_REQUIRED and not ctx.authenticated:
                run.question = ("Still seeing the sign-in page. Sign in "
                                "yourself, then say 'continue'.")
                self._observe_trace(run, "still on the login page — paused",
                                    ctx)
                return run
            run.status = BrowserStatus.RUNNING
            run.question = ""
            try:
                self.trace().confirmation_resolved(
                    True, action="login" if detect_login(ctx) else "resume",
                    detail="resolved state observed — resuming the same "
                           "task from the current page",
                    task_id=run.task_id)
            except Exception:
                pass
            self._observe_trace(
                run, f"Resolved state observed on "
                     f"{ctx.domain or 'page'} — resuming", ctx)
            self._phase(run, "ACTING", "resuming from the observed page")
            self._drive(run)
            return run
        return run

    def run_to_terminal(self, run: Optional[BrowserTaskRun]
                        ) -> Optional[BrowserTaskRun]:
        """Drive a started run until it reaches a terminal or paused state.

        COMPLETED / FAILED are terminal. ASKING_USER and WAITING_FOR_USER are
        resumable pauses: the task is kept (never restarted) and continues
        through :meth:`resume` once the user answers or finishes the manual
        step. The loop itself is bounded by :class:`BrowserLimits`.
        """
        if run is None:
            return None
        if not run.finished and run.status == BrowserStatus.RUNNING:
            self._drive(run)
        if not run.finished and run.status == BrowserStatus.RUNNING:
            self._finalize(run)
        return run

    def handle_command(self, text: str) -> Optional[str]:
        """Entry point used by the Brain.

        Returns the user-facing message when this text was (or resumed) a
        browser goal, or None when the existing pipeline should handle it.
        """
        run = self._pending_run
        if run is not None and run.status in (
                BrowserStatus.ASKING_USER, BrowserStatus.WAITING_FOR_USER):
            self.resume(text)
            return self._pending_run.user_message()
        goal = parse_browser_goal(text)
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
                and run.status in (BrowserStatus.ASKING_USER,
                                   BrowserStatus.WAITING_FOR_USER))

    def pending_run(self) -> Optional[BrowserTaskRun]:
        return self._pending_run

    def run_for(self, task_id: str) -> Optional[BrowserTaskRun]:
        return self._runs.get(task_id)

    def runs(self) -> Dict[str, BrowserTaskRun]:
        return dict(self._runs)


# Module-level singleton (the Brain imports this).
browser_goal_engine = BrowserGoalEngine()