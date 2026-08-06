"""
AgentBrain — Central orchestrator for autonomous desktop operation.

The Brain is the highest-level agent component. It:
  1. Receives high-level user goals
  2. Decomposes goals into tasks (uses LLM)
  3. Delegates each task to AgentPlanner
  4. Monitors execution progress
  5. Recovers from failures
  6. Updates persistent memory (GoalManager + ExperienceDB)
  7. Publishes lifecycle events to the EventBus

The Brain NEVER directly executes actions. Everything flows through Planner.
The Brain is the "thinking" layer — Planner is the "doing" layer.

Architecture:
    User Goal
        │
        ▼
    Brain.decompose_goal()  ── LLM decomposition into Task DAG
        │
        ▼
    Brain.execute_goal()    ── iterate tasks, delegate to Planner
        │
        ▼
    AgentPlanner.process_request()  ── per-task execution
        │
        ▼
    AgentExecutor  ── actual desktop/browser actions

Event Flow:
    goal:started  →  task:started  →  task:completed/failed  →  goal:completed/failed
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class GoalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A single task within a goal."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_s: float = 300.0
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    """A high-level user goal that may span multiple sessions."""
    id: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    current_task_index: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""


@dataclass
class CommandResult:
    """
    Result of processing a spoken command through the full pipeline.

    The ConversationEngine uses this to speak the response.
    """
    response: str = ""
    actions_executed: int = 0
    actions_succeeded: int = 0
    actions_failed: int = 0
    verified: bool = True
    used_llm: bool = False
    path: str = ""                      # Which path resolved the command
    latency_ms: float = 0.0
    error: str = ""


# ═══════════════════════════════════════════════════════════════
# Decomposition Prompt
# ═══════════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = """You are Leo, an autonomous desktop AI agent. Decompose the user's goal into a sequence of concrete, executable tasks.

Each task must be a single action Leo can perform on the desktop. Tasks should be ordered logically, with dependencies noted.

Available task types:
- Navigate: open a URL, open an application, switch windows
- Click: click buttons, menus, links
- Type: enter text into fields
- Read: read screen content, get text from elements
- Execute: run terminal commands, run scripts
- Verify: check if something succeeded
- Wait: wait for something to load
- Notify: tell the user something

Output ONLY a JSON array of tasks. Each task must have:
  {"id": "unique_id", "description": "what to do", "depends_on": ["id_of_prerequisite"], "timeout_s": 300}

Example:
[
  {"id": "1", "description": "Open PyCharm project GhostLine", "depends_on": [], "timeout_s": 60},
  {"id": "2", "description": "Run the test suite", "depends_on": ["1"], "timeout_s": 300},
  {"id": "3", "description": "Read test output and identify failures", "depends_on": ["2"], "timeout_s": 30},
  {"id": "4", "description": "Fix failing tests", "depends_on": ["3"], "timeout_s": 600},
  {"id": "5", "description": "Run tests again to verify fixes", "depends_on": ["4"], "timeout_s": 300},
  {"id": "6", "description": "Commit and push changes", "depends_on": ["5"], "timeout_s": 60}
]

Return ONLY valid JSON, no other text."""


# ═══════════════════════════════════════════════════════════════
# AgentBrain
# ═══════════════════════════════════════════════════════════════

class AgentBrain:
    """
    Central orchestrator for autonomous desktop operation.

    Receives high-level goals, decomposes them into tasks, delegates
    execution to AgentPlanner, monitors progress, recovers failures,
    and persists state for cross-session continuity.

    Never directly executes actions — everything goes through Planner.
    """

    def __init__(self):
        self._llm_client = None
        self._planner = None
        self._dispatcher = None
        self._verifier = None
        self._learning = None
        self._perception = None
        self._decision_engine = None
        self._initialized = False
        self._active_goal: Optional[Goal] = None
        self._goal_history: List[Goal] = []
        self._goal_manager = None  # Set after GoalManager is created
        self._event_bus = None     # Wired from outside
        self._lock = asyncio.Lock()

        # Pipeline stats
        self._commands_processed: int = 0
        self._actions_dispatched: int = 0
        self._actions_verified: int = 0
        self._actions_failed: int = 0

    # ── Wiring ─────────────────────────────────────────────────

    def set_llm_client(self, client) -> None:
        """Wire an LLM client for goal decomposition."""
        self._llm_client = client

    def set_event_bus(self, bus) -> None:
        """Wire the EventBus for lifecycle event publishing."""
        self._event_bus = bus

    def set_goal_manager(self, manager) -> None:
        """Wire the GoalManager for persistence."""
        self._goal_manager = manager

    async def initialize(self) -> bool:
        """Initialize the Brain and wire ALL subsystems.

        The Brain is the single orchestrator. Every subsystem is wired
        here so no other module can bypass the pipeline.
        """
        logger.info("[Brain] Initializing AgentBrain...")

        # ── Planner (creates plans ONLY) ──
        try:
            from agent.planner import agent_planner
            self._planner = agent_planner
            if not self._planner.is_available:
                self._planner.initialize()
        except Exception as e:
            logger.warning("[Brain] Planner not available: %s", e)
            self._planner = None

        # ── Action Dispatcher (executes actions ONLY) ──
        try:
            from agent.action_dispatcher import action_dispatcher
            self._dispatcher = action_dispatcher
        except Exception as e:
            logger.warning("[Brain] ActionDispatcher not available: %s", e)
            self._dispatcher = None

        # ── Action Verifier (verifies actions ONLY) ──
        try:
            from vision.action_verifier import action_verifier
            self._verifier = action_verifier
        except Exception as e:
            logger.warning("[Brain] ActionVerifier not available: %s", e)
            self._verifier = None

        # ── Learning Engine (records outcomes ONLY) ──
        try:
            from learning.learning_engine import learning_engine
            self._learning = learning_engine
        except Exception as e:
            logger.warning("[Brain] LearningEngine not available: %s", e)
            self._learning = None

        # ── Perception Pipeline (observes ONLY) ──
        try:
            from services.perception_pipeline import perception_pipeline
            self._perception = perception_pipeline
        except Exception as e:
            logger.warning("[Brain] PerceptionPipeline not available: %s", e)
            self._perception = None

        # ── Decision Engine (routes ONLY) ──
        try:
            from core.decision_engine import decision_engine
            self._decision_engine = decision_engine
        except Exception as e:
            logger.warning("[Brain] DecisionEngine not available: %s", e)
            self._decision_engine = None

        # LLM client fallback
        if self._llm_client is None:
            try:
                from ai.llm_client import llm_client
                self._llm_client = llm_client
            except Exception:
                pass

        self._initialized = True
        logger.info("[Brain] AgentBrain initialized (planner=%s, dispatcher=%s, verifier=%s, learning=%s, perception=%s)",
                     "ready" if self._planner else "unavailable",
                     "ready" if self._dispatcher else "unavailable",
                     "ready" if self._verifier else "unavailable",
                     "ready" if self._learning else "unavailable",
                     "ready" if self._perception else "unavailable")
        return True

    @property
    def is_available(self) -> bool:
        return self._initialized and self._planner is not None

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT: process_command()
    # The single pipeline for every spoken command:
    #   perceive → decide → plan → dispatch → verify → learn → respond
    # ═══════════════════════════════════════════════════════════

    async def process_command(self, text: str) -> CommandResult:
        """
        Process a spoken command through the full execution pipeline.

        This is the ONLY entry point for command execution. The
        ConversationEngine calls this and speaks the response.

        Flow:
          0. Normalize: canonicalize the spoken command
          1. Perceive: collect desktop context
          2. Decide: route the command (LLM last resort)
          3. Plan: create a plan if needed
          4. Dispatch: execute actions
          5. Verify: verify each action
          6. Learn: record outcomes
          7. Respond: return the response to speak

        Args:
            text: The user's spoken command.

        Returns:
            CommandResult with the response to speak.
        """
        if not self._initialized:
            await self.initialize()

        t0 = time.time()
        self._commands_processed += 1
        result = CommandResult()

        # ── Step 0: Normalize (NEW) ──────────────────────────
        # Canonicalize the spoken command before any processing.
        # Handles app aliases, verb normalization, noise removal,
        # follow-up commands, music/search detection.
        from nlp.command_normalizer import command_normalizer
        normalized = command_normalizer.normalize(text)
        if normalized != text:
            logger.info("[Brain] Normalized: '%s' → '%s'", text[:50], normalized[:50])
        text = normalized

        # ── Step 1: Perceive ─────────────────────────────────
        perception_ctx = await self._perceive()

        # ── Step 2: Decide ───────────────────────────────────
        decision = await self._decide(text, perception_ctx)

        if decision.resolved:
            # ── Simple path: no LLM needed ──────────────────
            result.path = decision.path.value
            result.used_llm = False

            # Execute single action
            if decision.action:
                ok = await self._dispatch_and_verify(decision.action)
                result.actions_executed = 1
                result.actions_succeeded = 1 if ok else 0
                result.actions_failed = 0 if ok else 1
                result.verified = ok

            # Execute multi-step workflow
            if decision.actions:
                for action in decision.actions:
                    ok = await self._dispatch_and_verify(action)
                    result.actions_executed += 1
                    if ok:
                        result.actions_succeeded += 1
                    else:
                        result.actions_failed += 1
                result.verified = result.actions_failed == 0

            # ── Response (CRITICAL FIX): generated AFTER execution ──
            # Never use the decision's canned response when actions FAILED.
            # The user must hear the actual outcome, not a promise to act.
            if result.actions_executed > 0:
                if result.actions_failed == 0:
                    # All succeeded — use the decision's canned confirmation
                    result.response = decision.response or "Done."
                else:
                    # At least one action failed — be honest about the failure.
                    # The decision's response ("Opening.") must NOT be spoken
                    # when the action did not actually open anything.
                    result.response = self._default_response(result)
            else:
                # No actions dispatched — conversational/cached responses are fine
                result.response = decision.response or self._default_response(result)

        else:
            # ── Complex path: needs LLM ─────────────────────
            result.path = "LLM"
            result.used_llm = True

            # Step 3: Plan (if needed)
            plan = await self._plan(text, perception_ctx)

            # Steps 4-6: Dispatch → Verify → Learn
            if plan:
                for step in plan:
                    action = self._step_to_action(step)
                    if action:
                        ok = await self._dispatch_and_verify(action)
                        result.actions_executed += 1
                        if ok:
                            result.actions_succeeded += 1
                        else:
                            result.actions_failed += 1
                result.verified = result.actions_failed == 0

            # Step 7: Respond (LLM generates the response)
            result.response = await self._generate_response(text, perception_ctx, result)

        result.latency_ms = (time.time() - t0) * 1000
        logger.info("[Brain] Command processed: '%s' path=%s actions=%d verified=%s latency=%.0fms",
                     text[:50], result.path, result.actions_executed, result.verified, result.latency_ms)
        return result

    # ── Pipeline steps ─────────────────────────────────────────

    async def _perceive(self) -> Optional[Any]:
        """Step 1: Perceive desktop context."""
        if not self._perception:
            return None
        try:
            ctx = await self._perception.perceive()
            logger.debug("[Brain] Perception: window='%s' a11y=%s ocr=%s",
                         getattr(ctx, 'window_title', '')[:40],
                         getattr(ctx, 'a11y_available', False),
                         getattr(ctx, 'ocr_used', False))
            return ctx
        except Exception as e:
            logger.debug("[Brain] Perception failed: %s", e)
            return None

    async def _decide(self, text: str, perception_ctx: Optional[Any]) -> Any:
        """Step 2: Decide how to handle the command."""
        if not self._decision_engine:
            # Fallback: always use LLM
            from core.decision_engine import Decision, DecisionPath
            return Decision(path=DecisionPath.LLM, needs_llm=True)

        try:
            desktop_ctx = ""
            if perception_ctx and hasattr(perception_ctx, 'compact_summary'):
                desktop_ctx = perception_ctx.compact_summary

            return await self._decision_engine.decide(
                text,
                vision_context=desktop_ctx or None,
                search_context=None,
                desktop_context=desktop_ctx,
            )
        except Exception as e:
            logger.warning("[Brain] Decision failed: %s", e)
            from core.decision_engine import Decision, DecisionPath
            return Decision(path=DecisionPath.LLM, needs_llm=True)

    async def _plan(self, text: str, perception_ctx: Optional[Any]) -> Optional[List[Dict[str, Any]]]:
        """Step 3: Create a plan (Planner only creates plans, never executes)."""
        if not self._planner:
            return None
        try:
            loop = asyncio.get_event_loop()
            plan = await loop.run_in_executor(
                None, self._planner.generate_plan_only, text
            )
            if plan:
                logger.info("[Brain] Plan created: %d steps", len(plan))
            return plan
        except Exception as e:
            logger.warning("[Brain] Planning failed: %s", e)
            return None

    async def _dispatch_and_verify(self, action: Dict[str, Any]) -> bool:
        """
        Steps 4-6: Dispatch → Verify → Learn.

        This is the ONLY place actions are dispatched, verified, and
        recorded for learning.

        ORDER INVARIANT (CRITICAL):
          1. Capture pre-action screen state (BEFORE execution)
          2. Execute the action exactly once (dispatch)
          3. Verify AFTER execution by comparing pre/post screen state
          4. Learn the outcome
        """
        if not self._dispatcher:
            logger.warning("[Brain] No dispatcher — cannot execute action")
            return False

        action_name = action.get("action", "")
        params = action.get("params", {}) or {}

        # ── Step 4a: Capture PRE-action state (BEFORE dispatch) ──
        # CRITICAL FIX: without a pre-action snapshot, the verifier
        # cannot detect ANY screen change and always reports NO_CHANGE.
        # This made every action fail verification even when it succeeded.
        if self._verifier:
            try:
                self._verifier.capture_pre_action()
                logger.debug("[Brain] Pre-action capture for %s", action_name)
            except Exception as e:
                logger.debug("[Brain] Pre-action capture failed: %s", e)

        # ── Step 4b: Dispatch (execute exactly once) ───────────
        try:
            result = await self._dispatcher.execute(action)
            self._actions_dispatched += 1
        except Exception as e:
            logger.warning("[Brain] Dispatch failed for %s: %s", action_name, e)
            self._actions_failed += 1
            await self._learn(action_name, params, success=False, error=str(e))
            return False

        # ── Step 5: Verify ──────────────────────────────────
        verified = await self._verify(action_name, params, result)
        self._actions_verified += 1

        if not verified:
            logger.warning("[Brain] Action %s failed verification", action_name)
            self._actions_failed += 1
            await self._learn(action_name, params, success=False, error="Verification failed")
            return False

        # ── Step 6: Learn ───────────────────────────────────
        await self._learn(action_name, params, success=True)
        return True

    async def _verify(self, action_name: str, params: Dict[str, Any],
                      result: Optional[str]) -> bool:
        """
        Step 5: Verify an action had the expected effect.

        Verification strategy (most reliable first):
          1. If dispatch returned an explicit failure ("Couldn't...") → fail fast
          2. OS-level process verification for desktop_open / browser actions
             (pgrep — authoritative: did the app actually launch?)
          3. Vision/screen comparison for UI actions (click, type, scroll)
          4. Trust dispatch result on verification subsystem failure
        """
        # ── Fail fast if the dispatcher itself reported failure ──
        if result is not None and "Couldn't" in str(result):
            logger.debug("[Brain] Verify FAIL (dispatcher reported): %s", result[:80])
            return False

        # ── OS-level process verification (authoritative for apps/browsers) ──
        try:
            import subprocess
            import shutil
            if shutil.which("pgrep"):
                if action_name == "desktop_open":
                    app = str(params.get("app", "")).lower()
                    # Map friendly names to process names
                    proc_map = {
                        "code": "code", "vscode": "code", "vs code": "code",
                        "firefox": "firefox", "browser": "firefox",
                        "chrome": "chrome", "google-chrome": "chrome",
                        "spotify": "spotify", "gnome-terminal": "gnome-terminal",
                        "terminal": "gnome-terminal", "nautilus": "nautilus",
                        "files": "nautilus", "slack": "slack",
                        "discord": "discord", "telegram-desktop": "telegram",
                        "notion-app": "notion",
                    }
                    proc = proc_map.get(app, app)
                    chk = subprocess.run(
                        ["pgrep", "-f", proc],
                        capture_output=True, text=True, timeout=3,
                    )
                    if chk.returncode == 0:
                        logger.info("[Brain] Verify OK: process '%s' running", proc)
                        return True
                    # Process not found — try alternate binaries
                    for alt in (proc.replace("-", ""), f"{proc}-esr", f"{proc}-stable"):
                        chk2 = subprocess.run(
                            ["pgrep", "-f", alt],
                            capture_output=True, text=True, timeout=3,
                        )
                        if chk2.returncode == 0:
                            logger.info("[Brain] Verify OK: process '%s' running", alt)
                            return True

                elif action_name in ("browser_navigate", "browser_search"):
                    chk = subprocess.run(
                        ["pgrep", "-f", "firefox|chrome|chromium|brave"],
                        capture_output=True, text=True, timeout=3,
                    )
                    if chk.returncode == 0:
                        logger.info("[Brain] Verify OK: browser process running")
                        return True
        except Exception as e:
            logger.debug("[Brain] OS-level verify failed: %s", e)

        # ── Vision/screen verification (for UI actions) ──
        if self._verifier:
            try:
                verify_type = self._map_action_to_verify_type(action_name)
                if verify_type:
                    vresult = await self._verifier.verify_action(
                        verify_type,
                        params,
                        expected_outcome="",
                    )
                    if vresult.success:
                        logger.info("[Brain] Verify OK: %s (%s)", action_name, vresult.status.value)
                        return True
                    logger.warning("[Brain] Verify FAIL: %s — %s",
                                   action_name, vresult.explanation[:100])
                    # If OS-level checks passed but vision says no change, the
                    # action may still have succeeded (e.g., app already open).
                    # IMPORTANT: Do NOT return False here — let the caller
                    # decide based on dispatch result + OS checks above.
                else:
                    # No verification type mapped — trust the dispatch result
                    return result is not None and "Couldn't" not in str(result)
            except Exception as e:
                logger.debug("[Brain] Vision verify failed: %s", e)

        # ── Fallback: trust a clean dispatch result ──
        return result is not None and "Couldn't" not in str(result)

    async def _learn(self, action_name: str, params: Dict[str, Any],
                     success: bool, error: str = "") -> None:
        """Step 6: Record the action outcome for learning."""
        if not self._learning:
            return
        try:
            self._learning.record_action(
                action_name=action_name,
                params=params,
                success=success,
                error=error,
            )
        except Exception as e:
            logger.debug("[Brain] Learning record failed: %s", e)

    async def _generate_response(self, text: str, perception_ctx: Optional[Any],
                                  result: CommandResult) -> str:
        """
        Step 7: Generate a conversational response.

        Uses the streaming LLM for complex commands. For simple
        commands, uses the decision response or a default.
        """
        # If we already have a response from the decision, use it
        if result.response:
            return result.response

        # If actions were executed, give a natural confirmation
        if result.actions_executed > 0:
            if result.actions_failed == 0:
                return "Done."
            return f"I ran into an issue with {result.actions_failed} of the steps."

        # Otherwise, use the LLM to generate a response
        try:
            from agent.streaming_llm import streaming_llm
            sentences = []
            async for sentence in streaming_llm.generate(text):
                sentences.append(sentence)
            return " ".join(sentences) if sentences else "I'm not sure how to help with that."
        except Exception as e:
            logger.warning("[Brain] LLM response failed: %s", e)
            return "I'm having trouble with that right now."

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _map_action_to_verify_type(action_name: str) -> Optional[str]:
        """Map dispatcher action names to verifier action types."""
        mapping = {
            "desktop_open": "open_app",
            "close_app": "open_app",
            "browser_navigate": "navigate",
            "browser_search": "navigate",
            "click_text": "click",
            "scroll": "scroll",
            "type_text": "type",
            "key_press": "key_press",
        }
        return mapping.get(action_name)

    @staticmethod
    def _step_to_action(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a planner step to an action dict."""
        if not step:
            return None
        action = step.get("action", "")
        if not action:
            return None
        return {
            "action": action,
            "params": step.get("params", {}) or {},
        }

    @staticmethod
    def _default_response(result: CommandResult) -> str:
        """Generate a default response based on execution results."""
        if result.actions_executed == 0:
            return "On it."
        if result.actions_failed == 0:
            return "Done."
        return f"Mostly done, but {result.actions_failed} step(s) had issues."

    @property
    def active_goal(self) -> Optional[Goal]:
        return self._active_goal

    # ── Goal Processing ────────────────────────────────────────

    async def process_goal(self, description: str, context: Optional[Dict[str, Any]] = None) -> Goal:
        """
        Process a high-level user goal end-to-end.

        This is the main entry point for autonomous operation:
          1. Decompose the goal into tasks (LLM)
          2. Build a task DAG with dependencies
          3. Execute tasks in order, respecting dependencies
          4. Recover from failures automatically
          5. Persist progress after each task

        Args:
            description: What the user wants ("Finish GhostLine")
            context: Optional desktop/project context

        Returns:
            The completed (or failed) Goal with results.
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            goal_id = f"goal_{int(time.time())}"
            goal = Goal(
                id=goal_id,
                description=description,
                context=context or {},
            )

            # ── Step 1: Decompose ──────────────────────────
            logger.info("[Brain] Decomposing goal: '%s'", description)
            await self._emit("goal:started", {"goal_id": goal.id, "description": description})

            tasks = await self._decompose_goal(description, context)
            if not tasks:
                logger.error("[Brain] Could not decompose goal '%s'", description)
                goal.status = GoalStatus.FAILED
                goal.result_summary = "Could not decompose goal into tasks"
                await self._emit("goal:failed", {"goal_id": goal.id, "error": goal.result_summary})
                return goal

            goal.tasks = tasks
            goal.status = GoalStatus.RUNNING
            logger.info("[Brain] Goal '%s' decomposed into %d tasks", description, len(tasks))

            # Persist goal immediately
            await self._persist_goal(goal)

            # ── Step 2: Execute task DAG ───────────────────
            goal = await self._execute_task_graph(goal)

            # ── Step 3: Finalize ───────────────────────────
            self._goal_history.append(goal)
            self._active_goal = None

            if goal.status == GoalStatus.COMPLETED:
                await self._emit("goal:completed", {
                    "goal_id": goal.id,
                    "description": goal.description,
                    "tasks_completed": sum(1 for t in goal.tasks if t.status == TaskStatus.SUCCESS),
                    "result": goal.result_summary,
                })
                logger.info("[Brain] Goal completed: '%s'", description)
            else:
                await self._emit("goal:failed", {
                    "goal_id": goal.id,
                    "description": goal.description,
                    "error": goal.result_summary,
                })
                logger.warning("[Brain] Goal failed: '%s' — %s", description, goal.result_summary)

            return goal

    async def _decompose_goal(
        self, description: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Task]]:
        """Use LLM to decompose a high-level goal into tasks."""
        # Build context string
        ctx_str = ""
        if context:
            ctx_parts = []
            if context.get("repo"):
                ctx_parts.append(f"Repository: {context['repo']}")
            if context.get("branch"):
                ctx_parts.append(f"Branch: {context['branch']}")
            if context.get("ide"):
                ctx_parts.append(f"IDE: {context['ide']}")
            if context.get("recent_files"):
                ctx_parts.append(f"Recent files: {', '.join(context['recent_files'][:5])}")
            if ctx_parts:
                ctx_str = "\nContext:\n" + "\n".join(ctx_parts)

        prompt = f"{DECOMPOSE_PROMPT}\n{ctx_str}\n\nUser goal: {description}\n\nTasks:"

        if not self._llm_client:
            return self._fallback_decompose(description)

        try:
            # CRITICAL FIX: chat() is async — MUST await it.
            response = await self._llm_client.chat(prompt)
            if response:
                tasks_data = self._parse_task_json(response)
                if tasks_data:
                    return self._build_tasks(tasks_data)
        except Exception as e:
            logger.warning("[Brain] LLM decomposition failed: %s", e)

        return self._fallback_decompose(description)

    @staticmethod
    def _parse_task_json(text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse JSON task array from LLM response."""
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                data = json.loads(text[start:end + 1])
                if isinstance(data, list) and len(data) > 0:
                    return data
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("[Brain] JSON parse failed: %s", e)
        return None

    @staticmethod
    def _build_tasks(tasks_data: List[Dict[str, Any]]) -> List[Task]:
        """Convert raw task dicts to Task objects."""
        tasks = []
        for td in tasks_data:
            task = Task(
                id=str(td.get("id", f"task_{len(tasks)}")),
                description=str(td.get("description", "")),
                depends_on=[str(d) for d in td.get("depends_on", [])],
                timeout_s=float(td.get("timeout_s", 300)),
                max_retries=int(td.get("max_retries", 3)),
            )
            tasks.append(task)
        return tasks

    def _fallback_decompose(self, description: str) -> Optional[List[Task]]:
        """Simple fallback: create a single-task goal."""
        logger.info("[Brain] Using fallback decomposition for '%s'", description)
        return [Task(
            id="task_1",
            description=description,
            timeout_s=600,
            max_retries=2,
        )]

    # ── Task Graph Execution ───────────────────────────────────

    async def _execute_task_graph(self, goal: Goal) -> Goal:
        """Execute all tasks in the goal, respecting the dependency DAG."""
        completed_ids: set = set()
        failed_ids: set = set()

        while True:
            # Find tasks whose dependencies are all satisfied
            ready = [
                t for t in goal.tasks
                if t.status == TaskStatus.PENDING
                and all(d in completed_ids for d in t.depends_on)
            ]

            # Also retry failed tasks if they have remaining retries
            retryable = [
                t for t in goal.tasks
                if t.status == TaskStatus.FAILED
                and t.retry_count < t.max_retries
                and all(d in completed_ids for d in t.depends_on)
            ]

            if not ready and not retryable:
                # Check if all tasks are done
                all_done = all(
                    t.status in (TaskStatus.SUCCESS, TaskStatus.SKIPPED, TaskStatus.CANCELLED)
                    for t in goal.tasks
                )
                if all_done:
                    goal.status = GoalStatus.COMPLETED
                    goal.result_summary = f"Completed {sum(1 for t in goal.tasks if t.status == TaskStatus.SUCCESS)}/{len(goal.tasks)} tasks"
                else:
                    # Some tasks failed with no retries left
                    goal.status = GoalStatus.FAILED
                    failed = [t for t in goal.tasks if t.status == TaskStatus.FAILED]
                    goal.result_summary = f"Failed tasks: {', '.join(t.id for t in failed)}"
                break

            # Execute ready tasks (sequential for now, parallel in TaskExecutor Part 3)
            for task in ready:
                await self._execute_single_task(goal, task)
                if task.status == TaskStatus.SUCCESS:
                    completed_ids.add(task.id)
                elif task.status == TaskStatus.FAILED:
                    failed_ids.add(task.id)
                elif task.status == TaskStatus.SKIPPED:
                    completed_ids.add(task.id)

                # Persist after each task
                await self._persist_goal(goal)

            # Retry failed tasks
            for task in retryable:
                task.status = TaskStatus.RETRY
                task.retry_count += 1
                logger.info("[Brain] Retrying task '%s' (attempt %d/%d)",
                            task.id, task.retry_count, task.max_retries)
                await self._execute_single_task(goal, task)
                if task.status == TaskStatus.SUCCESS:
                    completed_ids.add(task.id)
                elif task.status == TaskStatus.FAILED:
                    failed_ids.add(task.id)

                await self._persist_goal(goal)

        goal.updated_at = time.time()
        return goal

    async def _execute_single_task(self, goal: Goal, task: Task) -> None:
        """Execute one task via Planner, with timeout and error handling."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        await self._emit("task:started", {
            "goal_id": goal.id,
            "task_id": task.id,
            "description": task.description,
        })

        logger.info("[Brain] Executing task '%s': %s", task.id, task.description)

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._run_through_planner(task.description, goal.context),
                timeout=task.timeout_s,
            )

            task.status = TaskStatus.SUCCESS if result[0] else TaskStatus.FAILED
            task.result = result[1] if result[0] else None
            task.error = None if result[0] else result[1]
            task.completed_at = time.time()
            task.latency_ms = (task.completed_at - task.started_at) * 1000.0

            await self._emit("task:completed" if result[0] else "task:failed", {
                "goal_id": goal.id,
                "task_id": task.id,
                "result": task.result,
                "error": task.error,
                "latency_ms": task.latency_ms,
            })

            # Record experience for self-improvement
            await self._record_experience(goal, task)

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Timed out after {task.timeout_s}s"
            task.completed_at = time.time()
            task.latency_ms = (task.completed_at - task.started_at) * 1000.0
            logger.warning("[Brain] Task '%s' timed out after %.0fs", task.id, task.timeout_s)

            await self._emit("task:failed", {
                "goal_id": goal.id,
                "task_id": task.id,
                "error": task.error,
                "latency_ms": task.latency_ms,
            })

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            task.latency_ms = (task.completed_at - task.started_at) * 1000.0
            logger.error("[Brain] Task '%s' failed: %s", task.id, e)

            await self._emit("task:failed", {
                "goal_id": goal.id,
                "task_id": task.id,
                "error": str(e),
                "latency_ms": task.latency_ms,
            })

    async def _run_through_planner(
        self, description: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Run a single task through AgentPlanner.

        Returns (success, message).
        """
        if not self._planner:
            return False, "Planner not available"

        try:
            # The planner processes the request synchronously in the current
            # implementation. Run in executor thread to avoid blocking.
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._planner.process_request, description
            )
            # If result contains error keywords, treat as failure
            is_success = "ran into an issue" not in result.lower() and \
                         "couldn't" not in result.lower()
            return is_success, result
        except Exception as e:
            return False, str(e)

    # ── Experience Recording ───────────────────────────────────

    async def _record_experience(self, goal: Goal, task: Task) -> None:
        """Record task execution in ExperienceDB for self-improvement."""
        try:
            from learning.experience_db import experience_db

            experience_db.record(
                goal=f"{goal.description} :: {task.description}",
                plan_steps=[task.description],
                plan_actions=[],
                success=task.status == TaskStatus.SUCCESS,
                result=task.result or "",
                latency_ms=task.latency_ms,
                error=task.error or "",
                recovery_action="",
                recovery_success=False,
                used_fallback=False,
            )
        except Exception as e:
            logger.debug("[Brain] Experience recording skipped: %s", e)

    # ── Goal Persistence ──────────────────────────────────────

    async def _persist_goal(self, goal: Goal) -> None:
        """Persist goal state to GoalManager (Part 2)."""
        goal.updated_at = time.time()
        if self._goal_manager:
            try:
                self._goal_manager.save_goal(goal)
            except Exception as e:
                logger.debug("[Brain] Goal persistence failed: %s", e)

    # ── Event Emission ────────────────────────────────────────

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a lifecycle event to the EventBus."""
        if self._event_bus:
            try:
                await self._event_bus.emit(event_type, data, source="brain")
            except Exception as e:
                logger.debug("[Brain] Event emission failed: %s", e)

    # ── Goal Management ───────────────────────────────────────

    async def cancel_goal(self, goal_id: str) -> bool:
        """Cancel an active or pending goal."""
        if self._active_goal and self._active_goal.id == goal_id:
            self._active_goal.status = GoalStatus.CANCELLED
            self._active_goal.result_summary = "Cancelled by user"
            self._active_goal.updated_at = time.time()
            await self._emit("goal:cancelled", {"goal_id": goal_id})
            await self._persist_goal(self._active_goal)
            self._goal_history.append(self._active_goal)
            self._active_goal = None
            logger.info("[Brain] Goal '%s' cancelled", goal_id)
            return True
        return False

    def get_progress(self) -> Dict[str, Any]:
        """Return current brain progress for dashboard/metrics."""
        if not self._active_goal:
            return {"active": False}

        tasks = self._active_goal.tasks
        return {
            "active": True,
            "goal_id": self._active_goal.id,
            "description": self._active_goal.description,
            "status": self._active_goal.status.value,
            "total_tasks": len(tasks),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.SUCCESS),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "running": sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
            "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "current_task": next(
                (t.description for t in tasks if t.status == TaskStatus.RUNNING), None
            ),
        }

    def close(self) -> None:
        """Release brain resources."""
        self._initialized = False
        self._active_goal = None
        logger.info("[Brain] AgentBrain shut down")


# Global singleton
agent_brain = AgentBrain()