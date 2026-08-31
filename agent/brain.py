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
    speak_immediately: bool = False     # Speak now, verify in background
    followup_response: str = ""         # Spoken after verification completes


# ═══════════════════════════════════════════════════════════════
# Decomposition Prompt
# ═══════════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = """You are Diego, an autonomous desktop AI agent. Decompose the user's goal into a sequence of concrete, executable tasks.

Each task must be a single action Diego can perform on the desktop. Tasks should be ordered logically, with dependencies noted.

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
            # CRITICAL FIX: initialize the perception pipeline so the
            # screen capture backend is actually started. Without this,
            # the screen_capture._backend stays "none" and every
            # perceive() call logs "Stage 1 — Screen capture failed".
            if not getattr(perception_pipeline, '_initialized', False):
                await perception_pipeline.initialize()
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

    # ── Per-stage profiling (Priority 7) ─────────────────────
    # Records latency for each pipeline stage so bottlenecks can be
    # identified. Reset at the start of each command.
    _pipeline_timings: Dict[str, float] = {}

    def _stage_start(self) -> float:
        return time.time()

    def _stage_end(self, stage: str, t_start: float) -> None:
        ms = (time.time() - t_start) * 1000
        if stage not in self._pipeline_timings:
            self._pipeline_timings[stage] = ms
        else:
            self._pipeline_timings[stage] += ms

    async def process_command(self, text: str,
                              stt_confidence: Optional[float] = None,
                              audio_duration_ms: Optional[float] = None) -> CommandResult:
        """
        Process a spoken command through the full execution pipeline.

        This is the ONLY entry point for command execution. The
        ConversationEngine calls this and speaks the response.

        Flow:
          0. Normalize: canonicalize the spoken command
          0.2 Transcript-quality guard (cheap rejection)
          0.25 Intent sanity gate (quality + speech evidence + confidence)
          1. Perceive: collect desktop context (DEMAND-DRIVEN)
          2. Decide: route the command (LLM last resort)
          3. Plan: create a plan if needed
          4. Dispatch: execute actions
          5. Verify: verify each action
          6. Learn: record outcomes
          7. Respond: return the response to speak

        Args:
            text: The user's spoken command.
            stt_confidence: Whisper avg_logprob for the transcript
                (None for typed/internal input paths).
            audio_duration_ms: Duration of the captured utterance audio.

        Returns:
            CommandResult with the response to speak.
        """
        if not self._initialized:
            await self.initialize()

        t0 = time.time()
        self._commands_processed += 1
        result = CommandResult()
        # Reset per-stage profiling
        self._pipeline_timings = {}

        # ── Step 0: Normalize (NEW) ──────────────────────────
        # Canonicalize the spoken command before any processing.
        # Handles app aliases, verb normalization, noise removal,
        # follow-up commands, music/search detection.
        from nlp.command_normalizer import command_normalizer
        normalized = command_normalizer.normalize(text)
        if normalized != text:
            logger.info("[Brain] Normalized: '%s' → '%s'", text[:50], normalized[:50])
        text = normalized

        # ── Record user turn in conversation memory ──────────
        # CRITICAL FIX: Without this, Diego has no memory of what the
        # user said. Pronoun resolution, follow-ups, and context
        # awareness all depend on conv_memory having the turns.
        from agent.conversation_memory import conv_memory
        conv_memory.add_user(text)
        # Track the user's goal for "continue" / "go back" support
        if any(kw in text.lower() for kw in ("open", "play", "search", "close",
                                              "start", "run", "create", "write")):
            conv_memory.track_goal(text)

        # ── Step 0.2: Transcript-quality guard (2026-08-30) ────
        # DEFENSE IN DEPTH: low-quality / hallucinated transcripts ("you",
        # "too") must NEVER reach the decision engine, perception (OCR can
        # take ~20s), the planner, or the LLM. The command listener already
        # rejects them, but any transcript that slips through from another
        # input path is rejected CHEAPLY here with a gentle recovery
        # response — no perception, no planner, no LLM, no actions.
        try:
            from voice.command_listener import is_low_quality_transcript
            if is_low_quality_transcript(text):
                logger.info("[Brain] Transcript rejected as low-quality: '%s' — "
                            "no perception, no planner, no LLM", text[:50])
                result.path = "REJECTED_TRANSCRIPT"
                result.used_llm = False
                result.response = ("I'm not sure I heard you correctly. "
                                   "Could you say that again?")
                conv_memory.add_assistant(result.response)
                result.latency_ms = (time.time() - t0) * 1000
                return result
        except ImportError:
            pass  # voice subsystem unavailable — never block command processing

        # ── Step 0.25: FINAL INTENT AUTHORIZATION BOUNDARY (2026-08-30) ──
        # The transcript must be classified into one of the eight
        # authorized intent categories BEFORE any expensive work:
        #   DETERMINISTIC_COMMAND / VISION_COMMAND / SEARCH_REQUEST /
        #   CONVERSATIONAL / KNOWLEDGE_QUESTION / FOLLOW_UP /
        #   MULTI_STEP_TASK / UNCERTAIN
        # ONLY actionable categories may reach the dispatcher/planner.
        # UNCERTAIN pays for NOTHING (no perception, no search, no
        # planner, no LLM, no tools) — it gets a clarification.
        # CONVERSATIONAL / KNOWLEDGE_QUESTION may reach the LLM for a
        # spoken answer but NEVER desktop actions.
        raw_transcript = text
        try:
            from nlp.intent_authorizer import (
                authorize_intent, IntentCategory,
            )
            auth = authorize_intent(
                text,
                stt_confidence=stt_confidence,
                audio_duration_ms=audio_duration_ms,
            )
        except Exception:
            auth = None  # authorizer unavailable — never block the pipeline

        tool_execution_allowed = True
        if auth is not None:
            logger.info(
                "[Brain] INTENT-AUTH raw=%r normalized=%r category=%s "
                "intent_conf=%.2f actionable=%s llm=%s route=%s (%s)",
                raw_transcript[:60], text[:60], auth.category.value,
                auth.confidence, auth.actionable, auth.llm_allowed,
                auth.route, auth.reason)
            if auth.category == IntentCategory.UNCERTAIN:
                # Requirement 6: rejected/uncertain transcripts must not
                # invoke perception, planner, search, LLM, or tools.
                response = ("I'm not sure I heard you correctly. "
                            "Could you say that again?")
                result.path = "CLARIFICATION"
                result.used_llm = False
                result.response = response
                conv_memory.add_assistant(response)
                result.latency_ms = (time.time() - t0) * 1000
                return result
            if auth.category == IntentCategory.CONVERSATIONAL:
                from agent.personality import personality as _auth_personality
                response = (_auth_personality.contextual_response(text)
                            or _auth_personality.greeting())
                if response:
                    result.path = "CONVERSATION"
                    result.used_llm = False
                    result.response = response
                    conv_memory.add_assistant(response)
                    result.latency_ms = (time.time() - t0) * 1000
                    return result
                # No canned response — fall through to the LLM path,
                # but the transcript is NOT actionable (no planner, no
                # dispatcher, no tools).
                tool_execution_allowed = False
            elif auth.category == IntentCategory.KNOWLEDGE_QUESTION:
                # Factual questions reach the LLM for an answer —
                # never desktop actions.
                tool_execution_allowed = False

        if tool_execution_allowed:
            # Defense in depth: the original intent gate still applies
            # to anything that proceeds (cheap, pure-Python).
            try:
                from nlp.intent_gate import evaluate_intent
                verdict = evaluate_intent(
                    text,
                    stt_confidence=stt_confidence,
                    audio_duration_ms=audio_duration_ms,
                )
            except Exception:
                verdict = None  # gate unavailable — never block the pipeline
            if verdict is not None and not verdict.tool_execution_allowed:
                from agent.personality import personality as _gate_personality
                if verdict.mode == "conversational":
                    response = (_gate_personality.contextual_response(text)
                                or _gate_personality.greeting())
                    result.path = "CONVERSATION"
                else:
                    response = ("I'm not sure I heard you correctly. "
                                "Could you say that again?")
                    result.path = "CLARIFICATION"
                logger.info("[Brain] Intent gate: %s (%s) — no perception, no "
                            "planner, no LLM, no tools: '%s'",
                            verdict.mode, verdict.reason, text[:50])
                result.used_llm = False
                result.response = response
                conv_memory.add_assistant(response)
                result.latency_ms = (time.time() - t0) * 1000
                return result
            self._last_intent_verdict = verdict
        self._last_intent_authorization = auth

        # ── Step 0.5: Conversation First (NEW) ────────────────
        # Before ANY planning or action, check if this is just
        # conversation. Greetings, thanks, how-are-you, corrections,
        # and simple acknowledgments should NEVER invoke the planner
        # or the LLM. Diego is a companion first, a tool second.
        from agent.personality import personality
        conversational = personality.contextual_response(text)
        if conversational:
            result.path = "CONVERSATION"
            result.used_llm = False
            result.response = conversational
            conv_memory.add_assistant(conversational)
            result.latency_ms = (time.time() - t0) * 1000
            return result

        # ── Step 1: Decide (FAST — no perception needed for routing) ──
        # OPTIMIZATION: Run the decision engine FIRST without perception
        # context. The decision engine is <1ms for deterministic routes
        # (simple commands, conversation, cached responses). Only LLM-bound
        # commands need screen context, so we skip the expensive perception
        # pipeline (screen capture + OCR + accessibility tree) for the
        # majority of commands. This saves 500ms-2s per simple command.
        perception_ctx = None
        decision = await self._decide(text, None)

        # ── Step 1b: Perceive (ONLY if the decision needs screen context) ──
        # Perception is only needed for:
        #   - LLM path (screen context injected into the prompt)
        #   - Vision path (read_screen action runs perception internally)
        # Simple desktop commands (open app, volume, brightness, etc.)
        # do NOT need screen context — skip the expensive pipeline.
        # ── Web research context (2026-08-30 hardening) ──
        # When the LLM path is taken AND the request needs current web
        # information, fetch REAL search results and inject them so the
        # LLM answers from facts, not stale training knowledge.
        web_ctx = ""
        if decision.needs_llm:
            try:
                from core.decision_engine import decision_engine as _de
                if _de._needs_search(text):
                    from services.search_service import search_service
                    try:
                        if not search_service.is_ready:
                            await search_service.start()
                    except Exception:
                        pass
                    web_ctx = await asyncio.wait_for(
                        search_service.context_for_llm(text, max_results=3),
                        timeout=20.0,
                    )
                    if web_ctx:
                        logger.info("[Brain] Web context fetched (%d chars)", len(web_ctx))
            except Exception as e:
                logger.debug("[Brain] Web context fetch skipped: %s", e)
            web_ctx = web_ctx or None
            # Stored for _generate_response() to inject into the LLM prompt.
            self._last_web_context = web_ctx

        if decision.needs_llm:
            # DEMAND-DRIVEN PERCEPTION (2026-08-30): OCR is only invoked
            # when the request actually needs to READ the screen. A
            # conversational utterance that falls through to the LLM path
            # ("tell me a joke") must NOT pay the OCR cost (PaddleOCR can
            # take 16-21s when it fails).
            # BLOCKER 3 FIX (2026-08-30): perception is now FULLY
            # demand-driven. Ordinary conversation and knowledge questions
            # that fall through to the LLM path ("tell me a joke", "what
            # is the capital of France") do NOT need screen context and
            # must NOT pay the ~5s perception cost. Perception runs only
            # for explicit vision requests and deixis/UI references.
            if self._perception_needed(text):
                perception_ctx = await self._perceive(text)
            else:
                logger.debug("[Brain] Perception skipped (no screen/deixis "
                             "cues): '%s'", text[:50])
            # Re-decide with perception context now available (enables
            # L6 vision-context reuse and L7 search-context paths).
            decision = await self._decide(text, perception_ctx, search_context=web_ctx)

        if decision.resolved:
            # ── Simple path: no LLM needed ──────────────────
            result.path = decision.path.value
            result.used_llm = False

            from agent.personality import personality

            # ── Speak immediately for actions (NEW) ───────────
            # Generate the immediate response BEFORE dispatching so the
            # engine can speak "Opening Firefox." while the action runs.
            # Verification happens in the background.
            immediate_response = ""
            if decision.action:
                # Presence check: if the app is already running, say so
                # instead of "Opening X." — Diego feels present.
                already = self._check_already_running(decision.action)
                if already:
                    result.response = already
                    result.speak_immediately = False
                else:
                    immediate_response = self._immediate_response(decision.action, personality)
                    if immediate_response:
                        result.response = immediate_response
                        result.speak_immediately = True

            # Execute single action
            if decision.action:
                ok, action_result = await self._dispatch_and_verify(decision.action)
                result.actions_executed = 1
                result.actions_succeeded = 1 if ok else 0
                result.actions_failed = 0 if ok else 1
                result.verified = ok
                # CRITICAL FIX (2026-08-23): For read_screen, the dispatcher
                # returns the actual screen content ("On screen: ..."). Use
                # that as the response so the user hears what's on screen
                # instead of a generic confirmation.
                # 2026-08-30: extended to ALL informational actions whose
                # dispatch result IS the answer (web research, window list).
                if (ok and action_result
                        and decision.action.get("action") in self._RESULT_AS_RESPONSE_ACTIONS):
                    result.response = action_result
                    result.speak_immediately = False

            # Execute multi-step workflow
            if decision.actions:
                for action in decision.actions:
                    ok, action_result = await self._dispatch_and_verify(action)
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
                    # All succeeded — use a natural confirmation.
                    # CRITICAL FIX: Use personality for variety instead of
                    # always "Done." — Diego should sound alive, not robotic.
                    # CRITICAL FIX (2026-08-23): If read_screen already set
                    # the response to the actual screen content, do NOT
                    # overwrite it with a generic confirmation.
                    if (result.response and decision.action
                            and decision.action.get("action") in self._RESULT_AS_RESPONSE_ACTIONS):
                        pass  # Keep the informational content as the response
                    else:
                        detail = self._action_detail(decision)
                        confirmation = personality.task_confirmation(detail) if detail else personality.acknowledgment()
                        if result.speak_immediately:
                            # Keep the immediate response as the primary,
                            # set the confirmation as the followup.
                            result.followup_response = confirmation
                        else:
                            result.response = confirmation
                else:
                    # At least one action failed — be honest about the failure.
                    # The decision's response ("Opening.") must NOT be spoken
                    # when the action did not actually open anything.
                    if result.speak_immediately:
                        result.followup_response = self._default_response(result)
                    else:
                        result.response = self._default_response(result)
            else:
                # No actions dispatched — conversational/cached responses are fine
                result.response = decision.response or self._default_response(result)

        else:
            # ── Complex path: needs LLM ─────────────────────
            result.path = "LLM"
            result.used_llm = True

            # Step 3: Plan (ONLY for actionable intents — requirement 6:
            # non-actionable transcripts must never invoke the planner).
            plan = (await self._plan(text, perception_ctx)
                    if tool_execution_allowed else None)
            if plan is None and not tool_execution_allowed:
                logger.info("[Brain] Planner SKIPPED — intent is not "
                            "actionable (LLM-only answer): '%s'", text[:50])

            # Steps 4-6: Dispatch → Verify → Learn
            # BLOCKER 1 FIX (2026-08-30): every planner-generated action is
            # gated. A conversational/uncertain transcript must never
            # become desktop actions, and a type_text action that merely
            # echoes the transcript (planner hallucination, e.g.
            # type_text("Hello dear")) is blocked.
            if plan:
                for step in plan:
                    action = self._step_to_action(step)
                    if action and self._planner_action_allowed(text, action):
                        ok, _ = await self._dispatch_and_verify(action)
                        result.actions_executed += 1
                        if ok:
                            result.actions_succeeded += 1
                        else:
                            result.actions_failed += 1
                    elif action:
                        logger.info("[Brain] Planner action BLOCKED by intent "
                                    "gate: %s (transcript='%s')",
                                    action.get("action"), text[:50])
                result.verified = result.actions_failed == 0

            # Step 7: Respond (LLM generates the response)
            result.response = await self._generate_response(text, perception_ctx, result)

        # ── Record assistant turn in conversation memory ──────
        # CRITICAL FIX: Diego must remember what it said so follow-ups
        # like "what was that" / "say again" work.
        if result.response:
            conv_memory.add_assistant(result.response)

        result.latency_ms = (time.time() - t0) * 1000
        logger.info("[Brain] Command processed: '%s' path=%s actions=%d verified=%s latency=%.0fms",
                     text[:50], result.path, result.actions_executed, result.verified, result.latency_ms)
        return result

    # ── Pipeline steps ─────────────────────────────────────────

    async def _perceive(self, text: str = "",
                        include_ocr: Optional[bool] = None) -> Optional[Any]:
        """Step 1: Perceive desktop context.

        DEMAND-DRIVEN OCR (2026-08-30): OCR (PaddleOCR — up to ~20s when it
        fails) runs ONLY when the request needs to READ the screen
        ("what is on my screen?", "read this error"). Everything else —
        "open firefox", "how are you?", "tell me a joke" — gets window /
        accessibility context only and never invokes OCR.
        """
        if not self._perception:
            return None
        if include_ocr is None:
            include_ocr = self._ocr_required(text)
        try:
            # BLOCKER 3 FIX (2026-08-30): hard upper bound on the WHOLE
            # perception stage. OCR is internally bounded (OCR_TIMEOUT_S),
            # but capture / a11y / window enumeration can also stall. A
            # perception stall must never block the turn for tens of
            # seconds — on timeout we continue WITHOUT screen context.
            ctx = await asyncio.wait_for(
                self._perception.perceive(include_ocr=include_ocr),
                timeout=self.PERCEPTION_TIMEOUT_S,
            )
            logger.debug("[Brain] Perception: window='%s' a11y=%s ocr=%s",
                         getattr(ctx, 'window_title', '')[:40],
                         getattr(ctx, 'a11y_available', False),
                         getattr(ctx, 'ocr_used', False))
            return ctx
        except asyncio.TimeoutError:
            logger.warning("[Brain] Perception timed out after %.0fs — "
                           "continuing without screen context",
                           self.PERCEPTION_TIMEOUT_S)
            return None
        except Exception as e:
            logger.debug("[Brain] Perception failed: %s", e)
            return None

    # Perception hard budget (seconds). OCR itself is bounded at 10s
    # inside the pipeline; this is the total per-turn ceiling so a
    # stalled capture/a11y stage can never block the turn for tens of
    # seconds.
    PERCEPTION_TIMEOUT_S = 15.0

    # Deixis / UI-reference cues: when the request refers to "this",
    # "that", "here", the screen, or a UI element, the LLM needs to SEE
    # the desktop. Pure conversation / knowledge questions do not.
    _PERCEPTION_CUES = frozenset({
        "this", "that", "these", "those", "here", "there", "it",
        "screen", "display", "monitor", "window", "tab", "button",
        "dialog", "page", "click", "select", "highlight", "current",
        "visible", "active", "open", "close", "type", "write", "click",
    })

    @classmethod
    def _perception_needed(cls, text: str) -> bool:
        """True only when the request plausibly needs screen context.

        Explicit vision requests always need perception. Otherwise,
        perception runs only when the transcript references the screen
        or uses deixis ("click this", "close it", "what is that?").
        Ordinary conversation and knowledge questions skip perception.
        """
        if cls._ocr_required(text or ""):
            return True
        words = {
            w.strip(".,!?;:'\"")
            for w in (text or "").lower().split()
        }
        return bool(words & cls._PERCEPTION_CUES)

    @staticmethod
    def _ocr_required(text: str) -> bool:
        """True only when the request needs to actually READ screen pixels.

        Explicit vision phrases ("what is on my screen?", "read this
        error") require OCR. Everything else — "open firefox", "how are
        you?", "open chrome" — must NOT pay the OCR cost.
        """
        try:
            from core.decision_engine import DecisionEngine
            if DecisionEngine._needs_vision(text or ""):
                return True
        except Exception:
            pass
        t = " ".join((text or "").lower().split())
        screen_mentions = (
            "screen", "display", "monitor", "this error", "the error",
            "this page", "this window", "this dialog", "what do you see",
        )
        return any(m in t for m in screen_mentions)

    # Actions whose dispatch result IS the user-facing answer
    # (not just a confirmation). The Brain speaks the result verbatim.
    _RESULT_AS_RESPONSE_ACTIONS = {
        "read_screen",
        "web_search",
        "web_search_open_best",
        "list_windows",
        "music_status",
    }

    async def _decide(self, text: str, perception_ctx: Optional[Any],
                      search_context: Optional[str] = None) -> Any:
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
                search_context=search_context,
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

    async def _dispatch_and_verify(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Steps 4-6: Dispatch → Verify → Learn.

        This is the ONLY place actions are dispatched, verified, and
        recorded for learning.

        ORDER INVARIANT (CRITICAL):
          1. Capture pre-action screen state (BEFORE execution)
          2. Execute the action (dispatch)
          3. Verify AFTER execution by comparing pre/post screen state
          4. Retry on failure (up to MAX_ACTION_RETRIES)
          5. Learn the outcome

        RETRY POLICY (CRITICAL FIX):
          Previously this dispatched exactly once and never retried.
          If the first attempt failed (e.g. app not found, browser
          not ready), the action was reported as failed even though
          a retry with adjusted params would have succeeded. Now we
          retry up to MAX_ACTION_RETRIES with parameter adjustment.

        CRITICAL FIX (2026-08-23): Returns (bool, result_str) so the
        dispatcher's actual result text (e.g. read_screen's "On screen: ...")
        is preserved. Previously only a bool was returned, so the screen
        content was discarded and the user heard a generic confirmation
        instead of what was actually on screen.
        """
        if not self._dispatcher:
            logger.warning("[Brain] No dispatcher — cannot execute action")
            return False, ""

        action_name = action.get("action", "")
        params = action.get("params", {}) or {}

        MAX_ACTION_RETRIES = 2

        for attempt in range(MAX_ACTION_RETRIES + 1):
            # ── Step 4a: Capture PRE-action state (BEFORE dispatch) ──
            # CRITICAL FIX: without a pre-action snapshot, the verifier
            # cannot detect ANY screen change and always reports NO_CHANGE.
            # This made every action fail verification even when it succeeded.
            if self._verifier:
                try:
                    self._verifier.capture_pre_action()
                    logger.debug("[Brain] Pre-action capture for %s (attempt %d)",
                                 action_name, attempt + 1)
                except Exception as e:
                    logger.debug("[Brain] Pre-action capture failed: %s", e)

            # ── Step 4b: Dispatch (execute) ───────────────────
            try:
                result = await self._dispatcher.execute(action)
                self._actions_dispatched += 1
            except Exception as e:
                logger.warning("[Brain] Dispatch failed for %s (attempt %d): %s",
                               action_name, attempt + 1, e)
                if attempt < MAX_ACTION_RETRIES:
                    action = self._adjust_params_for_retry(action_name, action)
                    await asyncio.sleep(0.5)
                    continue
                self._actions_failed += 1
                await self._learn(action_name, params, success=False, error=str(e))
                return False, ""

            # ── Step 5: Verify ──────────────────────────────────
            verified = await self._verify(action_name, params, result)
            self._actions_verified += 1

            if verified:
                # ── Step 6: Learn ───────────────────────────────
                await self._learn(action_name, params, success=True)
                return True, result or ""

            # Verification failed — retry with adjusted params
            logger.warning("[Brain] Action %s failed verification (attempt %d)",
                           action_name, attempt + 1)
            if attempt < MAX_ACTION_RETRIES:
                action = self._adjust_params_for_retry(action_name, action)
                await asyncio.sleep(0.5)
                continue

            self._actions_failed += 1
            await self._learn(action_name, params, success=False, error="Verification failed")
            return False, result or ""

        return False, ""

    @staticmethod
    def _adjust_params_for_retry(action_name: str,
                                  action: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust action params for a retry attempt."""
        adjusted = dict(action)
        params = dict(action.get("params", {}) or {})

        if action_name == "desktop_open":
            app = params.get("app", "")
            alt_map = {
                "code": "code-insiders",
                "vscode": "code",
                "vs code": "code",
                "firefox": "firefox-esr",
                "chrome": "chromium-browser",
                "google-chrome": "chromium",
                "gnome-terminal": "xterm",
                "terminal": "xterm",
                "nautilus": "thunar",
                "files": "thunar",
                "file manager": "thunar",
            }
            if app.lower() in alt_map:
                params["app"] = alt_map[app.lower()]

        if action_name == "browser_navigate":
            url = params.get("url", "")
            if url.startswith("https://"):
                params["url"] = url.replace("https://", "http://")

        adjusted["params"] = params
        return adjusted

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
        # CRITICAL FIX (audit B3): music/media actions have no OS or
        # vision verifier — the dispatch result string is the ONLY
        # evidence. Failure strings like "Nothing to resume." (playerctl
        # exited non-zero, no player responded) must fail verification,
        # not be trusted as success.
        if result is not None:
            result_str = str(result)
            if "Couldn't" in result_str or self._is_dispatch_failure(result_str):
                logger.debug("[Brain] Verify FAIL (dispatcher reported): %s",
                             result_str[:80])
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
                    # ── SETTLE-WAIT (CRITICAL FIX) ──
                    # Desktop apps take 1-3 s to spawn after dispatch.
                    # Checking pgrep immediately after dispatch finds
                    # nothing, falsely failing the launch. Poll for up
                    # to ~1.5s before declaring failure.
                    # OPTIMIZATION: Reduced from 3.0s to 1.5s. Most desktop
                    # apps spawn within 1-1.5s; the extra 1.5s of polling
                    # added unnecessary latency to every app-open command.
                    settle_deadline = time.time() + 1.5
                    while time.time() < settle_deadline:
                        # CRITICAL FIX (2026-08-29): Use `pgrep -x` (exact
                        # process name) instead of `pgrep -f` (full command
                        # line). `pgrep -f` can match Diego's own process tree
                        # or wrapper shells, falsely verifying success.
                        chk = subprocess.run(
                            ["pgrep", "-x", proc],
                            capture_output=True, text=True, timeout=3,
                        )
                        if chk.returncode == 0:
                            logger.info("[Brain] Verify OK: process '%s' running "
                                        "(after %.1fs settle)", proc,
                                        time.time() - (settle_deadline - 1.5))
                            return True
                        # Also try alternate binaries
                        for alt in (proc.replace("-", ""), f"{proc}-esr", f"{proc}-stable"):
                            chk2 = subprocess.run(
                                ["pgrep", "-x", alt],
                                capture_output=True, text=True, timeout=3,
                            )
                            if chk2.returncode == 0:
                                logger.info("[Brain] Verify OK: process '%s' running",
                                            alt)
                                return True
                        time.sleep(0.3)
                    logger.warning("[Brain] Verify FAIL: process '%s' did not "
                                   "appear within 1.5s settle-wait", proc)

                elif action_name in ("browser_navigate", "browser_search"):
                    chk = subprocess.run(
                        ["pgrep", "-f", "firefox|chrome|chromium|brave"],
                        capture_output=True, text=True, timeout=3,
                    )
                    if chk.returncode == 0:
                        logger.info("[Brain] Verify OK: browser process running")
                        return True

                elif action_name == "close_app":
                    # CRITICAL FIX: verify the app process is GONE (not running).
                    # Previously close_app fell through to vision verification
                    # with the "open_app" type, which checked for a window
                    # APPEARING — the exact opposite of what close should do.
                    app = str(params.get("app", "")).lower()
                    proc_map = {
                        "vs code": "code", "vscode": "code", "code": "code",
                        "browser": "firefox", "firefox": "firefox",
                        "chrome": "chrome", "google-chrome": "chrome",
                        "spotify": "spotify", "terminal": "xterm",
                        "gnome-terminal": "gnome-terminal", "xterm": "xterm",
                        "konsole": "konsole", "alacritty": "alacritty",
                        "kitty": "kitty", "wezterm": "wezterm", "tilix": "tilix",
                        "files": "nautilus", "nautilus": "nautilus",
                        "calculator": "gnome-calculator",
                        "settings": "gnome-control-center", "slack": "slack",
                        "discord": "discord", "telegram": "telegram-desktop",
                        "notion": "notion-app", "pycharm": "pycharm",
                    }
                    proc = proc_map.get(app, app)
                    # Use -x (exact name) NOT -f (full cmdline) to avoid
                    # matching wrapper shells / the invoking process.
                    chk = subprocess.run(
                        ["pgrep", "-x", proc],
                        capture_output=True, text=True, timeout=3,
                    )
                    if chk.returncode != 0:
                        logger.info("[Brain] Verify OK: process '%s' is gone (closed)", proc)
                        return True
                    # pgrep matched — but check for ZOMBIES. A zombie (state
                    # 'Z') is already dead; its exit status just hasn't been
                    # reaped by its parent yet. Treat zombies as "closed".
                    live_pids = []
                    for pid_str in chk.stdout.split():
                        pid_str = pid_str.strip()
                        if not pid_str.isdigit():
                            continue
                        pid = int(pid_str)
                        try:
                            with open(f"/proc/{pid}/stat") as f:
                                state = f.read().split()[2]
                            if state != "Z":
                                live_pids.append(pid)
                        except (FileNotFoundError, ProcessLookupError, IndexError):
                            continue
                    if not live_pids:
                        logger.info("[Brain] Verify OK: process '%s' is gone (closed, only zombies remain)", proc)
                        return True
                    # Live process still running — close failed
                    logger.warning("[Brain] Verify FAIL: process '%s' still running after close (pids=%s)", proc, live_pids)
                    return False
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
                    # CRITICAL FIX (Task-failure root cause):
                    # When the vision subsystem ITSELF errors (screen
                    # capture unavailable, vision_service down, no ASR
                    # snapshot), the action did NOT necessarily fail.
                    # Previously an ERROR status was treated as a FAIL
                    # → the user was told "couldn't do it" even though
                    # the app launched fine. Vision ERROR now falls
                    # through to "trust dispatch result".
                    from vision.action_verifier import VerificationStatus
                    if vresult.status == VerificationStatus.ERROR:
                        logger.warning("[Brain] Verify ERROR (vision subsystem) — "
                                       "trusting dispatch result for %s: %s",
                                       action_name, vresult.explanation[:100])
                        return self._trust_dispatch_result(result)

                    if vresult.success:
                        logger.info("[Brain] Verify OK: %s (%s)", action_name, vresult.status.value)
                        return True
                    logger.warning("[Brain] Verify FAIL: %s — %s",
                                   action_name, vresult.explanation[:100])
                    # CRITICAL FIX: For UI actions (click, type, scroll,
                    # key_press), vision verification is AUTHORITATIVE.
                    # If the screen did not change, the action did not
                    # have its expected effect. Previously this fell
                    # through to "trust dispatch result" which made
                    # verification failures invisible — the user was
                    # told "Done." even when nothing happened.
                    # Exception: for desktop_app / browser actions, the
                    # OS-level process check above is authoritative and
                    # already returned True if the process is running.
                    # If we reach here, the OS check did NOT pass, so
                    # vision NO_CHANGE means the action truly failed.
                    return False
                else:
                    # No verification type mapped — trust the dispatch result
                    # (but only if it does not report a failure — audit B3)
                    return self._trust_dispatch_result(result)
            except Exception as e:
                logger.debug("[Brain] Vision verify failed: %s", e)

        # ── Fallback: trust a clean dispatch result ──
        return self._trust_dispatch_result(result)

    @staticmethod
    def _is_dispatch_failure(result: str) -> bool:
        """True if a dispatch result string reports a real failure.

        CRITICAL FIX (audit B3): media/music actions have no OS-level or
        vision verifier, so the dispatch string is the only evidence of
        what happened. MusicAgent now reports honest failure strings when
        playerctl exits non-zero (no player responded). Those strings —
        and other known no-op results — must NOT be treated as success.
        """
        if not result:
            return False
        markers = (
            "Nothing to resume",
            "Nothing to stop",
            "Nothing is playing",
            "no player responded",
            "isn't available right now",
            "not available",
            "MPV is not installed",
            "No local music found",
        )
        return any(marker in result for marker in markers)

    @staticmethod
    def _trust_dispatch_result(result: Optional[str]) -> bool:
        """Trust a dispatch result ONLY if it does not report failure."""
        if result is None:
            return False
        result_str = str(result)
        return ("Couldn't" not in result_str
                and not AgentBrain._is_dispatch_failure(result_str))

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
                from agent.personality import personality
                return personality.task_confirmation()
            return f"I ran into an issue with {result.actions_failed} of the steps."

        # Otherwise, use the LLM to generate a response.
        # CRITICAL FIX: inject the perception context (what Diego sees on
        # screen) so "what is going on" / "click here" actually work.
        # Without this, the LLM has no idea what's on the screen.
        try:
            from agent.streaming_llm import streaming_llm
            screen_ctx = ""
            if perception_ctx is not None:
                try:
                    screen_ctx = perception_ctx.compact_summary
                except Exception:
                    screen_ctx = ""
            # 2026-08-30: web_context is passed via the `web_context`
            # attribute set during process_command (search grounding).
            web_ctx = getattr(self, "_last_web_context", None)
            sentences = []
            async for sentence in streaming_llm.generate(
                    text, screen_context=screen_ctx, web_context=web_ctx):
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
            # CRITICAL FIX: close_app must NOT map to "open_app" — that
            # checks for a window APPEARING, the exact opposite of what
            # closing should do. close_app is verified at the OS level
            # (process gone) in _verify(), so it needs no vision mapping.
            "browser_navigate": "navigate",
            "browser_search": "navigate",
            "click_text": "click",
            "scroll": "scroll",
            "type_text": "type",
            "key_press": "key_press",
        }
        return mapping.get(action_name)

    # ── Planner action schema (2026-08-30) ────────────────────────────
    # Every planner-generated action must be in this schema AND the
    # transcript must contain explicit verb evidence for it. This kills
    # hallucinated actions like "Diego opened the tomb" -> close_app
    # (no "close" evidence in the transcript) and "Hello dear" ->
    # type_text + get_time.
    _PLANNER_ACTION_SCHEMA = {
        "desktop_open": ("open", "launch", "start", "run", "bring up",
                         "pull up", "load", "switch to"),
        "close_app": ("close", "quit", "exit", "kill", "shut down",
                      "shut", "turn off"),
        "browser_navigate": ("open", "go to", "navigate", "visit",
                             "browse", "switch to", "take me to"),
        "browser_search": ("search", "google", "look up", "find"),
        "web_search": ("search", "google", "look up", "find",
                       "what is", "who is", "how to", "tell me about",
                       "weather", "news"),
        "web_search_open_best": ("search", "google", "look up", "find",
                                 "open"),
        "youtube_search": ("search", "find", "youtube", "play"),
        "play_media": ("play", "put on", "start playing", "resume",
                       "music", "song", "track"),
        "click_text": ("click", "press", "tap", "select"),
        "scroll": ("scroll",),
        "key_press": ("press", "hit", "type", "enter"),
        "type_text": ("type", "write", "enter", "input"),
        "open_folder": ("open", "show", "folder", "files"),
        "volume_up": ("volume", "louder", "increase", "turn up"),
        "volume_down": ("volume", "quieter", "decrease", "lower",
                        "turn down"),
        "volume_set": ("volume", "set"),
        "volume_mute": ("mute", "volume"),
        "brightness_up": ("brightness", "brighter", "increase"),
        "brightness_down": ("brightness", "dimmer", "decrease", "lower"),
        "brightness_set": ("brightness", "set"),
        "lock_screen": ("lock",),
        "shutdown": ("shut down", "shutdown", "turn off", "power off"),
        "restart": ("restart", "reboot"),
        "get_time": ("time",),
        "get_date": ("date", "day", "today"),
        "minimize_window": ("minimize",),
        "maximize_window": ("maximize",),
        "switch_workspace": ("switch", "workspace"),
        "switch_workspace_prev": ("switch", "workspace", "previous",
                                  "back"),
        "switch_window": ("switch", "window"),
        "switch_window_prev": ("switch", "window", "previous", "back"),
        "switch_tab": ("switch", "tab"),
        "switch_tab_prev": ("switch", "tab", "previous", "back"),
        "read_screen": ("screen", "see", "read", "display", "monitor",
                        "looking at", "error", "button", "click"),
        "list_windows": ("what", "running", "open", "windows", "apps",
                         "list"),
        "music_status": ("music", "playing", "song", "track", "status"),
        "music_pause": ("pause", "stop", "music", "song", "track"),
        "music_resume": ("resume", "continue", "music", "play"),
        "music_next": ("next", "skip", "track", "song"),
        "music_previous": ("previous", "back", "track", "song"),
        "music_stop": ("stop", "music", "song", "track"),
        "music_shuffle": ("shuffle", "music"),
        "music_repeat": ("repeat", "music"),
        "music_volume": ("volume", "music"),
        "music_mute": ("mute", "music"),
        "screenshot": ("screenshot", "capture", "picture of the screen"),
    }

    @staticmethod
    def _planner_action_allowed(transcript: str,
                                action: Dict[str, Any]) -> bool:
        """BLOCKER 1 FIX (2026-08-30, hardened): gate a planner-generated
        action against the requested intent, the allowed action schema,
        the extracted entities, and the explicit user request.

        Guards:
          0. The action name must be in the allowed action schema AND
             the transcript must contain explicit verb evidence for it
             ("Diego opened the tomb" -> close_app is blocked: no
             "close" evidence in the transcript).
          1. The transcript itself must pass the intent gate (a
             conversational/uncertain transcript must never become
             desktop actions).
          2. A type_text action whose payload merely echoes the
             transcript is a planner hallucination ("Hello dear" ->
             type_text("Hello dear")) and is blocked unless the user
             explicitly asked to type.
        """
        name = action.get("action", "")

        # Guard 0: action schema + verb evidence.
        evidence = AgentBrain._PLANNER_ACTION_SCHEMA.get(name)
        if evidence is None:
            logger.info("[Brain] Planner action BLOCKED: '%s' is not in "
                        "the allowed action schema", name)
            return False
        tnorm = " ".join((transcript or "").lower().strip(" .!?").split())
        if not any(ev in tnorm for ev in evidence):
            logger.info("[Brain] Planner action BLOCKED: '%s' has no verb "
                        "evidence in transcript '%s'", name, tnorm[:60])
            return False

        # Guard 1: intent gate.
        try:
            from nlp.intent_gate import transcript_allows_tool_execution
            if not transcript_allows_tool_execution(transcript):
                return False
        except Exception:
            pass  # gate unavailable — never block the pipeline

        # Guard 2: type_text echo of the transcript.
        if name == "type_text":
            payload = str((action.get("params") or {}).get("text", "")).strip()
            pnorm = " ".join(payload.lower().split())
            explicit = tnorm.startswith(("type ", "write ", "enter ", "input "))
            if pnorm and not explicit and (
                    pnorm == tnorm or (len(pnorm) > 3 and pnorm in tnorm)):
                return False
        return True

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
        from agent.personality import personality
        if result.actions_executed == 0:
            return personality.acknowledgment()
        if result.actions_failed == 0:
            return personality.task_confirmation()
        return f"Mostly done, but {result.actions_failed} step(s) had issues."

    @staticmethod
    def _check_already_running(action: Dict[str, Any]) -> str:
        """Check if an app is already running and return a natural response.

        Returns "" if the app is not running (proceed with opening).
        Returns a natural "already open" response if it is.
        """
        try:
            name = action.get("action", "")
            if name != "desktop_open":
                return ""
            app = str(action.get("params", {}).get("app", "")).lower()
            if not app:
                return ""
            import shutil
            import subprocess
            if not shutil.which("pgrep"):
                return ""
            proc_map = {
                "code": "code", "vscode": "code", "vs code": "code",
                "firefox": "firefox", "browser": "firefox",
                "chrome": "chrome", "google-chrome": "chrome",
                "spotify": "spotify", "gnome-terminal": "gnome-terminal",
                "terminal": "gnome-terminal", "nautilus": "nautilus",
                "files": "nautilus", "slack": "slack",
                "discord": "discord", "telegram-desktop": "telegram",
                "notion-app": "notion", "pycharm": "pycharm",
            }
            proc = proc_map.get(app, app)
            # CRITICAL FIX (2026-08-29): Use `pgrep -x` (exact process name)
            # instead of `pgrep -f` (full command line). `pgrep -f` can match
            # Diego's own process tree or wrapper shells whose cmdline contains
            # the app name, falsely reporting the app as already running.
            chk = subprocess.run(
                ["pgrep", "-x", proc],
                capture_output=True, text=True, timeout=2,
            )
            if chk.returncode == 0:
                from agent.personality import personality
                return personality.already_running(app)
        except Exception:
            pass
        return ""

    @staticmethod
    def _immediate_response(action: Dict[str, Any], personality) -> str:
        """Generate an immediate spoken response for an action.

        Speaks BEFORE the action executes so the user hears
        "Opening Firefox." while Firefox launches.
        """
        try:
            name = action.get("action", "")
            params = action.get("params", {}) or {}
            if name == "desktop_open":
                app = params.get("app", "")
                return personality.opening(app) if app else ""
            if name == "browser_navigate":
                url = params.get("url", "")
                # Extract a friendly name from the URL
                if url:
                    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                    domain = domain.replace("www.", "")
                    return personality.opening(domain) if domain else ""
            if name == "browser_search":
                q = params.get("query", "")
                return personality.searching(q) if q else ""
            if name == "play_media":
                q = params.get("query", "")
                return personality.playing(q) if q else ""
            if name == "close_app":
                app = params.get("app", "")
                return personality.closed(app) if app else ""
            if name == "volume_up":
                return personality.volume_up()
            if name == "volume_down":
                return personality.volume_down()
            if name == "volume_mute":
                return personality.volume_mute()
            if name == "volume_set":
                return personality.volume_set(params.get("percent", 50))
            if name == "brightness_up":
                return personality.brightness_up()
            if name == "brightness_down":
                return personality.brightness_down()
            if name == "brightness_set":
                return personality.brightness_set(params.get("percent", 70))
            if name == "lock_screen":
                return personality.locked()
            if name == "music_pause":
                return personality.paused()
            if name == "music_resume":
                return personality.resumed()
            if name == "music_next":
                return personality.next_track()
            if name == "music_previous":
                return personality.previous_track()
            if name == "music_shuffle":
                return personality.shuffle()
            if name == "music_repeat":
                return personality.repeat()
        except Exception:
            pass
        return ""

    @staticmethod
    def _action_detail(decision) -> str:
        """Extract a natural detail string from a decision for confirmations."""
        try:
            if decision.action:
                action = decision.action.get("action", "")
                params = decision.action.get("params", {}) or {}
                if action == "desktop_open":
                    app = params.get("app", "")
                    return f"{app} is open" if app else ""
                if action == "browser_navigate":
                    url = params.get("url", "")
                    return f"opened {url}" if url else ""
                if action == "browser_search":
                    q = params.get("query", "")
                    return f"searched for {q}" if q else ""
                if action == "play_media":
                    q = params.get("query", "")
                    return f"playing {q}" if q else ""
                if action == "close_app":
                    app = params.get("app", "")
                    return f"closed {app}" if app else ""
                if action == "volume_up":
                    return "volume up"
                if action == "volume_down":
                    return "volume down"
                if action == "volume_mute":
                    return "muted"
                if action == "volume_set":
                    return f"volume at {params.get('percent', '')} percent"
                if action == "brightness_up":
                    return "brightness up"
                if action == "brightness_down":
                    return "brightness down"
                if action == "brightness_set":
                    return f"brightness at {params.get('percent', '')} percent"
                if action == "lock_screen":
                    return "screen locked"
                if action == "music_pause":
                    return "paused"
                if action == "music_resume":
                    return "resumed"
                if action == "music_next":
                    return "next track"
                if action == "music_previous":
                    return "previous track"
            if decision.actions:
                return f"{len(decision.actions)} steps done"
        except Exception:
            pass
        return ""

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