"""
GoalRuntime — general autonomous desktop goal runtime for Diego.

An ADDITIVE layer on top of the existing architecture. It composes — and
duplicates none of — the existing capabilities:

    wake/auth/session  → existing runtime + auth modules (untouched)
    ASR / VAD / TTS    → voice.* (untouched)
    browser agent      → agent.browser_goal_engine + computer.browser_controller
    desktop goals      → agent.desktop_goal_engine (still authoritative for
                         its goals; GoalRuntime handles everything else)
    computer use       → computer.computer_controller (the ACT boundary)
    perception         → computer.element_finder + vision.ocr_pipeline
    verification       → per-skill verifiers + existing goal verification
    task continuation  → agent.task_continuation vocabulary (reused)
    confirmation flows → goalruntime.permissions.PermissionManager

The runtime loop:

    PLAN → EXECUTE → OBSERVE → VERIFY → REPLAN (bounded) → COMPLETED

Goals are decomposed into subgoals; each subgoal is dispatched to a generic
skill (browser, visual UI, telegram, gmail, filesystem, coding, terminal,
security, vision). Artifacts produced by one subgoal flow into later ones,
enabling cross-application workflows.

Model policy (provider-agnostic — see goalruntime.llm):
    PLANNER  qwen2.5:7b            general planner/reasoner
    VISION   qwen2.5-vl:3b         visual perception
    CODER    deepseek-coder-v2:16b coding tasks
    LIGHT    qwen2.5:3b            lightweight decisions

Deterministic routing ALWAYS runs before any LLM reasoning.
"""

from goalruntime.models import (
    Artifact,
    GoalStatus,
    Observation,
    PermissionClass,
    SkillResult,
    Subgoal,
)
from goalruntime.permissions import (
    PermissionDecision,
    ScopedPermissionManager,
)
from goalruntime.runtime import GoalRuntime
from goalruntime.session import AutonomousSession, SessionState

__all__ = [
    "Artifact", "GoalStatus", "Observation", "PermissionClass",
    "SkillResult", "Subgoal",
    "PermissionDecision", "ScopedPermissionManager",
    "GoalRuntime",
    "AutonomousSession", "SessionState",
]
