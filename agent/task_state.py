"""
TaskState — Closed-loop task execution state machine for Diego.

This module turns Diego's one-shot "plan → execute → respond" flow into a
reliable CLOSED-LOOP TASK AGENT:

    USER GOAL
      → understand goal
      → create plan
      → execute STEP 1
      → observe real state
      → verify STEP 1
      → update task state
      → execute STEP 2 …
      → if failure: recover / re-plan from CURRENT state
      → repeat until GOAL COMPLETE
      → only then respond

"Recursive" here means ITERATIVE closed-loop planning/re-planning — never
uncontrolled recursion. Hard limits (MAX_TASK_STEPS, MAX_RETRIES_PER_STEP,
MAX_REPLANS, MAX_TOTAL_EXECUTION_TIME) and loop detection guarantee
termination.

Design constraints (respected, not redesigned):
  - The Brain remains the orchestrator; this module is a library the Brain
    drives. Execution still flows through the ActionDispatcher and the
    ActionVerifier via injected callables.
  - Tool/verification results are AUTHORITATIVE. The LLM never fabricates
    execution, state, or completion.
  - Task state survives across conversational follow-ups ("continue",
    "open the first result", "do the same for Chrome", "close that").

Logging contract (every task logs):
    [TASK] id=abc goal="..."
    [PLAN] v1 steps=4
    [STEP 1/4] open_firefox
    [VERIFY] success
    [REPLAN] state changed, remaining=2
    [TASK] COMPLETE
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Final states
# ═══════════════════════════════════════════════════════════════

class FinalStatus(str, Enum):
    """Explicit task final states. A task is SUCCESS only when the goal
    criteria are satisfied AND required steps are verified."""
    SUCCESS = "SUCCESS"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    NEEDS_INPUT = "NEEDS_INPUT"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"  # sensitive action requires user approval


# ═══════════════════════════════════════════════════════════════
# Human safety — sensitive action detection
# ═══════════════════════════════════════════════════════════════

# Actions that are DESTRUCTIVE, IRREVERSIBLE, or SECURITY-SENSITIVE.
# These require explicit user confirmation before execution.
# Read-only investigation is always allowed.
SENSITIVE_ACTIONS = frozenset({
    # Destructive / irreversible
    "shutdown", "restart", "delete_file", "delete_folder", "format_disk",
    "rm", "rmdir", "shred", "wipe",
    # Credential / security related
    "change_password", "reset_password", "delete_credentials",
    "modify_ssh_keys", "modify_gpg_keys", "clear_keyring",
    # Financial
    "make_payment", "transfer_money", "submit_order", "purchase",
    # Communication (sending on user's behalf)
    "send_email", "send_message", "post_message", "tweet", "submit_form",
    # Security-sensitive system changes
    "modify_firewall", "disable_security", "change_permissions",
    "modify_sudoers", "install_package", "uninstall_package",
    "modify_system_config", "change_network_settings",
})

# Actions that are READ-ONLY and always allowed without confirmation.
READ_ONLY_ACTIONS = frozenset({
    "read_screen", "list_windows", "get_time", "get_date", "music_status",
    "web_search", "screenshot", "desktop_open", "browser_navigate",
    "browser_search", "open_folder", "volume_up", "volume_down",
    "volume_set", "volume_mute", "brightness_up", "brightness_down",
    "brightness_set", "wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off",
})


def is_sensitive_action(action: str, params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determine if an action requires explicit user confirmation.

    Returns (requires_confirmation, reason).

    HUMAN SAFETY RULE:
    - Destructive, irreversible, credential-related, financial,
      communication, or security-sensitive actions require confirmation.
    - Read-only investigation is always allowed.
    """
    action_lower = action.lower()

    # Explicitly sensitive actions
    if action_lower in SENSITIVE_ACTIONS:
        return True, f"'{action}' is a sensitive action"

    # Heuristic: actions with destructive keywords in params
    params_str = str(params).lower()
    destructive_keywords = (
        "delete", "remove", "format", "wipe", "shred", "drop",
        "rm -rf", "sudo rm", "mkfs", "dd if=",
    )
    for kw in destructive_keywords:
        if kw in params_str:
            return True, f"action contains destructive keyword '{kw}'"

    # Heuristic: commands that might be destructive
    cmd = str(params.get("command", "")).lower()
    if cmd:
        sensitive_cmd = ("rm ", "rmdir", "shred", "dd ", "mkfs", "format",
                         "del ", "erase", "shutdown", "reboot", "halt",
                         "poweroff", "systemctl stop", "service stop")
        for kw in sensitive_cmd:
            if kw in cmd:
                return True, f"command contains sensitive keyword '{kw}'"

    return False, ""


class FailureKind(str, Enum):
    """Classification of a step failure, driving recovery strategy."""
    TRANSIENT = "transient"                    # retry is safe
    WRONG_PARAMS = "wrong_params"              # retry with adjusted params
    WRONG_TOOL = "wrong_tool"                  # re-plan with another action
    CHANGED_STATE = "changed_state"            # UI/state moved; re-observe
    UNAVAILABLE_CAPABILITY = "unavailable_capability"  # real blocker
    IMPOSSIBLE = "impossible"                  # cannot be done at all
    UNKNOWN = "unknown"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"          # verified success
    ALREADY_SATISFIED = "already_satisfied"  # idempotent skip (verified pre-existing)
    FAILED = "failed"
    SKIPPED_INVALID = "skipped_invalid"      # plan validation rejected it
    NEEDS_CONFIRMATION = "needs_confirmation"  # sensitive action awaiting user approval


# ═══════════════════════════════════════════════════════════════
# Configurable safety limits
# ═══════════════════════════════════════════════════════════════

class TaskLimits:
    """Hard termination guarantees. All values env-overridable."""

    def __init__(
        self,
        max_task_steps: Optional[int] = None,
        max_retries_per_step: Optional[int] = None,
        max_replans: Optional[int] = None,
        max_total_execution_time: Optional[float] = None,
    ):
        self.max_task_steps = int(
            max_task_steps
            if max_task_steps is not None
            else os.environ.get("DIEGO_MAX_TASK_STEPS", 12)
        )
        self.max_retries_per_step = int(
            max_retries_per_step
            if max_retries_per_step is not None
            else os.environ.get("DIEGO_MAX_RETRIES_PER_STEP", 2)
        )
        self.max_replans = int(
            max_replans
            if max_replans is not None
            else os.environ.get("DIEGO_MAX_REPLANS", 3)
        )
        self.max_total_execution_time = float(
            max_total_execution_time
            if max_total_execution_time is not None
            else os.environ.get("DIEGO_MAX_TASK_TIME_S", 300)
        )


# ═══════════════════════════════════════════════════════════════
# Step + task state records
# ═══════════════════════════════════════════════════════════════

class EvidenceSource(str, Enum):
    """Where a piece of evidence came from (priority of truth)."""
    LIVE_OBSERVATION = "live_observation"      # highest priority
    DETERMINISTIC_SYSTEM = "deterministic_system"  # OS-level facts
    ACTION_VERIFICATION = "action_verification"    # verified action result
    LOCAL_KNOWLEDGE = "local_knowledge"        # indexed knowledge with source
    DIAGNOSTICS = "diagnostics"                # measured diagnostics
    TOOL_RESULT = "tool_result"                # tool/action return value
    LLM_INFERENCE = "llm_inference"            # lowest priority
    UNKNOWN = "unknown"                        # no evidence available


@dataclass
class Evidence:
    """A piece of evidence with its source for priority-of-truth tracking."""
    fact: str
    source: EvidenceSource
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"{self.fact} [{self.source.value}]"


@dataclass
class StepRecord:
    """One step of a plan and its execution evidence."""
    index: int
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    result: str = ""                 # dispatcher result (evidence)
    verification: str = ""           # "success" / "failed: <reason>"
    verified: bool = False
    retries: int = 0
    error: str = ""
    latency_ms: float = 0.0
    evidence_source: EvidenceSource = EvidenceSource.UNKNOWN
    sensitive_reason: str = ""       # why this action needs confirmation

    def signature(self) -> str:
        """Canonical signature for duplicate/loop detection."""
        params = sorted((self.params or {}).items())
        return f"{self.action}:{params}"


@dataclass
class TaskExecutionState:
    """Explicit, inspectable state of one task execution."""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    original_request: str = ""
    normalized_goal: str = ""
    current_plan: List[Dict[str, Any]] = field(default_factory=list)
    completed_steps: List[StepRecord] = field(default_factory=list)
    current_step: Optional[StepRecord] = None
    failed_steps: List[StepRecord] = field(default_factory=list)
    observed_state: str = ""
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    replan_count: int = 0
    total_steps: int = 0
    final_status: Optional[FinalStatus] = None
    plan_version: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    # Real evidence gathered during execution (authoritative, never LLM):
    artifacts: Dict[str, Any] = field(default_factory=dict)
    # Human-readable execution log ([TASK]/[PLAN]/[STEP]/[VERIFY]/[REPLAN])
    log: List[str] = field(default_factory=list)
    blocker: str = ""                # honest explanation of a real blocker
    # Evidence tracking (priority of truth):
    evidence_log: List[Evidence] = field(default_factory=list)
    # Sensitive action awaiting confirmation:
    pending_confirmation: Optional[StepRecord] = None
    confirmation_reason: str = ""

    @property
    def total_latency_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return (end - self.started_at) * 1000.0

    def note(self, line: str) -> None:
        self.log.append(line)
        logger.info(line)

    def add_evidence(self, fact: str, source: EvidenceSource,
                     confidence: float = 1.0) -> None:
        """Record a piece of evidence with its source."""
        self.evidence_log.append(Evidence(
            fact=fact, source=source, confidence=confidence))

    def get_evidence(self, source: Optional[EvidenceSource] = None) -> List[Evidence]:
        """Get evidence, optionally filtered by source."""
        if source is None:
            return list(self.evidence_log)
        return [e for e in self.evidence_log if e.source == source]

    def summary(self) -> str:
        """Honest, status-aware summary for the spoken response."""
        done = len(self.completed_steps)
        failed = len(self.failed_steps)
        if self.final_status == FinalStatus.SUCCESS:
            if self.artifacts.get("answer"):
                return str(self.artifacts["answer"])
            if done <= 1:
                return (self.completed_steps[0].result
                        if self.completed_steps and self.completed_steps[0].result
                        else "Done.")
            return f"Done — {done} steps completed."
        if self.final_status == FinalStatus.PARTIAL_FAILURE:
            return (f"I completed {done} step(s) but {failed} step(s) failed: "
                    f"{self.blocker or 'the remaining steps could not be verified'}.")
        if self.final_status == FinalStatus.FAILED:
            return (f"I couldn't complete that. {self.blocker or 'The steps failed verification.'}")
        if self.final_status == FinalStatus.CANCELLED:
            return "Task cancelled."
        if self.final_status == FinalStatus.NEEDS_INPUT:
            return (f"I need your input to continue. {self.blocker}")
        return ""


# ═══════════════════════════════════════════════════════════════
# Failure classification
# ═══════════════════════════════════════════════════════════════

_TRANSIENT_MARKERS = (
    "timeout", "timed out", "temporarily", "try again", "busy",
    "not ready", "settle",
)
_UNAVAILABLE_MARKERS = (
    "couldn't find", "not installed", "isn't available", "not available",
    "no player responded", "mpv is not installed", "no local music",
    "no browser", "couldn't open the browser", "unavailable",
)
_WRONG_TOOL_MARKERS = (
    "unknown action", "not in the allowed action schema", "unsupported",
)


def classify_failure(action: str, result: str, error: str) -> FailureKind:
    """Deterministic failure classification from real evidence."""
    text = f"{result or ''} {error or ''}".lower()
    if any(m in text for m in _WRONG_TOOL_MARKERS):
        return FailureKind.WRONG_TOOL
    if any(m in text for m in _UNAVAILABLE_MARKERS):
        return FailureKind.UNAVAILABLE_CAPABILITY
    if any(m in text for m in _TRANSIENT_MARKERS):
        return FailureKind.TRANSIENT
    if "couldn't" in text or "failed" in text:
        # Dispatch executed but the effect was not achieved — most often
        # wrong parameters or a changed UI state.
        if action in ("click_text", "type_text", "scroll", "key_press"):
            return FailureKind.CHANGED_STATE
        return FailureKind.WRONG_PARAMS
    return FailureKind.UNKNOWN


def adjust_params_for_retry(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic parameter adjustment for a strategic retry."""
    adjusted = dict(params or {})
    if action == "desktop_open":
        alt_map = {
            "code": "code-insiders", "vscode": "code", "vs code": "code",
            "firefox": "firefox-esr", "chrome": "chromium-browser",
            "google-chrome": "chromium", "gnome-terminal": "xterm",
            "terminal": "xterm", "nautilus": "thunar", "files": "thunar",
        }
        app = str(adjusted.get("app", "")).lower()
        if app in alt_map:
            adjusted["app"] = alt_map[app]
    elif action == "browser_navigate":
        url = str(adjusted.get("url", ""))
        if url.startswith("https://"):
            adjusted["url"] = url.replace("https://", "http://", 1)
        elif url and not url.startswith(("http://", "https://")):
            adjusted["url"] = "https://" + url
    return adjusted


# ═══════════════════════════════════════════════════════════════
# Loop detection
# ═══════════════════════════════════════════════════════════════

class LoopDetector:
    """Detects repeated identical actions / states / failures, oscillating
    states, and planner producing the same plan repeatedly."""

    def __init__(self, repeat_threshold: int = 3):
        self.repeat_threshold = repeat_threshold
        self._action_sigs: List[str] = []
        self._state_sigs: List[str] = []
        self._failure_sigs: List[str] = []
        self._plan_sigs: List[Tuple[str, ...]] = []

    def record_action(self, sig: str) -> Optional[str]:
        self._action_sigs.append(sig)
        tail = self._action_sigs[-self.repeat_threshold:]
        if (len(tail) == self.repeat_threshold
                and len(set(tail)) == 1):
            return f"repeated identical action x{self.repeat_threshold}: {sig}"
        return None

    def record_state(self, sig: str) -> Optional[str]:
        if not sig:
            return None
        self._state_sigs.append(sig)
        # Repeated identical state after an action (no progress)
        if len(self._state_sigs) >= 3:
            tail = self._state_sigs[-3:]
            if len(set(tail)) == 1:
                return "repeated identical observed state (no progress)"
        # Oscillation A,B,A,B
        if len(self._state_sigs) >= 4:
            a, b, c, d = self._state_sigs[-4:]
            if a == c and b == d and a != b:
                return "oscillating states detected"
        return None

    def record_failure(self, sig: str) -> Optional[str]:
        self._failure_sigs.append(sig)
        tail = self._failure_sigs[-self.repeat_threshold:]
        if (len(tail) == self.repeat_threshold
                and len(set(tail)) == 1):
            return f"repeated identical failure x{self.repeat_threshold}: {sig}"
        return None

    def record_plan(self, plan: List[Dict[str, Any]]) -> Optional[str]:
        sig = tuple(str(s.get("action", "")) for s in (plan or []))
        self._plan_sigs.append(sig)
        if len(self._plan_sigs) >= 2 and self._plan_sigs[-1] == self._plan_sigs[-2]:
            return "planner produced the same plan repeatedly"
        return None


# ═══════════════════════════════════════════════════════════════
# Plan validation
# ═══════════════════════════════════════════════════════════════

# Actions the dispatcher can actually execute (authoritative list).
KNOWN_ACTIONS = frozenset({
    "desktop_open", "close_app", "browser_navigate", "browser_search",
    "web_search", "web_search_open_best", "youtube_search", "read_screen",
    "click_text", "scroll", "key_press", "type_text", "play_media",
    "open_folder", "volume_up", "volume_down", "volume_set", "volume_mute",
    "brightness_up", "brightness_down", "brightness_set", "lock_screen",
    "shutdown", "restart", "get_time", "get_date", "minimize_window",
    "maximize_window", "switch_workspace", "switch_workspace_prev",
    "switch_window", "switch_window_prev", "switch_tab", "switch_tab_prev",
    "list_windows", "music_status", "music_pause", "music_resume",
    "music_next", "music_previous", "music_stop", "music_shuffle",
    "music_repeat", "music_volume", "music_mute", "screenshot",
    "wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off",
    "focus_app", "close_window",
})

def registry_tool_available(name: str) -> bool:
    """True if `name` resolves to a tool in the general ToolRegistry.

    Phase 15B: the registry's built-in general tools (terminal, python,
    filesystem, git, docker, clipboard_*, notify, mouse, keyboard, …) are
    part of the canonical action namespace the autonomous loop may plan,
    alongside KNOWN_ACTIONS. Lazy + exception-safe: the registry is
    optional infrastructure and must never break plan validation.
    """
    try:
        from core.tool_registry import tool_registry
        tool_registry.install_builtin_tools()
        return tool_registry.is_available(name)
    except Exception:
        return False


# Required / validated parameters per action.
PARAM_REQUIREMENTS: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = {
    "desktop_open": lambda p: None if str(p.get("app", "")).strip() else "app",
    "close_app": lambda p: None if str(p.get("app", "")).strip() else "app",
    "browser_navigate": lambda p: None if str(p.get("url", "")).strip() else "url",
    "browser_search": lambda p: None if str(p.get("query", "")).strip() else "query",
    "web_search": lambda p: None if str(p.get("query", "")).strip() else "query",
    "web_search_open_best": lambda p: None if str(p.get("query", "")).strip() else "query",
    "youtube_search": lambda p: None if str(p.get("query", "")).strip() else "query",
    "click_text": lambda p: None if str(p.get("text", "")).strip() else "text",
    "type_text": lambda p: None if str(p.get("text", "")).strip() else "text",
    "key_press": lambda p: None if str(p.get("key", "")).strip() else "key",
    "scroll": lambda p: (None if str(p.get("direction", "down")).strip() else "direction"),
    "play_media": lambda p: None if str(p.get("query", "")).strip() else "query",
    "open_folder": lambda p: None if str(p.get("path", "")).strip() else "path",
    "volume_set": lambda p: None if _is_int(p.get("percent")) else "percent",
    "brightness_set": lambda p: None if _is_int(p.get("percent")) else "percent",
    "music_volume": lambda p: None if _is_int(p.get("percent", p.get("level"))) else "percent",
}


def _is_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


class PlanValidator:
    """Validates a planner-generated plan BEFORE dispatching anything.

    Checks:
      - action exists (not hallucinated)
      - action is allowed (optional external gate, e.g. Brain's
        intent-evidence gate)
      - arguments are valid
      - action is relevant to the current goal (non-empty plan only)
      - action is not a duplicate of an already-completed step
    """

    def __init__(self, action_gate: Optional[Callable[[str, Dict[str, Any]], bool]] = None):
        # action_gate(transcript, action_dict) -> bool (Brain's hallucination gate)
        self._action_gate = action_gate

    def validate_step(self, step: Dict[str, Any],
                      completed: List[StepRecord],
                      transcript: str = "") -> Tuple[bool, str]:
        if not isinstance(step, dict):
            return False, "step is not an action dict"
        name = str(step.get("action", "")).strip()
        if not name:
            return False, "step has no action name"
        if name not in KNOWN_ACTIONS and not registry_tool_available(name):
            return False, f"hallucinated/unknown action '{name}'"
        params = step.get("params") or {}
        if not isinstance(params, dict):
            return False, "params must be a dict"
        req = PARAM_REQUIREMENTS.get(name)
        if req:
            missing = req(params)
            if missing:
                return False, f"missing/invalid argument '{missing}' for {name}"
        if self._action_gate is not None and transcript:
            try:
                if not self._action_gate(transcript, {"action": name, "params": params}):
                    return False, f"action '{name}' not authorized for this request"
            except Exception:
                pass  # gate failure must never crash the loop
        sig = StepRecord(index=0, action=name, params=params).signature()
        for done in completed:
            if done.signature() == sig and done.status in (
                    StepStatus.COMPLETED, StepStatus.ALREADY_SATISFIED):
                return False, "duplicate of an already-completed step"
        return True, ""

    def validate_plan(self, plan: List[Dict[str, Any]],
                      completed: List[StepRecord],
                      transcript: str = "") -> List[Dict[str, Any]]:
        """Return only the valid steps of a plan (drops invalid ones)."""
        valid = []
        for step in (plan or []):
            ok, _reason = self.validate_step(step, completed, transcript)
            if ok:
                valid.append(step)
        return valid


# ═══════════════════════════════════════════════════════════════
# Idempotency — is the desired state already true?
# ═══════════════════════════════════════════════════════════════

_PROC_MAP = {
    "code": "code", "vscode": "code", "vs code": "code",
    "firefox": "firefox", "browser": "firefox",
    "chrome": "chrome", "google-chrome": "chrome",
    "spotify": "spotify", "gnome-terminal": "gnome-terminal",
    "terminal": "gnome-terminal", "nautilus": "nautilus",
    "files": "nautilus", "slack": "slack", "discord": "discord",
    "telegram-desktop": "telegram", "notion-app": "notion",
    "pycharm": "pycharm",
}


def radio_enabled(device: str) -> Optional[bool]:
    """OS-level check: is the wifi/bluetooth radio already enabled?

    Uses nmcli when available (authoritative NetworkManager state).
    Returns None when the state cannot be determined."""
    if not shutil.which("nmcli"):
        return None
    try:
        chk = subprocess.run(
            ["nmcli", "radio", device],
            capture_output=True, text=True, timeout=5)
        if chk.returncode != 0:
            return None
        out = (chk.stdout or "").strip().lower()
        if out.startswith("enabled"):
            return True
        if out.startswith("disabled"):
            return False
        return None
    except Exception:
        return None


def app_running(app: str) -> bool:
    """OS-level check: is this app's process already running? (pgrep -x)"""
    if not app or not shutil.which("pgrep"):
        return False
    proc = _PROC_MAP.get(app.lower(), app.lower())
    try:
        chk = subprocess.run(["pgrep", "-x", proc],
                             capture_output=True, text=True, timeout=3)
        return chk.returncode == 0
    except Exception:
        return False


class IdempotencyChecker:
    """Before executing an action, determine whether the desired state
    already exists. If so, the step is marked ALREADY_SATISFIED (verified
    by real observation) instead of executing a duplicate action."""

    async def check(self, action: str, params: Dict[str, Any],
                    state: TaskExecutionState) -> Optional[str]:
        """Return a reason string if the desired state already exists."""
        loop = asyncio.get_event_loop()
        if action == "desktop_open":
            app = str(params.get("app", ""))
            if app and await loop.run_in_executor(None, app_running, app):
                return f"{app} is already open"
        elif action == "browser_navigate":
            url = str(params.get("url", "")).strip()
            if url and state.artifacts.get("last_url") == url:
                return f"already on {url}"
        elif action == "web_search":
            query = str(params.get("query", "")).strip().lower()
            if query and query == str(state.artifacts.get("last_query", "")).lower():
                return "same search already performed"
        elif action == "music_pause":
            obs = (state.observed_state or "").lower()
            if "paused" in obs:
                return "music is already paused"
        elif action in ("wifi_on", "bluetooth_on"):
            device = "wifi" if action == "wifi_on" else "bluetooth"
            enabled = await loop.run_in_executor(None, radio_enabled, device)
            if enabled is True:
                display = "Wi-Fi" if device == "wifi" else device.capitalize()
                return f"{display} is already enabled"
        elif action in ("volume_set", "brightness_set", "music_volume"):
            key = "volume" if "volume" in action else "brightness"
            try:
                target = int(params.get("percent", -1))
            except (TypeError, ValueError):
                return None
            if state.artifacts.get(f"{key}_level") == target:
                return f"{key} already at {target}"
        return None


# ═══════════════════════════════════════════════════════════════
# Artifact extraction (authoritative evidence from dispatch results)
# ═══════════════════════════════════════════════════════════════

# Actions whose dispatch result IS the user-facing answer.
ANSWER_ACTIONS = frozenset({
    "read_screen", "web_search", "list_windows", "music_status",
    "get_time", "get_date",
})

_URL_RE = re.compile(r"https?://[^\s)\"']+")


def extract_artifacts(action: str, params: Dict[str, Any],
                      result: str, artifacts: Dict[str, Any]) -> None:
    """Update task artifacts from a REAL dispatch result (never invented)."""
    if not result:
        return
    urls = _URL_RE.findall(result)
    if urls:
        existing = artifacts.get("urls", [])
        for u in urls:
            if u not in existing:
                existing.append(u)
        artifacts["urls"] = existing
        # NOTE: last_url is ONLY set by actual navigation below — a URL
        # merely MENTIONED in a result (e.g. a search hit) is not a page
        # we are on.
    if action in ("web_search", "web_search_open_best", "browser_search"):
        artifacts["last_query"] = str(params.get("query", ""))
        # Search results (titles/urls) for "open the first result" follow-ups
        results = artifacts.get("search_results", [])
        for u in urls:
            if u not in results:
                results.append(u)
        artifacts["search_results"] = results
        # NOTE: a search RESULT list is not "being on" those pages —
        # last_url is only set by actual navigation below.
    if action in ("browser_navigate", "web_search_open_best"):
        # We actually OPENED this page — record it as the current location.
        if urls:
            artifacts["last_url"] = urls[-1]
    if action in ANSWER_ACTIONS:
        artifacts["answer"] = result
    if action == "desktop_open" and params.get("app"):
        artifacts["last_app"] = str(params.get("app"))
    if action in ("volume_set",) and _is_int(params.get("percent")):
        artifacts["volume_level"] = int(params["percent"])
    if action in ("brightness_set",) and _is_int(params.get("percent")):
        artifacts["brightness_level"] = int(params["percent"])


# ═══════════════════════════════════════════════════════════════
# The closed-loop TaskRunner
# ═══════════════════════════════════════════════════════════════

# Actions that are expected to change the real environment (used to flag
# "action succeeded but nothing changed" as an unverified effect).
STATEFUL_ACTIONS = frozenset({
    "desktop_open", "close_app", "browser_navigate", "browser_search",
    "click_text", "type_text", "scroll", "key_press", "play_media",
    "volume_up", "volume_down", "volume_set", "volume_mute",
    "brightness_up", "brightness_down", "brightness_set",
    "music_pause", "music_resume", "music_next", "music_previous",
    "music_stop", "music_volume", "music_mute", "lock_screen",
    "minimize_window", "maximize_window",
    "wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off",
    "focus_app", "close_window",
})

Executor = Callable[[Dict[str, Any]], Awaitable[Tuple[bool, str]]]
Observer = Callable[[], Awaitable[str]]
Planner = Callable[[str, Dict[str, Any]], Awaitable[Optional[List[Dict[str, Any]]]]]
# Confirmation callback: returns True if user approved the sensitive action.
ConfirmationCallback = Callable[[str, str, Dict[str, Any]], Awaitable[bool]]


class TaskRunner:
    """
    Closed-loop task executor.

    For every step:
      1. validate the step (plan validation)
      2. idempotency check (desired state already true?)
      3. execute exactly ONE actionable step
      4. observe the real environment
      5. verify the result
      6. mark completed ONLY when verification succeeds
      7. on failure: classify → retry (safe only) → re-plan from CURRENT state
      8. loop-detect and enforce hard limits
      9. continue until the goal is satisfied or a real blocker remains

    Dependencies are injected so production (Brain) and tests (fakes) share
    the exact same loop logic.
    """

    def __init__(
        self,
        executor: Executor,
        observer: Optional[Observer] = None,
        planner: Optional[Planner] = None,
        validator: Optional[PlanValidator] = None,
        limits: Optional[TaskLimits] = None,
        transcript: str = "",
        confirmation_callback: Optional[ConfirmationCallback] = None,
        approved_actions: Optional[frozenset] = None,
    ):
        self._executor = executor
        self._observer = observer
        self._planner = planner
        self._validator = validator or PlanValidator()
        self._idempotency = IdempotencyChecker()
        self._limits = limits or TaskLimits()
        self._transcript = transcript
        self._loops = LoopDetector()
        self._cancelled = False
        self._confirmation_callback = confirmation_callback
        # Action signatures the user has ALREADY approved (confirmation
        # resumption). A sensitive step whose signature is present here is
        # executed without re-asking — the user already said "yes".
        self._approved_actions = approved_actions or frozenset()

    # ── External control ──────────────────────────────────────

    def cancel(self) -> None:
        """Request cancellation (checked between steps — never mid-action)."""
        self._cancelled = True

    def _action_pre_approved(self, action: str, params: Dict[str, Any]) -> bool:
        """True if the user already approved this exact action (resumption)."""
        if not self._approved_actions:
            return False
        sig = StepRecord(index=0, action=action, params=params).signature()
        return sig in self._approved_actions

    # ── Lifecycle events (Phase 15F) ──────────────────────────
    # Task lifecycle transitions are published on the EXISTING event bus
    # (core/event_bus.py — the single shared bus) so UI/background
    # components can observe task start, progress, completion, failure,
    # cancellation, blocking, and replanning. No second event mechanism
    # is introduced.

    async def _emit_event(self, event_type: str,
                          state: TaskExecutionState,
                          **data: Any) -> None:
        """Emit one task lifecycle event on the existing event bus.

        NON-FATAL BY CONTRACT: bus unavailability, consumer errors, or any
        emission failure is logged and swallowed — event reporting must
        never change the task outcome or crash the runner. Payloads carry
        identifiers, counts, and SHORT reasons only; action params (which
        may hold user/sensitive content) are never included.
        """
        try:
            from core.event_bus import bus
            payload: Dict[str, Any] = {
                "task_id": state.task_id,
                "status": (state.final_status.value
                           if state.final_status is not None else ""),
                "completed_steps": len(state.completed_steps),
                "failed_steps": len(state.failed_steps),
                "retry_count": state.retry_count,
                "replan_count": state.replan_count,
            }
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 160:
                    value = value[:160] + "…"
                payload[key] = value
            await bus.emit(event_type, data=payload, source="task_runner")
        except Exception as e:
            logger.debug("[TaskRunner] lifecycle event '%s' not emitted: %s",
                         event_type, e)

    async def _emit_final_status(self, state: TaskExecutionState) -> None:
        """Emit the AUTHORITATIVE final lifecycle event for a task.

        Exactly one terminal event per task, derived from the final
        TaskExecutionState (never from optimistic intentions):
          SUCCESS → task.completed (only after verified success)
          CANCELLED → task.cancelled (never followed by task.completed)
          NEEDS_CONFIRMATION / NEEDS_INPUT → distinct events, never FAILED
          FAILED / PARTIAL_FAILURE → task.failed
        """
        status = state.final_status
        if status == FinalStatus.SUCCESS:
            await self._emit_event("task.completed", state)
        elif status == FinalStatus.CANCELLED:
            await self._emit_event("task.cancelled", state)
        elif status == FinalStatus.NEEDS_CONFIRMATION:
            await self._emit_event("task.needs_confirmation", state,
                                   reason=state.confirmation_reason)
        elif status == FinalStatus.NEEDS_INPUT:
            await self._emit_event("task.needs_input", state,
                                   blocker=state.blocker)
        elif status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE):
            await self._emit_event("task.failed", state,
                                   blocker=state.blocker)

    # ── Main loop ─────────────────────────────────────────────

    async def run(self, request: str,
                  initial_plan: Optional[List[Dict[str, Any]]] = None,
                  inherited: Optional[TaskExecutionState] = None) -> TaskExecutionState:
        """Run the closed loop until the goal is satisfied or a real
        blocker requires stopping. Returns the full TaskExecutionState."""
        state = TaskExecutionState(
            original_request=request,
            normalized_goal=request,
        )
        if inherited is not None:
            # Follow-up: inherit observed state + artifacts from the
            # previous task so references ("the first result") resolve.
            state.observed_state = inherited.observed_state
            state.artifacts = dict(inherited.artifacts)
            state.normalized_goal = inherited.normalized_goal or request
            # CONTINUATION (Phase 15G): inherit the AUTHORITATIVE record of
            # previously VERIFIED work so:
            #   - duplicate validation prevents re-executing verified
            #     steps (no repeated side effects),
            #   - re-plans receive the verified history in their context,
            #   - an already-satisfied task can finish honestly.
            # Merely-executed but UNVERIFIED steps are NOT trusted — they
            # are not inherited as completed and will execute/verify again
            # if the continuation plan contains them.
            state.completed_steps = [
                _dc_replace(s) for s in (inherited.completed_steps or [])
                if s.verified and s.status in (
                    StepStatus.COMPLETED, StepStatus.ALREADY_SATISFIED)
            ]
            state.failed_steps = [
                _dc_replace(s) for s in (inherited.failed_steps or [])
            ]
            # CANCELLED is authoritative across continuation: a persisted
            # cancelled task is never silently resumed (let alone as
            # SUCCESS) without an explicit new request.
            if inherited.final_status == FinalStatus.CANCELLED:
                state.final_status = FinalStatus.CANCELLED
                state.note("[TASK] CANCELLED (inherited persisted state — "
                           "not resumed)")
                state.ended_at = time.time()
                await self._emit_final_status(state)
                return state

        plan = self._validator.validate_plan(
            initial_plan or [], state.completed_steps, self._transcript)
        if not plan and initial_plan:
            # Report WHY every step was rejected (hallucinated action,
            # missing argument, unauthorized, duplicate…).
            reasons: List[str] = []
            for step in initial_plan:
                ok, reason = self._validator.validate_step(
                    step, state.completed_steps, self._transcript)
                if not ok:
                    reasons.append(reason)
            state.note(f"[TASK] id={state.task_id} goal=\"{request[:80]}\"")
            state.note("[PLAN] v0 rejected — " + "; ".join(reasons[:4]))
            # CONTINUATION: a plan whose EVERY step is a duplicate of
            # already-verified completed work means the task is DONE —
            # honest SUCCESS without repeating any side effect. (A plan
            # rejected for hallucination/params/auth is still a failure.)
            if (all(r == "duplicate of an already-completed step"
                    for r in reasons) and state.completed_steps):
                state.current_plan = list(initial_plan)
                state.final_status = FinalStatus.SUCCESS
                state.note("[TASK] COMPLETE (all steps already verified — "
                           "nothing to re-execute)")
                state.ended_at = time.time()
                await self._emit_final_status(state)
                return state
            state.final_status = FinalStatus.FAILED
            state.blocker = ("The generated plan contained no executable "
                             "actions: " + "; ".join(reasons[:3]))
            state.ended_at = time.time()
            await self._emit_final_status(state)
            return state

        state.current_plan = plan
        state.total_steps = len(plan)
        state.plan_version = 1
        state.note(f"[TASK] id={state.task_id} goal=\"{request[:80]}\"")
        state.note(f"[PLAN] v1 steps={len(plan)}")
        # NOTE: the raw request/goal text is deliberately NOT included —
        # it may contain sensitive user content (requirement I). Consumers
        # correlate via task_id and the inspectable TaskExecutionState.
        await self._emit_event("task.started", state, steps=len(plan))

        pending: List[Dict[str, Any]] = list(plan)
        step_counter = itertools.count(1)

        # Observe the initial state (baseline for change detection)
        state.observed_state = await self._safe_observe()

        # Self-register so the Brain / UI / voice can request cancellation
        # of this in-flight task from another async context.
        task_state_store.register_runner(self)
        try:
            return await self._run_loop_internal(state, pending, step_counter,
                                                 request, initial_plan)
        finally:
            task_state_store.unregister_runner()

    async def _run_loop_internal(
        self,
        state: TaskExecutionState,
        pending: List[Dict[str, Any]],
        step_counter: Any,
        request: str,
        initial_plan: Optional[List[Dict[str, Any]]] = None,
    ) -> TaskExecutionState:
        """Internal closed-loop execution (extracted so runner registration
        and cleanup are always paired)."""
        while True:
            # ── Safety gates ──────────────────────────────────
            if self._cancelled:
                state.final_status = FinalStatus.CANCELLED
                state.note("[TASK] CANCELLED")
                break

            executed = len(state.completed_steps) + len(state.failed_steps)
            if executed >= self._limits.max_task_steps:
                state.final_status = self._partial_or_failed(state)
                state.blocker = (f"Reached the task step limit "
                                 f"({self._limits.max_task_steps}).")
                state.note(f"[TASK] STOPPED — step limit {self._limits.max_task_steps} reached")
                break

            if (time.time() - state.started_at) > self._limits.max_total_execution_time:
                state.final_status = self._partial_or_failed(state)
                state.blocker = ("Task exceeded its execution time budget "
                                 f"({self._limits.max_total_execution_time:.0f}s).")
                state.note("[TASK] STOPPED — execution time budget exceeded")
                break

            # ── Plan supply ───────────────────────────────────
            if not pending:
                if self._goal_satisfied(state):
                    state.final_status = FinalStatus.SUCCESS
                    state.note("[TASK] COMPLETE")
                    break
                # Plan exhausted but goal not verified — re-plan from the
                # CURRENT observed state (never assume the original plan
                # is still valid).
                replanned = await self._replan(state, request)
                if replanned is None:
                    state.final_status = self._partial_or_failed(state)
                    if not state.blocker:
                        state.blocker = ("No further actions could be planned "
                                         "from the current state.")
                    state.note("[TASK] STOPPED — replanning exhausted")
                    break
                pending = replanned

            # ── Pick + validate the next step ─────────────────
            step_dict = pending.pop(0)
            ok, reason = self._validator.validate_step(
                step_dict, state.completed_steps, self._transcript)
            if not ok:
                idx = next(step_counter)
                if reason == "duplicate of an already-completed step":
                    # The work is already done — skipping is CORRECT, not
                    # a failure. Never re-execute a completed step.
                    state.note(f"[STEP {idx}] SKIPPED — {reason}")
                    continue
                rec = StepRecord(index=idx, action=str(step_dict.get("action", "?")),
                                 params=step_dict.get("params") or {},
                                 status=StepStatus.SKIPPED_INVALID)
                rec.verification = f"rejected: {reason}"
                state.failed_steps.append(rec)
                state.note(f"[STEP {idx}] REJECTED — {reason}")
                if not pending and not self._replans_left(state):
                    state.final_status = self._partial_or_failed(state)
                    state.blocker = state.blocker or reason
                    state.note("[TASK] STOPPED — no valid steps remain")
                    break
                continue

            idx = next(step_counter)
            action = str(step_dict.get("action"))
            params = dict(step_dict.get("params") or {})
            rec = StepRecord(index=idx, action=action, params=params,
                             description=str(step_dict.get("description", "")),
                             status=StepStatus.RUNNING)
            state.current_step = rec
            state.note(f"[STEP {idx}/{max(state.total_steps, idx)}] {action}")
            await self._emit_event("task.step.started", state,
                                   step=idx, action=action)

            # ── Idempotency: desired state already true? ──────
            already = await self._idempotency.check(action, params, state)
            if already:
                rec.status = StepStatus.ALREADY_SATISFIED
                rec.verified = True
                rec.verification = f"already satisfied: {already}"
                rec.result = already
                rec.evidence_source = EvidenceSource.DETERMINISTIC_SYSTEM
                state.completed_steps.append(rec)
                state.verification_results.append(
                    {"step": idx, "action": action, "verified": True,
                     "evidence": already})
                state.add_evidence(already, EvidenceSource.DETERMINISTIC_SYSTEM)
                state.note(f"[VERIFY] success (idempotent — {already})")
                await self._emit_event("task.step.completed", state,
                                       step=idx, action=action,
                                       evidence=already)
                continue

            # ── HUMAN SAFETY: sensitive action confirmation ────
            sensitive, sensitive_reason = is_sensitive_action(action, params)
            if sensitive and not self._action_pre_approved(action, params):
                rec.sensitive_reason = sensitive_reason
                approved = await self._request_confirmation(
                    action, sensitive_reason, params, state)
                if not approved:
                    rec.status = StepStatus.NEEDS_CONFIRMATION
                    rec.verification = f"awaiting confirmation: {sensitive_reason}"
                    state.pending_confirmation = rec
                    state.confirmation_reason = sensitive_reason
                    state.final_status = FinalStatus.NEEDS_CONFIRMATION
                    state.blocker = (f"'{action}' requires your explicit "
                                     f"confirmation: {sensitive_reason}")
                    state.note(f"[STEP {idx}] BLOCKED — sensitive action "
                               f"requires confirmation: {sensitive_reason}")
                    state.ended_at = time.time()
                    await self._emit_final_status(state)
                    return state

            # ── Execute ONE step ──────────────────────────────
            t0 = time.time()
            pre_state = state.observed_state
            try:
                exec_ok, exec_result = await self._executor(
                    {"action": action, "params": params})
            except Exception as e:
                exec_ok, exec_result = False, f"execution error: {e}"
            rec.latency_ms = (time.time() - t0) * 1000.0
            rec.result = exec_result or ""

            # ── Observe the real environment ──────────────────
            post_state = await self._safe_observe()
            if post_state:
                state.observed_state = post_state

            # ── Verify ────────────────────────────────────────
            # The executor (Brain._dispatch_and_verify) is AUTHORITATIVE:
            # it already verified via OS process state / vision / dispatch
            # evidence. The runner's own observation feeds state tracking
            # and loop detection (repeated identical state = no progress).
            verified = bool(exec_ok)
            verify_reason = ""
            if not verified:
                verify_reason = (exec_result or "execution reported failure")

            rec.verified = verified
            rec.verification = ("success" if verified
                                else f"failed: {verify_reason}")
            state.verification_results.append(
                {"step": idx, "action": action, "verified": verified,
                 "evidence": rec.result[:200], "reason": verify_reason})
            state.note(f"[VERIFY] {'success' if verified else 'failure'}"
                       + (f" — {verify_reason}" if verify_reason else ""))
            if not verified:
                await self._emit_event("task.step.failed", state,
                                       step=idx, action=action,
                                       reason=verify_reason)
                await self._emit_event("task.verification.failed", state,
                                       step=idx, action=action,
                                       reason=verify_reason)

            if verified:
                rec.status = StepStatus.COMPLETED
                state.completed_steps.append(rec)
                extract_artifacts(action, params, rec.result, state.artifacts)
                await self._emit_event("task.step.completed", state,
                                       step=idx, action=action,
                                       evidence=rec.result[:160])
                loop_hit = self._loops.record_action(rec.signature())
                state_hit = self._loops.record_state(
                    self._state_signature(post_state))
                if loop_hit or state_hit:
                    # Progress is repeating — stop safely, preserve state.
                    state.blocker = loop_hit or state_hit or ""
                    state.final_status = self._partial_or_failed(state)
                    state.note(f"[TASK] STOPPED — loop detected: {state.blocker}")
                    break
                state.current_step = None
                continue

            # ── Failure recovery ──────────────────────────────
            # 1. Capture the actual failure and classify it.
            # 2. Retry ONLY when safe (transient / wrong params / changed
            #    state), up to MAX_RETRIES_PER_STEP, with deterministically
            #    adjusted params — never the exact failed action forever.
            # 3. If retries are exhausted, re-plan from the CURRENT state.
            rec.status = StepStatus.FAILED
            rec.error = verify_reason
            kind = classify_failure(action, rec.result, verify_reason)
            # Retry attempts are EXPECTED repeats — include the attempt
            # number so legitimate retries are not flagged as loops.
            loop_hit = self._loops.record_failure(
                f"{action}:{kind}:{verify_reason[:80]}:r{rec.retries}")

            retried_ok = False
            while (kind in (FailureKind.TRANSIENT, FailureKind.WRONG_PARAMS,
                            FailureKind.CHANGED_STATE)
                   and rec.retries < self._limits.max_retries_per_step):
                # CANCELLED is authoritative: a cancellation requested
                # during a failing step must NOT burn the retry budget.
                # Breaking here falls through to the failed-step record,
                # and the pre-replan cancellation check below returns the
                # task to the top of the loop where CANCELLED is set.
                if self._cancelled:
                    break
                new_params = adjust_params_for_retry(action, params)
                if new_params == params and kind != FailureKind.TRANSIENT:
                    break  # no safe adjustment possible — stop retrying
                rec.retries += 1
                state.retry_count += 1
                state.note(f"[RETRY] {action} attempt {rec.retries}"
                           f"/{self._limits.max_retries_per_step} ({kind.value})")
                await self._emit_event("task.retry", state,
                                       step=idx, action=action,
                                       retry=rec.retries,
                                       max_retries=self._limits.max_retries_per_step,
                                       reason=kind.value)
                t0 = time.time()
                try:
                    exec_ok2, exec_result2 = await self._executor(
                        {"action": action, "params": new_params})
                except Exception as e:
                    exec_ok2, exec_result2 = False, f"execution error: {e}"
                rec.latency_ms += (time.time() - t0) * 1000.0
                post2 = await self._safe_observe()
                if post2:
                    state.observed_state = post2
                verified2 = bool(exec_ok2)
                retry_reason = ("" if verified2
                                else (exec_result2 or "retry unverified"))
                rec.result = exec_result2 or rec.result
                rec.verified = verified2
                rec.verification = ("success" if verified2
                                    else f"failed: {retry_reason}")
                state.verification_results.append(
                    {"step": idx, "action": action, "verified": verified2,
                     "retry": rec.retries, "evidence": rec.result[:200]})
                state.note(f"[VERIFY] {'success' if verified2 else 'failure'} (retry {rec.retries})")
                if verified2:
                    rec.status = StepStatus.COMPLETED
                    state.completed_steps.append(rec)
                    extract_artifacts(action, new_params, rec.result, state.artifacts)
                    retried_ok = True
                    break
                rec.status = StepStatus.FAILED
                rec.error = retry_reason
                kind = classify_failure(action, rec.result, retry_reason)
                loop_hit = self._loops.record_failure(
                    f"{action}:{kind}:{retry_reason[:80]}:r{rec.retries}")
                if loop_hit:
                    break

            if retried_ok:
                state.current_step = None
                continue

            # Retry failed or unsafe — record the failed step.
            state.failed_steps.append(rec)
            if loop_hit:
                state.blocker = loop_hit
                state.final_status = self._partial_or_failed(state)
                state.note(f"[TASK] STOPPED — loop detected: {loop_hit}")
                break

            # ── Re-plan from the CURRENT state ────────────────
            # CANCELLED is authoritative: never start a re-plan after the
            # user cancelled. `continue` returns to the top of the loop,
            # where the cancellation gate sets FinalStatus.CANCELLED.
            if self._cancelled:
                state.note("[TASK] cancellation requested — no re-plan")
                continue
            if not self._replans_left(state):
                state.final_status = self._partial_or_failed(state)
                state.blocker = (state.blocker or
                                 f"Step '{action}' failed ({kind.value}) and "
                                 f"no re-plan budget remains.")
                state.note("[TASK] STOPPED — replan limit reached")
                break
            replanned = await self._replan(
                state, request,
                failure_context=f"step '{action}' failed: {verify_reason}",
            )
            if replanned:
                pending = replanned
                state.current_step = None
                continue

            # Re-planning produced nothing usable — this is a real blocker.
            if kind in (FailureKind.UNAVAILABLE_CAPABILITY, FailureKind.IMPOSSIBLE):
                state.final_status = (
                    FinalStatus.NEEDS_INPUT
                    if state.completed_steps
                    else FinalStatus.FAILED
                )
                state.blocker = (rec.result or
                                 f"'{action}' is not available on this system.")
                if state.final_status == FinalStatus.NEEDS_INPUT:
                    state.blocker = (f"{state.blocker} Would you like me to "
                                     f"try a different approach?")
            else:
                state.final_status = self._partial_or_failed(state)
                state.blocker = (state.blocker or
                                 f"Step '{action}' failed and could not be recovered.")
            state.note("[TASK] STOPPED — unrecoverable step failure")
            break

        state.total_steps = max(state.total_steps, len(state.completed_steps)
                                + len(state.failed_steps))
        state.ended_at = time.time()
        state.current_step = None
        if state.final_status is None:
            state.final_status = self._partial_or_failed(state)
        state.note(f"[TASK] {state.final_status.value} "
                   f"(steps={len(state.completed_steps)} ok, "
                   f"{len(state.failed_steps)} failed, "
                   f"replans={state.replan_count}, "
                   f"latency={state.total_latency_ms:.0f}ms)")
        await self._emit_final_status(state)
        return state

    # ── Internals ─────────────────────────────────────────────

    async def _request_confirmation(self, action: str, reason: str,
                                    params: Dict[str, Any],
                                    state: TaskExecutionState) -> bool:
        """Request user confirmation for a sensitive action.

        Returns True if approved, False if denied or no callback available.
        HUMAN SAFETY RULE: without an explicit approval mechanism in place,
        sensitive actions are DENIED by default.
        """
        if self._confirmation_callback is None:
            # No confirmation mechanism — deny by default (safe).
            logger.info("[TaskRunner] Sensitive action '%s' denied: "
                        "no confirmation callback available", action)
            return False
        try:
            return await asyncio.wait_for(
                self._confirmation_callback(action, reason, params),
                timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("[TaskRunner] Confirmation timeout for '%s'", action)
            return False
        except Exception as e:
            logger.warning("[TaskRunner] Confirmation callback failed: %s", e)
            return False

    def _replans_left(self, state: TaskExecutionState) -> bool:
        return state.replan_count < self._limits.max_replans

    async def _safe_observe(self) -> str:
        """Observe the real environment; observation failure is non-fatal."""
        if not self._observer:
            return ""
        try:
            return await asyncio.wait_for(self._observer(), timeout=15.0) or ""
        except Exception:
            return ""

    @staticmethod
    def _state_signature(observed: str) -> str:
        return " ".join((observed or "").split())[:200]

    @staticmethod
    def _observation_includes_effect(observed: str, action: str,
                                     params: Dict[str, Any]) -> bool:
        """Best-effort check that the observation itself shows the desired
        effect (e.g. the app name appears in the window list)."""
        if not observed:
            return False
        obs = observed.lower()
        if action == "desktop_open":
            app = str(params.get("app", "")).lower()
            return bool(app) and app in obs
        if action in ("browser_navigate", "browser_search"):
            url = str(params.get("url", "")).lower()
            query = str(params.get("query", "")).lower()
            return (url and url in obs) or (query and query in obs)
        return False

    def _goal_satisfied(self, state: TaskExecutionState) -> bool:
        """Explicit completion criteria: every step of the CURRENT plan is
        verified complete (or already satisfied) AND at least one step
        produced a verified effect AND no required step remains incomplete.

        NOTE: earlier failed ATTEMPTS that were superseded by a successful
        re-plan do NOT block completion — the goal is what matters, and it
        is satisfied by verified evidence from the current plan.
        """
        if not state.completed_steps:
            return False
        done_sigs = {s.signature() for s in state.completed_steps}
        for step in state.current_plan or []:
            sig = StepRecord(index=0, action=str(step.get("action", "")),
                             params=step.get("params") or {}).signature()
            if sig not in done_sigs:
                return False  # a required step is still incomplete
        return True

    @staticmethod
    def _partial_or_failed(state: TaskExecutionState) -> FinalStatus:
        if state.completed_steps:
            return FinalStatus.PARTIAL_FAILURE
        return FinalStatus.FAILED

    async def _replan(self, state: TaskExecutionState, request: str,
                      failure_context: str = "") -> Optional[List[Dict[str, Any]]]:
        """Re-plan from the CURRENT observed state (iterative, bounded)."""
        if not self._planner or not self._replans_left(state):
            return None
        state.replan_count += 1
        context = {
            "goal": state.normalized_goal,
            "observed_state": state.observed_state,
            "completed": [f"{s.action} {s.params}" for s in state.completed_steps],
            "failed": [f"{s.action}: {s.error}" for s in state.failed_steps],
            "failure_context": failure_context,
            "artifacts": {k: v for k, v in state.artifacts.items()
                          if k in ("urls", "last_url", "last_app", "last_query")},
            "replan": state.replan_count,
        }
        state.note(f"[REPLAN] state changed, replan {state.replan_count}"
                   f"/{self._limits.max_replans}, "
                   f"remaining goal: {state.normalized_goal[:60]}")
        await self._emit_event("task.replan", state,
                               replan=state.replan_count,
                               max_replans=self._limits.max_replans,
                               reason=failure_context or "plan exhausted")
        try:
            plan = await self._planner(request, context)
        except Exception as e:
            logger.warning("[TaskRunner] replan failed: %s", e)
            plan = None
        if not plan:
            return None
        loop_hit = self._loops.record_plan(plan)
        if loop_hit:
            state.blocker = loop_hit
            return None
        valid = self._validator.validate_plan(
            plan, state.completed_steps, self._transcript)
        if not valid:
            state.blocker = "re-plan produced no valid new actions"
            return None
        state.current_plan = valid
        state.plan_version += 1
        state.total_steps = max(state.total_steps,
                                len(state.completed_steps) + len(valid))
        state.note(f"[PLAN] v{state.plan_version} steps={len(valid)} (replan)")
        return valid


# ═══════════════════════════════════════════════════════════════
# Follow-up resolution + cross-turn task state
# ═══════════════════════════════════════════════════════════════

class FollowUpResolver:
    """Maps conversational follow-ups to continuations of the previous
    task, using stored task state instead of starting from zero."""

    _PATTERNS: List[Tuple[str, str]] = [
        (r"^(continue|go on|keep going|carry on|proceed|finish (the )?task|"
         r"resume (the )?task)\b", "continue"),
        (r"open (the )?(first|top) (result|link|hit|one)", "open_result"),
        (r"open (the )?(second) (result|link|hit|one)", "open_result"),
        (r"open (the )?(third) (result|link|hit|one)", "open_result"),
        (r"^(try another|try the next|next result|try again|retry)\b", "retry_last"),
        (r"(do the same for|same for|also for|now for)\s+([a-z0-9 \-]+)", "repeat_for"),
        (r"^(close that|close it|close the (app|window|program))\b", "close_last"),
        (r"^(cancel( the task)?|stop the task|abort( the task)?)\b", "cancel"),
    ]

    _ORDINALS = {"first": 0, "top": 0, "second": 1, "third": 2}

    @classmethod
    def match(cls, text: str) -> Optional[Tuple[str, Any]]:
        t = " ".join((text or "").lower().strip(" .!?").split())
        if not t:
            return None
        for pattern, kind in cls._PATTERNS:
            m = re.match(pattern, t)
            if not m:
                continue
            if kind == "open_result":
                ordinal = (m.group(2) or "first").strip().lower()
                index = cls._ORDINALS.get(ordinal, 0)
                return ("open_result", index)
            if kind == "repeat_for":
                target = (m.group(2) or "").strip()
                return (kind, target) if target else None
            return (kind, None)
        return None


class TaskStateStore:
    """Preserves task execution state across conversational follow-ups."""

    # Directory for lightweight JSON snapshots of finished task states
    # (diagnosis / interruption recovery). No database is introduced.
    TASK_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "tasks",
    )

    def __init__(self):
        self.last: Optional[TaskExecutionState] = None
        self.active: Optional[TaskExecutionState] = None  # resumable task
        self._active_runner = None                        # running TaskRunner

    def register_runner(self, runner) -> None:
        """Register the currently executing TaskRunner so it can be
        cooperatively cancelled by the Brain / UI / voice."""
        self._active_runner = runner

    def unregister_runner(self) -> None:
        """Clear the runner reference when the in-flight task ends."""
        self._active_runner = None

    def has_running_task(self) -> bool:
        return self._active_runner is not None

    # ── Lightweight JSON persistence (diagnosis / resume) ────

    def persist(self, state: TaskExecutionState) -> Optional[str]:
        """Snapshot a finished task state to JSON under data/tasks/.

        Returns the snapshot path, or None when persistence is disabled
        or fails. Persistence failure is never fatal for the task loop.
        """
        if os.environ.get("DIEGO_TASK_PERSIST", "1") == "0":
            return None
        if state is None:
            return None
        try:
            os.makedirs(self.TASK_DIR, exist_ok=True)
            path = os.path.join(
                self.TASK_DIR, f"task_{state.task_id}_{int(time.time())}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    _task_state_to_dict(state), fh,
                    ensure_ascii=False, indent=2, default=str)
            return path
        except Exception as e:
            logger.debug("[TaskState] persistence skipped: %s", e)
            return None

    def load(self, task_id: str) -> Optional[TaskExecutionState]:
        """Load the most recent snapshot for a task id (if any)."""
        if not os.path.isdir(self.TASK_DIR):
            return None
        try:
            candidates = []
            for name in os.listdir(self.TASK_DIR):
                if name.startswith(f"task_{task_id}_") and name.endswith(".json"):
                    candidates.append(os.path.join(self.TASK_DIR, name))
            if not candidates:
                return None
            latest = max(candidates, key=os.path.getmtime)
            with open(latest, "r", encoding="utf-8") as fh:
                return _task_state_from_dict(json.load(fh))
        except Exception as e:
            logger.debug("[TaskState] load skipped: %s", e)
            return None

    def save(self, state: TaskExecutionState) -> None:
        self.last = state
        if state.final_status in (FinalStatus.PARTIAL_FAILURE,
                                  FinalStatus.NEEDS_INPUT,
                                  FinalStatus.NEEDS_CONFIRMATION):
            self.active = state   # "continue" / "yes" can resume this
        else:
            self.active = None    # SUCCESS/FAILED/CANCELLED — keep last for refs
        # Lightweight JSON snapshot of the finished/paused state.
        self.persist(state)

    def cancel_active(self) -> bool:
        if self.active is not None:
            self.active.final_status = FinalStatus.CANCELLED
            self.active.note("[TASK] CANCELLED (by user)")
            self.active = None
            return True
        return False

    def cancel_active_runner(self) -> bool:
        """Request cooperative cancellation of a running TaskRunner.

        The runner checks the cancel flag between steps (never mid-action)
        and sets the authoritative CANCELLED final status itself, so this
        method can never leave the task in a fabricated SUCCESS state.
        """
        runner = getattr(self, "_active_runner", None)
        if runner is None:
            return False
        try:
            runner.cancel()
            return True
        except Exception:
            return False

    def build_continuation(
        self, text: str,
    ) -> Optional[Tuple[str, List[Dict[str, Any]], TaskExecutionState]]:
        """Build a continuation (request, plan, inherited state) for a
        follow-up command, or None if there is no prior task to continue."""
        if self.last is None:
            return None
        match = FollowUpResolver.match(text)
        if match is None:
            return None
        kind, arg = match
        prev = self.last

        if kind == "cancel":
            self.cancel_active()
            return ("cancel", [], prev)

        if kind == "continue":
            if self.active is None:
                return None
            remaining = self._remaining_steps(self.active)
            if not remaining:
                return None
            return (f"continue: {self.active.normalized_goal}",
                    remaining, self.active)

        if kind == "open_result":
            results = prev.artifacts.get("search_results") or prev.artifacts.get("urls") or []
            if not results:
                return None
            idx = min(int(arg or 0), len(results) - 1)
            url = results[idx]
            return (f"open the {['first', 'second', 'third'][idx]} result ({url})",
                    [{"action": "browser_navigate", "params": {"url": url},
                      "description": f"Open result {idx + 1}"}], prev)

        if kind == "retry_last":
            if not prev.failed_steps:
                return None
            failed = prev.failed_steps[-1]
            return (f"retry: {failed.action}",
                    [{"action": failed.action, "params": dict(failed.params),
                      "description": f"Retry {failed.action}"}], prev)

        if kind == "repeat_for":
            target = str(arg or "").strip()
            if not target:
                return None
            # Substitute the target into the previous plan's app params.
            plan = []
            for s in prev.completed_steps or []:
                if s.action == "desktop_open":
                    plan.append({"action": "desktop_open",
                                 "params": {"app": target},
                                 "description": f"Open {target} (same as before)"})
                    break
            if not plan:
                plan = [{"action": "desktop_open", "params": {"app": target},
                         "description": f"Open {target}"}]
            return (f"do the same for {target}", plan, prev)

        if kind == "close_last":
            app = prev.artifacts.get("last_app")
            if not app:
                return None
            return (f"close {app}",
                    [{"action": "close_app", "params": {"app": app},
                      "description": f"Close {app}"}], prev)

        return None

    @staticmethod
    def _remaining_steps(state: TaskExecutionState) -> List[Dict[str, Any]]:
        """Steps of the active plan that are not yet verified complete."""
        done_sigs = {s.signature() for s in state.completed_steps}
        remaining = []
        for step in state.current_plan or []:
            sig = StepRecord(index=0, action=str(step.get("action", "")),
                             params=step.get("params") or {}).signature()
            if sig not in done_sigs:
                remaining.append(step)
        return remaining


# ═══════════════════════════════════════════════════════════════
# JSON serialization for lightweight state persistence
# ═══════════════════════════════════════════════════════════════

def _evidence_to_dict(ev: Evidence) -> Dict[str, Any]:
    return {
        "fact": getattr(ev, "fact", ""),
        "source": getattr(ev, "source", EvidenceSource.UNKNOWN).value,
        "timestamp": getattr(ev, "timestamp", time.time()),
        "confidence": getattr(ev, "confidence", 1.0),
    }


def _evidence_from_dict(data: Dict[str, Any]) -> Evidence:
    return Evidence(
        fact=str(data.get("fact", "")),
        source=EvidenceSource(str(data.get("source", EvidenceSource.UNKNOWN.value))),
        timestamp=float(data.get("timestamp", time.time())),
        confidence=float(data.get("confidence", 1.0)),
    )


def _step_record_to_dict(rec: StepRecord) -> Dict[str, Any]:
    return {
        "index": getattr(rec, "index", 0),
        "action": getattr(rec, "action", ""),
        "params": dict(getattr(rec, "params", {}) or {}),
        "description": getattr(rec, "description", ""),
        "status": getattr(rec, "status", StepStatus.PENDING).value,
        "result": getattr(rec, "result", ""),
        "verification": getattr(rec, "verification", ""),
        "verified": bool(getattr(rec, "verified", False)),
        "retries": getattr(rec, "retries", 0),
        "error": getattr(rec, "error", ""),
        "latency_ms": getattr(rec, "latency_ms", 0.0),
        "evidence_source": getattr(rec, "evidence_source", EvidenceSource.UNKNOWN).value,
        "sensitive_reason": getattr(rec, "sensitive_reason", ""),
    }


def _step_record_from_dict(data: Dict[str, Any]) -> StepRecord:
    return StepRecord(
        index=int(data.get("index", 0)),
        action=str(data.get("action", "")),
        params=dict(data.get("params") or {}),
        description=str(data.get("description", "")),
        status=StepStatus(str(data.get("status", StepStatus.PENDING.value))),
        result=str(data.get("result", "")),
        verification=str(data.get("verification", "")),
        verified=bool(data.get("verified", False)),
        retries=int(data.get("retries", 0)),
        error=str(data.get("error", "")),
        latency_ms=float(data.get("latency_ms", 0.0)),
        evidence_source=EvidenceSource(
            str(data.get("evidence_source", EvidenceSource.UNKNOWN.value))),
        sensitive_reason=str(data.get("sensitive_reason", "")),
    )


def _task_state_to_dict(state: TaskExecutionState) -> Dict[str, Any]:
    """Serialize a TaskExecutionState into a JSON-able dict.

    Used for lightweight persistence of task state so interrupted tasks
    can be diagnosed (and resumed when supported) without a database.
    """
    return {
        "task_id": getattr(state, "task_id", ""),
        "original_request": getattr(state, "original_request", ""),
        "normalized_goal": getattr(state, "normalized_goal", ""),
        "current_plan": list(getattr(state, "current_plan", []) or []),
        "completed_steps": [_step_record_to_dict(r) for r in getattr(state, "completed_steps", []) or []],
        "current_step": (_step_record_to_dict(state.current_step)
                         if getattr(state, "current_step", None) is not None else None),
        "failed_steps": [_step_record_to_dict(r) for r in getattr(state, "failed_steps", []) or []],
        "observed_state": getattr(state, "observed_state", ""),
        "verification_results": list(getattr(state, "verification_results", []) or []),
        "retry_count": getattr(state, "retry_count", 0),
        "replan_count": getattr(state, "replan_count", 0),
        "total_steps": getattr(state, "total_steps", 0),
        "final_status": (state.final_status.value
                         if getattr(state, "final_status", None) is not None else None),
        "plan_version": getattr(state, "plan_version", 0),
        "started_at": getattr(state, "started_at", time.time()),
        "ended_at": getattr(state, "ended_at", None),
        "artifacts": dict(getattr(state, "artifacts", {}) or {}),
        "log": list(getattr(state, "log", []) or []),
        "blocker": getattr(state, "blocker", ""),
        "evidence_log": [_evidence_to_dict(ev) for ev in getattr(state, "evidence_log", []) or []],
        "pending_confirmation": (_step_record_to_dict(state.pending_confirmation)
                                 if getattr(state, "pending_confirmation", None) is not None else None),
        "confirmation_reason": getattr(state, "confirmation_reason", ""),
    }


def _task_state_from_dict(data: Dict[str, Any]) -> TaskExecutionState:
    """Rebuild a TaskExecutionState from a JSON-able dict."""
    state = TaskExecutionState(
        task_id=str(data.get("task_id", "")),
        original_request=str(data.get("original_request", "")),
        normalized_goal=str(data.get("normalized_goal", "")),
        current_plan=list(data.get("current_plan") or []),
        completed_steps=[_step_record_from_dict(d) for d in data.get("completed_steps") or []],
        current_step=(_step_record_from_dict(data["current_step"])
                      if data.get("current_step") else None),
        failed_steps=[_step_record_from_dict(d) for d in data.get("failed_steps") or []],
        observed_state=str(data.get("observed_state", "")),
        verification_results=list(data.get("verification_results") or []),
        retry_count=int(data.get("retry_count", 0)),
        replan_count=int(data.get("replan_count", 0)),
        total_steps=int(data.get("total_steps", 0)),
        final_status=(FinalStatus(str(data["final_status"]))
                      if data.get("final_status") else None),
        plan_version=int(data.get("plan_version", 0)),
        started_at=float(data.get("started_at", time.time())),
        ended_at=(float(data["ended_at"]) if data.get("ended_at") is not None else None),
        artifacts=dict(data.get("artifacts") or {}),
        log=list(data.get("log") or []),
        blocker=str(data.get("blocker", "")),
        evidence_log=[_evidence_from_dict(d) for d in data.get("evidence_log") or []],
        pending_confirmation=(_step_record_from_dict(data["pending_confirmation"])
                              if data.get("pending_confirmation") else None),
        confirmation_reason=str(data.get("confirmation_reason", "")),
    )
    return state


# Global singleton — task state survives across conversational turns.
task_state_store = TaskStateStore()
