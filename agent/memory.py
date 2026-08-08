"""
AgentMemory — Tracks agent state across multi-step tasks.

Remembers:
- Active tab and browser state
- Last action performed
- Conversation context
- Current task and progress
- Pending confirmations
- Error history for recovery
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TaskStep:
    """A single step in a multi-step task."""
    action: str
    target: str
    status: str = "pending"  # pending, running, done, failed
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class AgentTask:
    """A multi-step task being executed by the agent."""
    id: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    current_step: int = 0
    status: str = "pending"  # pending, running, done, failed
    error: Optional[str] = None


class AgentMemory:
    """
    Tracks agent state across multi-step tasks.

    Provides:
    - Task tracking with step-by-step progress
    - Browser state (active tab, URL, session)
    - Last action for recovery
    - Conversation context
    - Error history
    """

    def __init__(self):
        self._current_task: Optional[AgentTask] = None
        self._task_history: List[AgentTask] = []
        self._last_action: Optional[str] = None
        self._last_screenshot: Optional[str] = None
        self._browser_url: Optional[str] = None
        self._browser_tabs: List[str] = []
        self._active_tab_index: int = 0
        self._conversation: List[Dict[str, str]] = []
        self._error_history: List[Dict[str, Any]] = []
        self._pending_confirmation: Optional[str] = None

    # ── Task Management ─────────────────────────────────

    def start_task(self, description: str) -> str:
        """Start a new task. Returns task ID."""
        task_id = f"task_{int(time.time())}_{len(self._task_history)}"
        self._current_task = AgentTask(
            id=task_id,
            description=description,
        )
        logger.info("Agent task started: %s — %s", task_id, description)
        return task_id

    def add_step(self, action: str, target: str) -> None:
        """Add a step to the current task."""
        if self._current_task is None:
            logger.warning("No active task to add step to")
            return
        step = TaskStep(action=action, target=target)
        self._current_task.steps.append(step)
        logger.debug("Task step added: %s → %s", action, target)

    def start_step(self) -> None:
        """Mark the current step as running."""
        if self._current_task is None:
            return
        step = self._current_step
        if step:
            step.status = "running"
            step.started_at = time.time()
            logger.info("Step %d/%d: %s %s",
                       self._current_task.current_step + 1,
                       len(self._current_task.steps),
                       step.action, step.target)

    def complete_step(self, result: Optional[str] = None) -> None:
        """Mark the current step as completed."""
        if self._current_task is None:
            return
        step = self._current_step
        if step:
            step.status = "done"
            step.result = result
            step.completed_at = time.time()
            self._current_task.current_step += 1
            logger.info("Step completed: %s", result or "ok")

    def fail_step(self, error: str) -> None:
        """Mark the current step as failed."""
        if self._current_task is None:
            return
        step = self._current_step
        if step:
            step.status = "failed"
            step.error = error
            step.completed_at = time.time()
            self._error_history.append({
                "step": self._current_task.current_step,
                "action": step.action,
                "error": error,
                "time": time.time(),
            })
            logger.warning("Step failed: %s", error)

    def complete_task(self, result: Optional[str] = None) -> None:
        """Mark the current task as completed."""
        if self._current_task is None:
            return
        self._current_task.status = "done"
        self._current_task.error = result
        self._task_history.append(self._current_task)
        logger.info("Agent task completed: %s", result or "done")
        self._current_task = None

    def fail_task(self, error: str) -> None:
        """Mark the current task as failed."""
        if self._current_task is None:
            return
        self._current_task.status = "failed"
        self._current_task.error = error
        self._task_history.append(self._current_task)
        logger.error("Agent task failed: %s", error)
        self._current_task = None

    # ── State Queries ───────────────────────────────────

    @property
    def has_active_task(self) -> bool:
        return self._current_task is not None

    @property
    def current_task(self) -> Optional[AgentTask]:
        return self._current_task

    @property
    def _current_step(self) -> Optional[TaskStep]:
        if self._current_task and self._current_task.steps:
            idx = self._current_task.current_step
            if idx < len(self._current_task.steps):
                return self._current_task.steps[idx]
        return None

    @property
    def current_step_description(self) -> Optional[str]:
        step = self._current_step
        if step:
            return f"{step.action} {step.target}"
        return None

    @property
    def task_progress(self) -> str:
        if not self._current_task:
            return "No active task"
        total = len(self._current_task.steps)
        done = sum(1 for s in self._current_task.steps if s.status == "done")
        return f"Step {done}/{total}"

    # ── Browser State ───────────────────────────────────

    def set_browser_url(self, url: str) -> None:
        self._browser_url = url
        logger.debug("Browser URL: %s", url)

    def set_browser_tabs(self, tabs: List[str]) -> None:
        self._browser_tabs = tabs

    def set_active_tab(self, index: int) -> None:
        self._active_tab_index = index

    @property
    def browser_url(self) -> Optional[str]:
        return self._browser_url

    @property
    def browser_tabs(self) -> List[str]:
        return self._browser_tabs

    # ── Last Action ─────────────────────────────────────

    def set_last_action(self, action: str) -> None:
        self._last_action = action

    @property
    def last_action(self) -> Optional[str]:
        return self._last_action

    # ── Screenshot ──────────────────────────────────────

    def set_last_screenshot(self, path: str) -> None:
        self._last_screenshot = path

    @property
    def last_screenshot(self) -> Optional[str]:
        return self._last_screenshot

    # ── Conversation ────────────────────────────────────

    def add_conversation(self, role: str, message: str) -> None:
        self._conversation.append({"role": role, "message": message, "time": time.time()})
        # Keep last 20 messages
        if len(self._conversation) > 20:
            self._conversation = self._conversation[-20:]

    @property
    def conversation(self) -> List[Dict[str, str]]:
        return list(self._conversation)

    # ── Confirmation ────────────────────────────────────

    def request_confirmation(self, prompt: str) -> None:
        self._pending_confirmation = prompt
        logger.info("Confirmation requested: %s", prompt)

    @property
    def pending_confirmation(self) -> Optional[str]:
        return self._pending_confirmation

    def confirm(self) -> None:
        self._pending_confirmation = None

    def reject(self) -> None:
        self._pending_confirmation = None
        if self._current_task:
            self.fail_task("User rejected confirmation")

    # ── Error Recovery ──────────────────────────────────

    @property
    def recent_errors(self) -> List[Dict[str, Any]]:
        return self._error_history[-5:]  # Last 5 errors

    def clear(self) -> None:
        """Reset all memory."""
        self._current_task = None
        self._last_action = None
        self._last_screenshot = None
        self._browser_url = None
        self._browser_tabs = []
        self._pending_confirmation = None
        logger.debug("Agent memory cleared")


# Global singleton
agent_memory = AgentMemory()