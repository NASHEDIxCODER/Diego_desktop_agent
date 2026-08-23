"""
ProactiveAgent — Event-driven proactive helper.

Diego monitors desktop events and proactively offers help:

  - Tests complete → summarize failures
  - Compilation failed → explain error
  - Git merge conflict → offer repair
  - Server crashed → offer restart
  - Download finished → notify
  - Long build finished → read results
  - Battery low → suggest power-saving
  - High CPU/memory → suggest cleanup
  - App crashed → offer to restart
  - Error in IDE → suggest fix

The ProactiveAgent subscribes to the EventBus and listens for
desktop events published by the DesktopObserver and other services.
When a trigger condition is detected, it generates a suggestion
and publishes a "proactive:suggestion" event.

Architecture:
    DesktopObserver → EventBus → ProactiveAgent → suggestion → Conversation/Brain

Usage:
    from agent.proactive_agent import proactive_agent
    await proactive_agent.start(event_bus)

Events subscribed:
    desktop:clipboard_changed  → detect build/compile output
    desktop:download_complete  → notify
    desktop:battery_low        → warn
    desktop:git_changed        → detect branch switching
    proactive:suggestion       → broadcast suggestions
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Trigger Conditions
# ═══════════════════════════════════════════════════════════════

@dataclass
class Suggestion:
    """A proactive suggestion from Diego to the user."""
    type: str  # "info", "warning", "error", "action"
    title: str
    description: str
    suggested_action: Optional[str] = None
    urgency: str = "low"  # low, medium, high, critical
    source_event: str = ""
    timestamp: float = field(default_factory=time.time)
    auto_dismiss_s: float = 30.0  # Seconds before auto-dismiss (0 = manual)
    id: str = ""


# ═══════════════════════════════════════════════════════════════
# Detection Patterns
# ═══════════════════════════════════════════════════════════════

# Compilation error patterns
COMPILE_ERROR_PATTERNS = [
    (re.compile(r"error[:\s]+(.+)", re.IGNORECASE), "error"),
    (re.compile(r"Error: (.+)"), "error"),
    (re.compile(r"FAILED: (.+)"), "error"),
    (re.compile(r"FAIL: (.+)"), "error"),
    (re.compile(r"PANIC: (.+)"), "error"),
    (re.compile(r"panic: (.+)"), "error"),
    (re.compile(r"undefined reference to (.+)"), "linker_error"),
    (re.compile(r"Cannot find module (.+)"), "import_error"),
    (re.compile(r"ModuleNotFoundError: (.+)"), "import_error"),
    (re.compile(r"ImportError: (.+)"), "import_error"),
]

# Test output patterns
TEST_PATTERNS = [
    (re.compile(r"(\d+) passed"), "passed"),
    (re.compile(r"(\d+) failed"), "failed"),
    (re.compile(r"FAILED \((.+?)\)"), "test_fail"),
    (re.compile(r"assert (.+?)$", re.MULTILINE), "assertion"),
    (re.compile(r"PASSED"), "all_pass"),
]

# Git conflict patterns
MERGE_CONFLICT_PATTERNS = [
    (re.compile(r"CONFLICT \((.+?)\)", re.IGNORECASE), "merge_conflict"),
    (re.compile(r"<<<<<<< HEAD"), "conflict_marker"),
    (re.compile(r"Automatic merge failed"), "merge_failed"),
]

# Server crash patterns
SERVER_CRASH_PATTERNS = [
    (re.compile(r"segmentation fault", re.IGNORECASE), "segfault"),
    (re.compile(r"core dump", re.IGNORECASE), "core_dump"),
    (re.compile(r"Killed"), "killed"),
    (re.compile(r"out of memory", re.IGNORECASE), "oom"),
    (re.compile(r"connection refused", re.IGNORECASE), "connection"),
    (re.compile(r"bind: address already in use", re.IGNORECASE), "port_conflict"),
    (re.compile(r"Address already in use"), "port_conflict"),
]

# Build patterns
BUILD_PATTERNS = [
    (re.compile(r"BUILD SUCCESS"), "build_success"),
    (re.compile(r"BUILD FAILED"), "build_failed"),
    (re.compile(r"Build finished in (.+)"), "build_finished"),
    (re.compile(r"Compilation successful"), "compilation_success"),
    (re.compile(r"Compilation failed"), "compilation_failed"),
]

# Success patterns
SUCCESS_PATTERNS = [
    (re.compile(r"done in (\d+\.?\d*)s"), "done"),
    (re.compile(r"Successfully (.+)"), "success"),
    (re.compile(r"Installed (.+)"), "installed"),
    (re.compile(r"Deployed (.+)"), "deployed"),
]


# ═══════════════════════════════════════════════════════════════
# ProactiveAgent
# ═══════════════════════════════════════════════════════════════

class ProactiveAgent:
    """
    Event-driven proactive helper.

    Subscribes to desktop events and generates suggestions when
    it detects situations where Diego can help.

    Suggestions are published to the EventBus and can be consumed
    by the Conversation Engine, Brain, or UI layers.
    """

    def __init__(self):
        self._event_bus = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._suggestions: List[Suggestion] = []
        self._suggestion_counter = 0
        self._suggestion_callback: Optional[Callable[[Suggestion], Any]] = None

        # Cooldown tracking to avoid spamming
        self._last_suggestion_time: Dict[str, float] = {}
        self._cooldown_s: float = 30.0  # Seconds between same-type suggestions

        # State tracking
        self._last_clipboard_hash: str = ""
        self._last_git_branch: str = ""
        self._last_build_output: str = ""
        self._last_test_output: str = ""
        self._detected_state: Dict[str, bool] = {}

    # ── Wiring ─────────────────────────────────────────────────

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    def set_suggestion_callback(self, fn: Callable[[Suggestion], Any]) -> None:
        """
        Wire a callback that receives proactive suggestions.
        This is how the Conversation Engine or Brain consumes suggestions.
        """
        self._suggestion_callback = fn

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self, event_bus=None) -> None:
        """Start the proactive agent and subscribe to events."""
        if self._running:
            return

        if event_bus:
            self._event_bus = event_bus

        if self._event_bus:
            # Subscribe to relevant event types
            self._event_bus.on("desktop:clipboard_changed", self._on_clipboard_changed)
            self._event_bus.on("desktop:download_complete", self._on_download_complete)
            self._event_bus.on("desktop:battery_low", self._on_battery_low)
            self._event_bus.on("desktop:git_changed", self._on_git_changed)
            self._event_bus.on("desktop:window_changed", self._on_window_changed)
            self._event_bus.on("goal:completed", self._on_goal_completed)
            self._event_bus.on("goal:failed", self._on_goal_failed)
            self._event_bus.on("task:completed", self._on_task_completed)
            self._event_bus.on("task:failed", self._on_task_failed)
            self._event_bus.on("desktop:audio_changed", self._on_audio_changed)

        self._running = True
        logger.info("[Proactive] ProactiveAgent started")

    async def stop(self) -> None:
        """Stop the proactive agent."""
        self._running = False
        logger.info("[Proactive] ProactiveAgent stopped")

    # ── Event Handlers ─────────────────────────────────────────

    async def _on_clipboard_changed(self, event: Any) -> None:
        """Detect build/test/error output in clipboard."""
        if not self._running:
            return

        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        text_preview = data.get("text_preview", "")
        if not text_preview or len(text_preview) < 10:
            return

        # Detect compilation errors
        for pattern, error_type in COMPILE_ERROR_PATTERNS:
            match = pattern.search(text_preview)
            if match:
                error_text = match.group(1).strip() if match.lastindex else match.group(0)
                if not self._is_cooldown("compile_error"):
                    await self._suggest(
                        type="error",
                        title="Compilation Error Detected",
                        description=f"I noticed this error in your output: {error_text[:200]}",
                        suggested_action="Would you like me to explain this error and suggest a fix?",
                        urgency="high",
                        source_event="desktop:clipboard_changed",
                        auto_dismiss_s=60.0,
                    )
                    self._record_cooldown("compile_error")
                return

        # Detect test failures
        for pattern, match_type in TEST_PATTERNS:
            match = pattern.search(text_preview)
            if match and match_type == "failed":
                if not self._is_cooldown("test_failure"):
                    failures = match.group(1)
                    await self._suggest(
                        type="warning",
                        title="Test Failures Detected",
                        description=f"I noticed {failures} test failure(s) in the output.",
                        suggested_action="Would you like me to read the failure details and suggest fixes?",
                        urgency="medium",
                        source_event="desktop:clipboard_changed",
                        auto_dismiss_s=45.0,
                    )
                    self._record_cooldown("test_failure")
                    return

        # Detect merge conflicts
        for pattern, match_type in MERGE_CONFLICT_PATTERNS:
            if pattern.search(text_preview):
                if not self._is_cooldown("merge_conflict"):
                    await self._suggest(
                        type="error",
                        title="Git Merge Conflict",
                        description="I detected a merge conflict. Files have conflicting changes.",
                        suggested_action="Would you like me to help resolve this merge conflict?",
                        urgency="high",
                        source_event="desktop:clipboard_changed",
                        auto_dismiss_s=60.0,
                    )
                    self._record_cooldown("merge_conflict")
                    return

        # Detect server crashes
        for pattern, match_type in SERVER_CRASH_PATTERNS:
            if pattern.search(text_preview):
                crash_type = match_type.replace("_", " ").title()
                if not self._is_cooldown("server_crash"):
                    await self._suggest(
                        type="error",
                        title=f"Server Issue: {crash_type}",
                        description=f"I detected a {crash_type}. The server or process may have crashed.",
                        suggested_action="Would you like me to restart the service or help debug?",
                        urgency="critical",
                        source_event="desktop:clipboard_changed",
                        auto_dismiss_s=0,
                    )
                    self._record_cooldown("server_crash")
                    return

        # Detect build completion
        for pattern, match_type in BUILD_PATTERNS:
            match = pattern.search(text_preview)
            if match:
                if match_type in ("build_failed", "compilation_failed"):
                    if not self._is_cooldown("build_failed"):
                        await self._suggest(
                            type="error",
                            title="Build Failed",
                            description="The build failed. There may be errors to review.",
                            suggested_action="Would you like me to read the build output and identify the errors?",
                            urgency="medium",
                            source_event="desktop:clipboard_changed",
                            auto_dismiss_s=45.0,
                        )
                        self._record_cooldown("build_failed")
                elif match_type in ("build_success", "build_finished", "compilation_success"):
                    if not self._is_cooldown("build_success"):
                        await self._suggest(
                            type="info",
                            title="Build Complete",
                            description="The build finished successfully.",
                            suggested_action=None,
                            urgency="low",
                            source_event="desktop:clipboard_changed",
                            auto_dismiss_s=10.0,
                        )
                        self._record_cooldown("build_success")
                return

        # Detect done/success
        for pattern, match_type in SUCCESS_PATTERNS:
            match = pattern.search(text_preview)
            if match and match_type == "done":
                time_taken = match.group(1) if match.lastindex else ""
                if time_taken:
                    try:
                        t = float(time_taken)
                        if t > 30:  # Long-running task
                            if not self._is_cooldown("long_task"):
                                await self._suggest(
                                    type="info",
                                    title="Long Task Completed",
                                    description=f"A task completed in {time_taken}s.",
                                    suggested_action=None,
                                    urgency="low",
                                    source_event="desktop:clipboard_changed",
                                    auto_dismiss_s=10.0,
                                )
                                self._record_cooldown("long_task")
                    except ValueError:
                        pass
                return

    async def _on_download_complete(self, event: Any) -> None:
        """Notify when a download completes."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        filename = data.get("filename", "")
        if filename and not self._is_cooldown(f"download:{filename}"):
            await self._suggest(
                type="info",
                title="Download Complete",
                description=f"File '{filename}' has finished downloading.",
                suggested_action=None,
                urgency="low",
                source_event="desktop:download_complete",
                auto_dismiss_s=10.0,
            )
            self._record_cooldown(f"download:{filename}")

    async def _on_battery_low(self, event: Any) -> None:
        """Warn when battery is low."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        percent = data.get("percent", 0)
        if percent <= 10 and not self._is_cooldown("battery_critical"):
            await self._suggest(
                type="warning",
                title=f"Battery Critical: {percent}%",
                description="Your battery is critically low. You should save your work.",
                suggested_action="Would you like me to suggest power-saving measures?",
                urgency="critical",
                source_event="desktop:battery_low",
                auto_dismiss_s=0,
            )
            self._record_cooldown("battery_critical")
        elif percent <= 15 and not self._is_cooldown("battery_low"):
            await self._suggest(
                type="warning",
                title=f"Battery Low: {percent}%",
                description="Your battery is getting low.",
                suggested_action=None,
                urgency="medium",
                source_event="desktop:battery_low",
                auto_dismiss_s=15.0,
            )
            self._record_cooldown("battery_low")

    async def _on_git_changed(self, event: Any) -> None:
        """React to git branch changes."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        branch = data.get("branch", "")
        previous = data.get("previous", "")

        if branch and previous and not self._is_cooldown("git_switch"):
            # Context switching detected
            logger.info("[Proactive] Git branch changed: %s → %s", previous, branch)

    async def _on_window_changed(self, event: Any) -> None:
        """React to window focus changes."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        app = data.get("application", "")
        title = data.get("title", "")

        # Detect IDE window
        ide_apps = ("pycharm", "code", "vim", "nvim", "emacs", "sublime_text",
                     "atom", "intellij", "android-studio", "rider", "rstudio")
        if app.lower() in ide_apps:
            self._detected_state["in_ide"] = True

        # Detect terminal
        terminal_apps = ("gnome-terminal", "konsole", "alacritty", "kitty",
                          "wezterm", "xfce4-terminal", "terminator", "tilix")
        if app.lower() in terminal_apps:
            self._detected_state["in_terminal"] = True

    async def _on_goal_completed(self, event: Any) -> None:
        """React to goal completion."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        goal_desc = data.get("description", "")
        tasks_completed = data.get("tasks_completed", 0)

        if not self._is_cooldown("goal_complete"):
            await self._suggest(
                type="info",
                title="Goal Completed",
                description=f"Goal '{goal_desc}' completed with {tasks_completed} tasks.",
                suggested_action=None,
                urgency="low",
                source_event="goal:completed",
                auto_dismiss_s=10.0,
            )
            self._record_cooldown("goal_complete")

    async def _on_goal_failed(self, event: Any) -> None:
        """React to goal failure."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        error = data.get("error", "")
        if error and not self._is_cooldown("goal_failed"):
            await self._suggest(
                type="error",
                title="Goal Failed",
                description=f"A goal failed: {error[:200]}",
                suggested_action="Would you like me to retry or adjust the approach?",
                urgency="medium",
                source_event="goal:failed",
                auto_dismiss_s=30.0,
            )
            self._record_cooldown("goal_failed")

    async def _on_task_completed(self, event: Any) -> None:
        """React to task completion."""
        pass  # Logged by the Brain; silence unless slow

    async def _on_task_failed(self, event: Any) -> None:
        """React to task failure."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        error = data.get("error", "")
        if error and "timeout" not in error.lower():
            if not self._is_cooldown("task_failed"):
                await self._suggest(
                    type="warning",
                    title="Task Failed",
                    description=f"A task failed: {error[:200]}",
                    suggested_action="Would you like me to retry?",
                    urgency="medium",
                    source_event="task:failed",
                    auto_dismiss_s=30.0,
                )
                self._record_cooldown("task_failed")

    async def _on_audio_changed(self, event: Any) -> None:
        """React to audio state changes."""
        data = event.data if hasattr(event, 'data') else event
        if not isinstance(data, dict):
            return

        volume = data.get("volume", 50)
        muted = data.get("muted", False)

        if muted and not self._is_cooldown("audio_muted"):
            # Don't proactively warn about muting, it's usually intentional
            pass

    # ── Suggestion System ──────────────────────────────────────

    async def _suggest(self, **kwargs) -> Optional[Suggestion]:
        """Create and publish a proactive suggestion."""
        self._suggestion_counter += 1
        suggestion = Suggestion(
            id=f"suggestion_{self._suggestion_counter}_{int(time.time())}",
            **{k: v for k, v in kwargs.items() if k in Suggestion.__dataclass_fields__},
        )

        self._suggestions.append(suggestion)
        if len(self._suggestions) > 100:
            self._suggestions = self._suggestions[-50:]

        logger.info("[Proactive] Suggestion (%s): %s — %s",
                     suggestion.urgency, suggestion.title, suggestion.description[:100])

        # Publish to EventBus
        if self._event_bus:
            try:
                await self._event_bus.emit("proactive:suggestion", {
                    "id": suggestion.id,
                    "type": suggestion.type,
                    "title": suggestion.title,
                    "description": suggestion.description,
                    "suggested_action": suggestion.suggested_action,
                    "urgency": suggestion.urgency,
                    "source_event": suggestion.source_event,
                    "timestamp": suggestion.timestamp,
                    "auto_dismiss_s": suggestion.auto_dismiss_s,
                }, source="proactive_agent")
            except Exception as e:
                logger.debug("[Proactive] EventBus publish failed: %s", e)

        # Dispatch to callback
        if self._suggestion_callback:
            try:
                self._suggestion_callback(suggestion)
            except Exception as e:
                logger.debug("[Proactive] Suggestion callback failed: %s", e)

        return suggestion

    # ── Cooldown Management ────────────────────────────────────

    def _is_cooldown(self, key: str) -> bool:
        """Check if a suggestion type is in cooldown."""
        last = self._last_suggestion_time.get(key, 0)
        return (time.time() - last) < self._cooldown_s

    def _record_cooldown(self, key: str) -> None:
        """Record that a suggestion type was just sent."""
        self._last_suggestion_time[key] = time.time()

    # ── Suggestion Management ──────────────────────────────────

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Dismiss a suggestion by ID."""
        for s in self._suggestions:
            if s.id == suggestion_id:
                self._suggestions.remove(s)
                return True
        return False

    def get_active_suggestions(self) -> List[Suggestion]:
        """Return currently active (non-expired) suggestions."""
        now = time.time()
        active = []
        for s in self._suggestions:
            if s.auto_dismiss_s == 0 or (now - s.timestamp) < s.auto_dismiss_s:
                active.append(s)
        return active

    def clear(self) -> None:
        """Clear all suggestions."""
        self._suggestions.clear()
        self._last_suggestion_time.clear()

    # ── Direct Suggestion Methods ──────────────────────────────

    async def suggest_test_summary(self, passed: int, failed: int, errors: List[str]) -> None:
        """Suggest a summary after test completion."""
        if failed > 0:
            await self._suggest(
                type="warning",
                title=f"Tests: {passed} passed, {failed} failed",
                description=f"Your tests have {failed} failure(s).",
                suggested_action="Would you like me to summarize the failures?",
                urgency="medium" if failed > 2 else "low",
                source_event="test_complete",
                auto_dismiss_s=45.0,
            )
        else:
            await self._suggest(
                type="info",
                title=f"All {passed} Tests Passed",
                description="All tests passed successfully.",
                suggested_action=None,
                urgency="low",
                source_event="test_complete",
                auto_dismiss_s=10.0,
            )

    async def suggest_compile_error(self, error_message: str) -> None:
        """Suggest help with a compilation error."""
        await self._suggest(
            type="error",
            title="Compilation Error",
            description=f"I noticed this error: {error_message[:200]}",
            suggested_action="Would you like me to explain this error?",
            urgency="high",
            source_event="compile_error",
            auto_dismiss_s=60.0,
        )

    async def suggest_merge_conflict(self, files: List[str]) -> None:
        """Suggest help with a merge conflict."""
        file_list = ", ".join(files[:3])
        await self._suggest(
            type="error",
            title="Merge Conflict",
            description=f"Conflicts detected in: {file_list}",
            suggested_action="Would you like me to help resolve this?",
            urgency="high",
            source_event="merge_conflict",
            auto_dismiss_s=0,
        )

    async def suggest_server_crash(self, process_name: str) -> None:
        """Suggest help after a server crash."""
        await self._suggest(
            type="error",
            title="Server Crashed",
            description=f"{process_name} appears to have crashed.",
            suggested_action="Would you like me to restart it?",
            urgency="critical",
            source_event="server_crash",
            auto_dismiss_s=0,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def close(self) -> None:
        self._running = False
        logger.info("[Proactive] ProactiveAgent shut down")


# Global singleton
proactive_agent = ProactiveAgent()