"""
BackgroundAgent — Leo's proactive observation system.

Leo should OBSERVE and ACT without being asked:

    Compilation finished.   → Notify user.
    Git push failed.        → Offer fix.
    Download completed.     → Announce.
    Battery low.            → Warn.

The background agent subscribes to perception events and runs rule-based
and LLM-assisted observers:

    * Terminal observer   — watch for compilation/build/process events
    * Git observer        — detect failed commits/pushes
    * Download observer   — detect download completion
    * Battery observer    — low battery warnings
    * Notification hook   — forward desktop notifications to Leo

Rules are lightweight observers with a `should_notify(data) -> bool`
and a `format_message(data) -> str`. This keeps the agent fast and
fully local (no LLM needed per event).
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from core.event_bus import bus, Event
from core.service import BaseService

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Rules
# ═══════════════════════════════════════════════════════════

class ObserverRule:
    """A background observation rule."""

    def __init__(self, name: str, event_pattern: str,
                 test: Callable[[dict], bool],
                 message: Callable[[dict], str]):
        self.name = name
        self.event_pattern = event_pattern
        self._test = test
        self._message = message

    def matches(self, event_type: str) -> bool:
        import fnmatch
        return fnmatch.fnmatchcase(event_type, self.event_pattern)

    def should_notify(self, data: dict) -> bool:
        try:
            return bool(self._test(data))
        except Exception:
            return False

    def format_message(self, data: dict) -> str:
        try:
            return self._message(data)
        except Exception as e:
            logger.debug("[BGAGENT] message formatting error: %s", e)
            return f"[{self.name}] — something happened."


# ═══════════════════════════════════════════════════════════
# Built-in rules
# ═══════════════════════════════════════════════════════════

def _terminal_compile_rule(data: dict) -> bool:
    """Detect compilation success/failure in terminal output."""
    text = (data.get("text", "") or data.get("output", "") or "").lower()
    return any(k in text for k in [
        "build succeeded", "compilation succeeded",
        "build failed", "compilation failed", "error: ",
        "tests passed", "tests failed", "process finished with exit code",
    ])


def _terminal_compile_msg(data: dict) -> str:
    text = (data.get("text", "") or data.get("output", "") or "").lower()
    if "build succeeded" in text or "compilation succeeded" in text:
        return "Your build succeeded."
    if "build failed" in text or "compilation failed" in text:
        return "Your build failed — want me to look at the error?"
    if "tests passed" in text:
        return "All tests passed."
    if "tests failed" in text:
        return "Some tests failed — want me to check?"
    if "error:" in text:
        return "I noticed an error in your terminal."
    if "process finished with exit code" in text:
        code = text.split("exit code")[-1].strip()
        return f"A process just finished with exit code {code}."
    return "I noticed something in your terminal."


def _git_failure_rule(data: dict) -> bool:
    """Detect git push/commit failures."""
    text = (data.get("text", "") or data.get("output", "") or "").lower()
    return any(k in text for k in [
        "failed to push", "push rejected", "!! [rejected]",
        "error: failed to push", "cannot lock ref",
        "fatal: unable to access",
    ])


def _git_failure_msg(data: dict) -> str:
    return "Your git push failed — want me to look at the error and fix it?"


def _battery_low_rule(data: dict) -> bool:
    """Detect low battery."""
    pct = data.get("percent", 100)
    return pct is not None and float(pct) < 20


def _battery_low_msg(data: dict) -> str:
    pct = int(float(data.get("percent", 0)))
    return f"Heads up — your battery is at {pct} percent."


def _download_rule(data: dict) -> bool:
    """Detect download completion."""
    text = (data.get("text", "") or "").lower()
    return "download complete" in text or "download finished" in text


def _download_msg(data: dict) -> str:
    return "Your download has finished."


def _notification_forward_rule(data: dict) -> bool:
    """Forward significant desktop notifications."""
    app = (data.get("app", "") or "").lower()
    summary = (data.get("summary", "") or "").lower()
    # Skip Leo's own notifications to avoid loops
    if "leo" in app or "leo" in summary:
        return False
    return bool(summary)


def _notification_forward_msg(data: dict) -> str:
    app = data.get("app", "")
    summary = data.get("summary", "")
    return f"Notification from {app}: {summary}"


# ═══════════════════════════════════════════════════════════
# BackgroundAgent
# ═══════════════════════════════════════════════════════════

class BackgroundAgent(BaseService):
    """
    Proactive background agent.

    Subscribes to perception/notification events and emits
    `background.notify` (with a voice-ready message) when a rule fires.
    """

    name = "background_agent"

    def __init__(self):
        super().__init__()
        self._rules: List[ObserverRule] = []
        self._notified: Dict[str, float] = {}  # rate-limit: rule → last time
        self._rate_limit_s: float = 60.0
        self._last_notify: Optional[str] = None

    async def _start(self) -> bool:
        """Install built-in rules and subscribe."""
        self.add_rule(ObserverRule(
            "terminal.compile", "terminal.output",
            _terminal_compile_rule, _terminal_compile_msg))
        self.add_rule(ObserverRule(
            "terminal.compile", "perception.ocr",
            _terminal_compile_rule, _terminal_compile_msg))
        self.add_rule(ObserverRule(
            "git.failure", "git.output",
            _git_failure_rule, _git_failure_msg))
        self.add_rule(ObserverRule(
            "battery.low", "system.battery",
            _battery_low_rule, _battery_low_msg))
        self.add_rule(ObserverRule(
            "download.complete", "perception.ocr",
            _download_rule, _download_msg))
        self.add_rule(ObserverRule(
            "notification", "desktop.notification",
            _notification_forward_rule, _notification_forward_msg))

        bus.on("terminal.output", self._on_event, priority=20)
        bus.on("perception.ocr", self._on_event, priority=20)
        bus.on("system.battery", self._on_event, priority=20)
        bus.on("git.output", self._on_event, priority=20)
        bus.on("desktop.notification", self._on_event, priority=20)

        self.set_health("background agent running",
                        {"rules": len(self._rules)})
        return True

    async def _stop(self) -> None:
        """Unsubscribe from the bus and clear rules (idempotent)."""
        try:
            bus.off("terminal.output", self._on_event)
            bus.off("perception.ocr", self._on_event)
            bus.off("system.battery", self._on_event)
            bus.off("git.output", self._on_event)
            bus.off("desktop.notification", self._on_event)
        except Exception as e:
            logger.debug("[BGAGENT] unsubscribe error: %s", e)
        self._rules.clear()
        self._last_notify = None

    def add_rule(self, rule: ObserverRule) -> None:
        """Register a custom observation rule."""
        self._rules.append(rule)

    async def _on_event(self, event: Event) -> None:
        """Check an incoming event against all rules."""
        for rule in self._rules:
            if not rule.matches(event.type):
                continue
            if not rule.should_notify(event.data or {}):
                continue
            # Rate-limit per rule
            now = time.time()
            last = self._notified.get(rule.name, 0.0)
            if now - last < self._rate_limit_s:
                continue
            self._notified[rule.name] = now

            message = rule.format_message(event.data or {})
            await self.notify(rule.name, message, event.data or {})

    async def notify(self, rule_name: str, message: str,
                     data: Optional[Dict[str, Any]] = None) -> None:
        """Emit a background.notify event (voice-ready)."""
        self._last_notify = message
        logger.info("[BGAGENT] %s: %s", rule_name, message)
        await bus.emit("background.notify", data={
            "rule": rule_name,
            "message": message,
            "source_data": data or {},
        }, source="background_agent")


# Global singleton
background_agent = BackgroundAgent()