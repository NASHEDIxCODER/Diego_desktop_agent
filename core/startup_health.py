"""
StartupHealth — Subsystem state tracking for Diego.

Each subsystem reports one of:
  READY    — Fully operational
  DEGRADED — Working with reduced functionality
  DISABLED — Skipped (optional, not available)
  FAILED   — Fatal error, blocks assistant startup

The assistant starts only after all states are finalized.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SubsystemState(Enum):
    READY = auto()
    DEGRADED = auto()
    DISABLED = auto()
    FAILED = auto()


@dataclass
class SubsystemHealth:
    """Health record for a single subsystem."""
    name: str
    state: SubsystemState
    message: str = ""
    details: dict = field(default_factory=dict)


class StartupHealth:
    """
    Central startup health registry.

    Usage:
        startup_health.register("nlp")
        startup_health.set_state("nlp", SubsystemState.READY, "Model loaded")
        if startup_health.finalize():
            print("Assistant ready")
    """

    def __init__(self):
        self._subsystems: Dict[str, SubsystemHealth] = {}
        self._finalized = False

    def register(self, name: str) -> None:
        """Register a subsystem (idempotent)."""
        if name not in self._subsystems:
            self._subsystems[name] = SubsystemHealth(
                name=name,
                state=SubsystemState.DISABLED,
                message="Not yet initialized",
            )

    def set_state(self, name: str, state: SubsystemState,
                  message: str = "", details: Optional[dict] = None) -> None:
        """Set subsystem state with a descriptive message."""
        self.register(name)
        self._subsystems[name].state = state
        self._subsystems[name].message = message
        if details:
            self._subsystems[name].details = details
        logger.info("Subsystem %s → %s: %s", name, state.name, message)

    def finalize(self) -> bool:
        """
        Finalize startup diagnostics.

        Prints a summary of all subsystem states.

        Returns:
            True if no FAILED subsystems exist (assistant can start).
        """
        self._finalized = True

        failed = [s for s in self._subsystems.values()
                  if s.state == SubsystemState.FAILED]
        ready = [s for s in self._subsystems.values()
                 if s.state == SubsystemState.READY]
        degraded = [s for s in self._subsystems.values()
                    if s.state == SubsystemState.DEGRADED]

        print()
        print("  Startup Diagnostics")
        print("  " + "=" * 40)

        for s in self._subsystems.values():
            icon = {
                SubsystemState.READY: "\u2713",
                SubsystemState.DEGRADED: "\u26A0",
                SubsystemState.DISABLED: "\u25CB",
                SubsystemState.FAILED: "\u2717",
            }.get(s.state, "?")
            print(f"  {icon} {s.name}: {s.state.name}  {s.message}")

        print()
        if failed:
            print(f"  \u2717 {len(failed)} subsystem(s) FAILED — cannot start")
        elif degraded:
            print(f"  \u2713 Assistant Ready (Degraded Mode, {len(degraded)} subsystem(s) degraded)")
        else:
            print(f"  \u2713 Assistant Ready ({len(ready)} subsystem(s) ready)")
        print()

        return len(failed) == 0

    @property
    def can_start(self) -> bool:
        """Check if startup conditions are met."""
        if not self._finalized:
            return False
        return not any(s.state == SubsystemState.FAILED
                       for s in self._subsystems.values())

    @property
    def is_degraded(self) -> bool:
        """Check if running in degraded mode."""
        return (self.can_start and
                any(s.state == SubsystemState.DEGRADED
                    for s in self._subsystems.values()))

    def get_report(self) -> Dict[str, dict]:
        """Get full health report as serializable dict."""
        return {
            name: {
                "state": health.state.name,
                "message": health.message,
                "details": health.details,
            }
            for name, health in self._subsystems.items()
        }


# Global singleton
startup_health = StartupHealth()