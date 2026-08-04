"""
SkillMemory — Remembers which actions work and which fail.

Every action Leo takes is recorded with its outcome. Over time this
builds an experience database that the planner can query to:

  - Choose a successful action over a failed one
  - Detect when a previously reliable action has degraded
  - Remember workarounds (Plan B succeeded when Plan A failed)
  - Estimate expected latency for each action type

Each entry stores:
  - goal: what the user asked for
  - action: what Leo did (app, command, URL, etc.)
  - success: True/False
  - latency_ms: how long it took
  - error: what went wrong (if failure)
  - recovery: what succeeded after failure (if any)
  - timestamp: when it happened

The adaptive planner (Phase 8) uses this data to prefer proven actions.

Data is persisted in DuckDB. No model weights are modified.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How many recent entries to keep per action type for scoring
SCORE_WINDOW = 20


@dataclass
class SkillEntry:
    """A single skill execution record."""
    goal: str                  # user intent (e.g. "open firefox")
    action: str                # action taken (e.g. "desktop_open:firefox")
    params: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    latency_ms: float = 0.0
    error: str = ""            # error message if failed
    recovery: str = ""         # what worked after failure
    timestamp: float = 0.0


class SkillMemory:
    """
    Experience database for Leo's actions.

    Records every action outcome. Queries provide:
      - Success rate per action pattern
      - Best action for a given goal
      - Recent failures (for proactive alerting)
      - Action latency statistics
    """

    def __init__(self):
        self._entries: List[SkillEntry] = []
        # Index: action -> list of entries (most recent last)
        self._by_action: Dict[str, List[SkillEntry]] = defaultdict(list)
        self._dirty = False
        self._max_entries = 10000  # hard cap

    # ── Record ──────────────────────────────────────────────────

    def record(self, goal: str, action: str, success: bool,
               latency_ms: float = 0.0, error: str = "",
               recovery: str = "",
               params: Optional[Dict[str, Any]] = None) -> SkillEntry:
        """
        Record an action execution.

        Args:
            goal: what the user asked for
            action: what Leo did (e.g. "desktop_open:code")
            success: did it work
            latency_ms: how long it took
            error: error message if failed
            recovery: what worked after failure
            params: action parameters
        """
        entry = SkillEntry(
            goal=goal,
            action=action,
            params=params or {},
            success=success,
            latency_ms=latency_ms,
            error=error,
            recovery=recovery,
            timestamp=time.time(),
        )

        self._entries.append(entry)
        self._by_action[action].append(entry)

        # Enforce cap
        if len(self._entries) > self._max_entries:
            removed = self._entries[:1000]
            self._entries = self._entries[1000:]
            for old in removed:
                lst = self._by_action.get(old.action, [])
                if old in lst:
                    lst.remove(old)

        self._dirty = True
        logger.debug("[Skill] Record %s success=%s latency=%.0fms",
                     action, success, latency_ms)
        return entry

    # ── Query ────────────────────────────────────────────────────

    def success_rate(self, action: str) -> float:
        """
        Return the success rate for a given action pattern (0.0 – 1.0).

        Only considers recent entries (last SCORE_WINDOW).
        """
        entries = self._by_action.get(action, [])
        if not entries:
            return 0.0

        recent = entries[-SCORE_WINDOW:]
        successes = sum(1 for e in recent if e.success)
        return successes / len(recent)

    def best_action(self, goal: str) -> Optional[str]:
        """
        Return the action with the highest success rate for a given goal.

        Returns None if no action has been tried for this goal.
        """
        # Find entries matching this goal (fuzzy prefix match)
        matching: Dict[str, List[SkillEntry]] = defaultdict(list)
        for entry in self._entries[-1000:]:  # look at recent 1000
            if goal.lower() in entry.goal.lower():
                matching[entry.action].append(entry)

        if not matching:
            return None

        best_action = None
        best_rate = -1.0
        for action, entries in matching.items():
            recent = entries[-SCORE_WINDOW:]
            successes = sum(1 for e in recent if e.success)
            rate = successes / len(recent) if recent else 0.0
            if rate > best_rate:
                best_rate = rate
                best_action = action

        return best_action

    def recent_failures(self, n: int = 10) -> List[SkillEntry]:
        """Return the most recent N failures."""
        failures = [e for e in reversed(self._entries) if not e.success]
        return failures[:n]

    def avg_latency(self, action: str) -> float:
        """Average latency in ms for a given action (recent 20)."""
        entries = self._by_action.get(action, [])
        if not entries:
            return 0.0
        recent = entries[-SCORE_WINDOW:]
        latencies = [e.latency_ms for e in recent if e.latency_ms > 0]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def has_degraded(self, action: str, threshold: float = 0.5) -> bool:
        """
        True if a previously successful action has recently started failing.

        Compares overall success rate vs. last 5 attempts.
        """
        entries = self._by_action.get(action, [])
        if len(entries) < 5:
            return False

        overall = self.success_rate(action)
        last5 = entries[-5:]
        recent_rate = sum(1 for e in last5 if e.success) / len(last5)

        return overall > 0.7 and recent_rate < threshold

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        entries_list = []
        for entry in self._entries:
            entries_list.append({
                "goal": entry.goal,
                "action": entry.action,
                "params": entry.params,
                "success": entry.success,
                "latency_ms": entry.latency_ms,
                "error": entry.error,
                "recovery": entry.recovery,
                "timestamp": entry.timestamp,
            })
        return {"entries": entries_list, "version": 1}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMemory":
        mem = cls()
        for edata in data.get("entries", []):
            entry = SkillEntry(
                goal=edata["goal"],
                action=edata["action"],
                params=edata.get("params", {}),
                success=edata.get("success", False),
                latency_ms=edata.get("latency_ms", 0.0),
                error=edata.get("error", ""),
                recovery=edata.get("recovery", ""),
                timestamp=edata.get("timestamp", 0.0),
            )
            mem._entries.append(entry)
            mem._by_action[entry.action].append(entry)
        return mem

    # ── LLM context ───────────────────────────────────────────────

    def llm_context(self) -> str:
        """Compact text block for LLM prompt injection."""
        lines = []

        # Recent failures the LLM should be aware of
        failures = self.recent_failures(5)
        if failures:
            lines.append("Recent action failures:")
            for f in failures:
                lines.append(f"  - {f.action}: {f.error or 'unknown error'}")

        # Actions that have degraded
        degraded = []
        for action in list(self._by_action.keys())[:20]:
            if self.has_degraded(action):
                rate = self.success_rate(action)
                degraded.append(f"{action} (recent: {rate:.0%})")

        if degraded:
            lines.append("Degraded actions:")
            for d in degraded:
                lines.append(f"  - {d}")

        return "\n".join(lines) if lines else ""

    @property
    def entry_count(self) -> int:
        return len(self._entries)