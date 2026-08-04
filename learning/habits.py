"""
HabitTracker — Frequency tracking over time windows.

Tracks what the user does repeatedly at different time scales:
  - hourly: what times of day they're active
  - daily: which days they use specific apps
  - weekly: patterns that repeat each week
  - project: which project they're currently working on

Each habit has a frequency count that increases with each observation
and decays exponentially when the habit stops being observed.

Used by the learning engine to detect:
  - "VS Code is the default editor" (most frequent app at 9am-5pm)
  - "User codes in Python most mornings"
  - "GhostLine is the current project" (most recent + frequent)

Data is persisted in DuckDB. No model weights are modified.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Time windows in seconds
HOUR = 3600
DAY = 86400
WEEK = 604800

# Decay half-life for each window
DECAY_HALF_LIFE = {
    "hourly": HOUR * 8,       # 8 hours
    "daily": DAY * 3,         # 3 days
    "weekly": WEEK * 2,       # 2 weeks
}

# Minimum observations to consider a habit "established"
ESTABLISHMENT_THRESHOLD = 5


@dataclass
class HabitEntry:
    """A single tracked habit entry."""
    category: str        # "app", "folder", "command", "project", "schedule"
    value: str           # e.g. "code", "/home/user/projects/ghostline"
    window: str          # "hourly", "daily", "weekly"
    count: int = 0       # weighted frequency (decay-adjusted)
    raw_count: int = 0   # total observations (never decays)
    first_seen: float = 0.0
    last_seen: float = 0.0


class HabitTracker:
    """
    Tracks user habits at multiple time scales.

    Each observation is recorded with a timestamp. When queried, the
    count is decay-adjusted: older observations contribute less.
    """

    def __init__(self):
        self._habits: Dict[str, Dict[str, Dict[str, HabitEntry]]] = defaultdict(
            lambda: defaultdict(dict)  # window -> {value: entry}
        )  # category -> window -> {value: entry}
        self._dirty = False

    # ── Observation ───────────────────────────────────────────────

    def observe(self, category: str, value: str) -> HabitEntry:
        """
        Record a habit observation. Should be called every time the user
        does something trackable (opens an app, visits a URL, etc.).

        The observation is recorded against all time windows.
        """
        now = time.time()
        entries = []

        for window in ("hourly", "daily", "weekly"):
            entry = self._observe_window(category, value, window, now)
            entries.append(entry)

        self._dirty = True
        return entries[1]  # return the daily entry as canonical

    def _observe_window(self, category: str, value: str, window: str,
                        now: float) -> HabitEntry:
        """Record an observation for a specific time window."""
        if value not in self._habits[category][window]:
            entry = HabitEntry(
                category=category,
                value=value,
                window=window,
                count=1,
                raw_count=1,
                first_seen=now,
                last_seen=now,
            )
            self._habits[category][window][value] = entry
        else:
            entry = self._habits[category][window][value]
            # Apply decay since last observation
            age = now - entry.last_seen
            half_life = DECAY_HALF_LIFE.get(window, DAY * 3)
            decay = 0.5 ** (age / half_life)
            entry.count = entry.count * decay + 1.0
            entry.raw_count += 1
            entry.last_seen = now

        return entry

    # ── Query ─────────────────────────────────────────────────────

    def get_top(self, category: str, window: str = "daily",
                n: int = 5) -> List[HabitEntry]:
        """
        Return the top N habits for a category + window, sorted by
        decay-adjusted count descending.
        """
        if category not in self._habits or window not in self._habits[category]:
            return []

        # First apply decay to all entries
        now = time.time()
        half_life = DECAY_HALF_LIFE.get(window, DAY * 3)
        for entry in self._habits[category][window].values():
            age = now - entry.last_seen
            entry.count = entry.count * (0.5 ** (age / half_life))

        entries = list(self._habits[category][window].values())
        entries.sort(key=lambda e: (-e.count, -e.raw_count))
        return entries[:n]

    def get_best(self, category: str, window: str = "daily") -> Optional[HabitEntry]:
        """Return the single highest-count habit."""
        top = self.get_top(category, window, n=1)
        return top[0] if top else None

    def is_established(self, category: str, value: str,
                       window: str = "daily") -> bool:
        """True if this habit has been observed >= ESTABLISHMENT_THRESHOLD times."""
        if (category not in self._habits or
                window not in self._habits[category] or
                value not in self._habits[category][window]):
            return False
        return self._habits[category][window][value].raw_count >= ESTABLISHMENT_THRESHOLD

    def most_active_hours(self) -> List[int]:
        """Return hours (0-23) sorted by activity level."""
        if "schedule" not in self._habits or "hourly" not in self._habits["schedule"]:
            return []

        hour_counts: Dict[int, float] = defaultdict(float)
        for value, entry in self._habits["schedule"]["hourly"].items():
            try:
                hour = int(value)
                if 0 <= hour <= 23:
                    hour_counts[hour] += entry.count
            except (ValueError, TypeError):
                continue

        sorted_hours = sorted(hour_counts.items(), key=lambda x: -x[1])
        return [h for h, _ in sorted_hours]

    # ── Decay ─────────────────────────────────────────────────────

    def decay(self) -> int:
        """Apply decay to all habits. Returns count of habits below threshold."""
        now = time.time()
        removed = 0

        for category in list(self._habits.keys()):
            for window in list(self._habits[category].keys()):
                half_life = DECAY_HALF_LIFE.get(window, DAY * 3)
                stale = []
                for value, entry in self._habits[category][window].items():
                    age = now - entry.last_seen
                    entry.count = entry.count * (0.5 ** (age / half_life))
                    if entry.count < 0.1 and entry.raw_count < 2:
                        stale.append(value)
                for value in stale:
                    del self._habits[category][window][value]
                    removed += 1
                if not self._habits[category][window]:
                    del self._habits[category][window]
            if not self._habits[category]:
                del self._habits[category]

        if removed:
            logger.info("[Habits] Decay removed %d stale habit entries", removed)
            self._dirty = True
        return removed

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        habits_list = []
        for category in self._habits:
            for window in self._habits[category]:
                for value, entry in self._habits[category][window].items():
                    habits_list.append({
                        "category": entry.category,
                        "value": entry.value,
                        "window": entry.window,
                        "count": round(entry.count, 2),
                        "raw_count": entry.raw_count,
                        "first_seen": entry.first_seen,
                        "last_seen": entry.last_seen,
                    })
        return {"habits": habits_list, "version": 1}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HabitTracker":
        tracker = cls()
        for hdata in data.get("habits", []):
            entry = HabitEntry(
                category=hdata["category"],
                value=hdata["value"],
                window=hdata["window"],
                count=hdata.get("count", 0),
                raw_count=hdata.get("raw_count", 0),
                first_seen=hdata.get("first_seen", 0.0),
                last_seen=hdata.get("last_seen", 0.0),
            )
            if entry.category not in tracker._habits:
                tracker._habits[entry.category] = {}
            if entry.window not in tracker._habits[entry.category]:
                tracker._habits[entry.category][entry.window] = {}
            tracker._habits[entry.category][entry.window][entry.value] = entry
        return tracker

    # ── LLM context ───────────────────────────────────────────────

    def llm_context(self) -> str:
        """Compact text block for LLM prompt injection."""
        lines = []

        # Top apps
        top_apps = self.get_top("app", "daily", n=5)
        if top_apps:
            lines.append("Most used apps today: " +
                         ", ".join(f"{h.value}({h.raw_count}x)" for h in top_apps))

        # Current project
        top_proj = self.get_top("project", "daily", n=3)
        if top_proj:
            lines.append("Active projects: " +
                         ", ".join(h.value for h in top_proj))

        # Top folders
        top_folders = self.get_top("folder", "daily", n=3)
        if top_folders:
            lines.append("Frequent folders: " +
                         ", ".join(h.value for h in top_folders))

        # Active hours
        hours = self.most_active_hours()[:3]
        if hours:
            lines.append("Most active hours: " +
                         ", ".join(f"{h}:00" for h in hours))

        return "\n".join(lines) if lines else ""

    @property
    def entry_count(self) -> int:
        return sum(
            len(window_habits)
            for cat in self._habits.values()
            for window_habits in cat.values()
        )