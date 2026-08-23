"""
UserProfile — Learned facts about Diego's owner.

Stores:
  - preferred apps (editor, browser, terminal, etc.)
  - coding habits (language, framework, git repos)
  - daily schedule (wake time, work hours, sleep time)
  - projects (names, paths, last opened)
  - speaking style (verbosity, formality)
  - common folders and commands

Everything is learned automatically from observed behavior.
No configuration file. No hardcoded defaults.
Confidence scores decay over time if a habit stops being observed.

Data is persisted in DuckDB. No model weights are modified.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Confidence thresholds
OBSERVATION_THRESHOLD = 3    # promote to "known" after N observations
CONFIDENCE_DECAY_DAYS = 14   # confidence halves after this many days without re-observation
META_CONFIDENCE_MIN = 0.3    # below this, the fact is considered stale


@dataclass
class UserFact:
    """A single learned fact about the user."""
    category: str          # "app", "editor", "browser", "project", "schedule", "style"
    key: str               # e.g. "preferred_app:editor"
    value: str             # e.g. "code"  (the learned value)
    confidence: float = 0.0  # 0.0 – 1.0
    observation_count: int = 0
    first_seen: float = 0.0  # epoch timestamp
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserProfile:
    """
    Profile of the user, learned from observed behavior.

    The profile is built incrementally: every time the user opens an app,
    visits a URL, runs a command, or mentions a project, the corresponding
    fact is observed and its confidence increases.

    After OBSERVATION_THRESHOLD (default 5) observations, a fact is promoted
    to "learned" and available for querying.
    """

    def __init__(self):
        self._facts: Dict[str, Dict[str, UserFact]] = {}  # category -> {key -> fact}
        self._store = None  # DuckDB store reference, set by LearningEngine
        self._dirty = False

    # ── Observation ───────────────────────────────────────────────

    def observe(self, category: str, key: str, value: str,
                metadata: Optional[Dict[str, Any]] = None) -> UserFact:
        """
        Record an observation. Each call increments the confidence.

        After OBSERVATION_THRESHOLD observations, the fact is "learned"
        and available for recommendation.

        Args:
            category: e.g. "app", "editor", "browser", "project"
            key: e.g. "preferred", "recent", "frequency"
            value: the observed value
            metadata: optional extra context
        """
        now = time.time()

        if category not in self._facts:
            self._facts[category] = {}

        fact_key = f"{key}:{value}"
        if fact_key in self._facts[category]:
            fact = self._facts[category][fact_key]
            fact.observation_count += 1
            fact.last_seen = now
            # Confidence: logistic curve from 0 to 1
            fact.confidence = min(1.0, fact.observation_count / OBSERVATION_THRESHOLD)
        else:
            fact = UserFact(
                category=category,
                key=key,
                value=value,
                confidence=1.0 / OBSERVATION_THRESHOLD,
                observation_count=1,
                first_seen=now,
                last_seen=now,
                metadata=metadata or {},
            )
            self._facts[category][fact_key] = fact

        self._dirty = True
        logger.debug("[Profile] Observe %s/%s=%s (count=%d, conf=%.2f)",
                     category, key, value, fact.observation_count, fact.confidence)
        return fact

    # ── Query ─────────────────────────────────────────────────────

    def get_best(self, category: str, key: str = "preferred") -> Optional[UserFact]:
        """
        Return the highest-confidence fact for a category + key.

        Returns None if no learned fact exists (below threshold).
        """
        if category not in self._facts:
            return None

        best: Optional[UserFact] = None
        for fact in self._facts[category].values():
            if fact.key == key and fact.confidence >= META_CONFIDENCE_MIN:
                if best is None or fact.confidence > best.confidence:
                    best = fact
        return best

    def get_top(self, category: str, key: str, n: int = 5) -> List[UserFact]:
        """Return the top N facts for a category + key, sorted by confidence."""
        if category not in self._facts:
            return []

        candidates = [f for f in self._facts[category].values()
                      if f.key == key and f.confidence >= META_CONFIDENCE_MIN]
        candidates.sort(key=lambda f: (-f.confidence, -f.observation_count))
        return candidates[:n]

    def get_recent(self, category: str, key: str, n: int = 5) -> List[UserFact]:
        """Return the most recently observed N facts for a category + key."""
        if category not in self._facts:
            return []

        candidates = [f for f in self._facts[category].values()
                      if f.key == key]
        candidates.sort(key=lambda f: -f.last_seen)
        return candidates[:n]

    def is_learned(self, category: str, key: str, value: str) -> bool:
        """True if this specific fact has been observed >= threshold times."""
        if category not in self._facts:
            return False
        fact_key = f"{key}:{value}"
        fact = self._facts[category].get(fact_key)
        return fact is not None and fact.observation_count >= OBSERVATION_THRESHOLD

    # ── Decay ─────────────────────────────────────────────────────

    def decay(self) -> int:
        """
        Reduce confidence for facts not recently observed.

        Confidence halves after CONFIDENCE_DECAY_DAYS without re-observation.
        Returns the number of facts that dropped below META_CONFIDENCE_MIN.
        """
        now = time.time()
        removed = 0
        decay_seconds = CONFIDENCE_DECAY_DAYS * 86400

        for category in list(self._facts.keys()):
            stale_keys = []
            for fact_key, fact in self._facts[category].items():
                age = now - fact.last_seen
                if age > decay_seconds:
                    # Halve confidence for every decay period elapsed
                    periods = max(1, int(age / decay_seconds))
                    fact.confidence = fact.confidence * (0.5 ** periods)
                    if fact.confidence < META_CONFIDENCE_MIN:
                        stale_keys.append(fact_key)
                        removed += 1
            for key in stale_keys:
                del self._facts[category][key]
            if not self._facts[category]:
                del self._facts[category]

        if removed:
            logger.info("[Profile] Decay removed %d stale facts", removed)
            self._dirty = True
        return removed

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all facts to a dict (for DuckDB persistence)."""
        facts_list = []
        for category in self._facts.values():
            for fact in category.values():
                facts_list.append({
                    "category": fact.category,
                    "key": fact.key,
                    "value": fact.value,
                    "confidence": round(fact.confidence, 3),
                    "observation_count": fact.observation_count,
                    "first_seen": fact.first_seen,
                    "last_seen": fact.last_seen,
                    "metadata": fact.metadata,
                })
        return {"facts": facts_list, "version": 1}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Deserialize from a dict."""
        profile = cls()
        for fdata in data.get("facts", []):
            fact = UserFact(
                category=fdata["category"],
                key=fdata["key"],
                value=fdata["value"],
                confidence=fdata.get("confidence", 0.0),
                observation_count=fdata.get("observation_count", 0),
                first_seen=fdata.get("first_seen", 0.0),
                last_seen=fdata.get("last_seen", 0.0),
                metadata=fdata.get("metadata", {}),
            )
            if fact.category not in profile._facts:
                profile._facts[fact.category] = {}
            profile._facts[fact.category][f"{fact.key}:{fact.value}"] = fact
        return profile

    # ── Convenience queries for the LLM context ───────────────────

    def llm_context(self) -> str:
        """
        Return a compact text block for LLM prompt injection.
        Only includes facts with confidence >= 0.5.
        """
        lines = []
        # Preferred apps
        editor = self.get_best("app", "editor")
        if editor:
            lines.append(f"Preferred editor: {editor.value}")
        browser = self.get_best("app", "browser")
        if browser:
            lines.append(f"Preferred browser: {browser.value}")
        terminal = self.get_best("app", "terminal")
        if terminal:
            lines.append(f"Preferred terminal: {terminal.value}")

        # Top apps by frequency
        top_apps = self.get_top("app", "frequency", n=5)
        if top_apps:
            lines.append("Frequently used apps: " +
                         ", ".join(f"{f.value}({f.observation_count}x)" for f in top_apps))

        # Projects
        top_projects = self.get_top("project", "current", n=3)
        if top_projects:
            lines.append("Active projects: " +
                         ", ".join(f.value for f in top_projects))

        # Coding
        lang = self.get_best("coding", "language")
        if lang:
            lines.append(f"Primary language: {lang.value}")
        framework = self.get_best("coding", "framework")
        if framework:
            lines.append(f"Framework: {framework.value}")

        # Schedule
        top_hours = self.get_top("schedule", "active_hour", n=3)
        if top_hours:
            hours_str = ", ".join(f"{f.value}:00" for f in top_hours)
            lines.append(f"Most active hours: {hours_str}")

        return "\n".join(lines) if lines else ""

    @property
    def fact_count(self) -> int:
        return sum(len(c) for c in self._facts.values())

    @property
    def learned_count(self) -> int:
        return sum(1 for c in self._facts.values()
                   for f in c.values()
                   if f.observation_count >= OBSERVATION_THRESHOLD)