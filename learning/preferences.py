"""
PreferenceStore — Key-value preferences with confidence decay.

Stores user preferences as (category, key, value, confidence) tuples.
Preferences are learned from observation, not configured.

Examples:
  - volume level (audio, volume, 0.7, conf=0.9)
  - brightness level (display, brightness, 0.8, conf=0.85)
  - reply verbosity (style, verbosity, brief, conf=0.6)
  - music provider (music, provider, spotify, conf=0.95)

Confidence increases with repeated observation and decays over time
if the user changes the preference.

Data is persisted in DuckDB. No model weights are modified.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Preference:
    """A single learned preference."""
    category: str           # e.g. "audio", "display", "music", "style"
    key: str                # e.g. "volume", "brightness", "provider"
    value: Any              # the preferred value
    confidence: float = 0.0  # 0.0 – 1.0
    observation_count: int = 0
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreferenceStore:
    """
    Learned user preferences with confidence tracking.

    When a user sets a value (e.g. volume to 70%), the preference is
    recorded. If the user consistently sets the same value, confidence
    rises. If the user changes it, the old preference decays and the
    new one gains confidence.

    After CONFIDENCE_THRESHOLD (default 0.6), a preference is
    considered "learned" and can be used automatically.
    """

    CONFIDENCE_THRESHOLD = 0.6   # promote to learned
    DECAY_DAYS = 7               # confidence halves every week without re-observation
    MIN_CONFIDENCE = 0.1

    def __init__(self):
        self._prefs: Dict[str, Dict[str, Preference]] = {}  # category -> {key -> pref}
        self._dirty = False

    # ── CRUD ──────────────────────────────────────────────────────

    def observe(self, category: str, key: str, value: Any,
                confidence_boost: float = 0.15) -> Preference:
        """
        Record an observation of a preference.

        If this is a NEW value for an existing key, the old preference's
        confidence is halved (the user changed their mind). The new value
        starts with boosted confidence.

        Args:
            category: e.g. "audio", "display", "music"
            key: e.g. "volume", "provider"
            value: the observed value
            confidence_boost: how much to boost confidence per observation
        """
        now = time.time()

        if category not in self._prefs:
            self._prefs[category] = {}

        if key in self._prefs[category]:
            existing = self._prefs[category][key]
            if str(existing.value) == str(value):
                # Same value — boost confidence
                existing.confidence = min(1.0, existing.confidence + confidence_boost)
                existing.observation_count += 1
                existing.last_seen = now
                self._dirty = True
                logger.debug("[Prefs] Reinforce %s/%s=%s (conf=%.2f, count=%d)",
                             category, key, value, existing.confidence, existing.observation_count)
                return existing
            else:
                # Changed value — decay old, boost new
                existing.confidence *= 0.5
                logger.debug("[Prefs] Override %s/%s: %s → %s (old conf dropped to %.2f)",
                             category, key, existing.value, value, existing.confidence)

        # New preference
        pref = Preference(
            category=category,
            key=key,
            value=value,
            confidence=0.3 + confidence_boost,  # start at ~0.45
            observation_count=1,
            last_seen=now,
        )
        self._prefs[category][key] = pref
        self._dirty = True
        logger.debug("[Prefs] New %s/%s=%s (conf=%.2f)",
                     category, key, value, pref.confidence)
        return pref

    def get(self, category: str, key: str) -> Optional[Any]:
        """
        Get the preference value if its confidence is above threshold.

        Returns None if not learned yet.
        """
        if category not in self._prefs or key not in self._prefs[category]:
            return None
        pref = self._prefs[category][key]
        if pref.confidence >= self.CONFIDENCE_THRESHOLD:
            return pref.value
        return None

    def get_with_confidence(self, category: str, key: str) -> Optional[Preference]:
        """Get the full Preference object, regardless of confidence."""
        if category not in self._prefs:
            return None
        return self._prefs[category].get(key)

    def set(self, category: str, key: str, value: Any,
            confidence: float = 1.0) -> Preference:
        """
        Explicitly set a preference (bypasses observation).

        Used when the user explicitly says "set volume to 70%".
        """
        now = time.time()
        if category not in self._prefs:
            self._prefs[category] = {}

        pref = Preference(
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            observation_count=1,
            last_seen=now,
        )
        self._prefs[category][key] = pref
        self._dirty = True
        return pref

    def all_learned(self) -> Dict[str, Dict[str, Any]]:
        """Return all preferences with confidence >= threshold."""
        result: Dict[str, Dict[str, Any]] = {}
        for cat, keys in self._prefs.items():
            for key, pref in keys.items():
                if pref.confidence >= self.CONFIDENCE_THRESHOLD:
                    if cat not in result:
                        result[cat] = {}
                    result[cat][key] = pref.value
        return result

    # ── Decay ─────────────────────────────────────────────────────

    def decay(self) -> int:
        """Decay old preferences. Returns count of preferences removed."""
        now = time.time()
        removed = 0
        decay_seconds = self.DECAY_DAYS * 86400

        for cat in list(self._prefs.keys()):
            for key in list(self._prefs[cat].keys()):
                pref = self._prefs[cat][key]
                age = now - pref.last_seen
                if age > decay_seconds:
                    periods = max(1, int(age / decay_seconds))
                    pref.confidence = pref.confidence * (0.5 ** periods)
                    if pref.confidence < self.MIN_CONFIDENCE:
                        del self._prefs[cat][key]
                        removed += 1
            if not self._prefs.get(cat):
                del self._prefs[cat]

        if removed:
            logger.info("[Prefs] Decay removed %d stale preferences", removed)
            self._dirty = True
        return removed

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        prefs_list = []
        for cat, keys in self._prefs.items():
            for key, pref in keys.items():
                prefs_list.append({
                    "category": pref.category,
                    "key": pref.key,
                    "value": pref.value,
                    "confidence": round(pref.confidence, 3),
                    "observation_count": pref.observation_count,
                    "last_seen": pref.last_seen,
                })
        return {"preferences": prefs_list, "version": 1}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceStore":
        store = cls()
        for pdata in data.get("preferences", []):
            pref = Preference(
                category=pdata["category"],
                key=pdata["key"],
                value=pdata["value"],
                confidence=pdata.get("confidence", 0.0),
                observation_count=pdata.get("observation_count", 0),
                last_seen=pdata.get("last_seen", 0.0),
            )
            if pref.category not in store._prefs:
                store._prefs[pref.category] = {}
            store._prefs[pref.category][pref.key] = pref
        return store

    # ── LLM context ───────────────────────────────────────────────

    def llm_context(self) -> str:
        """Compact text block for LLM prompt injection."""
        learned = self.all_learned()
        if not learned:
            return ""
        lines = []
        for cat in sorted(learned):
            for key in sorted(learned[cat]):
                lines.append(f"  {cat}.{key} = {learned[cat][key]}")
        return "User preferences:\n" + "\n".join(lines) if lines else ""

    @property
    def count(self) -> int:
        return sum(len(k) for k in self._prefs.values())