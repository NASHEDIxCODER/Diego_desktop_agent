"""
ExperienceDB — Self-improving agent experience store.

Every execution creates an experience record:

    - Goal: what the user asked for
    - Plan: the action plan that was generated
    - Result: success/failure + output
    - Latency: how long each step took
    - Failure: what went wrong (detailed)
    - Recovery: what worked as fallback

Future plans consult this database to:
    - Prefer previously successful actions
    - Avoid previously failed approaches
    - Estimate expected latency
    - Choose fallback strategies automatically

Example:
    Firefox failed → Use Chrome next time → Learned automatically.

Data persisted in DuckDB alongside other learning data.

Usage:
    from learning.experience_db import experience_db

    experience_db.record(
        goal="open browser",
        plan=[...],
        success=True,
        latency_ms=450.0,
    )

    best = experience_db.best_approach("open browser")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExperienceRecord:
    """A single execution experience."""
    goal: str                     # natural language goal
    plan_steps: List[str] = field(default_factory=list)   # step descriptions
    plan_actions: List[str] = field(default_factory=list)  # action names
    success: bool = False
    result: str = ""              # human-readable result
    latency_ms: float = 0.0
    per_step_latency: List[float] = field(default_factory=list)
    error: str = ""               # error message if failed
    error_step: int = -1          # which step failed (0-indexed)
    recovery_action: str = ""     # what recovered it
    recovery_success: bool = False
    used_fallback: bool = False   # was a fallback provider used
    timestamp: float = 0.0


class ExperienceDB:
    """
    Self-improving experience database.

    Every planner execution feeds this database. On future requests,
    the planner queries for the best approach based on past success.
    """

    def __init__(self):
        self._records: List[ExperienceRecord] = []
        self._max_records = 5000
        # Fast indexes
        self._by_goal: Dict[str, List[int]] = {}  # goal → list of indices
        # Patterns for normalization
        self._action_alternatives: Dict[str, List[str]] = {
            "desktop_open:firefox": ["desktop_open:google-chrome", "desktop_open:chromium",
                                      "desktop_open:brave", "desktop_open:firefox",
                                      "desktop_open:firefox-esr"],
            "desktop_open:google-chrome": ["desktop_open:google-chrome", "desktop_open:chromium",
                                            "desktop_open:brave", "desktop_open:firefox"],
            "browser_navigate:": [],  # generic
        }

    def record(self, goal: str, plan_steps: Optional[List[str]] = None,
               plan_actions: Optional[List[str]] = None,
               success: bool = False, result: str = "",
               latency_ms: float = 0.0,
               per_step_latency: Optional[List[float]] = None,
               error: str = "", error_step: int = -1,
               recovery_action: str = "",
               recovery_success: bool = False,
               used_fallback: bool = False) -> ExperienceRecord:
        """
        Record a complete execution experience.

        Called after every planner execution — successful or not.
        """
        record = ExperienceRecord(
            goal=goal,
            plan_steps=plan_steps or [],
            plan_actions=plan_actions or [],
            success=success,
            result=result,
            latency_ms=latency_ms,
            per_step_latency=per_step_latency or [],
            error=error,
            error_step=error_step,
            recovery_action=recovery_action,
            recovery_success=recovery_success,
            used_fallback=used_fallback,
            timestamp=time.time(),
        )

        idx = len(self._records)
        self._records.append(record)

        # Index by goal (normalized)
        key = self._normalize_goal(goal)
        if key not in self._by_goal:
            self._by_goal[key] = []
        self._by_goal[key].append(idx)

        # Cap
        if len(self._records) > self._max_records:
            removed = self._records[:1000]
            self._records = self._records[1000:]
            self._rebuild_index()

        logger.debug("[ExpDB] Recorded: goal='%s' success=%s latency=%.0fms",
                     goal[:60], success, latency_ms)

        # Persist via learning engine
        self._persist_implicitly(record)

        return record

    # ── Queries ────────────────────────────────────────────

    def best_approach(self, goal: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Return the best-scored approaches for a given goal.

        Scoring weighs: success rate × recency × throughput.
        Returns up to top_n approaches ranked by score.
        """
        key = self._normalize_goal(goal)
        indices = self._by_goal.get(key, [])

        if not indices:
            # Try fuzzy matching
            indices = self._fuzzy_match_goal(goal)

        if not indices:
            return []

        records = [self._records[i] for i in indices]

        # Score each record
        scored = []
        now = time.time()
        for r in records:
            score = self._score_record(r, now)
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_n]

        results = []
        for score, record in top:
            results.append({
                "goal": record.goal,
                "plan_steps": record.plan_steps[:],
                "plan_actions": record.plan_actions[:],
                "success": record.success,
                "result": record.result,
                "latency_ms": record.latency_ms,
                "score": round(score, 3),
                "recovery_action": record.recovery_action,
                "timestamp": record.timestamp,
            })

        return results

    def best_action_for(self, goal: str) -> Optional[str]:
        """Return the single best action for a simple goal."""
        approaches = self.best_approach(goal, top_n=1)
        if not approaches:
            return None
        # Return first action of best approach
        actions = approaches[0].get("plan_actions", [])
        return actions[0] if actions else None

    def avoid_actions(self, goal: str) -> List[str]:
        """Return actions that should be avoided for a given goal."""
        key = self._normalize_goal(goal)
        indices = self._by_goal.get(key, [])

        failing: Dict[str, int] = {}
        for i in indices[-20:]:  # look at recent 20
            r = self._records[i]
            if not r.success:
                for action in r.plan_actions:
                    failing[action] = failing.get(action, 0) + 1

        # Actions that failed more than succeeded recently
        avoid = []
        for action, fail_count in failing.items():
            success_count = sum(
                1 for i in indices[-20:]
                if self._records[i].success
                and action in self._records[i].plan_actions
            )
            if fail_count > success_count:
                avoid.append(action)
        return avoid

    def recent_failures(self, n: int = 10) -> List[ExperienceRecord]:
        """Return the N most recent failures."""
        failures = [r for r in reversed(self._records) if not r.success]
        return failures[:n]

    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics."""
        if not self._records:
            return {"total": 0, "success_rate": 0.0}

        total = len(self._records)
        successes = sum(1 for r in self._records if r.success)
        avg_latency = sum(r.latency_ms for r in self._records) / total

        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "unique_goals": len(self._by_goal),
        }

    def llm_context(self, max_items: int = 8) -> str:
        """Return a compact context block for the planner LLM."""
        lines = []

        # Recent failures the planner should avoid
        failures = self.recent_failures(3)
        if failures:
            lines.append("Recent action failures to avoid:")
            for f in failures:
                lines.append(f"  - {f.goal[:80]}: {f.error[:100]}" +
                             (f" → recovered with {f.recovery_action}" if f.recovery_action else ""))

        # Highly successful patterns to prefer
        all_goals = list(self._by_goal.keys())[:20]
        for goal_key in all_goals:
            approaches = self.best_approach(goal_key, top_n=1)
            if approaches and approaches[0]["success"] and approaches[0]["score"] > 0.7:
                actions = approaches[0].get("plan_actions", [])
                if actions:
                    lines.append(f"  ✓ {approaches[0]['goal'][:60]}")
                    break  # just the top one

        return "\n".join(lines) if lines else ""

    # ── Scoring ────────────────────────────────────────────

    def _score_record(self, record: ExperienceRecord, now: float) -> float:
        """Score a single experience record."""
        score = 0.0

        # Success: big weight
        if record.success:
            score += 0.5
        elif record.recovery_success:
            score += 0.3

        # Recency: exponential decay (half-life = 24 hours)
        age_hours = (now - record.timestamp) / 3600.0
        recency = max(0.05, 2.0 ** (-age_hours / 24.0))
        score += recency * 0.3

        # Throughput: faster is better
        if record.latency_ms > 0:
            throughput = max(0.0, 1.0 - (record.latency_ms / 30000.0))
        else:
            throughput = 0.5
        score += throughput * 0.1

        # Plan complexity: prefer simpler plans
        complexity = max(0.0, 1.0 - (len(record.plan_steps) / 10.0))
        score += complexity * 0.1

        return score

    # ── Normalization ──────────────────────────────────────

    @staticmethod
    def _normalize_goal(goal: str) -> str:
        """Normalize a goal string for indexing."""
        # Lowercase and trim
        key = goal.lower().strip()
        # Remove common filler words
        for word in ("please", "can you", "could you", "would you",
                      "i want to", "i need to", "just"):
            key = key.replace(word, "")
        # Normalize whitespace
        key = " ".join(key.split())
        return key[:120]

    def _fuzzy_match_goal(self, goal: str, threshold_pct: float = 0.4) -> List[int]:
        """Find records with goals similar to the given one."""
        key = self._normalize_goal(goal)
        key_words = set(key.split())
        if len(key_words) < 2:
            return []

        matches = []
        for existing_key, indices in self._by_goal.items():
            existing_words = set(existing_key.split())
            if not existing_words:
                continue
            overlap = key_words & existing_words
            pct = len(overlap) / max(len(key_words), len(existing_words))
            if pct >= threshold_pct:
                matches.extend(indices)

        return matches

    # ── Index maintenance ─────────────────────────────────

    def _rebuild_index(self) -> None:
        """Rebuild the goal index after culling."""
        self._by_goal.clear()
        for idx, record in enumerate(self._records):
            key = self._normalize_goal(record.goal)
            if key not in self._by_goal:
                self._by_goal[key] = []
            self._by_goal[key].append(idx)

    def _persist_implicitly(self, record: ExperienceRecord) -> None:
        """Store the experience via the learning engine for persistence."""
        try:
            from learning.learning_engine import learning_engine
            # Use record_action which is the stable public API
            learning_engine.record_action(
                action_name="|".join(record.plan_actions[:5]) or "unknown",
                params={
                    "goal": record.goal,
                    "plan_steps": record.plan_steps,
                    "result": record.result,
                    "used_fallback": record.used_fallback,
                },
                success=record.success,
                latency_ms=record.latency_ms,
                error=record.error,
            )
        except Exception:
            pass

    # ── Serialization ─────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        records_list = []
        for r in self._records:
            records_list.append({
                "goal": r.goal,
                "plan_steps": r.plan_steps,
                "plan_actions": r.plan_actions,
                "success": r.success,
                "result": r.result,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "error_step": r.error_step,
                "recovery_action": r.recovery_action,
                "recovery_success": r.recovery_success,
                "used_fallback": r.used_fallback,
                "timestamp": r.timestamp,
            })
        return {"records": records_list, "version": 1}

    def record_count(self) -> int:
        return len(self._records)


# Global singleton
experience_db = ExperienceDB()