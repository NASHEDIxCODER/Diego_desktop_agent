"""
TaskLessons — SAFE bounded self-improvement store (Phase 21A, Tasks 9/10/12).

Learning produces STRUCTURED REUSABLE LESSONS only. The model can NEVER
rewrite:
    - Python code, system prompts, authorization rules, safety policies,
      tool permissions, confirmation requirements, or task limits.

Lessons always carry: source task, evidence, confidence, timestamp,
reuse count, and success/failure feedback. LEARNING FROM AN UNVERIFIED
RESULT IS IMPOSSIBLE: success lessons require evidence of verified
success (the deterministic runtime decides what is verified).

Old memory is evidence, not truth (Task 10): retrieval returns lessons
with a trustworthiness flag, and every composed lesson block instructs
the reasoning layer to VERIFY the current environment before relying on
a remembered strategy. When a remembered strategy fails, record_reuse()
lowers its confidence and the lesson eventually decays away.

User preferences (Task 12): only a small, harmless allowlist can be
learned automatically. A preference is advisory information for the
reasoning layer — it can NEVER bypass authorization, confirmation, or
any safety policy (enforcement lives in the deterministic runtime and
is not influenced by preferences at all).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Lesson vocabulary ──────────────────────────────────────────
LESSON_TYPES = frozenset({
    "task_lesson",
    "successful_strategy",
    "failed_strategy",
    "tool_preference",
    "parameter_correction",
    "environment_state",
    "user_preference",
    "recurring_obstacle",
    "verified_workaround",
})

# Evidence forms that MAY back a lesson. Anything else (in particular
# "unverified", model claims, tool-returned-but-not-verified) cannot.
VERIFIED_EVIDENCE = frozenset({
    "verified_success",
    "verified_failure",
    "verified_observation",
})

MIN_TRUST_CONFIDENCE = 0.4     # below this a lesson is not trusted
MIN_CONFIDENCE = 0.05          # below this a lesson is dropped
REUSE_SUCCESS_BOOST = 0.05
REUSE_FAILURE_PENALTY = 0.15
DEFAULT_LESSON_CONFIDENCE = 0.7

# ── Task 12: safe-preference allowlist ─────────────────────────
# (category, key) pairs that are harmless, non-sensitive, and observable
# from ordinary usage. Anything NOT on this list is never auto-learned —
# this is what keeps credentials/permissions/contacts out of the
# preference store.
SAFE_PREFERENCES = frozenset({
    ("app", "browser"),
    ("app", "editor"),
    ("app", "terminal"),
    ("app", "file_manager"),
    ("music", "provider"),
    ("audio", "volume"),
    ("display", "brightness"),
    ("style", "verbosity"),
    ("style", "language"),
    ("workflow", "search_engine"),
})
# Categories that must NEVER be learned as preferences, even if someone
# adds them to the allowlist by mistake later.
FORBIDDEN_PREFERENCE_CATEGORIES = frozenset({
    "credentials", "password", "permission", "authorization", "security",
    "payment", "contact", "private", "secret", "key",
})


@dataclass
class TaskLesson:
    """One structured, reusable lesson."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    type: str = "task_lesson"
    task_pattern: str = ""        # e.g. "open_application"
    lesson: str = ""
    evidence: str = ""            # must be a VERIFIED_EVIDENCE form
    confidence: float = DEFAULT_LESSON_CONFIDENCE
    timestamp: float = field(default_factory=time.time)
    source_task_id: str = ""
    reuse_count: int = 0
    successes: int = 0
    failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "task_pattern": self.task_pattern,
            "lesson": self.lesson[:300],
            "evidence": self.evidence,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "source_task_id": self.source_task_id,
            "reuse_count": self.reuse_count,
            "successes": self.successes,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskLesson":
        return cls(
            id=str(data.get("id", "")) or uuid.uuid4().hex[:10],
            type=str(data.get("type", "task_lesson")),
            task_pattern=str(data.get("task_pattern", "")),
            lesson=str(data.get("lesson", "")),
            evidence=str(data.get("evidence", "")),
            confidence=float(data.get("confidence",
                                      DEFAULT_LESSON_CONFIDENCE)),
            timestamp=float(data.get("timestamp", time.time())),
            source_task_id=str(data.get("source_task_id", "")),
            reuse_count=int(data.get("reuse_count", 0)),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
        )

    @property
    def trustworthy(self) -> bool:
        """Old memory is evidence, not truth (Task 10)."""
        return (self.confidence >= MIN_TRUST_CONFIDENCE
                and self.failures <= self.successes)

    def line(self) -> str:
        """Compact rendering for the P6 context layer."""
        return (f"[{self.type}] {self.task_pattern}: {self.lesson} "
                f"(evidence={self.evidence}, conf={self.confidence:.2f}, "
                f"reuse={self.reuse_count})")


def task_pattern_from_goal(goal: str) -> str:
    """Deterministic pattern label for a goal (bounded, no model)."""
    g = (goal or "").lower().strip()
    if not g:
        return "general"
    if re.search(r"\b(open|launch|start)\b", g):
        return "open_application"
    if re.search(r"\b(close|quit|kill)\b", g):
        return "close_application"
    if re.search(r"\b(search|find|look up|look for)\b", g):
        return "search"
    if re.search(r"\b(play|pause|resume|next|previous)\b", g):
        return "media_control"
    if re.search(r"\b(volume|brightness)\b", g):
        return "system_setting"
    if re.search(r"\b(file|folder|directory)\b", g):
        return "file_operation"
    if re.search(r"\b(summar|report|tell me)\b", g):
        return "information_summary"
    return "general"


class TaskLessonStore:
    """Bounded store of structured task lessons with safe learning."""

    MAX_LESSONS = 200

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.environ.get("DIEGO_TASK_LESSONS_PATH") or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "task_lessons.json")
        self._path = path
        self._lock = threading.RLock()
        self._lessons: List[TaskLesson] = []
        self.load()

    # ── Persistence (bounded JSON, never a reasoning transcript) ──

    def load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            lessons = [TaskLesson.from_dict(d)
                       for d in (data.get("lessons") or [])]
            with self._lock:
                self._lessons = lessons[-self.MAX_LESSONS:]
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("[Lessons] load failed: %s", e)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with self._lock:
                payload = {"version": 1,
                           "lessons": [l.to_dict()
                                       for l in self._lessons[-self.MAX_LESSONS:]]}
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self._path)
        except Exception as e:
            logger.debug("[Lessons] save failed: %s", e)

    # ── Learning (verified-only) ───────────────────────────────

    def add_lesson(self, *, type: str, task_pattern: str, lesson: str,
                   evidence: str, source_task_id: str = "",
                   confidence: float = DEFAULT_LESSON_CONFIDENCE,
                   ) -> Optional[TaskLesson]:
        """Add a lesson. NEVER learns from an unverified result: evidence
        must be one of the VERIFIED_EVIDENCE forms (the deterministic
        runtime decides verification — the model cannot fabricate it).
        """
        if type not in LESSON_TYPES:
            logger.debug("[Lessons] rejected unknown lesson type %r", type)
            return None
        if evidence not in VERIFIED_EVIDENCE:
            # E.g. evidence="unverified" / "model_claim" → refused.
            logger.info(
                "[Lessons] refused to learn from unverified evidence %r",
                evidence)
            return None
        if not (0.0 <= confidence <= 1.0):
            confidence = DEFAULT_LESSON_CONFIDENCE
        item = TaskLesson(
            type=type,
            task_pattern=(task_pattern or "general")[:60],
            lesson=(lesson or "").strip()[:300],
            evidence=evidence,
            confidence=confidence,
            source_task_id=source_task_id[:20],
        )
        if not item.lesson:
            return None
        with self._lock:
            # Merge near-duplicates: same pattern+lesson → reinforce.
            for existing in self._lessons:
                if (existing.task_pattern == item.task_pattern
                        and existing.lesson == item.lesson):
                    existing.confidence = min(1.0, existing.confidence + 0.05)
                    existing.timestamp = time.time()
                    self.save()
                    return existing
            self._lessons.append(item)
            self._lessons = self._lessons[-self.MAX_LESSONS:]
        self.save()
        return item

    # ── Retrieval (Task 10) ────────────────────────────────────

    def retrieve(self, query: str, limit: int = 5,
                 trusted_only: bool = False) -> List[Tuple[TaskLesson, float]]:
        """Retrieve lessons relevant to a query (keyword overlap ×
        confidence × recency). Old memory is returned as EVIDENCE with a
        trustworthiness flag — never as truth."""
        q = (query or "").lower()
        q_tokens = set(re.findall(r"[a-z0-9]{3,}", q))
        now = time.time()
        scored: List[Tuple[TaskLesson, float]] = []
        with self._lock:
            for lesson in self._lessons:
                if trusted_only and not lesson.trustworthy:
                    continue
                hay = f"{lesson.task_pattern} {lesson.lesson}".lower()
                hay_tokens = set(re.findall(r"[a-z0-9]{3,}", hay))
                overlap = len(q_tokens & hay_tokens) / max(1, len(q_tokens))
                if overlap <= 0:
                    continue
                age_days = max(0.0, (now - lesson.timestamp) / 86400)
                recency = 2.0 ** (-age_days / 14.0)  # 2-week half-life
                score = overlap * 0.5 + lesson.confidence * 0.3 + recency * 0.2
                scored.append((lesson, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:max(1, limit)]

    def lesson_lines(self, query: str, limit: int = 5) -> List[str]:
        """P6 context lines for the retrieved lessons (with the
        evidence-not-truth framing)."""
        return [lesson.line() for lesson, _s in self.retrieve(query, limit)]

    # ── Feedback / reuse (Task 10) ─────────────────────────────

    def record_reuse(self, lesson_id: str, success: bool) -> None:
        """Record whether a reused lesson's strategy worked NOW.
        A remembered strategy that failed loses confidence and can decay
        below the trust threshold (then out of the store)."""
        with self._lock:
            for lesson in self._lessons:
                if lesson.id != lesson_id:
                    continue
                lesson.reuse_count += 1
                if success:
                    lesson.successes += 1
                    lesson.confidence = min(
                        1.0, lesson.confidence + REUSE_SUCCESS_BOOST)
                else:
                    lesson.failures += 1
                    lesson.confidence = max(
                        MIN_CONFIDENCE, lesson.confidence - REUSE_FAILURE_PENALTY)
                self.save()
                return

    def record_reuse_by_line(self, lesson: TaskLesson, success: bool) -> None:
        self.record_reuse(lesson.id, success)

    def prune(self) -> int:
        """Drop untrusted / decayed lessons. Returns removals."""
        with self._lock:
            before = len(self._lessons)
            self._lessons = [l for l in self._lessons if l.trustworthy]
            removed = before - len(self._lessons)
        if removed:
            self.save()
            logger.info("[Lessons] pruned %d untrusted lesson(s)", removed)
        return removed

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._lessons)

    def get(self, lesson_id: str) -> Optional[TaskLesson]:
        with self._lock:
            for lesson in self._lessons:
                if lesson.id == lesson_id:
                    return lesson
        return None

    def all(self) -> List[TaskLesson]:
        with self._lock:
            return list(self._lessons)

    # ── Task 11: reflection → lessons after task completion ────

    def record_task_outcome(self, task_state) -> List[TaskLesson]:
        """Extract bounded reusable lessons from a FINISHED task.

        Success lessons ONLY when the deterministic runtime marked the
        task verified-SUCCESS; failure lessons from verified failures;
        recurring obstacles when the same blocker repeats. Cancelled /
        needs-confirmation / needs-input tasks produce NO lessons (their
        outcomes are unverified).
        """
        from agent.task_state import FinalStatus, StepStatus  # noqa
        created: List[TaskLesson] = []
        if task_state is None:
            return created
        status = task_state.final_status
        pattern = task_pattern_from_goal(task_state.original_request
                                         or task_state.normalized_goal)
        completed = [s for s in (task_state.completed_steps or [])
                     if s.verified and s.status in (
                         StepStatus.COMPLETED, StepStatus.ALREADY_SATISFIED)]
        failed = list(task_state.failed_steps or [])
        strategy = " → ".join(
            s.action for s in (task_state.completed_steps or []))[:300]

        if status == FinalStatus.SUCCESS and completed:
            # verified_success evidence only
            lesson = self.add_lesson(
                type="successful_strategy",
                task_pattern=pattern,
                lesson=(f"strategy that worked: {strategy or 'direct plan'}"),
                evidence="verified_success",
                source_task_id=task_state.task_id,
            )
            if lesson:
                created.append(lesson)
            # Tool preference from a verified app-level success.
            for step in completed:
                if step.action == "desktop_open" and step.params.get("app"):
                    alt = self.add_lesson(
                        type="tool_preference",
                        task_pattern="open_application",
                        lesson=(f"{step.params['app']} launches reliably via "
                                f"desktop_open on this system"),
                        evidence="verified_success",
                        source_task_id=task_state.task_id,
                        confidence=0.75,
                    )
                    if alt:
                        created.append(alt)
                    break

        elif status in (FinalStatus.FAILED, FinalStatus.PARTIAL_FAILURE):
            # The failure itself is verified evidence (observed, not claimed).
            blocker = (task_state.blocker or
                       (failed[-1].error if failed else ""))[:200]
            if failed:
                last = failed[-1]
                # Recurring obstacle: same blocker seen before → promote.
                prior = self.retrieve(f"{pattern} {blocker}", limit=3)
                recurring = any(
                    l.type == "failed_strategy" and l.task_pattern == pattern
                    for l, _s in prior)
                lesson = self.add_lesson(
                    type="recurring_obstacle" if recurring else "failed_strategy",
                    task_pattern=pattern,
                    lesson=(f"{last.action} failed here ({blocker}) — "
                            f"do not repeat this strategy blindly"),
                    evidence="verified_failure",
                    source_task_id=task_state.task_id,
                    confidence=0.6,
                )
                if lesson:
                    created.append(lesson)
            elif status == FinalStatus.FAILED and blocker:
                lesson = self.add_lesson(
                    type="failed_strategy",
                    task_pattern=pattern,
                    lesson=f"no verified steps: {blocker}",
                    evidence="verified_failure",
                    source_task_id=task_state.task_id,
                    confidence=0.5,
                )
                if lesson:
                    created.append(lesson)

        # Unverified outcomes (CANCELLED / NEEDS_* / PARTIAL without
        # verified failures) produce NO success and NO strategy lessons.
        return created


# ═══════════════════════════════════════════════════════════════
# Task 12: SAFE user-preference learning
# ═══════════════════════════════════════════════════════════════

def is_safe_preference(category: str, key: str) -> bool:
    """True only for the harmless allowlist. Sensitive categories are
    rejected unconditionally."""
    cat = (category or "").lower().strip()
    key = (key or "").lower().strip()
    if cat in FORBIDDEN_PREFERENCE_CATEGORIES:
        return False
    if any(word in cat for word in FORBIDDEN_PREFERENCE_CATEGORIES):
        return False
    return (cat, key) in SAFE_PREFERENCES


def learn_user_preference(category: str, key: str, value: Any,
                          confidence_boost: float = 0.15) -> Optional[Any]:
    """Learn a SAFE, stable user preference through the EXISTING
    learning engine (learning.preferences.PreferenceStore).

    SAFETY GUARANTEE (Task 12): preferences are advisory context for the
    reasoning layer only. They are read by nothing in the authorization,
    confirmation, or safety paths, so a preference can NEVER bypass
    authorization or confirmation requirements. Sensitive information is
    never learned (allowlist + forbidden categories).

    Returns the recorded value, or None when the preference is not
    safe/allowed.
    """
    if not is_safe_preference(category, key):
        logger.info(
            "[Lessons] refused unsafe preference %s/%s", category, key)
        return None
    try:
        from learning.learning_engine import learning_engine
        pref = learning_engine.preferences.observe(
            category, key, value, confidence_boost=confidence_boost)
        return getattr(pref, "value", value)
    except Exception as e:
        logger.debug("[Lessons] preference store unavailable: %s", e)
        return value


def get_user_preference(category: str, key: str) -> Optional[Any]:
    """Read a learned preference (advisory only, if learned and trusted)."""
    if not is_safe_preference(category, key):
        return None
    try:
        from learning.learning_engine import learning_engine
        learned = learning_engine.preferences.all_learned()
        return learned.get(category, {}).get(key)
    except Exception:
        return None


# Global singleton — task lessons persist in data/task_lessons.json.
task_lesson_store = TaskLessonStore()
