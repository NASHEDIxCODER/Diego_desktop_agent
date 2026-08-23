"""
LearningEngine — Orchestrates all self-learning subsystems.

Hooks into the event bus and action dispatcher to observe user behavior
and build a continuously improving profile WITHOUT retraining any AI model.

Architecture:
    ActionDispatcher.execute()
        ↓
    LearningEngine.record_action()   (event bus subscriber)
        ↓
    ├── UserProfile.observe()        (apps, projects, coding habits)
    ├── HabitTracker.observe()       (frequencies over time windows)
    ├── PreferenceStore.observe()    (volume, brightness, style)
    └── SkillMemory.record()         (success/failure outcomes)

The learned context is injected into LLM prompts via:
    learning_engine.llm_context() → compact text block

Persisted in DuckDB via `memory/duckdb_store.py` (no new tables needed —
we use the existing `user_preferences` table extended with a JSON blob).

Usage:
    from learning.learning_engine import learning_engine

    # Wire into action dispatcher
    learning_engine.wire(action_dispatcher)

    # Inject into LLM
    prompt += learning_engine.llm_context()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.service import BaseService
from learning.user_profile import UserProfile
from learning.preferences import PreferenceStore
from learning.habits import HabitTracker
from learning.skill_memory import SkillMemory

logger = logging.getLogger(__name__)

# How often to persist to DuckDB (seconds)
PERSIST_INTERVAL_S = 60.0

# Path for local JSON backup (fallback if DuckDB unavailable)
PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "learner_profile.json"


class LearningEngine(BaseService):
    """
    Continuous self-learning engine for Diego.

    Extends BaseService for lifecycle management. Hooks into the
    event bus to observe every user action and builds a profile
    over time. Never modifies model weights.

    Service lifecycle:
        start()  → loads persisted profile, subscribes to event bus
        stop()   → persists profile to DuckDB, unsubscribes
    """

    name = "learning_engine"
    dependencies: List[str] = []

    # App name → category classification heuristics
    _EDITOR_APPS = {"code", "vscode", "pycharm", "sublime", "vim", "nvim", "neovim",
                     "emacs", "atom", "zed", "idea", "intellij", "notepad", "gedit"}
    _BROWSER_APPS = {"firefox", "chrome", "chromium", "brave", "edge", "safari",
                      "opera", "vivaldi", "floorp", "waterfox", "librewolf"}
    _TERMINAL_APPS = {"terminal", "gnome-terminal", "konsole", "alacritty",
                       "kitty", "wezterm", "iterm2", "tilix", "xfce4-terminal"}

    def __init__(self):
        super().__init__()
        self._profile = UserProfile()
        self._prefs = PreferenceStore()
        self._habits = HabitTracker()
        self._skills = SkillMemory()
        self._persist_task: Optional[asyncio.Task] = None
        self._action_count: int = 0

    # ── BaseService contract ──────────────────────────────────────

    async def _start(self) -> bool:
        """Load persisted profile and start periodic persistence."""
        self._load()
        self._start_persist_loop()
        logger.info("[Learn] Learning engine ready — %d facts, %d habits, %d skills",
                     self._profile.fact_count, self._habits.entry_count,
                     self._skills.entry_count)
        self.set_health("ready", {
            "facts": self._profile.fact_count,
            "habits": self._habits.entry_count,
            "skills_entries": self._skills.entry_count,
        })
        return True

    async def _stop(self) -> None:
        """Persist and stop."""
        if self._persist_task:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except (asyncio.CancelledError, Exception):
                pass
            self._persist_task = None
        self._save()
        # Apply decay on shutdown
        self._profile.decay()
        self._prefs.decay()
        self._habits.decay()
        logger.info("[Learn] Learning engine stopped — %d actions observed",
                     self._action_count)

    # ── Persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        """Load persisted data from DuckDB or JSON fallback."""
        # Try DuckDB first
        try:
            from memory.duckdb_store import DuckDBStore
            store = DuckDBStore()
            with store.connect() as conn:
                result = conn.execute(
                    "SELECT value FROM user_preferences WHERE key = 'learner_profile'"
                ).fetchone()
                if result and result[0]:
                    data = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    self._profile = UserProfile.from_dict(
                        data.get("profile", {"facts": []}))
                    self._prefs = PreferenceStore.from_dict(
                        data.get("preferences", {"preferences": []}))
                    self._habits = HabitTracker.from_dict(
                        data.get("habits", {"habits": []}))
                    self._skills = SkillMemory.from_dict(
                        data.get("skills", {"entries": []}))
                    logger.info("[Learn] Loaded profile from DuckDB")
                    return
        except Exception as e:
            logger.debug("[Learn] DuckDB load failed: %s — trying JSON", e)

        # JSON fallback
        try:
            if PROFILE_PATH.exists():
                with open(PROFILE_PATH, "r") as f:
                    data = json.load(f)
                self._profile = UserProfile.from_dict(
                    data.get("profile", {}))
                self._prefs = PreferenceStore.from_dict(
                    data.get("preferences", {}))
                self._habits = HabitTracker.from_dict(
                    data.get("habits", {}))
                self._skills = SkillMemory.from_dict(
                    data.get("skills", {}))
                logger.info("[Learn] Loaded profile from JSON fallback")
        except Exception as e:
            logger.debug("[Learn] No persisted profile found — starting fresh: %s", e)

    def _save(self) -> None:
        """Persist to DuckDB (primary) + JSON (fallback)."""
        data = {
            "profile": self._profile.to_dict(),
            "preferences": self._prefs.to_dict(),
            "habits": self._habits.to_dict(),
            "skills": self._skills.to_dict(),
            "updated_at": time.time(),
        }

        # DuckDB primary
        try:
            from memory.duckdb_store import DuckDBStore
            store = DuckDBStore()
            with store.connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
                    VALUES ('learner_profile', ?, CURRENT_TIMESTAMP)
                """, [json.dumps(data)])
            logger.debug("[Learn] Profile persisted to DuckDB")
        except Exception as e:
            logger.debug("[Learn] DuckDB persist failed: %s — using JSON", e)
            # JSON fallback
            try:
                PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(PROFILE_PATH, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e2:
                logger.warning("[Learn] JSON persist also failed: %s", e2)

    def _start_persist_loop(self) -> None:
        """Periodic persistence task."""
        async def persist_loop():
            try:
                while True:
                    await asyncio.sleep(PERSIST_INTERVAL_S)
                    self._save()
            except asyncio.CancelledError:
                pass
        try:
            loop = asyncio.get_running_loop()
            self._persist_task = loop.create_task(persist_loop())
        except RuntimeError:
            pass

    # ── Public observation API ────────────────────────────────────

    def record_action(self, action_name: str, params: Dict[str, Any],
                      success: bool = True, latency_ms: float = 0.0,
                      error: str = "") -> None:
        """
        Record an executed action for learning.

        Called by ActionDispatcher after every action.execute().

        Args:
            action_name: e.g. "desktop_open", "browser_navigate"
            params: action parameters (app name, URL, etc.)
            success: did the action succeed
            latency_ms: how long it took
            error: error message if failed
        """
        self._action_count += 1
        now = time.time()

        # ── User profile: classify app usage ──────────────
        app = params.get("app", "")
        if action_name == "desktop_open" and app:
            app_lower = app.lower().strip()
            self._profile.observe("app", "frequency", app_lower)
            self._habits.observe("app", app_lower)

            # Classify app type
            if app_lower in self._EDITOR_APPS:
                self._profile.observe("app", "editor", app_lower)
            elif app_lower in self._BROWSER_APPS:
                self._profile.observe("app", "browser", app_lower)
            elif app_lower in self._TERMINAL_APPS:
                self._profile.observe("app", "terminal", app_lower)

        # ── Browser / URL usage ───────────────────────────
        url = params.get("url", "")
        if action_name in ("browser_navigate", "browser_search") and url:
            self._habits.observe("url", url[:200])  # truncate long URLs

        # ── Volume / brightness preferences ───────────────
        if action_name in ("volume_up", "volume_down", "volume_set"):
            pct = params.get("percent", params.get("level", 50))
            self._prefs.observe("audio", "volume", pct)
        if action_name in ("brightness_up", "brightness_down", "brightness_set"):
            pct = params.get("percent", params.get("level", 70))
            self._prefs.observe("display", "brightness", pct)

        # ── Folder tracking ──────────────────────────────
        folder = params.get("path", "")
        if action_name == "open_folder" and folder:
            self._habits.observe("folder", folder)

        # ── Schedule tracking ────────────────────────────
        hour = time.localtime(now).tm_hour
        self._habits.observe("schedule", str(hour))

        # ── Skill memory: record outcome ─────────────────
        goal = f"{action_name}:{app or url or folder}"
        self._skills.record(
            goal=goal,
            action=f"{action_name}:{app or url}",
            success=success,
            latency_ms=latency_ms,
            error=error,
            params=params,
        )

    def record_conversation_turn(self, user_text: str,
                                  assistant_text: str) -> None:
        """
        Record a conversation turn to learn:
          - Projects mentioned ("I'm building GhostLine")
          - Coding language ("I'm writing Python")
          - Preferred response style (brief vs detailed)
        """
        # ── Project detection ────────────────────────────
        project_keywords = [
            "building", "working on", "project", "coding", "developing",
            "creating", "making", "writing", "the app", "my app",
        ]
        user_lower = user_text.lower()
        for keyword in project_keywords:
            idx = user_lower.find(keyword)
            if idx >= 0:
                # Extract the next 1-3 words as project name
                after = user_text[idx + len(keyword):].strip()
                words = after.split()[:3]
                project_name = " ".join(words).rstrip(".,!?;:-\"'")
                if len(project_name) > 1 and len(project_name) < 50:
                    self._profile.observe("project", "current", project_name)
                    self._habits.observe("project", project_name)
                break

        # ── Coding language detection ────────────────────
        lang_keywords = {
            "python": "python", "javascript": "javascript",
            "typescript": "typescript", "rust": "rust", "go": "go",
            "java": "java", "c++": "cpp", "c#": "csharp",
            "ruby": "ruby", "php": "php", "swift": "swift",
            "kotlin": "kotlin", "scala": "scala", "elixir": "elixir",
        }
        for word, lang in lang_keywords.items():
            if word in user_lower:
                self._profile.observe("coding", "language", lang)
                break

        # ── Reply length preference ──────────────────────
        asst_len = len(assistant_text.split())
        if asst_len < 10:
            self._prefs.observe("style", "verbosity", "brief")
        elif asst_len > 40:
            self._prefs.observe("style", "verbosity", "detailed")
        # Medium-length responses are neutral and don't move the needle

    # ── LLM context injection ─────────────────────────────────────

    def llm_context(self) -> str:
        """
        Return a compact text block for injection into LLM prompts.

        Includes learned facts, habits, and preferences that help
        the LLM personalize its responses.
        """
        parts = []

        profile_ctx = self._profile.llm_context()
        if profile_ctx:
            parts.append(profile_ctx)

        habits_ctx = self._habits.llm_context()
        if habits_ctx:
            parts.append(habits_ctx)

        prefs_ctx = self._prefs.llm_context()
        if prefs_ctx:
            parts.append(prefs_ctx)

        skills_ctx = self._skills.llm_context()
        if skills_ctx:
            parts.append(skills_ctx)

        return "\n".join(parts) if parts else ""

    # ── Queries for the planner ───────────────────────────────────

    def best_app(self, app_type: str) -> Optional[str]:
        """Return the user's preferred app of a given type."""
        fact = self._profile.get_best("app", app_type)
        if fact:
            return fact.value
        # Fall back to habit tracker
        best = self._habits.get_best("app", "daily")
        return best.value if best else None

    def best_action_for(self, goal: str) -> Optional[str]:
        """Return the historically most successful action for a goal."""
        return self._skills.best_action(goal)

    # ── Diagnostics ───────────────────────────────────────────────

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def profile(self) -> UserProfile:
        return self._profile

    @property
    def preferences(self) -> PreferenceStore:
        return self._prefs

    @property
    def habits(self) -> HabitTracker:
        return self._habits

    @property
    def skills(self) -> SkillMemory:
        return self._skills


# Global singleton
learning_engine = LearningEngine()