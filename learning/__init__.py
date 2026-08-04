"""
Leo Desktop Assistant — Self-Learning Engine

Leo continuously learns from its owner WITHOUT retraining any AI model.
All learning is database-backed: frequencies, preferences, habits, and
skill memories are stored in DuckDB and improve organically over time.

    learning_engine.py   — orchestrator, hooks into event bus + action dispatcher
    user_profile.py      — learned facts (name, editor, language, schedule, projects)
    preferences.py       — key-value preference store with confidence decay
    habits.py            — frequency tracking over time windows (hourly/daily/weekly)
    skill_memory.py      — remembers which actions work and which fail

Zero model retraining. All learning is database writes.
"""

from learning.learning_engine import LearningEngine
from learning.user_profile import UserProfile
from learning.preferences import PreferenceStore
from learning.habits import HabitTracker
from learning.skill_memory import SkillMemory

__all__ = [
    "LearningEngine",
    "UserProfile",
    "PreferenceStore",
    "HabitTracker",
    "SkillMemory",
]