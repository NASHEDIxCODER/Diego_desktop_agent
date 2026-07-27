"""
Context manager for Leo NLP pipeline.

Maintains conversation context across turns, enabling
context-aware commands like "send a message" followed by
"to John" (implicitly referencing the previous intent).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context across turns.

    Stores:
    - Last intent
    - Last entities
    - Last N turns
    - Active plugin/session
    - User preferences (in-memory)
    """

    def __init__(self, max_history: int = 20):
        self._history: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._current_intent: Optional[str] = None
        self._current_entities: Dict[str, Any] = {}
        self._active_session: Optional[str] = None
        self._preferences: Dict[str, Any] = {}
        self._slot_filling: Dict[str, Any] = {}

    @property
    def current_intent(self) -> Optional[str]:
        return self._current_intent

    @property
    def current_entities(self) -> Dict[str, Any]:
        return dict(self._current_entities)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def update(self, text: str, intent: str, entities: Dict[str, Any],
               confidence: float) -> None:
        """Update context with a new turn."""
        turn = {
            "text": text,
            "intent": intent,
            "entities": entities,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._history.append(turn)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        self._current_intent = intent
        self._current_entities = entities

    def get_last_intent(self, offset: int = 1) -> Optional[str]:
        """Get the intent from N turns ago."""
        if len(self._history) >= offset:
            return self._history[-offset].get("intent")
        return None

    def get_last_entities(self, offset: int = 1) -> Dict[str, Any]:
        """Get entities from N turns ago."""
        if len(self._history) >= offset:
            return self._history[-offset].get("entities", {})
        return {}

    def get_last_text(self, offset: int = 1) -> Optional[str]:
        """Get text from N turns ago."""
        if len(self._history) >= offset:
            return self._history[-offset].get("text")
        return None

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        self._preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self._preferences.get(key, default)

    def set_slot(self, slot_name: str, value: Any) -> None:
        """Set a slot value for slot-filling."""
        self._slot_filling[slot_name] = value

    def get_slot(self, slot_name: str, default: Any = None) -> Any:
        """Get a slot value."""
        return self._slot_filling.get(slot_name, default)

    def clear_slots(self) -> None:
        """Clear all slot-filling state."""
        self._slot_filling.clear()

    def has_slots(self) -> bool:
        """Check if there are pending slots."""
        return bool(self._slot_filling)

    def set_active_session(self, session: Optional[str]) -> None:
        """Set the active plugin session (e.g., 'youtube')."""
        self._active_session = session

    @property
    def active_session(self) -> Optional[str]:
        return self._active_session

    def is_in_session(self, session_name: str) -> bool:
        """Check if currently in a specific session."""
        return self._active_session == session_name

    def clear(self) -> None:
        """Reset all context."""
        self._history.clear()
        self._current_intent = None
        self._current_entities = {}
        self._active_session = None
        self._slot_filling.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dict."""
        return {
            "history": self._history,
            "current_intent": self._current_intent,
            "current_entities": self._current_entities,
            "active_session": self._active_session,
            "preferences": self._preferences,
            "slot_filling": self._slot_filling,
        }


# Global context manager
context_manager = ContextManager()