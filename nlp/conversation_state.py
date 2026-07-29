"""
ConversationStateManager — Tracks pending actions across turns.

When the assistant asks a follow-up question (e.g., "What would you like to play?"),
the next user utterance should NOT be classified as an intent. Instead, it should
be routed directly to the pending action handler.

States:
  waiting_for_youtube_query  — Next speech is a YouTube search query
  waiting_for_google_query   — Next speech is a Google search query
  waiting_for_note           — Next speech is a note to save
  waiting_for_confirmation   — Next speech is yes/no confirmation
  waiting_for_app_name       — Next speech is an app name to open
  none                       — No pending action, normal classification
"""

import asyncio
import logging
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class PendingAction(Enum):
    NONE = auto()
    YOUTUBE_QUERY = auto()
    GOOGLE_QUERY = auto()
    NOTE = auto()
    CONFIRMATION = auto()
    APP_NAME = auto()


class ConversationStateManager:
    """
    Manages conversation state across turns.

    When a pending action is set, the next user utterance bypasses
    intent classification and is routed directly to the action handler.
    """

    def __init__(self):
        self._pending_action: PendingAction = PendingAction.NONE
        self._pending_data: Dict[str, Any] = {}
        self._handlers: Dict[PendingAction, Callable] = {}

    @property
    def pending_action(self) -> PendingAction:
        return self._pending_action

    @property
    def has_pending_action(self) -> bool:
        return self._pending_action != PendingAction.NONE

    def set_pending(self, action: PendingAction, data: Optional[Dict[str, Any]] = None) -> None:
        """Set a pending action for the next user utterance."""
        self._pending_action = action
        self._pending_data = data or {}
        logger.info("Pending action set: %s", action.name)

    def clear(self) -> None:
        """Clear any pending action."""
        self._pending_action = PendingAction.NONE
        self._pending_data = {}

    def register_handler(self, action: PendingAction, handler: Callable) -> None:
        """Register a handler for a pending action."""
        self._handlers[action] = handler

    async def handle(self, text: str) -> Optional[str]:
        """
        Handle a user utterance based on pending action.

        Args:
            text: The user's spoken text.

        Returns:
            Response string, or None if no pending action.
        """
        if not self.has_pending_action:
            return None

        action = self._pending_action
        data = self._pending_data
        self.clear()  # Clear before handling to prevent re-entry

        handler = self._handlers.get(action)
        if handler is None:
            logger.warning("No handler registered for action: %s", action.name)
            return None

        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(text, data)
            return handler(text, data)
        except Exception as e:
            logger.error("Pending action handler error (%s): %s", action.name, e, exc_info=True)
            return None


# Global singleton
conversation_state = ConversationStateManager()