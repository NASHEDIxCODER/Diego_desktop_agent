"""
Async Event Bus for Leo Desktop Assistant.

Decouples components via publish/subscribe pattern.
Plugins and internal modules communicate through the bus
instead of direct imports.
"""

import asyncio
import logging
from typing import Callable, Coroutine, Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventHandler = Callable[..., Coroutine[Any, Any, None]]


@dataclass
class Event:
    """A single event on the bus."""
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: int = 0

    def __repr__(self) -> str:
        return f"Event({self.type}, source={self.source}, data={self.data})"


class EventBus:
    """
    Simple async event bus.

    Plugins register handlers for event types they care about.
    When an event is emitted, all matching handlers are awaited.
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._wildcard_handlers: List[EventHandler] = []
        self._event_counter: int = 0
        self._lock = asyncio.Lock()

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        if event_type == "*":
            self._wildcard_handlers.append(handler)
            return
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler %s for event %s", handler.__name__, event_type)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Unregister a handler."""
        if event_type == "*":
            self._wildcard_handlers.remove(handler)
            return
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def emit(self, event_type: str, data: Optional[Dict[str, Any]] = None,
                   source: Optional[str] = None) -> None:
        """Emit an event to all registered handlers."""
        async with self._lock:
            self._event_counter += 1
            event_id = self._event_counter

        event = Event(
            type=event_type,
            data=data or {},
            source=source,
            id=event_id,
        )

        logger.debug("Emitting event: %s", event)

        # Collect all handlers
        handlers: List[EventHandler] = list(self._wildcard_handlers)
        if event_type in self._handlers:
            handlers.extend(self._handlers[event_type])

        # Run all handlers concurrently
        if handlers:
            tasks = [handler(event) for handler in handlers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Handler %s failed on event %s: %s",
                                 handlers[i].__name__, event_type, result)


# Global singleton bus
bus = EventBus()