"""
Abstract base class for TTS engines.

All TTS engines must inherit from BaseTTSEngine and implement:
- initialize()
- speak(text) -> bool
- is_speaking() -> bool
- close()
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class BaseTTSEngine(ABC):
    """Abstract base class for all TTS engines."""

    def __init__(self):
        self._ready = False
        self._speaking = False
        self._warning_shown = False
        self._on_started: Optional[Callable] = None
        self._on_finished: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the TTS engine. Returns True if successful."""
        ...

    @abstractmethod
    def speak(self, text: str) -> bool:
        """Speak the given text. Blocks until speech completes. Returns True if audio was produced."""
        ...

    @abstractmethod
    def is_speaking(self) -> bool:
        """Check if speech is currently in progress."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release TTS resources."""
        ...

    def set_callbacks(
        self,
        on_started: Optional[Callable] = None,
        on_finished: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for speech events."""
        self._on_started = on_started
        self._on_finished = on_finished
        self._on_error = on_error

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Engine", "").lower()