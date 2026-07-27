"""
Base plugin interface for Leo Desktop Assistant.

All plugins must inherit from BasePlugin and implement
the required methods. The PluginManager discovers,
loads, and manages plugin lifecycle.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)


class BasePlugin(ABC):
    """
    Abstract base class for all Leo plugins.

    Plugins hook into the event bus and provide slot-based
    functionality. They can handle events, execute commands,
    and expose their own metadata.
    """

    def __init__(self):
        self.metadata: PluginMetadata = PluginMetadata(name=self.__class__.__name__)
        self._bus = None
        self._enabled = True

    @abstractmethod
    async def initialize(self) -> None:
        """Called when plugin is loaded. Register event handlers here."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Called when plugin is unloaded. Clean up resources."""
        ...

    async def handle_event(self, event_type: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Handle an incoming event. Return a response string if applicable.
        Default implementation dispatches to on_<event_type> methods.
        """
        handler_name = f"on_{event_type.replace('.', '_')}"
        handler = getattr(self, handler_name, None)
        if handler:
            return await handler(data)
        return None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def __repr__(self) -> str:
        return f"<Plugin {self.metadata.name} v{self.metadata.version}>"