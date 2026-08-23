"""
Base plugin interface for Diego Desktop Assistant.

All plugins must inherit from BasePlugin and implement
the required methods. The PluginManager discovers,
loads, and manages plugin lifecycle.

Features:
- Health checks
- Dependency validation
- Version checks
- Timeout configuration
- Event filtering
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
    min_core_version: str = "1.0.0"
    timeout: float = 30.0  # Default timeout in seconds


class BasePlugin(ABC):
    """
    Abstract base class for all Diego plugins.

    Plugins hook into the event bus and provide slot-based
    functionality. They can handle events, execute commands,
    and expose their own metadata.

    Lifecycle:
        initialize() → handle_event()* → shutdown()

    Health:
        health_check() — returns True if plugin is healthy
    """

    def __init__(self):
        self.metadata: PluginMetadata = PluginMetadata(name=self.__class__.__name__)
        self._bus = None
        self._enabled = True
        self._health_failures = 0
        self._max_health_failures = 3

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

    async def health_check(self) -> bool:
        """
        Check if the plugin is healthy.

        Returns:
            True if plugin is healthy, False otherwise.
        """
        # Default: check if plugin is enabled
        if not self._enabled:
            return False
        return True

    def validate_dependencies(self) -> List[str]:
        """
        Validate that all dependencies are available.

        Returns:
            List of missing dependency names (empty if all satisfied).
        """
        missing = []
        for dep in self.metadata.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        return missing

    def validate_core_version(self, core_version: str) -> bool:
        """
        Validate that the core version meets the plugin's minimum requirement.

        Args:
            core_version: Current core version string.

        Returns:
            True if core version is sufficient.
        """
        try:
            from packaging.version import Version
            return Version(core_version) >= Version(self.metadata.min_core_version)
        except Exception:
            # If we can't compare versions, assume compatible
            return True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        self._enabled = False
        self._health_failures = 0

    def enable(self) -> None:
        self._enabled = True

    def record_health_failure(self) -> bool:
        """
        Record a health check failure.

        Returns:
            True if plugin should be disabled (too many failures).
        """
        self._health_failures += 1
        if self._health_failures >= self._max_health_failures:
            self.disable()
            return True
        return False

    def __repr__(self) -> str:
        return f"<Plugin {self.metadata.name} v{self.metadata.version}>"