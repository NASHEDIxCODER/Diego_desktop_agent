"""
Plugin Manager for Leo Desktop Assistant.

Discovers, loads, and manages plugin lifecycle.
Plugins are loaded from the plugins/ directory and
registered with the event bus.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

from core.event_bus import bus, Event
from core.plugin_base import BasePlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages all plugins: discovery, loading, initialization,
    and shutdown.
    """

    def __init__(self, plugin_dir: Optional[Path] = None):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_dir = plugin_dir or Path(__file__).resolve().parent.parent / "plugins"

    @property
    def plugins(self) -> Dict[str, BasePlugin]:
        return dict(self._plugins)

    def discover(self) -> List[str]:
        """Scan the plugins directory for loadable plugins."""
        plugin_names = []
        if not self._plugin_dir.exists():
            logger.warning("Plugin directory %s does not exist", self._plugin_dir)
            return plugin_names

        for importer, modname, ispkg in pkgutil.iter_modules([str(self._plugin_dir)]):
            if ispkg:
                continue
            if modname.startswith("_"):
                continue
            plugin_names.append(modname)

        logger.info("Discovered plugins: %s", plugin_names)
        return plugin_names

    def load_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Load a single plugin by name."""
        try:
            module = importlib.import_module(f"plugins.{plugin_name}")
        except ImportError as e:
            logger.error("Failed to import plugin %s: %s", plugin_name, e)
            return None

        # Find all BasePlugin subclasses in the module
        plugin_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and issubclass(obj, BasePlugin)
                    and obj is not BasePlugin):
                plugin_classes.append(obj)

        if not plugin_classes:
            logger.warning("No BasePlugin subclass found in %s", plugin_name)
            return None

        # Instantiate the first plugin class
        plugin_class = plugin_classes[0]
        try:
            plugin = plugin_class()
            self._plugins[plugin.metadata.name] = plugin
            logger.info("Loaded plugin: %s v%s", plugin.metadata.name, plugin.metadata.version)
            return plugin
        except Exception as e:
            logger.error("Failed to instantiate plugin %s: %s", plugin_name, e)
            return None

    async def load_all(self) -> Dict[str, BasePlugin]:
        """Discover and load all plugins."""
        names = self.discover()
        for name in names:
            self.load_plugin(name)
        return self._plugins

    async def initialize_all(self) -> None:
        """Initialize all loaded plugins."""
        for name, plugin in self._plugins.items():
            try:
                await plugin.initialize()
                logger.info("Initialized plugin: %s", name)
            except Exception as e:
                logger.error("Failed to initialize plugin %s: %s", name, e)

    async def shutdown_all(self) -> None:
        """Shut down all plugins gracefully."""
        for name, plugin in self._plugins.items():
            try:
                await plugin.shutdown()
                logger.info("Shut down plugin: %s", name)
            except Exception as e:
                logger.error("Error shutting down plugin %s: %s", name, e)

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)

    async def handle_event(self, event: Event) -> None:
        """Dispatch an event to all enabled plugins."""
        for name, plugin in self._plugins.items():
            if not plugin.enabled:
                continue
            try:
                await plugin.handle_event(event.type, event.data)
            except Exception as e:
                logger.error("Plugin %s error handling event %s: %s",
                             name, event.type, e)


# Global singleton
plugin_manager = PluginManager()