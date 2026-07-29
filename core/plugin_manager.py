"""
Plugin Manager for Leo Desktop Assistant.

Discovers, loads, and manages plugin lifecycle.
Plugins are loaded from the plugins/ directory and
registered with the event bus.

All plugin operations have configurable timeouts.
A hanging plugin never blocks the main loop.
"""

import asyncio
import importlib
import inspect
import logging
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Type

from core.event_bus import bus, Event
from core.plugin_base import BasePlugin

logger = logging.getLogger(__name__)

# Default timeout for plugin operations
DEFAULT_PLUGIN_TIMEOUT = 10.0


class PluginManager:
    """
    Manages all plugins: discovery, loading, initialization,
    and shutdown.

    All plugin operations are isolated with timeouts.
    A failing plugin never affects other plugins or the main loop.
    """

    def __init__(self, plugin_dir: Optional[Path] = None):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_dir = plugin_dir or Path(__file__).resolve().parent.parent / "plugins"

    @property
    def plugins(self) -> Dict[str, BasePlugin]:
        return dict(self._plugins)

    def discover(self) -> List[str]:
        """Scan the plugins directory for loadable plugins."""
        import pkgutil
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
        """Initialize all loaded plugins with timeout isolation."""
        tasks = []
        for name, plugin in self._plugins.items():
            task = self._initialize_one(name, plugin)
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _initialize_one(self, name: str, plugin: BasePlugin) -> None:
        """Initialize a single plugin with timeout."""
        timeout = getattr(plugin.metadata, 'timeout', DEFAULT_PLUGIN_TIMEOUT)
        try:
            await asyncio.wait_for(plugin.initialize(), timeout=timeout)
            logger.info("Initialized plugin: %s", name)
        except asyncio.TimeoutError:
            logger.error("Plugin %s initialization timed out after %ss — disabling", name, timeout)
            plugin.disable()
        except Exception as e:
            logger.error("Failed to initialize plugin %s: %s — disabling", name, e)
            plugin.disable()

    async def shutdown_all(self) -> None:
        """Shut down all plugins gracefully with timeout isolation."""
        tasks = []
        for name, plugin in self._plugins.items():
            task = self._shutdown_one(name, plugin)
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _shutdown_one(self, name: str, plugin: BasePlugin) -> None:
        """Shut down a single plugin with timeout."""
        timeout = getattr(plugin.metadata, 'timeout', DEFAULT_PLUGIN_TIMEOUT)
        try:
            await asyncio.wait_for(plugin.shutdown(), timeout=timeout)
            logger.info("Shut down plugin: %s", name)
        except asyncio.TimeoutError:
            logger.warning("Plugin %s shutdown timed out after %ss", name, timeout)
        except Exception as e:
            logger.warning("Error shutting down plugin %s: %s", name, e)

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)

    async def handle_event(self, event: Event, timeout: float = DEFAULT_PLUGIN_TIMEOUT) -> None:
        """
        Dispatch an event to all enabled plugins with timeout isolation.

        Each plugin runs independently with a timeout.
        A hanging or crashing plugin does not affect other plugins or the main loop.

        Args:
            event: The event to dispatch.
            timeout: Maximum seconds to wait per plugin handler.
        """
        tasks = []
        for name, plugin in self._plugins.items():
            if not plugin.enabled:
                continue
            task = self._handle_one(name, plugin, event, timeout)
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_one(self, name: str, plugin: BasePlugin,
                          event: Event, timeout: float) -> None:
        """Handle an event for a single plugin with timeout."""
        logger.info(
            "[PLUGIN] received intent='%s' → selected plugin='%s'",
            event.type, name,
        )

        start_ts = _time.monotonic()
        try:
            result = await asyncio.wait_for(
                plugin.handle_event(event.type, event.data),
                timeout=timeout
            )
            elapsed = _time.monotonic() - start_ts
            status = "success" if result is not None else "no_return"
            logger.info(
                "[PLUGIN] plugin='%s' event='%s' execute() → status='%s' "
                "returned='%s' time=%.3fs",
                name, event.type, status, result, elapsed,
            )
        except asyncio.TimeoutError:
            elapsed = _time.monotonic() - start_ts
            logger.error(
                "[PLUGIN] plugin='%s' event='%s' execute() → status='timeout' "
                "time=%.3fs — FAILED: plugin timed out after %ss",
                name, event.type, elapsed, timeout,
            )
            plugin.disable()
        except Exception as e:
            elapsed = _time.monotonic() - start_ts
            logger.error(
                "[PLUGIN] plugin='%s' event='%s' execute() → status='error' "
                "time=%.3fs — FAILED: %s",
                name, event.type, elapsed, e,
            )
            plugin.disable()


# Global singleton
plugin_manager = PluginManager()