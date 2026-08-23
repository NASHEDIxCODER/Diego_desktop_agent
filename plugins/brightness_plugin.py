"""
Brightness Control Plugin for Diego Desktop Assistant.

Wraps the existing scripts/brightness.py functionality as a
BasePlugin with event bus integration.
"""

import logging
from typing import Any, Dict, Optional

from core.event_bus import bus, Event
from core.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger(__name__)

# Lazy-import brightness module
_brightness = None


def _get_brightness():
    global _brightness
    if _brightness is None:
        from scripts import brightness as _b
        _brightness = _b
    return _brightness


class BrightnessPlugin(BasePlugin):
    """
    Controls screen brightness.

    Responds to events:
    - brightness_set
    - brightness_up
    - brightness_down
    """

    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="Brightness",
            version="2.0.0",
            description="Control screen brightness via voice",
            author="Diego Team",
            commands=["set brightness to N", "brightness up", "brightness down"],
            events=["brightness_set", "brightness_up", "brightness_down"],
        )

    async def initialize(self) -> None:
        """Register event handlers."""
        bus.on("brightness_set", self._on_set)
        bus.on("brightness_up", self._on_up)
        bus.on("brightness_down", self._on_down)
        logger.info("Brightness plugin initialized")

    async def shutdown(self) -> None:
        pass

    async def _on_set(self, event: Event) -> Optional[str]:
        value = event.data.get("brightness") or event.data.get("value")
        if value is None:
            numbers = event.data.get("numbers", [])
            if numbers:
                value = numbers[0]
        if value is None:
            return "Please specify a brightness level between 0 and 100."

        value = max(0, min(100, int(value)))
        b = _get_brightness()
        b.set_brightness(value)
        return f"Brightness set to {value} percent."

    async def _on_up(self, event: Event) -> Optional[str]:
        b = _get_brightness()
        # Get current brightness (from dbus/brightnessctl)
        import subprocess
        try:
            result = subprocess.run(
                ["brightnessctl", "g"],
                capture_output=True, text=True, check=True
            )
            current = int(result.stdout.strip())
            result = subprocess.run(
                ["brightnessctl", "m"],
                capture_output=True, text=True, check=True
            )
            max_val = int(result.stdout.strip())
            new_val = min(current + 10, max_val)
            pct = int((new_val / max_val) * 100)
            b.set_brightness(pct)
            return f"Brightness increased to {pct} percent."
        except Exception as e:
            logger.error("Brightness up error: %s", e)
            return "I couldn't adjust the brightness."

    async def _on_down(self, event: Event) -> Optional[str]:
        b = _get_brightness()
        import subprocess
        try:
            result = subprocess.run(
                ["brightnessctl", "g"],
                capture_output=True, text=True, check=True
            )
            current = int(result.stdout.strip())
            result = subprocess.run(
                ["brightnessctl", "m"],
                capture_output=True, text=True, check=True
            )
            max_val = int(result.stdout.strip())
            new_val = max(current - 10, 0)
            pct = int((new_val / max_val) * 100)
            b.set_brightness(pct)
            return f"Brightness decreased to {pct} percent."
        except Exception as e:
            logger.error("Brightness down error: %s", e)
            return "I couldn't adjust the brightness."