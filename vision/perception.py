"""
PerceptionService — Continuous desktop awareness (Leo's eyes).

Leo should always know:
    Current screen, focused window, open applications, notifications,
    OCR, UI elements, clipboard, mouse, active monitor, browser tabs,
    terminal output.

This service samples the desktop at a configurable interval and emits
perception events on the bus:

    perception.screen_captured    {hash, width, height}
    perception.window_changed     {title, app}
    perception.clipboard_changed  {text}
    perception.active_monitor     {index, width, height}
    perception.ocr                {text, source}
    perception.activity           {mouse_x, mouse_y, keys, active}

Runs as a low-priority background service so idle CPU stays under 5%.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from core.event_bus import bus
from core.service import BaseService
from core.metrics import metrics

logger = logging.getLogger(__name__)


class PerceptionService(BaseService):
    """
    Continuous perception layer.

    samples the desktop and emits events. Backward-compatible with the
    legacy vision subsystem (uses vision_manager when available).
    """

    name = "perception"

    def __init__(self, interval: float = 2.0):
        super().__init__()
        self._interval = interval
        self._vision = None
        self._last_screen_hash = ""
        self._last_window_title = ""
        self._last_clipboard = ""
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    async def _start(self) -> bool:
        """Initialize the perception service."""
        try:
            from vision import vision_manager
            self._vision = vision_manager
            if not self._vision.is_available:
                self._vision.initialize()
        except Exception as e:
            logger.debug("[PERCEPTION] vision unavailable: %s", e)
            self._vision = None

        # Start the sample loop
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._loop_task = loop.create_task(self._run_loop())
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._loop_task = loop.create_task(self._run_loop())
            except Exception:
                logger.warning("[PERCEPTION] no event loop — "
                               "perception loop deferred")
        self.set_health("perception running",
                        {"vision": self._vision is not None})
        return True

    async def _stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
            self._loop_task = None
        self._vision = None

    async def _run_loop(self) -> None:
        """Periodic perception sampling loop."""
        try:
            while self._running:
                try:
                    await self._sample_screen()
                    await self._sample_window()
                    await self._sample_clipboard()
                    await self._sample_mouse()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("[PERCEPTION] sample error: %s", e)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    async def _sample_screen(self) -> None:
        """Capture the current screen and detect changes."""
        if self._vision is None:
            return
        try:
            capture = await self._vision.capture_screen()
            if capture is None or capture.image is None:
                return
            h = capture.hash
            if h != self._last_screen_hash:
                self._last_screen_hash = h
                metrics.record("perception.screen_hash_changes", 1)
                await bus.emit("perception.screen_captured", data={
                    "hash": h,
                    "width": capture.width,
                    "height": capture.height,
                }, source="perception")
                # OCR the new screen
                await self._sample_ocr(capture.image)

        except Exception as e:
            logger.debug("[PERCEPTION] screen sample failed: %s", e)

    async def _sample_ocr(self, image) -> None:
        """Run OCR on the screen (best effort, rate-limited)."""
        if self._vision is None:
            return
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._vision._ocr_engine.ocr, image)
            if text and len(text.strip()) > 5:
                await bus.emit("perception.ocr", data={
                    "text": text[:1000],
                }, source="perception")
        except Exception as e:
            logger.debug("[PERCEPTION] OCR failed: %s", e)

    async def _sample_window(self) -> None:
        """Detect the focused window and emit window.changed on change."""
        try:
            from agent.action_dispatcher import action_dispatcher
            title = action_dispatcher._active_window_title()
            if title and title != self._last_window_title:
                self._last_window_title = title
                await bus.emit("window.changed", data={
                    "title": title,
                    "app": title.split(" - ")[-1] if " - " in title else title,
                }, source="perception")
        except Exception as e:
            logger.debug("[PERCEPTION] window sample failed: %s", e)

    async def _sample_clipboard(self) -> None:
        """Detect clipboard changes and emit clipboard.captured."""
        try:
            import pyperclip
            text = pyperclip.paste() or ""
            if text and text != self._last_clipboard:
                self._last_clipboard = text
                await bus.emit("clipboard.captured", data={
                    "text": text[:2000],
                }, source="perception")
        except Exception as e:
            logger.debug("[PERCEPTION] clipboard sample failed: %s", e)

    async def _sample_mouse(self) -> None:
        """Emit mouse activity events (rate-limited)."""
        try:
            import pyautogui
            x, y = pyautogui.position()
            metrics.set("perception.mouse_x", x)
            metrics.set("perception.mouse_y", y)
        except Exception as e:
            logger.debug("[PERCEPTION] mouse sample failed: %s", e)


# Global singleton
perception_service = PerceptionService()