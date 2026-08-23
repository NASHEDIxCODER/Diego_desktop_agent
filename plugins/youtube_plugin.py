"""
YouTube Plugin for Diego Desktop Assistant.

Fully async with configurable timeouts.
Browser is lazy-initialized and reused across calls.
Never blocks the assistant main loop.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from core.event_bus import bus, Event
from core.plugin_base import BasePlugin, PluginMetadata
from nlp.context import context_manager
from nlp.conversation_state import conversation_state, PendingAction

logger = logging.getLogger(__name__)

# Lazy-import YouTube controls
_youtube = None

# Plugin-level timeout
PLUGIN_TIMEOUT = 10.0


def _get_youtube():
    global _youtube
    if _youtube is None:
        from scripts import youtube as _youtube_module
        _youtube = _youtube_module
    return _youtube


class YouTubePlugin(BasePlugin):
    """
    Controls YouTube via Selenium/web automation.

    All operations are async with configurable timeouts.
    Browser is initialized lazily on first use.
    """

    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="YouTube",
            version="3.0.0",
            description="Hands-free YouTube control via voice",
            author="Diego Team",
            commands=[
                "play music on youtube", "open youtube", "pause", "resume",
                "next song", "previous song", "volume up/down", "mute",
                "forward/rewind", "speed up/down", "close youtube",
            ],
            events=[
                "youtube_open", "youtube_search", "youtube_pause",
                "youtube_resume", "youtube_next", "youtube_previous",
                "youtube_volume_up", "youtube_volume_down", "youtube_mute",
                "youtube_unmute", "youtube_seek_forward",
                "youtube_seek_backward", "youtube_speed_up",
                "youtube_speed_down", "youtube_close",
            ],
            timeout=PLUGIN_TIMEOUT,
        )
        self._active = False

    async def initialize(self) -> None:
        """Register event handlers."""
        bus.on("youtube_open", self._on_open)
        bus.on("youtube_search", self._on_search)
        bus.on("youtube_pause", self._on_pause)
        bus.on("youtube_resume", self._on_resume)
        bus.on("youtube_next", self._on_next)
        bus.on("youtube_previous", self._on_previous)
        bus.on("youtube_volume_up", self._on_volume_up)
        bus.on("youtube_volume_down", self._on_volume_down)
        bus.on("youtube_mute", self._on_mute)
        bus.on("youtube_unmute", self._on_unmute)
        bus.on("youtube_seek_forward", self._on_seek_forward)
        bus.on("youtube_seek_backward", self._on_seek_backward)
        bus.on("youtube_speed_up", self._on_speed_up)
        bus.on("youtube_speed_down", self._on_speed_down)
        bus.on("youtube_close", self._on_close)
        logger.info("YouTube plugin initialized")

    async def shutdown(self) -> None:
        """Close YouTube and clean up."""
        if self._active:
            try:
                yt = _get_youtube()
                yt.close_youtube()
            except Exception as e:
                logger.error("Error closing YouTube: %s", e)
        self._active = False

    async def _run_with_timeout(self, func, *args, **kwargs) -> Any:
        """Run a synchronous function in a thread pool with timeout."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                timeout=PLUGIN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("YouTube operation timed out after %ss", PLUGIN_TIMEOUT)
            return None
        except Exception as e:
            logger.error("YouTube operation failed: %s", e)
            return None

    async def _on_open(self, event: Event) -> Optional[str]:
        yt = _get_youtube()
        result = await self._run_with_timeout(yt.youtube)
        if result is False:
            return "Failed to open YouTube"
        self._active = True
        context_manager.set_active_session("youtube")
        conversation_state.set_pending(PendingAction.YOUTUBE_QUERY)
        await bus.emit("speak", {"text": "YouTube is open. What would you like to play?"})
        return "YouTube opened"

    async def _on_search(self, event: Event) -> Optional[str]:
        query = event.data.get("query", "")
        if not query:
            return "No search query provided"
        yt = _get_youtube()
        result = await self._run_with_timeout(yt.search_song, query)
        if result is False:
            return f"Failed to play {query}"
        self._active = True
        # Skip ad in background (non-blocking)
        asyncio.create_task(self._skip_ad_async())
        return f"Playing {query}"

    async def _skip_ad_async(self):
        """Skip ad in background without blocking."""
        try:
            await asyncio.sleep(2)
            yt = _get_youtube()
            await self._run_with_timeout(yt.skip_ad)
        except Exception:
            pass

    async def _on_pause(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.pause_or_play)
        return "Playback paused"

    async def _on_resume(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.pause_or_play)
        return "Playback resumed"

    async def _on_next(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.play_next_song)
        return "Next song"

    async def _on_previous(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.play_previous_song)
        return "Previous song"

    async def _on_volume_up(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.set_volume, 1.0)
        return "Volume increased"

    async def _on_volume_down(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.set_volume, 0.0)
        return "Volume decreased"

    async def _on_mute(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.toggle_mute)
        return "Muted"

    async def _on_unmute(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.toggle_mute)
        return "Unmuted"

    async def _on_seek_forward(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        seconds = event.data.get("seconds", 10)
        yt = _get_youtube()
        await self._run_with_timeout(yt.seek_forward, seconds)
        return f"Forward {seconds} seconds"

    async def _on_seek_backward(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        seconds = event.data.get("seconds", 10)
        yt = _get_youtube()
        await self._run_with_timeout(yt.seek_backward, seconds)
        return f"Backward {seconds} seconds"

    async def _on_speed_up(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.increase_speed)
        return "Speed increased"

    async def _on_speed_down(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.decrease_speed)
        return "Speed decreased"

    async def _on_close(self, event: Event) -> Optional[str]:
        if not self._active:
            return "YouTube is not active"
        yt = _get_youtube()
        await self._run_with_timeout(yt.close_youtube)
        self._active = False
        context_manager.set_active_session(None)
        return "YouTube closed"