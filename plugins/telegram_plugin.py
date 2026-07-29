"""
Telegram Plugin for Leo Desktop Assistant.

Wraps the existing scripts/telegram_bot.py functionality as a
BasePlugin with event bus integration.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from core.event_bus import bus, Event
from core.plugin_base import BasePlugin, PluginMetadata
from nlp.context import context_manager

logger = logging.getLogger(__name__)

# Lazy-import telegram module
_telegram = None


def _get_telegram():
    global _telegram
    if _telegram is None:
        try:
            from scripts import telegram_bot as _t
            _telegram = _t
        except Exception as e:
            logger.warning("Failed to import telegram_bot: %s", e)
            _telegram = False
    return _telegram if _telegram is not False else None


class TelegramPlugin(BasePlugin):
    """
    Send, read, and reply to Telegram messages.

    Responds to events:
    - telegram_send
    - telegram_read
    - telegram_reply
    """

    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="Telegram",
            version="2.0.0",
            description="Send, read and reply to Telegram messages",
            author="Leo Team",
            commands=[
                "send message to [contact]", "read messages from [contact]",
                "reply to last message",
            ],
            events=["telegram_send", "telegram_read", "telegram_reply"],
        )

    async def initialize(self) -> None:
        """Register event handlers and init Telegram client in background."""
        bus.on("telegram_send", self._on_send)
        bus.on("telegram_read", self._on_read)
        bus.on("telegram_reply", self._on_reply)

        # Initialize Telegram client in background — NEVER block startup.
        # If session is invalid, disable instantly and show one warning.
        asyncio.ensure_future(self._init_background())

    async def _init_background(self) -> None:
        """Initialize Telegram in background. Never blocks startup."""
        try:
            tg = _get_telegram()
            if tg is None:
                self.disable()
                return
            # Quick check: if no session file exists, disable immediately
            import os as _os
            from pathlib import Path as _Path
            session_files = [
                _Path("leo_telegram.session"),
                _Path("leo_telegram.session-journal"),
            ]
            has_session = any(f.exists() for f in session_files)
            if not has_session:
                logger.warning(
                    "Telegram session file not found — Telegram disabled. "
                    "Run `python scripts/telegram_bot.py` interactively once to create a session."
                )
                self.disable()
                return

            await tg.init()
            if tg._available:
                logger.info("Telegram client initialized")
            else:
                logger.warning("Telegram session invalid — Telegram disabled")
                self.disable()
        except ValueError as e:
            logger.warning("Telegram init failed (API version mismatch): %s", e)
            self.disable()
        except ImportError as e:
            logger.warning("Telegram init failed (missing dependency): %s", e)
            self.disable()
        except Exception as e:
            logger.warning("Telegram init failed (unexpected): %s", e)
            self.disable()

    async def shutdown(self) -> None:
        """Cleanup."""
        pass

    async def _on_send(self, event: Event) -> Optional[str]:
        tg = _get_telegram()
        target = event.data.get("target", "")
        message = event.data.get("message", "")

        if not target:
            return "I need a contact name to send the message."
        if not message:
            return "What should I say?"

        ok, err = await tg.send_message(target, message)
        if ok:
            return f"Message sent to {target}."
        return err or "Failed to send message."

    async def _on_read(self, event: Event) -> Optional[str]:
        tg = _get_telegram()
        target = event.data.get("target", "")

        if not target:
            return "From whom should I read messages?"

        msg = await tg.read_latest_message(target)
        if msg:
            return f"Message from {target}: {msg}"
        return f"No messages found from {target}."

    async def _on_reply(self, event: Event) -> Optional[str]:
        tg = _get_telegram()
        message = event.data.get("message", "")

        if not message:
            return "What should I reply?"

        ok, err = await tg.reply_message(message)
        if ok:
            return "Reply sent."
        return err or "Failed to send reply."