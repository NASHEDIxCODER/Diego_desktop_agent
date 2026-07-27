"""
Telegram Plugin for Leo Desktop Assistant.

Wraps the existing scripts/telegram_bot.py functionality as a
BasePlugin with event bus integration.
"""

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
        from scripts import telegram_bot as _t
        _telegram = _t
    return _telegram


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
        """Register event handlers and init Telegram client."""
        bus.on("telegram_send", self._on_send)
        bus.on("telegram_read", self._on_read)
        bus.on("telegram_reply", self._on_reply)

        # Initialize Telegram client
        try:
            tg = _get_telegram()
            await tg.init()
            logger.info("Telegram client initialized")
        except Exception as e:
            logger.warning("Telegram init failed: %s", e)

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