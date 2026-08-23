# scripts/telegram_bot.py
#
# CRITICAL: This module must NEVER request interactive input (stdin).
# If credentials or session are missing, init() sets _available=False
# and all operations are no-ops. Startup never blocks.

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

API_ID = 35010936
API_HASH = "ebea5ed66cad2c023c000cc7e284ac21"
SESSION = "Diego_telegram"

# Lazy client — created only on first use
_client = None
DIALOG_CACHE = None
LAST_CONTACT: Optional[str] = None
_available = False


async def _get_client():
    """Get or create Telegram client. Never prompts stdin."""
    global _client, _available
    if _client is not None:
        return _client

    try:
        from telethon import TelegramClient
        from telethon.errors import RPCError
        _client = TelegramClient(SESSION, API_ID, API_HASH)
    except ImportError:
        logger.warning("Telethon not installed — Telegram unavailable")
        _available = False
        return None
    except Exception as e:
        logger.warning("Failed to create Telegram client: %s", e)
        _available = False
        return None

    return _client


async def _ensure_available() -> bool:
    """
    Check if Telegram is available without blocking on stdin.
    
    If no session file exists, we detect this and set _available=False.
    We NEVER call start() which prompts for phone number.
    """
    global _available
    if _available:
        return True

    client = await _get_client()
    if client is None:
        return False

    # If already connected, we're good
    if client.is_connected():
        _available = True
        return True

    # Check if session file exists — if not, we can't auth
    import os
    from pathlib import Path

    session_files = [
        Path(f"{SESSION}.session"),
        Path(f"{SESSION}.session-journal"),
    ]
    has_session = any(f.exists() for f in session_files)

    if not has_session:
        logger.warning(
            "Telegram session file '%s.session' not found. "
            "Telegram features disabled. "
            "To enable, run `python scripts/telegram_bot.py` interactively "
            "once to create a session.",
            SESSION
        )
        _available = False
        return False

    # Try to connect without stdin
    try:
        await client.connect()
        # Check if authorized without prompting
        if await client.is_user_authorized():
            _available = True
            logger.info("Telegram client connected and authorized")
            return True
        else:
            logger.warning(
                "Telegram session exists but not authorized. "
                "Delete %s.session and re-authorize interactively.",
                SESSION
            )
            _available = False
            return False
    except Exception as e:
        logger.warning("Telegram connect failed: %s", e)
        _available = False
        return False


async def _load_dialogs(force: bool = False):
    """Load dialogs once and reuse them."""
    global DIALOG_CACHE
    if not await _ensure_available():
        return

    if DIALOG_CACHE is None or force and _client is not None:
        try:
            DIALOG_CACHE = await _client.get_dialogs(limit=200)
        except Exception as e:
            logger.warning("Failed to load Telegram dialogs: %s", e)


async def find_dialog(name: str):
    """Return dialog whose name contains the given text (case-insensitive)."""
    if not name:
        return None

    await _load_dialogs()
    name = name.lower().strip()

    if DIALOG_CACHE is None:
        return None

    for d in DIALOG_CACHE:
        if d.name and name in d.name.lower():
            return d

    return None


async def send_message(receiver: str, text: str) -> Tuple[bool, Optional[str]]:
    """Send a message to a contact or chat."""
    global LAST_CONTACT
    if not await _ensure_available() or _client is None:
        return False, "Telegram is not available."

    dlg = await find_dialog(receiver)
    if not dlg:
        return False, "I couldn't find that contact on Telegram."

    try:
        from telethon.errors import FloodWaitError, RPCError
        await _client.send_message(dlg.id, text)
    except FloodWaitError as e:
        return False, f"Telegram is rate-limiting us. Try again after {e.seconds} seconds."
    except RPCError as e:
        return False, f"Telegram error: {e}"
    except Exception as e:
        return False, f"Unexpected Telegram error: {e}"

    LAST_CONTACT = dlg.name
    return True, None


async def read_latest_message(target: str) -> Optional[str]:
    """Read the latest incoming message from `target`."""
    global LAST_CONTACT
    if not await _ensure_available() or _client is None:
        return None

    dlg = await find_dialog(target)
    if not dlg:
        return None

    try:
        from telethon.errors import FloodWaitError, RPCError
        msgs = await _client.get_messages(dlg.id, limit=5)
    except FloodWaitError as e:
        return f"Telegram is rate-limiting us. Try again after {e.seconds} seconds."
    except RPCError as e:
        return f"Telegram error: {e}"
    except Exception as e:
        return f"Unexpected Telegram error: {e}"

    for m in msgs:
        if not m.out:
            LAST_CONTACT = dlg.name
            return m.text or "(message without text)"

    LAST_CONTACT = dlg.name
    return "No incoming messages."


async def reply_message(text: str) -> Tuple[bool, Optional[str]]:
    """Reply to the LAST_CONTACT."""
    global LAST_CONTACT
    if not await _ensure_available() or _client is None:
        return False, "Telegram is not available."

    if not LAST_CONTACT:
        return False, "There is no recent contact to reply to."

    dlg = await find_dialog(LAST_CONTACT)
    if not dlg:
        return False, "I can't find the last contact anymore."

    try:
        from telethon.errors import FloodWaitError, RPCError
        await _client.send_message(dlg.id, text)
    except FloodWaitError as e:
        return False, f"Telegram is rate-limiting us. Try again after {e.seconds} seconds."
    except RPCError as e:
        return False, f"Telegram error: {e}"
    except Exception as e:
        return False, f"Unexpected Telegram error: {e}"

    return True, None


async def init():
    """
    Initialize Telegram client. Never blocks on stdin.
    If session is missing, sets _available=False.
    """
    await _ensure_available()
    if _available:
        await _load_dialogs(force=True)
