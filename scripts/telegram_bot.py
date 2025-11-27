# scripts/telegram_bot.py

from typing import Optional, Tuple

from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError

API_ID = 35010936
API_HASH = "ebea5ed66cad2c023c000cc7e284ac21"
SESSION = "leo_telegram"

client = TelegramClient(SESSION, API_ID, API_HASH)

# Simple in-memory cache so we don't spam Telegram
DIALOG_CACHE = None   # will become a list of dialogs
LAST_CONTACT: Optional[str] = None


async def _ensure_client():
    """Make sure the client is connected & logged in."""
    if not client.is_connected():
        await client.start()


async def _load_dialogs(force: bool = False):
    """
    Load dialogs once and reuse them.
    This avoids calling get_dialogs() on every send/read.
    """
    global DIALOG_CACHE
    await _ensure_client()

    if DIALOG_CACHE is None or force:
        # limit=200 is usually enough; avoids unnecessary load
        DIALOG_CACHE = await client.get_dialogs(limit=200)


async def find_dialog(name: str):
    """Return dialog whose name contains the given text (case-insensitive)."""
    if not name:
        return None

    await _load_dialogs()
    name = name.lower().strip()

    for d in DIALOG_CACHE:
        if d.name and name in d.name.lower():
            return d

    return None


async def send_message(receiver: str, text: str) -> Tuple[bool, Optional[str]]:
    """
    Send a message to a contact or chat whose name contains `receiver`.
    Returns (ok, error_message_or_None).
    """
    global LAST_CONTACT
    await _ensure_client()

    dlg = await find_dialog(receiver)
    if not dlg:
        return False, "I couldn't find that contact on Telegram."

    try:
        await client.send_message(dlg.id, text)
    except FloodWaitError as e:
        return False, f"Telegram is rate-limiting us. Try again after {e.seconds} seconds."
    except RPCError as e:
        return False, f"Telegram error: {e}"
    except Exception as e:
        return False, f"Unexpected Telegram error: {e}"

    LAST_CONTACT = dlg.name
    return True, None


async def read_latest_message(target: str) -> Optional[str]:
    """
    Read the latest incoming message from `target`.
    Returns the text, or a friendly string, or None if not found.
    """
    global LAST_CONTACT
    await _ensure_client()

    dlg = await find_dialog(target)
    if not dlg:
        return None

    try:
        msgs = await client.get_messages(dlg.id, limit=5)
    except FloodWaitError as e:
        return f"Telegram is rate-limiting us. Try again after {e.seconds} seconds."
    except RPCError as e:
        return f"Telegram error: {e}"
    except Exception as e:
        return f"Unexpected Telegram error: {e}"

    for m in msgs:
        # incoming only
        if not m.out:
            LAST_CONTACT = dlg.name
            return m.text or "(message without text)"

    LAST_CONTACT = dlg.name
    return "No incoming messages."


async def reply_message(text: str) -> Tuple[bool, Optional[str]]:
    """
    Reply to the LAST_CONTACT.
    Returns (ok, error_message_or_None).
    """
    global LAST_CONTACT
    await _ensure_client()

    if not LAST_CONTACT:
        return False, "There is no recent contact to reply to."

    dlg = await find_dialog(LAST_CONTACT)
    if not dlg:
        return False, "I can't find the last contact anymore."

    try:
        await client.send_message(dlg.id, text)
    except FloodWaitError as e:
        return False, f"Telegram is rate-limiting us. Try again after {e.seconds} seconds."
    except RPCError as e:
        return False, f"Telegram error: {e}"
    except Exception as e:
        return False, f"Unexpected Telegram error: {e}"

    return True, None


async def init():
    """
    Call this once in main.py before using send/read/reply.
    """
    await _ensure_client()
    await _load_dialogs(force=True)
