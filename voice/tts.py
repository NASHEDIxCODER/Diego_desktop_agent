"""
Text-to-Speech module for Leo Desktop Assistant.

Uses Coqui TTS (Friday-style voice) with paplay for playback.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Lazy-load TTS
_tts = None


def _get_tts():
    global _tts
    if _tts is None:
        try:
            from TTS.api import TTS
            _tts = TTS(model_name=settings.COQUI_MODEL, progress_bar=False)
            logger.info("TTS model loaded: %s", settings.COQUI_MODEL)
        except Exception as e:
            logger.error("Failed to load TTS: %s", e)
            _tts = False  # Sentinel
    return _tts if _tts is not False else None


def speak(text: str) -> None:
    """
    Speak text using TTS. Falls back to print if TTS fails.
    """
    if not text:
        return

    tts = _get_tts()

    if tts is None:
        logger.info("[SPEAK] %s", text)
        return

    try:
        voice_file = Path(settings.VOICE_FILE)
        tts.tts_to_file(text=text, file_path=str(voice_file))
        subprocess.run(
            ["paplay", str(voice_file)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning("TTS failed: %s", e)
        logger.info("[SPEAK-FALLBACK] %s", text)