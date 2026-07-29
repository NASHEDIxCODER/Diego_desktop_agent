"""
Text-to-Speech module for Leo Desktop Assistant.

This module is maintained for backward compatibility.
All calls are delegated to voice.synthesizer.speech_synthesizer.

Uses pyttsx3 (espeak-based) with proper callbacks, timing, and error handling.
"""

import logging
from typing import Optional

from voice.synthesizer import speech_synthesizer

logger = logging.getLogger(__name__)


def _get_tts():
    """
    Ensure TTS is initialized.
    
    Returns:
        True if TTS is available.
    """
    if not speech_synthesizer._ready:
        speech_synthesizer.initialize()
    return speech_synthesizer._ready


def speak(text: str) -> None:
    """
    Speak text using the speech synthesizer.
    
    This delegates to SpeechSynthesizer.speak() which:
    1. Uses pyttsx3 (primary)
    2. Falls back to espeak directly
    3. Falls back to print with loud warning
    4. Blocks until speech completes (runAndWait)
    5. Logs timing (queued, started, finished)
    6. Fires callbacks (started-utterance, finished-utterance, error)
    
    Args:
        text: Text to speak.
    """
    if not text or not text.strip():
        return

    _get_tts()
    result = speech_synthesizer.speak(text)
    if not result:
        logger.warning("TTS speak returned False (no audio produced) for: %s", text)