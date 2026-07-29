"""
SpeechSynthesizer — TTS with pluggable engine support for Leo.

Delegates to TTSManager which provides:
1. Kokoro (lightweight, natural)
2. XTTS v2 (best quality, GPU)
3. Piper (fast, local)
4. pyttsx3 (espeak fallback)

Maintains backward compatibility with existing code.
"""

import logging
import time
from typing import Optional, Callable

from voice.settings import voice_settings
from voice.tts.manager import tts_manager

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """
    Text-to-speech engine with pluggable backends.

    Delegates to TTSManager which handles engine selection,
    initialization, and automatic fallback.
    """

    def __init__(self):
        self._ready = False
        self._warning_shown = False
        self._speaking = False  # Guard against overlapping speech

        # Callbacks
        self._on_started: Optional[Callable] = None
        self._on_finished: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    def initialize(self) -> None:
        """
        Preload TTS engine at startup.

        Delegates to TTSManager which tries engines in priority order.
        """
        logger.info("Preloading TTS engine...")
        result = tts_manager.initialize()
        if result:
            self._ready = True
            logger.info("TTS engine preloaded: %s", tts_manager.active_engine_name)
        else:
            logger.warning("No TTS engine available")
            self._ready = False

    def set_callbacks(
        self,
        on_started: Optional[Callable] = None,
        on_finished: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for speech events."""
        self._on_started = on_started
        self._on_finished = on_finished
        self._on_error = on_error
        tts_manager.set_callbacks(on_started, on_finished, on_error)

    def speak(self, text: str) -> bool:
        """
        Speak the given text. Blocks until speech completes.

        Prevents overlapping speech — if already speaking, waits.

        Args:
            text: Text to speak.

        Returns:
            True if speech was produced (audio played), False if only logged.
        """
        if not text or not text.strip():
            return False

        # Prevent overlapping speech
        if self._speaking:
            logger.debug("Already speaking, waiting...")
            import time as _time
            while self._speaking:
                _time.sleep(0.05)

        self._speaking = True
        try:
            result = tts_manager.speak(text)
            if not result:
                self._speak_fallback(text)
            return result
        finally:
            self._speaking = False

    def is_speaking(self) -> bool:
        """Check if speech is currently in progress."""
        if self._speaking:
            return True
        return tts_manager.is_speaking()

    def _speak_fallback(self, text: str) -> None:
        """Fallback: log the text with loud warning."""
        logger.error(
            "[TTS FATAL] NO TTS BACKEND AVAILABLE — cannot produce audio for: %s",
            text,
        )
        if not self._warning_shown:
            logger.warning(
                "No TTS backend available. Install Kokoro: pip install kokoro"
            )
            self._warning_shown = True
        logger.info("[SPEAK-FALLBACK] %s", text)

    def close(self) -> None:
        """Release TTS resources."""
        self._speaking = False
        tts_manager.close()
        self._ready = False
        logger.debug("TTS resources released")


# Global singleton
speech_synthesizer = SpeechSynthesizer()