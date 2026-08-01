"""
MicrophoneManager — LEGACY COMPATIBILITY WRAPPER.

DEPRECATED: The AudioManager (voice/audio_manager.py) owns the microphone.
This module exists ONLY for backward compatibility with legacy code that
imports `microphone` and calls `get_microphone()`.

It does NOT open a PyAudio stream or instantiate speech_recognition.Microphone().
All microphone access must go through AudioManager's single sounddevice.InputStream.

If legacy code calls get_microphone(), it returns None (no microphone available),
which causes the legacy code to gracefully fall back to the AudioManager path.
"""

import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class MicrophoneManager:
    """
    Legacy compatibility wrapper.

    The AudioManager owns the microphone. This class does NOT open
    any audio streams. It exists only so that legacy code importing
    `microphone` does not crash.
    """

    def __init__(self):
        self._mic = None
        self._backend = None
        self._available = False
        self._on_status_change: Optional[Callable[[bool], None]] = None

    def set_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Set a callback that is called when microphone availability changes."""
        self._on_status_change = callback

    def get_microphone(self):
        """
        LEGACY: Returns None.

        The AudioManager owns the microphone. This method does NOT
        open a PyAudio stream or instantiate speech_recognition.Microphone().
        Legacy code that calls this will get None and should fall back
        to the AudioManager path.
        """
        logger.debug(
            "[MIC] get_microphone() called — AudioManager owns the microphone. "
            "Returning None (legacy compatibility)."
        )
        return None

    def health_check(self) -> bool:
        """Legacy: microphone is managed by AudioManager."""
        return False

    def recover(self) -> bool:
        """Legacy: no recovery needed — AudioManager owns the microphone."""
        return False

    @property
    def available(self) -> bool:
        """Legacy: microphone is managed by AudioManager."""
        return False

    @property
    def backend(self) -> str:
        """Legacy: microphone is managed by AudioManager."""
        return "audio_manager"

    def close(self) -> None:
        """Legacy: nothing to release."""
        self._mic = None
        self._available = False
        logger.debug("Microphone legacy wrapper closed")


# Global singleton
microphone = MicrophoneManager()