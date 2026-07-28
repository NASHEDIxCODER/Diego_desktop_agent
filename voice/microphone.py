"""
MicrophoneManager — Manages microphone lifecycle with auto-recovery.

Features:
- Graceful handling of missing microphone
- Periodic health checks
- Automatic reconnection on failure
- Hotplug detection (microphone inserted/removed)
- Warning shown only once per failure
"""

import logging
import time
from typing import Optional, Callable

from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Maximum consecutive failures before entering recovery mode
MAX_CONSECUTIVE_FAILURES = 3
# Seconds between health checks
HEALTH_CHECK_INTERVAL = 5.0
# Seconds between recovery attempts (exponential backoff)
RECOVERY_BASE_DELAY = 2.0
RECOVERY_MAX_DELAY = 60.0


class MicrophoneManager:
    """
    Manages microphone lifecycle.

    Provides a microphone instance that is lazily created and
    automatically recovered on failure. The manager never raises
    exceptions to the caller — it returns None if unavailable.
    """

    def __init__(self):
        self._mic = None
        self._sr_module = None
        self._consecutive_failures = 0
        self._last_health_check = 0.0
        self._recovery_attempts = 0
        self._warning_shown = False
        self._available = False
        self._on_status_change: Optional[Callable[[bool], None]] = None

    def set_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Set a callback that is called when microphone availability changes."""
        self._on_status_change = callback

    def _import_sr(self):
        """Lazy-import speech_recognition."""
        if self._sr_module is not None:
            return self._sr_module
        try:
            self._sr_module = __import__(
                "speech_recognition",
                fromlist=["Recognizer", "Microphone", "WaitTimeoutError",
                          "UnknownValueError", "RequestError"]
            )
            return self._sr_module
        except ImportError as e:
            if not self._warning_shown:
                logger.warning("speech_recognition not available: %s", e)
                self._warning_shown = True
            return None

    def get_microphone(self):
        """
        Get or create a microphone instance.

        Returns:
            Microphone instance, or None if unavailable.
        """
        sr = self._import_sr()
        if sr is None:
            return None

        if self._mic is None:
            try:
                if voice_settings.device_index is not None:
                    self._mic = sr.Microphone(device_index=voice_settings.device_index)
                else:
                    self._mic = sr.Microphone()
                self._available = True
                self._consecutive_failures = 0
                self._recovery_attempts = 0
                logger.info("Microphone initialized (device_index=%s)",
                           voice_settings.device_index)
                if self._on_status_change:
                    self._on_status_change(True)
            except Exception as e:
                if not self._warning_shown:
                    logger.warning("Microphone creation failed: %s", e)
                    self._warning_shown = True
                self._available = False
                return None

        return self._mic

    def health_check(self) -> bool:
        """
        Check if the microphone is still available.

        Returns:
            True if microphone is available, False otherwise.
        """
        now = time.time()
        if now - self._last_health_check < HEALTH_CHECK_INTERVAL:
            return self._available

        self._last_health_check = now
        mic = self.get_microphone()
        if mic is None:
            self._available = False
            return False

        # Try to open the microphone briefly to verify it works
        try:
            with mic as source:
                pass  # Just check it opens
            self._consecutive_failures = 0
            self._available = True
            return True
        except Exception as e:
            self._consecutive_failures += 1
            logger.debug("Microphone health check failed (%d/%d): %s",
                        self._consecutive_failures, MAX_CONSECUTIVE_FAILURES, e)

            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._available = False
                self._mic = None  # Force re-creation
                if self._on_status_change:
                    self._on_status_change(False)
                logger.warning("Microphone unavailable after %d failures",
                              self._consecutive_failures)
                return False

            return True  # Still considered available until threshold

    def recover(self) -> bool:
        """
        Attempt to recover the microphone after failure.

        Uses exponential backoff between recovery attempts.

        Returns:
            True if recovery succeeded, False otherwise.
        """
        self._recovery_attempts += 1
        delay = min(
            RECOVERY_BASE_DELAY * (2 ** (self._recovery_attempts - 1)),
            RECOVERY_MAX_DELAY
        )

        logger.info("Attempting microphone recovery in %.1fs (attempt %d)...",
                   delay, self._recovery_attempts)
        time.sleep(delay)

        # Force re-creation
        self._mic = None
        mic = self.get_microphone()
        if mic is not None:
            self._available = True
            self._consecutive_failures = 0
            self._recovery_attempts = 0
            logger.info("Microphone recovered successfully")
            if self._on_status_change:
                self._on_status_change(True)
            return True

        return False

    @property
    def available(self) -> bool:
        """Check if microphone is currently available."""
        return self._available

    def close(self) -> None:
        """Release microphone resources."""
        self._mic = None
        self._available = False
        logger.debug("Microphone resources released")


# Global singleton
microphone = MicrophoneManager()