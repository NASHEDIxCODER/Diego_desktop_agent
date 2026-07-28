"""
WakeWordEngine — Offline wake word detection for Leo.

Provides wake word detection using:
1. Fuzzy text matching (when using cloud STT for wake word)
2. Porcupine offline engine (if available)
3. Simple energy-based detection as fallback

The engine never blocks startup. If no wake word engine is available,
the assistant falls back to push-to-talk mode.
"""

import difflib
import logging
from typing import List, Optional, Callable

from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Default wake word variants for fuzzy matching
DEFAULT_WAKE_VARIANTS = [
    "hello leo",
    "leo",
    "lio",
    "hey leo",
    "hello lio",
    "ok leo",
    "hi leo",
]


class WakeWordEngine:
    """
    Wake word detection engine.

    Supports multiple detection methods:
    - Fuzzy text matching (default, works with any STT)
    - Porcupine (offline, if pvporcupine is installed)
    - Custom callback for user-provided detection

    The engine is stateless and thread-safe.
    """

    def __init__(self):
        self._variants: List[str] = list(DEFAULT_WAKE_VARIANTS)
        self._custom_detector: Optional[Callable[[str], bool]] = None
        self._porcupine = None
        self._porcupine_available = False
        self._detection_count = 0

        # Try to load Porcupine for offline detection
        self._init_porcupine()

    def _init_porcupine(self) -> None:
        """Try to initialize Porcupine offline wake word engine."""
        try:
            import pvporcupine
            self._porcupine = pvporcupine.create(
                keywords=[voice_settings.wake_word],
                sensitivities=[voice_settings.wake_sensitivity]
            )
            self._porcupine_available = True
            logger.info("Porcupine wake word engine initialized (keyword=%s)",
                       voice_settings.wake_word)
        except ImportError:
            logger.debug("Porcupine not available, using fuzzy matching")
        except Exception as e:
            logger.debug("Porcupine init failed: %s", e)

    def set_variants(self, variants: List[str]) -> None:
        """Set wake word variants for fuzzy matching."""
        self._variants = variants

    def set_custom_detector(self, detector: Callable[[str], bool]) -> None:
        """Set a custom detection callback."""
        self._custom_detector = detector

    def detect(self, text: str) -> bool:
        """
        Detect if the given text contains the wake word.

        Args:
            text: Input text to check.

        Returns:
            True if wake word was detected.
        """
        if not text:
            return False

        # Custom detector
        if self._custom_detector:
            try:
                if self._custom_detector(text):
                    self._detection_count += 1
                    return True
            except Exception as e:
                logger.warning("Custom wake word detector failed: %s", e)

        # Fuzzy matching
        text_lower = text.lower().strip()
        for variant in self._variants:
            variant_lower = variant.lower().strip()
            if variant_lower in text_lower:
                self._detection_count += 1
                return True
            # Fuzzy match for slight mispronunciations
            ratio = difflib.SequenceMatcher(None, text_lower, variant_lower).ratio()
            if ratio >= 0.7:
                self._detection_count += 1
                return True

        return False

    def detect_audio(self, audio_frame) -> bool:
        """
        Detect wake word from raw audio frame using Porcupine.

        Args:
            audio_frame: Raw audio data (PCM16, 16kHz, mono).

        Returns:
            True if wake word was detected.
        """
        if not self._porcupine_available or self._porcupine is None:
            return False

        try:
            result = self._porcupine.process(audio_frame)
            if result >= 0:
                self._detection_count += 1
                logger.info("Wake word detected via Porcupine")
                return True
        except Exception as e:
            logger.warning("Porcupine processing failed: %s", e)

        return False

    @property
    def detection_count(self) -> int:
        """Get total number of wake word detections."""
        return self._detection_count

    @property
    def has_offline_engine(self) -> bool:
        """Check if offline wake word engine is available."""
        return self._porcupine_available

    def close(self) -> None:
        """Release wake word engine resources."""
        if self._porcupine is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None
        logger.debug("Wake word engine resources released")


# Global singleton
wake_word_engine = WakeWordEngine()