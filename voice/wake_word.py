"""
WakeWordEngine — Offline wake word detection for Leo.

Provides wake word detection using:
1. openWakeWord (offline, primary) — managed by WakeModelManager
2. Fuzzy text matching (fallback for text-based verification)
3. Porcupine offline engine (if available)

The engine never blocks startup. If no wake word engine is available,
the assistant falls back to push-to-talk mode.

Model selection (see WakeModelManager):
  WAKE_MODEL   → custom ONNX model → models/wake/*.onnx → bundled model
  WAKE_PHRASE  → phrase the model is expected to detect ("hello leo")

The bundled hey_jarvis model is NEVER hardcoded.

Google SpeechRecognition is NOT used for wake detection.
"""

import difflib
import logging
from typing import List, Optional, Callable

import numpy as np

from voice.settings import voice_settings
from voice.wake_model_manager import wake_model_manager

logger = logging.getLogger(__name__)

# Default wake word variants for fuzzy matching.
# Derived from the configured WAKE_PHRASE (default "hello leo") so the
# loaded model always matches the phrase the assistant waits for.
# Includes common mispronunciations (lio) and short forms (leo).
DEFAULT_WAKE_VARIANTS = [
    "hello leo",
    "hey leo",
    "ok leo",
    "hi leo",
    "leo",
    "lio",
    "hello lio",
]


class WakeWordEngine:
    """
    Wake word detection engine.

    Supports multiple detection methods:
    - openWakeWord (offline, primary, via WakeModelManager)
    - Fuzzy text matching (fallback)
    - Porcupine (offline, if pvporcupine is installed)
    - Custom callback for user-provided detection

    The engine is stateless and thread-safe.
    """

    def __init__(self):
        self._variants: List[str] = list(DEFAULT_WAKE_VARIANTS)
        self._custom_detector: Optional[Callable[[str], bool]] = None
        self._porcupine = None
        self._porcupine_available = False
        self._openwakeword_available = False
        self._detection_count = 0

        # Try to load Porcupine for offline detection
        self._init_porcupine()
        # openWakeWord is loaded lazily via WakeModelManager (never hardcoded).
        self._init_openwakeword()

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
            logger.debug("Porcupine not available, using openWakeWord")
        except Exception as e:
            logger.debug("Porcupine init failed: %s", e)

    def _init_openwakeword(self) -> None:
        """
        Initialize openWakeWord via WakeModelManager.

        The manager resolves the model from WAKE_MODEL → models/wake/*.onnx
        → bundled model matching WAKE_PHRASE. hey_jarvis is never hardcoded.
        """
        ok = wake_model_manager.load()
        if ok:
            self._openwakeword_available = True
        else:
            self._openwakeword_available = False
            logger.warning("openWakeWord init failed: %s",
                          wake_model_manager.load_error or "no model")

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
            # Exact substring match
            if variant_lower in text_lower:
                self._detection_count += 1
                return True
            # Fuzzy match for slight mispronunciations
            ratio = difflib.SequenceMatcher(None, text_lower, variant_lower).ratio()
            if ratio >= 0.75 and len(text_lower) >= len(variant_lower) * 0.5:
                self._detection_count += 1
                return True

        return False

    def detect_audio(self, audio_frame) -> bool:
        """
        Detect wake word from raw audio frame using openWakeWord or Porcupine.

        Args:
            audio_frame: Raw audio data (PCM16, 16kHz, mono).

        Returns:
            True if wake word was detected.
        """
        # Try openWakeWord first (primary) — via WakeModelManager
        if self._openwakeword_available:
            try:
                # Convert int16 to float32 in range [-1, 1] (manager handles both)
                if isinstance(audio_frame, np.ndarray):
                    audio_int16 = audio_frame
                else:
                    audio_int16 = np.frombuffer(audio_frame, dtype=np.int16)
                detected = wake_model_manager.detect(audio_int16)
                if detected:
                    self._detection_count += 1
                    model_name, score = wake_model_manager.highest_score()
                    logger.info("Wake word detected via openWakeWord ('%s', score=%.3f)",
                               model_name, score)
                    return True
            except Exception as e:
                logger.warning("openWakeWord processing failed: %s", e)

        # Try Porcupine as fallback
        if self._porcupine_available and self._porcupine is not None:
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
        return self._openwakeword_available or self._porcupine_available

    def close(self) -> None:
        """Release wake word engine resources."""
        if self._porcupine is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None
        self._openwakeword_available = False
        wake_model_manager.close()
        logger.debug("Wake word engine resources released")


# Global singleton
wake_word_engine = WakeWordEngine()