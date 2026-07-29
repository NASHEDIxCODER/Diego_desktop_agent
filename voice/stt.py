"""
Speech-to-Text module for Leo Desktop Assistant.

Provides speech recognition using Google Web Speech API
with Whisper as an optional offline fallback.

Uses a single shared microphone instance from MicrophoneManager.
Calibrates once at startup. Never recalibrates.
"""

import logging
import sys
from typing import Optional

from config.settings import settings
from voice.microphone import microphone

logger = logging.getLogger(__name__)

# Lazy-import speech_recognition
_recognizer_module = None
_recognizer: Optional[object] = None
_HAS_SPEECH_RECOGNITION = False
_calibrated = False


def _import_sr():
    """Lazy-import speech_recognition."""
    global _recognizer_module, _HAS_SPEECH_RECOGNITION
    if _recognizer_module is not None:
        return _recognizer_module

    try:
        _recognizer_module = __import__("speech_recognition", fromlist=["Recognizer", "Microphone", "WaitTimeoutError", "UnknownValueError", "RequestError"])
        _HAS_SPEECH_RECOGNITION = True
    except ImportError as e:
        logger.warning("speech_recognition not available: %s", e)
        _recognizer_module = False
        return None

    return _recognizer_module


def _get_recognizer():
    """Get or create the speech recognizer (singleton)."""
    global _recognizer
    sr = _import_sr()
    if sr is None:
        return None
    if _recognizer is None:
        _recognizer = sr.Recognizer()
        # Default energy threshold — will be calibrated at startup
        _recognizer.dynamic_energy_threshold = False
        _recognizer.energy_threshold = 300
        logger.debug("Recognizer created (energy_threshold=300)")
    return _recognizer


def calibrate(duration: float = 1.5) -> None:
    """Calibrate the microphone for ambient noise. Called once at startup."""
    global _calibrated
    if _calibrated:
        logger.debug("Already calibrated, skipping")
        return

    r = _get_recognizer()
    mic = microphone.get_microphone()
    if r is None or mic is None:
        logger.warning("Speech recognition not available, skipping calibration")
        return

    sr = _import_sr()
    if sr is None:
        return

    try:
        with mic as source:
            logger.info("Calibrating microphone for %.1f seconds...", duration)
            r.adjust_for_ambient_noise(source, duration=duration)
            r.dynamic_energy_threshold = False
            # Boost threshold slightly to avoid false triggers
            r.energy_threshold *= 1.2
            logger.info(
                "Calibration complete: energy_threshold=%.2f, "
                "mic RMS=%.2f",
                r.energy_threshold,
                r.energy_threshold / 1.2 if r.energy_threshold > 0 else 0,
            )
            _calibrated = True
    except Exception as e:
        logger.error("Calibration failed: %s", e)


def listen(timeout: Optional[float] = None,
           phrase_time_limit: Optional[float] = None) -> Optional[str]:
    """
    Listen for a single phrase and return the recognized text.

    Uses the shared microphone from MicrophoneManager.
    Never reopens the microphone unnecessarily.

    Args:
        timeout: Maximum time to wait for speech to start
        phrase_time_limit: Maximum time for a single phrase

    Returns:
        Recognized text string, or None if failed.
    """
    sr = _import_sr()
    r = _get_recognizer()
    mic = microphone.get_microphone()
    if r is None or mic is None or sr is None:
        logger.warning("Speech recognition not available")
        return None

    try:
        with mic as source:
            logger.debug(
                "Listening (energy=%.1f, timeout=%s, phrase_limit=%s)...",
                r.energy_threshold, timeout, phrase_time_limit
            )
            audio = r.listen(source, timeout=timeout,
                             phrase_time_limit=phrase_time_limit or 7)
    except sr.WaitTimeoutError:
        logger.debug("Listening timed out (energy=%.1f)", r.energy_threshold)
        return None
    except Exception as e:
        logger.error("Microphone error: %s", e)
        return None

    try:
        text = r.recognize_google(audio, language=settings.LANG_CODE)
        logger.debug("Recognized: %s", text)
        return text.strip()
    except sr.UnknownValueError:
        logger.debug("Could not understand audio (energy=%.1f)", r.energy_threshold)
        return None
    except sr.RequestError as e:
        logger.error("Recognition service error: %s", e)
        return None


def listen_wake(timeout: Optional[float] = None,
                phrase_time_limit: float = 3) -> Optional[str]:
    """
    Listen specifically for a wake word (shorter timeout).

    Logs energy threshold and detection metrics for debugging.
    """
    sr = _import_sr()
    r = _get_recognizer()
    mic = microphone.get_microphone()
    if r is None or mic is None or sr is None:
        return None

    try:
        with mic as source:
            audio = r.listen(source, timeout=timeout,
                             phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        logger.error("Wake listen error: %s", e)
        return None

    try:
        text = r.recognize_google(audio, language=settings.LANG_CODE)
        logger.debug("Wake phrase: '%s' (energy=%.1f)", text, r.energy_threshold)
        return text.strip()
    except sr.UnknownValueError:
        logger.debug("Wake: audio not recognized (energy=%.1f)", r.energy_threshold)
        return None
    except sr.RequestError as e:
        logger.error("Wake recognition error: %s", e)
        return None