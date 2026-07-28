"""
Speech-to-Text module for Leo Desktop Assistant.

Provides speech recognition using Google Web Speech API
with Whisper as an optional offline fallback.

Handles Python 3.14 compatibility (aifc module removed).
Compatibility stubs are injected by the compat module (imported in main.py).
"""

import logging
import sys
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Python 3.14 compatibility stubs are injected by compat module in main.py
# Do NOT import compat here — it's already injected at application startup

# Lazy-import speech_recognition
_recognizer_module = None
_recognizer: Optional[object] = None
_mic: Optional[object] = None
_HAS_SPEECH_RECOGNITION = False

# Alternative audio backends
_HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    pass


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
    """Get or create the speech recognizer."""
    global _recognizer
    sr = _import_sr()
    if sr is None:
        return None
    if _recognizer is None:
        _recognizer = sr.Recognizer()
    return _recognizer


def _get_mic():
    """Get or create the microphone."""
    global _mic
    sr = _import_sr()
    if sr is None:
        return None
    if _mic is None:
        try:
            if settings.WAKE_DEVICE_INDEX is not None:
                _mic = sr.Microphone(device_index=settings.WAKE_DEVICE_INDEX)
            else:
                _mic = sr.Microphone()
        except Exception as e:
            logger.warning("Microphone creation failed: %s", e)
            return None
    return _mic


def calibrate(duration: float = 1.5) -> None:
    """Calibrate the microphone for ambient noise."""
    r = _get_recognizer()
    mic = _get_mic()
    if r is None or mic is None:
        logger.warning("Speech recognition not available, skipping calibration")
        return
    try:
        with mic as source:
            logger.info("Calibrating microphone for %.1f seconds...", duration)
            r.adjust_for_ambient_noise(source, duration=duration)
            r.dynamic_energy_threshold = False
            r.energy_threshold *= 1.2
            logger.info("Energy threshold set to %.2f", r.energy_threshold)
    except Exception as e:
        logger.error("Calibration failed: %s", e)


def listen(timeout: Optional[float] = None,
           phrase_time_limit: Optional[float] = None) -> Optional[str]:
    """
    Listen for a single phrase and return the recognized text.

    Args:
        timeout: Maximum time to wait for speech to start
        phrase_time_limit: Maximum time for a single phrase

    Returns:
        Recognized text string, or None if failed.
    """
    sr = _import_sr()
    r = _get_recognizer()
    mic = _get_mic()
    if r is None or mic is None or sr is None:
        logger.warning("Speech recognition not available")
        return None

    try:
        with mic as source:
            logger.debug("Listening...")
            audio = r.listen(source, timeout=timeout,
                             phrase_time_limit=phrase_time_limit or 7)
    except sr.WaitTimeoutError:
        logger.debug("Listening timed out")
        return None
    except Exception as e:
        logger.error("Microphone error: %s", e)
        return None

    try:
        text = r.recognize_google(audio, language=settings.LANG_CODE)
        logger.debug("Recognized: %s", text)
        return text.strip()
    except sr.UnknownValueError:
        logger.debug("Could not understand audio")
        return None
    except sr.RequestError as e:
        logger.error("Recognition service error: %s", e)
        return None


def listen_wake(timeout: Optional[float] = None,
                phrase_time_limit: float = 3) -> Optional[str]:
    """
    Listen specifically for a wake word (shorter timeout).
    """
    return listen(timeout=timeout, phrase_time_limit=phrase_time_limit)