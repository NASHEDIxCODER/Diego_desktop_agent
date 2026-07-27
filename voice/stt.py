"""
Speech-to-Text module for Leo Desktop Assistant.

Provides speech recognition using Google Web Speech API
with Whisper as an optional offline fallback.
"""

import logging
from typing import Optional

import speech_recognition as sr

from config.settings import settings

logger = logging.getLogger(__name__)

# Global recognizer and mic
_recognizer: Optional[sr.Recognizer] = None
_mic: Optional[sr.Microphone] = None


def _get_recognizer() -> sr.Recognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = sr.Recognizer()
    return _recognizer


def _get_mic() -> sr.Microphone:
    global _mic
    if _mic is None:
        if settings.WAKE_DEVICE_INDEX is not None:
            _mic = sr.Microphone(device_index=settings.WAKE_DEVICE_INDEX)
        else:
            _mic = sr.Microphone()
    return _mic


def calibrate(duration: float = 1.5) -> None:
    """Calibrate the microphone for ambient noise."""
    r = _get_recognizer()
    mic = _get_mic()
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
    r = _get_recognizer()
    mic = _get_mic()

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