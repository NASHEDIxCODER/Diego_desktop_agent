"""
SpeechRecognizer — STT with fallback chain for Leo.

Provides speech-to-text with multiple recognition backends:
1. Google Web Speech API (primary, requires internet)
2. Whisper offline (if available)
3. Vosk offline (if available)
4. Fallback text input

Each backend is isolated. One failure falls through to the next.
"""

import logging
from typing import Optional

from voice.microphone import microphone
from voice.noise import noise_calibrator
from voice.settings import voice_settings

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Speech-to-text engine with automatic fallback.

    The recognizer:
    1. Creates a recognizer with cached noise profile
    2. Listens for speech with configurable timeout
    3. Tries Google STT first, falls through on failure
    4. Never raises — returns None on total failure
    """

    def __init__(self):
        self._sr = None
        self._sr_module = None
        self._whisper_model = None

    def _get_sr(self):
        """Lazy-import and create speech_recognition Recognizer."""
        if self._sr is not None:
            return self._sr

        if self._sr_module is None:
            try:
                self._sr_module = __import__(
                    "speech_recognition",
                    fromlist=["Recognizer", "Microphone", "WaitTimeoutError",
                              "UnknownValueError", "RequestError"]
                )
            except ImportError:
                return None

        self._sr = self._sr_module.Recognizer()
        # Apply cached noise profile
        noise_calibrator.apply(self._sr)
        return self._sr

    def listen(self, timeout: Optional[float] = None,
               phrase_time_limit: Optional[float] = None) -> Optional[str]:
        """
        Listen for speech and return recognized text.

        Args:
            timeout: Max seconds to wait for speech to start.
            phrase_time_limit: Max seconds for a single phrase.

        Returns:
            Recognized text, or None if no speech detected.
        """
        sr = self._get_sr()
        if sr is None:
            return None

        if timeout is None:
            timeout = voice_settings.stt_timeout
        if phrase_time_limit is None:
            phrase_time_limit = voice_settings.stt_phrase_limit

        mic = microphone.get_microphone()
        if mic is None:
            return None

        # Listen
        try:
            with mic as source:
                audio = sr.listen(source, timeout=timeout,
                                  phrase_time_limit=phrase_time_limit)
        except self._sr_module.WaitTimeoutError:
            return None
        except Exception as e:
            logger.debug("Listen error: %s", e)
            return None

        # Recognize with fallback chain
        text = self._recognize(sr, audio)
        return text

    def _recognize(self, recognizer, audio) -> Optional[str]:
        """
        Try multiple recognition backends in order.

        Args:
            recognizer: speech_recognition Recognizer instance.
            audio: Audio data to recognize.

        Returns:
            Recognized text, or None if all backends failed.
        """
        # 1. Google Web Speech API
        text = self._try_google(recognizer, audio)
        if text:
            return text

        # 2. Whisper (offline)
        text = self._try_whisper(recognizer, audio)
        if text:
            return text

        # 3. Vosk (offline)
        text = self._try_vosk(recognizer, audio)
        if text:
            return text

        return None

    def _try_google(self, recognizer, audio) -> Optional[str]:
        """Try Google Web Speech API."""
        try:
            text = recognizer.recognize_google(audio, language=voice_settings.lang_code)
            logger.debug("Google STT: %s", text)
            return text.strip()
        except self._sr_module.UnknownValueError:
            return None
        except self._sr_module.RequestError as e:
            logger.debug("Google STT error: %s", e)
            return None
        except Exception as e:
            logger.debug("Google STT unexpected error: %s", e)
            return None

    def _try_whisper(self, recognizer, audio) -> Optional[str]:
        """Try OpenAI Whisper offline."""
        try:
            text = recognizer.recognize_whisper(
                audio,
                model=voice_settings.whisper_model if hasattr(voice_settings, 'whisper_model') else "base"
            )
            logger.debug("Whisper STT: %s", text)
            return text.strip()
        except Exception as e:
            logger.debug("Whisper STT error: %s", e)
            return None

    def _try_vosk(self, recognizer, audio) -> Optional[str]:
        """Try Vosk offline."""
        try:
            text = recognizer.recognize_vosk(audio)
            logger.debug("Vosk STT: %s", text)
            return text.strip()
        except Exception as e:
            logger.debug("Vosk STT error: %s", e)
            return None

    def close(self) -> None:
        """Release resources."""
        self._sr = None
        self._whisper_model = None


# Global singleton
speech_recognizer = SpeechRecognizer()