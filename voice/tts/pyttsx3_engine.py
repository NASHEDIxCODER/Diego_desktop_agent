"""
pyttsx3 TTS Engine — final fallback using espeak.

Priority 4 (last resort) in the TTS pipeline.
Never removed — always available as fallback.
"""

import logging
import os
import time
from typing import Optional

from voice.tts.base import BaseTTSEngine
from voice.settings import voice_settings

logger = logging.getLogger(__name__)


def _suppress_stderr():
    """Redirect stderr to /dev/null and return a restore function."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    return lambda: (os.dup2(old_stderr, 2), os.close(old_stderr))


class Pyttsx3Engine(BaseTTSEngine):
    """pyttsx3 TTS engine — espeak-based fallback."""

    def __init__(self):
        super().__init__()
        self._engine = None

    def initialize(self) -> bool:
        """Initialize pyttsx3 engine. Returns True if successful."""
        logger.info("Initializing pyttsx3 TTS engine (fallback)...")

        restore_stderr = _suppress_stderr()
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
        except Exception as e:
            if not self._warning_shown:
                logger.warning("pyttsx3 unavailable: %s", e)
                self._warning_shown = True
            self._engine = None
        finally:
            restore_stderr()

        if self._engine is None:
            return False

        # Set speaking rate from voice_settings
        target_rate = voice_settings.voice_rate
        self._engine.setProperty('rate', target_rate)
        self._engine.setProperty('volume', voice_settings.voice_volume)

        voices = self._engine.getProperty('voices')
        if voices and voice_settings.voice_id != "default":
            for v in voices:
                if voice_settings.voice_id in v.id:
                    self._engine.setProperty('voice', v.id)
                    break

        self._ready = True
        logger.info("pyttsx3 initialized (rate=%d, volume=%.1f)",
                   target_rate, voice_settings.voice_volume)
        return True

    def speak(self, text: str) -> bool:
        """Speak using pyttsx3 (runAndWait blocks until speech completes)."""
        if not self._ready or self._engine is None:
            return False

        if not text or not text.strip():
            return False

        if self._speaking:
            logger.debug("pyttsx3 already speaking, waiting...")
            import time as _time
            while self._speaking:
                _time.sleep(0.05)

        self._speaking = True
        try:
            queue_time = time.time()
            logger.info("[TTS QUEUED] %.3f | %s", queue_time, text)

            if self._on_started:
                try:
                    self._on_started(text)
                except Exception as e:
                    logger.warning("started-utterance callback error: %s", e)

            logger.info("[TTS STARTED] pyttsx3: %s", text)
            self._engine.say(text)
            self._engine.runAndWait()

            finish_time = time.time()
            elapsed = finish_time - queue_time
            logger.info("[TTS FINISHED] %.3f (%.2fs) | %s", finish_time, elapsed, text)

            if self._on_finished:
                try:
                    self._on_finished(text)
                except Exception as e:
                    logger.warning("finished-utterance callback error: %s", e)

            return True
        except Exception as e:
            logger.error("[TTS ERROR] pyttsx3 failed for text='%s': %s", text, e, exc_info=True)
            if self._on_error:
                try:
                    self._on_error(text, e)
                except Exception as cb_e:
                    logger.warning("error callback error: %s", cb_e)
            return False
        finally:
            self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking

    def close(self) -> None:
        self._speaking = False
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = None
        self._ready = False
        logger.debug("pyttsx3 TTS resources released")