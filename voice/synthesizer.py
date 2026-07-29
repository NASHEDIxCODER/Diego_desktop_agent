"""
SpeechSynthesizer — TTS with multi-backend support for Leo.

Provides text-to-speech with configurable voice parameters:
1. pyttsx3 (primary, espeak-based on Linux)
2. subprocess espeak directly (fallback)
3. Print-only (last resort, warns loudly)

Voice parameters (rate, volume) are applied at synthesis time.

Callbacks:
  - started-utterance: called when audio playback begins
  - finished-utterance: called when audio playback completes
  - error: called when TTS fails with full exception

Timing:
  TTS queued  -> log before engine.say()
  TTS started -> log in started-utterance callback
  TTS finished -> log in finished-utterance callback / after runAndWait()
"""

import logging
import time
from pathlib import Path
from typing import Optional, Callable

from voice.settings import voice_settings

logger = logging.getLogger(__name__)


def _suppress_stderr():
    """Redirect stderr to /dev/null and return a restore function."""
    import os as _os
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    old_stderr = _os.dup(2)
    _os.dup2(devnull_fd, 2)
    _os.close(devnull_fd)
    return lambda: (_os.dup2(old_stderr, 2), _os.close(old_stderr))


class SpeechSynthesizer:
    """
    Text-to-speech engine with automatic fallback.

    The synthesizer:
    1. Tries pyttsx3 (system TTS via espeak)
    2. Falls back to subprocess espeak
    3. Falls back to print-only (never crashes, but logs loudly)
    4. Applies runtime voice settings
    5. Prevents overlapping speech via speaking flag
    6. Waits for speech completion before returning
    7. Emits started-utterance, finished-utterance, error callbacks
    8. Logs timing: TTS queued, TTS started, TTS finished
    """

    def __init__(self):
        self._pyttsx3 = None
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

        This ensures pyttsx3 is loaded before entering the main loop.
        No lazy loading during command handling.
        """
        logger.info("Preloading TTS engine...")
        self._init_pyttsx3()
        self._ready = True
        logger.info("TTS engine preloaded (pyttsx3)")

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

    def _init_pyttsx3(self):
        """Lazy-init pyttsx3 fallback. Suppresses ALSA/JACK stderr spam."""
        if self._pyttsx3 is not None:
            return self._pyttsx3

        # Suppress ALSA/JACK stderr spam during import (C library output)
        restore_stderr = _suppress_stderr()
        try:
            import pyttsx3
            self._pyttsx3 = pyttsx3.init()
        except Exception as e:
            if not self._warning_shown:
                logger.warning("pyttsx3 unavailable: %s", e)
                self._warning_shown = True
            self._pyttsx3 = None
        finally:
            restore_stderr()

        if self._pyttsx3 is None:
            return None

        # Set speaking rate to approximately 150 WPM
        TARGET_WPM = 150
        self._pyttsx3.setProperty('rate', TARGET_WPM)
        self._pyttsx3.setProperty('volume', voice_settings.voice_volume)

        voices = self._pyttsx3.getProperty('voices')
        if voices and voice_settings.voice_id != "default":
            for v in voices:
                if voice_settings.voice_id in v.id:
                    self._pyttsx3.setProperty('voice', v.id)
                    break

        logger.info("pyttsx3 initialized (rate=%d, volume=%.1f)",
                   TARGET_WPM,
                   voice_settings.voice_volume)
        return self._pyttsx3

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
            # 1. Try pyttsx3 (system TTS) — runAndWait blocks until done
            if self._speak_pyttsx3(text):
                return True

            # 2. Try espeak via subprocess directly
            if self._speak_espeak(text):
                return True

            # 3. Fallback: log the text and raise
            self._speak_fallback(text)
            return False
        finally:
            self._speaking = False

    def is_speaking(self) -> bool:
        """Check if speech is currently in progress."""
        return self._speaking

    def _speak_pyttsx3(self, text: str) -> bool:
        """Speak using pyttsx3 (runAndWait blocks until speech completes)."""
        engine = self._init_pyttsx3()
        if engine is None:
            return False

        try:
            queue_time = time.time()
            logger.info("[TTS QUEUED] %.3f | %s", queue_time, text)

            # Fire started-utterance callback before say()
            if self._on_started:
                try:
                    self._on_started(text)
                except Exception as e:
                    logger.warning("started-utterance callback error: %s", e)

            logger.info("[TTS STARTED] speaking: %s", text)
            engine.say(text)
            engine.runAndWait()  # Blocks until speech finishes

            finish_time = time.time()
            elapsed = finish_time - queue_time
            logger.info("[TTS FINISHED] %.3f (%.2fs) | %s", finish_time, elapsed, text)

            # Fire finished-utterance callback
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

    def _speak_espeak(self, text: str) -> bool:
        """Speak using espeak via subprocess directly."""
        import subprocess as _subprocess
        import shutil as _shutil

        espeak_cmd = _shutil.which("espeak") or _shutil.which("espeak-ng")
        if not espeak_cmd:
            return False

        try:
            queue_time = time.time()
            logger.info("[TTS QUEUED] %.3f | %s", queue_time, text)

            if self._on_started:
                try:
                    self._on_started(text)
                except Exception as e:
                    logger.warning("started-utterance callback error: %s", e)

            logger.info("[TTS STARTED] speaking (espeak): %s", text)
            result = _subprocess.run(
                [espeak_cmd, text],
                check=False,
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                timeout=30,
            )
            finish_time = time.time()
            elapsed = finish_time - queue_time
            logger.info("[TTS FINISHED] %.3f (%.2fs) | %s", finish_time, elapsed, text)

            if self._on_finished:
                try:
                    self._on_finished(text)
                except Exception as e:
                    logger.warning("finished-utterance callback error: %s", e)

            if result.returncode != 0:
                logger.warning("espeak returned non-zero: %d", result.returncode)
                return False
            return True
        except _subprocess.TimeoutExpired:
            logger.error("[TTS ERROR] espeak timed out for text='%s'", text)
            if self._on_error:
                try:
                    self._on_error(text, TimeoutError("espeak timed out"))
                except Exception as cb_e:
                    logger.warning("error callback error: %s", cb_e)
            return False
        except Exception as e:
            logger.error("[TTS ERROR] espeak failed for text='%s': %s", text, e, exc_info=True)
            if self._on_error:
                try:
                    self._on_error(text, e)
                except Exception as cb_e:
                    logger.warning("error callback error: %s", cb_e)
            return False

    def _speak_fallback(self, text: str) -> None:
        """Fallback: log the text with loud warning."""
        logger.error(
            "[TTS FATAL] NO TTS BACKEND AVAILABLE — cannot produce audio for: %s",
            text,
        )
        if not self._warning_shown:
            logger.warning(
                "No TTS backend available. Install pyttsx3 or espeak: "
                "pip install pyttsx3 && sudo apt install espeak"
            )
            self._warning_shown = True
        logger.info("[SPEAK-FALLBACK] %s", text)

    def close(self) -> None:
        """Release TTS resources."""
        self._speaking = False
        if self._pyttsx3 is not None:
            try:
                self._pyttsx3.stop()
            except Exception:
                pass
            self._pyttsx3 = None
        logger.debug("TTS resources released")


# Global singleton
speech_synthesizer = SpeechSynthesizer()