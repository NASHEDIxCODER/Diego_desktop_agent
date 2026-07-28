"""
SpeechSynthesizer — TTS with multi-backend support for Leo.

Provides text-to-speech with configurable voice parameters:
1. Coqui TTS (primary, if available)
2. pyttsx3 (fallback, espeak-based)
3. Print-only (last resort, never crashes)

Voice parameters (rate, volume, pitch) are applied at synthesis time.
"""

import logging
from pathlib import Path
from typing import Optional

from voice.audio_device import audio_device
from voice.settings import voice_settings

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """
    Text-to-speech engine with automatic fallback.

    The synthesizer:
    1. Tries Coqui TTS for neural voice
    2. Falls back to pyttsx3 for system TTS
    3. Falls back to print-only (never crashes)
    4. Applies runtime voice settings
    5. Uses auto-detected audio backend for playback
    6. Prevents overlapping speech
    7. Waits for speech completion before returning
    """

    def __init__(self):
        self._coqui = None
        self._pyttsx3 = None
        self._voice_file = Path(__file__).resolve().parent.parent / "leo.wav"
        self._ready = False
        self._warning_shown = False
        self._speaking = False  # Guard against overlapping speech
        self._lock = None  # asyncio.Lock set during initialize()

    def initialize(self) -> None:
        """
        Preload TTS engine at startup.
        
        This ensures all TTS models are loaded before entering the main loop.
        No lazy loading during command handling.
        """
        logger.info("Preloading TTS engine...")
        self._init_pyttsx3()
        self._init_coqui()
        self._ready = True
        logger.info("TTS engine preloaded")

    def _init_coqui(self):
        """Lazy-init Coqui TTS engine."""
        if self._coqui is not None:
            return self._coqui

        try:
            from TTS.api import TTS

            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            self._coqui = TTS(model_name=model_name, progress_bar=False)

            # Apply voice settings
            if hasattr(self._coqui, 'synthesizer'):
                if hasattr(self._coqui.synthesizer, 'voice_rate'):
                    self._coqui.synthesizer.voice_rate = voice_settings.voice_rate
                if hasattr(self._coqui.synthesizer, 'voice_volume'):
                    self._coqui.synthesizer.voice_volume = voice_settings.voice_volume
                if hasattr(self._coqui.synthesizer, 'voice_pitch'):
                    self._coqui.synthesizer.voice_pitch = voice_settings.voice_pitch

            logger.info("Coqui TTS loaded: %s (rate=%d, volume=%.1f, pitch=%.1f)",
                       model_name,
                       voice_settings.voice_rate,
                       voice_settings.voice_volume,
                       voice_settings.voice_pitch)
            return self._coqui
        except Exception as e:
            if not self._warning_shown:
                logger.warning("Coqui TTS unavailable: %s", e)
                self._warning_shown = True
            return None

    def _init_pyttsx3(self):
        """Lazy-init pyttsx3 fallback."""
        if self._pyttsx3 is not None:
            return self._pyttsx3

        try:
            import pyttsx3
            self._pyttsx3 = pyttsx3.init()

            # Set speaking rate to approximately 150 WPM
            # pyttsx3 default rate is 200. 150 WPM ≈ rate of 150.
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
        except Exception as e:
            if not self._warning_shown:
                logger.warning("pyttsx3 unavailable: %s", e)
                self._warning_shown = True
            return None

    def speak(self, text: str) -> bool:
        """
        Speak the given text. Blocks until speech completes.
        
        Prevents overlapping speech — if already speaking, waits.

        Args:
            text: Text to speak.

        Returns:
            True if speech was attempted, False otherwise.
        """
        if not text or not text.strip():
            return False

        # Prevent overlapping speech
        if self._speaking:
            logger.debug("Already speaking, waiting...")
            import time
            while self._speaking:
                time.sleep(0.05)

        self._speaking = True
        try:
            # 1. Try Coqui TTS (neural, high quality)
            if self._speak_coqui(text):
                return True

            # 2. Try pyttsx3 (system TTS) — runAndWait blocks until done
            if self._speak_pyttsx3(text):
                return True

            # 3. Fallback: log the text
            self._speak_fallback(text)
            return False
        finally:
            self._speaking = False

    def is_speaking(self) -> bool:
        """Check if speech is currently in progress."""
        return self._speaking

    def _speak_coqui(self, text: str) -> bool:
        """Speak using Coqui TTS."""
        tts = self._init_coqui()
        if tts is None:
            return False

        try:
            # Generate WAV file
            tts.tts_to_file(text=text, file_path=str(self._voice_file))

            # Play using detected audio backend (blocking)
            audio_device.play(str(self._voice_file))
            return True
        except Exception as e:
            logger.debug("Coqui TTS failed: %s", e)
            return False

    def _speak_pyttsx3(self, text: str) -> bool:
        """Speak using pyttsx3 (runAndWait blocks until speech completes)."""
        engine = self._init_pyttsx3()
        if engine is None:
            return False

        try:
            engine.say(text)
            engine.runAndWait()  # Blocks until speech finishes
            return True
        except Exception as e:
            logger.debug("pyttsx3 failed: %s", e)
            return False

    def _speak_fallback(self, text: str) -> None:
        """Fallback: just log the text."""
        if not self._warning_shown:
            logger.warning("No TTS backend available — speaking to console")
            self._warning_shown = True
        logger.info("[SPEAK] %s", text)

    def close(self) -> None:
        """Release TTS resources."""
        self._speaking = False
        if self._coqui is not None:
            try:
                del self._coqui
            except Exception:
                pass
            self._coqui = None
        if self._pyttsx3 is not None:
            try:
                self._pyttsx3.stop()
            except Exception:
                pass
            self._pyttsx3 = None
        logger.debug("TTS resources released")


# Global singleton
speech_synthesizer = SpeechSynthesizer()