"""
TTSManager — Pluggable TTS engine manager.

Orchestrates multiple TTS engines with automatic fallback.
Priority: Kokoro → XTTS v2 → Piper → pyttsx3

Loads the best available engine at startup.
Falls back automatically if the primary engine fails.
"""

import logging
import time
from typing import Optional, Callable

from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Engine priority order (best first)
ENGINE_PRIORITY = [
    "kokoro",
    "xtts",
    "piper",
    "pyttsx3",
]


class TTSManager:
    """
    Pluggable TTS engine manager with automatic fallback.

    Usage:
        tts_manager.initialize()  # Called once at startup
        tts_manager.speak("Hello")  # Blocks until speech completes
        tts_manager.close()  # Release resources
    """

    def __init__(self):
        self._engines = {}
        self._active_engine = None
        self._ready = False
        self._speaking = False
        self._warning_shown = False

        # Callbacks
        self._on_started: Optional[Callable] = None
        self._on_finished: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    def initialize(self) -> bool:
        """
        Initialize the best available TTS engine.

        Tries engines in priority order. Falls back automatically.
        Returns True if at least one engine is available.
        """
        logger.info("Initializing TTS engine manager...")

        # Determine which engines to try based on config
        forced_engine = voice_settings.tts_engine
        if forced_engine and forced_engine != "auto":
            # Try the forced engine first, then fall back
            engines_to_try = [forced_engine] + [e for e in ENGINE_PRIORITY if e != forced_engine]
        else:
            engines_to_try = ENGINE_PRIORITY

        for engine_name in engines_to_try:
            engine = self._create_engine(engine_name)
            if engine is None:
                continue

            logger.info("Trying TTS engine: %s", engine_name)
            if engine.initialize():
                self._engines[engine_name] = engine
                self._active_engine = engine
                self._ready = True
                logger.info("Active TTS engine: %s", engine_name)
                return True
            else:
                logger.info("TTS engine %s unavailable, trying next...", engine_name)

        logger.error("No TTS engine available!")
        self._ready = False
        return False

    def _create_engine(self, name: str):
        """Create a TTS engine instance by name."""
        try:
            if name == "kokoro":
                from voice.tts.kokoro_engine import KokoroEngine
                return KokoroEngine()
            elif name == "xtts":
                from voice.tts.xtts_engine import XTTSEngine
                return XTTSEngine()
            elif name == "piper":
                # Legacy: voice/tts/piper_engine.py does not exist in this
                # tree. Piper remains available through the active streaming
                # path (voice/streaming_tts.py _PiperSynth). Keep this branch
                # inert so a forced tts_engine=piper falls back cleanly.
                logger.warning("TTS engine 'piper' unavailable "
                               "(legacy engine module missing)")
                return None
            elif name == "pyttsx3":
                from voice.tts.pyttsx3_engine import Pyttsx3Engine
                return Pyttsx3Engine()
        except Exception as e:
            logger.warning("Failed to create engine %s: %s", name, e)
        return None

    def speak(self, text: str) -> bool:
        """
        Speak the given text using the active engine.

        Blocks until speech completes.
        Falls back to next engine if active engine fails.

        Args:
            text: Text to speak.

        Returns:
            True if audio was produced, False otherwise.
        """
        if not text or not text.strip():
            return False

        if not self._ready or self._active_engine is None:
            if not self._warning_shown:
                logger.warning("TTS not initialized — cannot speak")
                self._warning_shown = True
            return False

        # Prevent overlapping speech
        if self._speaking:
            logger.debug("TTS already speaking, waiting...")
            import time as _time
            while self._speaking:
                _time.sleep(0.05)

        self._speaking = True
        try:
            # Try active engine first
            result = self._active_engine.speak(text)
            if result:
                return True

            # Active engine failed — try fallback engines
            logger.warning("Active engine %s failed, trying fallback...",
                          self._active_engine.name)

            for engine_name, engine in self._engines.items():
                if engine is self._active_engine:
                    continue
                if not engine.ready:
                    continue
                logger.info("Trying fallback engine: %s", engine_name)
                result = engine.speak(text)
                if result:
                    # Promote this engine to active
                    self._active_engine = engine
                    logger.info("Promoted fallback engine: %s", engine_name)
                    return True

            # All engines failed
            logger.error("All TTS engines failed for: %s", text)
            return False

        except Exception as e:
            logger.error("[TTS ERROR] %s", e, exc_info=True)
            if self._on_error:
                try:
                    self._on_error(text, e)
                except Exception as cb_e:
                    logger.warning("error callback error: %s", cb_e)
            return False
        finally:
            self._speaking = False

    def is_speaking(self) -> bool:
        """Check if speech is currently in progress."""
        if self._speaking:
            return True
        if self._active_engine and self._active_engine.is_speaking():
            return True
        return False

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
        # Propagate to all engines
        for engine in self._engines.values():
            engine.set_callbacks(on_started, on_finished, on_error)

    def close(self) -> None:
        """Release all TTS engine resources."""
        self._speaking = False
        for name, engine in self._engines.items():
            try:
                engine.close()
                logger.debug("Closed TTS engine: %s", name)
            except Exception as e:
                logger.warning("Error closing TTS engine %s: %s", name, e)
        self._engines.clear()
        self._active_engine = None
        self._ready = False
        logger.info("TTS engine manager shut down")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def active_engine_name(self) -> str:
        if self._active_engine:
            return self._active_engine.name
        return "none"


# Global singleton
tts_manager = TTSManager()