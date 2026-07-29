"""
Piper TTS Engine — fast, local neural TTS.

Uses piper-tts (MIT license) which runs well on CPU.
Priority 3 in the TTS pipeline (after Kokoro and XTTS).

Install: pip install piper-tts
Model: voice-en-us-amy-low (auto-downloaded on first use)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from voice.tts.base import BaseTTSEngine
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Cache directory for generated audio
CACHE_DIR = Path(voice_settings.tts_cache_dir) / "piper"


class PiperEngine(BaseTTSEngine):
    """Piper TTS engine — fast, local neural TTS."""

    def __init__(self):
        super().__init__()
        self._tts = None
        self._voice = None
        self._sample_rate = 22050
        self._cache = {}

    def initialize(self) -> bool:
        """Load Piper model. Returns True if successful."""
        logger.info("Initializing Piper TTS engine...")

        try:
            # Suppress stderr during import
            import os as _os
            devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
            old_stderr = _os.dup(2)
            _os.dup2(devnull_fd, 2)
            _os.close(devnull_fd)

            try:
                import piper
                self._tts = piper
            finally:
                _os.dup2(old_stderr, 2)
                _os.close(old_stderr)

            self._ready = True
            logger.info("Piper TTS initialized")
            return True

        except ImportError:
            logger.warning("Piper not installed — skipping")
            return False
        except Exception as e:
            logger.warning("Piper init failed: %s", e)
            return False

    def speak(self, text: str) -> bool:
        """Synthesize and play text using Piper."""
        if not self._ready or self._tts is None:
            return False

        if not text or not text.strip():
            return False

        if self._speaking:
            logger.debug("Piper already speaking, waiting...")
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

            logger.info("[TTS STARTED] piper: %s", text)

            # Synthesize to WAV
            import subprocess as _subprocess
            import tempfile as _tempfile
            import shutil as _shutil

            with _tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wav_path = f.name

            # Use piper-tts CLI
            piper_cmd = _shutil.which("piper") or _shutil.which("piper-tts")
            if piper_cmd:
                result = _subprocess.run(
                    [piper_cmd, "--output-file", wav_path],
                    input=text.encode("utf-8"),
                    check=False,
                    stdout=_subprocess.DEVNULL,
                    stderr=_subprocess.DEVNULL,
                    timeout=30,
                )
                if result.returncode == 0 and Path(wav_path).stat().st_size > 100:
                    # Play the WAV
                    played = self._play_wav(wav_path)
                    finish_time = time.time()
                    elapsed = finish_time - queue_time
                    if played:
                        logger.info("[TTS FINISHED] %.3f (%.2fs) | %s", finish_time, elapsed, text)
                    else:
                        logger.warning("[TTS WARN] Piper synthesized but playback failed (%.2fs)", elapsed)

                    if played and self._on_finished:
                        try:
                            self._on_finished(text)
                        except Exception as e:
                            logger.warning("finished-utterance callback error: %s", e)
                    return played
                else:
                    logger.warning("Piper CLI returned non-zero: %d", result.returncode)
                    return False
            else:
                logger.warning("Piper CLI not found")
                return False

        except Exception as e:
            logger.error("[TTS ERROR] Piper failed for text='%s': %s", text, e, exc_info=True)
            if self._on_error:
                try:
                    self._on_error(text, e)
                except Exception as cb_e:
                    logger.warning("error callback error: %s", cb_e)
            return False
        finally:
            self._speaking = False
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _play_wav(self, wav_path: str) -> bool:
        """Play a WAV file through PipeWire or fallback."""
        import subprocess as _subprocess
        import shutil as _shutil

        # Prefer PipeWire
        pw_play = _shutil.which("pw-play")
        if pw_play:
            result = _subprocess.run(
                [pw_play, wav_path],
                check=False,
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                timeout=60,
            )
            if result.returncode == 0:
                return True

        # Fallback: PulseAudio
        paplay = _shutil.which("paplay")
        if paplay:
            result = _subprocess.run(
                [paplay, wav_path],
                check=False,
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                timeout=60,
            )
            if result.returncode == 0:
                return True

        # Fallback: ALSA
        aplay = _shutil.which("aplay")
        if aplay:
            result = _subprocess.run(
                [aplay, "-q", wav_path],
                check=False,
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.DEVNULL,
                timeout=60,
            )
            if result.returncode == 0:
                return True

        return False

    def is_speaking(self) -> bool:
        return self._speaking

    def close(self) -> None:
        self._speaking = False
        self._tts = None
        self._cache.clear()
        self._ready = False
        logger.debug("Piper TTS resources released")