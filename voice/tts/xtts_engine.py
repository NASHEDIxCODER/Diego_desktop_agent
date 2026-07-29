"""
XTTS v2 TTS Engine — best quality, GPU-accelerated.

Uses Coqui XTTS v2 (MIT license). Requires GPU for acceptable speed.
Priority 2 in the TTS pipeline.

Install: pip install TTS
Model: XTTS-v2 (auto-downloaded on first use, ~1.8GB)
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from voice.tts.base import BaseTTSEngine
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Cache directory for generated audio
CACHE_DIR = Path(voice_settings.tts_cache_dir) / "xtts"


class XTTSEngine(BaseTTSEngine):
    """XTTS v2 TTS engine — best quality, GPU-accelerated."""

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = "cpu"
        self._sample_rate = 24000
        self._cache = {}

    def initialize(self) -> bool:
        """Load XTTS v2 model. Returns True if successful."""
        logger.info("Initializing XTTS v2 TTS engine...")

        # Determine device
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if voice_settings.tts_device == "cpu":
            self._device = "cpu"
        elif voice_settings.tts_device == "cuda" and torch.cuda.is_available():
            self._device = "cuda"

        try:
            # Suppress stderr during import
            import os as _os
            devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
            old_stderr = _os.dup(2)
            _os.dup2(devnull_fd, 2)
            _os.close(devnull_fd)

            try:
                from TTS.api import TTS
                self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self._device)
            finally:
                _os.dup2(old_stderr, 2)
                _os.close(old_stderr)

            self._ready = True
            logger.info("XTTS v2 TTS initialized (device=%s)", self._device)
            return True

        except ImportError:
            logger.warning("XTTS (TTS package) not installed — skipping")
            return False
        except Exception as e:
            logger.warning("XTTS init failed: %s", e)
            return False

    def speak(self, text: str) -> bool:
        """Synthesize and play text using XTTS v2."""
        if not self._ready or self._model is None:
            return False

        if not text or not text.strip():
            return False

        if self._speaking:
            logger.debug("XTTS already speaking, waiting...")
            import time as _time
            while self._speaking:
                _time.sleep(0.05)

        self._speaking = True
        wav_path = None
        try:
            queue_time = time.time()
            logger.info("[TTS QUEUED] %.3f | %s", queue_time, text)

            if self._on_started:
                try:
                    self._on_started(text)
                except Exception as e:
                    logger.warning("started-utterance callback error: %s", e)

            logger.info("[TTS STARTED] xtts: %s", text)

            # Synthesize to temp WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wav_path = f.name

            # Use speaker embedding from a reference if configured, otherwise default
            speaker_wav = voice_settings.tts_speaker_wav
            if speaker_wav and Path(speaker_wav).exists():
                self._model.tts_to_file(
                    text=text,
                    file_path=wav_path,
                    speaker_wav=speaker_wav,
                    language="en",
                )
            else:
                # Use default speaker embedding
                self._model.tts_to_file(
                    text=text,
                    file_path=wav_path,
                    language="en",
                )

            # Play the WAV
            played = self._play_wav(wav_path)

            finish_time = time.time()
            elapsed = finish_time - queue_time
            if played:
                logger.info("[TTS FINISHED] %.3f (%.2fs) | %s", finish_time, elapsed, text)
            else:
                logger.warning("[TTS WARN] XTTS synthesized but playback failed (%.2fs)", elapsed)

            if played and self._on_finished:
                try:
                    self._on_finished(text)
                except Exception as e:
                    logger.warning("finished-utterance callback error: %s", e)

            return played

        except Exception as e:
            logger.error("[TTS ERROR] XTTS failed for text='%s': %s", text, e, exc_info=True)
            if self._on_error:
                try:
                    self._on_error(text, e)
                except Exception as cb_e:
                    logger.warning("error callback error: %s", cb_e)
            return False
        finally:
            self._speaking = False
            if wav_path:
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _play_wav(self, wav_path: str) -> bool:
        """Play a WAV file through PipeWire or fallback."""
        # Prefer PipeWire
        pw_play = shutil.which("pw-play")
        if pw_play:
            result = subprocess.run(
                [pw_play, wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            if result.returncode == 0:
                return True

        # Fallback: PulseAudio
        paplay = shutil.which("paplay")
        if paplay:
            result = subprocess.run(
                [paplay, wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            if result.returncode == 0:
                return True

        # Fallback: ALSA
        aplay = shutil.which("aplay")
        if aplay:
            result = subprocess.run(
                [aplay, "-q", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            if result.returncode == 0:
                return True

        return False

    def is_speaking(self) -> bool:
        return self._speaking

    def close(self) -> None:
        self._speaking = False
        if self._model is not None:
            try:
                del self._model
            except Exception:
                pass
            self._model = None
        self._cache.clear()
        self._ready = False
        logger.debug("XTTS TTS resources released")