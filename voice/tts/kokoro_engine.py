"""
Kokoro TTS Engine — lightweight, fast, natural-sounding.

Uses Kokoro-82M (MIT license) which runs well on CPU.
Priority 1 in the TTS pipeline.

Install: pip install kokoro
Model: kokoro-82M (auto-downloaded on first use)
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
CACHE_DIR = Path(voice_settings.tts_cache_dir) / "kokoro"


class KokoroEngine(BaseTTSEngine):
    """Kokoro TTS engine — lightweight, fast, natural."""

    def __init__(self):
        super().__init__()
        self._model = None
        self._pipe = None
        self._device = "cpu"
        self._sample_rate = 24000
        self._cache = {}

    def initialize(self) -> bool:
        """Load Kokoro model. Returns True if successful."""
        logger.info("Initializing Kokoro TTS engine...")

        # Determine device
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if voice_settings.tts_device == "cpu":
            self._device = "cpu"
        elif voice_settings.tts_device == "cuda" and torch.cuda.is_available():
            self._device = "cuda"

        try:
            # Suppress stderr during import (avoid ALSA noise)
            import os as _os
            devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
            old_stderr = _os.dup(2)
            _os.dup2(devnull_fd, 2)
            _os.close(devnull_fd)

            try:
                from kokoro import KPipeline
                self._model = KPipeline(lang_code='a', device=self._device)
            finally:
                _os.dup2(old_stderr, 2)
                _os.close(old_stderr)

            self._ready = True
            logger.info("Kokoro TTS initialized (device=%s)", self._device)
            return True

        except ImportError:
            logger.warning("Kokoro not installed — skipping")
            return False
        except Exception as e:
            logger.warning("Kokoro init failed: %s", e)
            return False

    def speak(self, text: str) -> bool:
        """Synthesize and play text using Kokoro."""
        if not self._ready or self._model is None:
            return False

        if not text or not text.strip():
            return False

        # Prevent overlapping speech
        if self._speaking:
            logger.debug("Kokoro already speaking, waiting...")
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

            logger.info("[TTS STARTED] kokoro: %s", text)

            # Synthesize
            audio_data = self._synthesize(text)

            if audio_data is None:
                logger.error("[TTS ERROR] Kokoro synthesis failed for: %s", text)
                return False

            # Play audio through PipeWire or fallback
            played = self._play_audio(audio_data)

            finish_time = time.time()
            elapsed = finish_time - queue_time

            if played:
                logger.info("[TTS FINISHED] %.3f (%.2fs) | %s", finish_time, elapsed, text)
            else:
                logger.warning("[TTS WARN] Kokoro synthesized but playback failed (%.2fs)", elapsed)

            if played and self._on_finished:
                try:
                    self._on_finished(text)
                except Exception as e:
                    logger.warning("finished-utterance callback error: %s", e)

            return played

        except Exception as e:
            logger.error("[TTS ERROR] Kokoro failed for text='%s': %s", text, e, exc_info=True)
            if self._on_error:
                try:
                    self._on_error(text, e)
                except Exception as cb_e:
                    logger.warning("error callback error: %s", cb_e)
            return False
        finally:
            self._speaking = False

    def _synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to audio bytes."""
        # Check cache first
        if voice_settings.tts_cache and text in self._cache:
            logger.debug("Kokoro cache hit for: %s", text[:50])
            return self._cache[text]

        try:
            import torch
            # Generate audio using Kokoro pipeline
            generator = self._model(text, voice='af_heart', speed=1.0)
            audio_chunks = []
            for result in generator:
                audio_chunks.append(result.audio)

            if not audio_chunks:
                return None

            # Concatenate all audio chunks
            audio_tensor = torch.cat(audio_chunks, dim=-1)
            audio_np = audio_tensor.cpu().numpy()

            # Convert float32 [-1, 1] to int16 PCM
            audio_int16 = (audio_np * 32767).astype('int16')
            audio_bytes = audio_int16.tobytes()

            # Cache if enabled
            if voice_settings.tts_cache and len(audio_bytes) < 5_000_000:
                self._cache[text] = audio_bytes
                logger.debug("Kokoro cached: %s (%d bytes)", text[:50], len(audio_bytes))

            return audio_bytes

        except Exception as e:
            logger.error("Kokoro synthesis error: %s", e)
            return None

    def _play_audio(self, audio_data: bytes) -> bool:
        """Play raw PCM audio data through PipeWire or fallback."""
        import subprocess as _subprocess
        import shutil as _shutil
        import struct as _struct
        import wave as _wave
        import tempfile as _tempfile

        # Write to temp WAV file
        try:
            with _tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wav_path = f.name
                with _wave.open(f, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self._sample_rate)
                    wf.writeframes(audio_data)
        except Exception as e:
            logger.warning("Failed to write temp WAV: %s", e)
            return False

        try:
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

            logger.warning("No audio playback backend available")
            return False
        except Exception as e:
            logger.warning("Audio playback error: %s", e)
            return False
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

    def is_speaking(self) -> bool:
        return self._speaking

    def close(self) -> None:
        self._speaking = False
        if self._model is not None:
            self._model = None
        self._cache.clear()
        self._ready = False
        logger.debug("Kokoro TTS resources released")