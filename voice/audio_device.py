"""
AudioDeviceManager — Auto-detect best audio backend for Leo.

Detects available audio playback backends:
  1. PipeWire (pw-play)
  2. PulseAudio (paplay)
  3. ALSA (aplay)
  4. JACK (jack_play)

Suppresses ALL ALSA/JACK stderr output via subprocess DEVNULL.
Provides a unified play() interface regardless of backend.
"""

import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Supported audio backends with their play commands
BACKENDS = {
    "pipewire": {"play": "pw-play", "args": []},
    "pulseaudio": {"play": "paplay", "args": []},
    "alsa": {"play": "aplay", "args": ["-q"]},
    "jack": {"play": "jack_play", "args": []},
}

# Ordered by preference
BACKEND_PREFERENCE = ["pipewire", "pulseaudio", "alsa", "jack"]


def _suppress_alsa_warnings() -> None:
    """Suppress harmless ALSA warnings that spam stderr."""
    os.environ.setdefault("ALSA_CONFIG_PATH", "")
    os.environ["DISPLAY_ALSA_OUTPUT"] = "0"
    os.environ["ALSA_OUTPUT_FORMAT"] = "0"
    os.environ["PYTTXS3_ALSA_DEBUG"] = "0"
    os.environ["SPEECH_RECOGNITION_ALSA_DEBUG"] = "0"
    os.environ["ALSA_DEBUG"] = "0"
    os.environ["ALSA_DEBUG_FILE"] = "/dev/null"
    os.environ["PULSE_LOG"] = "0"
    os.environ["PULSE_LOG_LEVEL"] = "0"
    os.environ["JACK_NO_AUDIO"] = "1"
    os.environ["JACK_NO_START_SERVER"] = "1"
    # Suppress pyttsx3 ALSA warnings specifically
    os.environ["PYTTXS3_ALSA_DEBUG"] = "0"


class AudioDeviceManager:
    """
    Manages audio playback backend selection and provides unified playback.

    Automatically detects the best available backend at startup.
    Falls back gracefully if backend becomes unavailable.
    Suppresses ALL stderr output from audio backends to prevent terminal flood.
    """

    def __init__(self):
        self._backend: Optional[str] = None
        self._play_cmd: Optional[str] = None
        self._backend_args: list = []
        self._warning_shown = False

        _suppress_alsa_warnings()

    def detect_backend(self) -> str:
        """
        Detect the best available audio playback backend.

        Probes once and caches the result. Never continuously retries
        failed initialization. Suppresses repeated ALSA/JACK warnings.

        Returns:
            Backend name: 'pipewire', 'pulseaudio', 'alsa', 'jack', or 'none'.
        """
        if self._backend is not None:
            return self._backend

        for backend_name in BACKEND_PREFERENCE:
            cmd = BACKENDS[backend_name]["play"]
            if shutil.which(cmd):
                self._backend = backend_name
                self._play_cmd = cmd
                self._backend_args = BACKENDS[backend_name]["args"]
                logger.info("Audio backend: %s", cmd)
                return self._backend

        self._backend = "none"
        if not self._warning_shown:
            logger.warning("No audio backend found (tried: %s)",
                           ", ".join(BACKEND_PREFERENCE))
            self._warning_shown = True
        return "none"

    @property
    def backend(self) -> str:
        """Get the current audio backend."""
        if self._backend is None:
            return self.detect_backend()
        return self._backend

    def play(self, wav_path: str) -> bool:
        """
        Play a WAV file using the detected backend.

        All stdout and stderr is suppressed to prevent terminal spam.

        Args:
            wav_path: Path to WAV file to play.

        Returns:
            True if playback was attempted, False if no backend available.
        """
        backend = self.backend
        if backend == "none":
            if not self._warning_shown:
                logger.warning("No audio backend — cannot play audio")
                self._warning_shown = True
            return False

        cmd = [self._play_cmd] + self._backend_args + [wav_path]
        try:
            subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except FileNotFoundError:
            logger.warning("Audio backend %s disappeared, re-detecting...", backend)
            self._backend = None
            return self.play(wav_path)
        except Exception as e:
            logger.warning("Audio playback failed: %s", e)
            return False

    def is_available(self) -> bool:
        """Check if any audio backend is available."""
        return self.backend != "none"


# Global singleton
audio_device = AudioDeviceManager()