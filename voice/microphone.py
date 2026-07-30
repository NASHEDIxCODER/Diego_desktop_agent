"""
MicrophoneManager — Manages microphone lifecycle with auto-recovery.

Features:
- Multiple backends: PyAudio → sounddevice → speech_recognition
- Graceful handling of missing microphone
- Periodic health checks
- Automatic reconnection on failure
- Hotplug detection (microphone inserted/removed)
- Warning shown only once per failure
"""

import logging
import time
from typing import Optional, Callable, Any

from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Maximum consecutive failures before entering recovery mode
MAX_CONSECUTIVE_FAILURES = 3
# Seconds between health checks
HEALTH_CHECK_INTERVAL = 5.0
# Seconds between recovery attempts (exponential backoff)
RECOVERY_BASE_DELAY = 2.0
RECOVERY_MAX_DELAY = 60.0


class SoundDeviceMic:
    """
    A drop-in microphone wrapper using sounddevice.
    
    Used when PyAudio is not available.
    Provides the same context manager interface as speech_recognition.Microphone.
    """

    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._audio_buffer = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self, chunk_size=None):
        """Read audio data (compatible with speech_recognition API)."""
        import sounddevice as sd
        if chunk_size is None:
            chunk_size = self.chunk_size
        duration = chunk_size / self.sample_rate
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate,
                          channels=1, dtype='int16', device=self.device_index)
        sd.wait()
        return recording.tobytes()


class MicrophoneManager:
    """
    Manages microphone lifecycle.

    Provides a microphone instance that is lazily created and
    automatically recovered on failure. The manager never raises
    exceptions to the caller — it returns None if unavailable.
    
    Backend priority:
    1. speech_recognition.Microphone (requires PyAudio)
    2. sounddevice (no PyAudio needed, works with PipeWire/PulseAudio)
    """

    def __init__(self):
        self._mic = None
        self._sr_module = None
        self._backend = None  # 'pyaudio' or 'sounddevice'
        self._consecutive_failures = 0
        self._last_health_check = 0.0
        self._recovery_attempts = 0
        self._warning_shown = False
        self._available = False
        self._on_status_change: Optional[Callable[[bool], None]] = None

    def set_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Set a callback that is called when microphone availability changes."""
        self._on_status_change = callback

    def _import_sr(self):
        """Lazy-import speech_recognition."""
        if self._sr_module is not None:
            return self._sr_module
        try:
            self._sr_module = __import__(
                "speech_recognition",
                fromlist=["Recognizer", "Microphone", "WaitTimeoutError",
                          "UnknownValueError", "RequestError"]
            )
            return self._sr_module
        except ImportError:
            return None

    def _try_sounddevice(self):
        """Try to initialize microphone via sounddevice."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                logger.warning("No input devices found via sounddevice")
                return False

            # Find the best input device:
            # 1. Prefer hardware devices with 'hw:' in name (direct ALSA access)
            # 2. Fall back to 'pipewire' or 'pulse' virtual devices
            # 3. Fall back to 'default' device
            # 4. Fall back to first input device
            default_idx = None
            if voice_settings.device_index is not None:
                default_idx = voice_settings.device_index
                logger.info("Using configured device index: %d", default_idx)
            else:
                # Try hardware devices first (hw: prefix)
                hw_devices = [d for d in input_devices if 'hw:' in str(d.get('name', ''))]
                if hw_devices:
                    default_idx = hw_devices[0]['index']
                    logger.info("Using hardware device: [%d] %s", default_idx, hw_devices[0]['name'])
                else:
                    # Try pipewire or pulse virtual devices
                    virtual_names = ['pipewire', 'pulse']
                    for vname in virtual_names:
                        vdevices = [d for d in input_devices if str(d.get('name', '')).lower() == vname]
                        if vdevices:
                            default_idx = vdevices[0]['index']
                            logger.info("Using virtual device: [%d] %s", default_idx, vdevices[0]['name'])
                            break
                
                if default_idx is None:
                    try:
                        default_idx = sd.default.device[0]
                        logger.info("Using system default device: [%d]", default_idx)
                    except Exception:
                        default_idx = input_devices[0]['index']
                        logger.info("Using first input device: [%d] %s", default_idx, input_devices[0]['name'])

            # Find the actual device info for the selected device
            selected = next((d for d in input_devices if d['index'] == default_idx), input_devices[0])
            sample_rate = int(selected['default_samplerate']) if selected.get('default_samplerate') else 16000
            logger.info("Microphone via sounddevice: [%d] %s (channels=%d, samplerate=%d)",
                       selected['index'], selected['name'], selected['max_input_channels'], sample_rate)

            self._mic = SoundDeviceMic(
                device_index=selected['index'],
                sample_rate=sample_rate,
            )
            self._backend = 'sounddevice'
            self._available = True
            self._consecutive_failures = 0
            self._recovery_attempts = 0
            if self._on_status_change:
                self._on_status_change(True)
            return True
        except ImportError:
            logger.debug("sounddevice not available")
            return False
        except Exception as e:
            logger.warning("sounddevice init failed: %s", e)
            return False

    def get_microphone(self):
        """Get or create a microphone instance."""
        if self._mic is not None:
            return self._mic

        # Try speech_recognition (requires PyAudio)
        sr = self._import_sr()
        if sr is not None:
            try:
                if voice_settings.device_index is not None:
                    self._mic = sr.Microphone(device_index=voice_settings.device_index)
                else:
                    self._mic = sr.Microphone()
                self._backend = 'pyaudio'
                self._available = True
                self._consecutive_failures = 0
                self._recovery_attempts = 0
                logger.info("Microphone initialized via PyAudio (device_index=%s)",
                           voice_settings.device_index)
                if self._on_status_change:
                    self._on_status_change(True)
                return self._mic
            except Exception as e:
                logger.debug("PyAudio microphone failed: %s", e)

        # Fallback: sounddevice (no PyAudio needed)
        if self._try_sounddevice():
            return self._mic

        if not self._warning_shown:
            logger.warning("No microphone backend available (tried PyAudio, sounddevice)")
            self._warning_shown = True
        return None

    def health_check(self) -> bool:
        """Check if the microphone is still available."""
        now = time.time()
        if now - self._last_health_check < HEALTH_CHECK_INTERVAL:
            return self._available
        self._last_health_check = now
        mic = self.get_microphone()
        if mic is None:
            self._available = False
            return False
        try:
            if self._backend == 'sounddevice':
                import sounddevice as sd
                sd.check_input_settings(device=mic.device_index)
            else:
                with mic as source:
                    pass
            self._consecutive_failures = 0
            self._available = True
            return True
        except Exception as e:
            self._consecutive_failures += 1
            logger.debug("Microphone health check failed (%d/%d): %s",
                        self._consecutive_failures, MAX_CONSECUTIVE_FAILURES, e)
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._available = False
                self._mic = None
                if self._on_status_change:
                    self._on_status_change(False)
                logger.warning("Microphone unavailable after %d failures", self._consecutive_failures)
                return False
            return True

    def recover(self) -> bool:
        """Attempt to recover the microphone after failure."""
        self._recovery_attempts += 1
        delay = min(RECOVERY_BASE_DELAY * (2 ** (self._recovery_attempts - 1)), RECOVERY_MAX_DELAY)
        logger.info("Attempting microphone recovery in %.1fs (attempt %d)...", delay, self._recovery_attempts)
        time.sleep(delay)
        self._mic = None
        mic = self.get_microphone()
        if mic is not None:
            self._available = True
            self._consecutive_failures = 0
            self._recovery_attempts = 0
            logger.info("Microphone recovered successfully")
            if self._on_status_change:
                self._on_status_change(True)
            return True
        return False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def backend(self) -> str:
        return self._backend or 'none'

    def close(self) -> None:
        self._mic = None
        self._available = False
        logger.debug("Microphone resources released")


# Global singleton
microphone = MicrophoneManager()