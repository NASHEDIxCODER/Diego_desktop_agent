"""
AudioManager — Unified audio capture for Leo Desktop Assistant.

ARCHITECTURE:
  One sounddevice.InputStream → Ring Buffer → Wake Detector / VAD / Command Recorder

The microphone opens ONCE at startup and stays open for the entire session.
Wake detection, command capture, and VAD all consume the SAME PCM stream.
Never switches backend mid-session.

Backend priority:
  1. sounddevice (preferred — no PyAudio needed)
  2. PyAudio (fallback via speech_recognition)
  3. Fail gracefully (text-only mode)

PCM format: 16-bit signed integer, 16 kHz, mono
"""

import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple, Callable

import numpy as np

from voice.audio_processing import audio_preprocessor, peak_monitor
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Global shutdown event. Set during graceful shutdown so ALL consumers
# (wake loop, command recorder, STT) stop accessing the AudioManager
# BEFORE the stream is closed. No component may access the AudioManager
# after this is set.
shutdown_event = threading.Event()

# ── Constants ──────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
FRAME_DURATION = 0.03  # 30ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION)  # 480 samples
RING_BUFFER_SECONDS = 10.0
RING_BUFFER_MAX_FRAMES = int(RING_BUFFER_SECONDS / FRAME_DURATION)

# VAD thresholds
VAD_ENERGY_THRESHOLD = 300.0
VAD_SILENCE_DURATION = 0.8  # seconds of silence to end speech
VAD_MIN_SPEECH_DURATION = 0.5  # minimum speech duration to accept


class RingBuffer:
    """
    Thread-safe ring buffer for audio frames.

    Stores audio as a deque of numpy arrays (each FRAME_SAMPLES long).
    Provides methods to read recent audio and get raw bytes.
    """

    def __init__(self, max_frames: int = RING_BUFFER_MAX_FRAMES):
        self._buffer: deque = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._total_samples = 0

    def put(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append(frame)
            self._total_samples += len(frame)

    def get_recent(self, duration_seconds: float) -> np.ndarray:
        """Get the most recent audio up to duration_seconds."""
        num_samples = int(duration_seconds * SAMPLE_RATE)
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.int16)
            # Collect frames from most recent backwards
            frames = []
            collected = 0
            for frame in reversed(self._buffer):
                frames.append(frame)
                collected += len(frame)
                if collected >= num_samples:
                    break
            frames.reverse()
            if not frames:
                return np.array([], dtype=np.int16)
            result = np.concatenate(frames)
            # Trim to requested duration
            if len(result) > num_samples:
                result = result[-num_samples:]
            return result

    def get_bytes(self, duration_seconds: float) -> bytes:
        """Get recent audio as raw bytes."""
        audio = self.get_recent(duration_seconds)
        return audio.tobytes()

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0

    @property
    def available_seconds(self) -> float:
        """How many seconds of audio are available."""
        with self._lock:
            return len(self._buffer) * FRAME_DURATION

    @property
    def total_samples(self) -> int:
        """Monotonic count of samples written since last clear()."""
        with self._lock:
            return self._total_samples

    def get_since(self, last_total: int):
        """Return (samples_written_after_last_total, new_total).

        Provides NON-OVERLAPPING sequential reads for streaming consumers
        (e.g. openWakeWord), so each sample is delivered exactly once.
        Samples older than the buffer capacity are dropped.
        """
        with self._lock:
            current = self._total_samples
            delta = current - last_total
            if delta <= 0:
                return np.array([], dtype=np.int16), current
            capacity = len(self._buffer) * FRAME_SAMPLES
            if delta > capacity:
                delta = capacity  # fell behind; drop old data and resync
            frames = []
            collected = 0
            for frame in reversed(self._buffer):
                frames.append(frame)
                collected += len(frame)
                if collected >= delta:
                    break
            frames.reverse()
            if not frames:
                return np.array([], dtype=np.int16), current
            result = np.concatenate(frames)
            if len(result) > delta:
                result = result[-delta:]
            return result, current


class VADState:
    """Tracks voice activity detection state."""

    SILENCE = "silence"
    SPEECH = "speech"

    def __init__(self):
        self.state = self.SILENCE
        self.speech_start: float = 0.0
        self.last_voice: float = 0.0
        self.speech_buffer: list = []

    def reset(self) -> None:
        self.state = self.SILENCE
        self.speech_start = 0.0
        self.last_voice = 0.0
        self.speech_buffer = []


class AudioManager:
    """
    Unified audio capture manager.

    Opens ONE sounddevice.InputStream at startup.
    All consumers (wake detector, VAD, command recorder) read from
    the same ring buffer. Never instantiates PyAudio or
    speech_recognition.Microphone() repeatedly.

    Usage:
        am = AudioManager()
        am.start()  # Opens stream
        # Wake detection:
        audio = am.get_recent_audio(2.0)  # 2 seconds
        # Command recording:
        audio_bytes = am.record_command(timeout=8.0, phrase_limit=7.0)
        am.stop()
    """

    def __init__(self):
        self._sd = None
        self._stream = None
        self._ring_buffer = RingBuffer()
        self._vad = VADState()
        self._running = False
        self._lock = threading.Lock()
        self._backend: str = "none"
        self._device_index: Optional[int] = None
        self._actual_sample_rate: int = SAMPLE_RATE
        self._energy_threshold: float = VAD_ENERGY_THRESHOLD
        self._initialized = False
        self._hp_zi = None  # State for lightweight high-pass filter in callback

        # Callback for wake word detection
        self._on_speech_detected: Optional[Callable] = None

    def _init_sounddevice(self) -> bool:
        """Initialize sounddevice backend. Returns True on success."""
        try:
            import sounddevice as sd
            self._sd = sd
            self._backend = "sounddevice"

            # Find best input device
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                logger.error("[AUDIO] No input devices found")
                return False

            # Use configured device, saved selection, or auto-select best
            from voice.mic_selector import select_best_microphone, load_saved_selection
            if voice_settings.device_index is not None:
                self._device_index = voice_settings.device_index
                logger.info("[AUDIO] Using configured device index: %d", self._device_index)
            else:
                # Try saved selection first
                saved = load_saved_selection()
                if saved and saved.get("index") is not None:
                    self._device_index = saved.get("index")
                    logger.info("[AUDIO] Using saved microphone selection: [%d] %s",
                                self._device_index, saved.get("name", "?"))
                else:
                    # Auto-select best microphone by measuring all devices
                    try:
                        best = select_best_microphone()
                        if best and best.get("index") is not None:
                            self._device_index = best.get("index")
                            logger.info("[AUDIO] Auto-selected best microphone: [%d] %s (SNR=%.1fdB)",
                                        self._device_index, best.get("name", "?"),
                                        best.get("snr_db", 0))
                        else:
                            # Fallback to default
                            try:
                                self._device_index = sd.default.device[0]
                            except Exception:
                                self._device_index = input_devices[0]['index']
                    except Exception as e:
                        logger.debug("[AUDIO] Auto mic selection failed: %s, using default", e)
                        try:
                            self._device_index = sd.default.device[0]
                        except Exception:
                            self._device_index = input_devices[0]['index']

            # ── Validate the chosen device still exists & has input channels ──
            # A stale saved selection (e.g. a disconnected USB/analog mic) must
            # not be used — fall back to the default input device.
            valid_indexes = {d['index'] for d in input_devices}
            if self._device_index not in valid_indexes:
                logger.warning(
                    "[AUDIO] Selected mic index %s is no longer available "
                    "(device disconnected?) — falling back to default input",
                    self._device_index)
                try:
                    self._device_index = sd.default.device[0]
                except Exception:
                    self._device_index = input_devices[0]['index']
                # Persist the corrected selection so the stale one is dropped.
                try:
                    from voice.mic_selector import save_selection
                    info = sd.query_devices(self._device_index)
                    save_selection({
                        "index": self._device_index,
                        "name": info['name'],
                        "channels": info['max_input_channels'],
                    })
                except Exception:
                    pass

            device_info = sd.query_devices(self._device_index)
            self._actual_sample_rate = int(device_info.get('default_samplerate', SAMPLE_RATE))

            logger.info("[AUDIO] sounddevice initialized — device[%d]: %s (%d Hz, %d ch)",
                        self._device_index, device_info['name'],
                        self._actual_sample_rate, device_info['max_input_channels'])
            return True

        except ImportError:
            logger.error("[AUDIO] sounddevice not available (ImportError)")
            return False
        except Exception as e:
            logger.error("[AUDIO] sounddevice init failed: %s", e, exc_info=True)
            return False

    def _resample_to_16k(self, audio_int16: np.ndarray) -> np.ndarray:
        """
        Resample audio from device sample rate to 16 kHz.

        Uses scipy's resample_poly for high-quality resampling.
        """
        if self._actual_sample_rate == SAMPLE_RATE:
            return audio_int16
        try:
            from scipy import signal as scipy_signal
            import math
            # Resample using rational ratio
            up = SAMPLE_RATE
            down = self._actual_sample_rate
            gcd = math.gcd(up, down)
            up //= gcd
            down //= gcd
            resampled = scipy_signal.resample_poly(
                audio_int16.astype(np.float64), up, down
            )
            return np.clip(resampled, -32768, 32767).astype(np.int16)
        except Exception as e:
            logger.debug("[AUDIO] Resample failed (%s), using raw audio", e)
            return audio_int16

    def _apply_light_highpass(self, audio_int16: np.ndarray) -> np.ndarray:
        """
        Apply a lightweight high-pass filter in the audio callback.

        This MUST be fast — it runs in the sounddevice callback thread.
        The full noise suppression chain (spectral gating) is applied
        later in get_recent_processed() when audio is consumed for
        wake detection / STT.

        NOTE: The audio is already resampled to 16 kHz at this point,
        so the filter reference rate is SAMPLE_RATE.
        """
        try:
            from scipy import signal as scipy_signal
            nyquist = SAMPLE_RATE / 2
            cutoff = 80.0
            if cutoff >= nyquist:
                return audio_int16
            sos = scipy_signal.butter(2, cutoff / nyquist, btype="highpass", output="sos")
            if self._hp_zi is None:
                # Initialize the filter state for streaming
                zi = scipy_signal.sosfilt_zi(sos) * 0
                filtered, self._hp_zi = scipy_signal.sosfilt(
                    sos, audio_int16.astype(np.float64), zi=zi
                )
            else:
                filtered, self._hp_zi = scipy_signal.sosfilt(
                    sos, audio_int16.astype(np.float64), zi=self._hp_zi
                )
            return np.clip(filtered, -32768, 32767).astype(np.int16)
        except Exception as e:
            logger.debug("[AUDIO] Light highpass failed: %s", e)
            return audio_int16

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Callback for sounddevice.InputStream — called for every audio frame."""
        if status:
            logger.debug("[AUDIO] Stream status: %s", status)

        # ── Peak monitoring: locate saturation at every stage ──
        # microphone (raw float32 source, range ~[-1, 1])
        peak_monitor.log("microphone", indata[:, 0] * 32767.0)

        # Convert float32 to int16.
        # sounddevice float32 is nominally in range [-1.0, 1.0], but hot /
        # overdriven sources exceed it. CLIP to [-1, 1] FIRST — otherwise
        # `(x * 32767).astype(int16)` OVERFLOW-WRAPS (e.g. 1.8*32767=58980
        # wraps to a NEGATIVE int16), flipping the waveform sign. Clipping
        # gives clean saturation at ±32767 instead of corrupting the signal.
        audio_clipped = np.clip(indata[:, 0], -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
        peak_monitor.log("int16_conversion", audio_int16)

        # Resample to 16 kHz if device uses a different rate
        if self._actual_sample_rate != SAMPLE_RATE:
            audio_int16 = self._resample_to_16k(audio_int16)
            peak_monitor.log("resampling", audio_int16)

        # Lightweight high-pass only inside the callback (fast).
        # Full noise suppression is done on read (get_recent_processed).
        # NOTE: The high-pass filter is unity-gain — it does NOT amplify.
        audio_filtered = self._apply_light_highpass(audio_int16)
        peak_monitor.log("vad_highpass", audio_filtered)

        # Put HIGH-PASSED audio in ring buffer (lightweight)
        self._ring_buffer.put(audio_filtered)

        # VAD: lightweight energy check on high-passed audio
        rms = float(np.sqrt(np.mean(audio_filtered.astype(float) ** 2)))
        now = time.time()

        if rms > self._energy_threshold:
            if self._vad.state == VADState.SILENCE:
                self._vad.state = VADState.SPEECH
                self._vad.speech_start = now
                self._vad.speech_buffer = []
                logger.debug("[VAD] Speech started (RMS=%.1f > threshold=%.1f)", rms, self._energy_threshold)
            self._vad.last_voice = now
            self._vad.speech_buffer.append(audio_filtered.copy())
        else:
            if self._vad.state == VADState.SPEECH:
                if now - self._vad.last_voice > VAD_SILENCE_DURATION:
                    # Speech ended
                    duration = now - self._vad.speech_start
                    if duration >= VAD_MIN_SPEECH_DURATION:
                        logger.debug("[VAD] Speech ended (duration=%.2fs, buffer=%d frames)",
                                     duration, len(self._vad.speech_buffer))
                        if self._on_speech_detected:
                            try:
                                self._on_speech_detected()
                            except Exception as e:
                                logger.debug("[VAD] Speech callback error: %s", e)
                    self._vad.reset()

    def start(self) -> bool:
        """
        Open the audio stream. Called ONCE at startup.

        Returns:
            True if stream was opened successfully.
        """
        if self._running:
            logger.debug("[AUDIO] Already running")
            return True

        if not self._init_sounddevice():
            logger.error("[AUDIO] Cannot start — no backend available")
            return False

        # Reset audio preprocessor noise profile for this session
        audio_preprocessor.reset_noise_profile()

        try:
            # Blocksize must match the DEVICE sample rate to produce FRAME_DURATION
            # of audio. After resampling to 16 kHz, this yields FRAME_SAMPLES samples.
            device_blocksize = int(self._actual_sample_rate * FRAME_DURATION)
            self._stream = self._sd.InputStream(
                samplerate=self._actual_sample_rate,
                device=self._device_index,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=device_blocksize,
            )
            self._stream.start()
            self._running = True
            self._initialized = True

            logger.info("[AUDIO] InputStream started — %d Hz, %d ch, backend=%s",
                        self._actual_sample_rate, CHANNELS, self._backend)
            return True

        except Exception as e:
            logger.error("[AUDIO] Failed to start stream: %s", e, exc_info=True)
            self._stream = None
            return False

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if not self._running:
            return

        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._running = False
            self._ring_buffer.clear()
            self._vad.reset()
            logger.info("[AUDIO] InputStream stopped")
        except Exception as e:
            logger.warning("[AUDIO] Stop error: %s", e)

    def get_recent_audio(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio from the ring buffer as numpy array.

        Args:
            duration_seconds: How many seconds of audio to retrieve.

        Returns:
            numpy array of int16 samples.
        """
        return self._ring_buffer.get_recent(duration_seconds)

    @property
    def total_samples(self) -> int:
        """Monotonic count of samples written to the ring buffer."""
        return self._ring_buffer.total_samples

    def read_since(self, last_total: int):
        """Return (new_audio_int16, new_total) written after last_total.

        NON-OVERLAPPING: each sample is returned exactly once across calls.
        Used by the continuous wake loop to feed openWakeWord streaming frames.
        """
        return self._ring_buffer.get_since(last_total)

    def get_recent_processed(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio and apply the FULL noise suppression chain.

        This is the production entry point for wake detection and STT.
        It applies:
          - High-pass filtering
          - Spectral gating (noise suppression)
          - NO AGC — the signal is never amplified beyond unity

        Args:
            duration_seconds: How many seconds of audio to retrieve.

        Returns:
            Fully processed int16 numpy array.
        """
        raw = self._ring_buffer.get_recent(duration_seconds)
        if len(raw) == 0:
            return raw
        return audio_preprocessor.process(raw)

    def get_recent_bytes(self, duration_seconds: float) -> bytes:
        """
        Get recent audio as raw PCM16 bytes.

        Args:
            duration_seconds: How many seconds of audio to retrieve.

        Returns:
            Raw bytes (PCM16, mono).
        """
        return self._ring_buffer.get_bytes(duration_seconds)

    def record_command(self, timeout: float = 8.0, phrase_limit: float = 7.0) -> Optional[bytes]:
        """
        Record a command from the shared audio stream (non-overlapping reads).

        Recording STOPS as soon as ANY of these is true:
          - silence exceeds VAD_SILENCE_DURATION, OR
          - phrase_limit expires, OR
          - timeout expires.
        The returned audio is NEVER longer than phrase_limit.

        Args:
            timeout: Max seconds to wait for speech to start.
            phrase_limit: Max seconds of audio to return.

        Returns:
            Raw PCM16 bytes of the command (<= phrase_limit), or None if no speech.
        """
        if not self._running:
            logger.error("[AUDIO] Cannot record — stream not running")
            return None
        if shutdown_event.is_set():
            logger.debug("[AUDIO] Shutdown in progress — record_command aborted")
            return None

        start_time = time.time()
        record_start: Optional[float] = None  # when speech actually started
        command_buffer: list = []
        speech_detected = False
        silence_start = 0.0
        max_samples = int(phrase_limit * SAMPLE_RATE)

        # Read only NEW audio from this point forward (non-overlapping).
        last_total = self._ring_buffer.total_samples

        logger.info("[AUDIO] Recording command (timeout=%.1fs, phrase_limit=%.1fs)",
                    timeout, phrase_limit)

        while time.time() - start_time < timeout:
            # Abort immediately if shutdown was requested.
            if shutdown_event.is_set():
                logger.debug("[AUDIO] Shutdown requested — aborting command recording")
                return None
            now = time.time()

            # phrase_limit expiry (measured from when speech actually started)
            if speech_detected and record_start is not None \
                    and (now - record_start) > phrase_limit:
                logger.debug("[AUDIO] phrase_limit reached (%.1fs)", phrase_limit)
                break

            # Non-overlapping read of new audio
            new_audio, last_total = self._ring_buffer.get_since(last_total)
            if len(new_audio) == 0:
                time.sleep(0.01)
                continue

            rms = float(np.sqrt(np.mean(new_audio.astype(float) ** 2)))

            if rms > self._energy_threshold:
                if not speech_detected:
                    speech_detected = True
                    record_start = now
                    logger.debug("[AUDIO] Command speech started (RMS=%.1f)", rms)
                command_buffer.append(new_audio.copy())
                silence_start = 0.0
            elif speech_detected:
                # silence: stop when it exceeds the configured threshold
                if silence_start == 0.0:
                    silence_start = now
                elif now - silence_start > VAD_SILENCE_DURATION:
                    logger.debug("[AUDIO] Command speech ended (silence=%.2fs)",
                                 now - silence_start)
                    break
                command_buffer.append(new_audio.copy())

            time.sleep(0.01)  # 10ms polling

        if not speech_detected or not command_buffer:
            logger.debug("[AUDIO] No speech detected during command recording")
            return None

        audio = np.concatenate(command_buffer)

        # NEVER return audio longer than phrase_limit.
        if len(audio) > max_samples:
            logger.debug("[AUDIO] Trimming command %.1fs -> %.1fs (phrase_limit)",
                         len(audio) / SAMPLE_RATE, phrase_limit)
            audio = audio[:max_samples]

        audio_bytes = audio.tobytes()
        if len(audio_bytes) < 512:
            logger.debug("[AUDIO] Command too short (%d bytes)", len(audio_bytes))
            return None

        duration = len(audio_bytes) / SAMPLE_RATE / 2
        logger.info("[AUDIO] Command recorded: %d bytes (%.2fs)", len(audio_bytes), duration)
        return audio_bytes

    def capture_duration(self, duration: float) -> Optional[bytes]:
        """
        Capture a fixed duration of audio from the ring buffer.

        Used for wake word detection — grabs the most recent audio.

        Args:
            duration: Seconds of audio to capture.

        Returns:
            Raw PCM16 bytes, or None if buffer is empty.
        """
        if not self._running:
            return None

        audio_bytes = self._ring_buffer.get_bytes(duration)
        if len(audio_bytes) < 256:
            return None
        return audio_bytes

    def set_energy_threshold(self, threshold: float) -> None:
        """Set the VAD energy threshold."""
        self._energy_threshold = threshold
        logger.debug("[AUDIO] Energy threshold set to %.1f", threshold)

    def calibrate(self, duration: float = 1.5) -> bool:
        """
        Calibrate energy threshold from ambient noise.

        Captures 'duration' seconds of audio and sets the threshold
        to 1.5x the measured RMS.

        Args:
            duration: Calibration duration in seconds.

        Returns:
            True if calibration succeeded.
        """
        if not self._running:
            logger.warning("[AUDIO] Cannot calibrate — stream not running")
            return False

        logger.info("[AUDIO] Calibrating ambient noise (%.1fs)...", duration)
        time.sleep(0.1)  # Let buffer fill

        audio = self._ring_buffer.get_recent(duration)
        if len(audio) == 0:
            logger.warning("[AUDIO] No audio for calibration")
            return False

        rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
        self._energy_threshold = max(300.0, rms * 1.5)
        logger.info("[AUDIO] Calibration complete — RMS=%.1f, threshold=%.1f",
                    rms, self._energy_threshold)
        return True

    def set_speech_callback(self, callback: Callable) -> None:
        """Set a callback that is called when VAD detects speech."""
        self._on_speech_detected = callback

    @property
    def is_running(self) -> bool:
        """Check if the audio stream is running."""
        return self._running

    @property
    def backend(self) -> str:
        """Get the current backend name."""
        return self._backend

    @property
    def sample_rate(self) -> int:
        """Get the actual sample rate."""
        return self._actual_sample_rate

    @property
    def device_index(self) -> Optional[int]:
        """Get the device index."""
        return self._device_index

    @property
    def energy_threshold(self) -> float:
        """Get the current energy threshold."""
        return self._energy_threshold

    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        return {
            "backend": self._backend,
            "running": self._running,
            "device_index": self._device_index,
            "sample_rate": self._actual_sample_rate,
            "channels": CHANNELS,
            "energy_threshold": self._energy_threshold,
            "buffer_seconds": self._ring_buffer.available_seconds,
            "vad_state": self._vad.state,
            "preprocessor": audio_preprocessor.get_metrics(),
        }


# Global singleton
audio_manager = AudioManager()