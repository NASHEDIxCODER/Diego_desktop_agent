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
        # Re-entrancy guard: only ONE command recorder may drain the ring
        # buffer at a time. asyncio.wait_for() timeouts do NOT kill executor
        # threads — without this lock, retried listen() calls overlap and
        # "recording" appears to run far beyond timeout/phrase_limit.
        self._record_lock = threading.Lock()
        # Cache for the polyphase resampling FIR so the real-time callback
        # does NOT redesign an 8821-tap filter every 30 ms.
        self._resample_filter_cache: dict = {}
        # Native channel count of the selected device and the count we open
        # the stream with (capped so multi-channel speech detection works
        # without wasting resources on huge virtual buses).
        self._device_max_channels: int = 1
        self._stream_channels: int = CHANNELS

        # ── Audio-capture instrumentation / channel detection ──
        # NEVER assume channel 0 contains the microphone. Auto-detect the
        # channel that carries speech and use it. See _detect_speech_channel.
        self._speech_channel: Optional[int] = None
        self._callback_count: int = 0
        self._dropped_frames: int = 0
        self._last_callback_time: float = 0.0
        self._callback_intervals: deque = deque(maxlen=200)
        self._rms_history: deque = deque(maxlen=200)
        self._peak_history: deque = deque(maxlen=200)
        self._channel_rms: dict = {}      # channel index -> latest RMS (float scale)
        self._last_raw_dump: float = 0.0  # last time a raw WAV was dumped

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

            # ── STEP 4: prefer REAL HARDWARE capture over silent virtual buses ──
            # If the selected device is a virtual bus (pipewire/pulse/default)
            # AND a real hardware input device exists, prefer the hardware
            # device — virtual buses often route to a monitor/silence source.
            try:
                VIRTUAL_KEYS = ("pipewire", "pulse", "default", "monitor")
                HARDWARE_KEYS = ("hw:", "usb", "analog", "alc", "mic", "microphone")
                cur_name = str(sd.query_devices(self._device_index)['name']).lower()
                is_virtual = any(k in cur_name for k in VIRTUAL_KEYS)
                hw_candidates = [
                    d for d in input_devices
                    if any(k in str(d['name']).lower() for k in HARDWARE_KEYS)
                ]
                if is_virtual and hw_candidates:
                    # Reuse the mic selector's measured hardware preference.
                    try:
                        best = select_best_microphone()
                        if best and best.get("index") in {d['index'] for d in hw_candidates}:
                            self._device_index = best.get("index")
                        else:
                            self._device_index = hw_candidates[0]['index']
                    except Exception:
                        self._device_index = hw_candidates[0]['index']
                    logger.info("[AUDIO] Preferring hardware mic over virtual bus: "
                                "[%d] %s", self._device_index,
                                sd.query_devices(self._device_index)['name'])
            except Exception as e:
                logger.debug("[AUDIO] hardware-preference check failed: %s", e)

            device_info = sd.query_devices(self._device_index)
            self._actual_sample_rate = int(device_info.get('default_samplerate', SAMPLE_RATE))
            self._device_max_channels = int(device_info.get('max_input_channels', 1))

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
            # Design the polyphase FIR filter EXACTLY ONCE per (up, down)
            # ratio. scipy's default designs a (2*10*max(up,down)+1)-tap
            # Kaiser filter on EVERY call — at 44100 Hz that is an 8821-tap
            # firwin running inside the 30 ms real-time callback, which
            # blows the callback budget and causes input overflows.
            key = (up, down)
            window = self._resample_filter_cache.get(key)
            if window is None:
                max_rate = max(up, down)
                half_len = 10  # scipy default
                num_taps = 2 * half_len * max_rate + 1
                window = scipy_signal.firwin(
                    num_taps, 1.0 / max_rate, window=("kaiser", 5.0))
                self._resample_filter_cache[key] = window
                logger.info("[AUDIO] Resample filter designed: %d->%d Hz (%d taps, cached)",
                            self._actual_sample_rate, SAMPLE_RATE, num_taps)
            resampled = scipy_signal.resample_poly(
                audio_int16.astype(np.float64), up, down, window=window
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
        # ── Callback instrumentation (STEP 1) ──
        now = time.time()
        if self._last_callback_time > 0:
            self._callback_intervals.append(now - self._last_callback_time)
        self._last_callback_time = now
        self._callback_count += 1
        if status:
            # Overflow/underflow or other stream status → dropped frames.
            self._dropped_frames += 1
            logger.debug("[AUDIO] Stream status (dropped): %s", status)

        # ── Per-channel RMS (STEP 2) — NEVER assume channel 0 is the mic ──
        n_channels = indata.shape[1] if indata.ndim > 1 else 1
        chan_rms = []
        for c in range(n_channels):
            col = indata[:, c] if indata.ndim > 1 else indata
            chan_rms.append(float(np.sqrt(np.mean(col.astype(np.float64) ** 2))))
        for c, r in enumerate(chan_rms):
            self._channel_rms[c] = r

        # Auto-detect the speech channel during the first callbacks if not set.
        if self._speech_channel is None:
            # Pick the channel with the highest RMS; if all silent, default 0.
            best_c = int(np.argmax(chan_rms)) if chan_rms else 0
            # Only lock in a non-zero channel once we actually see signal.
            if chan_rms and chan_rms[best_c] > 1e-4:
                self._speech_channel = best_c
                logger.info("[AUDIO] Speech channel auto-detected: ch%d "
                            "(rms=%.4f of %d channels)",
                            best_c, chan_rms[best_c], n_channels)
            else:
                self._speech_channel = 0  # silent so far — default, re-checked later
        src_channel = self._speech_channel if (
            self._speech_channel is not None and self._speech_channel < n_channels
        ) else 0

        mono = indata[:, src_channel] if indata.ndim > 1 else indata

        # Track RMS/peak history (STEP 8 diagnostics).
        mono_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        mono_peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        self._rms_history.append(mono_rms)
        self._peak_history.append(mono_peak)

        # Periodic per-callback dump (STEP 1) — every 50th callback.
        if self._callback_count % 50 == 0:
            ch_summary = " ".join(
                f"ch{c}={self._channel_rms[c] * 32768:.0f}" for c in sorted(self._channel_rms)
            )
            logger.info(
                "[CALLBACK #%d] shape=%s dtype=%s ch=%d src_ch=%d rms=%.1f peak=%.1f | %s",
                self._callback_count, indata.shape, indata.dtype, n_channels,
                src_channel, mono_rms * 32768, mono_peak * 32768, ch_summary)

        # ── Peak monitoring: locate saturation at every stage ──
        peak_monitor.log("microphone", mono * 32767.0)

        # Convert float32 to int16.
        # sounddevice float32 is nominally in range [-1.0, 1.0], but hot /
        # overdriven sources exceed it. CLIP to [-1, 1] FIRST — otherwise
        # `(x * 32767).astype(int16)` OVERFLOW-WRAPS (e.g. 1.8*32767=58980
        # wraps to a NEGATIVE int16), flipping the waveform sign. Clipping
        # gives clean saturation at ±32767 instead of corrupting the signal.
        audio_clipped = np.clip(mono, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
        peak_monitor.log("int16_conversion", audio_int16)

        # Resample to 16 kHz if device uses a different rate
        if self._actual_sample_rate != SAMPLE_RATE:
            audio_int16 = self._resample_to_16k(audio_int16)
            peak_monitor.log("resampling", audio_int16)

        # ── Raw dump buffer (STEP 3) — capture PRE-highpass int16 audio ──
        # so dump_raw_input() can save audio BEFORE noise suppression / VAD /
        # Whisper / openWakeWord / normalization / gain.
        if not hasattr(self, "_raw_dump_buffer"):
            self._raw_dump_buffer = deque(maxlen=int(6 * SAMPLE_RATE / FRAME_SAMPLES))
        self._raw_dump_buffer.append(audio_int16)

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
            # Open with the device's native channel count (capped at 4) so the
            # callback can AUTO-DETECT which channel carries the microphone
            # (never assume channel 0 — e.g. a stereo mic on the right channel,
            # or a multichannel interface). Downmix happens per-channel in the
            # callback via self._speech_channel.
            self._stream_channels = max(1, min(self._device_max_channels, 4))
            # Blocksize must match the DEVICE sample rate to produce FRAME_DURATION
            # of audio. After resampling to 16 kHz, this yields FRAME_SAMPLES samples.
            device_blocksize = int(self._actual_sample_rate * FRAME_DURATION)
            self._stream = self._sd.InputStream(
                samplerate=self._actual_sample_rate,
                device=self._device_index,
                channels=self._stream_channels,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=device_blocksize,
            )
            self._stream.start()
            self._running = True
            self._initialized = True

            logger.info("[AUDIO] InputStream started — %d Hz, %d/%d ch (native=%d), backend=%s",
                        self._actual_sample_rate, self._stream_channels,
                        self._device_max_channels, self._device_max_channels, self._backend)
            return True

        except Exception as e:
            # If the native channel count fails, retry with mono.
            logger.warning("[AUDIO] Open with %d ch failed (%s) — retrying mono",
                           getattr(self, "_stream_channels", 1), e)
            try:
                self._stream_channels = 1
                device_blocksize = int(self._actual_sample_rate * FRAME_DURATION)
                self._stream = self._sd.InputStream(
                    samplerate=self._actual_sample_rate,
                    device=self._device_index,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback,
                    blocksize=device_blocksize,
                )
                self._stream.start()
                self._running = True
                self._initialized = True
                logger.info("[AUDIO] InputStream started (mono fallback) — %d Hz", self._actual_sample_rate)
                return True
            except Exception as e2:
                logger.error("[AUDIO] Failed to start stream: %s", e2, exc_info=True)
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

    def _access_forbidden(self, caller: str) -> bool:
        """AudioManager must NEVER be accessed after shutdown begins or
        after the stream is stopped. All public read paths gate on this."""
        if shutdown_event.is_set():
            logger.debug("[AUDIO] %s blocked — shutdown in progress", caller)
            return True
        if not self._running:
            logger.debug("[AUDIO] %s blocked — stream not running", caller)
            return True
        return False

    def get_recent_audio(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio from the ring buffer as numpy array.

        Args:
            duration_seconds: How many seconds of audio to retrieve.

        Returns:
            numpy array of int16 samples.
        """
        if self._access_forbidden("get_recent_audio"):
            return np.array([], dtype=np.int16)
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
        if self._access_forbidden("read_since"):
            return np.array([], dtype=np.int16), last_total
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
        if self._access_forbidden("get_recent_processed"):
            return np.array([], dtype=np.int16)
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
        if self._access_forbidden("get_recent_bytes"):
            return b""
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

        # ── Re-entrancy guard ──────────────────────────────────────
        # asyncio.wait_for() does NOT kill executor threads: a timed-out
        # listen() keeps running while the main loop retries, and the retry
        # would start a SECOND recorder draining the same ring buffer.
        # Overlapping recorders are why "recording" appeared to run for
        # 39.6s with timeout=8s / phrase_limit=7s. Refuse overlaps.
        if not self._record_lock.acquire(blocking=False):
            logger.warning("[AUDIO] record_command already active — "
                           "refusing overlapping recording")
            return None
        try:
            return self._record_command_inner(timeout, phrase_limit)
        finally:
            self._record_lock.release()

    def _record_command_inner(self, timeout: float, phrase_limit: float) -> Optional[bytes]:
        """Instrumented command recorder. HARD guarantees:
          - wall-clock never exceeds `timeout` (hard deadline)
          - returned audio never exceeds `phrase_limit`
          - every exit logs its stop reason
        """
        t_record_start = time.time()
        record_deadline = t_record_start + timeout  # HARD wall-clock limit
        max_samples = int(phrase_limit * SAMPLE_RATE)

        command_buffer: list = []
        buffered_samples = 0
        speech_detected = False
        speech_start: Optional[float] = None   # wall time speech started
        speech_end: Optional[float] = None     # wall time speech ended
        silence_start = 0.0
        stop_reason = "timeout"                # refined as we exit

        # Read only NEW audio from this point forward (non-overlapping).
        last_total = self._ring_buffer.total_samples

        logger.info(
            "[AUDIO] record START t=%.3f timeout=%.1fs phrase_limit=%.1fs "
            "deadline=%.3f energy_threshold=%.1f",
            t_record_start, timeout, phrase_limit,
            record_deadline, self._energy_threshold)

        while True:
            if shutdown_event.is_set():
                logger.info("[AUDIO] record STOP reason=shutdown")
                return None

            now = time.time()

            # ── HARD timeout: the loop can NEVER pass the deadline ──
            if now >= record_deadline:
                stop_reason = "timeout"
                logger.info(
                    "[AUDIO] TIMEOUT at +%.2fs (limit=%.1fs, speech_detected=%s)",
                    now - t_record_start, timeout, speech_detected)
                break

            # ── phrase_limit: measured from actual speech start ──
            if speech_start is not None and (now - speech_start) >= phrase_limit:
                speech_end = now
                stop_reason = "phrase_limit"
                logger.info(
                    "[AUDIO] PHRASE_LIMIT at +%.2fs: %.2fs of speech (limit=%.1fs)",
                    now - t_record_start, now - speech_start, phrase_limit)
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
                    speech_start = now
                    logger.info(
                        "[AUDIO] SPEECH START at +%.2fs (RMS=%.1f > threshold=%.1f)",
                        now - t_record_start, rms, self._energy_threshold)
                command_buffer.append(new_audio.copy())
                buffered_samples += len(new_audio)
                silence_start = 0.0
            elif speech_detected:
                # silence: stop when it exceeds the configured threshold
                if silence_start == 0.0:
                    silence_start = now
                elif now - silence_start > VAD_SILENCE_DURATION:
                    speech_end = now
                    stop_reason = "silence"
                    logger.info(
                        "[AUDIO] SPEECH END at +%.2fs (silence=%.2fs >= %.2fs)",
                        now - t_record_start,
                        now - silence_start, VAD_SILENCE_DURATION)
                    break
                command_buffer.append(new_audio.copy())
                buffered_samples += len(new_audio)

            time.sleep(0.01)  # 10ms polling

        t_record_stop = time.time()
        wall_duration = t_record_stop - t_record_start

        logger.info(
            "[AUDIO] record STOP reason=%s wall=%.2fs buffered=%.2fs (%d samples) "
            "speech_start=%s speech_end=%s",
            stop_reason, wall_duration, buffered_samples / SAMPLE_RATE,
            buffered_samples,
            ("+%.2fs" % (speech_start - t_record_start)) if speech_start else "none",
            ("+%.2fs" % (speech_end - t_record_start)) if speech_end else "none")

        if not speech_detected or not command_buffer:
            logger.info("[AUDIO] No speech detected (reason=%s) — returning None",
                        stop_reason)
            return None

        audio = np.concatenate(command_buffer)

        # NEVER return audio longer than phrase_limit.
        if len(audio) > max_samples:
            logger.warning("[AUDIO] Trimming %.2fs -> %.2fs (phrase_limit invariant)",
                           len(audio) / SAMPLE_RATE, phrase_limit)
            audio = audio[:max_samples]

        audio_bytes = audio.tobytes()
        if len(audio_bytes) < 512:
            logger.info("[AUDIO] Command too short (%d bytes) — returning None",
                        len(audio_bytes))
            return None

        duration = len(audio_bytes) / SAMPLE_RATE / 2

        # ── HARD INVARIANT: recorded duration must NEVER exceed phrase_limit ──
        if duration > phrase_limit + 1e-6:
            logger.error(
                "[AUDIO] INVARIANT VIOLATION: duration=%.3fs > phrase_limit=%.3fs "
                "— forcing trim", duration, phrase_limit)
            audio_bytes = audio_bytes[: int(phrase_limit * SAMPLE_RATE) * 2]
            duration = len(audio_bytes) / SAMPLE_RATE / 2

        logger.info(
            "[AUDIO] Command recorded: %d bytes (%.2fs) stop_reason=%s wall=%.2fs",
            len(audio_bytes), duration, stop_reason, wall_duration)
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
        if not self._running or shutdown_event.is_set():
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

        VALIDATION (STEP 6): calibration is INVALID if the microphone is
        producing digital silence (RMS ≈ 0) — that means the selected device
        has no real mic routed. In that case this returns False so the caller
        can switch to another device. Leo must NEVER run on a silent mic.

        Args:
            duration: Calibration duration in seconds.

        Returns:
            True if calibration succeeded AND the mic delivers real signal.
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
        peak = float(np.max(np.abs(audio)))

        # ── Silence validation: RMS ≈ 0 means no real mic signal ──
        # Any real analog microphone (even in a quiet room) produces a small
        # non-zero room-tone/electronic-noise signal. Pure 0.0 is digital
        # silence from an unrouted/virtual device.
        if rms < 1.0:
            logger.error(
                "[AUDIO] CALIBRATION INVALID: RMS=%.2f (≈ digital silence). "
                "Selected device '%s' is NOT delivering a real microphone "
                "signal — pick another input device.",
                rms, self._device_index)
            return False

        self._energy_threshold = max(300.0, rms * 1.5)
        logger.info("[AUDIO] Calibration complete — RMS=%.1f peak=%.0f threshold=%.1f",
                    rms, peak, self._energy_threshold)
        return True

    # ── STEP 5: validate a capture device actually delivers speech ──
    def validate_capture(self, duration: float = 2.0) -> dict:
        """Record `duration` seconds from the CURRENT stream and measure
        speech/energy metrics. Used to verify the selected device works.

        Returns dict with rms, peak, max_channel_rms, speech_channel, valid.
        """
        if not self._running:
            return {"valid": False, "reason": "stream_not_running"}
        time.sleep(0.1)
        audio = self._ring_buffer.get_recent(duration)
        if len(audio) == 0:
            return {"valid": False, "reason": "no_audio"}
        rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
        peak = float(np.max(np.abs(audio)))
        max_ch = max(self._channel_rms.values()) * 32768 if self._channel_rms else 0.0
        valid = rms >= 1.0 or max_ch >= 5.0
        return {
            "valid": bool(valid),
            "rms": rms,
            "peak": peak,
            "max_channel_rms": max_ch,
            "speech_channel": self._speech_channel,
            "device_index": self._device_index,
            "sample_rate": self._actual_sample_rate,
        }

    # ── STEP 3: dump raw (pre-processing) microphone input to a WAV file ──
    def dump_raw_input(self, duration: float = 3.0) -> Optional[str]:
        """Save the most recent `duration` seconds of RAW (pre-highpass,
        pre-noise-suppression, pre-VAD, pre-Whisper, pre-openWakeWord, pre-
        normalization, pre-gain) microphone audio to debug/raw_input_<ts>.wav
        and print duration/rate/channels/RMS/peak. If the waveform is flat,
        the microphone stream is wrong.

        Returns the saved file path, or None on failure.
        """
        if not hasattr(self, "_raw_dump_buffer") or not self._raw_dump_buffer:
            logger.warning("[RAW] No raw audio captured yet")
            return None
        try:
            import wave
            from pathlib import Path as _P
            audio = np.concatenate(list(self._raw_dump_buffer))
            max_n = int(duration * SAMPLE_RATE)
            if len(audio) > max_n:
                audio = audio[-max_n:]
            rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            dbg = _P(__file__).resolve().parent.parent / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = dbg / f"raw_input_{ts}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(audio.astype(np.int16).tobytes())
            dur = len(audio) / SAMPLE_RATE
            logger.info(
                "[RAW] Saved %s: dur=%.2fs rate=%d ch=1 RMS=%.1f peak=%.0f %s",
                path.name, dur, SAMPLE_RATE, rms, peak,
                "FLAT!" if rms < 1.0 else "")
            self._last_raw_dump = time.time()
            return str(path)
        except Exception as e:
            logger.debug("[RAW] dump_raw_input failed: %s", e)
            return None

    # ── STEP 8: write a full runtime audio diagnostics report ──
    def write_audio_report(self, path: Optional[str] = None) -> dict:
        """Write debug/audio_report.json with chosen device, all devices,
        channel map, RMS/peak history, callback timing, dropped frames."""
        import json
        from pathlib import Path as _P
        try:
            devices = []
            if self._sd is not None:
                for d in self._sd.query_devices():
                    if d['max_input_channels'] > 0:
                        devices.append({
                            "index": d['index'], "name": d['name'],
                            "in_channels": d['max_input_channels'],
                            "default_samplerate": d['default_samplerate'],
                        })
            intervals = list(self._callback_intervals)
            report = {
                "chosen_device": self._device_index,
                "speech_channel": self._speech_channel,
                "sample_rate": self._actual_sample_rate,
                "backend": self._backend,
                "running": self._running,
                "all_devices": devices,
                "channel_rms": {str(k): round(v, 6) for k, v in self._channel_rms.items()},
                "rms_history": [round(x, 6) for x in list(self._rms_history)],
                "peak_history": [round(x, 6) for x in list(self._peak_history)],
                "callback_count": self._callback_count,
                "dropped_frames": self._dropped_frames,
                "callback_interval_ms": {
                    "mean": round(float(np.mean(intervals)) * 1000, 2) if intervals else 0.0,
                    "max": round(float(np.max(intervals)) * 1000, 2) if intervals else 0.0,
                },
                "peak_monitor": peak_monitor.report(),
                "energy_threshold": self._energy_threshold,
            }
            if path is None:
                path = str(_P(__file__).resolve().parent.parent / "debug" / "audio_report.json")
            _P(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("[AUDIO] Report written: %s", path)
            return report
        except Exception as e:
            logger.debug("[AUDIO] write_audio_report failed: %s", e)
            return {}

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