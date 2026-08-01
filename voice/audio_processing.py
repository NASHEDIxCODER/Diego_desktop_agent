"""
Audio Processing Pipeline for Leo Desktop Assistant.

Real-time audio preprocessing to make wake word detection robust in
noisy environments (laptop fan, background chatter, etc.).

Pipeline:
  Microphone
    ↓
  High-pass filter (~80–100 Hz) — removes rumble/DC offset
    ↓
  Noise suppression (spectral gating) — removes stationary noise
    ↓
  Voice Activity Detection (VAD) — identifies speech segments
    ↓
  Wake Word Detection / Speech Recognition

CRITICAL: NO automatic gain control (AGC) is applied.
The microphone signal must NEVER be amplified beyond unity.
Accidental amplification causes clipping and destroys speech quality.

All processing is implemented with scipy for production robustness.
No external ML dependencies required.
"""

import logging
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


class _PeakMonitor:
    """Track peak amplitude per pipeline stage to LOCATE saturation (32767).

    Only logs when a stage actually saturates (peak >= 32767) or when a
    stage reaches a new maximum — so it is safe to call on every frame.
    """

    def __init__(self):
        self.max_peaks: Dict[str, int] = {}
        self.sat_counts: Dict[str, int] = {}

    def log(self, stage: str, arr) -> int:
        """Record the peak of `arr` for `stage`. Returns the peak value."""
        if arr is None:
            return 0
        try:
            a = np.asarray(arr)
            if a.size == 0:
                return 0
            peak = int(np.max(np.abs(a.astype(np.float64))))
        except Exception:
            return 0
        if peak > self.max_peaks.get(stage, 0):
            self.max_peaks[stage] = peak
        if peak >= 32767:
            self.sat_counts[stage] = self.sat_counts.get(stage, 0) + 1
            c = self.sat_counts[stage]
            if c <= 5 or c % 100 == 0:
                logger.warning(
                    "[SATURATION] peak=%d at stage='%s' (count=%d) — clipping here",
                    peak, stage, c)
        return peak

    def report(self) -> Dict[str, dict]:
        return {
            s: {"max_peak": self.max_peaks[s],
                "saturation_events": self.sat_counts.get(s, 0)}
            for s in self.max_peaks
        }


peak_monitor = _PeakMonitor()

# ── Constants ──────────────────────────────────────────────────
SAMPLE_RATE = 16000
HIGH_PASS_CUTOFF = 80.0  # Hz — removes rumble/fan low-frequency noise
HIGH_PASS_ORDER = 4

# Spectral gating parameters
NOISE_FLOOR_SECONDS = 0.5  # Initial noise profile estimation time
NOISE_SMOOTHING = 0.9      # Smoothing factor for noise profile
SPECTRAL_FLOOR_DB = -30.0  # Minimum attenuation in dB
SPECTRAL_FFT_SIZE = 512
SPECTRAL_HOP = 128

# VAD parameters
VAD_ENERGY_THRESHOLD = 0.02    # RMS threshold for speech vs silence
VAD_SNR_THRESHOLD = 3.0        # Min SNR (speech/noise) in dB
VAD_MIN_SPEECH_FRAMES = 3      # Min consecutive frames for speech


class AudioPreprocessor:
    """
    Real-time audio preprocessing chain.

    NO AGC — the signal is never amplified beyond unity.
    Only noise suppression (spectral gating) is applied.

    Usage:
        proc = AudioPreprocessor(sample_rate=16000)
        processed = proc.process(audio_int16)
        is_speech = proc.is_speech(processed)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._init_filters()

        # Noise profile (estimated from initial audio)
        self._noise_profile: Optional[np.ndarray] = None
        self._noise_frames: list = []
        self._noise_total_samples = 0
        self._noise_est_frames = int(NOISE_FLOOR_SECONDS * sample_rate / SPECTRAL_HOP)

        # VAD state
        self._speech_frames = 0

        # Diagnostics
        self.diagnostics: Dict[str, Any] = {
            "highpass_cutoff": HIGH_PASS_CUTOFF,
            "noise_suppression": "spectral_gating",
            "noise_floor_db": None,
            "noise_floor_raw": None,
            "processed_count": 0,
            "last_processing_ms": 0.0,
            "input_rms": 0.0,
            "output_rms": 0.0,
            "gain_applied": 1.0,
        }

    def _init_filters(self) -> None:
        """Initialize the high-pass Butterworth filter."""
        nyquist = self.sample_rate / 2
        normalized_cutoff = HIGH_PASS_CUTOFF / nyquist
        self._sos = scipy_signal.butter(
            HIGH_PASS_ORDER,
            normalized_cutoff,
            btype="highpass",
            output="sos",
        )
        self._zi = scipy_signal.sosfilt_zi(self._sos) * 0

    def reset_noise_profile(self) -> None:
        """Reset the noise profile (call at start of session)."""
        self._noise_profile = None
        self._noise_frames = []
        self._noise_total_samples = 0
        self._speech_frames = 0
        self.diagnostics["noise_floor_db"] = None

    def _compute_spectral_magnitude(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT magnitude spectrum."""
        # Pad to at least FFT size
        if len(audio) < SPECTRAL_FFT_SIZE:
            audio = np.pad(audio, (0, SPECTRAL_FFT_SIZE - len(audio)))
        f, t, Zxx = scipy_signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=SPECTRAL_FFT_SIZE,
            noverlap=SPECTRAL_FFT_SIZE - SPECTRAL_HOP,
        )
        return np.abs(Zxx)

    def _estimate_noise(self, audio: np.ndarray) -> None:
        """Estimate noise profile from initial audio frames."""
        if len(audio) == 0:
            return

        mag = self._compute_spectral_magnitude(audio)
        if mag.size == 0:
            return

        # Use median spectrum of this chunk as a noise sample
        if mag.ndim == 2 and mag.shape[1] >= 1:
            frame_noise = np.median(mag, axis=1)  # shape (n_freq,)
            self._noise_frames.append(frame_noise)
            self._noise_total_samples += len(audio)

            # Build profile once we've accumulated enough audio (0.5s)
            if self._noise_total_samples >= self.sample_rate * NOISE_FLOOR_SECONDS:
                self._noise_profile = np.median(np.array(self._noise_frames), axis=0)
                noise_rms = float(np.mean(self._noise_profile ** 2)) ** 0.5
                self.diagnostics["noise_floor_raw"] = noise_rms
                if noise_rms > 0:
                    self.diagnostics["noise_floor_db"] = 20 * np.log10(noise_rms)
                logger.info("[AUDIO-PROC] Noise profile estimated (chunks=%d, samples=%d, floor=%.1f dB)",
                            len(self._noise_frames), self._noise_total_samples,
                            self.diagnostics["noise_floor_db"] or float('-inf'))

    def _apply_spectral_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral gating noise suppression."""
        if self._noise_profile is None:
            # Still estimating noise — output audio unchanged
            return audio

        if len(audio) == 0:
            return audio

        # Compute STFT
        f, t, Zxx = scipy_signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=SPECTRAL_FFT_SIZE,
            noverlap=SPECTRAL_FFT_SIZE - SPECTRAL_HOP,
        )

        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Ensure noise profile length matches
        n_freq = mag.shape[0]
        noise = self._noise_profile[:n_freq]
        if noise.shape[0] < n_freq:
            noise = np.pad(noise, (0, n_freq - noise.shape[0]))

        # Compute spectral gain (Wiener filter style)
        # gain = max(1 - noise^2 / signal^2, floor)
        eps = 1e-10
        gain = 1.0 - (noise[:, np.newaxis] ** 2) / (mag ** 2 + eps)
        gain = np.clip(gain, 10 ** (SPECTRAL_FLOOR_DB / 20), 1.0)

        # Apply gain with soft smoothing (avoid musical noise)
        # Use a mild temporal smoothing
        gain_smooth = scipy_signal.savgol_filter(
            gain, window_length=3, polyorder=1, axis=1
        ) if gain.shape[1] >= 3 else gain

        # Reconstruct
        filtered = gain_smooth * mag * np.exp(1j * phase)
        _, reconstructed = scipy_signal.istft(
            filtered,
            fs=self.sample_rate,
            nperseg=SPECTRAL_FFT_SIZE,
            noverlap=SPECTRAL_FFT_SIZE - SPECTRAL_HOP,
        )

        # Match length
        if len(reconstructed) < len(audio):
            reconstructed = np.pad(reconstructed, (0, len(audio) - len(reconstructed)))
        elif len(reconstructed) > len(audio):
            reconstructed = reconstructed[:len(audio)]

        return reconstructed

    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        if len(audio) == 0:
            return audio
        filtered, self._zi = scipy_signal.sosfilt(
            self._sos, audio, zi=self._zi
        )
        return filtered

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Run the complete preprocessing chain on audio data.

        NO AGC — the signal is never amplified beyond unity.
        Only high-pass filtering and spectral gating are applied.

        Args:
            audio: int16 PCM samples at self.sample_rate.

        Returns:
            Preprocessed int16 PCM samples.
        """
        t0 = time.time()

        if len(audio) == 0:
            return audio

        # Store input metrics
        self.diagnostics["input_rms"] = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
        peak_monitor.log("preproc_input", audio)

        # Convert to float for processing
        audio_float = audio.astype(np.float64) / 32768.0

        # 1. High-pass filter
        audio_filtered = self._apply_highpass(audio_float)
        peak_monitor.log("preproc_highpass", audio_filtered * 32768.0)

        # 2. Noise suppression (after initial noise estimation period)
        if self._noise_profile is None:
            self._estimate_noise(audio_filtered)
        audio_denoised = self._apply_spectral_gate(audio_filtered)
        peak_monitor.log("noise_suppression", audio_denoised * 32768.0)

        # 3. Convert back to int16 — NO AGC, NO amplification
        #    The spectral gate only attenuates noise, never amplifies speech.
        #    Clipping is impossible because gain is always <= 1.0.
        audio_int16 = np.clip(audio_denoised * 32768.0, -32768, 32767).astype(np.int16)
        peak_monitor.log("preproc_int16_output", audio_int16)

        # Update diagnostics
        self.diagnostics["output_rms"] = float(np.sqrt(np.mean(audio_int16.astype(float) ** 2)))
        self.diagnostics["processed_count"] += 1
        self.diagnostics["last_processing_ms"] = (time.time() - t0) * 1000
        self.diagnostics["gain_applied"] = 1.0  # Always unity — no AGC

        return audio_int16

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Simple VAD based on energy and SNR.

        Args:
            audio: Preprocessed int16 PCM samples.

        Returns:
            True if speech detected, False otherwise.
        """
        if len(audio) == 0:
            return False

        # Compute RMS
        rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))

        # Compute noise floor from noise profile if available
        if self._noise_profile is not None:
            noise_rms = float(np.mean(self._noise_profile ** 2)) ** 0.5
            # SNR in dB
            if noise_rms > 1e-10:
                snr_db = 20 * np.log10((rms + 1e-10) / (noise_rms + 1e-10))
            else:
                snr_db = 40.0
        else:
            snr_db = 40.0  # No noise profile yet — assume clean

        # Normalized RMS threshold (0-1 scale)
        norm_rms = rms / 32768.0

        # Speech detection
        is_speech = norm_rms > VAD_ENERGY_THRESHOLD and snr_db > VAD_SNR_THRESHOLD

        if is_speech:
            self._speech_frames += 1
        else:
            self._speech_frames = 0

        return self._speech_frames >= VAD_MIN_SPEECH_FRAMES

    def get_noise_floor_db(self) -> Optional[float]:
        """Get the current noise floor in dBFS."""
        return self.diagnostics.get("noise_floor_db")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics."""
        m = dict(self.diagnostics)
        m["noise_profile_ready"] = self._noise_profile is not None
        return m


# Global singleton
audio_preprocessor = AudioPreprocessor()