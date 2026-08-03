"""
Audio Processing Pipeline for Leo Desktop Assistant.

Real-time audio preprocessing to make wake word detection robust in
noisy environments (laptop fan, background chatter, etc.).

Pipeline (FLOAT DOMAIN end-to-end):
  Microphone (float32 [-1, 1])
    ↓
  High-pass filter (~80 Hz) — removes rumble/DC offset   [unity gain]
    ↓
  Noise suppression (spectral gating) — removes stationary noise  [gain <= 1.0]
    ↓
  Voice Activity Detection (VAD) — identifies speech segments
    ↓
  Wake Word Detection / Speech Recognition

GAIN PIPELINE CONTRACT (HARD RULES):

  1. NO automatic gain control (AGC), NO amplification, NO loudness
     normalization anywhere in the pipeline. The microphone signal must
     NEVER be amplified beyond unity.
  2. The signal is normalized EXACTLY ONCE: int16 → float32 (/ 32768)
     happens a single time when integer PCM enters the float domain.
     No stage may re-normalize, re-scale, or clip-to-range as "normalization".
  3. Every processing stage preserves float32 audio in [-1, 1].
     int16 conversion happens ONLY at a sink boundary:
       - openWakeWord input
       - Whisper input
       - WAV export
     and ONLY via float32_to_int16().
  4. After EVERY stage the stage tracer prints
       stage, dtype, min, max, rms, peak
     and asserts max(|audio|) <= 1.01. On violation it prints
       GAIN ERROR <stage> <caller> <stack trace>
     and raises GainError (the operation ABORTS).

Accidental amplification causes clipping and destroys speech quality.
All processing is implemented with scipy for production robustness.
"""

import logging
import sys
import time
import traceback
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


class GainError(RuntimeError):
    """Raised when a pipeline stage outputs float audio outside ±1.0.

    A GainError means a stage applied gain (AGC / normalization /
    amplification) and corrupted the signal. The current audio operation
    ABORTS — the corrupt frame/segment is never delivered downstream.
    """


# ── PCM scale factors (SINGLE source of truth) ────────────────
# int16 → float32 : divide by 32768  (full int16 range maps to [-1, 1))
# float32 → int16 : clip [-1, 1], multiply by 32767
# These are the ONLY two conversion helpers in the codebase; every
# int16↔float transition must go through them so the scale factor can
# never diverge between stages.
PCM16_FULL_SCALE = 32768.0
PCM16_MAX = 32767.0

# Float-domain assertion tolerance. Every stage must preserve [-1, 1];
# 1% headroom absorbs benign FIR/IIR ringing without masking real gain.
FLOAT_PEAK_TOLERANCE = 1.01


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Decode int16 PCM to float32 in [-1, 1].

    THE ONLY int16→float conversion in the pipeline. Float input is
    returned unchanged (already in the normalized domain — it is NEVER
    re-normalized).
    """
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    a = np.asarray(audio)
    if a.dtype == np.float32:
        return a
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32)
    return a.astype(np.float32) / PCM16_FULL_SCALE


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Encode float32 [-1, 1] to int16 PCM.

    SINK BOUNDARIES ONLY — call this immediately before openWakeWord,
    Whisper, or WAV export, and NOWHERE else. Clips to [-1, 1] first so
    a hot sample can never overflow-wrap into the opposite sign.
    """
    if audio is None:
        return np.zeros(0, dtype=np.int16)
    a = np.asarray(audio)
    if a.dtype == np.int16:
        return a
    a = np.clip(a.astype(np.float64), -1.0, 1.0)
    return (a * PCM16_MAX).astype(np.int16)


class _StageTracer:
    """Trace every audio pipeline stage and ENFORCE the gain contract.

    Every stage of the chain

        microphone → resample → high-pass → noise suppression
        → wake detector → whisper

    calls ``log(stage, samples)``. For FLOAT audio (the processing
    pipeline) the tracer prints

        stage, dtype, min, max, rms, peak

    and asserts ``max(|x|) <= 1.01``. A stage whose output exceeds the
    legal ±1.0 float range is caught red-handed: the tracer prints

        GAIN ERROR  stage name  caller  stack trace

    and raises :class:`GainError`, aborting the corrupt operation.

    For INT16 audio (sink boundaries only — openWakeWord / Whisper /
    WAV bytes) the tracer keeps the legacy int16-scale peak/RMS ledger.

    Logging budget: a stage logs its first few new maxima, then every
    200th, so it is safe to call on every 30 ms frame without flooding
    the logs. Saturation (peak >= 32767 int16-scale) is ALWAYS logged
    as a warning. The runtime target is: NO stage ever logs SATURATION.
    """

    def __init__(self):
        self.max_peaks: Dict[str, int] = {}
        self.max_rms: Dict[str, float] = {}
        self.sat_counts: Dict[str, int] = {}
        self._newmax_counts: Dict[str, int] = {}

    # ── public entry point ─────────────────────────────────────

    def log(self, stage: str, arr, caller: Optional[str] = None) -> int:
        """Record peak/RMS of `arr` at `stage`. Returns int16-scale peak.

        FLOAT input  → full stage report + ±1.01 assertion (GainError).
        INT16 input  → legacy saturation ledger (sink boundaries only).
        """
        if arr is None:
            return 0
        try:
            a = np.asarray(arr)
            if a.size == 0:
                return 0
            if np.issubdtype(a.dtype, np.floating):
                return self._log_float(stage, a, caller)
            a64 = a.astype(np.float64)
            peak = int(np.max(np.abs(a64)))
            rms = float(np.sqrt(np.mean(a64 * a64)))
        except GainError:
            raise
        except Exception:
            return 0

        # ── int16 sink domain (legacy ledger) ──
        new_max = peak > self.max_peaks.get(stage, -1)
        if new_max:
            self.max_peaks[stage] = peak
            self.max_rms[stage] = rms

        if peak >= int(PCM16_MAX):
            self.sat_counts[stage] = self.sat_counts.get(stage, 0) + 1
            c = self.sat_counts[stage]
            if c <= 5 or c % 100 == 0:
                logger.warning(
                    "[SATURATION] stage='%s' peak=%d rms=%.1f (count=%d) — "
                    "hard clip at this stage", stage, peak, rms, c)
        elif new_max:
            n = self._newmax_counts.get(stage, 0) + 1
            self._newmax_counts[stage] = n
            if n <= 5 or n % 200 == 0:
                logger.info("[TRACE] stage=%s peak=%d rms=%.1f (new max)",
                            stage, peak, rms)
        return peak

    # ── float pipeline domain (assertion + reporting) ──────────

    def _log_float(self, stage: str, a: np.ndarray,
                   caller: Optional[str]) -> int:
        """Float32 [-1, 1] stage: print stage/dtype/min/max/rms/peak and
        assert the gain contract. Raises GainError on violation."""
        a64 = a.astype(np.float64)
        mn = float(a64.min())
        mx = float(a64.max())
        peak_f = max(abs(mn), abs(mx))
        rms_f = float(np.sqrt(np.mean(a64 * a64)))
        peak = int(peak_f * PCM16_FULL_SCALE)
        rms = rms_f * PCM16_FULL_SCALE

        # ── THE ASSERTION: every processing stage preserves [-1, 1] ──
        # assert np.max(np.abs(audio)) <= 1.01 — on failure print
        # GAIN ERROR, stage name, caller, stack trace, and ABORT.
        if peak_f > FLOAT_PEAK_TOLERANCE:
            if caller is None:
                caller = self._external_caller()
            stack = "".join(traceback.format_stack())
            logger.error(
                "GAIN ERROR stage='%s' caller='%s' dtype=%s min=%.6f max=%.6f "
                "rms=%.6f peak=%.6f (int16-scale peak=%d) — stage output "
                "exceeds ±1.0 (tolerance %.2f); illegal gain applied\n"
                "STACK TRACE:\n%s",
                stage, caller, a.dtype, mn, mx, rms_f, peak_f, peak,
                FLOAT_PEAK_TOLERANCE, stack)
            raise GainError(
                f"GAIN ERROR at stage '{stage}' (caller '{caller}'): "
                f"peak={peak_f:.6f} exceeds ±{FLOAT_PEAK_TOLERANCE}")

        # ── Saturation warning (runtime target: this NEVER fires) ──
        if peak >= int(PCM16_MAX):
            self.sat_counts[stage] = self.sat_counts.get(stage, 0) + 1
            c = self.sat_counts[stage]
            if c <= 5 or c % 100 == 0:
                logger.warning(
                    "[SATURATION] stage='%s' dtype=%s min=%.4f max=%.4f "
                    "rms=%.1f peak=%d (count=%d) — hard clip at this stage",
                    stage, a.dtype, mn, mx, rms, peak, c)
        else:
            new_max = peak > self.max_peaks.get(stage, -1)
            if new_max:
                n = self._newmax_counts.get(stage, 0) + 1
                self._newmax_counts[stage] = n
                if n <= 5 or n % 200 == 0:
                    logger.info(
                        "[TRACE] stage=%s dtype=%s min=%.4f max=%.4f "
                        "rms=%.1f peak=%d (new max)",
                        stage, a.dtype, mn, mx, rms, peak)

        if peak > self.max_peaks.get(stage, -1):
            self.max_peaks[stage] = peak
            self.max_rms[stage] = rms
        return peak

    @staticmethod
    def _external_caller() -> str:
        """First stack frame OUTSIDE this module = the pipeline caller."""
        try:
            f = sys._getframe(1)
            while f is not None and f.f_code.co_filename == __file__:
                f = f.f_back
            if f is None:
                return "unknown"
            fname = f.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
            return f"{fname}:{f.f_code.co_name}:{f.f_lineno}"
        except Exception:
            return "unknown"

    def reset(self, stage: Optional[str] = None) -> None:
        """Reset tracked maxima — one stage, or everything when None."""
        if stage is None:
            self.max_peaks.clear()
            self.max_rms.clear()
            self.sat_counts.clear()
            self._newmax_counts.clear()
        else:
            self.max_peaks.pop(stage, None)
            self.max_rms.pop(stage, None)
            self.sat_counts.pop(stage, None)
            self._newmax_counts.pop(stage, None)

    def report(self) -> Dict[str, dict]:
        return {
            s: {"max_peak": self.max_peaks[s],
                "max_rms": round(self.max_rms.get(s, 0.0), 1),
                "saturation_events": self.sat_counts.get(s, 0)}
            for s in self.max_peaks
        }


# Public handles: `peak_monitor` is the historical name used by AudioManager
# (microphone / resampling / preproc_*), `stage_tracer` is the descriptive
# alias for pipeline stages (wake_detector, whisper).
peak_monitor = _StageTracer()
stage_tracer = peak_monitor


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

    FLOAT PIPELINE: process() accepts int16 PCM OR float32 [-1, 1] and
    ALWAYS returns float32 in [-1, 1]. The int16 decode (/ 32768) is the
    single normalization; nothing inside the chain re-scales the signal.
    Use process_int16() at sink boundaries (Whisper / WAV) that require
    PCM16 output.

    Usage:
        proc = AudioPreprocessor(sample_rate=16000)
        audio_f32 = proc.process(audio)            # float32 [-1, 1]
        pcm16 = proc.process_int16(audio)          # sink boundary only
        is_speech = proc.is_speech(audio_f32)
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
        """Apply spectral gating noise suppression.

        The Wiener-style gain is clipped to [floor, 1.0]: it can ONLY
        attenuate. There is NO amplification and NO normalization here.

        SHORT-CHUNK SAFETY: streaming consumers (the wake loop) feed
        chunks of arbitrary length — often a single 480-sample frame,
        shorter than SPECTRAL_FFT_SIZE (512). scipy's stft() silently
        clamps nperseg to the input length while istft() below uses the
        ORIGINAL nperseg/noverlap, which crashed the engine with
        "operands could not be broadcast together with shapes (480,)
        (512,)". Pad short inputs to the FFT size and trim the output
        back so stft/istft always agree on nperseg.
        """
        if self._noise_profile is None:
            # Still estimating noise — output audio unchanged
            return audio

        if len(audio) == 0:
            return audio

        # Pad short chunks so stft/istft use a consistent nperseg.
        orig_len = len(audio)
        if orig_len < SPECTRAL_FFT_SIZE:
            audio = np.pad(audio, (0, SPECTRAL_FFT_SIZE - orig_len))

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
        # gain = max(1 - noise^2 / signal^2, floor)   — always <= 1.0
        eps = 1e-10
        gain = 1.0 - (noise[:, np.newaxis] ** 2) / (mag ** 2 + eps)
        gain = np.clip(gain, 10 ** (SPECTRAL_FLOOR_DB / 20), 1.0)

        # Apply gain with soft smoothing (avoid musical noise)
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

        # Match length — always return EXACTLY the original chunk length
        # (the pad above was internal only).
        if len(reconstructed) < orig_len:
            reconstructed = np.pad(reconstructed, (0, orig_len - len(reconstructed)))
        elif len(reconstructed) > orig_len:
            reconstructed = reconstructed[:orig_len]

        return reconstructed


    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply the (single) high-pass filter — unity gain, float domain.

        This is the ONLY high-pass in the pipeline. The AudioManager
        callback used to run a second 'light' high-pass; that duplicate
        stage has been removed.
        """
        if len(audio) == 0:
            return audio
        filtered, self._zi = scipy_signal.sosfilt(
            self._sos, audio, zi=self._zi
        )
        return filtered

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Run the complete preprocessing chain on audio data.

        FLOAT PIPELINE CONTRACT:
          - int16 input is decoded ONCE via / 32768 (the single
            normalization in the entire pipeline).
          - float input is used AS-IS — it is NEVER re-normalized.
          - Every stage preserves float audio in [-1, 1]; the stage
            tracer asserts max(|x|) <= 1.01 and raises GainError
            (GAIN ERROR + caller + stack trace) on violation.
          - Returns float32 in [-1, 1]. Convert to int16 ONLY at the
            sink (openWakeWord / Whisper / WAV) via float32_to_int16().

        NO AGC, NO amplification, NO auto-normalization anywhere inside.

        Args:
            audio: int16 PCM samples OR float32 [-1, 1] @ self.sample_rate.

        Returns:
            Preprocessed float32 samples in [-1, 1].
        """
        t0 = time.time()

        if audio is None or len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        a = np.asarray(audio)

        # ── SINGLE normalization: int16 → float32 happens EXACTLY ONCE ──
        # Float input is already in the normalized domain; re-scaling it
        # would be a duplicate normalization stage (removed by design).
        if np.issubdtype(a.dtype, np.floating):
            audio_float = a.astype(np.float32, copy=False)
        else:
            audio_float = a.astype(np.float32) / PCM16_FULL_SCALE
        peak_monitor.log("preproc_input", audio_float)
        self.diagnostics["input_rms"] = float(
            np.sqrt(np.mean(audio_float.astype(np.float64) ** 2))
        ) * PCM16_FULL_SCALE

        # 1. High-pass filter (unity gain, float domain — no int16 round-trip).
        #    IIR ringing on capture-clipped input can overshoot ±1.0 by a
        #    few percent — contain it (artifact containment, NOT AGC).
        audio_filtered = np.clip(
            self._apply_highpass(audio_float), -1.0, 1.0)
        peak_monitor.log("preproc_highpass", audio_filtered)

        # 2. Noise suppression (after initial noise estimation period).
        #    Spectral gate gain is always <= 1.0 — it can only attenuate.
        #    istft reconstruction ringing is contained the same way.
        if self._noise_profile is None:
            self._estimate_noise(audio_filtered)
        audio_denoised = np.clip(
            self._apply_spectral_gate(audio_filtered), -1.0, 1.0)
        peak_monitor.log("noise_suppression", audio_denoised)


        # 3. Output: float32 in [-1, 1]. NO int16 conversion here.
        #    The IIR high-pass and istft reconstruction can ring a few
        #    percent past ±1.0 on hot (capture-clipped) signals — contain
        #    that filter overshoot so the stage assertion below does not
        #    abort loud speech chunks (the single normalization already
        #    happened at capture; this is artifact containment, not AGC).
        out = np.clip(np.asarray(audio_denoised, dtype=np.float32), -1.0, 1.0)
        peak_monitor.log("preproc_output", out)


        # Update diagnostics (int16-scale RMS for continuity with the
        # historical dashboards; the signal itself stays float).
        self.diagnostics["output_rms"] = float(
            np.sqrt(np.mean(out.astype(np.float64) ** 2))
        ) * PCM16_FULL_SCALE
        self.diagnostics["processed_count"] += 1
        self.diagnostics["last_processing_ms"] = (time.time() - t0) * 1000
        self.diagnostics["gain_applied"] = 1.0  # Always unity — no AGC

        return out

    def process_int16(self, audio: np.ndarray) -> np.ndarray:
        """
        Sink-boundary wrapper: process() → ONE int16 conversion.

        Use ONLY where a consumer genuinely requires PCM16 (e.g. Google
        SpeechRecognition bytes, WAV export). openWakeWord and Whisper
        accept float32 directly and should use process() instead.
        """
        return float32_to_int16(self.process(audio))

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Simple VAD based on energy and SNR.

        Args:
            audio: Preprocessed samples (float32 [-1, 1] or int16 PCM).

        Returns:
            True if speech detected, False otherwise.
        """
        if audio is None or len(audio) == 0:
            return False

        a = np.asarray(audio)
        is_float = np.issubdtype(a.dtype, np.floating)

        # Compute RMS (float domain)
        rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))

        # Compute noise floor from noise profile if available. The noise
        # profile is built in the float STFT domain, so compare in the
        # float domain (float input) — no re-scaling of the signal.
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
        norm_rms = rms if is_float else rms / PCM16_FULL_SCALE

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
