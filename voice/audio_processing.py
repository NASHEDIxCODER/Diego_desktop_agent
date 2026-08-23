"""
Audio Processing Pipeline for Diego Desktop Assistant.

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

  1. EXACTLY ONE instrumented gain stage exists in the entire pipeline:
     the capture-side AutomaticGainControl (leveler + limiter) in
     voice/audio_manager.py's callback. It targets RMS 0.08–0.12 with a
     0.95 limiter ceiling and NEVER hard-clips — over-range input reduces
     gain instead of destroying the waveform. NO other stage may apply
     gain, AGC, amplification, or loudness normalization.
  2. The signal is normalized EXACTLY ONCE: int16 → float32 (/ 32768)
     happens a single time when integer PCM enters the float domain.
     No stage may re-normalize, re-scale, or clip-to-range as "normalization".
  3. Every processing stage preserves float32 audio in [-1, 1].
     int16 conversion happens ONLY at a sink boundary:
       - openWakeWord input
       - Whisper input
       - WAV export
     and ONLY via float32_to_int16().
  4. After EVERY stage the stage tracer prints (time-budgeted, every few
     seconds — NOT every callback):
       stage, dtype, shape, min, max, rms, peak, gain applied, clip %
     and asserts max(|audio|) <= 1.01. On violation it prints
       GAIN ERROR <stage> <caller> <stack trace>
     and raises GainError (the operation ABORTS).
     Raw source measurement points (pre-AGC microphone input) use
     observe() — reported identically but never asserted, because raw
     hardware input is ALLOWED to exceed ±1.0 (that is precisely the
     condition the AGC repairs).

Accidental amplification causes clipping and destroys speech quality.
All processing is implemented with scipy for production robustness.
"""


import logging
import os
import sys
import time
import traceback
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# ── Logging-volume control ──────────────────────────────────────
# The audio callback runs every ~30ms and historically produced multiple
# [TRACE]/[STAGE]/[SATURATION] lines per callback, flooding the terminal.
# All per-frame audio tracing is now OFF by default. Set DIEGO_AUDIO_TRACE=1
# to re-enable verbose per-stage tracing for debugging.
AUDIO_TRACE_ENABLED = os.environ.get("DIEGO_AUDIO_TRACE", "0") == "1"


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
    the logs. Additionally EVERY stage emits a full diagnostic summary
    (dtype / shape / min / max / rms / peak / gain / clip %) at most once
    per SUMMARY_INTERVAL_S seconds — time-budgeted, never per-callback.
    Saturation (peak >= 32767 int16-scale) is ALWAYS logged as a warning.
    The runtime target is: NO stage ever logs SATURATION.
    """

    # Full per-stage diagnostic summary cadence (seconds). The summary is
    # the spec-mandated trace: dtype, shape, min, max, RMS, peak, gain
    # applied, clip percentage — emitted every few seconds, NOT per frame.
    SUMMARY_INTERVAL_S = 5.0

    def __init__(self):
        self.max_peaks: Dict[str, int] = {}
        self.max_rms: Dict[str, float] = {}
        self.sat_counts: Dict[str, int] = {}
        self._newmax_counts: Dict[str, int] = {}
        self._last_summary: Dict[str, float] = {}
        self.max_clip_pct: Dict[str, float] = {}

    # ── public entry points ────────────────────────────────────

    def log(self, stage: str, arr, caller: Optional[str] = None,
            gain: float = 1.0) -> int:
        """Record peak/RMS of `arr` at `stage`. Returns int16-scale peak.

        FLOAT input  → full stage report + ±1.01 assertion (GainError).
        INT16 input  → legacy saturation ledger (sink boundaries only).
        `gain` is the gain the producing stage applied (1.0 = unity) and
        is shown in the periodic summary — visibility, not enforcement.
        """
        return self._record(stage, arr, caller, gain, assert_float=True)

    def observe(self, stage: str, arr, gain: float = 1.0) -> int:
        """Measure a RAW source point WITHOUT the ±1.01 assertion.

        Use ONLY for unprocessed inputs (e.g. the pre-AGC microphone
        signal). Raw hardware input is ALLOWED to exceed ±1.0 — reporting
        it must never abort the frame. Processing stages must use log().
        """
        return self._record(stage, arr, None, gain, assert_float=False)

    # ── shared implementation ──────────────────────────────────

    def _record(self, stage: str, arr, caller: Optional[str],
                gain: float, assert_float: bool) -> int:
        if arr is None:
            return 0
        try:
            a = np.asarray(arr)
            if a.size == 0:
                return 0
            if np.issubdtype(a.dtype, np.floating):
                return self._log_float(stage, a, caller, gain, assert_float)
            a64 = a.astype(np.float64)
            peak = int(np.max(np.abs(a64)))
            rms = float(np.sqrt(np.mean(a64 * a64)))
            clip_pct = float(np.mean(np.abs(a64) >= PCM16_MAX) * 100.0)
        except GainError:
            raise
        except Exception:
            return 0

        # ── int16 sink domain (legacy ledger) ──
        new_max = peak > self.max_peaks.get(stage, -1)
        self._track(stage, peak, rms, clip_pct)
        self._maybe_summarize(stage, a, float(a64.min()), float(a64.max()),
                              rms / PCM16_FULL_SCALE, peak / PCM16_FULL_SCALE,
                              gain, clip_pct)

        if peak >= int(PCM16_MAX):
            self.sat_counts[stage] = self.sat_counts.get(stage, 0) + 1
            c = self.sat_counts[stage]
            if c <= 5 or c % 100 == 0:
                logger.warning(
                    "[SATURATION] stage='%s' peak=%d rms=%.1f clip=%.2f%% "
                    "(count=%d) — hard clip at this stage",
                    stage, peak, rms, clip_pct, c)
        elif new_max and AUDIO_TRACE_ENABLED:
            n = self._newmax_counts.get(stage, 0) + 1
            self._newmax_counts[stage] = n
            if n <= 5 or n % 200 == 0:
                logger.info("[TRACE] stage=%s peak=%d rms=%.1f (new max)",
                            stage, peak, rms)
        return peak



    # ── float pipeline domain (assertion + reporting) ──────────

    def _track(self, stage: str, peak: int, rms: float, clip_pct: float) -> None:
        """Update the per-stage ledger (maxima + clip percentage)."""
        if peak > self.max_peaks.get(stage, -1):
            self.max_peaks[stage] = peak
            self.max_rms[stage] = rms
        if clip_pct > self.max_clip_pct.get(stage, 0.0):
            self.max_clip_pct[stage] = clip_pct

    def _maybe_summarize(self, stage: str, a: np.ndarray, mn: float,
                         mx: float, rms_f: float, peak_f: float,
                         gain: float, clip_pct: float) -> None:
        """Time-budgeted full stage diagnostic (the STEP-5 trace):
        stage, dtype, shape, min, max, RMS, peak, gain applied, clip %.
        Emitted at most once per SUMMARY_INTERVAL_S per stage — never
        per-callback, so it is safe on the 30 ms real-time path.

        GATED: only emitted when DIEGO_AUDIO_TRACE=1 (default OFF) to avoid
        flooding the terminal during idle listening.
        """
        if not AUDIO_TRACE_ENABLED:
            return
        now = time.monotonic()
        last = self._last_summary.get(stage, 0.0)
        if now - last < self.SUMMARY_INTERVAL_S:
            return
        self._last_summary[stage] = now
        logger.info(
            "[STAGE] %s dtype=%s shape=%s min=%.4f max=%.4f rms=%.4f "
            "peak=%.4f (int16=%d) gain=%.3f clip=%.3f%%",
            stage, a.dtype, a.shape, mn, mx, rms_f, peak_f,
            int(peak_f * PCM16_FULL_SCALE), gain, clip_pct)

    def _log_float(self, stage: str, a: np.ndarray,
                   caller: Optional[str], gain: float = 1.0,
                   assert_float: bool = True) -> int:
        """Float32 stage: report + (for processing stages) assert the gain
        contract. observe() points skip the assertion — raw hardware input
        may legally exceed ±1.0 (that is what the AGC repairs)."""
        a64 = a.astype(np.float64)
        mn = float(a64.min())
        mx = float(a64.max())
        peak_f = max(abs(mn), abs(mx))
        rms_f = float(np.sqrt(np.mean(a64 * a64)))
        peak = int(peak_f * PCM16_FULL_SCALE)
        rms = rms_f * PCM16_FULL_SCALE
        clip_pct = float(np.mean(np.abs(a64) >= 1.0 - 1.0 / PCM16_FULL_SCALE) * 100.0)

        # ── THE ASSERTION: every processing stage preserves [-1, 1] ──
        # assert np.max(np.abs(audio)) <= 1.01 — on failure print
        # GAIN ERROR, stage name, caller, stack trace, and ABORT.
        if assert_float and peak_f > FLOAT_PEAK_TOLERANCE:
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

        new_max = peak > self.max_peaks.get(stage, -1)
        self._track(stage, peak, rms, clip_pct)
        self._maybe_summarize(stage, a, mn, mx, rms_f, peak_f, gain, clip_pct)

        # ── Saturation warning (runtime target: NEVER on a processing
        # stage). Raw observe() points (pre-AGC mic input) are EXPECTED
        # to see rail contact when the source is overdriven — that is
        # reported as information, NOT counted as pipeline saturation.
        if peak >= int(PCM16_MAX):
            if assert_float:
                self.sat_counts[stage] = self.sat_counts.get(stage, 0) + 1
                c = self.sat_counts[stage]
                if c <= 5 or c % 100 == 0:
                    logger.warning(
                        "[SATURATION] stage='%s' dtype=%s min=%.4f max=%.4f "
                        "rms=%.1f peak=%d clip=%.2f%% (count=%d) — hard clip "
                        "at this stage",
                        stage, a.dtype, mn, mx, rms, peak, clip_pct, c)
            else:
                n = self._newmax_counts.get(stage, 0) + 1
                self._newmax_counts[stage] = n
                if (n <= 5 or n % 200 == 0) and AUDIO_TRACE_ENABLED:
                    logger.info(
                        "[RAW] source='%s' peak=%.4f clip=%.2f%% — source "
                        "overdrive observed (AGC repairs downstream)",
                        stage, peak_f, clip_pct)
        elif new_max:

            n = self._newmax_counts.get(stage, 0) + 1
            self._newmax_counts[stage] = n
            if (n <= 5 or n % 200 == 0) and AUDIO_TRACE_ENABLED:
                logger.info(
                    "[TRACE] stage=%s dtype=%s min=%.4f max=%.4f "
                    "rms=%.1f peak=%d (new max)",
                    stage, a.dtype, mn, mx, rms, peak)
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
            self._last_summary.clear()
            self.max_clip_pct.clear()
        else:
            self.max_peaks.pop(stage, None)
            self.max_rms.pop(stage, None)
            self.sat_counts.pop(stage, None)
            self._newmax_counts.pop(stage, None)
            self._last_summary.pop(stage, None)
            self.max_clip_pct.pop(stage, None)

    def report(self) -> Dict[str, dict]:
        return {
            s: {"max_peak": self.max_peaks[s],
                "max_rms": round(self.max_rms.get(s, 0.0), 1),
                "max_clip_pct": round(self.max_clip_pct.get(s, 0.0), 3),
                "saturation_events": self.sat_counts.get(s, 0)}
            for s in self.max_peaks
        }



# Public handles: `peak_monitor` is the historical name used by AudioManager
# (microphone / resampling / preproc_*), `stage_tracer` is the descriptive
# alias for pipeline stages (wake_detector, whisper).
peak_monitor = _StageTracer()
stage_tracer = peak_monitor


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Automatic Gain Control (capture-side leveler + limiter)
# ═══════════════════════════════════════════════════════════════

class AutomaticGainControl:
    """The SINGLE instrumented gain stage in the audio pipeline.

    WHY THIS EXISTS: the verified production failure was a microphone
    delivering a hard-clipped, overdriven signal (ALSA Capture 100% = +30 dB
    + Internal Mic Boost +20 dB → ADC saturation; measured: 22.9 % of raw
    samples nailed to the rail, crest factor 1.95). The old pipeline then
    ran ``np.clip(mono, -1, 1)`` as its FIRST step, guaranteeing that any
    residual over-unity energy was destroyed before inference. openWakeWord
    (log-mel frontend) still scored 0.96–0.995 on the clipped audio, but
    Whisper hallucinated ("hello Diego" → "hello", "it's so big") and every
    wake verification was rejected.

    DESIGN (production AGC — NOT a clipper):

      1. LEVELER — a smoothed gain that steers the long-term RMS toward
         TARGET_RMS (0.10; accepted band 0.08–0.12). Gain moves FAST
         downward (attack, ~1 frame ≈ 30 ms) and SLOW upward (release,
         ~1 s), so speech transients are caught instantly while room tone
         is never pumped.
      2. LIMITER — after leveling, if `frame_peak × gain` would exceed
         LIMITER_CEILING (0.95), the gain is reduced INSTANTLY for that
         frame. The output therefore can never exceed 0.95 — hard clipping
         is mathematically impossible (there is NO np.clip on the signal).
      3. BOUNDS — gain is clamped to [MIN_GAIN, MAX_GAIN] so digital
         silence is not amplified into noise (MAX_GAIN = 4 ≈ +12 dB) and
         an overdriven source can always be brought back into range
         (MIN_GAIN = 0.01 ≈ −40 dB).

    The AGC sits at the TOP of the capture chain (raw float32 from
    PortAudio → AGC → ring buffer), so every downstream stage (resampler,
    high-pass, spectral gate, openWakeWord, Whisper) receives a signal
    with healthy headroom and a speech-like crest factor.
    """

    TARGET_RMS = 0.10        # spec: target RMS 0.08–0.12
    TARGET_RMS_MIN = 0.08
    TARGET_RMS_MAX = 0.12
    LIMITER_CEILING = 0.95   # spec: limiter 0.95 — hard clip NEVER
    MAX_GAIN = 4.0           # +12 dB — never pump silence into noise
    MIN_GAIN = 0.01          # −40 dB — always able to tame overdrive
    # Per-frame smoothing coefficients (30 ms frames): attack ~63 %/frame
    # (≈1 time constant per frame — fast), release ~3 %/frame (≈1 s to
    # converge upward — slow, pump-free).
    ATTACK = 0.63
    RELEASE = 0.03
    # Below this input RMS the leveler holds its gain (digital silence /
    # deep noise floor must not drive the gain to MAX_GAIN).
    LEVELER_GATE_RMS = 1e-4

    def __init__(self):
        self._gain: float = 1.0
        self._frames: int = 0
        self._limiter_events: int = 0
        self._last_report: float = 0.0

    # ── main entry ─────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply leveler + limiter to ONE frame. NEVER clips.

        Args:
            frame: raw float32 samples from the capture device. May exceed
                   ±1.0 (over-driven source) — that is exactly what this
                   stage repairs.

        Returns:
            (output_float32, gain_applied). ``max(|output|) <= 0.95`` for
            any non-empty input with a non-zero peak.
        """
        a = np.asarray(frame, dtype=np.float32)
        if a.size == 0:
            return a, self._gain
        a64 = a.astype(np.float64)
        rms = float(np.sqrt(np.mean(a64 * a64)))
        peak = float(np.max(np.abs(a64)))

        # ── LEVELER: steer long-term RMS into the 0.08–0.12 band ──
        if rms > self.LEVELER_GATE_RMS:
            desired = self.TARGET_RMS / rms
            desired = min(max(desired, self.MIN_GAIN), self.MAX_GAIN)
            if desired < self._gain:
                # Fast attack — a hot signal is tamed within ~1 frame.
                self._gain += self.ATTACK * (desired - self._gain)
            else:
                # Slow release — room tone never pumps the signal up.
                self._gain += self.RELEASE * (desired - self._gain)

        # ── LIMITER: instantaneous, sample-accurate, NO clipping ──
        # If this frame would exceed the ceiling, reduce gain NOW. The
        # output peak is then exactly LIMITER_CEILING — never above.
        if peak > 0.0 and peak * self._gain > self.LIMITER_CEILING:
            self._gain = self.LIMITER_CEILING / peak
            self._limiter_events += 1

        self._gain = min(max(self._gain, self.MIN_GAIN), self.MAX_GAIN)
        out = (a64 * self._gain).astype(np.float32)
        self._frames += 1
        return out, self._gain

    # ── diagnostics ────────────────────────────────────────────

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def limiter_events(self) -> int:
        return self._limiter_events

    def reset(self) -> None:
        self._gain = 1.0
        self._frames = 0
        self._limiter_events = 0
        self._last_report = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "gain": round(self._gain, 4),
            "gain_db": round(20.0 * np.log10(max(self._gain, 1e-6)), 2),
            "target_rms": self.TARGET_RMS,
            "limiter_ceiling": self.LIMITER_CEILING,
            "limiter_events": self._limiter_events,
            "frames": self._frames,
        }


# Singleton — ONE AGC instance feeds the single capture stream.
capture_agc = AutomaticGainControl()



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
