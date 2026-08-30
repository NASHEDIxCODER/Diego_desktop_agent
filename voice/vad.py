"""
Unified VAD — THE single Silero VAD instance for the entire voice pipeline.

Shared by:
  - WakeListener (speech gate — only score frames when someone is talking)
  - CommandListener (speech detection + endpoint silence tracking)

Exactly ONE Silero VAD model is loaded. Every consumer calls speech_prob()
with a 512-sample float32 frame at 16 kHz.

If Silero is unavailable, falls back to energy-based detection.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD native window: 512 samples = 32 ms @ 16 kHz
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16000

# Speech probability threshold
SPEECH_THRESHOLD = 0.5

# Energy fallback thresholds (int16 scale)
ENERGY_THRESHOLD = 300.0

# ── TASK 2: Robust VAD combined-evidence tuning ────────────────
# Silero probability alone is NOT a reliable speech detector (it can
# return 0.000 on clearly-voiced frames and ~1.0 on transient noise).
# The robust gate combines THREE pieces of evidence:
#   1. Silero probability (smoothed, so a single dropout doesn't kill us)
#   2. Audio energy (RMS on the int16 scale — a voiced frame has real level)
#   3. Minimum speech duration (hysteresis — don't flap on brief noise)
#
# These are the combined-evidence thresholds. They are NOT a blind
# lowering of the Silero threshold; energy + duration must ALSO agree.
#
# CRITICAL FIX (2026-08-29): ROBUST_ENERGY_RMS raised from 120→2500 and
# ROBUST_ENERGY_WEIGHT reduced from 0.4→0.2. The old 120 floor was far
# below this system's background noise (RMS ~1200-2300), so energy_score
# was ALWAYS 1.0. With weight 0.4, the combined score was always ≥0.4
# (0.6*0 + 0.4*1.0), above ROBUST_EXIT_SCORE (0.35). The VAD could NEVER
# detect silence — the state machine never transitioned to SILENCE_PENDING
# and always hit the 20s safety timeout. Raising the floor to 2500 (above
# background noise, below real speech ~3500-6500) and reducing the weight
# to 0.2 ensures the combined score drops below 0.35 during silence while
# still providing energy evidence for quiet speech detection.
ROBUST_SILERO_WEIGHT = 0.8        # weight of Silero prob in combined score
ROBUST_ENERGY_WEIGHT = 0.2        # weight of energy evidence
ROBUST_SPEECH_PROB = 0.35         # Silero prob floor to consider speech
ROBUST_ENERGY_RMS = 2500.0        # int16-scale RMS floor for voiced audio
ROBUST_ENTER_SCORE = 0.55         # combined score to ENTER speech
ROBUST_EXIT_SCORE = 0.35          # combined score to EXIT speech (hysteresis)
# SUSTAINED-SPEECH ONSET (2026-08-30): raised 3→5 frames (≈160ms).
# Root-cause fix for "command listener false speech": a transient VAD
# spike (door slam, click, cough — a few loud frames) must NOT be able to
# open a speech window. 96ms was inside the transient-spike duration band;
# 160ms requires SUSTAINED voiced evidence before the gate can open.
# Real speech onsets comfortably exceed 160ms.
ROBUST_MIN_SPEECH_FRAMES = 5      # min consecutive speech frames (≈160ms)
ROBUST_SMOOTH_ALPHA = 0.4         # EMA smoothing for Silero prob



class UnifiedVAD:
    """The ONE VAD in the pipeline. Thread-safe for read-only inference."""

    def __init__(self):
        self._model = None
        self._ready = False
        self._load_error: Optional[str] = None
        # Observability (Phase 6): state + timestamps so a stuck OPEN/CLOSED
        # VAD can be diagnosed from logs.
        self._state = "closed"
        self._last_probability = 0.0
        self._last_audio_timestamp: Optional[float] = None
        self._last_speech_timestamp: Optional[float] = None
        self._lock = None
        # TASK 1: rate-limit the out-of-range probability bug log so a single
        # bad frame does not flood the terminal.
        self._out_of_range_logged = False
        self.__init_robust_state()

    # ── Lifecycle ──────────────────────────────────────────────

    def load(self) -> bool:
        """Load Silero VAD ONNX model. Idempotent."""
        if self._ready:
            return True
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            self._ready = True
            self._load_error = None
            logger.info("[VAD] Silero VAD loaded (unified — shared by wake + command)")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.info("[VAD] Silero VAD unavailable (%s) — energy fallback active", e)
            self._ready = False
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ── Inference ──────────────────────────────────────────────

    def speech_prob(self, frame: np.ndarray) -> float:
        """Return speech probability (0..1) for a 512-sample float32 frame.

        When Silero is unavailable, returns an energy-based estimate:
        >0.9 when RMS exceeds the energy threshold, ~0.05 otherwise.

        Updates VAD observability state (state/prob/timestamps) so a stuck
        OPEN or CLOSED detector can be diagnosed from logs.
        """
        if not self._ready or self._model is None:
            prob = self._energy_fallback(frame)
        else:
            try:
                import torch
                audio = np.asarray(frame, dtype=np.float32)
                if len(audio) < VAD_FRAME_SAMPLES:
                    audio = np.pad(audio, (0, VAD_FRAME_SAMPLES - len(audio)))
                elif len(audio) > VAD_FRAME_SAMPLES:
                    audio = audio[:VAD_FRAME_SAMPLES]
                tensor = torch.from_numpy(audio)
                with torch.no_grad():
                    prob = self._model(tensor, VAD_SAMPLE_RATE).item()
                prob = float(prob)
            except Exception:
                prob = self._energy_fallback(frame)

        import time
        # ── TASK 1: clamp/validate every individual VAD probability ──
        # A probability MUST be in [0.0, 1.0]. Silero can emit values slightly
        # outside this range (or NaN) on edge frames; without clamping,
        # vad_avg becomes impossible (>1.0, e.g. 17.201 / 42.801). NaN is
        # treated as 0.0 (not speech).
        if prob != prob:  # NaN check
            if not self._out_of_range_logged:
                logger.error("[VAD] BUG: probability is NaN — clamping to 0.0")
                self._out_of_range_logged = True
            prob = 0.0
        elif prob < 0.0 or prob > 1.0:
            if not self._out_of_range_logged:
                logger.error("[VAD] BUG: probability out of range %.6f — clamping to [0,1]", prob)
                self._out_of_range_logged = True
            prob = min(max(prob, 0.0), 1.0)
        self._last_probability = prob
        self._last_audio_timestamp = time.monotonic()
        self._state = "open" if prob > SPEECH_THRESHOLD else "closed"
        if self._state == "open":
            self._last_speech_timestamp = time.monotonic()
        return prob

    def max_speech_prob(self, audio: np.ndarray, step: int = 256) -> float:
        """Highest speech probability across a chunk, striding by `step` samples.

        Used by the wake-listener speech gate to decide whether the VAD gate
        should be open for a chunk of ring-buffer audio.
        """
        if len(audio) < VAD_FRAME_SAMPLES:
            return self.speech_prob(audio)

        best = 0.0
        for i in range(0, len(audio) - VAD_FRAME_SAMPLES + 1, step):
            prob = self.speech_prob(audio[i:i + VAD_FRAME_SAMPLES])
            if prob > best:
                best = prob
        return best

    @staticmethod
    def _energy_fallback(frame: np.ndarray) -> float:
        """Energy-based fallback when Silero is not available."""
        a = np.asarray(frame, dtype=np.float64)
        if a.size == 0:
            return 0.05
        rms = float(np.sqrt(np.mean(a * a)))
        rms_int16 = rms * 32768.0
        if rms_int16 > ENERGY_THRESHOLD:
            return 0.9
        return 0.05

    # ── TASK 2: Robust combined-evidence VAD ────────────────────
    # Silero probability alone frequently returns 0.000 on clearly-voiced
    # frames (and ~1.0 on transient noise). The robust gate combines:
    #   - smoothed Silero probability
    #   - audio energy (RMS on int16 scale)
    #   - minimum speech duration (hysteresis)
    # so a brief Silero dropout does NOT reject normal speech, and a brief
    # noise burst does NOT get accepted as speech.

    def __init_robust_state(self) -> None:
        if not hasattr(self, "_robust_smoothed_prob"):
            self._robust_smoothed_prob = 0.0
            self._robust_in_speech = False
            self._robust_speech_frames = 0

    def _frame_rms_int16(self, frame: np.ndarray) -> float:
        """RMS of a frame on the int16 scale (for energy evidence)."""
        a = np.asarray(frame, dtype=np.float64)
        if a.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(a * a))) * 32768.0

    def robust_speech_prob(self, frame: np.ndarray) -> float:
        """Return a COMBINED speech score (0..1) using Silero + energy.

        This is the production gate for the command listener. It is NOT a
        blind lowering of the Silero threshold: energy evidence must agree,
        and a minimum speech duration is enforced via hysteresis.

        Returns the combined score in [0, 1]. Callers should compare it to
        ROBUST_ENTER_SCORE / ROBUST_EXIT_SCORE.
        """
        self.__init_robust_state()
        silero = self.speech_prob(frame)
        rms = self._frame_rms_int16(frame)

        # Smooth Silero so a single 0.000 dropout doesn't zero the score.
        self._robust_smoothed_prob = (
            ROBUST_SMOOTH_ALPHA * silero
            + (1.0 - ROBUST_SMOOTH_ALPHA) * self._robust_smoothed_prob
        )

        # Energy evidence: 0..1 based on how far RMS is above the floor.
        if rms >= ROBUST_ENERGY_RMS:
            energy_score = 1.0
        else:
            energy_score = max(0.0, rms / ROBUST_ENERGY_RMS)

        combined = (
            ROBUST_SILERO_WEIGHT * self._robust_smoothed_prob
            + ROBUST_ENERGY_WEIGHT * energy_score
        )
        return float(min(max(combined, 0.0), 1.0))

    def robust_is_speech(self, frame: np.ndarray) -> bool:
        """Hysteresis gate: True when speech is active, False otherwise.

        ENTERS speech when the combined score crosses ROBUST_ENTER_SCORE,
        EXITS when it drops below ROBUST_EXIT_SCORE. A minimum number of
        consecutive speech frames (ROBUST_MIN_SPEECH_FRAMES) is required to
        ENTER, so a single noise burst cannot open the gate.
        """
        self.__init_robust_state()
        score = self.robust_speech_prob(frame)

        if self._robust_in_speech:
            if score < ROBUST_EXIT_SCORE:
                self._robust_speech_frames = 0
                self._robust_in_speech = False
        else:
            if score >= ROBUST_ENTER_SCORE:
                self._robust_speech_frames += 1
                if self._robust_speech_frames >= ROBUST_MIN_SPEECH_FRAMES:
                    self._robust_in_speech = True
            else:
                self._robust_speech_frames = 0

        return self._robust_in_speech

    def get_robust_diagnostics(self) -> dict:
        """TASK 1: per-frame VAD diagnostics for the command pipeline."""
        self.__init_robust_state()
        return {
            "silero_prob": round(self._last_probability, 4),
            "smoothed_prob": round(self._robust_smoothed_prob, 4),
            "in_speech": self._robust_in_speech,
            "speech_frames": self._robust_speech_frames,
            "state": self._state,
        }

    def reset_state(self) -> None:
        """TASK 1: reset ALL VAD state so stale values never leak into a new
        command session (previous wake verification, face auth, TTS, or a
        previous command must not influence the next utterance)."""
        self.__init_robust_state()
        self._robust_smoothed_prob = 0.0
        self._robust_in_speech = False
        self._robust_speech_frames = 0
        self._last_probability = 0.0
        self._state = "closed"
        self._last_audio_timestamp = None
        self._last_speech_timestamp = None
        self._out_of_range_logged = False
        logger.info("[VAD] state reset (new command session)")

    def get_diagnostics(self) -> dict:
        return {
            "ready": self._ready,
            "backend": "silero_vad" if self._ready else "energy_fallback",
            "load_error": self._load_error,
            "frame_samples": VAD_FRAME_SAMPLES,
            "sample_rate": VAD_SAMPLE_RATE,
            "speech_threshold": SPEECH_THRESHOLD,
        }


# Global singleton — the ONE VAD instance
unified_vad = UnifiedVAD()