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


class UnifiedVAD:
    """The ONE VAD in the pipeline. Thread-safe for read-only inference."""

    def __init__(self):
        self._model = None
        self._ready = False
        self._load_error: Optional[str] = None

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
        """
        if not self._ready or self._model is None:
            return self._energy_fallback(frame)

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
            return float(prob)
        except Exception:
            return self._energy_fallback(frame)

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