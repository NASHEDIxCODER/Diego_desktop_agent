"""
WakeModelManager — Production wake-model lifecycle for Leo.

Responsibilities:
  - Load the wake model (WAKE_MODEL env → custom models/wake/*.onnx →
    bundled openWakeWord model matching WAKE_PHRASE). NEVER hardcodes hey_jarvis.
  - Reload the model.
  - Verify the model exists.
  - Print the loaded wake phrase + model name.
  - Warm up the openWakeWord prediction buffer (the first 4 predictions are
    silently zeroed by openWakeWord unless the buffer is primed).
  - Measure inference latency.
  - Track confidence, false positives, and false rejects.
  - Attach a custom speaker verifier (models/wake/verifier.pkl + metadata.json)
    automatically when present.

The verifier is a logistic-regression model trained with
openwakeword.train_custom_verifier on the USER's voice saying the wake phrase.
It is attached to the base openWakeWord model via `custom_verifier_models`.
"""

import difflib
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Bundled fallback used only when no model matches the wake phrase.
# NEVER hardcode hey_jarvis — models are selected dynamically below.
DEFAULT_BUNDLED_MODEL = "hey_marvin"

# Files in the openWakeWord resources dir that are not wake-word models
NON_WAKE_FILES = {"melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx"}

WARMUP_FRAMES = 5  # openWakeWord zeroes predictions until the buffer has ≥5 frames
WARMUP_FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz


class WakeModelManager:
    """Manages the openWakeWord base model + optional custom verifier."""

    def __init__(self):
        self._model = None                      # openwakeword.Model or None
        self._model_path: Optional[Path] = None
        self._model_name: Optional[str] = None
        self._verifier_path: Optional[Path] = None
        self._loaded = False
        self._load_error: Optional[str] = None

        self._wake_phrase: str = voice_settings.wake_phrase
        self._threshold: float = 0.5
        self._vad_threshold: float = 0.0        # openWakeWord VAD disabled; Silero VAD upstream

        # Metrics
        self._latency_samples: deque = deque(maxlen=100)
        self._last_prediction: Dict[str, float] = {}
        self._detections: int = 0
        self._false_positives: int = 0
        self._false_rejects: int = 0

    # ── Public lifecycle ───────────────────────────────────────────

    def load(self,
             model_path: Optional[str] = None,
             wake_phrase: Optional[str] = None) -> bool:
        """
        Load the wake model.

        Args:
            model_path: Explicit path to an ONNX model. If None, auto-resolve:
                        1. settings.WAKE_MODEL / voice_settings.wake_model
                        2. models/wake/*.onnx (custom trained model)
                        3. Bundled openWakeWord model best matching WAKE_PHRASE
            wake_phrase: Phrase to associate with the model (defaults to config).

        Returns:
            True if the model loaded successfully.
        """
        self.close()

        if wake_phrase:
            self._wake_phrase = wake_phrase
        elif voice_settings.wake_phrase:
            self._wake_phrase = voice_settings.wake_phrase

        try:
            resolved = Path(model_path).expanduser() if model_path else self._resolve_model_path()
            if resolved is None or not resolved.exists():
                self._load_error = (f"Wake model not found ({resolved or 'no candidate'}); "
                                    f"check WAKE_MODEL or run --train-wake")
                logger.error("[WAKE] %s", self._load_error)
                return False

            self._model_path = resolved
            self._model_name = resolved.stem

            from openwakeword import Model as OWWModel

            kwargs: dict = {"vad_threshold": self._vad_threshold}
            verifier = self._resolve_verifier(self._model_name)
            if verifier is not None:
                kwargs["custom_verifier_models"] = {self._model_name: str(verifier)}
                self._verifier_path = verifier
                # The verifier must fire on EVERY frame so that speaker
                # verification is always applied (even when the base model
                # scores the wake phrase < 0.1). Read the threshold from
                # verifier metadata (default 0.0 = always verify).
                meta = self._load_verifier_metadata()
                kwargs["custom_verifier_threshold"] = float(
                    meta.get("verifier_threshold", 0.0)
                )
                # Use the detection threshold computed during calibration so
                # the verifier's probabilities are compared against the
                # empirically optimal cut-off (not the generic 0.5).
                if "detection_threshold" in meta:
                    self._threshold = float(meta["detection_threshold"])

            self._model = OWWModel(wakeword_model_paths=[str(resolved)], **kwargs)
            self._loaded = True
            self._load_error = None

            # Prime the openWakeWord prediction buffer so the first real clip
            # is not silently zeroed (buffer must contain ≥5 frames).
            self._warmup()
            self.print_loaded_wake_phrase()
            return True

        except Exception as e:
            self._model = None
            self._loaded = False
            self._load_error = str(e)
            logger.error("[WAKE] Model load failed: %s", e, exc_info=True)
            return False

    def reload(self,
               model_path: Optional[str] = None,
               wake_phrase: Optional[str] = None) -> bool:
        """Close the current model and reload."""
        self.close()
        return self.load(model_path=model_path, wake_phrase=wake_phrase)

    def close(self) -> None:
        """Release the model and reset metrics."""
        self._model = None
        self._model_path = None
        self._model_name = None
        self._verifier_path = None
        self._loaded = False
        self._latency_samples.clear()
        self._last_prediction = {}
        self._load_error = None

    # ── Verification ───────────────────────────────────────────────

    def verify_model_exists(self) -> bool:
        """Check that a resolvable wake model exists on disk."""
        p = self._resolve_model_path()
        return p is not None and p.exists()

    def print_loaded_wake_phrase(self) -> None:
        """Print the loaded wake phrase and model to the console."""
        print(f"  Wake phrase: '{self._wake_phrase}'")
        print(f"  Wake model:  {self._model_name or 'none'}"
              f" ({'custom verifier attached' if self._verifier_path else 'no verifier'})")

    # ── Inference ──────────────────────────────────────────────────

    def predict(self, audio_int16: np.ndarray) -> Dict[str, float]:
        """
        Run the wake model on int16 PCM audio using STREAMING prediction.

        openWakeWord's feature buffer holds at most ~10s (120 frames); feeding
        a long clip in a single predict() call overflows it and raises an
        ONNX dimension error. Production wake detection therefore streams
        audio in chunks of 1280 samples (80 ms) and keeps the MAX score
        across all frames — identical to real-time usage.

        Args:
            audio_int16: int16 samples @ 16 kHz (any length ≥ 1 sample).

        Returns:
            Dict mapping model/class name → max score (0..1). Empty on failure.
        """
        if not self._loaded or self._model is None:
            return {}

        if audio_int16.dtype != np.float32:
            audio_float = audio_int16.astype(np.float32) / 32768.0
        else:
            audio_float = audio_int16

        if len(audio_float) == 0:
            return {}

        # Pad to a multiple of 1280 samples (openWakeWord frame size).
        remainder = len(audio_float) % WARMUP_FRAME_SAMPLES
        if remainder:
            pad = WARMUP_FRAME_SAMPLES - remainder
            audio_float = np.pad(audio_float, (0, pad))

        t0 = time.perf_counter()
        try:
            best: Dict[str, float] = {}
            for i in range(0, len(audio_float), WARMUP_FRAME_SAMPLES):
                frame = audio_float[i:i + WARMUP_FRAME_SAMPLES]
                preds = self._model.predict(frame)
                for name, score in preds.items():
                    s = float(score)
                    if s > best.get(name, 0.0):
                        best[name] = s
        except Exception as e:
            logger.debug("[WAKE] predict error: %s", e)
            self._latency_samples.append(time.perf_counter() - t0)
            return {}
        latency = time.perf_counter() - t0
        self._latency_samples.append(latency)
        self._last_prediction = best
        return self._last_prediction

    def predict_stream(self, frame: np.ndarray) -> Dict[str, float]:
        """Feed ONE 1280-sample frame to openWakeWord in continuous streaming
        mode. The model's internal feature/prediction buffers persist across
        calls, so this must be called with non-overlapping sequential frames
        (see AudioManager.read_since).

        Args:
            frame: int16 or float32 audio frame (exactly 1280 samples is ideal;
                   shorter/longer frames are padded/truncated to 1280).

        Returns:
            Dict mapping model/class name → score for this frame.
        """
        if not self._loaded or self._model is None:
            return {}
        if frame.dtype != np.float32:
            audio_float = frame.astype(np.float32) / 32768.0
        else:
            audio_float = frame
        if len(audio_float) == 0:
            return {}
        # Normalize to exactly 1280 samples (openWakeWord frame size).
        if len(audio_float) < WARMUP_FRAME_SAMPLES:
            audio_float = np.pad(audio_float, (0, WARMUP_FRAME_SAMPLES - len(audio_float)))
        elif len(audio_float) > WARMUP_FRAME_SAMPLES:
            audio_float = audio_float[:WARMUP_FRAME_SAMPLES]
        t0 = time.perf_counter()
        try:
            preds = self._model.predict(audio_float)
        except Exception as e:
            logger.debug("[WAKE] predict_stream error: %s", e)
            self._latency_samples.append(time.perf_counter() - t0)
            return {}
        self._latency_samples.append(time.perf_counter() - t0)
        self._last_prediction = {k: float(v) for k, v in preds.items()}
        return self._last_prediction

    def reset_stream(self) -> None:
        """Reset openWakeWord's prediction buffer.

        openWakeWord is designed for CONTINUOUS streaming: the preprocessor's
        rolling feature window naturally drops audio older than ~1.5s, so the
        production wake loop never resets between speech segments. This light
        reset (prediction_buffer only) is provided for external/rare use
        (e.g. after a reload). NOTE: after reset, the next 5 frames score 0
        (openWakeWord's init behavior), so prefer continuous streaming.
        """
        if self._model is None:
            return
        try:
            self._model.reset()
        except Exception as e:
            logger.debug("[WAKE] reset_stream model.reset error: %s", e)

    def detect(self,
               audio_int16: np.ndarray,
               threshold: Optional[float] = None) -> bool:
        """
        Detect the wake phrase in audio. The custom verifier (if attached)
        is applied inside openWakeWord's Model.predict.

        Returns:
            True if the highest-scoring class exceeded the threshold.
        """
        preds = self.predict(audio_int16)
        if not preds:
            return False
        thresh = threshold if threshold is not None else self._threshold
        _, best_score = self.highest_score()
        if best_score >= thresh:
            self._detections += 1
            return True
        return False

    def highest_score(self) -> tuple:
        """Return (model_name, score) of the highest-scoring class."""
        best_name, best_score = "", 0.0
        for name, score in self._last_prediction.items():
            if score > best_score:
                best_name, best_score = name, float(score)
        return best_name, best_score

    # ── Metrics / accounting ───────────────────────────────────────

    def record_false_positive(self) -> None:
        """Called by the pipeline when wake fired but transcript did not match."""
        self._false_positives += 1

    def record_false_reject(self) -> None:
        """Called by the pipeline when VAD heard speech but wake did not fire."""
        self._false_rejects += 1

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def model_path(self) -> Optional[Path]:
        return self._model_path

    @property
    def wake_phrase(self) -> str:
        return self._wake_phrase

    @property
    def verifier_path(self) -> Optional[Path]:
        return self._verifier_path

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def threshold(self) -> float:
        """Current detection threshold (from calibration metadata or 0.5)."""
        return self._threshold

    @property
    def detections(self) -> int:
        return self._detections

    @property
    def false_positives(self) -> int:
        return self._false_positives

    @property
    def false_rejects(self) -> int:
        return self._false_rejects

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if not self._latency_samples:
            return 0.0
        return (sum(self._latency_samples) / len(self._latency_samples)) * 1000.0

    def get_diagnostics(self) -> dict:
        """Serializable diagnostics for health reporting."""
        return {
            "loaded": self._loaded,
            "model_name": self._model_name,
            "model_path": str(self._model_path) if self._model_path else None,
            "verifier_path": str(self._verifier_path) if self._verifier_path else None,
            "wake_phrase": self._wake_phrase,
            "threshold": self._threshold,
            "load_error": self._load_error,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "detections": self._detections,
            "false_positives": self._false_positives,
            "false_rejects": self._false_rejects,
            "last_prediction": dict(self._last_prediction),
        }

    # ── Model resolution ───────────────────────────────────────────

    def _resolve_model_path(self) -> Optional[Path]:
        """Resolve the base wake model path in priority order."""
        # 1) Explicit WAKE_MODEL configuration
        cfg = (os.getenv("WAKE_MODEL")
               or getattr(settings, "WAKE_MODEL", None)
               or voice_settings.wake_model)
        if cfg:
            p = Path(str(cfg)).expanduser()
            if p.exists():
                return p
            p2 = settings.BASE_DIR / str(cfg)
            if p2.exists():
                return p2
            logger.warning("[WAKE] WAKE_MODEL='%s' not found at %s or %s",
                           cfg, p, p2)

        # 2) Custom ONNX model in models/wake/ (trained wake model)
        try:
            for f in sorted(settings.MODELS_WAKE_DIR.glob("*.onnx")):
                if f.name in NON_WAKE_FILES:
                    continue
                return f
        except Exception as e:
            logger.debug("[WAKE] models/wake glob error: %s", e)

        # 3) Bundled model recorded in verifier metadata
        meta = self._load_verifier_metadata()
        base = meta.get("base_model")
        if base:
            p = self._bundled_model_path(base)
            if p is not None:
                return p

        # 4) Bundled model best matching the wake phrase
        return self._select_bundled_model(self._wake_phrase)

    def _bundled_models(self) -> Dict[str, Path]:
        """Return {model_stem: Path} for all bundled openWakeWord models."""
        try:
            import openwakeword as _oww
        except ImportError:
            return {}
        d = Path(_oww.__file__).parent / "resources" / "models"
        out: Dict[str, Path] = {}
        if not d.exists():
            return out
        for f in d.glob("*.onnx"):
            if f.name in NON_WAKE_FILES:
                continue
            out[f.stem] = f
        return out

    def _bundled_model_path(self, model_stem: str) -> Optional[Path]:
        models = self._bundled_models()
        return models.get(model_stem)

    def _select_bundled_model(self, phrase: str) -> Optional[Path]:
        """Select the bundled model whose name best matches the wake phrase.

        Never hardcodes a specific bundled model. If the phrase has low
        similarity to all bundled models, DEFAULT_BUNDLED_MODEL is used as a
        generic base for the custom verifier.
        """
        models = self._bundled_models()
        if not models:
            return None
        phrase_lower = phrase.lower().strip()
        best_name = DEFAULT_BUNDLED_MODEL
        best_ratio = -1.0
        for name in models:
            pretty = name.replace("_", " ")
            ratio = difflib.SequenceMatcher(None, phrase_lower, pretty).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = name
        if best_ratio < 0.3:
            best_name = DEFAULT_BUNDLED_MODEL
        logger.info("[WAKE] Selected bundled model '%s' for phrase '%s' "
                    "(match ratio=%.2f)", best_name, phrase, best_ratio)
        return models.get(best_name)

    # ── Verifier resolution ────────────────────────────────────────

    def _load_verifier_metadata(self) -> dict:
        meta_path = settings.MODELS_WAKE_DIR / "metadata.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("[WAKE] Verifier metadata read failed: %s", e)
            return {}

    def _resolve_verifier(self, base_model_name: str) -> Optional[Path]:
        """Resolve a verifier valid for the given base model, or None."""
        meta = self._load_verifier_metadata()
        # Only attach when the verifier was trained on this base model
        if meta.get("base_model") and meta["base_model"] != base_model_name:
            logger.info("[WAKE] Verifier trained on '%s', current base model is "
                        "'%s' — verifier not attached", meta["base_model"], base_model_name)
            return None

        named = meta.get("verifier_path")
        candidates: List[Path] = []
        if named:
            candidates.append(settings.MODELS_WAKE_DIR / named)
        candidates.append(settings.MODELS_WAKE_DIR / "verifier.pkl")
        candidates.append(settings.MODELS_WAKE_DIR / "verifier.joblib")
        try:
            candidates += sorted(settings.MODELS_WAKE_DIR.glob("verifier.*"))
        except Exception:
            pass

        seen = set()
        for c in candidates:
            key = str(c)
            if key in seen or not c.exists():
                continue
            seen.add(key)
            return c
        return None

    # ── Warmup ─────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """Prime openWakeWord's prediction buffer.

        openWakeWord zeroes all predictions while its per-class buffer holds
        fewer than 5 frames. Without warmup, the first real clip is always
        missed — this is a common cause of "wake word never fires".
        """
        if self._model is None:
            return
        silence = np.zeros(WARMUP_FRAME_SAMPLES, dtype=np.float32)
        for _ in range(WARMUP_FRAMES + 1):
            try:
                self._model.predict(silence)
            except Exception as e:
                logger.debug("[WAKE] Warmup predict error: %s", e)
                break


# Global singleton
wake_model_manager = WakeModelManager()