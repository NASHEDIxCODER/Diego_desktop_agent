"""
WhisperProvider — faster-whisper backend wrapped behind ASRProvider.

Consumes the SAME normalized float32 [-1,1] 16 kHz mono audio that
AudioManager produces. This provider does NOT re-normalize or re-sample;
it converts to the float32 array faster-whisper expects (already in
[-1,1]) and runs inference in an executor.

Backend selection mirrors the existing pipeline: cached backend first,
then CUDA float16, then CPU int8. The model is the same "base" model used
by StreamingSTT/CommandListener so behavior is preserved.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from voice.asr_provider import ASRProvider, get_cpu_ram_stats, get_gpu_stats

logger = logging.getLogger(__name__)

_BACKEND_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "whisper_backend.json"


class WhisperProvider(ASRProvider):
    name = "faster-whisper"

    def __init__(self, model_size: str = "base"):
        self._model = None
        self._ready = False
        self._device = "cuda"
        self._compute = "int8_float16"
        self._model_size = model_size
        self._lock = threading.Lock()
        # Streaming state (partial hypotheses)
        self._stream_buffer: list = []
        self._stream_text = ""
        # Metrics
        self._total_transcriptions = 0
        self._total_latency_s = 0.0
        self._errors = 0

    # ── Lifecycle ──────────────────────────────────────────
    def start(self) -> bool:
        if self._ready:
            return True
        try:
            from faster_whisper import WhisperModel
            import torch

            cached = self._load_cached_backend()
            candidates = []
            if cached:
                candidates.append(cached)
            if torch.cuda.is_available():
                candidates.append(("cuda", "float16"))
            candidates.append(("cpu", "int8"))

            for device, compute in candidates:
                try:
                    model = WhisperModel(self._model_size, device=device, compute_type=compute)
                    warmup = np.zeros(16000, dtype=np.float32)
                    segments, _ = model.transcribe(warmup, beam_size=1, without_timestamps=True)
                    list(segments)
                    self._model = model
                    self._ready = True
                    self._device = device
                    self._compute = compute
                    self._save_backend_cache(device, compute)
                    logger.info("[WHISPER-PROVIDER] loaded (device=%s compute=%s size=%s)",
                                device, compute, self._model_size)
                    return True
                except Exception as e:
                    logger.warning("[WHISPER-PROVIDER] %s/%s unusable: %s", device, compute, e)
            logger.error("[WHISPER-PROVIDER] no working backend")
            return False
        except Exception as e:
            logger.warning("[WHISPER-PROVIDER] unavailable: %s", e)
            return False

    def stop(self) -> None:
        self._model = None
        self._ready = False
        self._stream_buffer = []
        self._stream_text = ""

    # ── ASRProvider interface ──────────────────────────────
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: Optional[str] = None) -> str:
        """Transcribe a complete utterance. Audio is float32 [-1,1] mono."""
        if not self._ready or audio is None or len(audio) == 0:
            return ""
        try:
            a = np.asarray(audio, dtype=np.float32)
            if len(a) < sample_rate * 0.15:
                return ""
            t0 = time.time()
            segments, _ = self._model.transcribe(
                a,
                beam_size=5,
                language=language or "en",
                temperature=0.0,
                best_of=5,
                condition_on_previous_text=False,
                compression_ratio_threshold=None,
                no_speech_threshold=0.9,
                vad_filter=False,
                without_timestamps=True,
            )
            segs = list(segments)
            text = " ".join(s.text.strip() for s in segs).strip()
            self._record_latency(time.time() - t0)
            return text
        except Exception as e:
            self._errors += 1
            logger.debug("[WHISPER-PROVIDER] transcribe error: %s", e)
            return ""

    def stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000,
               language: Optional[str] = None) -> str:
        """Accumulate chunks and return the current partial transcript."""
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._stream_buffer.append(np.asarray(audio_chunk, dtype=np.float32))
        if not self._stream_buffer:
            return self._stream_text
        full = np.concatenate(self._stream_buffer)
        if len(full) < sample_rate * 0.15:
            return self._stream_text
        # Fast beam=1 for partials
        try:
            t0 = time.time()
            segments, _ = self._model.transcribe(
                full, beam_size=1, language=language or "en",
                temperature=0.0, best_of=1, condition_on_previous_text=False,
                compression_ratio_threshold=None, no_speech_threshold=0.9,
                vad_filter=False, without_timestamps=True,
            )
            segs = list(segments)
            self._stream_text = " ".join(s.text.strip() for s in segs).strip()
            self._record_latency(time.time() - t0)
        except Exception as e:
            self._errors += 1
            logger.debug("[WHISPER-PROVIDER] stream error: %s", e)
        return self._stream_text

    def reset(self) -> None:
        self._stream_buffer = []
        self._stream_text = ""

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "device": self._device,
            "compute_type": self._compute,
            "model_size": self._model_size,
            "gpu_available": self._ready and self._device == "cuda",
        }

    def metrics(self) -> Dict[str, Any]:
        avg_latency = (self._total_latency_s / self._total_transcriptions
                       if self._total_transcriptions else 0.0)
        gpu = get_gpu_stats()
        cpu = get_cpu_ram_stats()
        return {
            "model": self.name,
            "total_transcriptions": self._total_transcriptions,
            "avg_latency_s": avg_latency,
            "errors": self._errors,
            **gpu,
            **cpu,
        }

    # ── Helpers ────────────────────────────────────────────
    def _record_latency(self, seconds: float) -> None:
        self._total_transcriptions += 1
        self._total_latency_s += seconds

    @classmethod
    def _load_cached_backend(cls):
        try:
            if _BACKEND_CACHE.exists():
                data = json.loads(_BACKEND_CACHE.read_text(encoding="utf-8"))
                device = data.get("device")
                compute = data.get("compute_type")
                if device and compute:
                    return device, compute
        except Exception:
            pass
        return None

    @classmethod
    def _save_backend_cache(cls, device: str, compute: str) -> None:
        try:
            _BACKEND_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _BACKEND_CACHE.write_text(
                json.dumps({"device": device, "compute_type": compute, "saved_at": time.time()}),
                encoding="utf-8")
        except Exception as e:
            logger.debug("[WHISPER-PROVIDER] cache write failed: %s", e)


# Global singleton for the benchmark / pipeline
whisper_provider = WhisperProvider()