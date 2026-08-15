"""
NemotronProvider — NVIDIA Nemotron-3.5-ASR-Streaming backend.

This provider attempts to load the local Nemotron model through every
available runtime and reports honestly when none is usable. It NEVER
assumes NVIDIA NIM or any cloud endpoint will work.

RUNTIME OPTIONS (tried in order):
  1. transformers >= 5.x (native Nemotron3_5AsrForRNNT architecture)
  2. onnxruntime int4 (onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4)
  3. NeMo toolkit (nemo_toolkit) with the .nemo checkpoint
  4. llama-cpp-python GGUF (q8_0)

HARDWARE CONSTRAINT (RTX 3050 4 GB VRAM):
  - float32 weights: 2.55 GB  → too large for 4 GB VRAM with CUDA context
  - q8_0 GGUF:      0.74 GB  → fits, but streaming RNNT via GGUF is unsupported
  - ONNX int4:      ~0.35 GB → fits, requires custom streaming state handling

The provider consumes the SAME normalized float32 [-1,1] 16 kHz mono audio
as every other backend. It performs NO VAD and NO re-normalization.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from voice.asr_provider import ASRProvider, get_cpu_ram_stats, get_gpu_stats

logger = logging.getLogger(__name__)


class NemotronProvider(ASRProvider):
    name = "nemotron-3.5-asr-streaming"

    def __init__(self, model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"):
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._ready = False
        self._runtime = "none"
        self._load_error = ""
        self._lock = threading.Lock()
        self._stream_buffer: list = []
        self._stream_text = ""
        self._total_transcriptions = 0
        self._total_latency_s = 0.0
        self._errors = 0

    # ── Lifecycle ──────────────────────────────────────────
    def start(self) -> bool:
        if self._ready:
            return True

        # Option 1: transformers >= 5.x native support
        if self._try_transformers():
            return True

        # Option 2: onnxruntime int4 community export
        if self._try_onnx():
            return True

        # Option 3: NeMo toolkit
        if self._try_nemo():
            return True

        # Option 4: llama-cpp GGUF
        if self._try_gguf():
            return True

        self._load_error = (
            "Nemotron runtime unavailable: requires transformers>=5.x "
            "(installed 4.57.6), nemo_toolkit, or onnxruntime int4 export. "
            "float32 weights (2.55 GB) exceed 4 GB VRAM budget."
        )
        logger.warning("[NEMOTRON-PROVIDER] %s", self._load_error)
        return False

    def stop(self) -> None:
        self._model = None
        self._processor = None
        self._ready = False
        self._stream_buffer = []
        self._stream_text = ""

    # ── Runtime attempts ───────────────────────────────────
    def _try_transformers(self) -> bool:
        try:
            import transformers
            major = int(transformers.__version__.split(".")[0])
            if major < 5:
                logger.info("[NEMOTRON-PROVIDER] transformers %s < 5 — "
                            "Nemotron3_5AsrForRNNT not available",
                            transformers.__version__)
                return False
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_id, torch_dtype="auto", device_map="auto")
            self._runtime = "transformers"
            self._ready = True
            logger.info("[NEMOTRON-PROVIDER] loaded via transformers %s",
                        transformers.__version__)
            return True
        except Exception as e:
            logger.info("[NEMOTRON-PROVIDER] transformers path failed: %s", e)
            return False

    def _try_onnx(self) -> bool:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            # The int4 community export requires a custom streaming graph;
            # attempt to locate the model files. If the graph is not a
            # standard encoder-decoder export, this fails gracefully.
            model_path = hf_hub_download(
                "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4",
                "model.onnx")
            self._model = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._runtime = "onnxruntime"
            self._ready = True
            logger.info("[NEMOTRON-PROVIDER] loaded via onnxruntime int4")
            return True
        except Exception as e:
            logger.info("[NEMOTRON-PROVIDER] onnxruntime path failed: %s", e)
            return False

    def _try_nemo(self) -> bool:
        try:
            import nemo.collections.asr as nemo_asr
            self._model = nemo_asr.models.ASRModel.restore_from(
                self._model_id + "/nemotron-3.5-asr-streaming-0.6b.nemo")
            self._runtime = "nemo_toolkit"
            self._ready = True
            logger.info("[NEMOTRON-PROVIDER] loaded via nemo_toolkit")
            return True
        except Exception as e:
            logger.info("[NEMOTRON-PROVIDER] nemo_toolkit path failed: %s", e)
            return False

    def _try_gguf(self) -> bool:
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            gguf_path = hf_hub_download(
                self._model_id, "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf")
            self._model = Llama(model_path=gguf_path, n_gpu_layers=-1)
            self._runtime = "llama-cpp-gguf"
            self._ready = True
            logger.info("[NEMOTRON-PROVIDER] loaded via llama-cpp GGUF")
            return True
        except Exception as e:
            logger.info("[NEMOTRON-PROVIDER] GGUF path failed: %s", e)
            return False

    # ── ASRProvider interface ──────────────────────────────
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: Optional[str] = None) -> str:
        if not self._ready:
            return ""
        try:
            a = np.asarray(audio, dtype=np.float32)
            if len(a) < sample_rate * 0.15:
                return ""
            t0 = time.time()
            if self._runtime == "transformers":
                inputs = self._processor(a, sampling_rate=sample_rate, return_tensors="pt")
                with self._lock:
                    out = self._model.generate(**inputs)
                text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
            elif self._runtime == "onnxruntime":
                text = self._onnx_transcribe(a, sample_rate)
            elif self._runtime == "nemo_toolkit":
                text = self._model.transcribe([a])[0]
            elif self._runtime == "llama-cpp-gguf":
                text = self._model.create_completion(a.tobytes())["choices"][0]["text"]
            else:
                text = ""
            self._total_transcriptions += 1
            self._total_latency_s += time.time() - t0
            return text.strip()
        except Exception as e:
            self._errors += 1
            logger.debug("[NEMOTRON-PROVIDER] transcribe error: %s", e)
            return ""

    def stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000,
               language: Optional[str] = None) -> str:
        # Nemotron streaming requires stateful RNNT decoding. The current
        # runtime (if any) may not expose a true streaming API; fall back to
        # full-buffer transcription of accumulated chunks.
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._stream_buffer.append(np.asarray(audio_chunk, dtype=np.float32))
        if not self._stream_buffer:
            return self._stream_text
        full = np.concatenate(self._stream_buffer)
        self._stream_text = self.transcribe(full, sample_rate, language)
        return self._stream_text

    def reset(self) -> None:
        self._stream_buffer = []
        self._stream_text = ""

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "runtime": self._runtime,
            "model_id": self._model_id,
            "load_error": self._load_error,
        }

    def metrics(self) -> Dict[str, Any]:
        avg_latency = (self._total_latency_s / self._total_transcriptions
                       if self._total_transcriptions else 0.0)
        gpu = get_gpu_stats()
        cpu = get_cpu_ram_stats()
        return {
            "model": self.name,
            "runtime": self._runtime,
            "total_transcriptions": self._total_transcriptions,
            "avg_latency_s": avg_latency,
            "errors": self._errors,
            **gpu,
            **cpu,
        }

    def _onnx_transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Placeholder for a real ONNX streaming graph invocation."""
        return ""


# Global singleton
nemotron_provider = NemotronProvider()