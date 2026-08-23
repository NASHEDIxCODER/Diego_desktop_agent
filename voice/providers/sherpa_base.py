"""
SherpaOnnxProvider — shared base for all sherpa-onnx ASR backends.

Every sherpa-onnx model (Qwen3-ASR, Parakeet RNNT, streaming Zipformer,
FireRedASR, SenseVoice) is wrapped behind the SAME ASRProvider interface
so the benchmark and Diego's command pipeline treat them interchangeably.

DESIGN RULES (identical to the rest of the ASR stack):
  - Providers consume the SAME normalized float32 [-1,1] 16 kHz mono audio
    that AudioManager produces. No re-normalization, no re-sampling, no
    own gain/VAD.
  - Providers NEVER implement their own VAD. Endpointing is owned by the
    unified VAD (voice/vad.py) and the command listener.
  - Providers NEVER invoke an LLM. ASR is a dedicated speech model.
  - Heavy inference runs in an executor; providers degrade gracefully and
    report health()/metrics().

Model files are downloaded ONCE via huggingface_hub into a local cache
(data/asr_models/<name>) and reused across runs.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from voice.asr_provider import ASRProvider, get_cpu_ram_stats, get_gpu_stats

logger = logging.getLogger(__name__)

# Local model cache root. Model files are large; keep them out of the repo.
MODEL_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "asr_models"


def _download_repo(repo_id: str, allow_patterns: List[str], dest: Path) -> bool:
    """Download required files from a HuggingFace repo into `dest`.

    Returns True when every requested pattern is present locally.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        logger.warning("[SHERPA-BASE] download failed for %s: %s", repo_id, e)
        return False
    # Verify every pattern resolved to at least one file.
    for pat in allow_patterns:
        matches = list(dest.glob(pat.lstrip("/")))
        if not matches:
            logger.warning("[SHERPA-BASE] missing %s in %s", pat, dest)
            return False
    return True


class SherpaOnnxProvider(ASRProvider):
    """Base class for sherpa-onnx offline/streaming ASR backends.

    Subclasses set:
      - name (provider id)
      - repo_id (HuggingFace repo)
      - file_patterns (list of glob patterns to download)
      - _build_recognizer() -> the sherpa_onnx recognizer object
      - streaming (bool) — True when the recognizer is an OnlineRecognizer
        with a persistent stream, False when offline chunk transcription is
        used for partials.

    transcribe() always runs full-utterance recognition.
    stream() feeds chunks and returns the current partial transcript.
    """

    name = "sherpa-onnx"
    repo_id = ""
    file_patterns: List[str] = []
    streaming = False

    def __init__(self, provider: str = "cpu", num_threads: int = 4):
        self._provider = provider
        self._num_threads = num_threads
        self._recognizer = None
        self._stream = None          # OnlineStream for streaming models
        self._ready = False
        self._load_error = ""
        self._lock = threading.Lock()
        # Streaming state
        self._stream_buffer: list = []
        self._stream_text = ""
        # Metrics
        self._total_transcriptions = 0
        self._total_latency_s = 0.0
        self._errors = 0
        self._model_dir = MODEL_ROOT / self.name

    # ── Lifecycle ──────────────────────────────────────────
    def start(self) -> bool:
        if self._ready:
            return True
        try:
            if not self._model_dir.exists() or not self._files_present():
                logger.info("[%s] downloading model from %s", self.name, self.repo_id)
                if not _download_repo(self.repo_id, self.file_patterns, self._model_dir):
                    self._load_error = f"download failed for {self.repo_id}"
                    return False
            self._recognizer = self._build_recognizer()
            self._ready = True
            self._load_error = ""
            logger.info("[%s] loaded (provider=%s threads=%d)",
                        self.name, self._provider, self._num_threads)
            return True
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            logger.warning("[%s] start failed: %s", self.name, e)
            return False

    def stop(self) -> None:
        self._recognizer = None
        self._stream = None
        self._ready = False
        self._stream_buffer = []
        self._stream_text = ""

    def _files_present(self) -> bool:
        return all(list(self._model_dir.glob(p.lstrip("/"))) for p in self.file_patterns)

    @abstractmethod
    def _build_recognizer(self):
        """Construct the sherpa_onnx recognizer. Called after download."""

    # ── ASRProvider interface ──────────────────────────────
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: Optional[str] = None) -> str:
        """Transcribe a complete utterance (float32 [-1,1] mono)."""
        if not self._ready or audio is None or len(audio) == 0:
            return ""
        a = np.asarray(audio, dtype=np.float32)
        if len(a) < sample_rate * 0.15:
            return ""
        try:
            t0 = time.time()
            text = self._recognize_offline(a, sample_rate)
            self._total_transcriptions += 1
            self._total_latency_s += time.time() - t0
            return text
        except Exception as e:
            self._errors += 1
            logger.debug("[%s] transcribe error: %s", self.name, e)
            return ""

    def _recognize_offline(self, audio: np.ndarray, sample_rate: int) -> str:
        """Run the offline recognizer and return the transcript."""
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(s)
        return (s.result.text or "").strip()

    def stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000,
               language: Optional[str] = None) -> str:
        """Consume a streaming chunk and return the current partial transcript."""
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._stream_buffer.append(np.asarray(audio_chunk, dtype=np.float32))
        if not self._stream_buffer:
            return self._stream_text
        full = np.concatenate(self._stream_buffer)
        if len(full) < sample_rate * 0.15:
            return self._stream_text
        try:
            t0 = time.time()
            self._stream_text = self._recognize_offline(full, sample_rate)
            self._total_transcriptions += 1
            self._total_latency_s += time.time() - t0
        except Exception as e:
            self._errors += 1
            logger.debug("[%s] stream error: %s", self.name, e)
        return self._stream_text

    def reset(self) -> None:
        self._stream_buffer = []
        self._stream_text = ""
        if self._stream is not None:
            try:
                self._recognizer.reset(self._stream)
            except Exception:
                pass

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "runtime": "sherpa-onnx",
            "provider": self._provider,
            "model_id": self.repo_id,
            "streaming": self.streaming,
            "load_error": self._load_error,
        }

    def metrics(self) -> Dict[str, Any]:
        avg_latency = (self._total_latency_s / self._total_transcriptions
                       if self._total_transcriptions else 0.0)
        gpu = get_gpu_stats()
        cpu = get_cpu_ram_stats()
        return {
            "model": self.name,
            "runtime": "sherpa-onnx",
            "provider": self._provider,
            "total_transcriptions": self._total_transcriptions,
            "avg_latency_s": avg_latency,
            "errors": self._errors,
            **gpu,
            **cpu,
        }


class StreamingSherpaOnnxProvider(SherpaOnnxProvider):
    """Base for TRUE streaming sherpa-onnx models (OnlineRecognizer).

    Uses a persistent OnlineStream so partial hypotheses are produced
    incrementally with the model's own streaming decoder — not by
    re-transcribing the whole buffer each time.
    """

    streaming = True

    def _recognize_offline(self, audio: np.ndarray, sample_rate: int) -> str:
        # For streaming models, offline transcription resets the stream and
        # feeds the whole buffer, then returns the final result.
        self._recognizer.reset(self._stream)
        self._stream.accept_waveform(sample_rate, audio)
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        return (self._recognizer.get_result(self._stream) or "").strip()

    def stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000,
               language: Optional[str] = None) -> str:
        """Feed a chunk into the persistent stream and return the partial."""
        if not self._ready:
            return ""
        try:
            if audio_chunk is not None and len(audio_chunk) > 0:
                a = np.asarray(audio_chunk, dtype=np.float32)
                self._stream.accept_waveform(sample_rate, a)
            t0 = time.time()
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            self._stream_text = (self._recognizer.get_result(self._stream) or "").strip()
            self._total_transcriptions += 1
            self._total_latency_s += time.time() - t0
        except Exception as e:
            self._errors += 1
            logger.debug("[%s] stream error: %s", self.name, e)
        return self._stream_text

    def reset(self) -> None:
        if self._stream is not None:
            try:
                self._recognizer.reset(self._stream)
            except Exception:
                pass
        self._stream_text = ""