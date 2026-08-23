"""
ASRProvider — Clean abstraction over speech recognition backends.

Diego's command/conversation ASR is being evaluated against multiple
backends (faster-whisper, NVIDIA Nemotron/Parakeet). This module defines
the single interface every backend must implement so the benchmark and the
command pipeline can treat them interchangeably.

DESIGN RULES (do not violate):
  - Providers consume the SAME normalized audio (float32 [-1,1] 16 kHz mono)
    that AudioManager already produces. No provider may re-normalize,
    re-sample, or apply its own gain/VAD.
  - Providers NEVER implement their own VAD. Endpointing is owned by the
    unified VAD (voice/vad.py) and the command listener.
  - Providers NEVER invoke an LLM. ASR is a dedicated speech model.
  - Providers MUST be non-blocking from the caller's perspective: heavy
    inference runs in an executor, never on the event loop.
  - Providers MUST degrade gracefully and report health()/metrics().

The old StreamingSTT/CommandListener Whisper transcriber logic is NOT
deleted; WhisperProvider wraps the same faster-whisper model so behavior
is preserved while the interface becomes swappable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Metric record for a single transcription ─────────────────
@dataclass
class ASRMetrics:
    """One record of measured ASR performance for a single utterance."""
    model: str = ""
    language: str = ""
    transcript: str = ""
    expected: str = ""
    wer: float = 0.0
    command_accuracy: bool = False
    first_word_accuracy: bool = False
    time_to_first_token_ms: float = 0.0
    finalization_latency_ms: float = 0.0
    real_time_factor: float = 0.0
    cpu_percent: float = 0.0
    gpu_vram_mb: float = 0.0
    gpu_util_percent: float = 0.0
    ram_mb: float = 0.0
    dropped_chunks: int = 0
    hallucinations: int = 0
    audio_duration_s: float = 0.0
    ok: bool = True
    error: str = ""


# ── Abstract interface ────────────────────────────────────────
class ASRProvider(ABC):
    """Abstract base class for all speech recognition backends."""

    name: str = "asr"

    @abstractmethod
    def start(self) -> bool:
        """Load the model and prepare for inference. Returns True on success."""

    @abstractmethod
    def stop(self) -> None:
        """Release the model and free resources."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: Optional[str] = None) -> str:
        """Transcribe a complete utterance (float32 [-1,1] mono)."""

    @abstractmethod
    def stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000,
               language: Optional[str] = None) -> str:
        """Consume a streaming chunk and return the current partial transcript."""

    @abstractmethod
    def reset(self) -> None:
        """Reset streaming state between turns (clear partial hypotheses)."""

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return health/readiness information."""

    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        """Return aggregate runtime metrics for this provider."""


# ── Shared helpers ─────────────────────────────────────────────
import re as _re


def normalize_transcript(text: str) -> str:
    """Lowercase and strip punctuation for fair comparison."""
    if not text:
        return ""
    t = text.lower().strip()
    t = _re.sub(r"[^\w\s]", "", t)
    t = _re.sub(r"\s+", " ", t).strip()
    return t


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate (0.0 = perfect) using Levenshtein distance.

    Punctuation and case are ignored so a trailing period or capitalization
    does not count as a word error.
    """
    ref = normalize_transcript(reference).split()
    hyp = normalize_transcript(hypothesis).split()
    if not ref:
        return float(len(hyp))
    # Levenshtein distance over words
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
    return d[len(ref)][len(hyp)] / len(ref)


def first_word_accuracy(reference: str, hypothesis: str) -> bool:
    """True if the first word matches (case/punctuation-insensitive)."""
    ref = normalize_transcript(reference).split()
    hyp = normalize_transcript(hypothesis).split()
    if not ref or not hyp:
        return False
    return ref[0] == hyp[0]


def get_gpu_stats() -> Dict[str, float]:
    """Best-effort GPU VRAM/utilization read via nvidia-smi."""
    out = {"gpu_vram_mb": 0.0, "gpu_util_percent": 0.0}
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(",")
            if len(parts) >= 2:
                out["gpu_vram_mb"] = float(parts[0].strip())
                out["gpu_util_percent"] = float(parts[1].strip())
    except Exception:
        pass
    return out


def get_cpu_ram_stats() -> Dict[str, float]:
    """Best-effort process CPU/RAM read via psutil."""
    out = {"cpu_percent": 0.0, "ram_mb": 0.0}
    try:
        import psutil
        p = psutil.Process()
        out["cpu_percent"] = p.cpu_percent(interval=None)
        out["ram_mb"] = p.memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    return out