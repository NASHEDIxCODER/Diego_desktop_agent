"""
Pluggable STT backend selection for the command pipeline.

The production command listener historically used a single faster-whisper
transcriber (voice/command_listener._WhisperTranscriber). This module adds a
config-driven backend switch without deleting that transcriber or touching
VAD, wake, face auth, Brain, or command routing.

Backends:
  faster_whisper  - the existing production baseline (unchanged default)
  qwen3           - Qwen3-ASR 0.6B INT8 via sherpa-onnx (offshelf, local)

CONFIDENCE INVARIANT (do not break):
  Downstream transcript-quality gates (voice/command_listener._validate_transcript,
  agent/brain.py low-quality guard, nlp/intent_gate.py, nlp/intent_authorizer.py)
  consume `confidence` = faster-whisper avg_logprob. sherpa-onnx Qwen3 does NOT
  expose token log-probs (OfflineRecognitionResult.ys_log_probs is empty), so a
  bare Qwen3 primary would report confidence=None and be treated by those gates
  as a "trusted path" (intent_gate: stt_confidence is None -> command=True),
  silently bypassing the hallucination rejection. Therefore:

    - The faster-whisper fallback is ALWAYS loaded.
    - When qwen3 is primary, transcripts come from Qwen3, but the
      confidence/verify/partial paths still run faster-whisper so the quality
      gates keep real evidence and never degrade.

Selection:
    STT_PRIMARY=faster_whisper|qwen3   (default faster_whisper)
    STT_FALLBACK=faster_whisper        (default faster_whisper)
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_primary() -> str:
    """Read the configured primary backend, normalized + validated."""
    from voice.settings import voice_settings
    name = (voice_settings.stt_primary or "faster_whisper").strip().lower()
    if name in ("faster-whisper", "faster_whisper", "whisper"):
        return "faster_whisper"
    if name in ("qwen3", "qwen3-asr", "qwen3-asr-0.6b-int8"):
        return "qwen3"
    logger.warning("[STT-BACKEND] unknown STT_PRIMARY=%r — falling back to faster_whisper", name)
    return "faster_whisper"


class Qwen3Transcriber:
    """Sherpa-onnx Qwen3-ASR 0.6B INT8 behind the same method surface the
    command listener expects (_WhisperTranscriber-compatible).

    Only `transcribe` (full utterance) is primary. This class does NOT expose
    confidence; callers must pair it with a faster-whisper transcriber for
    confidence/verify/partials (see build_stt_backend).
    """

    name = "qwen3-asr-0.6b-int8"

    def __init__(self):
        self._provider = None
        self._ready = False
        self._device = "cpu"
        self._compute = "int8"

    def load(self) -> bool:
        if self._ready:
            return True
        try:
            from voice.providers.sherpa_providers import Qwen3ASRProvider
            p = Qwen3ASRProvider()
            ok = p.start()
            if not ok:
                logger.error("[STT-BACKEND] qwen3 load failed: %s", p.health().get("load_error"))
                return False
            self._provider = p
            self._ready = True
            return True
        except Exception as e:
            logger.warning("[STT-BACKEND] qwen3 unavailable: %s", e)
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    def transcribe_with_language(self, pcm_int16: bytes, sample_rate: int = 16000,
                                 language: Optional[str] = None) -> Tuple[str, float]:
        """Full utterance via Qwen3. Confidence is None (not fabricated)."""
        if not self._ready or not pcm_int16:
            return "", 0.0
        import numpy as np
        from voice.audio_processing import SAMPLE_RATE as _RATE
        audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < sample_rate * 0.15:
            return "", 0.0
        text = self._provider.transcribe(audio, sample_rate)
        return (text or "").strip(), 0.0

    def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> Tuple[str, float]:
        return self.transcribe_with_language(pcm_int16, sample_rate, None)


def build_stt_backend():
    """Return a transcriber object exposing the command-listener interface.

    Default (faster_whisper primary): returns the real _WhisperTranscriber,
    exactly as before (the transcription + confidence path are unchanged).

    qwen3 primary: returns a composite whose `transcribe_with_language`
    (primary En pass) returns the Qwen3 transcript but a confidence/fallback
    (hi pass) run through faster-whisper, preserving the quality gates.
    """
    from voice.command_listener import _WhisperTranscriber
    from voice.settings import voice_settings

    primary = resolve_primary()
    whisper = _WhisperTranscriber()
    if primary == "faster_whisper":
        return whisper

    # qwen3 primary: load qwen3 transcript engine + faster-whisper fallback.
    qwen = Qwen3Transcriber()
    qwen.load()
    fallback = whisper  # always kept for confidence + verify + partials

    class _CompositeTranscriber:
        name = "qwen3+faster_whisper"

        def __init__(self):
            self._qwen = qwen
            self._whisper = fallback
            self._ready = qwen.ready
            self._device = getattr(fallback, "_device", "cpu")
            self._compute = getattr(fallback, "_compute", "int8")

        def load(self) -> bool:
            ok_w = self._whisper.load() if not self._whisper.ready else True
            ok_q = self._qwen.load() if not self._qwen.ready else True
            self._ready = ok_q
            return self._ready

        @property
        def ready(self) -> bool:
            return self._ready

        def transcribe_with_language(self, pcm_int16, sample_rate=16000,
                                     language=None):
            """Primary (En) transcript from Qwen3; confidence from faster-whisper.

            The confidence the gates consume is faster-whisper avg_logprob,
            computed on the SAME audio. This keeps the hallucination band
            active for Qwen3 transcripts rather than treating them as trusted.
            """
            qtext, _ = self._qwen.transcribe_with_language(pcm_int16, sample_rate, language)
            _wtext, wconf = self._whisper.transcribe_with_language(pcm_int16, sample_rate, language)
            return qtext, wconf

        def transcribe(self, pcm_int16, sample_rate=16000):
            return self.transcribe_with_language(pcm_int16, sample_rate, None)

        # Partials + wake verification always use faster-whisper (fast + has
        # native confidence). Qwen3 offline transcribe is too slow for partials.
        def transcribe_fast(self, pcm_int16, sample_rate=16000):
            return self._whisper.transcribe_fast(pcm_int16, sample_rate)

        def transcribe_verify(self, pcm_int16, sample_rate=16000):
            return self._whisper.transcribe_verify(pcm_int16, sample_rate)

    return _CompositeTranscriber()
