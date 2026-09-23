"""
Concrete sherpa-onnx ASR providers for Diego's benchmark.

Each provider wraps a specific pre-exported sherpa-onnx model behind the
ASRProvider interface. All models consume the SAME normalized float32
[-1,1] 16 kHz mono audio from AudioManager — no own VAD, no re-normalization.

Providers:
  - Qwen3ASR17BProvider     (Qwen3-ASR 1.7B INT8, offline, multilingual)
  - Qwen3ASRProvider        (Qwen3-ASR 0.6B INT8, offline, multilingual)
  - ParakeetProvider        (NVIDIA Parakeet CTC 1.1B INT8, offline, en)
  - ZipformerStreamProvider (streaming Zipformer EN INT8, TRUE streaming)
  - FireRedASRProvider      (FireRedASR2 CTC zh_en INT8, offline)
  - SenseVoiceProvider      (SenseVoiceSmall INT8, offline, multilingual)
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import sherpa_onnx

from voice.providers.sherpa_base import (
    SherpaOnnxProvider,
    StreamingSherpaOnnxProvider,
)

logger = logging.getLogger(__name__)


# ── Qwen3-ASR output format ───────────────────────────────────────────────
# The Qwen3-ASR export is a chat model, so the decoder can emit its
# chat-template preamble alongside the text:
#
#     "language English<asr_text>Hello, Diego."
#
# Only the part AFTER <asr_text> is the transcript; the prefix is model
# bookkeeping. This preamble is emitted NONDETERMINISTICALLY — int8
# multithreaded inference means an input differing by a single LSB (exactly
# what the float→int16→float round-trip in the command path produces) can
# flip the greedy decode between the two forms. Both forms are therefore
# reachable in production, and the preamble must never reach the intent
# parser ("language English<asr_text>turn on the lights" would not match any
# command).
_ASR_TEXT_MARKER = "<asr_text>"

# `language <Name><asr_text>`; the language name is free-form ("English",
# "Hindi", "Chinese", "None", ...), so capture non-greedily up to the marker.
_LANGUAGE_PREAMBLE_RE = re.compile(
    r"^\s*language\s+(?P<lang>[^<]*?)\s*" + re.escape(_ASR_TEXT_MARKER),
    re.IGNORECASE,
)

# Degenerate language values the model emits when it cannot decide.
_NO_LANGUAGE = frozenset({"", "none", "null", "unknown", "n/a", "auto", "undefined"})


def parse_qwen3_asr_output(raw: Optional[str]) -> Tuple[str, str]:
    """Split raw Qwen3-ASR output into ``(transcript, detected_language)``.

    Handles every form observed from the real 0.6B and 1.7B exports::

        'language English<asr_text>Hello, Diego.'  -> ('Hello, Diego.', 'English')
        'language Hindi<asr_text>बंद करो'            -> ('बंद करो', 'Hindi')
        '<asr_text>Hello'                          -> ('Hello', '')
        'language None<asr_text>'                  -> ('', '')
        'Hello, Diego.'                            -> ('Hello, Diego.', '')

    When no marker is present the text is returned UNCHANGED — a transcript is
    never discarded or altered on a guess. `detected_language` is returned as
    the model's own name ("English"), already normalised, or "" when absent.
    """
    text = (raw or "").strip()
    if not text:
        return "", ""
    if _ASR_TEXT_MARKER not in text:
        # No chat-template preamble: already a plain transcript.
        return text, ""

    # Everything after the FIRST marker is the transcript; the preamble (and
    # its optional `language <Name>` field) is before it.
    _preamble, _marker, transcript = text.partition(_ASR_TEXT_MARKER)
    match = _LANGUAGE_PREAMBLE_RE.match(text)
    language = (match.group("lang") or "").strip() if match else ""
    if language.lower() in _NO_LANGUAGE:
        language = ""
    return transcript.strip(), language


class _Qwen3ASROutputMixin:
    """Strips the Qwen3-ASR chat-template preamble off raw recognizer output.

    Mixed in BEFORE `SherpaOnnxProvider` so `_recognize_offline` wraps the
    shared sherpa-onnx recognizer call for both the 0.6B and 1.7B exports —
    identical transcript shape, only the model differs. Kept as a plain
    (non-ABC) mixin so instantiating it standalone never trips abstract
    checks; only the concrete provider subclasses are ever constructed.
    """

    _last_detected_language = ""

    def _recognize_offline(self, audio, sample_rate: int) -> str:  # noqa: ANN001,ANN202
        # noinspection PyProtectedMember — intended: this mixin wraps the base
        # SherpaOnnxProvider._recognize_offline via cooperative inheritance.
        raw = super()._recognize_offline(audio, sample_rate)  # type: ignore[misc]
        text, language = parse_qwen3_asr_output(raw)
        if language:
            self._last_detected_language = language
        return text


class Qwen3ASR17BProvider(_Qwen3ASROutputMixin, SherpaOnnxProvider):
    """Qwen3-ASR 1.7B INT8 via sherpa-onnx (offline, multilingual).

    Production default (Phase 24.7): larger model for better English / Hindi /
    Hinglish accuracy. Same architecture as the 0.6B variant but with 2.8x
    the parameters. ~4.4 GB on disk, ~3.4 GB peak RAM.

    DOWNLOAD TRAP (do not remove `decoder.int8.onnx.data`):
      In this export the decoder graph (`decoder.int8.onnx`, 5 MB) stores its
      weights OUTSIDE the graph, in the ONNX external-data sidecar
      `decoder.int8.onnx.data` (4.0 GB). onnxruntime cannot load the decoder
      without it, so the sidecar is a REQUIRED file — omitting it made the
      model fail to load and Diego silently degrade to faster-whisper. The
      0.6B export is self-contained and has no sidecar.
    """

    name = "qwen3-asr-1.7b-int8"
    repo_id = "solavr/sherpa-onnx-qwen3-asr-1.7B-int8"
    file_patterns = [
        "conv_frontend.onnx",
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        # ONNX external-data sidecar holding the decoder weights (4.0 GB).
        "decoder.int8.onnx.data",
        "tokenizer/*",
    ]

    def _build_recognizer(self):
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(self._model_dir / "conv_frontend.onnx"),
            encoder=str(self._model_dir / "encoder.int8.onnx"),
            decoder=str(self._model_dir / "decoder.int8.onnx"),
            tokenizer=str(self._model_dir / "tokenizer"),
            num_threads=self._num_threads,
            sample_rate=16000,
            feature_dim=128,
            decoding_method="greedy_search",
            debug=False,
            provider=self._provider,
            max_total_len=1024,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )


class Qwen3ASRProvider(_Qwen3ASROutputMixin, SherpaOnnxProvider):
    """Qwen3-ASR 0.6B INT8 via sherpa-onnx (offline, multilingual).

    Previous default (Phase 24.6). Kept as a lighter fallback for low-RAM
    devices. Select via STT_MODEL_SIZE=0.6b. Shares the chat-template
    preamble parsing with the 1.7B export (see `parse_qwen3_asr_output`).
    """

    name = "qwen3-asr-0.6b-int8"
    repo_id = "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    file_patterns = [
        "conv_frontend.onnx",
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "tokenizer/*",
    ]

    def _build_recognizer(self):
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(self._model_dir / "conv_frontend.onnx"),
            encoder=str(self._model_dir / "encoder.int8.onnx"),
            decoder=str(self._model_dir / "decoder.int8.onnx"),
            tokenizer=str(self._model_dir / "tokenizer"),
            num_threads=self._num_threads,
            sample_rate=16000,
            feature_dim=128,
            decoding_method="greedy_search",
            debug=False,
            provider=self._provider,
            max_total_len=1024,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )


class ParakeetProvider(SherpaOnnxProvider):
    """NVIDIA Parakeet CTC 1.1B INT8 via sherpa-onnx (offline, English).

    NOTE: The RNNT (transducer) 1.1B multilingual export is not available
    in sherpa-onnx; this is the closest 1.1B Parakeet export (CTC, English).
    """

    name = "parakeet-ctc-1.1b-int8"
    repo_id = "runanywhere/sherpa-onnx-nemo-parakeet-ctc-1.1b-int8"
    file_patterns = ["model.int8.onnx", "tokens.txt"]

    def _build_recognizer(self):
        return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(self._model_dir / "model.int8.onnx"),
            tokens=str(self._model_dir / "tokens.txt"),
            num_threads=self._num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider=self._provider,
        )


class ZipformerStreamProvider(StreamingSherpaOnnxProvider):
    """Streaming Zipformer EN INT8 via sherpa-onnx (TRUE streaming)."""

    name = "zipformer-streaming-en-int8"
    repo_id = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
    file_patterns = [
        "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "tokens.txt",
    ]

    def _build_recognizer(self):
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self._model_dir / "tokens.txt"),
            encoder=str(self._model_dir / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
            decoder=str(self._model_dir / "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
            joiner=str(self._model_dir / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
            num_threads=self._num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,  # Diego's unified VAD owns endpointing
            model_type="zipformer",
            provider=self._provider,
        )
        self._stream = recognizer.create_stream()
        return recognizer


class FireRedASRProvider(SherpaOnnxProvider):
    """FireRedASR2 CTC zh_en INT8 via sherpa-onnx (offline, zh+en)."""

    name = "firered-asr2-ctc-zh-en-int8"
    repo_id = "csukuangfj2/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"
    file_patterns = ["model.int8.onnx", "tokens.txt"]

    def _build_recognizer(self):
        return sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
            model=str(self._model_dir / "model.int8.onnx"),
            tokens=str(self._model_dir / "tokens.txt"),
            num_threads=self._num_threads,
            decoding_method="greedy_search",
            debug=False,
            provider=self._provider,
        )


class SenseVoiceProvider(SherpaOnnxProvider):
    """SenseVoiceSmall INT8 via sherpa-onnx (offline, zh/en/ja/ko/yue)."""

    name = "sensevoice-int8"
    repo_id = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    file_patterns = ["model.int8.onnx", "tokens.txt"]

    def _build_recognizer(self):
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            tokens=str(self._model_dir / "tokens.txt"),
            model=str(self._model_dir / "model.int8.onnx"),
            num_threads=self._num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider=self._provider,
            language="auto",
            use_itn=True,
        )


# Registry of all sherpa-onnx candidates for the benchmark.
ALL_SHERPA_PROVIDERS = [
    Qwen3ASR17BProvider,
    Qwen3ASRProvider,
    ParakeetProvider,
    ZipformerStreamProvider,
    FireRedASRProvider,
    SenseVoiceProvider,
]