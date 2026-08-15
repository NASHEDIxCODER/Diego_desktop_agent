"""
Concrete sherpa-onnx ASR providers for Leo's benchmark.

Each provider wraps a specific pre-exported sherpa-onnx model behind the
ASRProvider interface. All models consume the SAME normalized float32
[-1,1] 16 kHz mono audio from AudioManager — no own VAD, no re-normalization.

Providers:
  - Qwen3ASRProvider        (Qwen3-ASR 0.6B INT8, offline, multilingual)
  - ParakeetProvider        (NVIDIA Parakeet CTC 1.1B INT8, offline, en)
  - ZipformerStreamProvider (streaming Zipformer EN INT8, TRUE streaming)
  - FireRedASRProvider      (FireRedASR2 CTC zh_en INT8, offline)
  - SenseVoiceProvider      (SenseVoiceSmall INT8, offline, multilingual)
"""

from __future__ import annotations

import logging

import sherpa_onnx

from voice.providers.sherpa_base import (
    SherpaOnnxProvider,
    StreamingSherpaOnnxProvider,
)

logger = logging.getLogger(__name__)


class Qwen3ASRProvider(SherpaOnnxProvider):
    """Qwen3-ASR 0.6B INT8 via sherpa-onnx (offline, multilingual)."""

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
            enable_endpoint_detection=False,  # Leo's unified VAD owns endpointing
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
    Qwen3ASRProvider,
    ParakeetProvider,
    ZipformerStreamProvider,
    FireRedASRProvider,
    SenseVoiceProvider,
]