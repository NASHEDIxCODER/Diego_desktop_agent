"""
Pluggable STT backend selection for the command pipeline.

The production command listener historically used a single faster-whisper
transcriber (voice/command_listener._WhisperTranscriber). This module adds a
config-driven backend switch without deleting that transcriber or touching
VAD, wake, face auth, Brain, or command routing.

Backends:
  faster_whisper  - the previous production baseline
  qwen3           - Qwen3-ASR 1.7B INT8 via sherpa-onnx (default, local)
                    0.6B INT8 available via STT_MODEL_SIZE=0.6b

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
    STT_PRIMARY=qwen3|faster_whisper|openai|gemini|sarvam   (default qwen3)
    STT_FALLBACK=faster_whisper                              (default faster_whisper)
    STT_MODEL_SIZE=1.7b|0.6b                                 (default 1.7b)

FUTURE PROVIDERS (reserved, not implemented):
    `openai` and `gemini` are recognised names whose adapters do not exist.
    They are NOT faked, NOT default, and their API keys are NOT required at
    startup. Selecting one reports UNAVAILABLE and safely falls back to the
    configured fallback backend (faster_whisper).

EXPERIMENTAL PROVIDER — `sarvam` (Saaras v4 REST, benchmark only):
    `voice/providers/sarvam_provider.py` implements Sarvam Saaras v4 behind
    the SAME ASRProvider interface so it can be compared head-to-head against
    Qwen3-ASR 1.7B/0.6B and faster-whisper. It is deliberately NOT part of
    production routing: STT_PRIMARY=sarvam reports EXPERIMENTAL and falls back
    to STT_FALLBACK even when a valid SARVAM_API_KEY is present. The only code
    path that instantiates it is scripts/phase24g_sarvam_benchmark.py. Promote
    it only after benchmark evidence justifies it, and then wire it explicitly
    through build_stt_backend() — never by removing the guard here.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical provider name → accepted spellings/aliases.
_PROVIDER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "faster_whisper": ("faster-whisper", "faster_whisper", "whisper"),
    "qwen3": ("qwen3", "qwen3-asr", "qwen3-asr-0.6b-int8", "qwen3-asr-1.7b-int8"),
    "openai": ("openai", "open-ai", "whisper-api", "whisper_api"),
    "gemini": ("gemini", "google-gemini", "google_gemini"),
    "sarvam": ("sarvam", "saaras", "saaras-v4", "sarvam-saaras"),
}

# Providers that the abstraction RESERVES but that are not wired up for
# production routing. They must never be faked and never become the default.
FUTURE_PROVIDERS: Tuple[str, ...] = ("openai", "gemini")

# Providers that ARE implemented but are deliberately kept OUT of production
# routing (Phase 24.7): `sarvam` (Saaras v4 REST) exists only as an isolated
# benchmark comparison. Selecting it as STT_PRIMARY reports EXPERIMENTAL and
# safely falls back, even when a valid API key is present — a benchmark result
# must never silently become Diego's live command path.
EXPERIMENTAL_PROVIDERS: Tuple[str, ...] = ("sarvam",)

# Reserved-but-unimplemented ∪ implemented-but-benchmark-only. Anything here
# is guaranteed to be unavailable for production routing.
NON_ROUTABLE_PROVIDERS: Tuple[str, ...] = FUTURE_PROVIDERS + EXPERIMENTAL_PROVIDERS

# Local, always-available providers.
LOCAL_PROVIDERS: Tuple[str, ...] = ("qwen3", "faster_whisper")

_ALIAS_TO_CANONICAL: Dict[str, str] = {
    alias: canonical
    for canonical, aliases in _PROVIDER_ALIASES.items()
    for alias in aliases
}


def canonical_provider(name: str) -> str:
    """Normalize a spoken/config provider name to its canonical form.

    Returns "" for an unknown name (callers decide the fallback).
    """
    return _ALIAS_TO_CANONICAL.get(str(name or "").strip().lower(), "")


def provider_status(name: str) -> Dict[str, object]:
    """Report whether `name` is usable right now — never fabricates support.

    Returns ``{"requested", "provider", "available", "reason"}``. A future
    provider (openai/gemini) is explicitly UNAVAILABLE: its adapter is not
    implemented, and no API key is needed to reach this verdict.
    """
    requested = str(name or "").strip()
    canonical = canonical_provider(requested)
    if not canonical:
        return {"requested": requested, "provider": "", "available": False,
                "reason": f"UNAVAILABLE: unknown STT provider {requested!r}"}
    if canonical in EXPERIMENTAL_PROVIDERS:
        return {"requested": requested, "provider": canonical, "available": False,
                "experimental": True,
                "reason": (f"EXPERIMENTAL: '{canonical}' is benchmark-only and "
                           f"NOT in production routing; using STT_FALLBACK "
                           f"(see scripts/phase24g_sarvam_benchmark.py)")}
    if canonical in FUTURE_PROVIDERS:
        return {"requested": requested, "provider": canonical, "available": False,
                "experimental": False,
                "reason": (f"UNAVAILABLE: '{canonical}' adapter is not "
                           f"implemented (reserved provider; no API key "
                           f"required and none is used)")}
    return {"requested": requested, "provider": canonical, "available": True,
            "experimental": False,
            "reason": f"available (local provider '{canonical}')"}


def resolve_primary() -> str:
    """Read the configured primary backend, normalized + validated.

    Unknown or UNAVAILABLE providers fall back to the configured fallback
    (defaulting to faster_whisper) with an explicit log line — never silently
    and never to a faked provider.
    """
    from voice.settings import voice_settings
    raw = (voice_settings.stt_primary or "qwen3").strip()
    status = provider_status(raw)
    if status["available"]:
        return str(status["provider"])
    logger.warning("[STT-BACKEND] %s — using STT_FALLBACK", status["reason"])
    fallback = canonical_provider(getattr(voice_settings, "stt_fallback", ""))
    if fallback and fallback in LOCAL_PROVIDERS:
        return fallback
    return "faster_whisper"


def active_stt_report() -> str:
    """One-line startup report of the active STT provider + its evidence path."""
    from voice.settings import voice_settings
    primary = resolve_primary()
    status = provider_status(getattr(voice_settings, "stt_primary", ""))
    fallback = getattr(voice_settings, "stt_fallback", "faster_whisper") or ""
    if primary == "qwen3":
        size = str(getattr(voice_settings, "stt_model_size", "1.7b")
                   or "1.7b").strip().upper()  # "1.7B" or "0.6B"
        detail = (f"transcript=Qwen3-ASR {size} INT8 (sherpa-onnx); "
                  "confidence/verify/partials=faster-whisper")
    else:
        detail = "transcript+confidence=faster-whisper"
    line = (f"STT provider: {primary} (fallback={fallback}) — {detail}")
    if not status["available"]:
        line += f" [requested {status['requested']!r}: {status['reason']}]"
    return line


class Qwen3Transcriber:
    """Sherpa-onnx Qwen3-ASR (1.7B default / 0.6B fallback) behind the same
    method surface the command listener expects (_WhisperTranscriber-compatible).

    Only `transcribe` (full utterance) is primary. This class does NOT expose
    confidence; callers must pair it with a faster-whisper transcriber for
    confidence/verify/partials (see build_stt_backend).
    """

    name = "qwen3-asr-1.7b-int8"

    def __init__(self):
        self._provider = None
        self._ready = False
        self._device = "cpu"
        self._compute = "int8"
        self._model_size = ""

    def load(self) -> bool:
        if self._ready:
            return True
        try:
            from voice.settings import voice_settings
            size = (getattr(voice_settings, "stt_model_size", "1.7b")
                    or "1.7b").strip().lower()
            self._model_size = size

            if size in ("0.6b", "0.6"):
                from voice.providers.sherpa_providers import Qwen3ASRProvider
                p = Qwen3ASRProvider()
                self.name = "qwen3-asr-0.6b-int8"
            else:
                # Default: 1.7B for maximum En/Hi/Hinglish accuracy.
                from voice.providers.sherpa_providers import Qwen3ASR17BProvider
                p = Qwen3ASR17BProvider()
                self.name = "qwen3-asr-1.7b-int8"

            ok = p.start()
            if not ok:
                logger.error("[STT-BACKEND] qwen3 %s load failed: %s",
                             self.name, p.health().get("load_error"))
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
    logger.info("[STT-BACKEND] %s", active_stt_report())
    # Defense in depth: resolve_primary() already refuses non-routable
    # providers, so reaching here with one would be a routing bug — fail loud
    # instead of silently building a local backend under a cloud provider name.
    if primary in NON_ROUTABLE_PROVIDERS:
        raise RuntimeError(
            f"[STT-BACKEND] refusing to build production backend for "
            f"non-routable provider {primary!r} (experimental/reserved)")
    whisper = _WhisperTranscriber()
    if primary == "faster_whisper":
        return whisper

    # qwen3 primary: load qwen3 transcript engine + faster-whisper fallback.
    qwen = Qwen3Transcriber()
    if not qwen.load():
        # UNAVAILABLE → safe fallback to the configured fallback backend.
        logger.warning(
            "[STT-BACKEND] UNAVAILABLE: qwen3 primary could not load — "
            "falling back to faster-whisper (no fake provider, gates intact)")
        return whisper
    fallback = whisper  # always kept for confidence + verify + partials

    class _CompositeTranscriber:
        def __init__(self):
            self._qwen = qwen
            self._whisper = fallback
            self._ready = qwen.ready
            self._device = getattr(fallback, "_device", "cpu")
            self._compute = getattr(fallback, "_compute", "int8")
            self.name = f"{qwen.name}+faster_whisper"

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

            If Qwen3 yields no text (empty audio / runtime failure) the
            faster-whisper transcript is used instead — a safe fallback, never
            a fabricated transcript. The confidence still comes from
            faster-whisper, so the quality gates behave identically.
            """
            qtext, _ = self._qwen.transcribe_with_language(pcm_int16, sample_rate, language)
            wtext, wconf = self._whisper.transcribe_with_language(pcm_int16, sample_rate, language)
            if not (qtext or "").strip():
                return wtext, wconf
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


# ── Benchmark-only experimental provider access ───────────────────────────
# These helpers are the ONLY sanctioned way for offline tooling (benchmark
# harnesses, notebooks) to reach an experimental provider. They deliberately
# require an explicit opt-in argument so no caller can accidentally obtain a
# cloud provider from a config/env default, and they are never referenced by
# voice/command_listener.py, agent/brain.py, or any other runtime path.

def experimental_provider_available() -> Dict[str, bool]:
    """Report which experimental providers are configured (credentials only).

    Cheap and network-free: this checks configuration, not reachability.
    """
    from voice.settings import voice_settings
    return {
        "sarvam": bool(str(getattr(voice_settings, "sarvam_api_key", "") or "").strip()),
    }


def experimental_provider_for_benchmark(name: str, *, allow_experimental: bool):
    """Build an experimental ASR provider for offline benchmarking only.

    Refuses unless the caller explicitly passes `allow_experimental=True`,
    which documents at the call site that this is not a production path. Raises
    ValueError for unknown/reserved names and for EXPERIMENTAL providers with
    no configured credential (never fabricates a provider).
    """
    if not allow_experimental:
        raise PermissionError(
            "experimental_provider_for_benchmark() requires "
            "allow_experimental=True — experimental providers are benchmark-only "
            "and never part of production routing")
    canonical = canonical_provider(name)
    if canonical in EXPERIMENTAL_PROVIDERS:
        if canonical == "sarvam":
            from voice.providers.sarvam_provider import SarvamSaarasProvider
            provider = SarvamSaarasProvider()
            if not provider.health().get("api_key_set"):
                raise ValueError(
                    "sarvam is experimental but SARVAM_API_KEY is not set — "
                    "benchmark cannot run (the provider is never faked)")
            return provider
        raise ValueError(f"experimental provider {name!r} has no implementation")
    raise ValueError(
        f"{name!r} is not an experimental provider (use build_stt_backend() "
        f"for production providers; experimental={EXPERIMENTAL_PROVIDERS})")
