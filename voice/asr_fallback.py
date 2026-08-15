"""
ASR Fallback — Nemotron → Whisper → "I didn't catch that".

Implements the required fallback chain without touching the existing
command/conversation architecture:

    NemotronProvider
        ↓ (unavailable / error / timeout)
    WhisperProvider
        ↓ (both fail)
    "I didn't catch that."

Never silently fails. The fallback is a thin wrapper around ASRProvider
instances and does NOT duplicate VAD, AudioManager, or wake-word logic.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from voice.asr_provider import ASRProvider

logger = logging.getLogger(__name__)

FALLBACK_PHRASE = "I didn't catch that."


class ASRFallback:
    """Runs providers in priority order and returns the first usable result."""

    def __init__(self, providers: List[ASRProvider], timeout_s: float = 30.0):
        self._providers = providers
        self._timeout_s = timeout_s
        self._last_used: Optional[str] = None

    def start_all(self) -> None:
        """Start every provider (best-effort)."""
        for p in self._providers:
            try:
                ok = p.start()
                logger.info("[ASR-FALLBACK] %s start=%s", p.name, ok)
            except Exception as e:
                logger.warning("[ASR-FALLBACK] %s start failed: %s", p.name, e)

    def stop_all(self) -> None:
        for p in self._providers:
            try:
                p.stop()
            except Exception:
                pass

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: Optional[str] = None) -> str:
        """Transcribe using the first provider that succeeds.

        A provider "succeeds" when it is ready AND returns a non-empty
        transcript within the timeout. On total failure the fallback
        phrase is returned so the caller never silently drops a turn.
        """
        for provider in self._providers:
            if not provider.health().get("ready", False):
                logger.info("[ASR-FALLBACK] %s not ready — skipping", provider.name)
                continue
            try:
                t0 = time.time()
                text = provider.transcribe(audio, sample_rate, language)
                elapsed = time.time() - t0
                if elapsed > self._timeout_s:
                    logger.warning("[ASR-FALLBACK] %s timed out (%.1fs)", provider.name, elapsed)
                    continue
                if text and text.strip():
                    self._last_used = provider.name
                    logger.info("[ASR-FALLBACK] %s produced: %r", provider.name, text)
                    return text.strip()
                logger.info("[ASR-FALLBACK] %s returned empty — trying next", provider.name)
            except Exception as e:
                logger.warning("[ASR-FALLBACK] %s error: %s", provider.name, e)

        logger.warning("[ASR-FALLBACK] all providers failed — returning fallback phrase")
        return FALLBACK_PHRASE

    def health(self) -> Dict[str, Any]:
        return {
            "last_used": self._last_used,
            "providers": {p.name: p.health() for p in self._providers},
        }


# Default chain: Nemotron first, then Whisper.
def build_default_fallback() -> ASRFallback:
    from voice.providers.nemotron_provider import NemotronProvider
    from voice.providers.whisper_provider import WhisperProvider
    return ASRFallback([NemotronProvider(), WhisperProvider()])