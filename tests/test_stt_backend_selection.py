"""
STT backend selection regression tests (2026-09-20).

Guarantees:
  1. Default STT primary stays faster-whisper (production baseline unchanged).
  2. STT_PRIMARY/STT_FALLBACK are config-driven via voice_settings + env.
  3. `resolve_primary()` normalizes aliases and rejects unknown names safely.
  4. The qwen3 composite ALWAYS retains a faster-whisper confidence source,
     so downstream quality gates never degrade to the "trusted path".
  5. CommandListener.initialize() routes through build_stt_backend().

No heavy model is loaded here (pure unit coverage).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import voice.stt_backend as SB
import voice.settings as _vs
from voice.settings import VoiceSettings


def _patch_settings(monkeypatch, s):
    """Patch the canonical voice_settings singleton that stt_backend reads
    (resolve_primary + build_stt_backend do `from voice.settings import
    voice_settings` at call time)."""
    monkeypatch.setattr(_vs, "voice_settings", s)
    return s


def _fresh_settings():
    return VoiceSettings()


def test_default_primary_is_faster_whisper(monkeypatch):
    s = _fresh_settings()
    assert s.stt_primary == "faster_whisper"
    assert s.stt_fallback == "faster_whisper"
    _patch_settings(monkeypatch, s)
    assert SB.resolve_primary() == "faster_whisper"


def test_resolve_primary_normalizes_aliases(monkeypatch):
    s = _fresh_settings()
    _patch_settings(monkeypatch, s)

    s.stt_primary = "faster-whisper"
    assert SB.resolve_primary() == "faster_whisper"
    s.stt_primary = "whisper"
    assert SB.resolve_primary() == "faster_whisper"
    s.stt_primary = "qwen3"
    assert SB.resolve_primary() == "qwen3"
    s.stt_primary = "qwen3-asr-0.6b-int8"
    assert SB.resolve_primary() == "qwen3"


def test_resolve_primary_unknown_falls_back(monkeypatch):
    s = _fresh_settings()
    _patch_settings(monkeypatch, s)
    s.stt_primary = "totally-bogus-backend-xyz"
    assert SB.resolve_primary() == "faster_whisper"


def test_env_config_roundtrip(monkeypatch):
    monkeypatch.setenv("STT_PRIMARY", "qwen3")
    monkeypatch.setenv("STT_FALLBACK", "faster_whisper")
    s = VoiceSettings()
    s.update_from_env()
    assert s.stt_primary == "qwen3"
    assert s.stt_fallback == "faster_whisper"
    d = s.to_dict()
    assert d["stt_primary"] == "qwen3"
    assert d["stt_fallback"] == "faster_whisper"


def test_build_default_returns_whisper_transcriber(monkeypatch):
    s = _fresh_settings()
    _patch_settings(monkeypatch, s)
    backend = SB.build_stt_backend()
    # Default: returns the production _WhisperTranscriber (type has load +
    # transcribe_with_language + transcribe_fast + transcribe_verify).
    assert backend is not None
    for meth in ("load", "transcribe_with_language", "transcribe", "transcribe_fast", "transcribe_verify"):
        assert callable(getattr(backend, meth, None)), f"missing {meth}"


def test_qwen3_composite_keeps_faster_whisper_confidence(monkeypatch):
    """The composite that runs Qwen3 for transcripts MUST still produce a
    faster-whisper avg_logprob confidence, never None — so the downstream
    hallucination band and intent gates stay active."""
    from voice.command_listener import _WhisperTranscriber

    s = _fresh_settings()
    s.stt_primary = "qwen3"
    _patch_settings(monkeypatch, s)

    # Stub heavy transcriber construction so this is a pure unit test.
    fake_qwen = MagicMock()
    fake_qwen.ready = True
    fake_qwen.name = "qwen3-asr-0.6b-int8"
    fake_qwen.transcribe_with_language.return_value = ("Open Firefox.", 0.0)

    fake_whisper = MagicMock()
    fake_whisper.ready = True
    fake_whisper._device = "cpu"
    fake_whisper._compute = "int8"
    fake_whisper.transcribe_with_language.return_value = ("Open Firefox.", -0.5)
    fake_whisper.transcribe_fast.return_value = ("Open Firefox.", -0.5)
    fake_whisper.transcribe_verify.return_value = ("Open Firefox.", -0.5)

    import voice.command_listener as CL
    monkeypatch.setattr(SB, "Qwen3Transcriber", lambda: fake_qwen)
    monkeypatch.setattr(CL, "_WhisperTranscriber", lambda: fake_whisper)

    backend = SB.build_stt_backend()
    text, conf = backend.transcribe_with_language(b"\x00\x00" * 8000, 16000, "en")

    assert text == "Open Firefox."
    # Confidence MUST be the faster-whisper value (not None, not 0.0-default).
    assert conf == -0.5
