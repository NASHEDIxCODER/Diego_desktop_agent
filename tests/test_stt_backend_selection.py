"""
STT backend selection regression tests (Phase 24.6).

Guarantees:
  1. Default STT primary is QWEN3 (Phase 24.6 production flip).
  2. STT_FALLBACK defaults to faster_whisper and is ALWAYS the confidence /
     verification source, so downstream quality gates never degrade to the
     "trusted path" (never bypassed by missing Qwen confidence).
  3. `resolve_primary()` normalizes aliases and rejects unknown names safely.
  4. Future providers (openai / gemini) are recognized but UNAVAILABLE:
     clear status, no faking, no keys required → safe fallback.
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


def test_default_primary_is_qwen3_with_whisper_fallback(monkeypatch):
    """Phase 24.6: production default is Qwen3; faster-whisper is fallback."""
    s = _fresh_settings()
    assert s.stt_primary == "qwen3"
    assert s.stt_fallback == "faster_whisper"
    _patch_settings(monkeypatch, s)
    assert SB.resolve_primary() == "qwen3"


def test_qwen3_composite_is_default_backend(monkeypatch):
    """With default settings the built backend is the qwen3 composite whose
    confidence still comes from faster-whisper (never None)."""
    s = _fresh_settings()
    assert s.stt_primary == "qwen3"
    _patch_settings(monkeypatch, s)
    fake_qwen = MagicMock()
    fake_qwen.ready = True
    fake_qwen.transcribe_with_language.return_value = ("Open Firefox.", 0.0)
    fake_whisper = MagicMock()
    fake_whisper.ready = True
    fake_whisper._device = "cpu"
    fake_whisper._compute = "int8"
    fake_whisper.transcribe_with_language.return_value = ("Open Firefox.", -0.5)
    monkeypatch.setattr(SB, "Qwen3Transcriber", lambda: fake_qwen)
    import voice.command_listener as CL
    monkeypatch.setattr(CL, "_WhisperTranscriber", lambda: fake_whisper)
    backend = SB.build_stt_backend()
    text, conf = backend.transcribe_with_language(b"\x00\x00" * 8000, 16000, "en")
    assert text == "Open Firefox."
    assert conf == -0.5


def test_future_provider_openai_unavailable_safe_fallback(monkeypatch):
    """openai is reserved (no API key required, never faked): selecting it
    reports UNAVAILABLE and resolves to the safe faster_whisper fallback."""
    s = _fresh_settings()
    s.stt_primary = "openai"
    s.stt_fallback = "faster_whisper"
    _patch_settings(monkeypatch, s)
    status = SB.provider_status("openai")
    assert status["requested"] == "openai"
    assert status["available"] is False
    assert "UNAVAILABLE" in str(status.get("reason", "")).upper()
    assert SB.resolve_primary() == "faster_whisper"
    assert "openai" not in getattr(SB, "LOCAL_PROVIDERS", ("openai", "gemini"))


def test_future_provider_gemini_unavailable_safe_fallback(monkeypatch):
    s = _fresh_settings()
    s.stt_primary = "gemini"
    _patch_settings(monkeypatch, s)
    status = SB.provider_status("gemini")
    assert status["available"] is False
    assert SB.resolve_primary() == "faster_whisper"


# ── Phase 24.7: `sarvam` is experimental / benchmark-only ────────────────

def test_sarvam_is_experimental_not_future():
    """sarvam is an IMPLEMENTED provider that is deliberately kept out of
    production routing — it must be classified experimental, not reserved."""
    assert "sarvam" in SB.EXPERIMENTAL_PROVIDERS
    assert "sarvam" not in SB.FUTURE_PROVIDERS
    assert "sarvam" in SB.NON_ROUTABLE_PROVIDERS
    assert "sarvam" not in SB.LOCAL_PROVIDERS


def test_sarvam_primary_rejected_even_with_api_key(monkeypatch):
    """A configured key does NOT make sarvam routable: STT_PRIMARY=sarvam must
    still report EXPERIMENTAL and fall back, or a benchmark provider could
    silently become Diego's live command path."""
    s = _fresh_settings()
    s.stt_primary = "sarvam"
    s.stt_fallback = "faster_whisper"
    s.sarvam_api_key = "sk-live-looking-key"
    _patch_settings(monkeypatch, s)
    status = SB.provider_status("sarvam")
    assert status["available"] is False
    assert status["experimental"] is True
    assert "EXPERIMENTAL" in str(status["reason"]).upper()
    assert SB.resolve_primary() == "faster_whisper"


def test_sarvam_aliases_all_rejected(monkeypatch):
    """Every accepted spelling resolves to the guarded canonical name."""
    s = _fresh_settings()
    s.stt_primary = "saaras-v4"
    s.sarvam_api_key = "key"
    _patch_settings(monkeypatch, s)
    for alias in ("sarvam", "saaras", "saaras-v4", "sarvam-saaras"):
        st = SB.provider_status(alias)
        assert st["provider"] == "sarvam", alias
        assert st["available"] is False, alias
    assert SB.resolve_primary() == "faster_whisper"


def test_build_stt_backend_refuses_non_routable(monkeypatch):
    """Defense in depth: if resolve_primary() ever returned a non-routable
    provider, build_stt_backend() must fail loud rather than build a local
    backend under a cloud provider's name."""
    s = _fresh_settings()
    _patch_settings(monkeypatch, s)
    monkeypatch.setattr(SB, "resolve_primary", lambda: "sarvam")
    try:
        SB.build_stt_backend()
        assert False, "expected RuntimeError for non-routable provider"
    except RuntimeError as e:
        assert "non-routable" in str(e)


def test_active_stt_report_mentions_provider(monkeypatch):
    s = _fresh_settings()
    _patch_settings(monkeypatch, s)
    report = SB.active_stt_report()
    assert "qwen3" in report
    assert "faster_whisper" in report


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


def test_build_faster_whisper_primary_returns_whisper_transcriber(monkeypatch):
    s = _fresh_settings()
    s.stt_primary = "faster_whisper"
    _patch_settings(monkeypatch, s)
    backend = SB.build_stt_backend()
    # faster_whisper primary: returns the production _WhisperTranscriber
    # (type has load + transcribe_with_language + transcribe_fast +
    # transcribe_verify).
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


# ── Phase 24.7: benchmark seam + 1.7B default ────────────────────────────

def test_benchmark_seam_requires_explicit_optin():
    """The experimental seam must be impossible to reach by accident."""
    try:
        SB.experimental_provider_for_benchmark("sarvam", allow_experimental=False)
        assert False, "expected PermissionError without the explicit opt-in"
    except PermissionError as e:
        assert "allow_experimental" in str(e)


def test_benchmark_seam_rejects_non_experimental_names():
    """Production / reserved providers are not reachable through the seam."""
    for name in ("qwen3", "faster_whisper", "openai", "gemini", "nonsense"):
        try:
            SB.experimental_provider_for_benchmark(name, allow_experimental=True)
            assert False, f"expected ValueError for {name!r}"
        except ValueError:
            pass


def test_benchmark_seam_refuses_sarvam_without_key(monkeypatch):
    """No key → refuse (never fabricate a provider or a transcript)."""
    import voice.settings as _settings_mod
    s = _fresh_settings()
    s.sarvam_api_key = ""
    monkeypatch.setattr(_settings_mod, "voice_settings", s)
    try:
        SB.experimental_provider_for_benchmark("sarvam", allow_experimental=True)
        assert False, "expected ValueError when SARVAM_API_KEY is unset"
    except ValueError as e:
        assert "SARVAM_API_KEY" in str(e)


def test_benchmark_seam_returns_provider_with_key(monkeypatch):
    """With a key the seam yields the isolated provider, still flagged as
    experimental and outside production routing."""
    import voice.settings as _settings_mod
    s = _fresh_settings()
    s.sarvam_api_key = "test-key"
    monkeypatch.setattr(_settings_mod, "voice_settings", s)
    provider = SB.experimental_provider_for_benchmark("sarvam",
                                                     allow_experimental=True)
    assert provider.name == "sarvam-saaras-v4"
    assert provider.experimental is True
    assert provider.production_routing is False
    assert provider.health()["production_routing"] is False


def test_experimental_provider_available_reports_credential_only(monkeypatch):
    import voice.settings as _settings_mod
    s = _fresh_settings()
    monkeypatch.setattr(_settings_mod, "voice_settings", s)
    s.sarvam_api_key = ""
    assert SB.experimental_provider_available() == {"sarvam": False}
    s.sarvam_api_key = "abc"
    assert SB.experimental_provider_available() == {"sarvam": True}


def test_stt_model_size_default_is_1_7b():
    """Phase 24.7: Qwen3-ASR 1.7B is the default local ASR model."""
    assert _fresh_settings().stt_model_size == "1.7b"


def test_qwen3_transcriber_selects_1_7b_by_default(monkeypatch):
    """Default builds the 1.7B provider; 0.6B only on explicit opt-in."""
    import voice.settings as _settings_mod
    import voice.providers.sherpa_providers as SP
    built: list = []

    class _FakeProvider:
        def __init__(self, label):
            self._label = label

        def start(self):
            built.append(self._label)
            return True

        def health(self):
            return {"load_error": None}

    s = _fresh_settings()
    monkeypatch.setattr(_settings_mod, "voice_settings", s)
    monkeypatch.setattr(SP, "Qwen3ASR17BProvider", lambda: _FakeProvider("1.7b"))
    monkeypatch.setattr(SP, "Qwen3ASRProvider", lambda: _FakeProvider("0.6b"))

    t = SB.Qwen3Transcriber()
    assert t.load() is True
    assert built == ["1.7b"]
    assert t.name == "qwen3-asr-1.7b-int8"

    s.stt_model_size = "0.6b"
    t2 = SB.Qwen3Transcriber()
    assert t2.load() is True
    assert built == ["1.7b", "0.6b"]
    assert t2.name == "qwen3-asr-0.6b-int8"


def test_active_stt_report_names_1_7b_and_experimental_sarvam(monkeypatch):
    """The startup line states the model and, when sarvam is requested,
    explains why it is not used."""
    s = _fresh_settings()
    _patch_settings(monkeypatch, s)
    report = SB.active_stt_report()
    assert "qwen3" in report and "1.7B" in report

    s.stt_primary = "sarvam"
    s.sarvam_api_key = "key"
    report2 = SB.active_stt_report()
    assert "sarvam" in report2
    assert "EXPERIMENTAL" in report2.upper()

