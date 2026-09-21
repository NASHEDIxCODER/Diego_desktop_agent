"""
SarvamSaarasProvider (Saaras v4) — EXPERIMENTAL provider unit tests (Phase 24.7).

Scope: the isolated benchmark-only cloud provider. Nothing here touches
production routing; the routing guards themselves are covered in
tests/test_stt_backend_selection.py.

Guarantees verified here:
  1. No key → UNAVAILABLE, empty transcript, no network call, never faked.
  2. Correct Sarvam REST contract: POST, `api-subscription-key` header,
     `file`/`model`/`language_code` multipart fields, and `mode` is NOT sent
     for saaras:v4 (it is a saaras:v3-only parameter).
  3. Diego language hints map to BCP-47; unknown hints auto-detect.
  4. Audio > 30 s (REST limit) is reported, never silently truncated.
  5. HTTP errors / transport failures / malformed JSON degrade to "" with a
     recorded reason — no fabricated transcript, no exception leaked.
  6. stream() never fabricates partials and never hits the network.
  7. The provider is marked experimental and outside production routing.

The network seam (`_http_post_multipart`) is monkeypatched in every test, so
no test performs I/O.
"""

from __future__ import annotations

import sys
import wave
import io
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import voice.settings as _settings_mod
from voice.settings import VoiceSettings
import voice.providers.sarvam_provider as SV
from voice.providers.sarvam_provider import (
    SarvamSaarasProvider,
    map_language_code,
    to_wav_bytes,
)


def _provider(monkeypatch, key="test-key", **kw):
    """Provider wired to a private settings instance (no env leakage)."""
    s = VoiceSettings()
    s.sarvam_api_key = key
    s.sarvam_timeout_s = 5.0
    monkeypatch.setattr(_settings_mod, "voice_settings", s)
    return SarvamSaarasProvider(**kw)


def _recorder(monkeypatch, response=None, exc=None):
    """Capture requests and return a canned response from the network seam."""
    calls: list = []

    def fake_post(url, headers, files, data, timeout_s):
        calls.append({"url": url, "headers": headers, "files": files,
                      "data": data, "timeout_s": timeout_s})
        if exc is not None:
            raise exc
        return response if response is not None else {
            "status_code": 200,
            "json": {"request_id": "r1", "transcript": "Open Firefox",
                     "language_code": "en-IN"},
            "text": "{}",
        }

    monkeypatch.setattr(SV, "_http_post_multipart", fake_post)
    return calls


def _audio(seconds=2.0, value=0.1):
    return np.full(int(16000 * seconds), value, dtype=np.float32)


# ── Pure helpers ─────────────────────────────────────────────────────────

def test_language_hints_map_to_bcp47():
    assert map_language_code("en") == "en-IN"
    assert map_language_code("EN") == "en-IN"
    assert map_language_code("hi") == "hi-IN"
    assert map_language_code("hinglish") == "hi-IN"
    assert map_language_code("bn-IN") == "bn-IN"      # already BCP-47
    assert map_language_code("") == "unknown"
    assert map_language_code(None) == "unknown"
    assert map_language_code("klingon") == "unknown"  # never guess


def test_to_wav_bytes_is_16khz_mono_pcm16():
    data = to_wav_bytes(_audio(1.0), 16000)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 16000


def test_to_wav_bytes_clips_out_of_range_without_gain():
    """Values are clipped to [-1,1]; no gain is applied to quiet audio."""
    loud = to_wav_bytes(np.array([5.0, -5.0], dtype=np.float32), 16000)
    with wave.open(io.BytesIO(loud), "rb") as w:
        frames = np.frombuffer(w.readframes(2), dtype="<i2")
    assert frames[0] == 32767 and frames[1] == -32767
    quiet = to_wav_bytes(np.array([0.001], dtype=np.float32), 16000)
    with wave.open(io.BytesIO(quiet), "rb") as w:
        assert abs(int(np.frombuffer(w.readframes(1), dtype="<i2")[0])) < 100


# ── Credential / availability ─────────────────────────────────────────────

def test_missing_key_is_unavailable_and_never_faked(monkeypatch):
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch, key="")
    assert p.start() is False
    assert p.health()["ready"] is False
    assert "UNAVAILABLE" in p.health()["reason"].upper()
    assert p.health()["api_key_set"] is False
    assert p.transcribe(_audio(1.0), 16000, "en") == ""
    assert calls == []  # no network attempt without credentials


def test_verify_credentials_false_without_key(monkeypatch):
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch, key="")
    assert p.verify_credentials() is False
    assert calls == []


def test_verify_credentials_true_on_success(monkeypatch):
    _recorder(monkeypatch)
    p = _provider(monkeypatch)
    assert p.verify_credentials() is True


# ── REST contract ─────────────────────────────────────────────────────────

def test_request_uses_correct_endpoint_header_and_model(monkeypatch):
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch, key="secret-key")
    assert p.start() is True
    text = p.transcribe(_audio(2.0), 16000, "hinglish")
    assert text == "Open Firefox"
    assert len(calls) == 1
    c = calls[0]
    assert c["url"] == "https://api.sarvam.ai/speech-to-text"
    assert c["headers"]["api-subscription-key"] == "secret-key"
    assert c["timeout_s"] == 5.0
    assert c["data"]["model"] == "saaras:v4"
    assert c["data"]["language_code"] == "hi-IN"
    assert "file" in c["files"]
    fname, blob, mime = c["files"]["file"]
    assert fname.endswith(".wav") and mime == "audio/wav"
    assert blob[:4] == b"RIFF"  # a real WAV container


def test_mode_not_sent_for_saaras_v4(monkeypatch):
    """`mode` is documented as saaras:v3-only; sending it for v4 is wrong."""
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch)
    p.transcribe(_audio(2.0), 16000, "en")
    assert "mode" not in calls[0]["data"]


def test_mode_sent_for_saaras_v3(monkeypatch):
    """When explicitly pinned to v3 the documented `mode` field IS sent."""
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch, model="saaras:v3")
    p.transcribe(_audio(2.0), 16000, "en")
    assert calls[0]["data"]["mode"] == "transcribe"


def test_detected_language_recorded_in_metrics(monkeypatch):
    _recorder(monkeypatch)
    p = _provider(monkeypatch)
    p.transcribe(_audio(2.0), 16000, "hi")
    assert p.metrics()["last_language_code"] == "en-IN"
    assert p.health()["requests"] == 1


# ── Guard rails: limits and short audio ───────────────────────────────────

def test_audio_over_rest_limit_is_refused_not_truncated(monkeypatch):
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(31.0), 16000, "en") == ""
    assert calls == []  # refused before any upload
    assert "30s" in p.health()["reason"] or "limit" in p.health()["reason"]


def test_too_short_audio_short_circuits(monkeypatch):
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(0.05), 16000, "en") == ""
    assert calls == []


def test_empty_audio_is_safe(monkeypatch):
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch)
    assert p.transcribe(np.zeros(0, dtype=np.float32), 16000, "en") == ""
    assert p.transcribe(None, 16000, "en") == ""
    assert calls == []



# ── Failure modes: never fabricate, never raise ───────────────────────────

def test_http_error_returns_empty_with_api_detail(monkeypatch):
    _recorder(monkeypatch, response={
        "status_code": 422,
        "json": {"error": "audio too long", "detail": "invalid file"},
        "text": "unprocessable"})
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(2.0), 16000, "en") == ""
    reason = p.health()["reason"]
    assert "HTTP 422" in reason and "audio too long" in reason
    assert p.health()["errors"] == 1


def test_http_error_without_json_uses_raw_text(monkeypatch):
    _recorder(monkeypatch, response={"status_code": 401, "json": None,
                                     "text": "invalid api key"})
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(2.0), 16000, "en") == ""
    assert "401" in p.health()["reason"]


def test_transport_failure_is_caught(monkeypatch):
    _recorder(monkeypatch, exc=TimeoutError("connection timed out"))
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(2.0), 16000, "en") == ""
    assert "transport failure" in p.health()["reason"]
    assert p.health()["errors"] == 1


def test_malformed_success_body_is_not_trusted(monkeypatch):
    """HTTP 200 with a non-dict body must not be reported as a transcript."""
    _recorder(monkeypatch, response={"status_code": 200, "json": None,
                                     "text": "not json"})
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(2.0), 16000, "en") == ""
    assert p.health()["errors"] == 1


def test_empty_transcript_field_is_empty_string(monkeypatch):
    _recorder(monkeypatch, response={"status_code": 200, "json": {},
                                     "text": "{}"})
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(2.0), 16000, "en") == ""


def test_transcript_is_stripped(monkeypatch):
    _recorder(monkeypatch, response={"status_code": 200,
                                     "json": {"transcript": "  Open Firefox.  "},
                                     "text": "{}"})
    p = _provider(monkeypatch)
    assert p.transcribe(_audio(2.0), 16000, "en") == "Open Firefox."


def test_short_key_whitespace_only_is_unavailable(monkeypatch):
    p = _provider(monkeypatch, key="   ")
    assert p.start() is False


# ── Streaming honesty ─────────────────────────────────────────────────────

def test_stream_never_fabricates_and_never_hits_network(monkeypatch):
    calls = _recorder(monkeypatch)
    p = _provider(monkeypatch)
    assert p.stream(_audio(0.5), 16000, "en") == ""
    assert p.stream(_audio(0.5), 16000, "en") == ""
    assert calls == []
    p.reset()
    assert p.stream(np.zeros(0, dtype=np.float32), 16000, "en") == ""


# ── Isolation markers ─────────────────────────────────────────────────────

def test_provider_is_marked_experimental(monkeypatch):
    p = _provider(monkeypatch)
    assert p.experimental is True
    assert p.production_routing is False
    assert SV.PRODUCTION_ROUTING_ENABLED is False
    assert p.health()["experimental"] is True
    assert p.health()["production_routing"] is False
    assert p.metrics()["experimental"] is True
    assert p.name == "sarvam-saaras-v4"


def test_provider_not_in_local_providers():
    import voice.stt_backend as SB
    assert "sarvam" not in SB.LOCAL_PROVIDERS
    assert "sarvam" in SB.NON_ROUTABLE_PROVIDERS


def test_stop_clears_state(monkeypatch):
    p = _provider(monkeypatch)
    p.start()
    p.stream(_audio(0.5), 16000, "en")
    p.stop()
    assert p.health()["ready"] is False
    assert p.stream(np.zeros(0, dtype=np.float32), 16000, "en") == ""


# ── Secret hygiene ────────────────────────────────────────────────────────

def test_api_key_never_leaks_through_health_or_metrics(monkeypatch):
    """health()/metrics() are logged and surfaced — the key must never appear."""
    secret = "sk-super-secret-value"
    p = _provider(monkeypatch, key=secret)
    p.start()
    p.transcribe(_audio(2.0), 16000, "en")
    for blob in (repr(p.health()), repr(p.metrics()), str(p.health()),
                 str(p.metrics())):
        assert secret not in blob
    assert p.health()["api_key_set"] is True


def test_settings_to_dict_reports_key_presence_not_value(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "sk-do-not-serialize")
    s = VoiceSettings()
    s.update_from_env()
    assert s.sarvam_api_key == "sk-do-not-serialize"
    d = s.to_dict()
    assert d["sarvam_api_key_set"] is True
    assert "sk-do-not-serialize" not in repr(d)
    assert "sarvam_api_key" not in d


# ── Configuration defaults ────────────────────────────────────────────────

def test_benchmark_provider_settings_defaults():
    """Defaults must be safe: no key, the documented endpoint, saaras:v4."""
    s = VoiceSettings()
    assert s.sarvam_api_key == ""
    assert s.sarvam_stt_url == "https://api.sarvam.ai/speech-to-text"
    assert s.sarvam_stt_model == "saaras:v4"
    assert s.sarvam_timeout_s == 30.0


def test_sarvam_env_overrides(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "env-key")
    monkeypatch.setenv("SARVAM_STT_MODEL", "saaras:v3")
    monkeypatch.setenv("SARVAM_STT_URL", "https://example.test/stt")
    monkeypatch.setenv("SARVAM_TIMEOUT_S", "12.5")
    s = VoiceSettings()
    s.update_from_env()
    assert s.sarvam_api_key == "env-key"
    assert s.sarvam_stt_model == "saaras:v3"
    assert s.sarvam_stt_url == "https://example.test/stt"
    assert s.sarvam_timeout_s == 12.5


def test_sarvam_timeout_env_garbage_is_ignored(monkeypatch):
    """A malformed timeout must not crash startup or zero the timeout."""
    monkeypatch.setenv("SARVAM_TIMEOUT_S", "not-a-number")
    s = VoiceSettings()
    s.update_from_env()
    assert s.sarvam_timeout_s == 30.0


def test_provider_defaults_come_from_settings(monkeypatch):
    """With no explicit args the provider reads the configured endpoint/model."""
    s = VoiceSettings()
    s.sarvam_stt_model = "saaras:v3"
    s.sarvam_stt_url = "https://api.sarvam.ai/speech-to-text"
    monkeypatch.setattr(_settings_mod, "voice_settings", s)
    p = SarvamSaarasProvider()
    assert p.model == "saaras:v3"
    assert p.endpoint == "https://api.sarvam.ai/speech-to-text"


def test_instances_do_not_share_state(monkeypatch):
    """Two providers (e.g. v3 vs v4 benchmark rows) must be independent."""
    _recorder(monkeypatch)
    a = _provider(monkeypatch)
    b = _provider(monkeypatch, model="saaras:v3")
    a.transcribe(_audio(2.0), 16000, "en")
    assert a.health()["requests"] == 1
    assert b.health()["requests"] == 0
    assert a.model == "saaras:v4" and b.model == "saaras:v3"


# ── Isolation: no runtime module may import the experimental provider ─────

def test_no_runtime_module_imports_the_experimental_provider():
    """Guards the Phase 24.7 boundary: only benchmarks may reference this
    provider. If a runtime module ever imports it, this test fails."""
    runtime_files = [
        PROJECT_ROOT / "voice" / "command_listener.py",
        PROJECT_ROOT / "voice" / "asr_fallback.py",
        PROJECT_ROOT / "agent" / "brain.py",
        PROJECT_ROOT / "agent" / "streaming_llm.py",
        PROJECT_ROOT / "core" / "conversation_engine.py",
        PROJECT_ROOT / "main.py",
    ]
    offenders = []
    for path in runtime_files:
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        if "sarvam_provider" in src or "SarvamSaarasProvider" in src:
            offenders.append(path.name)
    assert offenders == [], f"experimental provider leaked into runtime: {offenders}"

