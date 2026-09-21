"""
Qwen3-ASR output-format regression tests (Phase 24.7).

Targets a REAL defect found by probing the live 1.7B export with the repo's own
human recordings:

    REQUIRED  'Hello, Diego.'
    OBSERVED  'language English<asr_text>Hello, Diego.'

The Qwen3-ASR export is a chat model, so the decoder can emit its
chat-template preamble in front of the transcript. Because int8 inference is
multithreaded, an input differing by one LSB — exactly what the
float -> int16 -> float round-trip in the command path produces — can flip the
greedy decode between the two forms. Both forms are therefore reachable in
production, and the preamble would have reached the intent parser verbatim
("language English<asr_text>turn on the lights" matches no command).

Guarantees:
  1. `parse_qwen3_asr_output()` strips the preamble and recovers the language.
  2. A plain transcript is returned UNCHANGED — never guessed at or discarded.
  3. The mixin is wired into BOTH the 1.7B and 0.6B providers, and wraps the
     base recognizer rather than replacing it.
  4. The fix holds across the production surface: provider.transcribe(),
     provider.stream() and stt_backend.Qwen3Transcriber.

No model is loaded — the recognizer is faked, so these run in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

sherpa_providers = pytest.importorskip(
    "voice.providers.sherpa_providers",
    reason="sherpa_onnx runtime not installed",
)
from voice.providers.sherpa_providers import (  # noqa: E402
    Qwen3ASRProvider,
    Qwen3ASR17BProvider,
    parse_qwen3_asr_output,
)

# The exact string the live 1.7B produced for models/wake/positives/positive_007.wav
LIVE_17B_WRAPPED = "language English<asr_text>Hello, Diego."
LIVE_17B_EXPECTED = "Hello, Diego."


# ── Fakes: exercise the real base recognizer path without loading a model ──

class _Result:
    def __init__(self, text):
        self.text = text


class _Stream:
    def __init__(self, text):
        self.result = _Result(text)
        self.accepted = 0
        self.sample_rates = []
        self.samples = []

    def accept_waveform(self, sample_rate, audio):
        self.accepted += 1
        self.sample_rates.append(sample_rate)
        self.samples.append(len(audio))


class _FakeRecognizer:
    """Mirrors the sherpa_onnx OfflineRecognizer surface used by the base."""

    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.decode_calls = 0
        self.streams = []

    def create_stream(self):
        s = _Stream(self.raw_text)
        self.streams.append(s)
        return s

    def decode_stream(self, stream):
        assert stream in self.streams, "decoded a stream it did not create"
        assert stream.accepted == 1, "audio was never fed to the stream"
        assert stream.result.text == self.raw_text
        self.decode_calls += 1


def _provider(cls, raw_text):
    """A provider with a faked recognizer, marked ready (no download/load)."""
    p = cls()
    p._recognizer = _FakeRecognizer(raw_text)
    p._ready = True
    return p


def _audio(seconds=1.0):
    return np.full(int(16000 * seconds), 0.05, dtype=np.float32)


def _pcm16(seconds=1.0):
    return (np.full(int(16000 * seconds), 0.05, dtype=np.float32)
            * 32767.0).astype(np.int16).tobytes()


# ── Pure parser ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected_text,expected_lang", [
    # The live 1.7B regression.
    (LIVE_17B_WRAPPED, LIVE_17B_EXPECTED, "English"),
    # Other real preamble shapes.
    ("language Hindi<asr_text>बंद करो", "बंद करो", "Hindi"),
    ("language Chinese<asr_text>你好", "你好", "Chinese"),
    ("language Spanish<asr_text>Buenos días", "Buenos días", "Spanish"),
    ("<asr_text>Hello", "Hello", ""),
    ("<asr_text>", "", ""),
    # Degenerate language fields must not be surfaced as a language.
    ("language None<asr_text>Hello", "Hello", ""),
    ("language none<asr_text>Hello", "Hello", ""),
    ("language <asr_text>Hello", "Hello", ""),
    # Plain transcripts pass through untouched.
    (LIVE_17B_EXPECTED, LIVE_17B_EXPECTED, ""),
    ("Good morning.", "Good morning.", ""),
    ("", "", ""),
])
def test_parse_qwen3_asr_output(raw, expected_text, expected_lang):
    assert parse_qwen3_asr_output(raw) == (expected_text, expected_lang)


def test_none_input_is_safe():
    assert parse_qwen3_asr_output(None) == ("", "")


def test_plain_transcript_is_never_altered():
    """No marker => byte-identical passthrough (no guessing, no stripping)."""
    for raw in (
        "open firefox and search for cats",
        "Language is a funny word",       # starts with 'language'
        "language English",               # prefix without a marker
        "Set the language to English.",
        "translate: language -> bhasha",
    ):
        text, lang = parse_qwen3_asr_output(raw)
        assert text == raw
        assert lang == ""


def test_preamble_parsing_is_idempotent():
    text, lang = parse_qwen3_asr_output(LIVE_17B_WRAPPED)
    assert parse_qwen3_asr_output(text) == (LIVE_17B_EXPECTED, "")
    assert lang == "English"


def test_surrounding_whitespace_is_trimmed():
    assert parse_qwen3_asr_output(
        "  language English  <asr_text>  hi  ") == ("hi", "English")
    assert parse_qwen3_asr_output("   spaced   ") == ("spaced", "")


def test_punctuation_and_case_preserved():
    raw = "language English<asr_text>Open Firefox, then search \"Python 3.11\"?!"
    text, lang = parse_qwen3_asr_output(raw)
    assert text == 'Open Firefox, then search "Python 3.11"?!'
    assert lang == "English"


# ── Mixin wiring ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_both_qwen3_providers_override_recognize_offline(cls):
    """The fix must cover the 1.7B (default) AND the 0.6B (low-RAM) export."""
    assert cls._recognize_offline is not \
        sherpa_providers.SherpaOnnxProvider._recognize_offline
    assert issubclass(cls, sherpa_providers._Qwen3ASROutputMixin)


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_transcribe_strips_preamble(cls):
    p = _provider(cls, LIVE_17B_WRAPPED)
    assert p.transcribe(_audio(), 16000) == LIVE_17B_EXPECTED


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_transcribe_passes_plain_text_through(cls):
    p = _provider(cls, LIVE_17B_EXPECTED)
    assert p.transcribe(_audio(), 16000) == LIVE_17B_EXPECTED


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_stream_partial_is_also_cleaned(cls):
    """Partials come from the same recognizer, so they need the same fix."""
    p = _provider(cls, LIVE_17B_WRAPPED)
    assert p.stream(_audio(0.5), 16000) == LIVE_17B_EXPECTED


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_recognizer_is_still_actually_used(cls):
    """Regression guard: the mixin must WRAP the base, not bypass it."""
    p = _provider(cls, LIVE_17B_EXPECTED)
    p.transcribe(_audio(), 16000)
    assert p._recognizer.decode_calls == 1


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_audio_and_sample_rate_still_reach_recognizer(cls):
    """The wrapper must not swallow the audio or alter the 16 kHz contract."""
    p = _provider(cls, LIVE_17B_WRAPPED)
    p.transcribe(_audio(), 16000)
    stream = p._recognizer.streams[-1]
    assert stream.accepted == 1
    assert stream.sample_rates == [16000]
    assert stream.samples == [16000]  # 1.0 s of audio forwarded intact


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_detected_language_is_recorded(cls):
    p = _provider(cls, "language Hindi<asr_text>नमस्ते")
    assert p.transcribe(_audio(), 16000) == "नमस्ते"
    assert p._last_detected_language == "Hindi"


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_language_not_set_for_plain_transcript(cls):
    p = _provider(cls, LIVE_17B_EXPECTED)
    p.transcribe(_audio(), 16000)
    assert p._last_detected_language == ""


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_empty_wrapped_decode_yields_no_transcript(cls):
    """'language None<asr_text>' must not become a bogus command."""
    p = _provider(cls, "language None<asr_text>")
    assert p.transcribe(_audio(), 16000) == ""


@pytest.mark.parametrize("cls", [Qwen3ASR17BProvider, Qwen3ASRProvider])
def test_too_short_audio_still_short_circuits(cls):
    """The base guard must survive the override (no recognizer call)."""
    p = _provider(cls, LIVE_17B_WRAPPED)
    assert p.transcribe(_audio(0.01), 16000) == ""
    assert p._recognizer.decode_calls == 0




# ── Full production path via stt_backend ──────────────────────────────────

def test_qwen3_transcriber_surface_is_clean():
    """The command listener talks to Qwen3Transcriber, not the provider."""
    from voice.stt_backend import Qwen3Transcriber

    t = Qwen3Transcriber()
    t._provider = _provider(Qwen3ASR17BProvider, LIVE_17B_WRAPPED)
    t._ready = True

    text, conf = t.transcribe_with_language(_pcm16(), 16000, "en")
    assert text == LIVE_17B_EXPECTED
    assert conf == 0.0  # Qwen3 exposes no log-probs: confidence is never faked


def test_qwen3_transcriber_does_not_mangle_plain_text():
    from voice.stt_backend import Qwen3Transcriber

    t = Qwen3Transcriber()
    t._provider = _provider(Qwen3ASR17BProvider, LIVE_17B_EXPECTED)
    t._ready = True

    assert t.transcribe_with_language(_pcm16(), 16000, "en")[0] == LIVE_17B_EXPECTED


def test_qwen3_transcriber_legacy_transcribe_signature():
    """`transcribe()` (2-arg) must stay clean for older callers."""
    from voice.stt_backend import Qwen3Transcriber

    t = Qwen3Transcriber()
    t._provider = _provider(Qwen3ASR17BProvider, LIVE_17B_WRAPPED)
    t._ready = True

    assert t.transcribe(_pcm16(), 16000)[0] == LIVE_17B_EXPECTED
