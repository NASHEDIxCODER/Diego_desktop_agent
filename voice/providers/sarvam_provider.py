"""
SarvamSaarasProvider — EXPERIMENTAL cloud ASR provider (BENCHMARK ONLY).

STATUS: isolated experiment. This provider is NOT wired into Diego's
production ASR routing. `STT_PRIMARY=sarvam` is treated as an experimental
provider by voice/stt_backend.py and safely falls back to the configured
fallback backend (faster-whisper) even when an API key is present. The only
code path that instantiates this class is the benchmark harness
(scripts/phase24g_sarvam_benchmark.py).

Why it exists: to measure Sarvam Saaras v4 (Global + Indian English and 22
Indic languages, native code-mixing) against the local Qwen3-ASR 1.7B/0.6B
and faster-whisper before any production decision is made.

DESIGN RULES (identical to the rest of the ASR stack):
  - Consumes the SAME normalized float32 [-1,1] 16 kHz mono audio that
    AudioManager already produces. No own VAD, no gain, no re-sampling
    (Sarvam REST only accepts 16 kHz PCM, which is exactly what we hold).
  - NEVER implements its own VAD. Endpointing stays owned by voice/vad.py and
    the command listener.
  - NEVER invokes an LLM. ASR is a dedicated speech model.
  - Network I/O is confined to one injectable module-level function so unit
    tests never touch the network.
  - Degrades gracefully: missing key / HTTP error / malformed JSON returns an
    empty transcript plus a recorded error reason — never a fake transcript.

API NOTES (docs.sarvam.ai, verified):
  POST https://api.sarvam.ai/speech-to-text   (multipart/form-data)
  header: `api-subscription-key: <key>`
  fields: `file` (required), `model`, `language_code`, plus `mode` —
          `mode` is ONLY supported by saaras:v3, so it is NOT sent for
          saaras:v4 (sending it would be rejected/misleading).
  REST limit: 30 seconds per request. Audio > 30 s is reported as an error
  instead of being silently truncated.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np

from voice.asr_provider import ASRProvider, get_cpu_ram_stats, get_gpu_stats

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "saaras:v4"
API_KEY_HEADER = "api-subscription-key"

#: Sarvam REST is synchronous and rejects audio longer than this.
MAX_AUDIO_S = 30.0

#: `mode` is documented as applicable only to saaras:v3.
_MODE_SUPPORTING_MODELS = ("saaras:v3",)

#: Diego passes short codes ("en"/"hi"); Sarvam wants BCP-47 (or "unknown").
_LANGUAGE_MAP: Dict[str, str] = {
    "": "unknown",
    "auto": "unknown",
    "unknown": "unknown",
    "en": "en-IN",
    "en-in": "en-IN",
    "english": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "hindi": "hi-IN",
    "hinglish": "hi-IN",  # code-mixed speech is scored against the hi path
}


def map_language_code(language: Optional[str]) -> str:
    """Map a Diego language hint to a Sarvam `language_code` value.

    Unknown hints become "unknown" so Sarvam auto-detects instead of us
    guessing a wrong Indic language.
    """
    raw = str(language or "").strip().lower()
    if raw in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[raw]
    # Already BCP-47 shaped (e.g. "bn-IN") — normalize casing (xx-YY) and
    # pass through unchanged.
    parts = raw.split("-")
    if len(parts) == 2 and len(parts[0]) in (2, 3) and len(parts[1]) == 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return "unknown"


def to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Serialize float32 [-1,1] mono audio to a 16-bit PCM WAV byte string.

    Sarvam REST accepts WAV directly, so no external encoder is needed.
    Values are clipped to [-1,1] before quantization (no gain applied).
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size:
        arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm)
    return buf.getvalue()


def _http_post_multipart(url: str, headers: Dict[str, str],
                         files: Dict[str, Tuple[str, bytes, str]],
                         data: Dict[str, str], timeout_s: float) -> Dict[str, Any]:
    """Single injectable network seam (monkeypatched in unit tests).

    Returns ``{"status_code": int, "json": dict|None, "text": str}``.
    Raises only on transport failure; HTTP error codes are returned so the
    caller can surface the API's own message.
    """
    import httpx

    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, headers=headers, files=files, data=data)
    payload: Optional[dict] = None
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None
    return {"status_code": int(resp.status_code), "json": payload,
            "text": (resp.text or "")[:2000]}


class SarvamSaarasProvider(ASRProvider):
    """Sarvam Saaras v4 REST provider — EXPERIMENTAL / BENCHMARK ONLY."""

    name = "sarvam-saaras-v4"

    #: Explicit markers so nothing can mistake this for a production backend.
    experimental = True
    production_routing = False

    def __init__(self, api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout_s: Optional[float] = None):
        s = self._settings()
        self._api_key = (api_key if api_key is not None
                         else str(getattr(s, "sarvam_api_key", "") or ""))
        self._endpoint = (endpoint if endpoint is not None
                          else str(getattr(s, "sarvam_stt_url", "")
                                   or DEFAULT_ENDPOINT))
        self._model = (model if model is not None
                       else str(getattr(s, "sarvam_stt_model", "")
                                or DEFAULT_MODEL))
        self._timeout_s = float(timeout_s if timeout_s is not None
                                else getattr(s, "sarvam_timeout_s", 30.0) or 30.0)
        self._ready = False
        self._reason = "not started"
        self._lock = threading.Lock()
        # Streaming state (REST has no partials — see stream()).
        self._stream_buffer: list = []
        self._stream_text = ""
        # Metrics
        self._requests = 0
        self._errors = 0
        self._total_latency_s = 0.0
        self._last_language_detected = ""

    # ── Helpers ────────────────────────────────────────────
    @staticmethod
    def _settings():
        from voice.settings import voice_settings
        return voice_settings

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return self._endpoint

    # ── Lifecycle ──────────────────────────────────────────
    def start(self) -> bool:
        """Validate configuration only — no network call at startup.

        A missing key is reported as UNAVAILABLE (never faked). Credential
        validity is checked on demand via verify_credentials().
        """
        if not self._api_key.strip():
            self._ready = False
            self._reason = ("UNAVAILABLE: SARVAM_API_KEY is not set "
                            "(experimental provider, benchmark only)")
            logger.info("[SARVAM] %s", self._reason)
            return False
        self._ready = True
        self._reason = f"ready (experimental/benchmark-only, model={self._model})"
        logger.info("[SARVAM] %s endpoint=%s", self._reason, self._endpoint)
        return True

    def stop(self) -> None:
        self._ready = False
        self._stream_buffer = []
        self._stream_text = ""

    def verify_credentials(self) -> bool:
        """Best-effort live check: send 0.5 s of silence and see whether the
        API accepts the key. Used by the benchmark harness (`--check`), never
        by production."""
        if not self.start():
            return False
        audio = np.zeros(int(16000 * 0.5), dtype=np.float32)
        self.transcribe(audio, 16000, "en")
        ok = self._errors == 0
        if not ok:
            self._reason = "UNAVAILABLE: credential/HTTP check failed"
        return ok

    # ── ASRProvider interface ──────────────────────────────
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: Optional[str] = None) -> str:
        """Full-utterance transcription via Sarvam REST.

        Returns "" on any failure (missing key, HTTP error, malformed
        response) and records the reason in health(). Never raises.
        """
        if not self._ready and not self.start():
            return ""
        arr = (np.asarray(audio, dtype=np.float32).reshape(-1)
               if audio is not None else np.zeros(0, dtype=np.float32))
        rate = int(sample_rate or 16000)
        duration_s = len(arr) / float(rate)
        if duration_s < 0.15:
            return ""
        if duration_s > MAX_AUDIO_S:
            self._errors += 1
            self._reason = (f"ERROR: audio {duration_s:.1f}s exceeds Sarvam REST "
                            f"limit of {MAX_AUDIO_S:.0f}s (not truncated)")
            logger.warning("[SARVAM] %s", self._reason)
            return ""

        data: Dict[str, str] = {"model": self._model,
                                "language_code": map_language_code(language)}
        # `mode` is documented as saaras:v3-only; sending it for v4 is wrong.
        if self._model in _MODE_SUPPORTING_MODELS:
            data["mode"] = "transcribe"
        files = {"file": ("diego_utterance.wav", to_wav_bytes(arr, rate),
                          "audio/wav")}
        headers = {API_KEY_HEADER: self._api_key.strip(),
                   "Accept": "application/json"}

        t0 = time.time()
        self._requests += 1
        try:
            resp = _http_post_multipart(self._endpoint, headers, files, data,
                                        self._timeout_s)
        except Exception as e:  # transport failure
            self._errors += 1
            self._reason = f"ERROR: transport failure: {type(e).__name__}: {e}"
            logger.warning("[SARVAM] %s", self._reason)
            return ""
        self._total_latency_s += time.time() - t0

        status = int(resp.get("status_code", 0))
        payload = resp.get("json")
        if status != 200 or not isinstance(payload, dict):
            self._errors += 1
            detail = ""
            if isinstance(payload, dict):
                detail = str(payload.get("error") or payload.get("detail")
                             or payload.get("message") or "")
            if not detail:
                detail = str(resp.get("text") or "")[:200]
            self._reason = f"ERROR: HTTP {status} {detail}".strip()
            logger.warning("[SARVAM] %s", self._reason)
            return ""

        text = str(payload.get("transcript") or "").strip()
        self._last_language_detected = str(payload.get("language_code") or "")
        self._reason = "ready (experimental/benchmark-only)"
        return text


    def stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000,
               language: Optional[str] = None) -> str:
        """Accumulate chunks and return the last PARTIAL transcript.

        Sarvam REST returns no partial hypotheses, and this provider
        deliberately does not issue one HTTP request per audio chunk (slow,
        costly, and not a real partial). Partials stay owned by the local
        faster-whisper path; transcribe() remains the only real result.
        """
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._stream_buffer.append(np.asarray(audio_chunk, dtype=np.float32))
        return self._stream_text

    def reset(self) -> None:
        self._stream_buffer = []
        self._stream_text = ""

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "reason": self._reason,
            "provider": self.name,
            "experimental": True,
            "production_routing": False,
            "model": self._model,
            "endpoint": self._endpoint,
            "api_key_set": bool(self._api_key.strip()),
            "requests": self._requests,
            "errors": self._errors,
        }

    def metrics(self) -> Dict[str, Any]:
        avg = (self._total_latency_s / self._requests) if self._requests else 0.0
        return {
            "model": self.name,
            "experimental": True,
            "total_requests": self._requests,
            "errors": self._errors,
            "avg_latency_s": avg,
            "last_language_code": self._last_language_detected,
            "api_key_set": bool(self._api_key.strip()),
            **get_gpu_stats(),
            **get_cpu_ram_stats(),
        }


#: Module-level marker, mirroring the class attributes.
PRODUCTION_ROUTING_ENABLED = False
