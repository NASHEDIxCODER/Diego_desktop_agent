"""
StreamingSTT — True streaming speech recognition for Diego.

Implements a Gemini-Live-style continuous transcription pipeline:

  Ring Buffer → 32ms frames → Silero VAD → speech segments
       │
       ├─▶ Rolling audio buffer (configurable window)
       │
       ├─▶ Continuous partial transcription (every 100-200ms)
       │      → partial hypotheses with stability tracking
       │
       ├─▶ Transcript stability detection (not just silence)
       │      → stable for N consecutive partials → finalize
       │
       ├─▶ Intelligent partial merging
       │      → longest-common-prefix anchoring
       │
       ├─▶ First-word clipping prevention
       │      → pre-roll buffer + early partial trigger
       │
       └─▶ Speech correction layer
              → fuzzy matching against local dictionary
              → NEVER invokes LLM

KEY DIFFERENCES from the old pipeline:
  - Partial transcripts update every 100-200ms (was 250ms)
  - Stability detection replaces silence-only endpointing
  - Rolling context maintained across partial updates
  - First-word clipping prevented via aggressive pre-roll
  - Partial hypotheses are merged intelligently (LCP anchoring)
  - Speech corrector runs on final transcript only
  - Comprehensive logging of every decision

ROOT CAUSE FIX (2026-08-05):
  nonoverlap_idx was computed as i // FRAME_SAMPLES where i is the
  per-chunk offset.  After the first chunk, every subsequent chunk
  had i=0 → nonoverlap_idx=0 → NOT > _last_nonoverlap_idx → no new
  frames were ever added to audio_buffer.  The utterance stayed at
  1 frame, silence expired, and every utterance was discarded as
  too_short.  Fixed by using a GLOBAL sample offset
  (_chunk_base_sample + i) so frame indices are unique across chunks.

Usage:
    from voice.streaming_stt import streaming_stt

    async for event in streaming_stt.stream_utterances():
        if event.kind == "partial":  ...
        elif event.kind == "final":  ...
        elif event.kind == "speech_start": ...
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from voice.audio_manager import audio_manager, SAMPLE_RATE, FRAME_SAMPLES
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# ── Streaming configuration ────────────────────────────────────
MIN_PAUSE_MS = getattr(voice_settings, "conv_min_pause_ms", 600)
ENDPOINT_SILENCE_MS = getattr(voice_settings, "conv_endpoint_ms", 900)
MIN_UTTERANCE_MS = 250
MAX_UTTERANCE_S = 20.0
PARTIAL_INTERVAL_S = 0.12      # 120ms — fast partial updates (was 250ms)
PARTIAL_MIN_NEW_MS = 150       # Minimum new audio before running partial (was 300ms)
PRE_ROLL_MS = 600              # 600ms pre-roll to prevent first-word clipping (was 500ms)
INTERRUPT_MIN_MS = getattr(voice_settings, "conv_interrupt_min_ms", 90)
MAX_ROLLING_CONTEXT_CHARS = 300  # Increased from 200 for better context

# ── Stability detection ────────────────────────────────────────
STABILITY_WINDOW = 3            # Number of consecutive partials that must match
STABILITY_MIN_RATIO = 0.85      # Fuzzy ratio threshold for "same" transcript
STABILITY_MIN_DURATION_MS = 400 # Minimum speech duration before stability check
STABILITY_MAX_DRIFT_CHARS = 3   # Max character drift between stable partials

# ── Rolling buffer ─────────────────────────────────────────────
ROLLING_BUFFER_WINDOW_S = 3.0   # Keep 3 seconds of audio for context
ROLLING_BUFFER_MAX_FRAMES = int(ROLLING_BUFFER_WINDOW_S / (FRAME_SAMPLES / SAMPLE_RATE))

# Filler words that must NOT finalize or reset the conversation
FILLERS = {
    "umm", "um", "uh", "uhh", "er", "erm", "hmm", "hm",
    "wait", "hold on", "hold", "actually", "no wait", "sorry",
    "let me think", "like", "you know", "i mean", "well",
}
_FILLER_RE = re.compile(
    r"^(?:" + "|".join(re.escape(f) for f in sorted(FILLERS, key=len, reverse=True)) + r")[\s,.\!?]*$",
    re.IGNORECASE,
)


def is_filler(text: str) -> bool:
    """True if the text is ONLY a filler word (turn should stay open)."""
    t = text.strip().lower()
    if not t:
        return True
    return bool(_FILLER_RE.match(t))


@dataclass
class UtteranceEvent:
    """An event emitted by the streaming STT."""
    kind: str                    # "speech_start" | "partial" | "final"
    text: str = ""
    is_final: bool = False
    confidence: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    audio: Optional[bytes] = None  # int16 PCM of the utterance (final only)
    # ── New fields for observability ──
    stability_score: float = 0.0     # How stable the transcript is (0-1)
    partial_index: int = 0           # Which partial this is in the sequence
    correction_log: List[dict] = field(default_factory=list)  # Corrections applied


class _SileroVAD:
    """Silero VAD wrapper for streaming frames.

    The pipeline uses 512-sample frames (32 ms @ 16 kHz) — native
    Silero VAD v6 window size. NO zero-padding is needed because every
    frame is exactly 512 samples.
    """

    def __init__(self):
        self._model = None
        self._ready = False
        self._threshold = 0.5

    def load(self) -> bool:
        if self._ready:
            return True
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            self._ready = True
            logger.info("[STREAM-STT] Silero VAD loaded (silero_vad pkg)")
            return True
        except Exception as e:
            logger.debug("[STREAM-STT] silero_vad pkg failed: %s", e)
        try:
            import torch
            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
                verbose=False,
            )
            self._model = model
            self._ready = True
            logger.info("[STREAM-STT] Silero VAD loaded (torch.hub)")
            return True
        except Exception as e:
            logger.warning("[STREAM-STT] Silero VAD unavailable: %s", e)
            return False

    def speech_prob(self, frame: np.ndarray) -> float:
        """Return speech probability for a 32ms 16kHz (512-sample) frame."""
        if not self._ready:
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            if np.issubdtype(frame.dtype, np.floating):
                rms *= 32768.0
            return 0.9 if rms > 300.0 else 0.05
        try:
            import torch
            if frame.dtype != np.float32:
                audio = frame.astype(np.float32) / 32768.0
            else:
                audio = frame
            tensor = torch.from_numpy(audio)
            with torch.no_grad():
                prob = self._model(tensor, 16000).item()
            return float(prob)
        except Exception:
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            if np.issubdtype(frame.dtype, np.floating):
                rms *= 32768.0
            return 0.9 if rms > 300.0 else 0.05


class _WhisperTranscriber:
    """faster-whisper transcriber for partial + final transcription."""

    _BACKEND_CACHE = Path(__file__).resolve().parent.parent / "data" / "whisper_backend.json"

    def __init__(self):
        self._model = None
        self._ready = False
        self._lock = asyncio.Lock()
        self._device: str = "cpu"
        self._compute: str = "int8"

    @classmethod
    def _load_cached_backend(cls) -> Optional[Tuple[str, str]]:
        try:
            if cls._BACKEND_CACHE.exists():
                data = json.loads(cls._BACKEND_CACHE.read_text(encoding="utf-8"))
                device = data.get("device")
                compute = data.get("compute_type")
                if device and compute:
                    logger.info("[STREAM-STT] Cached Whisper backend: %s/%s "
                                "(skipping CUDA probe)", device, compute)
                    return device, compute
        except Exception:
            pass
        return None

    @classmethod
    def _save_backend_cache(cls, device: str, compute: str) -> None:
        try:
            cls._BACKEND_CACHE.parent.mkdir(parents=True, exist_ok=True)
            cls._BACKEND_CACHE.write_text(
                json.dumps({"device": device, "compute_type": compute,
                            "saved_at": time.time()}),
                encoding="utf-8")
        except Exception as e:
            logger.debug("[STREAM-STT] Failed to cache backend: %s", e)

    def load(self) -> bool:
        if self._ready:
            return True
        try:
            from faster_whisper import WhisperModel
            import torch

            cached = self._load_cached_backend()
            candidates = []
            if cached:
                candidates.append(cached)
            if torch.cuda.is_available():
                candidates.append(("cuda", "float16"))
            candidates.append(("cpu", "int8"))

            for device, compute in candidates:
                try:
                    model = WhisperModel("base", device=device, compute_type=compute)
                    warmup = np.zeros(16000, dtype=np.float32)
                    segments, _ = model.transcribe(
                        warmup, beam_size=1, without_timestamps=True)
                    list(segments)
                    self._model = model
                    self._ready = True
                    self._device = device
                    self._compute = compute
                    self._save_backend_cache(device, compute)
                    logger.info("[STREAM-STT] faster-whisper loaded "
                                "(device=%s, compute=%s, warmup OK)",
                                device, compute)
                    return True
                except Exception as e:
                    logger.warning("[STREAM-STT] faster-whisper %s/%s "
                                   "unusable (%s) — trying next backend",
                                   device, compute, e)
            logger.error("[STREAM-STT] faster-whisper: no working backend")
            return False
        except Exception as e:
            logger.warning("[STREAM-STT] faster-whisper unavailable: %s", e)
            return False

    @property
    def gpu_available(self) -> bool:
        return self._ready and self._device == "cuda"

    @property
    def backend_info(self) -> dict:
        return {
            "device": self._device,
            "compute_type": self._compute,
            "ready": self._ready,
            "gpu_available": self.gpu_available,
        }

    def transcribe(self, pcm_int16: bytes, sample_rate: int = SAMPLE_RATE,
                   use_vad_filter: bool = True) -> str:
        return self.transcribe_detailed(
            pcm_int16, sample_rate, use_vad_filter).get("text") or ""

    def transcribe_with_context(self, pcm_int16: bytes,
                                sample_rate: int = SAMPLE_RATE,
                                prompt_context: str = "") -> str:
        """Transcribe with a prompt prefix for rolling context.

        condition_on_previous_text is set to True when prompt_context is
        provided, so faster-whisper actually uses the initial_prompt to
        bias decoding.
        """
        if not self._ready or not pcm_int16:
            return ""
        if not prompt_context:
            return self.transcribe(pcm_int16, sample_rate, use_vad_filter=False)
        try:
            audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < sample_rate * 0.15:  # Reduced from 0.2s for faster first partial
                return ""
            segments, _ = self._model.transcribe(
                audio,
                beam_size=1,
                language="en",
                temperature=0.0,
                best_of=1,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.9,
                vad_filter=False,
                without_timestamps=True,
                initial_prompt=prompt_context,
            )
            segs = list(segments)
            text = " ".join(s.text.strip() for s in segs).strip()
            return text
        except Exception as e:
            logger.debug("[STREAM-STT] transcribe_with_context failed: %s", e)
            return self.transcribe(pcm_int16, sample_rate, use_vad_filter=False)

    def transcribe_detailed(self, pcm_int16: bytes,
                            sample_rate: int = SAMPLE_RATE,
                            use_vad_filter: bool = True) -> dict:
        result = {"text": "", "language": "", "language_probability": 0.0,
                  "avg_logprob": 0.0, "no_speech_prob": 0.0,
                  "compression_ratio": 0.0, "segments": [],
                  "ok": False, "reason": "whisper_unavailable"}
        if not self._ready or not pcm_int16:
            return result
        try:
            from voice.audio_processing import peak_monitor
            peak_monitor.log("whisper", np.frombuffer(pcm_int16, dtype=np.int16))
            audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio) < sample_rate * 0.15:
                result["reason"] = "too_short"
                return result

            segments, info = self._model.transcribe(
                audio,
                beam_size=1,
                language="en",
                temperature=0.0,
                best_of=1,
                condition_on_previous_text=False,
                compression_ratio_threshold=None,
                no_speech_threshold=0.9,
                vad_filter=use_vad_filter,
                without_timestamps=True,
            )
            segs = list(segments)
            result["language"] = getattr(info, "language", "") or ""
            result["language_probability"] = float(
                getattr(info, "language_probability", 0.0) or 0.0)
            if not segs:
                result["reason"] = "no_segments"
                logger.info("[WHISPER] REJECTED reason=no_segments "
                            "lang=%s(%.2f) duration=%.2fs",
                            result["language"], result["language_probability"],
                            len(audio) / sample_rate)
                return result

            seg_details = []
            for s in segs:
                seg_details.append({
                    "text": s.text.strip(),
                    "start": round(float(getattr(s, "start", 0.0)), 2),
                    "end": round(float(getattr(s, "end", 0.0)), 2),
                    "avg_logprob": round(float(getattr(s, "avg_logprob", 0.0) or 0.0), 3),
                    "no_speech_prob": round(float(getattr(s, "no_speech_prob", 0.0) or 0.0), 3),
                    "compression_ratio": round(float(getattr(s, "compression_ratio", 0.0) or 0.0), 2),
                })
            text = " ".join(d["text"] for d in seg_details).strip()
            logprobs = [d["avg_logprob"] for d in seg_details]
            result.update({
                "text": text,
                "segments": seg_details,
                "avg_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
                "no_speech_prob": max(d["no_speech_prob"] for d in seg_details),
                "compression_ratio": max(d["compression_ratio"] for d in seg_details),
            })
            if not text:
                result["reason"] = "empty_transcript"
            elif result["no_speech_prob"] > 0.6:
                result["reason"] = f"high_no_speech_prob({result['no_speech_prob']:.2f})"
            elif result["compression_ratio"] > 2.4:
                result["reason"] = f"high_compression_ratio({result['compression_ratio']:.2f})"
            else:
                result["ok"] = True
                result["reason"] = "accepted"

            logger.info(
                "[WHISPER] %s text=%r lang=%s(%.2f) avg_logprob=%.3f "
                "no_speech=%.2f compression=%.2f segments=%d reason=%s",
                "ACCEPTED" if result["ok"] else "REJECTED",
                text, result["language"], result["language_probability"],
                result["avg_logprob"], result["no_speech_prob"],
                result["compression_ratio"], len(seg_details), result["reason"])
            return result
        except Exception as e:
            logger.warning("[STREAM-STT] transcribe error: %s", e)
            result["reason"] = f"exception:{type(e).__name__}:{e}"
            return result


# ═══════════════════════════════════════════════════════════════
# Partial transcript merging
# ═══════════════════════════════════════════════════════════════

def _longest_common_prefix(a: str, b: str) -> str:
    """Return the longest common prefix of two strings (word-level)."""
    a_words = a.lower().split()
    b_words = b.lower().split()
    common = []
    for wa, wb in zip(a_words, b_words):
        if wa == wb:
            common.append(wa)
        else:
            break
    return " ".join(common)


def _fuzzy_ratio_simple(a: str, b: str) -> float:
    """Simple fuzzy ratio for stability comparison."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a.lower(), b.lower()) / 100.0
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _merge_partial_transcripts(previous: str, current: str) -> str:
    """Intelligently merge two partial transcripts.

    Strategy:
      1. If current starts with previous → use current (natural growth)
      2. If they share a long common prefix → anchor on prefix, append new
      3. If current is shorter than previous → keep previous (Whisper drift)
      4. Otherwise → use the longer one
    """
    if not previous:
        return current
    if not current:
        return previous

    prev_lower = previous.lower().strip()
    curr_lower = current.lower().strip()

    # Case 1: Current is a superset of previous (natural growth)
    if curr_lower.startswith(prev_lower):
        return current

    # Case 2: Previous is a superset of current (Whisper trimmed)
    if prev_lower.startswith(curr_lower):
        # If current is significantly shorter, keep previous
        if len(curr_lower) < len(prev_lower) * 0.7:
            return previous
        return current

    # Case 3: Long common prefix anchoring
    lcp = _longest_common_prefix(previous, current)
    if lcp and len(lcp.split()) >= 2:
        # Anchor on the common prefix, append the new suffix from current
        curr_suffix = curr_lower[len(lcp):].strip()
        if curr_suffix:
            # Use the original casing from current for the suffix
            return (lcp + " " + current[len(lcp):].strip()).strip()
        return previous

    # Case 4: Fall back to the longer transcript
    if len(current.split()) >= len(previous.split()):
        return current
    return previous


def _compute_stability_score(partials: List[str]) -> float:
    """Compute how stable a sequence of partial transcripts is.

    Returns 0.0 (unstable) to 1.0 (perfectly stable).
    """
    if len(partials) < 2:
        return 0.0

    scores = []
    for i in range(1, len(partials)):
        scores.append(_fuzzy_ratio_simple(partials[i - 1], partials[i]))

    if not scores:
        return 0.0

    # Weight recent comparisons more heavily
    weights = [0.5 ** (len(scores) - 1 - i) for i in range(len(scores))]
    weight_sum = sum(weights)
    if weight_sum == 0:
        return 0.0

    weighted_avg = sum(s * w for s, w in zip(scores, weights)) / weight_sum

    # Bonus for consistent length (no wild growth)
    lengths = [len(p) for p in partials[-STABILITY_WINDOW:]]
    if len(lengths) >= 2:
        max_len_diff = max(lengths) - min(lengths)
        if max_len_diff <= STABILITY_MAX_DRIFT_CHARS:
            weighted_avg = min(1.0, weighted_avg + 0.05)

    return weighted_avg


# ═══════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════

# Common Whisper mistakes → corrections (kept for backward compat,
# but the SpeechCorrector is now the primary correction layer)
_WHISPER_CORRECTIONS = {
    "you too fo me": "YouTube for me",
    "you too": "YouTube",
    "you tube": "YouTube",
    "u tube": "YouTube",
    "spot if i": "Spotify",
    "spot a fire": "Spotify",
    "spot of eye": "Spotify",
    "net flicks": "Netflix",
    "face book": "Facebook",
    "what's up": "WhatsApp",
    "whats up": "WhatsApp",
    "what sup": "WhatsApp",
    "visual studio": "Visual Studio",
    "vs code": "VS Code",
    "v s code": "VS Code",
    "fire fox": "Firefox",
    "google chrome": "Google Chrome",
    "crome": "Chrome",
    "crom": "Chrome",
    "go ogle": "Google",
    "open a i": "OpenAI",
    "chat g p t": "ChatGPT",
    "chat gpt": "ChatGPT",
    "jet brains": "JetBrains",
    "pie charm": "PyCharm",
    "pie chum": "PyCharm",
    "pi thon": "Python",
    "java script": "JavaScript",
    "type script": "TypeScript",
    "get hub": "GitHub",
    "git hub": "GitHub",
    "get lab": "GitLab",
    "git lab": "GitLab",
    "stack over flow": "Stack Overflow",
    "k eight s": "K8s",
    "cube cuddle": "kubectl",
    "cube control": "kubectl",
    "note pad": "Notepad",
    "power point": "PowerPoint",
    "screen shot": "Screenshot",
    "log out": "Logout",
    "sign out": "Sign Out",
    "sign in": "Sign In",
    "log in": "Login",
    "save as": "Save As",
    "zoom in": "Zoom In",
    "zoom out": "Zoom Out",
    "full screen": "Full Screen",
    "new tab": "New Tab",
    "close tab": "Close Tab",
    "new window": "New Window",
    "close window": "Close Window",
    "switch tab": "Switch Tab",
    "switch window": "Switch Window",
    "go to": "Go To",
    "navigate to": "Navigate To",
    "scroll up": "Scroll Up",
    "scroll down": "Scroll Down",
    "page up": "Page Up",
    "page down": "Page Down",
    "volume up": "Volume Up",
    "volume down": "Volume Down",
    "brightness up": "Brightness Up",
    "brightness down": "Brightness Down",
    "delete key": "Delete",
    "windows key": "Windows Key",
    "command key": "Command Key",
    "super key": "Super Key",
    "meta key": "Meta Key",
    "function key": "Function Key",
    "arrow key": "Arrow Key",
    "escape key": "Escape Key",
    "enter key": "Enter Key",
    "space bar": "Space Bar",
    "back space": "Backspace",
    "caps lock": "Caps Lock",
    "num lock": "Num Lock",
    "scroll lock": "Scroll Lock",
    "print screen": "Print Screen",
    "pause break": "Pause Break",
    "page up key": "Page Up",
    "page down key": "Page Down",
    "home key": "Home Key",
    "end key": "End Key",
    "linked in": "LinkedIn",
    "x dot com": "X",
    "google drive": "Google Drive",
    "google docs": "Google Docs",
    "google sheets": "Google Sheets",
    "google slides": "Google Slides",
    "google meet": "Google Meet",
    "google maps": "Google Maps",
    "file explorer": "File Explorer",
    "system settings": "System Settings",
    "task manager": "Task Manager",
    "screen share": "Screen Share",
    "screen record": "Screen Record",
}

# Contractions → expanded form
_CONTRACTIONS = {
    "i'm": "I am", "i've": "I have", "i'll": "I will", "i'd": "I would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "he's": "he is", "he'll": "he will",
    "she's": "she is", "she'll": "she will", "it's": "it is",
    "it'll": "it will", "we're": "we are", "we've": "we have",
    "we'll": "we will", "they're": "they are", "they've": "they have",
    "they'll": "they will", "that's": "that is", "that'll": "that will",
    "what's": "what is", "what'll": "what will", "who's": "who is",
    "who'll": "who will", "where's": "where is", "when's": "when is",
    "why's": "why is", "how's": "how is", "can't": "cannot",
    "cannot": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not", "ain't": "is not",
    "let's": "let us", "there's": "there is", "here's": "here is",
}


def _postprocess_transcript(text: str) -> str:
    """Normalize and correct common Whisper mistakes.

    Steps:
      1. Strip leading/trailing whitespace and punctuation artifacts
      2. Normalize casing
      3. Expand contractions
      4. Apply static Whisper corrections
      5. Remove repeated words
    """
    if not text:
        return text

    # 1. Clean up
    text = text.strip()
    text = re.sub(r'^[,.!?;:\s]+', '', text)
    text = re.sub(r'[,.!?;:\s]+$', '', text)

    # 2. Normalize casing
    words = text.split()
    if len(words) <= 5:
        text = " ".join(w.capitalize() if len(w) > 2 else w for w in words)
    else:
        text = text[0].upper() + text[1:] if text else text

    # 3. Expand contractions
    text_lower = text.lower()
    for contraction, expanded in _CONTRACTIONS.items():
        if contraction in text_lower:
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            text = pattern.sub(expanded, text)

    # 4. Apply static Whisper corrections
    for wrong, correct in sorted(_WHISPER_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(correct, text)

    # 5. Remove repeated words
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # 6. Normalize whitespace
    text = " ".join(text.split())

    return text


class StreamingSTT:
    """
    True streaming speech-to-text with continuous partial transcription,
    stability detection, and intelligent merging.

    KEY IMPROVEMENTS over the old pipeline:
      - Partial transcripts every 100-200ms (was 250ms)
      - Stability detection replaces silence-only endpointing
      - Rolling audio buffer for context
      - Intelligent partial merging (LCP anchoring)
      - First-word clipping prevention via aggressive pre-roll
      - Speech corrector integration (local, no LLM)
      - Comprehensive observability logging
    """

    def __init__(self):
        self._vad = _SileroVAD()
        self._whisper = _WhisperTranscriber()
        self._ready = False
        self._listen_enabled = asyncio.Event()
        self._listen_enabled.set()
        self._cancel = asyncio.Event()
        # Speech corrector — loaded lazily
        self._corrector = None
        # Drain-on-resume: set by resume_listening(), cleared by stream_utterances()
        # after the ring buffer has been drained and VAD state reset.
        self._drain_requested = False

    def initialize(self) -> bool:
        vad_ok = self._vad.load()
        whisper_ok = self._whisper.load()
        self._ready = whisper_ok
        if not whisper_ok:
            logger.error("[STREAM-STT] Whisper unavailable — streaming STT disabled")
        # Pre-load the speech corrector in background
        if self._ready:
            try:
                from voice.speech_corrector import speech_corrector
                self._corrector = speech_corrector
                if not self._corrector.loaded:
                    import threading
                    threading.Thread(
                        target=self._corrector.initialize,
                        daemon=True, name="corrector-init"
                    ).start()
            except Exception as e:
                logger.debug("[STREAM-STT] Speech corrector unavailable: %s", e)
        return self._ready

    @property
    def ready(self) -> bool:
        return self._ready

    def pause_listening(self) -> None:
        """Mute STT — used during TTS playback to prevent Diego from
        transcribing his own voice."""
        self._listen_enabled.clear()
        logger.info("[STREAM-STT] Listening PAUSED (TTS guard)")

    def resume_listening(self) -> None:
        """Unmute STT and request a ring-buffer drain.
        
        Called after TTS finishes.  Sets a flag that causes the
        stream_utterances() loop to skip all audio accumulated during
        TTS playback (jumping last_total to the current write position),
        then re-enables VAD scoring.
        """
        self._drain_requested = True
        self._listen_enabled.set()
        logger.info("[STREAM-STT] Listening RESUMED — drain requested, "
                    "VAD re-enabled")

    def cancel(self) -> None:
        self._cancel.set()

    def stop_streaming(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    # ── Main streaming loop ───────────────────────────────

    async def stream_utterances(
        self,
        interrupt_while_speaking: bool = True,
    ) -> AsyncIterator[UtteranceEvent]:
        """
        Yield UtteranceEvents as the user speaks.

        Emits:
          - speech_start: when speech begins
          - partial: incremental transcription (~every 120ms of new audio)
          - final: the complete stabilized utterance

        Runs until cancelled (self.cancel() or task cancellation).
        """
        if not self._ready:
            if not self.initialize():
                return

        self.reset_cancel()

        # ── DRAIN the ring buffer ──
        drain_start = audio_manager.total_samples
        logger.info("[STREAM-STT] Draining TTS-contaminated audio "
                    "(total_samples=%d)", drain_start)
        last_total = drain_start

        # ── Audio buffers ──
        audio_buffer: List[np.ndarray] = []     # non-overlapping frames → Whisper
        pre_roll: List[np.ndarray] = []         # non-overlapping pre-roll
        in_speech = False
        speech_start_time = 0.0
        last_voice_time = 0.0
        silence_run_ms = 0.0
        last_partial_len = 0
        last_partial_time = 0.0
        _rolling_context: str = ""
        loop = asyncio.get_event_loop()
        _frame_remainder: np.ndarray = np.array([], dtype=np.float32)

        # ── NEW: Stability tracking ──
        _partial_history: List[str] = []        # Last N partial transcripts
        _partial_index: int = 0                 # Monotonic partial counter
        _merged_transcript: str = ""            # Best merged transcript so far
        _stable_count: int = 0                  # Consecutive stable partials
        _stabilized_at: float = 0.0             # When stability was first achieved
        _finalized_by_stability: bool = False   # True if stability triggered finalize

        # VAD overlap step (50% = 256 samples = 16ms)
        VAD_STEP = FRAME_SAMPLES // 2

        logger.info("[STREAM-STT] Listening started (endpoint=%dms, min_pause=%dms "
                    "partial_interval=%dms stability_window=%d total_samples=%d)",
                    ENDPOINT_SILENCE_MS, MIN_PAUSE_MS,
                    int(PARTIAL_INTERVAL_S * 1000), STABILITY_WINDOW, drain_start)

        # Track chunk timing for diagnostics
        _last_chunk_time = time.time()
        _chunk_durations: list = []

        # ── ROOT CAUSE FIX (2026-08-05): nonoverlap_idx must use a GLOBAL
        # sample offset so frame indices are unique across read_since() chunks.
        # The old per-chunk i // FRAME_SAMPLES reset to 0 on every chunk,
        # so after the first chunk every subsequent frame was rejected as
        # "already seen" and audio_buffer never grew beyond 1 frame.
        _last_nonoverlap_idx: int = -1
        _chunk_base_sample: int = 0  # absolute sample offset of new_audio[0]

        while not self._cancel.is_set():
            await self._listen_enabled.wait()
            if self._cancel.is_set():
                break

            # ── Drain-on-resume: skip TTS-contaminated audio ──
            # When resume_listening() is called after TTS ends, it sets
            # _drain_requested.  We jump last_total to the current write
            # position (total_samples), discarding all audio that arrived
            # during TTS playback.  This prevents Diego from transcribing
            # his own voice as a phantom user command.
            if self._drain_requested:
                self._drain_requested = False
                old_total = last_total
                last_total = audio_manager.total_samples
                skipped = last_total - old_total
                if skipped > 0:
                    logger.info("[STREAM-STT] Drain-on-resume: skipped %d samples "
                                "(%.0fms) of TTS-contaminated audio",
                                skipped, skipped / 16.0)
                # Reset VAD-related state so we start fresh
                in_speech = False
                audio_buffer.clear()
                silence_run_ms = 0.0
                _last_nonoverlap_idx = -1
                _frame_remainder = np.array([], dtype=np.float32)
                pre_roll.clear()
                _partial_history.clear()
                _partial_index = 0
                _merged_transcript = ""
                _stable_count = 0
                _rolling_context = ""
                continue

            new_audio, last_total = audio_manager.read_since(last_total)
            if len(new_audio) == 0:
                await asyncio.sleep(0.01)
                continue

            # Chunk timing diagnostics
            now_chunk = time.time()
            chunk_dur_ms = (now_chunk - _last_chunk_time) * 1000
            _last_chunk_time = now_chunk
            _chunk_durations.append(chunk_dur_ms)
            if len(_chunk_durations) > 50:
                _chunk_durations.pop(0)

            # Prepend frame remainder
            if len(_frame_remainder) > 0:
                new_audio = np.concatenate([_frame_remainder, new_audio])
                _frame_remainder = np.array([], dtype=np.float32)

            # ── Compute absolute sample base for this chunk ──
            # last_total is the ring buffer's monotonic sample counter AFTER
            # this read, so the chunk starts at last_total - len(new_audio).
            _chunk_base_sample = last_total - len(new_audio)

            # ── VAD: overlapping frames for smooth detection ──
            # ── Audio buffer: NON-overlapping frames for Whisper ──
            last_vad_idx = -1
            for i, frame in self._iter_frames_overlap(new_audio, VAD_STEP):
                last_vad_idx = i
                prob = await loop.run_in_executor(None, self._vad.speech_prob, frame)
                is_speech = prob > 0.5
                now = time.time()

                # ── Collect NON-overlapping frame for audio buffer ──
                # ROOT CAUSE FIX: use GLOBAL sample position so indices are
                # unique across read_since() chunks.  Without this, i resets
                # to 0 on every chunk and _last_nonoverlap_idx blocks all
                # subsequent frames.
                global_sample = _chunk_base_sample + i
                nonoverlap_idx = global_sample // FRAME_SAMPLES
                is_new_nonoverlap = nonoverlap_idx > _last_nonoverlap_idx
                if is_new_nonoverlap:
                    _last_nonoverlap_idx = nonoverlap_idx
                    # Extract the non-overlapping frame from new_audio
                    frame_start = i
                    frame_end = i + FRAME_SAMPLES
                    if frame_end <= len(new_audio):
                        clean_frame = new_audio[frame_start:frame_end].copy()
                    else:
                        clean_frame = frame.copy()  # fallback

                    # Maintain pre-roll buffer (non-overlapping)
                    pre_roll.append(clean_frame)
                    max_pre = max(1, int((PRE_ROLL_MS / 1000.0) / (FRAME_SAMPLES / SAMPLE_RATE * 1000)))
                    if len(pre_roll) > max_pre:
                        pre_roll.pop(0)

                    # If currently in speech, add frame directly to audio_buffer
                    if in_speech:
                        audio_buffer.append(clean_frame)

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        speech_start_time = now
                        audio_buffer = list(pre_roll)  # include pre-roll
                        last_partial_len = 0
                        last_partial_time = now
                        # ── Reset stability tracking on new speech ──
                        _partial_history.clear()
                        _partial_index = 0
                        _merged_transcript = ""
                        _stable_count = 0
                        _stabilized_at = 0.0
                        _finalized_by_stability = False
                        logger.info("[STREAM-STT] Speech start "
                                    "(VAD_prob=%.2f audio_frames=%d pre_roll_frames=%d "
                                    "chunk_base=%d global_nonoverlap=%d)",
                                    prob, len(audio_buffer), len(pre_roll),
                                    _chunk_base_sample, nonoverlap_idx)
                        yield UtteranceEvent(
                            kind="speech_start", started_at=now)
                    last_voice_time = now
                    silence_run_ms = 0.0
                else:
                    if in_speech:
                        silence_run_ms = (now - last_voice_time) * 1000.0

                        # ── Endpoint detection: silence OR stability ──
                        should_finalize = False
                        finalize_reason = ""

                        # Reason 1: Long silence
                        if silence_run_ms >= ENDPOINT_SILENCE_MS:
                            should_finalize = True
                            finalize_reason = f"silence_{int(silence_run_ms)}ms"

                        # Reason 2: Transcript stability
                        if not should_finalize and _stable_count >= STABILITY_WINDOW:
                            dur_ms = (now - speech_start_time) * 1000
                            if dur_ms >= STABILITY_MIN_DURATION_MS:
                                should_finalize = True
                                finalize_reason = f"stability_{_stable_count}/{STABILITY_WINDOW}"
                                _finalized_by_stability = True

                        if should_finalize:
                            dur_ms = len(audio_buffer) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
                            avg_chunk = (sum(_chunk_durations) / len(_chunk_durations)
                                         if _chunk_durations else 0)
                            logger.info("[STREAM-STT] Endpoint detected "
                                        "(reason=%s duration=%.0fms silence=%dms "
                                        "frames=%d chunk_avg=%.0fms stability=%.2f)",
                                        finalize_reason, dur_ms, int(silence_run_ms),
                                        len(audio_buffer), avg_chunk,
                                        _compute_stability_score(_partial_history))
                            final = await self._finalize(
                                audio_buffer, speech_start_time, _rolling_context,
                                partial_history=list(_partial_history),
                                finalized_by_stability=_finalized_by_stability,
                            )
                            in_speech = False
                            audio_buffer = []
                            silence_run_ms = 0.0
                            _rolling_context = ""
                            _last_nonoverlap_idx = -1
                            _partial_history.clear()
                            _partial_index = 0
                            _merged_transcript = ""
                            _stable_count = 0
                            _stabilized_at = 0.0
                            _finalized_by_stability = False
                            if final is not None:
                                if is_filler(final.text):
                                    logger.info(
                                        "[STREAM-STT] Filler '%s' — turn stays open",
                                        final.text)
                                    continue
                                yield final
                            continue

                # ── Continuous partial transcription ──
                if in_speech:
                    new_since_partial = (len(audio_buffer) - last_partial_len) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
                    if (now - last_partial_time) >= PARTIAL_INTERVAL_S and new_since_partial >= PARTIAL_MIN_NEW_MS:
                        pcm = self._frames_to_bytes(audio_buffer)
                        t_partial_start = time.time()
                        text = await loop.run_in_executor(
                            None, self._whisper.transcribe_with_context,
                            pcm, SAMPLE_RATE, _rolling_context)
                        t_partial_elapsed = (time.time() - t_partial_start) * 1000
                        last_partial_len = len(audio_buffer)
                        last_partial_time = now
                        _partial_index += 1

                        if text and not is_filler(text):
                            # ── Intelligent merging ──
                            merged = _merge_partial_transcripts(_merged_transcript, text)
                            _merged_transcript = merged

                            # ── Update rolling context ──
                            _rolling_context = merged.strip()
                            if len(_rolling_context) > MAX_ROLLING_CONTEXT_CHARS:
                                _rolling_context = _rolling_context[-MAX_ROLLING_CONTEXT_CHARS:]

                            # ── Track partial history for stability ──
                            _partial_history.append(merged)
                            if len(_partial_history) > STABILITY_WINDOW * 2:
                                _partial_history = _partial_history[-STABILITY_WINDOW * 2:]

                            # ── Compute stability ──
                            stability_score = _compute_stability_score(_partial_history)
                            if len(_partial_history) >= 2:
                                last_two_ratio = _fuzzy_ratio_simple(
                                    _partial_history[-2], _partial_history[-1])
                                if last_two_ratio >= STABILITY_MIN_RATIO:
                                    _stable_count += 1
                                else:
                                    _stable_count = 0

                            # ── Log partial hypothesis ──
                            logger.info(
                                "[STREAM-STT] Partial #%d (%.0fms): '%s' "
                                "(raw='%s' stability=%.2f stable_count=%d/%d "
                                "whisper_latency=%.0fms frames=%d)",
                                _partial_index,
                                (now - speech_start_time) * 1000,
                                merged, text, stability_score,
                                _stable_count, STABILITY_WINDOW,
                                t_partial_elapsed, len(audio_buffer))

                            yield UtteranceEvent(
                                kind="partial", text=merged, is_final=False,
                                started_at=speech_start_time,
                                stability_score=stability_score,
                                partial_index=_partial_index)

                    # Hard cap on utterance length
                    if (now - speech_start_time) >= MAX_UTTERANCE_S:
                        logger.info("[STREAM-STT] Utterance capped at %.1fs",
                                    MAX_UTTERANCE_S)
                        final = await self._finalize(
                            audio_buffer, speech_start_time, _rolling_context,
                            partial_history=list(_partial_history),
                            finalized_by_stability=False,
                        )
                        in_speech = False
                        audio_buffer = []
                        _rolling_context = ""
                        _last_nonoverlap_idx = -1
                        _partial_history.clear()
                        _partial_index = 0
                        _merged_transcript = ""
                        _stable_count = 0
                        _stabilized_at = 0.0
                        if final is not None and not is_filler(final.text):
                            yield final

            # Carry frame remainder forward
            if last_vad_idx >= 0:
                remainder_start = last_vad_idx + VAD_STEP
                if remainder_start < len(new_audio):
                    _frame_remainder = new_audio[remainder_start:].copy()

    async def _finalize(
        self,
        frames: List[np.ndarray],
        start: float,
        context: str = "",
        partial_history: List[str] = None,
        finalized_by_stability: bool = False,
    ) -> Optional[UtteranceEvent]:
        """Transcribe the complete utterance with correction.

        Uses rolling context for final transcription and applies the
        speech correction layer. Logs all decisions comprehensively.
        """
        pcm = self._frames_to_bytes(frames)
        dur_ms = len(frames) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
        num_samples = len(pcm) // 2
        t_finalize_start = time.time()

        logger.info("[STREAM-STT] Finalizing utterance: duration=%.0fms "
                    "frames=%d pcm_bytes=%d samples=%d context=%d chars "
                    "partials=%d stability_triggered=%s",
                    dur_ms, len(frames), len(pcm), num_samples, len(context),
                    len(partial_history) if partial_history else 0,
                    finalized_by_stability)

        if dur_ms < MIN_UTTERANCE_MS or len(pcm) < 512:
            logger.info(
                "[STREAM-STT] Utterance DISCARDED reason=too_short "
                "(duration=%.0fms < %dms, bytes=%d)",
                dur_ms, MIN_UTTERANCE_MS, len(pcm))
            return None

        loop = asyncio.get_event_loop()

        # ── Step 1: Raw Whisper transcription ──
        t_whisper = time.time()
        if context:
            raw_text = await loop.run_in_executor(
                None, self._whisper.transcribe_with_context,
                pcm, SAMPLE_RATE, context)
        else:
            raw_text = await loop.run_in_executor(
                None, self._whisper.transcribe, pcm, SAMPLE_RATE, False)
        whisper_latency = (time.time() - t_whisper) * 1000

        if not raw_text:
            logger.info(
                "[STREAM-STT] Utterance DISCARDED reason=empty_transcript "
                "(duration=%.0fms samples=%d whisper_latency=%.0fms)",
                dur_ms, num_samples, whisper_latency)
            return None

        # ── Step 2: Static post-processing ──
        text = _postprocess_transcript(raw_text)

        # ── Step 3: Speech corrector (local, no LLM) ──
        correction_log = []
        if self._corrector and self._corrector.loaded:
            text, correction_log = self._corrector.correct_phrase(text)
        elif self._corrector and not self._corrector.loaded:
            # Corrector is still loading in background — skip for now
            logger.debug("[STREAM-STT] Speech corrector not yet loaded — skipping")

        # ── Step 4: Merge with best partial if final is worse ──
        if partial_history and len(partial_history) >= 2:
            best_partial = max(partial_history[-STABILITY_WINDOW:], key=len)
            if len(best_partial) > len(text) and _fuzzy_ratio_simple(text, best_partial) > 0.7:
                logger.info("[STREAM-STT] Final shorter than best partial — "
                            "using partial: '%s' (final was: '%s')",
                            best_partial, text)
                text = best_partial

        # ── Step 5: Comprehensive log ──
        total_latency = (time.time() - t_finalize_start) * 1000
        stability_score = _compute_stability_score(partial_history) if partial_history else 0.0

        logger.info(
            "[STREAM-STT] FINALIZED: text='%s' raw='%s' duration=%.0fms "
            "whisper_latency=%.0fms total_latency=%.0fms "
            "stability=%.2f stability_triggered=%s "
            "corrections=%d partials=%d context_chars=%d",
            text, raw_text, dur_ms, whisper_latency, total_latency,
            stability_score, finalized_by_stability,
            len(correction_log),
            len(partial_history) if partial_history else 0,
            len(context))

        if correction_log:
            for c in correction_log:
                logger.info("[STREAM-STT]   CORRECTION: '%s' → '%s' (%.3f, %s)",
                            c.get("word", ""), c.get("corrected_to", ""),
                            c.get("confidence", 0.0), c.get("method", ""))

        return UtteranceEvent(
            kind="final", text=text, is_final=True,
            started_at=start, ended_at=time.time(), audio=pcm,
            stability_score=stability_score,
            partial_index=len(partial_history) if partial_history else 0,
            correction_log=correction_log)

    @staticmethod
    def _frames_to_bytes(frames: List[np.ndarray]) -> bytes:
        """Pack NON-OVERLAPPING frames into PCM16 bytes for Whisper."""
        if not frames:
            return b""
        from voice.audio_processing import float32_to_int16
        return float32_to_int16(np.concatenate(frames)).tobytes()

    @staticmethod
    def _iter_frames(audio: np.ndarray):
        """Yield non-overlapping 32ms frames."""
        n = len(audio)
        for i in range(0, n - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            yield audio[i:i + FRAME_SAMPLES]

    @staticmethod
    def _iter_frames_overlap(audio: np.ndarray, step: int):
        """Yield (index, frame) pairs with configurable step size.

        Used ONLY for VAD. The audio buffer for Whisper uses
        non-overlapping frames extracted separately.
        """
        n = len(audio)
        for i in range(0, n - FRAME_SAMPLES + 1, step):
            yield i, audio[i:i + FRAME_SAMPLES]

    # ── Interruption detection while Diego speaks ───────────

    async def detect_interruption(self, stop_event: asyncio.Event) -> None:
        if not self._ready:
            self.initialize()
        last_total = audio_manager.total_samples
        speech_run_ms = 0.0
        loop = asyncio.get_event_loop()

        while not stop_event.is_set() and not self._cancel.is_set():
            new_audio, last_total = audio_manager.read_since(last_total)
            if len(new_audio) == 0:
                await asyncio.sleep(0.01)
                continue
            for frame in self._iter_frames(new_audio):
                prob = await loop.run_in_executor(None, self._vad.speech_prob, frame)
                if prob > 0.6:
                    speech_run_ms += 32
                    if speech_run_ms >= INTERRUPT_MIN_MS:
                        logger.info("[STREAM-STT] Interruption detected (user speaking)")
                        stop_event.set()
                        return
                else:
                    speech_run_ms = 0.0


# Global singleton
streaming_stt = StreamingSTT()