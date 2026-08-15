"""
CommandListener — Clean streaming speech-to-text for command recognition.

Replaces the old streaming_stt.py. Key differences:
  - Uses the UNIFIED VAD (voice/vad.py) — no duplicate Silero instance
  - Whisper receives continuously growing context windows (minimum ~800 ms)
  - Simple silence-based endpoint only — no stability detection, no LCP merging
  - No SpeechCorrector integration
  - Essential post-processing only (common Whisper mistakes)
  - Comprehensive runtime diagnostics

Architecture:
  Ring Buffer → 32ms frames → Unified VAD → speech segments
       │
       ├─▶ Rolling audio buffer (grows from 800ms to utterance end)
       │
       ├─▶ Partial transcription every ~200ms (with full context)
       │
       └─▶ Silence-based endpoint → final transcription

Usage:
    from voice.command_listener import command_listener

    async for event in command_listener.stream_utterances():
        if event.kind == "partial":  ...
        elif event.kind == "final":  ...
        elif event.kind == "speech_start": ...
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

from voice.audio_manager import audio_manager, SAMPLE_RATE, FRAME_SAMPLES
from voice.audio_processing import float32_to_int16
from voice.vad import unified_vad

logger = logging.getLogger(__name__)

# ── Command-capture tuning ─────────────────────────────────────
# Single source of truth for the post-wake command pipeline.
# Every value is runtime-configurable (replace `command_config` or
# mutate it) so there are no scattered magic numbers.
@dataclass
class CommandListenerConfig:
    # VAD speech threshold (Silero probability 0..1). Logging is
    # rate-limited by vad_log_interval_s so we don't spam per-frame.
    vad_speech_threshold: float = 0.5
    vad_log_interval_s: float = 0.5

    # Endpoint detection (milliseconds of audio, sample-accurate).
    min_context_ms: int = 800         # Minimum audio before Whisper sees anything
    endpoint_silence_ms: int = 1200   # Trailing silence that finalizes an utterance
    min_utterance_ms: int = 600       # Shorter than this → discard
    max_utterance_s: float = 20.0     # Hard cap on utterance length
    min_speech_ms: int = 600          # Minimum speech before endpoint
    min_silence_ms: int = 600         # Minimum silence before endpoint
    low_confidence_silence_ms: int = 2000  # Longer silence for low-confidence audio

    # Streaming partials.
    partial_interval_s: float = 0.20  # Seconds between partial updates
    pre_roll_ms: int = 400            # Pre-roll to prevent first-word clipping

    # Transcript stabilization.
    stability_required: int = 2       # Consecutive matching partials before finalize
    stability_similarity: float = 0.85
    stability_min_ms: int = 1200      # Minimum speech before stability matters
    stability_max_ms: int = 4000      # After this, finalize even if unstable

    # Confidence.
    # TASK 4: confidence is NO LONGER a hard accept/reject gate. It is one
    # piece of evidence combined with transcript length, garbage detection,
    # speech duration, and repetition/hallucination detection. The old
    # unconditional `confidence < -0.3 => discard` is REMOVED.
    confidence_threshold: float = -0.3  # kept for diagnostics only (not a gate)

    # TASK 3: partial decoding must NOT run every 200ms on tiny windows.
    # Use a larger rolling context and require >=1.0-1.5s of speech before
    # the first partial. Partials are HINTS ONLY and never finalize.
    partial_min_context_ms: int = 1200   # min speech before FIRST partial
    partial_interval_s: float = 0.35     # seconds between partial updates
    partial_rolling_context_s: float = 2.5  # rolling context window for partials

    # TASK 2: robust VAD — combined Silero + energy + duration evidence.
    use_robust_vad: bool = True

    # TASK 6: failure responses — never silently return to wake mode.
    failure_response_ms: int = 0

    # Interruption.
    interrupt_min_ms: int = 90

    def samples(self, ms: int) -> int:
        """Convert milliseconds to sample count at SAMPLE_RATE."""
        return int(SAMPLE_RATE * ms / 1000.0)


# Runtime-configurable singleton. Tests and diagnostics may swap it.
command_config = CommandListenerConfig()

# Whisper inference must NEVER block the conversation engine indefinitely
# (Phase 7). Partial transcriptions get a shorter budget than finals because
# they run more frequently and a stall there should not freeze the turn.
WHISPER_FINAL_TIMEOUT_S = 30.0
WHISPER_PARTIAL_TIMEOUT_S = 5.0

# ── Garbage transcript rejection ──────────────────────────────
# Whisper sometimes hallucinates short, low-confidence fragments.
# These patterns are unlikely to be real user commands.
MIN_TRANSCRIPT_WORDS = 1      # Minimum words in a final transcript
MIN_TRANSCRIPT_CHARS = 2      # Minimum characters (reject single chars like "I", "a")
MAX_TRANSCRIPT_CHARS = 100    # Sanity cap on transcript length
GARBAGE_PATTERNS = [
    r"^i'?m? (sorry|gonna|going to|not sure|afraid|just|so)",
    r"^i don'?t (know|think|have|understand|see)",
    r"^i (can'?t|cannot)",
    r"^i (was|am|will)",
    r"^oh[,.\s]?$",
    r"^uh[,.\s]?$",
    r"^um[,.\s]?$",
    r"^hmm[,.\s]?$",
    r"^thank you[,.\s]?$",
    r"^that'?s (a|an|not|all|what)",
    r"^this is( |$)(a|an|the|not)?",
    r"^there is( |$)(a|an|no)?",
    r"^it'?s( |$)(a|an|the|not|just|like)?",
    r"^let me (think|see|check|look|try|know)",
    r"^(i|i'm|im|you|it|that|this)[.!?]?$",
]

# ── Filler words ───────────────────────────────────────────────
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
    """True if the text is ONLY a filler word."""
    t = text.strip().lower()
    if not t:
        return True
    return bool(_FILLER_RE.match(t))


def is_garbage(text: str) -> bool:
    """True if the transcript is unlikely to be a real user command.

    Rejects:
      - Empty or very short text
      - Whisper hallucinated phrases ("I'm sorry", "I don't know", etc.)
      - Single-word fragments that are unlikely commands
      - Very low-confidence transcripts (handled separately by confidence gate)

    NOTE: This is NOT a replacement for the confidence gate. Garbage
    transcripts can have high confidence (Whisper is confident it heard
    "I'm sorry" even when the user said nothing). This function catches
    those patterns by content, not by confidence.
    """
    t = text.strip().lower()
    if not t:
        return True
    if len(t) < MIN_TRANSCRIPT_CHARS:
        return True
    # Check against known garbage patterns
    import re as _re
    for pattern in GARBAGE_PATTERNS:
        if _re.match(pattern, t):
            return True
    return False


# ── Essential Whisper corrections (kept minimal) ───────────────
_WHISPER_CORRECTIONS = {
    "you too fo me": "YouTube for me",
    "you too": "YouTube",
    "you tube": "YouTube",
    "spot if i": "Spotify",
    "spot a fire": "Spotify",
    "net flicks": "Netflix",
    "face book": "Facebook",
    "visual studio": "Visual Studio",
    "vs code": "VS Code",
    "v s code": "VS Code",
    "fire fox": "Firefox",
    "google chrome": "Google Chrome",
    "crome": "Chrome",
    "crom": "Chrome",
    "pie charm": "PyCharm",
    "pie chum": "PyCharm",
    "get hub": "GitHub",
    "git hub": "GitHub",
    "ghost line": "GhostLine",
    "ghostline": "GhostLine",
    "ghost lime": "GhostLine",
    "ghost lyne": "GhostLine",
}


def _postprocess(text: str) -> str:
    """Minimal post-processing: clean up + common corrections."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r'^[,.!?;:\s]+', '', text)
    text = re.sub(r'[,.!?;:\s]+$', '', text)
    text_lower = text.lower()
    for wrong, correct in sorted(_WHISPER_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        if pattern.search(text_lower):
            text = pattern.sub(correct, text)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text


@dataclass
class UtteranceEvent:
    """An event emitted by the command listener."""
    kind: str                    # "speech_start" | "partial" | "final" | "failure"
    text: str = ""
    is_final: bool = False
    confidence: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    audio: Optional[bytes] = None  # int16 PCM (final only)
    # Diagnostics
    audio_duration_ms: float = 0.0
    whisper_latency_ms: float = 0.0
    endpoint_reason: str = ""
    # TASK 6: explicit failure classification
    failure_reason: str = ""     # MISUNDERSTOOD | LOW_CONFIDENCE | TRANSCRIPTION_FAILED | TIMEOUT | GARBAGE


# ── TASK 6: explicit failure reasons ──────────────────────────
# Leo must NEVER silently return to wake mode after a detected speech
# attempt. Each failure path yields a "failure" UtteranceEvent with one of
# these reasons so the ConversationEngine can speak a short response.
FAILURE_MISUNDERSTOOD = "MISUNDERSTOOD"
FAILURE_LOW_CONFIDENCE = "LOW_CONFIDENCE"
FAILURE_TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
FAILURE_TIMEOUT = "TIMEOUT"
FAILURE_GARBAGE = "GARBAGE"

# TASK 4: repetition/hallucination detection. A transcript that is just the
# same short fragment repeated many times (or a tiny fragment repeated) is a
# Whisper hallucination, not a real command.
_REPEATED_WORD_RE = re.compile(r"\b(\w+)\b(?:\s+\1\b){2,}", re.IGNORECASE)


def _is_repeated_hallucination(text: str) -> bool:
    """True if the transcript is a repeated/hallucinated fragment.

    Whisper sometimes emits the same word/fragment over and over (e.g.
    "you you you you"). A real command rarely repeats a single token 3+
    times in a row.
    """
    t = text.strip().lower()
    if not t:
        return False
    if _REPEATED_WORD_RE.search(t):
        return True
    # A transcript that is a single word repeated (with spaces) is a
    # hallucination.
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    return False


def _validate_transcript(
    text: str,
    confidence: float,
    speech_dur_ms: float,
    language_prob: Optional[float] = None,
) -> Tuple[bool, str]:
    """TASK 4: combined-evidence transcript validation.

    Confidence is NOT a binary accept/reject signal. It is combined with:
      - transcript length (garbage/short fragments)
      - garbage detection (content patterns)
      - speech duration (a real command has enough audio)
      - language probability (if available)
      - repetition/hallucination detection

    Returns (accepted, failure_reason). failure_reason is "" when accepted.
    """
    t = (text or "").strip()

    # 1. Empty transcript → transcription failed.
    if not t:
        return False, FAILURE_TRANSCRIPTION_FAILED

    # 2. Garbage content → reject regardless of confidence.
    if is_garbage(t):
        return False, FAILURE_GARBAGE

    # 3. Repetition/hallucination → reject.
    if _is_repeated_hallucination(t):
        return False, FAILURE_GARBAGE

    # 4. Too-short speech duration with a long transcript is suspicious, but
    #    a short command with short audio is fine. We only flag when the
    #    transcript is long but audio is implausibly short (hallucination).
    word_count = len(t.split())
    if word_count >= 6 and speech_dur_ms < 400:
        return False, FAILURE_GARBAGE

    # 5. Language probability (if the backend provides it) — a very low
    #    language probability suggests non-speech/hallucination.
    if language_prob is not None and language_prob < 0.2:
        return False, FAILURE_LOW_CONFIDENCE

    # 6. Confidence is now a SOFT signal. A low confidence alone is NOT a
    #    rejection. We only reject when confidence is EXTREMELY low AND the
    #    transcript is very short (a clear hallucination). Normal speech
    #    with confidence=-0.366 (e.g. "Now tell me can you see my screen?")
    #    is ACCEPTED.
    if confidence < -1.5 and word_count <= 2:
        return False, FAILURE_LOW_CONFIDENCE

    # Accepted.
    return True, ""


class _WhisperTranscriber:
    """faster-whisper transcriber for partial + final transcription."""

    _BACKEND_CACHE = Path(__file__).resolve().parent.parent / "data" / "whisper_backend.json"

    def __init__(self):
        self._model = None
        self._ready = False
        self._device: str = "cpu"
        self._compute: str = "int8"

    def load(self) -> bool:
        if self._ready:
            return True
        try:
            from faster_whisper import WhisperModel
            import torch
            import json

            cached = None
            try:
                if self._BACKEND_CACHE.exists():
                    data = json.loads(self._BACKEND_CACHE.read_text(encoding="utf-8"))
                    cached = (data.get("device"), data.get("compute_type"))
            except Exception:
                pass

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
                    segments, _ = model.transcribe(warmup, beam_size=1, without_timestamps=True)
                    list(segments)
                    self._model = model
                    self._ready = True
                    self._device = device
                    self._compute = compute
                    try:
                        self._BACKEND_CACHE.parent.mkdir(parents=True, exist_ok=True)
                        self._BACKEND_CACHE.write_text(
                            json.dumps({"device": device, "compute_type": compute, "saved_at": time.time()}),
                            encoding="utf-8")
                    except Exception:
                        pass
                    logger.info("[CMD-LISTEN] faster-whisper loaded (device=%s, compute=%s)", device, compute)
                    return True
                except Exception as e:
                    logger.warning("[CMD-LISTEN] faster-whisper %s/%s unusable: %s", device, compute, e)
            logger.error("[CMD-LISTEN] faster-whisper: no working backend")
            return False
        except Exception as e:
            logger.warning("[CMD-LISTEN] faster-whisper unavailable: %s", e)
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    def transcribe(self, pcm_int16: bytes, sample_rate: int = SAMPLE_RATE) -> Tuple[str, float]:
        """Transcribe PCM16 bytes. Returns (text, avg_logprob).

        Uses beam search (beam_size=5) for higher accuracy on final
        transcripts. Partial transcripts use beam_size=1 for speed.
        """
        if not self._ready or not pcm_int16:
            return "", 0.0
        try:
            audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < sample_rate * 0.15:
                return "", 0.0
            segments, _ = self._model.transcribe(
                audio,
                beam_size=5,          # Beam search for higher accuracy
                language="en",
                temperature=0.0,
                best_of=5,            # Best-of-N for better hypotheses
                condition_on_previous_text=False,
                compression_ratio_threshold=None,
                no_speech_threshold=0.9,
                vad_filter=False,
                without_timestamps=True,
            )
            segs = list(segments)
            if not segs:
                return "", 0.0
            text = " ".join(s.text.strip() for s in segs).strip()
            logprobs = [float(getattr(s, "avg_logprob", 0.0) or 0.0) for s in segs]
            avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0
            return text, avg_logprob
        except Exception as e:
            logger.debug("[CMD-LISTEN] transcribe error: %s", e)
            return "", 0.0

    def transcribe_fast(self, pcm_int16: bytes, sample_rate: int = SAMPLE_RATE) -> Tuple[str, float]:
        """Fast transcription for partials — beam_size=1 for low latency."""
        if not self._ready or not pcm_int16:
            return "", 0.0
        try:
            audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < sample_rate * 0.15:
                return "", 0.0
            segments, _ = self._model.transcribe(
                audio,
                beam_size=1,
                language="en",
                temperature=0.0,
                best_of=1,
                condition_on_previous_text=False,
                compression_ratio_threshold=None,
                no_speech_threshold=0.9,
                vad_filter=False,
                without_timestamps=True,
            )
            segs = list(segments)
            if not segs:
                return "", 0.0
            text = " ".join(s.text.strip() for s in segs).strip()
            logprobs = [float(getattr(s, "avg_logprob", 0.0) or 0.0) for s in segs]
            avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0
            return text, avg_logprob
        except Exception as e:
            logger.debug("[CMD-LISTEN] transcribe_fast error: %s", e)
            return "", 0.0


class CommandListener:
    """
    Clean streaming speech-to-text with continuously growing context windows.

    KEY PROPERTIES:
      - Whisper receives minimum ~800ms of audio (command_config.min_context_ms)
      - Context window grows continuously from speech start to endpoint
      - Simple silence-based endpoint — no stability heuristics
      - Single VAD instance (unified_vad)
      - No SpeechCorrector, no LCP merging, no stability tracking
    """

    def __init__(self):
        self._whisper = _WhisperTranscriber()
        self._ready = False
        self._listen_enabled = asyncio.Event()
        self._listen_enabled.set()
        self._cancel = asyncio.Event()
        self._drain_requested = False

    def initialize(self) -> bool:
        whisper_ok = self._whisper.load()
        vad_ok = unified_vad.load()
        self._ready = whisper_ok
        if not whisper_ok:
            logger.error("[CMD-LISTEN] Whisper unavailable — command STT disabled")
        return self._ready

    @property
    def ready(self) -> bool:
        return self._ready

    async def _transcribe_with_timeout(self, loop, fn, pcm: bytes, sample_rate: int,
                                       timeout_s: float) -> Tuple[str, float]:
        """Run a Whisper inference call in an executor with a hard timeout.

        A hung faster-whisper inference must not stall the conversation
        engine. On timeout we log a structured STT TIMEOUT record, return an
        empty result, and let the pipeline recover by staying in LISTEN.
        """
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, fn, pcm, sample_rate),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error(
                "[STT] TIMEOUT inference=%.0fms_budget_exceeded fn=%s "
                "pcm_bytes=%d — recovering (engine stays alive)",
                timeout_s * 1000.0, getattr(fn, "__name__", fn), len(pcm))
            return "", 0.0

    def pause_listening(self) -> None:
        """Mute STT during TTS playback."""
        self._listen_enabled.clear()
        logger.info("[CMD-LISTEN] Listening PAUSED (TTS guard)")

    def resume_listening(self) -> None:
        """Unmute STT after TTS finishes."""
        self._drain_requested = True
        self._listen_enabled.set()
        logger.info("[CMD-LISTEN] Listening RESUMED — drain requested")

    def cancel(self) -> None:
        self._cancel.set()

    def stop_streaming(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    # ── Main streaming loop ─────────────────────────────────

    async def stream_utterances(self) -> AsyncIterator[UtteranceEvent]:
        """
        Yield UtteranceEvents as the user speaks.

        Emits:
          - speech_start: when speech begins
          - partial: incremental transcription (~every 200ms)
          - final: the complete utterance (silence endpoint or max duration)

        POST-WAKE CAPTURE CONTRACT (sample-accurate, no wall-clock races):
          1. Establish a clear command_session_start boundary at the current
             ring-buffer write position. Samples OLDER than this boundary are
             ignored (they are wake-word / chime / TTS contamination).
          2. Consume only fresh samples after the boundary.
          3. Keep a small rolling pre-roll so the first syllable after the
             wake word is never clipped.
          4. Drive VAD / endpoint / partial timing from SAMPLE COUNTS, not
             time.time(), so the pipeline is deterministic and testable.
        """
        if not self._ready:
            if not self.initialize():
                return

        self.reset_cancel()
        loop = asyncio.get_event_loop()
        cfg = command_config
        logger.info("[CMD-DEBUG] listener_enter ready=%s", self._ready)

        # ── COMMAND_SESSION_START boundary ──────────────────────
        # The ring buffer is a non-destructive, cursor-based peek. We do NOT
        # copy the 20+ second historical buffer into recognition — we only
        # advance our read cursor to the current write position. Everything
        # written before this boundary (wake word, chime, TTS) is discarded.
        command_session_start = audio_manager.total_samples
        logger.info("[CMD] session_start total_samples=%d", command_session_start)
        logger.info("[CMD] buffer_before_flush=%d buffer_after_flush=0", command_session_start)
        last_total = command_session_start
        logger.info("[CMD-DEBUG] drain_complete current_write_head=%d", last_total)

        # ── CMD-AUDIO handoff diagnostics: record session-start state ──
        # We snapshot the AudioManager frame sequence at LISTEN entry so we
        # can PROVE fresh microphone frames arrive after this boundary
        # (frame_id/timestamp must increase). This distinguishes a STALE
        # ring buffer from FRESH microphone audio.
        _session_start_ts = time.time()
        _session_start_frame_id = getattr(audio_manager, "frame_id", 0)
        _session_start_last_frame_ts = getattr(audio_manager, "last_frame_timestamp", 0.0)
        _session_start_buffer_samples = command_session_start
        logger.info(
            "[CMD-AUDIO] session_start ts=%.3f frame_id=%d "
            "last_frame_ts=%.3f buffer_samples=%d",
            _session_start_ts, _session_start_frame_id,
            _session_start_last_frame_ts, _session_start_buffer_samples)
        # First fresh frame after LISTEN (for the 1s CMD-FATAL watchdog).
        _first_fresh_frame_ts = 0.0
        _fatal_reported = False

        # ── Sample-accurate state ───────────────────────────────
        audio_buffer: List[np.ndarray] = []   # collected speech frames (512 samples)
        pre_roll: List[np.ndarray] = []       # rolling pre-roll frames
        pending: np.ndarray = np.array([], dtype=np.float32)
        in_speech = False
        speech_samples = 0                    # speech samples captured (excluding pre-roll)
        silence_samples = 0                   # consecutive trailing silence samples
        last_partial_samples = 0              # samples captured since last partial
        speech_start_time = 0.0               # wall-clock (diagnostic only)

        # Transcript stabilization
        last_partial_text: str = ""
        stable_count: int = 0
        last_partial_confidence: float = 0.0

        frame_ms = FRAME_SAMPLES / SAMPLE_RATE * 1000.0  # 32 ms
        max_pre_frames = max(1, int(cfg.pre_roll_ms / frame_ms))
        endpoint_silence_samples = cfg.samples(cfg.endpoint_silence_ms)
        min_silence_samples = cfg.samples(cfg.min_silence_ms)
        min_speech_samples = cfg.samples(cfg.min_speech_ms)
        max_utterance_samples = cfg.samples(int(cfg.max_utterance_s * 1000))

        last_vad_log = 0.0
        last_received_log = 0.0
        _received_total = 0

        logger.info("[CMD] Listening started (endpoint=%dms, min_context=%dms, "
                    "partial_interval=%dms, pre_roll=%dms)",
                    cfg.endpoint_silence_ms, cfg.min_context_ms,
                    int(cfg.partial_interval_s * 1000), cfg.pre_roll_ms)

        while not self._cancel.is_set():
            # ── Diagnostic timeout on the listen gate ──
            # If LISTEN is paused (TTS guard) this wait may block. A hung
            # Event.wait() must not silently stall the pipeline — log when
            # the gate stays closed beyond 1s so a blocked consumer is
            # visible in the logs.
            try:
                await asyncio.wait_for(
                    self._listen_enabled.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                if not self._cancel.is_set():
                    logger.warning(
                        "[CMD] listen gate CLOSED for >1s (TTS guard active?) "
                        "— waiting; drain_requested=%s",
                        self._drain_requested)
                continue
            if self._cancel.is_set():
                break

            # Drain-on-resume: skip TTS-contaminated audio produced while
            # Leo was speaking. Reset the cursor to the current write head.
            if self._drain_requested:
                self._drain_requested = False
                old_total = last_total
                last_total = audio_manager.total_samples
                skipped = last_total - old_total
                if skipped > 0:
                    logger.info("[CMD] Drain-on-resume skipped=%d samples (%.0fms)",
                                skipped, skipped / 16.0)
                in_speech = False
                audio_buffer.clear()
                pre_roll.clear()
                pending = np.array([], dtype=np.float32)
                speech_samples = 0
                silence_samples = 0
                last_partial_samples = 0
                last_partial_text = ""
                stable_count = 0
                continue

            new_audio, last_total = audio_manager.read_since(last_total)
            if len(new_audio) == 0:
                # ── CMD-FATAL watchdog: no fresh microphone frame within 1s ──
                # After LISTEN begins, the AudioManager callback must keep
                # producing NEW frames. If nothing arrives for 1 second, dump
                # the full audio stream state so we can tell STALE BUFFER vs
                # FRESH MICROPHONE vs a blocked consumer.
                if (not _fatal_reported
                        and _first_fresh_frame_ts == 0.0
                        and (time.time() - _session_start_ts) >= 1.0):
                    _fatal_reported = True
                    frame_state = {}
                    try:
                        frame_state = audio_manager.get_frame_state()
                    except Exception as e:
                        frame_state = {"error": str(e)}
                    logger.error(
                        "[CMD-FATAL] No fresh microphone audio after LISTEN "
                        "elapsed=%.2fs session_frame_id=%d "
                        "audio_stream_state=%s producer_callback_count=%s "
                        "ring_buffer_samples=%s ring_buffer_frames=%s "
                        "ring_buffer_seconds=%s last_frame_timestamp=%.3f "
                        "last_frame_id=%s audio_running=%s dropped_frames=%s "
                        "digital_silence=%s zero_streak=%s",
                        time.time() - _session_start_ts, _session_start_frame_id,
                        frame_state.get("stream_state"),
                        frame_state.get("callback_count"),
                        frame_state.get("ring_buffer_samples"),
                        frame_state.get("ring_buffer_frames"),
                        frame_state.get("ring_buffer_seconds"),
                        frame_state.get("last_frame_timestamp", 0.0),
                        frame_state.get("frame_id"),
                        frame_state.get("stream_running"),
                        frame_state.get("dropped_frames"),
                        frame_state.get("digital_silence"),
                        frame_state.get("zero_streak"))
                await asyncio.sleep(0.01)
                continue

            # ── CMD-AUDIO: log every received audio chunk with frame identity ──
            # This proves whether the AudioManager callback is STILL producing
            # fresh frames after LISTEN. frame_id/timestamp must increase.
            _now_audio = time.time()
            _cur_frame_id = getattr(audio_manager, "frame_id", 0)
            _cur_last_frame_ts = getattr(audio_manager, "last_frame_timestamp", 0.0)
            _age_ms = (_now_audio - _cur_last_frame_ts) * 1000.0 if _cur_last_frame_ts else -1.0
            _rms = float(np.sqrt(np.mean(new_audio.astype(np.float64) ** 2))) * 32768.0
            _fresh = _cur_frame_id > _session_start_frame_id
            if _fresh and _first_fresh_frame_ts == 0.0:
                _first_fresh_frame_ts = _now_audio
                logger.info(
                    "[CMD-AUDIO] FIRST FRESH FRAME after LISTEN at +%.2fs "
                    "frame_id=%d (session_start=%d) age_ms=%.1f",
                    _now_audio - _session_start_ts, _cur_frame_id,
                    _session_start_frame_id, _age_ms)
            logger.info(
                "[CMD-AUDIO] frame_id=%d timestamp=%.3f age_ms=%.1f "
                "samples=%d rms=%.1f fresh=%s",
                _cur_frame_id, _cur_last_frame_ts, _age_ms,
                len(new_audio), _rms, _fresh)

            # Trace received samples + fresh-frame age (rate-limited).
            _received_total += len(new_audio)
            now_mono = time.monotonic()
            if now_mono - last_received_log >= 1.0:
                last_received_log = now_mono
                logger.info("[CMD] received_samples=%d (total since session=%d)",
                            len(new_audio), _received_total)
                logger.info("[CMD-DEBUG] audio_frame_received samples=%d "
                            "frame_age_ms=%.1f write_head=%d read_cursor=%d",
                            len(new_audio), _age_ms, last_total, last_total)

            pending = np.concatenate([pending, new_audio.astype(np.float32, copy=False)])

            # Process complete 512-sample frames. The remainder is carried
            # forward so no sample is ever dropped.
            n_frames = len(pending) // FRAME_SAMPLES
            for fi in range(n_frames):
                frame = pending[fi * FRAME_SAMPLES:(fi + 1) * FRAME_SAMPLES]
                prob = await loop.run_in_executor(None, unified_vad.speech_prob, frame)
                is_speech = prob > cfg.vad_speech_threshold

                now_mono = time.monotonic()
                if now_mono - last_vad_log >= cfg.vad_log_interval_s:
                    last_vad_log = now_mono
                    logger.info("[CMD] vad_probability=%.3f speech=%s", prob, bool(is_speech))
                    logger.info("[CMD-DEBUG] vad_probability=%.3f speech=%s "
                                "vad_state=%s last_speech_ts=%s",
                                prob, bool(is_speech),
                                getattr(unified_vad, "_state", "unknown"),
                                getattr(unified_vad, "_last_speech_timestamp", None))

                # Rolling pre-roll (always updated so onset keeps context).
                pre_roll.append(frame.copy())
                if len(pre_roll) > max_pre_frames:
                    pre_roll.pop(0)

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        speech_start_time = time.time()
                        speech_samples = 0
                        silence_samples = 0
                        last_partial_samples = 0
                        last_partial_text = ""
                        stable_count = 0
                        # Pre-roll (already contains the current frame) prevents
                        # first-syllable clipping.
                        audio_buffer = list(pre_roll)
                        logger.info("[CMD] speech_started vad_prob=%.2f pre_roll_frames=%d",
                                    prob, len(pre_roll))
                        logger.info("[CMD-DEBUG] speech_started vad_prob=%.3f", prob)
                        yield UtteranceEvent(kind="speech_start", started_at=speech_start_time)
                    else:
                        audio_buffer.append(frame.copy())
                    speech_samples += FRAME_SAMPLES
                    silence_samples = 0
                else:
                    if in_speech:
                        audio_buffer.append(frame.copy())
                        silence_samples += FRAME_SAMPLES
                        if silence_samples == FRAME_SAMPLES:
                            logger.info("[CMD] silence_started")
                        elif silence_samples % (FRAME_SAMPLES * 8) == 0:
                            logger.info("[CMD] silence_ms=%d",
                                        int(silence_samples / SAMPLE_RATE * 1000))

                        speech_dur_ms = speech_samples / SAMPLE_RATE * 1000.0

                        # ── Endpoint decision (sample-accurate) ──
                        # Do not finalize after tiny fragments; do not wait
                        # forever. Start on VAD speech, continue during
                        # speech, end after sustained silence.
                        can_endpoint = (
                            speech_samples >= min_speech_samples and
                            silence_samples >= min_silence_samples
                        )

                        if can_endpoint and last_partial_confidence < cfg.confidence_threshold:
                            can_endpoint = silence_samples >= cfg.samples(cfg.low_confidence_silence_ms)

                        # Endpoint is driven by sustained silence + minimum
                        # speech, NOT by partial-transcript stability. The old
                        # stability gate could block short commands that never
                        # produced enough matching partials, hanging the turn
                        # until the max-duration cap. (The confidence gate above
                        # already handles noisy/low-quality audio.)
                        if can_endpoint and silence_samples >= endpoint_silence_samples:
                            logger.info("[CMD] endpoint silence=%dms speech=%.0fms "
                                        "stable=%d conf=%.3f frames=%d",
                                        int(silence_samples / SAMPLE_RATE * 1000),
                                        speech_dur_ms, stable_count,
                                        last_partial_confidence, len(audio_buffer))
                            logger.info("[CMD-DEBUG] speech_ended silence=%dms",
                                        int(silence_samples / SAMPLE_RATE * 1000))
                            logger.info("[CMD-DEBUG] endpoint silence=%dms speech=%.0fms",
                                        int(silence_samples / SAMPLE_RATE * 1000),
                                        speech_dur_ms)
                            final = await self._finalize(
                                audio_buffer, speech_start_time,
                                endpoint_reason="silence_%dms" % int(silence_samples / SAMPLE_RATE * 1000))
                            in_speech = False
                            audio_buffer = []
                            silence_samples = 0
                            speech_samples = 0
                            last_partial_samples = 0
                            last_partial_text = ""
                            stable_count = 0
                            if final is not None:
                                if is_filler(final.text):
                                    logger.info("[CMD-LISTEN] Filler '%s' — turn stays open", final.text)
                                    continue
                                yield final
                            continue

                # Partial transcription (sample-gated, not wall-clock-gated)
                if in_speech and speech_samples >= cfg.samples(cfg.min_context_ms):
                    new_since_partial = speech_samples - last_partial_samples
                    partial_interval_samples = cfg.samples(int(cfg.partial_interval_s * 1000))
                    if new_since_partial >= partial_interval_samples:
                        last_partial_samples = speech_samples
                        pcm = self._frames_to_bytes(audio_buffer)
                        t_partial_start = time.time()
                        text, confidence = await self._transcribe_with_timeout(
                            loop, self._whisper.transcribe_fast, pcm, SAMPLE_RATE,
                            WHISPER_PARTIAL_TIMEOUT_S)
                        t_partial_elapsed = (time.time() - t_partial_start) * 1000

                        if text and not is_filler(text):
                            normalized = _postprocess(text).lower()
                            if last_partial_text and self._similarity(
                                    last_partial_text, normalized) >= cfg.stability_similarity:
                                stable_count += 1
                            else:
                                stable_count = 0
                            last_partial_text = normalized
                            last_partial_confidence = confidence

                            logger.info("[CMD-LISTEN] Partial (%.0fms audio, %.0fms latency): '%s' "
                                        "(conf=%.3f, stable=%d)",
                                        speech_samples / SAMPLE_RATE * 1000,
                                        t_partial_elapsed, text, confidence, stable_count)
                            yield UtteranceEvent(
                                kind="partial", text=text, is_final=False,
                                started_at=speech_start_time,
                                confidence=confidence,
                                audio_duration_ms=speech_samples / SAMPLE_RATE * 1000,
                                whisper_latency_ms=t_partial_elapsed)

                # Hard cap — never wait indefinitely
                if in_speech and speech_samples >= max_utterance_samples:
                    logger.info("[CMD] utterance capped at %.1fs", cfg.max_utterance_s)
                    final = await self._finalize(
                        audio_buffer, speech_start_time, endpoint_reason="max_duration")
                    in_speech = False
                    audio_buffer = []
                    silence_samples = 0
                    speech_samples = 0
                    last_partial_samples = 0
                    if final is not None and not is_filler(final.text):
                        yield final

            pending = pending[n_frames * FRAME_SAMPLES:]


    async def _finalize(
        self,
        frames: List[np.ndarray],
        start: float,
        endpoint_reason: str = "",
    ) -> Optional[UtteranceEvent]:
        """Transcribe the complete utterance."""
        cfg = command_config
        pcm = self._frames_to_bytes(frames)
        dur_ms = len(frames) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
        num_samples = len(pcm) // 2

        if dur_ms < cfg.min_utterance_ms or len(pcm) < 512:
            logger.info("[CMD-LISTEN] Utterance DISCARDED (too_short: %.0fms)", dur_ms)
            return None

        loop = asyncio.get_event_loop()
        logger.info("[CMD] whisper_start duration=%.2fs samples=%d", dur_ms / 1000.0, num_samples)
        logger.info("[CMD-DEBUG] whisper_started samples=%d", num_samples)
        t_whisper = time.time()
        raw_text, confidence = await self._transcribe_with_timeout(
            loop, self._whisper.transcribe, pcm, SAMPLE_RATE,
            WHISPER_FINAL_TIMEOUT_S)
        whisper_latency = (time.time() - t_whisper) * 1000
        logger.info("[CMD] whisper_result text=%r confidence=%.3f latency=%.0fms",
                    raw_text or "", confidence, whisper_latency)
        logger.info("[CMD-DEBUG] whisper_finished text=%r latency=%.0fms",
                    raw_text or "", whisper_latency)
        logger.info("[CMD-DEBUG] transcript=%r", raw_text or "")

        if not raw_text:
            logger.info("[CMD-LISTEN] Utterance DISCARDED (empty transcript, %.0fms audio)", dur_ms)
            return None

        text = _postprocess(raw_text)

        # ── Garbage rejection ──────────────────────────────
        # Reject Whisper hallucinated transcripts that are not real
        # user commands. These are logged but NOT yielded to the engine.
        if is_garbage(text):
            logger.info("[CMD-LISTEN] Utterance DISCARDED (garbage transcript: '%s', %.0fms, "
                        "conf=%.3f)", text, dur_ms, confidence)
            return None

        # ── Confidence gate ────────────────────────────────
        # Very low-confidence transcripts are likely noise.
        # But allow short commands ("stop", "yes", "no") with low confidence
        # since they are easy to mis-transcribe but critical to hear.
        short_commands = {"stop", "yes", "no", "go", "on", "off", "up", "down",
                          "open", "run", "play", "pause", "next", "back", "close",
                          "quit", "exit", "help", "menu", "home", "back", "cancel"}
        is_short_command = len(text.split()) <= 2 and text.lower().strip() in short_commands
        if not is_short_command and confidence < cfg.confidence_threshold:
            logger.info("[CMD-LISTEN] Utterance DISCARDED (low confidence: %.3f < %.3f, "
                        "text='%s', %.0fms)", confidence, cfg.confidence_threshold, text, dur_ms)
            return None

        logger.info("[CMD-LISTEN] FINALIZED: '%s' (raw='%s', duration=%.0fms, "
                    "whisper_latency=%.0fms, confidence=%.3f, endpoint=%s)",
                    text, raw_text, dur_ms, whisper_latency, confidence, endpoint_reason)

        return UtteranceEvent(
            kind="final", text=text, is_final=True,
            started_at=start, ended_at=time.time(), audio=pcm,
            confidence=confidence,
            audio_duration_ms=dur_ms,
            whisper_latency_ms=whisper_latency,
            endpoint_reason=endpoint_reason)

    @staticmethod
    def _frames_to_bytes(frames: List[np.ndarray]) -> bytes:
        if not frames:
            return b""
        return float32_to_int16(np.concatenate(frames)).tobytes()

    @staticmethod
    def _iter_frames_overlap(audio: np.ndarray, step: int):
        n = len(audio)
        for i in range(0, n - FRAME_SAMPLES + 1, step):
            yield i, audio[i:i + FRAME_SAMPLES]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute similarity between two transcript hypotheses.

        Uses word-level Jaccard similarity with a character-level
        fallback for short texts. Returns 0.0–1.0.
        """
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        # Word-level Jaccard
        words_a = set(a.split())
        words_b = set(b.split())
        if words_a and words_b:
            inter = len(words_a & words_b)
            union = len(words_a | words_b)
            if union > 0:
                jaccard = inter / union
                # For short commands, word overlap is very informative
                if len(words_a) <= 3 or len(words_b) <= 3:
                    return jaccard
                return jaccard

        # Character-level fallback (for very short texts)
        if len(a) < 3 or len(b) < 3:
            return 1.0 if a == b else 0.0
        # Simple character n-gram overlap
        def _ngrams(s: str, n: int = 2):
            return {s[i:i+n] for i in range(len(s) - n + 1)}
        na = _ngrams(a)
        nb = _ngrams(b)
        if not na or not nb:
            return 0.0
        return len(na & nb) / len(na | nb)

    # ── Interruption detection ─────────────────────────────

    async def detect_interruption(self, stop_event: asyncio.Event) -> None:
        """Detect user speech while Leo is speaking."""
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
            for i in range(0, len(new_audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                frame = new_audio[i:i + FRAME_SAMPLES]
                prob = await loop.run_in_executor(None, unified_vad.speech_prob, frame)
                if prob > 0.6:
                    speech_run_ms += 32
                    if speech_run_ms >= command_config.interrupt_min_ms:
                        logger.info("[CMD-LISTEN] Interruption detected (user speaking)")
                        stop_event.set()
                        return
                else:
                    speech_run_ms = 0.0


# Global singleton
command_listener = CommandListener()