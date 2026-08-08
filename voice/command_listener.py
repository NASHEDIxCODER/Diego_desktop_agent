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
from voice.vad import unified_vad, SPEECH_THRESHOLD

logger = logging.getLogger(__name__)

# ── Tuning ─────────────────────────────────────────────────────
MIN_CONTEXT_MS = 800          # Minimum audio before Whisper sees anything
ENDPOINT_SILENCE_MS = 1200    # Silence this long → finalize utterance (raised from 900)
MIN_UTTERANCE_MS = 600        # Shorter than this → discard (raised from 250)
MAX_UTTERANCE_S = 20.0        # Hard cap on utterance length
PARTIAL_INTERVAL_S = 0.20     # 200ms between partial updates
PARTIAL_MIN_NEW_MS = 200      # Minimum new audio before running partial
PRE_ROLL_MS = 600             # Pre-roll to prevent first-word clipping
INTERRUPT_MIN_MS = 90         # Speech duration to trigger interruption

# ── Transcript stabilization ───────────────────────────────────
# Do NOT trust the first transcript. Compare consecutive partial
# hypotheses and only finalize when the transcript is stable.
STABILITY_REQUIRED = 2        # Consecutive matching partials before finalize
STABILITY_SIMILARITY = 0.85   # Similarity threshold for "same" transcript
STABILITY_MIN_MS = 1200       # Minimum speech before stability matters
STABILITY_MAX_MS = 4000       # After this, finalize even if unstable

# ── Endpoint confidence ───────────────────────────────────────
# Finalize only when: min_speech + min_silence + stability + confidence
MIN_SPEECH_MS = 600           # Minimum speech duration before endpoint (raised from 400)
MIN_SILENCE_MS = 600          # Minimum silence before endpoint (raised from 500)
CONFIDENCE_THRESHOLD = -0.3   # Whisper avg_logprob threshold (raised from -0.5)
LOW_CONFIDENCE_SILENCE_MS = 2000  # Longer silence for low-confidence transcripts (raised from 1500)

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
    kind: str                    # "speech_start" | "partial" | "final"
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
      - Whisper receives minimum ~800ms of audio (MIN_CONTEXT_MS)
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
        """
        if not self._ready:
            if not self.initialize():
                return

        self.reset_cancel()
        loop = asyncio.get_event_loop()

        # Drain TTS-contaminated audio
        drain_start = audio_manager.total_samples
        logger.info("[CMD-LISTEN] Draining TTS-contaminated audio (total_samples=%d)", drain_start)
        last_total = drain_start

        # Audio buffers
        audio_buffer: List[np.ndarray] = []
        pre_roll: List[np.ndarray] = []
        in_speech = False
        speech_start_time = 0.0
        last_voice_time = 0.0
        silence_run_ms = 0.0
        last_partial_time = 0.0
        _frame_remainder: np.ndarray = np.array([], dtype=np.float32)

        # ── Transcript stabilization state (NEW) ─────────────
        # Track consecutive partial hypotheses to detect stability.
        # Do NOT trust the first transcript.
        last_partial_text: str = ""
        stable_count: int = 0
        last_partial_confidence: float = 0.0

        # VAD step (50% overlap = 256 samples = 16ms)
        VAD_STEP = FRAME_SAMPLES // 2

        # Non-overlap tracking (global sample offset)
        _last_nonoverlap_idx: int = -1
        _chunk_base_sample: int = 0

        logger.info("[CMD-LISTEN] Listening started (endpoint=%dms, min_context=%dms, "
                    "partial_interval=%dms)",
                    ENDPOINT_SILENCE_MS, MIN_CONTEXT_MS, int(PARTIAL_INTERVAL_S * 1000))

        while not self._cancel.is_set():
            await self._listen_enabled.wait()
            if self._cancel.is_set():
                break

            # Drain-on-resume: skip TTS-contaminated audio
            if self._drain_requested:
                self._drain_requested = False
                old_total = last_total
                last_total = audio_manager.total_samples
                skipped = last_total - old_total
                if skipped > 0:
                    logger.info("[CMD-LISTEN] Drain-on-resume: skipped %d samples (%.0fms)",
                                skipped, skipped / 16.0)
                in_speech = False
                audio_buffer.clear()
                silence_run_ms = 0.0
                _last_nonoverlap_idx = -1
                _frame_remainder = np.array([], dtype=np.float32)
                pre_roll.clear()
                continue

            new_audio, last_total = audio_manager.read_since(last_total)
            if len(new_audio) == 0:
                await asyncio.sleep(0.01)
                continue

            # Prepend frame remainder
            if len(_frame_remainder) > 0:
                new_audio = np.concatenate([_frame_remainder, new_audio])
                _frame_remainder = np.array([], dtype=np.float32)

            _chunk_base_sample = last_total - len(new_audio)

            # VAD: overlapping frames
            last_vad_idx = -1
            for i, frame in self._iter_frames_overlap(new_audio, VAD_STEP):
                last_vad_idx = i
                prob = await loop.run_in_executor(None, unified_vad.speech_prob, frame)
                is_speech = prob > SPEECH_THRESHOLD
                now = time.time()

                # Collect NON-overlapping frame for audio buffer
                global_sample = _chunk_base_sample + i
                nonoverlap_idx = global_sample // FRAME_SAMPLES
                is_new_nonoverlap = nonoverlap_idx > _last_nonoverlap_idx
                if is_new_nonoverlap:
                    _last_nonoverlap_idx = nonoverlap_idx
                    frame_start = i
                    frame_end = i + FRAME_SAMPLES
                    if frame_end <= len(new_audio):
                        clean_frame = new_audio[frame_start:frame_end].copy()
                    else:
                        clean_frame = frame.copy()

                    pre_roll.append(clean_frame)
                    max_pre = max(1, int((PRE_ROLL_MS / 1000.0) / (FRAME_SAMPLES / SAMPLE_RATE * 1000)))
                    if len(pre_roll) > max_pre:
                        pre_roll.pop(0)

                    if in_speech:
                        audio_buffer.append(clean_frame)

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        speech_start_time = now
                        audio_buffer = list(pre_roll)
                        last_partial_time = now
                        last_partial_text = ""
                        stable_count = 0
                        logger.info("[CMD-LISTEN] Speech start (VAD_prob=%.2f, pre_roll_frames=%d)",
                                    prob, len(pre_roll))
                        yield UtteranceEvent(kind="speech_start", started_at=now)
                    last_voice_time = now
                    silence_run_ms = 0.0
                else:
                    if in_speech:
                        silence_run_ms = (now - last_voice_time) * 1000.0
                        dur_ms = len(audio_buffer) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
                        speech_dur_ms = (now - speech_start_time) * 1000.0

                        # ── Endpoint decision (NEW) ─────────────
                        # Finalize only when:
                        #   min_speech + min_silence + stability + confidence
                        # Do NOT finalize immediately after silence.
                        can_endpoint = (
                            dur_ms >= MIN_SPEECH_MS and
                            silence_run_ms >= MIN_SILENCE_MS
                        )

                        # Low-confidence transcripts need longer silence
                        if can_endpoint and last_partial_confidence < CONFIDENCE_THRESHOLD:
                            can_endpoint = silence_run_ms >= LOW_CONFIDENCE_SILENCE_MS

                        # Transcript must be stable (or speech too long)
                        # CRITICAL FIX: For SHORT utterances (< STABILITY_MIN_MS),
                        # do NOT require stability. A 1-second command like
                        # "open firefox" only produces 1 partial (first partial
                        # needs 800ms of audio), so stable_count is always 0.
                        # Requiring stability here adds 3+ seconds of latency
                        # to every short command. Only require stability for
                        # longer utterances where the transcript may still be
                        # evolving.
                        if can_endpoint and speech_dur_ms >= STABILITY_MIN_MS:
                            if speech_dur_ms < STABILITY_MAX_MS:
                                can_endpoint = stable_count >= STABILITY_REQUIRED
                            # else: speech too long — finalize even if unstable
                        # else: short utterance — finalize on silence alone
                        # (confidence gate already applied above)

                        if can_endpoint and silence_run_ms >= ENDPOINT_SILENCE_MS:
                            logger.info("[CMD-LISTEN] Endpoint (silence=%dms, duration=%.0fms, "
                                        "stable=%d, conf=%.3f, frames=%d)",
                                        int(silence_run_ms), dur_ms, stable_count,
                                        last_partial_confidence, len(audio_buffer))
                            final = await self._finalize(
                                audio_buffer, speech_start_time,
                                endpoint_reason=f"silence_{int(silence_run_ms)}ms")
                            in_speech = False
                            audio_buffer = []
                            silence_run_ms = 0.0
                            _last_nonoverlap_idx = -1
                            last_partial_text = ""
                            stable_count = 0
                            if final is not None:
                                if is_filler(final.text):
                                    logger.info("[CMD-LISTEN] Filler '%s' — turn stays open", final.text)
                                    continue
                                yield final
                            continue

                # Partial transcription
                if in_speech:
                    context_ms = len(audio_buffer) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
                    if context_ms < MIN_CONTEXT_MS:
                        continue  # Not enough audio yet

                    if (now - last_partial_time) >= PARTIAL_INTERVAL_S:
                        pcm = self._frames_to_bytes(audio_buffer)
                        t_partial_start = time.time()
                        # Use fast transcription for partials (beam_size=1)
                        text, confidence = await loop.run_in_executor(
                            None, self._whisper.transcribe_fast, pcm, SAMPLE_RATE)
                        t_partial_elapsed = (time.time() - t_partial_start) * 1000
                        last_partial_time = now

                        if text and not is_filler(text):
                            # ── Transcript stabilization (NEW) ──
                            # Compare consecutive partial hypotheses.
                            # Only increment stable_count when they match.
                            normalized = _postprocess(text).lower()
                            if last_partial_text and self._similarity(
                                    last_partial_text, normalized) >= STABILITY_SIMILARITY:
                                stable_count += 1
                            else:
                                stable_count = 0
                            last_partial_text = normalized
                            last_partial_confidence = confidence

                            logger.info("[CMD-LISTEN] Partial (%.0fms audio, %.0fms latency): '%s' "
                                        "(conf=%.3f, stable=%d)",
                                        context_ms, t_partial_elapsed, text, confidence, stable_count)
                            yield UtteranceEvent(
                                kind="partial", text=text, is_final=False,
                                started_at=speech_start_time,
                                confidence=confidence,
                                audio_duration_ms=context_ms,
                                whisper_latency_ms=t_partial_elapsed)

                # Hard cap
                if in_speech and (now - speech_start_time) >= MAX_UTTERANCE_S:
                    logger.info("[CMD-LISTEN] Utterance capped at %.1fs", MAX_UTTERANCE_S)
                    final = await self._finalize(
                        audio_buffer, speech_start_time, endpoint_reason="max_duration")
                    in_speech = False
                    audio_buffer = []
                    _last_nonoverlap_idx = -1
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
        endpoint_reason: str = "",
    ) -> Optional[UtteranceEvent]:
        """Transcribe the complete utterance."""
        pcm = self._frames_to_bytes(frames)
        dur_ms = len(frames) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
        num_samples = len(pcm) // 2

        if dur_ms < MIN_UTTERANCE_MS or len(pcm) < 512:
            logger.info("[CMD-LISTEN] Utterance DISCARDED (too_short: %.0fms)", dur_ms)
            return None

        loop = asyncio.get_event_loop()
        t_whisper = time.time()
        raw_text, confidence = await loop.run_in_executor(
            None, self._whisper.transcribe, pcm, SAMPLE_RATE)
        whisper_latency = (time.time() - t_whisper) * 1000

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
        if not is_short_command and confidence < CONFIDENCE_THRESHOLD:
            logger.info("[CMD-LISTEN] Utterance DISCARDED (low confidence: %.3f < %.3f, "
                        "text='%s', %.0fms)", confidence, CONFIDENCE_THRESHOLD, text, dur_ms)
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
                    if speech_run_ms >= INTERRUPT_MIN_MS:
                        logger.info("[CMD-LISTEN] Interruption detected (user speaking)")
                        stop_event.set()
                        return
                else:
                    speech_run_ms = 0.0


# Global singleton
command_listener = CommandListener()