"""
StreamingSTT — Streaming VAD + Whisper with smart endpointing.

Replaces the old "record whole command, then transcribe" flow with a
streaming pipeline:

  ring buffer ──▶ 32ms frames ──▶ Silero VAD ──▶ speech segments
                                        │
                                        ├─▶ partial Whisper (every ~0.25s of new speech)
                                        │        → "first partial transcription <300ms"
                                        │
                                        └─▶ endpointing → final Whisper → utterance

ENDPOINTING RULES (the "feel" of Siri/Gemini Live):
  - Pause < 600ms            → do NOT cut off (user may continue)
  - Filler words ("umm", "wait", "hold on", "actually", "no")
                              → keep the turn open, don't finalize
  - Silence >= endpoint_ms   → finalize the utterance
  - While Leo speaks: any confident user speech → INTERRUPT signal

Everything runs from the shared AudioManager ring buffer, so it works
concurrently with wake detection and TTS playback (full duplex).

CRITICAL ARCHITECTURE (2026-08-05 fix):
  VAD uses 50% overlapping frames for smooth detection, but the AUDIO
  BUFFER sent to Whisper uses NON-OVERLAPPING frames. Overlapping frames
  concatenated together time-stretch the audio 2x, which is why Whisper
  returned "you too fo me" instead of "YouTube for me".

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
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

from voice.audio_manager import audio_manager, SAMPLE_RATE, FRAME_SAMPLES
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# ── Endpointing configuration (configurable via voice_settings) ──
MIN_PAUSE_MS = getattr(voice_settings, "conv_min_pause_ms", 600)        # <600ms pause does NOT end the turn
ENDPOINT_SILENCE_MS = getattr(voice_settings, "conv_endpoint_ms", 900)  # trailing silence finalizes the turn
MIN_UTTERANCE_MS = 250        # ignore blips shorter than this
MAX_UTTERANCE_S = 20.0        # hard cap on a single utterance
PARTIAL_INTERVAL_S = 0.25     # run partial Whisper every 250ms for fast first token
PRE_ROLL_MS = 500             # 500ms pre-roll to avoid cutting off the first word
INTERRUPT_MIN_MS = getattr(voice_settings, "conv_interrupt_min_ms", 90)  # sustained speech to interrupt
MAX_ROLLING_CONTEXT_CHARS = 200

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

        ROOT CAUSE FIX: condition_on_previous_text is set to True when
        prompt_context is provided, so faster-whisper actually uses the
        initial_prompt to bias decoding. Previously it was False, which
        caused the prompt to be ignored.
        """
        if not self._ready or not pcm_int16:
            return ""
        if not prompt_context:
            return self.transcribe(pcm_int16, sample_rate, use_vad_filter=False)
        try:
            audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < sample_rate * 0.2:
                return ""
            segments, _ = self._model.transcribe(
                audio,
                beam_size=1,
                language="en",
                temperature=0.0,
                best_of=1,
                condition_on_previous_text=True,   # ROOT CAUSE FIX: must be True for initial_prompt
                compression_ratio_threshold=None,
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

            if len(audio) < sample_rate * 0.2:
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


class StreamingSTT:
    """
    Streaming speech-to-text with VAD endpointing and partial results.

    CRITICAL: VAD uses overlapping frames for smooth detection, but the
    audio buffer sent to Whisper uses NON-OVERLAPPING frames. Overlapping
    frames concatenated together time-stretch the audio 2x, causing
    garbled transcripts.
    """

    def __init__(self):
        self._vad = _SileroVAD()
        self._whisper = _WhisperTranscriber()
        self._ready = False
        self._listen_enabled = asyncio.Event()
        self._listen_enabled.set()
        self._cancel = asyncio.Event()

    def initialize(self) -> bool:
        vad_ok = self._vad.load()
        whisper_ok = self._whisper.load()
        self._ready = whisper_ok
        if not whisper_ok:
            logger.error("[STREAM-STT] Whisper unavailable — streaming STT disabled")
        return self._ready

    @property
    def ready(self) -> bool:
        return self._ready

    def pause_listening(self) -> None:
        self._listen_enabled.clear()

    def resume_listening(self) -> None:
        self._listen_enabled.set()

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
          - partial: incremental transcription (~every 0.25s of new audio)
          - final: the complete utterance after endpointing

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

        # ── CRITICAL: Two separate buffers ──
        # audio_buffer: NON-OVERLAPPING frames for Whisper (clean audio)
        # vad uses overlapping frames for smooth detection only
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

        # VAD overlap step (50% = 256 samples = 16ms)
        VAD_STEP = FRAME_SAMPLES // 2

        logger.info("[STREAM-STT] Listening started (endpoint=%dms, min_pause=%dms "
                    "total_samples=%d)",
                    ENDPOINT_SILENCE_MS, MIN_PAUSE_MS, drain_start)

        # Track chunk timing for diagnostics
        _last_chunk_time = time.time()
        _chunk_durations: list = []

        # Track the last non-overlapping frame index to avoid double-counting
        _last_nonoverlap_idx = -1

        while not self._cancel.is_set():
            await self._listen_enabled.wait()
            if self._cancel.is_set():
                break

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

            # ── VAD: overlapping frames for smooth detection ──
            # ── Audio buffer: NON-overlapping frames for Whisper ──
            last_vad_idx = -1
            for i, frame in self._iter_frames_overlap(new_audio, VAD_STEP):
                last_vad_idx = i
                prob = await loop.run_in_executor(None, self._vad.speech_prob, frame)
                is_speech = prob > 0.5
                now = time.time()

                # ── Collect NON-overlapping frame for audio buffer ──
                # Only add every other frame (the ones at even multiples of FRAME_SAMPLES)
                # This ensures the audio buffer has no overlap
                nonoverlap_idx = i // FRAME_SAMPLES
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
                        # The current frame triggered speech — include it too
                        # (is_new_nonoverlap was True for this iteration, and
                        # clean_frame is already in pre_roll, but we need it in
                        # audio_buffer as well since audio_buffer was just set
                        # to a copy of pre_roll)
                        last_partial_len = 0
                        last_partial_time = now
                        logger.info("[STREAM-STT] Speech start "
                                    "(VAD_prob=%.2f audio_frames=%d)",
                                    prob, len(audio_buffer))
                        yield UtteranceEvent(
                            kind="speech_start", started_at=now)
                    last_voice_time = now
                    silence_run_ms = 0.0
                else:
                    if in_speech:
                        silence_run_ms = (now - last_voice_time) * 1000.0

                        if silence_run_ms >= ENDPOINT_SILENCE_MS:
                            dur_ms = len(audio_buffer) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
                            avg_chunk = (sum(_chunk_durations) / len(_chunk_durations)
                                         if _chunk_durations else 0)
                            logger.info("[STREAM-STT] Endpoint detected "
                                        "(duration=%.0fms silence=%dms frames=%d "
                                        "chunk_avg=%.0fms)",
                                        dur_ms, int(silence_run_ms),
                                        len(audio_buffer), avg_chunk)
                            final = await self._finalize(
                                audio_buffer, speech_start_time, _rolling_context)
                            in_speech = False
                            audio_buffer = []
                            silence_run_ms = 0.0
                            _rolling_context = ""
                            _last_nonoverlap_idx = -1
                            if final is not None:
                                if is_filler(final.text):
                                    logger.info(
                                        "[STREAM-STT] Filler '%s' — turn stays open",
                                        final.text)
                                    continue
                                yield final
                            continue

                # ── Partial transcription with rolling context ──
                if in_speech:
                    new_since_partial = (len(audio_buffer) - last_partial_len) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
                    if (now - last_partial_time) >= PARTIAL_INTERVAL_S and new_since_partial >= 300:
                        pcm = self._frames_to_bytes(audio_buffer)
                        text = await loop.run_in_executor(
                            None, self._whisper.transcribe_with_context,
                            pcm, SAMPLE_RATE, _rolling_context)
                        last_partial_len = len(audio_buffer)
                        last_partial_time = now
                        if text and not is_filler(text):
                            _rolling_context = text.strip()
                            if len(_rolling_context) > MAX_ROLLING_CONTEXT_CHARS:
                                _rolling_context = _rolling_context[-MAX_ROLLING_CONTEXT_CHARS:]
                            logger.info("[STREAM-STT] Partial (ctx=%d chars): '%s'",
                                       len(_rolling_context), text)
                            yield UtteranceEvent(
                                kind="partial", text=text, is_final=False,
                                started_at=speech_start_time)

                    # Hard cap on utterance length
                    if (now - speech_start_time) >= MAX_UTTERANCE_S:
                        logger.info("[STREAM-STT] Utterance capped at %.1fs",
                                    MAX_UTTERANCE_S)
                        final = await self._finalize(
                            audio_buffer, speech_start_time, _rolling_context)
                        in_speech = False
                        audio_buffer = []
                        _rolling_context = ""
                        _last_nonoverlap_idx = -1
                        if final is not None and not is_filler(final.text):
                            yield final

            # Carry frame remainder forward
            if last_vad_idx >= 0:
                remainder_start = last_vad_idx + VAD_STEP
                if remainder_start < len(new_audio):
                    _frame_remainder = new_audio[remainder_start:].copy()

    async def _finalize(self, frames: List[np.ndarray], start: float,
                        context: str = "") -> Optional[UtteranceEvent]:
        """Transcribe the complete utterance.

        ROOT CAUSE FIX: Uses rolling context for final transcription.
        Previously context was discarded and transcribe() was called
        with no prompt, losing all the accumulated partial context.
        """
        pcm = self._frames_to_bytes(frames)
        dur_ms = len(frames) * (FRAME_SAMPLES / SAMPLE_RATE * 1000)
        num_samples = len(pcm) // 2
        logger.info("[STREAM-STT] Finalizing utterance: duration=%.0fms "
                    "frames=%d pcm_bytes=%d samples=%d context=%d chars",
                    dur_ms, len(frames), len(pcm), num_samples, len(context))

        if dur_ms < MIN_UTTERANCE_MS or len(pcm) < 512:
            logger.info(
                "[STREAM-STT] Utterance DISCARDED reason=too_short "
                "(duration=%.0fms < %dms, bytes=%d)",
                dur_ms, MIN_UTTERANCE_MS, len(pcm))
            return None

        loop = asyncio.get_event_loop()
        # ROOT CAUSE FIX: Use context-aware transcription for final too
        if context:
            text = await loop.run_in_executor(
                None, self._whisper.transcribe_with_context,
                pcm, SAMPLE_RATE, context)
        else:
            text = await loop.run_in_executor(
                None, self._whisper.transcribe, pcm, SAMPLE_RATE, False)

        if not text:
            logger.info(
                "[STREAM-STT] Utterance DISCARDED reason=empty_transcript "
                "(duration=%.0fms samples=%d)",
                dur_ms, num_samples)
            return None

        # ── Post-process: normalize and correct common Whisper mistakes ──
        text = _postprocess_transcript(text)

        logger.info("[STREAM-STT] Final transcript (%.0fms): '%s'",
                    dur_ms, text)
        return UtteranceEvent(
            kind="final", text=text, is_final=True,
            started_at=start, ended_at=time.time(), audio=pcm)

    @staticmethod
    def _frames_to_bytes(frames: List[np.ndarray]) -> bytes:
        """Pack NON-OVERLAPPING frames into PCM16 bytes for Whisper.

        SINK BOUNDARY: frames are float32 [-1, 1]; the single int16
        conversion happens HERE, immediately before Whisper.
        """
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

    # ── Interruption detection while Leo speaks ───────────

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


# ═══════════════════════════════════════════════════════════════
# Post-processing: normalize and correct common Whisper mistakes
# ═══════════════════════════════════════════════════════════════

# Common Whisper mistakes → corrections
_WHISPER_CORRECTIONS = {
    # Common hallucination patterns
    "you too fo me": "YouTube for me",
    "you too": "YouTube",
    "you tube": "YouTube",
    "u tube": "YouTube",
    "spot if i": "Spotify",
    "spot a fire": "Spotify",
    "spot of eye": "Spotify",
    "net flicks": "Netflix",
    "netflix": "Netflix",
    "face book": "Facebook",
    "what's up": "WhatsApp",
    "whats up": "WhatsApp",
    "what sup": "WhatsApp",
    "visual studio": "Visual Studio",
    "vs code": "VS Code",
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
    "chat g p": "ChatGPT",
    "jet brains": "JetBrains",
    "pie charm": "PyCharm",
    "pycharm": "PyCharm",
    "pie chum": "PyCharm",
    "python": "Python",
    "pi thon": "Python",
    "java script": "JavaScript",
    "type script": "TypeScript",
    "get hub": "GitHub",
    "git hub": "GitHub",
    "get lab": "GitLab",
    "git lab": "GitLab",
    "stack over flow": "Stack Overflow",
    "stack overflow": "Stack Overflow",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k eight s": "K8s",
    "kubectl": "kubectl",
    "cube cuddle": "kubectl",
    "cube control": "kubectl",
    "terminal": "Terminal",
    "terminals": "Terminal",
    "file explorer": "File Explorer",
    "files": "Files",
    "settings": "Settings",
    "system settings": "System Settings",
    "task manager": "Task Manager",
    "calculator": "Calculator",
    "calendar": "Calendar",
    "notepad": "Notepad",
    "note pad": "Notepad",
    "word": "Word",
    "excel": "Excel",
    "power point": "PowerPoint",
    "outlook": "Outlook",
    "teams": "Teams",
    "slack": "Slack",
    "discord": "Discord",
    "zoom": "Zoom",
    "telegram": "Telegram",
    "signal": "Signal",
    "whatsapp": "WhatsApp",
    "messenger": "Messenger",
    "instagram": "Instagram",
    "twitter": "Twitter",
    "x dot com": "X",
    "reddit": "Reddit",
    "linkedin": "LinkedIn",
    "linked in": "LinkedIn",
    "amazon": "Amazon",
    "flipkart": "Flipkart",
    "swiggy": "Swiggy",
    "zomato": "Zomato",
    "uber": "Uber",
    "ola": "Ola",
    "gmail": "Gmail",
    "google drive": "Google Drive",
    "google docs": "Google Docs",
    "google sheets": "Google Sheets",
    "google slides": "Google Slides",
    "google meet": "Google Meet",
    "google maps": "Google Maps",
    "maps": "Maps",
    "photos": "Photos",
    "camera": "Camera",
    "music": "Music",
    "videos": "Videos",
    "documents": "Documents",
    "downloads": "Downloads",
    "desktop": "Desktop",
    "pictures": "Pictures",
    "home": "Home",
    "search": "Search",
    "open": "Open",
    "close": "Close",
    "start": "Start",
    "stop": "Stop",
    "pause": "Pause",
    "play": "Play",
    "next": "Next",
    "previous": "Previous",
    "volume up": "Volume Up",
    "volume down": "Volume Down",
    "mute": "Mute",
    "unmute": "Unmute",
    "brightness up": "Brightness Up",
    "brightness down": "Brightness Down",
    "screenshot": "Screenshot",
    "screen shot": "Screenshot",
    "screen share": "Screen Share",
    "screen record": "Screen Record",
    "lock": "Lock",
    "unlock": "Unlock",
    "shutdown": "Shutdown",
    "restart": "Restart",
    "sleep": "Sleep",
    "log out": "Logout",
    "sign out": "Sign Out",
    "sign in": "Sign In",
    "log in": "Login",
    "copy": "Copy",
    "paste": "Paste",
    "cut": "Cut",
    "delete": "Delete",
    "undo": "Undo",
    "redo": "Redo",
    "save": "Save",
    "save as": "Save As",
    "print": "Print",
    "export": "Export",
    "import": "Import",
    "refresh": "Refresh",
    "reload": "Reload",
    "back": "Back",
    "forward": "Forward",
    "zoom in": "Zoom In",
    "zoom out": "Zoom Out",
    "full screen": "Full Screen",
    "minimize": "Minimize",
    "maximize": "Maximize",
    "restore": "Restore",
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
    "home": "Home",
    "end": "End",
    "top": "Top",
    "bottom": "Bottom",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "enter": "Enter",
    "escape": "Escape",
    "tab": "Tab",
    "space": "Space",
    "backspace": "Backspace",
    "delete key": "Delete",
    "control": "Control",
    "alt": "Alt",
    "shift": "Shift",
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
    "insert": "Insert",
    "page up key": "Page Up",
    "page down key": "Page Down",
    "home key": "Home Key",
    "end key": "End Key",
}

# Contractions → expanded form
_CONTRACTIONS = {
    "i'm": "I am",
    "i've": "I have",
    "i'll": "I will",
    "i'd": "I would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "he'll": "he will",
    "she's": "she is",
    "she'll": "she will",
    "it's": "it is",
    "it'll": "it will",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "that's": "that is",
    "that'll": "that will",
    "what's": "what is",
    "what'll": "what will",
    "who's": "who is",
    "who'll": "who will",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "can't": "cannot",
    "cannot": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "ain't": "is not",
    "let's": "let us",
    "there's": "there is",
    "here's": "here is",
}


def _postprocess_transcript(text: str) -> str:
    """Normalize and correct common Whisper mistakes.

    Steps:
      1. Strip leading/trailing whitespace and punctuation artifacts
      2. Normalize casing (sentence case for commands)
      3. Expand contractions
      4. Apply fuzzy corrections for common Whisper mistakes
      5. Remove repeated words (hallucination artifact)
    """
    if not text:
        return text

    # 1. Clean up
    text = text.strip()
    # Remove leading punctuation artifacts
    text = re.sub(r'^[,.!?;:\s]+', '', text)
    text = re.sub(r'[,.!?;:\s]+$', '', text)

    # 2. Normalize casing — keep proper nouns capitalized, lowercase the rest
    # For short commands (< 5 words), use title case for readability
    words = text.split()
    if len(words) <= 5:
        # Short command — capitalize first letter of each significant word
        text = " ".join(w.capitalize() if len(w) > 2 else w for w in words)
    else:
        # Longer utterance — sentence case
        text = text[0].upper() + text[1:] if text else text

    # 3. Expand contractions
    text_lower = text.lower()
    for contraction, expanded in _CONTRACTIONS.items():
        if contraction in text_lower:
            # Case-preserving replacement
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            text = pattern.sub(expanded, text)

    # 4. Apply Whisper corrections (case-insensitive)
    for wrong, correct in sorted(_WHISPER_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(correct, text)

    # 5. Remove repeated words (common hallucination: "open open firefox")
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # 6. Normalize whitespace
    text = " ".join(text.split())

    return text


# Global singleton
streaming_stt = StreamingSTT()