"""
StreamingSTT — Streaming VAD + Whisper with smart endpointing.

Replaces the old "record whole command, then transcribe" flow with a
streaming pipeline:

  ring buffer ──▶ 30ms frames ──▶ Silero VAD ──▶ speech segments
                                        │
                                        ├─▶ partial Whisper (every ~0.4s of new speech)
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

Usage:
    from voice.streaming_stt import streaming_stt

    async for event in streaming_stt.stream_utterances():
        if event.kind == "partial":  ...
        elif event.kind == "final":  ...
        elif event.kind == "speech_start": ...
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

import numpy as np

from voice.audio_manager import audio_manager, SAMPLE_RATE, FRAME_SAMPLES
from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# ── Endpointing configuration (configurable via voice_settings) ──
MIN_PAUSE_MS = getattr(voice_settings, "conv_min_pause_ms", 600)        # <600ms pause does NOT end the turn
ENDPOINT_SILENCE_MS = getattr(voice_settings, "conv_endpoint_ms", 900)  # trailing silence finalizes the turn
MIN_UTTERANCE_MS = 250        # ignore blips shorter than this
MAX_UTTERANCE_S = 20.0        # hard cap on a single utterance
PARTIAL_INTERVAL_S = 0.4      # run partial Whisper every N seconds of new audio
PRE_ROLL_MS = 300             # audio kept before speech onset
INTERRUPT_MIN_MS = getattr(voice_settings, "conv_interrupt_min_ms", 90)  # sustained speech to interrupt

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

    silero-vad 6.x accepts ONLY 256/512/768-sample windows at 16 kHz —
    the legacy 480-sample (30 ms) frame raises "Input audio chunk is too
    short", which the old fallback path swallowed, silently degrading the
    VAD to an energy heuristic. Frames are padded to 512 samples here.
    """

    FRAME = 512  # 32 ms @ 16 kHz (valid silero-vad 6.x window)

    def __init__(self):
        self._model = None
        self._ready = False
        self._threshold = 0.5


    def load(self) -> bool:
        if self._ready:
            return True
        # Prefer the installed silero_vad package (offline, no download)
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            self._ready = True
            logger.info("[STREAM-STT] Silero VAD loaded (silero_vad pkg)")
            return True
        except Exception as e:
            logger.debug("[STREAM-STT] silero_vad pkg failed: %s", e)
        # Fallback: torch.hub (downloads on first use)
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
        """Return speech probability for a 30ms 16kHz frame.

        VAD preprocessing applies NO gain: float32 [-1, 1] frames are
        used AS-IS (no renormalization); legacy int16 frames are decoded
        once via /32768 at this model boundary.
        """
        if not self._ready:
            # Energy fallback (threshold stays on the int16 scale)
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
            # silero-vad 6.x requires 256/512/768-sample windows; the
            # pipeline's 480-sample frames are zero-padded to 512.
            if len(audio) < self.FRAME:
                audio = np.pad(audio, (0, self.FRAME - len(audio)))
            elif len(audio) > self.FRAME:
                audio = audio[: self.FRAME]
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

    def __init__(self):
        self._model = None
        self._ready = False
        self._lock = asyncio.Lock()

    def load(self) -> bool:
        """Load faster-whisper with a PROVEN backend.

        CUDA libraries (libcublas) load lazily at FIRST INFERENCE, so a
        model that "loaded" on cuda can still fail every transcribe call
        and silently return ''. Each candidate backend must pass a real
        warmup inference before it is accepted; otherwise we fall back
        to the next candidate (cuda → cpu).
        """
        if self._ready:
            return True
        try:
            from faster_whisper import WhisperModel
            import torch
            candidates = []
            if torch.cuda.is_available():
                candidates.append(("cuda", "float16"))
            candidates.append(("cpu", "int8"))
            for device, compute in candidates:
                try:
                    # tiny/base gives the best latency for partials
                    model = WhisperModel("base", device=device, compute_type=compute)
                    # Warmup inference: forces the backend libraries to
                    # load NOW. A backend that can't infer (missing
                    # libcublas, OOM) raises here and is rejected.
                    warmup = np.zeros(16000, dtype=np.float32)
                    segments, _ = model.transcribe(
                        warmup, beam_size=1, without_timestamps=True)
                    list(segments)  # consume the generator (runs inference)
                    self._model = model
                    self._ready = True
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


    def transcribe(self, pcm_int16: bytes, sample_rate: int = SAMPLE_RATE,
                   use_vad_filter: bool = True) -> str:
        """Transcribe int16 PCM to text. Returns '' on failure.

        use_vad_filter: faster-whisper's internal VAD. Disable it when the
        caller ALREADY gates speech upstream (wake confirmation): the
        internal filter aggressively drops quiet-but-real speech
        ("VAD filter removed 00:02.500 of audio" on a played wake phrase).
        """
        if not self._ready or not pcm_int16:
            return ""
        try:
            # Stage trace: peak/RMS of the exact audio handed to Whisper.
            from voice.audio_processing import peak_monitor
            peak_monitor.log("whisper", np.frombuffer(pcm_int16, dtype=np.int16))
            audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio) < sample_rate * 0.2:
                return ""
            segments, _info = self._model.transcribe(
                audio,
                beam_size=1,
                language="en",
                vad_filter=use_vad_filter,
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        except Exception as e:
            # No hidden exceptions: a failing backend must be visible.
            logger.warning("[STREAM-STT] transcribe error: %s", e)
            return ""




class StreamingSTT:
    """
    Streaming speech-to-text with VAD endpointing and partial results.

    Reads 30ms frames from the AudioManager ring buffer and emits
    UtteranceEvents. Fully async; supports cancellation.
    """

    def __init__(self):
        self._vad = _SileroVAD()
        self._whisper = _WhisperTranscriber()
        self._ready = False
        self._listen_enabled = asyncio.Event()
        self._listen_enabled.set()
        self._cancel = asyncio.Event()

    def initialize(self) -> bool:
        """Load VAD + Whisper models."""
        vad_ok = self._vad.load()
        whisper_ok = self._whisper.load()
        self._ready = whisper_ok  # VAD optional (energy fallback)
        if not whisper_ok:
            logger.error("[STREAM-STT] Whisper unavailable — streaming STT disabled")
        return self._ready

    @property
    def ready(self) -> bool:
        return self._ready

    def pause_listening(self) -> None:
        """Temporarily stop emitting (e.g. during face auth)."""
        self._listen_enabled.clear()

    def resume_listening(self) -> None:
        self._listen_enabled.set()

    def cancel(self) -> None:
        """Cancel any in-flight streaming loop."""
        self._cancel.set()

    def stop_streaming(self) -> None:
        """DESTROY the active streaming session (called when leaving
        COMMAND_LISTEN).

        The state machine guarantees streaming Whisper exists ONLY while a
        stream_utterances() consumer is active. Setting the cancel flag makes
        the in-flight generator exit immediately; the NEXT session re-arms
        itself via reset_cancel() at the top of stream_utterances().
        """
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
          - partial: incremental transcription (~every 0.4s of new audio)
          - final: the complete utterance after endpointing

        Runs until cancelled (self.cancel() or task cancellation).
        """
        if not self._ready:
            if not self.initialize():
                return

        self.reset_cancel()
        last_total = audio_manager.total_samples

        speech_frames: List[np.ndarray] = []
        pre_roll: List[np.ndarray] = []
        in_speech = False
        speech_start_time = 0.0
        last_voice_time = 0.0
        silence_run_ms = 0.0
        last_partial_len = 0
        last_partial_time = 0.0
        loop = asyncio.get_event_loop()

        logger.info("[STREAM-STT] Listening started (endpoint=%dms, min_pause=%dms)",
                    ENDPOINT_SILENCE_MS, MIN_PAUSE_MS)

        while not self._cancel.is_set():
            await self._listen_enabled.wait()
            if self._cancel.is_set():
                break

            # Pull new frames from the ring buffer (non-blocking-ish)
            new_audio, last_total = audio_manager.read_since(last_total)
            if len(new_audio) == 0:
                await asyncio.sleep(0.01)
                continue

            # Process in 30ms frames
            for frame in self._iter_frames(new_audio):
                prob = await loop.run_in_executor(None, self._vad.speech_prob, frame)
                is_speech = prob > 0.5
                now = time.time()

                # Maintain pre-roll buffer (audio just before speech onset)
                pre_roll.append(frame)
                max_pre = max(1, int((PRE_ROLL_MS / 1000.0) / 0.03))
                if len(pre_roll) > max_pre:
                    pre_roll.pop(0)

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        speech_start_time = now
                        speech_frames = list(pre_roll)  # include pre-roll
                        last_partial_len = 0
                        last_partial_time = now
                        logger.info("[STREAM-STT] Speech start")
                        yield UtteranceEvent(
                            kind="speech_start", started_at=now)
                    speech_frames.append(frame)
                    last_voice_time = now
                    silence_run_ms = 0.0
                else:
                    if in_speech:
                        speech_frames.append(frame)  # keep trailing audio for context
                        silence_run_ms = (now - last_voice_time) * 1000.0

                        # Check endpoint
                        if silence_run_ms >= ENDPOINT_SILENCE_MS:
                            final = await self._finalize(speech_frames, speech_start_time)
                            # Reset state
                            in_speech = False
                            speech_frames = []
                            silence_run_ms = 0.0
                            if final is not None:
                                # Filler-only utterance: keep turn open
                                if is_filler(final.text):
                                    logger.info(
                                        "[STREAM-STT] Filler '%s' — turn stays open",
                                        final.text)
                                    continue
                                yield final
                            continue

                # Partial transcription while in speech
                if in_speech:
                    audio_len_ms = len(speech_frames) * 30
                    new_since_partial = (len(speech_frames) - last_partial_len) * 30
                    if (now - last_partial_time) >= PARTIAL_INTERVAL_S and new_since_partial >= 300:
                        pcm = self._frames_to_bytes(speech_frames)
                        text = await loop.run_in_executor(
                            None, self._whisper.transcribe, pcm, SAMPLE_RATE)
                        last_partial_len = len(speech_frames)
                        last_partial_time = now
                        if text and not is_filler(text):
                            yield UtteranceEvent(
                                kind="partial", text=text, is_final=False,
                                started_at=speech_start_time)

                    # Hard cap on utterance length
                    if (now - speech_start_time) >= MAX_UTTERANCE_S:
                        final = await self._finalize(speech_frames, speech_start_time)
                        in_speech = False
                        speech_frames = []
                        if final is not None and not is_filler(final.text):
                            yield final

    async def _finalize(self, frames: List[np.ndarray], start: float) -> Optional[UtteranceEvent]:
        """Transcribe the complete utterance."""
        pcm = self._frames_to_bytes(frames)
        dur_ms = len(frames) * 30
        if dur_ms < MIN_UTTERANCE_MS or len(pcm) < 512:
            return None
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, self._whisper.transcribe, pcm, SAMPLE_RATE)
        if not text:
            return None
        logger.info("[STREAM-STT] Final (%dms): '%s'", dur_ms, text)
        return UtteranceEvent(
            kind="final", text=text, is_final=True,
            started_at=start, ended_at=time.time(), audio=pcm)

    @staticmethod
    def _frames_to_bytes(frames: List[np.ndarray]) -> bytes:
        """Pack frames into PCM16 bytes for Whisper.

        SINK BOUNDARY: frames are float32 [-1, 1]; the single int16
        conversion happens HERE, immediately before Whisper, and nowhere
        upstream in the pipeline.
        """
        if not frames:
            return b""
        from voice.audio_processing import float32_to_int16
        return float32_to_int16(np.concatenate(frames)).tobytes()

    @staticmethod
    def _iter_frames(audio: np.ndarray):
        """Yield 30ms frames from an arbitrary-length int16 array."""
        n = len(audio)
        for i in range(0, n - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            yield audio[i:i + FRAME_SAMPLES]

    # ── Interruption detection while Leo speaks ───────────

    async def detect_interruption(self, stop_event: asyncio.Event) -> None:
        """
        Watch for user speech while Leo is talking. On confident speech,
        set `stop_event` so the engine aborts TTS and listens.

        This is the full-duplex path: it runs concurrently with TTS playback.
        """
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
                    speech_run_ms += 30
                    # Require sustained speech to avoid TTS echo false-positives
                    if speech_run_ms >= INTERRUPT_MIN_MS:
                        logger.info("[STREAM-STT] Interruption detected (user speaking)")
                        stop_event.set()
                        return
                else:
                    speech_run_ms = 0.0


# Global singleton
streaming_stt = StreamingSTT()
