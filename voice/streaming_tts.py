"""
StreamingTTS — Interruptible, streaming text-to-speech playback.

KEY DIFFERENCE from the old TTS (subprocess pw-play/paplay):
  - Audio is synthesized sentence-by-sentence and played through a
    sounddevice.OutputStream that we OWN and can ABORT instantly.
  - Interruption is immediate: calling `stop()` aborts playback mid-sample
    and clears the queue. No waiting for a subprocess to finish.
  - Synthesis of the NEXT sentence happens WHILE the current one plays,
    so there's no gap between sentences.

Architecture:
  LLM sentences ──▶ sentence queue ──▶ synthesizer worker ──▶ audio chunk queue ──▶ playback worker ──▶ speakers
                         (async)              (thread)              (async queue)         (thread)

Interruption path:
  user speaks ──▶ engine calls stop() ──▶ playback aborted + queues cleared + synthesis cancelled

Usage:
    from voice.streaming_tts import streaming_tts

    streaming_tts.initialize()
    await streaming_tts.speak_sentences(sentence_generator, interrupt_event)
    streaming_tts.stop()   # instant abort
"""

import asyncio
import logging
import queue
import threading
import time
from typing import AsyncIterator, Optional

import numpy as np

from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# Playback sample rate (Kokoro/XTTS output 24kHz; we resample if needed)
PLAY_SAMPLE_RATE = 24000


class _InterruptiblePlayer:
    """
    Owns a sounddevice.OutputStream and plays int16 PCM chunks.

    OWNERSHIP CONTRACT: the OutputStream is created ONCE (lazily on first
    write) and destroyed ONCE (in close() at shutdown). It is NEVER
    destroyed during interruption — only stopped. This guarantees that
    the PortAudio/ALSA C memory is NEVER freed while the playback thread
    is inside write(), eliminating the use-after-free → heap corruption.

    The worker thread is PERSISTENT: it runs for the lifetime of the player
    (until close()), surviving utterance boundaries. `finish()` merely marks
    the end of one utterance so `is_playing` settles — it never kills the
    worker, so the next utterance's chunks are always played.
    """

    def __init__(self, sample_rate: int = PLAY_SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._sd = None
        self._stream_lock = threading.Lock()       # guards _stream create/destroy ONLY
        self._stream = None                        # persistent OutputStream (id fixed at log)
        self._stream_created: bool = False         # True once created, False until close()
        self._playing = False
        self._abort = threading.Event()             # interrupt current utterance
        self._shutdown = threading.Event()          # kill the worker thread
        self._chunk_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        # ── Race-free interruption protocol ──────────────────
        # NEVER call _stream.stop() from the caller thread.
        # The worker thread is the SOLE owner of stop().
        # interrupt() sets a flag; the worker executes stop()
        # after it has safely returned from write().
        self._stop_requested = threading.Event()    # caller → worker: "please stop"
        self._stop_executed = threading.Event()     # worker → caller: "stop complete"
        self._write_active = threading.Event()      # worker is inside write()

    # ── Stream lifecycle ───────────────────────────────────

    def _ensure_stream(self) -> bool:
        """Lazily create the OutputStream. Called ONLY from the playback
        worker thread (tts-playback). Creates once, never recreates after
        interruption — the stream lives until close() at shutdown."""
        with self._stream_lock:
            if self._stream is not None:
                return True
            if self._shutdown.is_set():
                return False
            try:
                if self._sd is None:
                    import sounddevice as sd
                    self._sd = sd
                self._stream = self._sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=0,  # let PortAudio pick for low latency
                )
                stream_id = id(self._stream)
                self._stream.start()
                self._stream_created = True
                logger.info("[PLAYER] OutputStream CREATED id=%s "
                            "thread=%s — persistent, lives until shutdown",
                            stream_id, threading.current_thread().name)
                return True
            except Exception as e:
                logger.error("[PLAYER] Failed to open output stream: %s", e)
                self._stream = None
                return False

    # ── Worker thread ──────────────────────────────────────

    def start_worker(self) -> None:
        """Start the persistent background playback thread."""
        if self._thread and self._thread.is_alive():
            logger.debug("[PLAYER] Worker thread already running (name=%s)",
                         self._thread.name)
            return
        self._shutdown.clear()
        self._abort.clear()
        self._thread = threading.Thread(
            target=self._run, name="tts-playback", daemon=True)
        self._thread.start()
        logger.info("[PLAYER] Worker thread STARTED name=%s",
                    self._thread.name)

    def _run(self) -> None:
        """Persistent playback loop. Exits only on close()/shutdown.

        NEVER touches self._stream_lock — only reads self._stream.
        The stream is never destroyed while this thread is alive,
        so a bare read is safe (no use-after-free possible).

        STOP PROTOCOL: only this thread ever calls _stream.stop().
        When the caller wants to interrupt, it sets _stop_requested.
        This thread checks the flag before write() (to skip the write)
        and after write() returns (to execute the stop), guaranteeing
        that stop() NEVER races with write() inside PortAudio/ALSA."""
        tid = threading.get_ident()
        logger.info("[PLAYER] Playback loop ENTERED thread=%s/%s",
                    threading.current_thread().name, tid)
        while not self._shutdown.is_set():
            # ── Check stop-requested BEFORE blocking on the queue ──
            self._service_stop_request()

            try:
                chunk = self._chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:  # end-of-utterance marker (NOT thread exit)
                self._playing = False
                logger.debug("[PLAYER] End-of-utterance marker received")
                continue

            if self._abort.is_set():
                # Drop chunks for an interrupted utterance.
                # Do NOT touch the stream — it stays alive.
                self._playing = False
                logger.debug("[PLAYER] Chunk dropped (abort set, queue_size=%d)",
                             self._chunk_queue.qsize())
                continue

            # ── Check stop-requested BEFORE calling write() ──
            if self._stop_requested.is_set():
                self._execute_stop()
                continue

            if not self._ensure_stream():
                # Can't play; drop the chunk but keep the worker alive
                continue

            try:
                data = np.frombuffer(chunk, dtype=np.int16)
                self._playing = True
                logger.debug("[PLAYER] WRITE START  thread=%s/%s  "
                             "chunk_samples=%d  stream_id=%s",
                             threading.current_thread().name, tid,
                             len(data), id(self._stream))
                # Signal that we are inside write().
                # interrupt() will wait for this to clear before
                # considering the stop protocol complete.
                self._write_active.set()
                self._stream.write(data)
            except Exception as e:
                logger.debug("[PLAYER] write error: %s", e)
                # Stream error — mark it for lazy recreation on next write.
                with self._stream_lock:
                    if self._stream is not None:
                        logger.warning("[PLAYER] Stream error on write — "
                                       "will recreate on next write. "
                                       "stream_id=%s error=%s",
                                       id(self._stream), e)
                        try:
                            self._stream.close()
                        except Exception:
                            pass
                        self._stream = None
            finally:
                self._write_active.clear()
                self._playing = False
                logger.debug("[PLAYER] WRITE END    thread=%s/%s  "
                             "stream_id=%s",
                             threading.current_thread().name, tid,
                             id(self._stream) if self._stream else "none")

            # ── Check stop-requested AFTER write() returned ──
            # This is the SAFE point: write() has definitely returned,
            # so calling stop() cannot race with PortAudio internals.
            self._service_stop_request()
        self._playing = False
        logger.info("[PLAYER] Playback loop EXITED thread=%s/%s",
                    threading.current_thread().name, tid)

    def _service_stop_request(self) -> None:
        """If the caller requested a stream stop, execute it HERE (worker thread).

        This is the ONLY place _stream.stop() is ever called during
        normal operation.  Guarantees single-thread ownership of
        PortAudio's stream-stop path."""
        if self._stop_requested.is_set():
            self._execute_stop()

    def _execute_stop(self) -> None:
        """Execute stream.stop() from the worker thread and signal completion."""
        tid = threading.get_ident()
        logger.info("[PLAYER] STOP EXECUTED thread=%s/%s  "
                    "stream_id=%s",
                    threading.current_thread().name, tid,
                    id(self._stream) if self._stream else "none")
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    logger.info("[PLAYER] Stream STOPPED — buffered audio dropped "
                                "(stream alive)  stream_id=%s", id(self._stream))
                except Exception as e:
                    logger.debug("[PLAYER] Stream stop error (benign): %s", e)
        self._stop_requested.clear()
        self._stop_executed.set()

    # ── Queue operations ────────────────────────────────────

    def enqueue(self, pcm_bytes: bytes) -> None:
        """Queue a chunk for playback (non-blocking)."""
        if not self._shutdown.is_set() and not self._abort.is_set():
            self._chunk_queue.put(pcm_bytes)

    def finish(self) -> None:
        """Mark the end of the current utterance (does NOT stop the worker)."""
        if not self._shutdown.is_set():
            self._chunk_queue.put(None)

    # ── Interruption (SAFE — never destroys the stream) ────

    def interrupt(self) -> None:
        """
        INSTANT interrupt of the CURRENT utterance only.

        RACE-FREE PROTOCOL:
          The caller thread NEVER calls _stream.stop() directly.
          Instead it sets _stop_requested and waits for the worker
          thread to execute stop() after it has safely returned from
          write().  This eliminates the PortAudio/ALSA race where
          Pa_StopStream (stop) and Pa_WriteStream (write) execute
          concurrently on different threads inside the ALSA backend,
          which corrupts internal ALSA state and causes:
            PaAlsaStreamComponent_EndProcessing  →  assertion / abort.

          Protocol:
            1. Set _abort     → worker skips queued chunks
            2. Drain queue    → remove pending work
            3. Set _stop_requested → delegate stop() to the worker
            4. Wait for _write_active to clear   → write() has returned
            5. Wait for _stop_executed           → stop() has been called
            6. Re-arm for next utterance
        """
        caller_tid = threading.get_ident()
        caller_thread = threading.current_thread().name
        logger.info("[PLAYER] Interrupt received from thread=%s/%s "
                    "(queue_size=%d stream_id=%s stream_created=%s)",
                    caller_thread, caller_tid, self._chunk_queue.qsize(),
                    id(self._stream) if self._stream else "none",
                    self._stream_created)

        # Step 1: Set abort flag so the worker skips all pending chunks
        self._abort.set()

        # Step 2: Drain the pending chunk queue
        drained = 0
        try:
            while True:
                self._chunk_queue.get_nowait()
                drained += 1
        except queue.Empty:
            pass
        if drained:
            logger.info("[PLAYER] Interrupt drained %d pending chunks", drained)

        # Step 3: Request stop — but do NOT execute it here.
        # The worker thread will execute _stream.stop() from its own
        # context, after write() has safely returned.
        logger.info("[PLAYER] STOP REQUESTED  thread=%s/%s  → delegating to worker",
                    caller_thread, caller_tid)
        self._stop_executed.clear()
        self._stop_requested.set()

        # Step 4: Wait for any in-flight write() to return.
        # _write_active is set by the worker just before write() and
        # cleared in the finally block after write() returns.
        # Event.wait() is a blocking OS primitive — NOT busy-waiting.
        if self._write_active.is_set():
            logger.info("[PLAYER] Waiting for in-flight write() to return...  "
                        "thread=%s/%s", caller_thread, caller_tid)
            self._write_active.wait(timeout=2.0)
            logger.info("[PLAYER] Write gate cleared  thread=%s/%s",
                        caller_thread, caller_tid)

        # Step 5: Wait for the worker to actually call _stream.stop().
        # The worker checks _stop_requested at the top of every loop
        # iteration and after every write() — so it will pick up the
        # request promptly.
        if not self._stop_executed.wait(timeout=2.0):
            logger.warning("[PLAYER] stop() not acknowledged by worker "
                           "within timeout — stream may still be running")

        logger.info("[PLAYER] STOP EXECUTED confirmed  thread=%s/%s",
                    caller_thread, caller_tid)

        self._playing = False

        # Step 6: Re-arm for the next utterance.
        self._abort.clear()
        logger.info("[PLAYER] Interrupt complete — re-armed for next utterance  "
                    "thread=%s/%s", caller_thread, caller_tid)


    def stop(self) -> None:
        """Alias for interrupt() — stops current playback immediately."""
        self.interrupt()

    @property
    def is_playing(self) -> bool:
        """True while audio is actively writing or chunks are queued."""
        return self._playing or not self._chunk_queue.empty()

    # ── Shutdown (the ONLY place the stream is destroyed) ──

    def close(self) -> None:
        """
        Kill the worker thread and DESTROY the stream.

        THIS IS THE ONLY PLACE THE OUTPUTSTREAM IS CLOSED.

        Guarantees:
          1. Tells the worker to exit (shutdown flag)
          2. Waits for the worker thread to join — at this point
             _stream.write() is guaranteed to have returned
          3. THEN closes and destroys the stream from the calling
             thread (always the main/asyncio thread at shutdown)
        """
        caller_thread = threading.current_thread().name
        logger.info("[PLAYER] Close requested from thread=%s "
                    "(stream_id=%s stream_created=%s worker_alive=%s)",
                    caller_thread,
                    id(self._stream) if self._stream else "none",
                    self._stream_created,
                    self._thread.is_alive() if self._thread else False)

        # Step 1: Tell worker thread to exit
        self._shutdown.set()
        self._abort.set()

        # Step 2: Drain the queue so the worker isn't blocked on put()
        drained = 0
        try:
            while True:
                self._chunk_queue.get_nowait()
                drained += 1
        except queue.Empty:
            pass
        if drained:
            logger.info("[PLAYER] Close drained %d remaining chunks", drained)

        # Step 3: Wait for the worker thread to exit.
        # The worker loop checks _shutdown on every iteration.
        # After join() returns, _stream.write() is guaranteed done.
        if self._thread and self._thread.is_alive():
            worker_name = self._thread.name
            logger.info("[PLAYER] Waiting for worker thread '%s' to exit...",
                        worker_name)
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("[PLAYER] Worker thread '%s' did not exit "
                               "within timeout — forcing stream close anyway",
                               worker_name)
            else:
                logger.info("[PLAYER] Worker thread '%s' confirmed EXITED",
                            worker_name)
        else:
            logger.info("[PLAYER] No worker thread to join (was never started "
                        "or already dead)")

        # Step 4: NOW it's safe to destroy the stream.
        # The worker thread is dead — no one else can touch _stream.
        with self._stream_lock:
            if self._stream is not None:
                stream_id = id(self._stream)
                try:
                    self._stream.abort()
                    self._stream.close()
                    logger.info("[PLAYER] OutputStream DESTROYED id=%s "
                                "— ALSA resources freed safely", stream_id)
                except Exception as e:
                    logger.debug("[PLAYER] Stream close error (benign): %s", e)
                self._stream = None
                self._stream_created = False
            else:
                logger.info("[PLAYER] No stream to destroy (was never created)")

        self._playing = False
        logger.info("[PLAYER] Close completed")


class StreamingTTS:
    """
    Streaming, interruptible TTS orchestrator.

    Synthesizes sentences with the active engine (Kokoro→XTTS→Piper→pyttsx3)
    and plays them through an interruptible sounddevice stream.
    """

    def __init__(self):
        self._player = _InterruptiblePlayer()
        self._engine = None  # active synthesis engine (with synthesize())
        self._ready = False
        self._synth_executor = None
        self._stop_event = threading.Event()
        self._speaking = threading.Event()
        self._sample_rate = PLAY_SAMPLE_RATE

    # ── Engine selection ──────────────────────────────────

    def initialize(self) -> bool:
        """Pick the best available synthesis engine."""
        logger.info("[STREAM-TTS] Initializing streaming TTS...")
        self._engine = self._pick_engine()
        if self._engine is None:
            logger.error("[STREAM-TTS] No synthesis engine available")
            self._ready = False
            return False
        self._player.start_worker()
        self._ready = True
        logger.info("[STREAM-TTS] Ready (engine=%s)", self._engine_name)
        return True

    def _pick_engine(self):
        """Return an object with .synthesize(text)->Optional[np.float32] and .sample_rate."""
        # Priority: Kokoro → XTTS → Piper → pyttsx3
        for name in ("kokoro", "xtts", "piper", "pyttsx3"):
            eng = self._make_engine(name)
            if eng is not None:
                return eng
        return None

    def _make_engine(self, name: str):
        try:
            if name == "kokoro":
                return _KokoroSynth()
            if name == "xtts":
                return _XTTSSynth()
            if name == "piper":
                return _PiperSynth()
            if name == "pyttsx3":
                return _Pyttsx3Synth()
        except Exception as e:
            logger.debug("[STREAM-TTS] engine %s unavailable: %s", name, e)
        return None

    @property
    def _engine_name(self) -> str:
        return type(self._engine).__name__ if self._engine else "none"

    # ── Public API ────────────────────────────────────────

    async def speak_sentences(
        self,
        sentences: AsyncIterator[str],
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> bool:
        """
        Consume an async stream of sentences; synthesize & play each.

        Synthesis of sentence N+1 overlaps playback of sentence N.
        If `interrupt_event` fires, playback and synthesis stop instantly.

        Returns:
            True if at least one audio chunk was queued for playback,
            False if nothing was spoken (TTS unavailable, all synthesis
            failed, or interrupted before any audio).
        """
        if not self._ready:
            # Lazy init
            ok = await asyncio.get_event_loop().run_in_executor(None, self.initialize)
            if not ok:
                return False

        self._stop_event.clear()
        self._speaking.set()
        loop = asyncio.get_event_loop()
        played_any = False

        try:
            async for sentence in sentences:
                if self._should_stop(interrupt_event):
                    break
                text = sentence.strip()
                if not text or text.startswith("ACTION:"):
                    continue

                # Synthesize (in thread so we don't block the loop)
                pcm = await loop.run_in_executor(None, self._synthesize, text)
                if self._should_stop(interrupt_event):
                    break
                if pcm is not None and len(pcm):
                    self._player.enqueue(pcm)
                    played_any = True
                    # Give the player a moment to start so is_playing is accurate
                    await asyncio.sleep(0)

            # Signal end of utterance
            if not self._should_stop(interrupt_event):
                self._player.finish()
        finally:
            self._speaking.clear()

        return played_any

    def _should_stop(self, interrupt_event: Optional[asyncio.Event]) -> bool:
        if self._stop_event.is_set():
            return True
        if interrupt_event is not None and interrupt_event.is_set():
            return True
        return False

    def _synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to int16 PCM bytes at the player sample rate."""
        if self._stop_event.is_set() or self._engine is None:
            return None
        try:
            audio = self._engine.synthesize(text)  # float32 [-1,1] @ engine rate
            if audio is None or len(audio) == 0:
                return None
            audio = np.asarray(audio, dtype=np.float32)
            # Resample if engine rate differs
            src_rate = getattr(self._engine, "sample_rate", self._sample_rate)
            if src_rate != self._sample_rate:
                audio = self._resample(audio, src_rate, self._sample_rate)
            audio = np.clip(audio, -1.0, 1.0)
            return (audio * 32767.0).astype(np.int16).tobytes()
        except Exception as e:
            logger.warning("[STREAM-TTS] synthesize error: %s", e)
            return None

    @staticmethod
    def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
        try:
            from scipy import signal as _sig
            import math
            g = math.gcd(src, dst)
            up, down = dst // g, src // g
            return _sig.resample_poly(audio, up, down).astype(np.float32)
        except Exception:
            # Fallback: naive linear interpolation
            duration = len(audio) / src
            n = int(duration * dst)
            x_old = np.linspace(0, duration, len(audio), endpoint=False)
            x_new = np.linspace(0, duration, n, endpoint=False)
            return np.interp(x_new, x_old, audio).astype(np.float32)

    # ── Interruption ──────────────────────────────────────

    def stop(self) -> None:
        """Immediately stop synthesis and playback (user interruption)."""
        logger.info("[STREAM-TTS] stop() called — interrupting playback")
        self._stop_event.set()
        self._player.stop()
        self._speaking.clear()
        logger.info("[STREAM-TTS] stop() complete")

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set() or self._player.is_playing

    @property
    def ready(self) -> bool:
        return self._ready

    def close(self) -> None:
        """Graceful shutdown: stop playback, kill worker, destroy stream."""
        logger.info("[STREAM-TTS] close() called — graceful shutdown")
        self.stop()
        self._player.close()
        self._ready = False
        logger.info("[STREAM-TTS] close() complete")


# ── Synthesis engine adapters ─────────────────────────────
# Each returns float32 audio in [-1, 1] and exposes .sample_rate.

class _KokoroSynth:
    """Kokoro-82M synthesis adapter (primary, CPU-friendly).

    Runs on CPU by default: the GPU is usually occupied by the local LLM
    (Ollama), and Kokoro-82M is fast enough on CPU for sentence streaming.
    Set TTS_DEVICE=cuda to override when the GPU is free.
    """
    sample_rate = 24000

    def __init__(self):
        from kokoro import KPipeline  # noqa
        import torch
        device = self._pick_device()
        self._pipe = KPipeline(lang_code="a", device=device)
        self._voice = getattr(voice_settings, "kokoro_voice", "af_heart") or "af_heart"

    @staticmethod
    def _pick_device() -> str:
        import torch
        cfg = getattr(voice_settings, "tts_device", "cpu")
        if cfg == "cuda" and torch.cuda.is_available():
            # Only use CUDA if there's actually free VRAM (>= ~1 GiB)
            try:
                free, _total = torch.cuda.mem_get_info()
                if free > 1 << 30:
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return "cpu"

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        import torch
        chunks = []
        for result in self._pipe(text, voice=self._voice, speed=1.0):
            chunks.append(result.audio)
        if not chunks:
            return None
        audio = torch.cat(chunks, dim=-1).cpu().numpy().astype(np.float32)
        return audio


class _XTTSSynth:
    """Coqui XTTS v2 synthesis adapter (GPU preferred)."""
    sample_rate = 24000

    def __init__(self):
        from TTS.api import TTS  # noqa
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        wav = self._tts.tts(text=text, speaker_wav=None, language="en")
        return np.asarray(wav, dtype=np.float32)


class _PiperSynth:
    """Piper synthesis adapter (fast, local)."""
    sample_rate = 22050

    def __init__(self):
        # Piper is used via subprocess normally; for streaming we use the
        # python API if available, else mark unavailable.
        from piper import PiperVoice  # noqa
        model = getattr(voice_settings, "piper_model", None)
        if not model:
            raise RuntimeError("piper model not configured")
        self._voice = PiperVoice.load(model)
        self.sample_rate = self._voice.config.sample_rate

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        audio = b"".join(
            chunk.audio_int16_bytes for chunk in self._voice.synthesize_stream_raw(text)
        )
        arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return arr


class _Pyttsx3Synth:
    """pyttsx3 emergency fallback — renders to a temp WAV then loads it."""
    sample_rate = 22050

    def __init__(self):
        import pyttsx3  # noqa
        self._pyttsx3 = pyttsx3

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        import tempfile, wave
        engine = self._pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        engine.save_to_file(text, path)
        engine.runAndWait()
        try:
            with wave.open(path, "rb") as w:
                self.sample_rate = w.getframerate()
                frames = w.readframes(w.getnframes())
                width = w.getsampwidth()
            if width == 2:
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                arr = (arr - 128.0) / 128.0
            return arr
        except Exception:
            return None
        finally:
            try:
                import os
                os.unlink(path)
            except Exception:
                pass
            try:
                engine.stop()
            except Exception:
                pass


# Global singleton
streaming_tts = StreamingTTS()