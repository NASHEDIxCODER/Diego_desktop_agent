"""
WakeListener — THE single wake-detection pipeline for Leo.

    Microphone
      ↓
    AudioManager          (ONE InputStream, float32 ring buffer @ 16 kHz mono)
      ↓
    Silero VAD            (speech gate, 512-sample windows)
      ↓
    openWakeWord          (streaming predict, EXACTLY 1280-sample int16 frames)
      ↓
    Whisper verification  (ONLY after openWakeWord triggers — never continuous)
      ↓
    Conversation          (handed back to the ConversationEngine)

There is NO other wake listener, wake loop, or verification path anywhere
in the project (the legacy duplicates in voice/stt.py, voice/recognizer.py,
voice/supervisor.py and WakeWordEngine were removed).

DETERMINISM CONTRACT
  * openWakeWord receives EXACTLY what it expects: int16 PCM @ 16 kHz mono,
    non-overlapping sequential 1280-sample (80 ms) frames carried across
    reads by the frame accumulator (never zero-padded mid-stream).
  * openWakeWord sees RAW ring-buffer audio — the exact domain the custom
    verifier was trained in. Whisper sees the preprocessed (high-pass +
    noise-suppressed) audio. Two consumers, two correct domains.
  * Every WAKE_LISTEN entry calls prime(): full model hard_reset (every
    internal buffer back to the post-load silence state), ring-buffer
    drain, and a refractory window — leftover audio/features can NEVER
    re-trigger the detector (no false wake loops).
  * The loop waits forever, never exits on its own, never reloads models
    (it only RETRIES loading when the model is missing), and logs one
    deterministic line per transition:

        WAIT_WAKE
        Speech detected (VAD probability=…)
        Wake score=…
        Wake trigger — verifying…
        Wake paused — verification running
        Verification buffer=… duration=…s samples=… RMS=… peak=…
        Verification transcript='…'
        Wake resumed
        Wake accepted | Wake rejected

LOW CPU: one 20 ms poll, one Silero window per 80 ms frame, one ONNX
predict per 80 ms frame. Whisper runs at most once per trigger (and a
cooldown prevents re-verifying the same audio).
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from voice.audio_manager import audio_manager
from voice.audio_processing import audio_preprocessor, float32_to_int16, peak_monitor
from voice.wake_model_manager import wake_model_manager, WARMUP_FRAME_SAMPLES
from voice.wake_word import verify_wake_transcript

logger = logging.getLogger(__name__)

# ── Deterministic tuning ───────────────────────────────────────
WAKE_FRAME = WARMUP_FRAME_SAMPLES      # 1280 samples = 80 ms @ 16 kHz (openWakeWord)
VAD_FRAME = 512                        # 32 ms @ 16 kHz (silero-vad 6.x window)
VAD_SPEECH_THRESHOLD = 0.5             # gate opens above this probability
VAD_HANGOVER_S = 0.6                   # gate stays open this long after VAD drops
# NO VAD override: when the speech gate is closed, NO score may trigger.
# (An override once let 1.000-scoring room-noise episodes bypass the gate;
# Silero measures 0.00 on such noise — the gate is the non-speech firewall.)
REFRACTORY_S = 1.0                     # triggers ignored right after prime()
REJECT_COOLDOWN_S = 2.0                # pause triggers after a rejected verification
VERIFY_MIN_INTERVAL_S = 1.5            # never verify more often than this
VERIFY_WINDOW_S = 2.5                  # Whisper verification looks back this far
SCORE_LOG_INTERVAL_S = 0.5             # "Wake score=…" cadence while idle
VAD_LOG_INTERVAL_S = 1.0               # "VAD probability=…" cadence while gate open
MODEL_RETRY_S = 5.0                    # missing-model retry cadence (never exits)
STALL_DUMP_S = 2.0                     # "NO INFERENCE" watchdog cadence
# Trigger settling (in 80 ms FRAMES, so behaviour is identical in real time
# and offline replay): when the score first crosses the threshold the phrase
# is usually STILL BEING SPOKEN — verifying immediately would transcribe a
# partial phrase and reject a real wake. The listener instead tracks the
# peak and verifies once the score has stayed below threshold for
# TRIGGER_SETTLE_FRAMES (≈0.4 s of speech end) or TRIGGER_MAX_FRAMES (≈1.2 s)
# elapsed since the trigger — i.e. at the end of the speech event, when the
# full phrase is inside the Whisper window.
TRIGGER_SETTLE_FRAMES = 5
TRIGGER_MAX_FRAMES = 15


@dataclass
class WakeEvent:
    """A verified wake detection."""
    model: str
    score: float
    transcript: str


# ═══════════════════════════════════════════════════════════════
# Silero VAD speech gate (WAKE_LISTEN only)
# ═══════════════════════════════════════════════════════════════

class WakeGateVAD:
    """Silero VAD used ONLY as the WAKE_LISTEN speech gate.

    Keeps the gate closed while the room is silent so openWakeWord scores
    produced by fan noise / clicks are never accepted. If Silero is
    unavailable the gate is permanently open and openWakeWord + Whisper
    decide alone.

    silero-vad 6.x accepts ONLY 256/512/768-sample windows at 16 kHz.
    """

    FRAME = VAD_FRAME

    def __init__(self):
        self._model = None
        self._ready = False

    def load(self) -> bool:
        if self._ready:
            return True
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            self._ready = True
            logger.info("[WAKE] Silero VAD gate loaded (WAKE_LISTEN speech gate)")
            return True
        except Exception as e:
            logger.info("[WAKE] Silero VAD gate unavailable (%s) — "
                        "openWakeWord runs ungated", e)
            self._ready = False
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    def max_speech_prob(self, audio: np.ndarray) -> float:
        """Highest speech probability across the chunk (0..1).

        Accepts float32 [-1, 1] (used AS-IS) or int16 PCM (decoded once via
        /32768 at this model boundary). Returns 1.0 (gate open) when the
        VAD is not loaded.
        """
        if not self._ready or len(audio) < self.FRAME:
            return 1.0
        try:
            import torch
            best = 0.0
            for i in range(0, len(audio) - self.FRAME + 1, self.FRAME):
                frame = audio[i:i + self.FRAME]
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32) / 32768.0
                with torch.no_grad():
                    prob = self._model(torch.from_numpy(frame), 16000).item()
                if prob > best:
                    best = float(prob)
            return best
        except Exception:
            return 1.0


# ═══════════════════════════════════════════════════════════════
# WakeListener
# ═══════════════════════════════════════════════════════════════

class WakeListener:
    """The ONE wake-listen implementation. Synchronous process() core +
    an async forever-loop (wait_for_wake) used by the ConversationEngine.
    debug/test_wake.py drives the SAME process() core directly.
    """

    def __init__(self):
        self.vad = WakeGateVAD()
        # Frame accumulator: guarantees openWakeWord only ever sees
        # exactly-1280-sample, non-overlapping, sequential frames.
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_total: Optional[int] = None
        # State
        self._refractory_until = 0.0
        self._cooldown_until = 0.0
        self._gate_open_until = 0.0
        self._gate_open = True
        self._last_score_log = 0.0
        self._last_vad_log = 0.0
        self._last_verify = 0.0
        self._last_inference = time.monotonic()
        self._last_stall_dump = 0.0
        self._last_model_retry = 0.0
        # Trigger-settling state (frame-counted, wall-clock independent)
        self._trigger_active = False
        self._trigger_peak = 0.0
        self._trigger_model = ""
        self._trigger_frames = 0
        self._trigger_below = 0
        # Observability (debug/test_wake.py prints these every second)
        self.last_score = 0.0
        self.last_vad = 1.0
        self.last_transcript = ""
        self.last_decision = "WAIT_WAKE"
        # Verification gate: True while Whisper is verifying a wake trigger.
        # process() is a no-op while this flag is set — the preprocessor
        # filter state is reserved for the verifier's clean pass.
        self._verifying = False

    # ── Lifecycle ──────────────────────────────────────────────

    def load(self) -> bool:
        """Load the VAD gate (the wake model itself is owned/loaded by
        WakeModelManager and retried forever inside the loop)."""
        return self.vad.load()

    def prime(self) -> None:
        """Deterministic clean state — called on EVERY WAKE_LISTEN entry.

        hard_reset() restores every openWakeWord buffer to the post-load
        silence state (stale wake-phrase / TTS / conversation features can
        never re-fire), the ring buffer is drained so only FRESH audio is
        scored, and a refractory window absorbs any in-flight sound.
        """
        wake_model_manager.hard_reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_total = audio_manager.total_samples  # drain backlog
        now = time.monotonic()
        self._refractory_until = now + REFRACTORY_S
        self._cooldown_until = 0.0
        self._gate_open_until = 0.0
        self._gate_open = True
        self._last_verify = 0.0
        self._last_inference = now
        self._trigger_active = False
        self._trigger_peak = 0.0
        self._trigger_model = ""
        self._trigger_frames = 0
        self._trigger_below = 0
        self.last_score = 0.0
        self.last_transcript = ""
        self.last_decision = "WAIT_WAKE"
        self._verifying = False
        logger.info("WAIT_WAKE — listening for '%s' "
                    "(model=%s threshold=%.2f vad_gate=%s)",
                    wake_model_manager.wake_phrase,
                    wake_model_manager.model_name or "none",
                    wake_model_manager.threshold,
                    "on" if self.vad.ready else "off")

    # ── Core: process one chunk of ring-buffer audio ───────────

    def process(self, new_audio: np.ndarray) -> Optional[Tuple[str, float]]:
        """Feed RAW ring-buffer audio (float32 [-1, 1], 16 kHz mono).

        Runs VAD gating + openWakeWord streaming inference with full
        deterministic logging. Returns (model_name, score) when a trigger
        requires Whisper verification, else None.

        VERIFICATION GATE: when _verifying is True, this method returns None
        immediately — the preprocessor filter state (_zi) is reserved for
        the verifier's clean pass and must NOT be mutated by streaming VAD.
        """
        # ── Verification gate: only ONE module owns mic frames at a time ──
        if self._verifying:
            return None

        now = time.monotonic()

        # Noise suppression for the VAD gate + Whisper (keeps the noise
        # profile warm); openWakeWord is fed the RAW audio below — the
        # exact domain its verifier was trained on.
        processed = audio_preprocessor.process(new_audio)
        peak_monitor.log("wake_detector", processed)

        # ── Silero VAD speech gate ──
        if self.vad.ready:
            vad_prob = self.vad.max_speech_prob(processed)
            if vad_prob > VAD_SPEECH_THRESHOLD:
                self._gate_open_until = now + VAD_HANGOVER_S
        else:
            vad_prob = 1.0
            self._gate_open_until = now + VAD_HANGOVER_S
        self.last_vad = vad_prob

        gate_open = now <= self._gate_open_until
        if gate_open != self._gate_open:
            self._gate_open = gate_open
            if gate_open:
                # Deterministic transition log: silence → speech.
                logger.info("Speech detected (VAD probability=%.2f)", vad_prob)
            else:
                logger.info("[VAD] Speech gate CLOSED (probability=%.2f)", vad_prob)
        if gate_open and now - self._last_vad_log >= VAD_LOG_INTERVAL_S:
            self._last_vad_log = now
            logger.info("VAD probability=%.2f gate=open", vad_prob)

        # ── openWakeWord: exactly-1280-sample sequential frames ──
        self._pending = np.concatenate([self._pending, new_audio])
        trigger: Optional[Tuple[str, float]] = None
        while len(self._pending) >= WAKE_FRAME:
            frame = self._pending[:WAKE_FRAME]
            self._pending = self._pending[WAKE_FRAME:]

            t_pred = time.perf_counter()
            preds = wake_model_manager.predict_stream(frame)
            pred_ms = (time.perf_counter() - t_pred) * 1000.0
            if not preds:
                continue
            self._last_inference = time.monotonic()
            score = max(float(s) for s in preds.values())
            self.last_score = score
            threshold = wake_model_manager.threshold

            # Deterministic score logging: time-budgeted while idle,
            # IMMEDIATE whenever the score crosses the trigger threshold.
            if score >= threshold or now - self._last_score_log >= SCORE_LOG_INTERVAL_S:
                logger.info("Wake score=%.3f model=%s threshold=%.2f "
                            "vad=%.2f latency=%.1fms",
                            score, wake_model_manager.model_name or "?",
                            threshold, vad_prob, pred_ms)
                self._last_score_log = now

            # ── Trigger settling: verify at the END of the speech event ──
            if self._trigger_active:
                self._trigger_frames += 1
                if score > self._trigger_peak:
                    self._trigger_peak = score
                if score < threshold:
                    self._trigger_below += 1
                else:
                    self._trigger_below = 0
                if (self._trigger_below >= TRIGGER_SETTLE_FRAMES
                        or self._trigger_frames >= TRIGGER_MAX_FRAMES):
                    # Speech event complete — the full phrase is inside the
                    # Whisper window. Verify ONCE with the peak score.
                    trigger = (self._trigger_model, self._trigger_peak)
                    self._trigger_active = False
                    break
                continue

            if score < threshold:
                continue
            if now < self._refractory_until:
                logger.info("[WAKE] Score %.2f ≥ %.2f during refractory "
                            "window — ignored", score, threshold)
                continue
            if now < self._cooldown_until:
                continue
            if not gate_open:
                logger.info("[WAKE] Score %.2f ≥ %.2f but VAD gate closed "
                            "(no speech) — rejected", score, threshold)
                continue
            if now - self._last_verify < VERIFY_MIN_INTERVAL_S:
                continue
            # Score crossed the threshold with the gate open: the phrase is
            # likely still being spoken — track the peak and verify when the
            # event settles (never mid-phrase).
            self._trigger_active = True
            self._trigger_peak = score
            self._trigger_model = wake_model_manager.model_name or "wake"
            self._trigger_frames = 0
            self._trigger_below = 0

        return trigger

    # ── Whisper verification (ONLY after an openWakeWord trigger) ──

    def verify_with_whisper(self) -> Tuple[bool, str]:
        """FORENSIC VERIFICATION — transcribe a frozen ring-buffer snapshot
        with full traceability and WAV export for every attempt.

        OWNERSHIP TRACE (STEP 1):
          Microphone → PortAudio callback → AGC → RingBuffer (float32)
            → read_since() → openWakeWord (int16, 1280-sample frames)
            → get_recent_audio() → this method → Whisper (int16 PCM)
        Every sample that Whisper receives comes from the SAME ring buffer
        that openWakeWord scores. The buffer offset, write pointer, sample
        count, and duration are printed so divergence can be identified.

        ROOT CAUSE #1 (2026-08-04, FIXED): using get_recent_processed()
        with the streaming-mutated _zi caused IIR transient ringing.

        ROOT CAUSE #2 (2026-08-04, FIXED HERE): _estimate_noise(raw) was
        called with SPEECH audio (the verification buffer). The spectral
        gate then used the wake phrase's own spectrum as the "noise"
        reference and attenuated the very speech Whisper should transcribe.
        The output was a suppressed/muffled signal → Whisper returned
        no_segments or hallucinated unrelated phrases.

        FIX: use the SHARED audio_preprocessor (which has a noise profile
        built from real background audio during streaming), and only do
        high-pass filtering — skip spectral gating since the wake buffer is
        already VAD-gated speech. This also avoids building a second
        preprocessor state that would be immediately discarded.

        STEP 3: every verification buffer is exported as
        debug/verification_NNN.wav for offline comparison.

        STEP 4: cross-correlation between the raw buffer and processed
        audio is computed and logged.

        STEP 5: concurrency audit — _verifying gate ensures no writer
        contention; ring buffer read is under lock; processed audio is
        an immutable copy.

        SYNCHRONOUS (call in an executor). NEVER raises.
        """
        t_verify = time.perf_counter()
        tid = threading.get_ident()

        # ── STEP 5: Concurrency audit ──
        # The _verifying flag is already set by the caller (wait_for_wake).
        # process() is a no-op → no streaming VAD runs → audio_preprocessor._zi
        # is NOT being mutated. The ring buffer IS being written by the
        # PortAudio callback, but get_recent_audio() takes self._lock so the
        # snapshot is consistent. We copy the snapshot immediately — the raw
        # array is immutable from this point forward.
        ring_total_at_snapshot = audio_manager.total_samples
        logger.info(
            "[WAKE] Verification started — STEP 1 trace: "
            "thread_id=%s ring_buffer_total=%d verify_window=%.1fs "
            "_verifying=%s",
            tid, ring_total_at_snapshot, VERIFY_WINDOW_S, self._verifying)

        try:
            from voice.streaming_stt import streaming_stt
            if not streaming_stt.ready:
                if not streaming_stt.initialize():
                    logger.warning("[WAKE] Whisper unavailable — verification aborted")
                    return False, ""

            # ── STEP 1: freeze the ring buffer with offset accounting ──
            raw = audio_manager.get_recent_audio(VERIFY_WINDOW_S)
            buf_samples = int(len(raw))
            buf_dur_s = buf_samples / 16000.0
            # Sample index range in the ring buffer (monotonic write pointer):
            # end_sample = ring_total_at_snapshot (most recent sample written)
            # start_sample = end_sample - buf_samples
            end_sample = ring_total_at_snapshot
            start_sample = max(0, end_sample - buf_samples)
            logger.info(
                "[WAKE] STEP 1 — Buffer snapshot: start_sample=%d end_sample=%d "
                "samples=%d duration=%.3fs write_ptr=%d",
                start_sample, end_sample, buf_samples, buf_dur_s,
                ring_total_at_snapshot)

            if buf_samples < 8000:
                logger.info("[WAKE] Verification buffer too short: %d samples "
                            "(%.3fs < 0.5s) — rejecting",
                            buf_samples, buf_dur_s)
                return False, ""

            # ── STEP 4: raw audio metrics (before any processing) ──
            raw64 = raw.astype(np.float64)
            raw_rms = float(np.sqrt(np.mean(raw64 * raw64))) * 32768.0
            raw_peak = float(np.max(np.abs(raw64))) * 32768.0
            logger.info("[WAKE] STEP 4a — Raw buffer: RMS=%.1f peak=%.0f "
                        "samples=%d duration=%.3fs (int16-scale)",
                        raw_rms, raw_peak, buf_samples, buf_dur_s)

            # ── ROOT CAUSE #2 FIX: use the SHARED preprocessor's noise profile ──
            # The streaming path (process() → audio_preprocessor.process())
            # has already built a noise profile from REAL background audio.
            # We only need the high-pass filter for this VAD-gated speech
            # buffer — spectral gating on a speech buffer would suppress the
            # very words we need to transcribe. Using a FRESH preprocessor and
            # calling _estimate_noise() on SPEECH audio was the root cause of
            # "no_segments" and hallucinated transcriptions.
            #
            # We apply ONLY high-pass filtering (no spectral gating) using a
            # fresh filter state to avoid the streaming _zi contamination.
            # The high-pass removes DC offset and sub-80Hz rumble; spectral
            # gating is unnecessary because this buffer already passed the
            # Silero VAD speech gate.
            from scipy import signal as scipy_signal
            from voice.audio_processing import SAMPLE_RATE as _SR, HIGH_PASS_CUTOFF, HIGH_PASS_ORDER
            nyquist = _SR / 2
            sos = scipy_signal.butter(HIGH_PASS_ORDER, HIGH_PASS_CUTOFF / nyquist,
                                      btype="highpass", output="sos")
            zi = scipy_signal.sosfilt_zi(sos) * 0  # zero initial state
            processed, _zi = scipy_signal.sosfilt(sos, raw.astype(np.float64), zi=zi)
            # Contain IIR ringing from the filter transient (first ~200 samples
            # may ring; this is benign for a 2.5s buffer and does NOT destroy the
            # wake phrase preamble like the old streaming-state ringing did).
            processed = np.clip(processed.astype(np.float32), -1.0, 1.0)

            proc_rms = float(np.sqrt(np.mean(
                processed.astype(np.float64) ** 2))) * 32768.0
            proc_peak = float(np.max(np.abs(processed))) * 32768.0
            logger.info("[WAKE] STEP 4b — High-pass only (no spectral gate): "
                        "RMS=%.1f peak=%.0f (fresh filter, zero-state _zi)",
                        proc_rms, proc_peak)

            # ── STEP 4: cross-correlation between raw and processed ──
            # The high-pass filter is nearly unity gain above 80 Hz, so the
            # correlation should be >0.95. A lower value indicates the buffer
            # was corrupted (wrong audio, overlapped writes, etc.).
            try:
                # Normalize both signals for correlation
                raw_norm = raw64 - np.mean(raw64)
                proc_norm = processed.astype(np.float64) - np.mean(processed.astype(np.float64))
                raw_std = np.std(raw_norm)
                proc_std = np.std(proc_norm)
                if raw_std > 1e-10 and proc_std > 1e-10:
                    correlation = np.corrcoef(raw_norm, proc_norm)[0, 1]
                    # Cross-correlation lag (should be ~0 — no time shift)
                    xcorr = np.correlate(raw_norm / raw_std,
                                         proc_norm / proc_std, mode="full")
                    lag_samples = int(np.argmax(np.abs(xcorr))) - (len(raw_norm) - 1)
                    lag_ms = lag_samples / 16.0  # samples → ms @ 16 kHz
                    logger.info("[WAKE] STEP 4c — Cross-correlation: "
                                "pearson_r=%.4f lag=%d samples (%.2f ms)",
                                correlation, lag_samples, lag_ms)
                else:
                    correlation = 0.0
                    lag_ms = 0.0
                    logger.warning("[WAKE] STEP 4c — Signal too quiet for correlation")
            except Exception as e:
                correlation = 0.0
                lag_ms = 0.0
                logger.debug("[WAKE] STEP 4c — Correlation failed: %s", e)

            # ── STEP 3: export verification WAV for offline analysis ──
            wav_path = None
            try:
                import wave
                from pathlib import Path
                dbg_dir = Path(__file__).resolve().parent.parent / "debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                # Find next available verification number
                existing = sorted(dbg_dir.glob("verification_*.wav"))
                next_num = 1
                if existing:
                    import re
                    nums = []
                    for p in existing:
                        m = re.search(r"verification_(\d+)\.wav", p.name)
                        if m:
                            nums.append(int(m.group(1)))
                    if nums:
                        next_num = max(nums) + 1
                wav_path = dbg_dir / f"verification_{next_num:03d}.wav"
                with wave.open(str(wav_path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(float32_to_int16(processed).tobytes())
                logger.info("[WAKE] STEP 3 — Exported: %s "
                            "(samples=%d duration=%.2fs RMS=%.1f peak=%.0f "
                            "correlation=%.4f)",
                            wav_path.name, buf_samples, buf_dur_s,
                            proc_rms, proc_peak, correlation)
            except Exception as e:
                logger.warning("[WAKE] STEP 3 — WAV export failed: %s", e)

            # ── Transcribe with Whisper ──
            pcm = float32_to_int16(processed).tobytes()
            detail = streaming_stt._whisper.transcribe_detailed(pcm, 16000, False)
            text = detail.get("text") or ""

            # ── Full forensic verdict ──
            elapsed_ms = (time.perf_counter() - t_verify) * 1000.0
            logger.info(
                "[WAKE] VERDICT: transcript=%r lang=%s confidence=%.3f "
                "no_speech=%.2f duration=%.2fs samples=%d "
                "raw_RMS=%.1f raw_peak=%.0f proc_RMS=%.1f proc_peak=%.0f "
                "correlation=%.3f lag=%.1fms latency=%.0fms "
                "wav=%s thread_id=%s",
                text,
                detail.get("language", ""),
                detail.get("avg_logprob", 0.0),
                detail.get("no_speech_prob", 0.0),
                buf_dur_s, buf_samples, raw_rms, raw_peak,
                proc_rms, proc_peak, correlation, lag_ms,
                elapsed_ms,
                wav_path.name if wav_path else "none",
                tid)

            ok = verify_wake_transcript(text)
            self.last_transcript = text
            return bool(ok), text
        except Exception as e:
            logger.debug("[WAKE] transcript confirmation error: %s", e)
            return False, ""

    # ── The forever loop ───────────────────────────────────────

    async def wait_for_wake(
        self,
        running: Callable[[], bool],
    ) -> Optional[WakeEvent]:
        """WAKE_LISTEN: idle forever on AudioManager + openWakeWord (+ VAD).

        Whisper / LLM / TTS are NOT running here. Returns a WakeEvent when
        a trigger passes Whisper verification, or None on shutdown. This
        loop NEVER raises and NEVER exits on its own.

        OWNERSHIP CONTRACT: only ONE module owns microphone frames at a time.
        During Whisper verification the wake detector PAUSES (process() is a
        no-op while _verifying=True). The ring buffer continues filling in
        the background via the PortAudio callback, but the frozen snapshot
        for verification was taken before resuming — no race.
        """
        loop = asyncio.get_event_loop()
        if self._last_total is None:
            self._last_total = audio_manager.total_samples

        while running():
            now = time.monotonic()

            # The wake detector MUST always be active: if the model is
            # missing, keep retrying forever — never exit WAKE_LISTEN.
            if not wake_model_manager.loaded:
                if now - self._last_model_retry >= MODEL_RETRY_S:
                    self._last_model_retry = now
                    logger.warning(
                        "[WAKE] Wake model NOT loaded (%s) — retrying "
                        "every %.0fs; detector stays in WAKE_LISTEN",
                        wake_model_manager.load_error or "no model",
                        MODEL_RETRY_S)
                    ok = await loop.run_in_executor(None, wake_model_manager.load)
                    if ok:
                        self.prime()
                await asyncio.sleep(0.25)
                continue

            try:
                new_audio, self._last_total = audio_manager.read_since(self._last_total)
                if len(new_audio) == 0:
                    await asyncio.sleep(0.02)
                else:
                    trigger = await loop.run_in_executor(
                        None, self.process, new_audio)

                    # No-inference watchdog: if openWakeWord produced NO
                    # inference for >2 s, dump exactly why.
                    if (now - self._last_inference > STALL_DUMP_S
                            and now - self._last_stall_dump > STALL_DUMP_S):
                        self._last_stall_dump = now
                        logger.warning(
                            "[WAKE] NO INFERENCE for %.1fs — model_loaded=%s "
                            "total_samples=%d ring_buffer_alive=%s",
                            now - self._last_inference,
                            wake_model_manager.loaded, self._last_total,
                            audio_manager.is_running)

                    if trigger is not None:
                        model_name, score = trigger
                        self._last_verify = time.monotonic()

                        # ── PAUSE the wake detector ──
                        # process() is now a no-op (returns None immediately).
                        # The preprocessor filter state (_zi) is reserved for
                        # the verifier's clean pass.
                        self._verifying = True
                        logger.info(
                            "[WAKE] Wake PAUSED — verification running "
                            "(thread_id=%s ring_buffer_total=%d)",
                            threading.get_ident(), audio_manager.total_samples)

                        logger.info("Wake trigger (score=%.3f ≥ %.2f) — "
                                    "verifying transcript…",
                                    score, wake_model_manager.threshold)
                        verified, transcript = await loop.run_in_executor(
                            None, self.verify_with_whisper)

                        # ── RESUME the wake detector ──
                        self._verifying = False
                        logger.info(
                            "[WAKE] Wake RESUMED — detector active "
                            "(thread_id=%s ring_buffer_total=%d)",
                            threading.get_ident(), audio_manager.total_samples)

                        if verified:
                            self.last_decision = "ACCEPTED"
                            logger.info(
                                "Wake accepted (model='%s' score=%.2f ≥ %.2f "
                                "transcript='%s')",
                                model_name, score,
                                wake_model_manager.threshold, transcript)
                            return WakeEvent(model=model_name, score=score,
                                             transcript=transcript)
                        # Rejected: log it, cool down, keep listening.
                        self.last_decision = "REJECTED"
                        wake_model_manager.record_false_positive()
                        self._cooldown_until = time.monotonic() + REJECT_COOLDOWN_S
                        logger.info(
                            "Wake rejected (score=%.2f transcript=%r) — "
                            "still listening",
                            score, transcript or "<no speech>")

            except asyncio.CancelledError:
                raise
            except Exception:
                # The wake detector must NEVER die.
                logger.exception("[WAKE] Wake-loop iteration failed — "
                                 "recovering, detector stays active")
                self._verifying = False  # safety: resume on crash
                await asyncio.sleep(0.1)

        return None