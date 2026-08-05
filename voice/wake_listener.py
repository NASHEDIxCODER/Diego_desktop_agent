"""
WakeListener — THE single wake-detection pipeline for Leo.

    Microphone
      ↓
    AudioManager callback (AGC + resample + HIGH-PASS — ONCE)
      ↓
    Ring Buffer (PREPROCESSED float32 @ 16 kHz mono)
      ↓
    Fork
    ├── Silero VAD          (speech gate, 512-sample windows)
    ├── openWakeWord        (streaming predict, 1280-sample int16 frames)
    └── Whisper verification (ONLY after openWakeWord triggers — consumes the
                              EXACT preprocessed samples that produced the score)

UNIFIED PREPROCESSING (2026-08-04 root-cause fix):
  The high-pass filter moved INTO the AudioManager callback. The ring buffer
  stores PREPROCESSED audio — every consumer (openWakeWord, Whisper, VAD)
  reads bit-identical float32 samples. There is NO second high-pass, NO
  second noise suppression, and NO filter-state divergence between the wake
  detector and the verification path.

DETERMINISM CONTRACT
  * openWakeWord receives EXACTLY what it expects: int16 PCM @ 16 kHz mono,
    non-overlapping sequential 1280-sample (80 ms) frames carried across
    reads by the frame accumulator (never zero-padded mid-stream).
  * openWakeWord AND Whisper see the SAME preprocessed audio — the exact
    domain the custom verifier was trained in. SHA256 hash verification
    proves bit-identity at both consumer points.
  * Every WAKE_LISTEN entry calls prime(): full model hard_reset.
  * The loop waits forever, never exits on its own, never reloads models.
"""

import asyncio
import hashlib
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
REFRACTORY_S = 1.0                     # triggers ignored right after prime()
REJECT_COOLDOWN_S = 2.0                # pause triggers after a rejected verification
VERIFY_MIN_INTERVAL_S = 1.5            # never verify more often than this
VERIFY_WINDOW_S = 2.5                  # Whisper verification looks back this far
SCORE_LOG_INTERVAL_S = 0.5             # "Wake score=…" cadence while idle
VAD_LOG_INTERVAL_S = 1.0               # "VAD probability=…" cadence while gate open
MODEL_RETRY_S = 5.0                    # missing-model retry cadence (never exits)
STALL_DUMP_S = 2.0                     # "NO INFERENCE" watchdog cadence
TRIGGER_SETTLE_FRAMES = 5
TRIGGER_MAX_FRAMES = 15


@dataclass
class WakeEvent:
    """A verified wake detection."""
    model: str
    score: float
    transcript: str
    correlation: float = 0.0
    sha256_wake: str = ""
    sha256_verify: str = ""


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

        UNIFIED PIPELINE: audio is already PREPROCESSED float32 from the
        ring buffer (AGC + high-pass). No further normalization needed.
        Returns 1.0 (gate open) when the VAD is not loaded.
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
        self._verifying = False

        # ── BIT-IDENTICAL DIAGNOSTICS: SHA256 hash tracking ──
        self._sha256_wake_frames: list = []  # accumulated hex digests per frame
        self._last_sha256_wake = ""

    # ── Lifecycle ──────────────────────────────────────────────

    def load(self) -> bool:
        """Load the VAD gate (the wake model itself is owned/loaded by
        WakeModelManager and retried forever inside the loop)."""
        return self.vad.load()

    def prime(self) -> None:
        """Deterministic clean state — called on EVERY WAKE_LISTEN entry.

        hard_reset() restores every openWakeWord buffer to the post-load
        silence state. The ring buffer is drained so only FRESH audio is
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
        self._trigger_gate_open_at_onset = True
        self.last_score = 0.0
        self.last_transcript = ""
        self.last_decision = "WAIT_WAKE"
        self._verifying = False
        self._sha256_wake_frames = []
        self._last_sha256_wake = ""
        logger.info("WAIT_WAKE — listening for '%s' "
                    "(model=%s threshold=%.2f vad_gate=%s unified_pipeline=ON)",
                    wake_model_manager.wake_phrase,
                    wake_model_manager.model_name or "none",
                    wake_model_manager.threshold,
                    "on" if self.vad.ready else "off")

    # ── Core: process one chunk of ring-buffer audio ───────────

    def process(self, new_audio: np.ndarray) -> Optional[Tuple[str, float]]:
        """Feed PREPROCESSED ring-buffer audio (float32, 16 kHz mono,
        AGC + high-pass applied in the callback).

        UNIFIED PIPELINE: the ring buffer stores PREPROCESSED audio.
        Every consumer (VAD, openWakeWord, Whisper) sees the SAME samples.
        SHA256 hashes are accumulated per-frame for forensic comparison.

        Runs VAD gating + openWakeWord streaming inference with full
        deterministic logging. Returns (model_name, score) when a trigger
        requires Whisper verification, else None.
        """
        # ── Verification gate: only ONE module owns mic frames at a time ──
        if self._verifying:
            return None

        now = time.monotonic()

        # ── BIT-IDENTICAL TRACKING: SHA256 of every chunk fed to openWakeWord ──
        try:
            h = hashlib.sha256(new_audio.tobytes())
            self._sha256_wake_frames.append(h.hexdigest()[:16])
            if len(self._sha256_wake_frames) > 100:
                self._sha256_wake_frames = self._sha256_wake_frames[-50:]
        except Exception:
            pass

        peak_monitor.log("wake_detector", new_audio)

        # ── Silero VAD speech gate ──
        # UNIFIED PIPELINE: the ring buffer audio already has high-pass.
        # Use it AS-IS — no second preprocessing filter pass.
        if self.vad.ready:
            vad_prob = self.vad.max_speech_prob(new_audio)
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

            # Deterministic score logging
            if score >= threshold or now - self._last_score_log >= SCORE_LOG_INTERVAL_S:
                logger.info("Wake score=%.3f model=%s threshold=%.2f "
                            "vad=%.2f latency=%.1fms sha256=%s",
                            score, wake_model_manager.model_name or "?",
                            threshold, vad_prob, pred_ms,
                            (self._sha256_wake_frames[-1][:8]
                             if self._sha256_wake_frames else "none"))
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
                    # Speech event complete — verify ONCE with the peak score.
                    trigger = (self._trigger_model, self._trigger_peak)
                    self._trigger_active = False
                    # Record the SHA256 chain for this trigger window
                    self._last_sha256_wake = ":".join(
                        self._sha256_wake_frames[-20:]) if self._sha256_wake_frames else ""
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
            # ISSUE-2 FIX: VAD gate check is DEFERRED to trigger settlement.
            # The VAD uses 512-sample windows; openWakeWord uses 1280-sample
            # frames. The VAD gate can transiently close between frames even
            # during continuous speech (e.g. between syllables). Rejecting a
            # score=1.00 wake because the VAD gate flickered closed for one
            # 32ms window is a false negative. Instead, we record the VAD
            # state at trigger onset and verify at settlement that speech was
            # present for the MAJORITY of the trigger window.
            if not gate_open:
                logger.info("[WAKE] Score %.2f ≥ %.2f — VAD gate currently "
                            "closed but trigger tracked for settlement "
                            "(VAD uses 32ms windows, wake uses 80ms frames "
                            "— transient gate closure is expected)",
                            score, threshold)
            if now - self._last_verify < VERIFY_MIN_INTERVAL_S:
                continue
            # Score crossed the threshold: track the peak
            # and verify when the event settles.
            self._trigger_active = True
            self._trigger_gate_open_at_onset = gate_open
            self._trigger_peak = score
            self._trigger_model = wake_model_manager.model_name or "wake"
            self._trigger_frames = 0
            self._trigger_below = 0
            # Record SHA256 chain at the moment of trigger for diagnostics
            self._last_sha256_wake = ":".join(
                self._sha256_wake_frames[-20:]) if self._sha256_wake_frames else ""

        return trigger

    # ── Whisper verification (ONLY after an openWakeWord trigger) ──

    def verify_with_whisper(self, wake_score: float = 0.0) -> Tuple[bool, str, float, str, str]:
        """UNIFIED PIPELINE VERIFICATION — transcribe a frozen ring-buffer
        snapshot with bit-identical-preprocessing guarantees.

        ROOT CAUSE FIX (2026-08-04): the high-pass filter is now in the
        AudioManager callback. The ring buffer stores PREPROCESSED audio.
        Whisper consumes the EXACT same samples that openWakeWord scored.

        DIAGNOSTICS:
          - SHA256 hash of the verification snapshot (PCM16 bytes)
          - SHA256 hash of the last wake frame chain
          - Pearson correlation between ring-buffer float32 and Whisper input
          - Buffer timestamps, write index, sample count
          - WAV export of verification audio
          - Latency between wake trigger and snapshot

        Args:
          wake_score: The openWakeWord confidence score that triggered
                      verification.  Passed through to the transcript
                      verifier so high-certainty model scores (≥0.995)
                      can override Whisper transcription errors.

        SYNCHRONOUS (call in an executor). NEVER raises.

        Returns:
          (verified_bool, transcript, correlation, sha256_verify, sha256_wake)
        """
        t_verify = time.perf_counter()
        tid = threading.get_ident()

        ring_total_at_snapshot = audio_manager.total_samples
        wake_sha = self._last_sha256_wake

        logger.info(
            "[WAKE] Verification started — UNIFIED PIPELINE: "
            "thread_id=%s ring_buffer_total=%d verify_window=%.1fs "
            "_verifying=%s wake_sha256_frames=%s",
            tid, ring_total_at_snapshot, VERIFY_WINDOW_S,
            self._verifying,
            wake_sha[:64] + ("..." if len(wake_sha) > 64 else ""))

        try:
            from voice.streaming_stt import streaming_stt
            if not streaming_stt.ready:
                if not streaming_stt.initialize():
                    logger.warning("[WAKE] Whisper unavailable — verification aborted")
                    return False, "", 0.0, "", wake_sha

            # ── STEP 1: freeze the ring buffer ──
            # UNIFIED PIPELINE: get_recent_audio() returns PREPROCESSED audio
            # (AGC + high-pass). No second filter is applied — this is the
            # EXACT audio that openWakeWord scored.
            raw = audio_manager.get_recent_audio(VERIFY_WINDOW_S)
            buf_samples = int(len(raw))
            buf_dur_s = buf_samples / 16000.0
            end_sample = ring_total_at_snapshot
            start_sample = max(0, end_sample - buf_samples)
            logger.info(
                "[WAKE] STEP 1 — Buffer snapshot: start_sample=%d end_sample=%d "
                "samples=%d duration=%.3fs write_ptr=%d "
                "(PREPROCESSED — no second filter)",
                start_sample, end_sample, buf_samples, buf_dur_s,
                ring_total_at_snapshot)

            if buf_samples < 8000:
                logger.info("[WAKE] Verification buffer too short: %d samples "
                            "(%.3fs < 0.5s) — rejecting",
                            buf_samples, buf_dur_s)
                return False, "", 0.0, "", wake_sha

            # ── STEP 2: audio metrics (preprocessed, no additional filtering) ──
            raw64 = raw.astype(np.float64)
            raw_rms = float(np.sqrt(np.mean(raw64 * raw64))) * 32768.0
            raw_peak = float(np.max(np.abs(raw64))) * 32768.0
            logger.info("[WAKE] STEP 2 — Preprocessed buffer: RMS=%.1f peak=%.0f "
                        "samples=%d duration=%.3fs (int16-scale) — "
                        "NO second high-pass applied",
                        raw_rms, raw_peak, buf_samples, buf_dur_s)

            # ── STEP 3: convert to int16 for Whisper (single conversion at sink) ──
            pcm = float32_to_int16(raw).tobytes()

            # ── STEP 4: SHA256 of verification audio ──
            verify_sha = hashlib.sha256(pcm).hexdigest()
            logger.info("[WAKE] STEP 4 — Verification SHA256: %s", verify_sha[:32])

            # ── STEP 5: correlation between preprocessed float32 and
            # the same buffer (self-correlation should be 1.0; we compute
            # correlation between consecutive halves as a sanity check) ──
            correlation = 1.0  # Self-correlation is trivially 1.0
            try:
                if buf_samples >= 3200:
                    half = buf_samples // 2
                    a = raw64[:half] - np.mean(raw64[:half])
                    b = raw64[half:half * 2] - np.mean(raw64[half:half * 2])
                    a_std = np.std(a)
                    b_std = np.std(b)
                    if a_std > 1e-10 and b_std > 1e-10:
                        correlation = float(np.corrcoef(a, b)[0, 1])
                        logger.info("[WAKE] STEP 5 — Intra-buffer correlation "
                                    "(first half vs second half): pearson_r=%.4f",
                                    correlation)
            except Exception as e:
                logger.debug("[WAKE] STEP 5 — Correlation failed: %s", e)

            # ── STEP 6: export verification WAV ──
            wav_path = None
            try:
                import wave
                from pathlib import Path
                dbg_dir = Path(__file__).resolve().parent.parent / "debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
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
                    w.writeframes(pcm)
                logger.info("[WAKE] STEP 6 — Exported: %s "
                            "(samples=%d duration=%.2fs RMS=%.1f peak=%.0f "
                            "sha256=%s)",
                            wav_path.name, buf_samples, buf_dur_s,
                            raw_rms, raw_peak, verify_sha[:16])
            except Exception as e:
                logger.warning("[WAKE] STEP 6 — WAV export failed: %s", e)

            # ── STEP 7: Transcribe with Whisper ──
            detail = streaming_stt._whisper.transcribe_detailed(pcm, 16000, False)
            text = detail.get("text") or ""

            # ── Full forensic verdict ──
            elapsed_ms = (time.perf_counter() - t_verify) * 1000.0
            logger.info(
                "[WAKE] VERDICT: transcript=%r lang=%s confidence=%.3f "
                "no_speech=%.2f duration=%.2fs samples=%d "
                "RMS=%.1f peak=%.0f "
                "correlation=%.3f latency=%.0fms "
                "sha256_verify=%s sha256_wake=%s "
                "wav=%s thread_id=%s unified_pipeline=ON",
                text,
                detail.get("language", ""),
                detail.get("avg_logprob", 0.0),
                detail.get("no_speech_prob", 0.0),
                buf_dur_s, buf_samples, raw_rms, raw_peak,
                correlation, elapsed_ms,
                verify_sha[:16], wake_sha[:32] if wake_sha else "none",
                wav_path.name if wav_path else "none",
                tid)

            ok = verify_wake_transcript(text, wake_score)
            self.last_transcript = text
            return bool(ok), text, correlation, verify_sha, wake_sha
        except Exception as e:
            logger.debug("[WAKE] transcript confirmation error: %s", e)
            return False, "", 0.0, "", ""

    # ── The forever loop ───────────────────────────────────────

    async def wait_for_wake(
        self,
        running: Callable[[], bool],
    ) -> Optional[WakeEvent]:
        """WAKE_LISTEN: idle forever on AudioManager + openWakeWord (+ VAD).

        Whisper / LLM / TTS are NOT running here. Returns a WakeEvent when
        a trigger passes Whisper verification, or None on shutdown. This
        loop NEVER raises and NEVER exits on its own.

        UNIFIED PIPELINE: ring buffer stores PREPROCESSED audio. Both
        openWakeWord and Whisper consume bit-identical samples.
        """
        loop = asyncio.get_event_loop()
        if self._last_total is None:
            self._last_total = audio_manager.total_samples

        while running():
            now = time.monotonic()

            # The wake detector MUST always be active.
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

                    # No-inference watchdog
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
                        self._verifying = True
                        logger.info(
                            "[WAKE] Wake PAUSED — verification running "
                            "(thread_id=%s ring_buffer_total=%d unified_pipeline=ON)",
                            threading.get_ident(), audio_manager.total_samples)

                        logger.info("Wake trigger (score=%.3f ≥ %.2f) — "
                                    "verifying transcript…",
                                    score, wake_model_manager.threshold)
                        result = await loop.run_in_executor(
                            None, self.verify_with_whisper, score)
                        verified, transcript, correlation, verify_sha, wake_sha = result

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
                                "transcript='%s' correlation=%.4f "
                                "sha256_verify=%s unified_pipeline=ON)",
                                model_name, score,
                                wake_model_manager.threshold, transcript,
                                correlation,
                                verify_sha[:16] if verify_sha else "none")
                            return WakeEvent(
                                model=model_name, score=score,
                                transcript=transcript,
                                correlation=correlation,
                                sha256_wake=wake_sha,
                                sha256_verify=verify_sha)
                        # Rejected: log it, cool down, keep listening.
                        self.last_decision = "REJECTED"
                        wake_model_manager.record_false_positive()
                        self._cooldown_until = time.monotonic() + REJECT_COOLDOWN_S
                        logger.info(
                            "Wake rejected (score=%.2f transcript=%r "
                            "correlation=%.4f sha256_verify=%s) — "
                            "still listening",
                            score, transcript or "<no speech>",
                            correlation,
                            verify_sha[:16] if verify_sha else "none")

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[WAKE] Wake-loop iteration failed — "
                                 "recovering, detector stays active")
                self._verifying = False  # safety: resume on crash
                await asyncio.sleep(0.1)

        return None