"""
WakeListener — THE single wake-detection pipeline for Leo.

    Microphone
      ↓
    AudioManager callback (AGC + resample + HIGH-PASS — ONCE)
      ↓
    Ring Buffer (PREPROCESSED float32 @ 16 kHz mono)
      ↓
    Fork
    ├── Unified VAD          (speech gate, shared with CommandListener)
    ├── openWakeWord         (streaming predict, 1280-sample int16 frames)
    └── Whisper verification (ONLY after openWakeWord triggers)

UNIFIED PREPROCESSING:
  The high-pass filter is in the AudioManager callback. The ring buffer
  stores PREPROCESSED audio — every consumer (openWakeWord, Whisper, VAD)
  reads bit-identical float32 samples.

DETERMINISM CONTRACT:
  * openWakeWord receives exactly-1280-sample, non-overlapping, sequential frames.
  * openWakeWord AND Whisper see the SAME preprocessed audio.
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
from voice.audio_processing import float32_to_int16, peak_monitor
from voice.vad import unified_vad, VAD_FRAME_SAMPLES, SPEECH_THRESHOLD
from voice.wake_model_manager import wake_model_manager, WARMUP_FRAME_SAMPLES
from voice.wake_word import verify_wake_transcript

logger = logging.getLogger(__name__)

# ── Tuning ─────────────────────────────────────────────────────
WAKE_FRAME = WARMUP_FRAME_SAMPLES      # 1280 samples = 80 ms @ 16 kHz (openWakeWord)
VAD_HANGOVER_S = 0.6                   # gate stays open this long after VAD drops
REFRACTORY_S = 1.0                     # triggers ignored right after prime()
REJECT_COOLDOWN_S = 2.0                # pause triggers after a rejected verification
VERIFY_MIN_INTERVAL_S = 1.5            # never verify more often than this
VERIFY_WINDOW_S = 2.5                  # Whisper verification looks back this far
SCORE_LOG_INTERVAL_S = 0.5             # "Wake score=…" cadence while idle
VAD_LOG_INTERVAL_S = 1.0               # "VAD probability=…" cadence while gate open
MODEL_RETRY_S = 5.0                    # missing-model retry cadence (never exits)
STALL_DUMP_S = 2.0                     # "NO INFERENCE" watchdog cadence
VERIFY_TIMEOUT_S = 30.0                # Whisper verification hard timeout
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


class WakeListener:
    """The ONE wake-listen implementation. Uses the unified VAD (voice/vad.py)
    instead of its own Silero instance."""

    def __init__(self):
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
        # Trigger-settling state
        self._trigger_active = False
        self._trigger_peak = 0.0
        self._trigger_model = ""
        self._trigger_frames = 0
        self._trigger_below = 0
        self._trigger_gate_open_at_onset = True
        # Observability
        self.last_score = 0.0
        self.last_vad = 1.0
        self.last_transcript = ""
        self.last_decision = "WAIT_WAKE"
        # Verification gate
        self._verifying = False
        # SHA256 tracking
        self._sha256_wake_frames: list = []
        self._last_sha256_wake = ""

    # ── Lifecycle ──────────────────────────────────────────────

    def load(self) -> bool:
        """Load the unified VAD (shared with CommandListener)."""
        return unified_vad.load()

    @property
    def vad(self):
        """Backward-compat: expose unified_vad as self.vad for callers
        that check vad.ready."""
        return unified_vad

    def prime(self) -> None:
        """Deterministic clean state — called on EVERY WAKE_LISTEN entry."""
        wake_model_manager.hard_reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_total = audio_manager.total_samples
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
                    "on" if unified_vad.ready else "off")

    # ── Core: process one chunk of ring-buffer audio ───────────

    def process(self, new_audio: np.ndarray) -> Optional[Tuple[str, float]]:
        """Feed PREPROCESSED ring-buffer audio (float32, 16 kHz mono).

        Uses the UNIFIED VAD (voice/vad.py) for speech gating — no duplicate
        Silero instance. Returns (model_name, score) when a trigger requires
        Whisper verification, else None.
        """
        if self._verifying:
            return None

        now = time.monotonic()

        # SHA256 tracking
        try:
            h = hashlib.sha256(new_audio.tobytes())
            self._sha256_wake_frames.append(h.hexdigest()[:16])
            if len(self._sha256_wake_frames) > 100:
                self._sha256_wake_frames = self._sha256_wake_frames[-50:]
        except Exception:
            pass

        peak_monitor.log("wake_detector", new_audio)

        # ── Unified VAD speech gate ──
        if unified_vad.ready:
            vad_prob = unified_vad.max_speech_prob(new_audio, step=VAD_FRAME_SAMPLES // 2)
            if vad_prob > SPEECH_THRESHOLD:
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

            # ── Logging-volume fix: do NOT log every detector frame ──
            # Only log when the score crosses the threshold (a real wake
            # candidate) or when verbose tracing is explicitly enabled.
            if score >= threshold:
                logger.info("Wake score=%.3f model=%s threshold=%.2f "
                            "vad=%.2f latency=%.1fms sha256=%s",
                            score, wake_model_manager.model_name or "?",
                            threshold, vad_prob, pred_ms,
                            (self._sha256_wake_frames[-1][:8]
                             if self._sha256_wake_frames else "none"))
                self._last_score_log = now

            # Trigger settling
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
                    trigger = (self._trigger_model, self._trigger_peak)
                    self._trigger_active = False
                    self._last_sha256_wake = ":".join(
                        self._sha256_wake_frames[-20:]) if self._sha256_wake_frames else ""
                    break
                continue

            if score < threshold:
                continue
            if now < self._refractory_until:
                logger.info("[WAKE] Score %.2f ≥ %.2f during refractory window — ignored",
                            score, threshold)
                continue
            if now < self._cooldown_until:
                continue
            if now - self._last_verify < VERIFY_MIN_INTERVAL_S:
                continue

            self._trigger_active = True
            self._trigger_gate_open_at_onset = gate_open
            self._trigger_peak = score
            self._trigger_model = wake_model_manager.model_name or "wake"
            self._trigger_frames = 0
            self._trigger_below = 0
            self._last_sha256_wake = ":".join(
                self._sha256_wake_frames[-20:]) if self._sha256_wake_frames else ""

        return trigger

    # ── Whisper verification ──────────────────────────────────

    def verify_with_whisper(self, wake_score: float = 0.0) -> Tuple[bool, str, float, str, str]:
        """UNIFIED PIPELINE VERIFICATION — transcribe a frozen ring-buffer
        snapshot. Whisper consumes the EXACT same samples that openWakeWord scored.

        Returns (verified_bool, transcript, correlation, sha256_verify, sha256_wake).
        """
        t_verify = time.perf_counter()
        tid = threading.get_ident()
        ring_total_at_snapshot = audio_manager.total_samples
        wake_sha = self._last_sha256_wake

        logger.info("[WAKE] Verification started — UNIFIED PIPELINE: "
                    "thread_id=%s ring_buffer_total=%d verify_window=%.1fs",
                    tid, ring_total_at_snapshot, VERIFY_WINDOW_S)

        try:
            from voice.command_listener import command_listener
            if not command_listener.ready:
                if not command_listener.initialize():
                    logger.warning("[WAKE] Whisper unavailable — verification aborted")
                    return False, "", 0.0, "", wake_sha

            raw = audio_manager.get_recent_audio(VERIFY_WINDOW_S)
            buf_samples = int(len(raw))
            buf_dur_s = buf_samples / 16000.0

            if buf_samples < 8000:
                logger.info("[WAKE] Verification buffer too short: %d samples (%.3fs)",
                            buf_samples, buf_dur_s)
                return False, "", 0.0, "", wake_sha

            # Convert to int16 for Whisper
            pcm = float32_to_int16(raw).tobytes()
            verify_sha = hashlib.sha256(pcm).hexdigest()

            # Correlation sanity check
            correlation = 1.0
            try:
                if buf_samples >= 3200:
                    raw64 = raw.astype(np.float64)
                    half = buf_samples // 2
                    a = raw64[:half] - np.mean(raw64[:half])
                    b = raw64[half:half * 2] - np.mean(raw64[half:half * 2])
                    a_std, b_std = np.std(a), np.std(b)
                    if a_std > 1e-10 and b_std > 1e-10:
                        correlation = float(np.corrcoef(a, b)[0, 1])
            except Exception:
                pass

            # Export verification WAV
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
            except Exception:
                pass

            # Transcribe with Whisper
            text, confidence = command_listener._whisper.transcribe(pcm, 16000)

            elapsed_ms = (time.perf_counter() - t_verify) * 1000.0
            logger.info(
                "[WAKE] VERDICT: transcript=%r confidence=%.3f "
                "duration=%.2fs samples=%d correlation=%.3f latency=%.0fms "
                "sha256_verify=%s sha256_wake=%s wav=%s",
                text, confidence, buf_dur_s, buf_samples, correlation, elapsed_ms,
                verify_sha[:16], wake_sha[:32] if wake_sha else "none",
                wav_path.name if wav_path else "none")

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

        Returns a WakeEvent when a trigger passes Whisper verification,
        or None on shutdown. This loop NEVER raises and NEVER exits on its own.
        """
        loop = asyncio.get_event_loop()
        if self._last_total is None:
            self._last_total = audio_manager.total_samples

        while running():
            now = time.monotonic()

            if not wake_model_manager.loaded:
                if now - self._last_model_retry >= MODEL_RETRY_S:
                    self._last_model_retry = now
                    logger.warning("[WAKE] Wake model NOT loaded (%s) — retrying",
                                   wake_model_manager.load_error or "no model")
                    try:
                        ok = await loop.run_in_executor(None, wake_model_manager.load)
                        if ok:
                            self.prime()
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        # A model reload failure must never kill the wake
                        # loop. Report it as a structured worker error and
                        # keep retrying on the next cadence.
                        logger.exception(
                            "[WORKER-CRASH] worker=wake_listener exception=%s "
                            "message=%s — model reload failed; retrying",
                            type(e).__name__, str(e),
                        )
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
                        logger.warning("[WAKE] NO INFERENCE for %.1fs — model_loaded=%s",
                                       now - self._last_inference, wake_model_manager.loaded)

                    if trigger is not None:
                        model_name, score = trigger
                        self._last_verify = time.monotonic()

                        self._verifying = True
                        logger.info("[WAKE] Wake PAUSED — verification running")

                        logger.info("Wake trigger (score=%.3f ≥ %.2f) — verifying transcript…",
                                    score, wake_model_manager.threshold)
                        try:
                            result = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None, self.verify_with_whisper, score),
                                timeout=VERIFY_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            logger.error(
                                "[WAKE] TIMEOUT verification=%.0fms_budget_exceeded "
                                "— rejecting trigger and resuming detector",
                                VERIFY_TIMEOUT_S * 1000.0)
                            self._verifying = False
                            await asyncio.sleep(0.1)
                            continue
                        verified, transcript, correlation, verify_sha, wake_sha = result

                        self._verifying = False
                        logger.info("[WAKE] Wake RESUMED — detector active")

                        if verified:
                            self.last_decision = "ACCEPTED"
                            logger.info("Wake accepted (model='%s' score=%.2f transcript='%s')",
                                        model_name, score, transcript)
                            return WakeEvent(
                                model=model_name, score=score,
                                transcript=transcript,
                                correlation=correlation,
                                sha256_wake=wake_sha,
                                sha256_verify=verify_sha)

                        self.last_decision = "REJECTED"
                        wake_model_manager.record_false_positive()
                        self._cooldown_until = time.monotonic() + REJECT_COOLDOWN_S
                        logger.info("Wake rejected (score=%.2f transcript=%r) — still listening",
                                    score, transcript or "<no speech>")

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[WAKE] Wake-loop iteration failed — recovering")
                self._verifying = False
                await asyncio.sleep(0.1)

        return None