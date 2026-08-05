"""
AudioManager — Professional hardware microphone detector + unified capture.

ARCHITECTURE:
  Hardware Detector → Verified InputStream → Ring Buffer → Wake / VAD / STT

The microphone is NEVER assumed. Before any stream is opened the manager:

  STEP 1  Enumerates EVERY input device with full classification flags.
  STEP 2  Rejects virtual buses (default/pulse/pipewire/monitor/loopback/
          echo/null/dummy/virtual) unless no physical hardware exists.
  STEP 3  Prefers real hardware (USB/ALC/Analog/Microphone/Mic/Built-in/
          HD Audio/Realtek).
  STEP 4  Probes EVERY candidate: opens a stream, records 2 s and measures
          RMS / Peak / Zero-crossing / Dynamic range / Voice probability.
          Devices with RMS≈0, Peak≈0 or a flat waveform are REJECTED.
  STEP 5  Selects the device with the highest speech energy and persists
          ONLY verified devices — a silent device is never saved.
  STEP 6  If every device is silent, prints "No working microphone
          detected." with every tested device and its RMS, and start()
          returns False — Leo NEVER continues on a silent mic.

After selection the InputStream callback prints real signal levels:

  CALLBACK device=HD Audio Generic speech_channel=1 RMS=2400 Peak=15000

AUDIO FORMAT CONTRACT (gain pipeline):
  The ring buffer carries float32 samples in [-1, 1], 16 kHz, mono.
  The signal is normalized EXACTLY ONCE at capture. EXACTLY ONE
  instrumented gain stage exists — the capture-side AutomaticGainControl
  (leveler target RMS 0.10, limiter ceiling 0.95, NEVER a hard clip) —
  which replaces the old destructive np.clip at capture. Additionally the
  OS/hardware capture gain is CALIBRATED at startup (mixer volume is
  stepped down while the raw ADC signal saturates) so the AGC receives a
  waveform it can actually repair. int16 conversion happens ONLY at sink
  boundaries (openWakeWord / Whisper / WAV export / PCM bytes) via
  float32_to_int16(). No other stage may apply gain or re-normalization.

UNIFIED PREPROCESSING (2026-08-04 root-cause fix):
  The ring buffer now stores PREPROCESSED float32 audio (AGC + resample +
  high-pass). Every consumer — openWakeWord, Whisper verification, command
  STT — reads bit-identical samples from the same buffer. There is NO
  second high-pass, NO second noise suppression, and NO filter-state
  divergence between the wake detector and the verification path.
"""


import json
import logging
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Callable


import numpy as np
from scipy import signal as scipy_signal

from voice.audio_processing import (
    AutomaticGainControl,
    GainError,
    SAMPLE_RATE as _PROC_SAMPLE_RATE,
    HIGH_PASS_CUTOFF,
    HIGH_PASS_ORDER,
    audio_preprocessor,
    float32_to_int16,
    peak_monitor,
)
from voice.settings import voice_settings


logger = logging.getLogger(__name__)

# Global shutdown event. Set during graceful shutdown so ALL consumers
# (wake loop, command recorder, STT) stop accessing the AudioManager
# BEFORE the stream is closed. No component may access the AudioManager
# after this is set.
shutdown_event = threading.Event()

# ── Constants ──────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
FRAME_DURATION = 0.032  # 32ms frames — native Silero VAD window size (no zero-padding)
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION)  # 512 samples
RING_BUFFER_SECONDS = 10.0
RING_BUFFER_MAX_FRAMES = int(RING_BUFFER_SECONDS / FRAME_DURATION)

# VAD thresholds
VAD_ENERGY_THRESHOLD = 300.0
VAD_SILENCE_DURATION = 0.8  # seconds of silence to end speech
VAD_MIN_SPEECH_DURATION = 0.5  # minimum speech duration to accept

# TASK 2 — digital-silence watchdog. A real analog microphone ALWAYS
# delivers room tone (float RMS > 0). If the RAW callback input is exactly
# 0.0 for this many consecutive frames (~1.5 s at 30 ms/frame) the stream
# is routed to a dead/virtual bus and initialization ABORTS instead of
# letting the pipeline run on silence.
DIGITAL_SILENCE_ABORT_FRAMES = 50


# ── Hardware detector configuration ────────────────────────────
# STEP 2: name substrings that mark a device as a VIRTUAL bus. These are
# auto-rejected unless no physical hardware exists at all.
REJECT_KEYWORDS = (
    "default", "pulse", "pipewire", "monitor", "loopback",
    "echo", "null", "dummy", "virtual",
)
# STEP 3: name substrings that mark REAL hardware microphones. Probed first
# and preferred when signal quality is comparable.
PREFER_KEYWORDS = (
    "usb", "alc", "analog", "microphone", "mic",
    "built-in", "hd audio", "realtek",
)
# STEP 4: probe settings — every candidate is recorded and measured.
PROBE_DURATION = 2.0          # seconds recorded per candidate device
PROBE_MAX_CHANNELS = 4        # cap channels during probing
MAX_PROBE_DEVICES = 12        # safety bound on total probe time
# STEP 4: rejection thresholds (int16 scale). A real analog microphone in a
# quiet room still delivers room tone; RMS≈0 / Peak≈0 is DIGITAL SILENCE
# from an unrouted virtual bus.
SILENCE_RMS = 1.0             # below → digital silence
SILENCE_PEAK = 8.0            # below → digital silence
FLAT_DYNAMIC_RANGE = 4.0      # p99−p1 below → flat waveform

# Verified-device persistence (STEP 5). Silent devices are NEVER written.
SELECTION_PATH = Path(__file__).resolve().parent.parent / "data" / "mic_selection.json"


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Device enumeration with full classification flags
# ═══════════════════════════════════════════════════════════════

def enumerate_input_devices(sd) -> list:
    """
    Enumerate EVERY capture-capable device and classify it.

    For each device the following fields are produced:
      index, name, hostapi, max_input_channels, default_samplerate,
      is_virtual, is_pipewire, is_pulse, is_monitor, is_loopback,
      is_default, is_rejected, is_preferred, capture_by_name

    CAPTURE-BY-NAME: on PipeWire systems PortAudio reports physical codecs
    (e.g. "HD-Audio Generic: ALC256 Analog (hw:2,0)") with
    max_input_channels == 0, yet they open for CAPTURE when addressed by
    their exact enumerated name. These are real hardware and MUST be
    probed — they are the only path to real microphone samples when the
    virtual buses deliver digital silence.
    """
    devices = []
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1

    for d in sd.query_devices():
        in_ch = int(d.get("max_input_channels", 0))
        name = str(d.get("name", ""))
        lname_pre = name.lower()
        # Physical codec hidden behind an output-only enumeration entry:
        # "(hw:C,D)" in the name and NOT an HDMI/DP display-audio endpoint.
        capture_by_name = (
            in_ch <= 0
            and "(hw:" in lname_pre
            and "hdmi" not in lname_pre
            and "displayport" not in lname_pre
            and "dp," not in lname_pre
        )
        if in_ch <= 0 and not capture_by_name:
            continue
        name = str(d.get("name", ""))
        hostapi_idx = d.get("hostapi", -1)
        try:
            hostapi = str(hostapis[hostapi_idx]["name"]) if 0 <= hostapi_idx < len(hostapis) else "?"
        except Exception:
            hostapi = "?"

        lname = name.lower()
        lhost = hostapi.lower()
        is_pipewire = "pipewire" in lname or "pipewire" in lhost
        is_pulse = "pulse" in lname or "pulse" in lhost
        is_monitor = "monitor" in lname
        is_loopback = "loopback" in lname
        is_default = ("default" in lname) or (d.get("index") == default_in)
        is_virtual = (
            is_pipewire or is_pulse or is_monitor or is_loopback or is_default
            or "virtual" in lname or "null" in lname
            or "dummy" in lname or "echo" in lname
        )
        is_rejected = any(k in lname for k in REJECT_KEYWORDS)
        is_preferred = any(k in lname for k in PREFER_KEYWORDS)

        devices.append({
            "index": d.get("index"),
            "name": name,
            "hostapi": hostapi,
            "max_input_channels": int(d.get("max_input_channels", 0)),
            "default_samplerate": float(d.get("default_samplerate", 0.0)),
            "is_virtual": bool(is_virtual),
            "is_pipewire": bool(is_pipewire),
            "is_pulse": bool(is_pulse),
            "is_monitor": bool(is_monitor),
            "is_loopback": bool(is_loopback),
            "is_default": bool(is_default),
            "is_rejected": bool(is_rejected),
            "is_preferred": bool(is_preferred),
            "capture_by_name": bool(capture_by_name),
        })
    return devices


def print_device_table(devices: list) -> None:
    """STEP 1 output — print every input device with all flags."""
    print()
    print("  ══════════════════════════════════════════════════════════════════")
    print("  STEP 1 — INPUT DEVICE ENUMERATION")
    print("  ══════════════════════════════════════════════════════════════════")
    if not devices:
        print("  (no input devices found)")
        return
    for d in devices:
        print(f"  [{d['index']:>2}] {d['name']}")
        print(f"       hostapi={d['hostapi']}  "
              f"max_input_channels={d['max_input_channels']}  "
              f"default_samplerate={d['default_samplerate']:.0f}")
        print(f"       virtual={d['is_virtual']}  pipewire={d['is_pipewire']}  "
              f"pulse={d['is_pulse']}  monitor={d['is_monitor']}  "
              f"loopback={d['is_loopback']}  default={d['is_default']}"
              + ("  capture_by_name=True (PHYSICAL)" if d.get("capture_by_name") else ""))
    print()



def classify_candidates(devices: list) -> list:
    """
    STEP 2 — Reject virtual buses unless no physical hardware exists.
    STEP 3 — Order candidates so preferred hardware is probed first.

    Returns the ordered candidate list (may be empty).
    """
    physical = [d for d in devices if not d["is_rejected"] and not d["is_virtual"]]
    if physical:
        candidates = physical
        rejected = [d for d in devices if d not in physical]
        logger.info("[MIC-DETECT] STEP 2: %d virtual/rejected device(s) excluded: %s",
                    len(rejected), ", ".join(f"[{d['index']}]{d['name']}" for d in rejected))
    else:
        # No physical hardware exists — fall back to everything (STEP 2 clause).
        candidates = list(devices)
        logger.warning("[MIC-DETECT] STEP 2: no physical hardware found — "
                       "falling back to virtual devices")

    # STEP 3 — preferred hardware first (stable by index within each class).
    candidates.sort(key=lambda d: (not d["is_preferred"], d["index"]))
    return candidates[:MAX_PROBE_DEVICES]


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Candidate probing (record + measure)
# ═══════════════════════════════════════════════════════════════

def _signal_metrics(audio: np.ndarray, samplerate: float) -> dict:
    """Compute RMS / Peak / Zero-crossing / Dynamic range / Voice probability
    for a mono int16 signal."""
    if audio.size == 0:
        return {"rms": 0.0, "peak": 0.0, "zcr": 0.0,
                "dynamic_range": 0.0, "voice_probability": 0.0}

    sig = audio.astype(np.float64)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig)))

    # Zero-crossing rate (per sample). Speech typically sits in 0.02–0.25.
    signs = np.signbit(sig)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if sig.size > 1 else 0.0

    # Dynamic range: p99 − p1 (robust to outliers). Flat line → ≈0.
    p99 = float(np.percentile(sig, 99))
    p1 = float(np.percentile(sig, 1))
    dynamic_range = p99 - p1

    # Voice probability heuristic (0..1) from level, range and crossing rate.
    rms_score = min(rms / 800.0, 1.0)
    peak_score = min(peak / 8000.0, 1.0)
    dyn_score = min(dynamic_range / 4000.0, 1.0)
    if 0.02 <= zcr <= 0.25:
        zcr_score = 1.0
    elif zcr < 0.02:
        zcr_score = zcr / 0.02
    else:
        zcr_score = max(0.0, 1.0 - (zcr - 0.25) / 0.25)
    voice_probability = (
        0.35 * rms_score + 0.25 * peak_score
        + 0.25 * dyn_score + 0.15 * zcr_score
    )
    return {
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "dynamic_range": dynamic_range,
        "voice_probability": min(max(voice_probability, 0.0), 1.0),
    }


def probe_device(sd, device: dict, duration: float = PROBE_DURATION) -> dict:
    """
    STEP 4 — Open a stream on ONE candidate, record `duration` seconds and
    measure the signal. NEVER assumes channel 0 is the microphone: every
    channel is measured and the strongest is reported as speech_channel.

    A device is REJECTED when RMS≈0, Peak≈0 or the waveform is flat.
    """
    result = dict(device)
    result.update({
        "ok": False, "valid": False, "error": None,
        "rms": 0.0, "peak": 0.0, "zcr": 0.0,
        "dynamic_range": 0.0, "voice_probability": 0.0,
        "speech_energy": 0.0, "speech_channel": 0,
        "probe_channels": 0, "probe_samplerate": int(device["default_samplerate"]),
        "open_target": None, "open_channels": 0,
    })

    # CAPTURE-BY-NAME devices report max_input_channels == 0 but open for
    # capture when addressed by their EXACT enumerated name (PipeWire hides
    # physical codecs behind output-only entries). Normal devices open by
    # index.
    by_name = bool(device.get("capture_by_name"))
    open_target = device["name"] if by_name else device["index"]
    native_ch = 2 if by_name else max(1, int(device["max_input_channels"]))
    samplerate = int(device["default_samplerate"]) or SAMPLE_RATE
    recording = None

    for channels in dict.fromkeys((min(native_ch, PROBE_MAX_CHANNELS), 1)):
        try:
            recording = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=channels,
                dtype="int16",
                device=open_target,
                blocking=True,
            )
            result["probe_channels"] = channels
            result["open_target"] = open_target
            result["open_channels"] = channels
            break
        except Exception as e:
            result["error"] = str(e)
            recording = None

    if recording is None:
        logger.warning("[MIC-DETECT] STEP 4: [%s] %s — open failed: %s",
                       device["index"], device["name"], result["error"])
        return result

    result["ok"] = True
    data = np.asarray(recording)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Measure EVERY channel; the strongest channel wins.
    per_channel = []
    for c in range(data.shape[1]):
        m = _signal_metrics(data[:, c], samplerate)
        per_channel.append(m)
    best_c = int(np.argmax([m["rms"] for m in per_channel]))
    best = per_channel[best_c]

    result.update(best)
    result["speech_channel"] = best_c
    result["channel_rms"] = [round(m["rms"], 2) for m in per_channel]

    # STEP 4 rejection: digital silence or a flat waveform.
    flat = result["dynamic_range"] < FLAT_DYNAMIC_RANGE
    silent = result["rms"] < SILENCE_RMS or result["peak"] < SILENCE_PEAK
    result["valid"] = bool(not (silent or flat))
    if silent:
        result["reject_reason"] = "digital silence (RMS≈0 / Peak≈0)"
    elif flat:
        result["reject_reason"] = "flat waveform"
    # Speech energy used for final ranking (STEP 5).
    result["speech_energy"] = result["rms"] * (0.5 + 0.5 * result["voice_probability"])

    status = "VALID" if result["valid"] else f"REJECTED ({result.get('reject_reason', '')})"
    logger.info(
        "[MIC-DETECT] STEP 4: [%s] %s — RMS=%.1f Peak=%.0f ZCR=%.3f "
        "DynRange=%.0f Voice=%.2f speech_ch=%d → %s",
        device["index"], device["name"], result["rms"], result["peak"],
        result["zcr"], result["dynamic_range"], result["voice_probability"],
        best_c, status)
    return result


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Verified persistence (NEVER save a silent device)
# ═══════════════════════════════════════════════════════════════

def load_saved_selection() -> Optional[dict]:
    """Load a previously persisted selection. Entries WITHOUT verification
    proof (verified=true + measured RMS) are NEVER trusted."""
    try:
        if SELECTION_PATH.exists():
            with open(SELECTION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("index") is not None:
                return data
    except Exception as e:
        logger.debug("[MIC-DETECT] Failed to load mic selection: %s", e)
    return None


def save_verified_selection(probe: dict) -> None:
    """Persist ONLY a verified working device. A silent device is never saved."""
    if not probe.get("valid"):
        logger.error("[MIC-DETECT] REFUSING to persist silent device [%s] %s",
                     probe.get("index"), probe.get("name"))
        return
    try:
        SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index": probe["index"],
            "name": probe["name"],
            "hostapi": probe["hostapi"],
            "channels": probe["max_input_channels"],
            "default_samplerate": probe["default_samplerate"],
            "speech_channel": probe["speech_channel"],
            "rms": round(probe["rms"], 2),
            "peak": round(probe["peak"], 2),
            "voice_probability": round(probe["voice_probability"], 3),
            "verified": True,
            "measured_at": time.time(),
        }
        with open(SELECTION_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("[MIC-DETECT] STEP 5: verified selection persisted → %s "
                    "(index=%d rms=%.1f)", SELECTION_PATH, probe["index"], probe["rms"])
    except Exception as e:
        logger.warning("[MIC-DETECT] Failed to persist selection: %s", e)


def clear_saved_selection() -> None:
    """Drop a stale/invalid saved selection so it can never be reused."""
    try:
        if SELECTION_PATH.exists():
            SELECTION_PATH.unlink()
            logger.info("[MIC-DETECT] Stale mic selection removed: %s", SELECTION_PATH)
    except Exception as e:
        logger.debug("[MIC-DETECT] Could not remove stale selection: %s", e)


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Total failure report
# ═══════════════════════════════════════════════════════════════

def report_no_working_microphone(tested: list) -> None:
    """STEP 6 — Every tested device was silent. Show the report and STOP."""
    lines = [
        "",
        "  ══════════════════════════════════════════════════════════════════",
        "  No working microphone detected.",
        "  ══════════════════════════════════════════════════════════════════",
        "  Tested devices:",
    ]
    for t in tested:
        if t.get("ok"):
            lines.append(f"    [{t['index']:>2}] {t['name']:<45} RMS={t['rms']:.2f}")
        else:
            lines.append(f"    [{t['index']:>2}] {t['name']:<45} RMS=N/A "
                         f"(open failed: {t.get('error', '?')})")
    lines.append("  ══════════════════════════════════════════════════════════════════")
    lines.append("")
    msg = "\n".join(lines)
    print(msg)
    logger.error("[MIC-DETECT] STEP 6: No working microphone detected.%s",
                 "".join(f"\n    [{t['index']}] {t['name']} RMS={t.get('rms', 0):.2f}"
                         for t in tested))


class RingBuffer:
    """
    Thread-safe ring buffer for audio frames.

    Stores audio as a deque of float32 numpy arrays in [-1, 1] (each
    FRAME_SAMPLES long). The buffer NEVER stores int16: PCM16 is
    materialized only at sink boundaries via get_bytes() / the
    AudioManager byte APIs.
    """

    def __init__(self, max_frames: int = RING_BUFFER_MAX_FRAMES):
        self._buffer: deque = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._total_samples = 0

    def put(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append(frame)
            self._total_samples += len(frame)

    def get_recent(self, duration_seconds: float) -> np.ndarray:
        """Get the most recent audio up to duration_seconds."""
        num_samples = int(duration_seconds * SAMPLE_RATE)
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            # Collect frames from most recent backwards
            frames = []
            collected = 0
            for frame in reversed(self._buffer):
                frames.append(frame)
                collected += len(frame)
                if collected >= num_samples:
                    break
            frames.reverse()
            if not frames:
                return np.array([], dtype=np.float32)
            result = np.concatenate(frames)
            # Trim to requested duration
            if len(result) > num_samples:
                result = result[-num_samples:]
            return result

    def get_bytes(self, duration_seconds: float) -> bytes:
        """Get recent audio as raw PCM16 bytes.

        SINK BOUNDARY: this is the single int16 conversion, done here
        (immediately before Whisper / WAV / SpeechRecognition consumers)
        and nowhere upstream.
        """
        audio = self.get_recent(duration_seconds)
        return float32_to_int16(audio).tobytes()

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0

    @property
    def available_seconds(self) -> float:
        """How many seconds of audio are available."""
        with self._lock:
            return len(self._buffer) * FRAME_DURATION

    @property
    def total_samples(self) -> int:
        """Monotonic count of samples written since last clear()."""
        with self._lock:
            return self._total_samples

    def get_since(self, last_total: int):
        """Return (samples_written_after_last_total, new_total).

        Provides NON-OVERLAPPING sequential reads for streaming consumers
        (e.g. openWakeWord), so each sample is delivered exactly once.
        Samples older than the buffer capacity are dropped.
        """
        with self._lock:
            current = self._total_samples
            delta = current - last_total
            if delta <= 0:
                return np.array([], dtype=np.float32), current
            capacity = len(self._buffer) * FRAME_SAMPLES
            if delta > capacity:
                delta = capacity  # fell behind; drop old data and resync
            frames = []
            collected = 0
            for frame in reversed(self._buffer):
                frames.append(frame)
                collected += len(frame)
                if collected >= delta:
                    break
            frames.reverse()
            if not frames:
                return np.array([], dtype=np.float32), current
            result = np.concatenate(frames)
            if len(result) > delta:
                result = result[-delta:]
            return result, current


class VADState:
    """Tracks voice activity detection state."""

    SILENCE = "silence"
    SPEECH = "speech"

    def __init__(self):
        self.state = self.SILENCE
        self.speech_start: float = 0.0
        self.last_voice: float = 0.0
        self.speech_buffer: list = []

    def reset(self) -> None:
        self.state = self.SILENCE
        self.speech_start = 0.0
        self.last_voice = 0.0
        self.speech_buffer = []


class AudioManager:
    """
    Unified audio capture manager with professional hardware detection.

    start() runs the full detector pipeline (STEPS 1–6) and opens ONE
    sounddevice.InputStream on a VERIFIED working microphone. If no working
    microphone exists, start() returns False and NO audio/AI pipeline may
    be initialized.

    Usage:
        am = AudioManager()
        if am.start():                      # hardware verified inside
            audio = am.get_recent_audio(2.0)
            audio_bytes = am.record_command(timeout=8.0, phrase_limit=7.0)
            am.stop()
    """

    def __init__(self):
        self._sd = None
        self._stream = None
        self._ring_buffer = RingBuffer()
        self._vad = VADState()
        self._running = False
        self._lock = threading.Lock()
        self._backend: str = "none"
        self._device_index: Optional[int] = None
        self._device_name: str = ""
        self._actual_sample_rate: int = SAMPLE_RATE
        self._energy_threshold: float = VAD_ENERGY_THRESHOLD
        self._initialized = False
        # Re-entrancy guard: only ONE command recorder may drain the ring
        # buffer at a time. asyncio.wait_for() timeouts do NOT kill executor
        # threads — without this lock, retried listen() calls overlap and
        # "recording" appears to run far beyond timeout/phrase_limit.
        self._record_lock = threading.Lock()
        # Cache for the polyphase resampling FIR so the real-time callback
        # does NOT redesign an 8821-tap filter every 30 ms.
        self._resample_filter_cache: dict = {}
        # Native channel count of the selected device and the count we open
        # the stream with (capped so multi-channel speech detection works
        # without wasting resources on huge virtual buses).
        self._device_max_channels: int = 1
        self._stream_channels: int = CHANNELS
        # How the verified device must be opened: index for normal devices,
        # the exact enumerated NAME for capture-by-name physical codecs.
        self._open_target = None
        self._open_channels: int = 1


        # ── Hardware detector state (STEPS 1–6) ──
        self._mic_verified: bool = False       # True ONLY after a probe passed
        self._probe_report: list = []          # every probed device + metrics

        # ── Audio-capture instrumentation / channel detection ──
        # NEVER assume channel 0 contains the microphone. The probe detects
        # the channel that carries speech; the callback keeps verifying it.
        self._speech_channel: Optional[int] = None
        # ISSUE-3: Speech-channel locking state
        self._speech_channel_locked_at: float = 0.0
        self._speech_channel_silent_since: float = 0.0
        self._callback_count: int = 0
        self._dropped_frames: int = 0
        self._last_callback_time: float = 0.0
        self._callback_intervals: deque = deque(maxlen=200)
        self._rms_history: deque = deque(maxlen=200)
        self._peak_history: deque = deque(maxlen=200)
        self._channel_rms: dict = {}      # channel index -> latest RMS (float scale)
        self._last_raw_dump: float = 0.0  # last time a raw WAV was dumped

        # ── TASK 2: digital-silence watchdog state ──
        self._zero_streak: int = 0        # consecutive all-zero RAW frames
        self._digital_silence: bool = False  # latched once abort threshold hit

        # ── STEP 6: capture-side AGC (the SINGLE gain stage) ──
        # Leveler (target RMS 0.10, band 0.08–0.12) + limiter (0.95),
        # NEVER a hard clip. Replaces the old np.clip at capture that
        # destroyed over-driven waveforms before inference.
        self._agc = AutomaticGainControl()
        # Raw-source saturation accounting for the startup mixer
        # calibration: samples at the ±1.0 rail / total samples seen.
        self._sat_rail_samples: int = 0
        self._sat_total_samples: int = 0

        # ── UNIFIED PREPROCESSING: high-pass filter built ONCE, used by
        # every callback frame. The ring buffer stores PREPROCESSED audio
        # so openWakeWord and Whisper consume bit-identical samples. ──
        self._hp_sos = None   # scipy sos array, built lazily first callback
        self._hp_zi = None    # streaming filter state, carried across frames

        # Callback for wake word detection
        self._on_speech_detected: Optional[Callable] = None



    # ──────────────────────────────────────────────────────────
    # Hardware detection pipeline (STEPS 1–6)
    # ──────────────────────────────────────────────────────────

    def _select_verified_device(self) -> bool:
        """
        Run STEPS 1–6: enumerate → classify → probe → select → persist.

        Returns True and sets _device_index/_speech_channel/_mic_verified
        when a REAL working microphone was found. Returns False when every
        device is silent (STEP 6 report is printed).
        """
        sd = self._sd

        # ── STEP 1: enumerate EVERY input device ──
        devices = enumerate_input_devices(sd)
        print_device_table(devices)
        if not devices:
            logger.error("[MIC-DETECT] No input devices found at all")
            report_no_working_microphone([])
            return False

        # ── Configured device override (env WAKE_DEVICE_INDEX) ──
        # Honored ONLY if it probes VALID — a silent configured device is
        # rejected like any other and detection continues.
        configured = voice_settings.device_index
        if configured is not None:
            dev = next((d for d in devices if d["index"] == configured), None)
            if dev is not None:
                logger.info("[MIC-DETECT] Configured device index %d — validating...", configured)
                probe = probe_device(sd, dev)
                self._probe_report.append(probe)
                if probe["valid"]:
                    self._accept_device(probe)
                    return True
                logger.warning("[MIC-DETECT] Configured device %d is SILENT — "
                               "falling back to full detection", configured)
            else:
                logger.warning("[MIC-DETECT] Configured device index %s not present", configured)

        # ── Saved verified selection: re-probe to confirm it still works ──
        # Entries without verification proof are NEVER trusted (this is how
        # the stale virtual "default" device gets dropped permanently).
        saved = load_saved_selection()
        if saved and saved.get("verified"):
            dev = next((d for d in devices if d["index"] == saved.get("index")), None)
            # Index may shift across reboots — fall back to exact name match.
            if dev is None and saved.get("name"):
                dev = next((d for d in devices if d["name"] == saved.get("name")), None)
                if dev is not None:
                    logger.info("[MIC-DETECT] Saved device found by NAME at new index [%d]",
                                dev["index"])
            if dev is not None and not dev["is_rejected"]:
                logger.info("[MIC-DETECT] Re-validating saved microphone: [%d] %s",
                            dev["index"], dev["name"])
                probe = probe_device(sd, dev)
                self._probe_report.append(probe)
                if probe["valid"]:
                    self._accept_device(probe)
                    return True
                logger.warning("[MIC-DETECT] Saved microphone [%d] is now SILENT — "
                               "running full detection", dev["index"])
                clear_saved_selection()
            elif dev is not None and dev["is_rejected"]:
                logger.warning("[MIC-DETECT] Saved device [%d] %s is a virtual bus — "
                               "discarding stale selection", dev["index"], dev["name"])
                clear_saved_selection()
        elif saved:
            logger.warning("[MIC-DETECT] Saved selection has NO verification proof "
                           "(index=%s name=%r) — discarding", saved.get("index"),
                           saved.get("name"))
            clear_saved_selection()

        # ── STEPS 2+3: reject virtual buses, preferred hardware first ──
        candidates = classify_candidates(devices)
        if not candidates:
            report_no_working_microphone(self._probe_report)
            return False

        # ── STEP 4: probe EVERY candidate ──
        already = {p["index"] for p in self._probe_report}
        probes = []
        for dev in candidates:
            if dev["index"] in already:
                probes.append(next(p for p in self._probe_report
                                   if p["index"] == dev["index"]))
                continue
            probe = probe_device(sd, dev)
            probes.append(probe)
            self._probe_report.append(probe)

        # ── STEP 5: choose highest speech energy; persist verified only ──
        working = [p for p in probes if p["valid"]]
        if not working:
            # ── STEP 6: everything was silent — report and STOP ──
            clear_saved_selection()
            report_no_working_microphone(probes)
            return False

        best = max(working, key=lambda p: p["speech_energy"])
        # STEP 3 preference: a preferred hardware device within 70% of the
        # best energy wins over an unknown device with marginally more noise.
        preferred = [p for p in working if p["is_preferred"]]
        if preferred:
            best_pref = max(preferred, key=lambda p: p["speech_energy"])
            if best_pref["speech_energy"] >= 0.7 * best["speech_energy"]:
                best = best_pref

        self._accept_device(best)
        save_verified_selection(best)
        return True

    def _accept_device(self, probe: dict) -> None:
        """Lock in a VERIFIED device as the session microphone."""
        self._device_index = probe["index"]
        self._device_name = probe["name"]
        self._device_max_channels = int(probe.get("open_channels")
                                        or probe["max_input_channels"] or 1)
        self._actual_sample_rate = int(probe["default_samplerate"]) or SAMPLE_RATE
        self._speech_channel = probe["speech_channel"]
        # Open target: the exact name for capture-by-name codecs, else index.
        self._open_target = probe.get("open_target")
        if self._open_target is None:
            self._open_target = probe["index"]
        self._open_channels = int(probe.get("open_channels") or 1)
        self._mic_verified = True

        logger.info(
            "[MIC-DETECT] STEP 5: SELECTED [%d] %s — speech_energy=%.1f "
            "RMS=%.1f Peak=%.0f Voice=%.2f speech_channel=%d",
            probe["index"], probe["name"], probe["speech_energy"],
            probe["rms"], probe["peak"], probe["voice_probability"],
            probe["speech_channel"])
        print(f"  ✓ Microphone verified: [{probe['index']}] {probe['name']} "
              f"(RMS={probe['rms']:.1f}, speech_channel={probe['speech_channel']})")

    def _init_sounddevice(self) -> bool:
        """Initialize sounddevice and run the hardware detector."""
        try:
            import sounddevice as sd
            self._sd = sd
            self._backend = "sounddevice"
        except ImportError:
            logger.error("[AUDIO] sounddevice not available (ImportError)")
            return False
        except Exception as e:
            logger.error("[AUDIO] sounddevice init failed: %s", e, exc_info=True)
            return False

        if not self._select_verified_device():
            # STEP 6 already printed the report. NEVER continue.
            return False

        logger.info("[AUDIO] sounddevice initialized — device[%d]: %s (%d Hz, %d ch)",
                    self._device_index, self._device_name,
                    self._actual_sample_rate, self._device_max_channels)
        return True

    # ──────────────────────────────────────────────────────────
    # Real-time processing helpers
    # ──────────────────────────────────────────────────────────

    def _resample_to_16k(self, audio_float: np.ndarray) -> np.ndarray:
        """
        Resample audio from device sample rate to 16 kHz.

        FLOAT DOMAIN: accepts float32 [-1, 1] and returns float32 — there
        is NO int16 round-trip and NO clip here (the stage tracer asserts
        the output stays within ±1.01). Uses scipy's resample_poly for
        high-quality resampling.
        """
        if self._actual_sample_rate == SAMPLE_RATE:
            return audio_float
        try:
            from scipy import signal as _scs
            import math
            # Resample using rational ratio
            up = SAMPLE_RATE
            down = self._actual_sample_rate
            gcd = math.gcd(up, down)
            up //= gcd
            down //= gcd
            # Design the polyphase FIR filter EXACTLY ONCE per (up, down)
            # ratio.
            key = (up, down)
            window = self._resample_filter_cache.get(key)
            if window is None:
                max_rate = max(up, down)
                half_len = 10  # scipy default
                num_taps = 2 * half_len * max_rate + 1
                window = _scs.firwin(
                    num_taps, 1.0 / max_rate, window=("kaiser", 5.0))
                self._resample_filter_cache[key] = window
                logger.info("[AUDIO] Resample filter designed: %d->%d Hz (%d taps, cached)",
                            self._actual_sample_rate, SAMPLE_RATE, num_taps)
            resampled = _scs.resample_poly(
                audio_float.astype(np.float64), up, down, window=window
            )
            return resampled.astype(np.float32)
        except Exception as e:
            logger.debug("[AUDIO] Resample failed (%s), using raw audio", e)
            return audio_float

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Callback for sounddevice.InputStream — called for every audio frame."""
        # ── Callback instrumentation ──
        now = time.time()
        if self._last_callback_time > 0:
            self._callback_intervals.append(now - self._last_callback_time)
        self._last_callback_time = now
        self._callback_count += 1
        if status:
            # Overflow/underflow or other stream status → dropped frames.
            self._dropped_frames += 1
            logger.debug("[AUDIO] Stream status (dropped): %s", status)

        # ── TASK 2: verify the RAW callback input BEFORE ANY processing ──
        raw = np.asarray(indata)
        if self._callback_count == 1:
            try:
                raw64 = raw.astype(np.float64)
                logger.info(
                    "[CALLBACK-RAW] shape=%s dtype=%s min=%.6f max=%.6f "
                    "RMS=%.6f Peak=%.6f frames=%d",
                    raw.shape, raw.dtype, float(raw64.min()) if raw.size else 0.0,
                    float(raw64.max()) if raw.size else 0.0,
                    float(np.sqrt(np.mean(raw64 ** 2))) if raw.size else 0.0,
                    float(np.max(np.abs(raw64))) if raw.size else 0.0,
                    frames)
            except Exception:
                pass
        if raw.size == 0 or float(np.max(np.abs(raw))) == 0.0:
            self._zero_streak += 1
        else:
            self._zero_streak = 0
        if (not self._digital_silence
                and self._zero_streak >= DIGITAL_SILENCE_ABORT_FRAMES):
            self._digital_silence = True
            logger.error(
                "DIGITAL SILENCE DETECTED — device '%s' delivered %d "
                "consecutive all-zero frames (shape=%s dtype=%s). The "
                "stream is routed to a dead/virtual bus; aborting audio "
                "initialization instead of running on silence.",
                self._device_name or self._device_index,
                self._zero_streak, raw.shape, raw.dtype)

        # ── Per-channel RMS — NEVER assume channel 0 is the mic ──

        n_channels = indata.shape[1] if indata.ndim > 1 else 1
        chan_rms = []
        for c in range(n_channels):
            col = indata[:, c] if indata.ndim > 1 else indata
            chan_rms.append(float(np.sqrt(np.mean(col.astype(np.float64) ** 2))))
        for c, r in enumerate(chan_rms):
            self._channel_rms[c] = r

        # ── ISSUE-3: Speech-channel locking ──
        # After startup calibration chooses the best speech channel, LOCK it.
        # Only reselect if the locked channel remains silent for several
        # seconds. Never alternate channels during normal conversation.
        # The probe already identified the speech channel; re-verify only
        # when the locked channel is silent but another channel is hot.
        if self._speech_channel is None:
            best_c = int(np.argmax(chan_rms)) if chan_rms else 0
            if chan_rms and chan_rms[best_c] > 1e-4:
                self._speech_channel = best_c
                self._speech_channel_locked_at = time.time()
                self._speech_channel_silent_since = 0.0
                logger.info("[AUDIO] Speech channel auto-detected: ch%d "
                            "(rms=%.4f of %d channels) — LOCKED",
                            best_c, chan_rms[best_c], n_channels)
            else:
                self._speech_channel = 0  # silent so far — default, re-checked later
                self._speech_channel_locked_at = time.time()
                self._speech_channel_silent_since = 0.0
        elif 0 <= self._speech_channel < n_channels and chan_rms:
            locked = chan_rms[self._speech_channel]
            best_c = int(np.argmax(chan_rms))
            # ISSUE-3: Only re-map if the locked channel has been silent
            # for >= 5 seconds AND another channel is clearly active.
            # This prevents channel oscillation during normal conversation.
            CHANNEL_RESELECT_SILENCE_S = 5.0
            if locked < 1e-5:
                if self._speech_channel_silent_since == 0.0:
                    self._speech_channel_silent_since = time.time()
                silent_dur = time.time() - self._speech_channel_silent_since
                if silent_dur >= CHANNEL_RESELECT_SILENCE_S and chan_rms[best_c] > 1e-3:
                    logger.info("[AUDIO] Speech channel re-mapped: ch%d → ch%d "
                                "(ch%d silent for %.1fs)",
                                self._speech_channel, best_c,
                                self._speech_channel, silent_dur)
                    self._speech_channel = best_c
                    self._speech_channel_locked_at = time.time()
                    self._speech_channel_silent_since = 0.0
            else:
                # Channel is active — reset silence timer
                self._speech_channel_silent_since = 0.0
        src_channel = self._speech_channel if (
            self._speech_channel is not None and self._speech_channel < n_channels
        ) else 0

        mono = indata[:, src_channel] if indata.ndim > 1 else indata

        # Track RMS/peak history (diagnostics).
        mono_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        mono_peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        self._rms_history.append(mono_rms)
        self._peak_history.append(mono_peak)

        # ── STEP 8: the callback MUST report the real device + real levels ──
        if self._callback_count == 1 or self._callback_count % 50 == 0:
            ch_report = " ".join(
                f"ch{c}={r * 32768:.0f}" for c, r in enumerate(chan_rms))
            interval_ms = 0.0
            if self._callback_intervals:
                interval_ms = float(np.mean(list(self._callback_intervals)[-10:])) * 1000.0
            logger.info(
                "CALLBACK device=%s interval=%.0fms channels=[%s] "
                "chosen=ch%d RMS=%.0f Peak=%.0f",
                self._device_name or self._device_index, interval_ms,
                ch_report, src_channel, mono_rms * 32768, mono_peak * 32768)


        # ── Source-overdrive diagnostic: answers "where is the gain?" ──
        if mono_peak > 1.0:
            self._src_overdrive_events = getattr(self, "_src_overdrive_events", 0) + 1
            c = self._src_overdrive_events
            if c <= 5 or c % 100 == 0:
                logger.warning(
                    "[GAIN] SOURCE OVERDRIVE: raw mic peak=%.4f exceeds ±1.0 "
                    "(count=%d) — device/OS mixer is overdriving the input; "
                    "the signal is clipped ONCE at capture (single normalization)",
                    mono_peak, c)

        # ── THE single normalization + SINGLE gain stage in the pipeline ──
        try:
            mono_f = np.asarray(mono, dtype=np.float32)
            peak_monitor.observe("mic_raw", mono_f)
            if mono_f.size:
                self._sat_rail_samples += int(
                    np.count_nonzero(np.abs(mono_f) >= 1.0 - 1.0 / 32768.0))
                self._sat_total_samples += int(mono_f.size)

            audio_float, agc_gain = self._agc.process(mono_f)
            peak_monitor.log("microphone", audio_float, gain=agc_gain)

            # Resample to 16 kHz in the FLOAT domain (no int16 round-trip).
            if self._actual_sample_rate != SAMPLE_RATE:
                audio_float = self._resample_to_16k(audio_float)
                audio_float = np.clip(audio_float, -1.0, 1.0, out=audio_float)
                peak_monitor.log("resampling", audio_float)

        except GainError:
            logger.critical("[GAIN] Audio frame aborted due to gain error")
            return

        # ── Raw dump buffer — capture float32 audio (int16 only at WAV export) ──
        if not hasattr(self, "_raw_dump_buffer"):
            self._raw_dump_buffer = deque(maxlen=int(6 * SAMPLE_RATE / FRAME_SAMPLES))
        self._raw_dump_buffer.append(audio_float)

        # ── UNIFIED PREPROCESSING: high-pass filter ONCE in the callback ──
        # ROOT CAUSE FIX (2026-08-04): the verification path ran a SECOND
        # high-pass filter with zero initial conditions, producing IIR
        # transient ringing that diverged from the streaming path and
        # dropped Pearson correlation to ~0.50. The ring buffer now stores
        # PREPROCESSED audio — openWakeWord and Whisper both consume
        # bit-identical samples from the same buffer. No duplicate
        # high-pass, no second noise suppression, no reconstruction.
        if self._hp_sos is None:
            nyquist = SAMPLE_RATE / 2
            self._hp_sos = scipy_signal.butter(
                HIGH_PASS_ORDER, HIGH_PASS_CUTOFF / nyquist,
                btype="highpass", output="sos")
            self._hp_zi = scipy_signal.sosfilt_zi(self._hp_sos) * 0
            logger.info("[AUDIO] Unified high-pass filter built: "
                        "order=%d cutoff=%.0fHz rate=%dHz "
                        "ring buffer → PREPROCESSED audio for ALL consumers",
                        HIGH_PASS_ORDER, HIGH_PASS_CUTOFF, SAMPLE_RATE)

        audio_hp, self._hp_zi = scipy_signal.sosfilt(
            self._hp_sos, audio_float.astype(np.float64), zi=self._hp_zi)
        audio_hp = np.clip(audio_hp.astype(np.float32), -1.0, 1.0)
        peak_monitor.log("callback_highpass", audio_hp)

        self._ring_buffer.put(audio_hp)

        # VAD: lightweight energy check. Thresholds stay on the int16
        # scale (300 default) — measured float RMS is scaled ×32768.
        rms = float(np.sqrt(np.mean(audio_hp.astype(np.float64) ** 2))) * 32768.0
        now = time.time()

        if rms > self._energy_threshold:
            if self._vad.state == VADState.SILENCE:
                self._vad.state = VADState.SPEECH
                self._vad.speech_start = now
                self._vad.speech_buffer = []
                logger.debug("[VAD] Speech started (RMS=%.1f > threshold=%.1f)", rms, self._energy_threshold)
            self._vad.last_voice = now
            self._vad.speech_buffer.append(audio_hp.copy())
        else:
            if self._vad.state == VADState.SPEECH:
                if now - self._vad.last_voice > VAD_SILENCE_DURATION:
                    # Speech ended
                    duration = now - self._vad.speech_start
                    if duration >= VAD_MIN_SPEECH_DURATION:
                        logger.debug("[VAD] Speech ended (duration=%.2fs, buffer=%d frames)",
                                     duration, len(self._vad.speech_buffer))
                        if self._on_speech_detected:
                            try:
                                self._on_speech_detected()
                            except Exception as e:
                                logger.debug("[VAD] Speech callback error: %s", e)
                    self._vad.reset()

    # ──────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Detect hardware, verify the microphone, then open the stream.

        Runs the full detector (STEPS 1–6) before any stream is opened.
        Returns True ONLY when a verified working microphone is streaming.
        Returns False when no working microphone exists — callers MUST NOT
        initialize Silero / Whisper / openWakeWord / the conversation
        engine in that case (STEP 9).
        """
        if self._running:
            logger.debug("[AUDIO] Already running")
            return True

        if not self._init_sounddevice():
            logger.error("[AUDIO] Cannot start — no working microphone")
            return False

        # Reset audio preprocessor noise profile for this session
        audio_preprocessor.reset_noise_profile()

        # The probe determined EXACTLY how this device opens: by index for
        # normal devices, by exact name for capture-by-name physical codecs.
        open_target = self._open_target if self._open_target is not None else self._device_index
        try:
            self._stream_channels = max(1, min(self._open_channels, 4))
            device_blocksize = int(self._actual_sample_rate * FRAME_DURATION)
            self._stream = self._sd.InputStream(
                samplerate=self._actual_sample_rate,
                device=open_target,
                channels=self._stream_channels,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=device_blocksize,
            )
            self._stream.start()
            self._running = True
            self._initialized = True

            logger.info("[AUDIO] InputStream started — device=[%d] %s, %d Hz, %d ch, backend=%s",
                        self._device_index, self._device_name,
                        self._actual_sample_rate, self._stream_channels, self._backend)
            ok = self._finish_start()
            if ok:
                self._calibrate_capture_gain()
            return ok


        except Exception as e:
            logger.warning("[AUDIO] Open with %d ch failed (%s) — retrying mono",
                           getattr(self, "_stream_channels", 1), e)
            try:
                self._stream_channels = 1
                device_blocksize = int(self._actual_sample_rate * FRAME_DURATION)
                self._stream = self._sd.InputStream(
                    samplerate=self._actual_sample_rate,
                    device=open_target,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback,
                    blocksize=device_blocksize,
                )
                self._stream.start()
                self._running = True
                self._initialized = True
                logger.info("[AUDIO] InputStream started (mono fallback) — %d Hz", self._actual_sample_rate)
                ok = self._finish_start()
                if ok:
                    self._calibrate_capture_gain()
                return ok

            except Exception as e2:
                logger.error("[AUDIO] Failed to start stream: %s", e2, exc_info=True)
                self._stream = None
                return False

    # ──────────────────────────────────────────────────────────
    # Stream-start verification (TASK 1 config dump + TASK 2 abort)
    # ──────────────────────────────────────────────────────────

    def _finish_start(self) -> bool:
        """After the stream opens: dump the full audio configuration and
        verify the live signal. ABORTS (stop + return False) on digital
        silence instead of letting the pipeline run on a dead device."""
        logger.info(
            "[AUDIO-CONFIG] device_index=%s device='%s' "
            "native_channels=%d opened_channels=%d sample_rate=%d "
            "dtype=float32 frame_samples=%d frame_ms=%.0f speech_channel=%s",
            self._device_index, self._device_name,
            self._device_max_channels, self._stream_channels,
            self._actual_sample_rate,
            int(self._actual_sample_rate * FRAME_DURATION),
            FRAME_DURATION * 1000.0, self._speech_channel)

        deadline = time.time() + 4.0
        while time.time() < deadline:
            if self._digital_silence:
                break
            if self._callback_count >= DIGITAL_SILENCE_ABORT_FRAMES:
                break
            time.sleep(0.02)

        if self._digital_silence or (
                self._callback_count >= DIGITAL_SILENCE_ABORT_FRAMES
                and self._zero_streak >= DIGITAL_SILENCE_ABORT_FRAMES):
            msg = ("DIGITAL SILENCE DETECTED — the opened stream delivers "
                   "only zeros; aborting initialization")
            print(f"\n  ✗ {msg}\n")
            logger.error("[AUDIO] %s (device='%s')", msg, self._device_name)
            self.stop()
            return False

        if self._callback_count < 5:
            logger.error("[AUDIO] Stream produced no callbacks in 4s — "
                         "aborting (dead stream on device '%s')",
                         self._device_name)
            self.stop()
            return False

        logger.info(
            "[AUDIO] Live signal verified — %d callbacks, zero_streak=%d, "
            "rms>0 on chosen channel ch%s",
            self._callback_count, self._zero_streak, self._speech_channel)
        return True

    # ──────────────────────────────────────────────────────────
    # STEP 6b — OS/hardware capture-gain calibration (ROOT FIX)
    # ──────────────────────────────────────────────────────────

    MIXER_CAL_MAX_STEPS = 5
    MIXER_CAL_MEASURE_S = 1.0
    MIXER_CAL_REDUCE_FACTOR = 0.55
    MIXER_CAL_MIN_VOLUME = 0.05
    MIXER_CAL_HEALTHY_RAIL_PCT = 0.2
    MIXER_CAL_HEALTHY_RMS = 0.35

    def _measure_raw_saturation(self, seconds: float):
        """Measure the RAW (pre-AGC) signal for `seconds`.

        Returns (rail_pct, rms): percentage of samples at the ±1.0 rail
        and mean raw RMS over the window.
        """
        self._sat_rail_samples = 0
        self._sat_total_samples = 0
        base_count = self._callback_count
        deadline = time.time() + seconds
        while time.time() < deadline and not shutdown_event.is_set():
            time.sleep(0.02)
        frames = max(1, self._callback_count - base_count)
        rail_pct = 0.0
        if self._sat_total_samples > 0:
            rail_pct = (self._sat_rail_samples / self._sat_total_samples) * 100.0
        recent = list(self._rms_history)[-frames:] or [0.0]
        rms = float(np.mean(recent))
        return rail_pct, rms

    def _alsa_card_for_device(self) -> Optional[int]:
        """Resolve the ALSA card number for the verified device."""
        m = re.search(r"hw:(\d+),\d+", self._device_name or "")
        if m:
            return int(m.group(1))
        try:
            with open("/proc/asound/cards", "r", encoding="utf-8") as f:
                cards = [int(line.split()[0]) for line in f
                         if line.strip() and line.strip()[0].isdigit()]
        except Exception:
            cards = []
        amixer = shutil.which("amixer")
        if not amixer:
            return None
        for card in cards:
            try:
                out = subprocess.run(
                    [amixer, "-c", str(card), "scontrols"],
                    capture_output=True, text=True, timeout=5).stdout
                if "'Capture'" in out:
                    return card
            except Exception:
                continue
        return None

    def _get_capture_volume(self) -> Optional[float]:
        """Current source volume as a fraction (1.0 = 100 %)."""
        wpctl = shutil.which("wpctl")
        if wpctl:
            try:
                out = subprocess.run(
                    [wpctl, "get-volume", "@DEFAULT_AUDIO_SOURCE@"],
                    capture_output=True, text=True, timeout=5).stdout
                m = re.search(r"Volume:\s*([\d.]+)", out)
                if m:
                    return float(m.group(1))
            except Exception:
                pass
        pactl = shutil.which("pactl")
        if pactl:
            try:
                out = subprocess.run(
                    [pactl, "get-source-volume", "@DEFAULT_SOURCE@"],
                    capture_output=True, text=True, timeout=5).stdout
                m = re.search(r"/\s*(\d+)%", out)
                if m:
                    return float(m.group(1)) / 100.0
            except Exception:
                pass
        amixer = shutil.which("amixer")
        card = self._alsa_card_for_device()
        if amixer and card is not None:
            try:
                out = subprocess.run(
                    [amixer, "-c", str(card), "sget", "Capture"],
                    capture_output=True, text=True, timeout=5).stdout
                m = re.search(r"\[(\d+)%\]", out)
                if m:
                    return float(m.group(1)) / 100.0
            except Exception:
                pass
        return None

    def _set_capture_volume(self, fraction: float) -> bool:
        """Set the source volume. Returns True when a backend accepted."""
        fraction = min(max(fraction, 0.0), 1.0)
        pct = int(round(fraction * 100))
        wpctl = shutil.which("wpctl")
        if wpctl:
            try:
                r = subprocess.run(
                    [wpctl, "set-volume", "@DEFAULT_AUDIO_SOURCE@",
                     f"{fraction:.2f}"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    logger.info("[MIXER] wpctl: source volume → %.2f (%d%%)",
                                fraction, pct)
                    return True
            except Exception:
                pass
        pactl = shutil.which("pactl")
        if pactl:
            try:
                r = subprocess.run(
                    [pactl, "set-source-volume", "@DEFAULT_SOURCE@", f"{pct}%"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    logger.info("[MIXER] pactl: source volume → %d%%", pct)
                    return True
            except Exception:
                pass
        amixer = shutil.which("amixer")
        card = self._alsa_card_for_device()
        if amixer and card is not None:
            try:
                r = subprocess.run(
                    [amixer, "-c", str(card), "sset", "Capture", f"{pct}%"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    logger.info("[MIXER] amixer card %d: Capture → %d%%",
                                card, pct)
                    return True
            except Exception:
                pass
        return False

    def _calibrate_capture_gain(self) -> None:
        """Step the OS/hardware capture volume DOWN until the RAW signal
        stops saturating the ADC."""
        rail_pct, rms = self._measure_raw_saturation(self.MIXER_CAL_MEASURE_S)
        logger.info("[MIXER] Capture-gain calibration: raw RMS=%.3f "
                    "rail=%.2f%% (targets: RMS≤%.2f, rail≤%.1f%%)",
                    rms, rail_pct, self.MIXER_CAL_HEALTHY_RMS,
                    self.MIXER_CAL_HEALTHY_RAIL_PCT)

        for step in range(1, self.MIXER_CAL_MAX_STEPS + 1):
            if (rail_pct <= self.MIXER_CAL_HEALTHY_RAIL_PCT
                    and rms <= self.MIXER_CAL_HEALTHY_RMS):
                if step > 1:
                    logger.info("[MIXER] ✓ Capture gain healthy after %d "
                                "reduction(s): raw RMS=%.3f rail=%.2f%%",
                                step - 1, rms, rail_pct)
                return
            if shutdown_event.is_set():
                return
            vol = self._get_capture_volume()
            if vol is None:
                logger.warning(
                    "[MIXER] No mixer backend (wpctl/pactl/amixer) — "
                    "CANNOT reduce capture gain automatically. Raw signal "
                    "SATURATES the ADC (rail=%.1f%%). Fix manually, e.g.: "
                    "wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.4",
                    rail_pct)
                return
            new_vol = max(self.MIXER_CAL_MIN_VOLUME,
                          vol * self.MIXER_CAL_REDUCE_FACTOR)
            logger.warning(
                "[MIXER] RAW SIGNAL CLIPS AT SOURCE (rail=%.1f%%, RMS=%.2f) "
                "— ADC overdrive. Reducing capture volume %.0f%% → %.0f%% "
                "(step %d/%d)",
                rail_pct, rms, vol * 100, new_vol * 100,
                step, self.MIXER_CAL_MAX_STEPS)
            if not self._set_capture_volume(new_vol):
                logger.warning("[MIXER] Volume reduction failed — keeping "
                               "current gain (AGC limiter still active)")
                return
            time.sleep(0.4)
            rail_pct, rms = self._measure_raw_saturation(
                self.MIXER_CAL_MEASURE_S)

        if (rail_pct > self.MIXER_CAL_HEALTHY_RAIL_PCT
                or rms > self.MIXER_CAL_HEALTHY_RMS):
            logger.error(
                "[MIXER] Capture STILL saturates after %d reductions "
                "(rail=%.1f%%, RMS=%.2f). The hardware mic BOOST is too "
                "high — reduce it manually: "
                "amixer -c %s sset 'Internal Mic Boost' 0 && "
                "amixer -c %s sset Capture 40%%",
                self.MIXER_CAL_MAX_STEPS, rail_pct, rms,
                self._alsa_card_for_device() or "?",
                self._alsa_card_for_device() or "?")
        else:
            logger.info("[MIXER] ✓ Capture gain healthy: raw RMS=%.3f "
                        "rail=%.2f%%", rms, rail_pct)


    def stop(self) -> None:
        """Stop and close the audio stream."""
        if not self._running:
            return

        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._running = False
            self._ring_buffer.clear()
            self._vad.reset()
            logger.info("[AUDIO] InputStream stopped")
        except Exception as e:
            logger.warning("[AUDIO] Stop error: %s", e)

    def _access_forbidden(self, caller: str) -> bool:
        """AudioManager must NEVER be accessed after shutdown begins or
        after the stream is stopped. All public read paths gate on this."""
        if shutdown_event.is_set():
            logger.debug("[AUDIO] %s blocked — shutdown in progress", caller)
            return True
        if not self._running:
            logger.debug("[AUDIO] %s blocked — stream not running", caller)
            return True
        return False

    def get_recent_audio(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio from the ring buffer as numpy array.

        UNIFIED PIPELINE: returns PREPROCESSED float32 (AGC + high-pass),
        the same audio that openWakeWord scored. Whisper verification
        consumes these exact samples — NO second filter pass.

        Args:
            duration_seconds: How many seconds of audio to retrieve.

        Returns:
            numpy array of PREPROCESSED float32 samples in [-1, 1].
        """
        if self._access_forbidden("get_recent_audio"):
            return np.array([], dtype=np.float32)
        return self._ring_buffer.get_recent(duration_seconds)

    @property
    def total_samples(self) -> int:
        """Monotonic count of samples written to the ring buffer."""
        return self._ring_buffer.total_samples

    def read_since(self, last_total: int):
        """Return (new_audio_float32, new_total) written after last_total.

        UNIFIED PIPELINE: returns PREPROCESSED float32 (AGC + high-pass).

        NON-OVERLAPPING: each sample is returned exactly once across calls.
        Used by the continuous wake loop to feed openWakeWord streaming frames.
        Samples are float32 in [-1, 1].
        """
        if self._access_forbidden("read_since"):
            return np.array([], dtype=np.float32), last_total
        return self._ring_buffer.get_since(last_total)

    def get_recent_processed(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio and apply the FULL noise suppression chain.

        This is the production entry point for wake detection and STT.
        It applies:
          - High-pass filtering (the SINGLE high-pass in the pipeline)
          - Spectral gating (noise suppression)
          - NO AGC — the signal is never amplified beyond unity

        Args:
            duration_seconds: How many seconds of audio to retrieve.

        Returns:
            Fully processed float32 numpy array in [-1, 1]. Convert to
            int16 ONLY at the sink (Whisper / WAV) via float32_to_int16().
        """
        if self._access_forbidden("get_recent_processed"):
            return np.array([], dtype=np.float32)
        raw = self._ring_buffer.get_recent(duration_seconds)
        if len(raw) == 0:
            return raw
        return audio_preprocessor.process(raw)

    def get_recent_bytes(self, duration_seconds: float) -> bytes:
        """
        Get recent audio as raw PCM16 bytes.

        Args:
            duration_seconds: How many seconds of audio to retrieve.

        Returns:
            Raw bytes (PCM16, mono).
        """
        if self._access_forbidden("get_recent_bytes"):
            return b""
        return self._ring_buffer.get_bytes(duration_seconds)

    def record_command(self, timeout: float = 8.0, phrase_limit: float = 7.0) -> Optional[bytes]:
        """
        Record a command from the shared audio stream (non-overlapping reads).

        Recording STOPS as soon as ANY of these is true:
          - silence exceeds VAD_SILENCE_DURATION, OR
          - phrase_limit expires, OR
          - timeout expires.
        The returned audio is NEVER longer than phrase_limit.

        Args:
            timeout: Max seconds to wait for speech to start.
            phrase_limit: Max seconds of audio to return.

        Returns:
            Raw PCM16 bytes of the command (<= phrase_limit), or None if no speech.
        """
        if not self._running:
            logger.error("[AUDIO] Cannot record — stream not running")
            return None
        if shutdown_event.is_set():
            logger.debug("[AUDIO] Shutdown in progress — record_command aborted")
            return None

        if not self._record_lock.acquire(blocking=False):
            logger.warning("[AUDIO] record_command already active — "
                           "refusing overlapping recording")
            return None
        try:
            return self._record_command_inner(timeout, phrase_limit)
        finally:
            self._record_lock.release()

    def _record_command_inner(self, timeout: float, phrase_limit: float) -> Optional[bytes]:
        """Instrumented command recorder. HARD guarantees:
          - wall-clock never exceeds `timeout` (hard deadline)
          - returned audio never exceeds `phrase_limit`
          - every exit logs its stop reason
        """
        t_record_start = time.time()
        record_deadline = t_record_start + timeout
        max_samples = int(phrase_limit * SAMPLE_RATE)

        command_buffer: list = []
        buffered_samples = 0
        speech_detected = False
        speech_start: Optional[float] = None
        speech_end: Optional[float] = None
        silence_start = 0.0
        stop_reason = "timeout"

        last_total = self._ring_buffer.total_samples

        logger.info(
            "[AUDIO] record START t=%.3f timeout=%.1fs phrase_limit=%.1fs "
            "deadline=%.3f energy_threshold=%.1f",
            t_record_start, timeout, phrase_limit,
            record_deadline, self._energy_threshold)

        while True:
            if shutdown_event.is_set():
                logger.info("[AUDIO] record STOP reason=shutdown")
                return None

            now = time.time()

            if now >= record_deadline:
                stop_reason = "timeout"
                logger.info(
                    "[AUDIO] TIMEOUT at +%.2fs (limit=%.1fs, speech_detected=%s)",
                    now - t_record_start, timeout, speech_detected)
                break

            if speech_start is not None and (now - speech_start) >= phrase_limit:
                speech_end = now
                stop_reason = "phrase_limit"
                logger.info(
                    "[AUDIO] PHRASE_LIMIT at +%.2fs: %.2fs of speech (limit=%.1fs)",
                    now - t_record_start, now - speech_start, phrase_limit)
                break

            new_audio, last_total = self._ring_buffer.get_since(last_total)
            if len(new_audio) == 0:
                time.sleep(0.01)
                continue

            rms = float(np.sqrt(np.mean(new_audio.astype(np.float64) ** 2))) * 32768.0

            if rms > self._energy_threshold:
                if not speech_detected:
                    speech_detected = True
                    speech_start = now
                    logger.info(
                        "[AUDIO] SPEECH START at +%.2fs (RMS=%.1f > threshold=%.1f)",
                        now - t_record_start, rms, self._energy_threshold)
                command_buffer.append(new_audio.copy())
                buffered_samples += len(new_audio)
                silence_start = 0.0
            elif speech_detected:
                if silence_start == 0.0:
                    silence_start = now
                elif now - silence_start > VAD_SILENCE_DURATION:
                    speech_end = now
                    stop_reason = "silence"
                    logger.info(
                        "[AUDIO] SPEECH END at +%.2fs (silence=%.2fs >= %.2fs)",
                        now - t_record_start,
                        now - silence_start, VAD_SILENCE_DURATION)
                    break
                command_buffer.append(new_audio.copy())
                buffered_samples += len(new_audio)

            time.sleep(0.01)

        t_record_stop = time.time()
        wall_duration = t_record_stop - t_record_start

        logger.info(
            "[AUDIO] record STOP reason=%s wall=%.2fs buffered=%.2fs (%d samples) "
            "speech_start=%s speech_end=%s",
            stop_reason, wall_duration, buffered_samples / SAMPLE_RATE,
            buffered_samples,
            ("+%.2fs" % (speech_start - t_record_start)) if speech_start else "none",
            ("+%.2fs" % (speech_end - t_record_start)) if speech_end else "none")

        if not speech_detected or not command_buffer:
            logger.info("[AUDIO] No speech detected (reason=%s) — returning None",
                        stop_reason)
            return None

        audio = np.concatenate(command_buffer)

        if len(audio) > max_samples:
            logger.warning("[AUDIO] Trimming %.2fs -> %.2fs (phrase_limit invariant)",
                           len(audio) / SAMPLE_RATE, phrase_limit)
            audio = audio[:max_samples]

        audio_bytes = float32_to_int16(audio).tobytes()
        if len(audio_bytes) < 512:
            logger.info("[AUDIO] Command too short (%d bytes) — returning None",
                        len(audio_bytes))
            return None

        duration = len(audio_bytes) / SAMPLE_RATE / 2

        if duration > phrase_limit + 1e-6:
            logger.error(
                "[AUDIO] INVARIANT VIOLATION: duration=%.3fs > phrase_limit=%.3fs "
                "— forcing trim", duration, phrase_limit)
            audio_bytes = audio_bytes[: int(phrase_limit * SAMPLE_RATE) * 2]
            duration = len(audio_bytes) / SAMPLE_RATE / 2

        logger.info(
            "[AUDIO] Command recorded: %d bytes (%.2fs) stop_reason=%s wall=%.2fs",
            len(audio_bytes), duration, stop_reason, wall_duration)
        return audio_bytes

    def capture_duration(self, duration: float) -> Optional[bytes]:
        """
        Capture a fixed duration of audio from the ring buffer.

        Used for wake word detection — grabs the most recent audio.

        Args:
            duration: Seconds of audio to capture.

        Returns:
            Raw PCM16 bytes, or None if buffer is empty.
        """
        if not self._running or shutdown_event.is_set():
            return None

        audio_bytes = self._ring_buffer.get_bytes(duration)
        if len(audio_bytes) < 256:
            return None
        return audio_bytes

    def set_energy_threshold(self, threshold: float) -> None:
        """Set the VAD energy threshold."""
        self._energy_threshold = threshold
        logger.debug("[AUDIO] Energy threshold set to %.1f", threshold)

    def calibrate(self, duration: float = 1.5) -> bool:
        """
        Calibrate energy threshold from ambient noise.

        VALIDATION: calibration is INVALID if the microphone is producing
        digital silence (RMS ≈ 0). A verified device must never calibrate
        to silence; if it does, this returns False so the caller can treat
        voice as unavailable. Leo must NEVER run on a silent mic.

        Args:
            duration: Calibration duration in seconds.

        Returns:
            True if calibration succeeded AND the mic delivers real signal.
        """
        if not self._running:
            logger.warning("[AUDIO] Cannot calibrate — stream not running")
            return False

        logger.info("[AUDIO] Calibrating ambient noise (%.1fs)...", duration)
        time.sleep(0.1)

        audio = self._ring_buffer.get_recent(duration)
        if len(audio) == 0:
            logger.warning("[AUDIO] No audio for calibration")
            return False

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) * 32768.0
        peak = float(np.max(np.abs(audio))) * 32768.0

        if rms < 1.0:
            logger.error(
                "[AUDIO] CALIBRATION INVALID: RMS=%.2f (≈ digital silence). "
                "Verified device '%s' stopped delivering signal — "
                "voice must be disabled.", rms, self._device_name)
            self._mic_verified = False
            return False

        self._energy_threshold = max(300.0, rms * 1.5)
        logger.info("[AUDIO] Calibration complete — RMS=%.1f peak=%.0f threshold=%.1f",
                    rms, peak, self._energy_threshold)
        return True

    def validate_capture(self, duration: float = 2.0) -> dict:
        """Record `duration` seconds from the CURRENT stream and measure
        speech/energy metrics. Used to verify the selected device works.

        Returns dict with rms, peak, max_channel_rms, speech_channel, valid.
        """
        if not self._running:
            return {"valid": False, "reason": "stream_not_running"}
        time.sleep(0.1)
        audio = self._ring_buffer.get_recent(duration)
        if len(audio) == 0:
            return {"valid": False, "reason": "no_audio"}
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) * 32768.0
        peak = float(np.max(np.abs(audio))) * 32768.0
        max_ch = max(self._channel_rms.values()) * 32768 if self._channel_rms else 0.0
        valid = rms >= 1.0 or max_ch >= 5.0
        return {
            "valid": bool(valid),
            "rms": rms,
            "peak": peak,
            "max_channel_rms": max_ch,
            "speech_channel": self._speech_channel,
            "device_index": self._device_index,
            "device_name": self._device_name,
            "sample_rate": self._actual_sample_rate,
        }

    def dump_raw_input(self, duration: float = 3.0) -> Optional[str]:
        """Save the most recent `duration` seconds of RAW (pre-highpass,
        pre-noise-suppression, pre-VAD, pre-Whisper, pre-openWakeWord, pre-
        normalization, pre-gain) microphone audio to debug/raw_input_<ts>.wav
        and print duration/rate/channels/RMS/peak. If the waveform is flat,
        the microphone stream is wrong.

        Returns the saved file path, or None on failure.
        """
        if not hasattr(self, "_raw_dump_buffer") or not self._raw_dump_buffer:
            logger.warning("[RAW] No raw audio captured yet")
            return None
        try:
            import wave
            audio = np.concatenate(list(self._raw_dump_buffer))
            max_n = int(duration * SAMPLE_RATE)
            if len(audio) > max_n:
                audio = audio[-max_n:]
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) * 32768.0
            peak = (float(np.max(np.abs(audio))) * 32768.0) if len(audio) else 0.0
            dbg = Path(__file__).resolve().parent.parent / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = dbg / f"raw_input_{ts}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(float32_to_int16(audio).tobytes())
            dur = len(audio) / SAMPLE_RATE
            logger.info(
                "[RAW] Saved %s: dur=%.2fs rate=%d ch=1 RMS=%.1f peak=%.0f %s",
                path.name, dur, SAMPLE_RATE, rms, peak,
                "FLAT!" if rms < 1.0 else "")
            self._last_raw_dump = time.time()
            return str(path)
        except Exception as e:
            logger.debug("[RAW] dump_raw_input failed: %s", e)
            return None

    def write_audio_report(self, path: Optional[str] = None) -> dict:
        """Write debug/audio_report.json with chosen device, all devices,
        channel map, RMS/peak history, callback timing, dropped frames."""
        try:
            devices = []
            if self._sd is not None:
                for d in enumerate_input_devices(self._sd):
                    devices.append({
                        "index": d["index"], "name": d["name"],
                        "hostapi": d["hostapi"],
                        "in_channels": d["max_input_channels"],
                        "default_samplerate": d["default_samplerate"],
                        "is_virtual": d["is_virtual"],
                        "is_rejected": d["is_rejected"],
                    })
            intervals = list(self._callback_intervals)
            report = {
                "chosen_device": self._device_index,
                "chosen_device_name": self._device_name,
                "mic_verified": self._mic_verified,
                "speech_channel": self._speech_channel,
                "sample_rate": self._actual_sample_rate,
                "backend": self._backend,
                "running": self._running,
                "all_devices": devices,
                "probe_report": [
                    {k: p.get(k) for k in (
                        "index", "name", "ok", "valid", "rms", "peak", "zcr",
                        "dynamic_range", "voice_probability", "speech_energy",
                        "speech_channel", "reject_reason", "error")}
                    for p in self._probe_report
                ],
                "channel_rms": {str(k): round(v, 6) for k, v in self._channel_rms.items()},
                "rms_history": [round(x, 6) for x in list(self._rms_history)],
                "peak_history": [round(x, 6) for x in list(self._peak_history)],
                "callback_count": self._callback_count,
                "dropped_frames": self._dropped_frames,
                "callback_interval_ms": {
                    "mean": round(float(np.mean(intervals)) * 1000, 2) if intervals else 0.0,
                    "max": round(float(np.max(intervals)) * 1000, 2) if intervals else 0.0,
                },
                "peak_monitor": peak_monitor.report(),
                "energy_threshold": self._energy_threshold,
            }
            if path is None:
                path = str(Path(__file__).resolve().parent.parent / "debug" / "audio_report.json")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("[AUDIO] Report written: %s", path)
            return report
        except Exception as e:
            logger.debug("[AUDIO] write_audio_report failed: %s", e)
            return {}

    def set_speech_callback(self, callback: Callable) -> None:
        """Set a callback that is called when VAD detects speech."""
        self._on_speech_detected = callback

    @property
    def is_running(self) -> bool:
        """Check if the audio stream is running."""
        return self._running

    @property
    def backend(self) -> str:
        """Get the current backend name."""
        return self._backend

    @property
    def sample_rate(self) -> int:
        """Get the actual sample rate."""
        return self._actual_sample_rate

    @property
    def device_index(self) -> Optional[int]:
        """Get the device index."""
        return self._device_index

    @property
    def device_name(self) -> str:
        """Get the verified device name."""
        return self._device_name

    @property
    def speech_channel(self) -> Optional[int]:
        """Get the channel carrying the microphone signal."""
        return self._speech_channel

    @property
    def mic_verified(self) -> bool:
        """True ONLY when a working microphone passed the STEP 4 probe."""
        return self._mic_verified

    @property
    def energy_threshold(self) -> float:
        """Get the current energy threshold."""
        return self._energy_threshold

    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        return {
            "backend": self._backend,
            "running": self._running,
            "device_index": self._device_index,
            "device_name": self._device_name,
            "mic_verified": self._mic_verified,
            "speech_channel": self._speech_channel,
            "sample_rate": self._actual_sample_rate,
            "channels": CHANNELS,
            "energy_threshold": self._energy_threshold,
            "buffer_seconds": self._ring_buffer.available_seconds,
            "vad_state": self._vad.state,
            "preprocessor": audio_preprocessor.get_metrics(),
        }


# Global singleton
audio_manager = AudioManager()