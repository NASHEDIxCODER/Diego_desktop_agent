"""
Speech-to-Text module for Leo Desktop Assistant.

ARCHITECTURE:
  Uses the unified AudioManager (single sounddevice.InputStream).
  Wake detection and command capture both read from the SAME ring buffer.
  Never opens a separate stream or instantiates PyAudio.

  AudioManager → Ring Buffer → Wake Detector / Command Recorder → SpeechRecognition.AudioData

WAKE DETECTION PIPELINE (offline, no Google):
  Microphone
    ↓
  Silero VAD (voice activity detection)
    ↓
  openWakeWord (wake word detection)
    ↓
  If wake detected
    ↓
  Whisper (offline STT)
    ↓
  Planner
    ↓
  Plugins

  Google SpeechRecognition is ONLY an optional fallback for command STT.

Python 3.14 compatibility:
  Imports compat module for aifc/audioop stubs needed by speech_recognition.
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config.settings import settings
from voice.audio_manager import audio_manager, SAMPLE_RATE, shutdown_event
from voice.audio_processing import audio_preprocessor
from voice.settings import voice_settings
from voice.wake_model_manager import wake_model_manager, WARMUP_FRAME_SAMPLES

# ═══════════════════════════════════════════════════════════════
# AUDIO BACKEND DETECTION (done once at module load, NEVER in loops)
# ═══════════════════════════════════════════════════════════════
HAS_SOUNDDEVICE = False
HAS_SPEECH_RECOGNITION = False
AUDIO_BACKEND = None
sd = None  # sounddevice module (or None if unavailable)
_BACKEND_INIT_ERROR = None  # Stores the full exception info if backend init fails

# Try sounddevice first — NEVER swallow ImportError silently
try:
    import sounddevice as _sd
    sd = _sd
    HAS_SOUNDDEVICE = True
    AUDIO_BACKEND = "sounddevice"
except ImportError as _e:
    _BACKEND_INIT_ERROR = traceback.format_exc()
    logging.getLogger(__name__).error(
        "[BACKEND] sounddevice import failed (ImportError):\n%s",
        _BACKEND_INIT_ERROR
    )
except ModuleNotFoundError as _e:
    _BACKEND_INIT_ERROR = traceback.format_exc()
    logging.getLogger(__name__).error(
        "[BACKEND] sounddevice not found (ModuleNotFoundError):\n%s",
        _BACKEND_INIT_ERROR
    )
except OSError as _e:
    _BACKEND_INIT_ERROR = traceback.format_exc()
    logging.getLogger(__name__).error(
        "[BACKEND] sounddevice OS error (missing PortAudio library?):\n%s",
        _BACKEND_INIT_ERROR
    )
except Exception as _e:
    _BACKEND_INIT_ERROR = traceback.format_exc()
    logging.getLogger(__name__).error(
        "[BACKEND] sounddevice unknown error:\n%s",
        _BACKEND_INIT_ERROR
    )

# Try speech_recognition (for STT recognition, NOT for microphone capture)
try:
    import speech_recognition as _sr_check
    HAS_SPEECH_RECOGNITION = True
    if AUDIO_BACKEND is None:
        AUDIO_BACKEND = "speech_recognition"
except ImportError as _e:
    if AUDIO_BACKEND is None:
        logging.getLogger(__name__).debug(
            "[BACKEND] speech_recognition not available: %s", _e
        )
except Exception as _e:
    if AUDIO_BACKEND is None:
        logging.getLogger(__name__).debug(
            "[BACKEND] speech_recognition init error: %s", _e
        )

logger = logging.getLogger(__name__)
logger.info("Audio backend: %s", AUDIO_BACKEND if AUDIO_BACKEND else "none")

if AUDIO_BACKEND is None:
    logger.warning("No microphone backend available. Voice commands disabled.")

# Debug directory for saving raw audio
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"

# Lazy-import speech_recognition
_recognizer_module = None
_recognizer: Optional[object] = None
_calibrated = False

# ═══════════════════════════════════════════════════════════════
# OFFLINE WAKE DETECTION (Silero VAD + openWakeWord)
# ═══════════════════════════════════════════════════════════════
_silero_vad = None
_silero_vad_available = False
_openwakeword = None
_openwakeword_available = False
_whisper_model = None
_whisper_available = False
_last_vad_confidence: float = 0.0


def init_wake_detection() -> dict:
    """Initialize ALL offline wake detection components at startup.

    This is called from _init_voice() BEFORE the health report is printed,
    so startup diagnostics reflect the REAL runtime state.

    Returns:
        dict with keys: silero_vad, openwakeword, whisper
    """
    return {
        "silero_vad": _init_silero_vad(),
        "openwakeword": _init_openwakeword(),
        "whisper": _init_whisper(),
    }


def _init_silero_vad() -> bool:
    """Initialize Silero VAD for voice activity detection."""
    global _silero_vad, _silero_vad_available
    if _silero_vad_available:
        return True
    try:
        import torch
        from silero_vad import load_silero_vad
        _silero_vad = load_silero_vad()
        _silero_vad_available = True
        logger.info("[VAD] Silero VAD initialized")
        return True
    except Exception as e:
        logger.warning("[VAD] Silero VAD init failed: %s", e)
        return False


def _init_openwakeword() -> bool:
    """Initialize openWakeWord via WakeModelManager.

    The WakeModelManager resolves the model from WAKE_MODEL → models/wake/*.onnx
    → bundled openWakeWord model best matching WAKE_PHRASE. The hey_jarvis
    model is NEVER hardcoded.
    """
    global _openwakeword, _openwakeword_available
    if _openwakeword_available:
        return True
    try:
        ok = wake_model_manager.load()
        if ok:
            _openwakeword = True
            _openwakeword_available = True
            logger.info("[WAKE] openWakeWord initialized (model: %s, phrase: '%s')",
                        wake_model_manager.model_name, wake_model_manager.wake_phrase)
            return True
        logger.warning("[WAKE] openWakeWord init failed: %s",
                       wake_model_manager.load_error or "no model")
        return False
    except Exception as e:
        logger.warning("[WAKE] openWakeWord init failed: %s", e)
        return False


def _init_whisper() -> bool:
    """Initialize faster-whisper for offline STT."""
    global _whisper_model, _whisper_available
    if _whisper_available:
        return True
    try:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        _whisper_available = True
        logger.info("[STT] faster-whisper initialized (model: base)")
        return True
    except Exception as e:
        logger.warning("[STT] faster-whisper init failed: %s", e)
        return False


# Silero VAD 6.x requires EXACTLY 512 samples per window at 16 kHz.
# Feeding any other length (e.g. the old 1600-sample chunks) raises a
# TorchScript error which was silently swallowed, returning False forever —
# this is why wake detection never fired.
_VAD_WINDOW = 512


def _silero_vad_confidence(audio_int16: np.ndarray) -> float:
    """Return max Silero VAD speech probability across 512-sample windows.

    Accepts any input length; internally slices into 512-sample windows
    (the only size Silero VAD 6.x accepts) and returns the maximum
    probability found. Returns 0.0 on error or when VAD is unavailable.
    """
    global _last_vad_confidence
    if not _silero_vad_available:
        return 0.0
    try:
        import torch
        audio_float = audio_int16.astype(np.float32) / 32768.0
        if len(audio_float) < _VAD_WINDOW:
            audio_float = np.pad(audio_float, (0, _VAD_WINDOW - len(audio_float)))
        max_prob = 0.0
        for i in range(0, len(audio_float) - _VAD_WINDOW + 1, _VAD_WINDOW):
            window = audio_float[i:i + _VAD_WINDOW]
            tensor = torch.from_numpy(window).unsqueeze(0)
            prob = float(_silero_vad(tensor, 16000).item())
            if prob > max_prob:
                max_prob = prob
        _last_vad_confidence = max_prob
        return max_prob
    except Exception as e:
        logger.debug("[VAD] Silero VAD error: %s", e)
        return 0.0


def _silero_vad_detect(audio_int16: np.ndarray) -> bool:
    """Detect voice activity using Silero VAD (512-sample windows)."""
    return _silero_vad_confidence(audio_int16) > 0.5


def _openwakeword_detect(audio_int16: np.ndarray) -> bool:
    """
    Detect wake word using the WakeModelManager (openWakeWord + verifier).

    Args:
        audio_int16: int16 PCM samples at 16 kHz.

    Returns:
        True if wake word detected.
    """
    if not _openwakeword_available:
        return False
    try:
        preds = wake_model_manager.predict(audio_int16)
        if not preds:
            return False
        best_name, best_score = wake_model_manager.highest_score()
        if best_score > 0.5:
            logger.info("[WAKE] openWakeWord detected '%s' (score=%.3f)", best_name, best_score)
            return True
        return False
    except Exception as e:
        logger.debug("[WAKE] openWakeWord error: %s", e)
        return False


def _whisper_transcribe_detailed(audio_int16: np.ndarray) -> dict:
    """
    Transcribe audio with faster-whisper and report the FULL decision.

    Returns a dict:
      text       — transcript (or None)
      confidence — mean segment avg_logprob (negative; closer to 0 = better)
      no_speech  — max segment no_speech_prob (0..1)
      ok         — True if the transcript is accepted
      reason     — why accepted, or why rejected (used to gate the Google fallback)
    """
    if not _whisper_available:
        return {"text": None, "confidence": 0.0, "no_speech": 0.0,
                "ok": False, "reason": "whisper_unavailable"}
    if audio_int16 is None or len(audio_int16) == 0:
        return {"text": None, "confidence": 0.0, "no_speech": 0.0,
                "ok": False, "reason": "empty_audio"}

    try:
        from voice.audio_processing import peak_monitor as _pm
        _pm.log("whisper_input", audio_int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        # Robustness settings (deterministic, no repetition loops, internal
        # VAD filter so silence/noise never reaches the decoder — this is
        # what previously produced no_segments / hallucinated transcripts
        # and pushed the pipeline into the Google fallback).
        segments, info = _whisper_model.transcribe(
            audio_float,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        segs = list(segments)
        if not segs:
            return {"text": None, "confidence": 0.0, "no_speech": 0.0,
                    "ok": False, "reason": "no_segments"}

        text = " ".join(s.text.strip() for s in segs).strip()
        logprobs = [s.avg_logprob for s in segs if getattr(s, "avg_logprob", None) is not None]
        confidence = float(np.mean(logprobs)) if logprobs else 0.0
        no_speech = max((getattr(s, "no_speech_prob", 0.0) or 0.0) for s in segs)

        if not text:
            return {"text": None, "confidence": confidence, "no_speech": no_speech,
                    "ok": False, "reason": "empty_transcript"}
        if no_speech > 0.6:
            return {"text": text, "confidence": confidence, "no_speech": no_speech,
                    "ok": False, "reason": f"high_no_speech_prob({no_speech:.2f})"}

        return {"text": text, "confidence": confidence, "no_speech": no_speech,
                "ok": True, "reason": "accepted"}
    except Exception as e:
        logger.warning("[STT] Whisper exception: %s: %s", type(e).__name__, e)
        return {"text": None, "confidence": 0.0, "no_speech": 0.0,
                "ok": False, "reason": f"exception:{type(e).__name__}:{e}"}


def _whisper_transcribe(audio_int16: np.ndarray) -> Optional[str]:
    """
    Transcribe audio using faster-whisper (offline).

    Returns the accepted transcript, or None if Whisper genuinely failed.
    Logs the full decision tree (transcript, confidence, accept/reject reason).
    """
    res = _whisper_transcribe_detailed(audio_int16)
    if res["ok"]:
        logger.info("[STT] Whisper ACCEPTED: '%s' (conf=%.3f)",
                    res["text"], res["confidence"])
        return res["text"]
    logger.info("[STT] Whisper REJECTED: reason=%s conf=%.3f no_speech=%.2f text=%r",
                res["reason"], res["confidence"], res["no_speech"], res["text"])
    return None


def get_backend_diagnostics() -> dict:
    """Return a detailed diagnostics report for the audio backend."""
    diag = {
        "has_sounddevice": HAS_SOUNDDEVICE,
        "has_speech_recognition": HAS_SPEECH_RECOGNITION,
        "audio_backend": AUDIO_BACKEND,
        "backend_init_error": _BACKEND_INIT_ERROR,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "audio_manager": audio_manager.get_diagnostics(),
        "silero_vad": _silero_vad_available,
        "openwakeword": _openwakeword_available,
        "whisper": _whisper_available,
    }

    if HAS_SOUNDDEVICE and sd is not None:
        try:
            diag["sounddevice_version"] = sd.__version__
        except Exception:
            diag["sounddevice_version"] = "unknown"
        try:
            diag["portaudio_version"] = sd.get_portaudio_version()
        except Exception:
            diag["portaudio_version"] = "unknown"
        try:
            diag["default_device"] = sd.default.device
        except Exception:
            diag["default_device"] = "unknown"
        try:
            devices = sd.query_devices()
            diag["all_devices"] = [
                {
                    "index": d['index'],
                    "name": d['name'],
                    "inputs": d['max_input_channels'],
                    "outputs": d['max_output_channels'],
                    "samplerate": d['default_samplerate'],
                }
                for d in devices
            ]
            diag["input_devices"] = [
                d for d in diag["all_devices"] if d["inputs"] > 0
            ]
        except Exception as e:
            diag["device_query_error"] = str(e)

    return diag


def print_startup_diagnostics() -> None:
    """Print comprehensive startup diagnostics for the voice subsystem."""
    print()
    print("  ═══════════════════════════════════════════")
    print("  VOICE SUBSYSTEM DIAGNOSTICS")
    print("  ═══════════════════════════════════════════")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version:    {sys.version.split()[0]}")
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX") or "none"
    print(f"  Virtualenv:        {venv}")

    if HAS_SOUNDDEVICE and sd is not None:
        try:
            print(f"  sounddevice:       {sd.__version__}")
        except Exception:
            print("  sounddevice:       unknown version")
        try:
            pa_ver = sd.get_portaudio_version()
            print(f"  PortAudio:         v{pa_ver}")
        except Exception:
            print("  PortAudio:         unknown")
        try:
            default_dev = sd.default.device
            print(f"  Default device:    {default_dev}")
        except Exception:
            print("  Default device:    unknown")
        try:
            devices = sd.query_devices()
            print(f"  All devices:       {len(devices)} found")
            input_devs = [d for d in devices if d['max_input_channels'] > 0]
            print(f"  Microphones:       {len(input_devs)} found")
            for d in input_devs:
                print(f"    [{d['index']}] {d['name']} "
                      f"({d['max_input_channels']} ch, {d['default_samplerate']} Hz)")
        except Exception as e:
            print(f"  Device query:      FAILED - {e}")
    else:
        print("  sounddevice:       NOT AVAILABLE")
        if _BACKEND_INIT_ERROR:
            print("  Init error:")
            for line in _BACKEND_INIT_ERROR.strip().split('\n'):
                print(f"    {line}")

    # AudioManager state
    am_diag = audio_manager.get_diagnostics()
    print(f"  AudioManager:      backend={am_diag['backend']}, running={am_diag['running']}")
    if am_diag['running']:
        print(f"    Device:          [{am_diag['device_index']}]")
        print(f"    Sample rate:     {am_diag['sample_rate']} Hz")
        print(f"    Channels:        {am_diag['channels']}")
        print(f"    Energy thresh:   {am_diag['energy_threshold']:.1f}")
        print(f"    Buffer:          {am_diag['buffer_seconds']:.1f}s")
        print(f"    VAD state:       {am_diag['vad_state']}")

    print(f"  Backend:           {AUDIO_BACKEND or 'none'}")
    print(f"  HAS_SOUNDDEVICE:   {HAS_SOUNDDEVICE}")
    print(f"  HAS_SR:            {HAS_SPEECH_RECOGNITION}")
    print(f"  Silero VAD:        {'READY' if _silero_vad_available else 'NOT AVAILABLE'}")
    print(f"  openWakeWord:      {'READY' if _openwakeword_available else 'NOT AVAILABLE'}")
    print(f"  faster-whisper:    {'READY' if _whisper_available else 'NOT AVAILABLE'}")
    print("  ═══════════════════════════════════════════")
    print()


def _import_sr():
    """Lazy-import speech_recognition (done once, not in loop)."""
    global _recognizer_module, HAS_SPEECH_RECOGNITION
    if _recognizer_module is not None:
        return _recognizer_module
    try:
        _recognizer_module = __import__(
            "speech_recognition",
            fromlist=["Recognizer", "Microphone", "WaitTimeoutError",
                      "UnknownValueError", "RequestError"]
        )
        HAS_SPEECH_RECOGNITION = True
    except ImportError as e:
        logger.warning("speech_recognition not available: %s", e)
        _recognizer_module = False
        return None
    return _recognizer_module


def _get_recognizer():
    """Get or create the speech recognizer (singleton)."""
    global _recognizer
    sr = _import_sr()
    if sr is None:
        return None
    if _recognizer is None:
        _recognizer = sr.Recognizer()
        _recognizer.dynamic_energy_threshold = False
        _recognizer.energy_threshold = 300
        _recognizer.pause_threshold = 0.8
        _recognizer.non_speaking_duration = 0.5
        _recognizer.operation_timeout = 10.0
        logger.debug("Recognizer created (energy_threshold=300, pause_threshold=0.8)")
    return _recognizer


def _save_debug_audio(audio_bytes: bytes, samplerate: int, prefix: str = "wake"):
    """Save raw PCM16 audio to a WAV file in debug/."""
    try:
        import wave
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        filepath = DEBUG_DIR / filename
        with wave.open(str(filepath), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(samplerate)
            wav.writeframes(audio_bytes)
        logger.debug("[DEBUG] Saved raw audio: %s (%d bytes, %d Hz, %.1fs)",
                    filepath, len(audio_bytes), samplerate, len(audio_bytes) / samplerate / 2)
        return filepath
    except Exception as e:
        logger.debug("[DEBUG] Failed to save debug audio: %s", e)
        return None


def _save_failed_audio(audio_bytes: bytes, samplerate: int, reason: str) -> Optional[Path]:
    """
    Save failed wake audio with spectrogram for debugging.

    Saves:
      - raw.wav (processed audio)
      - spectrogram.png (frequency-time visualization)

    Args:
        audio_bytes: PCM16 audio bytes.
        samplerate: Sample rate.
        reason: Why wake detection failed.

    Returns:
        Path to the saved WAV file, or None on failure.
    """
    try:
        import wave
        import numpy as _np
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_reason = "".join(c for c in reason if c.isalnum() or c in "_-")[:30]
        base = f"wake_fail_{timestamp}_{safe_reason}"

        # Save WAV
        wav_path = DEBUG_DIR / f"{base}.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(samplerate)
            wav.writeframes(audio_bytes)

        # Save spectrogram
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy import signal as scipy_signal

            samples = _np.frombuffer(audio_bytes, dtype=_np.int16).astype(_np.float64) / 32768.0
            if len(samples) > 0:
                f, t, Sxx = scipy_signal.spectrogram(
                    samples, fs=samplerate, nperseg=256, noverlap=128
                )
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.pcolormesh(t, f, 10 * _np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_xlabel("Time (s)")
                ax.set_title(f"Failed Wake: {reason}")
                plt.tight_layout()
                png_path = DEBUG_DIR / f"{base}.png"
                plt.savefig(png_path, dpi=100)
                plt.close(fig)
        except Exception as e:
            logger.debug("[DEBUG] Spectrogram save failed: %s", e)

        logger.info("[DEBUG] Saved failed wake audio: %s (reason: %s)", wav_path, reason)
        return wav_path
    except Exception as e:
        logger.debug("[DEBUG] Failed to save failed wake audio: %s", e)
        return None


def _classify_failure_reason(samples: np.ndarray, samplerate: int) -> str:
    """
    Classify why STT/wake detection failed.

    Uses the SAME clipping threshold as the clipping detector
    (>= 32760) to ensure consistency.

    Args:
        samples: int16 PCM samples.
        samplerate: Sample rate.

    Returns:
        Reason string.
    """
    if len(samples) == 0:
        return "empty_audio"

    rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
    peak = float(np.max(np.abs(samples)))
    # Same threshold as the clipping detector in _recognize_bytes
    clipping_pct = float(np.mean(np.abs(samples) >= 32760) * 100)

    # Clipping: use the SAME percentage-based detector as the diagnostics
    if clipping_pct > 0.0:
        return f"clipping_{clipping_pct:.1f}pct"

    if rms < 300:
        return "too_quiet"

    # Check noise floor
    preproc_m = audio_preprocessor.get_metrics()
    nf_db = preproc_m.get("noise_floor_db", None)
    if nf_db is not None and nf_db < -50:
        return "noise_too_high"

    return "stt_failed"


def _recognize_bytes(audio_bytes: bytes, samplerate: int) -> Optional[str]:
    """
    Recognize speech from raw PCM16 bytes.

    PRIMARY: faster-whisper (offline)
    FALLBACK: Google Web Speech API (optional, requires internet)

    On failure, prints COMPLETE diagnostics:
    - Backend, recognizer, language
    - Duration, sample rate, channels, PCM dtype, bytes
    - Exception type and full traceback
    """
    # ── PRIMARY: faster-whisper (offline, no internet needed) ──
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(samples) == 0:
        logger.info("[STT] Empty audio — nothing to recognize (Google NOT invoked)")
        return None

    res = _whisper_transcribe_detailed(samples)
    if res["ok"]:
        logger.info("[STT] RECOGNIZED (whisper): '%s' (avg_logprob=%.3f no_speech=%.2f)",
                    res["text"], res["confidence"], res["no_speech"])
        return res["text"]

    # ── Google fallback policy (HARD RULE) ──
    # Google Web Speech API may ONLY execute when Whisper itself FAILED:
    #   - an exception during inference (reason starts with "exception:"), or
    #   - Whisper was never initialized ("whisper_unavailable").
    # A clean Whisper REJECTION (no_segments / empty_transcript /
    # high_no_speech_prob / empty_audio) means the audio contains no usable
    # speech — Google must NEVER run for those. Every rejection is logged
    # with transcript, avg_logprob, no_speech_prob and the reject reason.
    whisper_failed = (
        res["reason"].startswith("exception:")
        or res["reason"] == "whisper_unavailable"
    )
    if not whisper_failed:
        logger.info(
            "[STT] Whisper REJECTED — Google fallback SKIPPED. "
            "transcript=%r avg_logprob=%.3f no_speech_prob=%.2f reject_reason=%s",
            res["text"], res["confidence"], res["no_speech"], res["reason"])
        return None

    logger.warning(
        "[STT] Whisper FAILED (%s) — invoking Google fallback. "
        "transcript=%r avg_logprob=%.3f no_speech_prob=%.2f",
        res["reason"], res["text"], res["confidence"], res["no_speech"])

    # Fallback: Google Web Speech API (ONLY on Whisper exception/unavailable)
    sr = _import_sr()
    r = _get_recognizer()
    if sr is None or r is None:
        logger.warning("[STT] Speech recognition not available")
        return None

    if not isinstance(audio_bytes, bytes):
        logger.error("[STT] _recognize_bytes: expected bytes, got %s", type(audio_bytes).__name__)
        return None
    if not isinstance(samplerate, (int, float)):
        logger.error("[STT] _recognize_bytes: expected int samplerate, got %s", type(samplerate).__name__)
        return None
    if len(audio_bytes) < 256:
        logger.debug("[STT] Audio too short for recognition: %d bytes", len(audio_bytes))
        return None

    # Compute audio metrics for diagnostics
    duration = len(samples) / samplerate
    rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
    peak = float(np.max(np.abs(samples)))
    dc_offset = float(np.mean(samples.astype(float)))
    clipping_pct = float(np.mean(np.abs(samples) >= 32760) * 100)

    try:
        t0 = time.time()
        audio_data = sr.AudioData(audio_bytes, samplerate, 2)
        logger.debug("[STT] Google recognition started: %d bytes, %d Hz, %.1fs",
                    len(audio_bytes), samplerate, len(audio_bytes) / samplerate / 2)

        # Google Web Speech API (fallback)
        text = r.recognize_google(audio_data, language=settings.LANG_CODE)
        stt_time = time.time() - t0
        if text and text.strip():
            logger.info("[STT] RECOGNIZED (google): '%s' (%.1fs)", text.strip(), stt_time)
            return text.strip()
        logger.debug("[STT] Empty transcript (%.1fs)", stt_time)
        return None
    except sr.UnknownValueError:
        logger.warning(
            "[STT] FAILED — Google could not understand audio\n"
            "  Backend:      SpeechRecognition\n"
            "  Recognizer:   Google Web Speech API\n"
            "  Language:     %s\n"
            "  Duration:     %.2f sec\n"
            "  Sample rate:  %d Hz\n"
            "  Channels:     1\n"
            "  PCM dtype:    int16\n"
            "  Bytes:        %d\n"
            "  RMS:          %.1f\n"
            "  Peak:         %.0f\n"
            "  DC offset:    %.1f\n"
            "  Clipping:     %.1f%%\n"
            "  Exception:    UnknownValueError",
            settings.LANG_CODE, duration, samplerate, len(audio_bytes),
            rms, peak, dc_offset, clipping_pct,
        )
        return None
    except sr.RequestError as e:
        logger.warning(
            "[STT] FAILED — Google STT request error\n"
            "  Backend:      SpeechRecognition\n"
            "  Recognizer:   Google Web Speech API\n"
            "  Language:     %s\n"
            "  Duration:     %.2f sec\n"
            "  Sample rate:  %d Hz\n"
            "  Bytes:        %d\n"
            "  RMS:          %.1f\n"
            "  Peak:         %.0f\n"
            "  Exception:    RequestError: %s",
            settings.LANG_CODE, duration, samplerate, len(audio_bytes),
            rms, peak, e,
        )
        return None
    except Exception as e:
        logger.error(
            "[STT] FAILED — Unexpected error\n"
            "  Backend:      SpeechRecognition\n"
            "  Recognizer:   Google Web Speech API\n"
            "  Language:     %s\n"
            "  Duration:     %.2f sec\n"
            "  Sample rate:  %d Hz\n"
            "  Bytes:        %d\n"
            "  RMS:          %.1f\n"
            "  Peak:         %.0f\n"
            "  Exception:    %s: %s\n"
            "  Traceback:\n%s",
            settings.LANG_CODE, duration, samplerate, len(audio_bytes),
            rms, peak, type(e).__name__, e, traceback.format_exc(),
        )
        return None


def calibrate(duration: float = 1.5) -> None:
    """Calibrate the microphone for ambient noise. Called once at startup."""
    global _calibrated
    if _calibrated:
        return

    if audio_manager.is_running:
        audio_manager.calibrate(duration)
        _calibrated = True
    else:
        logger.warning("[CALIBRATE] AudioManager not running — skipping calibration")


# ═══════════════════════════════════════════════════════════════
# WAKE WORD LISTENING (Silero VAD + openWakeWord)
# ═══════════════════════════════════════════════════════════════

def listen_wake_continuous(timeout: Optional[float] = None) -> Optional[str]:
    """
    Continuously listen for wake word using Silero VAD + openWakeWord.

    This is a BLOCKING call that runs in an executor thread.
    It reads audio from the shared ring buffer, detects voice activity
    via Silero VAD, then checks for the wake word via openWakeWord.

    The loop:
      1. Read audio from the ring buffer
      2. Silero VAD detects voice activity
      3. openWakeWord checks for wake word
      4. If wake word detected, capture the audio and transcribe with Whisper
      5. Return the transcribed text

    Google SpeechRecognition is NOT used for wake detection.

    Args:
        timeout: Optional timeout in seconds (None = no timeout).

    Returns:
        Recognized text if wake word detected, or None on timeout/error.
    """
    if not audio_manager.is_running:
        logger.error("[WAKE] AudioManager not running — cannot listen for wake word")
        return None

    if not HAS_SOUNDDEVICE:
        logger.error("[WAKE] sounddevice not available - wake detection disabled")
        return None

    # Initialize offline wake detection components
    _init_silero_vad()
    _init_openwakeword()
    _init_whisper()

    if not _silero_vad_available:
        logger.error("[WAKE] Silero VAD not available — wake detection disabled")
        return None

    if not _openwakeword_available:
        logger.error("[WAKE] openWakeWord not available — wake detection disabled")
        return None

    logger.info("[WAKE] Wake listener active - Silero VAD + openWakeWord "
                "(streaming, model=%s, phrase='%s')",
                wake_model_manager.model_name, wake_model_manager.wake_phrase)

    start_time = time.time()
    samplerate = audio_manager.sample_rate
    threshold = wake_model_manager.threshold

    # VAD / segmentation parameters
    silence_duration = 0.8   # seconds of silence to end a speech segment
    min_speech_duration = 0.4  # minimum speech to report as a segment

    # Non-overlapping streaming state. openWakeWord's preprocessor keeps a
    # rolling feature buffer, so every sample must be fed EXACTLY ONCE.
    last_total = audio_manager.total_samples
    pending = np.zeros(0, dtype=np.int16)

    # Per-speech-segment diagnostics (reset at each new segment).
    is_speaking = False
    speech_start = 0.0
    last_voice = 0.0
    seg_max_oww_score = 0.0
    seg_max_oww_model = ""
    seg_max_vad = 0.0
    # Only ONE wake-verification (Whisper call) per speech segment —
    # prevents Whisper spam while the model stays above threshold.
    seg_wake_attempted = False

    from voice.wake_word import verify_wake_transcript

    while True:
        # Abort immediately on shutdown — no AudioManager access after this.
        if shutdown_event.is_set():
            logger.info("[WAKE] Shutdown requested — stopping wake listener")
            return None

        # Check timeout
        if timeout is not None and time.time() - start_time > timeout:
            logger.debug("[WAKE] Timeout reached (%.1fs)", timeout)
            return None

        # Read ONLY the new audio since the last iteration (non-overlapping).
        new_audio, last_total = audio_manager.read_since(last_total)
        if len(new_audio) == 0:
            time.sleep(0.01)
            continue

        pending = np.concatenate([pending, new_audio])

        # Feed complete 1280-sample (80ms) frames to openWakeWord streaming.
        while len(pending) >= WARMUP_FRAME_SAMPLES:
            frame = pending[:WARMUP_FRAME_SAMPLES]
            pending = pending[WARMUP_FRAME_SAMPLES:]

            now = time.time()

            # ── Silero VAD (gate) — 512-sample windows internally ──
            vad_conf = _silero_vad_confidence(frame)
            is_voice = vad_conf > 0.5

            # ── openWakeWord + custom verifier (streaming) ──
            preds = wake_model_manager.predict_stream(frame)
            best_model, best_score = "", 0.0
            for name, score in preds.items():
                if score > best_score:
                    best_model, best_score = name, score

            # ── Track per-segment diagnostics ──
            if is_voice:
                if not is_speaking:
                    is_speaking = True
                    speech_start = now
                    seg_max_oww_score = 0.0
                    seg_max_oww_model = ""
                    seg_max_vad = 0.0
                    seg_wake_attempted = False
                    logger.debug("[WAKE] Speech started (VAD=%.2f)", vad_conf)
                last_voice = now
                if best_score > seg_max_oww_score:
                    seg_max_oww_score = best_score
                    seg_max_oww_model = best_model
                if vad_conf > seg_max_vad:
                    seg_max_vad = vad_conf

            # ── WAKE DETECTION (two-authority gate) ──
            # PRIMARY: the wake MODEL must fire (score >= threshold) while
            #          VAD confirms speech.
            # SECONDARY: the Whisper transcript of the segment must VERIFY
            #          against the wake phrase. Fuzzy matching alone is NOT
            #          enough, and a missing/failed Whisper transcript is
            #          NEVER substituted with the wake phrase — that was the
            #          hole that let "Thank you very much" wake Leo.
            if is_voice and best_score >= threshold and not seg_wake_attempted:
                seg_wake_attempted = True
                logger.info(
                    "[WAKE] Model fired: '%s' score=%.3f (threshold=%.2f, VAD=%.2f)"
                    " — verifying transcript",
                    best_model, best_score, threshold, vad_conf)

                # Grab the recent ~2.5s of audio and transcribe with Whisper.
                full_audio = audio_manager.get_recent_processed(2.5)
                if len(full_audio):
                    whisper_res = _whisper_transcribe_detailed(full_audio)
                else:
                    whisper_res = {"text": None, "confidence": 0.0,
                                   "no_speech": 0.0, "ok": False,
                                   "reason": "empty_audio"}
                transcript = whisper_res.get("text")
                verified = verify_wake_transcript(transcript)

                # Record the full decision for downstream logging/metrics.
                wake_model_manager.last_detection = {
                    "model": best_model,
                    "score": best_score,
                    "transcript": transcript,
                    "verified": verified,
                    "whisper_reason": whisper_res.get("reason"),
                    "whisper_avg_logprob": whisper_res.get("confidence", 0.0),
                    "whisper_no_speech": whisper_res.get("no_speech", 0.0),
                    "vad_confidence": vad_conf,
                    "at": time.time(),
                }

                # Continuous streaming: do NOT reset the model here — the
                # feature window rolls off naturally and the prediction_buffer
                # stays primed (avoiding a 5-frame blind spot next session).
                pending = np.zeros(0, dtype=np.int16)

                if verified:
                    wake_model_manager._detections += 1
                    is_speaking = False
                    logger.info(
                        "[WAKE] VERIFIED WAKE: model='%s' score=%.3f "
                        "transcript='%s' (model + transcript agree)",
                        best_model, best_score, transcript)
                    return transcript

                # Model fired but the transcript does NOT confirm the wake
                # phrase → FALSE WAKE. Log the full decision, save the audio,
                # count it, and KEEP LISTENING — never return the wake phrase.
                wake_model_manager.record_false_positive()
                logger.warning(
                    "[WAKE] FALSE WAKE rejected: model='%s' score=%.3f transcript=%r "
                    "(whisper: reason=%s avg_logprob=%.3f no_speech=%.2f) "
                    "— continuing to listen",
                    best_model, best_score, transcript,
                    whisper_res.get("reason"), whisper_res.get("confidence", 0.0),
                    whisper_res.get("no_speech", 0.0))
                if len(full_audio):
                    try:
                        _save_failed_audio(
                            full_audio.astype(np.int16).tobytes(), SAMPLE_RATE,
                            f"false_wake_{best_score:.2f}")
                    except Exception as _e:
                        logger.debug("[WAKE] failed-audio save error: %s", _e)

            # ── Segment end WITHOUT wake → failure diagnostics ──
            if is_speaking and (now - last_voice) > silence_duration:
                duration = now - speech_start
                is_speaking = False

                if duration >= min_speech_duration:
                    # Speech occurred but wake never fired → FALSE REJECT.
                    # Report raw confidence, highest scoring model, VAD
                    # confidence, and speech duration WITHOUT terminating
                    # listening — the loop continues.
                    wake_model_manager.record_false_reject()
                    logger.warning(
                        "[WAKE] Wake NOT detected — continuing listening\n"
                        "        raw_confidence=%.4f (threshold=%.2f)\n"
                        "        highest_scoring_model='%s'\n"
                        "        vad_confidence=%.4f\n"
                        "        speech_duration=%.2fs",
                        seg_max_oww_score, threshold, seg_max_oww_model,
                        seg_max_vad, duration,
                    )
                seg_max_oww_score = 0.0
                seg_max_oww_model = ""
                seg_max_vad = 0.0

        time.sleep(0.005)  # 5ms polling


# ═══════════════════════════════════════════════════════════════
# COMMAND LISTENING (uses AudioManager ring buffer)
# ═══════════════════════════════════════════════════════════════

def listen(timeout: Optional[float] = None,
           phrase_time_limit: Optional[float] = None) -> Optional[str]:
    """
    Listen for a command using the AudioManager ring buffer.

    This does NOT open a new stream — it reads from the shared ring buffer
    and uses VAD to detect speech followed by silence.

    Args:
        timeout: Max seconds to wait for speech to start (default: 8.0).
        phrase_time_limit: Max seconds for a single phrase (default: 7.0).

    Returns:
        Recognized text, or None if no speech detected.
    """
    if timeout is None:
        timeout = 8.0
    if phrase_time_limit is None:
        phrase_time_limit = 7.0

    if not audio_manager.is_running:
        logger.error("[CMD] AudioManager not running — cannot listen")
        return None
    if shutdown_event.is_set():
        logger.info("[CMD] Shutdown in progress — listen() aborted")
        return None

    logger.info("[CMD] Command listening started - timeout=%.1fs phrase_limit=%.1fs",
               timeout, phrase_time_limit)

    # Record command from the shared audio stream
    audio_bytes = audio_manager.record_command(timeout=timeout, phrase_limit=phrase_time_limit)
    if audio_bytes is None:
        logger.warning("[CMD] No audio captured")
        return None

    samplerate = audio_manager.sample_rate

    # Apply noise suppression (NO AGC — unity gain only)
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(samples) > 0:
        processed = audio_preprocessor.process(samples)
        audio_bytes = processed.tobytes()

    # Check audio quality
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
    if rms < 5.0:
        logger.info("[CMD] Silence detected (RMS=%.1f < 5.0)", rms)
        return None
    if len(audio_bytes) < 512:
        logger.info("[CMD] Audio too short (%d bytes)", len(audio_bytes))
        return None

    # Recognize
    t0 = time.time()
    text = _recognize_bytes(audio_bytes, samplerate)
    stt_time = time.time() - t0

    if text:
        logger.info("[CMD] Command received: '%s' (STT took %.1fs)", text, stt_time)
    else:
        logger.info("[CMD] No speech recognized (STT took %.1fs)", stt_time)

    return text


def listen_wake(timeout: Optional[float] = None,
                phrase_time_limit: float = 5.0) -> Optional[str]:
    """
    Listen for wake word using Silero VAD + openWakeWord.

    Entry point called from main loop executor.
    This is a BLOCKING call that runs in an executor thread.

    Args:
        timeout: Optional timeout in seconds.
        phrase_time_limit: Unused (kept for API compatibility).

    Returns:
        Recognized text if wake word detected, or None.
    """
    if not HAS_SOUNDDEVICE:
        logger.error("[WAKE] sounddevice not available - wake detection disabled")
        return None

    if not audio_manager.is_running:
        logger.error("[WAKE] AudioManager not running - wake detection disabled")
        return None

    logger.debug("[WAKE] Listen wake initiated - Silero VAD + openWakeWord")
    text = listen_wake_continuous(timeout)
    if text:
        logger.debug("[WAKE] Wake detected, returning to main loop")
    else:
        logger.debug("[WAKE] Wake listener interrupted")
    return text


def _capture_audio_sounddevice(duration: float = 1.0) -> Optional[Tuple[bytes, int]]:
    """
    LEGACY COMPATIBILITY: Capture audio from the AudioManager ring buffer.

    This function exists for backward compatibility with tests that
    reference the old sounddevice capture API. It reads from the
    AudioManager's shared ring buffer instead of opening a new stream.

    Args:
        duration: Seconds of audio to capture.

    Returns:
        Tuple of (audio_bytes, samplerate), or None if no audio available.
    """
    if not audio_manager.is_running:
        logger.warning("[CAPTURE] AudioManager not running — cannot capture")
        return None

    audio_bytes = audio_manager.capture_duration(duration)
    if audio_bytes is None:
        return None

    return audio_bytes, audio_manager.sample_rate