"""
Speech-to-Text module for Leo Desktop Assistant.

ARCHITECTURE:
  Continuous streaming for wake word detection.
  Wake listener NEVER terminates unless the application exits.
  After wake detection, microphone ownership switches to command recognition.
  Command recognition: 8s timeout, 12s phrase limit.
  After command processing, returns to wake mode.

AUDIO BACKEND:
  Detected ONCE at module load. Never re-imported inside loops.
  Priority: sounddevice → speech_recognition/PyAudio → none

BACKEND INITIALIZATION:
  If sounddevice fails to import, the FULL traceback is logged.
  ImportError, ModuleNotFoundError, OSError, and PortAudioError are
  never silently swallowed. Every failure is captured with complete traceback.
"""

import logging
import os
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config.settings import settings
from voice.microphone import microphone

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
    # Log immediately — this is a critical diagnostic
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

# Try speech_recognition (for PyAudio path)
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

# ── Continuous Wake Streaming ────────────────────────────────
_wake_stream: Optional[any] = None
_wake_stream_lock = threading.Lock()
_wake_audio_buffer = []
_wake_is_speaking = False
_wake_speech_start = 0.0
_wake_last_audio = 0.0
_wake_pause_threshold = 0.8
_wake_energy_threshold = 300
_wake_min_audio_duration = 0.5
_wake_max_audio_duration = 5.0


def get_backend_diagnostics() -> dict:
    """
    Return a detailed diagnostics report for the audio backend.
    
    This is called at startup to print the state of the audio subsystem.
    """
    diag = {
        "has_sounddevice": HAS_SOUNDDEVICE,
        "has_speech_recognition": HAS_SPEECH_RECOGNITION,
        "audio_backend": AUDIO_BACKEND,
        "backend_init_error": _BACKEND_INIT_ERROR,
        "python_executable": sys.executable,
        "python_version": sys.version,
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
    # Check virtualenv
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
    
    # Selected microphone
    try:
        mic = microphone.get_microphone()
        if mic is not None:
            print(f"  Selected mic:      {mic}")
            if hasattr(mic, 'device_index') and mic.device_index is not None:
                print(f"  Device index:      {mic.device_index}")
            if hasattr(mic, 'sample_rate') and mic.sample_rate is not None:
                print(f"  Sample rate:       {mic.sample_rate}")
            if hasattr(mic, 'channels'):
                print(f"  Channels:          {mic.channels}")
        else:
            print("  Selected mic:      NONE")
    except Exception as e:
        print(f"  Selected mic:      ERROR - {e}")
    
    print(f"  Backend:           {AUDIO_BACKEND or 'none'}")
    print(f"  HAS_SOUNDDEVICE:   {HAS_SOUNDDEVICE}")
    print(f"  HAS_SR:            {HAS_SPEECH_RECOGNITION}")
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


def _enumerate_microphones() -> list:
    """Enumerate all available microphones."""
    mics = []
    if HAS_SOUNDDEVICE and sd is not None:
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    mics.append({
                        'index': i, 'name': d['name'],
                        'channels': d['max_input_channels'],
                        'samplerate': d['default_samplerate'],
                    })
                    logger.debug("[MIC] device[%d]: %s (%d ch, %.0f Hz)",
                               i, d['name'], d['max_input_channels'], d['default_samplerate'])
        except Exception as e:
            logger.debug("[MIC] sounddevice enumeration failed: %s", e)

    sr = _import_sr()
    if sr and hasattr(sr, 'Microphone') and hasattr(sr.Microphone, 'list_microphone_names'):
        try:
            for i, name in enumerate(sr.Microphone.list_microphone_names()):
                if name:
                    mics.append({'index': i, 'name': name, 'source': 'pyaudio'})
                    logger.debug("[MIC] PyAudio device[%d]: %s", i, name)
        except Exception:
            pass

    if not mics:
        logger.warning("[MIC] No microphones found on this system")
    else:
        logger.info("[MIC] Found %d microphone(s): %s", len(mics),
                   [m['name'] for m in mics])
    return mics


def _get_device_index() -> Optional[int]:
    """Get the correct microphone device index (uses module-level sd)."""
    if not HAS_SOUNDDEVICE or sd is None:
        return None
    if settings.WAKE_DEVICE_INDEX is not None:
        return settings.WAKE_DEVICE_INDEX
    if (microphone._mic is not None and
        hasattr(microphone._mic, 'device_index') and
        microphone._mic.device_index is not None):
        return microphone._mic.device_index
    try:
        return sd.default.device[0]
    except Exception:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                return i
    return None


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


def _capture_audio_sounddevice(duration: float, samplerate: int = 16000) -> Optional[Tuple[bytes, int]]:
    """
    Capture raw audio using sounddevice (uses module-level sd, never re-imports).
    Returns (audio_bytes, actual_samplerate) tuple.
    """
    if not HAS_SOUNDDEVICE or sd is None:
        logger.error("[STT] sounddevice not available")
        return None

    try:
        device_idx = _get_device_index()
        if device_idx is None:
            logger.warning("[STT] No input device found")
            return None

        device_info = sd.query_devices(device_idx)
        actual_sr = int(device_info['default_samplerate']) if device_info['default_samplerate'] else samplerate
        samples = int(duration * actual_sr)

        t0 = time.time()
        recording = sd.rec(samples, samplerate=actual_sr, channels=1,
                          dtype='int16', device=device_idx, blocking=True)
        capture_time = time.time() - t0
        audio_bytes = recording.tobytes()

        if len(recording) > 0:
            rms = float(np.sqrt(np.mean(recording.astype(float)**2)))
            peak = float(np.max(np.abs(recording)))
            logger.debug("[STT] Audio metrics: RMS=%.1f, peak=%d, frames=%d, bytes=%d (%.2fs)",
                        rms, int(peak), len(recording), len(audio_bytes), capture_time)

        return (audio_bytes, actual_sr)
    except Exception as e:
        logger.error("[STT] sounddevice capture failed: %s", e, exc_info=True)
        return None


def _recognize_bytes(audio_bytes: bytes, samplerate: int) -> Optional[str]:
    """
    Recognize speech from raw PCM16 bytes using Google STT.
    """
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

    try:
        t0 = time.time()
        import speech_recognition as sr_module
        audio_data = sr_module.AudioData(audio_bytes, samplerate, 2)
        logger.debug("[STT] Recognition started: %d bytes, %d Hz, %.1fs",
                    len(audio_bytes), samplerate, len(audio_bytes) / samplerate / 2)
        text = r.recognize_google(audio_data, language=settings.LANG_CODE)
        stt_time = time.time() - t0
        if text and text.strip():
            logger.info("[STT] RECOGNIZED: '%s' (%.1fs)", text.strip(), stt_time)
            return text.strip()
        logger.debug("[STT] Empty transcript (%.1fs)", stt_time)
        return None
    except sr.UnknownValueError:
        logger.debug("[STT] Google could not understand audio (%d bytes, %d Hz, %.1fs)",
                    len(audio_bytes), samplerate, len(audio_bytes) / samplerate / 2)
        return None
    except sr.RequestError as e:
        logger.warning("[STT] Google STT request error: %s", e)
        return None
    except Exception as e:
        logger.error("[STT] Recognition error: %s", e, exc_info=True)
        return None


def calibrate(duration: float = 1.5) -> None:
    """Calibrate the microphone for ambient noise. Called once at startup."""
    global _calibrated
    if _calibrated:
        return
    _enumerate_microphones()
    r = _get_recognizer()
    if r is None:
        logger.warning("Speech recognition not available, skipping calibration")
        return
    if microphone.backend == 'sounddevice' and HAS_SOUNDDEVICE:
        try:
            logger.info("[CALIBRATE] Measuring ambient noise via sounddevice...")
            result = _capture_audio_sounddevice(duration)
            if result is None:
                return
            audio_bytes, actual_sr = result
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(samples) == 0:
                return
            rms = float(np.sqrt(np.mean(samples.astype(float)**2)))
            r.energy_threshold = max(300, rms * 1.5)
            global _wake_energy_threshold
            _wake_energy_threshold = r.energy_threshold
            logger.info("[CALIBRATE] Ambient RMS=%.1f, samples=%d, energy_threshold=%.1f",
                      rms, len(samples), r.energy_threshold)
            _calibrated = True
            return
        except Exception as e:
            logger.warning("[CALIBRATE] sounddevice calibration failed: %s", e)
            return
    _calibrated = True


# ═══════════════════════════════════════════════════════════════
# CONTINUOUS WAKE WORD STREAMING
# ═══════════════════════════════════════════════════════════════

def _wake_audio_callback(indata, frames, time_info, status):
    """Callback for sd.InputStream - processes audio frames in real-time."""
    global _wake_audio_buffer, _wake_is_speaking, _wake_last_audio
    if status:
        logger.debug("[WAKE] Stream status: %s", status)
    audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
    chunk_bytes = audio_chunk.tobytes()
    rms = float(np.sqrt(np.mean(audio_chunk.astype(float)**2)))
    now = time.time()
    if rms > _wake_energy_threshold:
        if not _wake_is_speaking:
            _wake_is_speaking = True
            _wake_audio_buffer = []
            logger.debug("[WAKE] Speech started (RMS=%.1f > threshold=%.1f)", rms, _wake_energy_threshold)
        _wake_last_audio = now
        _wake_audio_buffer.append(chunk_bytes)
    else:
        if _wake_is_speaking:
            if now - _wake_last_audio > _wake_pause_threshold:
                _wake_is_speaking = False
                logger.debug("[WAKE] Speech ended (silence=%.2fs, buffer=%d chunks)",
                           now - _wake_last_audio, len(_wake_audio_buffer))
            else:
                _wake_audio_buffer.append(chunk_bytes)


def listen_wake_continuous(timeout: Optional[float] = None) -> Optional[str]:
    """
    Continuously listen for wake word using streaming audio.
    Uses module-level sd import (never re-imports in loop).
    """
    global _wake_stream, _wake_audio_buffer, _wake_is_speaking

    if not HAS_SOUNDDEVICE or sd is None:
        logger.error("[WAKE] sounddevice not available - cannot start wake listener")
        return None

    device_idx = _get_device_index()
    if device_idx is None:
        logger.error("[WAKE] No microphone device available")
        return None

    device_info = sd.query_devices(device_idx)
    actual_sr = int(device_info['default_samplerate']) if device_info['default_samplerate'] else 16000
    logger.info("[WAKE] Wake listener active - device[%d]: %s (%d Hz, %d ch)",
               device_idx, device_info['name'], actual_sr, device_info['max_input_channels'])

    try:
        with sd.InputStream(
            samplerate=actual_sr, device=device_idx, channels=1,
            dtype='float32', callback=_wake_audio_callback, blocksize=1024,
        ):
            logger.info("[WAKE] Continuous streaming started - waiting for wake word...")
            while True:
                time.sleep(0.05)
                if not _wake_is_speaking and len(_wake_audio_buffer) > 0:
                    audio_bytes = b''.join(_wake_audio_buffer)
                    duration = len(audio_bytes) / actual_sr / 2
                    if duration < _wake_min_audio_duration:
                        _wake_audio_buffer = []
                        continue
                    if duration > _wake_max_audio_duration:
                        max_bytes = int(_wake_max_audio_duration * actual_sr * 2)
                        audio_bytes = audio_bytes[:max_bytes]
                    _wake_audio_buffer = []
                    _save_debug_audio(audio_bytes, actual_sr, "wake")
                    logger.debug("[WAKE] Audio frames received: %d bytes, %.2fs", len(audio_bytes), duration)
                    t0 = time.time()
                    text = _recognize_bytes(audio_bytes, actual_sr)
                    stt_time = time.time() - t0
                    if text:
                        from voice.wake_word import wake_word_engine
                        import difflib
                        for v in wake_word_engine._variants:
                            ratio = difflib.SequenceMatcher(None, text.lower(), v.lower()).ratio()
                            logger.debug("[WAKE]   vs '%s' similarity=%.3f threshold=0.85", v, ratio)
                        if wake_word_engine.detect(text):
                            logger.info("[WAKE] WAKE DETECTED: '%s' (STT=%.1fs)", text, stt_time)
                            return text
                    else:
                        logger.debug("[WAKE] No speech recognized (STT=%.1fs)", stt_time)
    except KeyboardInterrupt:
        logger.info("[WAKE] Wake listener interrupted")
    except Exception as e:
        logger.error("[WAKE] Wake stream error: %s", e, exc_info=True)
    finally:
        _wake_stream = None
        _wake_audio_buffer = []
        _wake_is_speaking = False
    return None


# ═══════════════════════════════════════════════════════════════
# COMMAND LISTENING
# ═══════════════════════════════════════════════════════════════

def listen(timeout: Optional[float] = None,
           phrase_time_limit: Optional[float] = None) -> Optional[str]:
    """Listen for a command. Timeout: 8s, Phrase limit: 12s."""
    if timeout is None:
        timeout = 8.0
    if phrase_time_limit is None:
        phrase_time_limit = 12.0

    logger.info("[CMD] Command listening started - timeout=%.1fs phrase_limit=%.1fs backend=%s",
               timeout, phrase_time_limit, microphone.backend)

    if microphone.backend == 'sounddevice' and HAS_SOUNDDEVICE:
        t0 = time.time()
        result = _capture_audio_sounddevice(phrase_time_limit)
        capture_time = time.time() - t0
        if result is None:
            logger.warning("[CMD] No audio captured (%.1fs)", capture_time)
            return None
        audio_bytes, samplerate = result
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(float)**2)))
        if rms < 5.0:
            logger.info("[CMD] Silence detected (RMS=%.1f < 5.0)", rms)
            return None
        if len(audio_bytes) < 512:
            logger.info("[CMD] Audio too short (%d bytes) — treating as silence", len(audio_bytes))
            return None
        t1 = time.time()
        text = _recognize_bytes(audio_bytes, samplerate)
        stt_time = time.time() - t1
        if text:
            logger.info("[CMD] Command received: '%s' (STT took %.1fs)", text, stt_time)
        else:
            logger.info("[CMD] No speech recognized (STT took %.1fs)", stt_time)
        return text

    # PyAudio fallback
    sr = _import_sr()
    r = _get_recognizer()
    mic = microphone.get_microphone()
    if r is None or mic is None or sr is None:
        logger.warning("[CMD] Speech recognition not available")
        return None
    try:
        with mic as source:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        logger.info("[CMD] Listening timed out (energy=%.1f, timeout=%.1fs)", r.energy_threshold, timeout)
        return None
    except Exception as e:
        logger.error("[CMD] Microphone error: %s", e)
        return None
    try:
        t0 = time.time()
        text = r.recognize_google(audio, language=settings.LANG_CODE)
        stt_time = time.time() - t0
        if text and text.strip():
            logger.info("[CMD] Command received: '%s' (STT took %.1fs)", text.strip(), stt_time)
            return text.strip()
        return None
    except sr.UnknownValueError:
        logger.info("[CMD] Could not understand audio (energy=%.1f)", r.energy_threshold)
        return None
    except sr.RequestError as e:
        logger.error("[CMD] Recognition service error: %s", e)
        return None
    except Exception as e:
        logger.error("[CMD] Recognition error: %s", e)
        return None


def listen_wake(timeout: Optional[float] = None,
                phrase_time_limit: float = 5.0) -> Optional[str]:
    """
    Listen for wake word using continuous streaming.
    Entry point called from main loop executor.
    """
    if not HAS_SOUNDDEVICE:
        logger.error("[WAKE] sounddevice not available - wake detection disabled")
        return None
    logger.debug("[WAKE] Listen wake initiated - continuous streaming")
    text = listen_wake_continuous(timeout)
    if text:
        logger.debug("[WAKE] Wake detected, returning to main loop")
    else:
        logger.debug("[WAKE] Wake listener interrupted")
    return text