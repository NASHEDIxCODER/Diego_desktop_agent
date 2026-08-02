"""
Leo Desktop Assistant — Production Entry Point

Architecture:
  Wake word → Face auth → Listen → STT → [PendingAction?] → NLP → Plugin → TTS

Startup diagnostics:
  Each subsystem reports READY, DEGRADED, DISABLED, or FAILED.
  Assistant starts only after all subsystem states are finalized.

Commands:
  python main.py              # Normal startup
  python main.py --train      # Train NLP model
  python main.py --status     # Show model status
  python main.py --benchmark  # Run NLP benchmarks

LLMs are OPTIONAL providers used only for unknown intents.
Startup NEVER retrains. Model must be pre-trained with --train.
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT-LEVEL FIXES (must happen before any imports)
# ═══════════════════════════════════════════════════════════════

# ── Fix 1: Suppress ALL ALSA/JACK/PulseAudio library noise ──
os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_OUTPUT_FORMAT"] = "0"
os.environ["ALSA_DEBUG"] = "0"
os.environ["ALSA_DEBUG_FILE"] = "/dev/null"
os.environ["PYTTXS3_ALSA_DEBUG"] = "0"
os.environ["SPEECH_RECOGNITION_ALSA_DEBUG"] = "0"
os.environ["DISPLAY_ALSA_OUTPUT"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["PULSE_LOG_LEVEL"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"
os.environ["JACK_NO_START_SERVER"] = "1"

# ── Fix 1b: Redirect stderr to /dev/null during third-party imports ──
# This catches ALSA/JACK/PulseAudio C library noise from cv2, face_recognition, etc.
_import_stderr_fd = os.dup(2)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)
os.close(_devnull_fd)
_import_stderr_restore = True

# ── Fix 2: Qt Font Configuration ──
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")

# ── Fix 3: HuggingFace offline-friendly settings ──
# The embedding model is cached locally. Normal startup must NOT contact
# HuggingFace. Only the first download (when model is not cached) may use
# the network. We set offline mode BEFORE any model imports happen.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
# Check if the embedding model is already cached locally
# sentence-transformers adds the "sentence-transformers/" prefix to model names
_hf_home = os.environ.get("HF_HOME") or os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache/huggingface")
_model_name = "all-MiniLM-L6-v2"
_model_cache_candidates = [
    os.path.join(_hf_home, "hub", f"models--{_model_name.replace('/', '--')}", "snapshots"),
    os.path.join(_hf_home, "hub", f"models--sentence-transformers--{_model_name.replace('/', '--')}", "snapshots"),
]
_model_is_cached = any(
    os.path.isdir(p) and bool(os.listdir(p))
    for p in _model_cache_candidates
)
if _model_is_cached:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("  [HF] Embedding model cached locally — HuggingFace offline mode enabled")
else:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

# ── Fix 4: Disable noisy TensorFlow/Keras warnings ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Fix 5: Ensure DISPLAY exists ──
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

try:
    subprocess.run(["xhost", "+local:"], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
except Exception:
    pass

# ── Python 3.14 compatibility: inject removed stdlib modules ──
import compat  # noqa: F401

from config.settings import settings
from telemetry.logger import setup_logging, set_correlation_id, set_subsystem_id

setup_logging()
logger = logging.getLogger(__name__)
set_subsystem_id("main")

# ── Core imports ──────────────────────────────────────────
from core.event_bus import bus, Event
from core.plugin_manager import plugin_manager

# ── Startup Health ────────────────────────────────────────
from core.startup_health import startup_health, SubsystemState

# ── NLP imports ───────────────────────────────────────────
from nlp.inference import inference
from nlp.model_metadata import is_model_ready, get_model_status
from nlp.context import context_manager
from nlp.entities import extract_entities
from nlp.trainer import trainer
from nlp.conversation_state import conversation_state, PendingAction

# ── AI imports ────────────────────────────────────────────
from ai.llm_client import llm_client, llm_chat

# ── Agent imports ─────────────────────────────────────────
from agent.planner import agent_planner
from agent.executor import agent_executor
from agent.browser import browser_controller

# ── Voice imports ─────────────────────────────────────────
# NOTE: voice.recognizer is DEPRECATED. All STT is in voice.stt.
from voice.supervisor import VoiceSupervisor, VoiceState, voice_supervisor
from voice.settings import voice_settings
from voice.synthesizer import speech_synthesizer
from voice.wake_word import wake_word_engine
from voice.microphone import microphone
from voice.noise import noise_calibrator
from voice.audio_device import audio_device
from voice.tts.manager import tts_manager
from voice.stt import get_backend_diagnostics, print_startup_diagnostics
from voice.audio_manager import audio_manager
from voice.wake_model_manager import wake_model_manager
from voice.wake_word import verify_wake_transcript

# ── Pending voice futures (wake workers / command workers / auth workers) ──
# run_in_executor futures are tracked here so shutdown can CANCEL every
# pending listen() before the AudioManager is destroyed. Futures that are
# already running exit via the global shutdown_event.
_PENDING_VOICE_FUTURES: set = set()


def _track_voice_future(fut) -> None:
    """Register an executor future for lifecycle management."""
    _PENDING_VOICE_FUTURES.add(fut)
    fut.add_done_callback(lambda f: _PENDING_VOICE_FUTURES.discard(f))

# ── Embeddings: preload SentenceTransformer at startup ───
from nlp.embeddings import preload_embedding_model

# ── Restore stderr after all third-party imports ──────────
if _import_stderr_restore:
    os.dup2(_import_stderr_fd, 2)
    os.close(_import_stderr_fd)
    _import_stderr_restore = False

# ── Wake word variants ────────────────────────────────────
WAKE_VARIANTS = [
    "hello leo",
    "leo",
    "lio",
    "hey leo",
    "hello lio",
]


def _get_voice():
    """Lazy-import voice module (deferred for Python 3.14 aifc compat)."""
    from voice.stt import calibrate, listen, listen_wake
    from voice.synthesizer import speech_synthesizer
    return calibrate, listen, listen_wake, speech_synthesizer.speak


async def handle_intent(parsed: dict) -> str:
    """Handle a classified intent. Returns response text or empty string."""
    intent = parsed.get("intent", "unknown")
    entities = parsed.get("entities", {})
    confidence = parsed.get("confidence", 0.0)

    if intent == "greeting":
        speech_synthesizer.speak("Hello! How can I help you?")
        return ""

    if intent == "exit":
        speech_synthesizer.speak("Goodbye, have a nice day.")
        return "__EXIT__"

    # ── Intent → Event mapping for plugin-bound intents ─────
    if intent == "youtube":
        await bus.emit("youtube_open", data=entities, source="nlp")
        return ""

    if intent.startswith("youtube_"):
        await bus.emit(intent, data=entities, source="nlp")
        return ""

    if intent == "telegram_read":
        await bus.emit("telegram_read", data=entities, source="nlp")
        return ""

    if intent == "telegram_reply":
        await bus.emit("telegram_reply", data=entities, source="nlp")
        return ""

    if intent == "telegram_send":
        await bus.emit("telegram_send", data=entities, source="nlp")
        return ""

    if intent == "brightness_up":
        await bus.emit("brightness_up", data=entities, source="nlp")
        return ""

    if intent == "brightness_down":
        await bus.emit("brightness_down", data=entities, source="nlp")
        return ""

    if intent == "brightness_set":
        await bus.emit("brightness_set", data=entities, source="nlp")
        return ""

    if intent == "time_query":
        import datetime
        now = datetime.datetime.now()
        speech_synthesizer.speak(f"The time is {now.strftime('%I:%M %p')}.")
        return ""

    if intent == "date_query":
        import datetime
        now = datetime.datetime.now()
        speech_synthesizer.speak(f"Today is {now.strftime('%A, %B %d, %Y')}.")
        return ""

    if intent == "help":
        speech_synthesizer.speak(
            "I can control YouTube, send Telegram messages, "
            "adjust brightness, tell the time and date, "
            "and have conversations. What would you like to do?"
        )
        return ""

    if intent == "joke":
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the AI break up with the database? Too many relationships!",
            "What do you call a fake noodle? An impasta!",
        ]
        import random
        speech_synthesizer.speak(random.choice(jokes))
        return ""

    if intent == "who_am_i":
        speech_synthesizer.speak("You are my user. I recognize you by your face.")
        return ""

    if intent == "unknown" or confidence < settings.SIMILARITY_THRESHOLD * 0.8:
        response = await llm_chat(parsed.get("text", ""))
        speech_synthesizer.speak(response)
        return ""

    response = await llm_chat(parsed.get("text", ""))
    speech_synthesizer.speak(response)
    return ""


async def _init_nlp(status: dict) -> None:
    """Initialize NLP subsystem."""
    startup_health.register("nlp")
    if inference.load():
        s = inference.get_status()
        status["nlp"] = True
        startup_health.set_state(
            "nlp", SubsystemState.READY,
            f"v{s.get('version', '?')}, {s.get('intents', 0)} intents, "
            f"{s.get('examples', 0)} examples",
        )
    else:
        startup_health.set_state(
            "nlp", SubsystemState.FAILED,
            "Model not found. Run: python main.py --train",
        )


async def _init_embeddings(status: dict) -> None:
    """Preload embedding model — runs in executor to avoid blocking event loop."""
    startup_health.register("embeddings")
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, preload_embedding_model)
        startup_health.set_state("embeddings", SubsystemState.READY,
                                 f"Model: {settings.MODEL_NAME}")
    except Exception as e:
        logger.error("Failed to preload embedding model: %s", e)
        startup_health.set_state("embeddings", SubsystemState.FAILED, str(e))
        status["nlp"] = False


async def _init_tts(status: dict) -> None:
    """Preload TTS engine."""
    startup_health.register("tts")
    try:
        speech_synthesizer.initialize()
        startup_health.set_state("tts", SubsystemState.READY, "TTS initialized")
    except Exception as e:
        logger.warning("TTS preload failed: %s", e)
        startup_health.set_state("tts", SubsystemState.DEGRADED, str(e))


async def _init_duckdb(status: dict) -> None:
    """Initialize DuckDB storage."""
    startup_health.register("duckdb")
    try:
        from memory.duckdb_store import store, DatabaseLockedError
        try:
            store.initialize()
            if store._conn is not None:
                status["duckdb"] = True
                startup_health.set_state("duckdb", SubsystemState.READY, "Connected")
            else:
                status["degraded"] = True
                startup_health.set_state("duckdb", SubsystemState.DEGRADED, "Locked")
        except DatabaseLockedError:
            status["degraded"] = True
            startup_health.set_state("duckdb", SubsystemState.DEGRADED, "Locked")
    except ImportError:
        startup_health.set_state("duckdb", SubsystemState.DISABLED, "Not installed")
    except Exception as e:
        status["degraded"] = True
        startup_health.set_state("duckdb", SubsystemState.DEGRADED, str(e))


async def _init_plugins(status: dict) -> None:
    """Initialize all plugins."""
    startup_health.register("plugins")
    try:
        await plugin_manager.load_all()
        await plugin_manager.initialize_all()
        names = list(plugin_manager.plugins.keys())
        if names:
            status["plugins"] = True
            failed = [n for n in names if not plugin_manager.plugins[n].enabled]
            if failed:
                status["degraded"] = True
                startup_health.set_state(
                    "plugins", SubsystemState.DEGRADED,
                    f"{len(names)} loaded, {len(failed)} disabled",
                )
            else:
                startup_health.set_state(
                    "plugins", SubsystemState.READY,
                    f"{len(names)} loaded",
                )
        else:
            status["plugins"] = True
            startup_health.set_state("plugins", SubsystemState.READY, "None discovered")
    except Exception as e:
        status["plugins"] = False
        startup_health.set_state("plugins", SubsystemState.FAILED, str(e))


async def _init_voice(status: dict) -> None:
    """Initialize voice subsystem — starts unified AudioManager."""
    startup_health.register("voice")
    try:
        voice_settings.update_from_env()
        audio_device.detect_backend()

        # Start the unified AudioManager (opens ONE InputStream for the entire session)
        loop = asyncio.get_running_loop()
        am_started = await loop.run_in_executor(None, audio_manager.start)

        if not am_started:
            status["voice"] = False
            startup_health.set_state("voice", SubsystemState.DISABLED,
                                     "AudioManager failed to start")
            return

        # Calibrate ambient noise — EXACTLY ONCE after stream has stabilized.
        # Wait for the ring buffer to fill with stable audio before measuring.
        await asyncio.sleep(0.5)  # Let the stream stabilize
        await loop.run_in_executor(None, audio_manager.calibrate, 1.5)

        # Pre-seed the noise suppression profile by processing ambient audio.
        # This ensures spectral gating has a noise reference BEFORE any
        # wake word detection begins.
        try:
            await loop.run_in_executor(None, audio_manager.get_recent_processed, 1.0)
        except Exception as e:
            logger.debug("[AUDIO] Noise profile pre-seed: %s", e)

        # AudioManager owns the microphone — do NOT call microphone.get_microphone()
        # which would open a second PyAudio stream. All STT reads from the
        # AudioManager ring buffer instead.
        noise_calibrator.load_profile()

        # ── Initialize ALL offline wake detection components NOW ──
        # Silero VAD, openWakeWord (+ verifier), and faster-whisper are
        # loaded eagerly here so the startup health report below reflects
        # the REAL runtime state instead of a race-condition false
        # "NOT AVAILABLE".
        from voice.stt import init_wake_detection
        wake_init = await loop.run_in_executor(None, init_wake_detection)

        # Register the wake-model subsystem with the true runtime state
        from voice.wake_model_manager import wake_model_manager as _wmm
        startup_health.register("wake_model")
        if _wmm.verify_model_exists() and _wmm.loaded:
            name = _wmm.model_name or "?"
            verifier = "custom verifier" if _wmm.verifier_path else "no verifier"
            startup_health.set_state(
                "wake_model", SubsystemState.READY,
                f"model={name}, phrase='{_wmm.wake_phrase}', {verifier}",
            )
        elif _wmm.verify_model_exists():
            status["degraded"] = True
            startup_health.set_state(
                "wake_model", SubsystemState.DEGRADED,
                f"model found but failed to load: {_wmm.load_error or 'unknown error'}",
            )
        else:
            status["degraded"] = True
            startup_health.set_state(
                "wake_model", SubsystemState.DEGRADED,
                "no wake model found — run: python main.py --train-wake",
            )

        status["voice"] = True
        startup_health.set_state("voice", SubsystemState.READY,
                                 f"AudioManager running (backend={audio_manager.backend})")
    except Exception as e:
        status["voice"] = False
        startup_health.set_state("voice", SubsystemState.DISABLED, str(e))


async def _init_telegram(status: dict) -> None:
    """Initialize Telegram client — non-blocking, checks session file only."""
    startup_health.register("telegram")
    try:
        from scripts.telegram_bot import _available as tg_available
        # Check .env for Telegram credentials
        import os as _os
        from pathlib import Path as _Path
        api_id = _os.environ.get("TELEGRAM_API_ID", "")
        api_hash = _os.environ.get("TELEGRAM_API_HASH", "")

        missing = []
        if not api_id:
            missing.append("TELEGRAM_API_ID")
        if not api_hash:
            missing.append("TELEGRAM_API_HASH")

        if missing:
            status["degraded"] = True
            startup_health.set_state(
                "telegram", SubsystemState.DEGRADED,
                f"Missing {' and '.join(missing)}",
            )
            return

        # Quick check: if no session file exists, disable immediately
        session_files = [
            _Path("leo_telegram.session"),
            _Path("leo_telegram.session-journal"),
        ]
        has_session = any(f.exists() for f in session_files)
        if not has_session:
            startup_health.set_state("telegram", SubsystemState.DISABLED,
                                     "No session — run interactively once")
            return

        # Defer actual connection to background (handled by TelegramPlugin)
        startup_health.set_state("telegram", SubsystemState.READY, "Session found")
    except ImportError:
        startup_health.set_state("telegram", SubsystemState.DISABLED, "Not installed")
    except Exception as e:
        startup_health.set_state("telegram", SubsystemState.DISABLED, str(e))


async def startup_diagnostics() -> dict:
    """Run startup diagnostics using StartupHealth with parallel initialization."""
    status = {
        "nlp": False,
        "voice": False,
        "plugins": False,
        "duckdb": False,
        "telegram": False,
        "face_auth": False,
        "vision": False,
        "degraded": False,
    }

    # Run all independent subsystems in parallel
    tasks = [
        _init_nlp(status),
        _init_embeddings(status),
        _init_tts(status),
        _init_duckdb(status),
        _init_plugins(status),
        _init_voice(status),
        _init_telegram(status),
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    # Check if NLP failed (core dependency)
    if not status["nlp"]:
        nlp_state = startup_health.get_state("nlp")
        if nlp_state and nlp_state.state == SubsystemState.FAILED:
            print()
            print("  NLP model not found. Run: python main.py --train")
            print()

    return status


async def main_loop():
    """Main assistant loop."""
    logger.info("Leo Desktop Assistant starting...")

    status = await startup_diagnostics()

    if not startup_health.finalize():
        logger.error("Core service NLP failed — cannot start")
        return

    if not startup_health.can_start:
        logger.error("Startup conditions not met — cannot start")
        return

    print()

    # Deferred imports for legacy voice functions
    calibrate, listen, listen_wake, speak = _get_voice()

    # Optional: Face authentication
    faceauth = None
    try:
        from auth import faceauth as _faceauth
        faceauth = _faceauth
        startup_health.set_state("face_auth", SubsystemState.READY, "Available")
    except Exception:
        startup_health.set_state("face_auth", SubsystemState.DISABLED, "Unavailable")

    # Optional: Vision
    vision = None
    try:
        from vision import vision_manager as _vision_manager
        vision = _vision_manager
        if vision.initialize():
            # Check if the reasoning model is actually available.
            # If screen capture/OCR work but the vision model failed,
            # report DEGRADED (not READY).
            diag = vision.get_diagnostics()
            model_ok = diag.get("vision_model") not in (None, "", "none")
            if model_ok:
                startup_health.set_state("vision", SubsystemState.READY, "Available")
            else:
                status["degraded"] = True
                startup_health.set_state(
                    "vision", SubsystemState.DEGRADED,
                    "Screen/OCR available, reasoning model unavailable",
                )
        else:
            startup_health.set_state("vision", SubsystemState.DISABLED, "No backends")
    except Exception as e:
        logger.debug("Vision init: %s", e)
        startup_health.set_state("vision", SubsystemState.DISABLED, "Unavailable")

    # Voice calibration — ALREADY DONE in _init_voice().
    # Do NOT recalibrate here. Calibration happens exactly once
    # after the microphone stream has stabilized.
    voice_ok = status.get("voice", False)

    # ── Print voice subsystem diagnostics ──────────────
    print_startup_diagnostics()

    # ── Speaking flag for STT feedback prevention ─────
    _is_speaking = False
    _orig_synthesizer_speak = speech_synthesizer.speak

    def _speak_with_flag(text: str) -> bool:
        nonlocal _is_speaking
        _is_speaking = True
        try:
            result = _orig_synthesizer_speak(text)
            return result
        finally:
            # Small pause to let audio finish through speakers before mic resumes
            import time as _time
            _time.sleep(0.3)
            _is_speaking = False

    speech_synthesizer.speak = _speak_with_flag

    async def on_speak(event: Event):
        text = event.data.get("text", "")
        if text:
            _speak_with_flag(text)
    bus.on("speak", on_speak)

    # ── Interruption commands ──────────────────────────
    _INTERRUPT_PHRASES = {"stop", "cancel", "shut up", "be quiet", "silence", "that's enough"}

    def _is_interruption(text: str) -> bool:
        """Check if the user is trying to interrupt the assistant."""
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in _INTERRUPT_PHRASES)

    # ── Register conversation state handlers ────────────
    async def handle_youtube_query(text: str, data: dict) -> str:
        """Handle a YouTube search query from conversation state."""
        from plugins.youtube_plugin import YouTubePlugin
        yt = YouTubePlugin()
        await yt._on_search(Event("youtube_search", {"query": text}))
        return ""

    conversation_state.register_handler(PendingAction.YOUTUBE_QUERY, handle_youtube_query)

    print()
    if voice_ok:
        print("  Listening for wake word...")
    else:
        print("  Voice unavailable — running in text-only mode")
    print()

    # ── Timing accumulators ─────────────────────────────
    timings = {}
    _interaction_count = 0
    _command_retry_count = 0
    _max_command_retries = 10  # Max silent retries before returning to WAIT_WAKE

    # ── State machine ───────────────────────────────────
    _STATE = "BOOT"
    _STATE_ENTER_TIME = time.time()
    logger.info("[BOOT] Leo Desktop Assistant booting")

    def _set_state(new_state: str) -> None:
        nonlocal _STATE, _STATE_ENTER_TIME
        elapsed = time.time() - _STATE_ENTER_TIME
        logger.info("[STATE] %s → %s (was in %s for %.1fs)", _STATE, new_state, _STATE, elapsed)
        _STATE = new_state
        _STATE_ENTER_TIME = time.time()

    def _get_state_duration() -> float:
        return time.time() - _STATE_ENTER_TIME

    def _check_state_timeout(max_seconds: float) -> bool:
        """Check if current state has exceeded max_seconds. Returns True if timed out."""
        if _get_state_duration() > max_seconds:
            logger.warning("[TIMEOUT] State %s exceeded max duration %.1fs", _STATE, max_seconds)
            return True
        return False

    # ── Thread monitoring ───────────────────────────────
    async def _thread_monitor():
        """Periodically log active threads, queue sizes, and detect leaks."""
        import threading
        while True:
            await asyncio.sleep(30.0)
            try:
                threads = threading.enumerate()
                active = [t for t in threads if t.is_alive()]
                daemon = [t for t in active if t.daemon]
                non_daemon = [t for t in active if not t.daemon]
                logger.info("[THREADS] Total=%d Active=%d Daemon=%d NonDaemon=%d",
                          len(threads), len(active), len(daemon), len(non_daemon))
                # Log suspicious thread counts
                if len(active) > 50:
                    logger.warning("[THREADS] High thread count: %d (possible leak)", len(active))
                # Log executor threads
                import concurrent.futures
                for name in dir(concurrent.futures):
                    obj = getattr(concurrent.futures, name, None)
                    if isinstance(obj, concurrent.futures.ThreadPoolExecutor):
                        logger.debug("[THREADS] Executor %s: %d workers", name, obj._max_workers)
            except Exception as e:
                logger.debug("[THREADS] Monitor error: %s", e)

    # Start thread monitor
    _thread_monitor_task = asyncio.create_task(_thread_monitor())

    # ── Background preload tasks (only for tasks NOT done in startup) ──
    _background_tasks = set()

    # Embeddings are already preloaded in _init_embeddings() during startup.
    # No need to preload again here.

    async def _bg_init_vision():
        """Initialize vision in background (if not already initialized)."""
        logger.info("[BG] Initializing vision...")
        try:
            if vision and not vision.is_available:
                vision.initialize()
        except Exception as e:
            logger.debug("[BG] Vision init: %s", e)

    if vision:
        _bg_t2 = asyncio.create_task(_bg_init_vision())
        _background_tasks.add(_bg_t2)
        _bg_t2.add_done_callback(_background_tasks.discard)

    _set_state("READY")
    logger.info("[READY] Assistant ready, entering wake loop")

    print()
    if voice_ok:
        print("  Listening for wake word...")
    else:
        print("  Voice unavailable — running in text-only mode")
    print()

    while True:
        if not voice_ok:
            await asyncio.sleep(1)
            continue

        # ── STATE: WAIT_WAKE ────────────────────────────
        _set_state("WAIT_WAKE")

        # Pause wake-word detection while TTS is speaking
        # Check both the local flag AND the synthesizer's internal flag
        _is_tts_busy = _is_speaking or speech_synthesizer.is_speaking() or tts_manager.is_speaking()
        if _is_tts_busy:
            logger.debug("[WAKE] TTS busy, skipping wake detection")
            await asyncio.sleep(0.1)
            continue

        # Timeout: if we've been in WAIT_WAKE too long (shouldn't happen, but safety)
        if _check_state_timeout(300.0):  # 5 minutes max
            logger.info("[WAKE] Resetting wake detection after timeout")
            _set_state("WAIT_WAKE")
            continue

        # Run continuous wake listener in executor
        # This blocks until wake word is detected (never times out)
        _wake_start = time.time()
        _loop = asyncio.get_running_loop()
        try:
            logger.info("[WAKE] Wake listener active - waiting for 'hello leo'...")
            _wake_fut = _loop.run_in_executor(
                None,
                lambda: listen_wake(phrase_time_limit=5.0)
            )
            _track_voice_future(_wake_fut)
            wake_text = await _wake_fut
        except Exception as e:
            logger.warning("[WAKE] Listen error: %s", e)
            # Sleep to prevent busy loop on persistent errors
            await asyncio.sleep(2.0)
            continue

        if not wake_text:
            # Wake listener returned without detection.
            # This happens when:
            # - HAS_SOUNDDEVICE is False (misconfigured backend)
            # - Microphone init failed at module level
            # - Stream error occurred
            # Sleep to prevent WAIT_WAKE → WAIT_WAKE busy loop.
            # The log will show WAIT_WAKE only once every 2 seconds,
            # not thousands of times per second.
            await asyncio.sleep(2.0)
            continue

        # ── STATE: WAKE_DETECTED ────────────────────────
        _set_state("WAKE_DETECTED")

        # ── TWO-AUTHORITY WAKE GATE ─────────────────────
        # The wake MODEL already fired (primary authority, enforced inside
        # listen_wake_continuous). As defense-in-depth, the transcript must
        # ALSO pass strict verification here. Fuzzy matching alone is never
        # sufficient, and "thank you very much" can NEVER wake Leo.
        _det = wake_model_manager.last_detection or {}
        if not verify_wake_transcript(wake_text):
            logger.warning(
                "[WAKE] REJECTED by transcript verification: phrase=%r "
                "model='%s' score=%.3f — NOT waking (false wake blocked)",
                wake_text, _det.get("model", "?"), _det.get("score", 0.0))
            continue

        timings["wake"] = time.time() - _wake_start
        logger.info(
            "[WAKE] ACCEPTED: model='%s' score=%.3f transcript='%s' "
            "verified=True (%.0fms) — model + transcript agree",
            _det.get("model", "?"), _det.get("score", 0.0),
            wake_text, timings["wake"] * 1000)

        set_correlation_id()

        # ── STATE: FACE_AUTH (MANDATORY) ────────────────
        _set_state("FACE_AUTH")
        t_auth = time.time()
        user_name = None
        if faceauth:
            try:
                _loop = asyncio.get_running_loop()
                _auth_budget = getattr(faceauth, "AUTH_TOTAL_BUDGET", 18.0)
                _auth_fut = _loop.run_in_executor(None, faceauth.recognize_faces)
                _track_voice_future(_auth_fut)
                user_name = await asyncio.wait_for(_auth_fut, timeout=_auth_budget)
                if user_name:
                    timings["face"] = time.time() - t_auth
                    logger.info("[AUTH] Authenticated: %s (%.0fms)", user_name, timings["face"]*1000)
            except asyncio.TimeoutError:
                logger.warning("[AUTH] Timed out after %.0fs", _auth_budget)
            except Exception as e:
                logger.warning("[AUTH] Failed: %s", e)

        # Authentication is MANDATORY: no verified face → no command session.
        if faceauth and not user_name:
            logger.warning("[AUTH] ACCESS DENIED — no registered face authenticated. "
                           "Returning to WAIT_WAKE.")
            try:
                _loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    _loop.run_in_executor(
                        None, lambda: _speak_with_flag(
                            "I could not verify your identity. Access denied.")),
                    timeout=15.0,
                )
            except Exception:
                pass
            continue

        # ── STATE: GREETING ─────────────────────────────
        _set_state("GREETING")
        greeting_text = f"Welcome back, {user_name}. How can I help you today?" if user_name else "Hello, how may I assist you?"
        logger.info("[GREETING] Speaking: '%s'", greeting_text)
        # Run TTS in executor to avoid blocking event loop
        try:
            _loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                _loop.run_in_executor(None, lambda: _speak_with_flag(greeting_text)),
                timeout=30.0  # 30s max for greeting TTS
            )
        except asyncio.TimeoutError:
            logger.warning("[GREETING] TTS timed out after 30s")
        except Exception as e:
            logger.warning("[GREETING] TTS error: %s", e)
        logger.info("[GREETING] Complete")

        # ── STATE: WAIT_COMMAND ─────────────────────────
        _set_state("WAIT_COMMAND")
        _command_retry_count = 0
        while True:
            # Check if TTS is still speaking
            _is_tts_busy = _is_speaking or speech_synthesizer.is_speaking() or tts_manager.is_speaking()
            if _is_tts_busy:
                logger.debug("[CMD] Waiting for TTS to finish (is_speaking=%s, synth=%s, tts=%s)",
                           _is_speaking, speech_synthesizer.is_speaking(), tts_manager.is_speaking())
                await asyncio.sleep(0.1)
                continue

            # Timeout: max 2 minutes in WAIT_COMMAND
            if _check_state_timeout(120.0):
                logger.warning("[CMD] WAIT_COMMAND timed out after 120s — returning to WAIT_WAKE")
                break

            # Max retries: if we get too many silent loops, return to WAIT_WAKE
            if _command_retry_count >= _max_command_retries:
                logger.warning("[CMD] Max retries (%d) reached without command — returning to WAIT_WAKE",
                              _max_command_retries)
                break

            t_stt = time.time()
            logger.info("[CMD] Calling listen() (retry=%d/%d)", _command_retry_count + 1, _max_command_retries)

            # Run listen in executor to avoid blocking event loop
            _loop = asyncio.get_running_loop()
            try:
                _cmd_fut = _loop.run_in_executor(
                    None, lambda: listen(phrase_time_limit=7))
                _track_voice_future(_cmd_fut)
                query = await asyncio.wait_for(
                    _cmd_fut,
                    timeout=15.0  # 15s total timeout for listen+STT
                )
            except asyncio.TimeoutError:
                logger.warning("[CMD] listen() timed out after 15s")
                _command_retry_count += 1
                continue
            except Exception as e:
                logger.error("[CMD] listen() error: %s", e, exc_info=True)
                _command_retry_count += 1
                continue

            if not query:
                logger.debug("[CMD] No speech detected (retry %d/%d)",
                           _command_retry_count + 1, _max_command_retries)
                _command_retry_count += 1
                continue

            # Reset retry counter on successful capture
            _command_retry_count = 0
            timings["stt"] = time.time() - t_stt
            logger.info("[CMD] STT: '%s' (%.0fms)", query, timings["stt"]*1000)

            query = query.lower().strip()

            # ── Check for interruption commands ─────────
            if _is_interruption(query):
                logger.info("[CMD] Interruption: '%s'", query)
                speech_synthesizer._speaking = False
                tts_manager._speaking = False
                _speak_with_flag("Stopped.")
                break

            # ── Check pending conversation action ───────
            if conversation_state.has_pending_action:
                logger.info("[CMD] Pending action: %s", conversation_state.pending_action.name)
                await conversation_state.handle(query)
                continue

            # ── STATE: EXECUTE ──────────────────────────
            _set_state("EXECUTE")

            # Try Agent Planner first (if available)
            t_plan = time.time()
            planner_used = False
            result = ""

            if agent_planner.is_available:
                try:
                    logger.info("[PLANNER] Processing request via Agent Planner...")
                    # Run planner in executor (it may use LLM)
                    _loop = asyncio.get_running_loop()
                    planner_result = await _loop.run_in_executor(
                        None, lambda: agent_planner.process_request(query)
                    )
                    if planner_result:
                        result = planner_result
                        planner_used = True
                        logger.info("[PLANNER] Result: '%s'", result[:100])
                    else:
                        logger.info("[PLANNER] No result, falling back to NLP")
                except Exception as e:
                    logger.warning("[PLANNER] Failed: %s, falling back to NLP", e)
            else:
                logger.debug("[PLANNER] Agent planner not available, using NLP")

            # NLP fallback (if planner didn't handle it)
            if not planner_used:
                t_nlp = time.time()
                results = inference.classify(query, top_k=1)
                if not results:
                    results = [{"intent": "unknown", "confidence": 0.0, "metadata": {}}]
                top = results[0]
                entities = extract_entities(query)
                timings["nlp"] = time.time() - t_nlp
                logger.info("[NLP] Classified: intent='%s' confidence=%.2f (%.0fms)",
                           top["intent"], top["confidence"], timings["nlp"]*1000)

                # Check if vision is needed
                if vision and vision.is_available:
                    source = vision.needs_vision(top.get("intent", "unknown"), query)
                    if source:
                        logger.info("[VISION] Needed: %s", source.name)
                        vision_result = await vision.analyze_screen(source)
                        if vision_result and vision_result.text:
                            query = f"{query}\n[Screen context: {vision_result.text[:500]}]"
                            logger.debug("[VISION] Context added (%d chars)", len(vision_result.text))

                parsed = {
                    "text": query,
                    "intent": top["intent"],
                    "confidence": top["confidence"],
                    "entities": entities,
                    "metadata": top.get("metadata", {}),
                }
                context_manager.update(query, top["intent"], entities, top["confidence"])
                logger.info("[NLP] Intent: %s (conf=%.2f)", top["intent"], top["confidence"])

                # Plugin execution
                t_plugin = time.time()
                result = await handle_intent(parsed)
                timings["plugin"] = time.time() - t_plugin
                logger.info("[PLUGIN] Result: '%s' (%.0fms)", result, timings["plugin"]*1000)

                if result == "__EXIT__":
                    logger.info("[CMD] Exit requested — shutting down")
                    return

            timings["planning"] = time.time() - t_plan

            # ── STATE: SPEAK → WAIT_WAKE ───────────────
            _set_state("SPEAK")

            # ── Performance report ──────────────────────
            _interaction_count += 1
            report_parts = []
            for key in ["wake", "face", "vision", "stt", "nlp", "plugin"]:
                if key in timings:
                    report_parts.append(f"{key}={timings[key]*1000:.0f}ms")
            total = sum(v for v in timings.values() if isinstance(v, (int, float)))
            report_parts.append(f"total={total*1000:.0f}ms")
            logger.info(
                "PERF [#%d] %s",
                _interaction_count,
                " | ".join(report_parts),
            )

            # ── Log to DuckDB if available ──────────────
            if status.get("duckdb"):
                try:
                    from memory.duckdb_store import store
                    store.add_command(
                        text=query,
                        intent=top["intent"],
                        confidence=top["confidence"],
                        response=result,
                    )
                except Exception as e:
                    logger.debug("[DB] Log failed: %s", e)

            logger.info("[STATE] → WAIT_WAKE (interaction #%d complete)", _interaction_count)
            break  # Return to wake word loop


async def shutdown_gracefully(sig: Optional[int] = None) -> None:
    """Perform graceful shutdown of all subsystems.

    Order is critical — a worker thread must NEVER touch the AudioManager
    after it is stopped:
      1. Set the global shutdown event so every voice worker (wake loop,
         command recorder, STT) exits its loop WITHOUT accessing AudioManager.
      2. Cancel all pending asyncio tasks (thread monitor, background init).
      3. Join worker threads — give blocked listen() calls a moment to see
         the shutdown event and return.
      4. ONLY THEN stop the AudioManager (closes the InputStream).
      5. Shut down the remaining subsystems.
    """
    import threading as _threading
    from voice.audio_manager import shutdown_event

    logger.info("Shutting down (signal=%s)...", sig)

    # ── 1. Signal all voice workers to stop ──────────────────────
    # Every wake worker / command worker / recorder checks this event and
    # exits WITHOUT touching the AudioManager again.
    shutdown_event.set()
    logger.info("[SHUTDOWN] shutdown_event set — voice workers exiting")

    # ── 2. Cancel every pending voice future (wake/command/auth workers) ──
    # Futures that have NOT started running yet are cancelled outright;
    # running ones observe shutdown_event and return on their own.
    try:
        pending = [f for f in list(_PENDING_VOICE_FUTURES) if not f.done()]
        for f in pending:
            f.cancel()
        if pending:
            logger.info("[SHUTDOWN] Cancelled %d pending voice future(s) "
                        "(wake/command/auth workers)", len(pending))
    except Exception as e:
        logger.debug("[SHUTDOWN] Voice future cancel error: %s", e)

    # ── 3. Cancel all pending asyncio tasks ─────────────────────
    try:
        current = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks()
                 if t is not current and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("[SHUTDOWN] Cancelled %d asyncio task(s)", len(tasks))
    except Exception as e:
        logger.debug("[SHUTDOWN] Task cancel error: %s", e)

    # ── 4. Join worker threads (bounded wait, stragglers logged) ──
    deadline = time.time() + 5.0
    alive: list = []
    while time.time() < deadline:
        alive = [t for t in _threading.enumerate()
                 if t.is_alive() and not t.daemon
                 and t is not _threading.current_thread()
                 and "MainThread" not in t.name]
        if not alive:
            break
        await asyncio.sleep(0.05)
    if alive:
        logger.warning("[SHUTDOWN] %d worker thread(s) still alive after join "
                       "deadline: %s", len(alive), [t.name for t in alive])
    else:
        logger.info("[SHUTDOWN] All worker threads joined")

    # ── 5. NOW stop AudioManager — no component touches it after ─
    try:
        audio_manager.stop()
        logger.info("[SHUTDOWN] AudioManager stopped")
    except Exception as e:
        logger.warning("AudioManager shutdown error: %s", e)

    # ── 5. Shut down remaining subsystems ───────────────────────
    try:
        await voice_supervisor.shutdown()
    except Exception as e:
        logger.warning("Voice shutdown error: %s", e)

    try:
        from memory.duckdb_store import store
        store.close()
    except Exception as e:
        logger.warning("DuckDB shutdown error: %s", e)

    await plugin_manager.shutdown_all()
    logger.info("Leo shutdown complete.")


async def run_conversational_mode(no_auth: bool = False) -> None:
    """Run Leo as a full-duplex conversational agent (the new default).

    Delegates to the conversational runtime in leo.py:
      face auth → wake → streaming VAD → streaming Whisper → streaming LLM
      → streaming TTS (interruptible), with continuous conversation.
    """
    from leo import run_leo
    await run_leo(no_auth=no_auth)


async def main(conversational: bool = True, no_auth: bool = False):
    """Entry point."""
    _shutdown_requested = False

    async def _on_signal(sig):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            return
        _shutdown_requested = True
        await shutdown_gracefully(sig)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_on_signal(s))
        )

    try:
        if conversational:
            await run_conversational_mode(no_auth=no_auth)
        else:
            await main_loop()
    except asyncio.CancelledError:
        logger.info("Main loop cancelled")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        if not _shutdown_requested:
            await shutdown_gracefully()



def cmd_train(args):
    """Run training and exit."""
    print(f"Training NLP with {args.examples} examples per intent...")
    total = asyncio.run(trainer.train(examples_per_intent=args.examples))
    print(f"\u2713 Training complete! {total} examples generated.")
    print(f"  Model:    {settings.CLASSIFIER_PATH}")
    print(f"  Metadata: {settings.METADATA_PATH}")


def cmd_status():
    """Show model status."""
    if not is_model_ready():
        print("NLP model not found. Run: python main.py --train")
        return
    status = get_model_status()
    print("Leo NLP Model Status")
    print("=" * 40)
    print(f"  Ready:        \u2713")
    print(f"  Version:      {status.get('version', '?')}")
    print(f"  Embedding:    {status.get('embedding_model', '?')}")
    print(f"  Intents:      {status.get('intents', 0)}")
    print(f"  Examples:     {status.get('examples', 0)}")
    print(f"  Trained at:   {status.get('trained_at', '?')}")
    print(f"  Threshold:    {status.get('threshold', 0.75)}")
    print(f"  Model path:   {settings.CLASSIFIER_PATH}")
    print(f"  Metadata:     {settings.METADATA_PATH}")


def cmd_benchmark():
    """Run NLP benchmarks."""
    if not inference.load():
        print("NLP model not found. Run: python main.py --train")
        return
    from nlp.evaluator import evaluator
    from nlp.classifier import BUILTIN_INTENTS
    test_cases = {}
    for intent_name, examples in BUILTIN_INTENTS.items():
        if intent_name == "unknown" or not examples:
            continue
        test_cases[intent_name] = examples[:5]
    print("Running NLP benchmarks...")
    results = evaluator.evaluate_classification(test_cases)
    print(evaluator.report())


def cmd_select_mic():
    """Interactive microphone selection."""
    from voice.mic_selector import select_microphone_interactive
    select_microphone_interactive()


def cmd_calibrate():
    """Auto-calibrate wake detection with 10 repetitions of 'hello leo'."""
    import numpy as _np
    from voice.audio_manager import audio_manager
    from voice.audio_processing import audio_preprocessor
    from voice.stt import _recognize_bytes
    from voice.wake_word import wake_word_engine

    print()
    print("  ═══════════════════════════════════════════")
    print("  AUTO-CALIBRATION")
    print("  ═══════════════════════════════════════════")
    print("  You will say 'hello leo' 10 times.")
    print("  Leo measures thresholds, gain, and wake scores.")
    print()

    if not audio_manager.start():
        print("  FAILED: cannot start AudioManager")
        sys.exit(1)

    time.sleep(1)  # Let buffer fill

    scores = []
    snrs = []
    energy_thresholds = []

    for i in range(10):
        print(f"  [{i+1}/10] Say 'hello leo'...")
        # Listen for speech (max 8s)
        audio_bytes = audio_manager.record_command(timeout=8.0, phrase_limit=7.0)
        if audio_bytes is None:
            print("    No speech detected, try again")
            continue

        # Process with noise suppression
        samples = _np.frombuffer(audio_bytes, dtype=_np.int16)
        processed = audio_preprocessor.process(samples)
        proc_bytes = processed.tobytes()

        # STT
        text = _recognize_bytes(proc_bytes, audio_manager.sample_rate)
        if not text:
            print("    No speech recognized, try again")
            continue

        print(f"    Recognized: '{text}'")

        # Compute wake score
        best_ratio = 0.0
        for variant in wake_word_engine._variants:
            import difflib
            ratio = difflib.SequenceMatcher(None, text.lower(), variant.lower()).ratio()
            best_ratio = max(best_ratio, ratio)

        scores.append(best_ratio)

        # Compute SNR
        rms = float(_np.sqrt(_np.mean(samples.astype(float) ** 2)))
        preproc_metrics = audio_preprocessor.get_metrics()
        nf = preproc_metrics.get('noise_floor_raw', 0)
        if nf and nf > 0:
            snr_db = 20 * _np.log10((rms / 32768.0 + 1e-10) / (nf + 1e-10))
        else:
            snr_db = 40.0
        snrs.append(snr_db)

        # Energy threshold
        energy_thresholds.append(max(300.0, rms * 1.5))

    audio_manager.stop()

    if scores:
        import statistics
        avg_score = statistics.mean(scores)
        avg_snr = statistics.mean(snrs)
        best_threshold = statistics.median(energy_thresholds) if energy_thresholds else 300.0

        print()
        print("  ═══════════════════════════════════════════")
        print("  CALIBRATION RESULTS")
        print("  ═══════════════════════════════════════════")
        print(f"  Successful phrases: {len(scores)}/10")
        print(f"  Average wake score: {avg_score:.3f}")
        print(f"  Average SNR:        {avg_snr:.1f} dB")
        print(f"  Energy threshold:   {best_threshold:.0f}")
        print()

        # Save calibration
        calibration = {
            "wake_score_threshold": max(0.7, avg_score * 0.8),
            "energy_threshold": best_threshold,
            "snr_db": avg_snr,
            "calibrated_at": time.time(),
        }
        try:
            from pathlib import Path as _Path
            cal_path = _Path("data/wake_calibration.json")
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(cal_path, "w") as f:
                json.dump(calibration, f, indent=2)
            print(f"  Calibration saved to: {cal_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save calibration: {e}")

        print()
    else:
        print()
        print("  Calibration failed — no phrases recognized.")
        print("  Check microphone levels and try again.")
        print()


def cmd_audio_debug():
    """Real-time audio level visualizer."""
    from voice.audio_manager import audio_manager
    from voice.audio_processing import audio_preprocessor
    import numpy as _np

    def _bar(value, max_val, width=40):
        pct = min(max(int(value / max_val * width), 0), width)
        return "█" * pct + "░" * (width - pct)

    print("\n  Real-time audio debug (Ctrl+C to stop)\n")

    if not audio_manager.start():
        print("  FAILED: cannot start AudioManager")
        sys.exit(1)

    try:
        while True:
            time.sleep(0.1)
            audio = audio_manager.get_recent_audio(0.1)
            if len(audio) == 0:
                continue

            rms = float(_np.sqrt(_np.mean(audio.astype(float) ** 2)))
            peak = float(_np.max(_np.abs(audio)))
            norm_rms = rms / 32768.0

            preproc = audio_preprocessor.get_metrics()
            noise_floor = preproc.get("noise_floor_raw", 0)
            gain = preproc.get("gain_applied", 1.0)

            # Compute noise floor dB
            nf_db = 20 * _np.log10(noise_floor / 32768.0) if noise_floor > 0 else -120.0

            # VAD state
            vad_state = audio_manager.get_diagnostics().get("vad_state", "?")

            import sys as _sys
            _sys.stdout.write(f"\r")
            _sys.stdout.write(
                f" In: {_bar(rms, 32768)} {norm_rms*100:5.1f}% "
                f"| RMS={rms:6.0f} Peak={peak:6.0f} "
                f"| Noise={nf_db:6.1f}dB Gain={gain:.2f}x "
                f"| VAD={vad_state}"
            )
            _sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n  Stopped")
    finally:
        audio_manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leo Desktop Assistant")
    parser.add_argument("--train", action="store_true",
                        help="Train the NLP model and exit")
    parser.add_argument("--examples", type=int, default=100,
                        help="Examples per intent for training (default: 100)")
    parser.add_argument("--status", action="store_true",
                        help="Show NLP model status")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run NLP benchmarks")
    parser.add_argument("--select-mic", action="store_true",
                        help="Interactively select the best microphone")
    parser.add_argument("--calibrate", action="store_true",
                        help="Auto-calibrate wake detection thresholds")
    parser.add_argument("--audio-debug", action="store_true",
                        help="Real-time audio level visualizer")
    parser.add_argument("--train-wake", action="store_true",
                        help="Record 100 wake phrases + train custom verifier")
    parser.add_argument("--legacy", action="store_true",
                        help="Run the legacy command-executor loop instead of conversational mode")
    parser.add_argument("--no-auth", action="store_true",
                        help="Skip face authentication (development only)")

    args = parser.parse_args()

    if args.train:
        cmd_train(args)
        sys.exit(0)
    if args.status:
        cmd_status()
        sys.exit(0)

    if args.benchmark:
        cmd_benchmark()
        sys.exit(0)

    if args.select_mic:
        cmd_select_mic()
        sys.exit(0)

    if args.calibrate:
        cmd_calibrate()
        sys.exit(0)

    if args.audio_debug:
        cmd_audio_debug()
        sys.exit(0)

    if args.train_wake:
        from voice.calibrate_wake import run_calibration
        ok = run_calibration()
        sys.exit(0 if ok else 1)

    asyncio.run(main(conversational=not args.legacy, no_auth=args.no_auth))
