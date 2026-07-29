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
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

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

# ── Voice imports ─────────────────────────────────────────
from voice.supervisor import VoiceSupervisor, VoiceState, voice_supervisor
from voice.settings import voice_settings
from voice.synthesizer import speech_synthesizer
from voice.recognizer import speech_recognizer
from voice.wake_word import wake_word_engine
from voice.microphone import microphone
from voice.noise import noise_calibrator
from voice.audio_device import audio_device

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
    """Preload embedding model."""
    startup_health.register("embeddings")
    try:
        preload_embedding_model()
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
    """Initialize voice subsystem."""
    startup_health.register("voice")
    try:
        voice_settings.update_from_env()
        audio_device.detect_backend()
        mic = microphone.get_microphone()
        mic_available = mic is not None
        if not mic_available:
            status["voice"] = False
            startup_health.set_state("voice", SubsystemState.DISABLED, "Mic unavailable")
        else:
            noise_calibrator.load_profile()
            status["voice"] = True
            startup_health.set_state("voice", SubsystemState.READY, "Mic available")
    except Exception as e:
        status["voice"] = False
        startup_health.set_state("voice", SubsystemState.DISABLED, str(e))


async def _init_telegram(status: dict) -> None:
    """Initialize Telegram client — non-blocking, checks session file only."""
    startup_health.register("telegram")
    try:
        from scripts.telegram_bot import _available as tg_available
        # Quick check: if no session file exists, disable immediately
        import os as _os
        from pathlib import Path as _Path
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
        from vision import vision as _vision
        vision = _vision
        startup_health.set_state("vision", SubsystemState.READY, "Available")
    except Exception:
        startup_health.set_state("vision", SubsystemState.DISABLED, "Unavailable")

    # Voice calibration
    voice_ok = status.get("voice", False)
    if voice_ok:
        try:
            calibrate()
        except Exception as e:
            logger.warning("Calibration failed: %s", e)
            voice_ok = False
            startup_health.set_state("voice", SubsystemState.DISABLED, "Calibration failed")

    # ── Speaking flag for STT feedback prevention ─────
    _is_speaking = False
    _orig_synthesizer_speak = speech_synthesizer.speak

    def _speak_with_flag(text: str) -> bool:
        nonlocal _is_speaking
        _is_speaking = True
        try:
            return _orig_synthesizer_speak(text)
        finally:
            _is_speaking = False

    speech_synthesizer.speak = _speak_with_flag

    async def on_speak(event: Event):
        text = event.data.get("text", "")
        if text:
            _speak_with_flag(text)
    bus.on("speak", on_speak)

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

    while True:
        if not voice_ok:
            await asyncio.sleep(1)
            continue

        if _is_speaking:
            await asyncio.sleep(0.1)
            continue

        t0 = time.time()
        wake_text = listen_wake(phrase_time_limit=3)
        if not wake_text:
            continue
        if not wake_word_engine.detect(wake_text):
            continue
        timings["wake"] = time.time() - t0
        logger.info("Wake detected")

        set_correlation_id()

        # ── Face authentication ─────────────────────────
        t_auth = time.time()
        if faceauth:
            try:
                user_name = faceauth.recognize_faces()
                if not user_name:
                    faceauth.Unknown_Face()
                    continue
                timings["face"] = time.time() - t_auth
                logger.info("Face authenticated: %s", user_name)
                _speak_with_flag(
                    f"Authentication successful. Welcome back {user_name}. "
                    "I am ready. How can I help you today?"
                )
            except Exception as e:
                logger.warning("Face auth failed: %s", e)
                _speak_with_flag("Hello, how may I assist you?")
        else:
            _speak_with_flag("Hello, how may I assist you?")

        # ── Command loop ────────────────────────────────
        while True:
            if _is_speaking:
                await asyncio.sleep(0.1)
                continue

            t_stt = time.time()
            query = listen(phrase_time_limit=7)
            if not query:
                continue
            timings["stt"] = time.time() - t_stt

            query = query.lower().strip()

            # ── Check pending conversation action ───────
            if conversation_state.has_pending_action:
                logger.info("Pending action: %s", conversation_state.pending_action.name)
                await conversation_state.handle(query)
                continue

            # ── NLP classification ──────────────────────
            t_nlp = time.time()
            results = inference.classify(query, top_k=1)
            if not results:
                results = [{"intent": "unknown", "confidence": 0.0, "metadata": {}}]
            top = results[0]
            entities = extract_entities(query)
            timings["nlp"] = time.time() - t_nlp

            parsed = {
                "text": query,
                "intent": top["intent"],
                "confidence": top["confidence"],
                "entities": entities,
                "metadata": top.get("metadata", {}),
            }

            context_manager.update(query, top["intent"], entities, top["confidence"])

            logger.info("Intent detected: %s", top["intent"])

            # ── Plugin execution ────────────────────────
            t_plugin = time.time()
            result = await handle_intent(parsed)
            timings["plugin"] = time.time() - t_plugin

            if result == "__EXIT__":
                return

            # ── Timing summary ──────────────────────────
            if all(k in timings for k in ("wake", "stt", "nlp", "plugin")):
                logger.info(
                    "Wake:%.1fs Face:%.1fs STT:%.1fs NLP:%.1fs Plugin:%.1fs Total:%.1fs",
                    timings.get("wake", 0),
                    timings.get("face", 0),
                    timings.get("stt", 0),
                    timings.get("nlp", 0),
                    timings.get("plugin", 0),
                    sum(timings.values()),
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
                except Exception:
                    pass

            logger.info("Interaction completed")


async def shutdown_gracefully(sig: Optional[int] = None) -> None:
    """Perform graceful shutdown of all subsystems."""
    logger.info("Shutting down (signal=%s)...", sig)

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


async def main():
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

    asyncio.run(main())