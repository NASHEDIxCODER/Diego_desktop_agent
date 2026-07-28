"""
Leo Desktop Assistant — Production Entry Point

Architecture:
  VoiceSupervisor (state machine) → Inference Engine → Plugin Router → TTS

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

Design: Resilient Service Manager
  Core services (NLP, Voice, Plugin Manager, Intent Router, Event Bus)
    - Missing core services STOP startup.

  Optional services (DuckDB, Telegram, Face Authentication, Vision, Weather, Calendar)
    - Missing optional services show warnings but startup continues.
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

# ── Python 3.14 compatibility: inject removed stdlib modules ──
import compat  # noqa: F401 — injects stubs before any third-party imports

# Ensure DISPLAY exists
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

try:
    subprocess.run(["xhost", "+local:"], check=False)
except Exception:
    pass

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

# ── AI imports ────────────────────────────────────────────
from ai.llm_client import llm_client, llm_chat

# ── Voice imports (new production pipeline) ──────────────
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
    from voice.tts import speak
    return calibrate, listen, listen_wake, speak


def wish_me():
    import datetime
    _, _, _, speak = _get_voice()
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good morning, sir.")
    elif 12 <= hour < 18:
        speak("Good afternoon, sir.")
    else:
        speak("Good evening, sir.")
    speak("I am Leo, your assistant. How may I help you?")


async def handle_intent(parsed: dict) -> str:
    _, _, _, speak = _get_voice()
    intent = parsed.get("intent", "unknown")
    entities = parsed.get("entities", {})
    confidence = parsed.get("confidence", 0.0)

    logger.info("Handling intent: %s (confidence=%.4f)", intent, confidence)

    if intent == "greeting":
        wish_me()
        return ""

    if intent == "exit":
        speak("Goodbye, have a nice day.")
        return "__EXIT__"

    # ── Intent → Event mapping for plugin-bound intents ─────
    # The NLP classifier returns intent names matching dataset filenames.
    # Plugin event handlers listen for specific event names.
    # Map intents to their corresponding event types.
    if intent == "youtube":
        await bus.emit("youtube_open", data=entities, source="nlp")
        return ""

    if intent.startswith("youtube_"):
        # e.g. youtube_pause, youtube_resume, etc. — direct passthrough
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
        speak(f"The time is {now.strftime('%I:%M %p')}.")
        return ""

    if intent == "date_query":
        import datetime
        now = datetime.datetime.now()
        speak(f"Today is {now.strftime('%A, %B %d, %Y')}.")
        return ""

    if intent == "help":
        speak("I can control YouTube, send Telegram messages, "
              "adjust brightness, tell the time and date, "
              "and have conversations. What would you like to do?")
        return ""

    if intent == "joke":
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the AI break up with the database? Too many relationships!",
            "What do you call a fake noodle? An impasta!",
        ]
        import random
        speak(random.choice(jokes))
        return ""

    if intent == "who_am_i":
        speak("You are my user. I recognize you by your face.")
        return ""

    if intent == "unknown" or confidence < settings.SIMILARITY_THRESHOLD * 0.8:
        response = await llm_chat(parsed.get("text", ""))
        speak(response)
        return ""

    response = await llm_chat(parsed.get("text", ""))
    speak(response)
    return ""


async def startup_diagnostics() -> dict:
    """
    Run startup diagnostics using StartupHealth.

    Each subsystem reports its true state:
      READY    — Fully operational
      DEGRADED — Working with reduced functionality
      DISABLED — Skipped (optional, not available)
      FAILED   — Fatal error, blocks assistant startup

    Returns a dict with status of each service.
    """
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

    # ── Core: NLP ──────────────────────────────────────────
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
        print()
        print("  NLP model not found. Run: python main.py --train")
        print()
        return status  # Fatal — no NLP means no assistant

    # ── Preload embedding model (SentenceTransformer singleton) ──
    # Load once at startup, never during command handling.
    # This prevents repeated HuggingFace downloads.
    startup_health.register("embeddings")
    try:
        preload_embedding_model()
        startup_health.set_state("embeddings", SubsystemState.READY,
                                 f"Model: {settings.MODEL_NAME}")
    except Exception as e:
        logger.error("Failed to preload embedding model: %s", e)
        startup_health.set_state("embeddings", SubsystemState.FAILED, str(e))
        return status

    # ── Preload TTS engine at startup ──────────────────────
    startup_health.register("tts")
    try:
        speech_synthesizer.initialize()
        startup_health.set_state("tts", SubsystemState.READY, "TTS initialized")
    except Exception as e:
        logger.warning("TTS preload failed: %s", e)
        startup_health.set_state("tts", SubsystemState.DEGRADED, str(e))

    # ── Optional: DuckDB ──────────────────────────────────
    startup_health.register("duckdb")
    try:
        from memory.duckdb_store import store
        from memory.duckdb_store import DatabaseLockedError
        try:
            store.initialize()
            if store._conn is not None:
                status["duckdb"] = True
                startup_health.set_state("duckdb", SubsystemState.READY, "Connected")
            else:
                status["duckdb"] = False
                status["degraded"] = True
                startup_health.set_state(
                    "duckdb", SubsystemState.DEGRADED,
                    "Locked — runtime memory disabled",
                )
        except DatabaseLockedError:
            status["duckdb"] = False
            status["degraded"] = True
            startup_health.set_state(
                "duckdb", SubsystemState.DEGRADED,
                "Locked — runtime memory disabled",
            )
    except ImportError:
        startup_health.set_state("duckdb", SubsystemState.DISABLED, "Not installed")
    except Exception as e:
        status["duckdb"] = False
        status["degraded"] = True
        startup_health.set_state("duckdb", SubsystemState.DEGRADED, str(e))

    # ── Core: Plugins ─────────────────────────────────────
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
                    f"{len(names)} loaded, {len(failed)} disabled: {', '.join(failed)}",
                )
            else:
                startup_health.set_state(
                    "plugins", SubsystemState.READY,
                    f"{len(names)} loaded: {', '.join(names)}",
                )
        else:
            status["plugins"] = True
            startup_health.set_state("plugins", SubsystemState.READY, "None discovered")
    except Exception as e:
        status["plugins"] = False
        startup_health.set_state("plugins", SubsystemState.FAILED, str(e))

    # ── Voice: new production pipeline ────────────────────
    startup_health.register("voice")
    try:
        voice_settings.update_from_env()
        audio_device.detect_backend()

        # Check microphone availability BEFORE calibration
        mic = microphone.get_microphone()
        mic_available = mic is not None

        if not mic_available:
            status["voice"] = False
            startup_health.set_state(
                "voice", SubsystemState.DISABLED,
                "Microphone unavailable — voice features disabled",
            )
        else:
            # Load cached noise profile (non-blocking, no calibration at startup)
            noise_calibrator.load_profile()
            status["voice"] = True
            startup_health.set_state("voice", SubsystemState.READY, "Microphone available")
    except Exception as e:
        status["voice"] = False
        startup_health.set_state(
            "voice", SubsystemState.DISABLED,
            f"Error: {e}",
        )

    # ── Optional: Telegram ────────────────────────────────
    startup_health.register("telegram")
    try:
        from scripts.telegram_bot import init as tg_init, _available as tg_available
        await tg_init()
        if tg_available:
            startup_health.set_state("telegram", SubsystemState.READY, "Connected")
        else:
            startup_health.set_state(
                "telegram", SubsystemState.DISABLED,
                "Session not found — run interactively to create",
            )
    except ImportError:
        startup_health.set_state("telegram", SubsystemState.DISABLED, "Not installed")
    except Exception as e:
        startup_health.set_state("telegram", SubsystemState.DISABLED, str(e))

    return status


async def main_loop():
    """Main assistant loop using StartupHealth."""
    logger.info("Leo Desktop Assistant starting...")

    status = await startup_diagnostics()

    # Finalize startup — prints summary and checks if we can start
    if not startup_health.finalize():
        logger.error("Core service NLP failed — cannot start")
        return

    if not startup_health.can_start:
        logger.error("Startup conditions not met — cannot start")
        return

    print()

    # Deferred imports for legacy voice functions
    calibrate, listen, listen_wake, speak = _get_voice()

    # Optional: Face authentication (cv2-dependent)
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

    # Voice calibration (non-blocking, only if mic available)
    voice_ok = status.get("voice", False)
    if voice_ok:
        try:
            calibrate()
            logger.info("Voice calibration complete")
        except Exception as e:
            logger.warning("Voice calibration failed: %s — voice disabled", e)
            voice_ok = False
            startup_health.set_state(
                "voice", SubsystemState.DISABLED,
                "Calibration failed",
            )

    async def on_speak(event: Event):
        text = event.data.get("text", "")
        if text:
            speak(text)
    bus.on("speak", on_speak)

    # ── Suspend STT flag for TTS feedback prevention ─────
    _speaking_lock = asyncio.Lock()
    _is_speaking = False

    async def speak_safe(text: str) -> None:
        """Speak text, preventing STT from hearing its own output."""
        nonlocal _is_speaking
        async with _speaking_lock:
            _is_speaking = True
            try:
                # Use the async-capable TTS
                from voice.tts import speak as tts_speak
                from voice.synthesizer import speech_synthesizer
                # Use synthesizer with wait-for-completion
                speech_synthesizer.speak(text)
            finally:
                _is_speaking = False

    # Patch the speak function to respect the speaking lock
    _orig_speak = speak

    def patched_speak(text: str) -> None:
        """Speak with audio feedback prevention."""
        if _is_speaking:
            return  # Prevent re-entrance
        _orig_speak(text)

    # Store for use by handle_intent and listen_wake guard
    main_loop_voice_ok = voice_ok

    # Replace the global speak function used by handle_intent
    import voice.tts as tts_module
    tts_module.speak = patched_speak

    print()
    if voice_ok:
        print("  Listening for wake word...")
    else:
        print("  Voice unavailable — running in text-only mode")
    print()

    while True:
        if not voice_ok:
            # No voice — just sleep to avoid busy loop
            await asyncio.sleep(1)
            continue

        # Skip listening while TTS is speaking (feedback prevention)
        if _is_speaking:
            await asyncio.sleep(0.1)
            continue

        wake_text = listen_wake(phrase_time_limit=3)
        if not wake_text:
            continue

        if not wake_word_engine.detect(wake_text):
            continue

        logger.info("Wake word detected: %s", wake_text)
        set_correlation_id()  # New correlation ID for this interaction

        if faceauth:
            try:
                user_name = faceauth.recognize_faces()
                if not user_name:
                    faceauth.Unknown_Face()
                    continue
                # Production greeting
                speech_synthesizer.speak(
                    f"Authentication successful. Welcome back {user_name}. "
                    "I am ready. How can I help you today?"
                )
            except Exception as e:
                logger.warning("Face auth failed: %s", e)
                speak("Hello, how may I assist you?")
        else:
            speak("Hello, how may I assist you?")

        while True:
            # Skip listening while TTS is speaking (feedback prevention)
            if _is_speaking:
                await asyncio.sleep(0.1)
                continue

            query = listen(phrase_time_limit=7)
            if not query:
                continue

            query = query.lower().strip()
            logger.info("Command: %s", query)

            results = inference.classify(query, top_k=1)
            if not results:
                results = [{"intent": "unknown", "confidence": 0.0, "metadata": {}}]

            top = results[0]
            entities = extract_entities(query)

            parsed = {
                "text": query,
                "intent": top["intent"],
                "confidence": top["confidence"],
                "entities": entities,
                "metadata": top.get("metadata", {}),
            }

            context_manager.update(query, top["intent"], entities, top["confidence"])

            result = await handle_intent(parsed)
            if result == "__EXIT__":
                return

            # Log to DuckDB if available (optional) — never block on failure
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


async def shutdown_gracefully(sig: Optional[int] = None) -> None:
    """Perform graceful shutdown of all subsystems."""
    logger.info("Shutting down (signal=%s)...", sig)

    # Shutdown voice supervisor
    try:
        await voice_supervisor.shutdown()
    except Exception as e:
        logger.warning("Voice shutdown error: %s", e)

    # Close DuckDB with WAL checkpoint
    try:
        from memory.duckdb_store import store
        store.close()
    except Exception as e:
        logger.warning("DuckDB shutdown error: %s", e)

    # Shutdown plugins
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