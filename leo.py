"""
Leo — Conversational Desktop Agent (production entry point).

This is the NEW Leo: a real conversational companion like Siri /
ChatGPT Voice / Gemini Live — NOT a command executor.

  python leo.py              # Start conversational Leo
  python leo.py --no-auth    # Skip face auth (dev only)
  python leo.py --status     # Show subsystem status

Pipeline:
  Wake ("leo" / "hey leo" / "hello leo")
    → Transcript verification
    → Face auth (mandatory — camera opens ONLY here, waits forever)
    → Streaming VAD
    → Streaming Whisper (partials)
    → Streaming LLM (sentence-by-sentence)
    → Streaming TTS (interruptible)
    → Full duplex: interrupt Leo any time by speaking
    → 60 s of silence / "goodbye" / "stop listening" / "cancel"
    → Back to wake listening (Leo NEVER exits on its own)

Everything is async and cancellable. Graceful shutdown on Ctrl+C.
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT FIXES (before any heavy imports)
# ═══════════════════════════════════════════════════════════════
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"
try:
    subprocess.run(["xhost", "+local:"], check=False,
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
except Exception:
    pass

# Python 3.14 compatibility shims
import compat  # noqa: F401, E402

from config.settings import settings  # noqa: E402
from telemetry.logger import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("leo")


# ═══════════════════════════════════════════════════════════════
# Face authentication — runs ONLY after the wake word, NEVER at startup.
# ═══════════════════════════════════════════════════════════════

async def authenticate_on_wake() -> Optional[str]:
    """
    Run live face authentication in the popup window.

    This is the engine's auth provider. It is called ONLY after the wake
    word is detected (and only when the previous session has expired).

    Returns the verified user's name, or None if denied/cancelled.
    NEVER raises, NEVER terminates Leo.
    """
    loop = asyncio.get_event_loop()
    try:
        from auth.live_auth import authenticate_live
    except Exception as e:
        logger.error("[AUTH] live_auth unavailable: %s", e)
        return None
    try:
        # authenticate_live blocks on the camera → run in an executor thread.
        # No timeout here: the popup waits forever for a face; the user can
        # close the popup to cancel (which returns None → deny → wake).
        name = await loop.run_in_executor(None, authenticate_live)
        return name
    except Exception as e:
        logger.warning("[AUTH] live authentication error: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# Conversational runtime
# ═══════════════════════════════════════════════════════════════

async def run_leo(no_auth: bool = False) -> None:
    """
    BOOT → LOAD MODELS → INIT AUDIO → WAIT_WAKE.

    Leo ALWAYS boots successfully and stays alive forever. Face auth is
    deferred until the wake word. There is NO startup authentication.

    The conversation engine owns the real boot sequence (audio → wake
    model → VAD → TTS → Whisper) and logs each stage as it completes.
    """
    from core.conversation_engine import conversation_engine
    from agent.action_dispatcher import action_dispatcher
    from services.vision_service import vision_service
    from services.search_service import search_service
    from services.screen_capture import screen_capture_service

    # ── Wire subsystems ───────────────────────────────────
    conversation_engine.set_action_executor(action_dispatcher.execute)

    # Vision context: Use the new VisionService (structured UI tree + OCR)
    # Falls back gracefully to the old vision module if the new service is
    # unavailable.
    conversation_engine.set_vision_context(vision_service.quick_context)

    # Search provider: Use the new SearchService (DuckDuckGo/Tavily + trafilatura)
    conversation_engine.set_search_provider(search_service.context_for_llm)

    # Learning context: Context Composer — smart, ranked, compact memory injection
    # Replaces the old naive learning_engine.llm_context() approach.
    from agent.context_composer import context_composer
    conversation_engine.set_learning_context(
        lambda: context_composer.compose("", max_tokens=500)
    )

    # ── Initialize new services ───────────────────────────
    # MusicAgent: unified music control (MPV, Spotify, YouTube, local)
    from services.music_agent import music_agent
    try:
        await music_agent.initialize()
        logger.info("[LEO] MusicAgent initialized")
    except Exception as e:
        logger.debug("[LEO] MusicAgent init skipped: %s", e)

    # Augment the learning engine callable with the composer for richer context
    from learning.learning_engine import learning_engine
    # Keep backward compatibility: the engine's llm_context() still works
    # but ContextComposer now handles ranking/compression in the LLM pipe.

    # ── Face auth provider (deferred to wake) ─────────────
    if no_auth:
        logger.warning("[AUTH] --no-auth: face authentication disabled (dev mode)")
        conversation_engine.set_authenticated(None)   # dev session, no camera
    else:
        conversation_engine.set_auth_provider(authenticate_on_wake)

    # ── Run engine + watchdog concurrently ────────────────
    engine_task = asyncio.create_task(conversation_engine.run())
    watchdog_task = asyncio.create_task(conversation_engine.timeout_watchdog())

    logger.info("Listening for wake word...")

    try:
        await asyncio.gather(engine_task, watchdog_task)
    except asyncio.CancelledError:
        pass
    finally:
        for t in (engine_task, watchdog_task):
            t.cancel()
        await asyncio.gather(engine_task, watchdog_task, return_exceptions=True)


# ═══════════════════════════════════════════════════════════════
# Graceful shutdown
# ═══════════════════════════════════════════════════════════════

async def shutdown() -> None:
    """Stop all subsystems cleanly."""
    logger.info("[LEO] Shutting down...")
    try:
        from core.conversation_engine import conversation_engine
        await conversation_engine.shutdown()
    except Exception as e:
        logger.debug("engine shutdown: %s", e)

    try:
        from voice.streaming_tts import streaming_tts
        streaming_tts.close()
    except Exception as e:
        logger.debug("tts shutdown: %s", e)

    try:
        from voice.streaming_stt import streaming_stt
        streaming_stt.cancel()
    except Exception as e:
        logger.debug("stt shutdown: %s", e)

    try:
        from voice.audio_manager import audio_manager, shutdown_event
        shutdown_event.set()
        audio_manager.stop()
    except Exception as e:
        logger.debug("audio shutdown: %s", e)

    try:
        from auth.faceauth import close as _auth_close
        _auth_close()
    except Exception:
        pass

    logger.info("[LEO] Shutdown complete.")


async def _main_async(no_auth: bool) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _signal(*_):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal)
        except (NotImplementedError, RuntimeError):
            pass

    run_task = asyncio.create_task(run_leo(no_auth=no_auth))
    stop_task = asyncio.create_task(stop.wait())

    done, pending = await asyncio.wait(
        {run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

    for t in pending:
        t.cancel()
    await shutdown()
    # Drain cancelled tasks
    await asyncio.gather(*pending, return_exceptions=True)


def cmd_status() -> None:
    """Print a quick subsystem status report."""
    print("Leo subsystem status")
    print("=" * 50)
    # Ollama
    try:
        import httpx
        r = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Ollama:        OK  ({len(models)} models: {', '.join(models[:3])})")
    except Exception as e:
        print(f"  Ollama:        DOWN ({e})")
    # Whisper
    try:
        import faster_whisper  # noqa
        print("  faster-whisper: installed")
    except ImportError:
        print("  faster-whisper: NOT installed (pip install faster-whisper)")
    # Kokoro
    try:
        import kokoro  # noqa
        print("  Kokoro TTS:    installed")
    except ImportError:
        print("  Kokoro TTS:    NOT installed (pip install kokoro)")
    # Silero VAD
    try:
        import torch  # noqa
        print("  torch (VAD):   installed")
    except ImportError:
        print("  torch (VAD):   NOT installed")
    # sounddevice
    try:
        import sounddevice  # noqa
        print("  sounddevice:   installed")
    except ImportError:
        print("  sounddevice:   NOT installed")
    # Face encodings
    ep = settings.KNOWN_ENCODINGS_PATH
    print(f"  Face encodings: {'present' if ep.exists() else 'MISSING'}")


def run(no_auth: bool = False) -> None:
    """Canonical blocking entry point — boots Leo and runs until Ctrl+C.

    Used by BOTH `python leo.py` and `python main.py` so there is exactly
    ONE runtime entry path.
    """
    try:
        asyncio.run(_main_async(no_auth=no_auth))
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Leo — Conversational Desktop Agent")
    parser.add_argument("--no-auth", action="store_true",
                        help="Skip face authentication (development only)")
    parser.add_argument("--status", action="store_true",
                        help="Show subsystem status and exit")
    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    run(no_auth=args.no_auth)


if __name__ == "__main__":
    main()
