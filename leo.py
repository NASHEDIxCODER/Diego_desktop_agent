"""
Leo — Conversational Desktop Agent (production entry point).

This is the NEW Leo: a real conversational companion like Siri /
ChatGPT Voice / Gemini Live — NOT a command executor.

  python leo.py              # Start conversational Leo
  python leo.py --no-auth    # Skip face auth (dev only)
  python leo.py --status     # Show subsystem status

Pipeline:
  Face auth (mandatory)
    → Wake ("leo" / "hey leo" / "hello leo")
    → Streaming VAD
    → Streaming Whisper (partials)
    → Streaming LLM (sentence-by-sentence)
    → Streaming TTS (interruptible)
    → Full duplex: interrupt Leo any time by speaking

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
# Face authentication (mandatory)
# ═══════════════════════════════════════════════════════════════

async def authenticate() -> tuple:
    """
    Run mandatory face authentication.

    Uses the robust pipeline (multi-frame voting, confidence averaging,
    head pose estimation, anti-spoofing). Falls back to the standard
    recognizer if the robust one is unavailable.

    Returns (ok: bool, name: Optional[str]).
    Runs the blocking camera pipeline in an executor thread.
    """
    logger.info("[AUTH] Starting face authentication (mandatory)...")
    loop = asyncio.get_event_loop()

    # Prefer the robust recognizer (voting + pose + anti-spoofing)
    recog_fn = None
    try:
        from auth.robust_auth import recognize_faces_robust
        recog_fn = recognize_faces_robust
        logger.info("[AUTH] Using robust face authentication")
    except Exception as e:
        logger.debug("[AUTH] Robust auth unavailable (%s); using standard", e)
        try:
            from auth.faceauth import recognize_faces
            recog_fn = recognize_faces
        except Exception as e2:
            logger.error("[AUTH] Face auth module unavailable: %s", e2)
            return False, None

    try:
        name = await asyncio.wait_for(
            loop.run_in_executor(None, recog_fn),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        logger.error("[AUTH] Face authentication timed out")
        return False, None
    except Exception as e:
        logger.error("[AUTH] Face authentication error: %s", e)
        return False, None

    if name:
        logger.info("[AUTH] Authenticated as: %s", name)
        return True, name
    logger.error("[AUTH] Face authentication FAILED — access denied")
    return False, None


# ═══════════════════════════════════════════════════════════════
# Conversational runtime
# ═══════════════════════════════════════════════════════════════

async def run_leo(no_auth: bool = False) -> None:
    """Initialize subsystems and run the conversation engine."""
    from core.conversation_engine import conversation_engine
    from agent.action_dispatcher import action_dispatcher

    # ── Face auth (mandatory unless --no-auth) ────────────
    if not no_auth:
        ok, name = await authenticate()
        if not ok:
            print("\n  Face authentication failed. Leo cannot start.\n")
            return
        conversation_engine.set_authenticated(name)
    else:
        logger.warning("[AUTH] --no-auth: skipping face authentication (dev mode)")
        conversation_engine.set_authenticated(None)

    # ── Wire vision context + action executor ─────────────
    conversation_engine.set_action_executor(action_dispatcher.execute)
    conversation_engine.set_vision_context(action_dispatcher._screen_context_sync)

    # ── Run engine + watchdog concurrently ────────────────
    engine_task = asyncio.create_task(conversation_engine.run())
    watchdog_task = asyncio.create_task(conversation_engine.timeout_watchdog())

    logger.info("[LEO] Conversational engine running. Say 'hey leo' to start.")
    print("\n  ═══════════════════════════════════════════════")
    print("  Leo is listening. Say 'hey leo' to start talking.")
    print("  Interrupt any time by speaking. Say 'bye' to sleep.")
    print("  ═══════════════════════════════════════════════\n")

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

    try:
        asyncio.run(_main_async(no_auth=args.no_auth))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
