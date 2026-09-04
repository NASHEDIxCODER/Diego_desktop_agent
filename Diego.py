"""
Diego — Conversational Desktop Agent (production entry point).

This is the NEW Diego: a real conversational companion like Siri /
ChatGPT Voice / Gemini Live — NOT a command executor.

  python Diego.py              # Start conversational Diego
  python Diego.py --no-auth    # Skip face auth (dev only)
  python Diego.py --status     # Show subsystem status

Pipeline:
  Wake ("Diego" / "hey Diego" / "hello Diego")
    → Transcript verification
    → Face auth (mandatory — camera opens ONLY here, waits forever)
    → Streaming VAD
    → Streaming Whisper (partials)
    → Streaming LLM (sentence-by-sentence)
    → Streaming TTS (interruptible)
    → Full duplex: interrupt Diego any time by speaking
    → 60 s of silence / "goodbye" / "stop listening" / "cancel"
    → Back to wake listening (Diego NEVER exits on its own)

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
# Tesseract language data: prefer the copy bundled with Diego
# (data/tessdata/eng.traineddata) when the system tessdata is missing.
_TESSDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "tessdata")
if os.path.exists(os.path.join(_TESSDATA_DIR, "eng.traineddata")):
    os.environ.setdefault("TESSDATA_PREFIX", _TESSDATA_DIR)
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
logger = logging.getLogger("Diego")


# ═══════════════════════════════════════════════════════════════
# Face authentication — runs ONLY after the wake word, NEVER at startup.
# ═══════════════════════════════════════════════════════════════

async def authenticate_on_wake() -> Optional[str]:
    """
    Run live face authentication in the popup window.

    This is the engine's auth provider. It is called ONLY after the wake
    word is detected (and only when the previous session has expired).

    Returns the verified user's name, or None if denied/cancelled.
    NEVER raises, NEVER terminates Diego.
    """
    loop = asyncio.get_event_loop()
    try:
        from auth.live_auth import authenticate_live
    except Exception as e:
        # Honest failure reporting: this auth ATTEMPT failed, but face
        # authentication is NOT disabled — the provider stays active and
        # the next gate retries. Never equivalent to --no-auth.
        logger.error("[AUTH] live_auth unavailable: %s — face authentication "
                     "FAILED for this attempt; auth remains ENABLED (this is "
                     "NOT --no-auth)", e)
        return None
    try:
        # authenticate_live blocks on the camera → run in an executor thread.
        # No timeout here: the popup waits forever for a face; the user can
        # close the popup to cancel (which returns None → deny → wake).
        name = await loop.run_in_executor(None, authenticate_live)
        return name
    except Exception as e:
        logger.error("[AUTH] live authentication error: %s — face authentication "
                     "FAILED for this attempt; auth remains ENABLED (this is "
                     "NOT --no-auth)", e)
        return None


# ═══════════════════════════════════════════════════════════════
# Conversational runtime
# ═══════════════════════════════════════════════════════════════

async def _warm_llm() -> None:
    """Asynchronously warm the local LLM BEFORE the first user command.

    Scheduled as a background task at startup (never per wake). Fully
    fail-safe: if Ollama is unavailable or slow, Diego still boots and
    the LLM simply loads on first use. Configurable via settings:
      LLM_WARMUP_ENABLED / LLM_WARMUP_TIMEOUT_S / LLM_WARMUP_VISION
    """
    from config.settings import settings
    if not settings.LLM_WARMUP_ENABLED:
        logger.info("[LLM-WARMUP] Disabled by configuration")
        return
    try:
        from agent.streaming_llm import streaming_llm
        t0 = time.time()
        ok = await asyncio.wait_for(
            streaming_llm.warm_up(), timeout=settings.LLM_WARMUP_TIMEOUT_S)
        if ok:
            logger.info("[LLM-WARMUP] Model ready before first command "
                        "(%.1fs, keep_alive=%s)",
                        time.time() - t0, settings.OLLAMA_KEEP_ALIVE)
        else:
            logger.info("[LLM-WARMUP] Skipped (Ollama unavailable) — "
                        "model will load on first use")
    except asyncio.TimeoutError:
        logger.warning("[LLM-WARMUP] Timed out after %.0fs (non-fatal)",
                       settings.LLM_WARMUP_TIMEOUT_S)
    except Exception as e:
        logger.warning("[LLM-WARMUP] Failed (non-fatal): %s", e)


async def run_Diego(no_auth: bool = False, no_wake: bool = False) -> None:
    """
    BOOT → LOAD MODELS → INIT AUDIO → WAIT_WAKE.

    Diego ALWAYS boots successfully and stays alive forever. Face auth is
    deferred until the wake word. There is NO startup authentication.

    The conversation engine owns the real boot sequence (audio → wake
    model → VAD → TTS → Whisper) and logs each stage as it completes.
    """
    # ── Runtime dependency/asset diagnostic ──────────────────────
    # Reports [HEALTH] component=OK/DEGRADED/MISSING for every
    # production component. Runs ONCE — never retries a permanently
    # missing component (no retry loop).
    from core.runtime_health import run_runtime_health
    run_runtime_health(no_wake=no_wake)

    from core.conversation_engine import conversation_engine
    from agent.action_dispatcher import action_dispatcher
    from services.vision_service import vision_service
    from services.search_service import search_service
    from services.screen_capture import screen_capture_service

    # ── Wire subsystems ───────────────────────────────────
    # ConversationEngine ONLY speaks. The Brain is the single
    # orchestrator for perceive → decide → plan → dispatch →
    # verify → learn → respond.

    # Brain — the single orchestrator
    from agent.brain import agent_brain
    await agent_brain.initialize()

    # ── LLM warm-up (async, non-blocking, fail-safe) ──
    # Loads the Ollama model in the background so it is resident BEFORE
    # the user's first command. Never blocks critical startup and never
    # runs per wake — scheduled exactly once here.
    asyncio.create_task(_warm_llm())

    # Wire the command router (classifies only — Brain dispatches)
    from core.command_router import command_router
    command_router.wire(action_dispatcher=action_dispatcher, conversation_engine=conversation_engine)

    # ── Initialize new services ───────────────────────────
    # MusicAgent: unified music control (MPV, Spotify, YouTube, local)
    from services.music_agent import music_agent
    try:
        await music_agent.initialize()
        logger.info("[DIEGO] MusicAgent initialized")
    except Exception as e:
        logger.debug("[DIEGO] MusicAgent init skipped: %s", e)

    # Augment the learning engine callable with the composer for richer context
    from learning.learning_engine import learning_engine
    # Keep backward compatibility: the engine's llm_context() still works
    # but ContextComposer now handles ranking/compression in the LLM pipe.

    # ── Face auth provider (deferred to wake) ─────────────
    if no_auth:
        logger.warning("[AUTH] --no-auth: face authentication disabled (dev mode)")
        # CRITICAL FIX (2026-08-31): set_auth_disabled() clears the auth
        # provider so _needs_auth() always returns False. The old
        # set_authenticated(None) left the provider set, so _needs_auth()
        # still returned True (because _auth_user is None) and the camera
        # would still open even with --no-auth.
        conversation_engine.set_auth_disabled()   # dev session, no camera
    else:
        conversation_engine.set_auth_provider(authenticate_on_wake)

    # ── Start background learner (idle-time self-improvement) ──
    from core.background_learning import background_learner
    background_learner.wire(conversation_engine=conversation_engine)
    await background_learner.start()

    # ── Run engine + watchdog concurrently ────────────────
    engine_task = asyncio.create_task(conversation_engine.run(no_wake=no_wake))

    logger.info("Listening for wake word...")

    try:
        await engine_task
    except asyncio.CancelledError:
        pass
    finally:
        engine_task.cancel()
        await asyncio.gather(engine_task, return_exceptions=True)
        await background_learner.stop()


# ═══════════════════════════════════════════════════════════════
# Graceful shutdown
# ═══════════════════════════════════════════════════════════════

async def shutdown() -> None:
    """Stop all subsystems cleanly."""
    logger.info("[DIEGO] Shutting down...")

    # ── Save session recording before tearing down ──
    try:
        from core.manual_session_recorder import session_recorder
        if session_recorder.enabled:
            saved_path = session_recorder.save()
            if saved_path:
                summary = session_recorder.get_summary()
                logger.info("[DIEGO] Session summary: %d turns, avg wake=%.0fms, avg turn=%.0fms",
                            summary.get("total_turns", 0),
                            summary.get("avg_wake_latency_ms", 0),
                            summary.get("avg_total_latency_ms", 0))
    except Exception as e:
        logger.debug("session recorder shutdown: %s", e)

    try:
        from core.conversation_engine import conversation_engine
        conversation_engine._running = False
    except Exception as e:
        logger.debug("engine shutdown: %s", e)

    try:
        from voice.streaming_tts import streaming_tts
        streaming_tts.close()
    except Exception as e:
        logger.debug("tts shutdown: %s", e)

    try:
        from voice.command_listener import command_listener
        command_listener.cancel()
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

    logger.info("[DIEGO] Shutdown complete.")


async def _main_async(no_auth: bool, no_wake: bool = False, record_session: bool = False) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _signal(*_):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal)
        except (NotImplementedError, RuntimeError):
            pass

    # ── Enable session recording if requested ──
    if record_session or os.environ.get("DIEGO_RECORD_SESSION", "").strip() in ("1", "true", "yes"):
        from core.manual_session_recorder import session_recorder
        session_recorder.enable()

    run_task = asyncio.create_task(run_Diego(no_auth=no_auth, no_wake=no_wake))
    stop_task = asyncio.create_task(stop.wait())

    done, pending = await asyncio.wait(
        {run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

    for t in pending:
        t.cancel()
    try:
        # Shutdown must have a hard upper timeout (Phase 12) so a hung
        # worker cannot prevent Diego from exiting.
        await asyncio.wait_for(shutdown(), timeout=15.0)
    except asyncio.TimeoutError:
        logger.error("[SHUTDOWN] Hard timeout exceeded 15s — exiting anyway")
    # Drain cancelled tasks
    await asyncio.gather(*pending, return_exceptions=True)


def cmd_status() -> None:
    """Print a quick subsystem status report."""
    print("Diego subsystem status")
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


async def cmd_debug_vision() -> None:
    """
    `Diego debug vision` — Open the live debug overlay.

    Runs the forensic vision pipeline continuously and renders
    the debug overlay showing:
      - Green boxes: detected UI buttons/interactables
      - Red boxes: discarded OCR
      - White boxes: layout regions
      - Magenta: planner target
      - Cyan: mouse position
      - Window name, frame hash, pipeline latency
      - OCR confidence per box
    """
    from services.vision_service import vision_service
    from services.screen_capture import screen_capture_service
    from vision.debug_overlay import debug_overlay
    from vision.forensic_logger import forensic_logger

    # Initialize capture
    print("Initializing screen capture...")
    await screen_capture_service._start()
    if not screen_capture_service.ready:
        print("ERROR: Screen capture backend not available. Install mss: pip install mss")
        return

    # Initialize vision service
    print("Initializing vision service...")
    await vision_service._start()

    # Enable overlay
    debug_overlay.toggle()
    print("\n" + "=" * 60)
    print("  DIEGO DEBUG VISION — Live Overlay Active")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()
    print("Overlay shows:")
    print("  Green boxes  = Buttons / clickable elements")
    print("  Red boxes    = Discarded OCR (didn't become UI elements)")
    print("  White boxes  = Layout regions (toolbar, sidebar, editor...)")
    print("  Magenta      = Planner target")
    print("  Cyan         = Mouse position")
    print()

    try:
        frame = 0
        while True:
            frame += 1
            print(f"\r  Frame #{frame} — capturing...", end="", flush=True)

            ctx, report = await vision_service.analyze_forensic(force=True)

            # Print per-stage summary
            print(f"\r  Frame #{frame}: {ctx.capture.width}x{ctx.capture.height} | "
                  f"App: {ctx.app_type}/{ctx.app_name} | "
                  f"OCR: {ctx.ocr_result.box_count_final if ctx.ocr_result else 0} boxes | "
                  f"UI: {len(ctx.desktop.walk()) if ctx.desktop else 0} elements | "
                  f"Page: {ctx.page_type} | "
                  f"Total: {report.total_latency_ms:.1f}ms")

            if report.error_count > 0:
                for s in report.stages:
                    if not s.success:
                        print(f"    ✗ Stage {s.stage} {s.name}: {s.failure_reason}")

            # Update overlay
            debug_overlay.update_from_context(
                ctx,
                planner_target="",
                status=f"Frame #{frame} | {ctx.app_type}/{ctx.app_name} | "
                       f"OCR: {ctx.ocr_result.box_count_final if ctx.ocr_result else 0} boxes | "
                       f"UI: {len(ctx.desktop.walk()) if ctx.desktop else 0} el | "
                       f"{report.total_latency_ms:.1f}ms"
            )

            await asyncio.sleep(0.5)  # 2 FPS is enough for debugging

    except KeyboardInterrupt:
        print("\n\nShutting down debug overlay...")
    finally:
        debug_overlay.close()
        await vision_service._stop()
        await screen_capture_service._stop()
        print("Debug vision session ended.")


async def cmd_inspect_screen() -> None:
    """
    `Diego inspect screen` — Comprehensive screen inspection report.

    Outputs:
      - Application name and type
      - Window title
      - Detected controls (by type)
      - Buttons with positions and confidence
      - Inputs
      - Menus
      - Dialogs
      - Notifications
      - OCR confidence and text preview
      - Layout regions
      - Pipeline stats
    """
    from services.vision_service import vision_service
    from services.screen_capture import screen_capture_service
    from vision.forensic_logger import inspect_screen

    # Initialize
    print("Initializing screen capture...")
    await screen_capture_service._start()
    if not screen_capture_service.ready:
        print("ERROR: Screen capture backend not available.")
        return

    print("Initializing vision service...")
    await vision_service._start()

    print("Capturing and analyzing screen (this may take 1-2 seconds)...")
    print()

    ctx, report = await vision_service.analyze_forensic(force=True)

    # Print forensic report
    print(report.to_text())
    print()

    # Print screen inspection
    print(inspect_screen(ctx))

    # Cleanup
    await vision_service._stop()
    await screen_capture_service._stop()


def run(no_auth: bool = False, no_wake: bool = False, record_session: bool = False) -> None:
    """Canonical blocking entry point — boots Diego and runs until Ctrl+C.

    Used by BOTH `python Diego.py` and `python main.py` so there is exactly
    ONE runtime entry path.
    """
    try:
        asyncio.run(_main_async(no_auth=no_auth, no_wake=no_wake, record_session=record_session))
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Diego — Conversational Desktop Agent")
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # `Diego debug vision` — live debug overlay
    sub.add_parser("debug", help="Debug tools").add_argument(
        "target", nargs="?", choices=["vision"], default="vision",
        help="Debug target (default: vision)")

    # `Diego inspect screen` — comprehensive screen report
    sub.add_parser("inspect", help="Inspect tools").add_argument(
        "target", nargs="?", choices=["screen"], default="screen",
        help="Inspect target (default: screen)")

    parser.add_argument("--no-auth", action="store_true",
                        help="Skip face authentication (development only)")
    parser.add_argument("--no-wake", action="store_true",
                        help="Bypass wake detection only — enter LISTEN directly "
                             "(face auth still runs unless --no-auth is also given; development only)")
    parser.add_argument("--status", action="store_true",
                        help="Show subsystem status and exit")
    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    if args.command == "debug":
        asyncio.run(cmd_debug_vision())
        return

    if args.command == "inspect":
        asyncio.run(cmd_inspect_screen())
        return

    run(no_auth=args.no_auth, no_wake=args.no_wake)


if __name__ == "__main__":
    main()
