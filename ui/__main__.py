"""
Diego UI Entry Point — python -m ui

Launches the Diego desktop UI on top of the existing production pipeline.

Usage:
    python -m ui                    # Full production mode (wake + auth)
    python -m ui --no-wake          # Skip wake word (dev)
    python -m ui --no-auth          # Skip face auth (dev)
    python -m ui --no-wake --no-auth  # Full dev mode

Integration:
    The UI runs the Qt event loop on the main thread. The existing
    ConversationEngine/Brain asyncio pipeline runs in a background thread
    with its own event loop. Events are bridged via EventBridge (thread-safe).

    This preserves the existing production architecture:
    - Voice pipeline (wake/VAD/STT/TTS) unchanged
    - Brain/DecisionEngine/TaskController unchanged
    - Face auth unchanged
    - Knowledge/vision/diagnostics unchanged
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
from typing import Optional

# Environment fixes (same as Diego.py)
os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

# Qt platform plugin (must be set before importing PySide6)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

logger = logging.getLogger("DiegoUI")


def setup_logging() -> None:
    """Configure logging for the UI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class DiegoRuntime:
    """
    Manages the Diego pipeline runtime alongside the Qt UI.

    The asyncio event loop runs in a background thread. The Qt event
    loop runs on the main thread. Events flow via EventBridge.
    """

    def __init__(self, no_wake: bool = False, no_auth: bool = False):
        self.no_wake = no_wake
        self.no_auth = no_auth
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._bridge = None

    def start(self, bridge) -> None:
        """Start the Diego pipeline in a background thread."""
        self._bridge = bridge
        self._running = True
        self._thread = threading.Thread(
            target=self._run_async_loop,
            name="DiegoPipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("[UI-RUNTIME] Diego pipeline thread started")

    def _run_async_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._run_pipeline())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("[UI-RUNTIME] Pipeline error: %s", e)
            if self._bridge:
                self._bridge.emit_error("Diego encountered a problem starting up.")
        finally:
            self.loop.close()

    async def _run_pipeline(self) -> None:
        """Initialize and run the production Diego pipeline."""
        # Import here to avoid heavy imports at UI startup
        from core.runtime_health import run_runtime_health
        run_runtime_health(no_wake=self.no_wake)

        # Restore persisted audio device selections BEFORE the AudioManager
        # opens its stream: input → voice_settings.device_index override
        # (still verified by the existing probe), output → StreamingTTS
        # target. Missing devices fall back to auto-detection safely.
        try:
            from voice.device_manager import device_manager
            device_manager.apply_persisted_devices()
        except Exception as e:
            logger.warning("[UI-RUNTIME] Device restore skipped: %s", e)

        from core.conversation_engine import conversation_engine
        from agent.brain import agent_brain

        # Wire the UI bridge to the pipeline
        from ui.event_bridge import wire_all
        wire_all(self._bridge)

        # Initialize Brain
        await agent_brain.initialize()

        # LLM warm-up (background, non-blocking)
        from config.settings import settings
        if settings.LLM_WARMUP_ENABLED:
            async def warm():
                try:
                    from agent.streaming_llm import streaming_llm
                    await asyncio.wait_for(
                        streaming_llm.warm_up(),
                        timeout=settings.LLM_WARMUP_TIMEOUT_S
                    )
                except Exception:
                    pass
            asyncio.create_task(warm())

        # Auth configuration
        if self.no_auth:
            logger.warning("[UI-RUNTIME] --no-auth: face authentication disabled")
            conversation_engine.set_auth_disabled()
        else:
            from Diego import authenticate_on_wake
            conversation_engine.set_auth_provider(authenticate_on_wake)

        # Wire command router
        from agent.action_dispatcher import action_dispatcher
        from core.command_router import command_router
        command_router.wire(
            action_dispatcher=action_dispatcher,
            conversation_engine=conversation_engine
        )

        # Start background learner
        from core.background_learning import background_learner
        background_learner.wire(conversation_engine=conversation_engine)
        await background_learner.start()

        # Emit ready state
        self._bridge.emit_state("Starting...")

        # Run the conversation engine
        logger.info("[UI-RUNTIME] Starting ConversationEngine (no_wake=%s)", self.no_wake)
        try:
            await conversation_engine.run(no_wake=self.no_wake)
        except asyncio.CancelledError:
            pass
        finally:
            await background_learner.stop()

    def stop(self) -> None:
        """Stop the Diego pipeline."""
        self._running = False
        if self.loop and self.loop.is_running():
            # Cancel all tasks
            async def cancel_all():
                tasks = [t for t in asyncio.all_tasks(self.loop) if not t.done()]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            try:
                asyncio.run_coroutine_threadsafe(cancel_all(), self.loop).result(timeout=5)
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("[UI-RUNTIME] Diego pipeline stopped")


class AudioLevelPoller:
    """
    Polls real audio levels from the existing pipeline for UI visualization.

    Uses the AudioManager's last_frame_rms (microphone) and StreamingTTS's
    is_speaking (TTS activity). Does NOT create a second audio pipeline.
    """

    # RMS normalization: int16 scale (0-32768) → 0-1 for UI
    # Typical speech RMS is 500-3000 on int16 scale
    RMS_MAX = 4000.0

    def __init__(self, bridge):
        self._bridge = bridge
        self._timer = None
        self._audio_manager = None
        self._streaming_tts = None

    def start(self) -> None:
        """Start polling audio levels (call from Qt thread)."""
        from PySide6.QtCore import QTimer

        # Try to get the audio manager and TTS from the pipeline
        try:
            from voice.audio_manager import AudioManager
            # The conversation engine creates the audio manager;
            # we access it via the global instance if available
            from core.conversation_engine import conversation_engine
            if hasattr(conversation_engine, '_audio_manager'):
                self._audio_manager = conversation_engine._audio_manager
        except Exception as e:
            logger.debug("[UI-AUDIO] Could not access AudioManager: %s", e)

        try:
            from voice.streaming_tts import streaming_tts
            self._streaming_tts = streaming_tts
        except Exception as e:
            logger.debug("[UI-AUDIO] Could not access StreamingTTS: %s", e)

        # Poll at ~30 FPS (33ms) — matches audio callback rate
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        logger.info("[UI-AUDIO] Audio level poller started")

    def stop(self) -> None:
        """Stop polling."""
        if self._timer:
            self._timer.stop()
            self._timer = None

    def _poll(self) -> None:
        """Poll audio levels and emit to the bridge."""
        # Microphone input level
        if self._audio_manager is not None:
            try:
                rms = self._audio_manager.last_frame_rms
                # Normalize to 0-1
                level = min(1.0, rms / self.RMS_MAX)
                self._bridge.emit_audio_level(level)
            except Exception:
                pass

        # TTS output level
        if self._streaming_tts is not None:
            try:
                if self._streaming_tts.is_speaking:
                    # While speaking, emit a simulated output level
                    # based on TTS activity. Real per-sample output
                    # level would require hooking into the audio stream.
                    import math
                    import time
                    # Smooth pulsing level while speaking
                    phase = time.time() * 4
                    level = 0.4 + 0.3 * math.sin(phase)
                    self._bridge.emit_tts_level(max(0.1, level))
                else:
                    self._bridge.emit_tts_level(0.0)
            except Exception:
                pass


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the Diego UI.

    `argv` lets other entry points (main.py) forward parsed flags without
    re-parsing sys.argv. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(description="Diego Desktop UI")
    parser.add_argument("--no-wake", action="store_true",
                        help="Bypass wake detection (development)")
    parser.add_argument("--no-auth", action="store_true",
                        help="Skip face authentication (development)")
    parser.add_argument("--ui-only", action="store_true",
                        help="Run UI only without the voice pipeline (testing)")
    args = parser.parse_args(argv)

    setup_logging()
    logger.info("[UI] Starting Diego Desktop UI (no_wake=%s, no_auth=%s, ui_only=%s)",
                args.no_wake, args.no_auth, args.ui_only)

    # Import Qt after environment setup
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt, QTimer

    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Diego")
    app.setOrganizationName("Diego")

    # ── Host the GUI dispatcher on the Qt MAIN thread ──
    # The ConversationEngine runs on the background DiegoPipeline thread
    # and marshals ALL Tk GUI work (face-auth popup) to this thread via
    # gui.submit(). A lightweight QTimer pumps the dispatcher (~30 ms) —
    # no long-running work ever runs on the Qt thread.
    from core.gui_dispatcher import gui
    gui_pump_timer = None
    if gui.start():
        gui_pump_timer = QTimer()
        gui_pump_timer.setInterval(30)
        gui_pump_timer.timeout.connect(gui.pump_once)
        gui_pump_timer.start()
        logger.info("[UI] GUI dispatcher hosted on the Qt main thread "
                    "(id=%s)", gui.main_thread_ident)
    else:
        logger.info("[UI] GUI dispatcher unavailable — face-auth popup "
                    "will fall back to headless mode")

    # Create the event bridge
    from ui.event_bridge import EventBridge
    bridge = EventBridge()

    # Create the main window
    from ui.main_window import DiegoMainWindow
    window = DiegoMainWindow(bridge=bridge, loop=None)

    # Connect TTS level to the window
    bridge.tts_level.connect(window.set_output_level)

    # Audio level poller (uses existing pipeline, no second capture)
    audio_poller: Optional[AudioLevelPoller] = None

    # Start the pipeline (unless UI-only mode)
    runtime: Optional[DiegoRuntime] = None
    if not args.ui_only:
        runtime = DiegoRuntime(no_wake=args.no_wake, no_auth=args.no_auth)
        runtime.start(bridge)

        # Update window with the runtime's loop once it's created
        def update_loop():
            if runtime.loop:
                window._loop = runtime.loop

        QTimer.singleShot(500, update_loop)

        # Start audio level polling after the pipeline initializes
        def start_audio_poller():
            nonlocal audio_poller
            audio_poller = AudioLevelPoller(bridge)
            audio_poller.start()

        QTimer.singleShot(2000, start_audio_poller)
    else:
        # UI-only mode: emit idle state
        bridge.emit_idle()

    window.show()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("[UI] Received signal %s, shutting down...", sig)
        if runtime:
            runtime.stop()
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the Qt event loop
    exit_code = app.exec()

    # Cleanup
    if runtime:
        runtime.stop()
    bridge.stop()
    if gui_pump_timer is not None:
        gui_pump_timer.stop()
    gui.stop()

    logger.info("[UI] Diego Desktop UI exited with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())