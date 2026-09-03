"""
EventBridge — Thread-safe bridge from Diego pipeline events to Qt signals.

This module connects the existing production pipeline (ConversationEngine,
Brain, EventBus, voice) to the PySide6 UI WITHOUT duplicating any logic.

Design:
    - The bridge is a QObject with Qt signals for each UI event type.
    - Pipeline events (asyncio/threading) are marshalled to the Qt thread
      via a thread-safe queue + QTimer polling (no direct signal emission
      from non-Qt threads).
    - The UI subscribes to Qt signals; the pipeline publishes via emit().

Event types (requirement 4):
    LISTENING, PARTIAL_TRANSCRIPT, FINAL_TRANSCRIPT, THINKING, PLANNING,
    EXECUTING, OBSERVING, VERIFYING, REPLANNING, RESPONSE, ERROR, IDLE

Threading model:
    - emit() can be called from ANY thread (asyncio executor, voice thread,
      Brain task, etc.). Events are queued and delivered on the Qt thread.
    - The Qt event loop is NEVER blocked by pipeline work.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)


class UIEventType(Enum):
    """High-level UI events (requirement 4)."""
    LISTENING = auto()
    PARTIAL_TRANSCRIPT = auto()
    FINAL_TRANSCRIPT = auto()
    THINKING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    OBSERVING = auto()
    VERIFYING = auto()
    REPLANNING = auto()
    SPEAKING = auto()
    RESPONSE = auto()
    RESPONSE_CHUNK = auto()  # Streaming response chunk
    ERROR = auto()
    IDLE = auto()
    STATE_CHANGE = auto()  # Generic state change (for status display)
    AUDIO_LEVEL = auto()   # Microphone audio level for waveform
    TTS_LEVEL = auto()     # TTS output level for speaking visualization
    WAKE_DETECTED = auto()  # Wake word detected
    AUTH_REQUIRED = auto()  # Face authentication required
    AUTH_COMPLETED = auto()  # Face authentication completed


@dataclass
class UIEvent:
    """A single UI event with type and payload."""
    type: UIEventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()


class EventBridge(QObject):
    """
    Thread-safe event bridge from pipeline to Qt UI.

    Usage (UI side):
        bridge = EventBridge()
        bridge.partial_transcript.connect(on_partial)
        bridge.final_transcript.connect(on_final)
        bridge.state_changed.connect(on_state)

    Usage (pipeline side — from any thread):
        bridge.emit(UIEvent(UIEventType.PARTIAL_TRANSCRIPT, {"text": "hel"}))
        bridge.emit_partial("hello wor")
        bridge.emit_final("hello world")
        bridge.emit_state("Thinking")
    """

    # Qt signals — emitted on the Qt thread only
    listening = Signal()
    partial_transcript = Signal(str)  # partial text
    final_transcript = Signal(str)    # final text
    thinking = Signal()
    planning = Signal()
    executing = Signal()
    observing = Signal()
    verifying = Signal()
    replanning = Signal()
    speaking = Signal()               # Diego is speaking (TTS active)
    response = Signal(str)            # complete response
    response_chunk = Signal(str)      # streaming chunk
    error = Signal(str)               # friendly error message
    idle = Signal()
    state_changed = Signal(str)       # high-level state name
    audio_level = Signal(float)       # 0.0 - 1.0 (microphone input)
    tts_level = Signal(float)         # 0.0 - 1.0 (TTS output)
    wake_detected = Signal()
    auth_required = Signal()
    auth_completed = Signal(str)      # authenticated user name

    # Internal signal for queue processing
    _process_queue = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._queue: queue.Queue[UIEvent] = queue.Queue()
        self._lock = threading.Lock()

        # Timer to poll the queue on the Qt thread (low CPU when idle)
        self._timer = QTimer(self)
        self._timer.setInterval(30)  # ~33 FPS is plenty for UI updates
        self._timer.timeout.connect(self._drain_queue)

        # Also use a signal for immediate wake-up when events arrive
        self._process_queue.connect(self._drain_queue)

        self._started = False

    def start(self) -> None:
        """Start the event bridge (call from Qt thread)."""
        if not self._started:
            self._timer.start()
            self._started = True
            logger.info("[UI-BRIDGE] Event bridge started")

    def stop(self) -> None:
        """Stop the event bridge."""
        if self._started:
            self._timer.stop()
            self._started = False
            logger.info("[UI-BRIDGE] Event bridge stopped")

    # ── Thread-safe emit (can be called from ANY thread) ──────────

    def emit(self, event: UIEvent) -> None:
        """
        Queue a UI event for delivery on the Qt thread.

        This method is THREAD-SAFE and can be called from:
        - asyncio tasks (ConversationEngine, Brain)
        - executor threads (voice, LLM)
        - any worker thread
        """
        self._queue.put(event)
        # Wake up the Qt event loop if we're running
        # (emit the signal — Qt handles cross-thread signal delivery)
        try:
            self._process_queue.emit()
        except RuntimeError:
            pass  # Object deleted — ignore

    # ── Convenience emitters for common events ────────────────────

    def emit_listening(self) -> None:
        self.emit(UIEvent(UIEventType.LISTENING))

    def emit_partial(self, text: str) -> None:
        self.emit(UIEvent(UIEventType.PARTIAL_TRANSCRIPT, {"text": text}))

    def emit_final(self, text: str) -> None:
        self.emit(UIEvent(UIEventType.FINAL_TRANSCRIPT, {"text": text}))

    def emit_thinking(self) -> None:
        self.emit(UIEvent(UIEventType.THINKING))

    def emit_planning(self) -> None:
        self.emit(UIEvent(UIEventType.PLANNING))

    def emit_executing(self, action: str = "") -> None:
        self.emit(UIEvent(UIEventType.EXECUTING, {"action": action}))

    def emit_observing(self) -> None:
        self.emit(UIEvent(UIEventType.OBSERVING))

    def emit_verifying(self) -> None:
        self.emit(UIEvent(UIEventType.VERIFYING))

    def emit_replanning(self) -> None:
        self.emit(UIEvent(UIEventType.REPLANNING))

    def emit_speaking(self) -> None:
        self.emit(UIEvent(UIEventType.SPEAKING))

    def emit_response(self, text: str) -> None:
        self.emit(UIEvent(UIEventType.RESPONSE, {"text": text}))

    def emit_response_chunk(self, chunk: str) -> None:
        self.emit(UIEvent(UIEventType.RESPONSE_CHUNK, {"chunk": chunk}))

    def emit_error(self, message: str) -> None:
        """Emit a friendly error message (never raw exceptions)."""
        self.emit(UIEvent(UIEventType.ERROR, {"message": message}))

    def emit_idle(self) -> None:
        self.emit(UIEvent(UIEventType.IDLE))

    def emit_state(self, state: str) -> None:
        self.emit(UIEvent(UIEventType.STATE_CHANGE, {"state": state}))

    def emit_audio_level(self, level: float) -> None:
        self.emit(UIEvent(UIEventType.AUDIO_LEVEL, {"level": level}))

    def emit_tts_level(self, level: float) -> None:
        self.emit(UIEvent(UIEventType.TTS_LEVEL, {"level": level}))

    def emit_wake_detected(self) -> None:
        self.emit(UIEvent(UIEventType.WAKE_DETECTED))

    def emit_auth_required(self) -> None:
        self.emit(UIEvent(UIEventType.AUTH_REQUIRED))

    def emit_auth_completed(self, name: str) -> None:
        self.emit(UIEvent(UIEventType.AUTH_COMPLETED, {"name": name}))

    # ── Qt-thread queue processing ────────────────────────────────

    def _drain_queue(self) -> None:
        """Process all queued events on the Qt thread."""
        processed = 0
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self._dispatch(event)
            processed += 1
            # Safety: don't process too many events in one batch
            if processed > 100:
                # Re-schedule to keep UI responsive
                QTimer.singleShot(0, self._drain_queue)
                break

    def _dispatch(self, event: UIEvent) -> None:
        """Dispatch a single event to the appropriate Qt signal."""
        try:
            if event.type == UIEventType.LISTENING:
                self.listening.emit()
                self.state_changed.emit("Listening")

            elif event.type == UIEventType.PARTIAL_TRANSCRIPT:
                self.partial_transcript.emit(event.data.get("text", ""))

            elif event.type == UIEventType.FINAL_TRANSCRIPT:
                self.final_transcript.emit(event.data.get("text", ""))

            elif event.type == UIEventType.THINKING:
                self.thinking.emit()
                self.state_changed.emit("Thinking")

            elif event.type == UIEventType.PLANNING:
                self.planning.emit()
                self.state_changed.emit("Planning")

            elif event.type == UIEventType.EXECUTING:
                self.executing.emit()
                action = event.data.get("action", "")
                self.state_changed.emit(f"Executing" + (f": {action}" if action else ""))

            elif event.type == UIEventType.OBSERVING:
                self.observing.emit()
                self.state_changed.emit("Observing")

            elif event.type == UIEventType.VERIFYING:
                self.verifying.emit()
                self.state_changed.emit("Verifying")

            elif event.type == UIEventType.REPLANNING:
                self.replanning.emit()
                self.state_changed.emit("Replanning")

            elif event.type == UIEventType.SPEAKING:
                self.speaking.emit()
                self.state_changed.emit("Speaking")

            elif event.type == UIEventType.RESPONSE:
                self.response.emit(event.data.get("text", ""))

            elif event.type == UIEventType.RESPONSE_CHUNK:
                self.response_chunk.emit(event.data.get("chunk", ""))

            elif event.type == UIEventType.ERROR:
                self.error.emit(event.data.get("message", "Something went wrong."))
                self.state_changed.emit("Error")

            elif event.type == UIEventType.IDLE:
                self.idle.emit()
                self.state_changed.emit("Idle")

            elif event.type == UIEventType.STATE_CHANGE:
                self.state_changed.emit(event.data.get("state", "Unknown"))

            elif event.type == UIEventType.AUDIO_LEVEL:
                self.audio_level.emit(event.data.get("level", 0.0))

            elif event.type == UIEventType.TTS_LEVEL:
                self.tts_level.emit(event.data.get("level", 0.0))

            elif event.type == UIEventType.WAKE_DETECTED:
                self.wake_detected.emit()

            elif event.type == UIEventType.AUTH_REQUIRED:
                self.auth_required.emit()
                self.state_changed.emit("Authenticating")

            elif event.type == UIEventType.AUTH_COMPLETED:
                self.auth_completed.emit(event.data.get("name", ""))

        except Exception as e:
            logger.error("[UI-BRIDGE] Event dispatch error: %s", e)


# ── Pipeline integration hooks ──────────────────────────────────────
# These functions wire the existing pipeline to the event bridge.
# They are called once during UI startup.

def wire_conversation_engine(bridge: EventBridge) -> None:
    """
    Wire the ConversationEngine state transitions to the UI bridge.

    This patches the engine's _set_state method to emit UI events.
    The original method is preserved — we only add event emission.
    """
    try:
        from core.conversation_engine import conversation_engine, EngineState

        original_set_state = conversation_engine._set_state

        def patched_set_state(new_state: EngineState, **diag) -> None:
            # Call the original first
            original_set_state(new_state, **diag)

            # Then emit UI events based on the new state
            if new_state == EngineState.IDLE:
                bridge.emit_idle()
            elif new_state == EngineState.WAKE:
                bridge.emit_state("Waiting for wake word")
            elif new_state == EngineState.FACE_AUTH:
                bridge.emit_auth_required()
            elif new_state == EngineState.LISTEN:
                bridge.emit_listening()
            elif new_state == EngineState.THINK:
                bridge.emit_thinking()
            elif new_state == EngineState.SPEAK:
                bridge.emit_state("Responding")

        conversation_engine._set_state = patched_set_state
        logger.info("[UI-BRIDGE] ConversationEngine wired to UI")

    except Exception as e:
        logger.warning("[UI-BRIDGE] Could not wire ConversationEngine: %s", e)


def wire_event_bus(bridge: EventBridge) -> None:
    """
    Wire the existing EventBus to forward relevant events to the UI.

    Subscribes to Brain/TaskController lifecycle events and maps them
    to UI state changes.
    """
    try:
        from core.event_bus import bus

        async def on_event(event) -> None:
            """Forward EventBus events to the UI bridge."""
            etype = event.type

            # Brain/Planner/Task events → UI states
            if etype in ("task:started", "goal:started"):
                bridge.emit_executing(event.data.get("description", ""))
            elif etype in ("task:completed", "goal:completed"):
                bridge.emit_verifying()
            elif etype in ("task:failed", "goal:failed"):
                bridge.emit_error(event.data.get("error", "Task failed"))
            elif etype == "planning:started":
                bridge.emit_planning()
            elif etype == "planning:replan":
                bridge.emit_replanning()
            elif etype == "observation:completed":
                bridge.emit_observing()

        # Register for all events (wildcard) and filter
        bus.on("*", on_event)
        logger.info("[UI-BRIDGE] EventBus wired to UI")

    except Exception as e:
        logger.warning("[UI-BRIDGE] Could not wire EventBus: %s", e)


def wire_stt_events(bridge: EventBridge) -> None:
    """
    Wire STT partial/final transcripts to the UI.

    This patches the command_listener's event emission to also
    send transcripts to the UI bridge.
    """
    # The ConversationEngine processes UtteranceEvents in _conversation_session.
    # We patch the engine's handling to emit UI events for partials/finals.
    try:
        from core.conversation_engine import conversation_engine

        original_stt_pump = conversation_engine._stt_event_pump

        async def patched_stt_pump(stream, events) -> None:
            """Patched STT pump that also emits UI events."""
            try:
                async for ev in stream:
                    if not conversation_engine._running:
                        break

                    # Emit UI events for transcripts
                    if ev.kind == "partial":
                        bridge.emit_partial(ev.text)
                    elif ev.kind == "final":
                        bridge.emit_final(ev.text)
                    elif ev.kind == "speech_start":
                        bridge.emit_listening()

                    # Forward to the original queue
                    events.put_nowait(ev)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning("[STT] Stream error: %s", e)

        conversation_engine._stt_event_pump = patched_stt_pump
        logger.info("[UI-BRIDGE] STT events wired to UI")

    except Exception as e:
        logger.warning("[UI-BRIDGE] Could not wire STT events: %s", e)


def wire_brain_events(bridge: EventBridge) -> None:
    """
    Wire Brain pipeline stages to UI state events.

    Patches Brain.process_command to emit thinking/planning/executing
    states as the pipeline progresses.
    """
    try:
        from agent.brain import agent_brain

        original_process = agent_brain.process_command

        async def patched_process(text: str, **kwargs):
            """Patched process_command that emits UI state events."""
            bridge.emit_thinking()
            try:
                result = await original_process(text, **kwargs)
                # Emit the final response for UI display
                if result.response:
                    bridge.emit_response(result.response)
                return result
            except Exception as e:
                bridge.emit_error("I had trouble processing that.")
                raise

        agent_brain.process_command = patched_process
        logger.info("[UI-BRIDGE] Brain events wired to UI")

    except Exception as e:
        logger.warning("[UI-BRIDGE] Could not wire Brain events: %s", e)


def wire_all(bridge: EventBridge) -> None:
    """Wire all pipeline components to the UI bridge."""
    wire_conversation_engine(bridge)
    wire_event_bus(bridge)
    wire_stt_events(bridge)
    # Note: Brain wiring is done carefully to avoid double-response
    # wire_brain_events(bridge)