"""
DiegoMainWindow — Premium voice-first desktop assistant HUD.

This is NOT a chat application. The primary interaction is:
    MICROPHONE → LIVE TRANSCRIPT → DIEGO RESPONSE PRINTED
    → DIEGO RESPONSE SPOKEN → LISTEN AGAIN

Layout:
    HEADER:          Diego identity + "Voice Assistant" + status + controls
    CENTRAL CORE:    Large circular voice visualizer (hero element)
    TRANSCRIPT:      "YOU SAID" region (live partial → final)
    RESPONSE:        "DIEGO" region (prominent, with speaking indicator)
    ACTIVITY PANEL:  Right-side high-level activity display
    METRICS:         Compact latency cards (STT/Agent/TTS/Total)
    SYSTEM STATUS:   Footer health display (STT ✓ Agent ✓ TTS ✓ Tools ✓)

Threading:
    - All pipeline events arrive via EventBridge (thread-safe)
    - The Qt event loop is NEVER blocked
    - No mouse/keyboard interaction required for normal operation
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QSizePolicy, QApplication, QSpacerItem,
)

from ui.event_bridge import EventBridge
from ui.styles import MAIN_WINDOW_QSS, COLORS, FONTS
from ui.visualizer import VoiceCoreVisualizer, VisualizerState
from ui.widgets import (
    ConnectionIndicator, TranscriptPanel, ResponsePanel,
    ActivityPanel, MetricsCards, SystemStatus, HistoryPanel,
)

logger = logging.getLogger(__name__)


class DiegoMainWindow(QMainWindow):
    """
    Premium voice-first Diego desktop assistant HUD.

    The window subscribes to EventBridge signals and displays:
    - Central voice core visualizer (reacts to real audio)
    - Current voice state (IDLE, LISTENING, THINKING, SPEAKING, etc.)
    - Live partial STT transcript (updating in real time)
    - Final user transcript (replaces partial)
    - Diego's response (printed before/during TTS, remains visible)
    - Speaking indicator (Diego is currently speaking)
    - Activity panel (human-readable high-level activity)
    - Latency metrics (STT/Agent/TTS/Total)
    - System status (component health)
    """

    # Signal for scheduling asyncio work from the Qt thread
    _run_async = Signal(object)

    def __init__(
        self,
        bridge: EventBridge,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._bridge = bridge
        self._loop = loop

        # Transcript state
        self._partial_text: str = ""
        self._final_text: str = ""
        self._response_text: str = ""

        # Latency tracking
        self._stt_latency_ms: float = 0.0
        self._agent_latency_ms: float = 0.0
        self._tts_latency_ms: float = 0.0
        self._total_latency_ms: float = 0.0
        self._turn_start_time: Optional[float] = None
        self._tts_start_time: Optional[float] = None

        # Speaking state
        self._is_speaking = False

        self._setup_ui()
        self._connect_signals()

        # Start the event bridge
        self._bridge.start()

    def _setup_ui(self) -> None:
        """Build the premium HUD layout."""
        self.setWindowTitle("Diego")
        self.setMinimumSize(680, 620)
        self.resize(780, 720)

        # Remove native title bar for custom controls
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        # Apply stylesheet
        self.setStyleSheet(MAIN_WINDOW_QSS)

        # Central widget
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── LEFT: Main content column ──────────────────────────
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # ── HEADER ──
        header = self._build_header()
        left_layout.addWidget(header)

        # ── CENTRAL VOICE CORE ──
        voice_core_area = self._build_voice_core()
        left_layout.addWidget(voice_core_area, 3)

        # ── TRANSCRIPT ──
        self._transcript_panel = TranscriptPanel()
        left_layout.addWidget(self._transcript_panel)

        # ── RESPONSE ──
        self._response_panel = ResponsePanel()
        left_layout.addWidget(self._response_panel, 1)

        # ── METRICS ──
        self._metrics = MetricsCards()
        left_layout.addWidget(self._metrics)

        # ── SYSTEM STATUS ──
        self._system_status = SystemStatus()
        left_layout.addWidget(self._system_status)

        main_layout.addWidget(left_column, 1)

        # ── RIGHT: Activity panel ──────────────────────────────
        self._activity_panel = ActivityPanel()
        main_layout.addWidget(self._activity_panel)

    def _build_header(self) -> QFrame:
        """Build the header: Diego identity + status + controls."""
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 16, 10)
        layout.setSpacing(12)

        # Logo/name
        title_container = QVBoxLayout()
        title_container.setSpacing(0)

        title_label = QLabel("DIEGO")
        title_label.setObjectName("titleLabel")
        title_container.addWidget(title_label)

        subtitle_label = QLabel("Voice Assistant")
        subtitle_label.setObjectName("subtitleLabel")
        title_container.addWidget(subtitle_label)

        layout.addLayout(title_container)
        layout.addStretch()

        # Connection indicator
        self._connection = ConnectionIndicator()
        layout.addWidget(self._connection)

        layout.addSpacing(12)

        # Window controls
        self._minimize_btn = QLabel("─")
        self._minimize_btn.setObjectName("minimizeButton")
        self._minimize_btn.setFixedSize(32, 32)
        self._minimize_btn.setAlignment(Qt.AlignCenter)
        self._minimize_btn.mousePressEvent = lambda e: self.showMinimized()
        layout.addWidget(self._minimize_btn)

        self._close_btn = QLabel("✕")
        self._close_btn.setObjectName("closeButton")
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.setAlignment(Qt.AlignCenter)
        self._close_btn.mousePressEvent = lambda e: self.close()
        layout.addWidget(self._close_btn)

        return header

    def _build_voice_core(self) -> QFrame:
        """Build the central voice core area (hero element)."""
        area = QFrame()
        area.setObjectName("voiceCoreArea")

        layout = QVBoxLayout(area)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(8)

        # State label (above visualizer)
        self._state_label = QLabel("IDLE")
        self._state_label.setObjectName("stateLabel")
        self._state_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._state_label)

        # Voice core visualizer (hero)
        self._visualizer = VoiceCoreVisualizer()
        layout.addWidget(self._visualizer, 1)

        return area

    def _connect_signals(self) -> None:
        """Connect EventBridge signals to UI handlers."""
        bridge = self._bridge

        # Transcript events
        bridge.partial_transcript.connect(self._on_partial_transcript)
        bridge.final_transcript.connect(self._on_final_transcript)
        bridge.response.connect(self._on_response)

        # State events
        bridge.state_changed.connect(self._on_state_changed)
        bridge.listening.connect(self._on_listening)
        bridge.thinking.connect(self._on_thinking)
        bridge.planning.connect(self._on_planning)
        bridge.executing.connect(self._on_executing)
        bridge.observing.connect(self._on_observing)
        bridge.verifying.connect(self._on_verifying)
        bridge.replanning.connect(self._on_replanning)
        bridge.speaking.connect(self._on_speaking)
        bridge.idle.connect(self._on_idle)
        bridge.error.connect(self._on_error)

        # Audio events
        bridge.audio_level.connect(self._on_audio_level)

        # Auth events
        bridge.auth_required.connect(self._on_auth_required)
        bridge.auth_completed.connect(self._on_auth_completed)

        # Async work signal
        self._run_async.connect(self._execute_async)

    # ── Transcript handlers ─────────────────────────────────────

    @Slot(str)
    def _on_partial_transcript(self, text: str) -> None:
        """
        Handle partial STT transcript.

        Updates the single live transcript region in real time.
        """
        if not text.strip():
            return

        self._partial_text = text
        self._transcript_panel.set_partial(text)

        # Update activity: Transcribing
        self._activity_panel.set_active("Transcribing")

        # Speech detected → stronger visualizer pulse
        self._visualizer.set_state(VisualizerState.SPEECH_DETECTED)

    @Slot(str)
    def _on_final_transcript(self, text: str) -> None:
        """
        Handle final STT transcript.

        Replaces the partial region with the final recognized sentence.
        """
        if not text.strip():
            return

        self._final_text = text
        self._partial_text = ""
        self._transcript_panel.set_final(text)

        # Start turn latency tracking
        self._turn_start_time = time.time()

        # Record STT latency (approximate: from turn start)
        # The actual STT latency would come from the pipeline

    @Slot(str)
    def _on_response(self, text: str) -> None:
        """
        Handle complete Diego response.

        The response is printed prominently and remains visible
        after TTS completes.
        """
        if not text.strip():
            return

        self._response_text = text
        self._response_panel.set_response(text)

        # Record agent latency
        if self._turn_start_time is not None:
            self._agent_latency_ms = (time.time() - self._turn_start_time) * 1000
            self._total_latency_ms = self._agent_latency_ms
            self._metrics.set_agent_latency(self._agent_latency_ms)
            self._metrics.set_total_latency(self._total_latency_ms)

    # ── State handlers ──────────────────────────────────────────

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        """Handle state change."""
        self._state_label.setText(state.upper())
        self._visualizer.set_state_by_name(state)
        self._activity_panel.set_active(state)

    @Slot()
    def _on_listening(self) -> None:
        """Handle listening state."""
        self._state_label.setText("LISTENING")
        self._visualizer.set_state(VisualizerState.LISTENING)
        self._activity_panel.set_active("Voice detected")
        self._response_panel.set_speaking(False)
        self._is_speaking = False

    @Slot()
    def _on_thinking(self) -> None:
        """Handle thinking state."""
        self._state_label.setText("THINKING")
        self._visualizer.set_state(VisualizerState.THINKING)
        self._activity_panel.set_active("Thinking")

    @Slot()
    def _on_planning(self) -> None:
        """Handle planning state."""
        self._state_label.setText("PLANNING")
        self._visualizer.set_state(VisualizerState.THINKING)
        self._activity_panel.set_active("Planning")

    @Slot()
    def _on_executing(self) -> None:
        """Handle executing state."""
        self._state_label.setText("EXECUTING")
        self._visualizer.set_state(VisualizerState.EXECUTING)
        self._activity_panel.set_active("Executing")

    @Slot()
    def _on_observing(self) -> None:
        """Handle observing state."""
        self._state_label.setText("OBSERVING")
        self._visualizer.set_state(VisualizerState.EXECUTING)
        self._activity_panel.set_active("Observing")

    @Slot()
    def _on_verifying(self) -> None:
        """Handle verifying state."""
        self._state_label.setText("VERIFYING")
        self._visualizer.set_state(VisualizerState.EXECUTING)
        self._activity_panel.set_active("Verifying")

    @Slot()
    def _on_replanning(self) -> None:
        """Handle replanning state."""
        self._state_label.setText("REPLANNING")
        self._visualizer.set_state(VisualizerState.THINKING)
        self._activity_panel.set_active("Planning")

    @Slot()
    def _on_speaking(self) -> None:
        """Handle speaking state."""
        self._state_label.setText("SPEAKING")
        self._visualizer.set_state(VisualizerState.SPEAKING)
        self._activity_panel.set_active("Responding")
        self._response_panel.set_speaking(True)
        self._is_speaking = True
        self._tts_start_time = time.time()

    @Slot()
    def _on_idle(self) -> None:
        """Handle idle state."""
        self._state_label.setText("IDLE")
        self._visualizer.set_state(VisualizerState.IDLE)
        self._activity_panel.reset()
        self._response_panel.set_speaking(False)

        # Record TTS latency if we were speaking
        if self._is_speaking and self._tts_start_time is not None:
            self._tts_latency_ms = (time.time() - self._tts_start_time) * 1000
            self._metrics.set_tts_latency(self._tts_latency_ms)
            if self._agent_latency_ms > 0:
                self._total_latency_ms = self._agent_latency_ms + self._tts_latency_ms
                self._metrics.set_total_latency(self._total_latency_ms)

        self._is_speaking = False
        self._tts_start_time = None

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Handle error — display friendly message."""
        self._state_label.setText("ERROR")
        self._visualizer.set_state(VisualizerState.ERROR)
        self._response_panel.set_error(message)
        self._response_panel.set_speaking(False)
        self._is_speaking = False

    # ── Audio handlers ──────────────────────────────────────────

    @Slot(float)
    def _on_audio_level(self, level: float) -> None:
        """Handle audio level update (real microphone RMS)."""
        self._visualizer.set_input_level(level)

    def set_output_level(self, level: float) -> None:
        """Set TTS output level for the visualizer."""
        self._visualizer.set_output_level(level)
        if self._is_speaking:
            self._response_panel.set_output_level(level)

    # ── Auth handlers ───────────────────────────────────────────

    @Slot()
    def _on_auth_required(self) -> None:
        """Handle face auth required."""
        self._state_label.setText("AUTHENTICATING")
        self._connection.set_status("connecting")

    @Slot(str)
    def _on_auth_completed(self, name: str) -> None:
        """Handle face auth completed."""
        self._connection.set_status("ready")
        if name:
            self._response_panel.set_response(f"Welcome back, {name}!")

    # ── Latency metrics ─────────────────────────────────────────

    def set_stt_latency(self, ms: float) -> None:
        """Set STT latency metric."""
        self._stt_latency_ms = ms
        self._metrics.set_stt_latency(ms)

    def set_tts_latency(self, ms: float) -> None:
        """Set TTS latency metric."""
        self._tts_latency_ms = ms
        self._metrics.set_tts_latency(ms)

    # ── System status ───────────────────────────────────────────

    def set_system_status(self, component: str, ok: bool) -> None:
        """Set the health status of a system component."""
        self._system_status.set_status(component, ok)

    # ── Async scheduling ────────────────────────────────────────

    def _schedule_async(self, coro) -> None:
        """Schedule an async coroutine on the asyncio loop (thread-safe)."""
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            # Fallback: run in a new thread with its own loop
            def run():
                try:
                    asyncio.run(coro)
                except Exception as e:
                    logger.error("[UI] Async fallback error: %s", e)
            threading.Thread(target=run, daemon=True).start()

    @Slot(object)
    def _execute_async(self, coro) -> None:
        """Execute async work (connected to _run_async signal)."""
        self._schedule_async(coro)

    # ── Window dragging (frameless window) ──────────────────────

    def mousePressEvent(self, event) -> None:
        """Enable window dragging from the header."""
        if event.button() == Qt.LeftButton:
            # Check if click is in header area
            if event.position().y() < 64:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Handle window drag movement."""
        if hasattr(self, '_drag_pos') and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """End window drag."""
        if hasattr(self, '_drag_pos'):
            del self._drag_pos

    # ── Cleanup ─────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Handle window close."""
        self._bridge.stop()
        super().closeEvent(event)

    # ── Public API ──────────────────────────────────────────────

    def set_runtime_state(self, state: str) -> None:
        """Set the runtime/connection state display."""
        self._connection.set_status(state)

    def get_transcript(self) -> list[dict]:
        """Get the current transcript as a list of messages."""
        messages = []
        if self._final_text:
            messages.append({"sender": "user", "text": self._final_text})
        if self._response_text:
            messages.append({"sender": "diego", "text": self._response_text})
        return messages

    def get_partial_text(self) -> str:
        """Get the current partial transcript text."""
        return self._partial_text

    def get_response_text(self) -> str:
        """Get the current Diego response text."""
        return self._response_text

    # ── Backward compatibility aliases for tests ────────────────

    @property
    def _state_indicator(self):
        """Backward compatibility: state indicator is now the state label."""
        return _StateIndicatorAdapter(self._state_label)

    @property
    def _waveform(self):
        """Backward compatibility: waveform is now the visualizer."""
        return _WaveformAdapter(self._visualizer)

    @property
    def _mic_indicator(self):
        """Backward compatibility: mic indicator adapter."""
        return _MicIndicatorAdapter(self._visualizer)

    @property
    def _transcript_label(self):
        """Backward compatibility: transcript label adapter."""
        return _TranscriptLabelAdapter(self._transcript_panel)

    @property
    def _response_label(self):
        """Backward compatibility: response label adapter."""
        return _ResponseLabelAdapter(self._response_panel)

    @property
    def _speaking_label(self):
        """Backward compatibility: speaking label adapter."""
        return _SpeakingLabelAdapter(self._response_panel)

    @property
    def _latency_metrics(self):
        """Backward compatibility: latency metrics adapter."""
        return _LatencyMetricsAdapter(self._metrics)


# ═══════════════════════════════════════════════════════════════
# Backward Compatibility Adapters
# ═══════════════════════════════════════════════════════════════
# These adapters allow existing tests to work with the new UI
# without modification. They map old widget APIs to new widgets.

class _StateIndicatorAdapter:
    """Adapter for backward compatibility with VoiceStateIndicator."""

    def __init__(self, label: QLabel):
        self._label = label

    def set_state(self, state: str) -> None:
        self._label.setText(state.upper())

    def state(self) -> str:
        # Read state from the label text (convert uppercase back to title case)
        text = self._label.text()
        if text == "IDLE":
            return "Idle"
        elif text == "LISTENING":
            return "Listening"
        elif text == "THINKING":
            return "Thinking"
        elif text == "PLANNING":
            return "Planning"
        elif text == "EXECUTING":
            return "Executing"
        elif text == "OBSERVING":
            return "Observing"
        elif text == "VERIFYING":
            return "Verifying"
        elif text == "REPLANNING":
            return "Replanning"
        elif text == "SPEAKING":
            return "Speaking"
        elif text == "ERROR":
            return "Error"
        elif text == "AUTHENTICATING":
            return "Authenticating"
        return text.capitalize() if text else "Idle"


class _WaveformAdapter:
    """Adapter for backward compatibility with WaveformWidget."""

    def __init__(self, visualizer: VoiceCoreVisualizer):
        self._visualizer = visualizer

    def set_level(self, level: float) -> None:
        self._visualizer.set_input_level(level)

    @property
    def _target_level(self) -> float:
        return self._visualizer._target_input_level


class _MicIndicatorAdapter:
    """Adapter for backward compatibility with MicIndicator."""

    def __init__(self, visualizer: VoiceCoreVisualizer):
        self._visualizer = visualizer

    @property
    def _active(self) -> bool:
        return self._visualizer.state in (
            VisualizerState.LISTENING,
            VisualizerState.SPEECH_DETECTED,
        )

    def set_active(self, active: bool) -> None:
        if active:
            self._visualizer.set_state(VisualizerState.LISTENING)
        else:
            self._visualizer.set_state(VisualizerState.IDLE)


class _TranscriptLabelAdapter:
    """Adapter for backward compatibility with TranscriptLabel."""

    def __init__(self, panel: TranscriptPanel):
        self._panel = panel

    @property
    def _text_label(self):
        return self._panel._text_label

    def set_partial(self, text: str) -> None:
        self._panel.set_partial(text)

    def set_final(self, text: str) -> None:
        self._panel.set_final(text)

    def clear(self) -> None:
        self._panel.clear()


class _ResponseLabelAdapter:
    """Adapter for backward compatibility with ResponseLabel."""

    def __init__(self, panel: ResponsePanel):
        self._panel = panel

    @property
    def _text_label(self):
        return self._panel._text_label

    def set_response(self, text: str) -> None:
        self._panel.set_response(text)

    def set_error(self, message: str) -> None:
        self._panel.set_error(message)

    def clear(self) -> None:
        self._panel.clear()


class _SpeakingLabelAdapter:
    """Adapter for backward compatibility with speaking label."""

    def __init__(self, panel: ResponsePanel):
        self._panel = panel
        self._visible = False

    def isVisible(self) -> bool:
        return self._panel._speaking_widget.isVisible()

    def show(self) -> None:
        self._panel.set_speaking(True)

    def hide(self) -> None:
        self._panel.set_speaking(False)

    def setVisible(self, visible: bool) -> None:
        self._panel.set_speaking(visible)


class _LatencyMetricsAdapter:
    """Adapter for backward compatibility with LatencyMetrics."""

    def __init__(self, metrics: MetricsCards):
        self._metrics = metrics

    def set_stt_latency(self, ms: float) -> None:
        self._metrics.set_stt_latency(ms)

    def set_agent_latency(self, ms: float) -> None:
        self._metrics.set_agent_latency(ms)

    def set_tts_latency(self, ms: float) -> None:
        self._metrics.set_tts_latency(ms)

    def set_total_latency(self, ms: float) -> None:
        self._metrics.set_total_latency(ms)