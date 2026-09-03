"""
DiegoMainWindow — Voice-first desktop assistant HUD.

This is NOT a chat application. The primary interaction is:
    MICROPHONE → LIVE TRANSCRIPT → DIEGO RESPONSE PRINTED
    → DIEGO RESPONSE SPOKEN → LISTEN AGAIN

Layout:
    HEADER: Diego identity + current voice state
    CENTER: Large animated waveform / audio visualizer
    TRANSCRIPT AREA: "You: ..." (live partial → final)
    RESPONSE AREA: "Diego: ..." (printed before/during TTS)
    FOOTER: Listening / Speaking / Processing + optional latency

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
    QFrame, QSizePolicy, QApplication,
)

from ui.event_bridge import EventBridge
from ui.styles import MAIN_WINDOW_QSS, COLORS
from ui.widgets import (
    VoiceStateIndicator, WaveformWidget, TranscriptLabel, ResponseLabel,
    LatencyMetrics, MicIndicator,
)

logger = logging.getLogger(__name__)


class DiegoMainWindow(QMainWindow):
    """
    Voice-first Diego desktop assistant HUD.

    The window subscribes to EventBridge signals and displays:
    - Current voice state (IDLE, LISTENING, THINKING, SPEAKING, etc.)
    - Live partial STT transcript (updating in real time)
    - Final user transcript (replaces partial)
    - Diego's response (printed before/during TTS, remains visible)
    - Speaking indicator (Diego is currently speaking)
    - Optional latency metrics
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

        self._setup_ui()
        self._connect_signals()

        # Start the event bridge
        self._bridge.start()

    def _setup_ui(self) -> None:
        """Build the voice-first HUD layout."""
        self.setWindowTitle("Diego")
        self.setMinimumSize(420, 560)
        self.resize(480, 680)

        # Remove native title bar for custom controls
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        # Apply stylesheet
        self.setStyleSheet(MAIN_WINDOW_QSS)

        # Central widget
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── HEADER: Diego identity + state ─────────────────────
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(72)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(12)

        # Mic indicator
        self._mic_indicator = MicIndicator()
        header_layout.addWidget(self._mic_indicator)

        # Title
        title_label = QLabel("DIEGO")
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Voice state indicator
        self._state_indicator = VoiceStateIndicator()
        header_layout.addWidget(self._state_indicator)

        header_layout.addSpacing(16)

        # Window controls
        self._minimize_btn = QLabel("─")
        self._minimize_btn.setObjectName("minimizeButton")
        self._minimize_btn.setFixedSize(32, 32)
        self._minimize_btn.mousePressEvent = lambda e: self.showMinimized()
        header_layout.addWidget(self._minimize_btn)

        self._close_btn = QLabel("✕")
        self._close_btn.setObjectName("closeButton")
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.mousePressEvent = lambda e: self.close()
        header_layout.addWidget(self._close_btn)

        main_layout.addWidget(header)

        # ── CENTER: Large waveform / audio visualizer ──────────
        waveform_area = QFrame()
        waveform_area.setObjectName("waveformArea")
        waveform_layout = QVBoxLayout(waveform_area)
        waveform_layout.setContentsMargins(20, 20, 20, 20)
        waveform_layout.setSpacing(8)

        self._waveform = WaveformWidget()
        self._waveform.setMinimumHeight(80)
        self._waveform.setMaximumHeight(120)
        waveform_layout.addWidget(self._waveform, 1)

        # Speaking indicator (hidden by default)
        self._speaking_label = QLabel("● Diego is speaking")
        self._speaking_label.setObjectName("speakingLabel")
        self._speaking_label.setAlignment(Qt.AlignCenter)
        self._speaking_label.hide()
        waveform_layout.addWidget(self._speaking_label)

        main_layout.addWidget(waveform_area, 1)

        # ── TRANSCRIPT AREA: "You: ..." ─────────────────────────
        transcript_area = QFrame()
        transcript_area.setObjectName("transcriptArea")
        transcript_layout = QVBoxLayout(transcript_area)
        transcript_layout.setContentsMargins(20, 8, 20, 8)
        transcript_layout.setSpacing(4)

        self._transcript_label = TranscriptLabel()
        transcript_layout.addWidget(self._transcript_label)

        main_layout.addWidget(transcript_area)

        # ── RESPONSE AREA: "Diego: ..." ─────────────────────────
        response_area = QFrame()
        response_area.setObjectName("responseArea")
        response_layout = QVBoxLayout(response_area)
        response_layout.setContentsMargins(20, 8, 20, 8)
        response_layout.setSpacing(4)

        self._response_label = ResponseLabel()
        response_layout.addWidget(self._response_label)

        main_layout.addWidget(response_area, 1)

        # ── FOOTER: status + latency ────────────────────────────
        footer = QFrame()
        footer.setObjectName("footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 8, 20, 12)
        footer_layout.setSpacing(12)

        self._status_label = QLabel("Listening for wake word...")
        self._status_label.setObjectName("statusLabel")
        footer_layout.addWidget(self._status_label)

        footer_layout.addStretch()

        self._latency_metrics = LatencyMetrics()
        footer_layout.addWidget(self._latency_metrics)

        main_layout.addWidget(footer)

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
        self._transcript_label.set_partial(text)

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
        self._transcript_label.set_final(text)

        # Start turn latency tracking
        self._turn_start_time = time.time()

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
        self._response_label.set_response(text)

        # Record agent latency
        if self._turn_start_time is not None:
            self._agent_latency_ms = (time.time() - self._turn_start_time) * 1000
            self._total_latency_ms = self._agent_latency_ms
            self._latency_metrics.set_agent_latency(self._agent_latency_ms)
            self._latency_metrics.set_total_latency(self._total_latency_ms)

    # ── State handlers ──────────────────────────────────────────

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        """Handle state change."""
        self._state_indicator.set_state(state)
        self._status_label.setText(state)

        # Update mic indicator
        listening = "listen" in state.lower()
        self._mic_indicator.set_active(listening)

        # Show/hide speaking indicator
        speaking = "speak" in state.lower() or "respond" in state.lower()
        self._speaking_label.setVisible(speaking)

    @Slot()
    def _on_listening(self) -> None:
        """Handle listening state."""
        self._state_indicator.set_state("Listening")
        self._status_label.setText("Listening...")
        self._mic_indicator.set_active(True)
        self._speaking_label.hide()

    @Slot()
    def _on_thinking(self) -> None:
        """Handle thinking state."""
        self._state_indicator.set_state("Thinking")
        self._status_label.setText("Thinking...")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()

    @Slot()
    def _on_planning(self) -> None:
        """Handle planning state."""
        self._state_indicator.set_state("Planning")
        self._status_label.setText("Planning...")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()

    @Slot()
    def _on_executing(self) -> None:
        """Handle executing state."""
        self._state_indicator.set_state("Executing")
        self._status_label.setText("Executing...")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()

    @Slot()
    def _on_observing(self) -> None:
        """Handle observing state."""
        self._state_indicator.set_state("Observing")
        self._status_label.setText("Observing...")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()

    @Slot()
    def _on_verifying(self) -> None:
        """Handle verifying state."""
        self._state_indicator.set_state("Verifying")
        self._status_label.setText("Verifying...")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()

    @Slot()
    def _on_replanning(self) -> None:
        """Handle replanning state."""
        self._state_indicator.set_state("Replanning")
        self._status_label.setText("Replanning...")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()

    @Slot()
    def _on_speaking(self) -> None:
        """Handle speaking state."""
        self._state_indicator.set_state("Speaking")
        self._status_label.setText("Speaking...")
        self._mic_indicator.set_active(False)
        self._speaking_label.show()

    @Slot()
    def _on_idle(self) -> None:
        """Handle idle state."""
        self._state_indicator.set_state("Idle")
        self._status_label.setText("Listening for wake word...")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Handle error — display friendly message."""
        self._state_indicator.set_state("Error")
        self._status_label.setText("Error")
        self._mic_indicator.set_active(False)
        self._speaking_label.hide()
        self._response_label.set_error(message)

    # ── Audio handlers ──────────────────────────────────────────

    @Slot(float)
    def _on_audio_level(self, level: float) -> None:
        """Handle audio level update."""
        self._waveform.set_level(level)

    # ── Auth handlers ───────────────────────────────────────────

    @Slot()
    def _on_auth_required(self) -> None:
        """Handle face auth required."""
        self._state_indicator.set_state("Authenticating")
        self._status_label.setText("Authenticating...")

    @Slot(str)
    def _on_auth_completed(self, name: str) -> None:
        """Handle face auth completed."""
        if name:
            self._response_label.set_response(f"Welcome back, {name}!")

    # ── Latency metrics ─────────────────────────────────────────

    def set_stt_latency(self, ms: float) -> None:
        """Set STT latency metric."""
        self._stt_latency_ms = ms
        self._latency_metrics.set_stt_latency(ms)

    def set_tts_latency(self, ms: float) -> None:
        """Set TTS latency metric."""
        self._tts_latency_ms = ms
        self._latency_metrics.set_tts_latency(ms)

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
            if event.position().y() < 72:
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
        self._state_indicator.set_state(state)

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