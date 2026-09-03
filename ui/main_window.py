"""
DiegoMainWindow — The main Diego desktop UI window.

Features:
    - Diego title/header with connection/runtime state
    - Conversation transcript (user + Diego messages)
    - Live partial transcript display (no duplication)
    - Current activity/status indicator
    - Microphone/listening indicator
    - Audio waveform area
    - Text input + send button (uses ConversationEngine)
    - Clear conversation button
    - Minimize/close controls

Threading:
    - All pipeline events arrive via EventBridge (thread-safe)
    - Text input is sent to ConversationEngine/Brain via asyncio
    - The Qt event loop is NEVER blocked
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QScrollArea, QFrame, QSizePolicy,
    QApplication, QSpacerItem,
)

from ui.event_bridge import EventBridge
from ui.styles import MAIN_WINDOW_QSS, COLORS
from ui.widgets import (
    MessageBubble, WaveformWidget, MicIndicator, StateIndicator, TypingIndicator,
)

logger = logging.getLogger(__name__)


class DiegoMainWindow(QMainWindow):
    """
    Main Diego conversational desktop window.

    The window subscribes to EventBridge signals and displays:
    - User messages (from FINAL_TRANSCRIPT events)
    - Diego responses (from RESPONSE events)
    - Partial transcripts (live STT updates)
    - State changes (Listening, Thinking, etc.)
    - Errors (friendly messages)
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
        self._partial_bubble: Optional[MessageBubble] = None
        self._partial_text: str = ""
        self._messages: list[MessageBubble] = []

        # Current Diego response being streamed
        self._streaming_bubble: Optional[MessageBubble] = None
        self._streaming_text: str = ""

        self._setup_ui()
        self._connect_signals()

        # Start the event bridge
        self._bridge.start()

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        self.setWindowTitle("Diego")
        self.setMinimumSize(480, 640)
        self.resize(520, 720)

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

        # ── Header ──────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(12)

        # Mic indicator
        self._mic_indicator = MicIndicator()
        header_layout.addWidget(self._mic_indicator)

        # Title
        title_label = QLabel("Diego")
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # State indicator
        self._state_indicator = StateIndicator()
        header_layout.addWidget(self._state_indicator)

        header_layout.addSpacing(16)

        # Window controls
        self._minimize_btn = QPushButton("─")
        self._minimize_btn.setObjectName("minimizeButton")
        self._minimize_btn.setFixedSize(32, 32)
        self._minimize_btn.clicked.connect(self.showMinimized)
        header_layout.addWidget(self._minimize_btn)

        self._close_btn = QPushButton("✕")
        self._close_btn.setObjectName("closeButton")
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.clicked.connect(self.close)
        header_layout.addWidget(self._close_btn)

        main_layout.addWidget(header)

        # ── Transcript area ─────────────────────────────────────
        self._transcript_scroll = QScrollArea()
        self._transcript_scroll.setObjectName("transcriptScroll")
        self._transcript_scroll.setWidgetResizable(True)
        self._transcript_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._transcript_scroll.setFrameShape(QFrame.NoFrame)

        self._transcript_container = QWidget()
        self._transcript_container.setObjectName("transcriptContainer")
        self._transcript_layout = QVBoxLayout(self._transcript_container)
        self._transcript_layout.setContentsMargins(20, 20, 20, 20)
        self._transcript_layout.setSpacing(12)
        self._transcript_layout.addStretch()

        self._transcript_scroll.setWidget(self._transcript_container)
        main_layout.addWidget(self._transcript_scroll, 1)

        # ── Audio/status area ───────────────────────────────────
        audio_area = QFrame()
        audio_area.setObjectName("audioArea")
        audio_layout = QHBoxLayout(audio_area)
        audio_layout.setContentsMargins(20, 8, 20, 8)
        audio_layout.setSpacing(12)

        # Waveform
        self._waveform = WaveformWidget()
        audio_layout.addWidget(self._waveform, 1)

        # Typing indicator (hidden by default)
        self._typing_indicator = TypingIndicator()
        self._typing_indicator.set_visible(False)
        audio_layout.addWidget(self._typing_indicator)

        main_layout.addWidget(audio_area)

        # ── Input area ──────────────────────────────────────────
        input_area = QFrame()
        input_area.setObjectName("inputArea")
        input_layout = QHBoxLayout(input_area)
        input_layout.setContentsMargins(20, 12, 20, 16)
        input_layout.setSpacing(12)

        # Clear button
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("clearButton")
        self._clear_btn.clicked.connect(self.clear_conversation)
        input_layout.addWidget(self._clear_btn)

        # Text input
        self._text_input = QLineEdit()
        self._text_input.setObjectName("textInput")
        self._text_input.setPlaceholderText("Type a message to Diego...")
        self._text_input.returnPressed.connect(self._on_send_clicked)
        input_layout.addWidget(self._text_input, 1)

        # Send button
        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendButton")
        self._send_btn.clicked.connect(self._on_send_clicked)
        input_layout.addWidget(self._send_btn)

        main_layout.addWidget(input_area)

        # ── Initial welcome message ─────────────────────────────
        self._add_message(
            "Hi! I'm Diego, your desktop assistant. "
            "You can talk to me or type a message below.",
            "diego"
        )

    def _connect_signals(self) -> None:
        """Connect EventBridge signals to UI handlers."""
        bridge = self._bridge

        # Transcript events
        bridge.partial_transcript.connect(self._on_partial_transcript)
        bridge.final_transcript.connect(self._on_final_transcript)
        bridge.response.connect(self._on_response)
        bridge.response_chunk.connect(self._on_response_chunk)

        # State events
        bridge.state_changed.connect(self._on_state_changed)
        bridge.listening.connect(self._on_listening)
        bridge.thinking.connect(self._on_thinking)
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

        Updates the existing partial bubble instead of creating duplicates.
        """
        if not text.strip():
            return

        self._partial_text = text

        if self._partial_bubble is None:
            # Create new partial bubble
            self._partial_bubble = MessageBubble(text, "partial")
            self._insert_before_stretch(self._partial_bubble)
        else:
            # Update existing bubble (no duplication)
            self._partial_bubble.update_text(text)

        self._scroll_to_bottom()

    @Slot(str)
    def _on_final_transcript(self, text: str) -> None:
        """
        Handle final STT transcript.

        Converts the partial bubble to a user message (no duplication).
        """
        if not text.strip():
            return

        # Remove/convert the partial bubble
        if self._partial_bubble is not None:
            # Convert partial to final user message
            self._partial_bubble.set_final(text)
            # Track it as a message now
            self._messages.append(self._partial_bubble)
            self._partial_bubble = None
        else:
            # No partial bubble — add as new user message
            self._add_message(text, "user")

        self._partial_text = ""
        self._scroll_to_bottom()

    @Slot(str)
    def _on_response(self, text: str) -> None:
        """Handle complete Diego response."""
        if not text.strip():
            return

        # If we were streaming, finalize the streaming bubble
        if self._streaming_bubble is not None:
            self._streaming_bubble.update_text(text)
            self._streaming_bubble = None
            self._streaming_text = ""
        else:
            self._add_message(text, "diego")

        self._typing_indicator.set_visible(False)
        self._scroll_to_bottom()

    @Slot(str)
    def _on_response_chunk(self, chunk: str) -> None:
        """Handle streaming response chunk."""
        if not chunk:
            return

        self._streaming_text += chunk

        if self._streaming_bubble is None:
            self._streaming_bubble = MessageBubble(self._streaming_text, "diego")
            self._insert_before_stretch(self._streaming_bubble)
        else:
            self._streaming_bubble.update_text(self._streaming_text)

        self._scroll_to_bottom()

    # ── State handlers ──────────────────────────────────────────

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        """Handle state change."""
        self._state_indicator.set_state(state)

        # Update mic indicator
        listening = "listen" in state.lower()
        self._mic_indicator.set_active(listening)

        # Show typing indicator when thinking/planning
        thinking = any(s in state.lower() for s in ("think", "plan", "execut", "observ", "verif", "replan"))
        self._typing_indicator.set_visible(thinking)

    @Slot()
    def _on_listening(self) -> None:
        """Handle listening state."""
        self._mic_indicator.set_active(True)
        self._typing_indicator.set_visible(False)

    @Slot()
    def _on_thinking(self) -> None:
        """Handle thinking state."""
        self._mic_indicator.set_active(False)
        self._typing_indicator.set_visible(True)

    @Slot()
    def _on_idle(self) -> None:
        """Handle idle state."""
        self._mic_indicator.set_active(False)
        self._typing_indicator.set_visible(False)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Handle error — display friendly message."""
        self._add_message(message, "error")
        self._typing_indicator.set_visible(False)
        self._scroll_to_bottom()

    # ── Audio handlers ──────────────────────────────────────────

    @Slot(float)
    def _on_audio_level(self, level: float) -> None:
        """Handle audio level update."""
        self._waveform.set_level(level)

    # ── Auth handlers ───────────────────────────────────────────

    @Slot()
    def _on_auth_required(self) -> None:
        """Handle face auth required."""
        self._add_message(
            "Please look at the camera for authentication.",
            "diego"
        )

    @Slot(str)
    def _on_auth_completed(self, name: str) -> None:
        """Handle face auth completed."""
        if name:
            self._add_message(f"Welcome back, {name}!", "diego")

    # ── Text input ──────────────────────────────────────────────

    def _on_send_clicked(self) -> None:
        """Handle send button click or Enter key."""
        text = self._text_input.text().strip()
        if not text:
            return

        # Clear input
        self._text_input.clear()

        # Add user message to transcript
        self._add_message(text, "user")
        self._scroll_to_bottom()

        # Send to ConversationEngine/Brain via asyncio
        self._submit_text_command(text)

    def _submit_text_command(self, text: str) -> None:
        """
        Submit typed text to the production ConversationEngine/Brain pipeline.

        This runs the Brain.process_command in the asyncio loop (off Qt thread).
        """
        async def process() -> None:
            try:
                from agent.brain import agent_brain

                # Ensure brain is initialized
                if not agent_brain._initialized:
                    await agent_brain.initialize()

                # Emit thinking state
                self._bridge.emit_thinking()

                # Process through the production pipeline
                result = await agent_brain.process_command(text)

                # Emit response
                if result.response:
                    self._bridge.emit_response(result.response)
                else:
                    self._bridge.emit_response("I'm not sure how to help with that.")

                # Return to idle
                self._bridge.emit_idle()

            except Exception as e:
                logger.error("[UI] Text command error: %s", e)
                self._bridge.emit_error("I had trouble processing that. Please try again.")

        self._schedule_async(process())

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

    # ── Message management ──────────────────────────────────────

    def _add_message(self, text: str, sender: str) -> MessageBubble:
        """Add a message bubble to the transcript."""
        bubble = MessageBubble(text, sender)
        self._messages.append(bubble)
        self._insert_before_stretch(bubble)
        return bubble

    def _insert_before_stretch(self, widget: QWidget) -> None:
        """Insert a widget before the trailing stretch."""
        count = self._transcript_layout.count()
        # Insert before the last item (stretch)
        self._transcript_layout.insertWidget(count - 1, widget)

    def _scroll_to_bottom(self) -> None:
        """Scroll the transcript to the bottom."""
        QTimer.singleShot(10, lambda: self._transcript_scroll.verticalScrollBar().setValue(
            self._transcript_scroll.verticalScrollBar().maximum()
        ))

    def clear_conversation(self) -> None:
        """Clear all messages from the transcript."""
        # Remove all message bubbles
        for bubble in self._messages:
            self._transcript_layout.removeWidget(bubble)
            bubble.deleteLater()
        self._messages.clear()

        # Clear partial/streaming state
        if self._partial_bubble is not None:
            self._transcript_layout.removeWidget(self._partial_bubble)
            self._partial_bubble.deleteLater()
            self._partial_bubble = None
        self._partial_text = ""

        if self._streaming_bubble is not None:
            self._transcript_layout.removeWidget(self._streaming_bubble)
            self._streaming_bubble.deleteLater()
            self._streaming_bubble = None
        self._streaming_text = ""

        # Add fresh welcome message
        self._add_message(
            "Conversation cleared. How can I help?",
            "diego"
        )

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
        self._state_indicator.set_state(state)

    def get_transcript(self) -> list[dict]:
        """Get the current transcript as a list of messages."""
        return [
            {"sender": b._sender, "text": b._text_label.text()}
            for b in self._messages
        ]