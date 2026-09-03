"""
Diego UI Widgets — Custom widgets for the conversational interface.

Includes:
    - MessageBubble: Chat-style message display (user/Diego)
    - WaveformWidget: Audio activity visualization
    - MicIndicator: Microphone/listening state indicator
    - StateIndicator: Current Diego state display
"""

from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Property, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QLinearGradient, QFont
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
)

from ui.styles import COLORS


class MessageBubble(QFrame):
    """
    A chat-style message bubble for user or Diego messages.

    Supports:
        - User messages (right-aligned, blue)
        - Diego messages (left-aligned, dark)
        - Partial transcripts (dashed border, muted)
        - Error messages (red accent)
    """

    def __init__(
        self,
        text: str,
        sender: str = "diego",  # "user" | "diego" | "partial" | "error"
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._sender = sender
        self._setup_ui(text)

    def _setup_ui(self, text: str) -> None:
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        # Sender label
        sender_label = QLabel(self._sender_name())
        sender_label.setObjectName("messageSender")
        sender_class = "userSender" if self._sender == "user" else "diegoSender"
        sender_label.setProperty("class", sender_class)
        layout.addWidget(sender_label)

        # Message text
        self._text_label = QLabel(text)
        self._text_label.setObjectName("messageLabel")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._text_label)

        # Styling based on sender type
        bubble_class = {
            "user": "userBubble",
            "diego": "diegoBubble",
            "partial": "partialBubble",
            "error": "errorBubble",
        }.get(self._sender, "diegoBubble")

        self.setProperty("class", f"messageBubble {bubble_class}")
        self._apply_style()

        # Size policy
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setMaximumWidth(500)

    def _sender_name(self) -> str:
        return {
            "user": "You",
            "diego": "Diego",
            "partial": "Listening...",
            "error": "Error",
        }.get(self._sender, "Diego")

    def _apply_style(self) -> None:
        """Apply inline styles based on sender type."""
        styles = {
            "user": f"""
                MessageBubble {{
                    background-color: {COLORS['user_bubble']};
                    border-radius: 12px;
                }}
                #messageLabel {{ color: white; font-size: 14px; }}
                #messageSender {{ color: rgba(255,255,255,0.7); font-size: 11px; font-weight: 600; }}
            """,
            "diego": f"""
                MessageBubble {{
                    background-color: {COLORS['diego_bubble']};
                    border-radius: 12px;
                }}
                #messageLabel {{ color: {COLORS['text_primary']}; font-size: 14px; }}
                #messageSender {{ color: {COLORS['accent_primary']}; font-size: 11px; font-weight: 600; }}
            """,
            "partial": f"""
                MessageBubble {{
                    background-color: {COLORS['bg_tertiary']};
                    border-radius: 12px;
                    border: 1px dashed {COLORS['border']};
                }}
                #messageLabel {{ color: {COLORS['text_muted']}; font-size: 14px; font-style: italic; }}
                #messageSender {{ color: {COLORS['text_muted']}; font-size: 11px; }}
            """,
            "error": f"""
                MessageBubble {{
                    background-color: rgba(247, 118, 142, 0.15);
                    border-radius: 12px;
                    border: 1px solid {COLORS['accent_error']};
                }}
                #messageLabel {{ color: {COLORS['accent_error']}; font-size: 14px; }}
                #messageSender {{ color: {COLORS['accent_error']}; font-size: 11px; font-weight: 600; }}
            """,
        }
        self.setStyleSheet(styles.get(self._sender, styles["diego"]))

    def update_text(self, text: str) -> None:
        """Update the message text (for partial → final transitions)."""
        self._text_label.setText(text)

    def set_final(self, text: str) -> None:
        """Convert a partial bubble to a final user message."""
        self._sender = "user"
        self._text_label.setText(text)
        # Update sender label
        sender_label = self.findChild(QLabel, "messageSender")
        if sender_label:
            sender_label.setText("You")
        self._apply_style()


class WaveformWidget(QWidget):
    """
    Audio waveform visualization for microphone activity.

    Displays a smooth, animated waveform based on audio level input.
    Low CPU usage when idle (no animation timer running).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("waveformWidget")
        self.setMinimumHeight(40)
        self.setMaximumHeight(60)

        self._level = 0.0
        self._target_level = 0.0
        self._bars = 24
        self._bar_values = [0.0] * self._bars
        self._active = False

        # Animation timer (only runs when active)
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._animate)

        self.setStyleSheet(f"""
            WaveformWidget {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 8px;
            }}
        """)

    def set_level(self, level: float) -> None:
        """Set the current audio level (0.0 - 1.0)."""
        self._target_level = max(0.0, min(1.0, level))
        if level > 0.05 and not self._active:
            self._active = True
            self._timer.start()
        elif level <= 0.02 and self._active:
            # Decay before stopping
            self._target_level = 0.0

    def _animate(self) -> None:
        """Animate the waveform bars."""
        # Smooth level transition
        self._level += (self._target_level - self._level) * 0.3

        # Generate bar values with some randomness for natural look
        import random
        for i in range(self._bars):
            # Center bars are taller
            center_factor = 1.0 - abs(i - self._bars / 2) / (self._bars / 2) * 0.5
            noise = random.uniform(0.7, 1.0)
            target = self._level * center_factor * noise
            self._bar_values[i] += (target - self._bar_values[i]) * 0.4

        self.update()

        # Stop animation when idle
        if self._level < 0.01 and self._target_level < 0.01:
            self._active = False
            self._timer.stop()
            self._bar_values = [0.0] * self._bars
            self.update()

    def paintEvent(self, event) -> None:
        """Draw the waveform bars."""
        from PySide6.QtCore import QRectF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_width = w / (self._bars * 1.5)
        spacing = bar_width * 0.5

        # Gradient for bars
        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0, QColor(COLORS['accent_primary']))
        gradient.setColorAt(0.5, QColor(COLORS['accent_secondary']))
        gradient.setColorAt(1, QColor(COLORS['accent_primary']))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)

        for i, value in enumerate(self._bar_values):
            bar_height = max(2, value * (h - 8))
            x = i * (bar_width + spacing) + spacing
            y = (h - bar_height) / 2
            rect = QRectF(x, y, bar_width, bar_height)
            painter.drawRoundedRect(rect, bar_width / 2, bar_width / 2)

        painter.end()


class MicIndicator(QWidget):
    """
    Microphone indicator showing listening state.

    Displays a pulsing dot when listening, static when idle.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("micIndicator")
        self.setFixedSize(24, 24)
        self._active = False
        self._pulse_phase = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._pulse)

    def set_active(self, active: bool) -> None:
        """Set the listening state."""
        if active != self._active:
            self._active = active
            if active:
                self._timer.start()
            else:
                self._timer.stop()
                self._pulse_phase = 0.0
            self.update()

    def _pulse(self) -> None:
        """Animate the pulse."""
        self._pulse_phase += 0.15
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the mic indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        center = QPointF(12, 12)

        if self._active:
            # Pulsing glow
            pulse = (math.sin(self._pulse_phase) + 1) / 2
            glow_radius = 8 + pulse * 3
            glow_color = QColor(COLORS['state_listening'])
            glow_color.setAlpha(int(80 + pulse * 60))
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, glow_radius, glow_radius)

            # Core dot
            painter.setBrush(QBrush(QColor(COLORS['state_listening'])))
            painter.drawEllipse(center, 5, 5)
        else:
            # Static muted dot
            painter.setBrush(QBrush(QColor(COLORS['text_muted'])))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, 5, 5)

        painter.end()


class StateIndicator(QFrame):
    """
    Displays the current Diego state with color coding.

    States: Listening, Thinking, Planning, Executing, Observing,
            Verifying, Replanning, Responding, Idle, Error
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("stateIndicator")
        self._state = "Idle"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        layout.addWidget(self._dot)

        self._label = QLabel("Idle")
        layout.addWidget(self._label)

        self.set_state("Idle")

    def set_state(self, state: str) -> None:
        """Update the displayed state."""
        self._state = state
        self._label.setText(state)

        color = self._state_color(state)
        self._dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.setStyleSheet(f"""
            StateIndicator {{
                background-color: {color}22;
                border-radius: 14px;
                border: 1px solid {color}44;
            }}
            QLabel {{
                color: {color};
                font-size: 13px;
                font-weight: 500;
                background: transparent;
            }}
        """)

    def _state_color(self, state: str) -> str:
        """Get the color for a state."""
        state_lower = state.lower()
        if "listen" in state_lower:
            return COLORS['state_listening']
        elif any(s in state_lower for s in ("think", "plan", "replan")):
            return COLORS['state_thinking']
        elif any(s in state_lower for s in ("execut", "observ", "verif")):
            return COLORS['state_executing']
        elif "respond" in state_lower or "speak" in state_lower:
            return COLORS['accent_secondary']
        elif "error" in state_lower:
            return COLORS['state_error']
        elif "auth" in state_lower:
            return COLORS['accent_warning']
        return COLORS['state_idle']

    def state(self) -> str:
        """Get the current state."""
        return self._state


class TypingIndicator(QWidget):
    """
    Animated typing indicator for when Diego is thinking/responding.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._phase = 0.0
        self._visible = False

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._animate)

    def set_visible(self, visible: bool) -> None:
        """Show/hide the typing indicator."""
        self._visible = visible
        if visible:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _animate(self) -> None:
        self._phase += 0.3
        self.update()

    def paintEvent(self, event) -> None:
        if not self._visible:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        dot_radius = 4
        spacing = 12
        start_x = 20

        for i in range(3):
            bounce = math.sin(self._phase + i * 0.5)
            y = 15 - bounce * 4
            alpha = int(150 + bounce * 100)
            color = QColor(COLORS['accent_primary'])
            color.setAlpha(max(50, min(255, alpha)))
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(start_x + i * spacing, y), dot_radius, dot_radius)

        painter.end()