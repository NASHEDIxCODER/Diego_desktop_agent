"""
Diego UI Widgets — Custom widgets for the voice-first assistant HUD.

Includes:
    - VoiceStateIndicator: Current Diego voice state display
    - WaveformWidget: Audio activity visualization
    - MicIndicator: Microphone/listening state indicator
    - TranscriptLabel: Live partial/final user transcript display
    - ResponseLabel: Diego response display
    - LatencyMetrics: Optional latency diagnostics
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


class VoiceStateIndicator(QFrame):
    """
    Displays the current Diego voice state with color coding.

    States: Idle, Listening, Speech Detected, Thinking, Planning,
            Executing, Observing, Verifying, Replanning, Speaking, Error
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("stateIndicator")
        self._state = "Idle"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(14)
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
            VoiceStateIndicator {{
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
        elif "speech" in state_lower:
            return COLORS['state_listening']
        elif any(s in state_lower for s in ("think", "plan", "replan")):
            return COLORS['state_thinking']
        elif any(s in state_lower for s in ("execut", "observ", "verif")):
            return COLORS['state_executing']
        elif "speak" in state_lower or "respond" in state_lower:
            return COLORS['accent_secondary']
        elif "error" in state_lower:
            return COLORS['state_error']
        elif "auth" in state_lower:
            return COLORS['accent_warning']
        return COLORS['state_idle']

    def state(self) -> str:
        """Get the current state."""
        return self._state


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
        self.setMaximumHeight(120)

        self._level = 0.0
        self._target_level = 0.0
        self._bars = 32
        self._bar_values = [0.0] * self._bars
        self._active = False

        # Animation timer (only runs when active)
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._animate)

        self.setStyleSheet(f"""
            WaveformWidget {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 12px;
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


class TranscriptLabel(QFrame):
    """
    Displays the user's transcript.

    Shows "You: ..." with live partial updates while speaking,
    then the final recognized sentence once.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("transcriptLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(2)

        # Sender label
        self._sender_label = QLabel("You")
        self._sender_label.setObjectName("transcriptSender")
        layout.addWidget(self._sender_label)

        # Text label
        self._text_label = QLabel("")
        self._text_label.setObjectName("transcriptText")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._text_label)

        self.setStyleSheet(f"""
            TranscriptLabel {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 12px;
                border: 1px solid {COLORS['border']};
            }}
            #transcriptSender {{
                color: {COLORS['accent_primary']};
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            #transcriptText {{
                color: {COLORS['text_primary']};
                font-size: 15px;
            }}
        """)

    def set_partial(self, text: str) -> None:
        """Update the live partial transcript."""
        self._text_label.setText(text)
        self._text_label.setStyleSheet(f"""
            #transcriptText {{
                color: {COLORS['text_secondary']};
                font-size: 15px;
                font-style: italic;
            }}
        """)

    def set_final(self, text: str) -> None:
        """Set the final recognized sentence (replaces partial)."""
        self._text_label.setText(text)
        self._text_label.setStyleSheet(f"""
            #transcriptText {{
                color: {COLORS['text_primary']};
                font-size: 15px;
            }}
        """)

    def clear(self) -> None:
        """Clear the transcript."""
        self._text_label.setText("")


class ResponseLabel(QFrame):
    """
    Displays Diego's response prominently.

    The response appears before/during TTS and remains visible after.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("responseLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(2)

        # Sender label
        self._sender_label = QLabel("Diego")
        self._sender_label.setObjectName("responseSender")
        layout.addWidget(self._sender_label)

        # Text label
        self._text_label = QLabel("")
        self._text_label.setObjectName("responseText")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._text_label)

        self.setStyleSheet(f"""
            ResponseLabel {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 12px;
                border: 1px solid {COLORS['accent_primary']}44;
            }}
            #responseSender {{
                color: {COLORS['accent_secondary']};
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            #responseText {{
                color: {COLORS['text_primary']};
                font-size: 16px;
                font-weight: 500;
            }}
        """)

    def set_response(self, text: str) -> None:
        """Set Diego's response text."""
        self._text_label.setText(text)
        self._text_label.setStyleSheet(f"""
            #responseText {{
                color: {COLORS['text_primary']};
                font-size: 16px;
                font-weight: 500;
            }}
        """)

    def set_error(self, message: str) -> None:
        """Display an error message."""
        self._text_label.setText(message)
        self._text_label.setStyleSheet(f"""
            #responseText {{
                color: {COLORS['accent_error']};
                font-size: 15px;
            }}
        """)

    def clear(self) -> None:
        """Clear the response."""
        self._text_label.setText("")


class LatencyMetrics(QWidget):
    """
    Optional small diagnostics display for latency metrics.

    Shows STT / agent / TTS / total turn latency in milliseconds.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("latencyMetrics")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._stt_label = QLabel("STT: --ms")
        self._agent_label = QLabel("Agent: --ms")
        self._tts_label = QLabel("TTS: --ms")
        self._total_label = QLabel("Total: --ms")

        for label in (self._stt_label, self._agent_label,
                      self._tts_label, self._total_label):
            label.setObjectName("latencyMetric")
            label.setStyleSheet(f"""
                #latencyMetric {{
                    color: {COLORS['text_muted']};
                    font-size: 11px;
                    font-family: 'JetBrains Mono', 'Fira Code', monospace;
                }}
            """)
            layout.addWidget(label)

    def set_stt_latency(self, ms: float) -> None:
        self._stt_label.setText(f"STT: {ms:.0f}ms")

    def set_agent_latency(self, ms: float) -> None:
        self._agent_label.setText(f"Agent: {ms:.0f}ms")

    def set_tts_latency(self, ms: float) -> None:
        self._tts_label.setText(f"TTS: {ms:.0f}ms")

    def set_total_latency(self, ms: float) -> None:
        self._total_label.setText(f"Total: {ms:.0f}ms")