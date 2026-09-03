"""
Diego UI Widgets — Premium HUD widgets for the voice-first assistant.

Includes:
    - ConnectionIndicator: Compact connection/status indicator
    - TranscriptPanel: "YOU SAID" live transcript region
    - ResponsePanel: "DIEGO" prominent response display
    - ActivityPanel: Right-side high-level activity display
    - MetricsCards: Compact latency metrics (STT/Agent/TTS/Total)
    - SystemStatus: Footer system health display
    - MiniWaveform: Small waveform for response/speaking indication
"""

from __future__ import annotations

import math
from typing import Optional, List

from PySide6.QtCore import Qt, QTimer, QPointF, Property, QEasingCurve
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
    QGraphicsOpacityEffect, QGridLayout,
)

from ui.styles import COLORS, FONTS


# ═══════════════════════════════════════════════════════════════
# Connection Indicator
# ═══════════════════════════════════════════════════════════════

class ConnectionIndicator(QFrame):
    """
    Compact connection/status indicator for the header.

    Shows a colored dot + text indicating pipeline status.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("connectionIndicator")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setObjectName("connectionDot")
        self._dot.setFixedWidth(10)
        layout.addWidget(self._dot)

        self._text = QLabel("Ready")
        self._text.setObjectName("connectionText")
        layout.addWidget(self._text)

        self.set_status("ready")

    def set_status(self, status: str) -> None:
        """Set the connection status: ready, connecting, error."""
        if status == "ready":
            color = COLORS["accent_success"]
            text = "Ready"
        elif status == "connecting":
            color = COLORS["accent_warning"]
            text = "Connecting"
        elif status == "error":
            color = COLORS["accent_error"]
            text = "Error"
        else:
            color = COLORS["text_muted"]
            text = status.capitalize()

        self._dot.setStyleSheet(f"color: {color}; font-size: 8px; background: transparent;")
        self._text.setText(text)

    def status(self) -> str:
        """Get the current status text."""
        return self._text.text()


# ═══════════════════════════════════════════════════════════════
# Transcript Panel ("YOU SAID")
# ═══════════════════════════════════════════════════════════════

class TranscriptPanel(QFrame):
    """
    Displays the user's transcript in a single "YOU SAID" region.

    During speech: shows live partial text (italic, secondary color).
    When final: replaces partial with final text (smooth transition).
    Never duplicates — one region only.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("transcriptPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(6)

        # Header: "YOU SAID"
        self._header = QLabel("You said")
        self._header.setObjectName("transcriptHeader")
        layout.addWidget(self._header)

        # Text label
        self._text_label = QLabel("")
        self._text_label.setObjectName("transcriptText")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text_label.setMinimumHeight(28)
        layout.addWidget(self._text_label)

        # Opacity effect for smooth transitions
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self._text_label.setGraphicsEffect(self._opacity)

        self._is_partial = False

    def set_partial(self, text: str) -> None:
        """Update the live partial transcript."""
        if not text.strip():
            return

        self._is_partial = True
        self._text_label.setObjectName("transcriptTextPartial")
        self._text_label.setText(text)
        self._apply_text_style()

    def set_final(self, text: str) -> None:
        """Set the final transcript (replaces partial with smooth transition)."""
        if not text.strip():
            return

        # Smooth transition: fade out, update, fade in
        self._transition_to(text, is_partial=False)

    def _transition_to(self, text: str, is_partial: bool) -> None:
        """Smoothly transition the text content."""
        self._is_partial = is_partial

        # Quick fade out
        self._opacity.setOpacity(0.3)

        # Update content
        self._text_label.setObjectName(
            "transcriptTextPartial" if is_partial else "transcriptText"
        )
        self._text_label.setText(text)
        self._apply_text_style()

        # Fade back in
        self._opacity.setOpacity(1.0)

    def _apply_text_style(self) -> None:
        """Apply the appropriate text style."""
        if self._is_partial:
            self._text_label.setStyleSheet(f"""
                #transcriptTextPartial {{
                    color: {COLORS['text_secondary']};
                    font-size: {FONTS['size_xl']};
                    font-style: italic;
                    background: transparent;
                }}
            """)
        else:
            self._text_label.setStyleSheet(f"""
                #transcriptText {{
                    color: {COLORS['text_primary']};
                    font-size: {FONTS['size_xl']};
                    background: transparent;
                }}
            """)

    def clear(self) -> None:
        """Clear the transcript."""
        self._text_label.setText("")
        self._is_partial = False

    def text(self) -> str:
        """Get the current transcript text."""
        return self._text_label.text()

    def is_partial(self) -> bool:
        """Check if the current text is a partial transcript."""
        return self._is_partial


# ═══════════════════════════════════════════════════════════════
# Response Panel ("DIEGO")
# ═══════════════════════════════════════════════════════════════

class ResponsePanel(QFrame):
    """
    Displays Diego's response prominently.

    Shows "DIEGO" header, large response text, and a small
    waveform/speaking indicator. The response appears immediately
    when available (before/during TTS) and remains visible.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("responsePanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(8)

        # Header row: "DIEGO" + speaking indicator
        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        self._header = QLabel("DIEGO")
        self._header.setObjectName("responseHeader")
        header_row.addWidget(self._header)

        header_row.addStretch()

        # Speaking indicator (animated dots + text)
        self._speaking_widget = SpeakingIndicator()
        self._speaking_widget.hide()
        header_row.addWidget(self._speaking_widget)

        layout.addLayout(header_row)

        # Response text
        self._text_label = QLabel("")
        self._text_label.setObjectName("responseText")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text_label.setMinimumHeight(28)
        layout.addWidget(self._text_label)

        # Mini waveform (shown during speaking)
        self._mini_waveform = MiniWaveform()
        self._mini_waveform.setFixedHeight(24)
        self._mini_waveform.hide()
        layout.addWidget(self._mini_waveform)

    def set_response(self, text: str) -> None:
        """Set Diego's response text."""
        self._text_label.setText(text)
        self._text_label.setStyleSheet(f"""
            #responseText {{
                color: {COLORS['text_primary']};
                font-size: {FONTS['size_xl']};
                font-weight: 500;
                background: transparent;
            }}
        """)

    def set_error(self, message: str) -> None:
        """Display an error message."""
        self._text_label.setText(message)
        self._text_label.setStyleSheet(f"""
            #responseText {{
                color: {COLORS['accent_error']};
                font-size: {FONTS['size_lg']};
                background: transparent;
            }}
        """)

    def set_speaking(self, speaking: bool) -> None:
        """Show/hide the speaking indicator."""
        self._speaking_widget.setVisible(speaking)
        self._mini_waveform.setVisible(speaking)
        if speaking:
            self._speaking_widget.start()
            self._mini_waveform.start()
        else:
            self._speaking_widget.stop()
            self._mini_waveform.stop()

    def set_output_level(self, level: float) -> None:
        """Set the TTS output level for the mini waveform."""
        self._mini_waveform.set_level(level)

    def clear(self) -> None:
        """Clear the response."""
        self._text_label.setText("")

    def text(self) -> str:
        """Get the current response text."""
        return self._text_label.text()


class SpeakingIndicator(QWidget):
    """Animated "Speaking" indicator with pulsing dots."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("speakingIndicator")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel("Speaking")
        self._label.setStyleSheet(f"""
            color: {COLORS['accent_primary']};
            font-size: {FONTS['size_sm']};
            font-weight: 600;
            background: transparent;
        """)
        layout.addWidget(self._label)

        self._dots = QLabel("···")
        self._dots.setStyleSheet(f"""
            color: {COLORS['accent_primary']};
            font-size: {FONTS['size_sm']};
            background: transparent;
        """)
        layout.addWidget(self._dots)

        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._animate)

    def start(self) -> None:
        """Start the animation."""
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        """Stop the animation."""
        self._timer.stop()
        self._dots.setText("···")

    def _animate(self) -> None:
        """Animate the dots."""
        self._phase = (self._phase + 1) % 4
        dots = "·" * (self._phase + 1)
        self._dots.setText(dots)


class MiniWaveform(QWidget):
    """Small waveform display for speaking/output indication."""

    NUM_BARS = 24

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("miniWaveform")
        self._level = 0.0
        self._target_level = 0.0
        self._phase = 0.0
        self._bar_values = [0.0] * self.NUM_BARS

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._animate)

    def set_level(self, level: float) -> None:
        """Set the output level (0-1)."""
        self._target_level = max(0.0, min(1.0, level))

    def start(self) -> None:
        """Start the animation."""
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        """Stop the animation."""
        self._timer.stop()
        self._bar_values = [0.0] * self.NUM_BARS
        self.update()

    def _animate(self) -> None:
        """Animate the waveform."""
        self._level += (self._target_level - self._level) * 0.3
        self._phase += 0.15

        for i in range(self.NUM_BARS):
            # Smooth wave pattern
            wave = math.sin(i * 0.5 + self._phase) * 0.5 + 0.5
            target = max(0.05, self._level * wave)
            self._bar_values[i] += (target - self._bar_values[i]) * 0.4

        self.update()

    def paintEvent(self, event) -> None:
        """Draw the mini waveform."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_width = w / (self.NUM_BARS * 1.8)
        spacing = bar_width * 0.8

        color = QColor(COLORS["accent_primary"])

        for i, value in enumerate(self._bar_values):
            bar_height = max(2, value * (h - 4))
            x = i * (bar_width + spacing) + spacing
            y = (h - bar_height) / 2

            bar_color = QColor(color)
            bar_color.setAlphaF(0.4 + value * 0.6)
            painter.setBrush(QBrush(bar_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(
                x, y, bar_width, bar_height,
                bar_width / 2, bar_width / 2
            )

        painter.end()


# ═══════════════════════════════════════════════════════════════
# Activity Panel
# ═══════════════════════════════════════════════════════════════

class ActivityPanel(QFrame):
    """
    Right-side panel showing human-readable high-level activity.

    Shows only: Voice detected, Transcribing, Thinking, Planning,
    Executing, Observing, Verifying, Responding.

    Does NOT show: paths, JSON, scores, stack traces, etc.
    """

    # Ordered activity steps
    ACTIVITIES = [
        "Voice detected",
        "Transcribing",
        "Thinking",
        "Planning",
        "Executing",
        "Observing",
        "Verifying",
        "Responding",
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("activityPanel")
        self.setFixedWidth(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(2)

        # Title
        self._title = QLabel("Activity")
        self._title.setObjectName("activityTitle")
        layout.addWidget(self._title)
        layout.addSpacing(12)

        # Activity items
        self._items: List[QLabel] = []
        for activity in self.ACTIVITIES:
            item = QLabel(activity)
            item.setObjectName("activityItem")
            item.setStyleSheet(f"""
                color: {COLORS['text_muted']};
                font-size: {FONTS['size_md']};
                padding: 4px 0px;
                background: transparent;
            """)
            layout.addWidget(item)
            self._items.append(item)

        layout.addStretch()

        self._active_index = -1

    def set_active(self, activity: str) -> None:
        """Set the currently active activity."""
        # Map state names to activity names
        activity_lower = activity.lower()

        if "listen" in activity_lower or "voice" in activity_lower:
            self._set_active_index(0)
        elif "transcrib" in activity_lower or "partial" in activity_lower:
            self._set_active_index(1)
        elif "think" in activity_lower:
            self._set_active_index(2)
        elif "plan" in activity_lower and "replan" not in activity_lower:
            self._set_active_index(3)
        elif "execut" in activity_lower:
            self._set_active_index(4)
        elif "observ" in activity_lower:
            self._set_active_index(5)
        elif "verif" in activity_lower:
            self._set_active_index(6)
        elif "speak" in activity_lower or "respond" in activity_lower:
            self._set_active_index(7)
        elif "replan" in activity_lower:
            self._set_active_index(3)  # Replanning → Planning
        else:
            self._set_active_index(-1)

    def _set_active_index(self, index: int) -> None:
        """Update item styles based on active index."""
        self._active_index = index

        for i, item in enumerate(self._items):
            if i == index:
                # Active: cyan, bold
                item.setStyleSheet(f"""
                    color: {COLORS['accent_primary']};
                    font-size: {FONTS['size_md']};
                    font-weight: 600;
                    padding: 4px 0px;
                    background: transparent;
                """)
            elif i < index:
                # Done: secondary color
                item.setStyleSheet(f"""
                    color: {COLORS['text_secondary']};
                    font-size: {FONTS['size_md']};
                    padding: 4px 0px;
                    background: transparent;
                """)
            else:
                # Pending: muted
                item.setStyleSheet(f"""
                    color: {COLORS['text_muted']};
                    font-size: {FONTS['size_md']};
                    padding: 4px 0px;
                    background: transparent;
                """)

    def reset(self) -> None:
        """Reset all activities to pending."""
        self._set_active_index(-1)

    def active_activity(self) -> str:
        """Get the currently active activity name."""
        if 0 <= self._active_index < len(self.ACTIVITIES):
            return self.ACTIVITIES[self._active_index]
        return ""


# ═══════════════════════════════════════════════════════════════
# Metrics Cards
# ═══════════════════════════════════════════════════════════════

class MetricsCards(QFrame):
    """
    Compact latency metrics cards: STT, Agent, TTS, Total.

    Uses existing latency/event data.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("metricsPanel")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        self._cards = {}
        for name in ("STT", "Agent", "TTS", "Total"):
            card = self._create_card(name)
            layout.addWidget(card)

        layout.addStretch()

    def _create_card(self, name: str) -> QFrame:
        """Create a single metric card."""
        card = QFrame()
        card.setObjectName("metricCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 8, 12, 8)
        card_layout.setSpacing(2)

        label = QLabel(name)
        label.setObjectName("metricLabel")
        label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: {FONTS['size_xs']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: transparent;
        """)
        card_layout.addWidget(label)

        value = QLabel("--")
        value.setObjectName("metricValue")
        value.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: {FONTS['size_lg']};
            font-weight: 600;
            font-family: {FONTS['mono']};
            background: transparent;
        """)
        card_layout.addWidget(value)

        self._cards[name] = value
        return card

    def set_stt_latency(self, ms: float) -> None:
        """Set STT latency in milliseconds."""
        self._cards["STT"].setText(f"{ms:.0f}ms")

    def set_agent_latency(self, ms: float) -> None:
        """Set Agent latency in milliseconds."""
        self._cards["Agent"].setText(f"{ms:.0f}ms")

    def set_tts_latency(self, ms: float) -> None:
        """Set TTS latency in milliseconds."""
        self._cards["TTS"].setText(f"{ms:.0f}ms")

    def set_total_latency(self, ms: float) -> None:
        """Set Total latency in milliseconds."""
        self._cards["Total"].setText(f"{ms:.0f}ms")

    def reset(self) -> None:
        """Reset all metrics."""
        for value in self._cards.values():
            value.setText("--")


# ═══════════════════════════════════════════════════════════════
# System Status
# ═══════════════════════════════════════════════════════════════

class SystemStatus(QFrame):
    """
    Footer system health display.

    Shows: STT ✓, Agent ✓, TTS ✓, Tools ✓
    Only shows current health, not logs.
    """

    COMPONENTS = ("STT", "Agent", "TTS", "Tools")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("systemStatus")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)

        self._items = {}
        for name in self.COMPONENTS:
            item = QLabel(f"{name} —")
            item.setObjectName("statusItem")
            item.setStyleSheet(f"""
                color: {COLORS['text_muted']};
                font-size: {FONTS['size_sm']};
                background: transparent;
            """)
            layout.addWidget(item)
            self._items[name] = item

        layout.addStretch()

    def set_status(self, component: str, ok: bool) -> None:
        """Set the health status of a component."""
        if component not in self._items:
            return

        item = self._items[component]
        if ok:
            item.setText(f"{component} ✓")
            item.setStyleSheet(f"""
                color: {COLORS['accent_success']};
                font-size: {FONTS['size_sm']};
                background: transparent;
            """)
        else:
            item.setText(f"{component} ✗")
            item.setStyleSheet(f"""
                color: {COLORS['accent_error']};
                font-size: {FONTS['size_sm']};
                background: transparent;
            """)

    def set_all_ok(self) -> None:
        """Set all components to OK status."""
        for name in self.COMPONENTS:
            self.set_status(name, True)

    def status(self, component: str) -> bool:
        """Get the status of a component (True = OK)."""
        if component in self._items:
            return "✓" in self._items[component].text()
        return False


# ═══════════════════════════════════════════════════════════════
# History Panel (collapsed/minimal)
# ═══════════════════════════════════════════════════════════════

class HistoryPanel(QFrame):
    """
    Minimal collapsed history of previous interactions.

    The current turn dominates the UI; history is kept minimal.
    """

    MAX_ITEMS = 3

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("historyPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(2)

        self._items: List[QLabel] = []
        self._history: List[str] = []

    def add_turn(self, user_text: str, response_text: str) -> None:
        """Add a completed turn to history."""
        # Keep only the summary
        summary = f"You: {user_text[:40]}{'...' if len(user_text) > 40 else ''}"
        self._history.insert(0, summary)
        self._history = self._history[:self.MAX_ITEMS]
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the displayed history."""
        # Clear existing
        for item in self._items:
            item.deleteLater()
        self._items.clear()

        layout = self.layout()
        for text in self._history:
            item = QLabel(text)
            item.setObjectName("historyItem")
            item.setStyleSheet(f"""
                color: {COLORS['text_muted']};
                font-size: {FONTS['size_sm']};
                padding: 2px 0px;
                background: transparent;
            """)
            layout.addWidget(item)
            self._items.append(item)

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._refresh()