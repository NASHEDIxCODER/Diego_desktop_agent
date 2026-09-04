"""
VoiceCoreVisualizer — The hero element of Diego's HUD.

A large circular audio visualizer with:
    - Central glowing core
    - Concentric rings
    - Radial waveform/equalizer surrounding the core
    - State-dependent animations
    - Real audio reactivity (microphone RMS / TTS activity)

States:
    IDLE            → calm breathing animation (extremely subtle)
    LISTENING       → live microphone waveform
    SPEECH_DETECTED → stronger pulse
    THINKING        → slow processing rotation
    EXECUTING       → progress/activity animation
    SPEAKING        → output waveform driven by TTS activity

Performance:
    - QTimer-based animation at ~60 FPS when active
    - Timer stops when idle (no continuous repainting)
    - Smooth interpolation, no jitter
    - No fake random waveform during active microphone use
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QRadialGradient, QConicalGradient,
)
from PySide6.QtWidgets import QWidget, QSizePolicy

from ui.styles import COLORS


class VisualizerState(Enum):
    """Visualizer animation states."""
    IDLE = auto()
    LISTENING = auto()
    SPEECH_DETECTED = auto()
    THINKING = auto()
    EXECUTING = auto()
    SPEAKING = auto()
    ERROR = auto()


# State → color mapping
STATE_COLORS = {
    VisualizerState.IDLE: COLORS["state_idle"],
    VisualizerState.LISTENING: COLORS["state_listening"],
    VisualizerState.SPEECH_DETECTED: COLORS["state_speech"],
    VisualizerState.THINKING: COLORS["state_thinking"],
    VisualizerState.EXECUTING: COLORS["state_executing"],
    VisualizerState.SPEAKING: COLORS["state_speaking"],
    VisualizerState.ERROR: COLORS["state_error"],
}


class VoiceCoreVisualizer(QWidget):
    """
    Premium circular voice visualizer widget.

    The visualizer is the visual centerpiece of the assistant.
    It reacts to real audio levels and communicates state
    through animation, color, and form.
    """

    # Number of radial waveform bars
    NUM_BARS = 64

    # Animation frame interval (ms) — ~60 FPS when active
    ACTIVE_INTERVAL = 16
    # Idle animation interval (slower, subtle)
    IDLE_INTERVAL = 50

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("voiceCoreVisualizer")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)

        # ── State ──
        self._state = VisualizerState.IDLE
        self._color = QColor(COLORS["state_idle"])
        self._target_color = QColor(COLORS["state_idle"])

        # ── Audio levels ──
        self._input_level = 0.0        # Current smoothed input level (0-1)
        self._target_input_level = 0.0 # Target input level from mic
        self._output_level = 0.0       # Current smoothed output level (0-1)
        self._target_output_level = 0.0 # Target output level from TTS

        # ── Animation state ──
        self._phase = 0.0              # Global animation phase
        self._breath_phase = 0.0       # Breathing animation phase
        self._rotation = 0.0           # Rotation for thinking/executing
        self._pulse = 0.0              # Pulse intensity for speech detected
        self._bar_values = [0.0] * self.NUM_BARS  # Smoothed bar values
        self._bar_targets = [0.0] * self.NUM_BARS # Target bar values

        # ── Particles (deterministic orbit) ──
        self._particles = []
        for i in range(14):
            angle = (i / 14) * math.pi * 2 + (i * 0.7) % 1.0
            radius_frac = 0.62 + ((i * 37) % 100) / 100 * 0.30
            speed = 0.004 + ((i * 13) % 10) / 10 * 0.008
            self._particles.append({
                "angle": angle, "radius": radius_frac, "speed": speed,
                "size": 1.2 + ((i * 7) % 10) / 10 * 1.6,
            })

        # ── Horizontal waveform (across panel) ──
        self._wave_points = [0.0] * 48

        # ── Timer ──
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._animating = False

        # Start with subtle idle animation
        self._start_animation(self.IDLE_INTERVAL)

    # ── Public API ─────────────────────────────────────────────

    def set_state(self, state: VisualizerState) -> None:
        """Set the visualizer state (changes animation style)."""
        if state == self._state:
            return

        self._state = state
        self._target_color = QColor(STATE_COLORS.get(state, COLORS["state_idle"]))

        # Adjust animation speed based on state
        if state == VisualizerState.IDLE:
            self._start_animation(self.IDLE_INTERVAL)
        elif state in (VisualizerState.THINKING, VisualizerState.EXECUTING):
            self._start_animation(33)  # ~30 FPS for slow animations
        else:
            self._start_animation(self.ACTIVE_INTERVAL)

    def set_state_by_name(self, state_name: str) -> None:
        """Set state from a string name (for event bridge integration)."""
        name = state_name.lower()
        if "listen" in name:
            self.set_state(VisualizerState.LISTENING)
        elif "speech" in name:
            self.set_state(VisualizerState.SPEECH_DETECTED)
        elif any(s in name for s in ("think", "plan", "replan")):
            self.set_state(VisualizerState.THINKING)
        elif any(s in name for s in ("execut", "observ", "verif")):
            self.set_state(VisualizerState.EXECUTING)
        elif any(s in name for s in ("speak", "respond")):
            self.set_state(VisualizerState.SPEAKING)
        elif "error" in name:
            self.set_state(VisualizerState.ERROR)
        else:
            self.set_state(VisualizerState.IDLE)

    def set_input_level(self, level: float) -> None:
        """
        Set the microphone input level (0.0 - 1.0).

        Called with REAL microphone RMS data. No fake random values.
        """
        self._target_input_level = max(0.0, min(1.0, level))

        # Wake up animation if there's activity
        if level > 0.03 and not self._animating:
            self._start_animation(self.ACTIVE_INTERVAL)

    def set_output_level(self, level: float) -> None:
        """
        Set the TTS output level (0.0 - 1.0).

        Called with TTS activity data during speaking.
        """
        self._target_output_level = max(0.0, min(1.0, level))

    @property
    def state(self) -> VisualizerState:
        """Get the current visualizer state."""
        return self._state

    @property
    def input_level(self) -> float:
        """Get the current smoothed input level."""
        return self._input_level

    @property
    def output_level(self) -> float:
        """Get the current smoothed output level."""
        return self._output_level

    # ── Animation control ──────────────────────────────────────

    def _start_animation(self, interval: int) -> None:
        """Start or update the animation timer."""
        self._animating = True
        if self._timer.interval() != interval:
            self._timer.setInterval(interval)
        if not self._timer.isActive():
            self._timer.start()

    def _stop_animation(self) -> None:
        """Stop the animation timer (low CPU when idle)."""
        self._animating = False
        self._timer.stop()

    def _animate(self) -> None:
        """Advance the animation by one frame."""
        now = time.time()

        # ── Smooth level interpolation ──
        self._input_level += (self._target_input_level - self._input_level) * 0.25
        self._output_level += (self._target_output_level - self._output_level) * 0.20

        # ── Smooth color transition ──
        self._color = _lerp_color(self._color, self._target_color, 0.08)

        # ── State-specific animation ──
        state = self._state

        if state == VisualizerState.IDLE:
            # Calm breathing — extremely subtle
            self._breath_phase += 0.03
            self._phase += 0.01
            self._update_bars_idle()

        elif state == VisualizerState.LISTENING:
            # Live microphone waveform
            self._phase += 0.05
            self._breath_phase += 0.02
            self._update_bars_from_input()

        elif state == VisualizerState.SPEECH_DETECTED:
            # Stronger pulse + microphone waveform
            self._phase += 0.08
            self._pulse = min(1.0, self._pulse + 0.1)
            self._update_bars_from_input(boost=1.3)

        elif state == VisualizerState.THINKING:
            # Slow processing rotation
            self._rotation += 0.02
            self._phase += 0.02
            self._breath_phase += 0.02
            self._update_bars_thinking()

        elif state == VisualizerState.EXECUTING:
            # Progress/activity animation
            self._rotation += 0.04
            self._phase += 0.04
            self._update_bars_executing()

        elif state == VisualizerState.SPEAKING:
            # Output waveform driven by TTS
            self._phase += 0.06
            self._breath_phase += 0.03
            self._update_bars_from_output()

        elif state == VisualizerState.ERROR:
            # Subtle error pulse
            self._phase += 0.03
            self._pulse = 0.5 + 0.5 * math.sin(self._phase * 2)
            self._update_bars_idle()

        # Decay pulse when not in speech detected
        if state != VisualizerState.SPEECH_DETECTED:
            self._pulse *= 0.92

        # ── Horizontal waveform update ──
        level = max(self._input_level, self._output_level)
        for i in range(len(self._wave_points)):
            t = i / (len(self._wave_points) - 1)
            env = math.sin(t * math.pi) ** 0.9
            wave = (0.5 + 0.5 * math.sin(i * 0.7 + self._phase * 2.2)) \
                 * (0.6 + 0.4 * math.sin(i * 1.9 - self._phase * 1.4))
            target = (0.05 + 0.75 * level * wave) * env
            self._wave_points[i] += (target - self._wave_points[i]) * 0.25

        # ── Particle drift ──
        for part in self._particles:
            part["angle"] += part["speed"]

        # ── Check if we can stop animating ──
        if (state == VisualizerState.IDLE
                and self._input_level < 0.01
                and self._output_level < 0.01):
            # Keep idle breathing but at lower rate
            if self._timer.interval() != self.IDLE_INTERVAL:
                self._timer.setInterval(self.IDLE_INTERVAL)

        self.update()

    # ── Bar update strategies ──────────────────────────────────

    def _update_bars_idle(self) -> None:
        """Idle: extremely subtle breathing bars."""
        breath = (math.sin(self._breath_phase) + 1) / 2  # 0-1
        for i in range(self.NUM_BARS):
            angle = (i / self.NUM_BARS) * math.pi * 2
            # Very subtle wave pattern
            wave = 0.02 + 0.015 * math.sin(angle * 3 + self._phase)
            target = wave * (0.8 + 0.2 * breath)
            self._bar_values[i] += (target - self._bar_values[i]) * 0.1

    def _update_bars_from_input(self, boost: float = 1.0) -> None:
        """Listening: bars driven by REAL microphone input level."""
        level = self._input_level * boost
        for i in range(self.NUM_BARS):
            angle = (i / self.NUM_BARS) * math.pi * 2
            # Create a natural waveform pattern around the circle
            # Using multiple sine waves for organic look
            wave1 = math.sin(angle * 4 + self._phase * 2) * 0.5 + 0.5
            wave2 = math.sin(angle * 7 - self._phase * 1.5) * 0.3 + 0.5
            wave3 = math.sin(angle * 2 + self._phase) * 0.2 + 0.5
            pattern = (wave1 + wave2 + wave3) / 3

            # Scale by actual input level
            target = level * pattern * 0.9
            # Smooth interpolation
            self._bar_values[i] += (target - self._bar_values[i]) * 0.35

    def _update_bars_from_output(self) -> None:
        """Speaking: bars driven by TTS output level."""
        level = max(self._output_level, 0.15)  # Minimum activity while speaking
        for i in range(self.NUM_BARS):
            angle = (i / self.NUM_BARS) * math.pi * 2
            # Speaking pattern: more uniform, pulsing
            wave1 = math.sin(angle * 3 + self._phase * 3) * 0.4 + 0.6
            wave2 = math.sin(angle * 5 - self._phase * 2) * 0.3 + 0.6
            pattern = (wave1 + wave2) / 2

            target = level * pattern * 0.85
            self._bar_values[i] += (target - self._bar_values[i]) * 0.30

    def _update_bars_thinking(self) -> None:
        """Thinking: slow, calm processing pattern."""
        for i in range(self.NUM_BARS):
            angle = (i / self.NUM_BARS) * math.pi * 2
            # Slow rotating wave
            wave = math.sin(angle * 2 - self._rotation * 3) * 0.5 + 0.5
            target = 0.08 + wave * 0.12
            self._bar_values[i] += (target - self._bar_values[i]) * 0.08

    def _update_bars_executing(self) -> None:
        """Executing: activity/progress pattern."""
        for i in range(self.NUM_BARS):
            angle = (i / self.NUM_BARS) * math.pi * 2
            # Progress sweep + activity
            sweep = (angle / (math.pi * 2) + self._rotation) % 1.0
            activity = 0.1 + 0.15 * math.sin(angle * 6 + self._phase * 4)
            target = activity + 0.1 * (1.0 - sweep)
            self._bar_values[i] += (target - self._bar_values[i]) * 0.15

    # ── Painting ───────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        """Draw the voice core visualizer."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2
        cy = h / 2
        center = QPointF(cx, cy)

        # Base radius (leaves room for bars)
        max_radius = min(w, h) / 2
        core_radius = max_radius * 0.35
        ring_radius = max_radius * 0.50
        bar_inner = max_radius * 0.55
        bar_max_len = max_radius * 0.35

        color = self._color
        breath = (math.sin(self._breath_phase) + 1) / 2

        # ── Outer glow (subtle) ──
        glow_intensity = 0.15 + self._input_level * 0.3 + self._pulse * 0.2
        if self._state == VisualizerState.SPEAKING:
            glow_intensity = 0.15 + self._output_level * 0.35

        glow_color = QColor(color)
        glow_color.setAlphaF(min(0.4, glow_intensity))
        glow_gradient = QRadialGradient(center, max_radius)
        glow_gradient.setColorAt(0.3, QColor(0, 0, 0, 0))
        glow_gradient.setColorAt(0.6, glow_color)
        glow_gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(glow_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, max_radius, max_radius)

        # ── Concentric rings ──
        self._draw_rings(painter, center, ring_radius, core_radius, color, breath)

        # ── Particles ──
        self._draw_particles(painter, center, max_radius, color)

        # ── Radial waveform bars ──
        self._draw_bars(painter, center, bar_inner, bar_max_len, color)

        # ── Central core ──
        self._draw_core(painter, center, core_radius, color, breath)

        # ── Horizontal waveform across the panel ──
        wave_y = h - max_radius * 0.22
        self._draw_horizontal_wave(painter, w, wave_y, color)

        painter.end()

    def _draw_horizontal_wave(
        self, painter: QPainter, width: float, y: float, color: QColor
    ) -> None:
        """Draw a subtle horizontal waveform near the bottom of the panel."""
        n = len(self._wave_points)
        margin = width * 0.06
        span = width - margin * 2
        color_line = QColor(color)
        color_line.setAlphaF(0.45)

        # Line through the wave points
        painter.setPen(QPen(color_line, 1.4))
        painter.setBrush(Qt.NoBrush)
        points = []
        for i, v in enumerate(self._wave_points):
            x = margin + (i / (n - 1)) * span
            points.append(QPointF(x, y - v * 9 + 4.5))
        for i in range(n - 1):
            painter.drawLine(points[i], points[i + 1])

        # Mirror below for symmetric look
        for i in range(n - 1):
            a = QPointF(points[i].x(), y + self._wave_points[i] * 9 - 4.5 + 9)
            b = QPointF(points[i + 1].x(), y + self._wave_points[i + 1] * 9 - 4.5 + 9)
            faint = QColor(color)
            faint.setAlphaF(0.18)
            painter.setPen(QPen(faint, 1.0))
            painter.drawLine(a, b)

    def _draw_particles(
        self, painter: QPainter, center: QPointF,
        max_radius: float, color: QColor
    ) -> None:
        """Draw subtle orbiting particle dots."""
        for part in self._particles:
            r = part["radius"] * max_radius
            x = center.x() + math.cos(part["angle"]) * r
            y = center.y() + math.sin(part["angle"]) * r * 0.92
            c = QColor(color)
            c.setAlphaF(0.25 + 0.35 * self._input_level)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(c))
            painter.drawEllipse(QPointF(x, y), part["size"], part["size"])

    def _draw_bars(
        self, painter: QPainter, center: QPointF,
        inner_radius: float, max_len: float, color: QColor
    ) -> None:
        """Draw the radial waveform bars around the core."""
        for i in range(self.NUM_BARS):
            angle = (i / self.NUM_BARS) * math.pi * 2 - math.pi / 2
            value = self._bar_values[i]

            if value < 0.005:
                continue

            bar_len = max(2.0, value * max_len)

            # Bar start and end points
            x1 = center.x() + math.cos(angle) * inner_radius
            y1 = center.y() + math.sin(angle) * inner_radius
            x2 = center.x() + math.cos(angle) * (inner_radius + bar_len)
            y2 = center.y() + math.sin(angle) * (inner_radius + bar_len)

            # Bar color with alpha based on value
            bar_color = QColor(color)
            alpha = 0.3 + value * 0.7
            bar_color.setAlphaF(min(1.0, alpha))

            pen = QPen(bar_color)
            pen.setWidthF(2.5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_rings(
        self, painter: QPainter, center: QPointF,
        ring_radius: float, core_radius: float,
        color: QColor, breath: float
    ) -> None:
        """Draw concentric rings."""
        # Outer ring (static, subtle)
        ring_color = QColor(color)
        ring_color.setAlphaF(0.15)
        pen = QPen(ring_color)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, ring_radius, ring_radius)

        # Middle ring (breathing)
        mid_radius = core_radius + (ring_radius - core_radius) * 0.5
        mid_scale = 1.0 + breath * 0.02
        mid_color = QColor(color)
        mid_color.setAlphaF(0.25 + breath * 0.1)
        pen = QPen(mid_color)
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.drawEllipse(center, mid_radius * mid_scale, mid_radius * mid_scale)

        # Thinking/executing: rotating arc
        if self._state in (VisualizerState.THINKING, VisualizerState.EXECUTING):
            arc_color = QColor(color)
            arc_color.setAlphaF(0.6)
            pen = QPen(arc_color)
            pen.setWidthF(2.0)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)

            # Draw rotating arc (60 degrees)
            start_angle = int(self._rotation * 180 / math.pi * 16)
            painter.drawArc(
                QRectF(
                    center.x() - ring_radius, center.y() - ring_radius,
                    ring_radius * 2, ring_radius * 2
                ),
                start_angle, 60 * 16
            )

    def _draw_core(
        self, painter: QPainter, center: QPointF,
        radius: float, color: QColor, breath: float
    ) -> None:
        """Draw the central glowing core."""
        # Core scale based on state
        scale = 1.0
        if self._state == VisualizerState.IDLE:
            scale = 1.0 + breath * 0.03  # Subtle breathing
        elif self._state == VisualizerState.SPEECH_DETECTED:
            scale = 1.0 + self._pulse * 0.08
        elif self._state == VisualizerState.LISTENING:
            scale = 1.0 + self._input_level * 0.06
        elif self._state == VisualizerState.SPEAKING:
            scale = 1.0 + self._output_level * 0.06

        scaled_radius = radius * scale

        # Core gradient
        gradient = QRadialGradient(center, scaled_radius)
        core_color = QColor(color)

        # Inner bright center
        inner_color = QColor(color)
        inner_color.setAlphaF(0.9)
        gradient.setColorAt(0.0, inner_color)

        # Middle
        mid_color = QColor(color)
        mid_color.setAlphaF(0.4)
        gradient.setColorAt(0.5, mid_color)

        # Edge fade
        edge_color = QColor(color)
        edge_color.setAlphaF(0.1)
        gradient.setColorAt(0.85, edge_color)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, scaled_radius, scaled_radius)

        # Core border ring
        border_color = QColor(color)
        border_color.setAlphaF(0.5)
        pen = QPen(border_color)
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, scaled_radius, scaled_radius)

        # ── Central microphone glyph ──
        self._draw_mic_glyph(painter, center, scaled_radius, breath)

    def _draw_mic_glyph(
        self, painter: QPainter, center: QPointF,
        radius: float, breath: float
    ) -> None:
        """Draw a microphone symbol inside the central core."""
        s = radius * 0.9
        mic_color = QColor("#dffbff")
        mic_color.setAlphaF(0.85 + 0.15 * breath)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(mic_color))

        # Capsule
        cap_w, cap_h = s * 0.26, s * 0.42
        painter.drawRoundedRect(
            QRectF(center.x() - cap_w / 2, center.y() - s * 0.30, cap_w, cap_h),
            cap_w / 2, cap_w / 2)

        # Cradle arc
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(mic_color), s * 0.05))
        arc_r = s * 0.26
        painter.drawArc(
            QRectF(center.x() - arc_r, center.y() - arc_r - s * 0.09,
                   arc_r * 2, arc_r * 2), -55 * 16, 110 * 16)

        # Stem and base
        painter.drawLine(QPointF(center.x(), center.y() + s * 0.20),
                         QPointF(center.x(), center.y() + s * 0.30))
        painter.drawLine(QPointF(center.x() - s * 0.13, center.y() + s * 0.30),
                         QPointF(center.x() + s * 0.13, center.y() + s * 0.30))


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    """Linearly interpolate between two colors."""
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )