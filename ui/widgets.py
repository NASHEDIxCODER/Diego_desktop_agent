"""
Diego UI Widgets — HUD widgets matching the Diego reference design.

Includes:
    - AvatarBadge: painted circular avatar (user / diego / logo)
    - MicGlyph: circular-glow microphone icon
    - ConnectionIndicator: compact LIVE state pill (header)
    - TranscriptPanel: "YOU SAID" glass card (avatar + Final ✓ + thin waveform)
    - ResponsePanel: "DIEGO" prominent glass card (avatar + project card + waveform)
    - VoiceStatePanel: mic icon + state progression (Listen → … → Respond)
    - ActivityPanel: right-side timeline with timestamps
    - MetricsCards: LATENCY (LIVE) compact cards
    - SystemStatus: SYSTEM STATUS component checks
    - FooterBar: version / status / activity indicator
    - SpeakingIndicator / ThinWaveform / HistoryPanel (kept for compatibility)
"""

from __future__ import annotations

import math
import re
from typing import Optional, List

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QRadialGradient, QLinearGradient,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
    QGraphicsOpacityEffect, QGridLayout,
)

from ui.tokens import (
    WINDOW_BACKGROUND, PANEL_BACKGROUND, PANEL_BACKGROUND_HI, PANEL_BORDER,
    PANEL_BORDER_SOFT, PRIMARY_ACCENT, PRIMARY_ACCENT_DIM, SECONDARY_ACCENT,
    SUCCESS, WARNING, ERROR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_FAMILY, FONT_MONO, FONT_XS, FONT_SM, FONT_MD, FONT_LG, FONT_XL,
    SPACING_SMALL, SPACING_MEDIUM, SPACING_LARGE,
)

# Backward-compatible aliases (ui.styles re-exports these too)
COLORS = {
    "accent_primary": PRIMARY_ACCENT,
    "accent_secondary": SECONDARY_ACCENT,
    "accent_success": SUCCESS,
    "accent_warning": WARNING,
    "accent_error": ERROR,
    "text_primary": TEXT_PRIMARY,
    "text_secondary": TEXT_SECONDARY,
    "text_muted": TEXT_MUTED,
}
FONTS = {
    "size_xs": f"{FONT_XS}px", "size_sm": f"{FONT_SM}px",
    "size_md": f"{FONT_MD}px", "size_lg": f"{FONT_LG}px",
    "size_xl": f"{FONT_XL}px",
    "mono": FONT_MONO, "family": FONT_FAMILY,
}


def _font(size: int, weight: int = QFont.Normal, mono: bool = False) -> QFont:
    f = QFont("DejaVu Sans" if not mono else "DejaVu Sans Mono")
    f.setPixelSize(size)
    f.setWeight(weight)
    return f


# ═══════════════════════════════════════════════════════════════
# Painted avatar / glyph widgets
# ═══════════════════════════════════════════════════════════════

class AvatarBadge(QWidget):
    """
    Painted circular avatar with soft glow.

    kind: "user" (person silhouette), "diego" (robot face), "logo" (letter D)
    """

    def __init__(self, kind: str = "user", diameter: int = 34,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._kind = kind
        self._diameter = diameter
        self.setFixedSize(diameter + 8, diameter + 8)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        d = self._diameter
        rect = QRectF((self.width() - d) / 2, (self.height() - d) / 2, d, d)
        center = rect.center()

        # Outer glow
        glow = QRadialGradient(center, d * 0.85)
        glow_color = QColor(PRIMARY_ACCENT if self._kind != "user" else "#3b82f6")
        glow_color.setAlphaF(0.22)
        glow.setColorAt(0.55, QColor(0, 0, 0, 0))
        glow.setColorAt(0.8, glow_color)
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(center, d * 0.85, d * 0.85)

        # Body
        body = QLinearGradient(rect.topLeft(), rect.bottomRight())
        if self._kind == "user":
            body.setColorAt(0.0, QColor("#16283c"))
            body.setColorAt(1.0, QColor("#0d1522"))
            ring = QColor("#3b82f6")
        elif self._kind == "diego":
            body.setColorAt(0.0, QColor("#0e2a35"))
            body.setColorAt(1.0, QColor("#0a141f"))
            ring = QColor(PRIMARY_ACCENT)
        else:  # logo
            body.setColorAt(0.0, QColor("#0e2a35"))
            body.setColorAt(1.0, QColor("#0a141f"))
            ring = QColor(PRIMARY_ACCENT)
        p.setBrush(QBrush(body))
        p.setPen(QPen(ring, 1.4))
        p.drawEllipse(rect)

        # Glyph
        p.setPen(Qt.NoPen)
        glyph = QColor(ring)
        glyph.setAlphaF(0.9)
        p.setBrush(QBrush(glyph))

        if self._kind == "user":
            # Head
            p.drawEllipse(QPointF(center.x(), center.y() - d * 0.13), d * 0.14, d * 0.14)
            # Shoulders
            shoulder = QPainterPath()
            shoulder.addRoundedRect(
                QRectF(center.x() - d * 0.26, center.y() + d * 0.06,
                       d * 0.52, d * 0.28), d * 0.14, d * 0.14)
            p.drawPath(shoulder)
        elif self._kind == "diego":
            # Robot: two eyes + mouth bar
            eye = QColor("#bff3fb")
            p.setBrush(QBrush(eye))
            p.drawEllipse(QPointF(center.x() - d * 0.14, center.y() - d * 0.08),
                          d * 0.07, d * 0.07)
            p.drawEllipse(QPointF(center.x() + d * 0.14, center.y() - d * 0.08),
                          d * 0.07, d * 0.07)
            p.setBrush(QBrush(glyph))
            mouth = QRectF(center.x() - d * 0.16, center.y() + d * 0.10,
                           d * 0.32, d * 0.05)
            p.drawRoundedRect(mouth, d * 0.02, d * 0.02)
            # Antenna
            p.setPen(QPen(QColor(glyph), 1.4))
            p.drawLine(QPointF(center.x(), center.y() - d * 0.26),
                       QPointF(center.x(), center.y() - d * 0.36))
            p.drawEllipse(QPointF(center.x(), center.y() - d * 0.39), d * 0.035, d * 0.035)
        else:  # logo "D"
            p.setPen(QPen(QColor(ring), 1))
            f = _font(int(d * 0.48), QFont.Bold)
            f.setFamily("DejaVu Sans")
            p.setFont(f)
            p.setPen(QPen(QColor("#bff3fb")))
            p.setBrush(Qt.NoBrush)
            p.drawText(rect, Qt.AlignCenter, "D")
        p.end()


class MicGlyph(QWidget):
    """Microphone icon inside a circular cyan glow (Voice State panel)."""

    def __init__(self, size: int = 44, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._size = size
        self._phase = 0.0
        self.setFixedSize(size + 12, size + 12)
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self._phase += 0.12
        self.update()

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self._size
        rect = QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s)
        center = rect.center()

        pulse = 0.5 + 0.5 * math.sin(self._phase)

        glow = QRadialGradient(center, s * 0.9)
        g1 = QColor(PRIMARY_ACCENT); g1.setAlphaF(0.10 + 0.10 * pulse)
        g2 = QColor(PRIMARY_ACCENT); g2.setAlphaF(0.0)
        glow.setColorAt(0.0, g1)
        glow.setColorAt(0.6, g1)
        glow.setColorAt(1.0, g2)
        p.setBrush(QBrush(glow)); p.setPen(Qt.NoPen)
        p.drawEllipse(center, s * 0.9, s * 0.9)

        p.setBrush(QBrush(QColor("#0d2230")))
        p.setPen(QPen(QColor(PRIMARY_ACCENT), 1.2))
        p.drawEllipse(rect)

        # Mic capsule
        mic = QColor("#bff3fb")
        p.setBrush(QBrush(mic)); p.setPen(Qt.NoPen)
        cap_w, cap_h = s * 0.24, s * 0.36
        p.drawRoundedRect(
            QRectF(center.x() - cap_w / 2, center.y() - s * 0.24, cap_w, cap_h),
            cap_w / 2, cap_w / 2)
        # Arc
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(mic), 1.6))
        arc_r = s * 0.22
        p.drawArc(QRectF(center.x() - arc_r, center.y() - arc_r - s * 0.06,
                         arc_r * 2, arc_r * 2), -50 * 16, 100 * 16)
        # Stem + base
        p.drawLine(QPointF(center.x(), center.y() + s * 0.16),
                   QPointF(center.x(), center.y() + s * 0.24))
        p.drawLine(QPointF(center.x() - s * 0.10, center.y() + s * 0.24),
                   QPointF(center.x() + s * 0.10, center.y() + s * 0.24))
        p.end()


class FolderGlyph(QWidget):
    """Small painted folder icon (project card)."""

    def __init__(self, size: int = 22, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self._size
        body = QRectF(1, s * 0.22, s - 2, s * 0.62)
        tab = QRectF(1, s * 0.12, s * 0.42, s * 0.16)
        p.setBrush(QBrush(QColor("#1b3a52")))
        p.setPen(QPen(QColor(PRIMARY_ACCENT), 1.1))
        p.drawRoundedRect(tab, 2, 2)
        p.drawRoundedRect(body, 3, 3)
        p.end()


# ═══════════════════════════════════════════════════════════════
# Thin waveform (transcript / footer / response)
# ═══════════════════════════════════════════════════════════════

class ThinWaveform(QWidget):
    """
    Slim waveform strip. Renders a subtle symmetric bar pattern.
    When animated (footer/response), bars react to `level` smoothly.
    """

    NUM_BARS = 40

    def __init__(self, color: str = PRIMARY_ACCENT, bar_height: int = 14,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = QColor(color)
        self._base_height = bar_height
        self._level = 0.0
        self._target_level = 0.0
        self._phase = 0.0
        self._animated = False
        self.setFixedHeight(bar_height + 4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def set_level(self, level: float) -> None:
        self._target_level = max(0.0, min(1.0, level))

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def start(self) -> None:
        self._animated = True
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._animated = False
        self._target_level = 0.0
        self._level = 0.0
        self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._level += (self._target_level - self._level) * 0.3
        self._phase += 0.14
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        n = self.NUM_BARS
        step = w / n
        bar_w = max(1.5, step * 0.45)
        lvl = max(self._level, 0.0)

        for i in range(n):
            t = i / (n - 1)
            # Gentle symmetric envelope: quiet at the edges
            env = math.sin(t * math.pi) ** 0.8
            wave = 0.5 + 0.5 * math.sin(i * 0.55 + self._phase)
            amp = (0.10 + 0.55 * lvl * wave) * env
            bar_h = max(1.5, amp * (self._base_height - 2))
            c = QColor(self._color)
            c.setAlphaF(0.25 + 0.55 * amp)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(c))
            x = i * step + (step - bar_w) / 2
            p.drawRoundedRect(QRectF(x, mid - bar_h / 2, bar_w, bar_h),
                              bar_w / 2, bar_w / 2)
        p.end()


# ═══════════════════════════════════════════════════════════════
# Connection Indicator (header LIVE pill)
# ═══════════════════════════════════════════════════════════════

class ConnectionIndicator(QFrame):
    """Compact LIVE state pill: '● LISTENING'."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("connectionIndicator")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setObjectName("connectionDot")
        self._dot.setFixedWidth(8)
        layout.addWidget(self._dot)

        self._text = QLabel("Ready")
        self._text.setObjectName("connectionText")
        layout.addWidget(self._text)

        self.set_status("ready")

    def set_status(self, status: str) -> None:
        if status == "ready":
            color, text = SUCCESS, "ONLINE"
        elif status == "connecting":
            color, text = WARNING, "CONNECTING"
        elif status == "listening":
            color, text = PRIMARY_ACCENT, "LISTENING"
        elif status == "speaking":
            color, text = PRIMARY_ACCENT, "SPEAKING"
        elif status == "error":
            color, text = ERROR, "ERROR"
        else:
            color, text = TEXT_SECONDARY, status.upper()

        self._dot.setStyleSheet(
            f"color: {color}; font-size: 8px; background: transparent;")
        self._text.setText(text)
        self._text.setStyleSheet(
            f"color: {color}; font-size: {FONT_XS}px; font-weight: 700; "
            f"letter-spacing: 1.2px; background: transparent;")

    def status(self) -> str:
        return self._text.text()


# ═══════════════════════════════════════════════════════════════
# Transcript Panel ("YOU SAID")
# ═══════════════════════════════════════════════════════════════

class TranscriptPanel(QFrame):
    """
    "YOU SAID" glass card:
        [user avatar] YOU SAID ......... Final ✓
                      "transcript text"
                      ~ thin waveform ~
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("transcriptPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(SPACING_MEDIUM, SPACING_SMALL + 2,
                                 SPACING_MEDIUM, SPACING_SMALL + 2)
        outer.setSpacing(SPACING_MEDIUM)

        self._avatar = AvatarBadge("user", 34)
        outer.addWidget(self._avatar, 0, Qt.AlignTop)

        body = QVBoxLayout()
        body.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self._header = QLabel("YOU SAID")
        self._header.setObjectName("transcriptHeader")
        header_row.addWidget(self._header)
        header_row.addStretch()

        self._final_badge = QLabel("Final ✓")
        self._final_badge.setObjectName("finalBadge")
        self._final_badge.hide()
        header_row.addWidget(self._final_badge)

        body.addLayout(header_row)

        self._text_label = QLabel("")
        self._text_label.setObjectName("transcriptText")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text_label.setMinimumHeight(24)
        body.addWidget(self._text_label)

        self._wave = ThinWaveform(PRIMARY_ACCENT, bar_height=10)
        body.addWidget(self._wave)

        outer.addLayout(body, 1)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self._text_label.setGraphicsEffect(self._opacity)

        self._is_partial = False

    def set_partial(self, text: str) -> None:
        if not text.strip():
            return
        self._is_partial = True
        self._final_badge.hide()
        self._text_label.setObjectName("transcriptTextPartial")
        self._text_label.setText(text)
        self._apply_text_style()
        self._wave.start()

    def set_final(self, text: str) -> None:
        if not text.strip():
            return
        self._transition_to(text, is_partial=False)

    def _transition_to(self, text: str, is_partial: bool) -> None:
        self._is_partial = is_partial
        self._opacity.setOpacity(0.3)
        self._text_label.setObjectName(
            "transcriptTextPartial" if is_partial else "transcriptText")
        self._text_label.setText(text)
        self._apply_text_style()
        self._opacity.setOpacity(1.0)
        self._final_badge.setVisible(not is_partial)
        if not is_partial:
            self._wave.stop()

    def _apply_text_style(self) -> None:
        if self._is_partial:
            self._text_label.setStyleSheet(f"""
                #transcriptTextPartial {{
                    color: {TEXT_SECONDARY};
                    font-size: {FONT_XL}px;
                    font-style: italic;
                    background: transparent;
                }}
            """)
        else:
            self._text_label.setStyleSheet(f"""
                #transcriptText {{
                    color: {TEXT_PRIMARY};
                    font-size: {FONT_XL}px;
                    background: transparent;
                }}
            """)

    def clear(self) -> None:
        self._text_label.setText("")
        self._is_partial = False
        self._final_badge.hide()

    def text(self) -> str:
        return self._text_label.text()

    def is_partial(self) -> bool:
        return self._is_partial


# ═══════════════════════════════════════════════════════════════
# Response Panel ("DIEGO") — the dominant card
# ═══════════════════════════════════════════════════════════════

class ResponsePanel(QFrame):
    """
    "DIEGO" prominent glass card:
        [diego avatar] DIEGO ........ speaking indicator
        response intro text
        [folder] highlighted project card (when a project path is present)
        path / trailing text ..... purple/cyan waveform (lower-right)
    """

    _PATH_RE = re.compile(r"(~[\w/\-.]*(?:projects|Projects)[\w/\-.]*/?[\w\-.]*)")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("responsePanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(SPACING_MEDIUM, SPACING_SMALL + 2,
                                 SPACING_MEDIUM, SPACING_SMALL + 2)
        outer.setSpacing(SPACING_MEDIUM)

        self._avatar = AvatarBadge("diego", 44)
        outer.addWidget(self._avatar, 0, Qt.AlignTop)

        body = QVBoxLayout()
        body.setSpacing(5)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self._header = QLabel("DIEGO")
        self._header.setObjectName("responseHeader")
        header_row.addWidget(self._header)
        header_row.addStretch()

        self._speaking_widget = SpeakingIndicator()
        self._speaking_widget.hide()
        header_row.addWidget(self._speaking_widget)
        body.addLayout(header_row)

        self._text_label = QLabel("")
        self._text_label.setObjectName("responseText")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text_label.setMinimumHeight(24)
        body.addWidget(self._text_label)

        # Highlighted project card (hidden unless a project path is found)
        self._project_card = QFrame()
        self._project_card.setObjectName("projectCard")
        pc_layout = QHBoxLayout(self._project_card)
        pc_layout.setContentsMargins(10, 7, 10, 7)
        pc_layout.setSpacing(10)
        self._folder_icon = FolderGlyph(22)
        pc_layout.addWidget(self._folder_icon, 0, Qt.AlignVCenter)
        pc_text = QVBoxLayout()
        pc_text.setSpacing(1)
        self._project_name = QLabel("")
        self._project_name.setObjectName("projectName")
        pc_text.addWidget(self._project_name)
        self._project_meta = QLabel("")
        self._project_meta.setObjectName("projectMeta")
        pc_text.addWidget(self._project_meta)
        pc_layout.addLayout(pc_text, 1)
        self._project_card.hide()
        body.addWidget(self._project_card)

        # Footer row: path text + waveform lower-right
        footer_row = QHBoxLayout()
        footer_row.setSpacing(10)
        self._path_label = QLabel("")
        self._path_label.setObjectName("responsePath")
        self._path_label.hide()
        footer_row.addWidget(self._path_label, 1, Qt.AlignBottom)
        self._mini_waveform = ThinWaveform(SECONDARY_ACCENT, bar_height=14)
        self._mini_waveform.setMinimumWidth(140)
        self._mini_waveform.setMaximumWidth(240)
        footer_row.addWidget(self._mini_waveform, 0, Qt.AlignBottom)
        body.addLayout(footer_row)

        outer.addLayout(body, 1)

    # ── API ──

    def set_response(self, text: str) -> None:
        self._text_label.setStyleSheet(f"""
            #responseText {{
                color: {TEXT_PRIMARY};
                font-size: {FONT_XL}px;
                font-weight: 500;
                background: transparent;
            }}
        """)
        self._text_label.setText(text)
        self._layout_response(text)

    def set_error(self, message: str) -> None:
        self._hide_project_card()
        self._text_label.setStyleSheet(f"""
            #responseText {{
                color: {ERROR};
                font-size: {FONT_LG}px;
                background: transparent;
            }}
        """)
        self._text_label.setText(message)

    def set_speaking(self, speaking: bool) -> None:
        self._speaking_widget.setVisible(speaking)
        if speaking:
            self._speaking_widget.start()
            self._mini_waveform.start()
        else:
            self._speaking_widget.stop()
            # Waveform settles back to a calm idle strip (still visible,
            # matching the reference's permanent lower-right waveform)
            self._mini_waveform.stop()

    def set_output_level(self, level: float) -> None:
        self._mini_waveform.set_level(level)

    def clear(self) -> None:
        self._text_label.setText("")
        self._hide_project_card()

    def text(self) -> str:
        return self._text_label.text()

    # ── Project-card extraction ──

    def _layout_response(self, text: str) -> None:
        match = self._PATH_RE.search(text)
        if not match:
            self._hide_project_card()
            return

        path = match.group(1).rstrip(".,!?;:")
        project_name = path.rstrip("/").split("/")[-1].replace("-", " ").title()
        # Intro = everything before the sentence containing the path
        intro_end = text.find(path)
        intro = text[:intro_end].strip().rstrip(":").strip()
        # Trailing = everything after the path's sentence end
        tail_idx = text.find(path) + len(match.group(0))
        tail = text[tail_idx:].lstrip(" .!:").strip()

        if intro:
            self._text_label.setText(intro + ("" if intro.endswith((":", ".")) else ":"))
        else:
            self._text_label.setText("")

        self._project_name.setText(project_name)
        self._project_meta.setText(path)
        self._project_card.show()
        self._path_label.hide()

        if tail:
            self._path_label.setText(tail)
            self._path_label.show()

    def _hide_project_card(self) -> None:
        self._project_card.hide()
        self._path_label.hide()


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
            color: {PRIMARY_ACCENT};
            font-size: {FONT_SM}px;
            font-weight: 600;
            background: transparent;
        """)
        layout.addWidget(self._label)

        self._dots = QLabel("···")
        self._dots.setStyleSheet(f"""
            color: {PRIMARY_ACCENT};
            font-size: {FONT_SM}px;
            background: transparent;
        """)
        layout.addWidget(self._dots)

        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._animate)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._dots.setText("···")

    def _animate(self) -> None:
        self._phase = (self._phase + 1) % 4
        self._dots.setText("·" * (self._phase + 1))


# ═══════════════════════════════════════════════════════════════
# Voice State Panel (right column, panel 1)
# ═══════════════════════════════════════════════════════════════

class VoiceStatePanel(QFrame):
    """
    VOICE STATE panel:
        [mic glyph]  Listening
        Listen → Think → Plan → Execute → Respond  (current highlighted)
    """

    STEPS = ("Listen", "Think", "Plan", "Execute", "Respond")

    # state name → highlighted step index
    _STATE_MAP = {
        "listen": 0, "speech": 0,
        "think": 1, "replan": 1,
        "plan": 2,
        "execut": 3, "observ": 3, "verif": 3,
        "speak": 4, "respond": 4,
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rightPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MEDIUM, SPACING_SMALL + 2,
                                  SPACING_MEDIUM, SPACING_SMALL + 2)
        layout.setSpacing(8)

        self._title = QLabel("VOICE STATE")
        self._title.setObjectName("panelTitle")
        layout.addWidget(self._title)

        row = QHBoxLayout()
        row.setSpacing(10)
        self._mic = MicGlyph(38)
        self._mic.start()
        row.addWidget(self._mic, 0, Qt.AlignVCenter)

        self._state_label = QLabel("Listening")
        self._state_label.setObjectName("voiceStateLabel")
        row.addWidget(self._state_label, 1, Qt.AlignVCenter)
        layout.addLayout(row)

        self._steps_row = QHBoxLayout()
        self._steps_row.setSpacing(2)
        self._step_labels: List[QLabel] = []
        for i, step in enumerate(self.STEPS):
            if i > 0:
                arrow = QLabel("›")
                arrow.setObjectName("stepArrow")
                arrow.setStyleSheet(
                    f"color: {TEXT_MUTED}; font-size: {FONT_MD}px; "
                    f"background: transparent;")
                self._steps_row.addWidget(arrow)
            lbl = QLabel(step)
            lbl.setObjectName("stepLabel")
            self._steps_row.addWidget(lbl)
            self._step_labels.append(lbl)
        self._steps_row.addStretch()
        layout.addLayout(self._steps_row)

        layout.addStretch()
        self.set_state("Idle")

    def set_state(self, state_name: str) -> None:
        self._state_label.setText(_friendly_state(state_name))
        name = state_name.lower()
        index = -1
        for key, idx in self._STATE_MAP.items():
            if key in name:
                index = idx
                break

        for i, lbl in enumerate(self._step_labels):
            if i == index:
                lbl.setStyleSheet(f"""
                    color: {PRIMARY_ACCENT}; font-size: {FONT_SM}px;
                    font-weight: 700; background: transparent;
                    padding: 2px 3px; border-radius: 4px;
                    background-color: rgba(34, 211, 238, 0.10);
                """)
            else:
                lbl.setStyleSheet(f"""
                    color: {TEXT_MUTED}; font-size: {FONT_SM}px;
                    background: transparent; padding: 2px 3px;
                """)

    def set_active(self, state_name: str) -> None:
        """Alias matching other panels."""
        self.set_state(state_name)


def _friendly_state(state: str) -> str:
    mapping = {
        "idle": "Standby", "listening": "Listening",
        "speech": "Speech detected", "thinking": "Thinking",
        "planning": "Planning", "replanning": "Replanning",
        "executing": "Executing", "observing": "Observing",
        "verifying": "Verifying", "speaking": "Speaking",
        "responding": "Responding", "error": "Error",
        "authenticating": "Authenticating",
    }
    return mapping.get(state.lower(), state)


# ═══════════════════════════════════════════════════════════════
# Activity Panel (right column, panel 2)
# ═══════════════════════════════════════════════════════════════

class ActivityPanel(QFrame):
    """
    Right-side compact timeline with right-aligned timestamps.

    Shows only human-readable activity items:
        Voice detected / Processing speech / Thinking /
        Executing / Responding
    """

    ACTIVITIES = [
        "Voice detected",
        "Processing speech",
        "Thinking",
        "Executing",
        "Responding",
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rightPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MEDIUM, SPACING_SMALL + 2,
                                  SPACING_MEDIUM, SPACING_SMALL + 2)
        layout.setSpacing(2)

        self._title = QLabel("ACTIVITY")
        self._title.setObjectName("panelTitle")
        layout.addWidget(self._title)
        layout.addSpacing(6)

        self._items: List[QLabel] = []
        self._times: List[QLabel] = []
        self._active_index = -1

        from PySide6.QtCore import QTime
        for i, activity in enumerate(self.ACTIVITIES):
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel("○")
            dot.setObjectName("activityDot")
            dot.setFixedWidth(10)
            row.addWidget(dot)
            item = QLabel(activity)
            item.setObjectName("activityItem")
            row.addWidget(item, 1)
            stamp = QLabel("—")
            stamp.setObjectName("activityStamp")
            row.addWidget(stamp)
            layout.addLayout(row)
            self._items.append(item)
            self._times.append(stamp)
            item._dot = dot  # type: ignore[attr-defined]

        layout.addStretch()

    def set_active(self, activity: str) -> None:
        activity_lower = activity.lower()

        from PySide6.QtCore import QTime
        if "listen" in activity_lower or "voice" in activity_lower:
            self._set_active_index(0)
        elif "transcrib" in activity_lower or "partial" in activity_lower \
                or "speech" in activity_lower:
            self._set_active_index(1)
        elif "think" in activity_lower:
            self._set_active_index(2)
        elif "speak" in activity_lower or "respond" in activity_lower:
            self._set_active_index(4)
        elif "execut" in activity_lower or "observ" in activity_lower \
                or "verif" in activity_lower:
            self._set_active_index(3)
        elif "plan" in activity_lower:
            self._set_active_index(2)
        else:
            self._set_active_index(-1)

    def _set_active_index(self, index: int) -> None:
        from PySide6.QtCore import QTime
        if index >= 0 and index != self._active_index:
            self._times[index].setText(QTime.currentTime().toString("hh:mm:ss"))

        self._active_index = index
        for i, item in enumerate(self._items):
            dot = getattr(item, "_dot", None)
            if i == index:
                item.setStyleSheet(f"""
                    color: {PRIMARY_ACCENT}; font-size: {FONT_MD}px;
                    font-weight: 600; padding: 3px 0px; background: transparent;
                """)
                if dot:
                    dot.setText("●")
                    dot.setStyleSheet(f"color: {PRIMARY_ACCENT}; "
                                      f"font-size: 7px; background: transparent;")
            elif 0 <= index > i:
                item.setStyleSheet(f"""
                    color: {TEXT_SECONDARY}; font-size: {FONT_MD}px;
                    padding: 3px 0px; background: transparent;
                """)
                if dot:
                    dot.setText("●")
                    dot.setStyleSheet(f"color: {TEXT_MUTED}; "
                                      f"font-size: 7px; background: transparent;")
            else:
                item.setStyleSheet(f"""
                    color: {TEXT_MUTED}; font-size: {FONT_MD}px;
                    padding: 3px 0px; background: transparent;
                """)
                if dot:
                    dot.setText("○")
                    dot.setStyleSheet(f"color: {TEXT_MUTED}; "
                                      f"font-size: 7px; background: transparent;")

    def reset(self) -> None:
        self._set_active_index(-1)

    def active_activity(self) -> str:
        if 0 <= self._active_index < len(self.ACTIVITIES):
            return self.ACTIVITIES[self._active_index]
        return ""


# ═══════════════════════════════════════════════════════════════
# Metrics Cards (right column, panel 3)
# ═══════════════════════════════════════════════════════════════

class MetricsCards(QFrame):
    """
    LATENCY (LIVE): four compact metric cards in a 2×2 grid:
        STT / Agent / TTS / Total
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rightPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MEDIUM, SPACING_SMALL + 2,
                                  SPACING_MEDIUM, SPACING_SMALL + 2)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        self._title = QLabel("LATENCY")
        self._title.setObjectName("panelTitle")
        title_row.addWidget(self._title)
        title_row.addStretch()
        live = QLabel("● LIVE")
        live.setObjectName("liveBadge")
        live.setStyleSheet(f"color: {SUCCESS}; font-size: {FONT_XS}px; "
                           f"font-weight: 700; letter-spacing: 1px; "
                           f"background: transparent;")
        title_row.addWidget(live)
        layout.addLayout(title_row)

        grid = QGridLayout()
        grid.setSpacing(6)
        self._cards = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for pos, name in zip(positions, ("STT", "Agent", "TTS", "Total")):
            card = self._create_card(name)
            grid.addWidget(card, pos[0], pos[1])
        layout.addLayout(grid)

    def _create_card(self, name: str) -> QFrame:
        card = QFrame()
        card.setObjectName("metricCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 5, 8, 5)
        card_layout.setSpacing(0)

        label = QLabel(name)
        label.setObjectName("metricLabel")
        label.setStyleSheet(f"""
            color: {TEXT_MUTED}; font-size: {FONT_XS}px;
            letter-spacing: 0.8px; background: transparent;
        """)
        card_layout.addWidget(label)

        value = QLabel("--")
        value.setObjectName("metricValue")
        value.setStyleSheet(f"""
            color: {TEXT_PRIMARY}; font-size: {FONT_MD}px;
            font-weight: 600; background: transparent;
        """)
        card_layout.addWidget(value)

        self._cards[name] = value
        return card

    def set_stt_latency(self, ms: float) -> None:
        self._cards["STT"].setText(f"{ms:.0f} ms")

    def set_agent_latency(self, ms: float) -> None:
        self._cards["Agent"].setText(f"{ms:.0f} ms")

    def set_tts_latency(self, ms: float) -> None:
        self._cards["TTS"].setText(f"{ms:.0f} ms")

    def set_total_latency(self, ms: float) -> None:
        self._cards["Total"].setText(f"{ms:.0f} ms")

    def reset(self) -> None:
        for value in self._cards.values():
            value.setText("--")


# ═══════════════════════════════════════════════════════════════
# System Status (right column, panel 4)
# ═══════════════════════════════════════════════════════════════

class SystemStatus(QFrame):
    """SYSTEM STATUS: compact 2×2 grid of STT ✓ / Agent ✓ / TTS ✓ / Tools ✓."""

    COMPONENTS = ("STT", "Agent", "TTS", "Tools")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rightPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MEDIUM, SPACING_SMALL + 2,
                                  SPACING_MEDIUM, SPACING_SMALL + 2)
        layout.setSpacing(6)

        self._title = QLabel("SYSTEM STATUS")
        self._title.setObjectName("panelTitle")
        layout.addWidget(self._title)

        grid = QGridLayout()
        grid.setSpacing(4)
        self._items = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for pos, name in zip(positions, self.COMPONENTS):
            item = QLabel(f"{name} —")
            item.setObjectName("statusItem")
            item.setStyleSheet(f"""
                color: {TEXT_MUTED}; font-size: {FONT_MD}px;
                background: transparent;
            """)
            grid.addWidget(item, pos[0], pos[1])
            self._items[name] = item
        layout.addLayout(grid)
        layout.addStretch()

    def set_status(self, component: str, ok: bool) -> None:
        if component not in self._items:
            return
        item = self._items[component]
        if ok:
            item.setText(f"{component} ✓")
            item.setStyleSheet(f"""
                color: {SUCCESS}; font-size: {FONT_MD}px; background: transparent;
            """)
        else:
            item.setText(f"{component} ✗")
            item.setStyleSheet(f"""
                color: {ERROR}; font-size: {FONT_MD}px; background: transparent;
            """)

    def set_all_ok(self) -> None:
        for name in self.COMPONENTS:
            self.set_status(name, True)

    def status(self, component: str) -> bool:
        if component in self._items:
            return "✓" in self._items[component].text()
        return False


# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════

class FooterBar(QFrame):
    """
    Footer:  DIEGO v2.1.0  ● ONLINE | All systems operational | [waveform]
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("footerBar")
        self.setFixedHeight(34)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING_LARGE, 4, SPACING_LARGE, 4)
        layout.setSpacing(SPACING_MEDIUM)

        version = QLabel("DIEGO v2.1.0")
        version.setObjectName("footerVersion")
        version.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FONT_XS}px; "
                              f"letter-spacing: 1px; background: transparent;")
        layout.addWidget(version)

        online = QLabel("● ONLINE")
        online.setObjectName("footerOnline")
        online.setStyleSheet(f"color: {SUCCESS}; font-size: {FONT_XS}px; "
                             f"font-weight: 700; letter-spacing: 1px; "
                             f"background: transparent;")
        layout.addWidget(online)

        layout.addStretch()

        center = QLabel("All systems operational")
        center.setObjectName("footerCenter")
        center.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FONT_XS}px; "
                             f"background: transparent;")
        layout.addWidget(center)

        layout.addStretch()

        self._wave = ThinWaveform(PRIMARY_ACCENT, bar_height=12)
        self._wave.setMinimumWidth(110)
        self._wave.setMaximumWidth(180)
        self._wave.start()
        layout.addWidget(self._wave)

    def set_level(self, level: float) -> None:
        self._wave.set_level(level)


# ═══════════════════════════════════════════════════════════════
# History Panel (collapsed/minimal — kept for compatibility)
# ═══════════════════════════════════════════════════════════════

class HistoryPanel(QFrame):
    """Minimal collapsed history of previous interactions."""

    MAX_ITEMS = 3

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("historyPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MEDIUM, 8, SPACING_MEDIUM, 8)
        layout.setSpacing(2)

        self._items: List[QLabel] = []
        self._history: List[str] = []

    def add_turn(self, user_text: str, response_text: str) -> None:
        summary = f"You: {user_text[:40]}{'...' if len(user_text) > 40 else ''}"
        self._history.insert(0, summary)
        self._history = self._history[:self.MAX_ITEMS]
        self._refresh()

    def _refresh(self) -> None:
        for item in self._items:
            item.deleteLater()
        self._items.clear()

        layout = self.layout()
        for text in self._history:
            item = QLabel(text)
            item.setObjectName("historyItem")
            item.setStyleSheet(f"""
                color: {TEXT_MUTED}; font-size: {FONT_SM}px;
                padding: 2px 0px; background: transparent;
            """)
            layout.addWidget(item)
            self._items.append(item)

    def clear(self) -> None:
        self._history.clear()
        self._refresh()