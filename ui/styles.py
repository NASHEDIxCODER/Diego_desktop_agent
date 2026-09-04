"""
Diego UI Styles — dark glass/HUD theme built from ui.tokens.

Design language:
    - Near-black navy background, glass-like translucent panels
    - Thin 1px borders, soft glow, consistent corner radius
    - Neon cyan primary accent, restrained violet secondary
"""

from __future__ import annotations

from ui.tokens import (
    WINDOW_BACKGROUND, PANEL_BACKGROUND, PANEL_BACKGROUND_HI, PANEL_BORDER,
    PANEL_BORDER_SOFT, PRIMARY_ACCENT, PRIMARY_ACCENT_DIM, SECONDARY_ACCENT,
    SUCCESS, WARNING, ERROR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_FAMILY, FONT_MONO,
    FONT_XS, FONT_SM, FONT_MD, FONT_LG, FONT_XL, FONT_2XL,
    PANEL_RADIUS, CARD_RADIUS,
    GLOW_INTENSITY,
    HEADER_HEIGHT, FOOTER_HEIGHT, SPACING_SMALL, SPACING_MEDIUM,
    SPACING_LARGE,
)

# ── Backward-compatible dicts (existing imports) ───────────────
COLORS = {
    "bg_primary": WINDOW_BACKGROUND,
    "bg_secondary": PANEL_BACKGROUND,
    "bg_tertiary": PANEL_BACKGROUND_HI,
    "bg_glass": PANEL_BACKGROUND,
    "bg_elevated": PANEL_BACKGROUND_HI,
    "text_primary": TEXT_PRIMARY,
    "text_secondary": TEXT_SECONDARY,
    "text_muted": TEXT_MUTED,
    "accent_primary": PRIMARY_ACCENT,
    "accent_blue": PRIMARY_ACCENT_DIM,
    "accent_secondary": SECONDARY_ACCENT,
    "accent_success": SUCCESS,
    "accent_warning": WARNING,
    "accent_error": ERROR,
    "border": PANEL_BORDER,
    "border_subtle": PANEL_BORDER_SOFT,
    "border_accent": PRIMARY_ACCENT,
    "state_idle": TEXT_MUTED,
    "state_listening": PRIMARY_ACCENT,
    "state_speech": SUCCESS,
    "state_thinking": SECONDARY_ACCENT,
    "state_executing": PRIMARY_ACCENT_DIM,
    "state_speaking": PRIMARY_ACCENT,
    "state_error": ERROR,
    "glow_cyan": PRIMARY_ACCENT,
    "glow_purple": SECONDARY_ACCENT,
}

FONTS = {
    "family": FONT_FAMILY,
    "mono": FONT_MONO,
    "size_xs": f"{FONT_XS}px",
    "size_sm": f"{FONT_SM}px",
    "size_md": f"{FONT_MD}px",
    "size_lg": f"{FONT_LG}px",
    "size_xl": f"{FONT_XL}px",
    "size_2xl": f"{FONT_2XL}px",
    "size_3xl": f"{FONT_2XL + 6}px",
}

# ── Main window stylesheet ─────────────────────────────────────
MAIN_WINDOW_QSS = f"""
QMainWindow {{
    background-color: {WINDOW_BACKGROUND};
}}

#centralWidget {{
    background-color: {WINDOW_BACKGROUND};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
}}

QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: {FONT_MD}px;
}}

/* ── Header ─────────────────────────────────────────────── */
#header {{
    background-color: {PANEL_BACKGROUND};
    border-bottom: 1px solid {PANEL_BORDER};
}}

#titleLabel {{
    font-size: {FONT_2XL}px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: 4px;
}}

#subtitleLabel {{
    font-size: {FONT_XS}px;
    color: {TEXT_MUTED};
    letter-spacing: 2px;
}}

#headerDivider {{
    background-color: {PANEL_BORDER};
    max-width: 1px;
}}

#minimizeButton, #maximizeButton, #settingsButton, #closeButton {{
    background-color: transparent;
    border: none;
    border-radius: 7px;
    color: {TEXT_MUTED};
    font-size: 13px;
}}

#minimizeButton:hover, #maximizeButton:hover, #settingsButton:hover {{
    background-color: {PANEL_BACKGROUND_HI};
    color: {TEXT_PRIMARY};
}}

#closeButton:hover {{
    background-color: {ERROR};
    color: white;
}}

#connectionIndicator {{
    background-color: {PANEL_BACKGROUND_HI};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
}}

/* ── Hero / voice core area ─────────────────────────────── */
#heroPanel {{
    background-color: rgba(11, 17, 28, 0.55);
    border: 1px solid {PANEL_BORDER};
    border-radius: {PANEL_RADIUS}px;
}}

#stateLabel {{
    font-size: {FONT_MD}px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    letter-spacing: 2px;
}}

/* ── Transcript panel ───────────────────────────────────── */
#transcriptPanel {{
    background-color: rgba(11, 17, 28, 0.55);
    border: 1px solid {PANEL_BORDER};
    border-radius: {PANEL_RADIUS}px;
}}

#transcriptHeader {{
    font-size: {FONT_XS}px;
    font-weight: 700;
    color: {TEXT_MUTED};
    letter-spacing: 2px;
}}

#finalBadge {{
    font-size: {FONT_XS}px;
    font-weight: 700;
    color: {SUCCESS};
    letter-spacing: 1px;
}}

/* ── Response panel (dominant) ──────────────────────────── */
#responsePanel {{
    background-color: rgba(14, 21, 34, 0.70);
    border: 1px solid rgba(34, 211, 238, 0.35);
    border-radius: {PANEL_RADIUS}px;
}}

#responseHeader {{
    font-size: {FONT_XS}px;
    font-weight: 800;
    color: {PRIMARY_ACCENT};
    letter-spacing: 2.5px;
}}

#projectCard {{
    background-color: rgba(34, 211, 238, 0.07);
    border: 1px solid rgba(34, 211, 238, 0.30);
    border-radius: {CARD_RADIUS}px;
}}

#projectName {{
    font-size: {FONT_LG}px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}

#projectMeta {{
    font-size: {FONT_XS}px;
    color: {TEXT_SECONDARY};
}}

#responsePath {{
    font-size: {FONT_MD}px;
    color: {TEXT_SECONDARY};
}}

/* ── Right column panels ────────────────────────────────── */
#rightPanel {{
    background-color: rgba(11, 17, 28, 0.55);
    border: 1px solid {PANEL_BORDER};
    border-radius: {PANEL_RADIUS}px;
}}

#panelTitle {{
    font-size: {FONT_XS}px;
    font-weight: 700;
    color: {TEXT_MUTED};
    letter-spacing: 1.8px;
}}

#voiceStateLabel {{
    font-size: {FONT_LG}px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

/* ── Metric cards ───────────────────────────────────────── */
#metricCard {{
    background-color: rgba(14, 21, 34, 0.85);
    border: 1px solid {PANEL_BORDER};
    border-radius: {CARD_RADIUS}px;
}}

/* ── Footer ─────────────────────────────────────────────── */
#footerBar {{
    background-color: {PANEL_BACKGROUND};
    border-top: 1px solid {PANEL_BORDER};
}}

/* ── Scrollbars ─────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {WINDOW_BACKGROUND};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {PANEL_BACKGROUND_HI};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background-color: {WINDOW_BACKGROUND};
    height: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: {PANEL_BACKGROUND_HI};
    border-radius: 4px;
    min-width: 30px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}
"""

# ── State color mapping ────────────────────────────────────────
STATE_COLORS = {
    "Idle": COLORS["state_idle"],
    "Waiting for wake word": COLORS["state_idle"],
    "Listening": COLORS["state_listening"],
    "Speech Detected": COLORS["state_speech"],
    "Thinking": COLORS["state_thinking"],
    "Planning": COLORS["state_thinking"],
    "Replanning": COLORS["state_thinking"],
    "Executing": COLORS["state_executing"],
    "Observing": COLORS["state_executing"],
    "Verifying": COLORS["state_executing"],
    "Speaking": COLORS["state_speaking"],
    "Responding": COLORS["state_speaking"],
    "Authenticating": WARNING,
    "Error": COLORS["state_error"],
}


def state_color(state: str) -> str:
    """Get the color for a given state."""
    return STATE_COLORS.get(state, COLORS["state_idle"])