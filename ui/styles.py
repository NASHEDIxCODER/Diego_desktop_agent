"""
Diego UI Styles — Premium dark glass/HUD assistant theme.

Design language:
    - Deep dark background with subtle glass panels
    - Neon cyan/blue primary accent, restrained purple secondary
    - Thin borders, soft shadows, rounded panels
    - Strong typographic hierarchy
    - Futuristic but professional
"""

# ── Color palette ──────────────────────────────────────────────
COLORS = {
    # Backgrounds (deep dark)
    "bg_primary": "#08090d",       # Main window background
    "bg_secondary": "#0d0f14",     # Panel backgrounds
    "bg_tertiary": "#12151c",      # Card backgrounds
    "bg_glass": "#0f1219",         # Glass panel base
    "bg_elevated": "#161a23",      # Elevated surfaces

    # Text
    "text_primary": "#e2e8f0",     # Main text (high contrast)
    "text_secondary": "#94a3b8",   # Secondary text
    "text_muted": "#475569",       # Muted/disabled text
    "text_inverse": "#08090d",     # Text on accent backgrounds

    # Accents
    "accent_primary": "#22d3ee",   # Neon cyan (primary)
    "accent_blue": "#3b82f6",      # Blue
    "accent_secondary": "#a78bfa", # Soft purple (secondary)
    "accent_success": "#34d399",   # Success (emerald)
    "accent_warning": "#fbbf24",   # Warning (amber)
    "accent_error": "#f87171",     # Error (red)

    # Borders
    "border": "#1e293b",           # Default border
    "border_subtle": "#162032",    # Subtle border
    "border_accent": "#22d3ee",    # Accent border

    # State colors
    "state_idle": "#475569",
    "state_listening": "#22d3ee",
    "state_speech": "#34d399",
    "state_thinking": "#a78bfa",
    "state_executing": "#3b82f6",
    "state_speaking": "#22d3ee",
    "state_error": "#f87171",

    # Glow effects
    "glow_cyan": "#22d3ee",
    "glow_purple": "#a78bfa",
}

# ── Typography ─────────────────────────────────────────────────
FONTS = {
    "family": "'Inter', 'Segoe UI', 'Ubuntu', 'Helvetica Neue', sans-serif",
    "mono": "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
    "size_xs": "10px",
    "size_sm": "11px",
    "size_md": "13px",
    "size_lg": "15px",
    "size_xl": "18px",
    "size_2xl": "22px",
    "size_3xl": "28px",
}

# ── Main window stylesheet ─────────────────────────────────────
MAIN_WINDOW_QSS = f"""
QMainWindow {{
    background-color: {COLORS['bg_primary']};
}}

QWidget {{
    background-color: transparent;
    color: {COLORS['text_primary']};
    font-family: {FONTS['family']};
    font-size: {FONTS['size_md']};
}}

/* ── Header ─────────────────────────────────────────────── */
#header {{
    background-color: {COLORS['bg_secondary']};
    border-bottom: 1px solid {COLORS['border']};
}}

#titleLabel {{
    font-size: {FONTS['size_2xl']};
    font-weight: 700;
    color: {COLORS['text_primary']};
    letter-spacing: 3px;
}}

#subtitleLabel {{
    font-size: {FONTS['size_sm']};
    color: {COLORS['text_muted']};
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* ── Window controls ────────────────────────────────────── */
#minimizeButton, #closeButton {{
    background-color: transparent;
    border: none;
    border-radius: 8px;
    color: {COLORS['text_muted']};
    font-size: 14px;
}}

#minimizeButton:hover {{
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
}}

#closeButton:hover {{
    background-color: {COLORS['accent_error']};
    color: white;
}}

/* ── Central voice core area ────────────────────────────── */
#voiceCoreArea {{
    background-color: {COLORS['bg_primary']};
}}

#stateLabel {{
    font-size: {FONTS['size_md']};
    font-weight: 600;
    color: {COLORS['text_secondary']};
    letter-spacing: 2px;
    text-transform: uppercase;
}}

/* ── Transcript panel ───────────────────────────────────── */
#transcriptPanel {{
    background-color: {COLORS['bg_glass']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 16px;
}}

#transcriptHeader {{
    font-size: {FONTS['size_xs']};
    font-weight: 600;
    color: {COLORS['text_muted']};
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

#transcriptText {{
    font-size: {FONTS['size_xl']};
    color: {COLORS['text_primary']};
    line-height: 1.4;
}}

#transcriptTextPartial {{
    font-size: {FONTS['size_xl']};
    color: {COLORS['text_secondary']};
    font-style: italic;
}}

/* ── Response panel ─────────────────────────────────────── */
#responsePanel {{
    background-color: {COLORS['bg_glass']};
    border: 1px solid {COLORS['border_accent']}33;
    border-radius: 16px;
}}

#responseHeader {{
    font-size: {FONTS['size_xs']};
    font-weight: 700;
    color: {COLORS['accent_primary']};
    letter-spacing: 2px;
    text-transform: uppercase;
}}

#responseText {{
    font-size: {FONTS['size_xl']};
    font-weight: 500;
    color: {COLORS['text_primary']};
    line-height: 1.4;
}}

#speakingIndicator {{
    font-size: {FONTS['size_sm']};
    color: {COLORS['accent_primary']};
    font-weight: 600;
}}

/* ── Activity panel ─────────────────────────────────────── */
#activityPanel {{
    background-color: {COLORS['bg_secondary']};
    border-left: 1px solid {COLORS['border']};
}}

#activityTitle {{
    font-size: {FONTS['size_xs']};
    font-weight: 600;
    color: {COLORS['text_muted']};
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

#activityItem {{
    font-size: {FONTS['size_md']};
    color: {COLORS['text_muted']};
    padding: 6px 0px;
}}

#activityItemActive {{
    font-size: {FONTS['size_md']};
    color: {COLORS['accent_primary']};
    font-weight: 600;
    padding: 6px 0px;
}}

#activityItemDone {{
    font-size: {FONTS['size_md']};
    color: {COLORS['text_secondary']};
    padding: 6px 0px;
}}

/* ── Metrics cards ──────────────────────────────────────── */
#metricsPanel {{
    background-color: {COLORS['bg_secondary']};
    border-top: 1px solid {COLORS['border']};
}}

#metricCard {{
    background-color: {COLORS['bg_tertiary']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 10px;
    padding: 8px 12px;
}}

#metricLabel {{
    font-size: {FONTS['size_xs']};
    color: {COLORS['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

#metricValue {{
    font-size: {FONTS['size_lg']};
    font-weight: 600;
    color: {COLORS['text_primary']};
    font-family: {FONTS['mono']};
}}

/* ── System status ──────────────────────────────────────── */
#systemStatus {{
    background-color: {COLORS['bg_secondary']};
    border-top: 1px solid {COLORS['border']};
}}

#statusItem {{
    font-size: {FONTS['size_sm']};
    color: {COLORS['text_muted']};
}}

#statusItemOk {{
    font-size: {FONTS['size_sm']};
    color: {COLORS['accent_success']};
}}

#statusItemError {{
    font-size: {FONTS['size_sm']};
    color: {COLORS['accent_error']};
}}

/* ── Connection indicator ───────────────────────────────── */
#connectionIndicator {{
    background-color: {COLORS['bg_tertiary']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 4px 10px;
}}

#connectionDot {{
    font-size: 8px;
}}

#connectionText {{
    font-size: {FONTS['size_xs']};
    color: {COLORS['text_secondary']};
    font-weight: 500;
}}

/* ── History (collapsed) ────────────────────────────────── */
#historyPanel {{
    background-color: {COLORS['bg_secondary']};
    border-top: 1px solid {COLORS['border_subtle']};
}}

#historyItem {{
    font-size: {FONTS['size_sm']};
    color: {COLORS['text_muted']};
    padding: 4px 0px;
}}

/* ── Scrollbars ─────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {COLORS['bg_primary']};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['bg_elevated']};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['border']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['bg_primary']};
    height: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['bg_elevated']};
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
    "Authenticating": COLORS["accent_warning"],
    "Error": COLORS["state_error"],
}


def state_color(state: str) -> str:
    """Get the color for a given state."""
    return STATE_COLORS.get(state, COLORS["state_idle"])