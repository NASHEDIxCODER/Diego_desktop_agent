"""
Diego UI Styles — Dark modern futuristic assistant theme.

Provides QSS styles for a polished, accessible dark UI.
Colors are chosen for good contrast and low eye strain.
"""

# Color palette
COLORS = {
    # Backgrounds
    "bg_primary": "#0f1117",      # Main window background (darker)
    "bg_secondary": "#1a1d27",    # Cards, panels
    "bg_tertiary": "#242836",     # Hover states
    "bg_input": "#1a1d27",        # Input fields

    # Text
    "text_primary": "#c0caf5",    # Main text
    "text_secondary": "#a9b1d6",  # Secondary text
    "text_muted": "#565f89",      # Muted/disabled text
    "text_inverse": "#0f1117",    # Text on accent backgrounds

    # Accents
    "accent_primary": "#7aa2f7",   # Primary accent (blue)
    "accent_secondary": "#bb9af7", # Secondary accent (purple)
    "accent_success": "#9ece6a",   # Success (green)
    "accent_warning": "#e0af68",   # Warning (yellow)
    "accent_error": "#f7768e",     # Error (red)

    # User/Diego message colors
    "user_bubble": "#3d59a1",      # User message background
    "diego_bubble": "#242836",     # Diego message background

    # Borders and dividers
    "border": "#3b4261",
    "border_focus": "#7aa2f7",

    # State indicator colors
    "state_listening": "#9ece6a",
    "state_thinking": "#e0af68",
    "state_executing": "#7aa2f7",
    "state_error": "#f7768e",
    "state_idle": "#565f89",
}

# Main window stylesheet
MAIN_WINDOW_QSS = f"""
QMainWindow {{
    background-color: {COLORS['bg_primary']};
}}

QWidget {{
    background-color: transparent;
    color: {COLORS['text_primary']};
    font-family: 'Inter', 'Segoe UI', 'Ubuntu', sans-serif;
    font-size: 14px;
}}

/* Header */
#header {{
    background-color: {COLORS['bg_secondary']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 12px 20px;
}}

#titleLabel {{
    font-size: 24px;
    font-weight: 700;
    color: {COLORS['text_primary']};
    letter-spacing: 2px;
}}

#statusLabel {{
    font-size: 13px;
    color: {COLORS['text_secondary']};
}}

/* State indicator */
#stateIndicator {{
    background-color: {COLORS['bg_tertiary']};
    border-radius: 16px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
}}

/* Waveform / audio area */
#waveformArea {{
    background-color: {COLORS['bg_primary']};
}}

#waveformWidget {{
    background-color: {COLORS['bg_secondary']};
    border-radius: 12px;
}}

#speakingLabel {{
    color: {COLORS['accent_secondary']};
    font-size: 13px;
    font-weight: 600;
}}

/* Transcript area */
#transcriptArea {{
    background-color: {COLORS['bg_primary']};
    border-top: 1px solid {COLORS['border']};
}}

#transcriptLabel {{
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

/* Response area */
#responseArea {{
    background-color: {COLORS['bg_primary']};
    border-top: 1px solid {COLORS['border']};
}}

#responseLabel {{
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

/* Footer */
#footer {{
    background-color: {COLORS['bg_secondary']};
    border-top: 1px solid {COLORS['border']};
    padding: 8px 20px;
}}

#statusLabel {{
    font-size: 12px;
    color: {COLORS['text_muted']};
}}

/* Window controls */
#minimizeButton, #closeButton {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 8px;
    min-width: 32px;
    min-height: 32px;
    color: {COLORS['text_muted']};
    font-size: 14px;
}}

#minimizeButton:hover {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
}}

#closeButton:hover {{
    background-color: {COLORS['accent_error']};
    color: white;
}}

/* Mic indicator */
#micIndicator {{
    border-radius: 8px;
    padding: 8px;
}}

#micIndicator[active="true"] {{
    background-color: rgba(158, 206, 106, 0.2);
}}

/* Latency metrics */
#latencyMetrics {{
    font-size: 11px;
    color: {COLORS['text_muted']};
}}

/* Scrollbars */
QScrollBar:vertical {{
    background-color: {COLORS['bg_primary']};
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['bg_tertiary']};
    border-radius: 5px;
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
"""

# State-specific colors for the indicator
STATE_COLORS = {
    "Listening": COLORS["state_listening"],
    "Speech Detected": COLORS["state_listening"],
    "Thinking": COLORS["state_thinking"],
    "Planning": COLORS["state_thinking"],
    "Executing": COLORS["state_executing"],
    "Observing": COLORS["state_executing"],
    "Verifying": COLORS["state_executing"],
    "Replanning": COLORS["state_thinking"],
    "Speaking": COLORS["accent_secondary"],
    "Responding": COLORS["accent_secondary"],
    "Idle": COLORS["state_idle"],
    "Error": COLORS["state_error"],
    "Waiting for wake word": COLORS["state_idle"],
    "Authenticating": COLORS["accent_warning"],
}


def state_indicator_qss(state: str) -> str:
    """Generate QSS for the state indicator based on current state."""
    color = STATE_COLORS.get(state, COLORS["state_idle"])
    return f"""
        #stateIndicator {{
            background-color: {color}22;
            color: {color};
            border: 1px solid {color}44;
        }}
    """