"""
Diego UI Styles — Dark modern assistant theme.

Provides QSS styles for a polished, accessible dark UI.
Colors are chosen for good contrast and low eye strain.
"""

# Color palette
COLORS = {
    # Backgrounds
    "bg_primary": "#1a1b26",      # Main window background
    "bg_secondary": "#24283b",    # Cards, panels
    "bg_tertiary": "#2f3349",     # Hover states
    "bg_input": "#1f2335",        # Input fields

    # Text
    "text_primary": "#c0caf5",    # Main text
    "text_secondary": "#a9b1d6",  # Secondary text
    "text_muted": "#565f89",      # Muted/disabled text
    "text_inverse": "#1a1b26",    # Text on accent backgrounds

    # Accents
    "accent_primary": "#7aa2f7",   # Primary accent (blue)
    "accent_secondary": "#bb9af7", # Secondary accent (purple)
    "accent_success": "#9ece6a",   # Success (green)
    "accent_warning": "#e0af68",   # Warning (yellow)
    "accent_error": "#f7768e",     # Error (red)

    # User/Diego message colors
    "user_bubble": "#3d59a1",      # User message background
    "diego_bubble": "#2f3349",     # Diego message background

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
    font-size: 22px;
    font-weight: 700;
    color: {COLORS['text_primary']};
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

/* Transcript area */
#transcriptScroll {{
    background-color: {COLORS['bg_primary']};
    border: none;
}}

#transcriptContainer {{
    background-color: {COLORS['bg_primary']};
}}

/* Message bubbles */
.messageBubble {{
    border-radius: 12px;
    padding: 12px 16px;
    margin: 4px 0px;
}}

.userBubble {{
    background-color: {COLORS['user_bubble']};
    color: white;
}}

.diegoBubble {{
    background-color: {COLORS['diego_bubble']};
    color: {COLORS['text_primary']};
}}

.partialBubble {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_muted']};
    border: 1px dashed {COLORS['border']};
}}

.errorBubble {{
    background-color: rgba(247, 118, 142, 0.15);
    color: {COLORS['accent_error']};
    border: 1px solid {COLORS['accent_error']};
}}

.messageLabel {{
    font-size: 14px;
    line-height: 1.5;
}}

.messageSender {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}}

.userSender {{
    color: rgba(255, 255, 255, 0.7);
}}

.diegoSender {{
    color: {COLORS['accent_primary']};
}}

/* Input area */
#inputArea {{
    background-color: {COLORS['bg_secondary']};
    border-top: 1px solid {COLORS['border']};
    padding: 12px 20px;
}}

#textInput {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    color: {COLORS['text_primary']};
    selection-background-color: {COLORS['accent_primary']};
    selection-color: {COLORS['text_inverse']};
}}

#textInput:focus {{
    border: 1px solid {COLORS['border_focus']};
}}

#textInput::placeholder {{
    color: {COLORS['text_muted']};
}}

/* Buttons */
QPushButton {{
    background-color: {COLORS['accent_primary']};
    color: {COLORS['text_inverse']};
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #89b4fa;
}}

QPushButton:pressed {{
    background-color: #5d87e8;
}}

QPushButton:disabled {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_muted']};
}}

#clearButton {{
    background-color: transparent;
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    padding: 8px 16px;
    font-size: 13px;
}}

#clearButton:hover {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
}}

/* Window controls */
#minimizeButton, #closeButton {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 8px;
    min-width: 32px;
    min-height: 32px;
}}

#minimizeButton:hover {{
    background-color: {COLORS['bg_tertiary']};
}}

#closeButton:hover {{
    background-color: {COLORS['accent_error']};
    color: white;
}}

/* Waveform / audio indicator */
#waveformWidget {{
    background-color: {COLORS['bg_secondary']};
    border-radius: 8px;
}}

/* Mic indicator */
#micIndicator {{
    border-radius: 8px;
    padding: 8px;
}}

#micIndicator[active="true"] {{
    background-color: rgba(158, 206, 106, 0.2);
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

/* Status bar */
#statusBar {{
    background-color: {COLORS['bg_secondary']};
    border-top: 1px solid {COLORS['border']};
    padding: 4px 12px;
    font-size: 12px;
    color: {COLORS['text_muted']};
}}
"""

# State-specific colors for the indicator
STATE_COLORS = {
    "Listening": COLORS["state_listening"],
    "Thinking": COLORS["state_thinking"],
    "Planning": COLORS["state_thinking"],
    "Executing": COLORS["state_executing"],
    "Observing": COLORS["state_executing"],
    "Verifying": COLORS["state_executing"],
    "Replanning": COLORS["state_thinking"],
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