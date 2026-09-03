"""
Diego Desktop UI — PySide6 conversational interface.

This package provides a native desktop UI that sits on top of the existing
production assistant pipeline (ConversationEngine / voice / Brain /
DecisionEngine / TaskController).

Architecture:
    ui/
    ├── __init__.py       # Package exports
    ├── __main__.py       # Entry point: python -m ui
    ├── event_bridge.py   # Thread-safe bridge from pipeline events to Qt signals
    ├── main_window.py    # Main Diego window (transcript, state, input)
    ├── widgets.py        # Custom widgets (message bubbles, waveform, indicators)
    └── styles.py         # Dark theme QSS styles

The UI subscribes to the existing EventBus and ConversationEngine state
transitions via the EventBridge. It does NOT duplicate any assistant logic.
"""

from ui.event_bridge import EventBridge, UIEvent, UIEventType
from ui.main_window import DiegoMainWindow

__all__ = [
    "EventBridge",
    "UIEvent",
    "UIEventType",
    "DiegoMainWindow",
]
