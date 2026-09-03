"""
Tests for the Diego Desktop Voice-First UI.

Covers:
    - UI startup
    - Event bridge thread safety
    - Partial transcript update (no duplication)
    - Final transcript replacement
    - User transcript rendering
    - Diego response rendering
    - State changes
    - Error rendering
    - Voice-first layout (no chat composer)
    - Speaking indicator
    - Long-running operations do not block UI
    - No duplicate transcript messages
"""

import asyncio
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Set offscreen platform for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Ensure a QApplication exists for tests
_app = None


def get_app() -> QApplication:
    """Get or create the QApplication instance."""
    global _app
    if _app is None:
        _app = QApplication.instance()
        if _app is None:
            _app = QApplication([])
    return _app


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Session-scoped QApplication fixture."""
    app = get_app()
    yield app


@pytest.fixture
def bridge(qapp):
    """Create a fresh EventBridge for each test."""
    from ui.event_bridge import EventBridge
    b = EventBridge()
    b.start()
    yield b
    b.stop()


@pytest.fixture
def window(qapp, bridge):
    """Create a DiegoMainWindow for testing."""
    from ui.main_window import DiegoMainWindow
    w = DiegoMainWindow(bridge=bridge, loop=None)
    w.show()
    yield w
    w.close()


def process_events(ms: int = 50) -> None:
    """Process Qt events for a short time."""
    app = get_app()
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


# ═══════════════════════════════════════════════════════════════
# UI Startup Tests
# ═══════════════════════════════════════════════════════════════

class TestUIStartup:
    """Tests for UI startup."""

    def test_event_bridge_creation(self, qapp):
        """EventBridge can be created and started."""
        from ui.event_bridge import EventBridge
        bridge = EventBridge()
        assert bridge is not None
        bridge.start()
        assert bridge._started
        bridge.stop()
        assert not bridge._started

    def test_main_window_creation(self, qapp, bridge):
        """DiegoMainWindow can be created."""
        from ui.main_window import DiegoMainWindow
        window = DiegoMainWindow(bridge=bridge)
        assert window is not None
        assert window.windowTitle() == "Diego"
        window.close()

    def test_main_window_initial_state(self, window):
        """Window starts in Idle state."""
        process_events()
        assert window._state_indicator.state() == "Idle"

    def test_voice_first_layout(self, window):
        """The UI is voice-first: no chat composer, no text input."""
        assert not hasattr(window, '_text_input')
        assert not hasattr(window, '_send_btn')
        # Voice-first widgets are present
        assert hasattr(window, '_state_indicator')
        assert hasattr(window, '_waveform')
        assert hasattr(window, '_transcript_label')
        assert hasattr(window, '_response_label')
        assert hasattr(window, '_speaking_label')


# ═══════════════════════════════════════════════════════════════
# Event Bridge Tests
# ═══════════════════════════════════════════════════════════════

class TestEventBridge:
    """Tests for the EventBridge."""

    def test_emit_from_same_thread(self, bridge, qapp):
        """Events emitted from the same thread are delivered."""
        received = []
        bridge.state_changed.connect(lambda s: received.append(s))

        bridge.emit_state("TestState")
        process_events(100)

        assert "TestState" in received

    def test_emit_from_different_thread(self, bridge, qapp):
        """Events emitted from a different thread are delivered safely."""
        received = []
        bridge.state_changed.connect(lambda s: received.append(s))

        def emit_from_thread():
            bridge.emit_state("ThreadState")

        t = threading.Thread(target=emit_from_thread)
        t.start()
        t.join()
        process_events(100)

        assert "ThreadState" in received

    def test_emit_partial_transcript(self, bridge, qapp):
        """Partial transcript events are delivered."""
        received = []
        bridge.partial_transcript.connect(lambda t: received.append(t))

        bridge.emit_partial("hello wor")
        process_events(100)

        assert received == ["hello wor"]

    def test_emit_final_transcript(self, bridge, qapp):
        """Final transcript events are delivered."""
        received = []
        bridge.final_transcript.connect(lambda t: received.append(t))

        bridge.emit_final("hello world")
        process_events(100)

        assert received == ["hello world"]

    def test_emit_response(self, bridge, qapp):
        """Response events are delivered."""
        received = []
        bridge.response.connect(lambda t: received.append(t))

        bridge.emit_response("Done!")
        process_events(100)

        assert received == ["Done!"]

    def test_emit_error(self, bridge, qapp):
        """Error events are delivered."""
        received = []
        bridge.error.connect(lambda m: received.append(m))

        bridge.emit_error("Something went wrong")
        process_events(100)

        assert received == ["Something went wrong"]

    def test_emit_speaking(self, bridge, qapp):
        """Speaking events are delivered."""
        received = []
        bridge.speaking.connect(lambda: received.append(True))

        bridge.emit_speaking()
        process_events(100)

        assert received == [True]

    def test_all_event_types(self, bridge, qapp):
        """All required event types can be emitted."""
        from ui.event_bridge import UIEvent, UIEventType

        events_received = []

        bridge.listening.connect(lambda: events_received.append("LISTENING"))
        bridge.thinking.connect(lambda: events_received.append("THINKING"))
        bridge.planning.connect(lambda: events_received.append("PLANNING"))
        bridge.executing.connect(lambda: events_received.append("EXECUTING"))
        bridge.observing.connect(lambda: events_received.append("OBSERVING"))
        bridge.verifying.connect(lambda: events_received.append("VERIFYING"))
        bridge.replanning.connect(lambda: events_received.append("REPLANNING"))
        bridge.speaking.connect(lambda: events_received.append("SPEAKING"))
        bridge.idle.connect(lambda: events_received.append("IDLE"))

        bridge.emit_listening()
        bridge.emit_thinking()
        bridge.emit_planning()
        bridge.emit_executing()
        bridge.emit_observing()
        bridge.emit_verifying()
        bridge.emit_replanning()
        bridge.emit_speaking()
        bridge.emit_idle()

        process_events(150)

        assert "LISTENING" in events_received
        assert "THINKING" in events_received
        assert "PLANNING" in events_received
        assert "EXECUTING" in events_received
        assert "OBSERVING" in events_received
        assert "VERIFYING" in events_received
        assert "REPLANNING" in events_received
        assert "SPEAKING" in events_received
        assert "IDLE" in events_received

    def test_rapid_events_no_crash(self, bridge, qapp):
        """Rapid event emission doesn't crash or block."""
        count = [0]
        bridge.state_changed.connect(lambda s: count.__setitem__(0, count[0] + 1))

        for i in range(100):
            bridge.emit_state(f"State{i}")

        process_events(200)
        assert count[0] >= 50  # Most events should be delivered


# ═══════════════════════════════════════════════════════════════
# Transcript Tests
# ═══════════════════════════════════════════════════════════════

class TestTranscript:
    """Tests for transcript handling."""

    def test_partial_transcript_updates_live_region(self, window, bridge):
        """Partial transcript updates the single live region."""
        bridge.emit_partial("hel")
        process_events(100)

        assert window.get_partial_text() == "hel"

    def test_partial_transcript_updates_same_region(self, window, bridge):
        """Multiple partials update the same region (no duplication)."""
        bridge.emit_partial("hel")
        process_events(50)
        bridge.emit_partial("hello")
        process_events(50)
        bridge.emit_partial("hello wor")
        process_events(50)

        # Single live region with latest text
        assert window.get_partial_text() == "hello wor"

    def test_final_transcript_replaces_partial(self, window, bridge):
        """Final transcript replaces the partial region."""
        bridge.emit_partial("hello")
        process_events(50)
        bridge.emit_final("hello world")
        process_events(100)

        # Partial is cleared
        assert window.get_partial_text() == ""

        # Final text is in the transcript
        transcript = window.get_transcript()
        user_msgs = [m for m in transcript if m["sender"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["text"] == "hello world"

    def test_final_without_partial_creates_user_message(self, window, bridge):
        """Final transcript without partial creates a user message."""
        bridge.emit_final("typed command")
        process_events(100)

        transcript = window.get_transcript()
        user_msgs = [m for m in transcript if m["sender"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["text"] == "typed command"

    def test_no_duplicate_transcript_messages(self, window, bridge):
        """Partial + final doesn't create duplicate messages."""
        # Simulate realistic STT flow
        bridge.emit_partial("open")
        process_events(30)
        bridge.emit_partial("open fire")
        process_events(30)
        bridge.emit_partial("open firefox")
        process_events(30)
        bridge.emit_final("open firefox")
        process_events(100)

        transcript = window.get_transcript()
        user_msgs = [m for m in transcript if m["sender"] == "user"]

        # Exactly one user message
        assert len(user_msgs) == 1
        assert user_msgs[0]["text"] == "open firefox"

    def test_diego_response_rendering(self, window, bridge):
        """Diego responses are rendered correctly."""
        bridge.emit_response("Opening Firefox for you.")
        process_events(100)

        transcript = window.get_transcript()
        diego_msgs = [m for m in transcript if m["sender"] == "diego"]

        assert len(diego_msgs) >= 1
        assert any("Firefox" in m["text"] for m in diego_msgs)

    def test_response_displayed_once(self, window, bridge):
        """Response is displayed once and remains visible."""
        bridge.emit_response("Opening Firefox for you.")
        process_events(100)

        transcript = window.get_transcript()
        diego_msgs = [m for m in transcript if m["sender"] == "diego"]
        assert len(diego_msgs) == 1
        assert diego_msgs[0]["text"] == "Opening Firefox for you."

    def test_error_rendering(self, window, bridge):
        """Errors are rendered with the error state."""
        bridge.emit_error("I couldn't find that application.")
        process_events(100)

        assert window._state_indicator.state() == "Error"
        assert "couldn't find" in window.get_response_text() or \
               "couldn't find" in window._response_label._text_label.text()


# ═══════════════════════════════════════════════════════════════
# State Tests
# ═══════════════════════════════════════════════════════════════

class TestStateChanges:
    """Tests for state display."""

    def test_listening_state(self, window, bridge):
        """Listening state is displayed."""
        bridge.emit_listening()
        process_events(100)

        assert window._state_indicator.state() == "Listening"
        assert window._mic_indicator._active

    def test_thinking_state(self, window, bridge):
        """Thinking state is displayed."""
        bridge.emit_thinking()
        process_events(100)

        assert window._state_indicator.state() == "Thinking"
        assert not window._mic_indicator._active

    def test_speaking_state(self, window, bridge):
        """Speaking state is displayed with speaking indicator."""
        bridge.emit_speaking()
        process_events(100)

        assert window._state_indicator.state() == "Speaking"
        assert window._speaking_label.isVisible()

    def test_executing_state(self, window, bridge):
        """Executing state is displayed."""
        bridge.emit_executing("desktop_open")
        process_events(100)

        state = window._state_indicator.state()
        assert "Executing" in state

    def test_idle_state(self, window, bridge):
        """Idle state is displayed."""
        bridge.emit_thinking()
        process_events(50)
        bridge.emit_idle()
        process_events(100)

        assert window._state_indicator.state() == "Idle"
        assert not window._mic_indicator._active
        assert not window._speaking_label.isVisible()

    def test_error_state(self, window, bridge):
        """Error state is displayed."""
        bridge.emit_error("Test error")
        process_events(100)

        assert window._state_indicator.state() == "Error"

    def test_all_high_level_states(self, window, bridge):
        """All required high-level states can be displayed."""
        states = [
            "Listening", "Thinking", "Planning", "Executing",
            "Observing", "Verifying", "Replanning", "Speaking",
            "Idle", "Error"
        ]

        for state in states:
            bridge.emit_state(state)
            process_events(50)
            assert window._state_indicator.state() == state


# ═══════════════════════════════════════════════════════════════
# Threading Tests
# ═══════════════════════════════════════════════════════════════

class TestThreading:
    """Tests for thread safety and non-blocking behavior."""

    def test_ui_responsive_during_events(self, window, bridge):
        """UI remains responsive during event processing."""
        # Emit many events
        for i in range(50):
            bridge.emit_state(f"State{i}")

        # UI should still process events
        start = time.time()
        process_events(100)
        elapsed = time.time() - start

        # Should not block for more than 1 second
        assert elapsed < 1.0

    def test_events_from_multiple_threads(self, bridge, qapp):
        """Events from multiple threads are all delivered safely."""
        received = []
        bridge.state_changed.connect(lambda s: received.append(s))

        def emit_worker(n):
            for i in range(10):
                bridge.emit_state(f"Thread{n}-{i}")

        threads = [threading.Thread(target=emit_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        process_events(200)

        # All 50 events should be delivered
        assert len(received) >= 45  # Allow small margin for timing

    def test_long_operation_does_not_block_ui(self, window, bridge):
        """Simulated long operation doesn't block the UI thread."""
        ui_responsive = [False]

        def check_responsive():
            # This runs on the Qt thread via timer
            ui_responsive[0] = True

        # Schedule a check
        QTimer.singleShot(50, check_responsive)

        # Simulate long operation in background
        def long_op():
            time.sleep(0.2)
            bridge.emit_response("Done with long task")

        t = threading.Thread(target=long_op)
        t.start()

        process_events(300)
        t.join()

        assert ui_responsive[0], "UI was blocked by long operation"


# ═══════════════════════════════════════════════════════════════
# Widget Tests
# ═══════════════════════════════════════════════════════════════

class TestWidgets:
    """Tests for custom widgets."""

    def test_voice_state_indicator(self, qapp):
        """Voice state indicator shows correct states."""
        from ui.widgets import VoiceStateIndicator
        indicator = VoiceStateIndicator()

        indicator.set_state("Listening")
        assert indicator.state() == "Listening"

        indicator.set_state("Thinking")
        assert indicator.state() == "Thinking"

        indicator.set_state("Speaking")
        assert indicator.state() == "Speaking"

    def test_waveform_level(self, qapp):
        """Waveform widget accepts audio levels."""
        from ui.widgets import WaveformWidget
        waveform = WaveformWidget()

        waveform.set_level(0.5)
        assert waveform._target_level == 0.5

        waveform.set_level(1.5)  # Should clamp
        assert waveform._target_level == 1.0

        waveform.set_level(-0.5)  # Should clamp
        assert waveform._target_level == 0.0

    def test_mic_indicator(self, qapp):
        """Mic indicator toggles active state."""
        from ui.widgets import MicIndicator
        mic = MicIndicator()

        assert not mic._active
        mic.set_active(True)
        assert mic._active
        mic.set_active(False)
        assert not mic._active

    def test_transcript_label(self, qapp):
        """Transcript label supports partial and final."""
        from ui.widgets import TranscriptLabel
        label = TranscriptLabel()

        label.set_partial("hel")
        assert label._text_label.text() == "hel"

        label.set_final("hello world")
        assert label._text_label.text() == "hello world"

    def test_response_label(self, qapp):
        """Response label supports response and error."""
        from ui.widgets import ResponseLabel
        label = ResponseLabel()

        label.set_response("Done.")
        assert label._text_label.text() == "Done."

        label.set_error("Failed.")
        assert label._text_label.text() == "Failed."

    def test_latency_metrics(self, qapp):
        """Latency metrics widget displays values."""
        from ui.widgets import LatencyMetrics
        metrics = LatencyMetrics()

        metrics.set_stt_latency(600)
        assert "600" in metrics._stt_label.text()

        metrics.set_agent_latency(1200)
        assert "1200" in metrics._agent_label.text()

        metrics.set_tts_latency(2000)
        assert "2000" in metrics._tts_label.text()

        metrics.set_total_latency(3800)
        assert "3800" in metrics._total_label.text()


# ═══════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for the full UI flow."""

    def test_full_voice_flow(self, window, bridge):
        """Test a complete voice interaction flow."""
        # User speaks (partial → final)
        bridge.emit_listening()
        process_events(30)

        bridge.emit_partial("what")
        process_events(20)
        bridge.emit_partial("what time")
        process_events(20)
        bridge.emit_final("what time is it")
        process_events(50)

        # Diego thinks
        bridge.emit_thinking()
        process_events(30)

        # Diego responds (printed)
        bridge.emit_response("It's 10:30 AM.")
        process_events(50)

        # Diego speaks
        bridge.emit_speaking()
        process_events(50)

        # Back to idle
        bridge.emit_idle()
        process_events(50)

        # Verify transcript
        transcript = window.get_transcript()
        user_msgs = [m for m in transcript if m["sender"] == "user"]
        diego_msgs = [m for m in transcript if m["sender"] == "diego"]

        assert len(user_msgs) == 1
        assert user_msgs[0]["text"] == "what time is it"
        assert any("10:30" in m["text"] for m in diego_msgs)

    def test_error_recovery_flow(self, window, bridge):
        """Test error display and recovery."""
        # Error occurs
        bridge.emit_error("I couldn't open that app.")
        process_events(50)

        assert window._state_indicator.state() == "Error"

        # Recovery
        bridge.emit_idle()
        process_events(50)

        assert window._state_indicator.state() == "Idle"


# ═══════════════════════════════════════════════════════════════
# Module Import Tests
# ═══════════════════════════════════════════════════════════════

class TestImports:
    """Tests for module imports."""

    def test_import_ui_package(self):
        """ui package can be imported."""
        import ui
        assert hasattr(ui, "EventBridge")
        assert hasattr(ui, "DiegoMainWindow")

    def test_import_event_bridge(self):
        """event_bridge module can be imported."""
        from ui.event_bridge import EventBridge, UIEvent, UIEventType
        assert EventBridge is not None
        assert UIEvent is not None
        assert UIEventType is not None

    def test_import_main_window(self):
        """main_window module can be imported."""
        from ui.main_window import DiegoMainWindow
        assert DiegoMainWindow is not None

    def test_import_widgets(self):
        """widgets module can be imported."""
        from ui.widgets import (
            VoiceStateIndicator, WaveformWidget, MicIndicator,
            TranscriptLabel, ResponseLabel, LatencyMetrics,
        )
        assert VoiceStateIndicator is not None
        assert TranscriptLabel is not None

    def test_import_styles(self):
        """styles module can be imported."""
        from ui.styles import COLORS, MAIN_WINDOW_QSS
        assert "bg_primary" in COLORS
        assert "QMainWindow" in MAIN_WINDOW_QSS