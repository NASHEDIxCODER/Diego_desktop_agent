"""
Tests for the Diego Desktop Voice-First HUD UI.

Covers:
    - UI startup
    - Event bridge thread safety
    - Voice visualizer states
    - Real audio level updates
    - Listening animation
    - Speaking animation
    - Partial transcript update (no duplication)
    - Final transcript replacement
    - Response rendering
    - State transitions
    - Resize behavior
    - No chat composer
    - UI non-blocking
    - Event bridge integration
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
        assert window._visualizer.state.name == "IDLE"

    def test_voice_first_layout(self, window):
        """The UI is voice-first: no chat composer, no text input."""
        assert not hasattr(window, '_text_input')
        assert not hasattr(window, '_send_btn')
        # Voice-first widgets are present
        assert hasattr(window, '_visualizer')
        assert hasattr(window, '_transcript_panel')
        assert hasattr(window, '_response_panel')
        assert hasattr(window, '_activity_panel')
        assert hasattr(window, '_metrics')
        assert hasattr(window, '_system_status')

    def test_no_chat_composer(self, window):
        """The UI does not have a chat composer or send button."""
        # Check that there's no text input widget
        from PySide6.QtWidgets import QLineEdit, QTextEdit
        line_edits = window.findChildren(QLineEdit)
        text_edits = window.findChildren(QTextEdit)
        assert len(line_edits) == 0, "UI should not have QLineEdit (chat composer)"
        assert len(text_edits) == 0, "UI should not have QTextEdit (chat composer)"


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

    def test_emit_audio_level(self, bridge, qapp):
        """Audio level events are delivered."""
        received = []
        bridge.audio_level.connect(lambda l: received.append(l))

        bridge.emit_audio_level(0.5)
        process_events(100)

        assert len(received) == 1
        assert abs(received[0] - 0.5) < 0.01

    def test_emit_tts_level(self, bridge, qapp):
        """TTS level events are delivered."""
        received = []
        bridge.tts_level.connect(lambda l: received.append(l))

        bridge.emit_tts_level(0.7)
        process_events(100)

        assert len(received) == 1
        assert abs(received[0] - 0.7) < 0.01

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
# Voice Visualizer Tests
# ═══════════════════════════════════════════════════════════════

class TestVoiceVisualizer:
    """Tests for the VoiceCoreVisualizer."""

    def test_visualizer_creation(self, qapp):
        """VoiceCoreVisualizer can be created."""
        from ui.visualizer import VoiceCoreVisualizer, VisualizerState
        viz = VoiceCoreVisualizer()
        assert viz is not None
        assert viz.state == VisualizerState.IDLE

    def test_visualizer_states(self, qapp):
        """Visualizer supports all required states."""
        from ui.visualizer import VoiceCoreVisualizer, VisualizerState
        viz = VoiceCoreVisualizer()

        viz.set_state(VisualizerState.IDLE)
        assert viz.state == VisualizerState.IDLE

        viz.set_state(VisualizerState.LISTENING)
        assert viz.state == VisualizerState.LISTENING

        viz.set_state(VisualizerState.SPEECH_DETECTED)
        assert viz.state == VisualizerState.SPEECH_DETECTED

        viz.set_state(VisualizerState.THINKING)
        assert viz.state == VisualizerState.THINKING

        viz.set_state(VisualizerState.EXECUTING)
        assert viz.state == VisualizerState.EXECUTING

        viz.set_state(VisualizerState.SPEAKING)
        assert viz.state == VisualizerState.SPEAKING

        viz.set_state(VisualizerState.ERROR)
        assert viz.state == VisualizerState.ERROR

    def test_visualizer_state_by_name(self, qapp):
        """Visualizer can set state from string names."""
        from ui.visualizer import VoiceCoreVisualizer, VisualizerState
        viz = VoiceCoreVisualizer()

        viz.set_state_by_name("Listening")
        assert viz.state == VisualizerState.LISTENING

        viz.set_state_by_name("Thinking")
        assert viz.state == VisualizerState.THINKING

        viz.set_state_by_name("Speaking")
        assert viz.state == VisualizerState.SPEAKING

        viz.set_state_by_name("Executing")
        assert viz.state == VisualizerState.EXECUTING

        viz.set_state_by_name("Idle")
        assert viz.state == VisualizerState.IDLE

    def test_visualizer_input_level(self, qapp):
        """Visualizer accepts real audio input levels."""
        from ui.visualizer import VoiceCoreVisualizer
        viz = VoiceCoreVisualizer()

        viz.set_input_level(0.5)
        assert viz._target_input_level == 0.5

        viz.set_input_level(1.5)  # Should clamp
        assert viz._target_input_level == 1.0

        viz.set_input_level(-0.5)  # Should clamp
        assert viz._target_input_level == 0.0

    def test_visualizer_output_level(self, qapp):
        """Visualizer accepts TTS output levels."""
        from ui.visualizer import VoiceCoreVisualizer
        viz = VoiceCoreVisualizer()

        viz.set_output_level(0.6)
        assert viz._target_output_level == 0.6

    def test_listening_animation(self, window, bridge):
        """Listening state triggers listening animation."""
        from ui.visualizer import VisualizerState

        bridge.emit_listening()
        process_events(100)

        assert window._visualizer.state == VisualizerState.LISTENING

    def test_speaking_animation(self, window, bridge):
        """Speaking state triggers speaking animation."""
        from ui.visualizer import VisualizerState

        bridge.emit_speaking()
        process_events(100)

        assert window._visualizer.state == VisualizerState.SPEAKING

    def test_thinking_animation(self, window, bridge):
        """Thinking state triggers thinking animation."""
        from ui.visualizer import VisualizerState

        bridge.emit_thinking()
        process_events(100)

        assert window._visualizer.state == VisualizerState.THINKING

    def test_audio_level_updates_visualizer(self, window, bridge):
        """Real audio levels update the visualizer."""
        bridge.emit_listening()
        process_events(50)

        bridge.emit_audio_level(0.7)
        process_events(100)

        # The visualizer should have received the level
        assert window._visualizer._target_input_level == pytest.approx(0.7, abs=0.01)

    def test_visualizer_smooth_interpolation(self, qapp):
        """Visualizer uses smooth interpolation, not jittery jumps."""
        from ui.visualizer import VoiceCoreVisualizer, VisualizerState
        viz = VoiceCoreVisualizer()
        viz.set_state(VisualizerState.LISTENING)

        # Set a target level
        viz.set_input_level(0.8)

        # Run a few animation frames
        for _ in range(5):
            viz._animate()

        # Level should be smoothed (not instantly at target)
        assert viz._input_level < 0.8
        assert viz._input_level > 0.0

    def test_visualizer_idle_subtle(self, qapp):
        """Idle animation is extremely subtle."""
        from ui.visualizer import VoiceCoreVisualizer, VisualizerState
        viz = VoiceCoreVisualizer()
        viz.set_state(VisualizerState.IDLE)

        # Run idle animation
        for _ in range(10):
            viz._animate()

        # Bar values should be very small (subtle)
        max_bar = max(viz._bar_values)
        assert max_bar < 0.1, "Idle animation should be extremely subtle"


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

        assert window._visualizer.state.name == "ERROR"
        assert "couldn't find" in window.get_response_text() or \
               "couldn't find" in window._response_panel._text_label.text()

    def test_transcript_panel_partial_style(self, qapp):
        """Transcript panel shows partial text with appropriate style."""
        from ui.widgets import TranscriptPanel
        panel = TranscriptPanel()

        panel.set_partial("hel")
        assert panel.text() == "hel"
        assert panel.is_partial()

    def test_transcript_panel_final_style(self, qapp):
        """Transcript panel shows final text with appropriate style."""
        from ui.widgets import TranscriptPanel
        panel = TranscriptPanel()

        panel.set_final("hello world")
        assert panel.text() == "hello world"
        assert not panel.is_partial()


# ═══════════════════════════════════════════════════════════════
# State Tests
# ═══════════════════════════════════════════════════════════════

class TestStateChanges:
    """Tests for state display."""

    def test_listening_state(self, window, bridge):
        """Listening state is displayed."""
        bridge.emit_listening()
        process_events(100)

        assert window._state_label.text() == "LISTENING"
        assert window._mic_indicator._active

    def test_thinking_state(self, window, bridge):
        """Thinking state is displayed."""
        bridge.emit_thinking()
        process_events(100)

        assert window._state_label.text() == "THINKING"
        assert not window._mic_indicator._active

    def test_speaking_state(self, window, bridge):
        """Speaking state is displayed with speaking indicator."""
        bridge.emit_speaking()
        process_events(100)

        assert window._state_label.text() == "SPEAKING"
        assert window._speaking_label.isVisible()

    def test_executing_state(self, window, bridge):
        """Executing state is displayed."""
        bridge.emit_executing("desktop_open")
        process_events(100)

        state = window._state_label.text()
        assert "EXECUTING" in state

    def test_idle_state(self, window, bridge):
        """Idle state is displayed."""
        bridge.emit_thinking()
        process_events(50)
        bridge.emit_idle()
        process_events(100)

        assert window._state_label.text() == "IDLE"
        assert not window._mic_indicator._active
        assert not window._speaking_label.isVisible()

    def test_error_state(self, window, bridge):
        """Error state is displayed."""
        bridge.emit_error("Test error")
        process_events(100)

        assert window._state_label.text() == "ERROR"

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
            assert window._state_label.text() == state.upper()

    def test_state_transition_sequence(self, window, bridge):
        """Test a realistic state transition sequence."""
        from ui.visualizer import VisualizerState

        # Idle → Listening → Thinking → Speaking → Idle
        bridge.emit_idle()
        process_events(30)
        assert window._visualizer.state == VisualizerState.IDLE

        bridge.emit_listening()
        process_events(30)
        assert window._visualizer.state == VisualizerState.LISTENING

        bridge.emit_thinking()
        process_events(30)
        assert window._visualizer.state == VisualizerState.THINKING

        bridge.emit_speaking()
        process_events(30)
        assert window._visualizer.state == VisualizerState.SPEAKING

        bridge.emit_idle()
        process_events(30)
        assert window._visualizer.state == VisualizerState.IDLE


# ═══════════════════════════════════════════════════════════════
# Activity Panel Tests
# ═══════════════════════════════════════════════════════════════

class TestActivityPanel:
    """Tests for the activity panel."""

    def test_activity_panel_creation(self, qapp):
        """ActivityPanel can be created."""
        from ui.widgets import ActivityPanel
        panel = ActivityPanel()
        assert panel is not None

    def test_activity_panel_shows_human_readable(self, qapp):
        """Activity panel shows only human-readable activities."""
        from ui.widgets import ActivityPanel
        panel = ActivityPanel()

        # All activities should be human-readable
        for activity in panel.ACTIVITIES:
            assert not any(c in activity.lower() for c in ['_', '{', '}', '/'])
            assert activity[0].isupper()

    def test_activity_panel_state_mapping(self, qapp):
        """Activity panel maps states to activities correctly."""
        from ui.widgets import ActivityPanel
        panel = ActivityPanel()

        panel.set_active("Listening")
        assert panel.active_activity() == "Voice detected"

        panel.set_active("Thinking")
        assert panel.active_activity() == "Thinking"

        panel.set_active("Executing")
        assert panel.active_activity() == "Executing"

        panel.set_active("Speaking")
        assert panel.active_activity() == "Responding"

    def test_activity_panel_reset(self, qapp):
        """Activity panel can be reset."""
        from ui.widgets import ActivityPanel
        panel = ActivityPanel()

        panel.set_active("Thinking")
        assert panel.active_activity() == "Thinking"

        panel.reset()
        assert panel.active_activity() == ""


# ═══════════════════════════════════════════════════════════════
# Metrics Tests
# ═══════════════════════════════════════════════════════════════

class TestMetrics:
    """Tests for latency metrics."""

    def test_metrics_cards_creation(self, qapp):
        """MetricsCards can be created."""
        from ui.widgets import MetricsCards
        metrics = MetricsCards()
        assert metrics is not None

    def test_metrics_display(self, qapp):
        """Metrics display latency values."""
        from ui.widgets import MetricsCards
        metrics = MetricsCards()

        metrics.set_stt_latency(600)
        assert "600" in metrics._cards["STT"].text()

        metrics.set_agent_latency(1200)
        assert "1200" in metrics._cards["Agent"].text()

        metrics.set_tts_latency(2000)
        assert "2000" in metrics._cards["TTS"].text()

        metrics.set_total_latency(3800)
        assert "3800" in metrics._cards["Total"].text()

    def test_metrics_reset(self, qapp):
        """Metrics can be reset."""
        from ui.widgets import MetricsCards
        metrics = MetricsCards()

        metrics.set_stt_latency(600)
        metrics.reset()
        assert metrics._cards["STT"].text() == "--"


# ═══════════════════════════════════════════════════════════════
# System Status Tests
# ═══════════════════════════════════════════════════════════════

class TestSystemStatus:
    """Tests for system status display."""

    def test_system_status_creation(self, qapp):
        """SystemStatus can be created."""
        from ui.widgets import SystemStatus
        status = SystemStatus()
        assert status is not None

    def test_system_status_ok(self, qapp):
        """System status shows OK state."""
        from ui.widgets import SystemStatus
        status = SystemStatus()

        status.set_status("STT", True)
        assert status.status("STT")

        status.set_all_ok()
        for comp in SystemStatus.COMPONENTS:
            assert status.status(comp)

    def test_system_status_error(self, qapp):
        """System status shows error state."""
        from ui.widgets import SystemStatus
        status = SystemStatus()

        status.set_status("TTS", False)
        assert not status.status("TTS")


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

    def test_audio_level_updates_non_blocking(self, window, bridge):
        """Rapid audio level updates don't block the UI."""
        start = time.time()

        # Simulate rapid audio level updates (like real mic callback)
        for i in range(100):
            bridge.emit_audio_level(i / 100.0)

        process_events(100)
        elapsed = time.time() - start

        assert elapsed < 1.0, "Audio level updates should not block UI"


# ═══════════════════════════════════════════════════════════════
# Resize Tests
# ═══════════════════════════════════════════════════════════════

class TestResize:
    """Tests for window resize behavior."""

    def test_window_resize(self, window):
        """Window can be resized."""
        window.resize(800, 700)
        process_events(50)
        assert window.width() == 800
        assert window.height() == 700

    def test_visualizer_scales_with_window(self, window):
        """Visualizer scales gracefully with window resize."""
        window.resize(900, 800)
        process_events(50)

        # Visualizer should still be visible and have reasonable size
        assert window._visualizer.width() > 100
        assert window._visualizer.height() > 100

    def test_minimum_size(self, window):
        """Window respects minimum size."""
        window.resize(400, 300)  # Below minimum
        process_events(50)

        # Should be clamped to minimum
        assert window.width() >= 680
        assert window.height() >= 620

    def test_small_window_readability(self, window):
        """Transcript/response remain readable at smaller sizes."""
        window.resize(680, 620)  # Minimum size
        process_events(50)

        # Panels should still be visible
        assert window._transcript_panel.isVisible()
        assert window._response_panel.isVisible()


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

        assert window._state_label.text() == "ERROR"

        # Recovery
        bridge.emit_idle()
        process_events(50)

        assert window._state_label.text() == "IDLE"

    def test_audio_reactive_flow(self, window, bridge):
        """Test audio-reactive visualization flow."""
        from ui.visualizer import VisualizerState

        # Start listening
        bridge.emit_listening()
        process_events(30)

        # Simulate real mic audio levels
        for level in [0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1]:
            bridge.emit_audio_level(level)
            process_events(20)

        # Visualizer should be in listening state with some activity
        assert window._visualizer.state == VisualizerState.LISTENING

        # Speech detected (stronger pulse)
        bridge.emit_partial("hello")
        process_events(30)
        assert window._visualizer.state == VisualizerState.SPEECH_DETECTED


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

    def test_import_visualizer(self):
        """visualizer module can be imported."""
        from ui.visualizer import VoiceCoreVisualizer, VisualizerState
        assert VoiceCoreVisualizer is not None
        assert VisualizerState is not None

    def test_import_widgets(self):
        """widgets module can be imported."""
        from ui.widgets import (
            ConnectionIndicator, TranscriptPanel, ResponsePanel,
            ActivityPanel, MetricsCards, SystemStatus,
        )
        assert TranscriptPanel is not None
        assert ResponsePanel is not None

    def test_import_styles(self):
        """styles module can be imported."""
        from ui.styles import COLORS, MAIN_WINDOW_QSS, FONTS
        assert "bg_primary" in COLORS
        assert "QMainWindow" in MAIN_WINDOW_QSS
        assert "family" in FONTS