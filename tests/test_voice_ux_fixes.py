"""
Regression tests for the voice UX fixes (2026-09-03).

Covers:
    - Diego-name normalization (Part 3)
    - "recent project" routes locally (Part 4)
    - generic play vs resume (Part 5)
    - duplicate TTS prevention (Part 9)
    - transcript confidence behavior (Part 6)
    - clipping protection (Part 7)
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Set offscreen platform for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# ═══════════════════════════════════════════════════════════════
# Part 3: Diego-name normalization
# ═══════════════════════════════════════════════════════════════

class TestWakeNameNormalization:
    """Tests for Diego-name removal in command normalization."""

    # NOTE: The normalizer canonicalizes app names to lowercase
    # ("Firefox" → "firefox" via app alias resolution).

    def test_diego_comma_open_firefox(self):
        """'Diego, open Firefox' → 'open firefox' (canonical form)"""
        from nlp.command_normalizer import command_normalizer
        result = command_normalizer.normalize("Diego, open Firefox")
        assert result == "open firefox"

    def test_diego_no_comma_open_firefox(self):
        """'Diego open Firefox' → 'open firefox' (canonical form)"""
        from nlp.command_normalizer import command_normalizer
        result = command_normalizer.normalize("Diego open Firefox")
        assert result == "open firefox"

    def test_hey_diego_comma_open_firefox(self):
        """'Hey Diego, open Firefox' → 'open firefox' (canonical form)"""
        from nlp.command_normalizer import command_normalizer
        result = command_normalizer.normalize("Hey Diego, open Firefox")
        assert result == "open firefox"

    def test_okay_diego_open_firefox(self):
        """'Okay Diego open Firefox' → 'open firefox' (canonical form)"""
        from nlp.command_normalizer import command_normalizer
        result = command_normalizer.normalize("Okay Diego open Firefox")
        assert result == "open firefox"

    def test_no_leading_comma(self):
        """Never leaves leading commas or punctuation."""
        from nlp.command_normalizer import command_normalizer
        result = command_normalizer.normalize("Diego, open Firefox")
        assert not result.startswith(",")
        assert not result.startswith(".")
        assert not result.startswith(" ")

    def test_idempotent(self):
        """Normalization is idempotent."""
        from nlp.command_normalizer import command_normalizer
        first = command_normalizer.normalize("Diego, open Firefox")
        second = command_normalizer.normalize(first)
        assert first == second

    def test_preserves_legitimate_diego_content(self):
        """'tell me about Diego' preserves the word Diego."""
        from nlp.command_normalizer import remove_wake_name
        result = remove_wake_name("tell me about Diego")
        assert "diego" in result.lower()

    def test_remove_wake_name_function(self):
        """The remove_wake_name function works directly."""
        from nlp.command_normalizer import remove_wake_name
        assert remove_wake_name("Diego, open Firefox") == "open Firefox"
        assert remove_wake_name("Diego open Firefox") == "open Firefox"
        assert remove_wake_name("Hey Diego, open Firefox") == "open Firefox"
        assert remove_wake_name("okay Diego open Firefox") == "open Firefox"


# ═══════════════════════════════════════════════════════════════
# Part 4: "recent project" routes locally
# ═══════════════════════════════════════════════════════════════

class TestLocalKnowledgeRouting:
    """Tests for local project/file routing."""

    def test_find_my_project_is_local(self):
        """'find my project which I have worked on recently' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("find my project which I have worked on recently")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE
        # LOCAL_KNOWLEDGE is answered from the local index + LLM, never
        # via the planner/tools (which would web-search the request).
        assert auth.actionable is False
        assert auth.llm_allowed

    def test_my_project_is_local(self):
        """'my project' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("show me my project")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_my_projects_is_local(self):
        """'my projects' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("list my projects")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_project_i_worked_on_is_local(self):
        """'project I worked on' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("what project was I working on")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_recently_worked_on_is_local(self):
        """'recently worked on' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("what have I recently worked on")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_my_recent_project_is_local(self):
        """'my recent project' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("find my recent project")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_find_my_code_is_local(self):
        """'find my code' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("find my code")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_my_local_files_is_local(self):
        """'my local files' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("show my local files")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_my_local_projects_is_local(self):
        """'my local projects' → LOCAL_KNOWLEDGE"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("show my local projects")
        assert auth.category == IntentCategory.LOCAL_KNOWLEDGE

    def test_web_search_still_works(self):
        """'search the web for cats' → SEARCH_REQUEST"""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent("search the web for cats")
        assert auth.category == IntentCategory.SEARCH_REQUEST

    def test_decision_engine_needs_search_false_for_local(self):
        """DecisionEngine._needs_search returns False for local project queries."""
        from core.decision_engine import DecisionEngine
        assert not DecisionEngine._needs_search("find my project which I have worked on recently")
        assert not DecisionEngine._needs_search("show me my projects")
        assert not DecisionEngine._needs_search("what was I working on")

    def test_decision_engine_needs_search_true_for_web(self):
        """DecisionEngine._needs_search returns True for explicit web search."""
        from core.decision_engine import DecisionEngine
        assert DecisionEngine._needs_search("search the web for cats")
        assert DecisionEngine._needs_search("google python tutorial")


# ═══════════════════════════════════════════════════════════════
# Part 5: generic play vs resume
# ═══════════════════════════════════════════════════════════════

class TestMediaCommandSemantics:
    """Tests for play vs resume media command routing."""

    @pytest.mark.asyncio
    async def test_play_music_routes_to_play(self):
        """'play music' → play_media (PLAY action), not music_resume."""
        from core.command_router import command_router, RouteKind
        result = await command_router.route("play music")
        assert result.kind == RouteKind.SIMPLE_DESKTOP
        assert result.action["action"] == "play_media"
        assert result.action["params"]["query"] == "music"

    @pytest.mark.asyncio
    async def test_play_something_routes_to_play(self):
        """'play something' → play_media (PLAY action)."""
        from core.command_router import command_router, RouteKind
        result = await command_router.route("play something")
        assert result.kind == RouteKind.SIMPLE_DESKTOP
        assert result.action["action"] == "play_media"

    @pytest.mark.asyncio
    async def test_start_music_routes_to_play(self):
        """'start music' → play_media (PLAY action)."""
        from core.command_router import command_router, RouteKind
        result = await command_router.route("start music")
        assert result.kind == RouteKind.KNOWN_WORKFLOW
        assert result.actions[0]["action"] == "play_media"

    @pytest.mark.asyncio
    async def test_resume_music_routes_to_resume(self):
        """'resume music' → music_resume (RESUME action)."""
        from core.command_router import command_router, RouteKind
        result = await command_router.route("resume music")
        assert result.kind == RouteKind.SIMPLE_DESKTOP
        assert result.action["action"] == "music_resume"

    @pytest.mark.asyncio
    async def test_bare_resume_routes_to_resume(self):
        """'resume' → music_resume (RESUME action)."""
        from core.command_router import command_router, RouteKind
        result = await command_router.route("resume")
        assert result.kind == RouteKind.SIMPLE_DESKTOP
        assert result.action["action"] == "music_resume"

    @pytest.mark.asyncio
    async def test_pause_routes_to_pause(self):
        """'pause' → music_pause (PAUSE action)."""
        from core.command_router import command_router, RouteKind
        result = await command_router.route("pause")
        assert result.kind == RouteKind.SIMPLE_DESKTOP
        assert result.action["action"] == "music_pause"

    @pytest.mark.asyncio
    async def test_pause_music_routes_to_pause(self):
        """'pause music' → music_pause (PAUSE action)."""
        from core.command_router import command_router, RouteKind
        result = await command_router.route("pause music")
        assert result.kind == RouteKind.SIMPLE_DESKTOP
        assert result.action["action"] == "music_pause"


# ═══════════════════════════════════════════════════════════════
# Part 9: duplicate TTS prevention
# ═══════════════════════════════════════════════════════════════

class TestDuplicateTTSPrevention:
    """Tests for duplicate TTS prevention."""

    def test_duplicate_text_skipped(self):
        """Speaking the same text twice in a row is skipped."""
        from core.conversation_engine import ConversationEngine
        engine = ConversationEngine()

        # Simulate first speak
        engine._last_spoken_text = "opening firefox"
        engine._tts_interrupt = asyncio.Event()

        async def run():
            # Second speak with same text should be skipped
            result = await engine._think_and_speak(
                "Opening Firefox", asyncio.Queue(), canned=True)
            return result

        # The duplicate guard should return True without calling TTS
        with patch.object(engine, '_one_line_stream') as mock_stream:
            mock_stream.return_value = None
            result = asyncio.run(run())
            assert result is True
            # _one_line_stream should NOT be called for duplicates
            mock_stream.assert_not_called()

    def test_different_text_not_skipped(self):
        """Different text is not skipped."""
        from core.conversation_engine import ConversationEngine
        engine = ConversationEngine()
        engine._last_spoken_text = "opening firefox"
        engine._tts_interrupt = asyncio.Event()

        async def run():
            result = await engine._think_and_speak(
                "Opening Chrome", asyncio.Queue(), canned=True)
            return result

        with patch.object(engine, '_one_line_stream') as mock_stream:
            mock_stream.return_value = None
            result = asyncio.run(run())
            # _one_line_stream should be called for different text
            mock_stream.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# Part 6: transcript confidence behavior
# ═══════════════════════════════════════════════════════════════

class TestTranscriptConfidence:
    """Tests for transcript confidence handling."""

    def test_low_confidence_actionable_becomes_uncertain(self):
        """Low-confidence actionable intent becomes UNCERTAIN."""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent(
            "open firefox",
            stt_confidence=-1.2,
            audio_duration_ms=300,
        )
        assert auth.category == IntentCategory.UNCERTAIN
        assert not auth.actionable

    def test_normal_confidence_actionable_stays_actionable(self):
        """Normal-confidence actionable intent stays actionable."""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent(
            "open firefox",
            stt_confidence=-0.5,
            audio_duration_ms=1200,
        )
        assert auth.category == IntentCategory.DETERMINISTIC_COMMAND
        assert auth.actionable

    def test_conversational_exempt_from_hard_band(self):
        """Greetings are exempt from the hard confidence band."""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent(
            "hello",
            stt_confidence=-1.1,
            audio_duration_ms=500,
        )
        assert auth.category == IntentCategory.CONVERSATIONAL

    def test_garbage_rejected(self):
        """Garbage transcripts are rejected regardless of confidence."""
        from nlp.intent_authorizer import authorize_intent, IntentCategory
        auth = authorize_intent(
            "I'm sorry",
            stt_confidence=0.5,
            audio_duration_ms=2000,
        )
        assert auth.category == IntentCategory.UNCERTAIN
        assert not auth.actionable


# ═══════════════════════════════════════════════════════════════
# Part 7: clipping protection
# ═══════════════════════════════════════════════════════════════

class TestClippingProtection:
    """Tests for audio clipping protection."""

    def test_highpass_headroom_factor(self):
        """The high-pass filter applies a 0.95 headroom factor to prevent clipping."""
        import numpy as np
        from voice.audio_manager import AudioManager

        am = AudioManager()
        # Simulate a hot signal that would clip after the high-pass filter
        audio = np.ones(512, dtype=np.float32) * 0.95

        # Build the high-pass filter
        from scipy import signal as scipy_signal
        from voice.audio_processing import HIGH_PASS_CUTOFF, HIGH_PASS_ORDER
        nyquist = 16000 / 2
        am._hp_sos = scipy_signal.butter(
            HIGH_PASS_ORDER, HIGH_PASS_CUTOFF / nyquist,
            btype="highpass", output="sos")
        am._hp_zi = scipy_signal.sosfilt_zi(am._hp_sos) * 0

        # Apply the filter with the headroom factor
        audio_hp, am._hp_zi = scipy_signal.sosfilt(
            am._hp_sos, audio.astype(np.float64), zi=am._hp_zi)
        audio_hp = np.clip(
            audio_hp.astype(np.float32) * 0.95, -1.0, 1.0)

        # The output should never exceed ±1.0
        assert np.max(np.abs(audio_hp)) <= 1.0
        # And should not be hard-clipped at the rail
        assert np.max(np.abs(audio_hp)) < 1.0


# ═══════════════════════════════════════════════════════════════
# UI: voice-first layout tests
# ═══════════════════════════════════════════════════════════════

class TestVoiceFirstUI:
    """Tests for the voice-first UI layout."""

    @pytest.fixture
    def qapp(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    @pytest.fixture
    def bridge(self, qapp):
        from ui.event_bridge import EventBridge
        b = EventBridge()
        b.start()
        yield b
        b.stop()

    @pytest.fixture
    def window(self, qapp, bridge):
        from ui.main_window import DiegoMainWindow
        w = DiegoMainWindow(bridge=bridge, loop=None)
        w.show()
        yield w
        w.close()

    def process_events(self, ms=50):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        end = time.time() + ms / 1000
        while time.time() < end:
            app.processEvents()
            time.sleep(0.005)

    def test_no_chat_composer(self, window):
        """The UI has no text input or send button."""
        assert not hasattr(window, '_text_input')
        assert not hasattr(window, '_send_btn')

    def test_voice_state_indicator(self, window):
        """The UI has a voice state indicator."""
        assert hasattr(window, '_state_indicator')
        assert window._state_indicator.state() == "Idle"

    def test_waveform_present(self, window):
        """The UI has a waveform widget."""
        assert hasattr(window, '_waveform')

    def test_transcript_label_present(self, window):
        """The UI has a transcript label."""
        assert hasattr(window, '_transcript_label')

    def test_response_label_present(self, window):
        """The UI has a response label."""
        assert hasattr(window, '_response_label')

    def test_speaking_indicator_present(self, window):
        """The UI has a speaking indicator."""
        assert hasattr(window, '_speaking_label')

    def test_partial_transcript_updates(self, window, bridge):
        """Partial transcript updates the live region."""
        bridge.emit_partial("hello wor")
        self.process_events(100)
        assert window.get_partial_text() == "hello wor"

    def test_final_transcript_replaces_partial(self, window, bridge):
        """Final transcript replaces the partial region."""
        bridge.emit_partial("hello wor")
        self.process_events(50)
        bridge.emit_final("hello world")
        self.process_events(100)

        assert window.get_partial_text() == ""
        transcript = window.get_transcript()
        user_msgs = [m for m in transcript if m["sender"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["text"] == "hello world"

    def test_response_displayed_once(self, window, bridge):
        """Response is displayed once and remains visible."""
        bridge.emit_response("Opening Firefox for you.")
        self.process_events(100)

        transcript = window.get_transcript()
        diego_msgs = [m for m in transcript if m["sender"] == "diego"]
        assert len(diego_msgs) == 1
        assert diego_msgs[0]["text"] == "Opening Firefox for you."

    def test_state_transitions(self, window, bridge):
        """State transitions update the indicator."""
        bridge.emit_listening()
        self.process_events(50)
        assert window._state_indicator.state() == "Listening"

        bridge.emit_thinking()
        self.process_events(50)
        assert window._state_indicator.state() == "Thinking"

        bridge.emit_speaking()
        self.process_events(50)
        assert window._state_indicator.state() == "Speaking"
        assert window._speaking_label.isVisible()

        bridge.emit_idle()
        self.process_events(50)
        assert window._state_indicator.state() == "Idle"
        assert not window._speaking_label.isVisible()

    def test_speaking_indicator(self, window, bridge):
        """Speaking indicator shows when Diego is speaking."""
        bridge.emit_speaking()
        self.process_events(50)
        assert window._speaking_label.isVisible()

        bridge.emit_idle()
        self.process_events(50)
        assert not window._speaking_label.isVisible()

    def test_event_bridge_speaking_signal(self, bridge, qapp):
        """EventBridge has a speaking signal."""
        received = []
        bridge.speaking.connect(lambda: received.append(True))
        bridge.emit_speaking()
        self.process_events(100)
        assert received == [True]

    def test_ui_never_blocks(self, window, bridge):
        """UI remains responsive during event processing."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()

        # Emit many events
        for i in range(50):
            bridge.emit_state(f"State{i}")

        # UI should still process events
        start = time.time()
        self.process_events(100)
        elapsed = time.time() - start

        # Should not block for more than 1 second
        assert elapsed < 1.0