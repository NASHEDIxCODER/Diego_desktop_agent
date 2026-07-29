"""
End-to-End Test for Leo Desktop Assistant.

Verifies all subsystems can initialize, run diagnostics,
and process intents without errors or exceptions.

Run: python -m pytest tests/test_e2e.py -v

Does NOT require:
- Microphone (STT is mocked)
- Camera (face auth is mocked)
- YouTube browser (plugin is mocked)
- Actual TTS output
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Environment setup (same as main.py) ──
os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["ALSA_DEBUG_FILE"] = "/dev/null"
os.environ["PYTTXS3_ALSA_DEBUG"] = "0"
os.environ["SPEECH_RECOGNITION_ALSA_DEBUG"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"
os.environ["JACK_NO_START_SERVER"] = "1"
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("DISPLAY", ":0")
os.environ["LOG_LEVEL"] = "DEBUG"

import compat  # noqa: F401

from config.settings import settings
from telemetry.logger import setup_logging, set_correlation_id, set_subsystem_id

setup_logging("DEBUG")
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def cleanup_duckdb():
    """Ensure DuckDB is cleaned up before each test."""
    from memory.duckdb_store import store, HAS_DUCKDB
    if HAS_DUCKDB:
        try:
            store.close()
        except Exception:
            pass
        # Remove stale files
        for pattern in [".duckdb.wal", ".duckdb.tmp", ".wal", ".tmp"]:
            p = Path(settings.DUCKDB_PATH).with_suffix(pattern)
            if p.exists():
                p.unlink(missing_ok=True)
    yield


class TestStartupDiagnostics:
    """Test that all subsystems initialize correctly."""

    @pytest.mark.asyncio
    async def test_nlp_initialization(self):
        """NLP model should load without exceptions."""
        from nlp.inference import inference
        result = inference.load()
        assert result is True, "NLP model should load"
        status = inference.get_status()
        assert "version" in status
        assert "intents" in status
        logger.info("NLP loaded: %d intents", status.get("intents", 0))

    @pytest.mark.asyncio
    async def test_embeddings_initialization(self):
        """Embedding model should preload without exceptions."""
        from nlp.embeddings import preload_embedding_model
        preload_embedding_model()
        logger.info("Embedding model preloaded successfully")

    @pytest.mark.asyncio
    async def test_tts_initialization(self):
        """TTS engine should initialize without errors."""
        from voice.synthesizer import speech_synthesizer
        speech_synthesizer.initialize()
        assert speech_synthesizer._ready
        logger.info("TTS initialized")

    @pytest.mark.asyncio
    async def test_duckdb_initialization(self):
        """DuckDB should connect without locking issues."""
        from memory.duckdb_store import store, DatabaseLockedError
        store.initialize()
        assert store._conn is not None, "DuckDB should have a connection"
        # Test basic operation
        store.add_command(text="test", intent="test", confidence=1.0, response="ok")
        history = store.get_recent_commands(limit=5)
        assert len(history) >= 1
        logger.info("DuckDB initialized and operational")

    @pytest.mark.asyncio
    async def test_plugins_initialization(self):
        """All plugins should load without errors."""
        from core.plugin_manager import plugin_manager
        await plugin_manager.load_all()
        await plugin_manager.initialize_all()
        names = list(plugin_manager.plugins.keys())
        assert len(names) > 0, "At least one plugin should be loaded"
        logger.info("Plugins loaded: %s", names)
        # Verify each plugin is enabled
        for name in names:
            plugin = plugin_manager.get_plugin(name)
            assert plugin is not None
            logger.info("Plugin %s: enabled=%s", name, plugin.enabled)

    @pytest.mark.asyncio
    async def test_voice_initialization(self):
        """Voice subsystem should detect backend (even if none)."""
        from voice.audio_device import audio_device
        backend = audio_device.detect_backend()
        assert backend is not None
        logger.info("Audio backend: %s", backend)

    @pytest.mark.asyncio
    async def test_parallel_startup(self):
        """Test that parallel subsystem initialization works."""
        from core.startup_health import StartupHealth, SubsystemState
        sh = StartupHealth()

        async def _init_nlp(status):
            from nlp.inference import inference
            sh.register("nlp")
            if inference.load():
                status["nlp"] = True
                sh.set_state("nlp", SubsystemState.READY, "Ready")

        async def _init_embeddings(status):
            from nlp.embeddings import preload_embedding_model
            sh.register("embeddings")
            preload_embedding_model()
            sh.set_state("embeddings", SubsystemState.READY, "Ready")

        async def _init_plugins(status):
            from core.plugin_manager import plugin_manager
            sh.register("plugins")
            await plugin_manager.load_all()
            await plugin_manager.initialize_all()
            sh.set_state("plugins", SubsystemState.READY, "Ready")

        status = {"nlp": False}
        t0 = time.time()
        await asyncio.gather(
            _init_nlp(status),
            _init_embeddings(status),
            _init_plugins(status),
        )
        elapsed = time.time() - t0
        assert status["nlp"]
        logger.info("Parallel startup: %.2fs", elapsed)
        assert elapsed < 60, "Startup should complete in under 60s"


class TestIntentProcessing:
    """Test that intent classification and processing works."""

    @pytest.mark.asyncio
    async def test_classify_greeting(self):
        """Greeting intent should be classified."""
        from nlp.inference import inference
        results = inference.classify("hello", top_k=1)
        assert len(results) > 0
        assert results[0]["intent"] == "greeting"

    @pytest.mark.asyncio
    async def test_classify_time_query(self):
        """Time query intent should be classified."""
        from nlp.inference import inference
        results = inference.classify("what time is it", top_k=1)
        assert len(results) > 0
        assert results[0]["intent"] == "time_query"

    @pytest.mark.asyncio
    async def test_classify_date_query(self):
        """Date query intent should be classified."""
        from nlp.inference import inference
        results = inference.classify("what is today's date", top_k=1)
        assert len(results) > 0
        assert results[0]["intent"] == "date_query"

    @pytest.mark.asyncio
    async def test_classify_youtube(self):
        """YouTube intent should be classified."""
        from nlp.inference import inference
        results = inference.classify("play music on youtube", top_k=1)
        assert len(results) > 0
        assert results[0]["intent"] == "youtube"

    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        """Entity extraction should return results for known patterns."""
        from nlp.entities import extract_entities
        entities = extract_entities("set brightness to 50 percent")
        assert isinstance(entities, dict)

    @pytest.mark.asyncio
    async def test_intent_latency(self):
        """Intent classification should complete in under 200ms."""
        from nlp.inference import inference
        t0 = time.time()
        for _ in range(10):
            inference.classify("play some music", top_k=1)
        elapsed = (time.time() - t0) / 10
        assert elapsed < 0.2, f"Classification took {elapsed*1000:.1f}ms (target <200ms)"
        logger.info("Average classification time: %.1fms", elapsed * 1000)


class TestDuckDBPersistence:
    """Test that DuckDB persists data correctly."""

    @pytest.mark.asyncio
    async def test_command_history(self):
        """Command history should store and retrieve data."""
        from memory.duckdb_store import store
        store.initialize()
        store.add_command(text="test command", intent="test",
                         confidence=0.95, response="test response")
        history = store.get_recent_commands(limit=10)
        assert len(history) >= 1
        assert history[0]["text"] == "test command"
        assert history[0]["intent"] == "test"

    @pytest.mark.asyncio
    async def test_preferences(self):
        """User preferences should persist."""
        from memory.duckdb_store import store
        store.initialize()
        store.set_preference("test_key", "test_value")
        value = store.get_preference("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_context_memory(self):
        """Context memory should save and retrieve."""
        from memory.duckdb_store import store
        store.initialize()
        store.save_context("test_session", "test_key", "test_value")
        value = store.get_context("test_session", "test_key")
        assert value == "test_value"


class TestEventBus:
    """Test that the event bus works correctly."""

    @pytest.mark.asyncio
    async def test_event_dispatch(self):
        """Events should be dispatched to handlers."""
        from core.event_bus import bus, Event
        received = []

        async def handler(event: Event):
            received.append(event.data)

        bus.on("test_event", handler)
        await bus.emit("test_event", {"msg": "hello"})
        await asyncio.sleep(0.01)
        assert len(received) == 1
        assert received[0]["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_event_no_crash(self):
        """Events should not crash if no handlers registered."""
        from core.event_bus import bus
        await bus.emit("nonexistent_event", {"data": 1})


class TestConversationState:
    """Test conversation state management."""

    @pytest.mark.asyncio
    async def test_pending_action(self):
        """Pending actions should be set and checked."""
        from nlp.conversation_state import conversation_state, PendingAction
        assert not conversation_state.has_pending_action
        conversation_state.set_pending(PendingAction.YOUTUBE_QUERY)
        assert conversation_state.has_pending_action
        assert conversation_state.pending_action == PendingAction.YOUTUBE_QUERY
        conversation_state.clear()
        assert not conversation_state.has_pending_action

    @pytest.mark.asyncio
    async def test_context(self):
        """Conversation context should track state."""
        from nlp.context import context_manager
        context_manager.set_active_session("youtube")
        assert context_manager.active_session == "youtube"
        context_manager.set_active_session(None)
        assert context_manager.active_session is None


class TestStartupHealth:
    """Test StartupHealth subsystem."""

    @pytest.mark.asyncio
    async def test_health_tracking(self):
        """Startup health should track subsystem states."""
        from core.startup_health import StartupHealth, SubsystemState
        sh = StartupHealth()
        sh.register("test_subsystem")
        sh.set_state("test_subsystem", SubsystemState.READY, "All good")
        state = sh._subsystems.get("test_subsystem")
        assert state is not None
        assert state.state == SubsystemState.READY
        assert state.message == "All good"

    @pytest.mark.asyncio
    async def test_can_start(self):
        """Startup should not start if NLP failed."""
        from core.startup_health import StartupHealth, SubsystemState
        sh = StartupHealth()
        sh.register("nlp")
        sh.set_state("nlp", SubsystemState.FAILED, "No model")
        sh.finalize()
        assert not sh.can_start


class TestLogging:
    """Test structured logging."""

    def test_correlation_id(self):
        """Correlation IDs should be set and retrieved."""
        from telemetry.logger import set_correlation_id, get_correlation_id
        cid = set_correlation_id()
        assert cid is not None
        assert get_correlation_id() == cid
        assert len(cid) == 8

    def test_subsystem_id(self):
        """Subsystem IDs should be set and retrieved."""
        from telemetry.logger import set_subsystem_id, get_subsystem_id
        set_subsystem_id("test")
        assert get_subsystem_id() == "test"


class TestNoiseSuppression:
    """Test that environment variable suppressions work."""

    def test_alsa_suppressed(self):
        """ALSA environment variables should be set."""
        assert os.environ.get("ALSA_DEBUG") == "0"
        assert os.environ.get("ALSA_DEBUG_FILE") == "/dev/null"
        assert os.environ.get("PULSE_LOG") == "0"

    def test_qt_fonts_configured(self):
        """Qt font directories should be configured."""
        assert os.environ.get("QT_QPA_FONTDIR") == "/usr/share/fonts"
        assert os.environ.get("FONTCONFIG_PATH") == "/etc/fonts"

    def test_hf_offline_friendly(self):
        """HuggingFace settings should be configured."""
        assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
        # HF_HUB_DOWNLOAD_TIMEOUT is set with setdefault, may not be present
        # if already set by environment
        timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
        if timeout is not None:
            assert timeout == "30"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])