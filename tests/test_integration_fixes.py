"""
Regression tests for the dead-code audit integration fixes (2026-08-31).

Covers:
  1. --no-auth genuinely bypasses face authentication (set_auth_disabled)
  2. _ocr_async is defined and uses the production vision service
  3. llm_client uses the configured OLLAMA_KEEP_ALIVE (not hard-coded 0)
  4. Dead provider callbacks (set_vision_context / set_search_provider /
     set_learning_context) are removed — the Brain uses direct paths
"""

import inspect

import pytest


# ═══════════════════════════════════════════════════════════════
# 1. --no-auth bypass
# ═══════════════════════════════════════════════════════════════

class TestNoAuthBypass:
    """The --no-auth flag must genuinely disable face authentication."""

    def test_set_auth_disabled_clears_provider(self):
        from core.conversation_engine import ConversationEngine

        engine = ConversationEngine()
        # Simulate the normal (auth-enabled) wiring
        engine.set_auth_provider(lambda: "test_user")
        assert engine._needs_auth() is True  # provider set, no user yet

        # --no-auth path
        engine.set_auth_disabled()
        assert engine._auth_provider is None
        assert engine._needs_auth() is False  # auth genuinely bypassed

    def test_set_authenticated_none_does_not_bypass(self):
        """The OLD buggy path: set_authenticated(None) must NOT be used
        to bypass auth — it leaves the provider set so _needs_auth() is True."""
        from core.conversation_engine import ConversationEngine

        engine = ConversationEngine()
        engine.set_auth_provider(lambda: "test_user")
        engine.set_authenticated(None)  # old buggy --no-auth path
        # This is the bug: _needs_auth() is still True because the provider
        # is set and _auth_user is None. The fix uses set_auth_disabled().
        assert engine._needs_auth() is True

    def test_diego_wiring_uses_set_auth_disabled(self):
        """Diego.py must call set_auth_disabled() for --no-auth, not
        set_authenticated(None)."""
        import Diego
        src = inspect.getsource(Diego.run_Diego)
        # The actual call site must use set_auth_disabled()
        assert "conversation_engine.set_auth_disabled()" in src
        # The old buggy call must NOT be an active call (only a comment
        # may mention it for documentation).
        assert "conversation_engine.set_authenticated(None)" not in src


# ═══════════════════════════════════════════════════════════════
# 2. _ocr_async defined
# ═══════════════════════════════════════════════════════════════

class TestOcrAsync:
    """_ocr_async must be defined and use the production vision service."""

    def test_ocr_async_is_defined(self):
        from agent.action_dispatcher import ActionDispatcher

        d = ActionDispatcher()
        assert hasattr(d, "_ocr_async")
        assert callable(d._ocr_async)

    def test_ocr_async_uses_vision_service(self):
        """_ocr_async must call the production vision_service.force_analyze()
        — no duplicate OCR logic."""
        from agent.action_dispatcher import ActionDispatcher

        src = inspect.getsource(ActionDispatcher._ocr_async)
        assert "vision_service" in src
        assert "force_analyze" in src

    def test_ocr_sync_calls_ocr_async(self):
        """_ocr_sync must reference the now-defined _ocr_async."""
        from agent.action_dispatcher import ActionDispatcher

        src = inspect.getsource(ActionDispatcher._ocr_sync)
        assert "_ocr_async" in src


# ═══════════════════════════════════════════════════════════════
# 3. llm_client keep_alive
# ═══════════════════════════════════════════════════════════════

class TestLLMClientKeepAlive:
    """The fallback LLM client must use the configured OLLAMA_KEEP_ALIVE."""

    def test_chat_uses_configured_keep_alive(self):
        from ai.llm_client import LLMClient
        from config.settings import settings

        src = inspect.getsource(LLMClient.chat)
        # Must reference the configured setting, not a hard-coded 0
        assert "settings.OLLAMA_KEEP_ALIVE" in src
        assert '"keep_alive": 0' not in src
        assert "'keep_alive': 0" not in src

    def test_keep_alive_matches_production(self):
        """The fallback client must use the SAME keep_alive as streaming_llm."""
        from ai.llm_client import LLMClient
        from agent.streaming_llm import StreamingLLM

        llm_src = inspect.getsource(LLMClient.chat)
        stream_src = inspect.getsource(StreamingLLM.generate)
        assert "settings.OLLAMA_KEEP_ALIVE" in llm_src
        assert "settings.OLLAMA_KEEP_ALIVE" in stream_src


# ═══════════════════════════════════════════════════════════════
# 4. Dead provider callbacks removed
# ═══════════════════════════════════════════════════════════════

class TestProviderCallbacksRemoved:
    """The engine's dead provider callbacks must be removed — the Brain
    uses direct paths (perception_pipeline, search_service, learning_engine)."""

    def test_setters_removed(self):
        from core.conversation_engine import ConversationEngine

        assert not hasattr(ConversationEngine, "set_vision_context")
        assert not hasattr(ConversationEngine, "set_search_provider")
        assert not hasattr(ConversationEngine, "set_learning_context")

    def test_engine_has_no_provider_state(self):
        from core.conversation_engine import ConversationEngine

        engine = ConversationEngine()
        assert not hasattr(engine, "_vision_context_fn")
        assert not hasattr(engine, "_search_provider_fn")
        assert not hasattr(engine, "_learning_context_fn")

    def test_diego_no_longer_wires_dead_callbacks(self):
        import Diego
        src = inspect.getsource(Diego.run_Diego)
        # No ACTIVE call sites (comments may mention the removed methods
        # for documentation, but there must be no `conversation_engine.` calls).
        assert "conversation_engine.set_vision_context(" not in src
        assert "conversation_engine.set_search_provider(" not in src
        assert "conversation_engine.set_learning_context(" not in src
