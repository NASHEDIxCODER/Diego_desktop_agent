"""
Regression tests for the 2026-08-30 desktop-agent capability hardening.

Covers:
  D1 — Desktop awareness routing: "what apps are running", "what windows
       are open", "switch window", "switch to Firefox/VS Code",
       "close this window", "minimize this", "maximize this".
  D2 — System control routing: Wi-Fi / Bluetooth on/off.
  D3 — Vision intent: "what is on my screen?", "read this error",
       "what button should I click?" route to the VISION path.
  D4 — Web research routing: "search for X", "search GitHub for X",
       "search the web for X", "search X and open the best result".
  D5 — Memory-override guard: stale semantic memory must never answer
       explicit current desktop / screen / web requests.
  D6 — Dispatcher: list_windows / focus_app / radio toggles / web_search
       produce honest, verified results (mocked OS layer).
  D7 — LLM performance: configurable keep_alive, fail-safe warm-up,
       live-request guard.
  D8 — Brain: informational action results are spoken verbatim.

Run with:
    python -m pytest tests/test_capability_hardening.py -v
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stubs)

from core.command_router import command_router, RouteKind
from core.decision_engine import decision_engine, DecisionPath
from agent.action_dispatcher import ActionDispatcher
from agent.brain import AgentBrain


def run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════
# D1 — Desktop awareness routing
# ═══════════════════════════════════════════════════════════════

class TestDesktopAwarenessRouting:

    def _route(self, text):
        return run(command_router.route(text))

    def test_what_apps_are_running(self):
        r = self._route("what apps are running")
        assert r.kind == RouteKind.SIMPLE_DESKTOP
        assert r.action["action"] == "list_windows"

    def test_what_windows_are_open(self):
        r = self._route("what windows are open")
        assert r.kind == RouteKind.SIMPLE_DESKTOP
        assert r.action["action"] == "list_windows"

    def test_what_is_open(self):
        r = self._route("what is open")
        assert r.action["action"] == "list_windows"

    def test_switch_window(self):
        r = self._route("switch window")
        assert r.action["action"] == "switch_window"

    def test_switch_to_next_window(self):
        r = self._route("switch to the next window")
        assert r.action["action"] == "switch_window"

    def test_switch_to_firefox(self):
        r = self._route("switch to firefox")
        assert r.action["action"] == "focus_app"
        assert r.action["params"]["app"] == "firefox"

    def test_switch_to_vscode(self):
        # The normalizer maps "vs code" → "vscode"; the router maps
        # "vscode" → the "code" binary.
        r = self._route("switch to vscode")
        assert r.action["action"] == "focus_app"
        assert r.action["params"]["app"] == "code"

    def test_normalizer_then_router_switch_to_vs_code(self):
        """Full production path: normalize('switch to VS Code') must still
        route to focus_app with the code binary."""
        from nlp.command_normalizer import command_normalizer
        normalized = command_normalizer.normalize("switch to VS Code")
        r = self._route(normalized)
        assert r.action["action"] == "focus_app"
        assert r.action["params"]["app"] == "code"

    def test_close_this_window(self):
        r = self._route("close this window")
        assert r.action["action"] == "close_window"

    def test_close_window(self):
        r = self._route("close window")
        assert r.action["action"] == "close_window"

    def test_minimize_this(self):
        r = self._route("minimize this")
        assert r.action["action"] == "minimize_window"

    def test_maximize_this(self):
        r = self._route("maximize this")
        assert r.action["action"] == "maximize_window"

    def test_minimize_window_still_works(self):
        r = self._route("minimize the window")
        assert r.action["action"] == "minimize_window"

    def test_change_volume_not_focus_app(self):
        """The generic switch pattern must not swallow volume control.
        'change the volume' must either route to a volume action or fall
        through to the LLM — never to focus_app."""
        r = self._route("change the volume")
        assert r.action is None or r.action["action"] != "focus_app"


# ═══════════════════════════════════════════════════════════════
# D2 — System control routing (Wi-Fi / Bluetooth)
# ═══════════════════════════════════════════════════════════════

class TestRadioRouting:

    def _route(self, text):
        return run(command_router.route(text))

    def test_wifi_off(self):
        r = self._route("wifi off")
        assert r.action["action"] == "wifi_off"

    def test_wifi_on(self):
        r = self._route("wifi on")
        assert r.action["action"] == "wifi_on"

    def test_turn_off_wifi(self):
        r = self._route("turn off wifi")
        assert r.action["action"] == "wifi_off"

    def test_turn_on_wifi(self):
        r = self._route("turn on wifi")
        assert r.action["action"] == "wifi_on"

    def test_bluetooth_off(self):
        r = self._route("bluetooth off")
        assert r.action["action"] == "bluetooth_off"

    def test_turn_on_bluetooth(self):
        r = self._route("turn on bluetooth")
        assert r.action["action"] == "bluetooth_on"


class TestNormalizerProductionPath:
    """The Brain normalizes BEFORE routing — the full path must hold."""

    def _full_route(self, text):
        from nlp.command_normalizer import command_normalizer
        normalized = command_normalizer.normalize(text)
        return run(command_router.route(normalized)), normalized

    def test_turn_off_wifi_full_path(self):
        """'turn off wifi' must NOT be corrupted into 'shutdown wifi' by
        the generic turn-off verb alias."""
        r, normalized = self._full_route("turn off wifi")
        assert normalized == "wifi off"
        assert r.action["action"] == "wifi_off"

    def test_turn_on_wifi_full_path(self):
        r, normalized = self._full_route("turn on wifi")
        assert normalized == "wifi on"
        assert r.action["action"] == "wifi_on"

    def test_turn_off_bluetooth_full_path(self):
        r, normalized = self._full_route("turn off bluetooth")
        assert r.action["action"] == "bluetooth_off"

    def test_switch_to_firefox_full_path(self):
        """'switch to Firefox' must focus the existing window, never
        spawn a fresh instance via the old switch-to→open alias."""
        r, normalized = self._full_route("switch to Firefox")
        assert normalized.startswith("switch to")
        assert r.action["action"] == "focus_app"
        assert r.action["params"]["app"] == "firefox"

    def test_turn_off_music_pauses(self):
        r, normalized = self._full_route("turn off the music")
        assert normalized == "pause music"
        assert r.action["action"] == "music_pause"

    def test_shutdown_still_works(self):
        r, normalized = self._full_route("shut down the computer")
        assert r.action["action"] == "shutdown"


# ═══════════════════════════════════════════════════════════════
# D3 — Vision intent detection
# ═══════════════════════════════════════════════════════════════

class TestVisionIntent:

    def _decide(self, text):
        decision_engine._intent_cache.clear()
        return run(decision_engine.decide(text))

    def test_what_is_on_my_screen(self):
        d = self._decide("what is on my screen?")
        assert d.path == DecisionPath.VISION
        assert d.action["action"] == "read_screen"
        assert "screen" in d.action["params"]["question"]

    def test_read_this_error(self):
        d = self._decide("read this error")
        assert d.path == DecisionPath.VISION

    def test_what_button_should_i_click(self):
        d = self._decide("what button should I click?")
        assert d.path == DecisionPath.VISION

    def test_read_this_page(self):
        d = self._decide("read this page")
        assert d.path == DecisionPath.VISION


# ═══════════════════════════════════════════════════════════════
# D4 — Web research routing
# ═══════════════════════════════════════════════════════════════

class TestWebResearchRouting:

    def _route(self, text):
        return run(command_router.route(text))

    def test_search_for_x(self):
        r = self._route("search for python asyncio tutorial")
        assert r.action["action"] == "web_search"
        assert r.action["params"]["query"] == "python asyncio tutorial"

    def test_search_github(self):
        r = self._route("search github for python websocket examples")
        assert r.action["action"] == "web_search"
        assert r.action["params"]["site"] == "github.com"
        assert r.action["params"]["query"] == "python websocket examples"

    def test_search_the_web(self):
        r = self._route("search the web for current information about rust")
        assert r.action["action"] == "web_search"
        assert "rust" in r.action["params"]["query"]

    def test_search_and_open_best_result(self):
        r = self._route("search rust async book and open the best result")
        assert r.action["action"] == "web_search_open_best"
        assert r.action["params"]["query"] == "rust async book"

    def test_search_youtube_still_search_only(self):
        r = self._route("search youtube for lofi beats")
        assert r.action["action"] == "youtube_search"


# ═══════════════════════════════════════════════════════════════
# D5 — Memory-override guard
# ═══════════════════════════════════════════════════════════════

class TestMemoryOverrideGuard:

    def setup_method(self):
        from agent.conversation_memory import conv_memory
        self.conv_memory = conv_memory
        self._saved_facts = list(conv_memory._facts)
        # Plant a stale fact that superficially overlaps desktop queries
        conv_memory._facts.append("my project is the running app")

    def teardown_method(self):
        self.conv_memory._facts.clear()
        self.conv_memory._facts.extend(self._saved_facts)
        decision_engine._intent_cache.clear()

    def test_desktop_query_not_answered_from_memory(self):
        d = run(decision_engine.decide("what apps are running"))
        assert d.path != DecisionPath.WORKING_MEMORY
        assert d.action is not None
        assert d.action["action"] == "list_windows"

    def test_search_request_not_answered_from_memory(self):
        d = run(decision_engine.decide("search the web for rust"))
        assert d.path != DecisionPath.WORKING_MEMORY

    def test_screen_request_not_answered_from_memory(self):
        d = run(decision_engine.decide("what is on my screen"))
        assert d.path == DecisionPath.VISION

    def test_live_desktop_query_helper(self):
        assert decision_engine._is_live_desktop_query("what apps are running")
        assert decision_engine._is_live_desktop_query("what windows are open")
        assert not decision_engine._is_live_desktop_query("what was my project called")

    def test_memory_still_answers_recall_questions(self):
        self.conv_memory._facts.clear()
        self.conv_memory._facts.append("my project is called Diego")
        d = run(decision_engine.decide("what was my project called"))
        # Either memory or LLM — but never a desktop action
        assert d.action is None or d.action.get("action") != "list_windows"


# ═══════════════════════════════════════════════════════════════
# D6 — Dispatcher: honest, verified results
# ═══════════════════════════════════════════════════════════════

class TestDispatcherDesktopAwareness:

    def setup_method(self):
        self.d = ActionDispatcher()

    def test_list_windows_wmctrl(self):
        fake_out = MagicMock(returncode=0, stdout=(
            "0x01  0  100  Firefox — Mozilla Firefox\n"
            "0x02  0  100  Visual Studio Code — Diego\n"
        ))
        with patch("shutil.which", return_value="/usr/bin/wmctrl"), \
             patch("subprocess.run", return_value=fake_out):
            result = self.d._list_windows()
        assert "Firefox" in result
        assert "Visual Studio Code" in result
        assert "2 windows" in result

    def test_list_windows_no_tools_honest_failure(self):
        with patch("shutil.which", return_value=None):
            result = self.d._list_windows()
        assert "couldn't detect" in result.lower()

    def test_focus_app_success(self):
        wm_out = MagicMock(returncode=0,
                           stdout="0x01  0  100  Mozilla Firefox\n")
        with patch("shutil.which", return_value="/usr/bin/wmctrl"), \
             patch("subprocess.run", return_value=wm_out), \
             patch.object(ActionDispatcher, "_active_window_title",
                          return_value="Mozilla Firefox"):
            result = self.d._focus_app("firefox")
        assert "Switched to firefox" in result

    def test_focus_app_not_found_honest(self):
        wm_out = MagicMock(returncode=0, stdout="")
        with patch("shutil.which", return_value="/usr/bin/wmctrl"), \
             patch("subprocess.run", return_value=wm_out):
            result = self.d._focus_app("nosuchapp")
        assert "couldn't find" in result.lower()

    def test_radio_on_verified(self):
        with patch("shutil.which", return_value="/usr/bin/nmcli"), \
             patch("subprocess.run"), \
             patch.object(ActionDispatcher, "_radio_state", return_value=True):
            result = self.d._radio("wifi", True)
        assert "Wi-Fi is on" in result

    def test_radio_off_verified(self):
        with patch("shutil.which", return_value="/usr/bin/nmcli"), \
             patch("subprocess.run"), \
             patch.object(ActionDispatcher, "_radio_state", return_value=False):
            result = self.d._radio("bluetooth", False)
        assert "Bluetooth is off" in result

    def test_radio_state_mismatch_reported_honestly(self):
        with patch("shutil.which", return_value="/usr/bin/nmcli"), \
             patch("subprocess.run"), \
             patch.object(ActionDispatcher, "_radio_state", return_value=True):
            result = self.d._radio("wifi", False)
        assert "still on" in result

    def test_radio_state_unknown(self):
        with patch("shutil.which", return_value=None):
            result = self.d._radio("wifi", True)
        assert "unavailable" in result.lower()


class TestDispatcherWebResearch:

    def setup_method(self):
        self.d = ActionDispatcher()

    def _make_result(self, pages=True):
        from services.search_service import SearchResult, SearchHit, FetchedPage
        hits = [SearchHit(title="Rust Book", url="https://example.com/rust",
                          snippet="Rust is a systems language.", source="duckduckgo")]
        pages_list = []
        if pages:
            pages_list = [FetchedPage(
                url="https://example.com/rust", title="Rust Book",
                content="Rust is a systems programming language. "
                        "It focuses on safety and speed. It is memory safe.",
                content_length=100, extractor="trafilatura")]
        r = SearchResult(query="rust", hits=hits, pages=pages_list)
        return r

    def test_web_search_answers_from_real_content(self):
        async def scenario():
            fake_svc = MagicMock()
            fake_svc.is_ready = True
            fake_svc.search = AsyncMock(return_value=self._make_result())
            with patch("services.search_service.search_service", fake_svc):
                return await self.d._web_search({"query": "rust"})

        result = run(scenario())
        assert "rust" in result.lower()
        assert "systems programming language" in result

    def test_web_search_no_results_is_honest(self):
        from services.search_service import SearchResult

        async def scenario():
            fake_svc = MagicMock()
            fake_svc.is_ready = True
            fake_svc.search = AsyncMock(
                return_value=SearchResult(query="x", error="no search results"))
            with patch("services.search_service.search_service", fake_svc):
                return await self.d._web_search({"query": "x"})

        result = run(scenario())
        assert "couldn't get any results" in result

    def test_web_search_github_scope(self):
        async def scenario():
            fake_svc = MagicMock()
            fake_svc.is_ready = True
            captured = {}

            async def search(q, max_results=5, **kw):
                captured["q"] = q
                return self._make_result()

            fake_svc.search = search
            with patch("services.search_service.search_service", fake_svc):
                result = await self.d._web_search(
                    {"query": "websocket examples", "site": "github.com"})
            return result, captured["q"]

        result, q = run(scenario())
        assert "site:github.com" in q
        assert "GitHub" in result

    def test_open_best_result_verifies_browser(self):
        async def scenario():
            fake_svc = MagicMock()
            fake_svc.is_ready = True
            fake_svc.search = AsyncMock(return_value=self._make_result())
            with patch("services.search_service.search_service", fake_svc), \
                 patch("shutil.which", return_value="/usr/bin/pgrep"), \
                 patch("subprocess.run",
                       return_value=MagicMock(returncode=0, stdout="123")):
                return await self.d._web_search_open_best({"query": "rust"})

        result = run(scenario())
        assert "opened" in result.lower()
        assert "Rust Book" in result

    def test_open_best_result_browser_unverified_is_honest(self):
        async def scenario():
            fake_svc = MagicMock()
            fake_svc.is_ready = True
            fake_svc.search = AsyncMock(return_value=self._make_result())
            with patch("services.search_service.search_service", fake_svc), \
                 patch("shutil.which", return_value="/usr/bin/pgrep"), \
                 patch("subprocess.run",
                       return_value=MagicMock(returncode=1, stdout="")):
                return await self.d._web_search_open_best({"query": "rust"})

        result = run(scenario())
        assert "couldn't verify" in result


# ═══════════════════════════════════════════════════════════════
# D7 — LLM performance: keep_alive, warm-up, live-request guard
# ═══════════════════════════════════════════════════════════════

class TestLLMPerformance:

    def test_keep_alive_configurable_not_zero(self):
        from config.settings import settings
        from agent.streaming_llm import streaming_llm
        # The default must NOT unload the model after every request.
        assert settings.OLLAMA_KEEP_ALIVE != "0"
        assert settings.OLLAMA_KEEP_ALIVE == "10m"

    def test_warm_up_fail_safe_when_ollama_down(self):
        from agent.streaming_llm import streaming_llm

        async def scenario():
            streaming_llm._checked = False
            streaming_llm._warmed = False
            with patch("agent.streaming_llm.httpx.AsyncClient",
                       side_effect=Exception("connection refused")):
                return await streaming_llm.warm_up()

        # Must return False — never raise.
        assert run(scenario()) is False

    def test_warm_up_success_marks_warmed(self):
        from agent.streaming_llm import streaming_llm

        async def scenario():
            streaming_llm._checked = False
            streaming_llm._warmed = False
            streaming_llm._available = True
            streaming_llm._model = "test-model"

            fake_client = MagicMock()
            fake_resp = MagicMock()
            fake_resp.raise_for_status = MagicMock()
            fake_client.__aenter__ = AsyncMock(return_value=fake_client)
            fake_client.__aexit__ = AsyncMock(return_value=False)
            fake_client.post = AsyncMock(return_value=fake_resp)

            with patch("agent.streaming_llm.httpx.AsyncClient",
                       return_value=fake_client):
                ok = await streaming_llm.warm_up()
            return ok

        assert run(scenario()) is True

    def test_live_request_guard(self):
        from agent.streaming_llm import StreamingLLM
        assert StreamingLLM._is_live_request("what apps are running")
        assert StreamingLLM._is_live_request("search the web for rust")
        assert StreamingLLM._is_live_request("what is on my screen")
        assert not StreamingLLM._is_live_request("what was my project called")
        assert not StreamingLLM._is_live_request("who is my brother")

    def test_warmup_settings_exist(self):
        from config.settings import settings
        assert hasattr(settings, "LLM_WARMUP_ENABLED")
        assert hasattr(settings, "LLM_WARMUP_TIMEOUT_S")
        assert hasattr(settings, "LLM_WARMUP_VISION")


# ═══════════════════════════════════════════════════════════════
# D8 — Brain: informational results spoken verbatim
# ═══════════════════════════════════════════════════════════════

class TestBrainInformationalResponses:

    def test_result_as_response_actions_include_new_actions(self):
        assert "web_search" in AgentBrain._RESULT_AS_RESPONSE_ACTIONS
        assert "web_search_open_best" in AgentBrain._RESULT_AS_RESPONSE_ACTIONS
        assert "list_windows" in AgentBrain._RESULT_AS_RESPONSE_ACTIONS
        assert "read_screen" in AgentBrain._RESULT_AS_RESPONSE_ACTIONS

    def test_dispatch_and_verify_returns_result_string(self):
        brain = AgentBrain()

        async def fake_execute(action):
            return "You have 3 windows open: Firefox; Code; Terminal."

        brain._dispatcher = MagicMock()
        brain._dispatcher.execute = fake_execute
        brain._verifier = None
        brain._learning = None

        ok, result = run(brain._dispatch_and_verify(
            {"action": "list_windows", "params": {}}))
        assert ok is True
        assert "3 windows" in result

    def test_search_command_speaks_real_results(self):
        """End-to-end routing: 'search for rust' → web_search action →
        the dispatcher's real answer becomes the spoken response."""
        brain = AgentBrain()
        brain._initialized = True
        brain._decision_engine = decision_engine
        brain._perception = None
        brain._verifier = None
        brain._learning = None

        async def fake_execute(action):
            assert action["action"] == "web_search"
            return "Here's what I found for rust: Rust is a systems language."

        brain._dispatcher = MagicMock()
        brain._dispatcher.execute = fake_execute

        result = run(brain.process_command("search for rust"))
        assert result.actions_executed == 1
        assert result.actions_failed == 0
        assert "Rust is a systems language" in result.response
        assert not result.used_llm


# ═══════════════════════════════════════════════════════════════
# Regression: existing behaviour must not break
# ═══════════════════════════════════════════════════════════════

class TestExistingBehaviourPreserved:

    def _route(self, text):
        return run(command_router.route(text))

    def test_open_firefox(self):
        r = self._route("open firefox")
        assert r.action["action"] == "desktop_open"
        assert r.action["params"]["app"] == "firefox"

    def test_volume_up(self):
        r = self._route("volume up")
        assert r.action["action"] == "volume_up"

    def test_play_music(self):
        r = self._route("play lofi beats")
        assert r.action["action"] == "play_media"
        assert r.action["params"]["query"] == "lofi beats"

    def test_greeting_is_conversation(self):
        r = self._route("hello")
        assert r.kind == RouteKind.CONVERSATION

    def test_time_query(self):
        r = self._route("what time is it")
        assert r.action["action"] == "get_time"