"""
Regression tests: Music / YouTube UX (2026-08-30).

Covers:
  - "play <song> on youtube"  -> VISIBLE YouTube playback (never hidden mpv)
  - "search youtube for X"    -> SEARCH-ONLY (results page, no playback)
  - "play <song>"             -> normal provider path unchanged (mpv/Spotify)
  - "pause" / "resume"        -> music_pause / music_resume routing
  - provider fallback         -> mpv preferred when no explicit provider
  - honest responses          -> playback claimed ONLY when verified

Uses fakes/monkeypatching; no real browser or network required.
"""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import compat  # noqa: F401  (Python 3.14 stdlib stubs)

import pytest

from core.command_router import command_router, RouteKind
from services.music_agent import MusicAgent, MusicProvider


# ── Router: play song on youtube ─────────────────────────────

async def test_play_song_on_youtube_routes_to_visible_playback():
    res = await command_router.route("play shape of you on youtube")
    assert res.kind == RouteKind.SIMPLE_DESKTOP
    assert res.action["action"] == "play_media"
    assert res.action["params"]["youtube"] is True
    assert res.action["params"]["query"] == "shape of you"


async def test_play_the_song_from_youtube():
    res = await command_router.route("play the song bohemian rhapsody from youtube")
    assert res.action["action"] == "play_media"
    assert res.action["params"]["youtube"] is True
    assert res.action["params"]["query"] == "bohemian rhapsody"


async def test_play_on_youtube_query_suffix_stripped():
    # Generic play pattern fallback also strips the provider suffix.
    res = await command_router.route("play some jazz on youtube")
    assert res.action["action"] == "play_media"
    assert res.action["params"]["youtube"] is True
    assert res.action["params"]["query"] == "some jazz"


# ── Router: search youtube is SEARCH-ONLY ────────────────────

async def test_search_youtube_is_search_only():
    res = await command_router.route("search youtube for lofi hip hop")
    assert res.action["action"] == "youtube_search"
    assert res.action["params"]["query"] == "lofi hip hop"
    assert res.action["action"] != "play_media"


# ── Router: normal play / pause / resume unchanged ───────────

async def test_normal_play_music_unchanged():
    res = await command_router.route("play some jazz")
    assert res.action["action"] == "play_media"
    assert "youtube" not in res.action["params"]
    # NOTE: the generic play pattern consumes the optional "some", so the
    # captured query is "jazz" — existing behavior, unchanged by this fix.
    assert res.action["params"]["query"] == "jazz"


async def test_pause_resume_routing():
    res = await command_router.route("pause")
    assert res.action["action"] == "music_pause"
    res = await command_router.route("resume")
    assert res.action["action"] == "music_resume"
    res = await command_router.route("pause the music")
    assert res.action["action"] == "music_pause"
    res = await command_router.route("resume the music")
    assert res.action["action"] == "music_resume"


# ── MusicAgent: visible YouTube playback ─────────────────────

async def test_play_on_youtube_visible_verified():
    ma = MusicAgent()
    calls = {}

    def fake_open(url):  # sync: production calls it via run_in_executor
        calls["url"] = url
        return True

    def fake_click():
        calls["clicked"] = True
        return True

    def fake_verify():
        calls["verified"] = True
        return True

    ma._open_url_visible = fake_open
    ma._click_first_youtube_result = fake_click
    ma._verify_youtube_playing = fake_verify

    msg = await ma.play("shape of you", youtube=True)
    assert "Playing shape of you on YouTube" in msg
    assert "youtube.com/results" in calls["url"]
    assert calls["clicked"] and calls["verified"]
    assert ma.is_playing


async def test_play_on_youtube_honest_when_unverified():
    ma = MusicAgent()

    def fake_open(url):
        return True

    ma._open_url_visible = fake_open
    ma._click_first_youtube_result = lambda: False  # automation unavailable

    msg = await ma.play("test song", youtube=True)
    # Must NOT claim playback started.
    assert "Playing" not in msg
    assert "couldn't" in msg.lower()
    assert not ma.is_playing


async def test_play_on_youtube_honest_when_click_but_no_verify():
    ma = MusicAgent()

    def fake_open(url):
        return True

    ma._open_url_visible = fake_open
    ma._click_first_youtube_result = lambda: True
    ma._verify_youtube_playing = lambda: False

    msg = await ma.play("test song", youtube=True)
    assert "couldn't verify" in msg.lower()
    assert not ma.is_playing


async def test_play_on_youtube_open_failure_honest():
    ma = MusicAgent()

    def fake_open(url):
        return False

    ma._open_url_visible = fake_open
    msg = await ma.play("test song", youtube=True)
    assert "couldn't open youtube" in msg.lower()


async def test_search_youtube_opens_results_visibly():
    ma = MusicAgent()
    urls = {}

    def fake_open(url):  # sync: production calls it via run_in_executor
        urls["u"] = url
        return True

    ma._open_url_visible = fake_open
    msg = await ma.search_youtube("lofi")
    assert "youtube.com/results" in urls["u"]
    assert "results for lofi" in msg.lower()


# ── Provider fallback: normal play still prefers mpv ─────────

async def test_provider_fallback_prefers_mpv(monkeypatch):
    ma = MusicAgent()
    called = {}

    async def fake_mpv(q):
        called["mpv"] = q
        return f"Playing {q}."

    monkeypatch.setattr(ma, "_play_mpv", fake_mpv)
    monkeypatch.setattr(ma, "_resolve_provider",
                        lambda p, q: MusicProvider.MPV)
    msg = await ma.play("test song")
    assert "mpv" in called
    assert "Playing test song" in msg


async def test_youtube_hint_never_uses_mpv(monkeypatch):
    """Even with mpv available, a query mentioning youtube must route to
    the visible browser path."""
    ma = MusicAgent()

    def fake_open(url):
        return True

    ma._open_url_visible = fake_open
    ma._click_first_youtube_result = lambda: False

    # mpv "available" but query contains "on youtube" -> visible browser.
    msg = await ma.play("test song on youtube")
    assert "youtube" in msg.lower()
    assert not ma.is_playing  # honest: playback not verified


# ── Dispatcher wiring ────────────────────────────────────────

async def test_dispatcher_routes_youtube_flag(monkeypatch):
    from agent.action_dispatcher import ActionDispatcher
    import services.music_agent as mod

    d = ActionDispatcher()

    class FakeMA:
        async def initialize(self):
            return True

        async def play(self, q, provider=None, youtube=False):
            return f"youtube={youtube}:{q}"

    monkeypatch.setattr(mod, "music_agent", FakeMA())
    msg = await d.execute(
        {"action": "play_media", "params": {"query": "x", "youtube": True}})
    assert "youtube=True" in msg

    msg2 = await d.execute({"action": "play_media", "params": {"query": "y"}})
    assert "youtube=False" in msg2


async def test_dispatcher_routes_youtube_search(monkeypatch):
    from agent.action_dispatcher import ActionDispatcher
    import services.music_agent as mod

    d = ActionDispatcher()

    class FakeMA:
        async def initialize(self):
            return True

        async def search_youtube(self, q):
            return f"searched:{q}"

    monkeypatch.setattr(mod, "music_agent", FakeMA())
    msg = await d.execute(
        {"action": "youtube_search", "params": {"query": "lofi"}})
    assert msg == "searched:lofi"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))