"""
MusicAgent — Unified music control for Diego.

Supports multiple backends:
    - YouTube Music (via browser automation)
    - Spotify (via spotify-tui / spotifyd or web)
    - MPV (local media player)
    - VLC (local media player)
    - Browser (YouTube web fallback)
    - Local Music (MPV on local paths)

Learns:
    - Favourite provider
    - Favourite artists
    - Favourite playlists

Allows conversation to continue while music is playing.

Usage:
    from services.music_agent import music_agent

    await music_agent.play("coding music")
    await music_agent.pause()
    await music_agent.next()
    await music_agent.set_volume(70)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import time
from urllib.parse import quote
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────

class MusicProvider(str, Enum):
    """Supported music providers."""
    YOUTUBE_MUSIC = "youtube_music"
    SPOTIFY = "spotify"
    MPV = "mpv"
    VLC = "vlc"
    BROWSER = "browser"
    LOCAL = "local"


class PlaybackState(str, Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class TrackInfo:
    """Information about the currently playing track."""
    title: str = ""
    artist: str = ""
    album: str = ""
    provider: MusicProvider = MusicProvider.BROWSER
    state: PlaybackState = PlaybackState.STOPPED
    volume: int = 50
    muted: bool = False
    position_ms: int = 0
    duration_ms: int = 0


# ── Predefined playlist mappings ──────────────────────────────

CODING_MUSIC_QUERIES = [
    "lofi hip hop coding",
    "coding music lofi",
    "study music lofi",
    "chill coding music",
    "instrumental coding music",
]

RELAXING_MUSIC_QUERIES = [
    "relaxing music",
    "ambient music",
    "calm piano",
    "chill music",
    "meditation music",
]


class MusicAgent:
    """
    Unified music playback and control agent.

    Supports multiple backends with automatic provider selection.
    Learns user preferences over time.
    """

    def __init__(self):
        self._current: TrackInfo = TrackInfo()
        self._preferred_provider: Optional[MusicProvider] = None
        self._favorite_artists: List[str] = []
        self._favorite_playlists: List[str] = []
        self._favorite_genres: List[str] = []
        self._mpv_process: Optional[subprocess.Popen] = None
        self._initialized = False

    # ── Initialization ─────────────────────────────────────────

    async def initialize(self) -> bool:
        """Detect available providers."""
        self._initialized = True

        available = []
        if shutil.which("mpv"):
            available.append("mpv")
        if shutil.which("vlc"):
            available.append("vlc")
        if shutil.which("spotify") or shutil.which("spotifyd"):
            available.append("spotify")

        logger.info("[Music] Providers available: %s",
                     ", ".join(available) if available else "browser-only")

        # Load learned preferences
        self._load_preferences()
        return True

    # ── Main API ───────────────────────────────────────────────

    @staticmethod
    def _playerctl(*args: str) -> bool:
        """Run a playerctl command; True ONLY if playerctl exists AND exits 0.

        CRITICAL FIX (audit B3): previous implementations fired
        `playerctl <cmd>` and returned a success string unconditionally,
        regardless of the exit status. With no MPRIS player running,
        playerctl exits non-zero — yet Diego reported "Music paused." /
        "Resumed." and verification trusted the string, producing false
        success. Now the exit status is authoritative.
        """
        if not shutil.which("playerctl"):
            return False
        try:
            result = subprocess.run(
                ["playerctl", *args],
                capture_output=True, timeout=2,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def play(self, query: str, provider: Optional[str] = None,
                   youtube: bool = False) -> str:
        """
        Play music matching the query.

        Args:
            query: What to play (artist, song, playlist, genre, etc.)
            provider: Optional specific provider, else auto-detect.
            youtube: True for an EXPLICIT "play X on youtube" request —
                forces the VISIBLE browser/YouTube flow (never hidden mpv).

        Returns:
            Human-readable status message.
        """
        query_lower = query.lower().strip()

        # ── Explicit YouTube request (2026-08-30 UX fix) ──
        # "play <song> on/from youtube" must open the browser VISIBLY,
        # search, select a result, start playback and verify it. It must
        # NEVER fall through to hidden mpv playback.
        if youtube or "youtube" in query_lower:
            clean = re.sub(r"\s+(?:on|from|in)\s+youtube\s*$", "",
                           query, flags=re.IGNORECASE).strip()
            return await self._play_youtube_visible(clean or query)

        # ── Handle special queries ─────────────────────────
        if any(w in query_lower for w in ("coding music", "coding playlist",
                                            "study music", "lofi", "lo-fi")):
            import random
            query = random.choice(CODING_MUSIC_QUERIES)

        if any(w in query_lower for w in ("relaxing music", "relaxing playlist",
                                            "ambient", "calm music", "meditation")):
            import random
            query = random.choice(RELAXING_MUSIC_QUERIES)

        # ── Handle "resume" ────────────────────────────────
        if query_lower in ("resume", "resume previous", "continue playing"):
            return await self.resume()

        # ── Auto-select provider ───────────────────────────
        selected = self._resolve_provider(provider, query_lower)

        # ── Execute ────────────────────────────────────────
        if selected == MusicProvider.MPV:
            result = await self._play_mpv(query)
        elif selected == MusicProvider.BROWSER or selected == MusicProvider.YOUTUBE_MUSIC:
            result = await self._play_browser(query)
        elif selected == MusicProvider.SPOTIFY:
            result = await self._play_spotify(query)
        elif selected == MusicProvider.LOCAL:
            result = await self._play_local(query)
        else:
            result = await self._play_browser(query)

        if "playing" in result.lower() or "started" in result.lower():
            self._current.state = PlaybackState.PLAYING
            self._current.title = query

        # ── Learn ──────────────────────────────────────────
        self._record_usage(query, selected, success=True)
        self._preferred_provider = selected

        return result

    async def pause(self) -> str:
        """Pause current playback."""
        if self._current.provider == MusicProvider.MPV and self._mpv_process:
            try:
                self._mpv_process.stdin.write(b'cycle pause\n')
                self._mpv_process.stdin.flush()
                self._current.state = PlaybackState.PAUSED
                return "Music paused."
            except Exception:
                pass

        # playerctl (controls mpv, vlc, spotify, any MPRIS player).
        # CRITICAL FIX (audit B3): only report success when playerctl
        # actually exited 0 (a player responded).
        if self._playerctl("pause"):
            self._current.state = PlaybackState.PAUSED
            return "Music paused."

        return "Couldn't pause — no player responded."

    async def resume(self) -> str:
        """Resume playback."""
        if self._current.provider == MusicProvider.MPV and self._mpv_process:
            try:
                self._mpv_process.stdin.write(b'cycle pause\n')
                self._mpv_process.stdin.flush()
                self._current.state = PlaybackState.PLAYING
                return "Resumed."
            except Exception:
                pass

        # CRITICAL FIX (audit B3): "Resumed." only when playerctl exited 0.
        if self._playerctl("play"):
            self._current.state = PlaybackState.PLAYING
            return "Resumed."

        return "Nothing to resume."

    async def next(self) -> str:
        """Skip to next track."""
        if self._current.provider == MusicProvider.MPV and self._mpv_process:
            try:
                self._mpv_process.stdin.write(b'playlist-next\n')
                self._mpv_process.stdin.flush()
                return "Skipped to next."
            except Exception:
                pass

        # CRITICAL FIX (audit B3): success only when playerctl exited 0.
        if self._playerctl("next"):
            return "Next track."

        return "Couldn't skip — no player responded."

    async def previous(self) -> str:
        """Go to previous track."""
        if self._current.provider == MusicProvider.MPV and self._mpv_process:
            try:
                self._mpv_process.stdin.write(b'playlist-prev\n')
                self._mpv_process.stdin.flush()
                return "Previous track."
            except Exception:
                pass

        # CRITICAL FIX (audit B3): success only when playerctl exited 0.
        if self._playerctl("previous"):
            return "Previous track."

        return "Couldn't go back — no player responded."

    async def stop(self) -> str:
        """Stop playback entirely."""
        if self._mpv_process:
            try:
                self._mpv_process.terminate()
                self._mpv_process.wait(timeout=3)
            except Exception:
                try:
                    self._mpv_process.kill()
                except Exception:
                    pass
            self._mpv_process = None
            self._current.state = PlaybackState.STOPPED
            return "Music stopped."

        # CRITICAL FIX (audit B3): success only when playerctl exited 0.
        if self._playerctl("stop"):
            self._current.state = PlaybackState.STOPPED
            return "Music stopped."

        return "Nothing to stop."

    async def set_volume(self, percent: int) -> str:
        """Set volume for the active music player."""
        percent = max(0, min(100, percent))

        # Try playerctl first — success only when it exits 0 (audit B3)
        if self._playerctl("volume", str(percent / 100)):
            self._current.volume = percent
            return f"Volume: {percent}%."

        # Fallback: system volume
        if shutil.which("pactl"):
            subprocess.Popen(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._current.volume = percent
            return f"Volume: {percent}%."

        return "Volume control unavailable."

    async def mute(self) -> str:
        """Toggle mute."""
        if shutil.which("pactl"):
            subprocess.Popen(
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._current.muted = not self._current.muted
            return "Muted." if self._current.muted else "Unmuted."
        return "Mute unavailable."

    async def shuffle(self) -> str:
        """Toggle shuffle mode."""
        # CRITICAL FIX (audit B3): success only when playerctl exited 0.
        if self._playerctl("shuffle", "toggle"):
            return "Shuffle toggled."
        return "Shuffle not available."

    async def repeat(self) -> str:
        """Toggle repeat mode."""
        # CRITICAL FIX (audit B3): success only when playerctl exited 0.
        if self._playerctl("loop", "toggle"):
            return "Repeat toggled."
        return "Repeat not available."

    async def status(self) -> str:
        """Get current playback status."""
        try:
            result = subprocess.run(
                ["playerctl", "metadata", "--format",
                 "{{ title }} by {{ artist }}"],
                capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                info = result.stdout.strip()
                return f"Playing: {info}."
        except Exception:
            pass

        if self._current.state == PlaybackState.PLAYING:
            return f"Playing {self._current.title or 'music'}."
        elif self._current.state == PlaybackState.PAUSED:
            return "Music is paused."
        return "Nothing is playing."

    # ── Provider-specific implementations ──────────────────────

    async def _play_mpv(self, query: str) -> str:
        """Play via MPV (local player with YouTube-DL support)."""
        # Stop any existing MPV
        if self._mpv_process:
            await self.stop()

        try:
            self._mpv_process = subprocess.Popen(
                ["mpv", "--no-video",
                 "--ytdl-format=bestaudio",
                 "--input-ipc-server=/tmp/Diego_mpv_socket",
                 f"ytdl://ytsearch:{query}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._current.provider = MusicProvider.MPV
            self._current.state = PlaybackState.PLAYING
            self._current.title = query
            logger.info("[Music] MPV started: %s", query)
            return f"Playing {query}."
        except FileNotFoundError:
            logger.debug("[Music] MPV not found")
            return await self._play_browser(query)
        except Exception as e:
            logger.warning("[Music] MPV error: %s", e)
            return await self._play_browser(query)

    # ── Visible YouTube playback (2026-08-30 UX fix) ───────────

    async def _play_youtube_visible(self, query: str) -> str:
        """EXPLICIT 'play X on YouTube' — VISIBLE browser playback.

        1. Open the YouTube search results page visibly.
        2. Select/click the first result via browser automation.
        3. Verify the player state where possible (watch URL / MPRIS).
        4. Report HONESTLY — never claim playback started if it could
           not be verified.
        """
        search_url = ("https://www.youtube.com/results?search_query="
                      + quote(query))

        opened = await asyncio.get_event_loop().run_in_executor(
            None, self._open_url_visible, search_url)
        if not opened:
            return f"I couldn't open YouTube to play {query}."

        # Try to select the first result and start playback.
        clicked = await asyncio.get_event_loop().run_in_executor(
            None, self._click_first_youtube_result)
        if clicked:
            verified = await asyncio.get_event_loop().run_in_executor(
                None, self._verify_youtube_playing)
            if verified:
                self._current.provider = MusicProvider.BROWSER
                self._current.state = PlaybackState.PLAYING
                self._current.title = query
                logger.info("[Music] YouTube playback verified: %s", query)
                return f"Playing {query} on YouTube."
            # Clicked but could not verify — be honest.
            logger.info("[Music] YouTube playback NOT verified: %s", query)
            return (f"I opened YouTube and selected a result for {query}, "
                    "but I couldn't verify that playback actually started.")

        # Automation unavailable — results are on screen; do NOT claim playback.
        return (f"I searched YouTube for {query} and the results are open "
                "in your browser, but I couldn't start playback "
                "automatically. Pick a result to play it.")

    async def search_youtube(self, query: str) -> str:
        """SEARCH-ONLY YouTube request — open the results page visibly.

        Never starts playback. Used for 'search youtube for X'.
        """
        url = ("https://www.youtube.com/results?search_query=" + quote(query))
        opened = await asyncio.get_event_loop().run_in_executor(
            None, self._open_url_visible, url)
        if opened:
            return f"Here are the YouTube results for {query}."
        return f"I couldn't open YouTube to search for {query}."

    def _open_url_visible(self, url: str) -> bool:
        """Open a URL in the user's VISIBLE browser. True on success."""
        # Preferred: the shared browser controller (real, visible browser).
        try:
            from agent.executor import agent_executor
            ex = agent_executor
            if not ex.is_available:
                ex.initialize()
            if ex.is_available:
                ok, _msg = ex.browser_navigate(url)
                if ok:
                    return True
        except Exception as e:
            logger.debug("[Music] Browser navigate failed: %s", e)

        # Fallback: default browser via xdg-open / gio (still visible).
        import subprocess
        for opener in ("xdg-open", "gio"):
            exe = shutil.which(opener)
            if exe:
                try:
                    argv = [exe, "open", url] if opener == "gio" else [exe, url]
                    subprocess.Popen(
                        argv,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True)
                    return True
                except Exception:
                    continue
        return False

    def _click_first_youtube_result(self) -> bool:
        """Click the first video result on the open YouTube search page.

        Uses the shared browser controller (visible browser). Returns
        True only if the click succeeded.
        """
        try:
            from agent.browser import browser_controller
            if not browser_controller.is_available:
                return False
            # Wait for results to render, then click the first video link.
            if not browser_controller.wait_for_selector("a#video-title", timeout=10):
                logger.debug("[Music] No YouTube results rendered")
                return False
            if browser_controller.click("a#video-title"):
                browser_controller.wait_for_load()
                return True
            return False
        except Exception as e:
            logger.debug("[Music] Click first result failed: %s", e)
            return False

    def _current_browser_url(self) -> Optional[str]:
        """Current URL of the visible browser tab, or None."""
        try:
            from agent.browser import browser_controller
            return browser_controller.get_current_url()
        except Exception:
            return None

    def _verify_youtube_playing(self) -> bool:
        """Best-effort verification that YouTube playback actually started.

        Evidence used (in order):
          1. The browser tab URL is a YouTube /watch URL.
          2. An MPRIS browser player reports 'Playing' via playerctl.
        """
        url = self._current_browser_url()
        if url and "/watch" in url:
            return True

        if shutil.which("playerctl"):
            try:
                r = subprocess.run(
                    ["playerctl", "--list-all"],
                    capture_output=True, text=True, timeout=2)
                players = [p.strip() for p in r.stdout.split() if p.strip()]
                browser_players = [
                    p for p in players
                    if any(b in p.lower() for b in
                           ("chrome", "chromium", "firefox", "brave",
                            "edge", "epiphany"))
                ]
                for p in browser_players:
                    r2 = subprocess.run(
                        ["playerctl", "-p", p, "status"],
                        capture_output=True, text=True, timeout=2)
                    if r2.returncode == 0 and "Playing" in r2.stdout:
                        return True
            except Exception:
                pass
        return False

    async def _play_browser(self, query: str) -> str:
        """Play via browser (YouTube web) — delegates to the VISIBLE
        playback flow so the response is always honest about whether
        playback actually started."""
        return await self._play_youtube_visible(query)

    async def _play_spotify(self, query: str) -> str:
        """Play via Spotify."""
        # Try spotify CLI
        if shutil.which("spotify"):
            try:
                subprocess.Popen(
                    ["spotify", "play", query],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True)
                self._current.provider = MusicProvider.SPOTIFY
                self._current.state = PlaybackState.PLAYING
                self._current.title = query
                return f"Playing {query} on Spotify."
            except Exception as e:
                logger.debug("[Music] spotify CLI failed: %s", e)

        # Fallback: open Spotify web search
        encoded = query.replace(" ", "%20")
        url = f"https://open.spotify.com/search/{encoded}"
        return await self._play_browser(query)

    async def _play_local(self, query: str) -> str:
        """Play local music files."""
        if not shutil.which("mpv"):
            return "MPV is not installed — can't play local files."

        # Search common music directories
        search_paths = [
            Path.home() / "Music",
            Path.home() / "music",
            Path.home() / "Downloads",
            Path("/media"),
            Path("/mnt"),
        ]

        import glob
        import random

        extensions = (".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".opus")
        found_files: List[Path] = []
        query_lower = query.lower()

        for base in search_paths:
            if not base.exists():
                continue
            for ext in extensions:
                pattern = str(base / "**" / f"*{ext}")
                try:
                    found = [Path(p) for p in glob.glob(pattern, recursive=True)]
                    if query_lower and query_lower != "music":
                        found = [f for f in found
                                 if query_lower in f.name.lower()]
                    found_files.extend(found)
                except Exception:
                    pass

            if len(found_files) >= 50:
                break

        if not found_files:
            return f"No local music found matching '{query}'."

        # Pick a random file or play a directory
        import random
        file = random.choice(found_files)

        if self._mpv_process:
            await self.stop()

        try:
            self._mpv_process = subprocess.Popen(
                ["mpv", "--no-video",
                 "--input-ipc-server=/tmp/Diego_mpv_socket",
                 str(file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._current.provider = MusicProvider.LOCAL
            self._current.state = PlaybackState.PLAYING
            self._current.title = file.stem
            return f"Playing {file.name}."
        except Exception as e:
            logger.warning("[Music] Local play error: %s", e)
            return f"Couldn't play {file.name}."

    # ── Provider selection ─────────────────────────────────────

    def _resolve_provider(self, provider: Optional[str],
                          query: str) -> MusicProvider:
        """Determine which provider to use."""
        if provider:
            try:
                return MusicProvider(provider.lower())
            except ValueError:
                pass

        # UX FIX (2026-08-30): explicit YouTube requests must use the
        # VISIBLE browser path — never hidden mpv playback.
        if "youtube" in query.lower():
            return MusicProvider.BROWSER

        # Use preferred provider if available
        if self._preferred_provider:
            return self._preferred_provider

        # Spotify-specific queries
        spotify_hints = ("on spotify", "spotify playlist", "spotify album")
        if any(h in query.lower() for h in spotify_hints):
            if shutil.which("spotify"):
                return MusicProvider.SPOTIFY

        # Local music hints
        local_hints = ("my music", "local file", "from my computer",
                        "local song")
        if any(h in query.lower() for h in local_hints):
            return MusicProvider.LOCAL

        # Prefer MPV if available (better UX, no browser)
        if shutil.which("mpv"):
            return MusicProvider.MPV

        # Fallback to browser
        return MusicProvider.BROWSER

    # ── Learning ───────────────────────────────────────────────

    def _record_usage(self, query: str, provider: MusicProvider,
                      success: bool) -> None:
        """Record music usage for learning."""
        try:
            from learning.learning_engine import learning_engine
            learning_engine.record_action(
                action_name="music_play",
                params={
                    "query": query,
                    "provider": provider.value,
                },
                success=success,
                latency_ms=0.0,
            )

            # Track artists (simple pattern: "play X by Y")
            query_lower = query.lower()
            if " by " in query_lower:
                artist = query_lower.split(" by ")[-1].strip()
                if artist and artist not in self._favorite_artists:
                    self._favorite_artists.append(artist)
                    if len(self._favorite_artists) > 20:
                        self._favorite_artists = self._favorite_artists[-20:]

            # Track playlist/genre queries
            if any(w in query_lower for w in ("playlist", "mix", "radio", "genre")):
                if query not in self._favorite_playlists:
                    self._favorite_playlists.append(query)
                    if len(self._favorite_playlists) > 20:
                        self._favorite_playlists = self._favorite_playlists[-20:]

            self._save_preferences()
        except Exception as e:
            logger.debug("[Music] Learn error: %s", e)

    def _load_preferences(self) -> None:
        """Load learned music preferences from DuckDB."""
        try:
            from memory.duckdb_store import DuckDBStore
            import json
            store = DuckDBStore()
            with store.connect() as conn:
                result = conn.execute(
                    "SELECT value FROM user_preferences WHERE key = 'music_preferences'"
                ).fetchone()
                if result and result[0]:
                    data = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    self._preferred_provider = (
                        MusicProvider(data["preferred_provider"])
                        if data.get("preferred_provider") else None
                    )
                    self._favorite_artists = data.get("favorite_artists", [])
                    self._favorite_playlists = data.get("favorite_playlists", [])
                    self._favorite_genres = data.get("favorite_genres", [])
        except Exception:
            pass

    def _save_preferences(self) -> None:
        """Persist music preferences to DuckDB."""
        try:
            from memory.duckdb_store import DuckDBStore
            import json
            data = {
                "preferred_provider": self._preferred_provider.value
                if self._preferred_provider else None,
                "favorite_artists": self._favorite_artists[-50:],
                "favorite_playlists": self._favorite_playlists[-50:],
                "favorite_genres": self._favorite_genres[-50:],
                "updated_at": time.time(),
            }
            store = DuckDBStore()
            with store.connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
                    VALUES ('music_preferences', ?, CURRENT_TIMESTAMP)
                """, [json.dumps(data)])
        except Exception:
            pass

    # ── Cleanup ────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Stop playback and clean up."""
        await self.stop()
        self._save_preferences()

    @property
    def is_playing(self) -> bool:
        return self._current.state == PlaybackState.PLAYING

    @property
    def current_track(self) -> TrackInfo:
        return self._current


# Global singleton
music_agent = MusicAgent()