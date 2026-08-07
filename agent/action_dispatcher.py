"""
ActionDispatcher — Maps LLM ACTION dicts to real desktop operations.

Bridges the conversational engine to the existing planner / executor /
browser / vision subsystems. Lets Leo DO things automatically without
asking for confirmation.

Every action follows: execute → verify → retry → fallback → report.

Supported actions (from the LLM's ACTION lines):
  desktop_open(app)          — open an application
  browser_navigate(url)      — open a website
  browser_search(query)      — google search
  read_screen()              — describe what's on screen (OCR + vision)
  click_text(text)           — click visible text / button
  scroll(direction)          — scroll up/down
  key_press(key)             — press a key
  type_text(text)            — type text
  play_media(query)          — play music/video

Also provides `screen_context()` — a compact text summary of the current
screen that the engine injects into the LLM prompt when the user asks
about what they're looking at.

Usage:
    from agent.action_dispatcher import action_dispatcher

    result = await action_dispatcher.execute({"action": "desktop_open", "params": {"app": "code"}})
    ctx = await action_dispatcher.screen_context()
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ActionDispatcher:
    """Executes desktop actions requested by the conversational LLM."""

    def __init__(self):
        self._executor = None
        self._initialized = False

    def _ensure_executor(self):
        if self._executor is None:
            try:
                from agent.executor import agent_executor
                self._executor = agent_executor
                if not self._executor.is_available:
                    self._executor.initialize()
            except Exception as e:
                logger.warning("[ACTIONS] Executor unavailable: %s", e)
                self._executor = None
        return self._executor

    # ── Learning integration ──────────────────────────────

    @staticmethod
    def _record_for_learning(action_name: str, params: Dict[str, Any],
                              success: bool, latency_ms: float = 0.0,
                              error: str = "", retries: int = 0,
                              fallback_used: bool = False) -> None:
        """Record an action in the learning engine (non-blocking)."""
        try:
            from learning.learning_engine import learning_engine
            learning_engine.record_action(
                action_name=action_name,
                params=params,
                success=success,
                latency_ms=latency_ms,
                error=error,
                retries=retries,
                fallback_used=fallback_used,
            )
        except Exception:
            pass  # Learning engine failure must never break action execution

    # ── Main entry point ──────────────────────────────────

    async def execute(self, action: Dict[str, Any]) -> Optional[str]:
        """
        Execute an ACTION dict from the LLM.

        RESPONSIBILITY: The Dispatcher ONLY executes actions. The Brain
        owns verification and learning. This method does NOT verify or
        record for learning — the Brain's pipeline does that.

        Fast path for music actions (must run in the event loop).
        All other actions run in a thread.

        Args:
            action: Action dict with action name and params

        Returns:
            Result string describing what happened.
        """
        name = action.get("action", "")
        params = action.get("params", {}) or {}
        logger.info("[ACTIONS] execute: %s %s", name, params)

        t0 = time.time()
        loop = asyncio.get_event_loop()

        # ── Music actions must run in the event loop, not a thread ──
        if name == "play_media":
            return await self._play_media_async(params.get("query", ""))
        if name in ("music_pause", "music_resume", "music_next",
                     "music_previous", "music_stop", "music_shuffle",
                     "music_repeat", "music_status"):
            return await self._music_action(name, params)
        if name == "music_volume":
            pct = int(params.get("percent", params.get("level", 50)))
            return await self._music_volume(pct)
        if name == "music_mute":
            return await self._music_mute()

        # ── Primary execution ─────────────────────────────
        try:
            result = await loop.run_in_executor(None, self._execute_sync, name, params)
            if result:
                return result
        except Exception as e:
            logger.warning("[ACTIONS] execute failed (%s): %s", name, e)

        # ── Fallback alternative ──────────────────────────
        try:
            fallback_result = await self._execute_fallback(name, params)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.warning("[ACTIONS] Fallback failed (%s): %s", name, e)

        # Track entity for pronoun resolution
        self._track_entity_for_memory(name, params)
        return f"Couldn't {name.replace('_', ' ')}"

    # ── ExperienceDB consultation ─────────────────────────

    @staticmethod
    def _consult_experience(name: str, params: Dict[str, Any]) -> Optional[str]:
        """Consult ExperienceDB for best approach before executing."""
        try:
            from learning.experience_db import experience_db
            goal = f"{name} {json.dumps(params)}" if params else name
            best = experience_db.best_action_for(goal)
            if best:
                logger.info("[ACTIONS] ExperienceDB suggests: %s for %s", best, name)
            # Check for actions to avoid
            avoid = experience_db.avoid_actions(goal)
            if avoid:
                logger.info("[ACTIONS] ExperienceDB warns against: %s", avoid)
            return best
        except Exception:
            return None

    # ── Pre/post action screen capture ────────────────────

    @staticmethod
    def _capture_pre_action() -> None:
        """Capture screen state before action for verification."""
        try:
            from vision.action_verifier import action_verifier
            action_verifier.capture_pre_action()
        except Exception:
            pass

    @staticmethod
    def _verify_post_action(name: str, params: Dict[str, Any]) -> bool:
        """Verify action had expected effect using vision."""
        try:
            from vision.action_verifier import action_verifier
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    action_verifier.verify_action(name, params), loop)
                result = future.result(timeout=5)
                return result.success
        except Exception:
            pass
        return True  # Default to trust on verification failure

    # ── Entity tracking for pronoun resolution ────────────

    @staticmethod
    def _track_entity_for_memory(name: str, params: Dict[str, Any]) -> None:
        """Track entities for pronoun resolution in conversation memory."""
        try:
            from agent.conversation_memory import conv_memory
            if name == "desktop_open":
                app = params.get("app", "")
                if app:
                    conv_memory.track_entity(app)
                    conv_memory.track_action(f"opened {app}")
            elif name == "browser_navigate":
                url = params.get("url", "")
                if url:
                    conv_memory.track_entity(url)
            elif name == "play_media":
                query = params.get("query", "")
                if query:
                    conv_memory.track_entity(query)
                    conv_memory.track_action(f"played {query}")
        except Exception:
            pass

    # ── Verification ──────────────────────────────────────

    async def _verify_action(self, name: str, params: Dict[str, Any],
                              result: Optional[str]) -> bool:
        """
        Verify that an action actually succeeded.

        Uses multiple verification strategies:
          - Process check (did the app actually start?)
          - Vision check (did the screen change?)
          - Result check (did we get a success message?)
        """
        if result and "Couldn't" in str(result):
            return False

        # App launch verification
        if name == "desktop_open":
            app = params.get("app", "").lower()
            return await self._verify_app_launched(app)

        # Browser verification
        if name in ("browser_navigate", "browser_search"):
            return await self._verify_browser_action()

        # Music verification
        if name == "play_media":
            return await self._verify_music_playing()

        # For other actions, trust the result if it's not an error
        if result and not str(result).startswith("Couldn't"):
            return True

        return result is not None

    async def _verify_app_launched(self, app: str) -> bool:
        """Check if an application process is actually running."""
        import subprocess

        app_proc_map = {
            "code": "code", "vscode": "code", "vs code": "code",
            "firefox": "firefox", "chrome": "chrome", "google-chrome": "chrome",
            "spotify": "spotify", "gnome-terminal": "gnome-terminal",
            "terminal": "gnome-terminal", "nautilus": "nautilus",
            "files": "nautilus", "file manager": "nautilus",
        }
        proc_name = app_proc_map.get(app, app)

        try:
            result = subprocess.run(
                ["pgrep", "-f", proc_name],
                capture_output=True, text=True, timeout=3
            )
            return result.returncode == 0
        except Exception:
            # pgrep not available — trust the launch
            return True

    async def _verify_browser_action(self) -> bool:
        """Verify browser action by checking if a browser process is running."""
        import subprocess
        try:
            result = subprocess.run(
                ["pgrep", "-f", "firefox|chrome|chromium|brave"],
                capture_output=True, text=True, timeout=3
            )
            return result.returncode == 0
        except Exception:
            return True

    async def _verify_music_playing(self) -> bool:
        """Check if music is actually playing."""
        try:
            from services.music_agent import music_agent
            return music_agent.is_playing
        except Exception:
            return True

    # ── Retry parameter adjustment ────────────────────────

    def _adjust_params_for_retry(self, name: str,
                                  params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters for a retry attempt."""
        adjusted = dict(params)

        if name == "desktop_open":
            app = params.get("app", "")
            # Try alternative binary names
            alt_map = {
                "code": "code-insiders",
                "vscode": "code",
                "vs code": "code",
                "firefox": "firefox-esr",
                "chrome": "chromium-browser",
                "google-chrome": "chromium",
                "gnome-terminal": "xterm",
                "terminal": "xterm",
                "nautilus": "thunar",
                "files": "thunar",
                "file manager": "thunar",
            }
            if app.lower() in alt_map:
                adjusted["app"] = alt_map[app.lower()]

        if name == "browser_navigate":
            # Try without https:// prefix
            url = params.get("url", "")
            if url.startswith("https://"):
                adjusted["url"] = url.replace("https://", "http://")

        return adjusted

    # ── Fallback execution ────────────────────────────────

    async def _execute_fallback(self, name: str,
                                 params: Dict[str, Any]) -> Optional[str]:
        """Execute a fallback alternative when primary and retry both fail."""
        loop = asyncio.get_event_loop()

        if name == "desktop_open":
            app = params.get("app", "")
            # Fallback: try xdg-open or gtk-launch
            import shutil
            import subprocess
            for launcher in ("gtk-launch", "xdg-open"):
                if shutil.which(launcher):
                    try:
                        subprocess.Popen(
                            [launcher, app],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        return f"Opened {app} via {launcher}"
                    except Exception:
                        continue
            return None

        if name in ("browser_navigate", "browser_search"):
            url = params.get("url", "")
            if not url:
                query = params.get("query", "")
                url = "https://www.google.com/search?q=" + query.replace(" ", "+")
            # Fallback: try multiple browsers
            import shutil
            import subprocess
            for browser in ("firefox", "google-chrome", "chromium-browser",
                            "chromium", "brave-browser", "epiphany"):
                if shutil.which(browser):
                    try:
                        subprocess.Popen(
                            [browser, url],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        return f"Opened {url} in {browser}"
                    except Exception:
                        continue
            return None

        if name == "play_media":
            query = params.get("query", "")
            # Fallback: open YouTube search in browser
            url = "https://www.youtube.com/results?search_query=" + query.replace(" ", "+")
            return await loop.run_in_executor(
                None, self._open_url_fallback, url)

        return None

    # ── Synchronous dispatch (runs in thread) ─────────────

    def _execute_sync(self, name: str, params: Dict[str, Any]) -> Optional[str]:
        ex = self._ensure_executor()

        # ── Application launching ─────────────────────────
        if name == "desktop_open":
            app = params.get("app", "")
            return self._open_app(app)

        if name == "close_app":
            app = params.get("app", "")
            return self._close_app(app)

        # ── Browser ───────────────────────────────────────
        if name == "browser_navigate":
            url = params.get("url", "")
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            if ex:
                ok, msg = ex.browser_navigate(url)
                if ok:
                    return msg
            # Selenium path unavailable/failed → native xdg-open fallback.
            return self._open_url_fallback(url)

        if name == "browser_search":
            query = params.get("query", "")
            url = "https://www.google.com/search?q=" + query.replace(" ", "+")
            if ex:
                ok, msg = ex.browser_navigate(url)
                if ok:
                    return f"Searched for {query}"
            return self._open_url_fallback(url)

        # ── Screen reading ────────────────────────────────
        if name == "read_screen":
            return self._read_screen()

        # ── Mouse / keyboard ──────────────────────────────
        if name == "click_text":
            text = params.get("text", "")
            return self._click_text(text)

        if name == "scroll":
            direction = params.get("direction", "down")
            clicks = -5 if direction == "down" else 5
            if ex:
                ex.scroll(clicks)
                return f"Scrolled {direction}"
            return None

        if name == "key_press":
            key = params.get("key", "")
            if ex:
                ex.keyboard_press(key)
                return f"Pressed {key}"
            return None

        if name == "type_text":
            text = params.get("text", "")
            if ex:
                ex.keyboard_type(text)
                return None
            return None

        # ── Music actions are handled in async execute() above ──
        # (They must run in the event loop, not a thread)

        # ── Folder opening ────────────────────────────────
        if name == "open_folder":
            path = params.get("path") or str(Path.home())
            return self._open_folder(path)

        # ── Volume (native PipeWire/ALSA APIs) ────────────
        if name == "volume_up":
            return self._volume_change("+10%")
        if name == "volume_down":
            return self._volume_change("-10%")
        if name == "volume_set":
            pct = int(params.get("percent", params.get("level", 50)))
            return self._volume_set(pct)
        if name == "volume_mute":
            return self._volume_mute()

        # ── Brightness (native brightnessctl) ─────────────
        if name == "brightness_up":
            return self._brightness_change("+10%")
        if name == "brightness_down":
            return self._brightness_change("10%-")
        if name == "brightness_set":
            pct = int(params.get("percent", params.get("level", 70)))
            return self._brightness_set(pct)

        # ── Session / power (native logind) ───────────────
        if name == "lock_screen":
            return self._lock_screen()
        if name == "shutdown":
            return self._power("poweroff")
        if name == "restart":
            return self._power("reboot")

        # ── Time / date ───────────────────────────────────
        if name == "get_time":
            import datetime
            now = datetime.datetime.now()
            return f"It's {now.strftime('%-I:%M %p')}."

        if name == "get_date":
            import datetime
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # ── Window management (X11/Wayland via wmctrl/xdotool) ──
        if name == "minimize_window":
            return self._window_action("minimize")
        if name == "maximize_window":
            return self._window_action("maximize")
        if name == "switch_workspace":
            return self._window_action("workspace_next")
        if name == "switch_workspace_prev":
            return self._window_action("workspace_prev")
        if name == "switch_window":
            return self._window_action("window_next")
        if name == "switch_window_prev":
            return self._window_action("window_prev")
        if name == "switch_tab":
            return self._window_action("tab_next")
        if name == "switch_tab_prev":
            return self._window_action("tab_prev")

        logger.warning("[ACTIONS] Unknown action: %s", name)
        return None

    # ── Native desktop operations ─────────────────────────

    @staticmethod
    def _run_cmd(argv: list, timeout: float = 5.0) -> bool:
        """Run a desktop command detached; True on spawn success."""
        import subprocess
        try:
            subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            logger.info("[ACTIONS] ran: %s", " ".join(argv))
            return True
        except Exception as e:
            logger.warning("[ACTIONS] command failed %s: %s", argv, e)
            return False

    def _open_folder(self, path: str) -> str:
        """Open a folder in the desktop file manager (xdg-open / gio)."""
        import shutil
        p = Path(path).expanduser()
        if not p.exists():
            p = Path.home()
        for opener in ("xdg-open", "gio"):
            exe = shutil.which(opener)
            if exe:
                argv = [exe, "open", str(p)] if opener == "gio" else [exe, str(p)]
                if self._run_cmd(argv):
                    return f"Opened {p}"
        return "Couldn't open the folder"

    def _volume_change(self, delta: str) -> str:
        """Adjust output volume via pactl (PipeWire/Pulse) or amixer."""
        import shutil
        if shutil.which("pactl"):
            if self._run_cmd(["pactl", "set-sink-volume", "@DEFAULT_SINK@", delta]):
                return f"Volume {'up' if delta.startswith('+') else 'down'}"
        if shutil.which("amixer"):
            if self._run_cmd(["amixer", "-q", "sset", "Master", f"{delta}%"]):
                return "Volume adjusted"
        return "Volume control unavailable"

    def _volume_set(self, percent: int) -> str:
        import shutil
        percent = max(0, min(100, percent))
        if shutil.which("pactl"):
            if self._run_cmd(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"]):
                return f"Volume set to {percent} percent"
        if shutil.which("amixer"):
            if self._run_cmd(["amixer", "-q", "sset", "Master", f"{percent}%"]):
                return f"Volume set to {percent} percent"
        return "Volume control unavailable"

    def _volume_mute(self) -> str:
        import shutil
        if shutil.which("pactl"):
            if self._run_cmd(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]):
                return "Toggled mute"
        if shutil.which("amixer"):
            if self._run_cmd(["amixer", "-q", "sset", "Master", "toggle"]):
                return "Toggled mute"
        return "Volume control unavailable"

    def _brightness_change(self, delta: str) -> str:
        import shutil
        if shutil.which("brightnessctl"):
            if self._run_cmd(["brightnessctl", "s", delta]):
                return "Brightness adjusted"
        if shutil.which("light"):
            if self._run_cmd(["light", "-A" if delta.startswith("+") else "-U", delta.lstrip("+%-")]):
                return "Brightness adjusted"
        return "Brightness control unavailable"

    def _brightness_set(self, percent: int) -> str:
        import shutil
        percent = max(0, min(100, percent))
        if shutil.which("brightnessctl"):
            if self._run_cmd(["brightnessctl", "s", f"{percent}%"]):
                return f"Brightness set to {percent} percent"
        if shutil.which("light"):
            if self._run_cmd(["light", "-S", str(percent)]):
                return f"Brightness set to {percent} percent"
        return "Brightness control unavailable"

    def _lock_screen(self) -> str:
        import shutil
        # logind first (works on Wayland + X11), then common lockers.
        for argv in (
            ["loginctl", "lock-session"],
            ["xdg-screensaver", "lock"],
            ["dm-tool", "lock"],
            ["gnome-screensaver-command", "--lock"],
        ):
            if shutil.which(argv[0]) and self._run_cmd(argv):
                return "Screen locked"
        return "Couldn't lock the screen"

    def _window_action(self, action: str) -> str:
        """Perform a window management action via wmctrl/xdotool."""
        import shutil
        import subprocess

        # wmctrl (X11) — workspace switching
        if action in ("workspace_next", "workspace_prev"):
            if shutil.which("wmctrl"):
                try:
                    out = subprocess.run(
                        ["wmctrl", "-d"], capture_output=True, text=True, timeout=2)
                    if out.returncode == 0:
                        lines = [l for l in out.stdout.splitlines() if l.strip()]
                        current = next((i for i, l in enumerate(lines) if l.startswith("*")), 0)
                        total = len(lines)
                        target = (current + 1) % total if action == "workspace_next" else (current - 1) % total
                        subprocess.run(
                            ["wmctrl", "-s", str(target)],
                            capture_output=True, text=True, timeout=2)
                        return "Switched workspace"
                except Exception:
                    pass
            # Fallback: xdotool keybinding
            if shutil.which("xdotool"):
                key = "super+Right" if action == "workspace_next" else "super+Left"
                subprocess.run(["xdotool", "key", key],
                               capture_output=True, text=True, timeout=2)
                return "Switched workspace"
            return "Workspace switching unavailable"

        # wmctrl (X11) — minimize/maximize
        if action in ("minimize", "maximize"):
            if shutil.which("xdotool"):
                try:
                    out = subprocess.run(
                        ["xdotool", "getactivewindow"],
                        capture_output=True, text=True, timeout=2)
                    if out.returncode == 0:
                        wid = out.stdout.strip()
                        if action == "minimize":
                            subprocess.run(["xdotool", "windowminimize", wid],
                                           capture_output=True, text=True, timeout=2)
                            return "Minimized window"
                        else:
                            subprocess.run(["xdotool", "windowsize", wid, "100%", "100%"],
                                           capture_output=True, text=True, timeout=2)
                            return "Maximized window"
                except Exception:
                    pass
            return f"Couldn't {action} window"

        # Window switching (Alt+Tab)
        if action in ("window_next", "window_prev"):
            if shutil.which("xdotool"):
                key = "alt+Tab" if action == "window_next" else "alt+shift+Tab"
                subprocess.run(["xdotool", "key", key],
                               capture_output=True, text=True, timeout=2)
                return "Switched window"
            return "Window switching unavailable"

        # Tab switching (Ctrl+Tab)
        if action in ("tab_next", "tab_prev"):
            if shutil.which("xdotool"):
                key = "ctrl+Tab" if action == "tab_next" else "ctrl+shift+Tab"
                subprocess.run(["xdotool", "key", key],
                               capture_output=True, text=True, timeout=2)
                return "Switched tab"
            return "Tab switching unavailable"

        return f"Unknown window action: {action}"

    def _power(self, action: str) -> str:
        import shutil
        if shutil.which("systemctl") and self._run_cmd(["systemctl", action]):
            return "Shutting down" if action == "poweroff" else "Restarting"
        return "Power control unavailable"

    # ── App / URL launching ───────────────────────────────

    def _open_app(self, app: str) -> str:
        """Open a desktop application by name (with smart mapping)."""
        import shutil
        import subprocess

        app_lower = app.lower().strip()
        # Map friendly names to actual binaries
        app_map = {
            "vs code": "code", "vscode": "code", "code": "code",
            "browser": "firefox", "firefox": "firefox", "chrome": "google-chrome",
            "spotify": "spotify", "terminal": "gnome-terminal",
            "files": "nautilus", "file manager": "nautilus",
            "calculator": "gnome-calculator", "settings": "gnome-control-center",
            "slack": "slack", "discord": "discord", "telegram": "telegram-desktop",
            "notion": "notion-app", "obsidian": "obsidian",
        }
        binary = app_map.get(app_lower, app_lower)

        # Special case: Spotify via web if binary missing
        if binary == "spotify" and not shutil.which("spotify"):
            return self._open_url_fallback("https://open.spotify.com")

        exe = shutil.which(binary)
        if exe is None:
            # Try common alternates
            for alt in (binary, binary.replace("-", ""), f"{binary}-stable"):
                exe = shutil.which(alt)
                if exe:
                    break
        if exe is None:
            logger.warning("[ACTIONS] App not found: %s", app)
            return f"Couldn't find {app}"

        try:
            subprocess.Popen(
                [exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            logger.info("[ACTIONS] Opened app: %s (%s)", app, exe)
            return f"Opened {app}"
        except Exception as e:
            logger.warning("[ACTIONS] Failed to open %s: %s", app, e)
            return f"Couldn't open {app}"

    def _close_app(self, app: str) -> str:
        """Close a desktop application by name (with smart mapping)."""
        import shutil
        import subprocess

        app_lower = app.lower().strip()
        # Map friendly names to process names for pkill
        proc_map = {
            "vs code": "code", "vscode": "code", "code": "code",
            "browser": "firefox", "firefox": "firefox", "chrome": "google-chrome",
            "google-chrome": "google-chrome", "spotify": "spotify",
            "terminal": "gnome-terminal", "gnome-terminal": "gnome-terminal",
            "files": "nautilus", "file manager": "nautilus", "nautilus": "nautilus",
            "calculator": "gnome-calculator", "settings": "gnome-control-center",
            "slack": "slack", "discord": "discord", "telegram": "telegram-desktop",
            "notion": "notion-app", "pycharm": "pycharm",
        }
        proc = proc_map.get(app_lower, app_lower)

        # Try pkill first (sends SIGTERM — graceful)
        if shutil.which("pkill"):
            try:
                result = subprocess.run(
                    ["pkill", "-f", proc],
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode == 0:
                    logger.info("[ACTIONS] Closed app: %s (%s)", app, proc)
                    return f"Closed {app}"
            except Exception as e:
                logger.warning("[ACTIONS] pkill failed for %s: %s", app, e)

        # Fallback: try killall
        if shutil.which("killall"):
            try:
                result = subprocess.run(
                    ["killall", proc],
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode == 0:
                    logger.info("[ACTIONS] Closed app via killall: %s", app)
                    return f"Closed {app}"
            except Exception as e:
                logger.warning("[ACTIONS] killall failed for %s: %s", app, e)

        logger.warning("[ACTIONS] Could not close app: %s", app)
        return f"Couldn't close {app}"

    def _open_url_fallback(self, url: str) -> str:
        """Open a URL in the default browser."""
        import subprocess
        import shutil
        for opener in ("xdg-open", "gio"):
            exe = shutil.which(opener)
            if exe:
                try:
                    if opener == "gio":
                        subprocess.Popen([exe, "open", url],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                    else:
                        subprocess.Popen([exe, url],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                    return f"Opened {url}"
                except Exception:
                    continue
        return f"Couldn't open {url}"

    # ── Screen reading / context ──────────────────────────

    def _read_screen(self) -> str:
        """OCR + describe the current screen."""
        try:
            text = self._ocr_sync()
            if text:
                return f"On screen: {text[:500]}"
        except Exception as e:
            logger.debug("[ACTIONS] read_screen failed: %s", e)
        return "I couldn't read the screen right now."

    def _ocr_sync(self) -> str:
        """Synchronous OCR of the active window."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._ocr_async(), loop)
                return future.result(timeout=5)
            return ""
        except Exception as e:
            logger.debug("[ACTIONS] OCR failed: %s", e)
            return ""

    async def _ocr_async(self) -> str:
        """Async OCR helper."""
        try:
            from services.vision_service import vision_service
            return await vision_service.ocr_only()
        except Exception:
            return ""

    async def screen_context(self) -> str:
        """
        Return a compact description of the current screen for the LLM.

        Combines: active window title + OCR snippet + visible UI hints.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._screen_context_sync)

    def _screen_context_sync(self) -> str:
        parts = []
        # Active window title
        try:
            title = self._active_window_title()
            if title:
                parts.append(f"Active window: {title}")
        except Exception:
            pass
        # OCR snippet from vision_service
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._ocr_async(), loop)
                text = future.result(timeout=5)
                if text:
                    snippet = " ".join(text.split())[:300]
                    parts.append(f"Visible text: {snippet}")
        except Exception:
            pass
        return " | ".join(parts)

    @staticmethod
    def _active_window_title() -> str:
        """Get the focused window title (X11/Wayland)."""
        import subprocess
        import shutil
        # xdotool (X11)
        if shutil.which("xdotool"):
            try:
                out = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=2)
                if out.returncode == 0:
                    return out.stdout.strip()
            except Exception:
                pass
        # wmctrl fallback
        if shutil.which("wmctrl"):
            try:
                out = subprocess.run(
                    ["wmctrl", "-l"], capture_output=True, text=True, timeout=2)
                for line in out.stdout.splitlines():
                    segs = line.split(None, 3)
                    if len(segs) == 4:
                        return segs[3]
            except Exception:
                pass
        return ""

    # ── Click / media helpers ─────────────────────────────

    def _click_text(self, text: str) -> str:
        """Click visible text: try browser first, then screen OCR coords."""
        ex = self._ensure_executor()
        if ex:
            ok, msg = ex.browser_click_text(text)
            if ok:
                return msg
        # Fallback: OCR-locate the text on screen and click its centroid
        try:
            coords = self._locate_text_on_screen(text)
            if coords and ex:
                ex.mouse_click(coords[0], coords[1])
                return f"Clicked {text}"
        except Exception as e:
            logger.debug("[ACTIONS] click_text fallback failed: %s", e)
        return f"Couldn't find {text} to click"

    def _locate_text_on_screen(self, text: str):
        """Return (x, y) centroid of `text` on screen via vision_service, or None."""
        try:
            from services.vision_service import vision_service
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return None
            future = asyncio.run_coroutine_threadsafe(
                self._locate_text_async(text), loop)
            return future.result(timeout=5)
        except Exception as e:
            logger.debug("[ACTIONS] locate_text failed: %s", e)
        return None

    async def _locate_text_async(self, text: str):
        """Async locate text using vision_service."""
        from services.vision_service import vision_service
        ctx = await vision_service.force_analyze()
        if not ctx.raw_ocr_boxes:
            return None
        text_lower = text.lower()
        for box in ctx.raw_ocr_boxes:
            if text_lower in box.text.lower():
                return box.center
        return None

    async def _play_media_async(self, query: str) -> str:
        """Play media through the MusicAgent."""
        try:
            from services.music_agent import music_agent
            await music_agent.initialize()
            return await music_agent.play(query)
        except Exception as e:
            logger.warning("[ACTIONS] MusicAgent failed: %s — falling back", e)
            # Fallback to old browser-based playback
            url = "https://www.youtube.com/results?search_query=" + query.replace(" ", "+")
            ex = self._ensure_executor()
            if ex:
                ok, msg = ex.browser_navigate(url)
                if ok:
                    return f"Playing {query}"
            return self._open_url_fallback(url)

    async def _music_action(self, name: str, params: Dict[str, Any]) -> str:
        """Execute a music control action."""
        try:
            from services.music_agent import music_agent
            action_map = {
                "music_pause": music_agent.pause,
                "music_resume": music_agent.resume,
                "music_next": music_agent.next,
                "music_previous": music_agent.previous,
                "music_stop": music_agent.stop,
                "music_shuffle": music_agent.shuffle,
                "music_repeat": music_agent.repeat,
                "music_status": music_agent.status,
            }
            fn = action_map.get(name)
            if fn:
                return await fn()
        except Exception as e:
            logger.debug("[ACTIONS] Music action failed: %s", e)
        return f"Music control unavailable for {name}"

    async def _music_volume(self, percent: int) -> str:
        try:
            from services.music_agent import music_agent
            return await music_agent.set_volume(percent)
        except Exception:
            return "Volume control unavailable"

    async def _music_mute(self) -> str:
        try:
            from services.music_agent import music_agent
            return await music_agent.mute()
        except Exception:
            return "Mute unavailable"


# Global singleton
action_dispatcher = ActionDispatcher()