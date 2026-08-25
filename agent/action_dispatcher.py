"""
ActionDispatcher — Maps LLM ACTION dicts to real desktop operations.

Bridges the conversational engine to the existing planner / executor /
browser / vision subsystems. Lets Diego DO things automatically without
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

        # Screen reading is async because it uses the VisionService event loop.
        if name == "read_screen":
            return await self._read_screen()

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
                return f"Typed {text}"
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
            "music_player": "spotify", "music": "spotify", "music player": "spotify",
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
        """Close a desktop application by name (with smart mapping).

        CRITICAL FIX: The previous implementation used `pkill -f <proc>`,
        which matches the FULL command line of every process. This caused
        two failure modes:
          1. Self-kill: the `pkill` subprocess (and any wrapper shell whose
             cmdline contained the pattern) matched itself and was killed,
             returning a misleading non-zero exit code.
          2. Wrong-target kill: `pkill -f` matched a wrapper/parent process
             whose cmdline contained the pattern, killing it while the real
             target survived.

        The fix uses, in order of preference:
          1. Graceful window close via wmctrl/xdotool (proper desktop way).
          2. `pkill -x` (exact process NAME match — never matches cmdline
             wrappers or the invoking process).
          3. `pgrep -x` + direct SIGTERM via os.kill (most robust).
        A self-kill guard excludes Diego's own process tree from any kill.
        """
        import shutil
        import subprocess
        import os
        import signal

        app_lower = app.lower().strip()
        # Map friendly names to process names for pkill
        proc_map = {
            "vs code": "code", "vscode": "code", "code": "code",
            "browser": "firefox", "firefox": "firefox", "chrome": "google-chrome",
            "google-chrome": "google-chrome", "spotify": "spotify",
            "terminal": "xterm", "gnome-terminal": "gnome-terminal",
            "xterm": "xterm", "konsole": "konsole", "alacritty": "alacritty",
            "kitty": "kitty", "wezterm": "wezterm", "tilix": "tilix",
            "files": "nautilus", "file manager": "nautilus", "nautilus": "nautilus",
            "calculator": "gnome-calculator", "settings": "gnome-control-center",
            "slack": "slack", "discord": "discord", "telegram": "telegram-desktop",
            "notion": "notion-app", "pycharm": "pycharm",
        }
        proc = proc_map.get(app_lower, app_lower)

        # ── Self-kill guard: never kill Diego's own process tree ──
        def _self_pids() -> set:
            """Return the set of PIDs in Diego's own process tree (ancestors)."""
            pids = {os.getpid()}
            try:
                ppid = os.getppid()
                # Walk up a few levels to catch the terminal/shell ancestors
                for _ in range(8):
                    if ppid <= 1:
                        break
                    pids.add(ppid)
                    try:
                        with open(f"/proc/{ppid}/stat") as f:
                            # Field 4 of /proc/PID/stat is the parent PID
                            ppid = int(f.read().split()[3])
                    except Exception:
                        break
            except Exception:
                pass
            return pids

        protected = _self_pids()

        # ── 1. Graceful window close (proper desktop method) ──
        # For terminal/window apps, closing the X11 window is cleaner than
        # SIGTERM. wmctrl -c sends a WM_DELETE_WINDOW close request.
        if shutil.which("wmctrl"):
            try:
                # Find window IDs matching the app's window title/class
                out = subprocess.run(
                    ["wmctrl", "-l"], capture_output=True, text=True, timeout=3,
                )
                if out.returncode == 0:
                    for line in out.stdout.splitlines():
                        parts = line.split(None, 3)
                        if len(parts) >= 4 and proc.lower() in parts[3].lower():
                            wid = parts[0]
                            subprocess.run(
                                ["wmctrl", "-ic", wid],
                                capture_output=True, text=True, timeout=3,
                            )
                            logger.info("[ACTIONS] Closed app window via wmctrl: %s (%s)",
                                        app, wid)
                            return f"Closed {app}"
            except Exception as e:
                logger.debug("[ACTIONS] wmctrl close failed for %s: %s", app, e)

        # ── 2. Signal the process and WAIT for actual termination ──
        # CRITICAL FIX: pkill -x signals the process and returns immediately,
        # but the process may take 1-2s to actually die (e.g. cleaning up a
        # child process). The Brain's verification then checks pgrep -x
        # immediately and finds the process still alive → false failure.
        # So we must poll until the process is GONE before reporting success.
        def _state_of(pid: int) -> Optional[str]:
            """Return the process state char from /proc/<pid>/stat.

            Returns None if the process no longer exists.
            A zombie ('Z') means the process is already dead but its
            exit status hasn't been reaped by its parent yet.
            """
            try:
                with open(f"/proc/{pid}/stat") as f:
                    return f.read().split()[2]
            except (FileNotFoundError, ProcessLookupError, IndexError):
                return None

        def _is_live(pid: int) -> bool:
            """True if the process exists AND is not a zombie."""
            state = _state_of(pid)
            return state is not None and state != "Z"

        def _signal_and_wait(proc_name: str) -> bool:
            """Signal all processes with exact name `proc_name` and wait
            until they terminate. Returns True if they are all gone.

            Zombie processes (state 'Z') are treated as already dead —
            signalling them again is pointless and they may linger if
            their parent hasn't reaped them.
            """
            # Find target PIDs (exact name match, skip Diego's own tree)
            targets = []
            try:
                r = subprocess.run(
                    ["pgrep", "-x", proc_name],
                    capture_output=True, text=True, timeout=3,
                )
                if r.returncode == 0:
                    for pid_str in r.stdout.split():
                        pid_str = pid_str.strip()
                        if not pid_str.isdigit():
                            continue
                        pid = int(pid_str)
                        if pid in protected:
                            logger.debug("[ACTIONS] Skipping protected PID %d", pid)
                            continue
                        # Skip zombies — already dead, no need to signal
                        if not _is_live(pid):
                            logger.debug("[ACTIONS] Skipping zombie PID %d", pid)
                            continue
                        targets.append(pid)
            except Exception as e:
                logger.warning("[ACTIONS] pgrep -x failed for %s: %s", proc_name, e)
                return False

            if not targets:
                # No matching live processes — already closed (or never existed)
                return True

            # Send SIGTERM (graceful) to all targets
            for pid in targets:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                except PermissionError as e:
                    logger.warning("[ACTIONS] No permission to kill %d: %s", pid, e)

            # Poll for termination (up to ~2s), then escalate to SIGKILL.
            # Treat zombies as dead — they are no longer running.
            deadline = time.time() + 2.0
            still_alive = []
            while time.time() < deadline:
                still_alive = [pid for pid in targets if _is_live(pid)]
                if not still_alive:
                    return True
                time.sleep(0.1)

            # Escalate to SIGKILL for stubborn survivors
            for pid in still_alive:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except PermissionError as e:
                    logger.warning("[ACTIONS] No permission to SIGKILL %d: %s", pid, e)

            # Final check
            time.sleep(0.2)
            for pid in still_alive:
                if _is_live(pid):
                    return False  # Still alive even after SIGKILL
            return True

        if _signal_and_wait(proc):
            logger.info("[ACTIONS] Closed app: %s (%s)", app, proc)
            return f"Closed {app}"

        # ── 4. killall (exact name match) as last resort ──
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

    async def _read_screen(self) -> str:
        """Read the current screen using the unified vision pipeline."""
        try:
            from services.vision_service import vision_service

            ctx = await vision_service.force_analyze()

            if ctx.capture is None or not ctx.capture.is_valid:
                logger.warning("[ACTIONS] Screen capture failed during read_screen")
                return "I couldn't capture the screen right now."

            if ctx.semantic_summary:
                return f"On screen: {ctx.semantic_summary}"

            if ctx.ocr_text:
                return f"On screen: {ctx.ocr_text[:2000]}"

            return "I captured the screen, but I couldn't extract useful information."

        except Exception as e:
            logger.exception("[ACTIONS] read_screen failed: %s", e)
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