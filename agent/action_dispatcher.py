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
import re
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
            return await self._read_screen(params)

        # ── Real web research (2026-08-30 hardening) ──
        # "search for X" must actually search, extract content, and answer —
        # never stop at opening the browser.
        if name == "web_search":
            return await self._web_search(params)
        if name == "web_search_open_best":
            return await self._web_search_open_best(params)

        # ── Music actions must run in the event loop, not a thread ──
        if name == "play_media":
            return await self._play_media_async(params)
        if name == "youtube_search":
            # UX FIX (2026-08-30): "search youtube for X" is SEARCH-ONLY —
            # open the results page visibly, never start playback.
            return await self._youtube_search(params.get("query", ""))
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
        # CRITICAL FIX (audit B2): _execute_sync reports failures as
        # human-readable strings (e.g. "Couldn't find gnome-terminal").
        # Those strings are truthy, so the old code returned them as if
        # the action had succeeded — and the fallback path (gtk-launch /
        # xdg-open / alternate browsers) was NEVER attempted. Now a
        # failure result is remembered and the fallback still runs; the
        # original failure string is only returned if the fallback also
        # fails, so Brain's verification and retry logic keep working.
        primary_failure: Optional[str] = None
        try:
            result = await loop.run_in_executor(None, self._execute_sync, name, params)
            if result and not self._is_failure_result(result):
                return result
            if result:
                primary_failure = result
        except Exception as e:
            logger.warning("[ACTIONS] execute failed (%s): %s", name, e)

        # ── Fallback alternative ──────────────────────────
        try:
            fallback_result = await self._execute_fallback(name, params)
            if fallback_result and not self._is_failure_result(fallback_result):
                return fallback_result
        except Exception as e:
            logger.warning("[ACTIONS] Fallback failed (%s): %s", name, e)

        # Track entity for pronoun resolution
        self._track_entity_for_memory(name, params)
        if primary_failure:
            return primary_failure
        return f"Couldn't {name.replace('_', ' ')}"

    @staticmethod
    def _is_failure_result(result: Any) -> bool:
        """True if a dispatch result string represents a FAILURE, not success.

        CRITICAL FIX (audit B2): _execute_sync / _open_url_fallback report
        failures as human-readable strings ("Couldn't find X", "Volume
        control unavailable"). Those strings are truthy, so callers must
        not treat them as a successful execution — otherwise the fallback
        path is never attempted and Brain's retry logic sees a "success".
        """
        if not result:
            return False
        text = str(result)
        return "Couldn't" in text or "unavailable" in text

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
            url = params.get("url") or ""
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
        if name == "focus_app":
            return self._focus_app(params.get("app", ""))
        if name == "close_window":
            return self._close_active_window()
        if name == "list_windows":
            return self._list_windows()

        # ── Wi-Fi / Bluetooth (2026-08-30 hardening) ──
        if name == "wifi_on":
            return self._radio("wifi", True)
        if name == "wifi_off":
            return self._radio("wifi", False)
        if name == "bluetooth_on":
            return self._radio("bluetooth", True)
        if name == "bluetooth_off":
            return self._radio("bluetooth", False)

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

    # ── Wi-Fi / Bluetooth (2026-08-30 hardening) ──────────

    def _radio(self, device: str, enable: bool) -> str:
        """Toggle Wi-Fi or Bluetooth via nmcli, with rfkill fallback.

        VERIFIED: after the toggle the state is read back with nmcli/rfkill
        so the response reflects the ACTUAL state, never a guess.
        """
        import shutil
        import subprocess

        state = "on" if enable else "off"
        display = "Wi-Fi" if device == "wifi" else device.capitalize()
        if shutil.which("nmcli"):
            try:
                argv = (["nmcli", "radio", "wifi", state] if device == "wifi"
                        else ["nmcli", "radio", "bluetooth", state])
                subprocess.run(argv, capture_output=True, text=True, timeout=8)
            except Exception as e:
                logger.debug("[ACTIONS] nmcli %s %s failed: %s", device, state, e)

        # 2. rfkill fallback (works without NetworkManager)
        if shutil.which("rfkill"):
            try:
                subprocess.run(
                    ["rfkill", "unblock" if enable else "block", device],
                    capture_output=True, text=True, timeout=5)
            except Exception as e:
                logger.debug("[ACTIONS] rfkill %s %s failed: %s", device, state, e)

        # 3. bluetoothctl power for Bluetooth when nmcli is absent
        if device == "bluetooth" and shutil.which("bluetoothctl"):
            try:
                subprocess.run(
                    ["bluetoothctl", "power", state],
                    capture_output=True, text=True, timeout=8)
            except Exception as e:
                logger.debug("[ACTIONS] bluetoothctl power %s failed: %s", state, e)

        # ── Verify the ACTUAL state ──
        actual = self._radio_state(device)
        if actual is None:
            return f"{display} control unavailable"
        if actual == enable:
            return f"{display} is {'on' if enable else 'off'}."
        return f"I tried to turn {display.lower()} {state}, but it is still " + \
               ("on" if actual else "off") + "."

    @staticmethod
    def _radio_state(device: str) -> Optional[bool]:
        """Read back the actual radio state. True=on, False=off, None=unknown."""
        import shutil
        import subprocess
        if shutil.which("rfkill"):
            try:
                out = subprocess.run(
                    ["rfkill", "list", device],
                    capture_output=True, text=True, timeout=4)
                if out.returncode == 0 and out.stdout.strip():
                    # Any entry NOT blocked (soft or hard) means enabled
                    blocked = [
                        line for line in out.stdout.splitlines()
                        if "Soft blocked" in line or "Hard blocked" in line
                    ]
                    if blocked:
                        return not any(
                            "yes" in line.lower() for line in blocked)
                    return True
                if out.returncode != 0:
                    # rfkill exits non-zero when the device does not exist
                    return None
            except Exception as e:
                logger.debug("[ACTIONS] rfkill state query failed: %s", e)
        if shutil.which("nmcli"):
            try:
                out = subprocess.run(
                    ["nmcli", "radio", device],
                    capture_output=True, text=True, timeout=4)
                if out.returncode == 0:
                    return "enabled" in out.stdout.lower()
            except Exception as e:
                logger.debug("[ACTIONS] nmcli state query failed: %s", e)
        return None

    # ── Desktop awareness (2026-08-30 hardening) ──────────

    def _list_windows(self) -> str:
        """List running applications / open windows (deterministic facts).

        Uses wmctrl (X11) first, then xdotool, then pgrep of common apps.
        The response is built ONLY from real OS data — never guessed.
        """
        import shutil
        import subprocess

        titles: list = []

        if shutil.which("wmctrl"):
            try:
                out = subprocess.run(
                    ["wmctrl", "-l"], capture_output=True, text=True, timeout=3)
                if out.returncode == 0:
                    for line in out.stdout.splitlines():
                        parts = line.split(None, 3)
                        if len(parts) >= 4 and parts[3].strip():
                            titles.append(parts[3].strip())
            except Exception as e:
                logger.debug("[ACTIONS] wmctrl -l failed: %s", e)

        if not titles and shutil.which("xdotool"):
            try:
                out = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", "",
                     "getwindowname", "%@"],
                    capture_output=True, text=True, timeout=3)
                if out.returncode == 0:
                    titles = [t.strip() for t in out.stdout.splitlines() if t.strip()]
            except Exception as e:
                logger.debug("[ACTIONS] xdotool window list failed: %s", e)

        if titles:
            # Deduplicate while preserving order
            seen, unique = set(), []
            for t in titles:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    unique.append(t)
            return f"You have {len(unique)} windows open: " + "; ".join(unique[:8]) + \
                   ("." if len(unique) <= 8 else f", and {len(unique) - 8} more.")

        # Last resort: running GUI processes (deterministic, no guessing)
        common = ("firefox", "chrome", "chromium", "code", "spotify",
                  "gnome-terminal", "xterm", "nautilus", "slack", "discord",
                  "telegram", "pycharm", "obsidian", "vlc", "gimp")
        running = []
        if shutil.which("pgrep"):
            for proc in common:
                try:
                    r = subprocess.run(
                        ["pgrep", "-x", proc], capture_output=True, timeout=2)
                    if r.returncode == 0:
                        running.append(proc)
                except Exception:
                    continue
        if running:
            return "Running apps: " + ", ".join(running) + "."
        return "I couldn't detect any open windows right now."

    def _focus_app(self, app: str) -> str:
        """Focus (switch to) an application's window by name.

        VERIFIED: after activation the active window title is read back;
        a mismatch is reported honestly instead of claiming success.
        """
        import shutil
        import subprocess

        app_lower = (app or "").lower().strip()
        if not app_lower:
            return "Couldn't switch windows"

        name_map = {
            "vs code": "code", "vscode": "code", "code": "code",
            "browser": "firefox", "firefox": "firefox",
            "chrome": "chrome", "google-chrome": "chrome",
            "spotify": "spotify", "terminal": "terminal",
            "gnome-terminal": "terminal", "files": "files",
            "nautilus": "files", "slack": "slack", "discord": "discord",
            "telegram-desktop": "telegram", "telegram": "telegram",
            "pycharm": "pycharm", "notion-app": "notion", "notion": "notion",
        }
        needle = name_map.get(app_lower, app_lower)

        # 1. wmctrl: find a window whose title contains the app name
        if shutil.which("wmctrl"):
            try:
                out = subprocess.run(
                    ["wmctrl", "-l"], capture_output=True, text=True, timeout=3)
                if out.returncode == 0:
                    for line in out.stdout.splitlines():
                        parts = line.split(None, 3)
                        if len(parts) >= 4 and needle in parts[3].lower():
                            subprocess.run(
                                ["wmctrl", "-i", "-a", parts[0]],
                                capture_output=True, text=True, timeout=3)
                            return self._verify_focus(needle, app)
            except Exception as e:
                logger.debug("[ACTIONS] wmctrl focus failed: %s", e)

        # 2. xdotool: search by window name and activate
        if shutil.which("xdotool"):
            try:
                out = subprocess.run(
                    ["xdotool", "search", "--name", needle],
                    capture_output=True, text=True, timeout=3)
                if out.returncode == 0 and out.stdout.strip():
                    wid = out.stdout.split()[0]
                    subprocess.run(
                        ["xdotool", "windowactivate", wid],
                        capture_output=True, text=True, timeout=3)
                    return self._verify_focus(needle, app)
            except Exception as e:
                logger.debug("[ACTIONS] xdotool focus failed: %s", e)

        return f"I couldn't find a {app} window to switch to."

    def _verify_focus(self, needle: str, app: str) -> str:
        """Read back the active window title to confirm the focus switch."""
        title = self._active_window_title()
        if title and needle in title.lower():
            return f"Switched to {app}."
        # Activation command ran but we could not confirm — be honest.
        if title:
            return f"I tried to switch to {app}; the active window is now '{title}'."
        return f"Switched to {app}."

    def _close_active_window(self) -> str:
        """Close the currently focused window (graceful WM close)."""
        import shutil
        import subprocess

        if shutil.which("wmctrl"):
            try:
                out = subprocess.run(
                    ["wmctrl", "-ic", ":ACTIVE:"],
                    capture_output=True, text=True, timeout=3)
                if out.returncode == 0:
                    return "Closed the window."
            except Exception as e:
                logger.debug("[ACTIONS] wmctrl close active failed: %s", e)
        if shutil.which("xdotool"):
            try:
                out = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True, text=True, timeout=2)
                if out.returncode == 0:
                    subprocess.run(
                        ["xdotool", "windowclose", out.stdout.strip()],
                        capture_output=True, text=True, timeout=3)
                    return "Closed the window."
            except Exception as e:
                logger.debug("[ACTIONS] xdotool close failed: %s", e)
        return "Couldn't close the window"

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

    # ── Real web research (2026-08-30 hardening) ──────────

    async def _web_search(self, params: Dict[str, Any]) -> str:
        """Search the web, extract REAL page content, and answer from it.

        Flow: request → search → obtain real results → extract content →
        answer. Never claims a search happened unless it actually did.
        """
        query = (params.get("query") or "").strip()
        site = (params.get("site") or "").strip()
        if not query:
            return "I couldn't figure out what to search for."

        full_query = f"{query} site:{site}" if site else query
        try:
            from services.search_service import search_service
            # Ensure backends are initialized (idempotent, fail-safe)
            try:
                if not search_service.is_ready:
                    await search_service.start()
            except Exception:
                pass  # search() works even without Playwright/Tavily

            result = await search_service.search(full_query, max_results=3)
        except Exception as e:
            logger.warning("[ACTIONS] web_search failed: %s", e)
            return f"I tried to search for {query}, but the search failed."

        if result.error or not result.pages:
            # Be honest — never claim a search that produced nothing.
            return f"I searched for {query} but couldn't get any results."

        # Compose the answer from the top extracted pages (deterministic —
        # real content, no LLM guessing).
        parts = []
        for page in result.pages[:2]:
            if not page.content:
                continue
            # First 2-3 sentences of the extracted content
            sentences = re.split(r"(?<=[.!?])\s+", page.content.strip())
            summary = " ".join(sentences[:3]).strip()
            if summary:
                src = f" (from {page.title or page.url})" if (page.title or page.url) else ""
                parts.append(f"{summary}{src}")
        if not parts:
            # Fall back to search snippets if extraction was empty
            for hit in result.hits[:2]:
                if hit.snippet:
                    parts.append(f"{hit.snippet} (from {hit.title or hit.url})")
        if not parts:
            return f"I searched for {query} but the pages I found had no readable content."

        scope = " on GitHub" if site == "github.com" else ""
        return f"Here's what I found{scope} for {query}: " + " ".join(parts)

    async def _web_search_open_best(self, params: Dict[str, Any]) -> str:
        """search → select best result → open → verify → respond.

        Never claims the page was opened unless the browser actually
        launched with the URL.
        """
        query = (params.get("query") or "").strip()
        if not query:
            return "I couldn't figure out what to search for."

        try:
            from services.search_service import search_service
            try:
                if not search_service.is_ready:
                    await search_service.start()
            except Exception:
                pass

            result = await search_service.search(query, max_results=3)
        except Exception as e:
            logger.warning("[ACTIONS] web_search_open_best search failed: %s", e)
            return f"I tried to search for {query}, but the search failed."

        if result.error or not result.hits:
            return f"I searched for {query} but couldn't get any results."

        # Select the best result: first hit whose page extracted real
        # content, otherwise the first hit.
        best_hit = result.hits[0]
        best_page = None
        for page in result.pages:
            if page.content and page.url == best_hit.url:
                best_page = page
                break
        if best_page is None:
            for page in result.pages:
                if page.content:
                    best_page = page
                    break
        if best_page is not None:
            for hit in result.hits:
                if hit.url == best_page.url:
                    best_hit = hit
                    break

        # Open the best result in the browser
        url = best_hit.url
        opened = False
        ex = self._ensure_executor()
        if ex:
            try:
                ok, _msg = ex.browser_navigate(url)
                opened = bool(ok)
            except Exception:
                opened = False
        if not opened:
            fallback_msg = self._open_url_fallback(url)
            opened = not self._is_failure_result(fallback_msg)
        if not opened:
            return f"I found a result ({best_hit.title or url}) but couldn't open the browser."

        # Verify the browser process is actually running
        import shutil
        import subprocess
        browser_running = False
        if shutil.which("pgrep"):
            try:
                chk = subprocess.run(
                    ["pgrep", "-f", "firefox|chrome|chromium|brave"],
                    capture_output=True, text=True, timeout=3)
                browser_running = (chk.returncode == 0)
            except Exception:
                browser_running = False

        title = best_hit.title or url
        summary = ""
        if best_page is not None and best_page.content:
            sentences = re.split(r"(?<=[.!?])\s+", best_page.content.strip())
            summary = " ".join(sentences[:2]).strip()

        if not browser_running:
            return (f"I opened {title}, but I couldn't verify the browser "
                    f"actually started.")
        if summary:
            return f"I opened {title}. Here's a quick look: {summary}"
        return f"I opened {title} in your browser."

    # ── Screen reading / context ──────────────────────────

    async def _screen_facts(self) -> Dict[str, Any]:
        """Collect DETERMINISTIC screen facts: window title + OCR text.

        Prefers the existing MSS/accessibility/OCR pipeline
        (vision_service → screen_capture + pytesseract). Never guesses.
        """
        facts: Dict[str, Any] = {"window_title": "", "ocr_text": "", "app": ""}

        # 1. Try the structured vision pipeline (MSS + OCR + a11y)
        try:
            from services.vision_service import vision_service
            ctx = await vision_service.force_analyze()
            if ctx is not None:
                facts["window_title"] = getattr(ctx, "window_title", "") or ""
                facts["app"] = getattr(ctx, "app_name", "") or ""
                if getattr(ctx, "raw_ocr_boxes", None):
                    text = " ".join(
                        box.text for box in ctx.raw_ocr_boxes if box.text)
                    facts["ocr_text"] = " ".join(text.split())[:1200]
                    return facts
        except Exception as e:
            logger.debug("[VISION] vision_service facts failed: %s", e)

        # 2. Fallback: MSS capture + direct OCR
        try:
            from services.screen_capture import screen_capture_service
            capture = await screen_capture_service.capture_fullscreen()
            if capture is not None and capture.image is not None:
                import pytesseract
                from PIL import Image
                text = pytesseract.image_to_string(
                    Image.fromarray(capture.image))
                facts["ocr_text"] = " ".join(text.split())[:1200]
        except Exception as e:
            logger.debug("[VISION] direct OCR failed: %s", e)

        if not facts["window_title"]:
            try:
                facts["window_title"] = self._active_window_title()
            except Exception:
                pass
        return facts

    async def _read_screen(self, params: dict | None = None) -> str:
        """Read the screen. OCR/accessibility facts FIRST, vision LLM second.

        The response is grounded in what is ACTUALLY visible — deterministic
        OCR facts are preferred over LLM interpretation, and the vision LLM
        (when available) only describes the real screenshot.
        """
        params = params or {}
        question = (
                params.get("question")
                or "What is currently visible on my screen?"
        )
        q = question.lower()

        try:
            # ── Step 1: deterministic facts (OCR + window title) ──
            facts = await self._screen_facts()
            ocr_text = facts.get("ocr_text", "")
            title = facts.get("window_title", "")

            # Reading requests ("read this page", "read this error") are
            # best answered with the raw OCR text — real content, no LLM.
            wants_reading = any(k in q for k in (
                "read this", "read the", "read my", "what error",
                "error say", "error is shown", "error do you see",
            ))
            if wants_reading:
                if ocr_text:
                    scope = f" on {title}" if title else ""
                    return f"Here's what I can read{scope}: {ocr_text[:600]}"
                return "I looked at the screen but couldn't read any text on it."

            # ── Step 2: vision LLM on the real screenshot (grounded) ──
            from services.screen_capture import screen_capture_service
            capture = await screen_capture_service.capture_fullscreen()
            if capture is not None and capture.image is not None:
                import io
                from PIL import Image
                buffer = io.BytesIO()
                Image.fromarray(capture.image).save(
                    buffer, format="JPEG", quality=85, optimize=True)

                from agent.streaming_llm import streaming_llm
                grounding = ""
                if title:
                    grounding += f"Active window: {title}. "
                if ocr_text:
                    grounding += f"OCR text visible: {ocr_text[:600]}. "
                # HARD TIMEOUT: the vision model can take minutes to load
                # on first use. Never let it hang the user's turn — fall
                # back to the deterministic OCR facts instead.
                try:
                    answer = await asyncio.wait_for(
                        streaming_llm.vision_generate(
                            prompt=(
                                "Look carefully at this desktop screenshot. "
                                "Describe what is visibly on the screen. "
                                "Identify the active application, important visible text, "
                                "errors, dialogs, buttons, and other relevant UI. "
                                "Do not invent anything that is not visible. "
                                + (f"User question: {question}" if params.get("question") else "")
                            ),
                            image_bytes=buffer.getvalue(),
                        ),
                        timeout=45.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[VISION] vision LLM timed out after 45s — "
                                   "falling back to OCR facts")
                    answer = ""
                if answer:
                    return answer

            # ── Step 3: fall back to the deterministic facts ──
            if ocr_text or title:
                parts = []
                if title:
                    parts.append(f"You're on '{title}'")
                if ocr_text:
                    parts.append(f"visible text includes: {ocr_text[:400]}")
                return ". ".join(parts) + "."
            return "I captured the screen, but I couldn't interpret it."

        except Exception as exc:
            logger.exception("[VISION] read_screen failed: %s", exc)
            return "I couldn't read the screen."
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

    async def _play_media_async(self, params: Any) -> str:
        """Play media through the MusicAgent.

        UX FIX (2026-08-30): accepts the full params dict so an explicit
        "play X on youtube" request (params["youtube"]=True) is routed to
        the VISIBLE YouTube playback flow instead of hidden mpv playback.
        """
        if isinstance(params, str):  # backward compatibility
            params = {"query": params}
        query = params.get("query", "")
        try:
            from services.music_agent import music_agent
            await music_agent.initialize()
            if params.get("youtube"):
                return await music_agent.play(query, youtube=True)
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

    async def _youtube_search(self, query: str) -> str:
        """SEARCH-ONLY YouTube request — open results visibly, no playback."""
        try:
            from services.music_agent import music_agent
            await music_agent.initialize()
            return await music_agent.search_youtube(query)
        except Exception as e:
            logger.warning("[ACTIONS] YouTube search failed: %s — falling back", e)
            url = "https://www.youtube.com/results?search_query=" + query.replace(" ", "+")
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