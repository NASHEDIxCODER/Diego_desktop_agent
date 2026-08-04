"""
ActionDispatcher — Maps LLM ACTION dicts to real desktop operations.

Bridges the conversational engine to the existing planner / executor /
browser / vision subsystems. Lets Leo DO things automatically without
asking for confirmation.

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
                              error: str = "") -> None:
        """Record an action in the learning engine (non-blocking)."""
        try:
            from learning.learning_engine import learning_engine
            learning_engine.record_action(
                action_name=action_name,
                params=params,
                success=success,
                latency_ms=latency_ms,
                error=error,
            )
        except Exception:
            pass  # Learning engine failure must never break action execution

    # ── Main entry point ──────────────────────────────────

    async def execute(self, action: Dict[str, Any]) -> Optional[str]:
        """
        Execute an ACTION dict from the LLM.

        Runs blocking desktop operations in a thread so the event loop
        stays responsive (full duplex keeps listening while acting).

        Every action is recorded in the learning engine for continuous
        self-improvement.
        """
        name = action.get("action", "")
        params = action.get("params", {}) or {}
        logger.info("[ACTIONS] execute: %s %s", name, params)

        t0 = time.time()
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._execute_sync, name, params)
            latency_ms = (time.time() - t0) * 1000
            self._record_for_learning(name, params, success=True, latency_ms=latency_ms)
            return result
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            self._record_for_learning(name, params, success=False, latency_ms=latency_ms, error=str(e))
            logger.warning("[ACTIONS] execute failed (%s): %s", name, e)
            return f"Couldn't {name.replace('_', ' ')}"

    # ── Synchronous dispatch (runs in thread) ─────────────

    def _execute_sync(self, name: str, params: Dict[str, Any]) -> Optional[str]:
        ex = self._ensure_executor()

        # ── Application launching ─────────────────────────
        if name == "desktop_open":
            app = params.get("app", "")
            return self._open_app(app)

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

        # ── Media (routed through MusicAgent) ────────────
        if name == "play_media":
            query = params.get("query", "")
            return asyncio.run(self._play_media_async(query))

        # ── Music control actions ─────────────────────────
        if name in ("music_pause", "music_resume", "music_next",
                     "music_previous", "music_stop", "music_shuffle",
                     "music_repeat", "music_status"):
            return asyncio.run(self._music_action(name, params))
        if name == "music_volume":
            pct = int(params.get("percent", params.get("level", 50)))
            return asyncio.run(self._music_volume(pct))
        if name == "music_mute":
            return asyncio.run(self._music_mute())

        # ── Folder opening (TASK 9) ───────────────────────
        if name == "open_folder":
            path = params.get("path") or str(Path.home())
            return self._open_folder(path)

        # ── Volume (TASK 9, native PipeWire/ALSA APIs) ────
        if name == "volume_up":
            return self._volume_change("+10%")
        if name == "volume_down":
            return self._volume_change("-10%")
        if name == "volume_set":
            pct = int(params.get("percent", params.get("level", 50)))
            return self._volume_set(pct)
        if name == "volume_mute":
            return self._volume_mute()

        # ── Brightness (TASK 9, native brightnessctl) ─────
        if name == "brightness_up":
            return self._brightness_change("+10%")
        if name == "brightness_down":
            return self._brightness_change("10%-")
        if name == "brightness_set":
            pct = int(params.get("percent", params.get("level", 70)))
            return self._brightness_set(pct)

        # ── Session / power (TASK 9, native logind) ───────
        if name == "lock_screen":
            return self._lock_screen()
        if name == "shutdown":
            return self._power("poweroff")
        if name == "restart":
            return self._power("reboot")

        logger.warning("[ACTIONS] Unknown action: %s", name)
        return None

    # ── TASK 9: native desktop operations ─────────────────

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
            from vision import vision_manager, CaptureSource
            if not vision_manager.is_available:
                vision_manager.initialize()
            # Try OCR first (fast, no model needed)
            import asyncio as _a
            text = _a.get_event_loop().run_until_complete(
                vision_manager.ocr_screen(CaptureSource.ACTIVE_WINDOW)
            ) if False else self._ocr_sync()
            if text:
                return f"On screen: {text[:500]}"
        except Exception as e:
            logger.debug("[ACTIONS] read_screen failed: %s", e)
        return "I couldn't read the screen right now."

    def _ocr_sync(self) -> str:
        """Synchronous OCR of the active window."""
        try:
            from vision import vision_manager, CaptureSource
            if not vision_manager.is_available:
                vision_manager.initialize()
            cap = vision_manager._screen_capturer.capture_active_window()
            if cap is None or cap.image is None:
                cap = vision_manager._screen_capturer.capture_full_screen()
            if cap is None or cap.image is None:
                return ""
            return vision_manager._ocr_engine.ocr(cap.image)
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
        # OCR snippet
        try:
            text = self._ocr_sync()
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
                # Best effort — return first line's title
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
        """Return (x, y) centroid of `text` on screen via OCR, or None."""
        try:
            import pytesseract
            from vision import vision_manager
            if not vision_manager.is_available:
                vision_manager.initialize()
            cap = vision_manager._screen_capturer.capture_full_screen()
            if cap is None or cap.image is None:
                return None
            data = pytesseract.image_to_data(cap.image, output_type=pytesseract.Output.DICT)
            text_lower = text.lower()
            n = len(data["text"])
            for i in range(n):
                if data["text"][i] and text_lower in data["text"][i].lower():
                    x = data["left"][i] + data["width"][i] // 2
                    y = data["top"][i] + data["height"][i] // 2
                    return (x, y)
        except Exception as e:
            logger.debug("[ACTIONS] locate_text failed: %s", e)
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
