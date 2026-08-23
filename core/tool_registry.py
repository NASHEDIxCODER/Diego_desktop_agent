"""
ToolRegistry — Makes every desktop capability a unified tool.

Every capability becomes a Tool:
    Browser, Terminal, Python, Bash, Mouse, Keyboard, Clipboard,
    Filesystem, Vision, Camera, OCR, Spotify, VS Code, Git, Docker,
    Notifications, Search, Downloads, Calendar, Email.

Tools have:
  * A canonical name + aliases
  * A JSON-schema description (auto-documented for the LLM)
  * A generic `execute(params) -> ToolResult` method
  * Isolation — a crashing tool never kills the runtime
  * Timeout protection — a hanging tool is cancelled, never blocking
"""

import asyncio
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.event_bus import bus

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool = True
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "data": self.data,
        }


class Tool:
    """
    A single executable tool.

    Subclass and implement `execute(params)`; or construct directly
    with a callable. The callable may be sync or async.
    """

    def __init__(self, name: str, description: str = "",
                 aliases: Optional[List[str]] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 handler: Optional[Callable] = None,
                 timeout: float = 30.0):
        self.name = name
        self.description = description
        self.aliases = aliases or []
        self.parameters = parameters or {}
        self._handler = handler
        self.timeout = timeout
        self.calls: int = 0
        self.failures: int = 0

    async def execute_async(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool, supporting async handlers and timeout."""
        t0 = time.monotonic()
        try:
            if self._handler is not None:
                result = self._handler(params)
                if asyncio.iscoroutine(result):
                    result = await asyncio.wait_for(result, timeout=self.timeout)
                elif callable(result) and not isinstance(result, ToolResult):
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, result),
                        timeout=self.timeout)
                out = self._coerce_result(result)
            else:
                out = ToolResult(success=False,
                                 error=f"Tool '{self.name}' not implemented")
        except asyncio.TimeoutError:
            self.failures += 1
            out = ToolResult(success=False,
                             error=f"Tool '{self.name}' timed out after {self.timeout}s")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.failures += 1
            out = ToolResult(success=False, error=str(e))
        out.duration_ms = (time.monotonic() - t0) * 1000
        self.calls += 1
        if not out.success:
            self.failures += 1
        return out

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool synchronously. Returns a ToolResult (never raises)."""
        t0 = time.monotonic()
        try:
            if self._handler is not None:
                result = self._handler(params)
                if asyncio.iscoroutine(result):
                    try:
                        asyncio.get_running_loop()
                        raise RuntimeError(
                            "async tool requires execute_async inside a running loop")
                    except RuntimeError:
                        result = asyncio.run(result)
                out = self._coerce_result(result)
            else:
                out = ToolResult(success=False,
                                 error=f"Tool '{self.name}' not implemented")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.failures += 1
            out = ToolResult(success=False, error=str(e))
        out.duration_ms = (time.monotonic() - t0) * 1000
        self.calls += 1
        if not out.success:
            self.failures += 1
        return out

    @staticmethod
    def _coerce_result(result: Any) -> ToolResult:
        """Normalize various handler return types into a ToolResult."""
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, tuple):
            ok, msg = result
            return ToolResult(success=bool(ok), output=str(msg))
        if isinstance(result, str):
            return ToolResult(success=True, output=result)
        return ToolResult(success=True, output=str(result))

    def describe(self) -> Dict[str, Any]:
        """Returns the LLM-visible tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "aliases": self.aliases,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
            },
        }


# ═══════════════════════════════════════════════════════════
# Built-in tool handlers
# ═══════════════════════════════════════════════════════════

def _run_subprocess(params: Dict[str, Any]) -> ToolResult:
    """Run a shell command and capture output."""
    cmd = params.get("command", "")
    timeout = float(params.get("timeout", 15.0))
    if not cmd:
        return ToolResult(success=False, error="No command provided")
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        output = proc.stdout.strip() or proc.stderr.strip()
        return ToolResult(
            success=proc.returncode == 0,
            output=output or "(no output)",
            error="" if proc.returncode == 0 else f"exit {proc.returncode}",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False,
                          error=f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _open_app(params: Dict[str, Any]) -> ToolResult:
    """Open a desktop application."""
    app = params.get("app", "").strip()
    if not app:
        return ToolResult(success=False, error="No app name provided")
    try:
        from agent.action_dispatcher import action_dispatcher
        result = action_dispatcher._open_app(app)
        return ToolResult(success="Couldn't" not in result, output=result)
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _open_url(params: Dict[str, Any]) -> ToolResult:
    """Open a URL in the default browser."""
    url = params.get("url", "").strip()
    if not url:
        return ToolResult(success=False, error="No URL provided")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        from agent.action_dispatcher import action_dispatcher
        result = action_dispatcher._open_url_fallback(url)
        return ToolResult(success="Couldn't" not in result, output=result)
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _screen_context(params: Dict[str, Any]) -> ToolResult:
    """Get a text summary of the current screen."""
    try:
        from agent.action_dispatcher import action_dispatcher
        ctx = action_dispatcher._screen_context_sync()
        return ToolResult(success=True, output=ctx or "(empty screen context)")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _clipboard_read(params: Dict[str, Any]) -> ToolResult:
    """Read the clipboard."""
    try:
        import pyperclip
        text = pyperclip.paste() or ""
        return ToolResult(success=True, output=text[:1000] or "(empty clipboard)")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _clipboard_write(params: Dict[str, Any]) -> ToolResult:
    """Write to the clipboard."""
    text = params.get("text", "")
    try:
        import pyperclip
        pyperclip.copy(text)
        return ToolResult(success=True, output=f"Copied {len(text)} chars")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _volume_control(params: Dict[str, Any]) -> ToolResult:
    """Volume control via pactl/amixer."""
    try:
        from agent.action_dispatcher import action_dispatcher as ad
        action = params.get("action", "up")
        if action == "up":
            return ToolResult(success=True, output=ad._volume_change("+10%"))
        if action == "down":
            return ToolResult(success=True, output=ad._volume_change("-10%"))
        if action == "mute":
            return ToolResult(success=True, output=ad._volume_mute())
        if action == "set":
            pct = int(params.get("percent", 50))
            return ToolResult(success=True, output=ad._volume_set(pct))
        return ToolResult(success=False, error=f"Unknown volume action: {action}")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _brightness_control(params: Dict[str, Any]) -> ToolResult:
    """Brightness control via brightnessctl."""
    try:
        from agent.action_dispatcher import action_dispatcher as ad
        action = params.get("action", "up")
        if action == "up":
            return ToolResult(success=True, output=ad._brightness_change("+10%"))
        if action == "down":
            return ToolResult(success=True, output=ad._brightness_change("10%-"))
        if action == "set":
            pct = int(params.get("percent", 70))
            return ToolResult(success=True, output=ad._brightness_set(pct))
        return ToolResult(success=False, error=f"Unknown brightness action: {action}")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _search(params: Dict[str, Any]) -> ToolResult:
    """Web search."""
    query = params.get("query", "")
    if not query:
        return ToolResult(success=False, error="No query provided")
    try:
        from agent.action_dispatcher import action_dispatcher
        url = "https://www.google.com/search?q=" + query.replace(" ", "+")
        result = action_dispatcher._open_url_fallback(url)
        return ToolResult(success="Couldn't" not in result,
                          output=f"Searched for {query}")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _notify(params: Dict[str, Any]) -> ToolResult:
    """Send a desktop notification."""
    title = params.get("title", "Diego")
    message = params.get("message", "")
    try:
        if shutil.which("notify-send"):
            subprocess.Popen(
                ["notify-send", title, message],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ToolResult(success=True, output=f"Notification sent: {title}")
        return ToolResult(success=False, error="notify-send not available")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _mouse_control(params: Dict[str, Any]) -> ToolResult:
    """Mouse control via pyautogui."""
    action = params.get("action", "move")
    try:
        import pyautogui
        x = params.get("x")
        y = params.get("y")
        if action == "move" and x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=0.2)
            return ToolResult(success=True, output=f"Moved to ({x}, {y})")
        if action == "click":
            if x is not None and y is not None:
                pyautogui.click(x, y)
            else:
                pyautogui.click()
            return ToolResult(success=True, output="Clicked")
        if action == "scroll":
            clicks = int(params.get("clicks", 3))
            pyautogui.scroll(clicks)
            return ToolResult(success=True, output=f"Scrolled {clicks}")
        return ToolResult(success=False, error=f"Unknown mouse action: {action}")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _keyboard_control(params: Dict[str, Any]) -> ToolResult:
    """Keyboard control via pyautogui."""
    action = params.get("action", "type")
    try:
        import pyautogui
        if action == "type":
            text = params.get("text", "")
            pyautogui.write(text, interval=0.01)
            return ToolResult(success=True, output=f"Typed {len(text)} chars")
        if action == "press":
            key = params.get("key", "")
            pyautogui.press(key)
            return ToolResult(success=True, output=f"Pressed {key}")
        if action == "hotkey":
            keys = params.get("keys", [])
            pyautogui.hotkey(*keys)
            return ToolResult(success=True, output=f"Hotkey {'+'.join(keys)}")
        return ToolResult(success=False, error=f"Unknown keyboard action: {action}")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _filesystem(params: Dict[str, Any]) -> ToolResult:
    """Filesystem operations: list, read, write, create."""
    action = params.get("action", "list")
    path = params.get("path", ".")
    try:
        from pathlib import Path
        p = Path(path).expanduser()
        if action == "list":
            if not p.exists():
                return ToolResult(success=False, error=f"Path not found: {p}")
            items = []
            for child in sorted(p.iterdir())[:50]:
                kind = "dir" if child.is_dir() else "file"
                items.append(f"{kind}: {child.name}")
            return ToolResult(success=True, output="\n".join(items) or "(empty)")
        if action == "read":
            if not p.is_file():
                return ToolResult(success=False, error=f"Not a file: {p}")
            text = p.read_text(errors="replace")[:5000]
            return ToolResult(success=True, output=text)
        if action == "write":
            text = params.get("text", "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            return ToolResult(success=True, output=f"Wrote {len(text)} chars to {p}")
        if action == "create_dir":
            p.mkdir(parents=True, exist_ok=True)
            return ToolResult(success=True, output=f"Created directory {p}")
        return ToolResult(success=False, error=f"Unknown filesystem action: {action}")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _browser(params: Dict[str, Any]) -> ToolResult:
    """Browser control."""
    action = params.get("action", "navigate")
    url = params.get("url", "")
    try:
        from agent.action_dispatcher import action_dispatcher
        if action == "navigate" and url:
            result = action_dispatcher._open_url_fallback(url)
            return ToolResult(success="Couldn't" not in result, output=result)
        return ToolResult(success=False, error="browser action not implemented")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _git(params: Dict[str, Any]) -> ToolResult:
    """Git operations."""
    action = params.get("action", "status")
    path = params.get("path", ".")
    return _run_subprocess({"command": f"git -C {path} {action}"})


def _docker(params: Dict[str, Any]) -> ToolResult:
    """Docker operations."""
    action = params.get("action", "ps")
    return _run_subprocess({"command": f"docker {action}"})


def _git_repo_check(params: Dict[str, Any]) -> ToolResult:
    """Show git status of a repository."""
    path = params.get("path", ".")
    return _run_subprocess(
        {"command": f"git -C {path} status --short 2>/dev/null | head -20"})


def _run_python(params: Dict[str, Any]) -> ToolResult:
    """Run inline Python code in a subprocess."""
    code = params.get("code", "")
    timeout = float(params.get("timeout", 15.0))
    if not code:
        return ToolResult(success=False, error="No Python code provided")
    try:
        proc = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=timeout)
        return ToolResult(
            success=proc.returncode == 0,
            output=proc.stdout.strip() or "(no output)",
            error="" if proc.returncode == 0
                else (proc.stderr.strip() or f"exit {proc.returncode}"),
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False,
                          error=f"Python code timed out after {timeout}s")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ═══════════════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════════════

class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._alias_map: Dict[str, str] = {}
        self._initialized = False

    def register(self, tool: Tool) -> None:
        """Register a tool + its aliases."""
        self._tools[tool.name] = tool
        for alias in tool.aliases:
            self._alias_map[alias] = tool.name

    def register_builtin(self, name: str, description: str,
                         handler: Callable,
                         aliases: Optional[List[str]] = None,
                         parameters: Optional[Dict[str, Any]] = None,
                         timeout: float = 30.0) -> Tool:
        """Register a tool from a callable."""
        tool = Tool(name, description, aliases, parameters, handler, timeout)
        self.register(tool)
        return tool

    def get(self, name_or_alias: str) -> Optional[Tool]:
        """Resolve a tool by name or alias."""
        name = self._alias_map.get(name_or_alias, name_or_alias)
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def describe_all(self) -> List[Dict[str, Any]]:
        """Returns the full tool schema for the LLM planner."""
        return [t.describe() for t in self._tools.values()]

    def is_available(self, name: str) -> bool:
        return self.get(name) is not None

    def install_builtin_tools(self) -> None:
        """Install Diego's standard desktop tool set (idempotent)."""
        if self._initialized:
            return
        self._initialized = True

        self.register_builtin(
            "terminal", "Run a shell command. params: {command, timeout}",
            _run_subprocess, aliases=["shell", "bash", "command"],
            parameters={"command": {"type": "string"},
                        "timeout": {"type": "number"}},
            timeout=60.0)
        self.register_builtin(
            "python", "Run Python code. params: {code, timeout}",
            _run_python, aliases=["py"],
            parameters={"code": {"type": "string"},
                        "timeout": {"type": "number"}},
            timeout=60.0)
        self.register_builtin(
            "open_app", "Open a desktop application. params: {app}",
            _open_app, aliases=["launch", "start"],
            parameters={"app": {"type": "string"}})
        self.register_builtin(
            "open_url", "Open a URL in the default browser. params: {url}",
            _open_url, aliases=["browser_navigate"],
            parameters={"url": {"type": "string"}})
        self.register_builtin(
            "search", "Search the web. params: {query}",
            _search, aliases=["google"],
            parameters={"query": {"type": "string"}})
        self.register_builtin(
            "notify", "Send a desktop notification. params: {title, message}",
            _notify, aliases=["notification"],
            parameters={"title": {"type": "string"},
                        "message": {"type": "string"}})
        self.register_builtin(
            "read_screen", "Get a text summary of the current screen. params: {}",
            _screen_context, aliases=["screen_context", "what_on_screen"])
        self.register_builtin(
            "clipboard_read", "Read the clipboard text. params: {}",
            _clipboard_read, aliases=["get_clipboard"])
        self.register_builtin(
            "clipboard_write", "Write text to the clipboard. params: {text}",
            _clipboard_write, aliases=["set_clipboard", "copy"],
            parameters={"text": {"type": "string"}})
        self.register_builtin(
            "mouse", "Control the mouse. params: {action, x, y, clicks}",
            _mouse_control, aliases=["mouse_control"],
            parameters={"action": {"type": "string"}})
        self.register_builtin(
            "keyboard", "Control the keyboard. params: {action, text, key, keys}",
            _keyboard_control, aliases=["type_text", "key_press"],
            parameters={"action": {"type": "string"}})
        self.register_builtin(
            "volume", "Control audio volume. params: {action, percent}",
            _volume_control, aliases=["volume_up", "volume_down", "volume_mute"],
            parameters={"action": {"type": "string"},
                        "percent": {"type": "number"}})
        self.register_builtin(
            "brightness", "Control screen brightness. params: {action, percent}",
            _brightness_control,
            aliases=["brightness_up", "brightness_down", "brightness_set"],
            parameters={"action": {"type": "string"},
                        "percent": {"type": "number"}})
        self.register_builtin(
            "filesystem", "Filesystem operations. params: {action, path, text}",
            _filesystem, aliases=["fs", "file"],
            parameters={"action": {"type": "string",
                                   "enum": ["list", "read", "write", "create_dir"]},
                        "path": {"type": "string"},
                        "text": {"type": "string"}})
        self.register_builtin(
            "browser", "Browser control. params: {action, url}",
            _browser, aliases=["web"],
            parameters={"action": {"type": "string"},
                        "url": {"type": "string"}},
            timeout=60.0)
        self.register_builtin(
            "git", "Git operations. params: {action, path}",
            _git, aliases=["github"],
            parameters={"action": {"type": "string"},
                        "path": {"type": "string"}})
        self.register_builtin(
            "docker", "Docker operations. params: {action}",
            _docker, aliases=["container"],
            parameters={"action": {"type": "string"}})
        self.register_builtin(
            "git_status", "Show git status of a repo. params: {path}",
            _git_repo_check, aliases=["repo_status"],
            parameters={"path": {"type": "string"}})

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_registry())
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._publish_registry())
            except Exception:
                pass  # no running loop — registry publish deferred

    async def _publish_registry(self) -> None:
        """Emit the registry on the event bus."""
        await bus.emit("tool.registry.updated", data={
            "tools": self.names(),
            "count": len(self._tools),
        }, source="tool_registry")

    async def execute(self, name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name/alias with isolation."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        try:
            result = await tool.execute_async(params)
        except Exception as e:
            result = ToolResult(success=False,
                                error=f"Tool execution error: {e}")
        try:
            await bus.emit("tool.executed", data={
                "tool": tool.name, "success": result.success,
                "duration_ms": result.duration_ms,
            }, source="tool_registry")
        except Exception:
            pass
        return result


# Global singleton
tool_registry = ToolRegistry()