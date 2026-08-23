"""
AutonomousReasoning — Auto-detect desktop context before asking questions.

When the user says "fix this", Diego should automatically determine:

  1. Active window (which app is focused)
  2. Project (which project/directory)
  3. Terminal (current directory, last command, error)
  4. Repository (current branch, recent commits)
  5. Current error (from terminal or screen OCR)
  6. Screen contents (what's visible)
  7. Clipboard (what was copied)

Without requiring the user to explain everything.

Architecture:
    User: "fix this"
        ↓
    AutoContext.collect() → desktop_state + terminal + screen + clipboard
        ↓
    Pass to decision_engine → LLM (with full context)
        ↓
    Response with action plan

Target: <50ms to collect all auto-context.

Usage:
    from core.autonomous_reasoning import auto_context

    ctx = await auto_context.collect()
    print(ctx.summary)  # "VS Code - Diego_desktop_agent | Terminal: ~/projects/Diego | Error: ImportError..."
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TerminalContext:
    """Current terminal state."""
    cwd: str = ""
    last_command: str = ""
    last_output: str = ""
    last_error: str = ""  # Extracted error message
    git_branch: str = ""
    git_status: str = ""
    python_env: str = ""  # Active venv/conda
    has_error: bool = False

    @property
    def summary(self) -> str:
        parts = []
        if self.cwd:
            parts.append(f"cwd={self.cwd}")
        if self.git_branch:
            parts.append(f"git={self.git_branch}")
        if self.last_error:
            parts.append(f"error={self.last_error[:80]}")
        if self.python_env:
            parts.append(f"env={self.python_env}")
        return " | ".join(parts) if parts else ""


@dataclass
class EditorContext:
    """Current editor/IDE state."""
    app: str = ""
    file_path: str = ""
    language: str = ""
    project_root: str = ""
    cursor_line: int = 0
    has_unsaved: bool = False

    @property
    def summary(self) -> str:
        parts = []
        if self.app:
            parts.append(self.app)
        if self.file_path:
            parts.append(os.path.basename(self.file_path))
        if self.project_root:
            parts.append(f"project={os.path.basename(self.project_root)}")
        return " | ".join(parts) if parts else ""


@dataclass
class BrowserContext:
    """Current browser state."""
    url: str = ""
    title: str = ""
    domain: str = ""

    @property
    def summary(self) -> str:
        if self.url:
            return f"{self.domain}: {self.title[:60]}"
        return ""


@dataclass
class AutoContext:
    """Complete autonomous context snapshot."""
    timestamp: float = field(default_factory=time.time)
    focused_window: str = ""
    focused_app: str = ""
    terminal: TerminalContext = field(default_factory=TerminalContext)
    editor: EditorContext = field(default_factory=EditorContext)
    browser: BrowserContext = field(default_factory=BrowserContext)
    clipboard: str = ""
    screen_summary: str = ""
    last_user_action: str = ""  # What the user just did
    errors_detected: List[str] = field(default_factory=list)

    # Latency tracking
    latency_ms: float = 0.0

    @property
    def summary(self) -> str:
        """One-line summary for the LLM context."""
        parts = []
        if self.focused_window:
            parts.append(f"Window: {self.focused_window[:80]}")
        if self.terminal.summary:
            parts.append(f"Terminal: {self.terminal.summary}")
        if self.editor.summary:
            parts.append(f"Editor: {self.editor.summary}")
        if self.browser.summary:
            parts.append(f"Browser: {self.browser.summary}")
        if self.errors_detected:
            parts.append(f"Errors: {'; '.join(self.errors_detected[:3])}")
        if self.clipboard:
            parts.append(f"Clipboard: {self.clipboard[:100]}")
        return "\n".join(parts) if parts else "No context available."

    @property
    def is_empty(self) -> bool:
        """True if no meaningful context was collected."""
        return not (self.focused_window or self.terminal.cwd or
                    self.editor.file_path or self.browser.url or
                    self.errors_detected)


class AutonomousContextCollector:
    """
    Collects desktop context for autonomous reasoning.

    Diego calls this BEFORE asking the user questions. If the context
    reveals enough information, Diego acts directly. Otherwise, it asks
    the minimal clarifying question.

    All methods are non-blocking and fast (<50ms total).
    """

    def __init__(self):
        self._last_collection_time: float = 0.0
        self._last_context: Optional[AutoContext] = None
        self._collection_count: int = 0

    # ── Main collection ─────────────────────────────────────

    async def collect(self, force: bool = False) -> AutoContext:
        """
        Collect all available desktop context.

        Args:
            force: If True, bypass cache and collect fresh data.

        Returns:
            AutoContext with all available information.
        """
        t0 = time.perf_counter_ns()

        # ── Cache check ──────────────────────────────────
        if not force and self._last_context is not None:
            age_ms = (time.time() - self._last_collection_time) * 1000
            if age_ms < 500:  # 500ms cache
                self._last_context.latency_ms = (
                    time.perf_counter_ns() - t0) / 1_000_000
                return self._last_context

        ctx = AutoContext()
        self._collection_count += 1

        # ── Collect in parallel where possible ────────────
        tasks = []

        # Desktop state (fast, sync)
        try:
            from services.desktop_state import desktop_state
            snap = desktop_state.snapshot()

            # Focused window
            if snap.focused_window:
                ctx.focused_window = snap.focused_window.title or ""
                ctx.focused_app = snap.focused_window.app_name or ""

            # Terminal context
            if snap.terminal:
                ctx.terminal.cwd = snap.terminal.cwd or ""
                ctx.terminal.git_branch = snap.terminal.git_branch or ""
                ctx.terminal.last_command = snap.terminal.last_command or ""

                # Try to get last error from terminal
                if snap.terminal.last_output:
                    ctx.terminal.last_output = snap.terminal.last_output[:500]
                    errors = self._extract_errors(snap.terminal.last_output)
                    if errors:
                        ctx.terminal.last_error = errors[0]
                        ctx.terminal.has_error = True
                        ctx.errors_detected.extend(errors[:3])

            # Browser context
            if snap.browser:
                ctx.browser.url = snap.browser.current_url or ""
                ctx.browser.title = snap.browser.title or ""
                if snap.browser.current_url:
                    ctx.browser.domain = self._extract_domain(
                        snap.browser.current_url)

        except Exception as e:
            logger.debug("[AUTO] Desktop state collection failed: %s", e)

        # ── Editor context ────────────────────────────────
        try:
            ctx.editor = self._detect_editor(ctx.focused_app, ctx.focused_window)
        except Exception as e:
            logger.debug("[AUTO] Editor detection failed: %s", e)

        # ── Clipboard ──────────────────────────────────────
        try:
            ctx.clipboard = await self._get_clipboard()
        except Exception as e:
            logger.debug("[AUTO] Clipboard collection failed: %s", e)

        # ── Terminal error (direct check) ──────────────────
        if not ctx.terminal.has_error:
            try:
                error = self._check_terminal_error_direct()
                if error:
                    ctx.terminal.last_error = error
                    ctx.terminal.has_error = True
                    ctx.errors_detected.append(error)
            except Exception as e:
                logger.debug("[AUTO] Terminal error check failed: %s", e)

        # ── Last user action ──────────────────────────────
        try:
            ctx.last_user_action = self._infer_last_action(ctx)
        except Exception as e:
            logger.debug("[AUTO] Action inference failed: %s", e)

        # Store and time
        ctx.latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
        self._last_context = ctx
        self._last_collection_time = time.time()

        if ctx.latency_ms > 100:
            logger.debug("[AUTO] Context collection slow: %.1fms", ctx.latency_ms)

        return ctx

    # ── Editor detection ────────────────────────────────────

    @staticmethod
    def _detect_editor(app_name: str, window_title: str) -> EditorContext:
        """Detect editor/IDE context from window information."""
        ctx = EditorContext()
        app_lower = app_name.lower()
        title = window_title or ""

        # VS Code / VSCodium
        if any(n in app_lower for n in ("code", "vscode", "vscodium")):
            ctx.app = "vscode"
            # Window title format: "filename.py - project - Visual Studio Code"
            parts = title.split(" - ")
            if parts:
                ctx.file_path = parts[0].strip()
                if len(parts) >= 2:
                    ctx.project_root = parts[-2].strip() if "Visual Studio" in parts[-1] else ""
            ctx.language = ctx._guess_language(ctx.file_path)

        # PyCharm
        elif "pycharm" in app_lower or "jetbrains" in app_lower:
            ctx.app = "pycharm"
            parts = title.split(" - ")
            if parts:
                file_part = parts[0].strip()
                if "[" in file_part:
                    # "filename.py [project] - ..."
                    file_match = re.match(r"^(.+?)\s*\[", file_part)
                    if file_match:
                        ctx.file_path = file_match.group(1).strip()
                    proj_match = re.search(r"\[(.+?)\]", file_part)
                    if proj_match:
                        ctx.project_root = proj_match.group(1)
                else:
                    ctx.file_path = file_part
                    if len(parts) >= 2:
                        ctx.project_root = parts[-1].strip().rstrip(".")
            ctx.language = ctx._guess_language(ctx.file_path)

        # Gedit / generic text editor
        elif "gedit" in app_lower or "editor" in app_lower:
            ctx.app = "gedit"
            ctx.file_path = title
            ctx.language = ctx._guess_language(title)

        # Terminal editors (vim, nano, emacs)
        elif any(e in app_lower for e in ("terminal", "gnome-terminal", "konsole",
                                           "alacritty", "xfce4-terminal")):
            if any(e in title for e in ("vim", "nvim", "nano", "emacs", "helix")):
                ctx.app = "terminal_editor"
                ctx.file_path = title.split(" - ")[0] if " - " in title else title
                ctx.language = ctx._guess_language(ctx.file_path)

        return ctx

    @staticmethod
    def _guess_language(filepath: str) -> str:
        """Guess language from file extension."""
        if not filepath:
            return ""
        ext = os.path.splitext(filepath)[1].lower()
        mapping = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "react", ".tsx": "react-ts", ".rs": "rust",
            ".go": "go", ".java": "java", ".cpp": "cpp", ".c": "c",
            ".h": "c", ".hpp": "cpp", ".css": "css", ".html": "html",
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".toml": "toml", ".md": "markdown", ".sql": "sql",
            ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
            ".rb": "ruby", ".php": "php", ".swift": "swift",
            ".kt": "kotlin", ".scala": "scala", ".dart": "dart",
            ".lua": "lua", ".r": "r", ".jl": "julia",
        }
        return mapping.get(ext, "")

    # ── Error extraction ────────────────────────────────────

    @staticmethod
    def _extract_errors(text: str) -> List[str]:
        """Extract error messages from terminal output."""
        errors = []

        # Python traceback
        if "Traceback (most recent call last)" in text:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if "Error:" in line or "Error " in line:
                    errors.append(line.strip())
                    # Get the last line (actual error message)
                if i == len(lines) - 1 and line.strip():
                    errors.append(line.strip())

        # Generic error patterns
        error_patterns = [
            r'(?:^|\n)((?:error|Error|ERROR)[: ].+?)(?:\n|$)',
            r'(?:^|\n)((?:fatal|Fatal|FATAL)[: ].+?)(?:\n|$)',
            r'(?:^|\n)((?:exception|Exception)[: ].+?)(?:\n|$)',
            r'(?:^|\n)((?:failed|Failed|FAILED)[: ].+?)(?:\n|$)',
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            errors.extend(m.strip() for m in matches)

        # Deduplicate and limit
        seen = set()
        unique = []
        for e in errors:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique[:5]

    @staticmethod
    def _check_terminal_error_direct() -> Optional[str]:
        """Check terminal directly for recent error (via wmctrl/xdotool)."""
        try:
            # Try to get the last terminal output via shell history
            result = subprocess.run(
                ["bash", "-c", "history 2 2>/dev/null | tail -1"],
                capture_output=True, text=True, timeout=2.0,
            )
            output = result.stdout.strip()
            if output and any(e in output.lower() for e in
                              ("error", "traceback", "failed", "exception")):
                return output[:200]
        except Exception:
            pass
        return None

    # ── Clipboard ───────────────────────────────────────────

    @staticmethod
    async def _get_clipboard() -> str:
        """Get clipboard content."""
        try:
            import subprocess
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=1.0,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:500]
        except Exception:
            pass

        try:
            import subprocess
            result = subprocess.run(
                ["wl-paste"],
                capture_output=True, text=True, timeout=1.0,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:500]
        except Exception:
            pass

        return ""

    # ── Domain extraction ───────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        match = re.search(r'https?://([^/]+)', url)
        if match:
            return match.group(1)
        match = re.search(r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', url)
        if match:
            return match.group(1)
        return url[:50]

    # ── Action inference ────────────────────────────────────

    @staticmethod
    def _infer_last_action(ctx: AutoContext) -> str:
        """Infer what the user was last doing."""
        if ctx.terminal.has_error:
            return f"User ran a command that produced an error: {ctx.terminal.last_error[:100]}"
        if ctx.terminal.last_command:
            return f"User last ran: {ctx.terminal.last_command[:100]}"
        if ctx.editor.file_path:
            return f"User was editing {os.path.basename(ctx.editor.file_path)}"
        if ctx.browser.url:
            return f"User was browsing {ctx.browser.domain}"
        if ctx.clipboard:
            return "User recently copied text"
        return ""

    # ── Quick query ─────────────────────────────────────────

    def quick_context_string(self) -> str:
        """Get a quick context string for the LLM (uses cached)."""
        ctx = self._last_context
        if ctx is None:
            return ""

        parts = []
        if ctx.focused_window:
            parts.append(f"[Desktop] {ctx.focused_window}")
        if ctx.editor.summary:
            parts.append(f"[Editor] {ctx.editor.summary}")
        if ctx.terminal.summary:
            parts.append(f"[Terminal] {ctx.terminal.summary}")
        if ctx.browser.summary:
            parts.append(f"[Browser] {ctx.browser.summary}")
        if ctx.errors_detected:
            parts.append(f"[Errors] {'; '.join(ctx.errors_detected[:2])}")

        return " | ".join(parts) if parts else ""

    # ── Problem detection ───────────────────────────────────

    def detect_problem(self) -> Optional[str]:
        """
        Auto-detect what problem the user is facing.

        Used when the user says "fix this" or "help".
        Returns a problem description or None.
        """
        ctx = self._last_context
        if ctx is None:
            return None

        # Error in terminal
        if ctx.terminal.has_error and ctx.terminal.last_error:
            return f"Terminal error in {ctx.terminal.cwd}: {ctx.terminal.last_error}"

        # Editor with unsaved changes + terminal error
        if ctx.editor.file_path and ctx.terminal.has_error:
            return (f"Error after editing {os.path.basename(ctx.editor.file_path)}: "
                    f"{ctx.terminal.last_error}")

        # Browser error page
        if ctx.browser.title and any(e in ctx.browser.title.lower()
                                      for e in ("error", "404", "500", "not found")):
            return f"Browser error: {ctx.browser.title} at {ctx.browser.url}"

        # No obvious problem
        return None

    # ── Stats ────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Return autonomous reasoning statistics."""
        ctx = self._last_context
        return {
            "collections": self._collection_count,
            "has_context": ctx is not None and not ctx.is_empty,
            "last_latency_ms": f"{ctx.latency_ms:.1f}" if ctx else "0",
            "focused_app": ctx.focused_app if ctx else "",
            "focused_window": (ctx.focused_window[:60] if ctx else ""),
            "terminal_active": bool(ctx.terminal.cwd) if ctx else False,
            "editor_active": bool(ctx.editor.file_path) if ctx else False,
            "browser_active": bool(ctx.browser.url) if ctx else False,
            "errors_detected": len(ctx.errors_detected) if ctx else 0,
            "clipboard_available": bool(ctx.clipboard) if ctx else False,
        }


# Global singleton
auto_context = AutonomousContextCollector()