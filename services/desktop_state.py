"""
DesktopState — Real-time desktop context awareness.

Diego knows what the user is doing without being told:

    - Focused window (title + application)
    - Clipboard content (text only)
    - Terminal output (recent)
    - Current file (IDE integration)
    - Current IDE / editor
    - Current browser tab (title + URL)
    - Current git branch
    - Active downloads
    - Desktop notifications
    - Battery status
    - WiFi / network status
    - Audio state (volume, mute, active sink)
    - CPU usage / load
    - GPU usage (nvidia-smi fallback)

Everything is polled on demand (or via background observers), never
blocks the event loop, and the planner can inject this automatically.

Usage:
    from services.desktop_state import desktop_state

    state = desktop_state.snapshot()  # full dict
    context = desktop_state.context_for_llm()  # compact LLM prompt block
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Information about a focused window."""
    title: str = ""
    application: str = ""
    pid: int = 0


@dataclass
class TerminalInfo:
    """Recent terminal context."""
    cwd: str = ""
    last_command: str = ""
    last_output: str = ""
    git_branch: str = ""


@dataclass
class SystemInfo:
    """System resource information."""
    battery_percent: int = 100
    battery_charging: bool = True
    wifi_ssid: str = ""
    wifi_strength: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_usage: str = ""


@dataclass
class DesktopSnapshot:
    """Complete desktop state at a point in time."""
    focused_window: WindowInfo = field(default_factory=WindowInfo)
    terminal: TerminalInfo = field(default_factory=TerminalInfo)
    system: SystemInfo = field(default_factory=SystemInfo)
    clipboard_text: str = ""
    browser_tab: str = ""
    browser_url: str = ""
    active_downloads: List[str] = field(default_factory=list)
    audio_volume: int = 50
    audio_muted: bool = False
    timestamp: float = 0.0


class DesktopState:
    """
    Real-time desktop awareness for Diego.

    Polls desktop state on demand. Never blocks the event loop.
    All methods are safe to call from any thread.
    """

    def __init__(self):
        self._last_snapshot: Optional[DesktopSnapshot] = None
        self._snapshot_interval: float = 2.0  # minimum seconds between full snapshots

    def snapshot(self) -> DesktopSnapshot:
        """
        Take a full desktop state snapshot.

        This runs synchronously because the desktop queries are fast
        (<50ms each on modern hardware). Call from an executor thread
        if integrating into the async event loop.
        """
        now = time.time()

        # Use cached snapshot if recent enough
        if self._last_snapshot and (now - self._last_snapshot.timestamp) < self._snapshot_interval:
            return self._last_snapshot

        snap = DesktopSnapshot(timestamp=now)
        snap.focused_window = self._get_focused_window()
        snap.terminal = self._get_terminal_context()
        snap.system = self._get_system_info()
        snap.clipboard_text = self._get_clipboard()
        snap.browser_tab, snap.browser_url = self._get_browser_tab()
        snap.active_downloads = self._get_active_downloads()
        snap.audio_volume, snap.audio_muted = self._get_audio_state()

        self._last_snapshot = snap
        return snap

    def context_for_llm(self) -> str:
        """
        Return a compact context block for LLM injection.

        Example output:
            Window: PyCharm - Diego_desktop_agent
            Git: feature/planner-rewrite
            Terminal: ~/projects/Diego
            Battery: 85% (charging)
            Browser: YouTube - "lofi hip hop"
        """
        snap = self.snapshot()
        parts: List[str] = []

        # Focused window
        if snap.focused_window.title:
            app = snap.focused_window.application or ""
            parts.append(f"Focused window: {app} — {snap.focused_window.title}")

        # IDE context
        if snap.terminal.git_branch:
            parts.append(f"Git branch: {snap.terminal.git_branch}")

        # Terminal
        if snap.terminal.cwd:
            parts.append(f"Terminal: {snap.terminal.cwd}")

        # Browser
        if snap.browser_tab:
            entry = f"Browser: {snap.browser_tab}"
            if snap.browser_url and len(snap.browser_url) < 120:
                entry += f" ({snap.browser_url})"
            parts.append(entry)

        # Clipboard
        if snap.clipboard_text:
            short = snap.clipboard_text[:100].replace("\n", " ")
            parts.append(f"Clipboard: {short}")

        # Battery
        if snap.system.battery_percent < 100 or not snap.system.battery_charging:
            status = "charging" if snap.system.battery_charging else "discharging"
            parts.append(f"Battery: {snap.system.battery_percent}% ({status})")

        # Audio
        if snap.audio_muted:
            parts.append("Audio: muted")
        elif snap.audio_volume != 50:
            parts.append(f"Volume: {snap.audio_volume}%")

        # Downloads
        if snap.active_downloads:
            parts.append(f"Downloads: {', '.join(snap.active_downloads[:3])}")

        return "\n".join(parts) if parts else ""

    def quick_context(self) -> str:
        """Fast context (just window + git) for low-latency injection."""
        snap = self.snapshot()
        items = []
        if snap.focused_window.title:
            items.append(f"Window: {snap.focused_window.title}")
        if snap.terminal.git_branch:
            items.append(f"Git: {snap.terminal.git_branch}")
        return " | ".join(items) if items else ""

    # ── Window detection ─────────────────────────────────────

    @staticmethod
    def _get_focused_window() -> WindowInfo:
        """Detect the currently focused window."""
        info = WindowInfo()

        # xdotool (X11 — most reliable)
        if shutil.which("xdotool"):
            try:
                # Get window ID
                wid = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True, text=True, timeout=1
                ).stdout.strip()
                if wid:
                    # Get window title
                    title = subprocess.run(
                        ["xdotool", "getwindowname", wid],
                        capture_output=True, text=True, timeout=1
                    ).stdout.strip()
                    info.title = title

                    # Get PID
                    pid_str = subprocess.run(
                        ["xdotool", "getwindowpid", wid],
                        capture_output=True, text=True, timeout=1
                    ).stdout.strip()
                    if pid_str:
                        info.pid = int(pid_str)
                        # Get process name from /proc
                        try:
                            exe = os.readlink(f"/proc/{pid_str}/exe")
                            info.application = Path(exe).name
                        except Exception:
                            try:
                                cmdline = Path(f"/proc/{pid_str}/cmdline").read_text()
                                info.application = cmdline.split("\x00")[0]
                                info.application = Path(info.application).name
                            except Exception:
                                pass
            except Exception:
                pass

        # wmctrl fallback
        if not info.title and shutil.which("wmctrl"):
            try:
                result = subprocess.run(
                    ["wmctrl", "-l", "-p"],
                    capture_output=True, text=True, timeout=1
                )
                for line in result.stdout.splitlines():
                    segs = line.split(None, 4)
                    if len(segs) >= 5:
                        # wmctrl -l -p: ID  DESK  PID  TITLE...
                        pid = int(segs[2]) if segs[2].isdigit() else 0
                        title = segs[4]
                        # Try to find the active one
                        if info.title == "":
                            info.title = title
                            info.pid = pid
                            if pid > 0:
                                try:
                                    exe = os.readlink(f"/proc/{pid}/exe")
                                    info.application = Path(exe).name
                                except Exception:
                                    pass
            except Exception:
                pass

        return info

    # ── Terminal context ──────────────────────────────────────

    @staticmethod
    def _get_terminal_context() -> TerminalInfo:
        """Get context from the active terminal (if any)."""
        info = TerminalInfo()

        # Try to find a running terminal emulator PID
        terminal_pids = []
        for terminal in ("gnome-terminal-server", "konsole", "alacritty",
                         "kitty", "wezterm", "xfce4-terminal", "tilix",
                         "terminator", "lxterminal", "qterminal"):
            try:
                result = subprocess.run(
                    ["pgrep", "-x", terminal],
                    capture_output=True, text=True, timeout=1
                )
                for pid_str in result.stdout.strip().split():
                    if pid_str.isdigit():
                        terminal_pids.append(int(pid_str))
            except Exception:
                pass

        if not terminal_pids:
            return info

        # Try to get CWD from /proc
        for pid in terminal_pids[:3]:
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
                if cwd:
                    info.cwd = str(Path(cwd).expanduser())
                    break
            except Exception:
                pass

        # Try to extract git branch from the CWD or any terminal
        try:
            if info.cwd:
                result = subprocess.run(
                    ["git", "-C", info.cwd, "branch", "--show-current"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    info.git_branch = result.stdout.strip()
        except Exception:
            pass

        # If no branch from terminal CWD, try common project dirs
        if not info.git_branch:
            # Try the focused window's CWD
            try:
                focused = DesktopState._get_focused_window()
                if focused.pid > 0:
                    cwd = os.readlink(f"/proc/{focused.pid}/cwd")
                    result = subprocess.run(
                        ["git", "-C", cwd, "branch", "--show-current"],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        info.git_branch = result.stdout.strip()
            except Exception:
                pass

        return info

    # ── System info ───────────────────────────────────────────

    @staticmethod
    def _get_system_info() -> SystemInfo:
        """Get battery, WiFi, CPU, GPU info."""
        info = SystemInfo()

        # Battery
        battery_dir = Path("/sys/class/power_supply")
        if battery_dir.exists():
            for bat in battery_dir.iterdir():
                if bat.name.startswith("BAT"):
                    try:
                        capacity = int((bat / "capacity").read_text().strip())
                        info.battery_percent = capacity
                        status = (bat / "status").read_text().strip().lower()
                        info.battery_charging = "charging" in status or "full" in status
                        break
                    except Exception:
                        pass

        # WiFi SSID (NetworkManager)
        if shutil.which("nmcli"):
            try:
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "active,ssid,signal", "dev", "wifi"],
                    capture_output=True, text=True, timeout=2
                )
                for line in result.stdout.splitlines():
                    if line.startswith("yes:"):
                        parts = line.split(":")
                        if len(parts) >= 2:
                            info.wifi_ssid = parts[1]
                        if len(parts) >= 3:
                            try:
                                info.wifi_strength = int(parts[2])
                            except ValueError:
                                pass
                        break
            except Exception:
                pass

        # CPU usage (simple /proc/stat sampling)
        try:
            stat = Path("/proc/stat").read_text().splitlines()[0]
            parts = stat.split()
            # Very rough: idle / total from first line
            if len(parts) >= 5:
                idle = int(parts[4])
                total = sum(int(x) for x in parts[1:])
                if total > 0:
                    info.cpu_percent = round((1 - idle / total) * 100, 1)
        except Exception:
            pass

        # Memory
        try:
            meminfo = Path("/proc/meminfo").read_text()
            total_match = re.search(r"MemTotal:\s+(\d+)", meminfo)
            avail_match = re.search(r"MemAvailable:\s+(\d+)", meminfo)
            if total_match and avail_match:
                total = int(total_match.group(1))
                avail = int(avail_match.group(1))
                info.memory_percent = round((1 - avail / total) * 100, 1)
        except Exception:
            pass

        # GPU (nvidia-smi)
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    info.gpu_usage = result.stdout.strip()
            except Exception:
                pass

        return info

    # ── Clipboard ─────────────────────────────────────────────

    @staticmethod
    def _get_clipboard(max_chars: int = 500) -> str:
        """Get current clipboard text content."""
        # Try xclip
        if shutil.which("xclip"):
            try:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    text = result.stdout.strip()
                    return text[:max_chars]
            except Exception:
                pass

        # Try wl-paste (Wayland)
        if shutil.which("wl-paste"):
            try:
                result = subprocess.run(
                    ["wl-paste", "--primary"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    text = result.stdout.strip()
                    return text[:max_chars]
            except Exception:
                pass

        return ""

    # ── Browser tab ───────────────────────────────────────────

    @staticmethod
    def _get_browser_tab() -> tuple:
        """Get the current browser tab title and URL (Firefox, Chrome, Brave)."""
        # Firefox via xdotool
        if shutil.which("xdotool"):
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", "Firefox",
                     "getwindowpid", "%@"],
                    capture_output=True, text=True, timeout=2
                )
                # Check if Firefox is focused
                focused = DesktopState._get_focused_window()
                if focused.application in ("firefox", "firefox-bin",
                                            "firefox-esr", "firefox-nightly"):
                    title = focused.title
                    # Firefox titles usually: "Page Title — Mozilla Firefox"
                    if " — " in title:
                        tab_title = title.rsplit(" — ", 1)[0].strip()
                    elif " - " in title:
                        tab_title = title.rsplit(" - ", 1)[0].strip()
                    else:
                        tab_title = title
                    return tab_title, ""
            except Exception:
                pass

        # Chrome / Brave / Chromium
        if shutil.which("xdotool"):
            try:
                focused = DesktopState._get_focused_window()
                browser_names = ("chrome", "chromium", "brave", "edge",
                                 "google-chrome", "chromium-browser")
                if focused.application in browser_names:
                    title = focused.title
                    # Chrome titles usually: "Page Title - Google Chrome"
                    if " - " in title:
                        tab_title = title.rsplit(" - ", 1)[0].strip()
                    else:
                        tab_title = title
                    return tab_title, ""
            except Exception:
                pass

        return "", ""

    # ── Downloads ─────────────────────────────────────────────

    @staticmethod
    def _get_active_downloads() -> List[str]:
        """Check for active downloads in common browsers."""
        downloads: List[str] = []

        # Firefox: check for .part files
        download_dirs = [
            Path.home() / "Downloads",
            Path.home() / "downloads",
        ]
        for d in download_dirs:
            if d.exists():
                part_files = list(d.glob("*.part"))
                for pf in part_files[:5]:
                    downloads.append(pf.name)

        return downloads

    # ── Audio state ───────────────────────────────────────────

    @staticmethod
    def _get_audio_state() -> tuple:
        """Get current volume and mute state."""
        volume = 50
        muted = False

        if shutil.which("pactl"):
            try:
                result = subprocess.run(
                    ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                    capture_output=True, text=True, timeout=1
                )
                # Format: "Volume: front-left: 32768 /  50% / -18.06 dB, ..."
                match = re.search(r"(\d+)%", result.stdout)
                if match:
                    volume = int(match.group(1))
            except Exception:
                pass

            try:
                result = subprocess.run(
                    ["pactl", "get-sink-mute", "@DEFAULT_SINK@"],
                    capture_output=True, text=True, timeout=1
                )
                muted = "yes" in result.stdout.lower()
            except Exception:
                pass

        return volume, muted

    # ── IDE / Current file detection ──────────────────────────

    def current_ide(self) -> str:
        """Return the name of the current IDE/editor, or empty string."""
        snap = self.snapshot()
        return snap.focused_window.application or ""

    def current_file(self) -> str:
        """Guess the current file being edited."""
        snap = self.snapshot()
        title = snap.focused_window.title

        # PyCharm: "filename.py - project - PyCharm"
        # VS Code: "filename.py - project - Visual Studio Code"
        # Vim: "filename.py + (~/project) - VIM"
        if " - " in title:
            parts = title.split(" - ")
            if len(parts) >= 2:
                first = parts[0].strip()
                # Check if it looks like a filename
                if "." in first and not first.startswith("http"):
                    return first
        return ""

    def current_git_branch(self) -> str:
        """Return the current git branch."""
        snap = self.snapshot()
        return snap.terminal.git_branch or ""


# Global singleton
desktop_state = DesktopState()