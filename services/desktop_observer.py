"""
DesktopObserver — Continuous background desktop monitoring.

Non-blocking observer that publishes change events to the EventBus:
  - Focused window changes (title, application)
  - Clipboard changes
  - Browser tab changes (title, URL)
  - Terminal activity (CWD changes, new commands)
  - IDE/file changes
  - Git branch changes
  - Download start/complete
  - Notification detection
  - System resource changes (CPU, memory, battery, WiFi, GPU)
  - Audio state changes (volume, mute)
  - Running application changes

The observer NEVER blocks — it runs in a background thread with
configurable sampling intervals. Change detection uses content hashing
to avoid spamming identical events.

Events published:
  desktop:window_changed   — {title, application, pid}
  desktop:clipboard_changed — {text_preview, length}
  desktop:browser_changed   — {tab_title, url}
  desktop:git_changed       — {branch, repo_path}
  desktop:download_started  — {filename}
  desktop:download_complete — {filename}
  desktop:battery_low       — {percent, charging}
  desktop:audio_changed     — {volume, muted}
  desktop:system_load       — {cpu_percent, ram_percent, gpu_util}
  desktop:app_launched      — {app_name}
  desktop:app_closed        — {app_name}

Usage:
    from services.desktop_observer import desktop_observer

    await desktop_observer.start(event_bus)
    await desktop_observer.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Observable State
# ═══════════════════════════════════════════════════════════════

@dataclass
class WindowState:
    """Last observed focused window state."""
    title: str = ""
    application: str = ""
    pid: int = 0
    hash: str = ""


@dataclass
class ClipboardState:
    """Last observed clipboard state."""
    text: str = ""
    hash: str = ""


@dataclass
class BrowserState:
    """Last observed browser tab state."""
    title: str = ""
    url: str = ""
    hash: str = ""


@dataclass
class SystemState:
    """Last observed system resource state."""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    battery_percent: int = 100
    battery_charging: bool = True
    wifi_ssid: str = ""
    wifi_strength: int = 0
    gpu_util: float = 0.0
    audio_volume: int = 50
    audio_muted: bool = False


@dataclass
class DesktopState:
    """Complete observable desktop state snapshot."""
    window: WindowState = field(default_factory=WindowState)
    clipboard: ClipboardState = field(default_factory=ClipboardState)
    browser: BrowserState = field(default_factory=BrowserState)
    system: SystemState = field(default_factory=SystemState)
    git_branch: str = ""
    terminal_cwd: str = ""
    downloads: Dict[str, float] = field(default_factory=dict)  # filename → last_seen_time
    running_apps: Set[str] = field(default_factory=set)

    def diff(self, previous: Optional["DesktopState"]) -> List[Tuple[str, Dict[str, Any]]]:
        """Compare with a previous state and return changed events."""
        if previous is None:
            return []

        events: List[Tuple[str, Dict[str, Any]]] = []

        # Window change
        if self.window.hash != previous.window.hash and self.window.title:
            events.append(("desktop:window_changed", {
                "title": self.window.title,
                "application": self.window.application,
                "pid": self.window.pid,
            }))

        # Clipboard change
        if self.clipboard.hash != previous.clipboard.hash and self.clipboard.text:
            preview = self.clipboard.text[:200]
            events.append(("desktop:clipboard_changed", {
                "text_preview": preview,
                "length": len(self.clipboard.text),
            }))

        # Browser tab change
        if self.browser.hash != previous.browser.hash and self.browser.title:
            events.append(("desktop:browser_changed", {
                "tab_title": self.browser.title,
                "url": self.browser.url,
            }))

        # Git branch change
        if self.git_branch and self.git_branch != previous.git_branch:
            events.append(("desktop:git_changed", {
                "branch": self.git_branch,
                "previous": previous.git_branch or "none",
            }))

        # New downloads
        new_downloads = set(self.downloads) - set(previous.downloads)
        for dl in new_downloads:
            events.append(("desktop:download_started", {"filename": dl}))

        # Completed downloads (removed from current but were in previous)
        completed = set(previous.downloads) - set(self.downloads)
        for dl in completed:
            events.append(("desktop:download_complete", {"filename": dl}))

        # Battery low warning
        if self.system.battery_percent <= 15 and not self.system.battery_charging:
            if previous.system.battery_percent > 15 or previous.system.battery_charging:
                events.append(("desktop:battery_low", {
                    "percent": self.system.battery_percent,
                    "charging": self.system.battery_charging,
                }))

        # Audio change
        if (self.system.audio_volume != previous.system.audio_volume or
                self.system.audio_muted != previous.system.audio_muted):
            events.append(("desktop:audio_changed", {
                "volume": self.system.audio_volume,
                "muted": self.system.audio_muted,
            }))

        # New applications launched
        new_apps = self.running_apps - previous.running_apps
        for app in new_apps:
            events.append(("desktop:app_launched", {"app_name": app}))

        # Applications closed
        closed_apps = previous.running_apps - self.running_apps
        for app in closed_apps:
            events.append(("desktop:app_closed", {"app_name": app}))

        return events


# ═══════════════════════════════════════════════════════════════
# DesktopObserver
# ═══════════════════════════════════════════════════════════════

class DesktopObserver:
    """
    Continuous background desktop state monitor.

    Runs in a background thread, sampling desktop state at configurable
    intervals. Detects changes by comparing state hashes and publishes
    events to the EventBus.

    NEVER blocks the main thread or event loop.
    """

    def __init__(
        self,
        window_interval: float = 0.5,
        system_interval: float = 5.0,
        clipboard_interval: float = 1.0,
        browser_interval: float = 2.0,
        git_interval: float = 10.0,
        downloads_interval: float = 3.0,
    ):
        self._window_interval = window_interval
        self._system_interval = system_interval
        self._clipboard_interval = clipboard_interval
        self._browser_interval = browser_interval
        self._git_interval = git_interval
        self._downloads_interval = downloads_interval

        self._event_bus = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Current state (thread-safe — only written by observer thread)
        self._state = DesktopState()
        self._previous_state: Optional[DesktopState] = None

        # Callbacks for extended monitoring
        self._on_window_change: Optional[Callable[[WindowState], Any]] = None
        self._on_clipboard_change: Optional[Callable[[ClipboardState], Any]] = None
        self._on_browser_change: Optional[Callable[[BrowserState], Any]] = None
        self._on_git_change: Optional[Callable[[str], Any]] = None
        self._on_download: Optional[Callable[[str, bool], Any]] = None  # (filename, started)
        self._on_battery_low: Optional[Callable[[int], Any]] = None  # (percent)
        self._on_audio_change: Optional[Callable[[int, bool], Any]] = None  # (volume, muted)
        self._on_app_change: Optional[Callable[[str, bool], Any]] = None  # (app_name, launched)

    # ── Wiring ─────────────────────────────────────────────────

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    def on_window_change(self, fn: Callable[[WindowState], Any]) -> None:
        self._on_window_change = fn

    def on_clipboard_change(self, fn: Callable[[ClipboardState], Any]) -> None:
        self._on_clipboard_change = fn

    def on_browser_change(self, fn: Callable[[BrowserState], Any]) -> None:
        self._on_browser_change = fn

    def on_git_change(self, fn: Callable[[str], Any]) -> None:
        self._on_git_change = fn

    def on_download(self, fn: Callable[[str, bool], Any]) -> None:
        self._on_download = fn

    def on_battery_low(self, fn: Callable[[int], Any]) -> None:
        self._on_battery_low = fn

    def on_audio_change(self, fn: Callable[[int, bool], Any]) -> None:
        self._on_audio_change = fn

    def on_app_change(self, fn: Callable[[str, bool], Any]) -> None:
        self._on_app_change = fn

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self, event_bus=None) -> None:
        """Start the background observer."""
        if self._running:
            return

        if event_bus:
            self._event_bus = event_bus

        self._running = True
        self._task = asyncio.create_task(self._observe_loop())
        logger.info("[Observer] Desktop observer started (window=%.1fs, system=%.1fs)",
                     self._window_interval, self._system_interval)

    async def stop(self) -> None:
        """Stop the observer gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[Observer] Desktop observer stopped")

    async def _observe_loop(self) -> None:
        """Main observation loop. Each sensor runs at its own interval."""
        last_window = 0.0
        last_clipboard = 0.0
        last_browser = 0.0
        last_system = 0.0
        last_git = 0.0
        last_downloads = 0.0

        try:
            while self._running:
                now = time.time()
                loop = asyncio.get_event_loop()

                # Window monitoring (fastest)
                if now - last_window >= self._window_interval:
                    await loop.run_in_executor(None, self._sample_window)
                    last_window = now

                # Clipboard monitoring
                if now - last_clipboard >= self._clipboard_interval:
                    await loop.run_in_executor(None, self._sample_clipboard)
                    last_clipboard = now

                # Browser monitoring
                if now - last_browser >= self._browser_interval:
                    await loop.run_in_executor(None, self._sample_browser)
                    last_browser = now

                # System monitoring (slowest)
                if now - last_system >= self._system_interval:
                    await loop.run_in_executor(None, self._sample_system)
                    last_system = now

                # Git monitoring
                if now - last_git >= self._git_interval:
                    await loop.run_in_executor(None, self._sample_git)
                    last_git = now

                # Download monitoring
                if now - last_downloads >= self._downloads_interval:
                    await loop.run_in_executor(None, self._sample_downloads)
                    last_downloads = now

                # Compute diff and publish events
                if self._previous_state is not None:
                    events = self._state.diff(self._previous_state)
                    for event_type, data in events:
                        await self._publish(event_type, data)
                        await self._dispatch_callbacks(event_type, data)

                # Save previous state
                self._previous_state = DesktopState(
                    window=WindowState(
                        title=self._state.window.title,
                        application=self._state.window.application,
                        pid=self._state.window.pid,
                        hash=self._state.window.hash,
                    ),
                    clipboard=ClipboardState(
                        text=self._state.clipboard.text,
                        hash=self._state.clipboard.hash,
                    ),
                    browser=BrowserState(
                        title=self._state.browser.title,
                        url=self._state.browser.url,
                        hash=self._state.browser.hash,
                    ),
                    system=SystemState(
                        cpu_percent=self._state.system.cpu_percent,
                        ram_percent=self._state.system.ram_percent,
                        battery_percent=self._state.system.battery_percent,
                        battery_charging=self._state.system.battery_charging,
                        wifi_ssid=self._state.system.wifi_ssid,
                        wifi_strength=self._state.system.wifi_strength,
                        gpu_util=self._state.system.gpu_util,
                        audio_volume=self._state.system.audio_volume,
                        audio_muted=self._state.system.audio_muted,
                    ),
                    git_branch=self._state.git_branch,
                    terminal_cwd=self._state.terminal_cwd,
                    downloads=dict(self._state.downloads),
                    running_apps=set(self._state.running_apps),
                )

                await asyncio.sleep(self._window_interval * 0.5)  # Half of fastest interval

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[Observer] Observe loop error: %s", e)

    # ── Sampling Methods (run in executor threads) ─────────────

    def _sample_window(self) -> None:
        """Sample the currently focused window."""
        try:
            info = self._detect_focused_window()
            new_hash = self._hash_data(f"{info.get('title', '')}"
                                       f"{info.get('application', '')}"
                                       f"{info.get('pid', 0)}")

            with self._lock:
                self._state.window.title = info.get("title", "")
                self._state.window.application = info.get("application", "")
                self._state.window.pid = info.get("pid", 0)
                self._state.window.hash = new_hash

        except Exception as e:
            logger.debug("[Observer] Window sample error: %s", e)

    def _sample_clipboard(self) -> None:
        """Sample clipboard text content."""
        try:
            text = self._get_clipboard_text()
            new_hash = self._hash_data(text)

            with self._lock:
                self._state.clipboard.text = text
                self._state.clipboard.hash = new_hash

        except Exception as e:
            logger.debug("[Observer] Clipboard sample error: %s", e)

    def _sample_browser(self) -> None:
        """Sample current browser tab."""
        try:
            title, url = self._get_browser_tab()
            new_hash = self._hash_data(f"{title}{url}")

            with self._lock:
                self._state.browser.title = title
                self._state.browser.url = url
                self._state.browser.hash = new_hash

        except Exception as e:
            logger.debug("[Observer] Browser sample error: %s", e)

    def _sample_system(self) -> None:
        """Sample system resources."""
        try:
            cpu, ram = self._get_cpu_ram()
            battery_pct, battery_charging = self._get_battery()
            wifi_ssid, wifi_strength = self._get_wifi()
            gpu = self._get_gpu_util()
            volume, muted = self._get_audio_state()

            with self._lock:
                self._state.system.cpu_percent = cpu
                self._state.system.ram_percent = ram
                self._state.system.battery_percent = battery_pct
                self._state.system.battery_charging = battery_charging
                self._state.system.wifi_ssid = wifi_ssid
                self._state.system.wifi_strength = wifi_strength
                self._state.system.gpu_util = gpu
                self._state.system.audio_volume = volume
                self._state.system.audio_muted = muted

        except Exception as e:
            logger.debug("[Observer] System sample error: %s", e)

    def _sample_git(self) -> None:
        """Sample current git branch."""
        try:
            branch = self._get_git_branch()
            with self._lock:
                self._state.git_branch = branch
        except Exception as e:
            logger.debug("[Observer] Git sample error: %s", e)

    def _sample_downloads(self) -> None:
        """Sample active downloads."""
        try:
            active = self._get_active_downloads()
            with self._lock:
                self._state.downloads = active
        except Exception as e:
            logger.debug("[Observer] Downloads sample error: %s", e)

    # ── Detection Helpers (reusing DesktopState patterns) ──────

    @staticmethod
    def _detect_focused_window() -> Dict[str, Any]:
        """Detect the currently focused window."""
        info: Dict[str, Any] = {"title": "", "application": "", "pid": 0}

        if shutil.which("xdotool"):
            try:
                wid = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True, text=True, timeout=1
                ).stdout.strip()
                if wid:
                    title = subprocess.run(
                        ["xdotool", "getwindowname", wid],
                        capture_output=True, text=True, timeout=1
                    ).stdout.strip()
                    info["title"] = title

                    pid_str = subprocess.run(
                        ["xdotool", "getwindowpid", wid],
                        capture_output=True, text=True, timeout=1
                    ).stdout.strip()
                    if pid_str:
                        info["pid"] = int(pid_str)
                        try:
                            exe = os.readlink(f"/proc/{pid_str}/exe")
                            info["application"] = Path(exe).name
                        except Exception:
                            pass
            except Exception:
                pass

        return info

    @staticmethod
    def _get_clipboard_text(max_chars: int = 500) -> str:
        """Get clipboard text content."""
        if shutil.which("xclip"):
            try:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    return result.stdout.strip()[:max_chars]
            except Exception:
                pass

        if shutil.which("wl-paste"):
            try:
                result = subprocess.run(
                    ["wl-paste", "--primary"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    return result.stdout.strip()[:max_chars]
            except Exception:
                pass

        return ""

    @staticmethod
    def _get_browser_tab() -> Tuple[str, str]:
        """Get current browser tab title and approximate URL."""
        focused = DesktopObserver._detect_focused_window()
        app = focused.get("application", "").lower()
        title = focused.get("title", "")

        browser_apps = (
            "firefox", "firefox-bin", "firefox-esr", "firefox-nightly",
            "chrome", "chromium", "brave", "edge", "google-chrome",
            "chromium-browser", "opera",
        )

        if app in browser_apps and title:
            # Strip browser suffix from title
            separators = [" — ", " - ", " – "]
            for sep in separators:
                if sep in title:
                    tab_title = title.rsplit(sep, 1)[0].strip()
                    return tab_title, ""

            return title, ""

        return "", ""

    @staticmethod
    def _get_cpu_ram() -> Tuple[float, float]:
        """Get approximate CPU and RAM usage."""
        cpu, ram = 0.0, 0.0

        try:
            stat = Path("/proc/stat").read_text().splitlines()[0]
            parts = stat.split()
            if len(parts) >= 5:
                idle = int(parts[4])
                total = sum(int(x) for x in parts[1:])
                if total > 0:
                    cpu = round((1 - idle / total) * 100, 1)
        except Exception:
            pass

        try:
            meminfo = Path("/proc/meminfo").read_text()
            total_m = re.search(r"MemTotal:\s+(\d+)", meminfo)
            avail_m = re.search(r"MemAvailable:\s+(\d+)", meminfo)
            if total_m and avail_m:
                total_val = int(total_m.group(1))
                avail_val = int(avail_m.group(1))
                ram = round((1 - avail_val / total_val) * 100, 1)
        except Exception:
            pass

        return cpu, ram

    @staticmethod
    def _get_battery() -> Tuple[int, bool]:
        """Get battery percentage and charging status."""
        battery_dir = Path("/sys/class/power_supply")
        if battery_dir.exists():
            for bat in battery_dir.iterdir():
                if bat.name.startswith("BAT"):
                    try:
                        capacity = int((bat / "capacity").read_text().strip())
                        status = (bat / "status").read_text().strip().lower()
                        charging = "charging" in status or "full" in status
                        return capacity, charging
                    except Exception:
                        pass
        return 100, True

    @staticmethod
    def _get_wifi() -> Tuple[str, int]:
        """Get WiFi SSID and signal strength."""
        if shutil.which("nmcli"):
            try:
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "active,ssid,signal", "dev", "wifi"],
                    capture_output=True, text=True, timeout=2
                )
                for line in result.stdout.splitlines():
                    if line.startswith("yes:"):
                        parts = line.split(":")
                        ssid = parts[1] if len(parts) >= 2 else ""
                        signal = int(parts[2]) if len(parts) >= 3 else 0
                        return ssid, signal
            except Exception:
                pass
        return "", 0

    @staticmethod
    def _get_gpu_util() -> float:
        """Get GPU utilization from nvidia-smi."""
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _get_audio_state() -> Tuple[int, bool]:
        """Get volume and mute state."""
        volume, muted = 50, False

        if shutil.which("pactl"):
            try:
                result = subprocess.run(
                    ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                    capture_output=True, text=True, timeout=1
                )
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

    @staticmethod
    def _get_git_branch() -> str:
        """Get current git branch from CWD or focused window PID."""
        try:
            # Try from focused window's CWD
            focused = DesktopObserver._detect_focused_window()
            pid = focused.get("pid", 0)
            if pid > 0:
                try:
                    cwd = os.readlink(f"/proc/{pid}/cwd")
                    result = subprocess.run(
                        ["git", "-C", cwd, "branch", "--show-current"],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                except Exception:
                    pass
        except Exception:
            pass

        return ""

    @staticmethod
    def _get_active_downloads() -> Dict[str, float]:
        """Get active download filenames with timestamps."""
        downloads: Dict[str, float] = {}
        download_dirs = [
            Path.home() / "Downloads",
            Path.home() / "downloads",
        ]
        for d in download_dirs:
            if d.exists():
                for part in d.glob("*.part"):
                    downloads[part.name] = part.stat().st_mtime
                # Also detect recent .crdownload (Chrome)
                for crd in d.glob("*.crdownload"):
                    downloads[crd.name] = crd.stat().st_mtime
        return downloads

    # ── Event Publication ──────────────────────────────────────

    async def _publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish a change event to the EventBus."""
        if self._event_bus:
            try:
                await self._event_bus.emit(event_type, data, source="desktop_observer")
            except Exception as e:
                logger.debug("[Observer] Event publish error: %s", e)

    async def _dispatch_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Dispatch change events to registered callbacks."""
        try:
            if event_type == "desktop:window_changed" and self._on_window_change:
                ws = WindowState(
                    title=data.get("title", ""),
                    application=data.get("application", ""),
                    pid=data.get("pid", 0),
                )
                self._on_window_change(ws)

            elif event_type == "desktop:clipboard_changed" and self._on_clipboard_change:
                cs = ClipboardState(text=data.get("text_preview", ""))
                self._on_clipboard_change(cs)

            elif event_type == "desktop:browser_changed" and self._on_browser_change:
                bs = BrowserState(
                    title=data.get("tab_title", ""),
                    url=data.get("url", ""),
                )
                self._on_browser_change(bs)

            elif event_type == "desktop:git_changed" and self._on_git_change:
                self._on_git_change(data.get("branch", ""))

            elif event_type == "desktop:download_started" and self._on_download:
                self._on_download(data.get("filename", ""), True)

            elif event_type == "desktop:download_complete" and self._on_download:
                self._on_download(data.get("filename", ""), False)

            elif event_type == "desktop:battery_low" and self._on_battery_low:
                self._on_battery_low(data.get("percent", 0))

            elif event_type == "desktop:audio_changed" and self._on_audio_change:
                self._on_audio_change(
                    data.get("volume", 50),
                    data.get("muted", False),
                )

            elif event_type == "desktop:app_launched" and self._on_app_change:
                self._on_app_change(data.get("app_name", ""), True)

            elif event_type == "desktop:app_closed" and self._on_app_change:
                self._on_app_change(data.get("app_name", ""), False)

        except Exception as e:
            logger.debug("[Observer] Callback dispatch error: %s", e)

    # ── Snapshot API ───────────────────────────────────────────

    def snapshot(self) -> DesktopState:
        """Return a copy of the current state."""
        with self._lock:
            return DesktopState(
                window=WindowState(
                    title=self._state.window.title,
                    application=self._state.window.application,
                    pid=self._state.window.pid,
                    hash=self._state.window.hash,
                ),
                clipboard=ClipboardState(
                    text=self._state.clipboard.text,
                    hash=self._state.clipboard.hash,
                ),
                browser=BrowserState(
                    title=self._state.browser.title,
                    url=self._state.browser.url,
                    hash=self._state.browser.hash,
                ),
                system=SystemState(
                    cpu_percent=self._state.system.cpu_percent,
                    ram_percent=self._state.system.ram_percent,
                    battery_percent=self._state.system.battery_percent,
                    battery_charging=self._state.system.battery_charging,
                    wifi_ssid=self._state.system.wifi_ssid,
                    wifi_strength=self._state.system.wifi_strength,
                    gpu_util=self._state.system.gpu_util,
                    audio_volume=self._state.system.audio_volume,
                    audio_muted=self._state.system.audio_muted,
                ),
                git_branch=self._state.git_branch,
                terminal_cwd=self._state.terminal_cwd,
                downloads=dict(self._state.downloads),
                running_apps=set(self._state.running_apps),
            )

    # ── Utilities ──────────────────────────────────────────────

    @staticmethod
    def _hash_data(data: str) -> str:
        """Create a short content hash for change detection."""
        if not data:
            return ""
        return hashlib.md5(data.encode("utf-8", errors="replace")).hexdigest()[:12]

    @property
    def is_running(self) -> bool:
        return self._running

    def close(self) -> None:
        self._running = False
        logger.info("[Observer] DesktopObserver closed")


# Global singleton
desktop_observer = DesktopObserver()