"""
LayoutAnalyzer — Application type detection + region segmentation.

Before OCR runs, this module determines:
  1. WHAT application is in focus (VSCode, Chrome, Terminal, etc.)
  2. WHERE the regions are (toolbar, sidebar, editor, tabs, status bar, etc.)

This allows OCR to focus ONLY on relevant regions and the UI tree builder
to assign correct semantic roles.

Application detection strategy:
  - Window title pattern matching (fastest, <1ms)
  - WM_CLASS property (X11 xprop, <5ms)
  - Window geometry heuristics (toolbar height, sidebar position)
  - Process name fingerprinting

Recognized applications:
  - IDEs: VSCode, JetBrains (PyCharm/IntelliJ/WebStorm/...)
  - Browsers: Chrome, Firefox, Edge, Brave
  - Terminals: GNOME Terminal, Konsole, Alacritty, Kitty, etc.
  - File Managers: Nautilus, Dolphin, Thunar
  - Communication: Discord, Telegram, Slack
  - Media: Spotify, YouTube (in browser), VLC
  - Document: PDF viewers, LibreOffice, Image viewers
  - System: Settings, GNOME Control Center
  - Generic: GTK apps, Qt apps, Electron apps

Layout segmentation:
  - Top → toolbar, menu bar, tab bar (top ~5-8% of window)
  - Left → sidebar, navigation panel (left ~15-25%)
  - Right → properties panel, inspector (right ~10-20%)
  - Center → editor, content area
  - Bottom → status bar, terminal panel (bottom ~3-5%)
  - Float → dialogs, popups, notifications

Structured logging: [LAYOUT]
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Application types
# ═══════════════════════════════════════════════════════════════

class ApplicationType(str, Enum):
    """Broad categories of desktop applications."""
    UNKNOWN = "unknown"
    VSCODE = "vscode"
    JETBRAINS = "jetbrains"            # PyCharm, IntelliJ, WebStorm, etc.
    CHROME = "chrome"
    FIREFOX = "firefox"
    EDGE = "edge"
    BRAVE = "brave"
    TERMINAL = "terminal"
    FILE_MANAGER = "file_manager"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    SPOTIFY = "spotify"
    YOUTUBE = "youtube"                # detected from browser title
    PDF_VIEWER = "pdf_viewer"
    IMAGE_VIEWER = "image_viewer"
    SETTINGS = "settings"
    LIBREOFFICE = "libreoffice"
    VLC = "vlc"
    GTK_APP = "gtk_app"
    QT_APP = "qt_app"
    ELECTRON_APP = "electron_app"
    SYSTEM_MONITOR = "system_monitor"
    CALCULATOR = "calculator"
    NOTEPAD = "notepad"                # gedit, kate, mousepad, etc.

    @property
    def is_browser(self) -> bool:
        return self in (
            ApplicationType.CHROME, ApplicationType.FIREFOX,
            ApplicationType.EDGE, ApplicationType.BRAVE,
        )

    @property
    def is_ide(self) -> bool:
        return self in (ApplicationType.VSCODE, ApplicationType.JETBRAINS)

    @property
    def is_terminal(self) -> bool:
        return self == ApplicationType.TERMINAL

    @property
    def is_electron(self) -> bool:
        """True for apps likely built on Electron."""
        return self in (
            ApplicationType.VSCODE, ApplicationType.DISCORD,
            ApplicationType.SLACK, ApplicationType.SPOTIFY,
            ApplicationType.ELECTRON_APP,
        )


# ═══════════════════════════════════════════════════════════════
# Layout regions
# ═══════════════════════════════════════════════════════════════

class RegionType(str, Enum):
    """Semantic regions within an application window."""
    DESKTOP = "desktop"
    WINDOW = "window"
    TOOLBAR = "toolbar"
    MENU_BAR = "menu_bar"
    TAB_BAR = "tab_bar"
    SIDEBAR = "sidebar"
    LEFT_PANEL = "left_panel"
    RIGHT_PANEL = "right_panel"
    BOTTOM_PANEL = "bottom_panel"
    EDITOR = "editor"
    CONTENT = "content"
    STATUS_BAR = "status_bar"
    NAVIGATION = "navigation"
    SCROLLBAR = "scrollbar"
    MINIMAP = "minimap"
    DIALOG = "dialog"
    POPUP = "popup"
    NOTIFICATION = "notification"
    TASKBAR = "taskbar"
    DOCK = "dock"
    SYSTEM_TRAY = "system_tray"
    TITLE_BAR = "title_bar"
    UNKNOWN = "unknown"


@dataclass
class LayoutRegion:
    """A detected region within an application window."""
    region_type: RegionType
    bounds: Tuple[int, int, int, int]  # x, y, w, h (relative to window)
    label: str = ""
    confidence: float = 0.0
    child_regions: List["LayoutRegion"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bounds[0]

    @property
    def y(self) -> int:
        return self.bounds[1]

    @property
    def width(self) -> int:
        return self.bounds[2]

    @property
    def height(self) -> int:
        return self.bounds[3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.region_type.value,
            "label": self.label,
            "bounds": list(self.bounds),
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
            "children": [r.to_dict() for r in self.child_regions],
        }

    def center(self) -> Tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )


@dataclass
class WindowLayout:
    """Complete layout analysis of a single window."""
    app_type: ApplicationType = ApplicationType.UNKNOWN
    app_name: str = ""                 # e.g. "PyCharm", "firefox"
    window_title: str = ""
    window_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    regions: List[LayoutRegion] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def find_region(self, region_type: RegionType) -> Optional[LayoutRegion]:
        """Find the first region of the given type."""
        for r in self.regions:
            result = self._find_in_region(r, region_type)
            if result is not None:
                return result
        return None

    @staticmethod
    def _find_in_region(region: LayoutRegion, region_type: RegionType) -> Optional[LayoutRegion]:
        if region.region_type == region_type:
            return region
        for child in region.child_regions:
            result = WindowLayout._find_in_region(child, region_type)
            if result is not None:
                return result
        return None

    def all_regions(self) -> List[LayoutRegion]:
        """Flatten all regions into a list."""
        flat: List[LayoutRegion] = []
        for r in self.regions:
            self._flatten(r, flat)
        return flat

    @staticmethod
    def _flatten(region: LayoutRegion, out: List[LayoutRegion]) -> None:
        out.append(region)
        for child in region.child_regions:
            WindowLayout._flatten(child, out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_type": self.app_type.value,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "window_bounds": list(self.window_bounds),
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
            "regions": [r.to_dict() for r in self.regions],
        }


# ═══════════════════════════════════════════════════════════════
# Application detection fingerprint database
# ═══════════════════════════════════════════════════════════════

# Title patterns: (regex, ApplicationType, app_name)
TITLE_PATTERNS: List[Tuple[re.Pattern, ApplicationType, str]] = [
    # JetBrains IDEs
    (re.compile(r"(.+?)\s*[-–—]\s*(.+?)\s*[-–—]\s*(PyCharm|IntelliJ\s*IDEA|WebStorm|PhpStorm|GoLand|CLion|Rider|DataGrip|RubyMine|RustRover)"), ApplicationType.JETBRAINS, ""),
    # VS Code / VSCodium
    (re.compile(r"(.+?)\s*[-–—]\s*(.+?)\s*[-–—]\s*(Visual\s+Studio\s+Code|VSCodium|Code\s*-\s*OSS)"), ApplicationType.VSCODE, ""),
    (re.compile(r"(.+?)\s+-\s+(Visual\s+Studio\s+Code|Code)"), ApplicationType.VSCODE, ""),
    # Browsers
    (re.compile(r"(.+?)\s*[-–—]\s*Mozilla\s+Firefox", re.IGNORECASE), ApplicationType.FIREFOX, "Firefox"),
    (re.compile(r"(.+?)\s*[-–—]\s*(Google\s+Chrome|Chrome)", re.IGNORECASE), ApplicationType.CHROME, "Chrome"),
    (re.compile(r"(.+?)\s*[-–—]\s*Microsoft\s+Edge", re.IGNORECASE), ApplicationType.EDGE, "Edge"),
    (re.compile(r"(.+?)\s*[-–—]\s*Brave", re.IGNORECASE), ApplicationType.BRAVE, "Brave"),
    # Terminals
    (re.compile(r"(.+?terminal)", re.IGNORECASE), ApplicationType.TERMINAL, ""),
    (re.compile(r"^(alacritty|kitty|wezterm|konsole|xfce4-terminal|terminator|tilix|qterminal)", re.IGNORECASE), ApplicationType.TERMINAL, ""),
    # File managers
    (re.compile(r"^(nautilus|dolphin|thunar|pcmanfm|nemo|caja)", re.IGNORECASE), ApplicationType.FILE_MANAGER, ""),
    # Communication
    (re.compile(r"(.+?)\s*[-–—]\s*Discord", re.IGNORECASE), ApplicationType.DISCORD, "Discord"),
    (re.compile(r"Telegram", re.IGNORECASE), ApplicationType.TELEGRAM, "Telegram"),
    (re.compile(r"(.+?)\s*[-–—]\s*Slack", re.IGNORECASE), ApplicationType.SLACK, "Slack"),
    # Media
    (re.compile(r"Spotify", re.IGNORECASE), ApplicationType.SPOTIFY, "Spotify"),
    (re.compile(r"YouTube\s*[-–—]\s*", re.IGNORECASE), ApplicationType.YOUTUBE, "YouTube"),
    (re.compile(r"VLC\s+media\s+player", re.IGNORECASE), ApplicationType.VLC, "VLC"),
    # PDF
    (re.compile(r"\.pdf\b", re.IGNORECASE), ApplicationType.PDF_VIEWER, ""),
    (re.compile(r"^(evince|okular|zathura|qpdfview)", re.IGNORECASE), ApplicationType.PDF_VIEWER, ""),
    # Image
    (re.compile(r"(eog|gthumb|gwenview|feh|nomacs)", re.IGNORECASE), ApplicationType.IMAGE_VIEWER, ""),
    # Settings
    (re.compile(r"(settings|preferences|gnome-control-center)", re.IGNORECASE), ApplicationType.SETTINGS, ""),
    # LibreOffice
    (re.compile(r"LibreOffice", re.IGNORECASE), ApplicationType.LIBREOFFICE, ""),
    # System
    (re.compile(r"(system\s+monitor|htop|btop|bpytop)", re.IGNORECASE), ApplicationType.SYSTEM_MONITOR, ""),
    (re.compile(r"(calculator|gnome-calculator|galculator)", re.IGNORECASE), ApplicationType.CALCULATOR, ""),
    (re.compile(r"(gedit|kate|mousepad|leafpad|xed|pluma)", re.IGNORECASE), ApplicationType.NOTEPAD, ""),
]

# Process name → ApplicationType mapping
PROCESS_MAP: Dict[str, Tuple[ApplicationType, str]] = {
    # IDEs
    "code": (ApplicationType.VSCODE, "VSCode"),
    "code-oss": (ApplicationType.VSCODE, "VSCodium"),
    "pycharm": (ApplicationType.JETBRAINS, "PyCharm"),
    "pycharm.sh": (ApplicationType.JETBRAINS, "PyCharm"),
    "idea": (ApplicationType.JETBRAINS, "IntelliJ IDEA"),
    "idea.sh": (ApplicationType.JETBRAINS, "IntelliJ IDEA"),
    "webstorm": (ApplicationType.JETBRAINS, "WebStorm"),
    "webstorm.sh": (ApplicationType.JETBRAINS, "WebStorm"),
    "phpstorm": (ApplicationType.JETBRAINS, "PhpStorm"),
    "goland": (ApplicationType.JETBRAINS, "GoLand"),
    "clion": (ApplicationType.JETBRAINS, "CLion"),
    "rider": (ApplicationType.JETBRAINS, "Rider"),
    "datagrip": (ApplicationType.JETBRAINS, "DataGrip"),
    "rubymine": (ApplicationType.JETBRAINS, "RubyMine"),
    "studio": (ApplicationType.JETBRAINS, "Android Studio"),
    # Browsers
    "firefox": (ApplicationType.FIREFOX, "Firefox"),
    "firefox-bin": (ApplicationType.FIREFOX, "Firefox"),
    "firefox-esr": (ApplicationType.FIREFOX, "Firefox"),
    "chrome": (ApplicationType.CHROME, "Chrome"),
    "google-chrome": (ApplicationType.CHROME, "Chrome"),
    "google-chrome-stable": (ApplicationType.CHROME, "Chrome"),
    "chromium": (ApplicationType.CHROME, "Chromium"),
    "chromium-browser": (ApplicationType.CHROME, "Chromium"),
    "brave": (ApplicationType.BRAVE, "Brave"),
    "brave-browser": (ApplicationType.BRAVE, "Brave"),
    "msedge": (ApplicationType.EDGE, "Edge"),
    "microsoft-edge": (ApplicationType.EDGE, "Edge"),
    # Terminals
    "gnome-terminal-server": (ApplicationType.TERMINAL, "GNOME Terminal"),
    "konsole": (ApplicationType.TERMINAL, "Konsole"),
    "alacritty": (ApplicationType.TERMINAL, "Alacritty"),
    "kitty": (ApplicationType.TERMINAL, "Kitty"),
    "wezterm": (ApplicationType.TERMINAL, "WezTerm"),
    "wezterm-gui": (ApplicationType.TERMINAL, "WezTerm"),
    "xfce4-terminal": (ApplicationType.TERMINAL, "Xfce Terminal"),
    "terminator": (ApplicationType.TERMINAL, "Terminator"),
    "tilix": (ApplicationType.TERMINAL, "Tilix"),
    "qterminal": (ApplicationType.TERMINAL, "QTerminal"),
    # File Managers
    "nautilus": (ApplicationType.FILE_MANAGER, "Files"),
    "dolphin": (ApplicationType.FILE_MANAGER, "Dolphin"),
    "thunar": (ApplicationType.FILE_MANAGER, "Thunar"),
    "pcmanfm": (ApplicationType.FILE_MANAGER, "PCManFM"),
    "nemo": (ApplicationType.FILE_MANAGER, "Nemo"),
    "caja": (ApplicationType.FILE_MANAGER, "Caja"),
    # Communication
    "discord": (ApplicationType.DISCORD, "Discord"),
    "telegram-desktop": (ApplicationType.TELEGRAM, "Telegram"),
    "slack": (ApplicationType.SLACK, "Slack"),
    # Media
    "spotify": (ApplicationType.SPOTIFY, "Spotify"),
    "vlc": (ApplicationType.VLC, "VLC"),
    # Document
    "evince": (ApplicationType.PDF_VIEWER, "Evince"),
    "okular": (ApplicationType.PDF_VIEWER, "Okular"),
    "zathura": (ApplicationType.PDF_VIEWER, "Zathura"),
    "eog": (ApplicationType.IMAGE_VIEWER, "Image Viewer"),
    "gthumb": (ApplicationType.IMAGE_VIEWER, "gThumb"),
    "gwenview": (ApplicationType.IMAGE_VIEWER, "Gwenview"),
    "soffice": (ApplicationType.LIBREOFFICE, "LibreOffice"),
    "soffice.bin": (ApplicationType.LIBREOFFICE, "LibreOffice"),
    # Settings
    "gnome-control-center": (ApplicationType.SETTINGS, "Settings"),
    # Notepad
    "gedit": (ApplicationType.NOTEPAD, "gedit"),
    "kate": (ApplicationType.NOTEPAD, "Kate"),
    "mousepad": (ApplicationType.NOTEPAD, "Mousepad"),
    # System
    "gnome-system-monitor": (ApplicationType.SYSTEM_MONITOR, "System Monitor"),
    "gnome-calculator": (ApplicationType.CALCULATOR, "Calculator"),
}

# YouTube detection in browser titles
YOUTUBE_TITLE_RE = re.compile(r"YouTube\s*[-–—]\s*", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════
# LayoutAnalyzer
# ═══════════════════════════════════════════════════════════════

class LayoutAnalyzer:
    """
    Detect application type and layout regions from window metadata.

    Target latencies:
      - App type detection: <5ms  (title pattern + process matching)
      - Layout segmentation:  <15ms (geometry heuristics)
      - Total:                 <20ms

    Structured logging: [LAYOUT]
    """

    def __init__(self):
        self._last_layout: Optional[WindowLayout] = None

    # ── Application type detection ─────────────────────────

    def detect_application(
        self,
        window_title: str,
        process_name: str = "",
        pid: int = 0,
    ) -> Tuple[ApplicationType, str, float]:
        """
        Determine the application type from window title + process name.

        Strategy (ordered by speed):
          1. Process name lookup (fastest, <0.5ms)
          2. Title pattern matching (<1ms)
          3. Window class via xprop (only if above failed, <5ms)

        Returns:
            (ApplicationType, app_name, confidence)
        """
        t0 = __import__("time").perf_counter_ns()

        # ── Step 1: Process name lookup ─────────────────
        if process_name:
            proc_key = process_name.lower().strip()
            if proc_key in PROCESS_MAP:
                app_type, app_name = PROCESS_MAP[proc_key]
                confidence = 0.95
                elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
                logger.info("[LAYOUT] App detected via process: %s → %s (%s) [%.1fms]",
                            proc_key, app_type.value, app_name, elapsed)
                self._record_layout(app_type, app_name, window_title)
                return app_type, app_name, confidence

            # Check fuzzy match (strip version suffixes)
            proc_base = proc_key.split(".")[0].split("-")[0]
            if proc_base in PROCESS_MAP:
                app_type, app_name = PROCESS_MAP[proc_base]
                confidence = 0.85
                elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
                logger.info("[LAYOUT] App detected via fuzzy process: %s → %s (%s) [%.1fms]",
                            proc_base, app_type.value, app_name, elapsed)
                self._record_layout(app_type, app_name, window_title)
                return app_type, app_name, confidence

        # ── Step 2: Title pattern matching ───────────────
        if window_title:
            for pattern, app_type, name_template in TITLE_PATTERNS:
                match = pattern.search(window_title)
                if match:
                    app_name = name_template
                    if not app_name:
                        # Try to extract app name from title groups
                        app_name = match.group(3) if match.lastindex and match.lastindex >= 3 else match.group(0)[:30]
                    confidence = 0.80 if process_name else 0.70
                    elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
                    logger.info("[LAYOUT] App detected via title pattern: '%s' → %s (%s) [%.1fms]",
                                window_title[:60], app_type.value, app_name, elapsed)
                    self._record_layout(app_type, app_name, window_title)
                    return app_type, app_name, confidence

            # YouTube detection (browser tab)
            if YOUTUBE_TITLE_RE.search(window_title):
                confidence = 0.70
                elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
                logger.info("[LAYOUT] App detected: YouTube (from browser title) [%.1fms]", elapsed)
                self._record_layout(ApplicationType.YOUTUBE, "YouTube", window_title)
                return ApplicationType.YOUTUBE, "YouTube", confidence

        # ── Step 3: WM_CLASS via xprop (X11 only) ────────
        if pid > 0 and shutil.which("xprop"):
            try:
                wm_class = self._get_wm_class(pid)
                if wm_class:
                    wm_lower = wm_class.lower()
                    for proc_key, (app_type, app_name) in PROCESS_MAP.items():
                        if proc_key in wm_lower:
                            confidence = 0.75
                            elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
                            logger.info("[LAYOUT] App detected via WM_CLASS: %s → %s [%.1fms]",
                                        wm_class, app_type.value, elapsed)
                            self._record_layout(app_type, app_name, window_title)
                            return app_type, app_name, confidence

                    # GTK/Qt/Electron fallback classification
                    if "gtk" in wm_lower or "gnome" in wm_lower:
                        app_type, app_name = ApplicationType.GTK_APP, wm_class
                    elif "qt" in wm_lower or "kde" in wm_lower:
                        app_type, app_name = ApplicationType.QT_APP, wm_class
                    elif "electron" in wm_lower or wm_lower in ("chromium-browser",):
                        app_type, app_name = ApplicationType.ELECTRON_APP, wm_class
                    else:
                        app_type, app_name = ApplicationType.UNKNOWN, wm_class
                    confidence = 0.50
                    elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
                    logger.info("[LAYOUT] App detected via WM_CLASS: %s → %s [%.1fms]",
                                wm_class, app_type.value, elapsed)
                    self._record_layout(app_type, app_name, window_title)
                    return app_type, app_name, confidence
            except Exception as e:
                logger.debug("[LAYOUT] xprop failed: %s", e)

        # ── Unknown ─────────────────────────────────────
        confidence = 0.20
        elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
        logger.debug("[LAYOUT] App type UNKNOWN for window '%s' proc='%s' [%.1fms]",
                     window_title[:60], process_name, elapsed)
        self._record_layout(ApplicationType.UNKNOWN, "", window_title)
        return ApplicationType.UNKNOWN, "", confidence

    @staticmethod
    def _get_wm_class(pid: int) -> str:
        """Get the WM_CLASS property for a window by PID."""
        try:
            # Find window IDs for the PID
            result = subprocess.run(
                ["xdotool", "search", "--pid", str(pid)],
                capture_output=True, text=True, timeout=2,
            )
            wid = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
            if not wid:
                return ""
            # Get WM_CLASS
            result = subprocess.run(
                ["xprop", "-id", wid, "WM_CLASS"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                match = re.search(r'WM_CLASS\(\w+\)\s*=\s*"([^"]*)",\s*"([^"]*)"', result.stdout)
                if match:
                    return match.group(2) or match.group(1)
                # Simpler match
                match = re.search(r'"([^"]*)"', result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return ""

    # ── Layout segmentation ───────────────────────────────

    def segment_layout(
        self,
        app_type: ApplicationType,
        window_width: int,
        window_height: int,
        window_title: str = "",
    ) -> WindowLayout:
        """
        Segment a window into semantic layout regions.

        Uses application-type-aware heuristics since different apps
        have different layout conventions.

        Returns a WindowLayout with regions for toolbar, sidebar,
        editor/content, status bar, etc.
        """
        t0 = __import__("time").perf_counter_ns()

        layout = WindowLayout(
            app_type=app_type,
            window_title=window_title,
            window_bounds=(0, 0, window_width, window_height),
            confidence=0.80,
        )

        w, h = window_width, window_height
        if w <= 0 or h <= 0:
            return layout

        # ── Common region definitions ──────────────────
        # These are Y/X fractions of the window dimensions

        # Title bar: top 3-5% (OS decoration, not always accessible)
        title_bar_h = int(h * 0.04)
        layout.regions.append(LayoutRegion(
            region_type=RegionType.TITLE_BAR,
            bounds=(0, 0, w, title_bar_h),
            label="Title Bar",
            confidence=0.90,
        ))

        # Menu bar: below title bar, ~4-6% of height
        menu_bar_y = title_bar_h
        menu_bar_h = int(h * 0.05)
        if app_type.is_ide or app_type in (ApplicationType.FILE_MANAGER, ApplicationType.LIBREOFFICE, ApplicationType.GTK_APP):
            layout.regions.append(LayoutRegion(
                region_type=RegionType.MENU_BAR,
                bounds=(0, menu_bar_y, w, menu_bar_h),
                label="Menu Bar",
                confidence=0.85,
            ))
            content_start_y = menu_bar_y + menu_bar_h
        elif app_type.is_browser:
            # Browsers: address bar + tab row
            layout.regions.append(LayoutRegion(
                region_type=RegionType.NAVIGATION,
                bounds=(0, menu_bar_y, w, int(h * 0.08)),
                label="Navigation Bar",
                confidence=0.85,
            ))
            content_start_y = menu_bar_y + int(h * 0.08)
        else:
            content_start_y = menu_bar_y

        # Toolbar region (IDEs, editors)
        if app_type.is_ide:
            toolbar_y = content_start_y
            toolbar_h = int(h * 0.05)
            layout.regions.append(LayoutRegion(
                region_type=RegionType.TOOLBAR,
                bounds=(0, toolbar_y, w, toolbar_h),
                label="Toolbar",
                confidence=0.80,
            ))
            content_start_y = toolbar_y + toolbar_h

        # Tab bar (browsers, IDEs)
        if app_type.is_browser or app_type.is_ide or app_type == ApplicationType.TERMINAL:
            tab_y = content_start_y
            tab_h = int(h * 0.04)
            layout.regions.append(LayoutRegion(
                region_type=RegionType.TAB_BAR,
                bounds=(0, tab_y, w, tab_h),
                label="Tab Bar",
                confidence=0.75,
            ))
            content_start_y = tab_y + tab_h

        # Sidebar (left panel — IDEs, file managers, Discord)
        sidebar_w = 0
        if app_type.is_ide:
            sidebar_w = int(w * 0.18)  # IDEs: wide project sidebar
        elif app_type == ApplicationType.FILE_MANAGER:
            sidebar_w = int(w * 0.22)  # File managers: sidebar with places
        elif app_type in (ApplicationType.DISCORD, ApplicationType.SLACK, ApplicationType.TELEGRAM):
            sidebar_w = int(w * 0.15)  # Chat apps: channel/server list
        elif app_type.is_terminal and w > 1200:
            sidebar_w = int(w * 0.10)  # Wide terminals sometimes have side panels

        if sidebar_w > 0:
            layout.regions.append(LayoutRegion(
                region_type=RegionType.SIDEBAR,
                bounds=(0, content_start_y, sidebar_w, h - content_start_y - int(h * 0.04)),
                label="Sidebar",
                confidence=0.75,
            ))

        # Right panel (IDEs, browsers with dev tools)
        right_panel_w = 0
        if app_type.is_ide and w > 1400:
            right_panel_w = int(w * 0.12)
        if right_panel_w > 0:
            layout.regions.append(LayoutRegion(
                region_type=RegionType.RIGHT_PANEL,
                bounds=(w - right_panel_w, content_start_y, right_panel_w, h - content_start_y - int(h * 0.08)),
                label="Right Panel",
                confidence=0.65,
            ))

        # Bottom panel (IDEs: terminal/output)
        bottom_panel_h = int(h * 0.06)
        if app_type.is_ide:
            bottom_panel_h = int(h * 0.20)  # IDEs: terminal, console, problems
            layout.regions.append(LayoutRegion(
                region_type=RegionType.BOTTOM_PANEL,
                bounds=(sidebar_w, h - bottom_panel_h - int(h * 0.035), w - sidebar_w - right_panel_w, bottom_panel_h),
                label="Bottom Panel",
                confidence=0.70,
            ))

        # Editor / Content area (center)
        editor_x = sidebar_w
        editor_w = w - sidebar_w - right_panel_w
        editor_y = content_start_y
        editor_h = h - content_start_y - int(h * 0.04)  # reserve for status bar
        if app_type.is_ide:
            editor_h -= bottom_panel_h  # IDEs have bottom panel inside

        if editor_w > 0 and editor_h > 0:
            region_type = RegionType.EDITOR if app_type.is_ide else RegionType.CONTENT
            layout.regions.append(LayoutRegion(
                region_type=region_type,
                bounds=(editor_x, editor_y, editor_w, editor_h),
                label="Editor" if app_type.is_ide else "Content",
                confidence=0.80,
            ))

        # Status bar (bottom strip — IDEs, browsers, editors)
        status_h = int(h * 0.035)
        status_y = h - status_h
        if app_type.is_ide or app_type.is_browser or app_type == ApplicationType.VSCODE:
            layout.regions.append(LayoutRegion(
                region_type=RegionType.STATUS_BAR,
                bounds=(0, status_y, w, status_h),
                label="Status Bar",
                confidence=0.85,
            ))
        elif app_type.is_terminal:
            # Terminal last line is the prompt/status
            layout.regions.append(LayoutRegion(
                region_type=RegionType.STATUS_BAR,
                bounds=(0, h - int(h * 0.03), w, int(h * 0.03)),
                label="Prompt Line",
                confidence=0.50,
            ))

        # Scrollbar hint (right edge)
        if _HAS_CV2:
            layout.regions.append(LayoutRegion(
                region_type=RegionType.SCROLLBAR,
                bounds=(w - 12, content_start_y, 12, h - content_start_y - status_h),
                label="Scrollbar",
                confidence=0.60,
            ))

        elapsed = (__import__("time").perf_counter_ns() - t0) / 1_000_000
        logger.info("[LAYOUT] Layout segmented: %s (%dx%d) → %d regions [%.1fms]",
                     app_type.value, w, h, len(layout.regions), elapsed)
        self._last_layout = layout
        return layout

    # ── Specialized layout functions ──────────────────────

    def detect_dialogs(self, ocr_boxes: List[Any], window_width: int, window_height: int) -> List[LayoutRegion]:
        """
        Detect dialog/modal windows from OCR box positions.

        Dialog characteristics:
          - Small bounding box (typically <60% of window)
          - Centered or near-centered
          - Contains OK/Cancel/Yes/No/Confirm/Dismiss buttons
          - Often has a title bar with specific text
        """
        dialogs: List[LayoutRegion] = []
        dialog_keywords = {"ok", "cancel", "yes", "no", "confirm", "dismiss", "apply", "save", "close",
                           "alert", "warning", "error", "notification", "info", "about", "preferences",
                           "settings", "open", "save as", "export", "import"}

        if not ocr_boxes:
            return dialogs

        # Check if a cluster of boxes forms a dialog
        try:
            from services.ui_tree import UIElement
            boxes_with_text = []
            for box in ocr_boxes:
                if hasattr(box, 'text') and hasattr(box, 'bbox'):
                    boxes_with_text.append(box)
                elif isinstance(box, dict):
                    boxes_with_text.append(type('obj', (object,), {
                        'text': box.get('text', ''),
                        'bbox': box.get('bbox', (0, 0, 0, 0))
                    })())

            if not boxes_with_text:
                return dialogs

            # Group boxes that are close together (dialog clustering)
            # Count dialog keywords
            dialog_count = sum(1 for b in boxes_with_text
                               if getattr(b, 'text', '').strip().lower() in dialog_keywords)

            if dialog_count >= 2:
                # Compute union bounding box of all close boxes
                xs = [getattr(b, 'bbox', (0, 0, 0, 0))[0] for b in boxes_with_text]
                ys = [getattr(b, 'bbox', (0, 0, 0, 0))[1] for b in boxes_with_text]
                right = [getattr(b, 'bbox', (0, 0, 0, 0))[0] + getattr(b, 'bbox', (0, 0, 0, 0))[2] for b in boxes_with_text]
                bottom = [getattr(b, 'bbox', (0, 0, 0, 0))[1] + getattr(b, 'bbox', (0, 0, 0, 0))[3] for b in boxes_with_text]

                if xs and ys:
                    d_x = min(xs)
                    d_y = min(ys)
                    d_w = max(right) - d_x
                    d_h = max(bottom) - d_y

                    # Dialog check: is this box centered-ish?
                    center_x = d_x + d_w // 2
                    center_y = d_y + d_h // 2
                    window_cx, window_cy = window_width // 2, window_height // 2

                    if (abs(center_x - window_cx) < window_width * 0.3 and
                            abs(center_y - window_cy) < window_height * 0.3 and
                            d_w < window_width * 0.7 and d_h < window_height * 0.7):
                        dialogs.append(LayoutRegion(
                            region_type=RegionType.DIALOG,
                            bounds=(d_x, d_y, d_w, d_h),
                            label="Dialog",
                            confidence=0.70,
                        ))
                        logger.info("[LAYOUT] Dialog detected: bounds=(%d,%d,%dx%d) keywords=%d",
                                    d_x, d_y, d_w, d_h, dialog_count)

        except Exception as e:
            logger.debug("[LAYOUT] Dialog detection error: %s", e)

        return dialogs

    def detect_notifications(self, ocr_boxes: List[Any], window_width: int, window_height: int) -> List[LayoutRegion]:
        """
        Detect notification/popup regions.

        Notifications typically appear in the top-right or bottom-right
        corner of the desktop and are small transient boxes.
        """
        notifications: List[LayoutRegion] = []
        notif_keywords = {"notification", "update", "installed", "downloaded", "completed",
                          "error", "warning", "success", "syncing", "backup"}

        if not ocr_boxes:
            return notifications

        try:
            for box in ocr_boxes:
                text = getattr(box, 'text', '') if hasattr(box, 'text') else str(box)
                text_lower = text.strip().lower()

                if any(kw in text_lower for kw in notif_keywords):
                    bbox = getattr(box, 'bbox', (0, 0, 0, 0)) if hasattr(box, 'bbox') else (0, 0, 0, 0)
                    x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]

                    # Check if in notification-typical position (top-right or bottom-right)
                    if ((x > window_width * 0.65 and y < window_height * 0.15) or
                            (x > window_width * 0.65 and y > window_height * 0.80)):
                        notifications.append(LayoutRegion(
                            region_type=RegionType.NOTIFICATION,
                            bounds=(x, y, bw, bh),
                            label=text.strip()[:50],
                            confidence=0.55,
                        ))
                        logger.debug("[LAYOUT] Notification hint: '%s' at (%d,%d)", text[:30], x, y)

        except Exception as e:
            logger.debug("[LAYOUT] Notification detection error: %s", e)

        return notifications

    # ── Diagnostics ───────────────────────────────────────

    def _record_layout(self, app_type: ApplicationType, app_name: str, window_title: str) -> None:
        """Record the latest layout for diagnostics."""
        pass  # self._last_layout is set in segment_layout

    @property
    def last_layout(self) -> Optional[WindowLayout]:
        return self._last_layout

    def report(self) -> Dict[str, Any]:
        """Return diagnostic summary."""
        return {
            "last_app_type": self._last_layout.app_type.value if self._last_layout else "none",
            "last_app_name": self._last_layout.app_name if self._last_layout else "",
            "last_regions": len(self._last_layout.regions) if self._last_layout else 0,
        }


# Re-import cv2 flag for layout analyzer use
_HAS_CV2 = False
try:
    import cv2  # noqa: F811

    _HAS_CV2 = True
except ImportError:
    pass


# Global singleton
layout_analyzer = LayoutAnalyzer()