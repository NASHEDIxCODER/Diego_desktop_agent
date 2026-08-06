"""
Desktop Layout Learning — Leo remembers common UI locations across sessions.

Instead of re-discovering layouts every time, Leo learns:

  - Common application layouts (VS Code, Firefox, Chrome, Terminal, etc.)
  - Where buttons typically are (toolbar position, status bar position)
  - How each app is structured (sidebar, editor, terminal positions)
  - Common UI element locations and their stability
  - Frequent interaction points

Learning is app-specific and persists across sessions via a JSON database.

Usage:
    from learning.desktop_layouts import layout_memory

    # Learn from a perception
    layout_memory.learn(ctx)

    # Predict where a button should be
    position = layout_memory.predict_location("vscode", "Run button")

    # Get layout template for an app
    template = layout_memory.get_template("vscode")

Logging: [LAYOUT]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

@dataclass
class RegionTemplate:
    """A learned region within an application layout."""
    region_type: str = ""               # "toolbar", "sidebar", "editor", "status_bar", etc.
    label: str = ""                     # descriptive label
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h (normalized 0-1)
    confidence: float = 0.0             # how confident we are in this region
    observation_count: int = 0          # how many times we've seen this
    last_seen: float = 0.0              # timestamp


@dataclass
class ElementLocation:
    """A learned UI element location within an application."""
    element_type: str = ""              # "button", "tab", "input", "menu", etc.
    label: str = ""
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # normalized 0-1
    screen_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # absolute pixel coords
    confidence: float = 0.0
    observation_count: int = 0
    stability: float = 0.0              # how much does position vary? (0=stable, 1=volatile)
    last_seen: float = 0.0
    last_screen_size: Tuple[int, int] = (1920, 1080)


@dataclass
class AppLayoutTemplate:
    """
    Learned layout template for a specific application.

    Contains:
      - Known regions (toolbar, sidebar, editor, status bar, etc.)
      - Common UI element locations
      - Layout stability score
      - App-specific interaction patterns
    """
    app_name: str = ""
    app_type: str = ""                  # from ApplicationType
    regions: List[RegionTemplate] = field(default_factory=list)
    elements: Dict[str, ElementLocation] = field(default_factory=dict)  # keyed by label
    observation_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    typical_window_size: Tuple[int, int] = (1920, 1080)
    stability_score: float = 0.0        # overall layout stability (0-1, higher=more stable)
    last_updated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "app_type": self.app_type,
            "regions": [
                {
                    "region_type": r.region_type,
                    "label": r.label,
                    "bounds": list(r.bounds),
                    "confidence": r.confidence,
                    "observation_count": r.observation_count,
                    "last_seen": r.last_seen,
                }
                for r in self.regions
            ],
            "elements": {
                key: {
                    "element_type": e.element_type,
                    "label": e.label,
                    "bounds": list(e.bounds),
                    "screen_bounds": list(e.screen_bounds),
                    "confidence": e.confidence,
                    "observation_count": e.observation_count,
                    "stability": e.stability,
                    "last_seen": e.last_seen,
                    "last_screen_size": list(e.last_screen_size),
                }
                for key, e in self.elements.items()
            },
            "observation_count": self.observation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "typical_window_size": list(self.typical_window_size),
            "stability_score": self.stability_score,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AppLayoutTemplate:
        template = cls(
            app_name=d.get("app_name", ""),
            app_type=d.get("app_type", ""),
            observation_count=d.get("observation_count", 0),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
            typical_window_size=tuple(d.get("typical_window_size", [1920, 1080])),
            stability_score=d.get("stability_score", 0.0),
            last_updated=d.get("last_updated", 0.0),
        )
        template.regions = [
            RegionTemplate(
                region_type=r["region_type"],
                label=r.get("label", ""),
                bounds=tuple(r["bounds"]),
                confidence=r.get("confidence", 0.0),
                observation_count=r.get("observation_count", 0),
                last_seen=r.get("last_seen", 0.0),
            )
            for r in d.get("regions", [])
        ]
        template.elements = {
            key: ElementLocation(
                element_type=e["element_type"],
                label=e.get("label", key),
                bounds=tuple(e["bounds"]),
                screen_bounds=tuple(e.get("screen_bounds", [0, 0, 0, 0])),
                confidence=e.get("confidence", 0.0),
                observation_count=e.get("observation_count", 0),
                stability=e.get("stability", 0.5),
                last_seen=e.get("last_seen", 0.0),
                last_screen_size=tuple(e.get("last_screen_size", [1920, 1080])),
            )
            for key, e in d.get("elements", {}).items()
        }
        return template


# ═══════════════════════════════════════════════════════════════
# Built-in layout templates for common applications
# ═══════════════════════════════════════════════════════════════

BUILTIN_TEMPLATES: Dict[str, AppLayoutTemplate] = {}


def _make_builtin(app_name: str, app_type: str, regions: List[Tuple[str, str, Tuple[float, float, float, float]]],
                  elements: List[Tuple[str, str, Tuple[float, float, float, float]]],
                  window_size: Tuple[int, int] = (1920, 1080)) -> AppLayoutTemplate:
    """Helper to create built-in templates."""
    template = AppLayoutTemplate(
        app_name=app_name,
        app_type=app_type,
        typical_window_size=window_size,
        observation_count=100,
        first_seen=time.time(),
        last_seen=time.time(),
        stability_score=0.95,
        last_updated=time.time(),
    )
    for rt, label, bounds in regions:
        template.regions.append(RegionTemplate(
            region_type=rt, label=label, bounds=tuple(bounds),
            confidence=0.95, observation_count=100, last_seen=time.time(),
        ))
    for et, label, bounds in elements:
        key = f"{et}:{label}"
        template.elements[key] = ElementLocation(
            element_type=et, label=label, bounds=tuple(bounds),
            screen_bounds=(0, 0, 0, 0),
            confidence=0.9, observation_count=100, stability=0.1,
            last_seen=time.time(), last_screen_size=window_size,
        )
    return template


# ── VS Code ─────────────────────────────────────────────────

BUILTIN_TEMPLATES["code"] = _make_builtin(
    "code", "vscode",
    regions=[
        ("menu_bar", "Menu Bar", (0.0, 0.0, 1.0, 0.03)),
        ("tab_bar", "Tab Bar", (0.0, 0.03, 1.0, 0.03)),
        ("toolbar", "Toolbar", (0.0, 0.03, 0.05, 0.05)),
        ("sidebar", "Sidebar", (0.0, 0.06, 0.15, 0.85)),
        ("editor", "Editor", (0.15, 0.06, 0.60, 0.70)),
        ("right_panel", "Right Panel", (0.75, 0.06, 0.25, 0.70)),
        ("bottom_panel", "Terminal", (0.0, 0.76, 1.0, 0.21)),
        ("status_bar", "Status Bar", (0.0, 0.97, 1.0, 0.03)),
    ],
    elements=[
        ("button", "Run", (0.88, 0.04, 0.03, 0.02)),
        ("button", "Debug", (0.91, 0.04, 0.03, 0.02)),
        ("button", "Build", (0.94, 0.04, 0.03, 0.02)),
        ("button", "Terminal", (0.03, 0.05, 0.02, 0.01)),
        ("button", "Extensions", (0.02, 0.08, 0.02, 0.01)),
        ("button", "Search", (0.02, 0.10, 0.02, 0.01)),
        ("button", "Source Control", (0.02, 0.12, 0.02, 0.01)),
        ("tab", "Problems", (0.01, 0.98, 0.04, 0.02)),
        ("tab", "Output", (0.05, 0.98, 0.04, 0.02)),
        ("tab", "Debug Console", (0.09, 0.98, 0.04, 0.02)),
        ("tab", "Terminal", (0.13, 0.98, 0.04, 0.02)),
    ],
)

# ── Firefox ─────────────────────────────────────────────────

BUILTIN_TEMPLATES["firefox"] = _make_builtin(
    "firefox", "browser",
    regions=[
        ("tab_bar", "Tab Bar", (0.0, 0.0, 1.0, 0.04)),
        ("toolbar", "Navigation Bar", (0.0, 0.04, 1.0, 0.04)),
        ("content", "Web Content", (0.0, 0.08, 1.0, 0.88)),
        ("status_bar", "Status Bar", (0.0, 0.96, 1.0, 0.04)),
    ],
    elements=[
        ("button", "Back", (0.01, 0.05, 0.02, 0.02)),
        ("button", "Forward", (0.03, 0.05, 0.02, 0.02)),
        ("button", "Reload", (0.05, 0.05, 0.02, 0.02)),
        ("input", "Address Bar", (0.08, 0.05, 0.60, 0.02)),
        ("button", "Menu", (0.97, 0.01, 0.02, 0.02)),
    ],
)

# ── Chrome ──────────────────────────────────────────────────

BUILTIN_TEMPLATES["chrome"] = _make_builtin(
    "chrome", "browser",
    regions=[
        ("tab_bar", "Tab Bar", (0.0, 0.0, 1.0, 0.04)),
        ("toolbar", "Navigation Bar", (0.0, 0.04, 1.0, 0.04)),
        ("content", "Web Content", (0.0, 0.08, 1.0, 0.92)),
    ],
    elements=[
        ("button", "Back", (0.01, 0.05, 0.02, 0.02)),
        ("button", "Forward", (0.03, 0.05, 0.02, 0.02)),
        ("button", "Reload", (0.05, 0.05, 0.02, 0.02)),
        ("input", "Address Bar", (0.08, 0.05, 0.55, 0.02)),
        ("button", "Extensions", (0.90, 0.05, 0.02, 0.02)),
        ("button", "Menu", (0.95, 0.05, 0.02, 0.02)),
    ],
)

# ── Terminal ────────────────────────────────────────────────

BUILTIN_TEMPLATES["terminal"] = _make_builtin(
    "terminal", "terminal",
    regions=[
        ("menu_bar", "Menu Bar", (0.0, 0.0, 1.0, 0.03)),
        ("content", "Terminal Output", (0.0, 0.03, 1.0, 0.94)),
        ("status_bar", "Status Bar", (0.0, 0.97, 1.0, 0.03)),
    ],
    elements=[
        ("tab", "Terminal Tab", (0.01, 0.01, 0.08, 0.02)),
        ("button", "New Tab", (0.90, 0.01, 0.02, 0.02)),
    ],
)

# ── PyCharm / JetBrains ─────────────────────────────────────

BUILTIN_TEMPLATES["pycharm"] = _make_builtin(
    "pycharm", "jetbrains",
    regions=[
        ("menu_bar", "Menu Bar", (0.0, 0.0, 1.0, 0.03)),
        ("toolbar", "Toolbar", (0.0, 0.03, 1.0, 0.03)),
        ("tab_bar", "Tab Bar", (0.0, 0.06, 1.0, 0.02)),
        ("navigation", "Navigation Bar", (0.0, 0.08, 1.0, 0.02)),
        ("sidebar", "Project", (0.0, 0.10, 0.15, 0.70)),
        ("editor", "Editor", (0.15, 0.10, 0.60, 0.70)),
        ("right_panel", "Right Panel", (0.75, 0.10, 0.25, 0.70)),
        ("bottom_panel", "Bottom Panel", (0.0, 0.80, 1.0, 0.18)),
        ("status_bar", "Status Bar", (0.0, 0.98, 1.0, 0.02)),
    ],
    elements=[
        ("button", "Run", (0.86, 0.05, 0.03, 0.02)),
        ("button", "Debug", (0.89, 0.05, 0.03, 0.02)),
        ("button", "Build", (0.92, 0.05, 0.03, 0.02)),
        ("button", "Stop", (0.84, 0.05, 0.02, 0.02)),
        ("tab", "Project", (0.0, 0.12, 0.06, 0.02)),
        ("tab", "Structure", (0.0, 0.14, 0.06, 0.02)),
        ("tab", "Terminal", (0.01, 0.99, 0.06, 0.01)),
        ("tab", "Problems", (0.08, 0.99, 0.06, 0.01)),
        ("tab", "Debug", (0.15, 0.99, 0.06, 0.01)),
        ("tab", "Git", (0.22, 0.99, 0.04, 0.01)),
    ],
)

# ── Discord ─────────────────────────────────────────────────

BUILTIN_TEMPLATES["discord"] = _make_builtin(
    "discord", "chat",
    regions=[
        ("sidebar", "Servers", (0.0, 0.0, 0.04, 1.0)),
        ("sidebar", "Channels", (0.04, 0.0, 0.15, 1.0)),
        ("content", "Messages", (0.19, 0.0, 0.62, 0.80)),
        ("right_panel", "Members", (0.81, 0.0, 0.19, 1.0)),
        ("bottom_panel", "Message Input", (0.19, 0.80, 0.62, 0.20)),
    ],
    elements=[
        ("input", "Message Input", (0.20, 0.82, 0.58, 0.15)),
        ("button", "Send", (0.79, 0.82, 0.02, 0.12)),
    ],
)

# ── Spotify ─────────────────────────────────────────────────

BUILTIN_TEMPLATES["spotify"] = _make_builtin(
    "spotify", "music",
    regions=[
        ("sidebar", "Library", (0.0, 0.0, 0.15, 0.85)),
        ("content", "Main Content", (0.15, 0.0, 0.85, 0.85)),
        ("bottom_panel", "Now Playing", (0.0, 0.85, 1.0, 0.15)),
    ],
    elements=[
        ("button", "Play", (0.48, 0.92, 0.02, 0.03)),
        ("button", "Previous", (0.44, 0.92, 0.02, 0.03)),
        ("button", "Next", (0.52, 0.92, 0.02, 0.03)),
        ("slider", "Progress", (0.10, 0.96, 0.80, 0.01)),
        ("slider", "Volume", (0.92, 0.96, 0.06, 0.01)),
    ],
)

# ── Gmail ───────────────────────────────────────────────────

BUILTIN_TEMPLATES["gmail"] = _make_builtin(
    "gmail", "email",
    regions=[
        ("sidebar", "Folders", (0.0, 0.08, 0.15, 0.92)),
        ("content", "Email List", (0.15, 0.12, 0.45, 0.88)),
        ("right_panel", "Reading Pane", (0.60, 0.12, 0.40, 0.88)),
    ],
    elements=[
        ("button", "Compose", (0.02, 0.10, 0.10, 0.03)),
        ("input", "Search", (0.35, 0.09, 0.25, 0.02)),
    ],
)

# ── YouTube ─────────────────────────────────────────────────

BUILTIN_TEMPLATES["youtube"] = _make_builtin(
    "youtube", "video",
    regions=[
        ("sidebar", "Sidebar", (0.0, 0.05, 0.15, 0.95)),
        ("content", "Video Player / Feed", (0.15, 0.05, 0.85, 0.95)),
    ],
    elements=[
        ("button", "Search", (0.62, 0.01, 0.04, 0.02)),
        ("input", "Search", (0.25, 0.01, 0.35, 0.02)),
        ("button", "Play", (0.45, 0.70, 0.04, 0.04)),
        ("button", "Like", (0.35, 0.68, 0.03, 0.02)),
    ],
)

# ── GitHub ──────────────────────────────────────────────────

BUILTIN_TEMPLATES["github"] = _make_builtin(
    "github", "code_review",
    regions=[
        ("toolbar", "Navigation", (0.0, 0.0, 1.0, 0.05)),
        ("content", "Content", (0.0, 0.05, 1.0, 0.95)),
    ],
    elements=[
        ("input", "Search", (0.45, 0.01, 0.20, 0.03)),
        ("button", "Code", (0.03, 0.06, 0.04, 0.02)),
        ("button", "Issues", (0.07, 0.06, 0.04, 0.02)),
        ("button", "Pull Requests", (0.11, 0.06, 0.04, 0.02)),
        ("button", "Actions", (0.15, 0.06, 0.04, 0.02)),
    ],
)


# ═══════════════════════════════════════════════════════════════
# LayoutMemory — learns and remembers desktop layouts
# ═══════════════════════════════════════════════════════════════

class LayoutMemory:
    """
    Learns and remembers desktop application layouts.

    Persists to a JSON file so Leo retains layout knowledge
    across restarts. Combines built-in templates with observed
    learning from real desktop usage.
    """

    STORAGE_PATH = Path.home() / ".leo" / "desktop_layouts.json"

    def __init__(self):
        self._templates: Dict[str, AppLayoutTemplate] = {}
        self._loaded: bool = False

    # ── Load / Save ──────────────────────────────────────────

    def load(self) -> None:
        """Load learned layouts from disk and merge with builtins."""
        # Start with built-in templates
        self._templates = dict(BUILTIN_TEMPLATES)

        # Load learned data from disk
        try:
            if self.STORAGE_PATH.exists():
                data = json.loads(self.STORAGE_PATH.read_text())
                for key, d in data.items():
                    template = AppLayoutTemplate.from_dict(d)
                    # Merge with builtin if exists, otherwise add
                    if key in self._templates:
                        self._merge_template(self._templates[key], template)
                    else:
                        self._templates[key] = template
                logger.info("[LAYOUT] Loaded %d layouts from %s",
                             len(data), self.STORAGE_PATH)
        except Exception as e:
            logger.warning("[LAYOUT] Failed to load layouts: %s", e)

        self._loaded = True
        logger.info("[LAYOUT] %d layout templates available (%d builtin, %d learned)",
                     len(self._templates),
                     len(BUILTIN_TEMPLATES),
                     len(self._templates) - len(BUILTIN_TEMPLATES))

    def save(self) -> None:
        """Save all learned layouts to disk."""
        try:
            self.STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                key: template.to_dict()
                for key, template in self._templates.items()
            }
            self.STORAGE_PATH.write_text(json.dumps(data, indent=2))
            logger.debug("[LAYOUT] Saved %d layout templates", len(data))
        except Exception as e:
            logger.warning("[LAYOUT] Failed to save layouts: %s", e)

    # ── Learning ─────────────────────────────────────────────

    def learn(self, app_name: str, app_type: str,
              regions: List[Dict[str, Any]],
              elements: List[Dict[str, Any]],
              window_size: Tuple[int, int] = (1920, 1080)) -> None:
        """
        Learn from a perception observation.

        Updates the layout template for an application with new
        region and element observations. Uses exponential moving
        average for stability.

        Args:
            app_name: Process name (e.g., "code", "firefox")
            app_type: Application type (e.g., "vscode", "browser")
            regions: List of region dicts with region_type, label, bounds
            elements: List of element dicts with element_type, label, bounds
            window_size: Current window dimensions for normalization
        """
        if not self._loaded:
            self.load()

        key = app_name.lower()
        now = time.time()

        # Get or create template
        if key not in self._templates:
            self._templates[key] = AppLayoutTemplate(
                app_name=app_name,
                app_type=app_type,
                first_seen=now,
            )

        template = self._templates[key]
        template.last_seen = now
        template.observation_count += 1
        template.last_updated = now

        w, h = window_size
        if w > 0 and h > 0:
            template.typical_window_size = window_size

        # ── Update regions ───────────────────────────────
        for region_data in regions:
            region_type = region_data.get("region_type", "")
            label = region_data.get("label", "")
            bounds = region_data.get("bounds", (0, 0, 0, 0))

            # Normalize bounds
            if w > 0 and h > 0:
                nx, ny = bounds[0] / w, bounds[1] / h
                nw, nh = bounds[2] / w, bounds[3] / h
                norm_bounds = (nx, ny, nw, nh)
            else:
                norm_bounds = tuple(bounds)

            # Find existing region or create new
            existing = None
            for r in template.regions:
                if r.region_type == region_type:
                    existing = r
                    break

            if existing:
                # Update with exponential moving average
                alpha = 0.3  # learning rate
                old = existing.bounds
                new = tuple(
                    alpha * n + (1 - alpha) * o
                    for n, o in zip(norm_bounds, old)
                )
                existing.bounds = new
                existing.confidence = min(1.0, existing.confidence + 0.05)
                existing.observation_count += 1
                existing.last_seen = now
            else:
                template.regions.append(RegionTemplate(
                    region_type=region_type,
                    label=label,
                    bounds=norm_bounds,
                    confidence=0.5,
                    observation_count=1,
                    last_seen=now,
                ))

        # ── Update elements ──────────────────────────────
        for elem_data in elements:
            element_type = elem_data.get("element_type", "")
            label = elem_data.get("label", "")
            bounds = elem_data.get("bounds", (0, 0, 0, 0))

            elem_key = f"{element_type}:{label}"

            # Normalize bounds
            if w > 0 and h > 0:
                nx, ny = bounds[0] / w, bounds[1] / h
                nw, nh = bounds[2] / w, bounds[3] / h
                norm_bounds = (nx, ny, nw, nh)
            else:
                norm_bounds = tuple(bounds)

            if elem_key in template.elements:
                existing = template.elements[elem_key]
                # Calculate stability (how much position varies)
                old = existing.bounds
                dist = sum(abs(n - o) for n, o in zip(norm_bounds, old))
                existing.stability = min(1.0, existing.stability * 0.9 + dist * 0.1)

                # Update position with EMA
                alpha = 0.3
                new_bounds = tuple(
                    alpha * n + (1 - alpha) * o
                    for n, o in zip(norm_bounds, old)
                )
                existing.bounds = new_bounds
                existing.screen_bounds = tuple(bounds)
                existing.confidence = min(1.0, existing.confidence + 0.05)
                existing.observation_count += 1
                existing.last_seen = now
                existing.last_screen_size = window_size
            else:
                template.elements[elem_key] = ElementLocation(
                    element_type=element_type,
                    label=label,
                    bounds=norm_bounds,
                    screen_bounds=tuple(bounds),
                    confidence=0.5,
                    observation_count=1,
                    stability=0.5,
                    last_seen=now,
                    last_screen_size=window_size,
                )

        # ── Update stability score ───────────────────────
        if template.regions:
            avg_confidence = sum(r.confidence for r in template.regions) / len(template.regions)
            template.stability_score = avg_confidence

        logger.debug("[LAYOUT] Learned: %s (obs=%d, stability=%.2f)",
                      key, template.observation_count, template.stability_score)

        # Auto-save periodically
        if template.observation_count % 10 == 0:
            self.save()

    # ── Querying ─────────────────────────────────────────────

    def get_template(self, app_name: str) -> Optional[AppLayoutTemplate]:
        """Get the layout template for an application."""
        if not self._loaded:
            self.load()
        return self._templates.get(app_name.lower())

    def predict_location(
        self, app_name: str, element_label: str,
        window_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Predict where a UI element should be on screen.

        Args:
            app_name: Process name (e.g., "code")
            element_label: Label of the element (e.g., "Run")
            window_size: Current window size for denormalization

        Returns:
            (x, y) center coordinates in screen pixels, or None
        """
        template = self.get_template(app_name)
        if template is None:
            return None

        # Search elements by label
        for key, elem in template.elements.items():
            if element_label.lower() in elem.label.lower():
                w, h = window_size or template.typical_window_size
                if w > 0 and h > 0:
                    nx, ny, nw, nh = elem.bounds
                    x = int(nx * w + nw * w / 2)
                    y = int(ny * h + nh * h / 2)
                    logger.debug("[LAYOUT] Predicted '%s' at (%d, %d) in %s (conf=%.2f)",
                                  element_label, x, y, app_name, elem.confidence)
                    return (x, y)
                elif elem.screen_bounds != (0, 0, 0, 0):
                    sx, sy, sw, sh = elem.screen_bounds
                    return (sx + sw // 2, sy + sh // 2)

        return None

    def predict_region(
        self, app_name: str, region_type: str,
        window_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Predict where a layout region should be.

        Args:
            app_name: Process name
            region_type: Region type ("toolbar", "sidebar", etc.)
            window_size: Current window size

        Returns:
            (x, y, w, h) in screen pixels, or None
        """
        template = self.get_template(app_name)
        if template is None:
            return None

        for region in template.regions:
            if region.region_type == region_type:
                w, h = window_size or template.typical_window_size
                if w > 0 and h > 0:
                    nx, ny, nw, nh = region.bounds
                    return (
                        int(nx * w), int(ny * h),
                        int(nw * w), int(nh * h),
                    )
                return region.bounds
        return None

    def get_clickable_elements(self, app_name: str) -> List[Dict[str, Any]]:
        """Get all known clickable elements for an app."""
        template = self.get_template(app_name)
        if template is None:
            return []

        results = []
        clickable_types = {"button", "tab", "link", "menu_item", "checkbox"}
        for key, elem in template.elements.items():
            if elem.element_type in clickable_types:
                results.append({
                    "type": elem.element_type,
                    "label": elem.label,
                    "confidence": elem.confidence,
                    "stability": elem.stability,
                    "observation_count": elem.observation_count,
                })
        return results

    def get_all_known_apps(self) -> List[str]:
        """List all applications with known layouts."""
        if not self._loaded:
            self.load()
        return sorted(self._templates.keys())

    def get_template_summary(self, app_name: str) -> str:
        """Get a text summary of a layout template for LLM context."""
        template = self.get_template(app_name)
        if template is None:
            return f"No layout known for {app_name}"

        lines = [
            f"Layout for {template.app_name} ({template.app_type}):",
            f"  Observed {template.observation_count} times, stability={template.stability_score:.1%}",
            f"  Typical window: {template.typical_window_size[0]}x{template.typical_window_size[1]}",
        ]

        if template.regions:
            lines.append("  Regions:")
            for r in template.regions:
                lines.append(f"    {r.region_type}: {r.label} (conf={r.confidence:.1%})")

        clickable = self.get_clickable_elements(app_name)
        if clickable:
            lines.append("  Common clickable elements:")
            for e in sorted(clickable, key=lambda x: x["confidence"], reverse=True)[:10]:
                lines.append(f"    {e['type']}: {e['label']} (conf={e['confidence']:.1%})")

        return "\n".join(lines)

    # ── Merge ─────────────────────────────────────────────────

    @staticmethod
    def _merge_template(target: AppLayoutTemplate, source: AppLayoutTemplate) -> None:
        """Merge learned template into target (builtin)."""
        target.observation_count = max(target.observation_count, source.observation_count)
        target.last_seen = max(target.last_seen, source.last_seen)
        target.last_updated = max(target.last_updated, source.last_updated)

        # Merge regions
        for src_region in source.regions:
            found = False
            for tgt_region in target.regions:
                if tgt_region.region_type == src_region.region_type:
                    tgt_region.observation_count += src_region.observation_count
                    tgt_region.confidence = max(tgt_region.confidence, src_region.confidence)
                    tgt_region.last_seen = max(tgt_region.last_seen, src_region.last_seen)
                    found = True
                    break
            if not found:
                target.regions.append(src_region)

        # Merge elements
        for key, src_elem in source.elements.items():
            if key in target.elements:
                tgt_elem = target.elements[key]
                tgt_elem.observation_count += src_elem.observation_count
                tgt_elem.confidence = max(tgt_elem.confidence, src_elem.confidence)
                tgt_elem.last_seen = max(tgt_elem.last_seen, src_elem.last_seen)
            else:
                target.elements[key] = src_elem

        # Recalculate stability
        if target.regions:
            avg_conf = sum(r.confidence for r in target.regions) / len(target.regions)
            target.stability_score = max(target.stability_score, avg_conf)

    # ── Diagnostics ──────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Return comprehensive diagnostic report."""
        if not self._loaded:
            self.load()

        app_summaries = {}
        for key, template in self._templates.items():
            app_summaries[key] = {
                "app_type": template.app_type,
                "observations": template.observation_count,
                "regions": len(template.regions),
                "elements": len(template.elements),
                "stability": round(template.stability_score, 2),
                "last_seen": template.last_seen,
            }

        return {
            "total_apps": len(self._templates),
            "builtin_apps": len(BUILTIN_TEMPLATES),
            "learned_apps": len(self._templates) - len(BUILTIN_TEMPLATES),
            "apps": app_summaries,
        }


# Global singleton
layout_memory = LayoutMemory()