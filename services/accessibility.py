"""
AccessibilityTree — Structured accessibility data from AT-SPI, GTK, Qt, and browsers.

Diego's accessibility-first strategy:
  1. AT-SPI (Linux accessibility bus) — richest structured data
  2. GTK accessibility — GTK app widgets
  3. Qt accessibility — Qt app widgets
  4. Browser accessibility — Chrome/Firefox DevTools Protocol
  5. OCR — FALLBACK ONLY when none of the above are available

This module provides a unified interface to all accessibility backends.
NEVER run OCR if structured accessibility data is available.

Usage:
    from services.accessibility import accessibility_tree

    tree = accessibility_tree.get_tree()  # returns A11yNode tree
    if tree and tree.has_meaningful_content():
        # Use structured data — skip OCR
        elements = tree.find_clickable()
    else:
        # Fall back to OCR
        ...

Logging: [A11Y]
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

class A11yRole(str, Enum):
    """Accessibility roles (WAI-ARIA + AT-SPI)."""
    WINDOW = "window"
    DIALOG = "dialog"
    ALERT = "alert"
    BUTTON = "button"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    MENU_BAR = "menu_bar"
    TOOLBAR = "toolbar"
    TAB = "tab"
    TAB_LIST = "tab_list"
    TAB_PANEL = "tab_panel"
    LINK = "link"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    COMBO_BOX = "combo_box"
    TEXT_BOX = "text_box"
    TEXT = "text"
    LABEL = "label"
    HEADING = "heading"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    TABLE_ROW = "table_row"
    TREE = "tree"
    TREE_ITEM = "tree_item"
    PANEL = "panel"
    SCROLL_BAR = "scroll_bar"
    SLIDER = "slider"
    PROGRESS_BAR = "progress_bar"
    STATUS_BAR = "status_bar"
    TOOL_TIP = "tool_tip"
    NOTIFICATION = "notification"
    SEPARATOR = "separator"
    IMAGE = "image"
    ICON = "icon"
    DOCUMENT = "document"
    SECTION = "section"
    GROUP = "group"
    UNKNOWN = "unknown"


class A11yState(str, Enum):
    """Accessibility states."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    FOCUSED = "focused"
    SELECTED = "selected"
    CHECKED = "checked"
    PRESSED = "pressed"
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"
    VISIBLE = "visible"
    HIDDEN = "hidden"
    ACTIVE = "active"
    BUSY = "busy"
    READ_ONLY = "read_only"
    REQUIRED = "required"
    INVALID = "invalid"


class A11yBackend(str, Enum):
    """Which accessibility backend provided the data."""
    AT_SPI = "at-spi"
    GTK = "gtk"
    QT = "qt"
    BROWSER_CDP = "browser_cdp"
    X11_WINDOW = "x11_window"
    NONE = "none"


@dataclass
class A11yNode:
    """
    A single node in the accessibility tree.

    Mirrors the structure of AT-SPI's Accessible object but
    unified across all backends.
    """
    role: A11yRole = A11yRole.UNKNOWN
    name: str = ""                      # accessible name (label)
    description: str = ""               # accessible description
    value: str = ""                     # current value (for text boxes, sliders, etc.)
    states: Set[A11yState] = field(default_factory=set)
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    children: List[A11yNode] = field(default_factory=list)
    parent: Optional[A11yNode] = None
    backend: A11yBackend = A11yBackend.NONE
    confidence: float = 1.0             # 1.0 for real a11y data, lower for heuristics
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Properties ──────────────────────────────────────────

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

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def enabled(self) -> bool:
        return A11yState.ENABLED in self.states and A11yState.DISABLED not in self.states

    @property
    def focused(self) -> bool:
        return A11yState.FOCUSED in self.states

    @property
    def visible(self) -> bool:
        return A11yState.VISIBLE in self.states and A11yState.HIDDEN not in self.states

    @property
    def is_clickable(self) -> bool:
        return self.role in (
            A11yRole.BUTTON, A11yRole.MENU_ITEM, A11yRole.LINK,
            A11yRole.CHECKBOX, A11yRole.RADIO, A11yRole.TAB,
            A11yRole.COMBO_BOX, A11yRole.LIST_ITEM, A11yRole.TREE_ITEM,
        )

    @property
    def is_text_input(self) -> bool:
        return self.role in (A11yRole.TEXT_BOX,)

    @property
    def is_container(self) -> bool:
        return self.role in (
            A11yRole.WINDOW, A11yRole.DIALOG, A11yRole.PANEL,
            A11yRole.MENU, A11yRole.MENU_BAR, A11yRole.TOOLBAR,
            A11yRole.TAB_LIST, A11yRole.LIST, A11yRole.TABLE,
            A11yRole.TREE, A11yRole.GROUP, A11yRole.SECTION,
            A11yRole.DOCUMENT, A11yRole.STATUS_BAR,
        )

    # ── Tree operations ─────────────────────────────────────

    def add_child(self, child: A11yNode) -> None:
        child.parent = self
        self.children.append(child)

    def walk(self) -> List[A11yNode]:
        """Depth-first traversal of all nodes."""
        nodes: List[A11yNode] = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes

    def find_by_role(self, role: A11yRole) -> List[A11yNode]:
        """Find all nodes with a given role."""
        return [n for n in self.walk() if n.role == role]

    def find_by_name(self, name: str, case_sensitive: bool = False) -> List[A11yNode]:
        """Find all nodes whose name contains the given text."""
        compare = self.name if case_sensitive else self.name.lower()
        target = name if case_sensitive else name.lower()
        return [n for n in self.walk() if target in compare]

    def find_clickable(self) -> List[A11yNode]:
        """Return all clickable, enabled, visible elements."""
        return [n for n in self.walk() if n.is_clickable and n.enabled and n.visible]

    def find_focused(self) -> Optional[A11yNode]:
        """Return the currently focused element, if any."""
        for n in self.walk():
            if n.focused:
                return n
        return None

    def find_text_inputs(self) -> List[A11yNode]:
        """Return all text input fields."""
        return [n for n in self.walk() if n.is_text_input and n.enabled and n.visible]

    def has_meaningful_content(self) -> bool:
        """Check if this tree has enough structured data to skip OCR."""
        all_nodes = self.walk()
        # X11 window tree provides window-level info — accept if we have
        # a named window with process info (even a single node is meaningful)
        if self.backend == A11yBackend.X11_WINDOW:
            return len(all_nodes) >= 1 and bool(self.name) and bool(self.metadata.get("process"))
        if len(all_nodes) < 3:
            return False
        # Must have at least some interactive elements or text
        clickable = sum(1 for n in all_nodes if n.is_clickable)
        text_inputs = sum(1 for n in all_nodes if n.is_text_input)
        named = sum(1 for n in all_nodes if n.name)
        return (clickable + text_inputs + named) >= 5

    def to_compact_str(self, indent: int = 0) -> str:
        """Compact string representation for LLM context."""
        prefix = "  " * indent
        role_icon = self._role_icon()
        name_part = f' "{self.name}"' if self.name else ""
        bounds_part = f" @({self.x},{self.y},{self.width}x{self.height})" if self.bounds != (0, 0, 0, 0) else ""
        state_str = ""
        if not self.enabled:
            state_str += " [disabled]"
        if self.focused:
            state_str += " [FOCUSED]"
        if A11yState.CHECKED in self.states:
            state_str += " [checked]"
        if A11yState.EXPANDED in self.states:
            state_str += " [expanded]"
        backend_str = f" [{self.backend.value}]" if self.backend != A11yBackend.NONE else ""
        line = f"{prefix}{role_icon} {self.role.value}{name_part}{bounds_part}{state_str}{backend_str}"
        for child in self.children:
            line += "\n" + child.to_compact_str(indent + 1)
        return line

    @staticmethod
    def _role_icon() -> str:
        return "♿"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "role": self.role.value,
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "states": [s.value for s in self.states],
            "bounds": list(self.bounds),
            "backend": self.backend.value,
            "confidence": self.confidence,
            "children": [c.to_dict() for c in self.children],
        }


# ═══════════════════════════════════════════════════════════════
# AccessibilityTree — unified access to all backends
# ═══════════════════════════════════════════════════════════════

class AccessibilityTree:
    """
    Unified accessibility tree across AT-SPI, GTK, Qt, and browsers.

    Priority order:
      1. AT-SPI (via pyatspi or at-spi2 tools)
      2. Browser CDP (Chrome DevTools Protocol)
      3. X11 window tree (xdotool + xprop)
      4. None (fall back to OCR)

    Caches the tree for the current focused window. Invalidates when
    the focused window changes.
    """

    def __init__(self):
        self._last_tree: Optional[A11yNode] = None
        self._last_window_id: str = ""
        self._last_backend: A11yBackend = A11yBackend.NONE
        self._available_backends: List[A11yBackend] = []
        self._initialized: bool = False

    def initialize(self) -> bool:
        """Detect which accessibility backends are available."""
        self._available_backends = []

        # Check AT-SPI
        if self._check_at_spi():
            self._available_backends.append(A11yBackend.AT_SPI)
            logger.info("[A11Y] AT-SPI backend available")

        # Check browser CDP
        if self._check_browser_cdp():
            self._available_backends.append(A11yBackend.BROWSER_CDP)
            logger.info("[A11Y] Browser CDP backend available")

        # X11 window tree (always available on X11)
        if os.environ.get("XDG_SESSION_TYPE", "").lower() in ("x11", "") or shutil.which("xdotool"):
            self._available_backends.append(A11yBackend.X11_WINDOW)
            logger.info("[A11Y] X11 window tree backend available")

        self._initialized = True
        logger.info("[A11Y] Initialized — backends: %s",
                     ", ".join(b.value for b in self._available_backends))
        return len(self._available_backends) > 0

    @property
    def ready(self) -> bool:
        return self._initialized and len(self._available_backends) > 0

    @property
    def available_backends(self) -> List[A11yBackend]:
        return list(self._available_backends)

    # ── Main API ────────────────────────────────────────────

    def get_tree(self, force_refresh: bool = False) -> Optional[A11yNode]:
        """
        Get the accessibility tree for the currently focused window.

        Returns None if no accessibility data is available (OCR fallback needed).
        Caches the result until the focused window changes.
        """
        current_window_id = self._get_focused_window_id()

        # Return cached if window hasn't changed
        if not force_refresh and self._last_tree is not None and current_window_id == self._last_window_id:
            logger.debug("[A11Y] Returning cached tree for window %s", current_window_id[:20])
            return self._last_tree

        self._last_window_id = current_window_id

        # Try backends in priority order
        for backend in self._available_backends:
            try:
                if backend == A11yBackend.AT_SPI:
                    tree = self._get_at_spi_tree()
                elif backend == A11yBackend.BROWSER_CDP:
                    tree = self._get_browser_cdp_tree()
                elif backend == A11yBackend.X11_WINDOW:
                    tree = self._get_x11_window_tree()
                else:
                    continue

                if tree is not None and tree.has_meaningful_content():
                    self._last_tree = tree
                    self._last_backend = backend
                    logger.info("[A11Y] Tree from %s: %d nodes, %d clickable",
                                 backend.value, len(tree.walk()),
                                 len(tree.find_clickable()))
                    return tree
            except Exception as e:
                logger.debug("[A11Y] Backend %s failed: %s", backend.value, e)

        logger.debug("[A11Y] No accessibility data available — OCR fallback needed")
        self._last_backend = A11yBackend.NONE
        return None

    def get_focused_element(self) -> Optional[A11yNode]:
        """Get the currently focused UI element from the accessibility tree."""
        tree = self.get_tree()
        if tree is None:
            return None
        return tree.find_focused()

    def find_element(self, name: str, role: Optional[A11yRole] = None) -> List[A11yNode]:
        """Search for elements by name and optional role."""
        tree = self.get_tree()
        if tree is None:
            return []
        results = tree.find_by_name(name)
        if role is not None:
            results = [n for n in results if n.role == role]
        return results

    def get_clickable_elements(self) -> List[A11yNode]:
        """Get all clickable elements on screen."""
        tree = self.get_tree()
        if tree is None:
            return []
        return tree.find_clickable()

    def get_text_inputs(self) -> List[A11yNode]:
        """Get all text input fields on screen."""
        tree = self.get_tree()
        if tree is None:
            return []
        return tree.find_text_inputs()

    def should_skip_ocr(self) -> bool:
        """
        Determine if OCR can be skipped because structured accessibility
        data is available and sufficient.
        """
        tree = self.get_tree()
        if tree is None:
            return False
        return tree.has_meaningful_content()

    # ── Backend: AT-SPI ─────────────────────────────────────

    @staticmethod
    def _check_at_spi() -> bool:
        """Check if AT-SPI is available."""
        # Check for pyatspi2 Python module
        try:
            import gi
            gi.require_version('Atspi', '2.0')
            from gi.repository import Atspi
            return True
        except (ImportError, ValueError):
            pass

        # Check for at-spi2 command-line tools
        if shutil.which("at-spi-bus-launcher"):
            return True

        # Check if at-spi bus is running
        try:
            result = subprocess.run(
                ["busctl", "--user", "status", "org.a11y.Bus"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

        return False

    def _get_at_spi_tree(self) -> Optional[A11yNode]:
        """Get accessibility tree via AT-SPI."""
        try:
            import gi
            gi.require_version('Atspi', '2.0')
            from gi.repository import Atspi

            # Get the desktop
            desktop = Atspi.get_desktop(0)
            if desktop is None:
                return None

            # Find the focused window
            focused_app = None
            for i in range(desktop.get_child_count()):
                app = desktop.get_child_at_index(i)
                if app is None:
                    continue
                for j in range(app.get_child_count()):
                    window = app.get_child_at_index(j)
                    if window is None:
                        continue
                    state_set = window.get_state_set()
                    if state_set and state_set.contains(Atspi.StateType.FOCUSED):
                        focused_app = window
                        break
                if focused_app:
                    break

            if focused_app is None:
                # Fall back to first window
                for i in range(desktop.get_child_count()):
                    app = desktop.get_child_at_index(i)
                    if app and app.get_child_count() > 0:
                        focused_app = app.get_child_at_index(0)
                        break

            if focused_app is None:
                return None

            return self._convert_at_spi_node(focused_app)

        except Exception as e:
            logger.debug("[A11Y] AT-SPI error: %s", e)
            return None

    def _convert_at_spi_node(self, atspi_obj, depth: int = 0) -> Optional[A11yNode]:
        """Convert an AT-SPI accessible object to an A11yNode."""
        if depth > 50:  # Prevent infinite recursion
            return None

        try:
            import gi
            gi.require_version('Atspi', '2.0')
            from gi.repository import Atspi

            # Get role
            role = A11yRole.UNKNOWN
            try:
                atspi_role = atspi_obj.get_role()
                role = self._map_at_spi_role(atspi_role)
            except Exception:
                pass

            # Get name
            name = ""
            try:
                name = atspi_obj.get_name() or ""
            except Exception:
                pass

            # Get description
            description = ""
            try:
                description = atspi_obj.get_description() or ""
            except Exception:
                pass

            # Get states
            states: Set[A11yState] = set()
            try:
                state_set = atspi_obj.get_state_set()
                states = self._map_at_spi_states(state_set)
            except Exception:
                pass

            # Get bounds
            bounds = (0, 0, 0, 0)
            try:
                component = atspi_obj.query_component()
                if component:
                    extents = component.get_extents(Atspi.CoordType.SCREEN)
                    bounds = (extents.x, extents.y, extents.width, extents.height)
            except Exception:
                pass

            # Get value
            value = ""
            try:
                if role in (A11yRole.TEXT_BOX, A11yRole.COMBO_BOX, A11yRole.SLIDER):
                    value_iface = atspi_obj.query_value()
                    if value_iface:
                        value = str(value_iface.get_current_value())
            except Exception:
                pass

            node = A11yNode(
                role=role,
                name=name,
                description=description,
                value=value,
                states=states,
                bounds=bounds,
                backend=A11yBackend.AT_SPI,
                confidence=1.0,
            )

            # Recurse children
            try:
                for i in range(atspi_obj.get_child_count()):
                    child = atspi_obj.get_child_at_index(i)
                    if child is not None:
                        child_node = self._convert_at_spi_node(child, depth + 1)
                        if child_node is not None:
                            node.add_child(child_node)
            except Exception:
                pass

            return node

        except Exception as e:
            logger.debug("[A11Y] AT-SPI node conversion error: %s", e)
            return None

    @staticmethod
    def _map_at_spi_role(atspi_role) -> A11yRole:
        """Map AT-SPI role enum to A11yRole."""
        role_name = str(atspi_role).lower().replace("role_", "")
        mapping = {
            "window": A11yRole.WINDOW,
            "frame": A11yRole.WINDOW,
            "dialog": A11yRole.DIALOG,
            "alert": A11yRole.ALERT,
            "push_button": A11yRole.BUTTON,
            "toggle_button": A11yRole.BUTTON,
            "button": A11yRole.BUTTON,
            "menu": A11yRole.MENU,
            "menu_item": A11yRole.MENU_ITEM,
            "menu_bar": A11yRole.MENU_BAR,
            "tool_bar": A11yRole.TOOLBAR,
            "page_tab": A11yRole.TAB,
            "page_tab_list": A11yRole.TAB_LIST,
            "link": A11yRole.LINK,
            "check_box": A11yRole.CHECKBOX,
            "radio_button": A11yRole.RADIO,
            "combo_box": A11yRole.COMBO_BOX,
            "text": A11yRole.TEXT_BOX,
            "label": A11yRole.LABEL,
            "heading": A11yRole.HEADING,
            "list": A11yRole.LIST,
            "list_item": A11yRole.LIST_ITEM,
            "table": A11yRole.TABLE,
            "table_cell": A11yRole.TABLE_CELL,
            "table_row": A11yRole.TABLE_ROW,
            "tree": A11yRole.TREE,
            "tree_item": A11yRole.TREE_ITEM,
            "panel": A11yRole.PANEL,
            "scroll_bar": A11yRole.SCROLL_BAR,
            "slider": A11yRole.SLIDER,
            "progress_bar": A11yRole.PROGRESS_BAR,
            "status_bar": A11yRole.STATUS_BAR,
            "tool_tip": A11yRole.TOOL_TIP,
            "notification": A11yRole.NOTIFICATION,
            "separator": A11yRole.SEPARATOR,
            "image": A11yRole.IMAGE,
            "icon": A11yRole.ICON,
            "document": A11yRole.DOCUMENT,
            "section": A11yRole.SECTION,
            "group": A11yRole.GROUP,
        }
        return mapping.get(role_name, A11yRole.UNKNOWN)

    @staticmethod
    def _map_at_spi_states(state_set) -> Set[A11yState]:
        """Map AT-SPI state set to A11yState set."""
        states: Set[A11yState] = set()
        try:
            import gi
            gi.require_version('Atspi', '2.0')
            from gi.repository import Atspi

            state_map = {
                Atspi.StateType.ENABLED: A11yState.ENABLED,
                Atspi.StateType.SENSITIVE: A11yState.ENABLED,
                Atspi.StateType.FOCUSED: A11yState.FOCUSED,
                Atspi.StateType.SELECTED: A11yState.SELECTED,
                Atspi.StateType.CHECKED: A11yState.CHECKED,
                Atspi.StateType.PRESSED: A11yState.PRESSED,
                Atspi.StateType.EXPANDED: A11yState.EXPANDED,
                Atspi.StateType.COLLAPSED: A11yState.COLLAPSED,
                Atspi.StateType.VISIBLE: A11yState.VISIBLE,
                Atspi.StateType.SHOWING: A11yState.VISIBLE,
                Atspi.StateType.ACTIVE: A11yState.ACTIVE,
                Atspi.StateType.BUSY: A11yState.BUSY,
                Atspi.StateType.READ_ONLY: A11yState.READ_ONLY,
                Atspi.StateType.REQUIRED: A11yState.REQUIRED,
                Atspi.StateType.INVALID: A11yState.INVALID,
            }
            for atspi_state, a11y_state in state_map.items():
                if state_set.contains(atspi_state):
                    states.add(a11y_state)
        except Exception:
            pass
        return states

    # ── Backend: Browser CDP ────────────────────────────────

    @staticmethod
    def _check_browser_cdp() -> bool:
        """Check if any browser with CDP is running."""
        # Check for Chrome DevTools Protocol port
        for port in [9222, 9223, 9224]:  # Common CDP ports
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                result = s.connect_ex(('127.0.0.1', port))
                s.close()
                if result == 0:
                    return True
            except Exception:
                pass
        return False

    def _get_browser_cdp_tree(self) -> Optional[A11yNode]:
        """Get accessibility tree from browser via Chrome DevTools Protocol."""
        try:
            import json
            import urllib.request

            # Try common CDP ports
            for port in [9222, 9223, 9224]:
                try:
                    # Get list of pages
                    resp = urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/json", timeout=2
                    )
                    pages = json.loads(resp.read())

                    for page in pages:
                        if page.get("type") == "page":
                            ws_url = page.get("webSocketDebuggerUrl")
                            if not ws_url:
                                continue

                            # We need websocket for full CDP — use subprocess with node
                            # for a lightweight CDP client
                            title = page.get("title", "")
                            url = page.get("url", "")

                            # Build a minimal tree from page metadata
                            root = A11yNode(
                                role=A11yRole.DOCUMENT,
                                name=title,
                                description=url,
                                backend=A11yBackend.BROWSER_CDP,
                                confidence=0.9,
                            )

                            # Try to get accessibility tree via CDP
                            cdp_tree = self._get_cdp_accessibility_tree(port, page.get("id", ""))
                            if cdp_tree:
                                return cdp_tree

                            # Fallback: return page-level info
                            if title:
                                return root

                except Exception:
                    continue

        except Exception as e:
            logger.debug("[A11Y] Browser CDP error: %s", e)

        return None

    def _get_cdp_accessibility_tree(self, port: int, page_id: str) -> Optional[A11yNode]:
        """Get full accessibility tree via CDP Accessibility.getFullAXTree."""
        try:
            import json
            import subprocess

            # Use a small Node.js script to get the a11y tree via CDP
            script = f"""
            const WebSocket = require('ws');
            const ws = new WebSocket('ws://127.0.0.1:{port}/devtools/page/{page_id}');
            ws.on('open', () => {{
                ws.send(JSON.stringify({{
                    id: 1,
                    method: 'Accessibility.getFullAXTree',
                    params: {{}}
                }}));
            }});
            ws.on('message', (data) => {{
                const msg = JSON.parse(data);
                if (msg.id === 1) {{
                    console.log(JSON.stringify(msg.result || {{}}));
                    ws.close();
                    process.exit(0);
                }}
            }});
            setTimeout(() => process.exit(1), 3000);
            """

            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0 or not result.stdout.strip():
                return None

            data = json.loads(result.stdout)
            nodes = data.get("nodes", [])
            if not nodes:
                return None

            # Convert CDP AX nodes to A11yNode tree
            node_map: Dict[str, A11yNode] = {}
            root = None

            for ax_node in nodes:
                node_id = ax_node.get("nodeId", "")
                role_str = ax_node.get("role", {}).get("value", "unknown")
                name = ax_node.get("name", {}).get("value", "")
                description = ax_node.get("description", {}).get("value", "")

                role = self._map_cdp_role(role_str)

                a11y_node = A11yNode(
                    role=role,
                    name=name,
                    description=description,
                    backend=A11yBackend.BROWSER_CDP,
                    confidence=0.95,
                )

                # Parse properties for states
                for prop in ax_node.get("properties", []):
                    prop_name = prop.get("name", "")
                    prop_value = prop.get("value", {}).get("value", "")
                    if prop_name == "disabled" and prop_value:
                        a11y_node.states.add(A11yState.DISABLED)
                    elif prop_name == "focused" and prop_value:
                        a11y_node.states.add(A11yState.FOCUSED)
                    elif prop_name == "checked" and prop_value:
                        a11y_node.states.add(A11yState.CHECKED)
                    elif prop_name == "expanded" and prop_value:
                        a11y_node.states.add(A11yState.EXPANDED)
                    elif prop_name == "selected" and prop_value:
                        a11y_node.states.add(A11yState.SELECTED)

                if A11yState.DISABLED not in a11y_node.states:
                    a11y_node.states.add(A11yState.ENABLED)
                a11y_node.states.add(A11yState.VISIBLE)

                node_map[node_id] = a11y_node

                if root is None:
                    root = a11y_node

            # Build parent-child relationships
            for ax_node in nodes:
                node_id = ax_node.get("nodeId", "")
                for child_id in ax_node.get("childIds", []):
                    if node_id in node_map and child_id in node_map:
                        node_map[node_id].add_child(node_map[child_id])

            return root

        except Exception as e:
            logger.debug("[A11Y] CDP accessibility tree error: %s", e)
            return None

    @staticmethod
    def _map_cdp_role(role_str: str) -> A11yRole:
        """Map CDP accessibility role to A11yRole."""
        mapping = {
            "window": A11yRole.WINDOW,
            "dialog": A11yRole.DIALOG,
            "alert": A11yRole.ALERT,
            "button": A11yRole.BUTTON,
            "menu": A11yRole.MENU,
            "menuitem": A11yRole.MENU_ITEM,
            "menubar": A11yRole.MENU_BAR,
            "toolbar": A11yRole.TOOLBAR,
            "tab": A11yRole.TAB,
            "tablist": A11yRole.TAB_LIST,
            "tabpanel": A11yRole.TAB_PANEL,
            "link": A11yRole.LINK,
            "checkbox": A11yRole.CHECKBOX,
            "radio": A11yRole.RADIO,
            "combobox": A11yRole.COMBO_BOX,
            "textbox": A11yRole.TEXT_BOX,
            "statictext": A11yRole.TEXT,
            "text": A11yRole.TEXT,
            "label": A11yRole.LABEL,
            "heading": A11yRole.HEADING,
            "list": A11yRole.LIST,
            "listitem": A11yRole.LIST_ITEM,
            "table": A11yRole.TABLE,
            "cell": A11yRole.TABLE_CELL,
            "row": A11yRole.TABLE_ROW,
            "tree": A11yRole.TREE,
            "treeitem": A11yRole.TREE_ITEM,
            "scrollbar": A11yRole.SCROLL_BAR,
            "slider": A11yRole.SLIDER,
            "progressbar": A11yRole.PROGRESS_BAR,
            "status": A11yRole.STATUS_BAR,
            "tooltip": A11yRole.TOOL_TIP,
            "image": A11yRole.IMAGE,
            "document": A11yRole.DOCUMENT,
            "section": A11yRole.SECTION,
            "group": A11yRole.GROUP,
            "separator": A11yRole.SEPARATOR,
        }
        return mapping.get(role_str.lower(), A11yRole.UNKNOWN)

    # ── Backend: X11 Window Tree ────────────────────────────

    def _get_x11_window_tree(self) -> Optional[A11yNode]:
        """Build a basic accessibility tree from X11 window hierarchy."""
        if not shutil.which("xdotool"):
            return None

        try:
            # Get active window
            wid = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=1
            ).stdout.strip()

            if not wid:
                return None

            # Get window name
            name = subprocess.run(
                ["xdotool", "getwindowname", wid],
                capture_output=True, text=True, timeout=1
            ).stdout.strip()

            # Get window geometry
            geom = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", wid],
                capture_output=True, text=True, timeout=1
            ).stdout

            x, y, w, h = 0, 0, 0, 0
            for line in geom.splitlines():
                if line.startswith("X="):
                    x = int(line.split("=")[1])
                elif line.startswith("Y="):
                    y = int(line.split("=")[1])
                elif line.startswith("WIDTH="):
                    w = int(line.split("=")[1])
                elif line.startswith("HEIGHT="):
                    h = int(line.split("=")[1])

            # Get PID
            pid_str = subprocess.run(
                ["xdotool", "getwindowpid", wid],
                capture_output=True, text=True, timeout=1
            ).stdout.strip()
            pid = int(pid_str) if pid_str.isdigit() else 0

            # Get process name
            process_name = ""
            if pid > 0:
                try:
                    exe = os.readlink(f"/proc/{pid}/exe")
                    process_name = Path(exe).name
                except Exception:
                    pass

            # Get window class
            wm_class = ""
            try:
                class_result = subprocess.run(
                    ["xprop", "-id", wid, "WM_CLASS"],
                    capture_output=True, text=True, timeout=1
                )
                match = re.search(r'WM_CLASS.*"([^"]*)",\s*"([^"]*)"', class_result.stdout)
                if match:
                    wm_class = match.group(1)
            except Exception:
                pass

            # Get window type
            window_type = ""
            try:
                type_result = subprocess.run(
                    ["xprop", "-id", wid, "_NET_WM_WINDOW_TYPE"],
                    capture_output=True, text=True, timeout=1
                )
                match = re.search(r'_NET_WM_WINDOW_TYPE.*_NET_WM_WINDOW_TYPE_(\w+)', type_result.stdout)
                if match:
                    window_type = match.group(1).lower()
            except Exception:
                pass

            # Determine role from window type
            role = A11yRole.WINDOW
            if window_type in ("dialog", "modal"):
                role = A11yRole.DIALOG
            elif window_type == "notification":
                role = A11yRole.NOTIFICATION
            elif window_type == "menu":
                role = A11yRole.MENU
            elif window_type == "tooltip":
                role = A11yRole.TOOL_TIP

            root = A11yNode(
                role=role,
                name=name,
                description=f"{process_name} ({wm_class})" if process_name else wm_class,
                bounds=(x, y, w, h),
                states={A11yState.ENABLED, A11yState.VISIBLE, A11yState.FOCUSED, A11yState.ACTIVE},
                backend=A11yBackend.X11_WINDOW,
                confidence=0.8,
                metadata={
                    "pid": pid,
                    "process": process_name,
                    "wm_class": wm_class,
                    "window_id": wid,
                },
            )

            # Try to get child windows
            try:
                children = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--all", "--pid", str(pid)],
                    capture_output=True, text=True, timeout=1
                ).stdout.strip().splitlines()

                for child_wid in children[:20]:  # Limit to 20 children
                    if child_wid == wid:
                        continue
                    try:
                        child_name = subprocess.run(
                            ["xdotool", "getwindowname", child_wid],
                            capture_output=True, text=True, timeout=0.5
                        ).stdout.strip()
                        if child_name:
                            child_node = A11yNode(
                                role=A11yRole.PANEL,
                                name=child_name,
                                backend=A11yBackend.X11_WINDOW,
                                confidence=0.6,
                                states={A11yState.ENABLED, A11yState.VISIBLE},
                            )
                            root.add_child(child_node)
                    except Exception:
                        pass
            except Exception:
                pass

            return root

        except Exception as e:
            logger.debug("[A11Y] X11 window tree error: %s", e)
            return None

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _get_focused_window_id() -> str:
        """Get a unique identifier for the currently focused window."""
        if shutil.which("xdotool"):
            try:
                wid = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True, text=True, timeout=1
                ).stdout.strip()
                if wid:
                    return wid
            except Exception:
                pass
        return ""

    def invalidate_cache(self) -> None:
        """Force the next get_tree() call to refresh."""
        self._last_tree = None
        self._last_window_id = ""

    def report(self) -> Dict[str, Any]:
        """Return diagnostic information."""
        return {
            "initialized": self._initialized,
            "available_backends": [b.value for b in self._available_backends],
            "last_backend": self._last_backend.value,
            "last_window_id": self._last_window_id[:20] if self._last_window_id else "",
            "cached_tree_nodes": len(self._last_tree.walk()) if self._last_tree else 0,
            "cached_clickable": len(self._last_tree.find_clickable()) if self._last_tree else 0,
        }


# Global singleton
accessibility_tree = AccessibilityTree()