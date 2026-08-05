"""
UITree — Structured UI element hierarchy data model.

Every screen is represented as a tree of UI elements:

    UIDesktop (root)
      └── UIWindow "PyCharm"
            ├── UIPanel "Toolbar"
            │     ├── UIButton "Run"
            │     └── UIButton "Stop"
            ├── UISidebar "Project"
            ├── UIEditorArea
            │     └── UITab "main.py"
            ├── UIBottomPanel "Terminal"
            │     └── UIText "$ python main.py"
            └── UIStatusBar

This is NOT raw OCR text — it's a structured, queryable tree that the LLM
can reason about for accurate desktop automation.

Each element carries:
  - type (desktop, window, button, text, menu, tab, icon, input, link, dialog,
    toolbar, sidebar, panel, status_bar, checkbox, radio, dropdown, table, tree,
    notification, image, divider, scrollbar, unknown)
  - label (visible text)
  - bounding box (x, y, w, h)
  - confidence
  - role ("button", "menu", "tab", "textbox", "checkbox", "link", "heading", etc.)
  - enabled (bool) — is the element interactable?
  - visible (bool) — is the element visible on screen?
  - parent (reference to parent element)
  - children (sub-elements)
  - metadata (any additional info)

Serialization:
  - to_dict() for JSON embedding in LLM prompts
  - to_compact_str() for terse text representation
  - from_dict() for deserialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class ElementType(str, Enum):
    """Types of UI elements that can appear in the tree."""
    # Root
    DESKTOP = "desktop"

    # Windows / dialogs
    WINDOW = "window"
    DIALOG = "dialog"
    POPUP = "popup"
    NOTIFICATION = "notification"

    # Layout containers
    PANEL = "panel"                 # generic panel/container
    TOOLBAR = "toolbar"
    SIDEBAR = "sidebar"
    LEFT_PANEL = "left_panel"
    RIGHT_PANEL = "right_panel"
    BOTTOM_PANEL = "bottom_panel"
    EDITOR = "editor"
    CONTENT = "content"
    MENU_BAR = "menu_bar"
    TAB_BAR = "tab_bar"
    STATUS_BAR = "status_bar"
    TITLE_BAR = "title_bar"
    NAVIGATION = "navigation"
    SCROLLBAR = "scrollbar"
    MINIMAP = "minimap"
    TASKBAR = "taskbar"
    DOCK = "dock"
    SYSTEM_TRAY = "system_tray"

    # Interactive elements
    BUTTON = "button"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    TAB_GROUP = "tab_group"
    LINK = "link"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DROPDOWN = "dropdown"           # select/combo box
    INPUT = "input"
    TEXTBOX = "textbox"
    SLIDER = "slider"
    TOGGLE = "toggle"
    SWITCH = "switch"
    ICON = "icon"

    # Display elements
    TEXT = "text"
    LABEL = "label"
    IMAGE = "image"
    TABLE = "table"
    TREE = "tree"                   # tree view widget
    LIST = "list"
    LIST_ITEM = "list_item"
    DIVIDER = "divider"
    PROGRESS_BAR = "progress_bar"

    # Unknown
    UNKNOWN = "unknown"

    @property
    def is_container(self) -> bool:
        """True if this element type is a container for children."""
        return self in (
            ElementType.DESKTOP, ElementType.WINDOW, ElementType.DIALOG,
            ElementType.POPUP, ElementType.NOTIFICATION, ElementType.PANEL,
            ElementType.TOOLBAR, ElementType.SIDEBAR, ElementType.LEFT_PANEL,
            ElementType.RIGHT_PANEL, ElementType.BOTTOM_PANEL, ElementType.EDITOR,
            ElementType.CONTENT, ElementType.MENU_BAR, ElementType.TAB_BAR,
            ElementType.STATUS_BAR, ElementType.TITLE_BAR, ElementType.NAVIGATION,
            ElementType.TAB_GROUP, ElementType.LIST, ElementType.TABLE,
            ElementType.TREE, ElementType.MENU,
        )

    @property
    def is_clickable(self) -> bool:
        """True if this element type is typically clickable."""
        return self in (
            ElementType.BUTTON, ElementType.MENU_ITEM, ElementType.TAB,
            ElementType.LINK, ElementType.CHECKBOX, ElementType.RADIO,
            ElementType.ICON, ElementType.TOGGLE, ElementType.SWITCH,
            ElementType.DROPDOWN, ElementType.LIST_ITEM,
        )

    @property
    def is_text_input(self) -> bool:
        """True if this element type accepts text input."""
        return self in (
            ElementType.INPUT, ElementType.TEXTBOX,
        )


@dataclass
class UIElement:
    """
    A single UI element with type, label, position, and optional children.

    The bounding box is in screen coordinates (pixels).
    Text content (when type is TEXT) is stored in `label`.

    Metadata fields:
      - role: accessibility role ("button", "menu", "tab", "textbox", etc.)
      - enabled: whether the element is interactable
      - visible: whether the element is visible on screen
      - active: for tabs — which one is active
      - checked: for checkboxes — whether checked
      - selected: for list items — whether selected
      - state: "normal", "hovered", "pressed", "disabled", "focused"
    """
    element_type: ElementType
    label: str = ""
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    confidence: float = 0.0
    children: List["UIElement"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bounding_box[0] if self.bounding_box else 0

    @property
    def y(self) -> int:
        return self.bounding_box[1] if self.bounding_box else 0

    @property
    def width(self) -> int:
        return self.bounding_box[2] if self.bounding_box else 0

    @property
    def height(self) -> int:
        return self.bounding_box[3] if self.bounding_box else 0

    @property
    def center(self) -> Tuple[int, int]:
        """Return the center point of this element."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def enabled(self) -> bool:
        """Whether the element is enabled/interactable."""
        return self.metadata.get("enabled", True)

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.metadata["enabled"] = value

    @property
    def visible(self) -> bool:
        """Whether the element is visible on screen."""
        return self.metadata.get("visible", True)

    @visible.setter
    def visible(self, value: bool) -> None:
        self.metadata["visible"] = value

    @property
    def role(self) -> str:
        """Accessibility role."""
        return self.metadata.get("role", self.element_type.value)

    @role.setter
    def role(self, value: str) -> None:
        self.metadata["role"] = value

    def add_child(self, child: "UIElement") -> None:
        """Append a child element."""
        self.children.append(child)
        child.metadata["parent"] = self

    def find_by_type(self, element_type: ElementType) -> List["UIElement"]:
        """Recursively find all elements of a given type."""
        results: List[UIElement] = []
        if self.element_type == element_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_type(element_type))
        return results

    def find_by_label(self, label: str, case_sensitive: bool = False) -> List["UIElement"]:
        """Recursively find all elements whose label contains the given text."""
        results: List[UIElement] = []
        compare = self.label if case_sensitive else self.label.lower()
        target = label if case_sensitive else label.lower()
        if target in compare:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_label(target, case_sensitive=True))
        return results

    def find_by_role(self, role: str) -> List["UIElement"]:
        """Recursively find all elements with a given accessibility role."""
        results: List[UIElement] = []
        if self.role == role:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_role(role))
        return results

    def find_clickable(self) -> List["UIElement"]:
        """Return all clickable elements (buttons, links, tabs, etc.)."""
        return [e for e in self.walk() if e.element_type.is_clickable and e.enabled]

    def find_by_region(self, x: int, y: int, w: int, h: int) -> List["UIElement"]:
        """Find all elements that fall within a region (x, y, w, h)."""
        results: List[UIElement] = []
        for e in self.walk():
            if e.x >= x and e.y >= y and (e.x + e.width) <= (x + w) and (e.y + e.height) <= (y + h):
                results.append(e)
        return results

    def walk(self) -> List["UIElement"]:
        """Flatten the subtree into a depth-first list."""
        flat: List[UIElement] = [self]
        for child in self.children:
            flat.extend(child.walk())
        return flat

    def parent(self) -> Optional["UIElement"]:
        """Return the parent element, or None."""
        return self.metadata.get("parent")

    def ancestors(self) -> List["UIElement"]:
        """Return the chain from root to this element."""
        chain: List[UIElement] = [self]
        parent = self.parent()
        while parent is not None:
            chain.append(parent)
            parent = parent.parent()
        chain.reverse()
        return chain

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": self.element_type.value,
            "label": self.label,
            "bounding_box": list(self.bounding_box) if self.bounding_box else None,
            "confidence": round(self.confidence, 3),
            "role": self.role,
            "enabled": self.enabled,
            "visible": self.visible,
            "metadata": {k: v for k, v in self.metadata.items() if k != "parent"},
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UIElement":
        """Deserialize from a dict."""
        bbox = tuple(d["bounding_box"]) if d.get("bounding_box") else None
        meta = d.get("metadata", {})
        # Restore enabled/visible/role from dedicated fields
        if "enabled" in d:
            meta["enabled"] = d["enabled"]
        if "visible" in d:
            meta["visible"] = d["visible"]
        if "role" in d:
            meta["role"] = d["role"]
        element = cls(
            element_type=ElementType(d["type"]),
            label=d.get("label", ""),
            bounding_box=bbox,
            confidence=d.get("confidence", 0.0),
            metadata=meta,
            children=[UIElement.from_dict(c) for c in d.get("children", [])],
        )
        # Re-set parent references
        for child in element.children:
            child.metadata["parent"] = element
        return element

    def to_compact_str(self, indent: int = 0) -> str:
        """Single-line compact representation for LLM context."""
        prefix = "  " * indent
        type_icon = self._icon(self.element_type)
        label_part = f' "{self.label}"' if self.label else ""
        bbox_part = ""
        if self.bounding_box:
            bbox_part = f" @({self.x},{self.y},{self.width}x{self.height})"
        status = ""
        if not self.enabled:
            status = " [disabled]"
        elif not self.visible:
            status = " [hidden]"
        conf_part = f" [{self.confidence:.0%}]" if self.confidence > 0 else ""
        line = f"{prefix}{type_icon} {self.element_type.value}{label_part}{bbox_part}{status}{conf_part}"
        for child in self.children:
            line += "\n" + child.to_compact_str(indent + 1)
        return line

    @staticmethod
    def _icon(element_type: ElementType) -> str:
        """Emoji icon per element type for human readability."""
        return {
            ElementType.DESKTOP: "🖥️",
            ElementType.WINDOW: "🪟",
            ElementType.DIALOG: "💬",
            ElementType.POPUP: "🔔",
            ElementType.NOTIFICATION: "📢",
            ElementType.PANEL: "📦",
            ElementType.TOOLBAR: "🧰",
            ElementType.SIDEBAR: "📂",
            ElementType.LEFT_PANEL: "⬅️",
            ElementType.RIGHT_PANEL: "➡️",
            ElementType.BOTTOM_PANEL: "⬇️",
            ElementType.EDITOR: "📝",
            ElementType.CONTENT: "📄",
            ElementType.MENU_BAR: "📋",
            ElementType.TAB_BAR: "🗂️",
            ElementType.STATUS_BAR: "📊",
            ElementType.TITLE_BAR: "🏷️",
            ElementType.NAVIGATION: "🧭",
            ElementType.SCROLLBAR: "📏",
            ElementType.MINIMAP: "🗺️",
            ElementType.TASKBAR: "📌",
            ElementType.DOCK: "🚢",
            ElementType.SYSTEM_TRAY: "🔧",
            ElementType.BUTTON: "🔘",
            ElementType.MENU: "📋",
            ElementType.MENU_ITEM: "📌",
            ElementType.TAB: "📑",
            ElementType.TAB_GROUP: "🗂️",
            ElementType.LINK: "🔗",
            ElementType.CHECKBOX: "☑️",
            ElementType.RADIO: "🔘",
            ElementType.DROPDOWN: "🔽",
            ElementType.INPUT: "⌨️",
            ElementType.TEXTBOX: "📝",
            ElementType.SLIDER: "🎚️",
            ElementType.TOGGLE: "🔀",
            ElementType.SWITCH: "🔛",
            ElementType.ICON: "🖼️",
            ElementType.TEXT: "💬",
            ElementType.LABEL: "🏷️",
            ElementType.IMAGE: "🖼️",
            ElementType.TABLE: "📊",
            ElementType.TREE: "🌳",
            ElementType.LIST: "📃",
            ElementType.LIST_ITEM: "•",
            ElementType.DIVIDER: "➖",
            ElementType.PROGRESS_BAR: "⏳",
            ElementType.UNKNOWN: "❓",
        }.get(element_type, "❓")

    def __repr__(self) -> str:
        return f"UIElement({self.element_type.value}, label={self.label!r}, children={len(self.children)})"


# ── Convenience constructors for the tree ──────────────────────────


@dataclass
class UIDesktop:
    """
    Root of the UI tree representing the entire desktop.
    Contains one or more UIWindow children.
    """
    root: UIElement = field(default_factory=lambda: UIElement(
        element_type=ElementType.DESKTOP,
        label="Desktop",
    ))

    @property
    def windows(self) -> List[UIElement]:
        return self.root.children

    def add_window(self, window: "UIWindow") -> None:
        self.root.add_child(window.element)

    def to_dict(self) -> Dict[str, Any]:
        return self.root.to_dict()

    def to_compact_str(self) -> str:
        lines = ["Desktop"]
        for w in self.root.children:
            lines.append(w.to_compact_str(indent=1))
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UIDesktop":
        desktop = cls()
        desktop.root = UIElement.from_dict(d)
        return desktop

    def find_by_type(self, element_type: ElementType) -> List[UIElement]:
        return self.root.find_by_type(element_type)

    def find_by_label(self, label: str) -> List[UIElement]:
        return self.root.find_by_label(label)

    def find_clickable(self) -> List[UIElement]:
        return self.root.find_clickable()

    def walk(self) -> List[UIElement]:
        return self.root.walk()


@dataclass
class UIWindow:
    """
    A single application window in the desktop tree.
    """
    element: UIElement

    def __init__(self, title: str = "", bounding_box: Optional[Tuple[int, int, int, int]] = None,
                 app_type: str = "", app_name: str = ""):
        self.element = UIElement(
            element_type=ElementType.WINDOW,
            label=title,
            bounding_box=bounding_box,
            metadata={"app_type": app_type, "app_name": app_name},
        )

    def add_child(self, child: UIElement) -> None:
        self.element.add_child(child)

    def add_button(self, label: str, bbox: Optional[Tuple[int, int, int, int]] = None,
                   confidence: float = 0.0, enabled: bool = True,
                   metadata: Optional[Dict[str, Any]] = None) -> UIElement:
        btn = UIElement(
            element_type=ElementType.BUTTON,
            label=label,
            bounding_box=bbox,
            confidence=confidence,
            metadata=metadata or {},
        )
        btn.enabled = enabled
        btn.role = "button"
        self.element.add_child(btn)
        return btn

    def add_text(self, label: str, bbox: Optional[Tuple[int, int, int, int]] = None,
                 confidence: float = 0.0) -> UIElement:
        txt = UIElement(
            element_type=ElementType.TEXT,
            label=label,
            bounding_box=bbox,
            confidence=confidence,
        )
        self.element.add_child(txt)
        return txt

    def add_panel(self, region_type: ElementType, label: str = "",
                  bbox: Optional[Tuple[int, int, int, int]] = None) -> UIElement:
        panel = UIElement(
            element_type=region_type,
            label=label,
            bounding_box=bbox,
        )
        self.element.add_child(panel)
        return panel

    def add_menu(self, label: str, bbox: Optional[Tuple[int, int, int, int]] = None,
                 items: Optional[List[str]] = None) -> UIElement:
        menu = UIElement(
            element_type=ElementType.MENU,
            label=label,
            bounding_box=bbox,
        )
        if items:
            for item_label in items:
                menu.add_child(UIElement(
                    element_type=ElementType.MENU_ITEM,
                    label=item_label,
                ))
        self.element.add_child(menu)
        return menu

    def add_tab(self, label: str, bbox: Optional[Tuple[int, int, int, int]] = None,
                active: bool = False) -> UIElement:
        tab = UIElement(
            element_type=ElementType.TAB,
            label=label,
            bounding_box=bbox,
            metadata={"active": active},
        )
        self.element.add_child(tab)
        return tab

    def add_icon(self, label: str, bbox: Optional[Tuple[int, int, int, int]] = None) -> UIElement:
        icon = UIElement(
            element_type=ElementType.ICON,
            label=label,
            bounding_box=bbox,
        )
        self.element.add_child(icon)
        return icon

    def add_input(self, label: str = "", bbox: Optional[Tuple[int, int, int, int]] = None) -> UIElement:
        inp = UIElement(
            element_type=ElementType.INPUT,
            label=label,
            bounding_box=bbox,
        )
        self.element.add_child(inp)
        return inp

    @property
    def title(self) -> str:
        return self.element.label

    @property
    def buttons(self) -> List[UIElement]:
        return self.element.find_by_type(ElementType.BUTTON)

    @property
    def text_elements(self) -> List[UIElement]:
        return self.element.find_by_type(ElementType.TEXT)

    @property
    def tabs(self) -> List[UIElement]:
        return self.element.find_by_type(ElementType.TAB)

    def to_compact_str(self, indent: int = 0) -> str:
        return self.element.to_compact_str(indent)