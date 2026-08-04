
"""
UITree — Structured UI element hierarchy data model.

Every screen is represented as a tree of UI elements:

    UIDesktop (root)
      └── UIWindow "PyCharm"
            ├── UIText "README.md"
            ├── UIButton "Run"
            ├── UIButton "Stop"
            ├── UITab "Terminal"
            └── UIText "Fatal Error"

This is NOT raw OCR text — it's a structured, queryable tree that the LLM
can reason about for accurate desktop automation.

Each element carries:
  - type (window, button, text, menu, tab, icon, input, link, dialog, toolbar)
  - label (visible text)
  - bounding box (x, y, w, h)
  - confidence
  - children (sub-elements)

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
    DESKTOP = "desktop"
    WINDOW = "window"
    BUTTON = "button"
    TEXT = "text"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    TAB_GROUP = "tab_group"
    ICON = "icon"
    INPUT = "input"
    LINK = "link"
    DIALOG = "dialog"
    TOOLBAR = "toolbar"
    SCROLLBAR = "scrollbar"
    LIST = "list"
    LIST_ITEM = "list_item"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    LABEL = "label"
    IMAGE = "image"
    DIVIDER = "divider"
    UNKNOWN = "unknown"


@dataclass
class UIElement:
    """
    A single UI element with type, label, position, and optional children.

    The bounding box is in screen coordinates (pixels).
    Text content (when type is TEXT) is stored in `label`.
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

    def add_child(self, child: "UIElement") -> None:
        """Append a child element."""
        self.children.append(child)

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

    def walk(self) -> List["UIElement"]:
        """Flatten the subtree into a depth-first list."""
        flat: List[UIElement] = [self]
        for child in self.children:
            flat.extend(child.walk())
        return flat

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": self.element_type.value,
            "label": self.label,
            "bounding_box": list(self.bounding_box) if self.bounding_box else None,
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UIElement":
        """Deserialize from a dict."""
        bbox = tuple(d["bounding_box"]) if d.get("bounding_box") else None
        return cls(
            element_type=ElementType(d["type"]),
            label=d.get("label", ""),
            bounding_box=bbox,
            confidence=d.get("confidence", 0.0),
            metadata=d.get("metadata", {}),
            children=[UIElement.from_dict(c) for c in d.get("children", [])],
        )

    def to_compact_str(self, indent: int = 0) -> str:
        """Single-line compact representation for LLM context."""
        prefix = "  " * indent
        type_icon = self._icon(self.element_type)
        label_part = f' "{self.label}"' if self.label else ""
        bbox_part = ""
        if self.bounding_box:
            bbox_part = f" @({self.x},{self.y},{self.width}x{self.height})"
        conf_part = f" [{self.confidence:.0%}]" if self.confidence > 0 else ""
        line = f"{prefix}{type_icon} {self.element_type.value}{label_part}{bbox_part}{conf_part}"
        for child in self.children:
            line += "\n" + child.to_compact_str(indent + 1)
        return line

    @staticmethod
    def _icon(element_type: ElementType) -> str:
        """Emoji icon per element type for human readability."""
        return {
            ElementType.DESKTOP: "🖥️",
            ElementType.WINDOW: "🪟",
            ElementType.BUTTON: "🔘",
            ElementType.TEXT: "📝",
            ElementType.MENU: "📋",
            ElementType.MENU_ITEM: "📌",
            ElementType.TAB: "📑",
            ElementType.TAB_GROUP: "🗂️",
            ElementType.ICON: "🖼️",
            ElementType.INPUT: "⌨️",
            ElementType.LINK: "🔗",
            ElementType.DIALOG: "💬",
            ElementType.TOOLBAR: "🧰",
            ElementType.SCROLLBAR: "📊",
            ElementType.LIST: "📃",
            ElementType.LIST_ITEM: "•",
            ElementType.CHECKBOX: "☑️",
            ElementType.RADIO: "🔘",
            ElementType.LABEL: "🏷️",
            ElementType.IMAGE: "🖼️",
            ElementType.DIVIDER: "➖",
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


@dataclass
class UIWindow:
    """
    A single application window in the desktop tree.
    """
    element: UIElement

    def __init__(self, title: str = "", bounding_box: Optional[Tuple[int, int, int, int]] = None):
        self.element = UIElement(
            element_type=ElementType.WINDOW,
            label=title,
            bounding_box=bounding_box,
        )

    def add_button(self, label: str, bbox: Optional[Tuple[int, int, int, int]] = None,
                   confidence: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> UIElement:
        btn = UIElement(
            element_type=ElementType.BUTTON,
            label=label,
            bounding_box=bbox,
            confidence=confidence,
            metadata=metadata or {},
        )
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