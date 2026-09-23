"""
Phase 22: accessibility perception tier (tier 2 of the hierarchy).

Thin adapter over services.accessibility.accessibility_tree (AT-SPI / GTK /
Qt backends — never reimplemented here). Converts A11yNode objects into flat,
JSON-able A11yElement records that element_finder and the ComputerController
reason about and log.

Attribute access on wrapped nodes is defensive (getattr): the underlying
node shape may evolve without breaking this tier.

Logging: [PERCEPTION]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from computer.perception import PerceptionMethod

logger = logging.getLogger(__name__)


@dataclass
class A11yElement:
    """Flat, serializable view of one accessibility node."""

    name: str = ""
    role: str = ""
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.9
    enabled: bool = True
    clickable: bool = False
    text_input: bool = False
    method: str = PerceptionMethod.ACCESSIBILITY.value

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "role": self.role, "bounds": list(self.bounds),
            "center": list(self.center), "confidence": self.confidence,
            "enabled": self.enabled, "clickable": self.clickable,
            "text_input": self.text_input, "method": self.method,
        }


def _tree_api():
    """Duck-load the existing accessibility_tree singleton (never rebinds it)."""
    try:
        from services.accessibility import accessibility_tree
        return accessibility_tree
    except Exception as e:
        logger.debug("[PERCEPTION] accessibility tier unavailable: %s", e)
        return None


def available() -> bool:
    tree = _tree_api()
    if tree is None:
        return False
    try:
        return tree.get_tree() is not None
    except Exception:
        return False


def _convert(node: Any, clickable: bool) -> Optional[A11yElement]:
    """Best-effort conversion of an A11yNode-like object."""
    if node is None:
        return None
    try:
        name = str(getattr(node, "name", "") or getattr(node, "label", "") or "")
        role = str(getattr(node, "role", "") or
                   getattr(getattr(node, "node_type", None), "name", "") or "")
        try:
            bounds = (int(node.x), int(node.y),
                      int(node.width), int(node.height))
        except Exception:
            bounds = tuple(getattr(node, "bounds", (0, 0, 0, 0)) or (0, 0, 0, 0))
        return A11yElement(
            name=name, role=role,
            bounds=bounds,  # type: ignore[arg-type]
            confidence=0.9,
            enabled=bool(getattr(node, "enabled", True)),
            clickable=bool(getattr(node, "is_clickable", clickable)),
            text_input=bool(getattr(node, "is_text_input", False)),
        )
    except Exception as e:
        logger.debug("[PERCEPTION] a11y node conversion failed: %s", e)
        return None


def _walk(root: Any, max_nodes: int = 400):
    """Yield nodes depth-first, defensively, with a hard node budget."""
    seen = 0
    stack: List[Any] = [root]
    while stack and seen < max_nodes:
        node = stack.pop(0)
        seen += 1
        yield node
        children = getattr(node, "children", None) or []
        for child in children:
            stack.append(child)


def clickable(limit: int = 40) -> List[A11yElement]:
    """All interactable nodes of the focused window's accessibility tree."""
    tree = _tree_api()
    if tree is None:
        return []
    try:
        root = tree.get_tree()
    except Exception as e:
        logger.debug("[PERCEPTION] a11y get_tree failed: %s", e)
        return []
    if root is None:
        return []

    out: List[A11yElement] = []
    try:
        raw = root.find_clickable() or []
    except Exception:
        raw = []
    for node in raw[:limit]:
        el = _convert(node, clickable=True)
        if el is not None:
            out.append(el)
    if not out:
        # find_clickable may be unavailable — filter the full walk instead.
        for node in _walk(root):
            el = _convert(node, clickable=False)
            if el is not None and (el.clickable or el.text_input):
                out.append(el)
            if len(out) >= limit:
                break
    return out


def find(query: str, limit: int = 10) -> List[A11yElement]:
    """Nodes whose name contains `query` (case-insensitive), best first."""
    tree = _tree_api()
    if tree is None:
        return []
    try:
        root = tree.get_tree()
    except Exception as e:
        logger.debug("[PERCEPTION] a11y get_tree failed: %s", e)
        return []
    if root is None:
        return []

    q = str(query or "").strip().lower()
    if not q:
        return []
    hits: List[A11yElement] = []
    for node in _walk(root):
        el = _convert(node, clickable=False)
        if el is None or not el.name:
            continue
        if q in el.name.lower():
            hits.append(el)
        if len(hits) >= limit:
            break
    # Prefer clickable / text-input hits for acting.
    hits.sort(key=lambda e: (not (e.clickable or e.text_input), not e.enabled))
    return hits
