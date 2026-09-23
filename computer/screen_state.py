"""Phase 22: normalized ComputerState (additive; never raw screenshots only)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ComputerState:
    """Normalized desktop truth: structured fields + bounded text excerpts."""
    active_application: str = ""
    focused_window: str = ""
    window_title: str = ""
    window_class: str = ""
    active_url: str = ""
    page_title: str = ""
    visible_text: str = ""                       # bounded excerpt (not full dump)
    visible_elements: List[Dict[str, Any]] = field(default_factory=list)
    screen_hash: str = ""
    last_action: str = ""
    last_observation: str = ""
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        parts = []
        if self.active_application:
            parts.append(f"app={self.active_application}")
        if self.window_title:
            parts.append(f"window='{self.window_title[:80]}'")
        if self.active_url:
            parts.append(f"url={self.active_url[:100]}")
        if self.page_title:
            parts.append(f"page='{self.page_title[:80]}'")
        parts.append(f"elements={len(self.visible_elements)}")
        if self.screen_hash:
            parts.append(f"hash={self.screen_hash[:12]}")
        return " ".join(parts) or "empty state"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_application": self.active_application,
            "focused_window": self.focused_window,
            "window_title": self.window_title,
            "window_class": self.window_class,
            "active_url": self.active_url,
            "page_title": self.page_title,
            "visible_text": self.visible_text[:2000],
            "visible_elements": self.visible_elements[:50],
            "screen_hash": self.screen_hash,
            "last_action": self.last_action,
            "last_observation": self.last_observation,
            "timestamp": self.timestamp,
        }
