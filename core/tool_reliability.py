"""
ToolReliability — Confidence scores for every tool/action Leo can perform.

Maintains per-tool confidence scores based on success/failure history.
The planner automatically prefers higher-confidence tools.

Architecture:
    Each tool has:
      - base_confidence: initial confidence (0.0-1.0)
      - success_count: number of successful executions
      - failure_count: number of failed executions
      - last_success: timestamp of last success
      - last_failure: timestamp of last failure
      - avg_latency_ms: average execution time
      - failure_reasons: dict of {reason: count}

    Confidence decays over time if a tool hasn't been used recently.
    Repeated failures reduce confidence exponentially.

Usage:
    from core.tool_reliability import tool_reliability

    # Before executing:
    if tool_reliability.confidence("desktop_open") < 0.5:
        # Try alternative tool

    # After executing:
    tool_reliability.record_success("desktop_open", latency_ms=120)
    tool_reliability.record_failure("desktop_open", "app not found")

    # Get best tool for a category:
    best = tool_reliability.best_for("browser_launch")
    # → ("google-chrome", 0.95)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RELIABILITY_PATH = Path(__file__).resolve().parent.parent / "data" / "tool_reliability.json"

# ── Default base confidences ──────────────────────────────────────

_DEFAULT_CONFIDENCES: Dict[str, float] = {
    # App launching
    "desktop_open:code": 0.99,
    "desktop_open:firefox": 0.71,
    "desktop_open:google-chrome": 0.95,
    "desktop_open:gnome-terminal": 0.99,
    "desktop_open:spotify": 0.80,
    "desktop_open:slack": 0.75,
    "desktop_open:discord": 0.75,
    "desktop_open:telegram-desktop": 0.80,
    "desktop_open:nautilus": 0.99,
    "desktop_open:gnome-calculator": 0.99,
    "desktop_open:gnome-control-center": 0.99,

    # Volume
    "volume_up": 0.99,
    "volume_down": 0.99,
    "volume_mute": 0.99,
    "volume_set": 0.95,

    # Brightness
    "brightness_up": 0.90,
    "brightness_down": 0.90,
    "brightness_set": 0.85,

    # Music
    "music_pause": 0.98,
    "music_resume": 0.98,
    "music_next": 0.98,
    "music_previous": 0.98,
    "music_shuffle": 0.95,
    "music_repeat": 0.95,
    "music_status": 0.95,
    "play_media": 0.85,

    # Screen
    "read_screen": 0.90,
    "lock_screen": 0.99,
    "shutdown": 0.95,
    "restart": 0.95,

    # Scroll
    "scroll": 0.95,

    # Browser
    "browser_navigate": 0.90,
    "browser_click": 0.75,
    "browser_click_text": 0.70,
    "browser_type": 0.85,
    "browser_screenshot": 0.95,
    "browser_get_url": 0.99,
    "browser_get_text": 0.80,
    "browser_list_tabs": 0.99,
    "browser_new_tab": 0.95,
    "browser_close_tab": 0.99,
    "browser_switch_tab": 0.90,

    # Mouse
    "mouse_move": 0.99,
    "mouse_click": 0.95,
    "mouse_double_click": 0.90,

    # Keyboard
    "keyboard_type": 0.95,
    "keyboard_press": 0.99,
    "keyboard_hotkey": 0.95,

    # Clipboard
    "clipboard_copy": 0.99,
    "clipboard_paste": 0.95,

    # Vision
    "vision_ocr": 0.85,
    "vision_ui_detect": 0.80,
    "vision_layout": 0.90,
    "vision_screen_capture": 0.99,

    # Search
    "search_web": 0.85,
    "search_local": 0.70,
}

# ── Tool categories for alternative selection ─────────────────────

_TOOL_CATEGORIES: Dict[str, List[str]] = {
    "browser_launch": ["desktop_open:firefox", "desktop_open:google-chrome",
                        "desktop_open:chromium", "desktop_open:brave"],
    "terminal_launch": ["desktop_open:gnome-terminal", "desktop_open:konsole",
                        "desktop_open:xfce4-terminal", "desktop_open:alacritty"],
    "editor_launch": ["desktop_open:code", "desktop_open:gedit",
                      "desktop_open:nano", "desktop_open:vim"],
    "music_control": ["music_pause", "music_resume", "music_next",
                      "music_previous", "music_shuffle", "music_repeat"],
    "volume_control": ["volume_up", "volume_down", "volume_mute", "volume_set"],
    "brightness_control": ["brightness_up", "brightness_down", "brightness_set"],
    "browser_action": ["browser_navigate", "browser_click", "browser_click_text",
                       "browser_type", "browser_screenshot"],
    "mouse_action": ["mouse_move", "mouse_click", "mouse_double_click", "scroll"],
    "keyboard_action": ["keyboard_type", "keyboard_press", "keyboard_hotkey"],
    "vision_action": ["vision_ocr", "vision_ui_detect", "vision_layout",
                      "vision_screen_capture"],
}


@dataclass
class ToolStats:
    """Statistics for a single tool."""
    name: str
    base_confidence: float = 0.80
    success_count: int = 0
    failure_count: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    consecutive_failures: int = 0

    @property
    def total_executions(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        total = self.total_executions
        if total == 0:
            return self.base_confidence
        return self.success_count / total

    @property
    def confidence(self) -> float:
        """
        Current confidence score (0.0-1.0).

        Factors:
          - Base confidence (initial estimate)
          - Success rate (from actual executions)
          - Recency decay (unused tools lose confidence over time)
          - Consecutive failure penalty
        """
        # Start with base confidence
        score = self.base_confidence

        # Blend with actual success rate if we have data
        total = self.total_executions
        if total >= 3:
            # Weight: 30% base, 70% actual
            score = 0.3 * self.base_confidence + 0.7 * self.success_rate

        # Recency decay: unused for >1 hour → slight decay
        now = time.time()
        last_use = max(self.last_success, self.last_failure)
        if last_use > 0:
            hours_since_use = (now - last_use) / 3600
            if hours_since_use > 1:
                decay = min(0.2, hours_since_use * 0.01)
                score *= (1.0 - decay)

        # Consecutive failure penalty
        if self.consecutive_failures >= 3:
            penalty = min(0.5, self.consecutive_failures * 0.1)
            score *= (1.0 - penalty)

        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_confidence": self.base_confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": f"{self.success_rate:.1%}",
            "confidence": f"{self.confidence:.1%}",
            "avg_latency_ms": f"{self.avg_latency_ms:.1f}",
            "consecutive_failures": self.consecutive_failures,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "failure_reasons": dict(self.failure_reasons),
        }


class ToolReliability:
    """
    Maintains confidence scores for every tool Leo can use.

    The planner queries this before selecting tools. Higher-confidence
    tools are preferred. Failed tools are deprioritized.
    """

    def __init__(self):
        self._tools: Dict[str, ToolStats] = {}
        self._loaded = False

    # ── Lifecycle ───────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load persisted reliability data."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self) -> None:
        """Load tool reliability from disk."""
        try:
            if RELIABILITY_PATH.exists():
                with open(RELIABILITY_PATH, "r") as f:
                    data = json.load(f)
                for name, stats_data in data.get("tools", {}).items():
                    ts = ToolStats(
                        name=name,
                        base_confidence=stats_data.get("base_confidence",
                                                       _DEFAULT_CONFIDENCES.get(name, 0.80)),
                        success_count=stats_data.get("success_count", 0),
                        failure_count=stats_data.get("failure_count", 0),
                        last_success=stats_data.get("last_success", 0.0),
                        last_failure=stats_data.get("last_failure", 0.0),
                        avg_latency_ms=stats_data.get("avg_latency_ms", 0.0),
                        total_latency_ms=stats_data.get("total_latency_ms", 0.0),
                        failure_reasons=stats_data.get("failure_reasons", {}),
                        consecutive_failures=stats_data.get("consecutive_failures", 0),
                    )
                    self._tools[name] = ts
                logger.debug("[RELIABILITY] Loaded %d tool stats", len(self._tools))
        except Exception as e:
            logger.debug("[RELIABILITY] Load failed: %s", e)

    def _save(self) -> None:
        """Persist tool reliability to disk."""
        try:
            RELIABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "tools": {
                    name: ts.to_dict()
                    for name, ts in self._tools.items()
                },
                "updated_at": time.time(),
            }
            with open(RELIABILITY_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug("[RELIABILITY] Save failed: %s", e)

    # ── Tool lookup ─────────────────────────────────────────

    def _get_or_create(self, tool_name: str) -> ToolStats:
        """Get tool stats, creating with defaults if new."""
        self._ensure_loaded()
        if tool_name not in self._tools:
            base = _DEFAULT_CONFIDENCES.get(tool_name, 0.80)
            self._tools[tool_name] = ToolStats(
                name=tool_name,
                base_confidence=base,
            )
        return self._tools[tool_name]

    # ── Recording ───────────────────────────────────────────

    def record_success(self, tool_name: str, latency_ms: float = 0.0) -> None:
        """Record a successful tool execution."""
        ts = self._get_or_create(tool_name)
        ts.success_count += 1
        ts.last_success = time.time()
        ts.consecutive_failures = 0

        if latency_ms > 0:
            ts.total_latency_ms += latency_ms
            ts.avg_latency_ms = ts.total_latency_ms / ts.total_executions

        logger.debug("[RELIABILITY] %s success (conf=%.2f, latency=%.0fms)",
                     tool_name, ts.confidence, latency_ms)
        self._save()

    def record_failure(self, tool_name: str, reason: str = "unknown") -> None:
        """Record a failed tool execution."""
        ts = self._get_or_create(tool_name)
        ts.failure_count += 1
        ts.last_failure = time.time()
        ts.consecutive_failures += 1

        reason_key = reason[:50]
        ts.failure_reasons[reason_key] = ts.failure_reasons.get(reason_key, 0) + 1

        logger.info("[RELIABILITY] %s FAILED (reason=%s, conf=%.2f, consecutive=%d)",
                    tool_name, reason, ts.confidence, ts.consecutive_failures)
        self._save()

    # ── Queries ─────────────────────────────────────────────

    def confidence(self, tool_name: str) -> float:
        """Get current confidence for a tool (0.0-1.0)."""
        ts = self._get_or_create(tool_name)
        return ts.confidence

    def is_reliable(self, tool_name: str, threshold: float = 0.70) -> bool:
        """Check if a tool is reliable enough to use."""
        return self.confidence(tool_name) >= threshold

    def best_for(self, category: str) -> Optional[Tuple[str, float]]:
        """
        Get the best tool for a category.

        Returns (tool_name, confidence) for the highest-confidence tool
        in the category, or None if no tools are available.
        """
        tools = _TOOL_CATEGORIES.get(category, [])
        if not tools:
            return None

        best_tool = None
        best_conf = -1.0

        for tool_name in tools:
            conf = self.confidence(tool_name)
            if conf > best_conf:
                best_conf = conf
                best_tool = tool_name

        if best_tool is None:
            return None
        return (best_tool, best_conf)

    def alternatives(self, tool_name: str, min_confidence: float = 0.5) -> List[Tuple[str, float]]:
        """
        Get alternative tools for a given tool.

        Returns list of (tool_name, confidence) sorted by confidence descending.
        """
        # Find which category this tool belongs to
        category = None
        for cat, tools in _TOOL_CATEGORIES.items():
            if tool_name in tools:
                category = cat
                break

        if category is None:
            return []

        alternatives = []
        for alt in _TOOL_CATEGORIES[category]:
            if alt == tool_name:
                continue
            conf = self.confidence(alt)
            if conf >= min_confidence:
                alternatives.append((alt, conf))

        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives

    def should_avoid(self, tool_name: str) -> bool:
        """Check if a tool should be avoided (confidence < 0.3 or 3+ consecutive failures)."""
        ts = self._get_or_create(tool_name)
        return ts.confidence < 0.3 or ts.consecutive_failures >= 3

    def get_stats(self, tool_name: str) -> Optional[ToolStats]:
        """Get full stats for a tool."""
        self._ensure_loaded()
        return self._tools.get(tool_name)

    # ── Bulk queries ────────────────────────────────────────

    def all_confidences(self) -> Dict[str, float]:
        """Get confidence for all known tools."""
        self._ensure_loaded()
        return {name: ts.confidence for name, ts in self._tools.items()}

    def unreliable_tools(self, threshold: float = 0.5) -> List[str]:
        """List all tools below the confidence threshold."""
        return [
            name for name, ts in self._tools.items()
            if ts.confidence < threshold
        ]

    def top_tools(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the N most reliable tools."""
        confidences = self.all_confidences()
        sorted_tools = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        return sorted_tools[:n]

    # ── Report ──────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Generate a comprehensive reliability report."""
        self._ensure_loaded()

        all_tools = sorted(self._tools.values(),
                          key=lambda t: t.confidence, reverse=True)

        return {
            "total_tools": len(self._tools),
            "reliable_tools": sum(1 for t in self._tools.values() if t.confidence >= 0.7),
            "unreliable_tools": sum(1 for t in self._tools.values() if t.confidence < 0.5),
            "top_10": [
                {"name": t.name, "confidence": f"{t.confidence:.1%}",
                 "success_rate": f"{t.success_rate:.1%}",
                 "executions": t.total_executions}
                for t in all_tools[:10]
            ],
            "bottom_5": [
                {"name": t.name, "confidence": f"{t.confidence:.1%}",
                 "failures": t.failure_count,
                 "reasons": dict(list(t.failure_reasons.items())[:3])}
                for t in all_tools[-5:]
            ],
            "categories": {
                cat: {
                    "best": (best[0] if (best := self.best_for(cat)) else "none"),
                    "confidence": f"{best[1]:.1%}" if (best := self.best_for(cat)) else "0%",
                }
                for cat in _TOOL_CATEGORIES
            },
        }

    def print_report(self) -> None:
        """Print a human-readable reliability report."""
        r = self.report()
        print("\n" + "=" * 60)
        print("  TOOL RELIABILITY REPORT")
        print("=" * 60)
        print(f"  Total tools:     {r['total_tools']}")
        print(f"  Reliable (≥70%): {r['reliable_tools']}")
        print(f"  Unreliable (<50%): {r['unreliable_tools']}")
        print()
        print("  ── Top 10 Tools ──")
        for t in r['top_10']:
            print(f"  {t['name']:<30s} conf={t['confidence']} "
                  f"rate={t['success_rate']} n={t['executions']}")
        print()
        if r['bottom_5']:
            print("  ── Bottom 5 Tools ──")
            for t in r['bottom_5']:
                print(f"  {t['name']:<30s} conf={t['confidence']} "
                      f"failures={t['failures']} reasons={t['reasons']}")
        print()
        print("  ── Category Best ──")
        for cat, info in r['categories'].items():
            print(f"  {cat:<25s} → {info['best']} ({info['confidence']})")
        print("=" * 60)


# Global singleton
tool_reliability = ToolReliability()