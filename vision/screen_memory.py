"""
ScreenMemory — Frame history with state comparison for the vision pipeline.

Remembers previous frames, their UI trees, OCR results, and semantic
summaries. The Planner can compare previous vs current state to understand
what changed after an action.

Stores:
  - window title
  - frame hash (pHash)
  - UI tree (full structured tree)
  - OCR boxes (raw + processed)
  - semantic summary (compact LLM-ready string)
  - timestamp
  - action history (what was done to get here)

Capacity: rolling window of N frames (default 20).
Old frames are evicted FIFO.

Enables:
  - "What changed?" queries
  - "Where was the login button before?"
  - Action verification (did the click actually change the UI?)
  - Context recovery after actions

Structured logging: [MEMORY]
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScreenSnapshot:
    """A single frame in screen memory."""
    # Identity
    frame_id: int = 0
    timestamp: float = 0.0                          # monotonic time when captured
    wall_time: str = ""                              # ISO 8601 wall clock

    # Window
    window_title: str = ""
    app_type: str = ""                               # from LayoutAnalyzer
    app_name: str = ""

    # Frame
    frame_hash: str = ""                             # pHash
    full_hash: str = ""                              # SHA-256 (for precise dedup)
    width: int = 0
    height: int = 0

    # Vision results (expensive — only stored when processing happens)
    ocr_boxes_json: str = ""                         # JSON-serialized OCRBox list
    ocr_text: str = ""                               # raw concatenated text
    ui_tree_json: str = ""                           # JSON-serialized UIDesktop
    ui_tree_text: str = ""                           # compact string representation
    layout_json: str = ""                            # JSON-serialized WindowLayout

    # Semantic
    semantic_summary: str = ""                        # human-readable description
    page_type: str = ""                               # from ScreenReasoner
    interactive_elements: str = ""                    # JSON-serialized ScreenElement list
    error_elements: str = ""                          # JSON-serialized error list

    # Action context
    action_before: str = ""                           # action that produced this state
    action_after: str = ""                            # action taken after this state

    # Metadata
    total_pipeline_ms: float = 0.0
    from_cache: bool = False

    @property
    def has_vision(self) -> bool:
        """True if this snapshot has full vision results."""
        return bool(self.ui_tree_text or self.ocr_text)

    @property
    def age_ms(self) -> float:
        """Age of this snapshot in milliseconds."""
        return (time.monotonic() - self.timestamp) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict for storage/transmission."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "wall_time": self.wall_time,
            "window_title": self.window_title,
            "app_type": self.app_type,
            "app_name": self.app_name,
            "frame_hash": self.frame_hash,
            "full_hash": self.full_hash,
            "width": self.width,
            "height": self.height,
            "ocr_text": self.ocr_text[:1000],
            "ui_tree_text": self.ui_tree_text,
            "semantic_summary": self.semantic_summary,
            "page_type": self.page_type,
            "action_before": self.action_before,
            "action_after": self.action_after,
            "total_pipeline_ms": self.total_pipeline_ms,
            "from_cache": self.from_cache,
        }

    def quick_diff(self, previous: "ScreenSnapshot") -> Dict[str, Any]:
        """
        Compute a quick comparison with a previous snapshot.

        Returns a dict describing what changed.
        """
        changes: Dict[str, Any] = {
            "frame_ids": (previous.frame_id, self.frame_id),
            "elapsed_ms": (self.timestamp - previous.timestamp) * 1000,
        }

        # Window changed?
        if self.window_title != previous.window_title:
            changes["window_changed"] = {
                "from": previous.window_title,
                "to": self.window_title,
            }

        # App changed?
        if self.app_type != previous.app_type:
            changes["app_changed"] = {
                "from": f"{previous.app_type}/{previous.app_name}",
                "to": f"{self.app_type}/{self.app_name}",
            }

        # Frame changed?
        if self.frame_hash != previous.frame_hash:
            changes["frame_changed"] = True

        # UI elements added/removed (quick count)
        if previous.ui_tree_text and self.ui_tree_text:
            prev_lines = previous.ui_tree_text.split("\n")
            curr_lines = self.ui_tree_text.split("\n")
            changes["ui_tree_lines"] = {
                "previous": len(prev_lines),
                "current": len(curr_lines),
                "delta": len(curr_lines) - len(prev_lines),
            }

        # OCR text length changed?
        prev_len = len(previous.ocr_text)
        curr_len = len(self.ocr_text)
        if abs(curr_len - prev_len) > 50:
            changes["ocr_text_changed"] = {
                "previous_chars": prev_len,
                "current_chars": curr_len,
                "delta": curr_len - prev_len,
            }

        # Page type changed?
        if self.page_type and previous.page_type and self.page_type != previous.page_type:
            changes["page_type_changed"] = {
                "from": previous.page_type,
                "to": self.page_type,
            }

        # Semantic summary different?
        if self.semantic_summary and previous.semantic_summary and self.semantic_summary != previous.semantic_summary:
            changes["semantic_changed"] = True

        if len(changes) <= 2:  # only frame_ids + elapsed_ms
            changes["unchanged"] = True
        else:
            changes["unchanged"] = False

        return changes


class ScreenMemory:
    """
    Rolling memory of recent screen snapshots.

    Capacity: 20 frames (configurable).
    Stores full vision results for the most recent N frames.
    Older frames are evicted (FIFO).

    Provides:
      - Snapshot storage and retrieval
      - Previous/current comparison (for action verification)
      - Context for "what changed?" queries
      - History for planner state recovery
    """

    DEFAULT_CAPACITY: int = 20

    def __init__(self, capacity: int = 20):
        self._capacity = max(1, capacity)
        self._frames: deque[ScreenSnapshot] = deque(maxlen=self._capacity)
        self._frame_counter: int = 0
        self._last_action: str = ""

    # ── Storage ──────────────────────────────────────────

    def store(self, snapshot: ScreenSnapshot) -> None:
        """
        Store a screen snapshot in memory.

        Args:
            snapshot: The ScreenSnapshot to store (must have frame_id set).
        """
        if snapshot.frame_id <= 0:
            self._frame_counter += 1
            snapshot.frame_id = self._frame_counter
        else:
            self._frame_counter = max(self._frame_counter, snapshot.frame_id)

        snapshot.timestamp = time.monotonic()
        from datetime import datetime, timezone
        snapshot.wall_time = datetime.now(timezone.utc).isoformat()
        snapshot.action_before = self._last_action

        self._frames.append(snapshot)
        logger.debug("[MEMORY] Stored frame #%d: hash=%s window='%s' app=%s ocr_chars=%d tree_lines=%d",
                     snapshot.frame_id, snapshot.frame_hash[:8],
                     snapshot.window_title[:40], snapshot.app_type,
                     len(snapshot.ocr_text),
                     len(snapshot.ui_tree_text.split("\n")) if snapshot.ui_tree_text else 0)

    def record_action(self, action_description: str) -> None:
        """
        Record an action that was just performed.
        The NEXT stored snapshot will have this as its action_before.
        """
        self._last_action = action_description
        if self._frames:
            self._frames[-1].action_after = action_description
        logger.debug("[MEMORY] Recorded action: %s", action_description[:80])

    # ── Retrieval ────────────────────────────────────────

    def latest(self) -> Optional[ScreenSnapshot]:
        """Return the most recent snapshot, or None."""
        return self._frames[-1] if self._frames else None

    def previous(self) -> Optional[ScreenSnapshot]:
        """Return the second most recent snapshot, or None."""
        return self._frames[-2] if len(self._frames) >= 2 else None

    def get(self, frame_id: int) -> Optional[ScreenSnapshot]:
        """Return a snapshot by frame_id, or None."""
        for frame in self._frames:
            if frame.frame_id == frame_id:
                return frame
        return None

    def get_range(self, start_id: int, end_id: int) -> List[ScreenSnapshot]:
        """Return snapshots with frame_id in [start_id, end_id]."""
        return [f for f in self._frames if start_id <= f.frame_id <= end_id]

    def all(self) -> List[ScreenSnapshot]:
        """Return all stored snapshots (most recent last)."""
        return list(self._frames)

    # ── Comparison ───────────────────────────────────────

    def compare_latest(self) -> Optional[Dict[str, Any]]:
        """
        Compare the two most recent snapshots.

        Returns a diff dict, or None if there aren't two snapshots.
        """
        prev = self.previous()
        curr = self.latest()
        if prev is None or curr is None:
            return None
        return curr.quick_diff(prev)

    def last_action_changed_ui(self) -> Tuple[bool, str]:
        """
        Check whether the last action actually changed the UI.

        Returns:
            (changed: bool, explanation: str)
        """
        diff = self.compare_latest()
        if diff is None:
            return False, "no previous frame to compare"

        if diff.get("unchanged", True):
            return False, "UI appears unchanged after action"

        reasons = []
        if diff.get("frame_changed"):
            reasons.append("frame content changed")
        if diff.get("window_changed"):
            reasons.append(f"window changed to '{diff['window_changed']['to']}'")
        if diff.get("ocr_text_changed"):
            reasons.append(f"text changed ({diff['ocr_text_changed']['delta']:+d} chars)")
        if diff.get("ui_tree_lines"):
            dt = diff["ui_tree_lines"]
            reasons.append(f"UI elements changed ({dt['delta']:+d} lines)")
        if diff.get("page_type_changed"):
            reasons.append(f"page type changed to '{diff['page_type_changed']['to']}'")

        if reasons:
            return True, "; ".join(reasons)
        return False, "no detectable UI change"

    # ── Context for LLM ──────────────────────────────────

    def context_for_llm(self, max_frames: int = 3) -> str:
        """
        Build a compact context block for LLM injection.

        Includes recent window transitions, UI state, and actions.
        """
        recent = list(self._frames)[-max_frames:]
        if not recent:
            return ""

        lines: List[str] = ["[RECENT SCREEN HISTORY]"]
        for i, snap in enumerate(recent):
            age_s = snap.age_ms / 1000
            header = f"  Frame #{snap.frame_id} ({age_s:.1f}s ago):"
            lines.append(header)
            if snap.window_title:
                lines.append(f"    Window: {snap.window_title[:80]}")
            if snap.app_type:
                lines.append(f"    App: {snap.app_type}/{snap.app_name}")
            if snap.page_type:
                lines.append(f"    Page type: {snap.page_type}")
            if snap.semantic_summary:
                lines.append(f"    Summary: {snap.semantic_summary[:200]}")
            elif snap.ui_tree_text:
                # Truncate tree to first 5 elements
                tree_lines = snap.ui_tree_text.split("\n")[:5]
                lines.append(f"    UI: {len(tree_lines)} elements visible")
            if snap.action_before and i > 0:
                lines.append(f"    ↳ Action before: {snap.action_before}")
            if snap.action_after:
                lines.append(f"    ↳ Action after: {snap.action_after}")

        return "\n".join(lines)

    def what_changed(self) -> str:
        """
        Human-readable description of what changed since the previous frame.

        Used to answer "What changed?" queries.
        """
        diff = self.compare_latest()
        if diff is None:
            return "I haven't seen a previous screen state yet."

        if diff.get("unchanged", True):
            return "Nothing has changed since the last time I looked."

        parts: List[str] = []
        if diff.get("window_changed"):
            wc = diff["window_changed"]
            parts.append(f"The active window changed from '{wc['from']}' to '{wc['to']}'")
        if diff.get("app_changed"):
            ac = diff["app_changed"]
            parts.append(f"The application changed from {ac['from']} to {ac['to']}")
        if diff.get("frame_changed"):
            parts.append("The screen content changed")
        if diff.get("ocr_text_changed"):
            oc = diff["ocr_text_changed"]
            parts.append(f"The visible text changed ({oc['delta']:+d} characters)")
        if diff.get("ui_tree_lines"):
            ut = diff["ui_tree_lines"]
            parts.append(f"The number of UI elements changed ({ut['delta']:+d})")
        if diff.get("page_type_changed"):
            pt = diff["page_type_changed"]
            parts.append(f"The page type changed from '{pt['from']}' to '{pt['to']}'")

        return ". ".join(parts) + "."

    # ── Memory recall ────────────────────────────────────

    def find_element_in_history(self, label: str, element_type: Optional[str] = None,
                                 max_lookback: int = 5) -> Optional[Tuple[ScreenSnapshot, Dict[str, Any]]]:
        """
        Search recent frames for a UI element by label.

        Useful for "Where was the login button?" queries.

        Returns (snapshot, element_dict) or None.
        """
        recent = list(self._frames)[-max_lookback:]
        for snap in reversed(recent):
            if not snap.ui_tree_json:
                continue
            try:
                tree = json.loads(snap.ui_tree_json)
                found = self._search_tree(tree, label, element_type)
                if found:
                    return snap, found
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _search_tree(node: Dict[str, Any], label: str,
                      element_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Recursively search a UI tree node for a matching element."""
        node_label = (node.get("label", "") or "").lower()
        node_type = node.get("type", "")

        if label.lower() in node_label:
            if element_type is None or node_type == element_type:
                return node

        for child in node.get("children", []):
            result = ScreenMemory._search_tree(child, label, element_type)
            if result is not None:
                return result
        return None

    # ── Visual memory (recurring interfaces) ─────────────

    def find_similar_window(self, window_title: str, app_type: str,
                             max_age_s: float = 3600.0) -> Optional[ScreenSnapshot]:
        """
        Find a recent snapshot of the same window/app for comparison.

        Returns the most recent matching snapshot, or None.
        """
        now = time.monotonic()
        for snap in reversed(self._frames):
            if (now - snap.timestamp) > max_age_s:
                continue
            if snap.app_type == app_type and snap.window_title == window_title:
                return snap
            # Fuzzy match: same app type, similar title (first N chars)
            if snap.app_type == app_type and window_title[:30] in snap.window_title:
                return snap
        return None

    # ── Management ───────────────────────────────────────

    def clear(self) -> None:
        """Clear all stored snapshots."""
        self._frames.clear()
        self._frame_counter = 0
        self._last_action = ""
        logger.info("[MEMORY] Screen memory cleared")

    def trim(self, max_age_s: float = 300.0) -> int:
        """
        Remove snapshots older than max_age_s seconds.

        Returns the number of removed snapshots.
        """
        now = time.monotonic()
        removed = 0
        while self._frames and (now - self._frames[0].timestamp) > max_age_s:
            self._frames.popleft()
            removed += 1
        if removed:
            logger.debug("[MEMORY] Trimmed %d old snapshots (>%.0fs)", removed, max_age_s)
        return removed

    # ── Diagnostics ──────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._frames)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def last_frame_id(self) -> int:
        return self._frame_counter

    @property
    def last_action(self) -> str:
        return self._last_action

    def report(self) -> Dict[str, Any]:
        """Return diagnostic summary."""
        latest = self.latest()
        return {
            "frame_count": self.count,
            "capacity": self._capacity,
            "last_frame_id": self._frame_counter,
            "last_action": self._last_action,
            "latest_window": latest.window_title[:60] if latest else "",
            "latest_app": f"{latest.app_type}/{latest.app_name}" if latest else "",
            "latest_age_ms": latest.age_ms if latest else 0,
            "latest_has_vision": latest.has_vision if latest else False,
        }


# Global singleton
screen_memory = ScreenMemory()