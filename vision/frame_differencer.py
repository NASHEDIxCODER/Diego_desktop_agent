"""
FrameDifferencer — High-speed frame hashing + motion detection + gating.

THE GATEKEEPER of the vision pipeline. No OCR/UI detection runs unless
this module decides the frame is "interesting."

Gating rules (ONLY process vision when):
  1. Desktop frame changed (pHash differs)
  2. Active window title changed
  3. Mouse clicked (detected via event bus)
  4. Keyboard input happened (detected via event bus)
  5. User explicitly requested vision (force=True)
  6. Planner requires updated state

Otherwise: return cached result immediately.

Supports:
  - Perceptual hash (pHash) in <2ms
  - Motion mask (which regions changed) in <10ms
  - Mouse/keyboard event tracking
  - Structured logging: [FRAME]

Architecture:
  Screen Capture → FrameHash → MotionDetection → GatingDecision
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)

_HAS_CV2 = False
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

@dataclass
class MotionRegion:
    """A rectangular region where motion was detected."""
    x: int
    y: int
    width: int
    height: int
    change_ratio: float = 0.0  # 0.0-1.0, how much this region changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "change_ratio": round(self.change_ratio, 4),
        }


@dataclass
class FrameDiffResult:
    """Complete frame difference analysis."""
    frame_hash: str = ""
    previous_hash: str = ""
    same: bool = True
    hash_differed: bool = False
    pixel_change_ratio: float = 0.0  # global pixel change fraction
    motion_regions: List[MotionRegion] = field(default_factory=list)
    motion_detected: bool = False
    elapsed_hash_us: float = 0.0    # microseconds
    elapsed_motion_us: float = 0.0   # microseconds
    elapsed_total_us: float = 0.0    # microseconds

    @property
    def interesting(self) -> bool:
        """True if the frame should trigger full vision processing."""
        return not self.same or self.motion_detected

    @property
    def summary(self) -> str:
        parts = []
        if self.same:
            parts.append("unchanged")
        else:
            parts.append(f"changed (pixel_ratio={self.pixel_change_ratio:.4f})")
        if self.motion_detected:
            parts.append(f"motion in {len(self.motion_regions)} regions")
        return ", ".join(parts)


@dataclass
class GatingDecision:
    """Whether to run the full vision pipeline."""
    should_process: bool = False
    reason: str = ""          # WHY we decided to process or skip
    force: bool = False       # user explicitly requested
    window_changed: bool = False
    frame_changed: bool = False
    mouse_clicked: bool = False
    keyboard_input: bool = False
    planner_requested: bool = False
    quality: str = ""         # "fresh", "cached", "stale"
    cache_age_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════
# FrameDifferencer
# ═══════════════════════════════════════════════════════════════

class FrameDifferencer:
    """
    High-speed frame comparison and gating.

    Target latencies:
      - pHash:      <2ms
      - Motion:     <10ms
      - Gating:     <0.5ms (logic only)

    Design:
      - pHash computed immediately on capture
      - Motion mask computed only if pHash differs
      - Gating decision based on hash + events + window state
    """

    # Tunable thresholds
    HASH_BLOCK_SIZE: int = 8                  # 8×8 downsample for pHash
    MOTION_GRID_CELLS: int = 16               # divide frame into N×N grid cells
    MOTION_CELL_THRESHOLD: float = 0.02       # 2% change in a cell → "motion"
    PIXEL_CHANGE_THRESHOLD: float = 0.005     # 0.5% global → "changed"
    CACHE_MAX_AGE_MS: float = 5000.0          # 5s max cache before auto-refresh

    def __init__(self):
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_hash: str = ""
        self._prev_full_hash: str = ""        # full SHA-256 for cache key
        self._prev_window_title: str = ""
        self._last_process_time: float = 0.0
        self._mouse_clicks: int = 0
        self._keyboard_events: int = 0
        self._cache_age: float = 0.0

        # Diagnostics
        self.hash_count: int = 0
        self.motion_count: int = 0
        self.skip_count: int = 0
        self.process_count: int = 0

    # ── Hash computation ───────────────────────────────────

    @staticmethod
    def compute_phash(image: np.ndarray) -> str:
        """
        Compute a fast 64-bit perceptual hash.

        Downscales to 8×8 grayscale, compares to the mean, and packs
        into 16 hex characters. Sub-2ms on 4K.

        Args:
            image: RGB or grayscale numpy array (H, W, 3) or (H, W).

        Returns:
            16-character hex string.
        """
        t0 = time.perf_counter_ns()

        if _HAS_CV2:
            try:
                if image.ndim == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_NEAREST)
                mean = small.mean()
                bits = (small > mean).flatten()
                phash = "".join(
                    format(int("".join(str(int(b)) for b in bits[i:i + 4]), 2), "x")
                    for i in range(0, 64, 4)
                )
                elapsed = (time.perf_counter_ns() - t0) / 1000
                if elapsed > 2000:
                    logger.debug("[FRAME] pHash slow: %.0fµs", elapsed)
                return phash
            except Exception as e:
                logger.warning("[FRAME] OpenCV pHash failed: %s — falling back to numpy", e)

        # Pure-numpy fallback
        try:
            if image.ndim == 3:
                gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                gray = image
            h, w = gray.shape
            small = np.zeros((8, 8), dtype=np.uint8)
            bh, bw = max(h // 8, 1), max(w // 8, 1)
            for i in range(8):
                for j in range(8):
                    block = gray[i * bh: min((i + 1) * bh, h), j * bw: min((j + 1) * bw, w)]
                    small[i, j] = int(np.mean(block))
            mean = small.mean()
            bits = (small > mean).flatten()
            phash = "".join(
                format(int("".join(str(int(b)) for b in bits[i:i + 4]), 2), "x")
                for i in range(0, 64, 4)
            )
            return phash
        except Exception as e:
            logger.error("[FRAME] pHash failed entirely: %s", e)
            return "0000000000000000"

    @staticmethod
    def compute_full_hash(image: np.ndarray) -> str:
        """Compute a full SHA-256 hash for cache isolation (different images with same pHash)."""
        try:
            return hashlib.sha256(image.tobytes()).hexdigest()[:16]
        except Exception:
            return ""

    # ── Motion detection ───────────────────────────────────

    def compute_motion(self, current: np.ndarray, previous: Optional[np.ndarray] = None,
                       grid_cells: Optional[int] = None) -> FrameDiffResult:
        """
        Compare current frame against previous to detect motion regions.

        Works at two levels:
          1. Global pixel change ratio (fast, <2ms)
          2. Grid-cell motion regions (slightly slower, <10ms)

        Args:
            current: Current frame as numpy array.
            previous: Previous frame. Uses self._prev_frame if None.
            grid_cells: Number of grid cells per side. Default MOTION_GRID_CELLS.

        Returns:
            FrameDiffResult with motion analysis.
        """
        t0 = time.perf_counter_ns()
        prev = previous if previous is not None else self._prev_frame
        cells = grid_cells if grid_cells is not None else self.MOTION_GRID_CELLS

        result = FrameDiffResult()
        result.frame_hash = self.compute_phash(current)
        result.previous_hash = self._prev_hash

        # Fast path: hash match → no change
        if prev is not None and result.frame_hash == self._prev_hash:
            result.same = True
            result.hash_differed = False
            result.elapsed_total_us = (time.perf_counter_ns() - t0) / 1000
            self.hash_count += 1
            return result

        result.hash_differed = True
        self.hash_count += 1

        if prev is None:
            result.same = False
            result.pixel_change_ratio = 1.0
            result.elapsed_total_us = (time.perf_counter_ns() - t0) / 1000
            return result

        # ── Global pixel change ─────────────────────────
        try:
            t_pixel = time.perf_counter_ns()

            # Downsample for speed (step=4)
            if current.ndim == 3 and prev.ndim == 3 and current.shape == prev.shape:
                step = 4
                a_small = current[::step, ::step].astype(np.float32)
                b_small = prev[::step, ::step].astype(np.float32)
                diff = np.abs(a_small - b_small).mean() / 255.0
                result.pixel_change_ratio = float(diff)
            else:
                result.pixel_change_ratio = 1.0

            result.elapsed_motion_us = (time.perf_counter_ns() - t_pixel) / 1000

        except Exception as e:
            logger.debug("[FRAME] Pixel change computation failed: %s", e)
            result.pixel_change_ratio = 1.0

        result.same = result.pixel_change_ratio < self.PIXEL_CHANGE_THRESHOLD

        # ── Grid-cell motion regions ─────────────────────
        if not result.same and _HAS_CV2 and current.ndim == 3 and prev is not None:
            try:
                t_grid = time.perf_counter_ns()
                result.motion_regions = self._grid_motion(
                    current, prev, cells, self.MOTION_CELL_THRESHOLD)
                result.motion_detected = len(result.motion_regions) > 0
                result.elapsed_motion_us += (time.perf_counter_ns() - t_grid) / 1000
                self.motion_count += 1
            except Exception as e:
                logger.debug("[FRAME] Grid motion failed: %s", e)

        result.elapsed_total_us = (time.perf_counter_ns() - t0) / 1000
        return result

    @staticmethod
    def _grid_motion(current: np.ndarray, previous: np.ndarray,
                     cells: int, threshold: float) -> List[MotionRegion]:
        """Divide frame into grid cells and detect which changed."""
        h, w = current.shape[:2]
        cell_h = max(h // cells, 1)
        cell_w = max(w // cells, 1)

        # Grayscale for faster comparison
        if current.ndim == 3:
            curr_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray, prev_gray = current, previous

        regions: List[MotionRegion] = []
        for row in range(cells):
            for col in range(cells):
                y1 = row * cell_h
                y2 = min((row + 1) * cell_h, h)
                x1 = col * cell_w
                x2 = min((col + 1) * cell_w, w)

                curr_cell = curr_gray[y1:y2, x1:x2].astype(np.float32)
                prev_cell = prev_gray[y1:y2, x1:x2].astype(np.float32)

                if curr_cell.shape != prev_cell.shape:
                    continue

                diff = np.abs(curr_cell - prev_cell).mean() / 255.0
                if diff > threshold:
                    regions.append(MotionRegion(
                        x=x1, y=y1,
                        width=x2 - x1, height=y2 - y1,
                        change_ratio=float(diff),
                    ))

        return regions

    # ── Frame storage ──────────────────────────────────────

    def store_frame(self, image: np.ndarray, window_title: str = "") -> None:
        """Store the current frame as the new baseline for future comparisons."""
        self._prev_frame = image.copy() if image is not None else None
        self._prev_hash = self.compute_phash(image) if image is not None else ""
        self._prev_full_hash = self.compute_full_hash(image) if image is not None else ""
        self._prev_window_title = window_title
        self._last_process_time = time.monotonic()

    # ── Event tracking ─────────────────────────────────────

    def record_mouse_click(self) -> None:
        """Record a mouse click event for gating."""
        self._mouse_clicks += 1

    def record_keyboard_input(self) -> None:
        """Record keyboard input for gating."""
        self._keyboard_events += 1

    def consume_events(self) -> Tuple[int, int]:
        """Get and reset the pending mouse/keyboard event counters."""
        clicks = self._mouse_clicks
        keys = self._keyboard_events
        self._mouse_clicks = 0
        self._keyboard_events = 0
        return clicks, keys

    # ── Gating decision ────────────────────────────────────

    def should_process(
        self,
        force: bool = False,
        window_title: str = "",
        planner_requested: bool = False,
    ) -> GatingDecision:
        """
        Decide whether to run the full vision pipeline.

        This is THE gate that prevents unnecessary OCR/vision work.

        Args:
            force: User/planner explicitly requested vision.
            window_title: Current active window title.
            planner_requested: Planner needs updated state.

        Returns:
            GatingDecision with should_process=True/False and reason.
        """
        clicks, keys = self.consume_events()
        decision = GatingDecision()

        # ── Force always runs ───────────────────────────
        if force:
            decision.should_process = True
            decision.force = True
            decision.reason = "force requested"
            decision.quality = "fresh"
            self.process_count += 1
            logger.info("[FRAME] Gating: FORCE — full pipeline requested")
            return decision

        # ── User interacted → process ────────────────────
        if clicks > 0:
            decision.should_process = True
            decision.mouse_clicked = True
            decision.reason = f"mouse click ({clicks} events)"
            decision.quality = "fresh"
            self.process_count += 1
            logger.info("[FRAME] Gating: PROCESS — mouse click event (%d)", clicks)
            return decision

        if keys > 0:
            decision.should_process = True
            decision.keyboard_input = True
            decision.reason = f"keyboard input ({keys} events)"
            decision.quality = "fresh"
            self.process_count += 1
            logger.info("[FRAME] Gating: PROCESS — keyboard input (%d)", keys)
            return decision

        # ── Planner requested → process ──────────────────
        if planner_requested:
            decision.should_process = True
            decision.planner_requested = True
            decision.reason = "planner requested updated state"
            decision.quality = "fresh"
            self.process_count += 1
            logger.info("[FRAME] Gating: PROCESS — planner requested")
            return decision

        # ── Window title changed → process ───────────────
        if window_title and window_title != self._prev_window_title:
            decision.should_process = True
            decision.window_changed = True
            decision.reason = f"window changed to '{window_title}'"
            decision.quality = "fresh"
            self.process_count += 1
            logger.info("[FRAME] Gating: PROCESS — window changed: '%s'", window_title)
            return decision

        # ── Frame changed → process ──────────────────────
        if not self._prev_hash:
            # No baseline — must process
            decision.should_process = True
            decision.frame_changed = True
            decision.reason = "no baseline frame"
            decision.quality = "fresh"
            self.process_count += 1
            logger.info("[FRAME] Gating: PROCESS — no baseline (first frame)")
            return decision

        # Check cache age
        self._cache_age = (time.monotonic() - self._last_process_time) * 1000
        if self._cache_age > self.CACHE_MAX_AGE_MS:
            decision.should_process = True
            decision.reason = f"cache expired ({self._cache_age:.0f}ms > {self.CACHE_MAX_AGE_MS:.0f}ms)"
            decision.quality = "stale"
            self.process_count += 1
            logger.info("[FRAME] Gating: PROCESS — cache expired (%.0fms)", self._cache_age)
            return decision

        # ── Nothing changed → SKIP ───────────────────────
        decision.should_process = False
        decision.reason = "frame unchanged"
        decision.quality = "cached"
        decision.cache_age_ms = self._cache_age
        self.skip_count += 1
        logger.debug("[FRAME] Gating: SKIP — frame unchanged (cache %.0fms old)", self._cache_age)
        return decision

    # ── Diagnostics ───────────────────────────────────────

    @property
    def prev_hash(self) -> str:
        return self._prev_hash

    @property
    def prev_window_title(self) -> str:
        return self._prev_window_title

    def reset(self) -> None:
        """Reset all state (for testing / recovery)."""
        self._prev_frame = None
        self._prev_hash = ""
        self._prev_full_hash = ""
        self._prev_window_title = ""
        self._last_process_time = 0.0
        self._mouse_clicks = 0
        self._keyboard_events = 0
        self._cache_age = 0.0

    def report(self) -> Dict[str, Any]:
        """Return diagnostic summary."""
        return {
            "hash_count": self.hash_count,
            "motion_count": self.motion_count,
            "skip_count": self.skip_count,
            "process_count": self.process_count,
            "skip_ratio": f"{self.skip_count / max(self.skip_count + self.process_count, 1):.1%}",
            "cache_age_ms": round(self._cache_age, 1),
            "prev_hash": self._prev_hash,
            "prev_window": self._prev_window_title,
        }


# Global singleton
frame_differencer = FrameDifferencer()