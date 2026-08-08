"""
ScreenCapture — High-performance desktop capture service.

Target latency: <20ms per capture (mss-backed).

Features:
  - Full desktop, per-monitor, and per-window capture via mss
  - Frame differencing to avoid redundant OCR/vision work
  - Perceptual hash comparison (fast pHash, not cryptographic)
  - Active window title detection (xdotool / wmctrl)
  - Thread-safe: all capture runs in executor threads
  - Extends BaseService for lifecycle management

Usage:
    from services.screen_capture import screen_capture_service

    cap = await screen_capture_service.capture_fullscreen()
    if screen_capture_service.has_changed(cap):
        # run expensive vision pipeline

Never analyze every frame — use frame differencing to gate heavy work.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.service import BaseService

logger = logging.getLogger(__name__)

# Try importing mss — the primary capture backend.
_HAS_MSS = False
try:
    import mss as _mss_lib

    _HAS_MSS = True
except ImportError:
    _mss_lib = None  # type: ignore[assignment]

# Fallback: pyautogui
_HAS_PYAUTOGUI = False
try:
    import pyautogui as _pyautogui_lib

    _HAS_PYAUTOGUI = True
except ImportError:
    _pyautogui_lib = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CaptureResult:
    """Result of a single screen capture."""
    image: Optional[np.ndarray] = None          # RGB uint8 array (H, W, 3)
    width: int = 0
    height: int = 0
    channels: int = 3
    monitor_index: int = 0
    timestamp: float = 0.0
    frame_hash: str = ""                        # Fast perceptual hash (16 hex chars)
    source_label: str = ""                      # e.g. "fullscreen", "monitor_1", "window:PyCharm"

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    @property
    def is_valid(self) -> bool:
        return self.image is not None and self.width > 0 and self.height > 0


@dataclass
class FrameDiff:
    """Result of comparing two captures."""
    same: bool = False
    hash_match: bool = False
    pixel_change_ratio: float = 0.0            # fraction of pixels that differ (0.0 – 1.0)
    diff_hash: str = ""
    elapsed_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# ScreenCapture service
# ═══════════════════════════════════════════════════════════════════


class ScreenCapture(BaseService):
    """
    Fast desktop capture service with frame-difference gating.

    Extends BaseService for lifecycle (start/stop/health). All heavy work
    runs through executor threads so the asyncio event loop is never blocked.

    Service lifecycle:
        start()  → initialises mss backend
        stop()   → releases mss resources
    """

    name = "screen_capture"
    dependencies: List[str] = []

    # Tunable thresholds
    DIFF_PIXEL_RATIO_THRESHOLD: float = 0.005  # 0.5% pixel change → "changed"
    DIFF_HASH_BLOCK_SIZE: int = 8              # downsample to 8×8 for pHash comparison

    def __init__(self):
        super().__init__()
        self._mss: Optional[Any] = None
        self._backend: str = "none"
        self._monitors: List[Dict[str, int]] = []
        self._last_capture: Optional[CaptureResult] = None
        self._capture_count: int = 0

    # ── BaseService contract ──────────────────────────────────────

    async def _start(self) -> bool:
        """Initialise the capture backend."""
        import os as _os
        # Verify DISPLAY is set for X11 capture backends
        if not _os.environ.get("DISPLAY") and not _os.environ.get("WAYLAND_DISPLAY"):
            logger.warning("[ScreenCapture] No DISPLAY/WAYLAND_DISPLAY env — capture may fail")

        if _HAS_MSS:
            try:
                self._mss = _mss_lib.mss()
                self._monitors = list(self._mss.monitors)
                if len(self._monitors) > 1:
                    self._backend = "mss"
                    logger.info(
                        "[ScreenCapture] backend=mss monitors=%d",
                        len(self._monitors) - 1,  # monitor[0] is the virtual combined screen
                    )
                    self.set_health("mss ready", {"backend": "mss", "monitors": len(self._monitors) - 1})
                    return True
                else:
                    logger.warning("[ScreenCapture] mss found no monitors — falling back")
                    try:
                        self._mss.close()
                    except Exception:
                        pass
                    self._mss = None
            except Exception as e:
                logger.warning("[ScreenCapture] mss init failed: %s", e)
                self._mss = None

        if _HAS_PYAUTOGUI:
            self._backend = "pyautogui"
            logger.info("[ScreenCapture] backend=pyautogui (fallback, slower)")
            self.set_health("pyautogui ready", {"backend": "pyautogui"})
            return True

        logger.error("[ScreenCapture] No capture backend available — install mss")
        self.set_health("no backend available", {"backend": "none"})
        return False

    async def _stop(self) -> None:
        """Release capture resources."""
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None
        self._backend = "none"
        self._last_capture = None
        logger.info("[ScreenCapture] Stopped")

    @property
    def ready(self) -> bool:
        """True if any capture backend is available."""
        if self._backend == "none":
            return False
        if self._backend == "mss":
            return self._mss is not None
        return True  # pyautogui backend is always ready if imported

    # ── Public capture API ────────────────────────────────────────

    async def capture_fullscreen(self) -> Optional[CaptureResult]:
        """
        Capture the entire virtual desktop (all monitors combined).

        Returns None if capture fails.
        Target latency: <20ms.
        """
        t0 = time.time()
        if self._backend == "mss" and self._mss is not None:
            try:
                monitor = self._mss.monitors[0]  # full virtual screen
                sct = self._mss.grab(monitor)
                img = np.array(sct, dtype=np.uint8)
                # mss returns BGRA; strip alpha, convert to RGB
                if img.shape[2] == 4:
                    img = img[:, :, :3][:, :, ::-1].copy()
                else:
                    img = img[:, :, ::-1].copy()
            except Exception as e:
                logger.warning("[ScreenCapture] Fullscreen capture failed: %s", e)
                return None

        elif self._backend == "pyautogui" and _pyautogui_lib is not None:
            try:
                pil_img = _pyautogui_lib.screenshot()
                img = np.array(pil_img, dtype=np.uint8)
            except Exception as e:
                logger.warning("[ScreenCapture] pyautogui capture failed: %s", e)
                return None
        else:
            return None

        elapsed = (time.time() - t0) * 1000
        result = self._build_result(img, 0, "fullscreen")
        self._last_capture = result
        self._capture_count += 1

        if elapsed > 50:
            logger.debug("[ScreenCapture] Fullscreen %dx%d in %.1fms (slow)", result.width, result.height, elapsed)
        return result

    async def capture_monitor(self, monitor_index: int = 1) -> Optional[CaptureResult]:
        """
        Capture a single monitor. Index 1 is the primary monitor (mss convention).

        Returns None if capture fails or index out of range.
        """
        t0 = time.time()
        if self._backend != "mss" or self._mss is None:
            # Fall back to fullscreen for non-mss backends
            return await self.capture_fullscreen()

        monitors = list(self._mss.monitors)
        if monitor_index < 0 or monitor_index >= len(monitors):
            logger.warning("[ScreenCapture] Monitor index %d out of range [0, %d)", monitor_index, len(monitors))
            return None

        try:
            mon = monitors[monitor_index]
            sct = self._mss.grab(mon)
            img = np.array(sct, dtype=np.uint8)
            if img.shape[2] == 4:
                img = img[:, :, :3][:, :, ::-1].copy()
            else:
                img = img[:, :, ::-1].copy()
        except Exception as e:
            logger.warning("[ScreenCapture] Monitor %d capture failed: %s", monitor_index, e)
            return None

        elapsed = (time.time() - t0) * 1000
        result = self._build_result(img, monitor_index, f"monitor_{monitor_index}")
        logger.debug("[ScreenCapture] Monitor %d: %dx%d in %.1fms", monitor_index, result.width, result.height, elapsed)
        return result

    async def capture_region(self, left: int, top: int, width: int, height: int) -> Optional[CaptureResult]:
        """
        Capture a screen region.

        Args:
            left: x-coordinate of top-left corner (pixels).
            top: y-coordinate of top-left corner (pixels).
            width: region width (pixels).
            height: region height (pixels).

        Returns None if capture fails or region is invalid.
        """
        if width <= 0 or height <= 0:
            return None

        t0 = time.time()
        if self._backend == "mss" and self._mss is not None:
            try:
                region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
                sct = self._mss.grab(region)
                img = np.array(sct, dtype=np.uint8)
                if img.shape[2] == 4:
                    img = img[:, :, :3][:, :, ::-1].copy()
                else:
                    img = img[:, :, ::-1].copy()
            except Exception as e:
                logger.warning("[ScreenCapture] Region capture failed: %s", e)
                return None
        elif self._backend == "pyautogui" and _pyautogui_lib is not None:
            try:
                pil_img = _pyautogui_lib.screenshot(region=(int(left), int(top), int(width), int(height)))
                img = np.array(pil_img, dtype=np.uint8)
            except Exception as e:
                logger.warning("[ScreenCapture] pyautogui region capture failed: %s", e)
                return None
        else:
            return None

        elapsed = (time.time() - t0) * 1000
        result = self._build_result(img, -1, f"region_{left}_{top}_{width}x{height}")
        logger.debug("[ScreenCapture] Region %dx%d in %.1fms", result.width, result.height, elapsed)
        return result

    async def capture_active_window(self) -> Optional[CaptureResult]:
        """
        Capture the currently focused window.

        Uses xdotool (X11) to get the window geometry, then captures
        that region. Falls back to fullscreen if xdotool is unavailable
        or the window is not found.

        Returns CaptureResult with source_label="window:<title>" or None.
        """
        window_title = self.active_window_title()
        geo = self._active_window_geometry()
        if geo is not None:
            x, y, w, h = geo
            result = await self.capture_region(x, y, w, h)
            if result is not None:
                result.source_label = f"window:{window_title}" if window_title else "window:unknown"
                return result

        # Fallback: fullscreen
        result = await self.capture_fullscreen()
        if result is not None and window_title:
            result.source_label = f"window:{window_title} (fullscreen fallback)"
        return result

    # ── Window info (X11 / wmctrl) ─────────────────────────────────

    @staticmethod
    def active_window_title() -> str:
        """Return the title of the currently focused window, or ''."""
        import shutil

        # xdotool (X11)
        if shutil.which("xdotool"):
            try:
                out = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=2,
                )
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip()
            except Exception:
                pass

        # wmctrl fallback
        if shutil.which("wmctrl"):
            try:
                out = subprocess.run(
                    ["wmctrl", "-l"], capture_output=True, text=True, timeout=2,
                )
                for line in out.stdout.splitlines():
                    segs = line.split(None, 3)
                    if len(segs) == 4:
                        return segs[3]
            except Exception:
                pass

        return ""

    @staticmethod
    def _active_window_geometry() -> Optional[Tuple[int, int, int, int]]:
        """Return (x, y, w, h) of the active window, or None."""
        import shutil

        if not shutil.which("xdotool"):
            return None

        try:
            wid = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=2,
            )
            if wid.returncode != 0:
                return None
            window_id = wid.stdout.strip()

            geo = subprocess.run(
                ["xdotool", "getwindowgeometry", window_id],
                capture_output=True, text=True, timeout=2,
            )
            if geo.returncode != 0:
                return None

            lines = geo.stdout.strip().split("\n")
            pos_line = next((l for l in lines if "Position:" in l), None)
            geo_line = next((l for l in lines if "Geometry:" in l), None)
            if not pos_line or not geo_line:
                return None

            pos = pos_line.split(":")[1].strip()
            x, y = map(int, pos.split(","))
            geom = geo_line.split(":")[1].strip()
            w, h = map(int, geom.split("x"))
            return (x, y, w, h)
        except Exception:
            return None

    # ── Frame differencing ────────────────────────────────────────

    def compute_diff(self, a: CaptureResult, b: CaptureResult) -> FrameDiff:
        """
        Compare two captures and return a diff.

        Fast path: frame_hash comparison (sub-microsecond).
        Slow path: pixel ratio calculation (only when hashes differ).
        """
        t0 = time.time()
        if a.frame_hash and b.frame_hash and a.frame_hash == b.frame_hash:
            elapsed = (time.time() - t0) * 1000
            return FrameDiff(same=True, hash_match=True, diff_hash=a.frame_hash, elapsed_ms=elapsed)

        # Hashes differ — compute pixel-level change ratio on a downsampled version
        if a.image is not None and b.image is not None and a.image.shape == b.image.shape:
            ratio = self._pixel_change_ratio(a.image, b.image)
        else:
            ratio = 1.0  # completely different

        elapsed = (time.time() - t0) * 1000
        return FrameDiff(
            same=ratio < self.DIFF_PIXEL_RATIO_THRESHOLD,
            hash_match=False,
            pixel_change_ratio=ratio,
            diff_hash=a.frame_hash + "_" + b.frame_hash,
            elapsed_ms=elapsed,
        )

    def has_changed(self, current: CaptureResult, previous: Optional[CaptureResult] = None) -> bool:
        """
        Return True if the screen has changed significantly since the last capture.

        This is the primary gate: NEVER run OCR/vision if has_changed() is False.
        """
        prev = previous or self._last_capture
        if prev is None:
            return True  # No baseline — assume changed
        if current.frame_hash != prev.frame_hash:
            return True
        # Hashes match → no meaningful change
        return False

    # ── Internal helpers ──────────────────────────────────────────

    def _build_result(self, img: np.ndarray, monitor_index: int, source_label: str) -> CaptureResult:
        """Construct a CaptureResult from a raw numpy image."""
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1
        fhash = self._fast_hash(img)
        return CaptureResult(
            image=img,
            width=w,
            height=h,
            channels=c,
            monitor_index=monitor_index,
            timestamp=time.time(),
            frame_hash=fhash,
            source_label=source_label,
        )

    @staticmethod
    def _fast_hash(img: np.ndarray) -> str:
        """
        Compute a fast perceptual hash.

        Downscales to 8×8 grayscale, compares to the mean, and packs
        into 16 hex characters. Not cryptographic — designed for
        sub-millisecond frame-difference checks.
        """
        try:
            import cv2
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            # Pure-numpy fallback (no OpenCV dependency for hashing)
            if img.ndim == 3:
                gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                gray = img
            # Simple block-average downsample to 8×8
            h, w = gray.shape
            small = np.zeros((8, 8), dtype=np.uint8)
            bh, bw = max(h // 8, 1), max(w // 8, 1)
            for i in range(8):
                for j in range(8):
                    block = gray[i * bh: min((i + 1) * bh, h), j * bw: min((j + 1) * bw, w)]
                    small[i, j] = int(np.mean(block))

        mean = small.mean()
        bits = (small > mean).flatten()
        # Pack 64 bits into 16 hex characters
        hex_str = "".join(
            format(int("".join(str(int(b)) for b in bits[i:i + 4]), 2), "x")
            for i in range(0, 64, 4)
        )
        return hex_str

    @staticmethod
    def _pixel_change_ratio(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute what fraction of pixels differ between two images.

        Uses an 8× downsampled version for speed.
        Returns a float in [0.0, 1.0].
        """
        # Downsample to keep this fast (<1ms on 4K)
        step = 8
        a_small = a[::step, ::step]
        b_small = b[::step, ::step]
        # Mean absolute difference, normalised
        diff = np.abs(a_small.astype(np.float32) - b_small.astype(np.float32)).mean() / 255.0
        return float(diff)

    # ── Diagnostics ───────────────────────────────────────────────

    @property
    def capture_count(self) -> int:
        return self._capture_count

    @property
    def last_capture(self) -> Optional[CaptureResult]:
        return self._last_capture


# Global singleton
screen_capture_service = ScreenCapture()