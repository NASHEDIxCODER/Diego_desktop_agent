"""
VisionService — Structured screen understanding for desktop automation.

Target latencies:
  - Desktop capture:   <20ms  (via mss, delegated to ScreenCapture)
  - Frame differencing: <1ms  (perceptual hash comparison)
  - OCR:                <200ms (PaddleOCR/EasyOCR/Tesseract)
  - UI tree building:   <800ms (element detection + heuristics)
  - Vision model (VL):  <2s    (Qwen2.5-VL, optional)

Gating strategy (NEVER analyze every frame):
  1. Capture screen (fast)
  2. Frame differencing (hash check — skip if unchanged)
  3. Active window check (skip if same window title)
  4. User explicitly asked (always run)

Only run OCR + vision when:
  - active window changed
  - frame changed significantly (pHash differs)
  - user explicitly asked (force=True)

Produces a structured UITree, NOT raw OCR text:

    Desktop
      Window "PyCharm"
        Button "Run"
        Button "Stop"
        Tab "Terminal"
        Text "Fatal Error"

The tree is JSON-serializable for LLM prompt injection.

Extends BaseService. Conversational engine accesses through:
  - vision_service.analyze() → builds full UITree
  - vision_service.quick_context() → compact text summary for LLM
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.service import BaseService
from services.screen_capture import screen_capture_service, CaptureResult, FrameDiff
from services.ui_tree import (
    UIDesktop,
    UIWindow,
    UIElement,
    ElementType,
)

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────

_HAS_CV2 = False
try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]

_HAS_PADDLEOCR = False
try:
    from paddleocr import PaddleOCR as _PaddleOCR

    _HAS_PADDLEOCR = True
except ImportError:
    _PaddleOCR = None  # type: ignore[assignment]

_HAS_EASYOCR = False
try:
    import easyocr as _easyocr  # noqa: F401

    _HAS_EASYOCR = True
except ImportError:
    _easyocr = None  # type: ignore[assignment]

_HAS_TESSERACT = False
try:
    import pytesseract as _pytesseract

    _HAS_TESSERACT = True
except ImportError:
    _pytesseract = None  # type: ignore[assignment]

_HAS_QWEN_VL = False
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _QWEN_MODEL  # noqa: F401
    from transformers import AutoProcessor as _QWEN_PROCESSOR  # noqa: F401

    _HAS_QWEN_VL = True
except ImportError:
    _QWEN_MODEL = None  # type: ignore[assignment]
    _QWEN_PROCESSOR = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class OCRBox:
    """A single OCR-detected text box."""
    text: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    confidence: float = 0.0


@dataclass
class VisionContext:
    """The full result of vision analysis, ready for LLM injection."""
    desktop: Optional[UIDesktop] = None
    active_window_title: str = ""
    ocr_text: str = ""
    raw_ocr_boxes: List[OCRBox] = field(default_factory=list)
    ui_tree_text: str = ""           # Compact string representation
    caption: str = ""                # VL model scene description (optional)
    capture_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    vision_time_ms: float = 0.0
    total_time_ms: float = 0.0
    from_cache: bool = False
    error: str = ""

    @property
    def compact_summary(self) -> str:
        """A compact text summary suitable for LLM prompt injection."""
        parts = []
        if self.active_window_title:
            parts.append(f"Active window: {self.active_window_title}")
        if self.ui_tree_text:
            parts.append(self.ui_tree_text)
        elif self.ocr_text:
            parts.append(f"Visible text: {self.ocr_text[:500]}")
        if self.caption:
            parts.append(f"Scene: {self.caption}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# OCR Engine (PaddleOCR → EasyOCR → Tesseract)
# ═══════════════════════════════════════════════════════════════════


class OCREngine:
    """
    Multi-backend OCR with automatic failover.

    Prioritises accuracy: PaddleOCR > EasyOCR > Tesseract.
    All backends return per-box coordinates + confidence, not just raw text.
    """

    def __init__(self):
        self._paddle: Optional[Any] = None
        self._easyocr_reader: Optional[Any] = None
        self._active: str = "none"

    def initialize(self) -> bool:
        """Try to initialise the best available OCR backend."""
        # PaddleOCR
        if _HAS_PADDLEOCR:
            try:
                self._paddle = _PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=False,
                    show_log=False,
                )
                self._active = "paddleocr"
                logger.info("[VisionService] OCR backend: PaddleOCR")
                return True
            except Exception as e:
                logger.warning("[VisionService] PaddleOCR init failed: %s", e)

        # EasyOCR
        if _HAS_EASYOCR:
            try:
                self._easyocr_reader = _easyocr.Reader(["en"], gpu=False)
                self._active = "easyocr"
                logger.info("[VisionService] OCR backend: EasyOCR")
                return True
            except Exception as e:
                logger.warning("[VisionService] EasyOCR init failed: %s", e)

        # Tesseract
        if _HAS_TESSERACT:
            try:
                # Verify tesseract binary is available
                import subprocess
                result = subprocess.run(
                    ["tesseract", "--version"],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    self._active = "tesseract"
                    logger.info("[VisionService] OCR backend: Tesseract")
                    return True
            except Exception:
                pass

        logger.warning("[VisionService] No OCR backend available")
        return False

    @property
    def ready(self) -> bool:
        return self._active != "none"

    def ocr(self, image: np.ndarray) -> List[OCRBox]:
        """
        Extract text boxes from an RGB image.

        Returns a list of OCRBox with per-word/per-line coordinates and
        confidence scores.
        """
        if self._active == "paddleocr" and self._paddle is not None:
            return self._ocr_paddle(image)
        if self._active == "easyocr" and self._easyocr_reader is not None:
            return self._ocr_easyocr(image)
        if self._active == "tesseract":
            return self._ocr_tesseract(image)
        return []

    def _ocr_paddle(self, image: np.ndarray) -> List[OCRBox]:
        try:
            results = self._paddle.ocr(image, cls=True)
            boxes: List[OCRBox] = []
            if results and results[0]:
                for line in results[0]:
                    bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0

                    if text and conf > 0.3:
                        xs = [p[0] for p in bbox_points]
                        ys = [p[1] for p in bbox_points]
                        x, y = int(min(xs)), int(min(ys))
                        w, h = int(max(xs) - x), int(max(ys) - y)
                        boxes.append(OCRBox(text=str(text), bbox=(x, y, w, h), confidence=float(conf)))
            return boxes
        except Exception as e:
            logger.debug("[VisionService] PaddleOCR error: %s", e)
            return []

    def _ocr_easyocr(self, image: np.ndarray) -> List[OCRBox]:
        try:
            results = self._easyocr_reader.readtext(image)
            boxes: List[OCRBox] = []
            for bbox_points, text, conf in results:
                if text and conf > 0.3:
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    boxes.append(OCRBox(text=str(text), bbox=(x, y, w, h), confidence=float(conf)))
            return boxes
        except Exception as e:
            logger.debug("[VisionService] EasyOCR error: %s", e)
            return []

    def _ocr_tesseract(self, image: np.ndarray) -> List[OCRBox]:
        try:
            import cv2 as _cv
            bgr = _cv.cvtColor(image, _cv.COLOR_RGB2BGR)
            data = _pytesseract.image_to_data(bgr, output_type=_pytesseract.Output.DICT)
            boxes: List[OCRBox] = []
            n = len(data["text"])
            for i in range(n):
                text = (data["text"][i] or "").strip()
                conf = int(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.0
                if text and conf > 0.3:
                    x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                    if w > 0 and h > 0:
                        boxes.append(OCRBox(text=text, bbox=(x, y, w, h), confidence=conf))
            return boxes
        except Exception as e:
            logger.debug("[VisionService] Tesseract error: %s", e)
            return []

    def close(self) -> None:
        self._paddle = None
        self._easyocr_reader = None


# ═══════════════════════════════════════════════════════════════════
# UI element detector — heuristics to classify OCR boxes
# ═══════════════════════════════════════════════════════════════════


class UIDetector:
    """
    Classifies OCR text boxes into UI element types using heuristics.

    Each box is examined for:
      - Position (top bar → window title, toolbar; bottom → status bar)
      - Text patterns (short + verb-like → button; hierarchical → menu item)
      - Spatial relationships (boxes in a row → tabs; close together → toolbar)
      - Neighbor analysis (boxes near icons → icon labels)
    """

    # Common UI button text patterns
    _BUTTON_PATTERNS = re.compile(
        r"^(OK|Cancel|Apply|Submit|Save|Delete|Close|Yes|No|Next|Back|"
        r"Finish|Run|Stop|Start|Pause|Resume|Retry|Skip|"
        r"Open|New|Edit|Copy|Paste|Undo|Redo|"
        r"Send|Search|Clear|Refresh|Reload|"
        r"Login|Logout|Sign In|Sign Up|Register|"
        r"Download|Upload|Install|Update|"
        r"Play|Pause|Stop|Mute|"
        r"Add|Remove|Create|Delete|Modify|"
        r"Accept|Decline|Reject|Allow|Deny|"
        r"Enable|Disable|On|Off|"
        r"Confirm|Dismiss|Ignore|"
        r"Settings|Preferences|Options|"
        r"Help|About|Exit|Quit)$",
        re.IGNORECASE,
    )

    # Common menu labels
    _MENU_PATTERNS = re.compile(
        r"^(File|Edit|View|Tools|Window|Help|Navigate|Code|Refactor|"
        r"Run|Debug|Profile|Build|VCS|Git|Bookmarks|"
        r"Format|Project|Settings|Preferences|Plugins|"
        r"Terminal|Run|Debug|Stop|"
        r"History|Bookmarks|Favorites|"
        r"Analyze|Inspect|Generate|"
        r"Recent|New|Open|Save)$",
        re.IGNORECASE,
    )

    # Tab label indicators
    _TAB_PATTERNS = re.compile(
        r"^(.*\.(py|js|ts|html|css|json|yaml|yml|md|txt|java|cpp|c|h|rs|go|rb|"
        r"php|sql|xml|toml|cfg|ini|sh|bash|zsh|fish|ps1))$",
        re.IGNORECASE,
    )

    def __init__(self):
        pass

    def build_tree(
        self,
        ocr_boxes: List[OCRBox],
        active_window_title: str,
        image_width: int,
        image_height: int,
    ) -> UIDesktop:
        """
        Build a structured UI tree from OCR boxes + heuristics.

        The tree is:
          UIDesktop
            └── UIWindow (active window)
                  ├── UIButton / UIText / UITab / UIMenu / UIIcon / etc.

        Returns a UIDesktop with one UIWindow child (the active window).
        """
        desktop = UIDesktop()
        window = UIWindow(
            title=active_window_title,
            bounding_box=(0, 0, image_width, image_height),
        )

        if not ocr_boxes:
            desktop.add_window(window)
            return desktop

        # ── Classify each OCR box ──────────────────────────
        for box in sorted(ocr_boxes, key=lambda b: (b.bbox[1], b.bbox[0])):
            element_type = self._classify(box, image_width, image_height)
            element = UIElement(
                element_type=element_type,
                label=box.text,
                bounding_box=box.bbox,
                confidence=box.confidence,
            )
            window.element.add_child(element)

        # ── Post-processing: detect tab groups ─────────────
        self._detect_tab_groups(window.element)

        # ── Post-processing: detect menus from menu items ──
        self._detect_menus(window.element)

        desktop.add_window(window)
        return desktop

    def _classify(
        self, box: OCRBox, img_w: int, img_h: int
    ) -> ElementType:
        """Classify a single OCR box."""
        text = box.text.strip()
        x, y, w, h = box.bbox
        text_len = len(text)

        # Position-based hints
        at_top = y < img_h * 0.08
        at_bottom = y > img_h * 0.92
        at_left = x < img_w * 0.05
        at_right = x > img_w * 0.95
        is_short = text_len <= 20
        is_very_short = text_len <= 5

        # Tab: file extensions or short at top
        if self._TAB_PATTERNS.match(text):
            return ElementType.TAB

        # Button: short, verb-like, clickable
        if is_very_short and self._BUTTON_PATTERNS.match(text):
            return ElementType.BUTTON

        # Button heuristics for short, action-oriented text
        if is_very_short and not at_top:
            # Single-word capitalized often = button
            if text[0].isupper() and text.isalpha():
                return ElementType.BUTTON

        # Menu: common menu bar labels, at top
        if at_top and self._MENU_PATTERNS.match(text):
            return ElementType.MENU

        # Menu item: short text inside a menu region
        if is_short and at_top and not self._MENU_PATTERNS.match(text):
            # Individual items under a menu
            return ElementType.MENU_ITEM

        # Tab at top (not menu-like)
        if at_top and is_very_short:
            return ElementType.TAB

        # Input: short text near the bottom (terminal input line)
        if at_bottom and is_short:
            return ElementType.INPUT

        # Link: starts with http or common link patterns
        if text.startswith(("http://", "https://", "www.", "ftp://")):
            return ElementType.LINK

        # Checkbox / radio: starts with common markers
        if text.startswith(("☐", "☑", "☒", "○", "●", "◉", "[ ]", "[x]", "[X]", "( )", "(*)")):
            return ElementType.CHECKBOX

        # Icon: very short text in a small, isolated box
        if is_very_short and w < 50 and h < 50:
            return ElementType.ICON

        # Label: short text, often before input fields
        if is_very_short and text.endswith(":"):
            return ElementType.LABEL

        # Default: text
        return ElementType.TEXT

    @staticmethod
    def _detect_tab_groups(window_element: UIElement) -> None:
        """
        Cluster adjacent tabs into a TabGroup.

        Tabs are typically in a horizontal row with similar y-coordinates.
        """
        tabs = window_element.find_by_type(ElementType.TAB)
        if len(tabs) < 2:
            return

        # Sort by x position
        tabs_sorted = sorted(tabs, key=lambda t: t.x)
        groups: List[List[UIElement]] = []
        current_group = [tabs_sorted[0]]

        for i in range(1, len(tabs_sorted)):
            prev = tabs_sorted[i - 1]
            curr = tabs_sorted[i]
            # Same y-line (within 10px) and close horizontally (within 200px)
            if abs(curr.y - prev.y) <= 10 and (curr.x - (prev.x + prev.width)) <= 200:
                current_group.append(curr)
            else:
                groups.append(current_group)
                current_group = [curr]
        groups.append(current_group)

        # For each group > 1, wrap in TAB_GROUP
        for group in groups:
            if len(group) > 1:
                # Check if any tab metadata says "active" based on visual contrast
                # (simplified: the first tab is active)
                group[0].metadata["active"] = True
                # The group relationship is implicit from the adjacency metadata
                for tab in group[1:]:
                    tab.metadata["active"] = False

    @staticmethod
    def _detect_menus(window_element: UIElement) -> None:
        """
        Group adjacent MENU and MENU_ITEM elements.

        Menus at the top bar are often followed by their items.
        """
        menus = window_element.find_by_type(ElementType.MENU)
        menu_items = window_element.find_by_type(ElementType.MENU_ITEM)

        if not menus or not menu_items:
            return

        # For each menu, find menu items that are vertically close below
        for menu in menus:
            mx, my, mw, mh = menu.x, menu.y, menu.width, menu.height
            nearby_items = []
            for item in menu_items[:]:
                if abs(item.x - mx) < mw + 50 and item.y - (my + mh) < 30:
                    nearby_items.append(item)
                    menu_items.remove(item)
            for item in nearby_items:
                menu.add_child(item)

    # ── Image preprocessing for better OCR ─────────────────────────

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """
        Apply OpenCV preprocessing to improve OCR accuracy.

        Steps:
          1. Convert to grayscale
          2. Denoise (fast Non-Local Means)
          3. Adaptive thresholding (binarization)
          4. Morphological close (connect broken text)
          5. Sharpen (unsharp mask)

        Returns the preprocessed image (RGB format, same size).
        """
        if not _HAS_CV2:
            return image

        import cv2 as _cv

        try:
            gray = _cv.cvtColor(image, _cv.COLOR_RGB2GRAY)
        except Exception:
            return image

        # Denoise
        denoised = _cv.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Adaptive threshold
        binary = _cv.adaptiveThreshold(
            denoised, 255, _cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            _cv.THRESH_BINARY, 11, 2,
        )

        # Morphological close (connect text components)
        kernel = _cv.getStructuringElement(_cv.MORPH_RECT, (2, 2))
        closed = _cv.morphologyEx(binary, _cv.MORPH_CLOSE, kernel)

        # Sharpen
        blur = _cv.GaussianBlur(closed, (0, 0), 3)
        sharpened = _cv.addWeighted(closed, 1.5, blur, -0.5, 0)

        # Convert back to RGB for OCR engines
        result = _cv.cvtColor(sharpened, _cv.COLOR_GRAY2RGB)
        return result


# ═══════════════════════════════════════════════════════════════════
# VisionService — main orchestrator
# ═══════════════════════════════════════════════════════════════════


class VisionService(BaseService):
    """
    Structured screen understanding service.

    Extends BaseService for lifecycle. Provides:
      - analyze() — full capture → OCR → UI tree pipeline
      - quick_context() — compact LLM-ready text summary
      - ui_tree() — get current UIDesktop directly
      - force_analyze() — bypass all gating, always run

    Gating:
      - If frame hasn't changed → skip (return cached)
      - If window title unchanged + frame same → skip
      - force=True → always run full pipeline
    """

    name = "vision_service"
    dependencies: List[str] = ["screen_capture"]

    def __init__(self):
        super().__init__()
        self._ocr = OCREngine()
        self._detector = UIDetector()
        self._last_context: Optional[VisionContext] = None
        self._last_window_title: str = ""
        self._last_frame_hash: str = ""
        self._analyze_count: int = 0
        self._skip_count: int = 0

        # Tunables
        self.preprocess_for_ocr: bool = True
        self.use_vl_model: bool = False  # Qwen2.5-VL (optional, heavy)

    # ── BaseService contract ──────────────────────────────────────

    async def _start(self) -> bool:
        """Initialise OCR backend."""
        ocr_ok = self._ocr.initialize()
        details = {
            "ocr": self._ocr._active,
            "preprocess": self.preprocess_for_ocr and _HAS_CV2,
            "vl_model": self.use_vl_model and _HAS_QWEN_VL,
            "opencv": _HAS_CV2,
        }
        self.set_health("ready" if ocr_ok else "degraded (no OCR)", details)
        logger.info(
            "[VisionService] Backends — ocr=%s preprocess=%s vl=%s",
            self._ocr._active if ocr_ok else "none",
            "✓" if _HAS_CV2 else "✗",
            "✓" if (_HAS_QWEN_VL and self.use_vl_model) else "✗",
        )
        return True  # Always "ready" — OCR is optional, capture still works

    async def _stop(self) -> None:
        """Release OCR resources."""
        self._ocr.close()
        self._last_context = None
        logger.info("[VisionService] Stopped")

    @property
    def ready(self) -> bool:
        return self._ocr.ready

    # ── Public API ─────────────────────────────────────────────────

    async def analyze(
        self,
        force: bool = False,
        source: str = "active_window",
        include_ocr: bool = True,
        include_tree: bool = True,
    ) -> VisionContext:
        """
        Run the full vision pipeline: capture → OCR → UI tree.

        Args:
            force: Bypass frame-difference gating (always run full pipeline).
            source: "active_window" or "fullscreen".
            include_ocr: Run OCR (otherwise only capture).
            include_tree: Build structured UI tree (otherwise raw OCR only).

        Returns:
            VisionContext with structured results. Never returns raw HTML.

        Target latency: <800ms total (capture + OCR + tree).
        """
        t0 = time.time()
        ctx = VisionContext()

        # ── Step 1: Capture ────────────────────────────────
        # Determine source
        if source == "active_window":
            cap = await screen_capture_service.capture_active_window()
        else:
            cap = await screen_capture_service.capture_fullscreen()

        if cap is None or not cap.is_valid:
            ctx.error = "capture failed"
            ctx.total_time_ms = (time.time() - t0) * 1000
            return ctx

        ctx.capture_time_ms = (time.time() - t0) * 1000
        ctx.active_window_title = cap.source_label.replace("window:", "").split(" (fullscreen")[0]

        # ── Frame-difference gating ────────────────────────
        if not force and self._last_frame_hash and cap.frame_hash == self._last_frame_hash:
            # Frame hasn't changed — return cached if we have it
            if self._last_context is not None and self._last_context.ui_tree_text:
                self._skip_count += 1
                cached = self._last_context
                cached.total_time_ms = (time.time() - t0) * 1000
                cached.from_cache = True
                logger.debug("[VisionService] Frame unchanged — returning cached context")
                return cached

        self._last_frame_hash = cap.frame_hash
        self._last_window_title = ctx.active_window_title

        # ── Step 2: OCR ────────────────────────────────────
        if include_ocr and self._ocr.ready:
            t_ocr = time.time()
            image = cap.image

            # Optional preprocessing
            if self.preprocess_for_ocr and _HAS_CV2:
                image = self._detector.preprocess(image)

            ocr_boxes = await asyncio.get_event_loop().run_in_executor(
                None, self._ocr.ocr, image,
            )
            ctx.ocr_time_ms = (time.time() - t_ocr) * 1000
            ctx.raw_ocr_boxes = ocr_boxes
            ctx.ocr_text = " ".join(b.text for b in ocr_boxes) if ocr_boxes else ""

            # ── Step 3: Build UI tree ──────────────────────
            if include_tree and ocr_boxes:
                t_tree = time.time()
                desktop = self._detector.build_tree(
                    ocr_boxes,
                    ctx.active_window_title,
                    cap.width,
                    cap.height,
                )
                ctx.desktop = desktop
                ctx.ui_tree_text = desktop.to_compact_str()
                ctx.vision_time_ms = (time.time() - t_tree) * 1000
        else:
            # No OCR — at least note what window is active
            if include_tree:
                desktop = UIDesktop()
                window = UIWindow(
                    title=ctx.active_window_title,
                    bounding_box=(0, 0, cap.width, cap.height),
                )
                desktop.add_window(window)
                ctx.desktop = desktop
                ctx.ui_tree_text = desktop.to_compact_str()

        # ── Cache ──────────────────────────────────────────
        self._last_context = ctx
        self._analyze_count += 1

        ctx.total_time_ms = (time.time() - t0) * 1000
        logger.info(
            "[VisionService] Analysis complete: %.0fms (capture=%.0f ocr=%.0f tree=%.0f) "
            "window='%s' ocr_boxes=%d",
            ctx.total_time_ms,
            ctx.capture_time_ms,
            ctx.ocr_time_ms,
            ctx.vision_time_ms,
            ctx.active_window_title,
            len(ctx.raw_ocr_boxes),
        )
        return ctx

    async def force_analyze(self) -> VisionContext:
        """Always run the full pipeline (bypasses gating)."""
        return await self.analyze(force=True)

    async def quick_context(self) -> str:
        """
        Return a compact text summary for LLM prompt injection.

        Uses frame-difference gating: returns cached text if the screen
        hasn't changed since the last analysis.

        This is the primary interface for the conversation engine —
        called when a user asks about their screen.
        """
        ctx = await self.analyze(force=False)
        return ctx.compact_summary

    async def ocr_only(self) -> str:
        """Run OCR only (no UI tree) and return raw text."""
        ctx = await self.analyze(force=False, include_tree=False)
        return ctx.ocr_text

    async def ui_tree(self, force: bool = False) -> Optional[UIDesktop]:
        """Return the current structured UI tree, or None."""
        ctx = await self.analyze(force=force)
        return ctx.desktop

    async def find_element(self, label: str, element_type: Optional[ElementType] = None) -> List[UIElement]:
        """
        Search the UI tree for elements matching a label and optional type.

        Returns a list of matching elements (may be empty).
        """
        ctx = await self.analyze(force=True)
        if ctx.desktop is None:
            return []

        results: List[UIElement] = []
        for window in ctx.desktop.windows:
            if element_type is not None:
                results.extend(window.find_by_type(element_type))
            results.extend(window.find_by_label(label))
        return results

    async def click_element(self, label: str) -> Optional[Tuple[int, int]]:
        """
        Find a clickable element by label and return its centre coordinates.

        Returns (x, y) or None if no matching clickable element is found.
        """
        elements = await self.find_element(label)
        # Prefer buttons, then any clickable type
        clickable_types = {ElementType.BUTTON, ElementType.LINK, ElementType.TAB,
                           ElementType.MENU_ITEM, ElementType.CHECKBOX, ElementType.ICON}
        for el in elements:
            if el.element_type in clickable_types and el.bounding_box:
                cx = el.x + el.width // 2
                cy = el.y + el.height // 2
                logger.info("[VisionService] click_element '%s' → (%d, %d)", label, cx, cy)
                return (cx, cy)
        # Fallback: any element with matching label
        for el in elements:
            if el.bounding_box:
                cx = el.x + el.width // 2
                cy = el.y + el.height // 2
                return (cx, cy)
        return None

    # ── Diagnostics ───────────────────────────────────────────────

    @property
    def analyze_count(self) -> int:
        return self._analyze_count

    @property
    def skip_count(self) -> int:
        return self._skip_count

    @property
    def last_tree(self) -> Optional[str]:
        """Return the last UI tree as a compact string (for debugging)."""
        if self._last_context:
            return self._last_context.ui_tree_text
        return None

    def clear_cache(self) -> None:
        """Clear the last analysis cache (force next analyze to re-run)."""
        self._last_context = None
        self._last_frame_hash = ""
        self._last_window_title = ""


# Global singleton
vision_service = VisionService()