"""
OCRPipeline — Enhanced multi-backend OCR with box merging, dedup, confidence.

Wraps and enhances the existing OCREngine from vision_service.py.
Adds:
  - Box merging (merge overlapping/similar boxes)
  - Duplicate removal (identical text at same position)
  - Confidence estimation (per-box, per-region, global)
  - Text hierarchy (heading, body, caption, button-text, code)
  - Region-aware OCR (only run on relevant regions)
  - Self-healing (retry with preprocessing on failure)
  - Confidence-based filtering

OCR backends (priority order):
  1. PaddleOCR (best accuracy)
  2. EasyOCR (good accuracy, GPU optional)
  3. Tesseract (universal fallback)

Structured logging: [OCR]
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────

_HAS_CV2 = False
try:
    import cv2

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


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

class TextClass(str):
    """Semantic classification of OCR text."""
    HEADING = "heading"
    BODY = "body"
    CAPTION = "caption"
    BUTTON_TEXT = "button_text"
    CODE = "code"
    ERROR = "error"
    LABEL = "label"
    LINK = "link"
    MENU_ITEM = "menu_item"
    STATUS = "status"
    UNKNOWN = "unknown"


@dataclass
class OCRBox:
    """A single OCR-detected text box with enhanced metadata."""
    text: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    confidence: float = 0.0
    text_class: str = TextClass.UNKNOWN
    backend: str = ""           # "paddleocr", "easyocr", "tesseract"
    region: str = ""            # layout region this text belongs to
    is_merged: bool = False     # was this box created by merging?
    source_boxes: int = 1       # how many original boxes were merged
    line_number: int = 0        # estimated line number (top to bottom)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def overlap_ratio(self, other: "OCRBox") -> float:
        """Compute IoU (intersection over union) with another box."""
        x_overlap = max(0, min(self.right, other.right) - max(self.x, other.x))
        y_overlap = max(0, min(self.bottom, other.bottom) - max(self.y, other.y))
        intersection = x_overlap * y_overlap
        area_a = self.width * self.height
        area_b = other.width * other.height
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
            "text_class": self.text_class,
            "backend": self.backend,
            "region": self.region,
        }


@dataclass
class OCRResult:
    """Enhanced OCR result with metadata."""
    boxes: List[OCRBox] = field(default_factory=list)
    text: str = ""                       # concatenated text
    backend: str = ""                    # which backend was used
    preprocess_applied: bool = False
    elapsed_raw_ms: float = 0.0         # raw OCR time
    elapsed_postprocess_ms: float = 0.0  # box merging / dedup time
    elapsed_total_ms: float = 0.0
    box_count_raw: int = 0             # boxes before post-processing
    box_count_final: int = 0           # boxes after post-processing
    merged_count: int = 0              # how many boxes were merged
    dedup_count: int = 0               # how many boxes were removed
    error: str = ""
    retry_count: int = 0

    @property
    def avg_confidence(self) -> float:
        if not self.boxes:
            return 0.0
        return sum(b.confidence for b in self.boxes) / len(self.boxes)

    @property
    def high_confidence_boxes(self) -> List[OCRBox]:
        """Return boxes with confidence >= 0.7."""
        return [b for b in self.boxes if b.confidence >= 0.7]

    def text_in_region(self, x: int, y: int, w: int, h: int) -> str:
        """Return only text from boxes within the given region."""
        region_boxes = [
            b for b in self.boxes
            if b.x >= x and b.y >= y
            and b.right <= (x + w) and b.bottom <= (y + h)
        ]
        region_boxes.sort(key=lambda b: (b.y, b.x))
        return " ".join(b.text for b in region_boxes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:2000],
            "box_count": self.box_count_final,
            "avg_confidence": round(self.avg_confidence, 3),
            "backend": self.backend,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════
# Enhanced OCREngine
# ═══════════════════════════════════════════════════════════════

class EnhancedOCREngine:
    """
    Multi-backend OCR with box merging, dedup, and self-healing.

    Target latency: <150ms for a single region on modern hardware.
    """

    # Merging thresholds
    MERGE_OVERLAP_THRESHOLD: float = 0.3      # IoU > this → merge
    MERGE_VERTICAL_CLOSE_PX: int = 5           # vertically closer than this → merge
    DEDUP_TEXT_SIMILARITY: float = 0.8         # text similarity > this → dedup
    DEDUP_POSITION_CLOSE_PX: int = 3           # position within this → dedup

    # Confidence thresholds
    MIN_CONFIDENCE: float = 0.3
    HIGH_CONFIDENCE: float = 0.7

    def __init__(self):
        self._paddle: Optional[Any] = None
        self._easyocr_reader: Optional[Any] = None
        self._active: str = "none"
        self._retry_count: int = 0

    def initialize(self) -> bool:
        """Try to initialise the best available OCR backend."""
        # PaddleOCR
        if _HAS_PADDLEOCR:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
                logger.info(
                    "[OCR] CUDA available=%s device=%s",
                    cuda_available,
                    torch.cuda.get_device_name(0) if cuda_available else "CPU",
                )

                paddle_kwargs = {
                    "lang": "en",
                }

                # Newer PaddleOCR versions use device instead of use_gpu.
                try:
                    paddle_kwargs["device"] = "gpu:0" if cuda_available else "cpu"
                    self._paddle = _PaddleOCR(**paddle_kwargs)
                except (TypeError, ValueError):
                    # Compatibility with older PaddleOCR versions.
                    paddle_kwargs.pop("device", None)
                    paddle_kwargs["use_gpu"] = cuda_available
                    self._paddle = _PaddleOCR(**paddle_kwargs)

                self._active = "paddleocr"
                logger.info(
                    "[OCR] Backend: PaddleOCR device=%s",
                    "GPU" if cuda_available else "CPU",
                )
                return True

            except Exception as e:
                logger.warning("[OCR] PaddleOCR init failed: %s", e)

        # EasyOCR
        if _HAS_EASYOCR:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())

                self._easyocr_reader = _easyocr.Reader(
                    ["en"],
                    gpu=cuda_available,
                )

                self._active = "easyocr"

                logger.info(
                    "[OCR] Backend: EasyOCR device=%s",
                    "GPU" if cuda_available else "CPU",
                )

                return True
                # self._easyocr_reader = _easyocr.Reader(["en"], gpu=False)
                self._active = "easyocr"
                logger.info("[OCR] Backend: EasyOCR")
                return True
            except Exception as e:
                logger.warning("[OCR] EasyOCR init failed: %s", e)

        # Tesseract
        if _HAS_TESSERACT:
            try:
                import subprocess
                result = subprocess.run(
                    ["tesseract", "--version"],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    self._active = "tesseract"
                    self._configure_tesseract_data()
                    logger.info("[OCR] Backend: Tesseract")
                    return True
            except Exception:
                pass

        logger.warning("[OCR] No backend available")
        return False

    @staticmethod
    def _configure_tesseract_data() -> None:
        """
        Locate the tesseract 'eng.traineddata' and set TESSDATA_PREFIX.

        On some systems the traineddata is not in the default
        /usr/share/tessdata location (e.g. bundled with another app).
        Without this, pytesseract fails with "Error opening data file".
        """
        import os
        import glob

        # If TESSDATA_PREFIX is already set and valid, leave it.
        existing = os.environ.get("TESSDATA_PREFIX", "")
        if existing and os.path.exists(os.path.join(existing, "eng.traineddata")):
            return

        # Common locations to search
        candidates = [
            "/usr/share/tessdata",
            "/usr/share/tesseract-ocr/5/tessdata",
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/local/share/tessdata",
        ]

        # CRITICAL FIX (2026-08-23): The system tessdata dir may be missing
        # eng.traineddata (only afr/osd installed). We bundle a copy in the
        # project's data/tessdata directory so OCR works out of the box.
        try:
            project_tessdata = Path(__file__).resolve().parent.parent / "data" / "tessdata"
            if project_tessdata.exists():
                candidates.insert(0, str(project_tessdata))
        except Exception:
            pass

        # Also search for any eng.traineddata on the system (bounded).
        try:
            for path in glob.glob("/opt/**/tessdata/eng.traineddata", recursive=True):
                candidates.append(os.path.dirname(path))
            for path in glob.glob("/usr/**/tessdata/eng.traineddata", recursive=True):
                candidates.append(os.path.dirname(path))
        except Exception:
            pass

        for d in candidates:
            if os.path.exists(os.path.join(d, "eng.traineddata")):
                os.environ["TESSDATA_PREFIX"] = d
                logger.info("[OCR] TESSDATA_PREFIX set to %s", d)
                return

        logger.warning("[OCR] eng.traineddata not found — OCR will fail")

    @property
    def ready(self) -> bool:
        return self._active != "none"

    @property
    def active_backend(self) -> str:
        return self._active

    def ocr(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        region_bounds: Optional[Tuple[int, int, int, int]] = None,
        retry_on_failure: bool = True,
    ) -> OCRResult:
        """
        Extract text boxes from an image with post-processing.

        Args:
            image: RGB or grayscale numpy array.
            preprocess: Apply OpenCV preprocessing before OCR.
            region_bounds: (x, y, w, h) to crop before OCR (None = full image).
            retry_on_failure: Retry with different preprocessing if first attempt fails.

        Returns:
            OCRResult with merged, deduped boxes and metadata.
        """
        t0 = time.perf_counter_ns()
        result = OCRResult()

        # Crop to region if specified
        work_image = image
        region_offset = (0, 0)
        if region_bounds is not None:
            x, y, w, h = region_bounds
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                if (y + h) <= image.shape[0] and (x + w) <= image.shape[1]:
                    work_image = image[y:y + h, x:x + w]
                    region_offset = (x, y)

        # ── Raw OCR ──────────────────────────────────
        if preprocess and _HAS_CV2:
            work_image = self._preprocess(work_image)
            result.preprocess_applied = True

        raw_boxes = self._raw_ocr(work_image)
        result.elapsed_raw_ms = (time.perf_counter_ns() - t0) / 1_000_000
        result.box_count_raw = len(raw_boxes)
        result.backend = self._active

        # Offset boxes back to full image coordinates
        for box in raw_boxes:
            ox, oy, ow, oh = box.bbox
            box.bbox = (ox + region_offset[0], oy + region_offset[1], ow, oh)

        # ── Self-healing: retry with different preprocessing ──
        if len(raw_boxes) == 0 and retry_on_failure:
            logger.info("[OCR] No boxes found — retrying with alternative preprocessing")
            self._retry_count += 1
            result.retry_count = 1

            # Retry without preprocessing (raw image)
            raw_boxes = self._raw_ocr(image)
            if len(raw_boxes) == 0 and _HAS_CV2:
                # Retry with inverted colors
                try:
                    inverted = cv2.bitwise_not(image) if image.ndim == 3 else cv2.bitwise_not(
                        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                    raw_boxes = self._raw_ocr(inverted)
                except Exception:
                    pass

            if len(raw_boxes) == 0:
                result.error = "No text detected after retries"
                result.elapsed_total_ms = (time.perf_counter_ns() - t0) / 1_000_000
                logger.warning("[OCR] ERROR: No text found after %d retries", result.retry_count)
                return result

        # ── Post-processing ───────────────────────────
        t_post = time.perf_counter_ns()

        # Filter low confidence
        boxes = [b for b in raw_boxes if b.confidence >= self.MIN_CONFIDENCE]

        # Merge overlapping boxes
        boxes, merge_count = self._merge_boxes(boxes)

        # Remove duplicates
        boxes, dedup_count = self._deduplicate_boxes(boxes)

        # Sort by position (top-to-bottom, left-to-right)
        boxes.sort(key=lambda b: (b.y, b.x))

        # Assign line numbers
        for i, box in enumerate(boxes):
            box.line_number = i

        # Classify text
        for box in boxes:
            box.text_class = self._classify_text(box)

        result.boxes = boxes
        result.box_count_final = len(boxes)
        result.merged_count = merge_count
        result.dedup_count = dedup_count
        result.text = " ".join(b.text for b in boxes) if boxes else ""
        result.elapsed_postprocess_ms = (time.perf_counter_ns() - t_post) / 1_000_000
        result.elapsed_total_ms = (time.perf_counter_ns() - t0) / 1_000_000

        if result.elapsed_total_ms > 200:
            logger.debug("[OCR] Slow: %.1fms total (raw=%.1fms post=%.1fms) for %d boxes",
                         result.elapsed_total_ms, result.elapsed_raw_ms,
                         result.elapsed_postprocess_ms, result.box_count_final)

        logger.info("[OCR] Complete: %d boxes (merged %d, deduped %d) backend=%s avg_conf=%.2f [%.1fms]",
                     result.box_count_final, merge_count, dedup_count,
                     result.backend, result.avg_confidence, result.elapsed_total_ms)
        return result

    def ocr_region(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> OCRResult:
        """Run OCR on a specific region of the image."""
        return self.ocr(image, region_bounds=(x, y, w, h))

    # ── Raw OCR backends ───────────────────────────────

    def _raw_ocr(self, image: np.ndarray) -> List[OCRBox]:
        """Run raw OCR and return boxes (no post-processing)."""
        boxes: List[OCRBox] = []

        if self._active == "paddleocr" and self._paddle is not None:
            boxes = self._ocr_paddle(image)
        elif self._active == "easyocr" and self._easyocr_reader is not None:
            boxes = self._ocr_easyocr(image)
        elif self._active == "tesseract":
            boxes = self._ocr_tesseract(image)

        return boxes

    def _ocr_paddle(self, image: np.ndarray) -> List[OCRBox]:
        try:
            results = self._paddle.ocr(image, cls=false)
            boxes: List[OCRBox] = []
            if results and results[0]:
                for line in results[0]:
                    bbox_points = line[0]
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0

                    if text:
                        xs = [p[0] for p in bbox_points]
                        ys = [p[1] for p in bbox_points]
                        x, y = int(min(xs)), int(min(ys))
                        w, h = int(max(xs) - x), int(max(ys) - y)
                        boxes.append(OCRBox(
                            text=str(text), bbox=(x, y, w, h),
                            confidence=float(conf), backend="paddleocr",
                        ))
            return boxes
        except Exception as e:
            logger.warning("[OCR] PaddleOCR error: %s", e)
            return []

    def _ocr_easyocr(self, image: np.ndarray) -> List[OCRBox]:
        try:
            results = self._easyocr_reader.readtext(image)
            boxes: List[OCRBox] = []
            for bbox_points, text, conf in results:
                if text:
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    boxes.append(OCRBox(
                        text=str(text), bbox=(x, y, w, h),
                        confidence=float(conf), backend="easyocr",
                    ))
            return boxes
        except Exception as e:
            logger.warning("[OCR] EasyOCR error: %s", e)
            return []

    def _ocr_tesseract(self, image: np.ndarray) -> List[OCRBox]:
        try:
            import cv2 as _cv
            if image.ndim == 3:
                bgr = _cv.cvtColor(image, _cv.COLOR_RGB2BGR)
            else:
                bgr = _cv.cvtColor(image, _cv.COLOR_GRAY2BGR)
            data = _pytesseract.image_to_data(bgr, output_type=_pytesseract.Output.DICT)
            boxes: List[OCRBox] = []
            n = len(data["text"])
            for i in range(n):
                text = (data["text"][i] or "").strip()
                conf = int(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.0
                if text:
                    x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                    if w > 0 and h > 0:
                        boxes.append(OCRBox(
                            text=text, bbox=(x, y, w, h),
                            confidence=conf, backend="tesseract",
                        ))
            return boxes
        except Exception as e:
            logger.warning("[OCR] Tesseract error: %s", e)
            return []

    # ── Box merging ───────────────────────────────────

    def _merge_boxes(self, boxes: List[OCRBox]) -> Tuple[List[OCRBox], int]:
        """
        Merge overlapping or vertically-close OCR boxes.

        Fixes cases where OCR engines split single text blocks
        into multiple horizontal boxes on the same line.
        """
        if len(boxes) < 2:
            return boxes, 0

        merged: List[OCRBox] = []
        used: Set[int] = set()
        merge_count = 0

        boxes_sorted = sorted(boxes, key=lambda b: (b.y, b.x))

        for i, box in enumerate(boxes_sorted):
            if i in used:
                continue

            current = box
            merged_group = [box]
            used.add(i)

            # Find all boxes that should merge with this one
            for j in range(i + 1, len(boxes_sorted)):
                if j in used:
                    continue
                other = boxes_sorted[j]

                # Check: vertically close (same horizontal line)
                vert_close = abs(other.y - current.y) < self.MERGE_VERTICAL_CLOSE_PX
                # Check: horizontally overlapping or adjacent
                horiz_overlap = (other.x >= current.x and other.x <= current.right + 20) or \
                                (current.x >= other.x and current.x <= other.right + 20)

                if vert_close and horiz_overlap:
                    merged_group.append(other)
                    used.add(j)
                    # Expand current bounding box
                    new_x = min(current.x, other.x)
                    new_y = min(current.y, other.y)
                    new_r = max(current.right, other.right)
                    new_b = max(current.bottom, other.bottom)
                    current = OCRBox(
                        text="",
                        bbox=(new_x, new_y, new_r - new_x, new_b - new_y),
                        confidence=min(current.confidence, other.confidence),
                        backend=current.backend,
                        is_merged=True,
                        source_boxes=current.source_boxes + other.source_boxes,
                    )

            if len(merged_group) > 1:
                # Concatenate text
                group_text = " ".join(b.text for b in sorted(merged_group, key=lambda b: b.x))
                current.text = group_text
                current.is_merged = True
                current.source_boxes = len(merged_group)
                merge_count += len(merged_group) - 1

            merged.append(current)

        return merged, merge_count

    # ── Deduplication ─────────────────────────────────

    def _deduplicate_boxes(self, boxes: List[OCRBox]) -> Tuple[List[OCRBox], int]:
        """
        Remove duplicate boxes (same text at nearly the same position).

        OCR engines sometimes produce duplicate detections for the
        same text block.
        """
        if len(boxes) < 2:
            return boxes, 0

        deduped: List[OCRBox] = []
        dedup_count = 0

        for box in boxes:
            is_duplicate = False
            for existing in deduped:
                # Same text?
                text_sim = self._text_similarity(box.text, existing.text)
                if text_sim < self.DEDUP_TEXT_SIMILARITY:
                    continue
                # Same position?
                pos_diff = abs(box.x - existing.x) + abs(box.y - existing.y)
                if pos_diff < self.DEDUP_POSITION_CLOSE_PX:
                    is_duplicate = True
                    break

            if is_duplicate:
                dedup_count += 1
            else:
                deduped.append(box)

        return deduped, dedup_count

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple text similarity based on character overlap."""
        a_clean = re.sub(r'\s+', '', a.lower())
        b_clean = re.sub(r'\s+', '', b.lower())
        if not a_clean or not b_clean:
            return 0.0
        common = sum(1 for c in a_clean if c in b_clean)
        return common / max(len(a_clean), len(b_clean))

    # ── Text classification ───────────────────────────

    @staticmethod
    def _classify_text(box: OCRBox) -> str:
        """Classify OCR text into semantic categories."""
        text = box.text.strip()
        text_len = len(text)
        text_lower = text.lower()

        # Short all-caps or title-case = button
        if text_len <= 15 and (text.isupper() or text.istitle()):
            return TextClass.BUTTON_TEXT

        # Common code indicators
        if any(indicator in text for indicator in (
            "def ", "class ", "import ", "from ", "return ", "print(", "function",
            "const ", "let ", "var ", "=>", ".py", ".js", ".ts", ".go", ".rs",
            "Error:", "Traceback", "Exception", "Warning:",
        )):
            return TextClass.CODE

        # Error indicators
        if any(word in text_lower for word in (
            "error", "failed", "failure", "traceback", "exception",
            "cannot", "invalid", "missing", "denied",
        )):
            return TextClass.ERROR

        # Links
        if text.startswith(("http://", "https://", "www.", "ftp://")):
            return TextClass.LINK

        # Menu items (short text at top of screen)
        if text_len <= 20 and box.y < 100:
            return TextClass.MENU_ITEM

        # Status bar text (bottom of screen, short)
        if text_len <= 30 and box.y > 900:
            return TextClass.STATUS

        # Labels (short, ends with colon)
        if text_len <= 20 and text.endswith(":"):
            return TextClass.LABEL

        # Headings (short, all-caps, or large font estimate)
        if text_len <= 40 and (text.isupper() or text.istitle()):
            return TextClass.HEADING

        # Caption (very short, low on screen)
        if text_len <= 20 and box.y > 500:
            return TextClass.CAPTION

        return TextClass.BODY

    # ── Preprocessing ─────────────────────────────────

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """
        Apply OpenCV preprocessing to improve OCR accuracy.

        Steps:
          1. Convert to grayscale
          2. Denoise (fast Non-Local Means)
          3. Adaptive thresholding
          4. Morphological close
          5. Sharpen (unsharp mask)
          6. Convert back to RGB
        """
        if not _HAS_CV2:
            return image

        try:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )

            # Morphological close
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Sharpen
            blur = cv2.GaussianBlur(closed, (0, 0), 3)
            sharpened = cv2.addWeighted(closed, 1.5, blur, -0.5, 0)

            # Back to RGB
            result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            return result
        except Exception as e:
            logger.debug("[OCR] Preprocessing failed: %s — returning original", e)
            return image

    # ── Diagnostics ────────────────────────────────────

    def close(self) -> None:
        self._paddle = None
        self._easyocr_reader = None
        self._active = "none"

    def report(self) -> Dict[str, Any]:
        return {
            "backend": self._active,
            "ready": self.ready,
            "retry_count": self._retry_count,
        }


# Global singleton
ocr_pipeline = EnhancedOCREngine()