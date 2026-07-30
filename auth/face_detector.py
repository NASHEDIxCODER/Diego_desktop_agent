"""
FaceDetector — Multi-backend face detection abstraction for Leo.

Detection pipeline (best available backend is auto-selected):
  1. OpenCV YuNet (FaceDetectorYN) — fast, accurate, DNN-based
  2. face_recognition HOG — fallback
  3. face_recognition CNN — slower but more accurate fallback

The detector automatically chooses the best available backend.
Never relies only on HOG.

Frame preprocessing is applied before detection:
  - Brightness measurement
  - Laplacian blur measurement
  - CLAHE (if brightness < 90)
  - Gamma correction (if brightness < 90)
  - Auto contrast (if brightness < 90)

If blur < 25, the frame is rejected and a new one should be captured.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)

# ── YuNet model path ───────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_YUNET_MODEL = _PROJECT_ROOT / "face_detection_yunet_2023mar.onnx"

# ── Quality thresholds ─────────────────────────────────────────
BRIGHTNESS_THRESHOLD = 90.0
BLUR_THRESHOLD = 25.0
MIN_FACE_WIDTH = 120  # pixels
TRACKING_STABLE_FRAMES = 5


class FaceBox:
    """Represents a detected face with bounding box and metadata."""

    def __init__(self, x: int, y: int, w: int, h: int,
                 confidence: float = 1.0, backend: str = "unknown"):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.backend = backend

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def left(self) -> int:
        return self.x

    def to_face_recognition_format(self) -> Tuple[int, int, int, int]:
        """Convert to (top, right, bottom, left) format used by face_recognition."""
        return (self.y, self.x + self.w, self.y + self.h, self.x)

    def __repr__(self):
        return f"FaceBox({self.x},{self.y},{self.w}x{self.h}, conf={self.confidence:.3f}, {self.backend})"


class FrameQuality:
    """Represents the quality metrics of a frame."""

    def __init__(self, brightness: float, blur: float):
        self.brightness = brightness
        self.blur = blur

    @property
    def is_acceptable(self) -> bool:
        """Check if frame quality is acceptable for detection."""
        return self.blur >= BLUR_THRESHOLD

    @property
    def needs_preprocessing(self) -> bool:
        """Check if frame needs brightness preprocessing."""
        return self.brightness < BRIGHTNESS_THRESHOLD

    def __repr__(self):
        return f"FrameQuality(brightness={self.brightness:.1f}, blur={self.blur:.2f})"


def measure_brightness(frame: np.ndarray) -> float:
    """Measure mean brightness (0-255)."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def measure_blur(frame: np.ndarray) -> float:
    """Measure Laplacian blur score. Lower = more blurry."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def measure_quality(frame: np.ndarray) -> FrameQuality:
    """Measure both brightness and blur of a frame."""
    return FrameQuality(measure_brightness(frame), measure_blur(frame))


def apply_clahe(frame: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def apply_gamma_correction(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply gamma correction to brighten dark frames."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(frame, table)


def apply_auto_contrast(frame: np.ndarray) -> np.ndarray:
    """Apply automatic contrast enhancement."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculate histogram
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    # Flatten histogram to 1D array
    hist_flat = hist.flatten()
    # Find the 1st and 99th percentile
    total = gray.size
    low = 0
    high = 255
    cumsum = 0
    for i in range(256):
        cumsum += hist_flat[i]
        if cumsum > total * 0.01:
            low = i
            break
    cumsum = 0
    for i in range(255, -1, -1):
        cumsum += hist_flat[i]
        if cumsum > total * 0.01:
            high = i
            break
    if high <= low:
        return frame
    # Stretch contrast
    alpha = 255.0 / (high - low)
    beta = -low * alpha
    return cv.convertScaleAbs(frame, alpha=alpha, beta=beta)


def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, FrameQuality]:
    """
    Preprocess a frame for face detection.

    Steps:
      1. Measure brightness and blur
      2. If brightness < 90: apply CLAHE, gamma correction, auto contrast
      3. Return processed frame and quality metrics

    If blur < 25, the frame is still returned but quality.is_acceptable is False.
    The caller should capture another frame in that case.
    """
    quality = measure_quality(frame)

    processed = frame.copy()

    if quality.needs_preprocessing:
        logger.debug("[PREPROCESS] Brightness=%.1f < %.1f, applying enhancement",
                     quality.brightness, BRIGHTNESS_THRESHOLD)
        processed = apply_clahe(processed)
        processed = apply_gamma_correction(processed, gamma=1.5)
        processed = apply_auto_contrast(processed)

        # Re-measure after preprocessing
        new_brightness = measure_brightness(processed)
        logger.debug("[PREPROCESS] Brightness after enhancement: %.1f → %.1f",
                     quality.brightness, new_brightness)
        quality.brightness = new_brightness

    return processed, quality


class FaceDetector:
    """
    Multi-backend face detector.

    Automatically selects the best available backend:
      1. OpenCV YuNet (FaceDetectorYN) — fast, accurate, DNN-based
      2. face_recognition HOG — fallback
      3. face_recognition CNN — slower but more accurate fallback

    The detector applies frame preprocessing before detection.
    """

    def __init__(self):
        self._yunet_detector = None
        self._yunet_available = False
        self._face_recognition_available = False
        self._backend = "none"
        self._input_size = (640, 480)

        self._init_backends()

    def _init_backends(self):
        """Initialize available detection backends."""
        # Try YuNet first (best option)
        try:
            if _YUNET_MODEL.exists():
                self._yunet_detector = cv.FaceDetectorYN.create(
                    model=str(_YUNET_MODEL),
                    config="",
                    input_size=self._input_size,
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=10,
                )
                self._yunet_available = True
                self._backend = "yunet"
                logger.info("[DETECTOR] YuNet (FaceDetectorYN) initialized — model=%s", _YUNET_MODEL.name)
            else:
                logger.warning("[DETECTOR] YuNet model not found at %s", _YUNET_MODEL)
        except Exception as e:
            logger.warning("[DETECTOR] YuNet init failed: %s", e)

        # Check face_recognition availability
        try:
            import face_recognition
            self._face_recognition_available = True
            if not self._yunet_available:
                self._backend = "hog"
                logger.info("[DETECTOR] face_recognition HOG available (fallback)")
        except ImportError:
            logger.warning("[DETECTOR] face_recognition not available")

        if self._backend == "none":
            logger.error("[DETECTOR] No face detection backend available!")

    @property
    def backend(self) -> str:
        """Get the current detection backend name."""
        return self._backend

    @property
    def is_available(self) -> bool:
        """Check if any detection backend is available."""
        return self._yunet_available or self._face_recognition_available

    def set_input_size(self, width: int, height: int):
        """Set the input size for the YuNet detector."""
        self._input_size = (width, height)
        if self._yunet_detector is not None:
            self._yunet_detector.setInputSize((width, height))

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces in a frame.

        Applies preprocessing first, then runs the best available backend.

        Args:
            frame: BGR frame from camera.

        Returns:
            List of FaceBox objects.
        """
        if not self.is_available:
            logger.warning("[DETECTOR] No backend available")
            return []

        # Preprocess frame
        processed, quality = preprocess_frame(frame)

        if not quality.is_acceptable:
            logger.debug("[DETECTOR] Frame too blurry (blur=%.2f < %.2f) — skipping",
                         quality.blur, BLUR_THRESHOLD)
            return []

        # Try YuNet first
        if self._yunet_available:
            faces = self._detect_yunet(processed)
            if faces:
                return faces
            # If YuNet finds nothing, try HOG as fallback
            logger.debug("[DETECTOR] YuNet found 0 faces, trying HOG fallback")

        # Fallback to face_recognition HOG
        if self._face_recognition_available:
            faces = self._detect_hog(processed)
            if faces:
                return faces

            # Last resort: CNN (slow but more sensitive)
            logger.debug("[DETECTOR] HOG found 0 faces, trying CNN fallback")
            faces = self._detect_cnn(processed)
            return faces

        return []

    def _detect_yunet(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces using OpenCV YuNet (FaceDetectorYN)."""
        h, w = frame.shape[:2]
        if (w, h) != self._input_size:
            self.set_input_size(w, h)

        try:
            t0 = time.time()
            _, faces = self._yunet_detector.detect(frame)
            latency = time.time() - t0

            if faces is None:
                logger.debug("[DETECTOR] YuNet: 0 faces (%.3fs)", latency)
                return []

            result = []
            for face in faces:
                x, y, fw, fh = face[:4].astype(int)
                confidence = float(face[-1])
                result.append(FaceBox(x, y, fw, fh, confidence, "yunet"))

            logger.debug("[DETECTOR] YuNet: %d face(s) (%.3fs)", len(result), latency)
            return result
        except Exception as e:
            logger.error("[DETECTOR] YuNet error: %s", e, exc_info=True)
            return []

    def _detect_hog(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces using face_recognition HOG model."""
        try:
            import face_recognition

            # Use full resolution for HOG (half res misses faces in dark conditions)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            t0 = time.time()
            locations = face_recognition.face_locations(rgb, model="hog")
            latency = time.time() - t0

            result = []
            for (top, right, bottom, left) in locations:
                w = right - left
                h = bottom - top
                result.append(FaceBox(left, top, w, h, 1.0, "hog"))

            logger.debug("[DETECTOR] HOG: %d face(s) (%.3fs)", len(result), latency)
            return result
        except Exception as e:
            logger.error("[DETECTOR] HOG error: %s", e, exc_info=True)
            return []

    def _detect_cnn(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces using face_recognition CNN model (slow but accurate)."""
        try:
            import face_recognition

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            t0 = time.time()
            locations = face_recognition.face_locations(rgb, model="cnn")
            latency = time.time() - t0

            result = []
            for loc in locations:
                if len(loc) == 4:
                    top, right, bottom, left = loc
                    confidence = 1.0
                else:
                    top, right, bottom, left = loc[:4]
                    confidence = float(loc[4]) if len(loc) > 4 else 1.0
                w = right - left
                h = bottom - top
                result.append(FaceBox(left, top, w, h, confidence, "cnn"))

            logger.debug("[DETECTOR] CNN: %d face(s) (%.3fs)", len(result), latency)
            return result
        except Exception as e:
            logger.error("[DETECTOR] CNN error: %s", e, exc_info=True)
            return []


# Global singleton
face_detector = FaceDetector()