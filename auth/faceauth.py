"""
Fast Face Authentication for Leo Desktop Assistant.

Optimized for speed:
- Camera is initialized once and reused
- Face encodings are cached in memory
- Model is preloaded at import time
- No blocking OpenCV GUI windows
- No pyttsx3 dependency (uses speech_synthesizer)
- Firebase is optional and lazy-loaded
- Target: authentication under 2 seconds
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional, List, Tuple

import cv2 as cv
import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
CAM_INDEX = 0
BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_PATH = BASE_DIR / "Known_encodings.p"
IMAGES_DIR = BASE_DIR / "images"

# ── Cached state ───────────────────────────────────────────────
_camera: Optional[cv.VideoCapture] = None
_known_encodings: List[np.ndarray] = []
_known_names: List[str] = []
_encodings_loaded = False
_encodings_mtime: float = 0


def _get_camera() -> Optional[cv.VideoCapture]:
    """Get or create the camera instance. Reuses existing instance."""
    global _camera
    if _camera is not None:
        # Check if camera is still alive
        ret, _ = _camera.read()
        if ret:
            _camera.set(cv.CAP_PROP_POS_FRAMES, 0)
            return _camera
        # Camera died, release and recreate
        _release_camera()

    try:
        _camera = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
        if not _camera.isOpened():
            _camera = cv.VideoCapture(CAM_INDEX)
        if _camera.isOpened():
            # Optimize for speed: lower resolution, faster FPS
            _camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            _camera.set(cv.CAP_PROP_FPS, 30)
            _camera.set(cv.CAP_PROP_BUFFERSIZE, 1)
            logger.info("Camera initialized")
            return _camera
    except Exception as e:
        logger.warning("Camera init failed: %s", e)

    _camera = None
    return None


def _release_camera():
    """Release the camera instance."""
    global _camera
    if _camera is not None:
        try:
            _camera.release()
        except Exception:
            pass
        _camera = None


def _load_encodings(force: bool = False) -> bool:
    """Load face encodings from disk. Caches in memory."""
    global _known_encodings, _known_names, _encodings_loaded, _encodings_mtime

    if not force and _encodings_loaded:
        # Check if file has changed on disk
        if ENCODINGS_PATH.exists():
            mtime = ENCODINGS_PATH.stat().st_mtime
            if mtime <= _encodings_mtime:
                return True

    if not ENCODINGS_PATH.exists():
        logger.warning("Encodings file not found: %s", ENCODINGS_PATH)
        return False

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        # Support both tuple and list formats (encode.py saves as list)
        if isinstance(data, (tuple, list)) and len(data) == 2:
            _known_encodings, _known_names = data
        else:
            logger.warning("Unexpected encodings format: type=%s len=%s", type(data).__name__, len(data) if hasattr(data, '__len__') else 'N/A')
            return False

        _encodings_loaded = True
        _encodings_mtime = ENCODINGS_PATH.stat().st_mtime
        logger.info("Loaded %d face encodings", len(_known_encodings))
        return True
    except Exception as e:
        logger.warning("Failed to load encodings: %s", e)
        return False


def recognize_faces() -> Optional[str]:
    """
    Recognize a face from the camera.

    Returns:
        Name of recognized person, or None if not recognized.
    """
    t0 = time.time()

    # Ensure encodings are loaded
    if not _load_encodings():
        logger.warning("No face encodings available")
        return None

    # Get camera
    cam = _get_camera()
    if cam is None:
        logger.warning("Camera not available")
        return None

    # Read a single frame
    ret, frame = cam.read()
    if not ret or frame is None:
        logger.warning("Empty frame from camera")
        return None

    # Downscale for speed (0.5x = 4x faster face detection)
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    if not face_locations:
        logger.debug("No face detected in frame")
        elapsed = time.time() - t0
        logger.info("Face auth: no face (%.2fs)", elapsed)
        return None

    # Encode detected faces
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if not _known_encodings:
        logger.warning("No known encodings to compare against")
        return None

    # Compare against known faces
    for encode_face in face_encodings:
        distances = face_recognition.face_distance(_known_encodings, encode_face)
        if len(distances) == 0:
            continue

        best_match_idx = int(np.argmin(distances))
        best_distance = float(distances[best_match_idx])

        # Threshold for recognition (lower = more strict)
        if best_distance < 0.5:
            name = _known_names[best_match_idx]
            elapsed = time.time() - t0
            logger.info("Face auth: %s (dist=%.3f, %.2fs)", name, best_distance, elapsed)
            return name

    elapsed = time.time() - t0
    logger.info("Face auth: unknown face (%.2fs)", elapsed)
    return None


def Unknown_Face() -> bool:
    """
    Enroll a new face: capture photo, save to images/, re-encode.

    Returns:
        True if enrollment was successful.
    """
    cam = _get_camera()
    if cam is None:
        logger.warning("Camera not available for enrollment")
        return False

    # Read a frame
    ret, frame = cam.read()
    if not ret or frame is None:
        logger.warning("Empty frame for enrollment")
        return False

    # Detect face in frame
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    if not face_locations:
        logger.warning("No face detected for enrollment")
        return False

    # Save the frame
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    img_path = IMAGES_DIR / f"unknown_{timestamp}.jpg"
    cv.imwrite(str(img_path), frame)
    logger.info("Saved face image: %s", img_path)

    # Re-encode all faces
    try:
        from auth.encode import encode_and_upload_faces
        encode_and_upload_faces()
        # Force reload encodings
        _load_encodings(force=True)
        logger.info("Face enrollment complete")
        return True
    except Exception as e:
        logger.warning("Face re-encoding failed: %s", e)
        return False


def close():
    """Release camera resources."""
    _release_camera()
    logger.info("Face auth resources released")


if __name__ == "__main__":
    name = recognize_faces()
    if name:
        print(f"Recognized: {name}")
    else:
        print("No face recognized")
    close()