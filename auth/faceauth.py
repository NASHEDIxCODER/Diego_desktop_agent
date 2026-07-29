"""
Face Authentication for Leo Desktop Assistant.

Fully local authentication. Firebase is ONLY for optional sync.
Authentication works with no internet, no Firebase credentials.

Pipeline:
  Camera → Face detection → 128D embedding → Compare → Decision

Features:
- Multiple embeddings per user (different lighting/angles)
- Configurable tolerance from .env (FACE_TOLERANCE)
- Automatic format migration
- Detailed logging of every stage
- 3-second timeout
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2 as cv
import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
CAM_INDEX = 0
BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_PATH = BASE_DIR / "Known_encodings.p"
IMAGES_DIR = BASE_DIR / "images"

# Default tolerance — overridden by .env
FACE_TOLERANCE = float(os.environ.get("FACE_TOLERANCE", "0.55"))

# ── Cached state ───────────────────────────────────────────────
_camera: Optional[cv.VideoCapture] = None
_known_encodings: List[np.ndarray] = []  # Flat list of all encodings
_known_names: List[str] = []             # Corresponding names (one per encoding)
_encodings_loaded = False
_encodings_mtime: float = 0


def _get_camera() -> Optional[cv.VideoCapture]:
    """Get or create the camera instance. Reuses existing instance."""
    global _camera
    if _camera is not None:
        ret, _ = _camera.read()
        if ret:
            _camera.set(cv.CAP_PROP_POS_FRAMES, 0)
            return _camera
        _release_camera()

    try:
        _camera = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)
        if not _camera.isOpened():
            _camera = cv.VideoCapture(CAM_INDEX)
        if _camera.isOpened():
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


def _validate_encodings(data) -> bool:
    """
    Validate that pickle data contains valid face encodings.

    Supports formats:
    - [encodings_list, names_list]  (list of lists)
    - (encodings_list, names_list)  (tuple of lists)
    - {name: [encoding, ...]}       (dict with multiple per user)

    Returns True if valid.
    """
    if isinstance(data, dict):
        # Dict format: {name: [encoding_ndarray, ...]}
        if len(data) == 0:
            logger.warning("Encodings dict is empty")
            return False
        for name, encs in data.items():
            if not isinstance(encs, list) or len(encs) == 0:
                logger.warning("Invalid encoding entry for '%s'", name)
                return False
            if not hasattr(encs[0], 'shape') or encs[0].shape != (128,):
                logger.warning("Invalid encoding shape for '%s': %s", name, encs[0].shape if hasattr(encs[0], 'shape') else 'N/A')
                return False
        return True

    if isinstance(data, (tuple, list)) and len(data) == 2:
        encs, names = data
        if not isinstance(encs, list) or not isinstance(names, list):
            return False
        if len(encs) != len(names):
            logger.warning("Encoding/name count mismatch: %d vs %d", len(encs), len(names))
            return False
        if len(encs) > 0 and (not hasattr(encs[0], 'shape') or encs[0].shape != (128,)):
            logger.warning("Invalid encoding shape: %s", encs[0].shape if hasattr(encs[0], 'shape') else 'N/A')
            return False
        return True

    return False


def _migrate_to_flat(data) -> Tuple[List[np.ndarray], List[str]]:
    """
    Migrate any supported format to flat lists.

    Returns (encodings_flat, names_flat).
    """
    encodings_flat: List[np.ndarray] = []
    names_flat: List[str] = []

    if isinstance(data, dict):
        # Dict format: {name: [encoding, ...]}
        for name, encs in data.items():
            for enc in encs:
                encodings_flat.append(enc)
                names_flat.append(name)
        logger.info("Migrated dict format: %d users, %d samples", len(data), len(encodings_flat))
        return encodings_flat, names_flat

    if isinstance(data, (tuple, list)) and len(data) == 2:
        encs, names = data
        # If encodings is a list of arrays (old format), use as-is
        if isinstance(encs, list) and len(encs) > 0 and hasattr(encs[0], 'shape'):
            return list(encs), list(names)
        # If encodings is a numpy array, convert
        if isinstance(encs, np.ndarray):
            for i in range(len(encs)):
                encodings_flat.append(encs[i])
                names_flat.append(names[i] if i < len(names) else f"user_{i}")
            return encodings_flat, names_flat

    return encodings_flat, names_flat


def _load_encodings(force: bool = False) -> bool:
    """
    Load face encodings from disk. Caches in memory.
    Validates format and migrates if needed.

    Returns True if encodings are available.
    """
    global _known_encodings, _known_names, _encodings_loaded, _encodings_mtime

    if not force and _encodings_loaded:
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
    except Exception as e:
        logger.warning("Failed to read encodings file: %s", e)
        return False

    # Validate format
    if not _validate_encodings(data):
        logger.warning(
            "Invalid encodings format in %s (type=%s). Rebuild with encode.py.",
            ENCODINGS_PATH, type(data).__name__
        )
        return False

    # Migrate to flat format
    _known_encodings, _known_names = _migrate_to_flat(data)

    _encodings_loaded = True
    _encodings_mtime = ENCODINGS_PATH.stat().st_mtime

    # Log summary
    unique_users = len(set(_known_names))
    logger.info(
        "Loaded %d face encodings (%d users, tolerance=%.2f): %s",
        len(_known_encodings), unique_users, FACE_TOLERANCE,
        sorted(set(_known_names))
    )
    return True


def recognize_faces() -> Optional[str]:
    """
    Recognize a face from the camera. Maximum 3 second timeout.

    Pipeline:
      1. Load encodings
      2. Capture frame
      3. Detect face locations
      4. Generate 128D embedding
      5. Compare against all known encodings
      6. Return best match if within tolerance

    Returns:
        Name of recognized person, or None if not recognized.
    """
    t0 = time.time()

    # Step 1: Load encodings
    if not _load_encodings():
        logger.warning("Face auth: no encodings loaded")
        return None

    # Step 2: Get camera
    cam = _get_camera()
    if cam is None:
        logger.warning("Face auth: camera not available")
        return None

    # Step 3: Capture frame
    ret, frame = cam.read()
    if not ret or frame is None:
        logger.warning("Face auth: empty frame from camera")
        return None

    # Check timeout
    if time.time() - t0 > 3.0:
        logger.warning("Face auth: timed out after 3s")
        return None

    # Step 4: Downscale and convert to RGB
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

    # Step 5: Detect face locations
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    if not face_locations:
        elapsed = time.time() - t0
        logger.info("Face auth: no face detected (%.2fs)", elapsed)
        return None

    # Step 6: Generate 128D embeddings
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if not _known_encodings:
        logger.warning("Face auth: no known encodings to compare against")
        return None

    # Step 7: Compare against all known encodings
    tolerance = FACE_TOLERANCE
    best_overall_name = None
    best_overall_distance = float('inf')

    for encode_face in face_encodings:
        distances = face_recognition.face_distance(_known_encodings, encode_face)
        if len(distances) == 0:
            continue

        best_match_idx = int(np.argmin(distances))
        best_distance = float(distances[best_match_idx])

        if best_distance < best_overall_distance:
            best_overall_distance = best_distance
            best_overall_name = _known_names[best_match_idx]

    # Step 8: Decision
    elapsed = time.time() - t0
    if best_overall_name and best_overall_distance < tolerance:
        logger.info(
            "Face auth: %s (dist=%.4f, tolerance=%.2f, time=%.2fs)",
            best_overall_name, best_overall_distance, tolerance, elapsed
        )
        return best_overall_name

    logger.info(
        "Face auth: unknown face (best_dist=%.4f, tolerance=%.2f, time=%.2fs)",
        best_overall_distance if best_overall_distance < float('inf') else -1,
        tolerance, elapsed
    )
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

    ret, frame = cam.read()
    if not ret or frame is None:
        logger.warning("Empty frame for enrollment")
        return False

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

    # Re-encode all faces (local only, Firebase is optional)
    try:
        from auth.encode import encode_and_upload_faces
        encode_and_upload_faces()
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