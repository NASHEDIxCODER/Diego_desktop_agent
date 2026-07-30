"""
Face Authentication for Leo Desktop Assistant.

Fully local authentication. Firebase is ONLY for optional sync.
Authentication works with no internet, no Firebase credentials.

Pipeline:
  Camera → Continuous capture → Frame preprocessing → Face detection
  → Face tracking (5 stable frames) → 128D embedding → Compare → Decision

ROOT CAUSE FIX:
  The original implementation used only HOG face detection on a single
  dark frame (brightness=64). HOG and CNN both returned 0 faces.
  The fix uses a multi-backend detector (YuNet + HOG + CNN) with
  automatic frame preprocessing (CLAHE + gamma + auto-contrast).

Features:
  - Multi-backend face detection (YuNet → HOG → CNN)
  - Automatic frame preprocessing (CLAHE, gamma, auto-contrast)
  - Continuous capture at 30 FPS (max 5 seconds)
  - Face tracking (5 stable frames before encoding)
  - Quality gates (face width > 120px, blur > 35, brightness > 90)
  - Camera improvements (V4L2, 640x480, 30 FPS, MJPEG)
  - Debug overlay (press D to toggle)
  - Comprehensive logging at every stage
  - No silent failures
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

from auth.face_detector import (
    FaceDetector, FaceBox, FrameQuality,
    measure_brightness, measure_blur, measure_quality,
    preprocess_frame, face_detector,
    BRIGHTNESS_THRESHOLD, BLUR_THRESHOLD, MIN_FACE_WIDTH, TRACKING_STABLE_FRAMES,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
CAM_INDEX = 0
BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_PATH = BASE_DIR / "Known_encodings.p"
IMAGES_DIR = BASE_DIR / "images"
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"

# Default tolerance — overridden by .env
FACE_TOLERANCE = float(os.environ.get("FACE_TOLERANCE", "0.55"))

# ── Camera configuration ───────────────────────────────────────
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
MAX_CAPTURE_DURATION = 5.0  # seconds
WARMUP_FRAMES = 5

# ── Tracking quality gates ─────────────────────────────────────
TRACKING_MIN_BLUR = 35.0
TRACKING_MIN_BRIGHTNESS = 90.0

# ── Debug overlay ──────────────────────────────────────────────
_debug_overlay_enabled = False

# ── CameraManager Singleton ────────────────────────────────────
_camera: Optional[cv.VideoCapture] = None
_camera_refcount = 0


def _get_camera() -> Optional[cv.VideoCapture]:
    """
    Get or create the camera instance (singleton).

    Uses V4L2 backend on Linux with optimized settings:
      - 640x480 resolution
      - 30 FPS
      - MJPEG codec
      - Buffer size 1 (lowest latency)
      - Manual autofocus if supported
    """
    global _camera, _camera_refcount

    if _camera is not None:
        try:
            ret, _ = _camera.read()
            if ret:
                _camera_refcount += 1
                logger.debug("[CAMERA] Reusing existing instance (refcount=%d)", _camera_refcount)
                return _camera
            else:
                logger.warning("[CAMERA] Existing camera returned no frame — reinitializing")
                _release_camera()
        except Exception as e:
            logger.warning("[CAMERA] Existing camera error: %s — reinitializing", e)
            _release_camera()

    try:
        logger.info("[CAMERA] Opening /dev/video%d with V4L2...", CAM_INDEX)
        _camera = cv.VideoCapture(CAM_INDEX, cv.CAP_V4L2)

        if not _camera.isOpened():
            logger.warning("[CAMERA] CAP_V4L2 failed, trying default backend")
            _camera = cv.VideoCapture(CAM_INDEX)

        if not _camera.isOpened():
            logger.error("[CAMERA] Failed to open /dev/video%d with any backend", CAM_INDEX)
            _camera = None
            return None

        # Configure camera
        _camera.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        _camera.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        _camera.set(cv.CAP_PROP_FPS, CAM_FPS)
        _camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

        # Try MJPEG codec (reduces bandwidth, improves FPS)
        _camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

        # Try manual autofocus (not all cameras support this)
        try:
            _camera.set(cv.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        except Exception:
            pass

        actual_w = int(_camera.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_h = int(_camera.get(cv.CAP_PROP_FRAME_HEIGHT))
        actual_fps = _camera.get(cv.CAP_PROP_FPS)
        backend = _camera.get(cv.CAP_PROP_BACKEND)

        logger.info("[CAMERA] Initialized: %dx%d @ %.1f FPS (backend=%d, MJPEG=%s)",
                    actual_w, actual_h, actual_fps, int(backend),
                    _camera.get(cv.CAP_PROP_FOURCC) == cv.VideoWriter_fourcc(*'MJPG'))

        # Set detector input size
        face_detector.set_input_size(actual_w, actual_h)

        _camera_refcount = 1
        return _camera

    except Exception as e:
        logger.error("[CAMERA] Camera init exception: %s", e, exc_info=True)
        _camera = None
        return None


def _release_camera():
    """Release the camera instance."""
    global _camera, _camera_refcount
    if _camera is not None:
        _camera_refcount -= 1
        logger.debug("[CAMERA] Release request (refcount=%d)", _camera_refcount)
        if _camera_refcount <= 0:
            try:
                _camera.release()
                logger.info("[CAMERA] Released")
            except Exception as e:
                logger.warning("[CAMERA] Release error: %s", e)
            _camera = None
            _camera_refcount = 0


def _validate_encodings(data) -> bool:
    """Validate that pickle data contains valid face encodings."""
    if isinstance(data, dict):
        if len(data) == 0:
            return False
        for name, encs in data.items():
            if not isinstance(encs, list) or len(encs) == 0:
                return False
            if not hasattr(encs[0], 'shape') or encs[0].shape != (128,):
                return False
        return True

    if isinstance(data, (tuple, list)) and len(data) == 2:
        encs, names = data
        if not isinstance(encs, list) or not isinstance(names, list):
            return False
        if len(encs) != len(names):
            return False
        if len(encs) > 0 and (not hasattr(encs[0], 'shape') or encs[0].shape != (128,)):
            return False
        return True

    return False


def _migrate_to_flat(data) -> Tuple[List[np.ndarray], List[str]]:
    """Migrate any supported format to flat lists."""
    encodings_flat: List[np.ndarray] = []
    names_flat: List[str] = []

    if isinstance(data, dict):
        for name, encs in data.items():
            for enc in encs:
                encodings_flat.append(enc)
                names_flat.append(name)
        logger.info("Migrated dict format: %d users, %d samples", len(data), len(encodings_flat))
        return encodings_flat, names_flat

    if isinstance(data, (tuple, list)) and len(data) == 2:
        encs, names = data
        if isinstance(encs, list) and len(encs) > 0 and hasattr(encs[0], 'shape'):
            return list(encs), list(names)

    return encodings_flat, names_flat


# ── Cached state ───────────────────────────────────────────────
_known_encodings: List[np.ndarray] = []
_known_names: List[str] = []
_encodings_loaded = False
_encodings_mtime: float = 0


def _load_encodings(force: bool = False) -> bool:
    """Load face encodings from disk. Caches in memory."""
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

    if not _validate_encodings(data):
        logger.warning("Invalid encodings format in %s (type=%s)", ENCODINGS_PATH, type(data).__name__)
        return False

    _known_encodings, _known_names = _migrate_to_flat(data)
    _encodings_loaded = True
    _encodings_mtime = ENCODINGS_PATH.stat().st_mtime

    unique_users = len(set(_known_names))
    logger.info("Loaded %d face encodings (%d users, tolerance=%.2f): %s",
                len(_known_encodings), unique_users, FACE_TOLERANCE,
                sorted(set(_known_names)))
    return True


def _save_debug_frame(frame: np.ndarray, prefix: str = "auth", extra_info: str = ""):
    """Save a debug frame to debug/ directory with metadata."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = DEBUG_DIR / filename
    cv.imwrite(str(filepath), frame)

    blur = measure_blur(frame)
    brightness = measure_brightness(frame)
    meta_path = DEBUG_DIR / f"{prefix}_{timestamp}.txt"
    with open(meta_path, "w") as f:
        f.write(f"time: {timestamp}\n")
        f.write(f"blur_score: {blur:.2f}\n")
        f.write(f"brightness: {brightness:.2f}\n")
        f.write(f"resolution: {frame.shape[1]}x{frame.shape[0]}\n")
        f.write(f"detector_backend: {face_detector.backend}\n")
        if extra_info:
            f.write(f"info: {extra_info}\n")

    logger.debug("[AUTH] Debug frame saved: %s (blur=%.2f, brightness=%.2f)",
                 filepath, blur, brightness)
    return filepath


def _draw_debug_overlay(frame: np.ndarray, faces: List[FaceBox],
                        quality: FrameQuality, fps: float,
                        recognition_result: str = "") -> np.ndarray:
    """Draw debug overlay on frame."""
    overlay = frame.copy()

    # Semi-transparent background for text
    cv.rectangle(overlay, (5, 5), (350, 160), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Info text
    y = 25
    cv.putText(frame, f"FPS: {fps:.1f}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv.putText(frame, f"Brightness: {quality.brightness:.1f}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv.putText(frame, f"Blur: {quality.blur:.2f}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv.putText(frame, f"Backend: {face_detector.backend}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv.putText(frame, f"Faces: {len(faces)}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if recognition_result:
        y += 20
        cv.putText(frame, f"Result: {recognition_result}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw face boxes
    for face in faces:
        color = (0, 255, 0) if face.confidence > 0.7 else (0, 165, 255)
        cv.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2)
        label = f"{face.confidence:.2f} ({face.backend})"
        cv.putText(frame, label, (face.x, face.y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def _is_face_stable(face: FaceBox, quality: FrameQuality) -> bool:
    """Check if a face meets the quality gates for encoding."""
    if face.w < MIN_FACE_WIDTH:
        logger.debug("[TRACKING] Face too small: %dpx < %dpx", face.w, MIN_FACE_WIDTH)
        return False
    if quality.blur < TRACKING_MIN_BLUR:
        logger.debug("[TRACKING] Face too blurry: %.2f < %.2f", quality.blur, TRACKING_MIN_BLUR)
        return False
    if quality.brightness < TRACKING_MIN_BRIGHTNESS:
        logger.debug("[TRACKING] Face too dark: %.1f < %.1f", quality.brightness, TRACKING_MIN_BRIGHTNESS)
        return False
    return True


def recognize_faces() -> Optional[str]:
    """
    Recognize a face from the camera using continuous capture.

    PIPELINE:
      1. Load encodings
      2. Open camera (V4L2, 640x480, 30 FPS, MJPEG)
      3. Warmup (discard 5 frames for auto-exposure)
      4. Continuous capture at 30 FPS (max 5 seconds)
      5. For each frame:
         a. Measure brightness and blur
         b. Preprocess (CLAHE + gamma + auto-contrast if dark)
         c. Detect faces (YuNet → HOG → CNN)
         d. Track face (require 5 stable frames)
         e. Encode face (128D embedding)
         f. Compare against known encodings
         g. Decision

    Returns:
        Name of recognized person, or None if not recognized.
    """
    global _debug_overlay_enabled

    t0 = time.time()
    logger.info("[AUTH] ════════════════════════════════════════════════════")
    logger.info("[AUTH] Face recognition started (backend=%s)", face_detector.backend)

    # Step 1: Load encodings
    t1 = time.time()
    if not _load_encodings():
        logger.warning("[AUTH] No encodings loaded (%.2fs)", time.time() - t1)
        return None
    logger.info("[AUTH] Encodings loaded: %d samples, %d users (%.2fs)",
                len(_known_encodings), len(set(_known_names)), time.time() - t1)

    # Step 2: Get camera
    t2 = time.time()
    cam = _get_camera()
    if cam is None:
        logger.error("[AUTH] Camera not available (%.2fs)", time.time() - t2)
        return None
    logger.info("[AUTH] Camera opened (%.2fs)", time.time() - t2)

    # Step 3: Warmup
    logger.info("[AUTH] Warming up camera (discarding %d frames)...", WARMUP_FRAMES)
    for i in range(WARMUP_FRAMES):
        ret, _ = cam.read()
        if not ret:
            logger.warning("[AUTH] Warmup frame %d failed", i + 1)
    logger.info("[AUTH] Warmup complete (%.2fs)", time.time() - t2)

    # Step 4: Continuous capture
    stable_count = 0
    best_face: Optional[FaceBox] = None
    best_frame: Optional[np.ndarray] = None
    best_quality: Optional[FrameQuality] = None
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0

    logger.info("[AUTH] Continuous capture started (max %.1fs)...", MAX_CAPTURE_DURATION)

    while time.time() - t0 < MAX_CAPTURE_DURATION:
        ret, frame = cam.read()
        if not ret or frame is None:
            logger.warning("[AUTH] Frame %d capture failed", frame_count)
            continue

        frame_count += 1
        fps_counter += 1

        # Calculate FPS
        fps_elapsed = time.time() - fps_start
        if fps_elapsed >= 1.0:
            current_fps = fps_counter / fps_elapsed
            fps_counter = 0
            fps_start = time.time()

        # Measure quality
        quality = measure_quality(frame)

        # Check if frame is too blurry
        if quality.blur < BLUR_THRESHOLD:
            logger.debug("[AUTH] Frame %d too blurry (blur=%.2f < %.2f) — capturing another",
                         frame_count, quality.blur, BLUR_THRESHOLD)
            if _debug_overlay_enabled:
                display = _draw_debug_overlay(frame, [], quality, current_fps, "BLURRY")
                cv.imshow("Leo Face Auth", display)
                if cv.waitKey(1) & 0xFF == ord('d'):
                    _debug_overlay_enabled = not _debug_overlay_enabled
            continue

        # Detect faces
        faces = face_detector.detect(frame)

        # Debug overlay
        if _debug_overlay_enabled:
            display = _draw_debug_overlay(frame, faces, quality, current_fps)
            cv.imshow("Leo Face Auth", display)
            key = cv.waitKey(1) & 0xFF
            if key == ord('d'):
                _debug_overlay_enabled = not _debug_overlay_enabled
            elif key == 27:  # ESC
                break

        if not faces:
            stable_count = 0
            best_face = None
            best_frame = None
            logger.debug("[AUTH] Frame %d: 0 faces (brightness=%.1f, blur=%.2f, fps=%.1f)",
                         frame_count, quality.brightness, quality.blur, current_fps)
            continue

        # Get the largest face
        largest_face = max(faces, key=lambda f: f.w * f.h)

        logger.debug("[AUTH] Frame %d: %d face(s), largest=%dx%d conf=%.3f backend=%s "
                     "(brightness=%.1f, blur=%.2f, fps=%.1f, stable=%d/%d)",
                     frame_count, len(faces), largest_face.w, largest_face.h,
                     largest_face.confidence, largest_face.backend,
                     quality.brightness, quality.blur, current_fps,
                     stable_count, TRACKING_STABLE_FRAMES)

        # Check quality gates
        if not _is_face_stable(largest_face, quality):
            stable_count = 0
            continue

        # Face is stable enough — increment tracking counter
        stable_count += 1
        best_face = largest_face
        best_frame = frame.copy()
        best_quality = quality

        if stable_count < TRACKING_STABLE_FRAMES:
            logger.debug("[AUTH] Tracking: %d/%d stable frames", stable_count, TRACKING_STABLE_FRAMES)
            continue

        # Face has been stable for 5 consecutive frames — encode it
        logger.info("[AUTH] Face stable for %d frames — encoding (backend=%s, %dx%d, conf=%.3f)",
                    stable_count, best_face.backend, best_face.w, best_face.h, best_face.confidence)

        # Save debug frame
        _save_debug_frame(best_frame, "auth_stable",
                          f"stable={stable_count}, backend={best_face.backend}")

        # Step 5: Generate 128D embedding
        t6 = time.time()
        rgb_frame = cv.cvtColor(best_frame, cv.COLOR_BGR2RGB)
        face_loc = best_face.to_face_recognition_format()
        face_encodings = face_recognition.face_encodings(rgb_frame, [face_loc])
        encode_time = time.time() - t6

        logger.info("[AUTH] Face encoding: %d encoding(s) generated (%.2fs)",
                    len(face_encodings), encode_time)

        if not face_encodings:
            logger.warning("[AUTH] No encodings generated despite face detection")
            stable_count = 0
            continue

        if not _known_encodings:
            logger.warning("[AUTH] No known encodings to compare against")
            break

        # Step 6: Compare against all known encodings
        t7 = time.time()
        tolerance = FACE_TOLERANCE
        encode_face = face_encodings[0]
        distances = face_recognition.face_distance(_known_encodings, encode_face)
        face_distances = list(zip(_known_names, distances))

        if len(distances) == 0:
            logger.warning("[AUTH] No distances computed")
            break

        best_match_idx = int(np.argmin(distances))
        best_distance = float(distances[best_match_idx])
        best_name = _known_names[best_match_idx]

        compare_time = time.time() - t7
        logger.info("[AUTH] Comparison: best match='%s' dist=%.4f tolerance=%.2f (%.2fs)",
                    best_name, best_distance, tolerance, compare_time)

        # Log all distances
        unique_names = sorted(set(_known_names))
        for uname in unique_names:
            name_distances = [d for n, d in face_distances if n == uname]
            if name_distances:
                min_dist = min(name_distances)
                avg_dist = sum(name_distances) / len(name_distances)
                logger.info("[AUTH]   vs '%s': min=%.4f, avg=%.4f, samples=%d",
                            uname, min_dist, avg_dist, len(name_distances))

        # Step 7: Decision
        elapsed = time.time() - t0
        if best_distance < tolerance:
            logger.info("[AUTH] ✅ AUTHENTICATED: '%s' (dist=%.4f < tolerance=%.2f, total=%.2fs)",
                        best_name, best_distance, tolerance, elapsed)
            logger.info("[AUTH] ════════════════════════════════════════════════════")

            if _debug_overlay_enabled:
                display = _draw_debug_overlay(best_frame, [best_face], best_quality,
                                               current_fps, f"AUTH: {best_name}")
                cv.imshow("Leo Face Auth", display)
                cv.waitKey(1000)

            _release_camera()
            return best_name

        logger.info("[AUTH] ❌ REJECTED: '%s' (dist=%.4f >= tolerance=%.2f, total=%.2fs)",
                    best_name, best_distance, tolerance, elapsed)

        # Log top 5 closest matches
        face_distances.sort(key=lambda x: x[1])
        logger.info("[AUTH] Top 5 closest matches:")
        for i, (name, dist) in enumerate(face_distances[:5]):
            logger.info("[AUTH]   %d. '%s' (dist=%.4f)", i + 1, name, dist)

        break

    # Timeout or no face detected
    elapsed = time.time() - t0
    if best_face is None:
        logger.info("[AUTH] No face detected within %.1fs (%d frames captured)", elapsed, frame_count)
        if frame is not None:
            _save_debug_frame(frame, "auth_no_face", f"frames={frame_count}, backend={face_detector.backend}")
    else:
        logger.info("[AUTH] Face detected but not recognized within %.1fs", elapsed)

    logger.info("[AUTH] ════════════════════════════════════════════════════")

    if _debug_overlay_enabled:
        cv.destroyAllWindows()

    _release_camera()
    return None


def Unknown_Face() -> bool:
    """Enroll a new face: capture photo, save to images/, re-encode."""
    cam = _get_camera()
    if cam is None:
        logger.warning("Camera not available for enrollment")
        return False

    # Warmup
    for i in range(WARMUP_FRAMES):
        cam.read()

    # Capture frame
    ret, frame = cam.read()
    if not ret or frame is None:
        logger.warning("Empty frame for enrollment")
        _release_camera()
        return False

    # Detect face
    faces = face_detector.detect(frame)
    if not faces:
        logger.warning("No face detected for enrollment")
        _release_camera()
        return False

    # Save the frame
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    img_path = IMAGES_DIR / f"unknown_{timestamp}.jpg"
    cv.imwrite(str(img_path), frame)
    logger.info("Saved face image: %s", img_path)

    _release_camera()

    # Re-encode all faces
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


def toggle_debug_overlay():
    """Toggle the debug overlay on/off."""
    global _debug_overlay_enabled
    _debug_overlay_enabled = not _debug_overlay_enabled
    logger.info("[AUTH] Debug overlay: %s", "ON" if _debug_overlay_enabled else "OFF")


if __name__ == "__main__":
    name = recognize_faces()
    if name:
        print(f"Recognized: {name}")
    else:
        print("No face recognized")
    close()