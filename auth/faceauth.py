"""
Face Authentication for Diego Desktop Assistant.

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
CAM_INDEX = 2
BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_PATH = BASE_DIR / "Known_encodings.p"
IMAGES_DIR = BASE_DIR / "images"
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"

# Default tolerance — overridden by .env
# The face_recognition library's standard tolerance is 0.6. A stricter
# 0.55 rejected legitimate users whose distance sat just above it
# (e.g. dist=0.5578), so the default is aligned with the library standard.
FACE_TOLERANCE = float(os.environ.get("FACE_TOLERANCE", "0.6"))

# ── Camera configuration ───────────────────────────────────────
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
MAX_CAPTURE_DURATION = 5.0  # seconds (legacy single-shot budget)
# Intelligent retry: several capture rounds per authentication session.
AUTH_ROUNDS = 3
ROUND_DURATION = 4.0        # seconds per capture round
WARMUP_FRAMES = 5

# ── Tracking quality gates ─────────────────────────────────────
TRACKING_MIN_BLUR = 35.0
TRACKING_MIN_BRIGHTNESS = 90.0
# Lenient gates used in later retry rounds (rounds are 0-indexed).
RELAXED_MIN_FACE_WIDTH = 80
RELAXED_TRACKING_MIN_BLUR = 25.0
# Total auth budget exposed to main.py for its wait_for timeout.
AUTH_TOTAL_BUDGET = AUTH_ROUNDS * ROUND_DURATION + 6.0

# ── Multi-frame authentication ──────────────────────────────────
# Number of high-quality stable frames to collect for multi-frame recognition.
# Encoding costs ~200ms/frame, so keep this bounded (3-4 frames = ~800ms total).
MAX_ENCODE_FRAMES = 4

# Minimum time gap between collected frames (ms).
# Frames captured within this window are nearly identical (~67ms apart at ~15 FPS).
# Skipping near-duplicate frames gives more diverse poses per encoding cost,
# or fewer encodings for the same diversity.
MIN_FRAME_GAP_MS = 120

# ── Debug overlay ──────────────────────────────────────────────
_debug_overlay_enabled = False


def _overlay_allowed() -> bool:
    """OpenCV GUI calls (imshow / waitKey / destroyAllWindows) may ONLY ever
    execute on the MAIN thread. Face auth normally runs on an executor
    worker thread — calling them there violates Qt/Tcl thread affinity and
    crashes the process. On any non-main thread the overlay is forced OFF."""
    import threading
    return _debug_overlay_enabled and (
        threading.current_thread() is threading.main_thread())


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


def _get_exposure_props(cam) -> Dict[str, Optional[float]]:
    """Read camera exposure-related properties for diagnostics."""
    props: Dict[str, Optional[float]] = {}
    for name, prop in (("exposure", cv.CAP_PROP_EXPOSURE),
                       ("auto_exposure", cv.CAP_PROP_AUTO_EXPOSURE),
                       ("gain", cv.CAP_PROP_GAIN),
                       ("brightness", cv.CAP_PROP_BRIGHTNESS),
                       ("contrast", cv.CAP_PROP_CONTRAST),
                       ("gamma", cv.CAP_PROP_GAMMA),
                       ("backlight", cv.CAP_PROP_BACKLIGHT)):
        try:
            props[name] = round(float(cam.get(prop)), 3)
        except Exception:
            props[name] = None
    return props


def _auto_adjust_exposure(cam, current_brightness: float) -> float:
    """
    If the scene is too dark, brighten it via CAMERA controls (not just
    software preprocessing) BEFORE the capture round starts.

    Returns the measured brightness after adjustment.
    """
    logger.warning(
        "[CAMERA] Scene too dark (brightness=%.1f < %.1f) — auto-adjusting exposure",
        current_brightness, BRIGHTNESS_THRESHOLD)
    before = _get_exposure_props(cam)
    logger.info("[CAMERA] Exposure props BEFORE: %s", before)

    adjusted = False
    # 1) Ensure AUTO exposure is enabled (V4L2: 0.75 = auto, 0.25 = manual).
    try:
        if cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75):
            adjusted = True
    except Exception:
        pass
    # 2) Raise gain / brightness / backlight compensation moderately.
    for prop, values in ((cv.CAP_PROP_BACKLIGHT, (1,)),
                         (cv.CAP_PROP_GAIN, (32, 64, 128)),
                         (cv.CAP_PROP_BRIGHTNESS, (0.6, 0.8, 1.0)),
                         (cv.CAP_PROP_CONTRAST, (0.6, 0.8))):
        for v in values:
            try:
                if cam.set(prop, v):
                    adjusted = True
                    break
            except Exception:
                continue
    # 3) Manual exposure bump when auto-exposure is unavailable.
    try:
        cur = cam.get(cv.CAP_PROP_EXPOSURE)
        if cur is not None and cur > 0:
            for mult in (2.0, 4.0):
                if cam.set(cv.CAP_PROP_EXPOSURE, cur * mult):
                    adjusted = True
                    break
    except Exception:
        pass

    # Let the sensor settle (auto-exposure needs several frames).
    for _ in range(10):
        cam.read()

    after = _get_exposure_props(cam)
    logger.info("[CAMERA] Exposure props AFTER: %s (adjusted=%s)", after, adjusted)

    ret, frame = cam.read()
    if ret and frame is not None:
        new_b = measure_brightness(frame)
        logger.info("[CAMERA] Brightness after exposure adjust: %.1f → %.1f",
                    current_brightness, new_b)
        return new_b
    return current_brightness


def _measure_ambient_brightness(cam, frames: int = 5) -> float:
    """Average brightness over a few frames (camera already warmed up)."""
    vals = []
    for _ in range(frames):
        ret, frame = cam.read()
        if ret and frame is not None:
            vals.append(measure_brightness(frame))
    return float(np.mean(vals)) if vals else 0.0


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


def _is_face_stable(face: FaceBox, quality: FrameQuality,
                    min_width: float = MIN_FACE_WIDTH,
                    min_blur: float = TRACKING_MIN_BLUR) -> Tuple[bool, str]:
    """Check if a face meets the quality gates for encoding.

    Gates on POST-preprocessing brightness (quality.brightness is already
    the enhanced value when preprocessing ran) and processed-frame blur —
    this is what previously failed: gates used RAW brightness, so a dark
    room NEVER produced a stable frame even though detection ran on the
    brightened image.

    Returns (stable, reject_reason).
    """
    if face.w < min_width:
        return False, f"face_too_small({face.w}px<{min_width}px)"
    blur = getattr(quality, "proc_blur", None) or quality.blur
    if blur < min_blur:
        return False, f"too_blurry({blur:.1f}<{min_blur})"
    if quality.brightness < TRACKING_MIN_BRIGHTNESS:
        return False, f"too_dark({quality.brightness:.1f}<{TRACKING_MIN_BRIGHTNESS})"
    return True, "ok"


def _encode_single_frame(frame: np.ndarray, face: FaceBox) -> Optional[np.ndarray]:
    """Encode a single face frame to 128-D embedding. Returns None on failure."""
    try:
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_loc = face.to_face_recognition_format()
        encs = face_recognition.face_encodings(rgb_frame, [face_loc])
        return encs[0] if encs else None
    except Exception as e:
        logger.debug("[AUTH] Encoding failed: %s", e)
        return None


def _aggregate_and_decide(
    frame_face_pairs: List[Tuple[np.ndarray, FaceBox]],
    t0: float,
    max_encode: int = 4,
) -> Optional[str]:
    """
    Multi-frame authentication decision.

    Encodes up to max_encode frames, computes per-frame distances, then
    aggregates using median distance per identity. Requires consistent
    evidence: the winning identity must have median distance < tolerance
    and must be the majority winner across frames.

    This prevents a single anomalous frame from dominating the decision.
    """
    if not frame_face_pairs or not _known_encodings:
        logger.warning("[AUTH] Aggregate: no frames or no known encodings")
        return None

    # Encode frames (bounded)
    tolerance = FACE_TOLERANCE
    embeddings: List[np.ndarray] = []
    encode_times: List[float] = []

    for frame, face in frame_face_pairs[:max_encode]:
        t_enc = time.perf_counter()
        enc = _encode_single_frame(frame, face)
        encode_times.append(time.perf_counter() - t_enc)
        if enc is not None:
            embeddings.append(enc)

    total_encode_ms = sum(encode_times) * 1000 if encode_times else 0.0
    logger.info("[AUTH] Aggregate: %d/%d frames encoded (total %.0fms, avg %.0fms, max %.0fms)",
                len(embeddings), len(frame_face_pairs[:max_encode]),
                total_encode_ms,
                np.mean(encode_times) * 1000 if encode_times else 0.0,
                max(encode_times) * 1000 if encode_times else 0.0)

    if not embeddings:
        logger.warning("[AUTH] Aggregate: no valid embeddings generated")
        return None

    # Per-frame distances: list of (name, distance) lists
    per_frame_results: List[List[Tuple[str, float]]] = []
    for enc in embeddings:
        distances = face_recognition.face_distance(_known_encodings, enc)
        frame_results = list(zip(_known_names, distances))
        per_frame_results.append(frame_results)

    # Aggregate: median distance per identity
    identity_median_dist: Dict[str, float] = {}
    identity_win_count: Dict[str, int] = {}
    for uname in sorted(set(_known_names)):
        all_dists = []
        for frame_results in per_frame_results:
            for name, dist in frame_results:
                if name == uname:
                    all_dists.append(dist)
        if all_dists:
            identity_median_dist[uname] = float(np.median(all_dists))
            identity_win_count[uname] = sum(
                1 for frame_results in per_frame_results
                if min(frame_results, key=lambda x: x[1])[0] == uname
            )

    if not identity_median_dist:
        logger.warning("[AUTH] Aggregate: no distances computed")
        return None

    # Find best identity by median distance
    best_name = min(identity_median_dist, key=identity_median_dist.get)
    best_median = identity_median_dist[best_name]
    best_wins = identity_win_count.get(best_name, 0)

    # Log per-identity aggregates
    for uname in sorted(identity_median_dist.keys()):
        logger.info("[AUTH]   vs '%s': median_dist=%.4f wins=%d/%d",
                    uname, identity_median_dist[uname],
                    identity_win_count.get(uname, 0), len(embeddings))

    elapsed_ms = (time.time() - t0) * 1000

    # Decision: median distance must be within tolerance AND majority of frames agree
    majority_threshold = max(1, len(embeddings) // 2)
    if best_median < tolerance and best_wins > majority_threshold:
        logger.info("[AUTH] ✅ AUTHENTICATED: '%s' (median_dist=%.4f < tol=%.2f, "
                    "wins=%d/%d, total=%.0fms)",
                    best_name, best_median, tolerance, best_wins, len(embeddings), elapsed_ms)
        return best_name

    # Rejection with reason category
    if best_median >= tolerance:
        reject_reason = f"distance_too_high(median={best_median:.4f}>={tolerance})"
    else:
        reject_reason = f"insufficient_consensus(wins={best_wins}<={majority_threshold})"

    logger.info("[AUTH] ❌ REJECTED: '%s' (%s, total=%.0fms)",
                best_name, reject_reason, elapsed_ms)
    return None


def _compare_and_decide(best_frame: np.ndarray, best_face: FaceBox, t0: float) -> Optional[str]:
    """Encode the stable face and compare against known encodings (single-frame legacy)."""
    result = _aggregate_and_decide([(best_frame, best_face)], t0, max_encode=1)
    return result


def recognize_faces() -> Optional[str]:
    """
    Recognize a face from the camera — MANDATORY authentication.

    Session structure (intelligent retry):
      ROUND 0: strict gates (face ≥120px, blur ≥35)
      ROUND 1: re-warm + exposure re-check, strict gates
      ROUND 2: relaxed gates (face ≥80px, blur ≥25)

    Each round:
      - auto-adjusts camera exposure if the scene is too dark
      - captures at up to 30 FPS with CONTINUOUS diagnostics
        (fps / brightness / blur / face confidence / face size / backend /
        camera exposure) logged every 10 frames
      - tracks the largest face for TRACKING_STABLE_FRAMES stable frames
      - encodes + compares on stability, or on the best face seen when
        the round ends with ≥2 stable frames (intelligent retry)
      - saves failed frames automatically for offline analysis

    Returns:
        Name of recognized person, or None if not recognized.
    """
    global _debug_overlay_enabled

    t0 = time.time()
    logger.info("[AUTH] ════════════════════════════════════════════════════")
    logger.info("[AUTH] Face recognition started (backend=%s, rounds=%d × %.1fs)",
                face_detector.backend, AUTH_ROUNDS, ROUND_DURATION)

    # Step 1: Load encodings
    if not _load_encodings():
        logger.warning("[AUTH] No encodings loaded")
        return None
    logger.info("[AUTH] Encodings loaded: %d samples, %d users",
                len(_known_encodings), len(set(_known_names)))

    # Step 2: Get camera
    cam = _get_camera()
    if cam is None:
        logger.error("[AUTH] Camera not available")
        return None

    last_frame: Optional[np.ndarray] = None
    saved_fail_frames = 0
    final_reject_reason = "no_face"

    for round_idx in range(AUTH_ROUNDS):
        relaxed = round_idx >= AUTH_ROUNDS - 1
        min_width = RELAXED_MIN_FACE_WIDTH if relaxed else MIN_FACE_WIDTH
        min_blur = RELAXED_TRACKING_MIN_BLUR if relaxed else TRACKING_MIN_BLUR
        logger.info("[AUTH] ── Round %d/%d (gates: width≥%d, blur≥%.0f) ──",
                    round_idx + 1, AUTH_ROUNDS, min_width, min_blur)

        # Step 3: Warmup (let auto-exposure settle) — every round.
        for i in range(WARMUP_FRAMES):
            ret, f = cam.read()
            if ret and f is not None:
                last_frame = f

        # Step 3b: Exposure auto-adjust BEFORE capture if too dark.
        ambient = _measure_ambient_brightness(cam)
        logger.info("[AUTH] Ambient brightness: %.1f (threshold=%.1f) exposure=%s",
                    ambient, BRIGHTNESS_THRESHOLD, _get_exposure_props(cam))
        if ambient < BRIGHTNESS_THRESHOLD:
            ambient = _auto_adjust_exposure(cam, ambient)

        # Step 4: Continuous capture for this round
        stable_count = 0
        best_face: Optional[FaceBox] = None
        best_frame: Optional[np.ndarray] = None
        best_stable = 0
        # Multi-frame collection: store (frame, face) pairs for aggregation
        stable_frame_pairs: List[Tuple[np.ndarray, FaceBox]] = []
        last_frame_collect_time: float = 0.0  # timestamp of last collected frame
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        current_fps = 0.0
        exposure_props = _get_exposure_props(cam)
        round_start = time.time()

        while time.time() - round_start < ROUND_DURATION:
            ret, frame = cam.read()
            if not ret or frame is None:
                logger.warning("[AUTH] Round %d frame %d capture failed",
                               round_idx + 1, frame_count)
                continue

            frame_count += 1
            fps_counter += 1
            last_frame = frame

            fps_elapsed = time.time() - fps_start
            if fps_elapsed >= 1.0:
                current_fps = fps_counter / fps_elapsed
                fps_counter = 0
                fps_start = time.time()

            # Detect faces — get POST-preprocess quality for gating.
            faces, quality = face_detector.detect(frame, return_quality=True)
            if quality is None:
                continue

            largest_face = max(faces, key=lambda f: f.w * f.h) if faces else None

            # ── CONTINUOUS DIAGNOSTICS (every 10 frames) ──
            if frame_count % 10 == 0:
                if frame_count % 50 == 0:
                    exposure_props = _get_exposure_props(cam)
                logger.info(
                    "[AUTH-DIAG] r%d f%d: fps=%.1f brightness=%.1f blur=%.1f "
                    "proc_blur=%.1f faces=%d conf=%.2f size=%s backend=%s "
                    "exposure=%s stable=%d/%d",
                    round_idx + 1, frame_count, current_fps,
                    quality.brightness, quality.blur,
                    getattr(quality, "proc_blur", quality.blur),
                    len(faces),
                    largest_face.confidence if largest_face else 0.0,
                    f"{largest_face.w}x{largest_face.h}" if largest_face else "0x0",
                    largest_face.backend if largest_face else face_detector.backend,
                    exposure_props, stable_count, TRACKING_STABLE_FRAMES)

            if _overlay_allowed():
                display = _draw_debug_overlay(frame, faces, quality, current_fps)

                cv.imshow("Diego Face Auth", display)
                key = cv.waitKey(1) & 0xFF
                if key == ord('d'):
                    _debug_overlay_enabled = not _debug_overlay_enabled
                elif key == 27:
                    break

            if not faces:
                if stable_count > 0:
                    logger.debug("[AUTH] Lost face after %d stable frames", stable_count)
                stable_count = 0
                continue

            stable, reason = _is_face_stable(largest_face, quality,
                                             min_width=min_width, min_blur=min_blur)
            if not stable:
                final_reject_reason = reason
                stable_count = 0
                # Save the FIRST few gate-rejected frames automatically.
                if saved_fail_frames < 3:
                    saved_fail_frames += 1
                    _save_debug_frame(
                        frame, f"auth_reject_r{round_idx + 1}",
                        f"reason={reason} conf={largest_face.confidence:.2f} "
                        f"size={largest_face.w}x{largest_face.h} "
                        f"brightness={quality.brightness:.1f} blur={quality.blur:.1f}")
                continue

            stable_count += 1
            if stable_count > best_stable:
                best_stable = stable_count
                best_face = largest_face
                best_frame = frame.copy()

            # Collect multi-frame pairs for aggregation (bounded + temporal gap)
            # Skip near-duplicate frames captured within MIN_FRAME_GAP_MS
            now_ts = time.time()
            if (len(stable_frame_pairs) < MAX_ENCODE_FRAMES and
                    (not stable_frame_pairs or
                     (now_ts - last_frame_collect_time) * 1000 >= MIN_FRAME_GAP_MS)):
                stable_frame_pairs.append((frame.copy(), largest_face))
                last_frame_collect_time = now_ts

            if stable_count < TRACKING_STABLE_FRAMES:
                continue

            # ── Face stable for N consecutive frames → multi-frame encode + aggregate ──
            logger.info("[AUTH] Face stable %d frames — multi-frame encoding "
                        "(%d frames collected/%d max, backend=%s, %dx%d, conf=%.3f)",
                        stable_count, len(stable_frame_pairs), MAX_ENCODE_FRAMES,
                        best_face.backend, best_face.w, best_face.h, best_face.confidence)
            _save_debug_frame(best_frame, "auth_stable",
                              f"stable={stable_count}, frames={len(stable_frame_pairs)}, "
                              f"backend={best_face.backend}")
            name = _aggregate_and_decide(stable_frame_pairs, t0, max_encode=MAX_ENCODE_FRAMES)
            if name:
                if _overlay_allowed():
                    cv.destroyAllWindows()
                _release_camera()
                logger.info("[AUTH] ════════════════════════════════════════════════════")
                return name
            # Face recognized as someone unknown — no point retrying rounds.

            logger.info("[AUTH] Face found but NOT a registered user — ending session")
            final_reject_reason = "unknown_face"
            if best_frame is not None:
                _save_debug_frame(best_frame, "auth_unknown_face",
                                  f"backend={best_face.backend}")
            if _overlay_allowed():
                cv.destroyAllWindows()
            _release_camera()
            logger.info("[AUTH] ════════════════════════════════════════════════════")
            return None


        # ── Round ended. Intelligent retry: if we had a decent face for
        # ≥2 (non-consecutive-end) stable frames, try multi-frame encoding
        # instead of throwing the round away.
        if best_face is not None and best_frame is not None and best_stable >= 2:
            logger.info("[AUTH] Round %d ended with %d stable frames — "
                        "attempting best-effort multi-frame encoding",
                        round_idx + 1, best_stable)
            # Use collected stable frames if available, otherwise fall back to best frame
            retry_pairs = stable_frame_pairs if stable_frame_pairs else [(best_frame, best_face)]
            name = _aggregate_and_decide(retry_pairs, t0, max_encode=MAX_ENCODE_FRAMES)
            if name:
                if _overlay_allowed():
                    cv.destroyAllWindows()
                _release_camera()
                logger.info("[AUTH] ════════════════════════════════════════════════════")
                return name
            logger.info("[AUTH] Best-effort multi-frame encoding rejected — next round")

        else:
            logger.info("[AUTH] Round %d complete: no stable face "
                        "(%d frames, best_stable=%d)",
                        round_idx + 1, frame_count, best_stable)

    # ── All rounds exhausted ──
    elapsed_ms = (time.time() - t0) * 1000
    logger.warning("[AUTH] No face authenticated within %.0fms (%d rounds, reason=%s)",
                   elapsed_ms, AUTH_ROUNDS, final_reject_reason)
    if last_frame is not None:
        _save_debug_frame(last_frame, "auth_no_face",
                          f"rounds={AUTH_ROUNDS} reason={final_reject_reason} "
                          f"backend={face_detector.backend}")
    logger.info("[AUTH] ════════════════════════════════════════════════════")

    if _overlay_allowed():
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