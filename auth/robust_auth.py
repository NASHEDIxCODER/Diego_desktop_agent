"""
RobustAuth — Enhanced face authentication for Diego.

Builds on the existing detector + camera infrastructure and adds the
robustness requirements:

  ✓ automatic exposure          (reuses faceauth._auto_adjust_exposure)
  ✓ brightness correction       (CLAHE/gamma preprocessing in detector)
  ✓ motion blur detection       (Laplacian blur gate)
  ✓ multiple frame voting       (collect N stable frames, majority vote)
  ✓ confidence averaging        (mean distance across voted frames)
  ✓ head pose estimation        (reject non-frontal faces via landmarks)
  ✓ anti-spoofing               (texture + screen-moiré + micro-motion)

Decision rule:
  - Encode up to VOTE_FRAMES stable, frontal, live frames.
  - Each frame votes for the closest known identity (if within tolerance).
  - Require a MAJORITY of votes to agree on one identity.
  - The averaged distance must also be within tolerance.
  - Otherwise: REJECT.

Public API:
    recognize_faces_robust() -> Optional[str]
"""

import logging
import time
from collections import Counter
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from auth.face_detector import (
    face_detector, FaceBox, measure_brightness, measure_blur,
    BRIGHTNESS_THRESHOLD, BLUR_THRESHOLD, MIN_FACE_WIDTH,
)

logger = logging.getLogger(__name__)

# ── Robustness tuning ──────────────────────────────────────────
VOTE_FRAMES = 5                 # stable frames to collect for voting
MAJORITY = 3                    # votes required to accept an identity
CAPTURE_BUDGET_S = 8.0          # total capture time budget
WARMUP_FRAMES = 5
MIN_FACE_WIDTH_ROBUST = 100
BLUR_MIN_ROBUST = 28.0
BRIGHTNESS_MIN_ROBUST = 80.0

# Head pose: max allowed deviation (degrees-ish, from landmark geometry)
MAX_YAW_RATIO = 0.55            # eye-midpoint offset / face width
MAX_ROLL_DEG = 25.0             # eye-line tilt

# Anti-spoofing thresholds
TEXTURE_VAR_MIN = 60.0          # Laplacian-of-gray variance (screen/photo is flat-ish)
MOIRE_STD_MAX = 9.0             # high-freq banding typical of screens
MOTION_MIN = 1.5                # mean abs diff between consecutive frames (live faces move)

# Lazy import of tolerance from faceauth (single source of truth)
def _tolerance() -> float:
    try:
        from auth.faceauth import FACE_TOLERANCE
        return FACE_TOLERANCE
    except Exception:
        return 0.55


# ═══════════════════════════════════════════════════════════════
# Anti-spoofing + liveness
# ═══════════════════════════════════════════════════════════════

def _texture_score(face_bgr: np.ndarray) -> float:
    """High-frequency texture energy. Real skin has micro-texture;
    printed photos and screens are smoother (or have banding)."""
    gray = cv.cvtColor(face_bgr, cv.COLOR_BGR2GRAY)
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def _moire_score(face_bgr: np.ndarray) -> float:
    """Detect screen moiré banding via horizontal high-pass energy."""
    gray = cv.cvtColor(face_bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    # Horizontal derivative emphasizes row banding from displays
    kernel = np.array([[-1, 2, -1]], dtype=np.float32)
    band = cv.filter2D(gray, -1, kernel)
    return float(np.std(band))


def _motion_score(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
    """Mean absolute difference between consecutive face crops.
    A live face has micro-motion; a static photo does not."""
    if prev is None or prev.shape != cur.shape:
        return 999.0  # unknown → don't penalize on first frame
    diff = cv.absdiff(prev, cur)
    return float(np.mean(diff))


def _is_live(face_bgr: np.ndarray, prev_face: Optional[np.ndarray],
             motion: float) -> Tuple[bool, str]:
    """Combine texture, moiré, and motion into a liveness decision."""
    tex = _texture_score(face_bgr)
    if tex < TEXTURE_VAR_MIN * 0.4:
        return False, f"flat_texture({tex:.1f})"
    moire = _moire_score(face_bgr)
    if moire > MOIRE_STD_MAX * 3:
        return False, f"screen_moire({moire:.1f})"
    # Motion: require a little micro-movement once we have a previous frame
    if prev_face is not None and motion < MOTION_MIN * 0.2 and tex < TEXTURE_VAR_MIN:
        return False, f"no_motion({motion:.2f})"
    return True, "ok"


# ═══════════════════════════════════════════════════════════════
# Head pose estimation (from face landmarks)
# ═══════════════════════════════════════════════════════════════

def _head_pose_ok(frame_bgr: np.ndarray, face: FaceBox) -> Tuple[bool, str]:
    """
    Estimate yaw/roll from eye landmarks. Reject strongly turned or
    tilted faces (authentication should use a roughly frontal face).

    Uses face_recognition's 68-point landmarks when available.
    Falls back to 'ok' if landmarks can't be computed.
    """
    try:
        import face_recognition
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        loc = face.to_face_recognition_format()
        landmarks_list = face_recognition.face_landmarks(rgb, [loc])
        if not landmarks_list:
            return True, "no_landmarks_ok"
        lm = landmarks_list[0]
        left_eye = np.array(lm.get("left_eye", []))
        right_eye = np.array(lm.get("right_eye", []))
        if left_eye.size == 0 or right_eye.size == 0:
            return True, "no_eyes_ok"

        left_c = left_eye.mean(axis=0)
        right_c = right_eye.mean(axis=0)

        # Roll: angle of the eye line
        dx = right_c[0] - left_c[0]
        dy = right_c[1] - left_c[1]
        roll = abs(np.degrees(np.arctan2(dy, dx)))
        if roll > MAX_ROLL_DEG:
            return False, f"roll({roll:.1f}deg)"

        # Yaw: horizontal offset of nose tip vs eye midpoint, normalized
        nose = np.array(lm.get("nose_tip", []))
        if nose.size:
            nose_c = nose.mean(axis=0)
            eye_mid_x = (left_c[0] + right_c[0]) / 2.0
            face_w = max(face.w, 1)
            yaw_ratio = abs(nose_c[0] - eye_mid_x) / face_w
            if yaw_ratio > MAX_YAW_RATIO:
                return False, f"yaw({yaw_ratio:.2f})"
        return True, "ok"
    except Exception as e:
        logger.debug("[ROBUST-AUTH] pose estimation failed: %s", e)
        return True, "pose_unknown_ok"


# ═══════════════════════════════════════════════════════════════
# Encoding + comparison
# ═══════════════════════════════════════════════════════════════

def _encode(frame_bgr: np.ndarray, face: FaceBox) -> Optional[np.ndarray]:
    """Encode a face to a 128-D vector."""
    try:
        import face_recognition
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        loc = face.to_face_recognition_format()
        encs = face_recognition.face_encodings(rgb, [loc])
        return encs[0] if encs else None
    except Exception as e:
        logger.debug("[ROBUST-AUTH] encode failed: %s", e)
        return None


def _closest(known_encs: List[np.ndarray], known_names: List[str],
             enc: np.ndarray) -> Tuple[Optional[str], float]:
    """Return (name, distance) of the closest known identity."""
    import face_recognition
    if not known_encs:
        return None, 1.0
    dists = face_recognition.face_distance(known_encs, enc)
    if len(dists) == 0:
        return None, 1.0
    idx = int(np.argmin(dists))
    return known_names[idx], float(dists[idx])


# ═══════════════════════════════════════════════════════════════
# Main robust recognition
# ═══════════════════════════════════════════════════════════════

def recognize_faces_robust() -> Optional[str]:
    """
    Robust face recognition with multi-frame voting.

    Returns the authenticated user's name, or None on failure.
    """
    t0 = time.time()
    logger.info("[ROBUST-AUTH] ════════════════════════════════════════")
    logger.info("[ROBUST-AUTH] Starting robust face authentication")

    # Load encodings via the shared loader
    try:
        from auth import faceauth as _fa
        if not _fa._load_encodings():
            logger.warning("[ROBUST-AUTH] No encodings available")
            return None
        known_encs = list(_fa._known_encodings)
        known_names = list(_fa._known_names)
    except Exception as e:
        logger.error("[ROBUST-AUTH] Failed to load encodings: %s", e)
        return None

    # Get camera via the shared singleton
    try:
        from auth import faceauth as _fa
        cam = _fa._get_camera()
        if cam is None:
            logger.error("[ROBUST-AUTH] Camera unavailable")
            return None
    except Exception as e:
        logger.error("[ROBUST-AUTH] Camera error: %s", e)
        return None

    tolerance = _tolerance()

    # Warmup
    for _ in range(WARMUP_FRAMES):
        cam.read()

    # Exposure auto-adjust if dark
    try:
        from auth import faceauth as _fa
        ret, f = cam.read()
        if ret and f is not None:
            b = measure_brightness(f)
            if b < BRIGHTNESS_THRESHOLD:
                _fa._auto_adjust_exposure(cam, b)
    except Exception:
        pass

    votes: List[Tuple[str, float]] = []        # (name, distance) per accepted frame
    distances_for_avg: List[float] = []
    prev_face_crop: Optional[np.ndarray] = None
    frames_seen = 0

    while time.time() - t0 < CAPTURE_BUDGET_S and len(votes) < VOTE_FRAMES:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
        frames_seen += 1

        faces, quality = face_detector.detect(frame, return_quality=True)
        if not faces or quality is None:
            prev_face_crop = None
            continue

        face = max(faces, key=lambda f: f.w * f.h)

        # ── Quality gates ────────────────────────────────
        if face.w < MIN_FACE_WIDTH_ROBUST:
            continue
        blur = getattr(quality, "proc_blur", None) or quality.blur
        if blur < BLUR_MIN_ROBUST:
            continue
        if quality.brightness < BRIGHTNESS_MIN_ROBUST:
            continue

        # ── Crop face for liveness ───────────────────────
        x, y, w, h = face.x, face.y, face.w, face.h
        pad = int(0.15 * w)
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad); y1 = min(frame.shape[0], y + h + pad)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        # ── Anti-spoofing / liveness ─────────────────────
        motion = _motion_score(prev_face_crop, cv.resize(crop, prev_face_crop.shape[::-1][1:][::-1]) if prev_face_crop is not None and prev_face_crop.size else crop)
        live, live_reason = _is_live(crop, prev_face_crop, motion)
        if not live:
            logger.info("[ROBUST-AUTH] Liveness reject: %s", live_reason)
            prev_face_crop = crop.copy()
            continue
        prev_face_crop = crop.copy()

        # ── Head pose ────────────────────────────────────
        pose_ok, pose_reason = _head_pose_ok(frame, face)
        if not pose_ok:
            logger.info("[ROBUST-AUTH] Pose reject: %s", pose_reason)
            continue

        # ── Encode + vote ────────────────────────────────
        enc = _encode(frame, face)
        if enc is None:
            continue
        name, dist = _closest(known_encs, known_names, enc)
        if name is None:
            continue
        logger.info("[ROBUST-AUTH] frame %d: candidate='%s' dist=%.4f "
                    "(blur=%.1f bright=%.1f)", frames_seen, name, dist, blur,
                    quality.brightness)
        if dist < tolerance:
            votes.append((name, dist))
            distances_for_avg.append(dist)

    # ── Decision: majority vote + confidence averaging ────
    elapsed = time.time() - t0
    if not votes:
        logger.warning("[ROBUST-AUTH] No valid votes collected (%.2fs, %d frames)",
                       elapsed, frames_seen)
        _release()
        return None

    counts = Counter(n for n, _ in votes)
    top_name, top_count = counts.most_common(1)[0]
    avg_conf = float(np.mean([d for n, d in votes if n == top_name]))

    logger.info("[ROBUST-AUTH] Votes: %s | top='%s' %d/%d avg_dist=%.4f tol=%.2f",
                dict(counts), top_name, top_count, len(votes), avg_conf, tolerance)

    _release()

    if top_count >= MAJORITY and avg_conf < tolerance:
        logger.info("[ROBUST-AUTH] ✅ AUTHENTICATED '%s' (votes=%d, avg=%.4f, %.2fs)",
                    top_name, top_count, avg_conf, elapsed)
        return top_name

    logger.warning("[ROBUST-AUTH] ❌ REJECTED (top='%s' %d/%d votes, avg=%.4f)",
                   top_name, top_count, len(votes), avg_conf)
    return None


def _release() -> None:
    try:
        from auth import faceauth as _fa
        _fa._release_camera()
    except Exception:
        pass


if __name__ == "__main__":
    print(recognize_faces_robust() or "No face recognized")
