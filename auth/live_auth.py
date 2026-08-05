"""
LiveAuth — Face authentication with a live popup UI.

This is the NEW authentication flow. It runs ONLY after the wake word,
NEVER at startup, and NEVER blocks Leo from booting.

NEW BEHAVIOUR (single-capture):
  - Opens the FaceAuthPopup and streams the webcam at ~30 FPS.
  - Shows "Looking for your face..." while waiting.
  - As soon as ONE high-quality face is detected:
      • Face is centered
      • Face is sharp (not blurry)
      • Eyes open
      • Lighting acceptable
      • Face large enough
  - Captures ONE high-quality image.
  - Closes popup IMMEDIATELY.
  - Runs face recognition on the captured image.
  - If match: returns name → conversation continues.
  - If no match: returns None → "Sorry, I couldn't verify you."
  - If no face ever appears: popup stays open forever (no timeout).

Public API:
    authenticate_live(stop_event=None) -> Optional[str]

Designed to be run inside an executor thread (it blocks on the camera).
"""

import logging
import threading
import time
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from auth.face_detector import face_detector, measure_brightness, measure_blur
from auth.face_popup import FaceAuthPopup, FaceStatus

logger = logging.getLogger(__name__)

# ── Quality gates (single best frame) ───────────────────
MIN_FACE_WIDTH = 110           # px — below this: "Move closer"
CENTER_TOL = 0.25              # face center must be within ±25% of frame center
BRIGHTNESS_MIN = 70.0          # below: "Too dark"
BLUR_MIN = 30.0                # below: "Too blurry" (raised for single-capture quality)
MAX_YAW_DEG = 18.0             # tightened for single-capture
MAX_PITCH_DEG = 18.0
MAX_ROLL_DEG = 12.0

# Eye aspect ratio: below this → eyes likely closed
EAR_THRESHOLD = 0.22

# Frame pacing
FRAME_DELAY = 0.033            # ~30 FPS


def _tolerance() -> float:
    try:
        from auth.faceauth import FACE_TOLERANCE
        return FACE_TOLERANCE
    except Exception:
        return 0.55


# ═══════════════════════════════════════════════════════════
# Eye Aspect Ratio (EAR) — detects closed eyes
# ═══════════════════════════════════════════════════════════

def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Compute the eye aspect ratio (EAR). Lower = eyes more closed."""
    if len(eye_points) < 6:
        return 1.0  # can't compute → assume open
    # Vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def _eyes_open(lm: dict) -> Tuple[bool, float]:
    """Check if both eyes are open using EAR. Returns (open, min_ear)."""
    left = lm.get("left_eye", [])
    right = lm.get("right_eye", [])
    if len(left) < 6 or len(right) < 6:
        return True, 1.0  # can't compute → don't block
    left_ear = _eye_aspect_ratio(np.array(left))
    right_ear = _eye_aspect_ratio(np.array(right))
    min_ear = min(left_ear, right_ear)
    return min_ear >= EAR_THRESHOLD, min_ear


# ═══════════════════════════════════════════════════════════
# Head pose from landmarks (yaw / pitch / roll in degrees)
# ═══════════════════════════════════════════════════════════

def _head_pose_deg(frame_bgr: np.ndarray, face) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """
    Estimate (yaw, pitch, roll) in degrees from 68-pt landmarks.
    Returns ((yaw,pitch,roll), reason). reason 'ok' if within limits.
    Falls back to (None, 'ok') if landmarks unavailable (don't block).
    """
    try:
        import face_recognition
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        loc = face.to_face_recognition_format()
        lms = face_recognition.face_landmarks(rgb, [loc])
        if not lms:
            return None, "ok"
        lm = lms[0]
        le = np.array(lm.get("left_eye", []))
        re_ = np.array(lm.get("right_eye", []))
        nose = np.array(lm.get("nose_tip", []))
        chin = np.array(lm.get("chin", []))
        if le.size == 0 or re_.size == 0:
            return None, "ok"

        le_c = le.mean(axis=0)
        re_c = re_.mean(axis=0)

        # Roll: tilt of the eye line
        dx = re_c[0] - le_c[0]
        dy = re_c[1] - le_c[1]
        roll = float(np.degrees(np.arctan2(dy, dx)))

        # Yaw: horizontal offset of nose vs eye midpoint
        yaw = 0.0
        if nose.size:
            nose_c = nose.mean(axis=0)
            eye_mid_x = (le_c[0] + re_c[0]) / 2.0
            inter_eye = max(np.hypot(dx, dy), 1.0)
            yaw = float(np.degrees(np.arcsin(np.clip((nose_c[0]-eye_mid_x)/inter_eye, -1, 1))) * 0.8)

        # Pitch: vertical position of chin vs eye line (rough)
        pitch = 0.0
        if chin.size and nose.size:
            chin_c = chin.mean(axis=0)
            nose_c = nose.mean(axis=0)
            face_h = max(face.h, 1)
            pitch = float(np.degrees(np.arctan2((chin_c[1]-nose_c[1]) - face_h*0.35, face_h)) * 0.5)

        if abs(yaw) > MAX_YAW_DEG:
            return (yaw, pitch, roll), f"turn_too_much(yaw={yaw:.0f})"
        if abs(pitch) > MAX_PITCH_DEG:
            return (yaw, pitch, roll), f"tilt_too_much(pitch={pitch:.0f})"
        if abs(roll) > MAX_ROLL_DEG:
            return (yaw, pitch, roll), f"head_tilted(roll={roll:.0f})"
        return (yaw, pitch, roll), "ok"
    except Exception as e:
        logger.debug("[LIVE-AUTH] pose failed: %s", e)
        return None, "ok"


def _landmark_points(frame_bgr: np.ndarray, face) -> Optional[list]:
    """Flattened 68-pt face landmarks [(x, y), ...] for the popup overlay."""
    try:
        import face_recognition
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        lms = face_recognition.face_landmarks(rgb, [face.to_face_recognition_format()])
        if not lms:
            return None
        pts = []
        for feature_points in lms[0].values():
            pts.extend(feature_points)
        return pts or None
    except Exception:
        return None


def _encode(frame_bgr: np.ndarray, face) -> Optional[np.ndarray]:
    try:
        import face_recognition
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb, [face.to_face_recognition_format()])
        return encs[0] if encs else None
    except Exception:
        return None


def _closest(known_encs, known_names, enc) -> Tuple[Optional[str], float]:
    import face_recognition
    if not known_encs:
        return None, 1.0
    dists = face_recognition.face_distance(known_encs, enc)
    if len(dists) == 0:
        return None, 1.0
    i = int(np.argmin(dists))
    return known_names[i], float(dists[i])


# ═══════════════════════════════════════════════════════════
# Main live authentication — SINGLE CAPTURE flow
# ═══════════════════════════════════════════════════════════

def authenticate_live(stop_event: Optional[threading.Event] = None,
                      use_popup: bool = True) -> Optional[str]:
    """
    Run live face authentication with popup UI.

    NEW SINGLE-CAPTURE FLOW:
      1. Open popup → "Looking for your face..."
      2. Wait for ONE frame meeting ALL quality gates:
         - Face centered, sharp, eyes open, good lighting, large enough
      3. Capture that ONE high-quality image
      4. Close popup IMMEDIATELY
      5. Run face recognition on the captured image
      6. Match → return name; No match → return None

    Blocks until a face is verified or stop_event set.
    No timeout on "no face" — popup stays open forever.
    """
    t0 = time.time()
    logger.info("[LIVE-AUTH] Opening authentication popup (single-capture mode)...")

    # Load known encodings
    try:
        from auth import faceauth as _fa
        if not _fa._load_encodings():
            logger.warning("[LIVE-AUTH] No encodings available")
            return None
        known_encs = list(_fa._known_encodings)
        known_names = list(_fa._known_names)
    except Exception as e:
        logger.error("[LIVE-AUTH] Encoding load failed: %s", e)
        return None

    # Camera
    cam = _fa._get_camera()
    if cam is None:
        logger.error("[LIVE-AUTH] Camera unavailable")
        return None

    # Popup (fall back to headless if it can't open)
    popup = None
    if use_popup:
        try:
            popup = FaceAuthPopup()
            if not popup.show():
                logger.info("[LIVE-AUTH] Popup unavailable — headless mode")
                popup = None
        except Exception as e:
            logger.debug("[LIVE-AUTH] popup error: %s", e)
            popup = None

    tolerance = _tolerance()
    captured_frame: Optional[np.ndarray] = None
    captured_face = None
    name_result: Optional[str] = None
    last_landmarks: Optional[list] = None

    def _ui(frame, state, msg, box=None, sub="", landmarks=None, confidence=-1.0):
        if popup is not None:
            popup.update(frame, FaceStatus(
                state, msg, box, sub,
                landmarks=landmarks, confidence=confidence))

    try:
        # ── Show initial "Looking for your face..." ──
        _ui(None, "searching", "Looking for your face...",
            sub="Face centered • sharp • eyes open • good lighting")

        # ── WAIT FOREVER for ONE high-quality face ──
        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("[LIVE-AUTH] Cancelled by stop_event")
                break
            if popup is not None and not popup.is_open:
                logger.info("[LIVE-AUTH] Popup closed by user — cancelling")
                break

            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(FRAME_DELAY)
                continue

            faces, quality = face_detector.detect(frame, return_quality=True)
            H, W = frame.shape[:2]

            # ── No face → guidance, keep waiting ──
            if not faces:
                last_landmarks = None
                _ui(frame, "no_face", "No face detected",
                    sub="Please look at the camera")
                time.sleep(FRAME_DELAY)
                continue

            face = max(faces, key=lambda f: f.w * f.h)
            box = (face.x, face.y, face.w, face.h)

            # ── Quality gate: brightness ──
            bright = quality.brightness if quality else measure_brightness(frame)
            if bright < BRIGHTNESS_MIN:
                _ui(frame, "guidance", "Too dark", box, "Increase lighting")
                time.sleep(FRAME_DELAY)
                continue

            # ── Quality gate: face size ──
            if face.w < MIN_FACE_WIDTH:
                _ui(frame, "guidance", "Move closer", box, "Face too small")
                time.sleep(FRAME_DELAY)
                continue

            # ── Quality gate: centering ──
            cx = face.x + face.w / 2
            cy = face.y + face.h / 2
            if abs(cx - W / 2) > W * CENTER_TOL or abs(cy - H / 2) > H * CENTER_TOL:
                _ui(frame, "guidance", "Center your face", box)
                time.sleep(FRAME_DELAY)
                continue

            # ── Quality gate: sharpness ──
            blur = getattr(quality, "proc_blur", None) or (quality.blur if quality else measure_blur(frame))
            if blur < BLUR_MIN:
                _ui(frame, "guidance", "Too blurry", box, "Hold still")
                time.sleep(FRAME_DELAY)
                continue

            # ── Quality gate: head pose + eyes open ──
            # Get landmarks once for both checks
            try:
                import face_recognition
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                loc = face.to_face_recognition_format()
                lms_list = face_recognition.face_landmarks(rgb, [loc])
                if lms_list:
                    lm = lms_list[0]
                    # Check eyes open
                    eyes_ok, ear = _eyes_open(lm)
                    if not eyes_ok:
                        _ui(frame, "guidance", "Eyes closed", box,
                            "Please open your eyes")
                        time.sleep(FRAME_DELAY)
                        continue

                    # Check head pose
                    pose_ok, pose_reason = _check_pose_from_landmarks(lm, face)
                    if not pose_ok:
                        _ui(frame, "guidance", "Look at the camera", box,
                            pose_reason.split("(")[0].replace("_", " "))
                        time.sleep(FRAME_DELAY)
                        continue

                    # Landmarks for overlay
                    last_landmarks = []
                    for feature_points in lm.values():
                        last_landmarks.extend(feature_points)
            except Exception:
                pass  # can't check eyes/pose → proceed anyway

            # ── ALL QUALITY GATES PASSED — capture this frame ──
            captured_frame = frame.copy()
            captured_face = face
            logger.info("[LIVE-AUTH] ✓ High-quality face captured "
                        "(w=%d bright=%.1f blur=%.1f)",
                        face.w, bright, blur)

            # Show verified briefly
            _ui(frame, "detected", "Face captured — verifying...", box,
                "Processing...", landmarks=last_landmarks)

            # ── Close popup IMMEDIATELY ──
            break

    except Exception as e:
        logger.warning("[LIVE-AUTH] error: %s", e, exc_info=True)
    finally:
        # ── GUI + camera teardown — STRICT ORDER ─────────────────
        # 1. Release the camera FIRST (no more frames will be read).
        try:
            _fa._release_camera()
            logger.info("[LIVE-AUTH] Camera released")
        except Exception as e:
            logger.warning("[LIVE-AUTH] Camera release error: %s", e)
        # 2. Destroy the popup ON THE MAIN THREAD and WAIT for completion.
        if popup is not None:
            try:
                popup.close(wait=True)
                logger.info("[LIVE-AUTH] Popup destroyed")
            except Exception as e:
                logger.warning("[LIVE-AUTH] Popup destroy error: %s", e)
        # 3. Teardown fully complete.
        logger.info("[LIVE-AUTH] GUI cleanup complete")

    # ── Run face recognition on the captured image ──
    if captured_frame is not None and captured_face is not None:
        logger.info("[LIVE-AUTH] Running face recognition on captured image...")
        enc = _encode(captured_frame, captured_face)
        if enc is not None:
            name, dist = _closest(known_encs, known_names, enc)
            conf = 1.0 - dist
            logger.info("[LIVE-AUTH] Recognition result: name=%s dist=%.4f conf=%.2f tol=%.2f",
                        name or "unknown", dist, conf, tolerance)
            if name is not None and dist < tolerance:
                name_result = name
                logger.info("[LIVE-AUTH] ✅ AUTHENTICATED '%s' (conf=%.2f, %.1fs)",
                            name, conf, time.time() - t0)
            else:
                logger.warning("[LIVE-AUTH] ❌ Face not recognized "
                               "(best=%s dist=%.4f tol=%.2f)",
                               name or "unknown", dist, tolerance)
        else:
            logger.warning("[LIVE-AUTH] ❌ Failed to encode captured face")
    else:
        logger.info("[LIVE-AUTH] No face captured — authentication not completed")

    if name_result:
        logger.info("[LIVE-AUTH] Authentication successful")
    else:
        logger.info("[LIVE-AUTH] Authentication failed — 'Sorry, I couldn't verify you.'")
    return name_result


def _check_pose_from_landmarks(lm: dict, face) -> Tuple[bool, str]:
    """Check head pose using pre-computed landmarks. Returns (ok, reason)."""
    try:
        le = np.array(lm.get("left_eye", []))
        re_ = np.array(lm.get("right_eye", []))
        nose = np.array(lm.get("nose_tip", []))
        chin = np.array(lm.get("chin", []))
        if le.size == 0 or re_.size == 0:
            return True, "ok"

        le_c = le.mean(axis=0)
        re_c = re_.mean(axis=0)
        dx = re_c[0] - le_c[0]
        dy = re_c[1] - le_c[1]
        roll = float(np.degrees(np.arctan2(dy, dx)))

        yaw = 0.0
        if nose.size:
            nose_c = nose.mean(axis=0)
            eye_mid_x = (le_c[0] + re_c[0]) / 2.0
            inter_eye = max(np.hypot(dx, dy), 1.0)
            yaw = float(np.degrees(np.arcsin(np.clip((nose_c[0] - eye_mid_x) / inter_eye, -1, 1))) * 0.8)

        pitch = 0.0
        if chin.size and nose.size:
            chin_c = chin.mean(axis=0)
            nose_c = nose.mean(axis=0)
            face_h = max(face.h, 1)
            pitch = float(np.degrees(np.arctan2((chin_c[1] - nose_c[1]) - face_h * 0.35, face_h)) * 0.5)

        if abs(yaw) > MAX_YAW_DEG:
            return False, f"turn_too_much(yaw={yaw:.0f})"
        if abs(pitch) > MAX_PITCH_DEG:
            return False, f"tilt_too_much(pitch={pitch:.0f})"
        if abs(roll) > MAX_ROLL_DEG:
            return False, f"head_tilted(roll={roll:.0f})"
        return True, "ok"
    except Exception:
        return True, "ok"


if __name__ == "__main__":
    print(authenticate_live() or "not authenticated")