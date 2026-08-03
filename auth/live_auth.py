"""
LiveAuth — Face authentication with a live popup UI.

This is the NEW authentication flow. It runs ONLY after the wake word,
NEVER at startup, and NEVER blocks Leo from booting.

Behaviour:
  - Opens the FaceAuthPopup and streams the webcam at ~30 FPS.
  - WAITS FOREVER for a face to enter the frame (no "no face" timeout).
    If the user walks away, it keeps showing "No face detected".
  - Runs face-quality checks and shows GUIDANCE instead of failing:
      "Move closer", "Too dark", "Too blurry", "Look at camera",
      "Center your face", "Face too small".
  - Collects MULTI_FRAME consecutive good frames, votes by majority,
    averages confidence, and authenticates only on a stable identity.
  - On success: shows "Identity verified — welcome <name>", closes, returns name.
  - On failure/cancel: returns None. Leo then DENIES the interaction and
    returns to WAIT_WAKE — it NEVER exits.

Public API:
    authenticate_live(stop_event=None) -> Optional[str]

Designed to be run inside an executor thread (it blocks on the camera).
"""

import logging
import threading
import time
from collections import Counter
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from auth.face_detector import face_detector, measure_brightness, measure_blur
from auth.face_popup import FaceAuthPopup, FaceStatus

logger = logging.getLogger(__name__)

# ── Quality gates (guidance, not failure) ────────────────
MIN_FACE_WIDTH = 110           # px — below this: "Move closer"
CENTER_TOL = 0.28              # face center must be within ±28% of frame center
BRIGHTNESS_MIN = 70.0          # below: "Too dark"
BLUR_MIN = 22.0                # below: "Too blurry"
MAX_YAW_DEG = 20.0
MAX_PITCH_DEG = 20.0
MAX_ROLL_DEG = 15.0

# ── Multi-frame verification ─────────────────────────────
VOTE_FRAMES = 15               # consecutive good frames to collect
MAJORITY = 8                   # votes needed to accept identity
VERIFY_CONFIDENCE = 0.62       # (1 - distance) threshold for a frame to "count"

# Frame pacing
FRAME_DELAY = 0.033            # ~30 FPS


def _tolerance() -> float:
    try:
        from auth.faceauth import FACE_TOLERANCE
        return FACE_TOLERANCE
    except Exception:
        return 0.55


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
# Main live authentication
# ═══════════════════════════════════════════════════════════

def authenticate_live(stop_event: Optional[threading.Event] = None,
                      use_popup: bool = True) -> Optional[str]:
    """
    Run live face authentication with popup UI.

    Blocks until a face is verified (no "no face" timeout) or stop_event set.

    Returns the verified user's name, or None if cancelled/unavailable.
    """
    t0 = time.time()
    logger.info("[LIVE-AUTH] Opening authentication popup...")

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
    votes: List[Tuple[str, float]] = []
    good_streak = 0
    frame_i = 0
    name_result: Optional[str] = None
    last_landmarks: Optional[list] = None   # 68-pt overlay for the popup
    last_conf: float = -1.0                 # latest recognition confidence

    def _ui(frame, state, msg, box=None, sub="", landmarks=None, confidence=-1.0):
        if popup is not None:
            popup.update(frame, FaceStatus(
                state, msg, box, sub,
                landmarks=landmarks, confidence=confidence))


    try:
        # ── WAIT FOREVER for a face + verification ──
        # No timeout on "no face". Only stop_event or success breaks the loop.
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
            frame_i += 1

            faces, quality = face_detector.detect(frame, return_quality=True)
            H, W = frame.shape[:2]

            # ── No face → guidance, keep waiting (NO timeout) ──
            # TASK 6: if the face disappears mid-verification the assistant
            # PAUSES — votes and streak reset, camera stays open forever.
            if not faces:
                good_streak = 0
                votes.clear()
                last_landmarks = None
                last_conf = -1.0
                _ui(frame, "no_face", "No face detected",
                    sub="Please look at the camera")
                time.sleep(FRAME_DELAY)
                continue


            face = max(faces, key=lambda f: f.w * f.h)
            box = (face.x, face.y, face.w, face.h)

            # ── Quality gates → guidance instead of failure ──
            # Brightness
            bright = quality.brightness if quality else measure_brightness(frame)
            if bright < BRIGHTNESS_MIN:
                good_streak = 0
                _ui(frame, "guidance", "Too dark", box, "Increase lighting")
                time.sleep(FRAME_DELAY); continue

            # Face size
            if face.w < MIN_FACE_WIDTH:
                good_streak = 0
                _ui(frame, "guidance", "Move closer", box, "Face too small")
                time.sleep(FRAME_DELAY); continue

            # Centering
            cx = face.x + face.w/2
            cy = face.y + face.h/2
            if abs(cx - W/2) > W*CENTER_TOL or abs(cy - H/2) > H*CENTER_TOL:
                good_streak = 0
                _ui(frame, "guidance", "Center your face", box)
                time.sleep(FRAME_DELAY); continue

            # Blur
            blur = getattr(quality, "proc_blur", None) or (quality.blur if quality else measure_blur(frame))
            if blur < BLUR_MIN:
                good_streak = 0
                _ui(frame, "guidance", "Too blurry", box, "Hold still")
                time.sleep(FRAME_DELAY); continue

            # Head pose
            pose, reason = _head_pose_deg(frame, face)
            if reason != "ok":
                good_streak = 0
                _ui(frame, "guidance", "Look at the camera", box,
                    reason.split("(")[0].replace("_", " "))
                time.sleep(FRAME_DELAY); continue

            # ── Good frame → landmarks + encode + vote ──
            good_streak += 1
            last_landmarks = _landmark_points(frame, face)

            enc = _encode(frame, face)
            if enc is not None:
                name, dist = _closest(known_encs, known_names, enc)
                if name is not None:
                    last_conf = 1.0 - dist
                    if dist < tolerance:
                        votes.append((name, dist))
                        logger.info("[LIVE-AUTH] vote %d: %s conf=%.2f",
                                    len(votes), name, last_conf)
                    else:
                        logger.debug("[LIVE-AUTH] candidate=%s conf=%.2f < tolerance",
                                     name, last_conf)

            # TASK 6: GREEN box + landmarks + live confidence while verifying.
            _ui(frame, "detected", "Face detected — hold still...", box,
                f"Verifying {min(good_streak, VOTE_FRAMES)}/{VOTE_FRAMES}",
                landmarks=last_landmarks, confidence=last_conf)

            # ── Decision once enough good frames collected ──
            if good_streak >= VOTE_FRAMES:
                if votes:
                    counts = Counter(n for n, _ in votes)
                    top, top_n = counts.most_common(1)[0]
                    avg_dist = float(np.mean([d for n, d in votes if n == top]))
                    if top_n >= MAJORITY and avg_dist < tolerance:
                        name_result = top
                        # TASK 6: continue ONLY after recognized face AND
                        # confidence above threshold (majority + tolerance).
                        _ui(frame, "verified", "Identity verified",
                            box, f"Welcome {top}",
                            landmarks=last_landmarks,
                            confidence=1.0 - avg_dist)
                        logger.info("[LIVE-AUTH] Identity verified: %s "
                                    "(votes=%d/%d avg=%.3f conf=%.2f %.1fs)",
                                    top, top_n, len(votes), avg_dist,
                                    1.0 - avg_dist, time.time()-t0)
                        break

                    else:
                        # Not a stable known identity → keep trying, reset
                        logger.info("[LIVE-AUTH] unstable/unknown (top=%s %d/%d) — resuming",
                                    top, top_n, len(votes))
                        votes.clear()
                        good_streak = 0
                else:
                    votes.clear()
                    good_streak = 0

            time.sleep(FRAME_DELAY)

    except Exception as e:
        logger.warning("[LIVE-AUTH] error: %s", e, exc_info=True)
    finally:
        # ── GUI + camera teardown — STRICT ORDER ─────────────────
        # On success, let the popup show "Identity verified" briefly first.
        if popup is not None and name_result:
            time.sleep(1.2)
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
        # 3. Teardown fully complete — safe to enter WAKE_LISTEN.
        logger.info("[LIVE-AUTH] GUI cleanup complete")


    if name_result:
        logger.info("[LIVE-AUTH] Greeting")
    else:
        logger.info("[LIVE-AUTH] Authentication not completed (denied) — returning to wake")
    return name_result


if __name__ == "__main__":
    print(authenticate_live() or "not authenticated")
