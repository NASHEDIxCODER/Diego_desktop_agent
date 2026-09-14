"""
Camera device selection for face authentication.

Enumerates V4L2 capture devices, identifies each by a STABLE identity
(the /dev/v4l/by-id/* symlink + USB device path -- stable across
unplug/replug, unlike the /dev/videoN index which can shift), validates
each can actually produce frames, persists the user's choice, and
prefers an EXTERNAL USB camera over the internal webcam when nothing
is explicitly configured.

Selection priority (highest first):
  1. Explicit configured camera  (DIEGO_CAMERA env var = name/label/by-id)
  2. Persisted selection          (matched by stable by-id/USB identity)
  3. External USB camera (preferred over internal when both available)
  4. Existing default (CAM_INDEX constant, if it still works)
  5. Safe working-capture fallback (first device that yields a real frame)

Safety contract:
  - Metadata-only nodes are NEVER selected -- they cannot produce frames.
  - The chosen device is probed (open -> read frame -> release) BEFORE being
    handed to the face-auth pipeline; the probe is released cleanly.
  - If the selected device disappears, selection falls through to the next
    priority level rather than failing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2 as cv

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SELECTION_PATH = DATA_DIR / "camera_selection.json"
V4L2_CAP_VIDEO_CAPTURE = 0x00000001

EXTERNAL_NAME_CUES = (
    "external",
)
INTERNAL_NAME_CUES = (
    "integrated camera", "internal", "built-in", "builtin",
    "uvc webcam", "hd uvc",
)
# DEPRECATED: friendly labels are derived from each device's OWN runtime
# metadata (see `CameraDevice.label`). Kept as an empty table so existing
# imports keep working — product names are NEVER hardcoded here.
DEVICE_ALIASES: dict = {}
PROBE_MAX_FRAMES = 12
PROBE_TIMEOUT_S = 3.0
RESOLVE_TTL_S = 30.0


@dataclass
class CameraDevice:
    """One enumerated camera, identified by stable device info."""
    index: int
    path: str
    name: str
    caps: int
    capture_capable: bool
    by_id_path: str = ""
    usb_path: str = ""
    external: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0

    @property
    def label(self) -> str:
        """Human-friendly label derived from the device's OWN runtime
        metadata (product token of the V4L2 name / by-id symlink).

        No alias tables, no hardcoded product names: whatever the connected
        hardware reports is what the user sees.
        """
        label = ""
        try:
            from voice.runtime_devices import camera_friendly_name
            label = camera_friendly_name(self.name, self.by_id_path)
        except Exception:
            label = ""
        return label or (self.name or "").strip() or f"Camera {self.index}"

    @property
    def identity(self) -> str:
        """Stable identity (by-id preferred, else USB path, else name).
        Survives unplug/replug when /dev/videoN changes."""
        if self.by_id_path:
            return self.by_id_path
        if self.usb_path:
            return self.usb_path
        return self.name.strip() or f"camera-{self.index}"


def _read_sysfs(index: int, filename: str) -> str:
    try:
        return (Path("/sys/class/video4linux") / f"video{index}" / filename).read_text().strip()
    except Exception:
        return ""


def _read_sysfs_caps(index: int) -> int:
    raw = _read_sysfs(index, "capabilities")
    try:
        return int(raw, 16) if raw else 0
    except ValueError:
        return 0


def _video_indices() -> List[int]:
    indices: List[int] = []
    try:
        for entry in Path("/dev").glob("video*"):
            m = re.search(r"(\d+)", entry.name)
            if m:
                indices.append(int(m.group(1)))
    except Exception:
        pass
    return sorted(set(indices))


def _by_id_for_index(index: int) -> str:
    """Resolve the /dev/v4l/by-id/* symlink pointing at /dev/videoN."""
    by_id_dir = Path("/dev/v4l/by-id")
    if not by_id_dir.is_dir():
        return ""
    target = f"../../video{index}"
    candidates: List[Path] = []
    try:
        for link in by_id_dir.iterdir():
            try:
                if link.is_symlink() and os.readlink(str(link)) == target:
                    candidates.append(link)
            except OSError:
                continue
    except OSError:
        return ""
    candidates.sort(key=lambda p: p.name)
    return str(candidates[0]) if candidates else ""


def _usb_path_for_index(index: int) -> str:
    try:
        dev_link = Path("/sys/class/video4linux") / f"video{index}" / "device"
        if dev_link.is_symlink():
            return os.readlink(str(dev_link))
    except OSError:
        pass
    return ""


def _integration_for_index(index: int, name: str) -> str:
    """udev ID_INTEGRATION semantics from sysfs (best effort, no subprocess).

    Trusted ONLY when the sysfs name matches the enumerated name, so a
    stale/faked topology can never inject foreign metadata.
    """
    try:
        from voice.runtime_devices import read_video_metadata
        meta = read_video_metadata(int(index))
    except Exception:
        return ""
    sysfs_name = str(meta.get("v4l_name") or "").strip()
    if not sysfs_name or sysfs_name != (name or "").strip():
        return ""
    return str(meta.get("integration") or "")


def _classify_external(name: str, by_id: str, usb_path: str,
                       integration: str = "") -> bool:
    """Generic external/internal classification (no product hardcodes).

    Priority: runtime integration metadata (USB `removable` / bus) →
    structural name cues → presence of a USB/by-id identity.
    """
    try:
        from voice.runtime_devices import classify_camera_external
        return classify_camera_external(name, by_id, usb_path, integration)
    except Exception:
        pass
    blob = f"{name} {by_id} {usb_path}".lower()
    if any(c in blob for c in EXTERNAL_NAME_CUES):
        return True
    if any(c in blob for c in INTERNAL_NAME_CUES):
        return False
    return bool(usb_path or by_id)

class CameraSelector:
    """Enumerate, validate, select and persist a camera device."""

    def __init__(self, default_index: int = 2,
                 selection_path: Path = SELECTION_PATH):
        self.default_index = default_index
        self.selection_path = selection_path
        self._resolved_index: Optional[int] = None
        self._resolved_at: float = 0.0

    def enumerate_devices(self) -> List[CameraDevice]:
        """All present /dev/video* devices with stable identity + caps.

        NOTE: /sys/.../capabilities is absent on some kernels (this
        system). When unavailable caps reads as 0, treated as UNKNOWN
        (not metadata-only) -- the probe is authoritative in that case.
        The -index1 metadata node of a multi-node device is skipped.
        """
        devices: List[CameraDevice] = []
        for idx in _video_indices():
            name = _read_sysfs(idx, "name")
            caps = _read_sysfs_caps(idx)
            by_id = _by_id_for_index(idx)
            usb = _usb_path_for_index(idx)
            if by_id and "-index1" in by_id:
                continue   # metadata-only node
            capture_capable = caps == 0 or bool(caps & V4L2_CAP_VIDEO_CAPTURE)
            external = _classify_external(name, by_id, usb,
                                          _integration_for_index(idx, name))
            devices.append(CameraDevice(
                index=idx, path=f"/dev/video{idx}", name=name, caps=caps,
                capture_capable=capture_capable, by_id_path=by_id,
                usb_path=usb, external=external))
        return devices

    def probe_device(self, index: int) -> Optional[tuple]:
        """Open device, read one frame, release. Returns (w, h, fps) or None.

        The capture is initialized the SAME way the production face-auth
        pipeline initializes it (V4L2 backend, 640x480@30, MJPEG, buffersize
        1). Some UVC cameras select()-time-out if the
        format is not set before the first read, so probing WITHOUT these
        settings would falsely reject a camera the pipeline CAN use.

        The temporary capture is always released, so this never leaves a
        stream open and never conflicts with the pipeline singleton.
        """
        cap = None
        try:
            cap = cv.VideoCapture(index, cv.CAP_V4L2)
            if not cap.isOpened():
                cap = cv.VideoCapture(index)
            if not cap.isOpened():
                return None
            # Match the production capture init so the UVC driver is primed.
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv.CAP_PROP_FPS, 30)
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
            deadline = time.time() + PROBE_TIMEOUT_S
            frames = 0
            while time.time() < deadline and frames < PROBE_MAX_FRAMES:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    fps = float(cap.get(cv.CAP_PROP_FPS))
                    return (w or 640, h or 480, fps or 30.0)
                frames += 1
            return None
        except Exception as e:
            logger.debug("[CAMERA-SELECT] probe %d error: %s", index, e)
            return None
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    def list_working_cameras(self) -> List[CameraDevice]:
        """Enumerate + probe; return only capture-capable devices that
        yield frames, with res/fps recorded."""
        working: List[CameraDevice] = []
        for dev in self.enumerate_devices():
            caps_ok = (dev.caps == 0) or bool(dev.caps & V4L2_CAP_VIDEO_CAPTURE)
            if not caps_ok:
                logger.info("[CAMERA-SELECT] skip %s (%s): metadata-only (caps=0x%x)",
                            dev.index, dev.path, dev.caps)
                continue
            res = self.probe_device(dev.index)
            if res is None:
                logger.info("[CAMERA-SELECT] skip %s (%s): no frames produced",
                            dev.index, dev.path)
                continue
            dev.width, dev.height, dev.fps = res
            working.append(dev)
        return working

    def load_selection(self) -> Optional[dict]:
        try:
            if self.selection_path.exists():
                data = json.loads(self.selection_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and (data.get("by_id_path")
                                               or data.get("device_name")):
                    return data
        except Exception as e:
            logger.warning("[CAMERA-SELECT] load selection failed: %s", e)
        return None

    def save_selection(self, device: CameraDevice) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "device_name": device.name,
                "device_label": device.label,
                "device_path": device.path,
                "index": device.index,  # snapshot only; not used for matching
                "by_id_path": device.by_id_path,
                "usb_path": device.usb_path,
                "external": device.external,
                "caps": f"0x{device.caps:x}",
                "resolution": f"{device.width}x{device.height}",
                "fps": device.fps,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            self.selection_path.write_text(json.dumps(payload, indent=2),
                                           encoding="utf-8")
            logger.info("[CAMERA-SELECT] persisted selection: %s (%s)",
                        device.label, device.identity)
        except Exception as e:
            logger.warning("[CAMERA-SELECT] save selection failed: %s", e)
    # -- Matching (stable identity first) ---------------------------
    def _match_by_identity(self, devices, by_id="", usb="", name="") -> Optional[CameraDevice]:
        # 1. by-id symlink (authoritative across reconnects).
        if by_id:
            for d in devices:
                if d.by_id_path and d.by_id_path == by_id:
                    return d
        # 2. USB device path.
        if usb:
            for d in devices:
                if d.usb_path and d.usb_path == usb:
                    return d
        # 3. Stable name (last resort).
        if name:
            want = name.strip().lower()
            for d in devices:
                if d.name and d.name.strip().lower() == want:
                    return d
            for d in devices:
                if d.label.lower() == want:
                    return d
            for d in devices:
                if d.name and want in d.name.strip().lower():
                    return d
        return None

    def _match_by_name(self, devices, name: str) -> Optional[CameraDevice]:
        if not name:
            return None
        want = name.strip().lower()
        for d in devices:
            if d.name and d.name.strip().lower() == want:
                return d
        for d in devices:
            if d.label.lower() == want:
                return d
        for d in devices:
            if d.name and want in d.name.strip().lower():
                return d
        return None

    def _match_by_index(self, devices, index: int) -> Optional[CameraDevice]:
        for d in devices:
            if d.index == index:
                return d
        return None

    def _match_by_by_id(self, devices, by_id: str) -> Optional[CameraDevice]:
        if not by_id:
            return None
        for d in devices:
            if d.by_id_path and d.by_id_path == by_id:
                return d
        return None

    # -- Public selection API ---------------------------------------
    def select_camera(self, name: str) -> Optional[CameraDevice]:
        """Validate + persist an explicit choice by name/label/by-id."""
        working = self.list_working_cameras()
        chosen = self._match_by_name(working, name) or self._match_by_by_id(working, name)
        if chosen is None:
            logger.warning("[CAMERA-SELECT] %r not found among working cameras", name)
            return None
        self.save_selection(chosen)
        self._resolved_index = chosen.index
        self._resolved_at = time.time()
        return chosen

    def select_camera_by_index(self, index: int) -> Optional[CameraDevice]:
        """Validate + persist an explicit choice by current OpenCV index."""
        working = self.list_working_cameras()
        chosen = self._match_by_index(working, index)
        if chosen is None:
            logger.warning("[CAMERA-SELECT] /dev/video%d not a working camera", index)
            return None
        self.save_selection(chosen)
        self._resolved_index = chosen.index
        self._resolved_at = time.time()
        return chosen
    # -- Runtime interactive selector ------------------------------
    def list_for_display(self, devices=None) -> List[dict]:
        """Return cameras as plain dicts for a runtime UI / CLI selector.
        Each entry: number, name, device path, external/internal, cap.
        """
        if devices is None:
            devices = self.list_working_cameras()
        rows: List[dict] = []
        for i, d in enumerate(devices, 1):
            rows.append({
                "number": i,
                "name": d.label,
                "device_path": d.path,
                "by_id": d.by_id_path,
                "external": d.external,
                "capability": (f"{d.width}x{d.height}@{int(d.fps)}fps"
                               if d.width else "unknown"),
                "index": d.index,
            })
        return rows

    def interactive_select(self, input_fn=input, print_fn=print) -> Optional[CameraDevice]:
        """Runtime CLI selector. Lists cameras, prompts the user, validates.

        If the user cancels (empty/invalid), returns None -- the caller then
        uses the default selection behavior. The selected camera is probed
        (validates a frame is produced) and persisted.
        """
        working = self.list_working_cameras()
        if not working:
            print_fn("[CAMERA-SELECT] No working cameras detected.")
            return None
        rows = self.list_for_display(working)
        print_fn("[CAMERA-SELECT] Detected cameras:")
        print_fn("  #  name                                  device     ext  cap")
        for r in rows:
            print_fn(f"  {r['number']:<2} {r['name']:<38} {r['device_path']:<9}"
                     f"  {'Y' if r['external'] else 'N'}   {r['capability']}")
        try:
            choice = input_fn("Select camera # (Enter to keep default): ").strip()
        except EOFError:
            return None
        if not choice:
            return None
        try:
            n = int(choice)
        except ValueError:
            print_fn(f"[CAMERA-SELECT] invalid selection {choice!r}; keeping default")
            return None
        if n < 1 or n > len(working):
            print_fn(f"[CAMERA-SELECT] # {n} out of range; keeping default")
            return None
        chosen = working[n - 1]
        self.save_selection(chosen)
        self._resolved_index = chosen.index
        self._resolved_at = time.time()
        print_fn(f"[CAMERA-SELECT] selected {chosen.label} ({chosen.path})")
        return chosen

    def invalidate_cache(self) -> None:
        """Forget the resolved index (e.g. after the camera is fully released)."""
        self._resolved_index = None
        self._resolved_at = 0.0
    def resolve_camera_index(self) -> Optional[int]:
        """Apply selection priority, validate, return a working index or None.

        Result is cached for RESOLVE_TTL_S to avoid re-probing on every
        open/close, while still detecting device changes shortly after.
        """
        now = time.time()
        if (self._resolved_index is not None
                and (now - self._resolved_at) < RESOLVE_TTL_S):
            return self._resolved_index

        working = self.list_working_cameras()
        if not working:
            logger.error("[CAMERA-SELECT] no working camera found")
            self._resolved_index = None
            self._resolved_at = now
            return None

        chosen: Optional[CameraDevice] = None
        label = ""

        # Priority 1: explicit configured camera (env var).
        explicit = (os.environ.get("DIEGO_CAMERA") or "").strip()
        if explicit:
            chosen = self._match_by_name(working, explicit) or self._match_by_by_id(working, explicit)
            if chosen:
                label = f"explicit env DIEGO_CAMERA={explicit!r}"
            else:
                logger.warning("[CAMERA-SELECT] explicit camera %r not found/working;"
                               " falling through", explicit)

        # Priority 2: persisted selection (matched by stable by-id/USB/name).
        if chosen is None:
            sel = self.load_selection()
            if sel:
                chosen = self._match_by_identity(
                    working, sel.get("by_id_path", ""),
                    sel.get("usb_path", ""), sel.get("device_name", ""))
                if chosen:
                    label = f"persisted ({sel.get('device_label') or sel.get('device_name')})"
                else:
                    logger.info("[CAMERA-SELECT] persisted camera %r not present;"
                                " falling through", sel.get("device_name"))

        # Priority 3: external USB camera preferred over internal.
        if chosen is None:
            externals = [d for d in working if d.external]
            if externals:
                chosen = externals[0]
                label = f"external-USB preference ({chosen.label})"

        # Priority 4: existing default (CAM_INDEX) if it still works.
        if chosen is None:
            chosen = self._match_by_index(working, self.default_index)
            if chosen:
                label = f"default CAM_INDEX={self.default_index}"

        # Priority 5: safe fallback -- first working capture device.
        if chosen is None:
            chosen = working[0]
            label = f"fallback (first working: {chosen.path})"

        if chosen:
            self.save_selection(chosen)
            self._resolved_index = chosen.index
            self._resolved_at = now
            logger.info("[CAMERA-SELECT] selected %s (%s) [%s]",
                        chosen.label, chosen.path, label)
            return chosen.index

        self._resolved_index = None
        self._resolved_at = now
        return None


# Module-level singleton selector (default index overridden at import time
# by faceauth via set_default_index).
_camera_selector = CameraSelector(default_index=2)


def set_default_index(index: int) -> None:
    """Let faceauth set the default CAM_INDEX for the shared selector."""
    _camera_selector.default_index = index


def get_selector() -> CameraSelector:
    return _camera_selector
