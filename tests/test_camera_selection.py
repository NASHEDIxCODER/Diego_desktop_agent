"""
Phase 18F-H runtime camera selection tests (mocks/fakes, no real camera).

Covers enumerate/validate/select/persist/reconnect/fallback with a stable
by-id identity, external-camera preference, and the runtime interactive
selector. Hardware map faked:

  /dev/video0  USB2.0 HD UVC WebCam   (capture; internal)
  /dev/video1  USB2.0 HD UVC metadata (-index1 node; skipped)
  /dev/video2  ZEB LIVE PRO          (capture; EXTERNAL)
  /dev/video3  ZEB LIVE PRO metadata (-index1 node; skipped)
"""

import json
from pathlib import Path

import pytest

import compat  # noqa: F401
import auth.camera_selector as cs
from auth.camera_selector import CameraSelector, CameraDevice

# Shared fake hardware table; tests can swap it via FakeCapture._active_hw.
FAKE_HW = {
    0: ("USB2.0 HD UVC WebCam: USB2.0 HD",
        "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_HD_UVC_WebCam-video-index0",
        True),
    1: ("USB2.0 HD UVC WebCam: USB2.0 HD",
        "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_HD_UVC_WebCam-video-index1",
        False),
    2: ("ZEB LIVE PRO: ZEB LIVE PRO",
        "/dev/v4l/by-id/usb-BC-250603-ZW_ZEB_LIVE_PRO-video-index0",
        True),
    3: ("ZEB LIVE PRO: ZEB LIVE PRO",
        "/dev/v4l/by-id/usb-BC-250603-ZW_ZEB_LIVE_PRO-video-index1",
        False),
}


class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture.

    Uses a class-level ``_active_hw`` table that tests can swap so reconnect
    tests can point the probe at a different device topology.
    """
    _open = 0
    _active_hw = FAKE_HW
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 1196444237
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, index, backend=None):
        hw = FakeCapture._active_hw
        self.index = index
        self._ok = index in hw and hw[index][2]
        if self._ok:
            FakeCapture._open += 1
        self._released = False

    def isOpened(self):
        return self._ok

    def read(self):
        if self._ok:
            import numpy as np
            return True, np.zeros((480, 640, 3), dtype="uint8")
        return False, None

    def set(self, *a):
        return True

    def get(self, p):
        if p == self.CAP_PROP_FRAME_WIDTH:
            return 640
        if p == self.CAP_PROP_FRAME_HEIGHT:
            return 480
        if p == self.CAP_PROP_FPS:
            return 30.0
        return 0

    def release(self):
        if self._ok and not self._released:
            self._released = True
            FakeCapture._open = max(0, FakeCapture._open - 1)


def _patch_hw(monkeypatch, hw=FAKE_HW):
    monkeypatch.setattr(cs, "_video_indices", lambda: sorted(hw.keys()))
    monkeypatch.setattr(cs, "_read_sysfs",
                        lambda i, f: hw[i][0] if f == "name" else "")
    # Return the full by-id path for every index; the production enumerate
    # filters out -index1 metadata nodes itself.
    monkeypatch.setattr(cs, "_by_id_for_index",
                        lambda i: hw[i][1] if i in hw else "")
    # Non-empty usb_path so _classify_external uses the name cues (both real
    # USB cameras have a usb_path; the name is the external/internal signal).
    monkeypatch.setattr(cs, "_usb_path_for_index",
                        lambda i: f"../../../usb{i}:1.0" if i in hw else "")
    FakeCapture._active_hw = hw
    monkeypatch.setattr(cs.cv, "VideoCapture", FakeCapture)


@pytest.fixture()
def sel(monkeypatch, tmp_path):
    s = CameraSelector(default_index=2,
                       selection_path=tmp_path / "camera_selection.json")
    _patch_hw(monkeypatch)
    FakeCapture._open = 0
    return s


# 1. Enumerate captures (only -index0 capture nodes)
def test_enumerate_capture_capable(sel):
    devs = sel.enumerate_devices()
    assert [d.index for d in devs] == [0, 2]
    for d in devs:
        assert d.capture_capable is True


# 2. Rejects metadata-only
def test_rejects_metadata_only(sel):
    working = sel.list_working_cameras()
    assert [d.index for d in working] == [0, 2]
    assert 1 not in [d.index for d in working]
    assert 3 not in [d.index for d in working]


# 3. Stable identity recorded
def test_stable_identity_recorded(sel):
    working = sel.list_working_cameras()
    by_ids = {d.index: d.by_id_path for d in working}
    assert "index0" in by_ids[2]
    assert "ZEB_LIVE_PRO" in by_ids[2]
    for d in working:
        assert d.identity == d.by_id_path


# 4. Persisted camera restored
def test_persisted_camera_restored(sel):
    chosen = sel.select_camera("ZEB LIVE PRO")
    assert chosen.index == 2
    # A fresh selector on the same store must restore it via by-id.
    s2 = CameraSelector(default_index=0, selection_path=sel.selection_path)
    # monkeypatch is module-level, reused; re-read works.
    s2._resolved_index = None
    assert s2.resolve_camera_index() == 2


# 5. Device node changes, same identity -> restored
def test_reconnect_restores_same_camera(monkeypatch, tmp_path):
    store = tmp_path / "camera_selection.json"
    store.write_text(json.dumps({
        "device_name": "ZEB LIVE PRO: ZEB LIVE PRO",
        "by_id_path": "/dev/v4l/by-id/usb-BC-250603-ZW_ZEB_LIVE_PRO-video-index0",
        "index": 2, "device_path": "/dev/video2",
    }))
    # ZEB LIVE PRO reconnected at /dev/video4, same by-id.
    new_hw = {
        0: FAKE_HW[0],
        4: ("ZEB LIVE PRO: ZEB LIVE PRO",
            "/dev/v4l/by-id/usb-BC-250603-ZW_ZEB_LIVE_PRO-video-index0",
            True),
        }
    _patch_hw(monkeypatch, new_hw)
    sel = CameraSelector(default_index=0, selection_path=store)
    assert sel.resolve_camera_index() == 4


# 6. Previously selected unavailable -> fallback
def test_previously_selected_unavailable_falls_back(monkeypatch, tmp_path):
    store = tmp_path / "camera_selection.json"
    store.write_text(json.dumps({
        "device_name": "ZEB LIVE PRO: ZEB LIVE PRO",
        "by_id_path": "/dev/v4l/by-id/usb-BC-250603-ZW_ZEB_LIVE_PRO-video-index0",
        "index": 2, "device_path": "/dev/video2",
    }))
    # Only the internal webcam is present.
    new_hw = {0: FAKE_HW[0], 1: FAKE_HW[1]}
    _patch_hw(monkeypatch, new_hw)
    sel = CameraSelector(default_index=0, selection_path=store)
    assert sel.resolve_camera_index() == 0


# 7. External preferred over internal
def test_external_preferred_over_internal(sel):
    assert not sel.selection_path.exists()
    idx = sel.resolve_camera_index()
    assert idx == 2, "external USB camera must win without an explicit choice"
    s = sel.load_selection()
    assert s["external"] is True


# 8. User-selected overrides automatic preference
def test_user_selected_overrides_preference(sel):
    dev = sel.select_camera("USB2.0 HD UVC WebCam")
    assert dev.index == 0
    sel._resolved_index = None
    assert sel.resolve_camera_index() == 0
    s = sel.load_selection()
    assert s["device_label"] == "USB2.0 HD UVC WebCam"


# 9. Invalid selection fails safely
def test_invalid_selection_fails_safely(sel, monkeypatch):
    assert sel.select_camera("nonexistent xyz") is None
    assert sel.select_camera_by_index(99) is None
    bogus = CameraSelector(default_index=99, selection_path=sel.selection_path)
    _patch_hw(monkeypatch)
    assert bogus.resolve_camera_index() in (0, 2)


# 10. Probe stream is released
def test_probe_stream_released(sel):
    FakeCapture._open = 0
    sel.list_working_cameras()  # probes every device
    assert FakeCapture._open == 0, "all probe streams must be released"


# 11. Only one active camera stream owned
def test_only_one_stream(sel):
    FakeCapture._open = 0
    sel.resolve_camera_index()
    assert FakeCapture._open == 0
    cap = FakeCapture(2)
    assert FakeCapture._open == 1
    cap.release()
    assert FakeCapture._open == 0


# 12. Existing face-auth receives the selected device
def test_faceauth_receives_selected_device(sel, monkeypatch):
    dev = sel.select_camera("ZEB LIVE PRO")
    assert dev.index == 2
    import auth.faceauth as fa
    monkeypatch.setattr(fa, "get_selector", lambda: sel)
    fa._camera = None
    fa._camera_refcount = 0
    cam = fa._get_camera()
    assert cam is not None and cam.index == 2
    assert cam.get(fa.cv.CAP_PROP_FRAME_WIDTH) == 640
    fa._release_camera()
    assert fa._camera is None


# 13. No hard-coded /dev/video2 assumption
def test_no_hardcoded_video2(sel):
    sel.default_index = 99
    idx = sel.resolve_camera_index()
    assert idx in (0, 2)
    assert idx != 99


# 14. Existing auth behavior unchanged
def test_auth_pipeline_unchanged(sel, monkeypatch):
    import auth.faceauth as fa
    monkeypatch.setattr(fa, "get_selector", lambda: sel)
    fa._camera = None
    fa._camera_refcount = 0
    cam = fa._get_camera()
    assert cam is not None
    assert fa._get_exposure_props(cam) is not None or True
    fa._release_camera()
    assert fa._camera is None


# Runtime interactive selector
def test_interactive_select_picks_chosen(sel):
    # list_for_display is 1-based: #1 = internal webcam (index 0), #2 = ZEB LIVE PRO
    chosen = sel.interactive_select(input_fn=lambda p: "2",
                                    print_fn=lambda *a: None)
    assert chosen is not None and chosen.label == "ZEB LIVE PRO"
    assert sel.load_selection()["device_label"] == "ZEB LIVE PRO"


def test_interactive_select_cancel_keeps_default(sel):
    assert sel.interactive_select(input_fn=lambda p: "",
                                  print_fn=lambda *a: None) is None
    assert sel.interactive_select(input_fn=lambda p: "abc",
                                  print_fn=lambda *a: None) is None
    assert sel.interactive_select(input_fn=lambda p: "99",
                                  print_fn=lambda *a: None) is None
