"""Runtime hardware device discovery / name-based selection tests.

Covers (all with fakes — NO hardware, NO real persistence files):

  DEVICE DISCOVERY   physical input/output/camera discovery, camera-associated
                     microphone detection, virtual/wrapper classification
  FRIENDLY NAMING    deterministic runtime names, missing manufacturer/model,
                     missing advanced Linux metadata, raw-name fallback,
                     no hardcoded product names
  HARDWARE IDENTITY  stability across runtime-index / ALSA renumbering,
                     same physical device → same logical identity,
                     deterministic fallback identity
  PERSISTENCE        selection survives reconnect/index changes,
                     stale indices are never canonical
  INDEPENDENCE       input / output / camera never influence each other
  FALLBACK           preferred device disappears → next valid physical device,
                     camera microphone never steals the primary microphone
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice import runtime_devices as rd  # noqa: E402

# ═══════════════════════════════════════════════════════════════
# Synthetic hardware (invented names — deliberately NOT the
# developer's hardware, so hardcoding would fail these tests).
# ═══════════════════════════════════════════════════════════════

HEADSET_META = {
    "bus": "usb", "usb_vid": "0d8c", "usb_pid": "0012",
    "usb_product": "Aurora Headset", "usb_manufacturer": "Aurora Audio",
    "usb_serial": "SN-77", "removable": "removable",
    "integration": "external", "card_id": "Headset", "usb_dir": "1-4",
}
CAMERA_COMBO_META = {
    "bus": "usb", "usb_vid": "1bcf", "usb_pid": "2281",
    "usb_product": "VistaCam Pro", "usb_manufacturer": "Vista",
    "usb_serial": "", "removable": "removable", "integration": "external",
    "usb_dir": "1-2", "card_id": "Pro",
}
LAPTOP_CODEC_META = {
    "bus": "pci", "pci_id": "1022:15e3", "card_id": "Generic_1",
    "integration": "internal",
}
GPU_HDMI_META = {
    "bus": "pci", "pci_id": "10de:2291", "card_id": "NVidia",
    "integration": "internal",
}
META_BY_CARD = {0: CAMERA_COMBO_META, 1: GPU_HDMI_META,
                2: LAPTOP_CODEC_META, 3: HEADSET_META}

CAMERA_RAW = "VistaCam Pro: USB Audio (hw:0,0)"
HDMI_RAW = "HDA VendorX: HDMI 0 (hw:1,3)"
CODEC_RAW = "HD-Audio Generic: ALC999 Analog (hw:2,0)"
HEADSET_RAW = "Aurora Headset: USB Audio (hw:3,0)"

CAMERA_BY_ID = "/dev/v4l/by-id/usb-Vista_VistaCam_Pro-video-index0"


def _entry(index, name, in_ch, out_ch, rate=48000.0):
    return {"index": index, "name": name, "hostapi": 0,
            "max_input_channels": in_ch, "max_output_channels": out_ch,
            "default_samplerate": rate}


BASE_DEVICES = [
    _entry(0, CAMERA_RAW, 1, 0, 44100.0),      # camera-bundled microphone
    _entry(1, HDMI_RAW, 0, 8),                 # display audio (GPU)
    _entry(2, CODEC_RAW, 2, 0),                # built-in codec (capture-only)
    _entry(3, HEADSET_RAW, 1, 2),              # external headset (duplex)
    _entry(4, "sysdefault", 128, 0),           # virtual bus
    _entry(5, "lavrate", 128, 0),              # rate-conversion wrapper
    _entry(6, "default", 32, 32),              # virtual default route
    _entry(7, "Monitor of VistaCam Pro", 2, 0),  # monitor of a sink
    _entry(8, "Loopback Capture", 2, 0),       # loopback
]


class FakeSD:
    """Minimal sounddevice stand-in (enumeration only)."""

    def __init__(self, devices=None, hostapis=None, default=(6, 6)):
        self._devices = list(BASE_DEVICES if devices is None else devices)
        self._hostapis = list(hostapis or [{"name": "ALSA"}])
        self.default = SimpleNamespace(device=tuple(default))

    def query_devices(self):
        return [dict(d) for d in self._devices]

    def query_hostapis(self):
        return list(self._hostapis)


def fake_meta(raw):
    """Synthetic /sys + /proc/asound metadata, keyed by ALSA card number."""
    card, _sub = rd.extract_alsa_card(raw or "")
    return dict(META_BY_CARD.get(card, {}))


def empty_meta(_raw):
    """Advanced Linux metadata completely unavailable."""
    return {}


CAMERA_IDENTITIES = rd.CameraIdentitySet(identities=[
    rd.CameraIdentity(name="VistaCam Pro: VistaCam Pro",
                      product="VistaCam Pro", usb_dir="1-2",
                      usb_id="1bcf:2281",
                      name_token=rd.normalize_raw_name("VistaCam Pro")),
])


def _camera_runtime(index=0, name="VistaCam Pro: VistaCam Pro",
                    by_id=CAMERA_BY_ID, external=True):
    return rd.RuntimeDevice(
        runtime_index=index, raw_name=name, direction="camera",
        device_type="camera", normalized_name=rd.normalize_raw_name(name),
        friendly_name=rd.camera_friendly_name(name, by_id),
        sample_rate=30.0, category="PHYSICAL", is_physical=True,
        is_virtual=False, is_usb=True, is_camera=True,
        is_internal=(not external),
        integration="external" if external else "internal",
        model=rd.camera_friendly_name(name, by_id),
        hardware_identity=rd.camera_hardware_identity(name, by_id),
        family_id=f"family:{rd.camera_hardware_identity(name, by_id)}",
        extra={"path": f"/dev/video{index}", "by_id_path": by_id,
               "usb_path": "../../../1-2:1.0", "external": external,
               "width": 640, "height": 480, "fps": 30.0},
    )


def _internal_camera_runtime():
    return _camera_runtime(index=2, name="SonixCam: SonixCam HD",
                           by_id="/dev/v4l/by-id/usb-Sonix_SonixCam-video-index0",
                           external=False)


@pytest.fixture()
def registry(tmp_path):
    return rd.DeviceRegistry(selection_path=tmp_path / "hardware_selection.json")


@pytest.fixture()
def discovered(registry):
    registry.discover(sd_module=FakeSD(), meta_provider=fake_meta,
                      cameras=[_camera_runtime(), _internal_camera_runtime()],
                      camera_identities=CAMERA_IDENTITIES)
    return registry


def _by_index(devices, index):
    return next((d for d in devices if d.runtime_index == index), None)


# ═══════════════════════════════════════════════════════════════
# DEVICE DISCOVERY
# ═══════════════════════════════════════════════════════════════

def test_physical_input_discovered(discovered):
    physical = [d for d in discovered.inputs if d.is_physical]
    raws = {d.raw_name for d in physical}
    assert HEADSET_RAW in raws
    assert CODEC_RAW in raws
    assert all(d.category == "PHYSICAL" for d in physical)
    assert all(d.direction == "input" for d in physical)


def test_physical_output_discovered(discovered):
    physical = [d for d in discovered.outputs if d.is_physical]
    raws = {d.raw_name for d in physical}
    assert HEADSET_RAW in raws        # headset headphones
    assert CODEC_RAW in raws          # built-in speakers (play-by-name)
    assert HDMI_RAW in raws           # display audio endpoint
    headset_out = next(d for d in physical if d.raw_name == HEADSET_RAW)
    assert headset_out.direction == "output"
    assert headset_out.friendly_name.endswith("Headphones")
    codec_out = next(d for d in physical if d.raw_name == CODEC_RAW)
    assert codec_out.friendly_name.endswith("Speakers")
    assert codec_out.extra["playback_by_name"] is True


def test_camera_discovered(discovered):
    assert len(discovered.cameras) == 2
    assert all(c.is_camera and c.is_physical for c in discovered.cameras)
    assert discovered.cameras[0].hardware_identity == CAMERA_BY_ID
    assert discovered.cameras[0].friendly_name == "VistaCam Pro"


def test_camera_associated_microphone_detected(discovered):
    cam_mic = _by_index(discovered.inputs, 0)
    assert cam_mic.is_camera_associated_audio is True
    assert cam_mic.friendly_name.endswith("Camera Microphone")
    headset = _by_index(discovered.inputs, 3)
    assert headset.is_camera_associated_audio is False


def test_camera_association_detected_from_runtime_metadata():
    # Same USB device node as an enumerated camera → bundled microphone.
    assert rd.detect_camera_associated_audio(
        "Unnamed Audio: USB Audio (hw:5,0)", {"usb_dir": "1-2"},
        CAMERA_IDENTITIES)
    # Same USB vid:pid as an enumerated camera → bundled microphone.
    assert rd.detect_camera_associated_audio(
        "Unnamed Audio (hw:6,0)", {"usb_id": "1bcf:2281"}, CAMERA_IDENTITIES)
    # Camera product token present in the audio name → bundled microphone.
    assert rd.detect_camera_associated_audio(
        "VistaCam Pro: USB Audio (hw:7,0)", {}, CAMERA_IDENTITIES)
    # Unrelated USB hardware → dedicated microphone.
    assert not rd.detect_camera_associated_audio(
        HEADSET_RAW, HEADSET_META, CAMERA_IDENTITIES)


def test_virtual_device_classification(discovered):
    for index in (4, 6):
        dev = _by_index(discovered.inputs, index)
        assert dev.is_virtual is True
        assert dev.is_physical is False
        assert dev.category == "VIRTUAL"


def test_wrapper_device_classification(discovered):
    dev = _by_index(discovered.inputs, 5)
    assert dev.category == "WRAPPER"
    assert dev.is_virtual is True


def test_monitor_and_loopback_classification(discovered):
    assert _by_index(discovered.inputs, 7).category == "MONITOR"
    assert _by_index(discovered.inputs, 8).category == "LOOPBACK"


def test_classification_is_structural_not_brand_based():
    # An unknown vendor's concrete ALSA codec is physical hardware.
    cls = rd.classify_audio_name("SomeVendor Codec: Analog (hw:9,0)",
                                 "ALSA", 2, 2)
    assert cls["category"] == "PHYSICAL"
    assert cls["is_virtual"] is False
    # Every ALSA plugin/route is virtual, whatever it happens to be called.
    for raw in ("sysdefault", "pulse", "pipewire", "default", "dmix",
                "plughw", "speexrate", "upmix", "vdownmix"):
        assert rd.classify_audio_name(raw, "ALSA", 2, 2)["is_virtual"] is True


def test_camera_enumeration_reuses_existing_selector(registry, monkeypatch):
    import auth.camera_selector as cs
    fake = SimpleNamespace(
        index=0, path="/dev/video0", name="VistaCam Pro: VistaCam Pro",
        caps=0, capture_capable=True, by_id_path=CAMERA_BY_ID,
        usb_path="../../../1-2:1.0", external=True,
        width=640, height=480, fps=30.0)

    class FakeSelector:
        def list_working_cameras(self):
            return [fake]

    monkeypatch.setattr(cs, "get_selector", lambda: FakeSelector())
    monkeypatch.setattr(rd, "read_video_metadata", lambda i: {})
    cams = registry._enumerate_cameras()
    assert len(cams) == 1
    assert cams[0].hardware_identity == CAMERA_BY_ID
    assert cams[0].friendly_name == "VistaCam Pro"
    assert cams[0].extra["external"] is True


# ═══════════════════════════════════════════════════════════════
# FRIENDLY NAMING
# ═══════════════════════════════════════════════════════════════

def test_friendly_names_derived_from_runtime_metadata(discovered):
    assert (_by_index(discovered.inputs, 3).friendly_name
            == "Aurora Headset — Microphone")
    headset_out = next(d for d in discovered.outputs
                       if d.raw_name == HEADSET_RAW)
    assert headset_out.friendly_name == "Aurora Headset — Headphones"
    assert (_by_index(discovered.inputs, 2).friendly_name
            == "HD-Audio Generic ALC999 — Microphone")
    assert (_by_index(discovered.inputs, 0).friendly_name
            == "VistaCam Pro — Camera Microphone")
    hdmi = next(d for d in discovered.outputs if d.raw_name == HDMI_RAW)
    assert hdmi.friendly_name == "HDA VendorX HDMI 0 — Display Audio"


def test_friendly_names_are_deterministic(registry):
    kwargs = dict(meta=fake_meta(HEADSET_RAW), cameras=CAMERA_IDENTITIES)
    first = rd.build_audio_views(3, HEADSET_RAW, "ALSA", 1, 2, 48000.0, **kwargs)
    second = rd.build_audio_views(3, HEADSET_RAW, "ALSA", 1, 2, 48000.0, **kwargs)
    assert first[0].friendly_name == second[0].friendly_name
    assert first[1].friendly_name == second[1].friendly_name

    def _snapshot():
        registry.discover(sd_module=FakeSD(), meta_provider=fake_meta,
                          cameras=[_camera_runtime()],
                          camera_identities=CAMERA_IDENTITIES)
        snap = registry.describe_for_commands()
        return ([d["friendly_name"] for d in snap["inputs"]],
                [d["friendly_name"] for d in snap["outputs"]],
                [c["friendly_name"] for c in snap["cameras"]])

    assert _snapshot() == _snapshot()


def test_missing_manufacturer_is_handled():
    meta = dict(HEADSET_META)
    meta.pop("usb_manufacturer")
    in_view, _out = rd.build_audio_views(3, HEADSET_RAW, "ALSA", 1, 2, meta=meta)
    assert in_view.manufacturer == ""
    assert in_view.friendly_name == "Aurora Headset — Microphone"


def test_missing_model_is_handled():
    meta = dict(HEADSET_META)
    meta.pop("usb_product")
    meta.pop("card_id")
    in_view, _out = rd.build_audio_views(3, HEADSET_RAW, "ALSA", 1, 2, meta=meta)
    assert in_view.model == "Aurora Headset"          # raw product token
    assert in_view.friendly_name == "Aurora Headset — Microphone"


def test_missing_advanced_linux_metadata_is_handled(registry):
    registry.discover(sd_module=FakeSD(), meta_provider=empty_meta,
                      cameras=[_camera_runtime()],
                      camera_identities=CAMERA_IDENTITIES)
    physical = [d for d in registry.inputs if d.is_physical]
    assert physical, "physical hardware must be classified without sysfs"
    assert all(d.friendly_name for d in physical)
    assert all(d.hardware_identity for d in physical)


def test_raw_name_fallback_works():
    in_view, out_view = rd.build_audio_views(9, "Mystery Device (hw:9,0)",
                                             "ALSA", 1, 1, meta={})
    assert in_view.friendly_name == "Mystery Device — Microphone"
    assert out_view.friendly_name == "Mystery Device — Headphones"


def test_virtual_devices_keep_their_raw_names(discovered):
    assert _by_index(discovered.inputs, 4).friendly_name == "sysdefault"
    assert _by_index(discovered.inputs, 5).friendly_name == "lavrate"


def test_no_hardcoded_product_names_in_device_sources():
    forbidden = ("evofox", "zeb live pro", "zeb_live_pro", "logitech",
                 "laptop microphone", "laptop speakers", "alc256")
    sources = (PROJECT_ROOT / "voice" / "runtime_devices.py",
               PROJECT_ROOT / "voice" / "device_manager.py",
               PROJECT_ROOT / "auth" / "camera_selector.py")
    for path in sources:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"{path.name} hardcodes {token!r}"


# ═══════════════════════════════════════════════════════════════
# HARDWARE IDENTITY
# ═══════════════════════════════════════════════════════════════

def test_identity_stable_when_runtime_index_changes():
    meta = fake_meta(HEADSET_RAW)
    a, _ = rd.build_audio_views(3, HEADSET_RAW, "ALSA", 1, 2, meta=meta)
    b, _ = rd.build_audio_views(11, HEADSET_RAW, "ALSA", 1, 2, meta=meta)
    assert a.runtime_index != b.runtime_index
    assert a.hardware_identity == b.hardware_identity


def test_identity_survives_alsa_renumbering():
    a, _ = rd.build_audio_views(3, "Aurora Headset: USB Audio (hw:3,0)",
                                "ALSA", 1, 2, meta=fake_meta(HEADSET_RAW))
    b, _ = rd.build_audio_views(8, "Aurora Headset: USB Audio (hw:7,0)",
                                "ALSA", 1, 2, meta=fake_meta(HEADSET_RAW))
    assert a.hardware_identity == b.hardware_identity


def test_same_physical_device_shares_identity_and_family(discovered):
    headset_in = _by_index(discovered.inputs, 3)
    headset_out = next(d for d in discovered.outputs
                       if d.raw_name == HEADSET_RAW)
    assert headset_in.hardware_identity == headset_out.hardware_identity
    assert headset_in.family_id == headset_out.family_id
    assert headset_in.friendly_name != headset_out.friendly_name


def test_serial_is_preferred_in_identity():
    with_serial = dict(HEADSET_META)
    without_serial = dict(HEADSET_META)
    without_serial["usb_serial"] = ""
    a = rd.hardware_identity_for_audio(HEADSET_RAW, with_serial)
    b = rd.hardware_identity_for_audio(HEADSET_RAW, without_serial)
    assert a.startswith("usb:0d8c:0012:sn 77")
    assert b.startswith("usb:0d8c:0012|")
    assert a != b


def test_insufficient_metadata_gets_deterministic_fallback_identity():
    a = rd.hardware_identity_for_audio("Mystery Device (hw:9,0)", {})
    b = rd.hardware_identity_for_audio("Mystery Device (hw:4,0)", {})
    assert a == b
    assert a.startswith("audio|")


def test_distinct_endpoints_have_distinct_identities(discovered):
    ids = [d.hardware_identity for d in discovered.outputs if d.is_physical]
    assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════════════════
# SELECTION + PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def _rediscover(registry, devices=None, cameras=None, probe_output=None):
    registry.discover(
        sd_module=FakeSD(devices), meta_provider=fake_meta,
        cameras=([_camera_runtime(), _internal_camera_runtime()]
                 if cameras is None else cameras),
        camera_identities=CAMERA_IDENTITIES, probe_output=probe_output)
    return registry


def test_automatic_selection_prefers_external_dedicated_hardware(discovered):
    assert discovered.selected_input.raw_name == HEADSET_RAW
    assert discovered.selected_input.runtime_index == 3
    assert "external" in discovered.input_reason
    assert discovered.selected_output.raw_name == HEADSET_RAW
    assert discovered.selected_output.direction == "output"
    assert discovered.selected_camera.friendly_name == "VistaCam Pro"
    assert discovered.selected_camera.extra["external"] is True


def test_selection_is_persisted_as_logical_identity(discovered):
    store = json.loads(discovered.selection_path.read_text(encoding="utf-8"))
    assert store["input"]["hardware_identity"] == (
        discovered.selected_input.hardware_identity)
    assert store["input"]["friendly_name"] == (
        discovered.selected_input.friendly_name)
    assert store["input"]["manufacturer"] == "Aurora Audio"
    assert store["input"]["model"] == "Aurora Headset"
    assert store["output"]["hardware_identity"] == (
        discovered.selected_output.hardware_identity)
    assert store["camera"]["hardware_identity"] == (
        discovered.selected_camera.hardware_identity)
    # Independent slots, direction-specific friendly names.
    assert store["input"]["friendly_name"] != store["output"]["friendly_name"]
    assert store["camera"]["friendly_name"] != store["input"]["friendly_name"]


def test_stale_runtime_index_is_never_canonical(discovered):
    store = json.loads(discovered.selection_path.read_text(encoding="utf-8"))
    store["input"]["runtime_index"] = 999      # bogus stale index
    store["output"]["runtime_index"] = 999
    discovered.selection_path.write_text(json.dumps(store), encoding="utf-8")
    _rediscover(discovered)
    assert discovered.selected_input.runtime_index == 3
    assert discovered.selected_output.runtime_index == 3
    assert "persisted-identity" in discovered.input_reason


def test_selection_survives_reconnect_with_new_indices(discovered):
    discovered.select_input_device("Aurora Headset — Microphone")
    discovered.select_output_device("Aurora Headset — Headphones")
    # Same hardware reconnects with DIFFERENT runtime indices and order.
    reshuffled = [
        _entry(0, "sysdefault", 128, 0),
        _entry(7, HEADSET_RAW, 1, 2),
        _entry(9, CODEC_RAW, 2, 0),
        _entry(11, CAMERA_RAW, 1, 0, 44100.0),
        _entry(12, HDMI_RAW, 0, 8),
    ]
    _rediscover(discovered, devices=reshuffled,
                cameras=[_camera_runtime(index=5)])
    assert discovered.selected_input.runtime_index == 7
    assert discovered.selected_input.raw_name == HEADSET_RAW
    assert discovered.selected_output.runtime_index == 7
    assert "persisted-identity(explicit)" in discovered.input_reason
    assert "persisted-identity(explicit)" in discovered.output_reason


def test_persisted_identity_beats_ranking(discovered):
    # The user explicitly chose the built-in codec; it must stay selected
    # even though the external headset ranks higher automatically.
    discovered.select_input_device("HD-Audio Generic ALC999 — Microphone")
    _rediscover(discovered)
    assert discovered.selected_input.raw_name == CODEC_RAW
    assert "persisted-identity(explicit)" in discovered.input_reason


def test_explicit_choice_survives_temporary_disconnect(discovered):
    # The user explicitly selects the headset → it is persisted as the
    # EXPLICIT hardware identity.
    discovered.select_input_device("Aurora Headset — Microphone")
    headset_identity = discovered.selected_input.hardware_identity
    assert _store(discovered)["input"]["source"] == "explicit"

    # Headset is physically REMOVED. Automatic discovery must NOT rewrite
    # the explicit slot with the fallback device.
    without = [d for d in BASE_DEVICES if d["name"] != HEADSET_RAW]
    _rediscover(discovered, devices=without)
    assert discovered.selected_input.raw_name == CODEC_RAW  # runtime fallback
    saved = _store(discovered)["input"]
    assert saved["source"] == "explicit"
    assert saved["hardware_identity"] == headset_identity

    # Headset reconnects at a DIFFERENT runtime index → the persisted
    # logical identity re-selects the SAME physical device.
    reconnected = [_entry(7, HEADSET_RAW, 1, 2), _entry(9, CODEC_RAW, 2, 0)]
    _rediscover(discovered, devices=reconnected)
    assert discovered.selected_input.raw_name == HEADSET_RAW
    assert discovered.selected_input.hardware_identity == headset_identity
    assert "persisted-identity(explicit)" in discovered.input_reason


# ═══════════════════════════════════════════════════════════════
# INDEPENDENCE (input / output / camera never touch each other)
# ═══════════════════════════════════════════════════════════════

def _store(registry):
    return json.loads(registry.selection_path.read_text(encoding="utf-8"))


def test_changing_input_does_not_change_output_or_camera(discovered):
    before_out = discovered.selected_output.hardware_identity
    before_cam = discovered.selected_camera.hardware_identity
    chosen = discovered.select_input_device(
        "HD-Audio Generic ALC999 — Microphone")
    assert chosen is not None
    assert discovered.selected_input.raw_name == CODEC_RAW
    assert discovered.selected_output.hardware_identity == before_out
    assert discovered.selected_camera.hardware_identity == before_cam
    store = _store(discovered)
    assert store["output"]["hardware_identity"] == before_out
    assert store["camera"]["hardware_identity"] == before_cam


def test_changing_output_does_not_change_input_or_camera(discovered):
    before_in = discovered.selected_input.hardware_identity
    before_cam = discovered.selected_camera.hardware_identity
    chosen = discovered.select_output_device(
        "HDA VendorX HDMI 0 — Display Audio")
    assert chosen is not None
    assert discovered.selected_output.raw_name == HDMI_RAW
    assert discovered.selected_input.hardware_identity == before_in
    assert discovered.selected_camera.hardware_identity == before_cam
    store = _store(discovered)
    assert store["input"]["hardware_identity"] == before_in
    assert store["camera"]["hardware_identity"] == before_cam


def test_changing_camera_does_not_change_audio(discovered, monkeypatch):
    import auth.camera_selector as cs
    internal = SimpleNamespace(
        index=2, path="/dev/video2", name="SonixCam: SonixCam HD", caps=0,
        capture_capable=True,
        by_id_path="/dev/v4l/by-id/usb-Sonix_SonixCam-video-index0",
        usb_path="../../../1-4:1.0", external=False,
        width=640, height=480, fps=30.0)

    class FakeSelector:
        def select_camera(self, name):
            return internal if "SonixCam" in (name or "") else None

        def select_camera_by_index(self, index):
            return internal if index == 2 else None

    monkeypatch.setattr(cs, "get_selector", lambda: FakeSelector())
    monkeypatch.setattr(rd, "read_video_metadata", lambda i: {})

    before_in = discovered.selected_input.hardware_identity
    before_out = discovered.selected_output.hardware_identity
    chosen = discovered.select_camera("SonixCam")
    assert chosen is not None
    assert chosen["friendly_name"] == "SonixCam"
    assert discovered.selected_camera.hardware_identity == internal.by_id_path
    assert discovered.selected_input.hardware_identity == before_in
    assert discovered.selected_output.hardware_identity == before_out
    store = _store(discovered)
    assert store["input"]["hardware_identity"] == before_in
    assert store["output"]["hardware_identity"] == before_out


# ═══════════════════════════════════════════════════════════════
# FALLBACK
# ═══════════════════════════════════════════════════════════════

def test_preferred_headset_disappears_input_falls_back_to_physical(discovered):
    discovered.select_input_device("Aurora Headset — Microphone")
    without = [d for d in BASE_DEVICES if d["name"] != HEADSET_RAW]
    _rediscover(discovered, devices=without)
    assert discovered.selected_input.raw_name == CODEC_RAW
    assert discovered.selected_input.is_physical is True
    assert discovered.selected_input.is_camera_associated_audio is False
    assert discovered.selected_input.is_virtual is False


def test_preferred_headset_disappears_output_falls_back_to_physical(discovered):
    discovered.select_output_device("Aurora Headset — Headphones")
    without = [d for d in BASE_DEVICES if d["name"] != HEADSET_RAW]
    _rediscover(discovered, devices=without, probe_output=lambda dev: True)
    assert discovered.selected_output.raw_name == CODEC_RAW
    assert discovered.selected_output.is_physical is True
    assert discovered.selected_output.is_display_audio is False


def test_speculative_output_needs_validation(discovered):
    # Without a playback probe the capture-only codec is NOT auto-selected;
    # the OS default route (which follows the desktop sink) is used instead
    # of a monitor's HDMI jack.
    without = [d for d in BASE_DEVICES if d["name"] != HEADSET_RAW]
    _rediscover(discovered, devices=without)
    assert discovered.selected_output.raw_name != CODEC_RAW
    assert discovered.selected_output.raw_name == "default"
    assert "virtual" in discovered.output_reason


def test_camera_microphone_never_steals_a_dedicated_microphone(discovered):
    assert discovered.selected_input.raw_name == HEADSET_RAW
    # Only the camera microphone + virtual buses remain → it becomes the
    # explicit last-resort fallback, and the reason says so.
    only = [_entry(0, CAMERA_RAW, 1, 0, 44100.0),
            _entry(4, "sysdefault", 128, 0),
            _entry(6, "default", 32, 32)]
    _rediscover(discovered, devices=only)
    assert discovered.selected_input.raw_name == CAMERA_RAW
    assert "camera-associated" in discovered.input_reason


def test_auto_saved_camera_mic_defers_to_dedicated_microphone(discovered):
    cam_mic = _by_index(discovered.inputs, 0)
    rd.save_hardware_slot(rd.INPUT_SLOT, {
        "hardware_identity": cam_mic.hardware_identity,
        "friendly_name": cam_mic.friendly_name,
        "family_id": cam_mic.family_id,
        "source": "auto",
    }, discovered.selection_path)
    _rediscover(discovered)
    assert discovered.selected_input.raw_name == HEADSET_RAW


def test_explicit_camera_mic_choice_is_respected(discovered):
    chosen = discovered.select_input_device("VistaCam Pro — Camera Microphone")
    assert chosen is not None
    _rediscover(discovered)
    assert discovered.selected_input.raw_name == CAMERA_RAW
    assert "persisted-identity(explicit)" in discovered.input_reason


def test_virtual_devices_are_only_a_last_resort(discovered):
    only = [_entry(4, "sysdefault", 128, 0), _entry(6, "default", 32, 32)]
    _rediscover(discovered, devices=only)
    assert discovered.selected_input.is_virtual is True
    assert "virtual" in discovered.input_reason


# ═══════════════════════════════════════════════════════════════
# RESOLUTION API
# ═══════════════════════════════════════════════════════════════

def test_select_by_friendly_name_identity_and_index(discovered):
    headset = _by_index(discovered.inputs, 3)
    by_name = discovered.select_input_device(headset.friendly_name)
    assert by_name["hardware_identity"] == headset.hardware_identity
    by_identity = discovered.select_input_device(headset.hardware_identity)
    assert by_identity["runtime_index"] == 3
    by_index = discovered.select_input_device("3")   # internal fallback only
    assert by_index["runtime_index"] == 3


def test_unknown_identifier_is_rejected(discovered):
    before = discovered.selected_input.hardware_identity
    assert discovered.select_input_device("No Such Microphone") is None
    assert discovered.select_output_device("No Such Speaker") is None
    assert discovered.selected_input.hardware_identity == before


def test_failed_probe_rejects_selection(discovered):
    before = discovered.selected_input.hardware_identity
    chosen = discovered.select_input_device(
        "HD-Audio Generic ALC999 — Microphone", probe=lambda dev: False)
    assert chosen is None
    assert discovered.selected_input.hardware_identity == before
    assert _store(discovered)["input"]["hardware_identity"] == before


def test_describe_for_commands_exposes_runtime_device_data(discovered):
    snap = discovered.describe_for_commands()
    assert snap["selected"]["input"] == "Aurora Headset — Microphone"
    assert snap["selected"]["output"] == "Aurora Headset — Headphones"
    assert snap["selected"]["camera"] == "VistaCam Pro"
    assert snap["selected"]["input_reason"]
    mic = next(d for d in snap["inputs"]
               if d["friendly_name"] == "Aurora Headset — Microphone")
    assert "microphone" in mic["tags"] and "external" in mic["tags"]
    cam_mic = next(d for d in snap["inputs"]
                   if d["is_camera_associated_audio"])
    assert "camera microphone" in cam_mic["tags"]
    headphones = next(d for d in snap["outputs"]
                      if d["friendly_name"] == "Aurora Headset — Headphones")
    assert "headphones" in headphones["tags"]


def test_module_helpers_resolve_identity_from_plain_enumerations(tmp_path):
    path = tmp_path / "hardware_selection.json"
    payload = rd.record_active_audio(rd.INPUT_SLOT, HEADSET_RAW,
                                     runtime_index=3, source="explicit",
                                     path=path, meta=fake_meta(HEADSET_RAW))
    assert payload["hardware_identity"].startswith("usb:0d8c:0012")
    # The same hardware re-appears at a DIFFERENT runtime index.
    devices = [{"index": 12, "name": HEADSET_RAW, "hostapi": "ALSA",
                "max_input_channels": 1, "max_output_channels": 2,
                "default_samplerate": 48000.0}]
    hit = rd.resolve_persisted_index(rd.INPUT_SLOT, devices,
                                     meta_provider=fake_meta, path=path)
    assert hit is not None
    assert hit[0] == 12
    assert "persisted-identity(explicit)" in hit[1]
    # Disconnected hardware resolves to nothing (never to a stale index).
    assert rd.resolve_persisted_index(rd.INPUT_SLOT, [],
                                      meta_provider=fake_meta,
                                      path=path) is None
