"""Runtime hardware device discovery + name-based selection for Diego.

ONE canonical runtime device layer for audio input / audio output / camera:

  * Enumerates the hardware that is ACTUALLY connected (never hardcoded
    product names, never hardcoded device indices).
  * Normalizes raw PortAudio/ALSA/V4L2 metadata into deterministic,
    human-friendly runtime names ("<product> — Microphone", …).
  * Classifies devices generically: PHYSICAL / VIRTUAL / MONITOR /
    LOOPBACK / WRAPPER / UNKNOWN (structural rules, not brand blacklists).
  * Detects camera-associated microphones and hardware families from
    runtime metadata (USB topology / vid:pid / product identity).
  * Builds stable HARDWARE IDENTITIES that survive reboot, USB reconnect,
    PipeWire restart and ALSA renumbering. The runtime index is only a
    transient optimization — never the canonical identity.
  * Persists LOGICAL identities in data/hardware_selection.json and
    re-resolves them to whatever runtime index the hardware now has.
  * Resolves input / output / camera COMPLETELY INDEPENDENTLY.

Advanced Linux metadata (/sys/class/sound, /sys/bus/usb, /proc/asound,
/sys/class/video4linux, udev integration semantics) is best-effort ONLY:
when it is unavailable everything still works from plain sounddevice /
PortAudio metadata, so the existing AudioBackend keeps functioning.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SELECTION_PATH = DATA_DIR / "hardware_selection.json"

INPUT_SLOT = "input"
OUTPUT_SLOT = "output"
CAMERA_SLOT = "camera"

# ═══════════════════════════════════════════════════════════════
# Generic structural vocabularies (transport/plugin concepts only —
# NEVER product or vendor names).
# ═══════════════════════════════════════════════════════════════

# Whole-name virtual bus/pseudo devices.
VIRTUAL_EXACT_TOKENS = frozenset({
    "default", "sysdefault", "pulse", "pipewire", "pipewire-pulse",
    "wireplumber", "dummy", "null", "virtual", "monitor", "combine",
})
# ALSA plugin / software-conversion wrappers (substring cues).
WRAPPER_SUBSTRINGS = (
    "lavrate", "samplerate", "speexrate", "speex", "upmix", "vdownmix",
    "dmix", "dsnoop", "dshare", "plughw", "shm", "jack", "oss", "a52",
    "softvol", "equal", "ladspa",
)
MONITOR_SUBSTRINGS = ("monitor",)
LOOPBACK_SUBSTRINGS = ("loopback",)
# Display (HDMI/DP) audio endpoints: they validate but usually have no
# speaker attached, so they rank below real speakers/headphones.
DISPLAY_AUDIO_CUES = ("hdmi", "displayport", "dp,", "spdif", "iec958")
# Camera-associated audio cues (a lens+mic combo unit).
CAMERA_AUDIO_CUES = ("uvc", "webcam", "camera", "video", "cam ")
# Generic detail words stripped when composing a friendly name.
GENERIC_DETAIL_TOKENS = frozenset({
    "usb audio", "audio", "usb", "analog", "digital", "stereo", "mono",
    "mic", "microphone", "speaker", "speakers", "headphone", "headphones",
})
# Internal-attached buses (udev 65-integration.rules semantics).
INTERNAL_BUSES = frozenset({"pci", "platform", "acpi", "i8042", "i2c",
                            "rmi", "spi", "isa"})

_ALSA_CARD_RE = re.compile(r"\(hw\s*:\s*(\d+)\s*,\s*(\d+)\)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_USB_MODALIAS_RE = re.compile(r"usb:v([0-9A-Fa-f]{4})p([0-9A-Fa-f]{4})")

SYSFS_SOUND_ROOT = Path("/sys/class/sound")
SYSFS_USB_ROOT = Path("/sys/bus/usb/devices")
SYSFS_VIDEO_ROOT = Path("/sys/class/video4linux")
PROC_ASOUND_ROOT = Path("/proc/asound")

# Metadata cache (short TTL; reconnects must be visible quickly).
_METADATA_TTL_S = 20.0
_meta_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_meta_lock = threading.Lock()


def reset_metadata_cache() -> None:
    """Drop cached sysfs/procfs metadata (used after hot-plug events/tests)."""
    with _meta_lock:
        _meta_cache.clear()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _meta_lock:
        hit = _meta_cache.get(key)
        if hit and (time.time() - hit[0]) < _METADATA_TTL_S:
            return dict(hit[1])
        if hit:
            _meta_cache.pop(key, None)
    return None


def _cache_put(key: str, value: Dict[str, Any]) -> None:
    with _meta_lock:
        _meta_cache[key] = (time.time(), dict(value))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════
# Name normalization helpers
# ═══════════════════════════════════════════════════════════════

def normalize_raw_name(raw: str) -> str:
    """Lowercase alphanumeric normalization (identity/family matching)."""
    return _NON_ALNUM_RE.sub(" ", (raw or "").strip().lower()).strip()


def strip_alsa_suffix(name: str) -> str:
    """Remove the volatile "(hw:C,D)" suffix so identities survive
    ALSA renumbering, and collapse whitespace."""
    base = _ALSA_CARD_RE.sub("", name or "")
    base = re.sub(r"\s{2,}", " ", base).strip(" :,;-")
    return base.strip()


def extract_alsa_card(raw: str) -> Tuple[Optional[int], Optional[int]]:
    m = _ALSA_CARD_RE.search(raw or "")
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None, None


def split_product_detail(name: str) -> Tuple[str, str]:
    """'PRODUCT: detail (hw:c,d)' → ('PRODUCT', 'detail')."""
    base = strip_alsa_suffix(name or "")
    if ":" in base:
        product, detail = base.split(":", 1)
        return product.strip(), detail.strip()
    return base.strip(), ""


def clean_detail(detail: str, base: str = "") -> str:
    """Keep only the informative part of a device detail string.

    'CODEC Analog' → 'CODEC'; 'USB Audio' → ''; 'HDMI 0' → 'HDMI 0'.
    """
    text = (detail or "").strip()
    if not text:
        return ""
    low = text.lower()
    if low in GENERIC_DETAIL_TOKENS:
        return ""
    if base and (low == base.lower() or low in base.lower()):
        return ""
    # Drop trailing generic words ("CODEC Analog" → "CODEC").
    words = text.split()
    while words and " ".join(words[-2:]).lower() in GENERIC_DETAIL_TOKENS:
        words = words[:-2]
    while words and words[-1].lower() in GENERIC_DETAIL_TOKENS:
        words = words[:-1]
    cleaned = " ".join(words).strip(" ,:-")
    if not cleaned:
        return ""
    if base and cleaned.lower() in base.lower():
        return ""
    return cleaned


def humanize_token(token: str) -> str:
    """Make an underscore/kebab sysfs token readable WITHOUT re-casing
    vendor-provided strings (deterministic, no brand knowledge)."""
    text = (token or "").strip()
    if not text:
        return ""
    if "_" in text or text.islower():
        spaced = text.replace("_", " ").replace("-", " ")
        spaced = re.sub(r"\s{2,}", " ", spaced).strip()
        return " ".join(w[:1].upper() + w[1:] for w in spaced.split())
    return text


def friendly_role(direction: str, *, is_camera_assoc: bool = False,
                  is_display: bool = False, is_internal: bool = False) -> str:
    """Deterministic role suffix derived from runtime classification."""
    if is_display:
        return "Display Audio"
    if is_camera_assoc:
        return "Camera Microphone"
    if direction == "input":
        return "Microphone"
    if direction == "output":
        return "Speakers" if is_internal else "Headphones"
    return "Audio"


def compose_friendly_name(base: str, detail: str, role: str) -> str:
    """'<base>[ <detail>] — <role>' (deterministic; no hardcoded names)."""
    base = (base or "").strip() or "Audio Device"
    detail = (detail or "").strip()
    label = f"{base} {detail}".strip() if detail else base
    return f"{label} — {role}" if role else label


# ═══════════════════════════════════════════════════════════════
# Best-effort Linux metadata (sysfs / procfs). NEVER mandatory.
# ═══════════════════════════════════════════════════════════════

def _usb_device_dir_for(sysfs_path: str) -> Optional[Path]:
    """Walk up from a sysfs device/interface dir to its USB device dir."""
    try:
        p = Path(sysfs_path)
        for _ in range(5):
            if (p / "idVendor").exists():
                return p
            if p.parent == p:
                break
            p = p.parent
    except Exception:
        return None
    return None


def read_usb_metadata(sysfs_path: str) -> Dict[str, Any]:
    """USB descriptor metadata for a sysfs device path (best effort)."""
    if not sysfs_path:
        return {}
    cache_key = f"usb:{sysfs_path}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    meta: Dict[str, Any] = {}
    try:
        dev_dir = _usb_device_dir_for(str(Path(sysfs_path).resolve()))
        if dev_dir is not None:
            vid = _read_text(dev_dir / "idVendor").lower()
            pid = _read_text(dev_dir / "idProduct").lower()
            meta["bus"] = "usb"
            meta["usb_dir"] = dev_dir.name
            meta["usb_vid"] = vid
            meta["usb_pid"] = pid
            meta["usb_id"] = f"{vid}:{pid}" if vid and pid else ""
            meta["usb_manufacturer"] = _read_text(dev_dir / "manufacturer")
            meta["usb_product"] = _read_text(dev_dir / "product")
            meta["usb_serial"] = _read_text(dev_dir / "serial")
            meta["removable"] = _read_text(dev_dir / "removable").lower()
            meta["maxchild"] = _read_text(dev_dir / "maxchild")
            meta["devpath"] = _read_text(dev_dir / "devpath")
            meta["busnum"] = _read_text(dev_dir / "busnum")
            meta["speed"] = _read_text(dev_dir / "speed")
    except Exception as e:
        logger.debug("[DEVICE-DISCOVERY] USB metadata unavailable for %s: %s",
                     sysfs_path, e)
    _cache_put(cache_key, meta)
    return dict(meta)


def integration_from_metadata(meta: Dict[str, Any]) -> str:
    """udev ID_INTEGRATION semantics from sysfs only (no subprocess).

    internal → laptop-attached hardware; external → pluggable hardware;
    unknown → metadata unavailable (caller falls back to name cues).
    """
    bus = str(meta.get("bus") or "").lower()
    if bus == "usb":
        removable = str(meta.get("removable") or "").lower()
        if removable == "fixed":
            return "internal"
        if removable in ("removable", "unknown"):
            return "external"
        return "unknown"
    if bus in INTERNAL_BUSES:
        return "internal"
    explicit = str(meta.get("integration") or "").lower()
    if explicit in ("internal", "external"):
        return explicit
    return "unknown"


def read_alsa_card_metadata(card: Optional[int]) -> Dict[str, Any]:
    """ALSA card + USB/PCI metadata for a card number (best effort)."""
    if card is None:
        return {}
    cache_key = f"card:{card}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    meta: Dict[str, Any] = {"card": card}
    try:
        card_dir = SYSFS_SOUND_ROOT / f"card{card}"
        if card_dir.exists():
            meta["card_id"] = _read_text(card_dir / "id")
            try:
                link = os.readlink(str(card_dir / "device"))
                meta["device_link"] = link
                meta["device_dir"] = str((card_dir / "device").resolve())
            except OSError:
                pass
            uevent = _read_text(card_dir / "device" / "uevent")
            for line in uevent.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[f"uevent_{k.strip()}"] = v.strip()
            modalias = str(meta.get("uevent_MODALIAS", ""))
            m = _USB_MODALIAS_RE.search(modalias)
            if m:
                meta.setdefault("usb_vid", m.group(1).lower())
                meta.setdefault("usb_pid", m.group(2).lower())
                meta["usb_id"] = (f"{meta['usb_vid']}:{meta['usb_pid']}")
            pci_id = str(meta.get("uevent_PCI_ID", "")).strip()
            if pci_id:
                meta["bus"] = "pci"
                meta["pci_id"] = pci_id.lower()
            if meta.get("device_dir"):
                usb = read_usb_metadata(meta["device_dir"])
                if usb:
                    for k, v in usb.items():
                        meta.setdefault(k, v)
        for fname, key in (("usbid", "asound_usbid"), ("id", "asound_id"),
                           ("stream0", "asound_stream0")):
            text = _read_text(PROC_ASOUND_ROOT / f"card{card}" / fname)
            if text:
                meta[key] = text[:2000]
        if meta.get("asound_usbid"):
            m = re.match(r"\s*([0-9a-fA-F]{4}):([0-9a-fA-F]{4})",
                         str(meta["asound_usbid"]))
            if m:
                meta.setdefault("usb_vid", m.group(1).lower())
                meta.setdefault("usb_pid", m.group(2).lower())
                meta["usb_id"] = f"{meta['usb_vid']}:{meta['usb_pid']}"
                meta.setdefault("bus", "usb")
        meta["integration"] = integration_from_metadata(meta)
    except Exception as e:
        logger.debug("[DEVICE-DISCOVERY] ALSA metadata unavailable for card %s: %s",
                     card, e)
    _cache_put(cache_key, meta)
    return dict(meta)


def read_video_metadata(index: int) -> Dict[str, Any]:
    """V4L2 device metadata (name + USB topology), best effort."""
    cache_key = f"video:{index}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    meta: Dict[str, Any] = {"index": index}
    try:
        vdir = SYSFS_VIDEO_ROOT / f"video{index}"
        if vdir.exists():
            meta["v4l_name"] = _read_text(vdir / "name")
            try:
                link = os.readlink(str(vdir / "device"))
                meta["device_link"] = link
                meta["device_dir"] = str((vdir / "device").resolve())
            except OSError:
                pass
            if meta.get("device_dir"):
                usb = read_usb_metadata(meta["device_dir"])
                if usb:
                    for k, v in usb.items():
                        meta.setdefault(k, v)
            meta["integration"] = integration_from_metadata(meta)
    except Exception as e:
        logger.debug("[DEVICE-DISCOVERY] video metadata unavailable for %s: %s",
                     index, e)
    _cache_put(cache_key, meta)
    return dict(meta)


# ═══════════════════════════════════════════════════════════════
# Generic classification (PHYSICAL / VIRTUAL / MONITOR / LOOPBACK /
# WRAPPER / UNKNOWN) — structural rules, no product blacklists.
# ═══════════════════════════════════════════════════════════════

def classify_audio_name(raw: str, host_api: str = "",
                        in_ch: int = 0, out_ch: int = 0) -> Dict[str, Any]:
    """Classify one PortAudio device entry from its runtime metadata.

    Structural rules (work on ANY machine / ANY hardware names):
      * ALSA concrete hardware always carries a "(hw:C,D)" token; every
        other ALSA entry is a plugin/route (sysdefault, pulse, default,
        dmix, lavrate, …) → VIRTUAL/WRAPPER.
      * monitor / loopback names are capture-of-playback, never a mic.
      * display (HDMI/DP) endpoints are real hardware but rank last for
        speech playback.
    """
    lname = (raw or "").strip().lower()
    lhost = (host_api or "").strip().lower()
    has_hw = "(hw:" in lname
    is_alsa = lhost.startswith("alsa") or has_hw

    is_monitor = any(s in lname for s in MONITOR_SUBSTRINGS)
    is_loopback = any(s in lname for s in LOOPBACK_SUBSTRINGS)
    is_wrapper = any(s in lname for s in WRAPPER_SUBSTRINGS)
    is_exact_virtual = lname.strip() in VIRTUAL_EXACT_TOKENS
    is_route = any(k in lname for k in ("default", "pulse", "pipewire",
                                        "wireplumber"))
    is_display = any(c in lname for c in DISPLAY_AUDIO_CUES)

    is_virtual = bool(
        is_monitor or is_loopback or is_wrapper or is_exact_virtual
        or (is_alsa and not has_hw)
        or (is_route and not has_hw)
        or "virtual" in lname or "dummy" in lname or "null" in lname
    )
    if is_monitor:
        category = "MONITOR"
    elif is_loopback:
        category = "LOOPBACK"
    elif is_wrapper:
        category = "WRAPPER"
    elif is_virtual:
        category = "VIRTUAL"
    elif has_hw or not is_alsa:
        category = "PHYSICAL"
    else:
        category = "UNKNOWN"

    # PipeWire hides physical codecs behind output-only enumeration
    # entries: "(hw:C,D)" + not display audio ⇒ capture works BY NAME.
    capture_by_name = (
        int(in_ch) <= 0 and has_hw and not is_display and not is_virtual
    )
    # Mirror case: a physical codec enumerated as capture-only can still
    # PLAY when addressed by its exact name (speculative — must be
    # validated by a real open before it is used for playback).
    playback_by_name = (
        int(out_ch) <= 0 and has_hw and not is_display and not is_virtual
    )
    return {
        "is_virtual": is_virtual,
        "is_physical": category == "PHYSICAL",
        "category": category,
        "is_monitor": is_monitor,
        "is_loopback": is_loopback,
        "is_wrapper": is_wrapper,
        "is_display_audio": is_display,
        "capture_by_name": capture_by_name,
        "playback_by_name": playback_by_name,
    }


@dataclass
class CameraIdentity:
    """Runtime identity of a camera, used to detect its bundled mic."""
    name: str = ""
    product: str = ""
    usb_dir: str = ""
    usb_id: str = ""
    name_token: str = ""


@dataclass
class CameraIdentitySet:
    identities: List[CameraIdentity] = field(default_factory=list)

    @property
    def usb_dirs(self) -> set:
        return {i.usb_dir for i in self.identities if i.usb_dir}

    @property
    def usb_ids(self) -> set:
        return {i.usb_id for i in self.identities if i.usb_id}

    @property
    def name_tokens(self) -> set:
        return {i.name_token for i in self.identities if len(i.name_token) >= 4}


def collect_camera_identities() -> CameraIdentitySet:
    """Camera identities of the currently connected cameras (no probing)."""
    out = CameraIdentitySet()
    try:
        from auth import camera_selector as cs
        for dev in cs.get_selector().enumerate_devices():
            meta = read_video_metadata(int(dev.index))
            product, _detail = split_product_detail(dev.name or "")
            out.identities.append(CameraIdentity(
                name=dev.name or "",
                product=product or (dev.name or ""),
                usb_dir=str(meta.get("usb_dir") or ""),
                usb_id=str(meta.get("usb_id") or ""),
                name_token=normalize_raw_name(product or dev.name or ""),
            ))
    except Exception as e:
        logger.debug("[DEVICE-DISCOVERY] camera identities unavailable: %s", e)
    return out


def detect_camera_associated_audio(raw: str, meta: Optional[Dict[str, Any]] = None,
                                   cameras: Optional[CameraIdentitySet] = None,
                                   ) -> bool:
    """Is this audio device the microphone bundled inside a camera?

    Detected from runtime metadata only (no product hardcodes):
      1. same USB device node (e.g. 1-2) as an enumerated camera, or
      2. same USB vid:pid as an enumerated camera, or
      3. the camera product token appears in the audio device name, or
      4. explicit camera/UVC/webcam cues in the audio device name.
    """
    meta = meta or {}
    usb_dir = str(meta.get("usb_dir") or "")
    usb_id = str(meta.get("usb_id") or "")
    if cameras is not None:
        if usb_dir and usb_dir in cameras.usb_dirs:
            return True
        if usb_id and usb_id in cameras.usb_ids:
            return True
        norm = normalize_raw_name(strip_alsa_suffix(raw or ""))
        for token in cameras.name_tokens:
            if token and token in norm:
                return True
    lname = (raw or "").lower()
    return any(c in lname for c in CAMERA_AUDIO_CUES)


# ═══════════════════════════════════════════════════════════════
# Hardware identity + family (index-independent, deterministic)
# ═══════════════════════════════════════════════════════════════

def _digest(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def hardware_identity_for_audio(raw: str,
                                meta: Optional[Dict[str, Any]] = None) -> str:
    """Stable logical identity for one audio endpoint.

    Strongest available metadata wins: USB vid:pid(+serial) → ALSA card id
    → PCI id → normalized name. The volatile "(hw:C,D)" suffix and the
    runtime index are NEVER part of the identity.
    """
    meta = meta or {}
    vid = str(meta.get("usb_vid") or "").lower()
    pid = str(meta.get("usb_pid") or "").lower()
    serial = normalize_raw_name(str(meta.get("usb_serial") or ""))
    if vid and pid:
        prefix = f"usb:{vid}:{pid}" + (f":{serial}" if serial else "")
    elif meta.get("card_id") or meta.get("asound_id"):
        card_id = str(meta.get("card_id") or meta.get("asound_id"))
        prefix = f"alsa:{normalize_raw_name(card_id)}"
    elif meta.get("pci_id"):
        prefix = f"pci:{normalize_raw_name(str(meta['pci_id']))}"
    else:
        prefix = "audio"
    return f"{prefix}|{_digest(normalize_raw_name(strip_alsa_suffix(raw)))}"


def family_id_for_audio(raw: str,
                        meta: Optional[Dict[str, Any]] = None) -> str:
    """Physical hardware family shared by a device's input+output sides."""
    meta = meta or {}
    vid = str(meta.get("usb_vid") or "").lower()
    pid = str(meta.get("usb_pid") or "").lower()
    if vid and pid:
        return f"family:usb:{vid}:{pid}"
    product, _detail = split_product_detail(raw or "")
    base = str(meta.get("usb_product") or product or strip_alsa_suffix(raw or ""))
    return f"family:{_digest(normalize_raw_name(base))}"


def audio_metadata_fields(raw: str, meta: Dict[str, Any]) -> Dict[str, str]:
    """manufacturer / model derived from runtime metadata (best effort)."""
    manufacturer = (str(meta.get("usb_manufacturer") or "").strip()
                    or str(meta.get("uevent_DRIVER") or "").strip())
    model = (str(meta.get("usb_product") or "").strip()
             or str(meta.get("card_id") or "").strip())
    if not model:
        model = split_product_detail(raw or "")[0]
    return {"manufacturer": manufacturer, "model": model}


def audio_base_token(raw: str, meta: Dict[str, Any]) -> str:
    """Human-readable product token (hardware metadata first, raw last)."""
    product, _detail = split_product_detail(raw or "")
    candidates = (str(meta.get("usb_product") or "").strip(),
                  product.strip(),
                  str(meta.get("card_id") or "").strip(),
                  strip_alsa_suffix(raw or "").strip())
    for candidate in candidates:
        if candidate:
            return humanize_token(candidate)
    return "Audio Device"


def camera_friendly_name(raw_name: str, by_id: str = "") -> str:
    """Friendly camera label from runtime metadata (no alias tables)."""
    product, _detail = split_product_detail(raw_name or "")
    if product:
        return product.strip()
    if (raw_name or "").strip():
        return raw_name.strip()
    if by_id:
        stem = re.sub(r"-video-index\d*$", "", Path(by_id).name)
        stem = re.sub(r"^usb-", "", stem)
        return humanize_token(stem) or "Camera"
    return "Camera"


def camera_hardware_identity(raw_name: str, by_id: str = "", usb_id: str = "",
                             usb_serial: str = "", usb_path: str = "") -> str:
    """Stable camera identity: by-id symlink → USB vid:pid(+serial) → name."""
    if by_id:
        return by_id.strip()
    if usb_id:
        serial = normalize_raw_name(usb_serial)
        return f"usb:{usb_id}" + (f":{serial}" if serial else "")
    if usb_path:
        return f"camusb:{_digest(normalize_raw_name(f'{raw_name} {usb_path}'))}"
    return f"cam:{_digest(normalize_raw_name(raw_name))}"


def classify_camera_external(raw_name: str, by_id: str = "", usb_path: str = "",
                             integration: str = "") -> bool:
    """Generic external/internal camera classification (no brand lists)."""
    integ = (integration or "").strip().lower()
    if integ == "external":
        return True
    if integ == "internal":
        return False
    blob = f"{raw_name} {by_id} {usb_path}".lower()
    if any(c in blob for c in ("integrated", "internal", "built-in",
                               "builtin", "uvc webcam", "hd uvc")):
        return False
    if "external" in blob:
        return True
    return bool((usb_path or "").strip() or (by_id or "").strip())


# ═══════════════════════════════════════════════════════════════
# Normalized runtime device model
# ═══════════════════════════════════════════════════════════════

@dataclass
class RuntimeDevice:
    """Normalized runtime view of ONE hardware endpoint.

    `runtime_index` is transient (it changes on reboot/reconnect); the
    canonical identity is `hardware_identity`.
    """
    runtime_index: Optional[int]
    raw_name: str
    direction: str = ""                 # input | output | camera
    device_type: str = "audio"          # audio | camera
    normalized_name: str = ""
    friendly_name: str = ""
    host_api: str = ""
    channels: int = 0
    sample_rate: float = 0.0
    category: str = "UNKNOWN"           # PHYSICAL|VIRTUAL|MONITOR|LOOPBACK|WRAPPER|UNKNOWN
    is_physical: bool = False
    is_virtual: bool = True
    is_usb: bool = False
    is_camera: bool = False
    is_camera_associated_audio: bool = False
    is_display_audio: bool = False
    is_internal: bool = False
    integration: str = "unknown"        # internal | external | unknown
    manufacturer: str = ""
    model: str = ""
    hardware_identity: str = ""
    family_id: str = ""
    alsa_card: Optional[int] = None
    reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in (
            "runtime_index", "raw_name", "direction", "device_type",
            "normalized_name", "friendly_name", "host_api", "channels",
            "sample_rate", "category", "is_physical", "is_virtual", "is_usb",
            "is_camera", "is_camera_associated_audio", "is_display_audio",
            "is_internal", "integration", "manufacturer", "model",
            "hardware_identity", "family_id", "alsa_card", "reason")}
        out["extra"] = dict(self.extra)
        return out


def build_audio_views(index: Optional[int], raw: str, host_api: str = "",
                      in_ch: int = 0, out_ch: int = 0, sample_rate: float = 0.0,
                      *, meta: Optional[Dict[str, Any]] = None,
                      is_default: bool = False,
                      cameras: Optional[CameraIdentitySet] = None,
                      ) -> Tuple[Optional[RuntimeDevice], Optional[RuntimeDevice]]:
    """Build the INDEPENDENT input and output views of one PortAudio entry.

    A duplex codec yields BOTH views (same hardware identity + family, two
    direction-specific friendly names) so input and output can be selected
    completely independently.
    """
    raw = raw or ""
    card, _sub = extract_alsa_card(raw)
    if meta is None:
        meta = read_alsa_card_metadata(card)
    cls = classify_audio_name(raw, host_api, in_ch, out_ch)
    integration = str(meta.get("integration")
                      or integration_from_metadata(meta) or "unknown")
    is_usb = (str(meta.get("bus") or "").lower() == "usb"
              or bool(meta.get("usb_id")) or "usb" in raw.lower())
    is_display = bool(cls["is_display_audio"])
    # "Internal" requires POSITIVE evidence (bus metadata). With no metadata
    # we stay neutral instead of guessing laptop-attached hardware.
    is_internal = (integration == "internal")
    cam_assoc = detect_camera_associated_audio(raw, meta, cameras)
    base = audio_base_token(raw, meta)
    detail = clean_detail(split_product_detail(raw)[1], base)
    fields = audio_metadata_fields(raw, meta)
    identity = hardware_identity_for_audio(raw, meta)
    family = family_id_for_audio(raw, meta)

    def _make(direction: str, channels: int) -> RuntimeDevice:
        assoc = cam_assoc if direction == "input" else False
        role = friendly_role(direction, is_camera_assoc=assoc,
                             is_display=is_display, is_internal=is_internal)
        friendly = (compose_friendly_name(base, detail, role)
                    if cls["is_physical"] else (strip_alsa_suffix(raw) or raw))
        return RuntimeDevice(
            runtime_index=index, raw_name=raw, direction=direction,
            device_type="audio", normalized_name=normalize_raw_name(raw),
            friendly_name=friendly, host_api=host_api or "",
            channels=int(channels), sample_rate=float(sample_rate or 0.0),
            category=cls["category"], is_physical=bool(cls["is_physical"]),
            is_virtual=bool(cls["is_virtual"]), is_usb=is_usb, is_camera=False,
            is_camera_associated_audio=assoc, is_display_audio=is_display,
            is_internal=is_internal, integration=integration,
            manufacturer=fields["manufacturer"], model=fields["model"],
            hardware_identity=identity, family_id=family, alsa_card=card,
            extra={"is_default": bool(is_default),
                   "max_input_channels": int(in_ch),
                   "max_output_channels": int(out_ch),
                   "capture_by_name": bool(cls["capture_by_name"]),
                   "playback_by_name": bool(cls["playback_by_name"]),
                   "alsa_subdevice": _sub},
        )

    input_view = None
    if int(in_ch) > 0 or cls["capture_by_name"]:
        input_view = _make("input", max(int(in_ch), 1))
    output_view = None
    # A camera-bundled audio function is a MICROPHONE: never invent a
    # speculative playback endpoint for a camera unit.
    if int(out_ch) > 0 or (cls["playback_by_name"] and not cam_assoc):
        output_view = _make("output", max(int(out_ch), 1))
    return input_view, output_view


def is_os_default_route(dev: RuntimeDevice) -> bool:
    """True for the OS default audio route (follows the desktop's sink)."""
    return normalize_raw_name(strip_alsa_suffix(dev.raw_name)) in {
        "default", "pulse", "pipewire", "pipewire pulse", "wireplumber"}


def rank_key(dev: RuntimeDevice, direction: str) -> tuple:
    """Automatic-selection ranking (lower = better), per direction.

    Input order:  physical → dedicated (non camera-mic) → external →
    unknown → internal → virtual.

    Output order: physical speakers/headphones → speculative play-by-name
    codec → the OS default route → display (HDMI/DP) audio → other
    virtual/wrapper devices. Display audio ranks BELOW the OS default
    route because a monitor jack usually has no speakers attached and
    would silently swallow speech (the same heuristic the existing
    DeviceManager applies).

    Runtime indices NEVER influence ranking.
    """
    cam_pen = 5 if (direction == "input" and dev.is_camera_associated_audio) else 0
    # Speculative playback (physical codec enumerated without output
    # channels): only usable when a real open/validation pass accepts it.
    spec_pen = 3 if (direction == "output"
                     and dev.extra.get("playback_by_name")) else 0
    if direction == "output":
        if not dev.is_physical:
            cls_pen = 5 if is_os_default_route(dev) else 9
        elif dev.is_display_audio:
            cls_pen = 6
        else:
            cls_pen = 0
    else:
        cls_pen = 0 if dev.is_physical else 9
    if not dev.is_physical:
        integ_pen = 3
    else:
        integ_pen = {"external": 0, "unknown": 1, "internal": 2}.get(
            dev.integration or ("external" if dev.is_usb else "unknown"), 1)
    return (cls_pen, cam_pen, spec_pen, integ_pen,
            (dev.friendly_name or "").lower())


# ═══════════════════════════════════════════════════════════════
# Persistence — LOGICAL identities (runtime index is only a hint)
# ═══════════════════════════════════════════════════════════════

def load_hardware_store(path: Path = SELECTION_PATH) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.debug("[DEVICE-SELECT] hardware store load failed: %s", e)
    return {}


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_hardware_slot(slot: str, payload: Dict[str, Any],
                       path: Path = SELECTION_PATH) -> None:
    """Persist ONE slot (input|output|camera); other slots are untouched."""
    store = load_hardware_store(path)
    store[slot] = dict(payload)
    try:
        _atomic_write(path, store)
    except Exception as e:
        logger.warning("[DEVICE-SELECT] could not persist %s slot: %s", slot, e)


def identity_payload_for_audio(raw_name: str, direction: str,
                               runtime_index: Optional[int] = None,
                               meta: Optional[Dict[str, Any]] = None,
                               source: str = "auto",
                               cameras: Optional[CameraIdentitySet] = None,
                               ) -> Dict[str, Any]:
    """Logical identity payload for a raw audio device name."""
    card, _sub = extract_alsa_card(raw_name or "")
    if meta is None:
        meta = read_alsa_card_metadata(card)
    in_view, out_view = build_audio_views(
        runtime_index, raw_name or "", "",
        in_ch=1 if direction == "input" else 0,
        out_ch=1 if direction == "output" else 0,
        meta=meta, cameras=cameras)
    dev = in_view if direction == "input" else (out_view or in_view)
    if dev is None:
        return {}
    return {
        "hardware_identity": dev.hardware_identity,
        "friendly_name": dev.friendly_name,
        "manufacturer": dev.manufacturer,
        "model": dev.model,
        "family_id": dev.family_id,
        "raw_name": dev.raw_name,
        "category": dev.category,
        "is_camera_associated_audio": dev.is_camera_associated_audio,
        "runtime_index": runtime_index,   # transient hint only
        "source": source,                 # "explicit" | "auto"
        "saved_at": time.time(),
    }


def record_active_audio(slot: str, raw_name: str,
                        runtime_index: Optional[int] = None,
                        source: str = "auto",
                        path: Path = SELECTION_PATH,
                        meta: Optional[Dict[str, Any]] = None,
                        ) -> Optional[Dict[str, Any]]:
    """Mirror a live/verified device into the logical hardware store."""
    if not raw_name:
        return None
    direction = "output" if slot == OUTPUT_SLOT else "input"
    payload = identity_payload_for_audio(raw_name, direction,
                                         runtime_index=runtime_index,
                                         meta=meta, source=source)
    if not payload:
        return None
    # An existing EXPLICIT user choice is sacred: automatic re-verification
    # must NEVER overwrite it — not even when the chosen hardware is currently
    # absent (the discovery slot would otherwise point at a fallback device,
    # and the explicit choice would not be re-selected on reconnect).
    try:
        existing = load_hardware_store(path).get(slot)
        if (source != "explicit" and isinstance(existing, dict)
                and existing.get("source") == "explicit"):
            logger.debug("[DEVICE-SELECT] %s has an explicit user choice — "
                         "auto discovery will not overwrite it", slot)
            return None
    except Exception:
        pass
    save_hardware_slot(slot, payload, path)
    logger.info("[DEVICE-SELECT] %s hardware identity persisted: %r "
                "(identity=%s runtime_index=%s source=%s)", slot,
                payload.get("friendly_name"), payload.get("hardware_identity"),
                runtime_index, source)
    return payload


def _legacy_saved_name(slot: str) -> str:
    """Best-effort migration from the pre-identity stores (name based)."""
    try:
        if slot == INPUT_SLOT:
            for fname, key in (("mic_selection.json", None),
                               ("audio_devices.json", "input_device")):
                p = DATA_DIR / fname
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8"))
                    entry = data.get(key) if key else data
                    if isinstance(entry, dict) and entry.get("name"):
                        return str(entry["name"])
        elif slot == OUTPUT_SLOT:
            p = DATA_DIR / "audio_devices.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                entry = data.get("output_device")
                if isinstance(entry, dict) and entry.get("name"):
                    return str(entry["name"])
    except Exception as e:
        logger.debug("[DEVICE-SELECT] legacy store read failed: %s", e)
    return ""


def resolve_saved_entry_index(slot: str, devices: Iterable[Dict[str, Any]],
                              saved_identity: Optional[str] = None,
                              saved_name: Optional[str] = None,
                              source: str = "auto",
                              meta_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
                              cameras: Optional[CameraIdentitySet] = None,
                              ) -> Optional[Tuple[int, str]]:
    """Map a saved LOGICAL identity onto the CURRENT runtime index.

    Unlike `resolve_persisted_index`, this does NOT read any store file —
    the caller owns the persistence and passes the saved entry's identity
    (canonical, preferred) and/or legacy name. This keeps test-isolated
    stores isolated and lets every subsystem resolve with its OWN store.

    `devices` may carry an enriched "hardware_identity" key (from
    DeviceManager._enrich / the registry); otherwise the identity is
    computed from each device's raw name + Linux metadata.
    Returns (runtime_index, reason) or None when the saved hardware is not
    connected. A saved runtime index is NEVER trusted — only the identity.
    """
    devices = list(devices or [])
    if not devices:
        return None
    provider = meta_provider or (
        lambda raw: read_alsa_card_metadata(extract_alsa_card(raw)[0]))
    saved_identity = str(saved_identity or "")
    if not saved_identity and saved_name:
        saved_identity = hardware_identity_for_audio(
            saved_name, provider(saved_name))
    if not saved_identity:
        return None
    for dev in devices:
        raw = str(dev.get("name") or "")
        index = dev.get("index")
        if index is None or not raw:
            continue
        enriched = dev.get("hardware_identity")
        dev_identity = (str(enriched) if enriched
                        else hardware_identity_for_audio(raw, provider(raw)))
        if dev_identity != saved_identity:
            continue
        in_ch = int(dev.get("max_input_channels", 0) or 0)
        out_ch = int(dev.get("max_output_channels", 0) or 0)
        in_view, out_view = build_audio_views(
            index, raw, str(dev.get("hostapi") or ""), in_ch, out_ch,
            float(dev.get("default_samplerate", 0.0) or 0.0),
            meta=provider(raw), cameras=cameras)
        candidate = in_view if slot == INPUT_SLOT else (out_view or in_view)
        if candidate is None:
            continue
        # A camera-bundled microphone never wins AUTOMATICALLY over a
        # dedicated microphone; an explicit user choice always wins.
        if (slot == INPUT_SLOT and source != "explicit"
                and candidate.is_camera_associated_audio):
            others = [d for d in devices
                      if str(d.get("name") or "") != raw
                      and "(hw:" in str(d.get("name") or "").lower()]
            if others:
                logger.info("[DEVICE-SELECT] persisted input %r is a "
                            "camera-associated microphone — deferring to "
                            "dedicated hardware", candidate.friendly_name)
                continue
        logger.info("[DEVICE-SELECT] persisted %s identity %r resolved to "
                    "runtime index %s (%s)", slot, saved_identity, index,
                    candidate.friendly_name)
        return int(index), f"persisted-identity({source})"
    logger.info("[DEVICE-SELECT] persisted %s identity %r is not connected",
                slot, saved_identity)
    return None


def resolve_persisted_index(slot: str, devices: Iterable[Dict[str, Any]],
                            meta_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
                            cameras: Optional[CameraIdentitySet] = None,
                            path: Path = SELECTION_PATH,
                            ) -> Optional[Tuple[int, str]]:
    """Map the LOGICAL identity persisted in `path` onto the CURRENT index.

    Thin convenience wrapper over `resolve_saved_entry_index` for callers
    that use the canonical hardware-selection store (the audio pipeline
    mirror / camera path). A saved runtime index is NEVER trusted.
    """
    provider = meta_provider or (
        lambda raw: read_alsa_card_metadata(extract_alsa_card(raw)[0]))
    store = load_hardware_store(path)
    saved = store.get(slot) if isinstance(store.get(slot), dict) else None
    saved_identity = str((saved or {}).get("hardware_identity") or "")
    source = str((saved or {}).get("source") or "auto")
    if not saved_identity:
        legacy_name = _legacy_saved_name(slot)
        if legacy_name:
            saved_identity = hardware_identity_for_audio(
                legacy_name, provider(legacy_name))
            source = "auto"
    if not saved_identity:
        return None
    return resolve_saved_entry_index(slot, devices,
                                     saved_identity=saved_identity,
                                     saved_name=None, source=source,
                                     meta_provider=meta_provider,
                                     cameras=cameras)


# ═══════════════════════════════════════════════════════════════
# Registry — discovery → classification → ranking → selection
# ═══════════════════════════════════════════════════════════════

class DeviceRegistry:
    """Canonical runtime device registry.

    Input, output and camera are enumerated, ranked, validated, selected
    and persisted COMPLETELY INDEPENDENTLY. Selection is by hardware
    identity / friendly name; the runtime index is an internal fallback.
    """

    def __init__(self, selection_path: Path = SELECTION_PATH):
        self.selection_path = selection_path
        self._lock = threading.Lock()
        self.inputs: List[RuntimeDevice] = []
        self.outputs: List[RuntimeDevice] = []
        self.cameras: List[RuntimeDevice] = []
        self.selected_input: Optional[RuntimeDevice] = None
        self.selected_output: Optional[RuntimeDevice] = None
        self.selected_camera: Optional[RuntimeDevice] = None
        self.input_reason = ""
        self.output_reason = ""
        self.camera_reason = ""
        self.discovered_at = 0.0

    # ── Discovery ────────────────────────────────────────────────

    def discover(self, sd_module: Any = None, *, include_cameras: bool = True,
                 meta_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
                 cameras: Optional[List[RuntimeDevice]] = None,
                 camera_identities: Optional[CameraIdentitySet] = None,
                 probe_input: Optional[Callable[[RuntimeDevice], bool]] = None,
                 probe_output: Optional[Callable[[RuntimeDevice], bool]] = None,
                 ) -> Dict[str, Any]:
        """Enumerate the connected hardware and resolve every slot."""
        reset_metadata_cache()
        cam_ids = camera_identities
        if cam_ids is None:
            cam_ids = collect_camera_identities()
        inputs, outputs = self._enumerate_audio(sd_module, meta_provider, cam_ids)
        cam_devices = list(cameras) if cameras is not None else (
            self._enumerate_cameras() if include_cameras else [])
        with self._lock:
            self.inputs = inputs
            self.outputs = outputs
            self.cameras = cam_devices
            self.discovered_at = time.time()
        self._log_discovery()
        self._resolve_selections(probe_input=probe_input,
                                 probe_output=probe_output)
        return {"inputs": list(inputs), "outputs": list(outputs),
                "cameras": list(cam_devices)}

    def _enumerate_audio(self, sd_module: Any,
                         meta_provider: Optional[Callable[[str], Dict[str, Any]]],
                         cam_ids: CameraIdentitySet,
                         ) -> Tuple[List[RuntimeDevice], List[RuntimeDevice]]:
        inputs: List[RuntimeDevice] = []
        outputs: List[RuntimeDevice] = []
        if sd_module is None:
            try:
                import sounddevice as sd_module  # type: ignore
            except Exception as e:
                logger.warning("[DEVICE-DISCOVERY] sounddevice unavailable: %s", e)
                return inputs, outputs
        try:
            hostapis = list(sd_module.query_hostapis())
        except Exception:
            hostapis = []
        try:
            default_in, default_out = sd_module.default.device
        except Exception:
            default_in, default_out = -1, -1
        try:
            entries = list(sd_module.query_devices())
        except Exception as e:
            logger.error("[DEVICE-DISCOVERY] audio enumeration failed: %s", e)
            return inputs, outputs
        for entry in entries:
            try:
                idx = entry.get("index")
                raw = str(entry.get("name", ""))
                in_ch = int(entry.get("max_input_channels", 0) or 0)
                out_ch = int(entry.get("max_output_channels", 0) or 0)
                host = ""
                hidx = entry.get("hostapi", -1)
                if isinstance(hidx, int) and 0 <= hidx < len(hostapis):
                    host = str(hostapis[hidx].get("name", ""))
                if meta_provider is not None:
                    meta = meta_provider(raw) or {}
                else:
                    meta = read_alsa_card_metadata(extract_alsa_card(raw)[0])
                in_view, out_view = build_audio_views(
                    idx, raw, host, in_ch, out_ch,
                    float(entry.get("default_samplerate", 0.0) or 0.0),
                    meta=meta,
                    is_default=(idx == default_in or idx == default_out),
                    cameras=cam_ids)
                if in_view is not None:
                    in_view.extra["is_default"] = bool(idx == default_in)
                    inputs.append(in_view)
                if out_view is not None:
                    out_view.extra["is_default"] = bool(idx == default_out)
                    outputs.append(out_view)
            except Exception as e:
                logger.debug("[DEVICE-DISCOVERY] skipped audio entry %r: %s",
                             entry, e)
        return inputs, outputs

    def _enumerate_cameras(self) -> List[RuntimeDevice]:
        """Cameras that were enumerated AND validated (real frames)."""
        out: List[RuntimeDevice] = []
        try:
            from auth import camera_selector as cs
            working = cs.get_selector().list_working_cameras()
        except Exception as e:
            logger.debug("[DEVICE-DISCOVERY] camera enumeration unavailable: %s", e)
            return out
        for dev in working:
            try:
                meta = read_video_metadata(int(dev.index))
                usb_id = str(meta.get("usb_id") or "")
                serial = str(meta.get("usb_serial") or "")
                integration = str(meta.get("integration") or "")
                external = classify_camera_external(
                    dev.name, dev.by_id_path, dev.usb_path, integration)
                if not integration:
                    external = bool(dev.external)
                label = camera_friendly_name(dev.name, dev.by_id_path)
                identity = camera_hardware_identity(
                    dev.name, dev.by_id_path, usb_id, serial, dev.usb_path)
                out.append(RuntimeDevice(
                    runtime_index=int(dev.index), raw_name=dev.name,
                    direction="camera", device_type="camera",
                    normalized_name=normalize_raw_name(dev.name),
                    friendly_name=label, channels=0,
                    sample_rate=float(dev.fps or 0.0), category="PHYSICAL",
                    is_physical=True, is_virtual=False,
                    is_usb=bool(usb_id or dev.usb_path), is_camera=True,
                    is_internal=(not external),
                    integration=(integration or
                                 ("external" if external else "internal")),
                    manufacturer=str(meta.get("usb_manufacturer") or ""),
                    model=label, hardware_identity=identity,
                    family_id=f"family:{identity}",
                    extra={"path": dev.path, "by_id_path": dev.by_id_path,
                           "usb_path": dev.usb_path, "external": external,
                           "width": dev.width, "height": dev.height,
                           "fps": dev.fps, "usb_id": usb_id},
                ))
            except Exception as e:
                logger.debug("[DEVICE-DISCOVERY] skipped camera %r: %s", dev, e)
        return out

    def _log_discovery(self) -> None:
        for dev in self.inputs:
            log = logger.info if dev.is_physical else logger.debug
            log("[DEVICE-DISCOVERY] audio input: raw_name=%r friendly_name=%r "
                "hardware_identity=%r manufacturer=%r model=%r usb=%s "
                "virtual=%s category=%s integration=%s camera_mic=%s",
                dev.raw_name, dev.friendly_name, dev.hardware_identity,
                dev.manufacturer, dev.model, dev.is_usb, dev.is_virtual,
                dev.category, dev.integration, dev.is_camera_associated_audio)
        for dev in self.outputs:
            log = logger.info if dev.is_physical else logger.debug
            log("[DEVICE-DISCOVERY] audio output: raw_name=%r friendly_name=%r "
                "hardware_identity=%r manufacturer=%r model=%r usb=%s "
                "virtual=%s category=%s integration=%s",
                dev.raw_name, dev.friendly_name, dev.hardware_identity,
                dev.manufacturer, dev.model, dev.is_usb, dev.is_virtual,
                dev.category, dev.integration)
        for cam in self.cameras:
            logger.info("[DEVICE-DISCOVERY] camera: raw_name=%r friendly_name=%r "
                        "hardware_identity=%r external=%s usb=%s",
                        cam.raw_name, cam.friendly_name, cam.hardware_identity,
                        cam.extra.get("external"), cam.is_usb)

    # ── Selection resolution ─────────────────────────────────────

    def _resolve_selections(self, probe_input=None, probe_output=None) -> None:
        store = load_hardware_store(self.selection_path)
        self.selected_input, self.input_reason = self._resolve_slot(
            INPUT_SLOT, self.inputs, "input", store, probe_input)
        self.selected_output, self.output_reason = self._resolve_slot(
            OUTPUT_SLOT, self.outputs, "output", store, probe_output)
        self.selected_camera, self.camera_reason = self._resolve_camera_slot(store)
        # Persist the LOGICAL identity of every resolved slot (the runtime
        # index is stored only as a transient hint). An existing EXPLICIT
        # user choice for the same hardware is never downgraded.
        if self.selected_input is not None:
            self._persist_slot(INPUT_SLOT, self.selected_input,
                               self.selected_input.friendly_name,
                               source="auto")
        if self.selected_output is not None:
            self._persist_slot(OUTPUT_SLOT, self.selected_output,
                               self.selected_output.friendly_name,
                               source="auto")
        if self.selected_camera is not None:
            self._persist_slot(CAMERA_SLOT, self.selected_camera,
                               self.selected_camera.friendly_name,
                               source="auto")
        if self.selected_input is not None:
            logger.info("[DEVICE-SELECT] input=%r reason=%s identity=%s "
                        "runtime_index=%s", self.selected_input.friendly_name,
                        self.input_reason,
                        self.selected_input.hardware_identity,
                        self.selected_input.runtime_index)
        else:
            logger.warning("[DEVICE-SELECT] input=NONE reason=%s",
                           self.input_reason)
        if self.selected_output is not None:
            logger.info("[DEVICE-SELECT] output=%r reason=%s identity=%s "
                        "runtime_index=%s", self.selected_output.friendly_name,
                        self.output_reason,
                        self.selected_output.hardware_identity,
                        self.selected_output.runtime_index)
        else:
            logger.warning("[DEVICE-SELECT] output=NONE reason=%s",
                           self.output_reason)
        if self.selected_camera is not None:
            logger.info("[CAMERA-SELECT] camera=%r reason=%s identity=%s",
                        self.selected_camera.friendly_name, self.camera_reason,
                        self.selected_camera.hardware_identity)
        else:
            logger.info("[CAMERA-SELECT] camera=NONE reason=%s",
                        self.camera_reason)

    def _resolve_slot(self, slot: str, candidates: List[RuntimeDevice],
                      direction: str, store: Dict[str, Any],
                      probe: Optional[Callable[[RuntimeDevice], bool]],
                      ) -> Tuple[Optional[RuntimeDevice], str]:
        """Resolve ONE slot independently (input never touches output)."""
        if not candidates:
            return None, "no-devices-present"
        saved = store.get(slot) if isinstance(store.get(slot), dict) else None
        saved_identity = str((saved or {}).get("hardware_identity") or "")
        source = str((saved or {}).get("source") or "auto")
        ranked = sorted(candidates, key=lambda d: rank_key(d, direction))
        physical = [d for d in ranked if d.is_physical]

        # 1-2. Previously saved LOGICAL hardware identity (index-independent).
        if saved_identity:
            match = next((d for d in candidates
                          if d.hardware_identity == saved_identity), None)
            if match is not None:
                defer = (slot == INPUT_SLOT and source != "explicit"
                         and match.is_camera_associated_audio
                         and any(not d.is_camera_associated_audio
                                 for d in physical))
                speculative = (slot == OUTPUT_SLOT and probe is None
                               and bool(match.extra.get("playback_by_name")))
                if defer:
                    logger.info("[DEVICE-SELECT] saved input %r is a "
                                "camera-associated microphone — preferring "
                                "dedicated hardware", match.friendly_name)
                elif speculative:
                    logger.info("[DEVICE-SELECT] saved output %r exposes no "
                                "enumerated playback channels — needs "
                                "validation, re-ranking", match.friendly_name)
                elif probe is None or self._safe_probe(probe, match):
                    return match, f"persisted-identity({source})"
                else:
                    logger.warning("[DEVICE-SELECT] saved %s %r failed "
                                   "validation — re-ranking", slot,
                                   match.friendly_name)
            else:
                logger.info("[DEVICE-SELECT] saved %s identity %r is not "
                            "connected — re-ranking", slot, saved_identity)
            family = str((saved or {}).get("family_id") or "")
            if family:
                for dev in [d for d in physical if d.family_id == family]:
                    if (direction == "output" and probe is None
                            and dev.extra.get("playback_by_name")):
                        continue
                    # Never re-pick a camera-bundled microphone that was
                    # deferred above in favour of dedicated hardware.
                    if (slot == INPUT_SLOT and source != "explicit"
                            and dev.is_camera_associated_audio
                            and any(not d.is_camera_associated_audio
                                    for d in physical)):
                        continue
                    if probe is None or self._safe_probe(probe, dev):
                        return dev, "persisted-family-fallback"

        # 3-7. Ranked automatic selection (rank_key encodes the preference:
        # physical → dedicated → external → OS default route → display →
        # virtual). Candidates that fail validation are skipped, so the
        # next-ranked device wins.
        for dev in ranked:
            if (direction == "output" and probe is None
                    and dev.extra.get("playback_by_name")):
                # Speculative playback endpoint: never auto-selected without
                # a real open/validation pass.
                continue
            if probe is not None and not self._safe_probe(probe, dev):
                continue
            if not dev.is_physical:
                if is_os_default_route(dev):
                    reason = ("auto-fallback-virtual-os-default-route(follows "
                              "the desktop default sink)")
                else:
                    reason = "auto-fallback-virtual(no physical hardware present)"
            elif direction == "input" and dev.is_camera_associated_audio:
                reason = ("auto-physical(camera-associated microphone — no "
                          "dedicated microphone present)")
            elif dev.integration == "external":
                reason = "auto-physical-external(dedicated hardware)"
            elif dev.integration == "internal":
                reason = "auto-physical-internal(built-in hardware)"
            else:
                reason = "auto-physical(best ranked hardware)"
            return dev, reason
        return None, "no-valid-device"

    def _resolve_camera_slot(self, store: Dict[str, Any],
                             ) -> Tuple[Optional[RuntimeDevice], str]:
        if not self.cameras:
            return None, "no-camera-present"
        saved = store.get(CAMERA_SLOT) if isinstance(
            store.get(CAMERA_SLOT), dict) else None
        saved_identity = str((saved or {}).get("hardware_identity") or "")
        if saved_identity:
            match = next((c for c in self.cameras
                          if c.hardware_identity == saved_identity), None)
            if match is not None:
                label = ((saved or {}).get("friendly_name")
                         or match.friendly_name)
                return match, f"persisted-identity({label})"
            logger.info("[CAMERA-SELECT] saved camera identity %r is not "
                        "connected — re-ranking", saved_identity)
        externals = [c for c in self.cameras if c.extra.get("external")]
        if externals:
            return externals[0], "external-preference"
        return self.cameras[0], "fallback-first-validated-camera"

    @staticmethod
    def _safe_probe(probe: Callable[[RuntimeDevice], bool],
                    dev: RuntimeDevice) -> bool:
        try:
            return bool(probe(dev))
        except Exception as e:
            logger.debug("[DEVICE-SELECT] probe failed for %r: %s",
                         dev.raw_name, e)
            return False

    # ── Public API ───────────────────────────────────────────────

    def list_input_devices(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self.inputs]

    def list_output_devices(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self.outputs]

    def list_cameras(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.cameras]

    def get_selected_input(self) -> Optional[Dict[str, Any]]:
        return self.selected_input.to_dict() if self.selected_input else None

    def get_selected_output(self) -> Optional[Dict[str, Any]]:
        return self.selected_output.to_dict() if self.selected_output else None

    def get_selected_camera(self) -> Optional[Dict[str, Any]]:
        return self.selected_camera.to_dict() if self.selected_camera else None

    def friendly_name_for_index(self, direction: str,
                                runtime_index: Optional[int]) -> str:
        """Friendly runtime name for a runtime index (UI / voice commands)."""
        if runtime_index is None:
            return ""
        pool = self.inputs if direction == "input" else self.outputs
        for dev in pool:
            if dev.runtime_index == int(runtime_index):
                return dev.friendly_name
        return ""

    @staticmethod
    def _match(candidates: List[RuntimeDevice],
               identifier: str) -> Optional[RuntimeDevice]:
        """Resolve an identifier: friendly name → hardware identity →
        normalized name → (internal fallback) runtime index."""
        ident = (identifier or "").strip()
        if not ident:
            return None
        norm = normalize_raw_name(ident)
        for dev in candidates:
            if ident in (dev.friendly_name, dev.hardware_identity, dev.raw_name):
                return dev
        for dev in candidates:
            if norm and norm in (dev.normalized_name,
                                 normalize_raw_name(dev.friendly_name)):
                return dev
        for dev in candidates:
            if norm and (norm in dev.normalized_name
                         or norm in normalize_raw_name(dev.friendly_name)):
                return dev
        if ident.isdigit():
            for dev in candidates:
                if dev.runtime_index == int(ident):
                    return dev
        return None

    def _persist_slot(self, slot: str, dev: RuntimeDevice, identifier: str,
                      source: str = "explicit") -> None:
        payload = {
            "hardware_identity": dev.hardware_identity,
            "friendly_name": dev.friendly_name,
            "manufacturer": dev.manufacturer,
            "model": dev.model,
            "family_id": dev.family_id,
            "raw_name": dev.raw_name,
            "category": dev.category,
            "is_camera_associated_audio": dev.is_camera_associated_audio,
            "runtime_index": dev.runtime_index,   # transient hint only
            "source": source,
            "saved_at": time.time(),
        }
        if source != "explicit":
            # NEVER overwrite an explicit user choice — even when that
            # hardware is currently missing (keeps the user's selection
            # sticky across temporary disconnects).
            existing = load_hardware_store(self.selection_path).get(slot)
            if isinstance(existing, dict) and existing.get("source") == "explicit":
                logger.debug("[DEVICE-SELECT] %s keeps its explicit user "
                             "choice — auto discovery will not overwrite it",
                             slot)
                return
        save_hardware_slot(slot, payload, self.selection_path)
        if source == "explicit":
            tag = "[CAMERA-SELECT]" if slot == CAMERA_SLOT else "[DEVICE-SELECT]"
            logger.info("%s %s=%r reason=explicit user selection (%s) "
                        "identity=%s", tag, slot, dev.friendly_name,
                        identifier, dev.hardware_identity)

    def select_input_device(self, identifier: str,
                            probe: Optional[Callable[[RuntimeDevice], bool]] = None,
                            ) -> Optional[Dict[str, Any]]:
        """Select the microphone by friendly name / identity. NEVER touches
        the output or camera slots."""
        dev = self._match(self.inputs, identifier)
        if dev is None:
            logger.warning("[DEVICE-SELECT] input %r not found among runtime "
                           "input devices", identifier)
            return None
        if probe is not None and not self._safe_probe(probe, dev):
            logger.warning("[DEVICE-SELECT] input %r failed validation — "
                           "selection rejected", identifier)
            return None
        with self._lock:
            self.selected_input = dev
            self.input_reason = f"explicit user selection ({identifier})"
        self._persist_slot(INPUT_SLOT, dev, identifier)
        return dev.to_dict()

    def select_output_device(self, identifier: str,
                             probe: Optional[Callable[[RuntimeDevice], bool]] = None,
                             ) -> Optional[Dict[str, Any]]:
        """Select the speakers/headphones by friendly name / identity.
        NEVER touches the input or camera slots."""
        dev = self._match(self.outputs, identifier)
        if dev is None:
            logger.warning("[DEVICE-SELECT] output %r not found among runtime "
                           "output devices", identifier)
            return None
        if probe is not None and not self._safe_probe(probe, dev):
            logger.warning("[DEVICE-SELECT] output %r failed validation — "
                           "selection rejected", identifier)
            return None
        with self._lock:
            self.selected_output = dev
            self.output_reason = f"explicit user selection ({identifier})"
        self._persist_slot(OUTPUT_SLOT, dev, identifier)
        return dev.to_dict()

    def select_camera(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Select a camera by friendly name / identity.

        Validation is delegated to the EXISTING CameraSelector (open →
        frame capture → release); nothing is persisted unless the camera
        actually produces frames. NEVER touches the audio slots.
        """
        matched = self._match(self.cameras, identifier)
        chosen = None
        validator_available = True
        try:
            from auth import camera_selector as cs
            selector = cs.get_selector()
            chosen = selector.select_camera(identifier)
            if chosen is None and str(identifier).strip().isdigit():
                chosen = selector.select_camera_by_index(int(identifier))
            if (chosen is None and matched is not None
                    and matched.runtime_index is not None):
                chosen = selector.select_camera_by_index(int(matched.runtime_index))
        except ImportError:
            validator_available = False
        except Exception as e:
            logger.warning("[CAMERA-SELECT] camera validation failed: %s", e)

        if chosen is not None:
            meta = read_video_metadata(int(chosen.index))
            usb_id = str(meta.get("usb_id") or "")
            serial = str(meta.get("usb_serial") or "")
            label = camera_friendly_name(chosen.name, chosen.by_id_path)
            identity = camera_hardware_identity(chosen.name, chosen.by_id_path,
                                                usb_id, serial, chosen.usb_path)
            external = classify_camera_external(
                chosen.name, chosen.by_id_path, chosen.usb_path,
                str(meta.get("integration") or ""))
            dev = next((c for c in self.cameras
                        if c.hardware_identity == identity), None)
            if dev is None:
                dev = RuntimeDevice(
                    runtime_index=int(chosen.index), raw_name=chosen.name,
                    direction="camera", device_type="camera",
                    normalized_name=normalize_raw_name(chosen.name),
                    friendly_name=label, sample_rate=float(chosen.fps or 0.0),
                    category="PHYSICAL", is_physical=True, is_virtual=False,
                    is_usb=bool(usb_id or chosen.usb_path), is_camera=True,
                    is_internal=(not external),
                    integration=(str(meta.get("integration") or "")
                                 or ("external" if external else "internal")),
                    manufacturer=str(meta.get("usb_manufacturer") or ""),
                    model=label, hardware_identity=identity,
                    family_id=f"family:{identity}",
                    extra={"path": chosen.path,
                           "by_id_path": chosen.by_id_path,
                           "usb_path": chosen.usb_path, "external": external,
                           "width": chosen.width, "height": chosen.height,
                           "fps": chosen.fps, "usb_id": usb_id})
                self.cameras.append(dev)
            with self._lock:
                self.selected_camera = dev
                self.camera_reason = f"explicit user selection ({identifier})"
            self._persist_slot(CAMERA_SLOT, dev, identifier)
            return dev.to_dict()

        if not validator_available and matched is not None:
            # No camera validator on this platform: keep the discovered device.
            with self._lock:
                self.selected_camera = matched
                self.camera_reason = f"explicit user selection ({identifier})"
            self._persist_slot(CAMERA_SLOT, matched, identifier)
            return matched.to_dict()

        logger.warning("[CAMERA-SELECT] camera %r not found or failed frame "
                       "validation — selection rejected", identifier)
        return None

    # ── Structured data for the UI / command system ──────────────

    @staticmethod
    def command_tags(dev: RuntimeDevice) -> List[str]:
        """Generic descriptors a future intent handler can match on."""
        tags = [dev.direction or dev.device_type]
        if dev.device_type == "camera":
            tags += ["camera", "webcam"]
        if dev.is_camera_associated_audio:
            tags += ["camera", "camera microphone"]
        if dev.direction == "input" and not dev.is_camera_associated_audio:
            tags += ["microphone", "mic"]
        if dev.direction == "output":
            tags += ["speakers" if dev.is_internal else "headphones"]
        if dev.is_internal:
            tags += ["internal", "built-in"]
        if dev.integration == "external":
            tags += ["external", "usb"]
        if dev.is_display_audio:
            tags += ["display"]
        if not dev.is_physical:
            tags += ["virtual"]
        return tags

    def describe_for_commands(self) -> Dict[str, Any]:
        """Runtime device snapshot (friendly names + identities)."""
        def _entry(dev: RuntimeDevice) -> Dict[str, Any]:
            return {
                "friendly_name": dev.friendly_name,
                "hardware_identity": dev.hardware_identity,
                "family_id": dev.family_id,
                "runtime_index": dev.runtime_index,
                "raw_name": dev.raw_name,
                "manufacturer": dev.manufacturer,
                "model": dev.model,
                "category": dev.category,
                "is_physical": dev.is_physical,
                "is_virtual": dev.is_virtual,
                "is_usb": dev.is_usb,
                "is_internal": dev.is_internal,
                "integration": dev.integration,
                "is_display_audio": dev.is_display_audio,
                "is_camera_associated_audio": dev.is_camera_associated_audio,
                "tags": self.command_tags(dev),
            }
        return {
            "inputs": [_entry(d) for d in self.inputs],
            "outputs": [_entry(d) for d in self.outputs],
            "cameras": [_entry(c) for c in self.cameras],
            "selected": {
                "input": (self.selected_input.friendly_name
                          if self.selected_input else None),
                "input_reason": self.input_reason,
                "output": (self.selected_output.friendly_name
                           if self.selected_output else None),
                "output_reason": self.output_reason,
                "camera": (self.selected_camera.friendly_name
                           if self.selected_camera else None),
                "camera_reason": self.camera_reason,
            },
        }


# ═══════════════════════════════════════════════════════════════
# Module-level singleton + convenience API
# ═══════════════════════════════════════════════════════════════

_registry = DeviceRegistry()


def get_registry() -> DeviceRegistry:
    return _registry


def discover_devices(sd_module: Any = None, **kwargs) -> Dict[str, Any]:
    return _registry.discover(sd_module=sd_module, **kwargs)


def list_input_devices() -> List[Dict[str, Any]]:
    return _registry.list_input_devices()


def list_output_devices() -> List[Dict[str, Any]]:
    return _registry.list_output_devices()


def list_cameras() -> List[Dict[str, Any]]:
    return _registry.list_cameras()


def get_selected_input() -> Optional[Dict[str, Any]]:
    return _registry.get_selected_input()


def get_selected_output() -> Optional[Dict[str, Any]]:
    return _registry.get_selected_output()


def get_selected_camera() -> Optional[Dict[str, Any]]:
    return _registry.get_selected_camera()


def select_input_device(identifier: str,
                        probe: Optional[Callable[[RuntimeDevice], bool]] = None,
                        ) -> Optional[Dict[str, Any]]:
    return _registry.select_input_device(identifier, probe=probe)


def select_output_device(identifier: str,
                         probe: Optional[Callable[[RuntimeDevice], bool]] = None,
                         ) -> Optional[Dict[str, Any]]:
    return _registry.select_output_device(identifier, probe=probe)


def select_camera(identifier: str) -> Optional[Dict[str, Any]]:
    return _registry.select_camera(identifier)


def describe_devices() -> Dict[str, Any]:
    return _registry.describe_for_commands()


def friendly_name_for_index(direction: str,
                            runtime_index: Optional[int]) -> str:
    return _registry.friendly_name_for_index(direction, runtime_index)


def record_active_input(raw_name: str, runtime_index: Optional[int] = None,
                        source: str = "auto") -> Optional[Dict[str, Any]]:
    return record_active_audio(INPUT_SLOT, raw_name, runtime_index, source)


def record_active_output(raw_name: str, runtime_index: Optional[int] = None,
                         source: str = "auto") -> Optional[Dict[str, Any]]:
    return record_active_audio(OUTPUT_SLOT, raw_name, runtime_index, source)


def resolve_persisted_input_index(devices: Iterable[Dict[str, Any]],
                                  **kwargs) -> Optional[Tuple[int, str]]:
    return resolve_persisted_index(INPUT_SLOT, devices, **kwargs)


def resolve_persisted_output_index(devices: Iterable[Dict[str, Any]],
                                   **kwargs) -> Optional[Tuple[int, str]]:
    return resolve_persisted_index(OUTPUT_SLOT, devices, **kwargs)


def startup_device_discovery(sd_module: Any = None, *,
                             include_cameras: bool = True,
                             probe_output: Optional[Callable[[RuntimeDevice], bool]] = None,
                             ) -> Dict[str, Any]:
    """Startup entry point: enumerate → classify → rank → resolve → log."""
    try:
        return _registry.discover(sd_module=sd_module,
                                  include_cameras=include_cameras,
                                  probe_output=probe_output)
    except Exception as e:
        logger.error("[DEVICE-DISCOVERY] startup discovery failed: %s", e)
        return {"inputs": [], "outputs": [], "cameras": []}
