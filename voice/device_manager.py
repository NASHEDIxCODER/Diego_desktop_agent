"""
DeviceManager — Independent audio INPUT/OUTPUT device selection for Diego.

Responsibilities:
    - Enumerate capture-capable (input) and playback-capable (output)
      sounddevice devices for the HUD AUDIO panel.
    - Persist the user's selections INDEPENDENTLY:
        data/audio_devices.json = {
            "input_device":  {"index": ..., "name": "..."},
            "output_device": {"index": ..., "name": "..."}
        }
      Saving the speaker NEVER touches the microphone entry (and vice versa).
      Input persistence reuses the existing verified-microphone mechanism
      (voice_settings.device_index override + AudioManager verified probe +
      data/mic_selection.json) on top of this store.
    - Apply persisted devices at startup: restore them when still present,
      otherwise let the existing auto-detection pick a working device.
    - Switch devices at runtime WITHOUT restarting the assistant:
        input  → AudioManager.switch_input_device() (single capture stream)
        output → StreamingTTS.set_output_device()  (playback only)
    - Safe validation: short silent input test (no loud playback).

This module creates NO audio streams of its own — it drives the EXISTING
AudioManager (one InputStream) and StreamingTTS (one OutputStream).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Persistence store (separate keys for input/output; one file so the pair
# is easy to inspect, but every write touches ONLY its own key).
DEVICES_PATH = Path(__file__).resolve().parent.parent / "data" / "audio_devices.json"

INPUT_KEY = "input_device"
OUTPUT_KEY = "output_device"

# Virtual ALSA/Pulse "default" pseudo-devices. These indices are NOT
# concrete hardware: playing through a saved virtual `default` index is
# unreliable (the user may hear nothing). When a saved/selected output is
# virtual, the REAL current OS default playback device is resolved and
# validated instead (see resolve_default_output_device).
VIRTUAL_OUTPUT_NAMES = {"default"}


class DeviceManager:
    """Independent input/output audio device selection and persistence."""

    def __init__(self):
        self._lock = threading.Lock()

    # ── sounddevice access ────────────────────────────────────────

    def _sd(self):
        import sounddevice as sd
        return sd

    # ── Virtual-output detection ──────────────────────────────────

    @staticmethod
    def _is_virtual_output(name: str) -> bool:
        """True for virtual ALSA/Pulse pseudo-devices (e.g. `default`).

        A virtual `default` index is never stored/played blindly: a
        concrete working output device is resolved instead.
        """
        return (name or "").strip().lower() in VIRTUAL_OUTPUT_NAMES

    # ── Enumeration ───────────────────────────────────────────────

    def list_input_devices(self) -> List[Dict[str, Any]]:
        """All capture-capable devices (input channels > 0 or capture-by-name
        physical codecs), classified by the existing AudioManager detector
        and enriched with runtime friendly names / hardware identities."""
        try:
            from voice.audio_manager import enumerate_input_devices
            return self._enrich(enumerate_input_devices(self._sd()), "input")
        except Exception as e:
            logger.error("[DEVICES] Input enumeration failed: %s", e)
            return []

    def list_output_devices(self) -> List[Dict[str, Any]]:
        """All playback-capable devices (output channels > 0 or physical
        codecs that play by name), enriched with runtime friendly names."""
        out = []
        try:
            sd = self._sd()
            try:
                hostapis = sd.query_hostapis()
            except Exception:
                hostapis = []
            try:
                default_out = sd.default.device[1]
            except Exception:
                default_out = -1
            for d in sd.query_devices():
                out_ch = int(d.get("max_output_channels", 0))
                name = str(d.get("name", ""))
                # Physical ALSA codecs enumerated as capture-only can still
                # play when addressed by name (validated before use).
                physical_by_name = (out_ch <= 0 and "(hw:" in name.lower()
                                    and not self._is_display_output(name))
                if out_ch <= 0 and not physical_by_name:
                    continue
                idx = d.get("index")
                hostapi_idx = d.get("hostapi", -1)
                try:
                    hostapi = str(hostapis[hostapi_idx]["name"]) \
                        if 0 <= hostapi_idx < len(hostapis) else "?"
                except Exception:
                    hostapi = "?"
                out.append({
                    "index": idx,
                    "name": name,
                    "hostapi": hostapi,
                    "max_output_channels": out_ch,
                    "default_samplerate": float(d.get("default_samplerate", 0.0)),
                    "is_default": (idx == default_out),
                    "playback_by_name": bool(physical_by_name),
                })
        except Exception as e:
            logger.error("[DEVICES] Output enumeration failed: %s", e)
        return self._enrich(out, "output")

    @staticmethod
    def _is_display_output(name: str) -> bool:
        lname = (name or "").lower()
        return any(c in lname for c in ("hdmi", "displayport", "dp,"))

    @staticmethod
    def _enrich(devices: List[Dict[str, Any]],
                direction: str) -> List[Dict[str, Any]]:
        """Attach runtime friendly names + hardware identities (best effort).

        The runtime index stays available for the existing pipeline; the
        friendly name/identity come from voice.runtime_devices and are what
        the user-facing surfaces and future voice commands use.
        """
        try:
            from voice import runtime_devices as rd
        except Exception:
            return devices
        for dev in devices:
            try:
                in_view, out_view = rd.build_audio_views(
                    dev.get("index"), str(dev.get("name") or ""),
                    str(dev.get("hostapi") or ""),
                    int(dev.get("max_input_channels", 0) or 0),
                    int(dev.get("max_output_channels", 0) or 0),
                    float(dev.get("default_samplerate", 0.0) or 0.0))
                view = in_view if direction == "input" else (out_view or in_view)
                if view is None:
                    continue
                dev["friendly_name"] = view.friendly_name
                dev["hardware_identity"] = view.hardware_identity
                dev["family_id"] = view.family_id
                dev["category"] = view.category
                dev["integration"] = view.integration
                dev["is_camera_associated_audio"] = (
                    view.is_camera_associated_audio)
            except Exception:
                continue
        return devices

    def find_input_device(self, index: int) -> Optional[Dict[str, Any]]:
        return next((d for d in self.list_input_devices()
                     if d.get("index") == index), None)

    def find_output_device(self, index: int) -> Optional[Dict[str, Any]]:
        return next((d for d in self.list_output_devices()
                     if d.get("index") == index), None)

    # ── Persistence (independent keys) ────────────────────────────

    def _load_store(self) -> Dict[str, Any]:
        try:
            if DEVICES_PATH.exists():
                with open(DEVICES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.debug("[DEVICES] Failed to load device store: %s", e)
        return {}

    def _save_store(self, store: Dict[str, Any]) -> None:
        try:
            DEVICES_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = DEVICES_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2)
            tmp.replace(DEVICES_PATH)
        except Exception as e:
            logger.warning("[DEVICES] Failed to persist device store: %s", e)

    def save_input_device(self, index: int, name: str,
                          source: str = "auto") -> None:
        """Persist the microphone selection. NEVER touches output_device.

        The LOGICAL hardware identity is stored NEXT TO the runtime index
        (index is only a transient hint; identity is canonical).
        """
        with self._lock:
            store = self._load_store()
            store[INPUT_KEY] = self._entry_with_identity(
                "input", index, name, source)
            self._save_store(store)
        logger.info("[DEVICES] input_device persisted: [%s] %s", index, name)

    def save_output_device(self, index: int, name: str,
                           source: str = "auto") -> None:
        """Persist the speaker selection. NEVER touches input_device.

        The LOGICAL hardware identity is stored NEXT TO the runtime index
        (index is only a transient hint; identity is canonical).
        """
        with self._lock:
            store = self._load_store()
            store[OUTPUT_KEY] = self._entry_with_identity(
                "output", index, name, source)
            self._save_store(store)
        logger.info("[DEVICES] output_device persisted: [%s] %s", index, name)

    def _entry_with_identity(self, slot: str, index: Optional[int],
                             name: str, source: str) -> Dict[str, Any]:
        """Build the persisted entry (index + name + LOGICAL identity).

        The identity comes from voice.runtime_devices but is written into
        THIS store's entry (audio_devices.json) — never into a separate
        global file — so test-isolated stores stay isolated and the
        identity always travels with the selection it describes.
        """
        entry: Dict[str, Any] = {
            "index": index, "name": name, "saved_at": time.time(),
        }
        try:
            from voice import runtime_devices as rd
            direction = "output" if slot == "output" else "input"
            payload = rd.identity_payload_for_audio(name, direction,
                                                    runtime_index=index,
                                                    source=source)
            entry.update({
                "hardware_identity": payload.get("hardware_identity", ""),
                "friendly_name": payload.get("friendly_name", ""),
                "family_id": payload.get("family_id", ""),
                "manufacturer": payload.get("manufacturer", ""),
                "model": payload.get("model", ""),
                "category": payload.get("category", ""),
                "is_camera_associated_audio": payload.get(
                    "is_camera_associated_audio", False),
                "source": source,
            })
        except Exception as e:
            logger.debug("[DEVICES] hardware-identity payload failed: %s", e)
        return entry

    def get_saved_input_device(self) -> Optional[Dict[str, Any]]:
        return self._load_store().get(INPUT_KEY)

    def get_saved_output_device(self) -> Optional[Dict[str, Any]]:
        return self._load_store().get(OUTPUT_KEY)

    # ── Current devices (from the live pipeline) ──────────────────

    def get_current_input_device(self) -> Dict[str, Any]:
        try:
            from voice.audio_manager import audio_manager
            index = audio_manager.device_index
            name = audio_manager.device_name
            return {"index": index,
                    "name": name,
                    "friendly_name": self._friendly("input", index, name),
                    "running": audio_manager.is_running,
                    "verified": audio_manager.mic_verified}
        except Exception as e:
            logger.debug("[DEVICES] current input unavailable: %s", e)
            return {"index": None, "name": "", "friendly_name": "",
                    "running": False, "verified": False}

    def get_current_output_device(self) -> Dict[str, Any]:
        try:
            from voice.streaming_tts import streaming_tts
            index = streaming_tts.output_device
            name = streaming_tts.output_device_name
            return {"index": index,
                    "name": name,
                    "friendly_name": self._friendly("output", index, name),
                    "ready": streaming_tts.ready}
        except Exception as e:
            logger.debug("[DEVICES] current output unavailable: %s", e)
            return {"index": None, "name": "", "friendly_name": "",
                    "ready": False}

    @staticmethod
    def _friendly(direction: str, index, raw_name: str = "") -> str:
        """Runtime friendly name for the live device (user-facing label)."""
        try:
            from voice import runtime_devices as rd
            name = rd.friendly_name_for_index(direction, index)
            if name:
                return name
            if raw_name:
                in_view, out_view = rd.build_audio_views(index, raw_name)
                view = (in_view if direction == "input"
                        else (out_view or in_view))
                if view is not None:
                    return view.friendly_name
        except Exception:
            pass
        return raw_name or ""

    # ── Startup restore ───────────────────────────────────────────

    def apply_persisted_devices(self) -> Dict[str, Any]:
        """Restore saved devices at startup when they are still available.

        Input:  sets voice_settings.device_index BEFORE the AudioManager
                opens its stream — the existing verified-detection honours
                that override and still probes/verifies the device.
        Output: targets StreamingTTS's playback stream at the saved device
                (applied lazily on the next synthesis / stream creation).

        Missing/unavailable devices are safely skipped — the existing
        auto-detection then selects a working default.
        """
        result = {"input_restored": None, "output_restored": None}

        # ── Runtime hardware-identity resolution (index-independent) ──
        # The saved LOGICAL hardware identity (in THIS store's entry) is
        # re-resolved against the CURRENT enumeration first so the same
        # physical device keeps working after reboot/reconnect. A saved
        # runtime index is NEVER canonical — only the identity is.
        saved_in = self.get_saved_input_device()
        hit = None
        if saved_in and (saved_in.get("hardware_identity")
                         or saved_in.get("name")):
            try:
                from voice import runtime_devices as rd
                hit = rd.resolve_saved_entry_index(
                    rd.INPUT_SLOT, self.list_input_devices(),
                    saved_identity=saved_in.get("hardware_identity"),
                    saved_name=(None if saved_in.get("hardware_identity")
                                else saved_in.get("name")),
                    source=str(saved_in.get("source") or "auto"))
            except Exception as e:
                logger.debug("[DEVICES] identity-based input resolution "
                             "unavailable: %s", e)
        if hit:
            index, reason = hit
            dev = self.find_input_device(int(index))
            from voice.settings import voice_settings
            voice_settings.device_index = int(index)
            result["input_restored"] = {
                "index": int(index),
                "name": (dev or {}).get("name", ""),
                "friendly_name": (dev or {}).get("friendly_name", ""),
                "hardware_identity": (dev or {}).get("hardware_identity", ""),
                "reason": reason,
            }
            logger.info("[DEVICES] Input resolved by hardware identity → "
                        "[%s] %s (%s)", index, (dev or {}).get("name"),
                        reason)

        if result["input_restored"] is None and saved_in \
                and saved_in.get("index") is not None:
            if self.find_input_device(saved_in["index"]) is not None:
                from voice.settings import voice_settings
                voice_settings.device_index = int(saved_in["index"])
                result["input_restored"] = saved_in
                logger.info("[DEVICES] Restored input device: [%s] %s",
                            saved_in["index"], saved_in.get("name"))
            else:
                logger.warning("[DEVICES] Saved input device [%s] %s is "
                               "unavailable — auto-detection will pick one",
                               saved_in["index"], saved_in.get("name"))

        restored = False

        # ── Output: hardware-identity resolution first (index-independent) ──
        saved_out = self.get_saved_output_device()
        hit = None
        if saved_out and (saved_out.get("hardware_identity")
                          or saved_out.get("name")):
            try:
                from voice import runtime_devices as rd
                hit = rd.resolve_saved_entry_index(
                    rd.OUTPUT_SLOT, self.list_output_devices(),
                    saved_identity=saved_out.get("hardware_identity"),
                    saved_name=(None if saved_out.get("hardware_identity")
                                else saved_out.get("name")),
                    source=str(saved_out.get("source") or "auto"))
            except Exception as e:
                logger.debug("[DEVICES] identity-based output resolution "
                             "unavailable: %s", e)
        if hit:
            index, reason = hit
            dev = self.find_output_device(int(index))
            if (dev is not None
                    and not self._is_virtual_output(str(dev.get("name") or ""))
                    and self.test_output_device(int(index)).get("ok")):
                from voice.streaming_tts import streaming_tts
                streaming_tts.set_output_device(int(index))
                result["output_restored"] = {
                    "index": int(index),
                    "name": dev.get("name", ""),
                    "friendly_name": dev.get("friendly_name", ""),
                    "hardware_identity": dev.get("hardware_identity", ""),
                    "reason": reason,
                }
                restored = True
                logger.info("[DEVICES] Output resolved by hardware "
                            "identity → [%s] %s (%s)", index,
                            dev.get("name"), reason)
            elif dev is not None and self._is_virtual_output(
                    str(dev.get("name") or "")):
                logger.warning("[DEVICES] Identity-resolved output [%s] "
                               "'%s' is a virtual pseudo-device — resolving "
                               "a concrete working output instead",
                               index, dev.get("name"))
            else:
                logger.warning("[DEVICES] Identity-resolved output [%s] "
                               "failed playback validation — falling back "
                               "to the saved index / OS default", index)

        if not restored and saved_out and saved_out.get("index") is not None:
            dev = self.find_output_device(int(saved_out["index"]))
            if dev is None:
                logger.warning("[DEVICES] Saved output device [%s] %s is "
                               "unavailable — resolving the current OS "
                               "default output",
                               saved_out["index"], saved_out.get("name"))
            elif self._is_virtual_output(dev["name"]):
                logger.warning("[DEVICES] Saved output device [%s] '%s' is a "
                               "virtual pseudo-device — resolving a concrete "
                               "working output instead",
                               saved_out["index"], dev["name"])
            elif not self.test_output_device(int(saved_out["index"])).get("ok"):
                logger.warning("[DEVICES] Saved output device [%s] %s failed "
                               "playback validation — resolving the current "
                               "OS default output",
                               saved_out["index"], saved_out.get("name"))
            else:
                try:
                    from voice.streaming_tts import streaming_tts
                    streaming_tts.set_output_device(int(saved_out["index"]))
                    result["output_restored"] = saved_out
                    restored = True
                    logger.info("[DEVICES] Restored output device: [%s] %s",
                                saved_out["index"], saved_out.get("name"))
                except Exception as e:
                    logger.warning("[DEVICES] Could not restore output "
                                   "device: %s", e)

        if not restored:
            # No (valid) user selection: resolve the REAL current OS
            # default playback device, validate it and use it for TTS.
            # This is an automatic resolution, NOT a user selection — it
            # is never persisted over the user's saved choice.
            try:
                idx = self.resolve_default_output_device()
            except Exception as e:
                logger.warning("[DEVICES] Default output resolution "
                               "failed: %s", e)
                idx = None
            if idx is not None:
                try:
                    from voice.streaming_tts import streaming_tts
                    streaming_tts.set_output_device(int(idx))
                    dev = self.find_output_device(int(idx))
                    result["output_resolved_default"] = {
                        "index": int(idx),
                        "name": dev["name"] if dev else f"Device {idx}",
                    }
                    logger.info("[DEVICES] Resolved concrete default output: "
                                "[%s] %s", idx,
                                dev["name"] if dev else f"Device {idx}")
                except Exception as e:
                    logger.warning("[DEVICES] Could not apply resolved "
                                   "default output: %s", e)
            else:
                logger.warning("[DEVICES] No concrete output device could be "
                               "validated — TTS will use the PortAudio "
                               "default (device=None)")

        return result

    # ── Runtime switching ─────────────────────────────────────────

    def set_input_device(self, index: int) -> Dict[str, Any]:
        """
        Switch the live microphone (STT/VAD input) to `index`.

        Drives the EXISTING AudioManager's single capture stream:
        stop → reconfigure (verified probe honours the selection) → restart.
        VAD/STT consumers keep working (monotonic ring-buffer counters are
        preserved). On failure the previous working device is restored.
        Persisted ONLY on success.
        """
        dev = self.find_input_device(index)
        if dev is None:
            return {"ok": False,
                    "error": f"Input device {index} not found",
                    "device_index": None, "device_name": "", "fallback": False}
        try:
            from voice.audio_manager import audio_manager
            result = audio_manager.switch_input_device(index)
        except Exception as e:
            logger.error("[DEVICES] Input switch to [%s] failed: %s", index, e)
            return {"ok": False, "error": str(e),
                    "device_index": None, "device_name": "", "fallback": False}

        if result.get("ok"):
            name = result.get("device_name") or dev["name"]
            # Explicit user switch → identity persisted as explicit choice
            # (survives reboot / ALSA renumbering). Writes ONLY into THIS
            # store's input_device entry — output/camera are untouched.
            self.save_input_device(int(result.get("device_index") or index),
                                   name, source="explicit")
        return result

    def set_output_device(self, index: int) -> Dict[str, Any]:
        """
        Switch the TTS playback device to `index`.

        Updates StreamingTTS's output target for future synthesis/playback.
        NEVER restarts microphone capture and NEVER alters STT. Persisted
        on success (input_device is untouched).
        """
        dev = self.find_output_device(index)
        if dev is None:
            return {"ok": False,
                    "error": f"Output device {index} not found",
                    "device_index": None, "device_name": "", "fallback": False}

        # Never blindly store/play through a virtual ALSA `default` index:
        # resolve a concrete working output device instead.
        if self._is_virtual_output(dev["name"]):
            concrete = self.resolve_default_output_device()
            if concrete is None:
                return {"ok": False,
                        "error": ("'default' is a virtual device and no "
                                  "concrete output device could be validated"),
                        "device_index": None, "device_name": "",
                        "fallback": False}
            logger.info("[DEVICES] Virtual output '%s' selected — using "
                        "concrete default output [%s] instead",
                        dev["name"], concrete)
            index = int(concrete)
            dev = self.find_output_device(index)
            if dev is None:
                return {"ok": False, "error": "resolved device not found",
                        "device_index": None, "device_name": "",
                        "fallback": False}

        # Validate playback BEFORE switching/persisting.
        validation = self.test_output_device(index)
        if not validation.get("ok"):
            logger.warning("[DEVICES] Output [%s] %s failed playback "
                           "validation (%s) — selection rejected",
                           index, dev["name"], validation.get("error"))
            return {"ok": False, "error": validation.get("error"),
                    "device_index": None, "device_name": "",
                    "fallback": False}

        try:
            from voice.streaming_tts import streaming_tts
            result = streaming_tts.set_output_device(index)
        except Exception as e:
            logger.error("[DEVICES] Output switch to [%s] failed: %s", index, e)
            return {"ok": False, "error": str(e),
                    "device_index": None, "device_name": "", "fallback": False}

        if result.get("ok"):
            # Persist ONLY the speaker selection — the microphone entry is
            # never touched (independent keys). The LOGICAL identity is
            # persisted as an explicit choice (survives reboot/renumbering).
            self.save_output_device(index, dev["name"], source="explicit")
        else:
            logger.warning("[DEVICES] Output switch to [%s] %s failed: %s "
                           "— staying on the previous output",
                           index, dev["name"], result.get("error"))
        return result

    # ── Safe validation (requirement 9) ───────────────────────────

    def test_input_device(self, index: int, duration: float = 0.5) -> Dict[str, Any]:
        """Open the input device, capture a SHORT test and verify a
        non-zero signal. No loud audio, no persistent state changes."""
        try:
            import numpy as np
            sd = self._sd()
            dev = self.find_input_device(index)
            if dev is None:
                return {"ok": False, "error": "device not found"}
            samplerate = int(dev.get("default_samplerate") or 16000)
            rec = sd.rec(int(duration * samplerate), samplerate=samplerate,
                         channels=1, dtype="int16", device=index, blocking=True)
            data = np.asarray(rec)
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) \
                if data.size else 0.0
            return {"ok": rms > 1.0, "rms": rms,
                    "error": None if rms > 1.0 else "silent signal"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def test_output_device(self, index: int,
                           samplerate: Optional[int] = None) -> Dict[str, Any]:
        """Validate that the output device actually supports playback.

        Opens an OutputStream on the device, writes a 10 ms SILENT buffer
        (inaudible, non-destructive) and closes again. When `samplerate`
        is given (e.g. the TTS playback rate) the device must accept THAT
        rate — a device that only opens at its own default rate is not a
        valid TTS output.
        """
        try:
            import numpy as np
            sd = self._sd()
            dev = self.find_output_device(index)
            if dev is None:
                return {"ok": False, "error": "device not found"}
            rate = int(samplerate or dev.get("default_samplerate") or 24000)
            try:
                stream = sd.OutputStream(samplerate=rate, channels=1,
                                         dtype="int16", blocksize=0,
                                         device=index)
            except Exception:
                if samplerate is None:
                    raise
                # Retry at the device's own default rate so a device that
                # merely lacks the TTS rate is distinguishable from a
                # device that cannot play at all.
                rate = int(dev.get("default_samplerate") or rate)
                stream = sd.OutputStream(samplerate=rate, channels=1,
                                         dtype="int16", blocksize=0,
                                         device=index)
            stream.start()
            # 10 ms of digital silence — safe playback validation.
            stream.write(np.zeros(int(rate * 0.01), dtype=np.int16))
            stream.stop()
            stream.close()
            return {"ok": True, "error": None, "samplerate": rate}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def resolve_default_output_device(
            self, sample_rate: int = 24000) -> Optional[int]:
        """Resolve the REAL current OS default playback device.

        The PortAudio `default` index may itself be a virtual ALSA
        pseudo-device (name `default`) which can be silent/unreliable.
        This returns a CONCRETE, playback-validated device:

          1. the PortAudio default output — when it is concrete
          2. pulse/pipewire host-api devices (they route to the real
             OS default sink and accept arbitrary sample rates)
          3. any other concrete output device

        Every candidate is validated with a short silent playback test at
        the TTS sample rate. Returns None when nothing validates (the
        caller then keeps the PortAudio default, device=None).
        """
        devices = self.list_output_devices()
        if not devices:
            return None
        try:
            default_idx = int(self._sd().default.device[1])
        except Exception:
            default_idx = -1

        ordered: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(dev: Optional[Dict[str, Any]]) -> None:
            if dev is None or dev["index"] in seen:
                return
            if self._is_virtual_output(dev["name"]):
                return
            seen.add(dev["index"])
            ordered.append(dev)

        # 1) PortAudio default — only when concrete
        _add(next((d for d in devices if d["index"] == default_idx), None))
        # 2) ALSA pulse/pipewire PLUGIN devices (by NAME) — these route to
        #    the REAL OS default sink configured by the desktop, which is
        #    exactly "the current OS default playback device".
        for d in devices:
            name = (d.get("name") or "").lower()
            if "pulse" in name or "pipewire" in name:
                _add(d)
        # 3) any other concrete output device — HDMI outputs LAST, since a
        #    monitor HDMI jack usually has no speakers attached and would
        #    silently swallow TTS even though it "validates".
        for d in devices:
            name = (d.get("name") or "").lower()
            if "hdmi" not in name:
                _add(d)
        for d in devices:
            _add(d)

        for dev in ordered:
            if self.test_output_device(dev["index"],
                                       samplerate=sample_rate).get("ok"):
                return dev["index"]
        return None

    # ── Runtime name-based selection (friendly names / identities) ──

    def runtime_registry(self):
        """The canonical runtime device registry (discovered on demand)."""
        from voice import runtime_devices as rd
        registry = rd.get_registry()
        if not registry.inputs and not registry.outputs:
            registry.discover(include_cameras=False)
        return registry

    def list_runtime_devices(self) -> Dict[str, Any]:
        """Structured runtime device data (friendly names + identities)."""
        try:
            return self.runtime_registry().describe_for_commands()
        except Exception as e:
            logger.debug("[DEVICES] runtime device snapshot failed: %s", e)
            return {"inputs": [], "outputs": [], "cameras": [], "selected": {}}

    def get_selected_input_device(self) -> Optional[Dict[str, Any]]:
        try:
            return self.runtime_registry().get_selected_input()
        except Exception:
            return None

    def get_selected_output_device(self) -> Optional[Dict[str, Any]]:
        try:
            return self.runtime_registry().get_selected_output()
        except Exception:
            return None

    def get_selected_camera_device(self) -> Optional[Dict[str, Any]]:
        try:
            from voice import runtime_devices as rd
            registry = rd.get_registry()
            if not registry.cameras:
                registry.discover(include_cameras=True)
            return registry.get_selected_camera()
        except Exception:
            return None

    def select_input_device(self, identifier: str) -> Dict[str, Any]:
        """Select the microphone BY RUNTIME FRIENDLY NAME (or identity).

        Resolves through the runtime registry (never a hardcoded index),
        then drives the EXISTING AudioManager switch (live verification).
        Output and camera selections are untouched.
        """
        try:
            registry = self.runtime_registry()
        except Exception as e:
            return {"ok": False, "error": str(e), "device_index": None,
                    "device_name": "", "fallback": False}
        chosen = registry.select_input_device(identifier)
        if chosen is None:
            return {"ok": False,
                    "error": f"input device {identifier!r} not found",
                    "device_index": None, "device_name": "", "fallback": False}
        index = chosen.get("runtime_index")
        if index is None:
            return {"ok": False, "error": "resolved device has no runtime index",
                    "device_index": None, "device_name": "", "fallback": False}
        result = self.set_input_device(int(index))
        result["friendly_name"] = chosen.get("friendly_name", "")
        result["hardware_identity"] = chosen.get("hardware_identity", "")
        return result

    def select_output_device(self, identifier: str) -> Dict[str, Any]:
        """Select speakers/headphones BY RUNTIME FRIENDLY NAME (or identity).

        Playback is validated BEFORE persisting; input and camera selections
        are untouched.
        """
        try:
            registry = self.runtime_registry()
        except Exception as e:
            return {"ok": False, "error": str(e), "device_index": None,
                    "device_name": "", "fallback": False}

        def _probe(dev) -> bool:
            if dev.runtime_index is None:
                return False
            return bool(self.test_output_device(int(dev.runtime_index)).get("ok"))

        chosen = registry.select_output_device(identifier, probe=_probe)
        if chosen is None:
            return {"ok": False,
                    "error": (f"output device {identifier!r} not found or "
                              "failed playback validation"),
                    "device_index": None, "device_name": "", "fallback": False}
        index = chosen.get("runtime_index")
        if index is None:
            return {"ok": False, "error": "resolved device has no runtime index",
                    "device_index": None, "device_name": "", "fallback": False}
        result = self.set_output_device(int(index))
        result["friendly_name"] = chosen.get("friendly_name", "")
        result["hardware_identity"] = chosen.get("hardware_identity", "")
        return result

    def select_camera(self, identifier: str) -> Dict[str, Any]:
        """Select a camera BY RUNTIME FRIENDLY NAME (or stable identity).

        Validated (open + frame capture) by the existing CameraSelector.
        Audio selections are untouched.
        """
        try:
            from voice import runtime_devices as rd
            registry = rd.get_registry()
            if not registry.cameras:
                registry.discover(include_cameras=True)
            chosen = registry.select_camera(identifier)
        except Exception as e:
            logger.warning("[CAMERA-SELECT] selection failed: %s", e)
            return {"ok": False, "error": str(e)}
        if chosen is None:
            return {"ok": False,
                    "error": (f"camera {identifier!r} not found or failed "
                              "frame validation")}
        logger.info("[CAMERA-SELECT] camera=%r reason=explicit user selection "
                    "(%s)", chosen.get("friendly_name"), identifier)
        return {"ok": True, "camera": chosen,
                "friendly_name": chosen.get("friendly_name", ""),
                "hardware_identity": chosen.get("hardware_identity", ""),
                "index": chosen.get("runtime_index")}


# Global singleton
device_manager = DeviceManager()


# ═══════════════════════════════════════════════════════════════
# CLI interactive microphone selection (`python main.py --select-mic`)
# ═══════════════════════════════════════════════════════════════

def select_microphone_interactive() -> None:
    """Interactive microphone picker: list capture devices, let the user
    choose one, verify it with the existing AudioManager probe and persist
    through the SAME verified-selection mechanism used at startup."""
    devices = device_manager.list_input_devices()
    if not devices:
        print("  No input devices found.")
        return
    print("\n  Available input devices:")
    for d in devices:
        marker = " (default)" if d.get("is_default") else ""
        print(f"    [{d['index']:>2}] {d['name']}{marker}")

    raw = input("\n  Select microphone index: ").strip()
    try:
        index = int(raw)
    except ValueError:
        print("  Invalid index — aborted.")
        return

    dev = device_manager.find_input_device(index)
    if dev is None:
        print(f"  Device [{index}] not found — aborted.")
        return

    # Verify with the existing probe (records 2 s, measures signal).
    try:
        from voice.audio_manager import probe_device
        probe = probe_device(device_manager._sd(), dev)
    except Exception as e:
        print(f"  Probe failed: {e} — aborted.")
        return
    if not probe.get("valid"):
        print(f"  ✗ Device rejected: {probe.get('reject_reason', 'no signal')}"
              " — NOT saved.")
        return

    # Persist through both mechanisms (UI store + verified selection).
    device_manager.save_input_device(index, dev["name"])
    from voice.settings import voice_settings
    voice_settings.device_index = index
    try:
        from voice.audio_manager import save_verified_selection
        save_verified_selection(probe)
    except Exception:
        pass
    print(f"  ✓ Microphone selected and persisted: [{index}] {dev['name']}")
