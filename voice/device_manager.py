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


class DeviceManager:
    """Independent input/output audio device selection and persistence."""

    def __init__(self):
        self._lock = threading.Lock()

    # ── sounddevice access ────────────────────────────────────────

    def _sd(self):
        import sounddevice as sd
        return sd

    # ── Enumeration ───────────────────────────────────────────────

    def list_input_devices(self) -> List[Dict[str, Any]]:
        """All capture-capable devices (input channels > 0 or capture-by-name
        physical codecs), classified by the existing AudioManager detector."""
        try:
            from voice.audio_manager import enumerate_input_devices
            return enumerate_input_devices(self._sd())
        except Exception as e:
            logger.error("[DEVICES] Input enumeration failed: %s", e)
            return []

    def list_output_devices(self) -> List[Dict[str, Any]]:
        """All playback-capable devices (output channels > 0)."""
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
                if out_ch <= 0:
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
                    "name": str(d.get("name", "")),
                    "hostapi": hostapi,
                    "max_output_channels": out_ch,
                    "default_samplerate": float(d.get("default_samplerate", 0.0)),
                    "is_default": (idx == default_out),
                })
        except Exception as e:
            logger.error("[DEVICES] Output enumeration failed: %s", e)
        return out

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

    def save_input_device(self, index: int, name: str) -> None:
        """Persist the microphone selection. NEVER touches output_device."""
        with self._lock:
            store = self._load_store()
            store[INPUT_KEY] = {"index": index, "name": name,
                                "saved_at": time.time()}
            self._save_store(store)
        logger.info("[DEVICES] input_device persisted: [%s] %s", index, name)

    def save_output_device(self, index: int, name: str) -> None:
        """Persist the speaker selection. NEVER touches input_device."""
        with self._lock:
            store = self._load_store()
            store[OUTPUT_KEY] = {"index": index, "name": name,
                                 "saved_at": time.time()}
            self._save_store(store)
        logger.info("[DEVICES] output_device persisted: [%s] %s", index, name)

    def get_saved_input_device(self) -> Optional[Dict[str, Any]]:
        return self._load_store().get(INPUT_KEY)

    def get_saved_output_device(self) -> Optional[Dict[str, Any]]:
        return self._load_store().get(OUTPUT_KEY)

    # ── Current devices (from the live pipeline) ──────────────────

    def get_current_input_device(self) -> Dict[str, Any]:
        try:
            from voice.audio_manager import audio_manager
            return {"index": audio_manager.device_index,
                    "name": audio_manager.device_name,
                    "running": audio_manager.is_running,
                    "verified": audio_manager.mic_verified}
        except Exception as e:
            logger.debug("[DEVICES] current input unavailable: %s", e)
            return {"index": None, "name": "", "running": False, "verified": False}

    def get_current_output_device(self) -> Dict[str, Any]:
        try:
            from voice.streaming_tts import streaming_tts
            return {"index": streaming_tts.output_device,
                    "name": streaming_tts.output_device_name,
                    "ready": streaming_tts.ready}
        except Exception as e:
            logger.debug("[DEVICES] current output unavailable: %s", e)
            return {"index": None, "name": "", "ready": False}

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

        saved_in = self.get_saved_input_device()
        if saved_in and saved_in.get("index") is not None:
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

        saved_out = self.get_saved_output_device()
        if saved_out and saved_out.get("index") is not None:
            if self.find_output_device(saved_out["index"]) is not None:
                try:
                    from voice.streaming_tts import streaming_tts
                    streaming_tts.set_output_device(int(saved_out["index"]))
                    result["output_restored"] = saved_out
                    logger.info("[DEVICES] Restored output device: [%s] %s",
                                saved_out["index"], saved_out.get("name"))
                except Exception as e:
                    logger.warning("[DEVICES] Could not restore output "
                                   "device: %s", e)
            else:
                logger.warning("[DEVICES] Saved output device [%s] %s is "
                               "unavailable — system default will be used",
                               saved_out["index"], saved_out.get("name"))

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
            self.save_input_device(index, result.get("device_name") or dev["name"])
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
        try:
            from voice.streaming_tts import streaming_tts
            result = streaming_tts.set_output_device(index)
        except Exception as e:
            logger.error("[DEVICES] Output switch to [%s] failed: %s", index, e)
            return {"ok": False, "error": str(e),
                    "device_index": None, "device_name": "", "fallback": False}

        if result.get("ok"):
            self.save_output_device(index, dev["name"])
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

    def test_output_device(self, index: int) -> Dict[str, Any]:
        """Open/validate the output device by creating (and immediately
        closing) an OutputStream on it. NO test audio is played."""
        try:
            sd = self._sd()
            dev = self.find_output_device(index)
            if dev is None:
                return {"ok": False, "error": "device not found"}
            samplerate = int(dev.get("default_samplerate") or 24000)
            stream = sd.OutputStream(samplerate=samplerate, channels=1,
                                     dtype="int16", blocksize=0, device=index)
            stream.start()
            stream.stop()
            stream.close()
            return {"ok": True, "error": None}
        except Exception as e:
            return {"ok": False, "error": str(e)}


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
