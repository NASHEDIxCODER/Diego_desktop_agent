"""
Production-readiness tests.

Covers:
    - main.py launches the UI by default; --headless keeps the CLI runtime;
      --no-wake/--no-auth are forwarded
    - independent input/output device enumeration
    - separate persisted input/output settings (no cross-overwrites)
    - input switching does not touch output (and vice versa)
    - AudioManager input switch: stop → reconfigure → restart, ring-buffer
      counters preserved, missing-device fallback to the previous device
    - StreamingTTS output device targeting + safe stream recreation
    - device persistence + startup restore with missing-device fallback
    - runtime health status model (READY/DEGRADED/MISSING/FAILED/BYPASSED)
    - wake-unavailable does not create an infinite retry loop
    - engine degrades to LISTEN when the wake model is unavailable
    - UI event bridge remains functional + AUDIO panel behaviour
"""

import asyncio
import contextlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Headless Qt for the UI tests in this module
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from voice.device_manager import DeviceManager
import voice.device_manager as device_manager_module


# ═══════════════════════════════════════════════════════════════
# Helpers / fakes
# ═══════════════════════════════════════════════════════════════

def _fake_sd(input_indices=(0, 1), output_indices=(2, 3), names=None):
    """Minimal sounddevice stand-in for enumeration tests."""
    names = names or {}
    devices = []
    for i in range(max(max(input_indices, default=0),
                       max(output_indices, default=0)) + 1):
        if i in input_indices:
            in_ch, out_ch = 2, 0
        elif i in output_indices:
            in_ch, out_ch = 0, 2
        else:
            in_ch, out_ch = 0, 0
        devices.append({
            "index": i,
            "name": names.get(i, f"Device {i}"),
            "hostapi": 0,
            "max_input_channels": in_ch,
            "max_output_channels": out_ch,
            "default_samplerate": 44100.0,
        })
    hostapis = [{"name": "ALSA"}]

    class FakeDefault:
        device = (input_indices[0] if input_indices else -1,
                  output_indices[0] if output_indices else -1)

    sd = SimpleNamespace(
        query_devices=lambda: devices,
        query_hostapis=lambda: hostapis,
        default=FakeDefault(),
    )
    return sd


# ═══════════════════════════════════════════════════════════════
# 1. main.py entrypoint behaviour
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def main_module(monkeypatch):
    """Import main.py with the startup mic-probe neutered."""
    import main as main_mod
    from voice.audio_manager import audio_manager
    monkeypatch.setattr(type(audio_manager), "start", lambda self: True)
    monkeypatch.setattr(type(audio_manager), "stop", lambda self: None)
    audio_manager._mic_verified = False
    return main_mod


def test_main_default_launches_ui(main_module, monkeypatch):
    """`python main.py` (no args) launches the existing voice-first UI."""
    calls = {"diego": None, "ui": None}
    monkeypatch.setattr(main_module.Diego, "run",
                        lambda **kw: calls.__setitem__("diego", kw))
    import ui.__main__ as ui_main_mod
    monkeypatch.setattr(ui_main_mod, "main",
                        lambda argv=None: calls.__setitem__("ui", argv) or 0)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit):
        main_module.main()
    assert calls["ui"] is not None, "UI entrypoint must run by default"
    assert calls["diego"] is None, "headless runtime must NOT run by default"


def test_main_no_wake_no_auth_launches_ui(main_module, monkeypatch):
    """`python main.py --no-wake --no-auth` launches the UI with flags."""
    calls = {"diego": None, "ui": None}
    monkeypatch.setattr(main_module.Diego, "run",
                        lambda **kw: calls.__setitem__("diego", kw))
    import ui.__main__ as ui_main_mod
    monkeypatch.setattr(ui_main_mod, "main",
                        lambda argv=None: calls.__setitem__("ui", argv) or 0)
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--no-wake", "--no-auth"])
    with pytest.raises(SystemExit):
        main_module.main()
    assert calls["ui"] is not None
    assert "--no-wake" in calls["ui"]
    assert "--no-auth" in calls["ui"]
    assert calls["diego"] is None


def test_main_headless_preserved(main_module, monkeypatch):
    """`python main.py --headless` keeps the existing CLI runtime."""
    calls = {"diego": None, "ui": None}
    monkeypatch.setattr(main_module.Diego, "run",
                        lambda **kw: calls.__setitem__("diego", kw))
    import ui.__main__ as ui_main_mod
    monkeypatch.setattr(ui_main_mod, "main",
                        lambda argv=None: calls.__setitem__("ui", argv) or 0)
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--headless", "--no-wake", "--no-auth"])
    main_module.main()  # headless path returns without SystemExit
    assert calls["diego"] is not None
    assert calls["diego"].get("no_wake") is True
    assert calls["diego"].get("no_auth") is True
    assert calls["ui"] is None


# ═══════════════════════════════════════════════════════════════
# 2. Device enumeration
# ═══════════════════════════════════════════════════════════════

def test_input_device_enumeration(monkeypatch):
    mgr = DeviceManager()
    monkeypatch.setattr(mgr, "_sd", lambda: _fake_sd(input_indices=(0, 1),
                                                     output_indices=(2, 3)))
    inputs = mgr.list_input_devices()
    assert {d["index"] for d in inputs} == {0, 1}
    assert all(d["max_input_channels"] > 0 for d in inputs)


def test_output_device_enumeration(monkeypatch):
    mgr = DeviceManager()
    monkeypatch.setattr(mgr, "_sd", lambda: _fake_sd(input_indices=(0, 1),
                                                     output_indices=(2, 3)))
    outputs = mgr.list_output_devices()
    assert {d["index"] for d in outputs} == {2, 3}
    assert all(d["max_output_channels"] > 0 for d in outputs)


# ═══════════════════════════════════════════════════════════════
# 3. Separate persisted settings
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def dev_store(tmp_path, monkeypatch):
    monkeypatch.setattr(device_manager_module, "DEVICES_PATH",
                        tmp_path / "audio_devices.json")
    return DeviceManager()


def test_separate_input_output_settings(dev_store):
    dev_store.save_input_device(2, "ALC256 Analog")
    assert dev_store.get_saved_input_device()["index"] == 2
    assert dev_store.get_saved_output_device() is None

    # Changing the speaker must NOT overwrite the microphone selection.
    dev_store.save_output_device(7, "USB Headset")
    assert dev_store.get_saved_input_device()["index"] == 2
    assert dev_store.get_saved_input_device()["name"] == "ALC256 Analog"
    assert dev_store.get_saved_output_device()["index"] == 7

    # And changing the microphone must NOT overwrite the speaker.
    dev_store.save_input_device(1, "USB Mic")
    assert dev_store.get_saved_output_device()["index"] == 7
    assert dev_store.get_saved_input_device()["index"] == 1


def test_persistence_roundtrip(dev_store):
    dev_store.save_input_device(3, "Mic X")
    dev_store.save_output_device(5, "Spk Y")
    # Re-read through a NEW manager instance (simulates restart).
    fresh = DeviceManager()
    assert fresh.get_saved_input_device()["index"] == 3
    assert fresh.get_saved_output_device()["index"] == 5


def test_missing_input_device_not_persisted(dev_store, monkeypatch):
    mgr = dev_store
    monkeypatch.setattr(mgr, "list_input_devices", lambda: [])
    res = mgr.set_input_device(42)
    assert res["ok"] is False
    assert mgr.get_saved_input_device() is None


def test_missing_output_device_not_persisted(dev_store, monkeypatch):
    mgr = dev_store
    monkeypatch.setattr(mgr, "list_output_devices", lambda: [])
    res = mgr.set_output_device(99)
    assert res["ok"] is False
    assert mgr.get_saved_output_device() is None


# ═══════════════════════════════════════════════════════════════
# 4. Input switching — independent from output
# ═══════════════════════════════════════════════════════════════

def _make_running_manager():
    """AudioManager that looks like a live stream on device 5."""
    from voice.audio_manager import AudioManager
    am = AudioManager()
    am._running = True
    am._device_index = 5
    am._device_name = "Old Mic"
    am._mic_verified = True
    old_stream = MagicMock(name="old_stream")
    am._stream = old_stream
    # Pre-existing ring-buffer content → monotonic counters must survive.
    import numpy as np
    am._ring_buffer.put(np.zeros(512, dtype=np.float32))
    return am, old_stream


def test_input_switch_restarts_single_stream(monkeypatch):
    am, old_stream = _make_running_manager()
    started_with = []

    def fake_start():
        from voice.settings import voice_settings
        started_with.append(voice_settings.device_index)
        am._running = True
        am._device_index = started_with[-1]
        am._device_name = f"Dev {started_with[-1]}"
        am._mic_verified = True
        am._stream = MagicMock(name="new_stream")
        return True

    monkeypatch.setattr(am, "start", fake_start)
    res = am.switch_input_device(7)

    assert res["ok"] is True
    assert res["device_index"] == 7
    assert res["fallback"] is False
    # Old stream stopped/closed exactly once (stop → reconfigure → restart).
    assert old_stream.stop.called and old_stream.close.called
    assert started_with == [7]
    # Ring-buffer counters preserved (VAD/STT consumers keep cursors).
    assert am._ring_buffer.total_samples == 512


def test_input_switch_does_not_touch_output(monkeypatch):
    am, _ = _make_running_manager()
    monkeypatch.setattr(am, "start", lambda: True)

    import voice.streaming_tts as st
    tts_spy = MagicMock()
    monkeypatch.setattr(st.streaming_tts, "set_output_device", tts_spy)
    monkeypatch.setattr(st.streaming_tts, "stop", tts_spy.stop)

    am.switch_input_device(7)
    assert tts_spy.set_output_device.call_count == 0


def test_input_switch_missing_device_falls_back(monkeypatch):
    am, old_stream = _make_running_manager()

    def fake_start():
        from voice.settings import voice_settings
        if voice_settings.device_index == 9:
            return False  # target device dead/missing
        am._running = True
        am._device_index = voice_settings.device_index
        am._device_name = f"Dev {voice_settings.device_index}"
        am._mic_verified = True
        am._stream = MagicMock()
        return True

    monkeypatch.setattr(am, "start", fake_start)
    res = am.switch_input_device(9)

    assert res["ok"] is True, "must fall back to the previous working device"
    assert res["fallback"] is True
    assert res["device_index"] == 5


def test_output_switch_does_not_touch_microphone(monkeypatch):
    from voice.streaming_tts import StreamingTTS
    tts = StreamingTTS()
    am, old_stream = _make_running_manager()

    res = tts.set_output_device(3)
    assert res["ok"] is True
    assert tts.output_device == 3
    # Microphone capture untouched: old stream NOT stopped, still running.
    assert not old_stream.stop.called
    assert am._running is True


def test_output_switch_applied_safely_by_worker():
    """The player's stream is recreated by the worker path, never mid-write."""
    from voice.streaming_tts import _InterruptiblePlayer
    player = _InterruptiblePlayer()
    fake_stream = MagicMock()
    player._stream = fake_stream
    player._stream_created = True

    player.set_output_device(3)
    assert player.output_device == 3
    player._apply_pending_device_switch()
    assert fake_stream.abort.called and fake_stream.close.called
    assert player._stream is None
    assert player._stream_created is False


def test_output_switch_invalid_device_reports_error(dev_store, monkeypatch):
    mgr = dev_store
    monkeypatch.setattr(mgr, "list_output_devices", lambda: [])
    res = mgr.set_output_device(123)
    assert res["ok"] is False
    assert "not found" in res["error"]


# ═══════════════════════════════════════════════════════════════
# 5. Startup restore (persistence + missing-device fallback)
# ═══════════════════════════════════════════════════════════════

def test_startup_restore_both_devices(dev_store, monkeypatch):
    mgr = dev_store
    mgr.save_input_device(1, "USB Mic")
    mgr.save_output_device(3, "USB Headset")

    monkeypatch.setattr(mgr, "list_input_devices", lambda: [
        {"index": 1, "name": "USB Mic", "max_input_channels": 1}])
    monkeypatch.setattr(mgr, "list_output_devices", lambda: [
        {"index": 3, "name": "USB Headset", "max_output_channels": 2}])

    # Playback validation (new requirement): the saved output is restored
    # only after a successful playback check.
    monkeypatch.setattr(mgr, "test_output_device",
                        lambda idx, samplerate=None: {"ok": True})

    from voice.settings import voice_settings
    import voice.streaming_tts as st
    tts_spy = MagicMock()
    monkeypatch.setattr(st.streaming_tts, "set_output_device", tts_spy)

    result = mgr.apply_persisted_devices()
    assert result["input_restored"]["index"] == 1
    assert voice_settings.device_index == 1
    assert result["output_restored"]["index"] == 3
    tts_spy.assert_called_once_with(3)


def test_startup_restore_missing_devices_fall_back_safely(dev_store, monkeypatch):
    mgr = dev_store
    mgr.save_input_device(50, "Gone Mic")
    mgr.save_output_device(60, "Gone Speaker")

    # Neither device exists anymore.
    monkeypatch.setattr(mgr, "list_input_devices", lambda: [
        {"index": 1, "name": "Other Mic", "max_input_channels": 1}])
    monkeypatch.setattr(mgr, "list_output_devices", lambda: [])

    from voice.settings import voice_settings
    import voice.streaming_tts as st
    tts_spy = MagicMock()
    monkeypatch.setattr(st.streaming_tts, "set_output_device", tts_spy)
    voice_settings.device_index = None

    result = mgr.apply_persisted_devices()
    assert result["input_restored"] is None
    assert result["output_restored"] is None
    assert voice_settings.device_index is None  # auto-detection will pick
    tts_spy.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# 6. Runtime health status model
# ═══════════════════════════════════════════════════════════════

def _patch_health(monkeypatch, versions=None, imports=None):
    from core import runtime_health as rh
    versions = versions or {}
    imports = imports or {}

    def fake_import_version(name):
        return versions.get(name)

    def fake_check_import(name):
        return bool(imports.get(name))

    import httpx
    monkeypatch.setattr(rh, "_import_version", fake_import_version)
    monkeypatch.setattr(rh, "_check_import", fake_check_import)
    monkeypatch.setattr(httpx, "get",
                        MagicMock(side_effect=OSError("unreachable")))
    return rh


def test_health_ready_components_show_no_fallback(monkeypatch):
    rh = _patch_health(monkeypatch, versions={
        "sounddevice": "0.5.5", "silero_vad": "1.0",
        "faster_whisper": "1.0", "kokoro": "0.1",
        "openwakeword": "0.6.0",
    }, imports={"openwakeword": True})
    # The wake check uses the canonical resolver (voice/wake_resolver.py);
    # stub it so this test exercises the health logic deterministically.
    from types import SimpleNamespace
    import voice.wake_resolver as wr
    fake_res = SimpleNamespace(
        found=True,
        path=Path("/fake/wake_model.onnx"),
        verifier_path=Path("/fake/verifier.pkl"),
        reason="resolved /fake/wake_model.onnx (source=test)")
    monkeypatch.setattr(wr, "resolve_wake_model", lambda *a, **k: fake_res)
    h = rh.RuntimeHealth()
    comps = {c.name: c for c in h.run(no_wake=False)}

    assert comps["microphone"].status == rh.READY
    assert comps["vad"].status == rh.READY
    assert comps["stt"].status == rh.READY
    assert comps["tts"].status == rh.READY
    assert comps["wake"].status == rh.READY
    # READY lines must NOT advertise fallbacks (the misleading log fix).
    for name in ("microphone", "vad", "stt", "tts", "wake"):
        assert "fallback=" not in comps[name].to_line(), comps[name].to_line()


def test_health_silero_missing_is_degraded_with_fallback(monkeypatch):
    rh = _patch_health(monkeypatch, versions={
        "sounddevice": "0.5.5", "faster_whisper": "1.0", "kokoro": "0.1",
        "openwakeword": "0.6.0",
    }, imports={"openwakeword": True})
    from types import SimpleNamespace
    import voice.wake_resolver as wr
    fake_res = SimpleNamespace(
        found=True,
        path=Path("/fake/wake_model.onnx"),
        verifier_path=None,
        reason="resolved /fake/wake_model.onnx (source=test)")
    monkeypatch.setattr(wr, "resolve_wake_model", lambda *a, **k: fake_res)
    h = rh.RuntimeHealth()
    comps = {c.name: c for c in h.run(no_wake=False)}
    assert comps["vad"].status == rh.DEGRADED
    line = comps["vad"].to_line()
    assert "fallback=energy-based VAD" in line
    # And the honest aggregate: voice is available but degraded.
    assert h.voice_ready is True


def test_health_whisper_missing_is_failed(monkeypatch):
    rh = _patch_health(monkeypatch, versions={
        "sounddevice": "0.5.5", "silero_vad": "1.0", "kokoro": "0.1",
    }, imports={"openwakeword": True})
    h = rh.RuntimeHealth()
    comps = {c.name: c for c in h.run(no_wake=False)}
    assert comps["stt"].status == rh.FAILED
    assert h.voice_ready is False


def test_health_no_wake_is_bypassed(monkeypatch):
    rh = _patch_health(monkeypatch, versions={
        "sounddevice": "0.5.5", "silero_vad": "1.0",
        "faster_whisper": "1.0", "kokoro": "0.1",
    }, imports={"openwakeword": True})
    h = rh.RuntimeHealth()
    comps = {c.name: c for c in h.run(no_wake=True)}
    assert comps["wake"].status == rh.BYPASSED
    assert "fallback=" not in comps["wake"].to_line()


def test_health_wake_missing_when_enabled(monkeypatch):
    rh = _patch_health(monkeypatch, versions={
        "sounddevice": "0.5.5", "silero_vad": "1.0",
        "faster_whisper": "1.0", "kokoro": "0.1",
    }, imports={})
    h = rh.RuntimeHealth()
    comps = {c.name: c for c in h.run(no_wake=False)}
    assert comps["wake"].status == rh.MISSING


def test_health_ollama_unavailable_is_fail_safe(monkeypatch):
    rh = _patch_health(monkeypatch, versions={
        "sounddevice": "0.5.5", "silero_vad": "1.0",
        "faster_whisper": "1.0", "kokoro": "0.1",
    }, imports={"openwakeword": True})
    h = rh.RuntimeHealth()
    comps = {c.name: c for c in h.run(no_wake=False)}
    assert comps["llm"].status == rh.UNAVAILABLE
    # Ollama down must NOT block voice operation.
    assert h.voice_ready is True


# ═══════════════════════════════════════════════════════════════
# 7. Wake failure: no infinite retry loop
# ═══════════════════════════════════════════════════════════════

def test_wake_retry_loop_is_bounded(monkeypatch):
    """After MODEL_MAX_RETRIES the listener stops retrying (no spin)."""
    import voice.wake_listener as wl

    listener = wl.WakeListener()
    fake_mm = MagicMock()
    fake_mm.loaded = False
    fake_mm.load = MagicMock(return_value=False)
    fake_mm.load_error = "no model"
    fake_mm.wake_phrase = "hello diego"
    fake_mm.model_name = None
    fake_mm.threshold = 0.5
    monkeypatch.setattr(wl, "wake_model_manager", fake_mm)
    monkeypatch.setattr(wl, "audio_manager",
                        SimpleNamespace(total_samples=0, read_since=lambda t: ([], t)))
    listener._last_total = 0
    listener._model_retry_count = wl.MODEL_MAX_RETRIES  # retries exhausted

    async def scenario():
        task = asyncio.create_task(listener.wait_for_wake(lambda: True))
        await asyncio.sleep(0.5)
        # In the idle window the model must NOT be reloaded repeatedly.
        assert fake_mm.load.call_count == 0
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_wake_retry_cadence_is_slow_not_tight(monkeypatch):
    """Before the cap: retries exist but are cadence-limited (5 s apart)."""
    import voice.wake_listener as wl

    listener = wl.WakeListener()
    fake_mm = MagicMock()
    fake_mm.loaded = False
    fake_mm.load = MagicMock(return_value=False)
    fake_mm.load_error = "no model"
    fake_mm.wake_phrase = "hello diego"
    fake_mm.model_name = None
    fake_mm.threshold = 0.5
    monkeypatch.setattr(wl, "wake_model_manager", fake_mm)
    monkeypatch.setattr(wl, "audio_manager",
                        SimpleNamespace(total_samples=0, read_since=lambda t: ([], t)))
    listener._last_total = 0

    async def scenario():
        task = asyncio.create_task(listener.wait_for_wake(lambda: True))
        await asyncio.sleep(0.7)
        calls = fake_mm.load.call_count
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # At most one retry within <MODEL_RETRY_S — a tight loop would
        # have hammered `load` dozens of times.
        assert calls <= 2, f"retry loop too aggressive: {calls} calls in 0.7s"

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_engine_degrades_to_listen_when_wake_unavailable(monkeypatch):
    """Wake model missing at boot (and --no-wake not set) → engine enters
    the LISTEN path instead of spinning in WAKE with endless retries."""
    import core.conversation_engine as ce

    # Stub the heavy module-level dependencies the run loop touches.
    monkeypatch.setattr(ce, "audio_manager",
                        SimpleNamespace(is_running=True, total_samples=0))
    monkeypatch.setattr(ce, "streaming_tts", MagicMock())
    monkeypatch.setattr(ce, "command_listener", MagicMock())
    monkeypatch.setattr(ce, "wake_model_manager", MagicMock(load_error="gone"))
    import voice.vad as vad_mod
    monkeypatch.setattr(vad_mod, "unified_vad", MagicMock())
    fake_ks = MagicMock()
    monkeypatch.setitem(sys.modules, "knowledge.service",
                        SimpleNamespace(knowledge_service=fake_ks))

    eng = ce.ConversationEngine()
    monkeypatch.setattr(eng, "_ensure_wake_model", lambda: False)

    visited = []

    async def fake_session():
        visited.append(eng._state)
        eng._running = False  # end the forever loop after one session
    monkeypatch.setattr(eng, "_conversation_session", fake_session)

    await asyncio.wait_for(eng.run(no_wake=False), timeout=10)

    assert eng._wake_active is False
    assert visited == [ce.EngineState.LISTEN], (
        "engine must skip WAKE and proceed directly to LISTEN when the "
        "wake model is unavailable")


# ═══════════════════════════════════════════════════════════════
# 8. UI: AUDIO panel + event bridge
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _patch_panel_devices(monkeypatch, cur_in=None, cur_out=None):
    import voice.device_manager as dm
    inputs = [{"index": 0, "name": "ALC256 Analog", "is_default": True},
              {"index": 1, "name": "USB Mic", "is_default": False}]
    outputs = [{"index": 2, "name": "ALC256 Analog", "is_default": True},
               {"index": 3, "name": "USB Headset", "is_default": False}]
    monkeypatch.setattr(dm.device_manager, "list_input_devices",
                        lambda: list(inputs))
    monkeypatch.setattr(dm.device_manager, "list_output_devices",
                        lambda: list(outputs))
    monkeypatch.setattr(dm.device_manager, "get_current_input_device",
                        lambda: cur_in or {"index": 0, "name": "ALC256 Analog",
                                           "running": True, "verified": True})
    monkeypatch.setattr(dm.device_manager, "get_current_output_device",
                        lambda: cur_out or {"index": 3, "name": "USB Headset",
                                            "ready": True})
    return dm


def _drain_qt(qapp, seconds=2.0):
    from PySide6.QtCore import QEventLoop, QTimer
    end = time.time() + seconds
    while time.time() < end:
        loop = QEventLoop()
        QTimer.singleShot(50, loop.quit)
        loop.exec()
        if qapp:
            qapp.processEvents()


def test_audio_panel_enumerates_independent_lists(qapp, monkeypatch):
    _patch_panel_devices(monkeypatch)
    from ui.widgets import AudioDevicePanel
    panel = AudioDevicePanel()
    try:
        assert panel.input_combo.count() == 2
        assert panel.output_combo.count() == 2
        # Input list ≠ output list (independent enumeration).
        in_idx = {panel.input_combo.itemData(i)
                  for i in range(panel.input_combo.count())}
        out_idx = {panel.output_combo.itemData(i)
                   for i in range(panel.output_combo.count())}
        assert in_idx == {0, 1}
        assert out_idx == {2, 3}
        # Current selection shown clearly.
        assert panel.input_combo.currentData() == 0
        assert panel.output_combo.currentData() == 3
        assert "ALC256 Analog" in panel.input_status.text()
        assert "USB Headset" in panel.output_status.text()
    finally:
        panel.deleteLater()


def test_audio_panel_input_switch_leaves_output_alone(qapp, monkeypatch):
    dm = _patch_panel_devices(monkeypatch)
    calls = {"input": [], "output": []}
    monkeypatch.setattr(dm.device_manager, "set_input_device",
                        lambda idx: calls["input"].append(idx)
                        or {"ok": True, "device_index": idx,
                            "device_name": f"Dev {idx}", "fallback": False})
    monkeypatch.setattr(dm.device_manager, "set_output_device",
                        lambda idx: calls["output"].append(idx)
                        or {"ok": True, "device_index": idx,
                            "device_name": f"Dev {idx}", "fallback": False})

    from ui.widgets import AudioDevicePanel
    panel = AudioDevicePanel()
    try:
        panel.input_combo.setCurrentIndex(1)  # select "USB Mic"
        _drain_qt(qapp)
        assert calls["input"] == [1]
        assert calls["output"] == [], "input switch must not touch output"
    finally:
        panel.deleteLater()


def test_audio_panel_output_switch_leaves_input_alone(qapp, monkeypatch):
    dm = _patch_panel_devices(monkeypatch)
    calls = {"input": [], "output": []}
    monkeypatch.setattr(dm.device_manager, "set_input_device",
                        lambda idx: calls["input"].append(idx)
                        or {"ok": True, "device_index": idx,
                            "device_name": f"Dev {idx}", "fallback": False})
    monkeypatch.setattr(dm.device_manager, "set_output_device",
                        lambda idx: calls["output"].append(idx)
                        or {"ok": True, "device_index": idx,
                            "device_name": f"Dev {idx}", "fallback": False})

    from ui.widgets import AudioDevicePanel
    panel = AudioDevicePanel()
    try:
        panel.output_combo.setCurrentIndex(0)  # select "ALC256 Analog"
        _drain_qt(qapp)
        assert calls["output"] == [2]
        assert calls["input"] == [], "output switch must not touch input"
    finally:
        panel.deleteLater()


def test_audio_panel_missing_device_shows_error(qapp, monkeypatch):
    dm = _patch_panel_devices(monkeypatch)
    monkeypatch.setattr(dm.device_manager, "set_input_device",
                        lambda idx: {"ok": False, "error": "device gone",
                                     "device_index": None,
                                     "device_name": "", "fallback": False})

    from ui.widgets import AudioDevicePanel
    panel = AudioDevicePanel()
    try:
        panel.input_combo.setCurrentIndex(1)
        _drain_qt(qapp)
        status = panel.input_status.text()
        assert "device gone" in status
    finally:
        panel.deleteLater()


def test_audio_panel_in_main_window(qapp):
    """The AUDIO panel is present in the HUD's right column."""
    from ui.event_bridge import EventBridge
    from ui.main_window import DiegoMainWindow
    b = EventBridge()
    b.start()
    try:
        w = DiegoMainWindow(bridge=b, loop=None)
        try:
            panel = getattr(w, "_audio_panel", None)
            assert panel is not None, "AUDIO panel missing from right column"
            assert panel.input_combo is not None
            assert panel.output_combo is not None
        finally:
            w.close()
    finally:
        b.stop()


def test_ui_event_bridge_remains_functional(qapp):
    """Existing bridge behaviour still works (regression guard)."""
    from ui.event_bridge import EventBridge, UIEventType, UIEvent
    bridge = EventBridge()
    bridge.start()
    received = []
    bridge.final_transcript.connect(received.append)
    bridge.state_changed.connect(lambda s: received.append(("state", s)))
    try:
        bridge.emit(UIEvent(UIEventType.FINAL_TRANSCRIPT, {"text": "hello"}))
        bridge.emit(UIEvent(UIEventType.LISTENING))
        _drain_qt(qapp, 0.5)
        assert "hello" in received
        assert ("state", "Listening") in received
    finally:
        bridge.stop()