"""
Production-blocker regression tests (2026-09-04).

Covers the runtime integration defects reported in the latest
`python main.py` startup log:

  A. Wake model actual asset resolution — the canonical resolver
     (voice/wake_resolver.py) must find the REAL bundled ONNX asset
     (hey_jarvis_v0.1.onnx) and health + runtime must select the SAME
     model (deeper coverage in tests/test_wake_resolver.py).
  B. Wake startup succeeds and WAKE → FACE_AUTH ordering is preserved.
  C. Default audio OUTPUT handling:
       - a virtual ALSA `default` index is never blindly restored/played
       - the real OS default playback device is resolved + validated
       - a missing saved output falls back safely
       - input and output selections are fully independent
  D. GUI threading:
       - Qt main thread hosts the GUI dispatcher
       - ConversationEngine on the background thread no longer logs
         "Engine not on main thread — GUI disabled"
  E. EventBridge thread safety (emit from any thread).
"""

import asyncio
import inspect
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════
# A. Wake model — actual asset resolution (health == runtime)
# ═══════════════════════════════════════════════════════════════

def test_wake_asset_resolution_finds_real_onnx_file():
    """The canonical resolver must find an EXISTING hey_jarvis ONNX
    asset on disk (the exact production-blocker symptom was
    'wake FAILED reason=no wake model found')."""
    from voice.wake_resolver import resolve_wake_model

    res = resolve_wake_model()
    assert res.found, f"resolver failed: {res.reason()}"
    assert res.path is not None
    assert res.path.exists(), f"resolved path does not exist: {res.path}"
    assert res.path.suffix == ".onnx"
    assert res.path.name == "hey_jarvis_v0.1.onnx"
    assert res.source == "metadata.base_model"
    assert res.base_model_requested == "hey_jarvis_v0.1"


def test_health_and_runtime_loader_select_same_wake_model():
    """Health probe and runtime loader must resolve the SAME asset."""
    from voice.wake_resolver import resolve_wake_model
    from voice.wake_model_manager import WakeModelManager
    from core import runtime_health as rh

    canon = resolve_wake_model()
    assert canon.found

    mgr = WakeModelManager()
    assert mgr.load() is True
    assert str(mgr.model_path) == str(canon.path)

    comps = rh.RuntimeHealth().run(no_wake=False)
    wake = [c for c in comps if c.name == "wake"][0]
    assert wake.status == rh.READY
    assert wake.asset == str(canon.path)


def test_wake_startup_succeeds_and_scores():
    """The wake listener's model loads and can SCORE audio (the runtime
    path, not just health)."""
    from voice.wake_model_manager import wake_model_manager
    import numpy as np

    assert wake_model_manager.load() is True
    assert wake_model_manager.loaded
    preds = wake_model_manager.predict(np.zeros(16000, dtype=np.int16))
    assert isinstance(preds, dict)
    # The model produces a score for its class even on silence.
    assert any(v >= 0.0 for v in preds.values())


def test_resolver_covers_legacy_openwakeword_layouts():
    """The bundled-dir candidates must include BOTH the new
    resources/models layout and the legacy models/ layout."""
    from voice import wake_resolver as wr

    dirs = [str(d) for d in wr._bundled_model_dirs()]
    assert any(d.endswith("openwakeword/resources/models") for d in dirs)
    assert any(
        d.endswith("openwakeword/models") and "resources" not in
        d.rsplit("openwakeword/", 1)[1]
        for d in dirs
    )


# ═══════════════════════════════════════════════════════════════
# B. Wake → FACE_AUTH ordering
# ═══════════════════════════════════════════════════════════════

async def test_wake_transition_to_face_auth_is_allowed_and_used():
    """WAKE → FACE_AUTH is a legal transition and the post-wake path
    runs the auth gate (deeper behavioural coverage in
    tests/test_startup_auth_independence.py)."""
    from core.conversation_engine import (
        ConversationEngine, EngineState, ALLOWED_TRANSITIONS,
    )

    assert EngineState.FACE_AUTH in ALLOWED_TRANSITIONS[EngineState.WAKE]

    eng = ConversationEngine()
    eng._set_state(EngineState.WAKE)

    async def provider():
        return "Tester"

    eng.set_auth_provider(provider)
    # Greeting speak is stubbed (no TTS hardware in tests).
    async def _noop_speak(text):
        return True
    eng._speak_guarded = _noop_speak

    await eng._face_auth_gate(trigger="wake")

    assert eng._state == EngineState.FACE_AUTH
    assert eng._auth_user == "Tester"


# ═══════════════════════════════════════════════════════════════
# C. Default audio output handling
# ═══════════════════════════════════════════════════════════════

def _make_dev(index, name, hostapi="ALSA", out=2, rate=44100.0,
              is_default=False):
    return {"index": index, "name": name, "hostapi": hostapi,
            "max_output_channels": out, "default_samplerate": float(rate),
            "is_default": is_default}


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    """Point the device store at a temp file (never touch real data)."""
    import voice.device_manager as dm
    store_path = tmp_path / "audio_devices.json"
    monkeypatch.setattr(dm, "DEVICES_PATH", store_path)
    return store_path


@pytest.fixture
def fake_tts(monkeypatch):
    """Capture set_output_device calls on the StreamingTTS singleton."""
    from voice.streaming_tts import streaming_tts
    calls = []
    monkeypatch.setattr(streaming_tts, "set_output_device",
                        lambda idx: calls.append(idx) or
                        {"ok": True, "device_index": idx})
    return calls


def test_resolve_default_output_skips_virtual_default(monkeypatch):
    """The PortAudio default may be the virtual ALSA `default` device —
    resolution must skip it and return a CONCRETE validated device."""
    from voice.device_manager import device_manager as dmgr

    devices = [
        _make_dev(5, "HD-Audio Generic: ALC256 Analog (hw:2,0)"),
        _make_dev(8, "pulse", hostapi="pulse", out=32),
        _make_dev(9, "default", hostapi="ALSA", out=32, is_default=True),
    ]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)
    monkeypatch.setattr(dmgr, "_sd", lambda: SimpleNamespace(
        default=SimpleNamespace(device=[5, 9])))
    # Every concrete device validates.
    monkeypatch.setattr(dmgr, "test_output_device",
                        lambda idx, samplerate=None: {"ok": True})

    resolved = dmgr.resolve_default_output_device()
    assert resolved == 8  # pulse — concrete, routes to the real OS sink
    assert resolved != 9  # never the virtual `default` index


def test_resolve_default_output_returns_none_without_devices(monkeypatch):
    from voice.device_manager import device_manager as dmgr
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: [])
    assert dmgr.resolve_default_output_device() is None


def test_concrete_output_device_validation(monkeypatch):
    """test_output_device opens the stream and reports failure when the
    device cannot be opened (playback validation)."""
    from voice.device_manager import device_manager as dmgr

    devices = [_make_dev(3, "Broken Speaker")]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)
    monkeypatch.setattr(dmgr, "find_output_device",
                        lambda idx: next((d for d in devices
                                          if d["index"] == idx), None))

    class FakeSD:
        def OutputStream(self, **kwargs):
            raise OSError("device unavailable")

    monkeypatch.setattr(dmgr, "_sd", lambda: FakeSD())
    result = dmgr.test_output_device(3)
    assert result["ok"] is False
    assert "unavailable" in (result.get("error") or "")


def test_apply_persisted_bypasses_virtual_saved_output(
        monkeypatch, isolated_store, fake_tts):
    """A saved output of [9] 'default' (virtual) must NOT be restored —
    the concrete OS default output is resolved and used instead."""
    import voice.device_manager as dm
    from voice.device_manager import device_manager as dmgr

    isolated_store.write_text(json.dumps({
        "input_device": {"index": 5, "name": "Mic"},
        "output_device": {"index": 9, "name": "default"},
    }))

    devices = [
        _make_dev(5, "HD-Audio Generic: ALC256 Analog (hw:2,0)"),
        _make_dev(8, "pulse", hostapi="pulse", out=32),
        _make_dev(9, "default", out=32, is_default=True),
    ]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)
    monkeypatch.setattr(
        dmgr, "find_input_device",
        lambda idx: next((d for d in devices if d["index"] == idx), None))
    monkeypatch.setattr(dmgr, "_sd", lambda: SimpleNamespace(
        default=SimpleNamespace(device=[5, 9])))
    monkeypatch.setattr(dmgr, "test_output_device",
                        lambda idx, samplerate=None: {"ok": True})

    result = dmgr.apply_persisted_devices()

    # Input restored untouched
    assert result["input_restored"]["index"] == 5
    # Virtual saved output NOT restored
    assert result["output_restored"] is None
    # Concrete default resolved and applied to TTS
    assert result["output_resolved_default"]["index"] == 8
    assert fake_tts == [8]


def test_missing_saved_output_falls_back_to_resolved_default(
        monkeypatch, isolated_store, fake_tts):
    """A saved output that disappeared must fall back to the resolved
    OS default without crashing."""
    import voice.device_manager as dm
    from voice.device_manager import device_manager as dmgr

    isolated_store.write_text(json.dumps({
        "output_device": {"index": 42, "name": "Gone USB Speaker"},
    }))

    devices = [_make_dev(8, "pulse", hostapi="pulse", out=32)]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)
    monkeypatch.setattr(dmgr, "_sd", lambda: SimpleNamespace(
        default=SimpleNamespace(device=[8, 8])))
    monkeypatch.setattr(dmgr, "test_output_device",
                        lambda idx, samplerate=None: {"ok": True})

    result = dmgr.apply_persisted_devices()
    assert result["output_restored"] is None
    assert result["output_resolved_default"]["index"] == 8
    assert fake_tts == [8]


def test_valid_saved_output_is_restored(monkeypatch, isolated_store, fake_tts):
    """A concrete, validated saved output IS restored."""
    import voice.device_manager as dm
    from voice.device_manager import device_manager as dmgr

    isolated_store.write_text(json.dumps({
        "output_device": {"index": 5, "name": "Analog Speaker"},
    }))
    devices = [_make_dev(5, "Analog Speaker")]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)
    monkeypatch.setattr(dmgr, "test_output_device",
                        lambda idx, samplerate=None: {"ok": True})

    result = dmgr.apply_persisted_devices()
    assert result["output_restored"]["index"] == 5
    assert fake_tts == [5]
    assert "output_resolved_default" not in result


def test_set_output_device_rejects_unplayable_device(
        monkeypatch, isolated_store, fake_tts):
    """A device that fails playback validation is never applied or
    persisted."""
    import json as _json
    from voice.device_manager import device_manager as dmgr

    devices = [
        _make_dev(5, "Analog Speaker"),
        _make_dev(6, "Broken HDMI"),
    ]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)

    def fake_test(index, samplerate=None):
        return {"ok": index != 6,
                "error": None if index != 6 else "no playback"}
    monkeypatch.setattr(dmgr, "test_output_device", fake_test)

    result = dmgr.set_output_device(6)
    assert result["ok"] is False
    assert fake_tts == []                      # never applied
    import voice.device_manager as dm_mod
    if isolated_store.exists():
        stored = _json.loads(isolated_store.read_text())
        assert dm_mod.OUTPUT_KEY not in stored  # never persisted


def test_set_output_device_persists_and_keeps_mic(
        monkeypatch, isolated_store, fake_tts):
    """Selecting a speaker persists it WITHOUT touching the microphone
    selection (input/output independence)."""
    import json as _json
    import voice.device_manager as dm_mod
    from voice.device_manager import device_manager as dmgr

    isolated_store.write_text(_json.dumps({
        "input_device": {"index": 5, "name": "My Mic"},
    }))
    devices = [_make_dev(5, "Analog Speaker")]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)
    monkeypatch.setattr(dmgr, "test_output_device",
                        lambda idx, samplerate=None: {"ok": True})

    result = dmgr.set_output_device(5)
    assert result["ok"] is True
    assert fake_tts == [5]

    stored = _json.loads(isolated_store.read_text())
    assert stored[dm_mod.OUTPUT_KEY]["index"] == 5
    # Microphone selection untouched
    assert stored[dm_mod.INPUT_KEY]["index"] == 5
    assert stored[dm_mod.INPUT_KEY]["name"] == "My Mic"


def test_set_output_device_maps_virtual_default_to_concrete(
        monkeypatch, isolated_store, fake_tts):
    """Selecting the virtual 'default' entry stores/plays a CONCRETE
    resolved device instead of the virtual index."""
    import json as _json
    import voice.device_manager as dm_mod
    from voice.device_manager import device_manager as dmgr

    devices = [
        _make_dev(8, "pulse", hostapi="pulse", out=32),
        _make_dev(9, "default", out=32, is_default=True),
    ]
    monkeypatch.setattr(dmgr, "list_output_devices", lambda: devices)
    monkeypatch.setattr(dmgr, "_sd", lambda: SimpleNamespace(
        default=SimpleNamespace(device=[9, 9])))
    monkeypatch.setattr(dmgr, "test_output_device",
                        lambda idx, samplerate=None: {"ok": True})

    result = dmgr.set_output_device(9)  # user clicked the virtual default
    assert result["ok"] is True
    assert fake_tts == [8]              # concrete device used

    stored = _json.loads(isolated_store.read_text())
    assert stored[dm_mod.OUTPUT_KEY]["index"] == 8
    assert stored[dm_mod.OUTPUT_KEY]["name"] != "default"


def test_input_output_independence_store_keys(isolated_store):
    """save_output_device never modifies input_device and vice versa."""
    from voice.device_manager import device_manager as dmgr
    import json as _json

    dmgr.save_input_device(5, "Mic A")
    dmgr.save_output_device(8, "Speaker B")

    stored = _json.loads(isolated_store.read_text())
    assert stored[dm_mod_INPUT_KEY()]["index"] == 5
    assert stored["output_device"]["index"] == 8

    # Re-saving output leaves input intact
    dmgr.save_output_device(3, "Speaker C")
    stored = _json.loads(isolated_store.read_text())
    assert stored["input_device"]["index"] == 5
    assert stored["output_device"]["index"] == 3


def dm_mod_INPUT_KEY():
    from voice.device_manager import INPUT_KEY
    return INPUT_KEY


# ═══════════════════════════════════════════════════════════════
# D. GUI threading
# ═══════════════════════════════════════════════════════════════

OLD_WARNING = "Engine not on main thread — GUI disabled"


def test_old_gui_warning_removed_from_source():
    """The forbidden log line must be gone from the engine entirely."""
    import core.conversation_engine as ce
    src = inspect.getsource(ce)
    assert OLD_WARNING not in src


async def test_engine_uses_host_owned_gui_on_background_thread(caplog):
    """Engine on a background thread with a host-owned GUI dispatcher:
    no warning, no gui.start() call from the engine."""
    import core.conversation_engine as ce

    started = {"calls": 0}

    class FakeGUI:
        available = True
        main_thread_ident = 12345

        @staticmethod
        def start():
            started["calls"] += 1
            return True

    monkey_gui = FakeGUI()
    with patch.object(ce, "gui", monkey_gui), caplog.at_level("INFO"):
        eng = ce.ConversationEngine()
        await asyncio.get_event_loop().run_in_executor(
            None, eng._setup_gui)

    assert started["calls"] == 0, \
        "engine must NOT start a host-owned GUI dispatcher"
    assert not any(OLD_WARNING in r.message for r in caplog.records)
    assert any("hosted by the UI main thread" in r.message
               for r in caplog.records)


async def test_engine_starts_gui_on_main_thread(monkeypatch):
    """Headless CLI mode: engine on the MAIN thread starts + pumps the
    dispatcher itself."""
    import core.conversation_engine as ce

    calls = {"start": 0}

    class FakeGUI:
        available = False
        main_thread_ident = None

        @staticmethod
        def start():
            calls["start"] += 1
            return True

        @staticmethod
        async def pump(interval=0.03):
            await asyncio.sleep(3600)

    monkeypatch.setattr(ce, "gui", FakeGUI())
    eng = ce.ConversationEngine()
    eng._setup_gui()  # synchronous method
    assert calls["start"] == 1
    assert eng._gui_pump_task is not None
    eng._gui_pump_task.cancel()


async def test_engine_background_thread_without_host_runs_headless(caplog):
    """Background engine without a GUI host: headless fallback, and the
    OLD misleading warning text is never emitted."""
    import core.conversation_engine as ce

    class FakeGUI:
        available = False
        main_thread_ident = None

        @staticmethod
        def start():
            raise AssertionError("gui.start() must not be called off the "
                                 "main thread")

    with patch.object(ce, "gui", FakeGUI()), caplog.at_level("INFO"):
        eng = ce.ConversationEngine()
        await asyncio.get_event_loop().run_in_executor(
            None, eng._setup_gui)

    assert not any(OLD_WARNING in r.message for r in caplog.records)
    assert any("running headless" in r.message for r in caplog.records)


def test_ui_hosts_gui_dispatcher_on_qt_main_thread():
    """Architecture: ui.__main__ starts the GUI dispatcher on the Qt
    main thread BEFORE the pipeline thread starts, and pumps it with a
    QTimer."""
    import ui.__main__ as ui_main

    src = inspect.getsource(ui_main.main)
    assert "gui.start()" in src
    assert "gui.pump_once" in src
    # The GUI host must be set up before the pipeline thread starts.
    assert src.index("gui.start()") < src.index("runtime.start(")
    # Cleanup stops the pump and the dispatcher.
    assert "gui.stop()" in src


async def test_conversation_engine_runs_on_background_thread():
    """DiegoRuntime runs the ConversationEngine pipeline on a background
    thread (never the Qt/main thread)."""
    from ui.__main__ import DiegoRuntime

    seen = {}

    async def fake_pipeline():
        seen["thread"] = threading.current_thread()
        seen["is_main"] = threading.current_thread() is threading.main_thread()

    rt = DiegoRuntime(no_wake=True, no_auth=True)
    rt._run_pipeline = fake_pipeline

    bridge = MagicMock()
    rt.start(bridge)
    rt._thread.join(timeout=10)

    assert seen.get("is_main") is False
    assert seen.get("thread").name == "DiegoPipeline"


# ═══════════════════════════════════════════════════════════════
# E. EventBridge thread safety
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_event_bridge_emit_from_worker_thread_is_queued(qapp):
    """emit() from a NON-Qt thread must queue the event; draining on the
    Qt thread delivers it via the signal."""
    from ui.event_bridge import EventBridge, UIEvent, UIEventType

    bridge = EventBridge()
    received = []
    bridge.state_changed.connect(received.append)

    def worker():
        bridge.emit(UIEvent(UIEventType.STATE_CHANGE, {"state": "FromWorker"}))

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    # Nothing delivered until the Qt thread drains the queue.
    assert received == []
    bridge._drain_queue()
    assert received == ["FromWorker"]


def test_event_bridge_concurrent_emitters_lose_no_events(qapp):
    """Many concurrent emitter threads: every event is delivered exactly
    once (thread-safe queue)."""
    from ui.event_bridge import EventBridge, UIEvent, UIEventType

    bridge = EventBridge()
    received = []
    bridge.state_changed.connect(received.append)

    N_THREADS, N_EVENTS = 8, 25

    def worker(tid):
        for i in range(N_EVENTS):
            bridge.emit(UIEvent(UIEventType.STATE_CHANGE,
                                {"state": f"t{tid}-{i}"}))

    threads = [threading.Thread(target=worker, args=(tid,))
               for tid in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # Drain until empty — 300 events < the 100/batch cap × 3 passes, but
    # loop defensively until the queue is exhausted.
    for _ in range(10):
        if bridge._queue.empty():
            break
        bridge._drain_queue()
    assert bridge._queue.empty()

    assert len(received) == N_THREADS * N_EVENTS
    assert len(set(received)) == N_THREADS * N_EVENTS


def test_event_bridge_convenience_emitters_thread_safe(qapp):
    """emit_error / emit_state from a worker thread deliver the right
    payloads after a Qt-thread drain."""
    from ui.event_bridge import EventBridge

    bridge = EventBridge()
    errors, states = [], []
    bridge.error.connect(errors.append)
    bridge.state_changed.connect(states.append)

    t = threading.Thread(
        target=lambda: (bridge.emit_error("boom"), bridge.emit_state("S")))
    t.start()
    t.join(timeout=5)
    bridge._drain_queue()

    assert errors == ["boom"]
    assert "S" in states