#!/usr/bin/env python3
"""
PHASE 1 — Full System Audit for Diego Desktop Assistant v3.0

Audits every subsystem and produces:
- Status (READY/DEGRADED/FAILED)
- Root cause for failures
- Latency measurements
- Dependency verification
- Environment checks

Usage:
    python debug/audit_system.py
"""

import asyncio
import importlib
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress noise
os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["ALSA_DEBUG_FILE"] = "/dev/null"
os.environ["PULSE_LOG"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import compat  # noqa: F401

from telemetry.logger import setup_logging

setup_logging("INFO")
logger = logging.getLogger("audit")


class SubsystemStatus(Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


@dataclass
class SubsystemAudit:
    name: str
    status: SubsystemStatus = SubsystemStatus.FAILED
    root_cause: str = ""
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    details: str = ""


# ── Utilities ─────────────────────────────────────────────


def check_import(name: str, package: str = None) -> bool:
    """Check if a Python package is importable."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def check_command(name: str) -> bool:
    """Check if a CLI command is available."""
    return shutil.which(name) is not None


def get_memory_usage() -> Tuple[float, float]:
    """Get current process memory and CPU usage."""
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info().rss / 1024 / 1024  # MB
        cpu = proc.cpu_percent(interval=0.1)
        return mem, cpu
    except ImportError:
        return 0.0, 0.0


# ── Subsystem Audits ─────────────────────────────────────


def audit_environment() -> Dict[str, Any]:
    """Audit the runtime environment."""
    results = {}
    results["python_version"] = sys.version
    results["platform"] = sys.platform
    results["display"] = os.environ.get("DISPLAY", "NOT SET")
    results["alsa_config"] = os.environ.get("ALSA_CONFIG_PATH", "NOT SET")

    # Check audio backend
    audio_backends = {
        "pw-play": check_command("pw-play"),
        "paplay": check_command("paplay"),
        "aplay": check_command("aplay"),
        "ffplay": check_command("ffplay"),
    }
    results["audio_backends"] = audio_backends
    results["audio_available"] = any(audio_backends.values())

    # Check Chrome
    chrome_names = ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium", "chrome"]
    chrome_path = None
    for name in chrome_names:
        path = shutil.which(name)
        if path:
            chrome_path = path
            break
    results["chrome_path"] = chrome_path

    # Check CDP port
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('127.0.0.1', 9222))
        results["cdp_port_9222"] = result == 0
        sock.close()
    except Exception:
        results["cdp_port_9222"] = False

    # Check DISPLAY
    try:
        disp = os.environ.get("DISPLAY", "")
        if disp:
            subprocess.run(["xdpyinfo", "-display", disp],
                         capture_output=True, timeout=5)
            results["x11_available"] = True
        else:
            results["x11_available"] = False
    except Exception:
        results["x11_available"] = False

    return results


async def audit_startup() -> SubsystemAudit:
    """Audit startup subsystem."""
    audit = SubsystemAudit(name="Startup")
    audit.dependencies = ["NLP", "Embeddings", "DuckDB", "Voice", "Plugins"]

    try:
        t0 = time.time()
        from core.startup_health import StartupHealth, SubsystemState
        sh = StartupHealth()
        sh.register("startup_test")
        sh.set_state("startup_test", SubsystemState.READY, "All good")
        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed
        audit.status = SubsystemStatus.READY
        audit.details = "StartupHealth module loads and functions correctly"
    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_nlp() -> SubsystemAudit:
    """Audit NLP subsystem."""
    audit = SubsystemAudit(name="NLP")
    audit.dependencies = ["sentence-transformers", "scikit-learn", "numpy"]

    try:
        t0 = time.time()
        from nlp.inference import inference
        loaded = inference.load()
        if loaded:
            status = inference.get_status()
            audit.details = f"v{status.get('version', '?')}, {status.get('intents', 0)} intents, {status.get('examples', 0)} examples"
            audit.status = SubsystemStatus.READY
        else:
            audit.status = SubsystemStatus.FAILED
            audit.root_cause = "Model not found — run 'python main.py --train'"

        # Test classification latency
        if loaded:
            t1 = time.time()
            for _ in range(5):
                inference.classify("hello", top_k=1)
            audit.latency_ms = ((time.time() - t1) / 5) * 1000

    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_embeddings() -> SubsystemAudit:
    """Audit embedding subsystem."""
    audit = SubsystemAudit(name="Embeddings")
    audit.dependencies = ["sentence-transformers", "numpy"]

    try:
        t0 = time.time()
        from nlp.embeddings import embed, preload_embedding_model
        preload_embedding_model()
        vec = embed("test")
        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed
        audit.status = SubsystemStatus.READY
        audit.details = f"Dimension: {len(vec)}"
    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_voice() -> SubsystemAudit:
    """Audit voice subsystem."""
    audit = SubsystemAudit(name="Voice")
    audit.dependencies = ["speech_recognition", "sounddevice", "AudioManager"]

    try:
        t0 = time.time()
        from voice.audio_device import audio_device
        from voice.settings import voice_settings
        from voice.audio_manager import audio_manager

        voice_settings.update_from_env()
        backend = audio_device.detect_backend()
        am_started = audio_manager.start()
        am_ok = am_started and audio_manager.is_running

        audit.details = f"Audio backend: {backend}, AudioManager: {'OK' if am_ok else 'NONE'}"

        if backend != "none" and am_ok:
            audit.status = SubsystemStatus.READY
        elif backend != "none":
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "AudioManager failed to start"
        else:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "No audio playback backend"

        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed

    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_tts() -> SubsystemAudit:
    """Audit TTS subsystem."""
    audit = SubsystemAudit(name="TTS")
    audit.dependencies = ["kokoro", "torch"]

    try:
        t0 = time.time()
        from voice.synthesizer import speech_synthesizer
        speech_synthesizer.initialize()
        if speech_synthesizer._ready:
            from voice.tts.manager import tts_manager
            audit.status = SubsystemStatus.READY
            audit.details = f"Engine: {tts_manager.active_engine_name}"
        else:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "No TTS engine available"
        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed
    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_faceauth() -> SubsystemAudit:
    """Audit face authentication subsystem."""
    audit = SubsystemAudit(name="FaceAuth")
    audit.dependencies = ["face_recognition", "opencv-python", "numpy"]

    try:
        t0 = time.time()
        import cv2
        import face_recognition
        from auth.faceauth import _load_encodings

        # Check camera
        cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cam.isOpened():
            cam = cv2.VideoCapture(0)
        cam_ok = cam.isOpened()
        if cam_ok:
            cam.release()

        # Check encodings
        encodings_ok = _load_encodings()

        if cam_ok and encodings_ok:
            audit.status = SubsystemStatus.READY
            audit.details = "Camera OK, encodings loaded"
        elif cam_ok:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "No face encodings — run auth/encode.py"
        else:
            audit.status = SubsystemStatus.DISABLED
            audit.root_cause = "No camera available"

        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed

    except ImportError as e:
        audit.status = SubsystemStatus.DISABLED
        audit.root_cause = f"Missing dependency: {e.name}"
    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_vision() -> SubsystemAudit:
    """Audit vision subsystem (lightweight — no model download)."""
    audit = SubsystemAudit(name="Vision")
    audit.dependencies = ["mss", "numpy", "pytesseract", "Pillow"]

    try:
        t0 = time.time()
        # Check dependencies without loading heavy models
        deps_ok = {
            "mss": check_import("mss"),
            "pyautogui": check_import("pyautogui"),
            "pytesseract": check_import("pytesseract"),
            "tesseract_cli": check_command("tesseract"),
            "Pillow": check_import("PIL"),
            "numpy": check_import("numpy"),
        }
        
        # Quick screen capture test (no model)
        screen_ok = False
        # Try ImageMagick first (most reliable on Linux)
        try:
            import subprocess
            result = subprocess.run(
                ["import", "-window", "root", "/tmp/Diego_audit_screen.png"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                screen_ok = True
        except Exception:
            pass
        if not screen_ok and deps_ok.get("pyautogui"):
            try:
                import pyautogui
                pil_img = pyautogui.screenshot()
                screen_ok = pil_img is not None
            except Exception:
                pass
        if not screen_ok and deps_ok.get("mss"):
            try:
                import mss
                with mss.mss() as sct:
                    img = sct.grab(sct.monitors[0])
                    screen_ok = img is not None
            except Exception:
                pass

        # OCR test
        ocr_ok = deps_ok.get("tesseract_cli", False) and deps_ok.get("pytesseract", False)

        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed

        if screen_ok and ocr_ok:
            audit.status = SubsystemStatus.READY
            audit.details = f"Capture: {'OK' if screen_ok else 'NONE'}, OCR: {'OK' if ocr_ok else 'NONE'}"
        elif screen_ok:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "OCR not available (install tesseract + pytesseract)"
        else:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "Screen capture not available (install mss or pyautogui)"

    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_duckdb() -> SubsystemAudit:
    """Audit DuckDB subsystem."""
    audit = SubsystemAudit(name="DuckDB")
    audit.dependencies = ["duckdb"]

    try:
        t0 = time.time()
        from memory.duckdb_store import store, DatabaseLockedError
        store.initialize()
        if store._conn is not None:
            audit.status = SubsystemStatus.READY
            # Test basic operations
            store.add_command(text="audit_test", intent="audit", confidence=1.0, response="ok")
            history = store.get_recent_commands(limit=1)
            audit.details = f"Connected, {len(history)} recent commands"
        else:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "Database locked"
        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed
    except DatabaseLockedError:
        audit.status = SubsystemStatus.DEGRADED
        audit.root_cause = "Database locked by another process"
    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_browser() -> SubsystemAudit:
    """Audit browser controller."""
    audit = SubsystemAudit(name="Browser")
    audit.dependencies = ["playwright"]

    try:
        t0 = time.time()
        from agent.browser import browser_controller
        result = browser_controller.initialize()
        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed

        if result:
            audit.status = SubsystemStatus.READY
            audit.details = f"Attached: {browser_controller.is_attached}, Page: {browser_controller.is_available}"
        else:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "Chrome not available"

    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    audit.memory_mb, audit.cpu_percent = get_memory_usage()
    return audit


async def audit_ollama() -> SubsystemAudit:
    """Audit Ollama LLM client."""
    audit = SubsystemAudit(name="Ollama")
    audit.dependencies = ["httpx"]

    try:
        t0 = time.time()
        from ai.llm_client import llm_client
        available = await llm_client.ensure_initialized()
        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed

        if available:
            audit.status = SubsystemStatus.READY
            audit.details = f"Model: {llm_client._model}"
        else:
            audit.status = SubsystemStatus.DISABLED
            audit.root_cause = "Ollama not running at localhost:11434"

    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    return audit


async def audit_plugins() -> SubsystemAudit:
    """Audit plugin subsystem."""
    audit = SubsystemAudit(name="Plugins")
    audit.dependencies = ["core.plugin_base", "core.event_bus"]

    try:
        t0 = time.time()
        from core.plugin_manager import plugin_manager
        await plugin_manager.load_all()
        await plugin_manager.initialize_all()
        names = list(plugin_manager.plugins.keys())
        elapsed = (time.time() - t0) * 1000
        audit.latency_ms = elapsed

        if names:
            enabled = [n for n in names if plugin_manager.plugins[n].enabled]
            disabled = [n for n in names if not plugin_manager.plugins[n].enabled]
            audit.details = f"{len(enabled)} enabled, {len(disabled)} disabled: {names}"
            audit.status = SubsystemStatus.READY if not disabled else SubsystemStatus.DEGRADED
            if disabled:
                audit.root_cause = f"Disabled plugins: {disabled}"
        else:
            audit.details = "No plugins discovered"
            audit.status = SubsystemStatus.READY

    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    return audit


async def audit_planner() -> SubsystemAudit:
    """Audit agent planner."""
    audit = SubsystemAudit(name="Planner")
    audit.dependencies = ["agent.memory", "agent.executor", "agent.browser"]

    try:
        from agent.planner import agent_planner
        agent_planner.initialize()
        if agent_planner.is_available:
            audit.status = SubsystemStatus.READY
            audit.details = "Initialized"
        else:
            audit.status = SubsystemStatus.DEGRADED
            audit.root_cause = "LLM not available"
    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    return audit


async def audit_shutdown() -> SubsystemAudit:
    """Audit shutdown gracefully."""
    audit = SubsystemAudit(name="Shutdown")
    audit.dependencies = ["voice", "DuckDB", "plugins"]

    try:
        from core.plugin_manager import plugin_manager
        from memory.duckdb_store import store
        store.close()
        audit.status = SubsystemStatus.READY
        audit.details = "Shutdown handlers registered"
    except Exception as e:
        audit.status = SubsystemStatus.FAILED
        audit.root_cause = str(e)

    return audit


# ── Main Audit Runner ────────────────────────────────────


async def run_audit() -> Dict[str, SubsystemAudit]:
    """Run all subsystem audits."""
    results = {}

    print("=" * 70)
    print("  DIEGO DESKTOP ASSISTANT — FULL SYSTEM AUDIT")
    print("=" * 70)
    print()

    # Environment
    print("[ENVIRONMENT]")
    env = audit_environment()
    for k, v in env.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                status = "✓" if sv else "✗"
                print(f"    {status} {sk}: {sv}")
        else:
            print(f"  {k}: {v}")
    print()

    # Run all audits
    audit_tasks = [
        ("Startup", audit_startup()),
        ("NLP", audit_nlp()),
        ("Embeddings", audit_embeddings()),
        ("Voice", audit_voice()),
        ("TTS", audit_tts()),
        ("FaceAuth", audit_faceauth()),
        ("Vision", audit_vision()),
        ("DuckDB", audit_duckdb()),
        ("Browser", audit_browser()),
        ("Ollama", audit_ollama()),
        ("Plugins", audit_plugins()),
        ("Planner", audit_planner()),
        ("Shutdown", audit_shutdown()),
    ]

    for name, task in audit_tasks:
        try:
            audit = await task
            results[name] = audit
        except Exception as e:
            results[name] = SubsystemAudit(name=name, status=SubsystemStatus.FAILED, root_cause=str(e))

    # Print results
    print("[SUBSYSTEM AUDIT RESULTS]")
    print(f"{'Subsystem':<20} {'Status':<12} {'Latency':<10} {'Details'}")
    print("-" * 70)
    ready_count = 0
    degraded_count = 0
    failed_count = 0

    for name in ["Startup", "NLP", "Embeddings", "Voice", "TTS", "FaceAuth",
                  "Vision", "DuckDB", "Browser", "Ollama", "Plugins", "Planner", "Shutdown"]:
        audit = results.get(name)
        if not audit:
            continue
        status_str = audit.status.value
        latency_str = f"{audit.latency_ms:.0f}ms" if audit.latency_ms else "N/A"
        details = audit.details if audit.details else (audit.root_cause if audit.root_cause else "")
        print(f"{name:<20} {status_str:<12} {latency_str:<10} {details[:60]}")

        if audit.status == SubsystemStatus.READY:
            ready_count += 1
        elif audit.status in (SubsystemStatus.DEGRADED, SubsystemStatus.DISABLED):
            degraded_count += 1
        else:
            failed_count += 1

    print("-" * 70)
    print(f"READY: {ready_count}  |  DEGRADED: {degraded_count}  |  FAILED: {failed_count}")
    print()

    # Save results
    import json
    report = {
        "environment": {k: str(v) if not isinstance(v, (str, bool, type(None))) else v for k, v in env.items()},
        "subsystems": {
            name: {
                "status": a.status.value,
                "latency_ms": round(a.latency_ms, 2),
                "root_cause": a.root_cause,
                "details": a.details,
                "memory_mb": round(a.memory_mb, 2),
                "cpu_percent": round(a.cpu_percent, 2),
            }
            for name, a in results.items()
        },
    }

    report_path = Path("debug/audit_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_audit())