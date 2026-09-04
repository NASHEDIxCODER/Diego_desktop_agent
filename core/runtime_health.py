"""
RuntimeHealth — Production dependency/asset diagnostic for Diego.

Prints a structured [HEALTH] report at startup so every component's
status is visible in one place:

    [HEALTH] status=READY      name=microphone  package=sounddevice/0.5.5  ...
    [HEALTH] status=DEGRADED   name=vad         package=silero_vad/MISSING fallback=energy-based VAD
    [HEALTH] status=BYPASSED   name=wake        reason=--no-wake active

Status model (exactly one of):
    READY      — component loaded and fully operational
    DEGRADED   — component unavailable but a functional fallback is active
    MISSING    — component not installed (feature absent)
    FAILED     — component present but failed to initialise/load
    BYPASSED   — intentionally disabled (e.g. wake via --no-wake)
    UNAVAILABLE— external service unreachable (e.g. Ollama down); Diego
                 keeps running with deterministic/local capabilities

An earlier revision printed component=OK together with a fallback field
like "energy-based VAD (degraded)" — a READY line NEVER shows a fallback
now: the fallback text only appears on DEGRADED/FAILED/MISSING lines.

Classification:
  REQUIRED  — normal voice operation depends on this (mic + VAD + STT + TTS)
  OPTIONAL  — degraded feature (screen/OCR/search/wake)
  DEVONLY   — development/benchmark only (sherpa-onnx, etc.)

The diagnostic NEVER retries a permanently-missing component. It reports
the status once and lets the caller decide. This prevents the retry-loop
behaviour that occurs when a required component is permanently absent.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Status values ──────────────────────────────────────────────────
READY = "READY"
OK = READY            # backward-compatible alias
DEGRADED = "DEGRADED"
MISSING = "MISSING"
FAILED = "FAILED"
BYPASSED = "BYPASSED"
UNAVAILABLE = "UNAVAILABLE"

# ── Component classes ──────────────────────────────────────────────
REQUIRED = "REQUIRED"      # normal voice operation
OPTIONAL = "OPTIONAL"      # degraded feature
DEVONLY = "DEVONLY"        # development/benchmark only


@dataclass
class ComponentHealth:
    """Health record for a single runtime component."""
    name: str
    status: str = MISSING
    package: str = ""
    version: str = ""
    asset: str = ""
    reason: str = ""
    fallback: str = ""
    component_class: str = OPTIONAL

    def to_line(self) -> str:
        # READY/BYPASSED lines NEVER advertise a fallback: showing
        # "fallback=energy-based VAD (degraded)" on an OK line was
        # misleading (a component cannot be fully OK and degraded).
        fallback_shown = "" if self.status in (READY, BYPASSED) \
            else f" fallback={self.fallback or 'none'}"
        return (
            f"[HEALTH] status={self.status:<11} "
            f"name={self.name} "
            f"package={self.package or 'none'}"
            f"{'/' + self.version if self.version else ''} "
            f"asset={self.asset or 'none'} "
            f"class={self.component_class} "
            f"reason={self.reason or 'ready'}"
            f"{fallback_shown}"
        )


def _import_version(module_name: str) -> Optional[str]:
    """Import a module and return its __version__ (or '?' if unknown)."""
    try:
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "?"))
    except Exception:
        return None


def _check_import(module_name: str) -> bool:
    """True if the module imports successfully."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _check_asset(path: str) -> bool:
    """True if the file/dir exists."""
    return Path(path).exists()


class RuntimeHealth:
    """Runs the full dependency/asset diagnostic and prints [HEALTH] lines."""

    def __init__(self):
        self._components: List[ComponentHealth] = []
        self._base_dir = Path(__file__).resolve().parent.parent

    # ── Component registration helpers ────────────────────────────

    def _add(self, name: str, status: str, package: str, version: str,
             asset: str, reason: str, fallback: str,
             component_class: str) -> None:
        self._components.append(ComponentHealth(
            name=name, status=status, package=package, version=version,
            asset=asset, reason=reason, fallback=fallback,
            component_class=component_class,
        ))

    def _check_pkg(self, name: str, module: str, component_class: str,
                   fallback: str = "", asset: str = "") -> None:
        """Check a Python package import."""
        ver = _import_version(module)
        if ver is not None:
            self._add(name, OK, module, ver, asset, "ready", fallback, component_class)
        else:
            self._add(name, MISSING, module, "", asset,
                      "package not installed", fallback, component_class)

    # ── Full diagnostic ───────────────────────────────────────────

    def run(self, no_wake: bool = False) -> List[ComponentHealth]:
        """Run every check and return the component list.

        `no_wake` marks the wake detector BYPASSED (--no-wake / dev mode)
        instead of READY — the component is intentionally skipped, not
        operational.
        """
        self._components = []
        base = self._base_dir

        # ══ REQUIRED — normal voice operation ══════════════════════
        # 1. Microphone capture
        self._check_pkg("microphone", "sounddevice", REQUIRED,
                        fallback="none")
        # 2. VAD — Silero missing means the energy fallback is ACTIVE,
        #    so the honest status is DEGRADED, not MISSING.
        silero_ver = _import_version("silero_vad")
        if silero_ver is not None:
            self._add("vad", READY, "silero_vad", silero_ver, "",
                      "ready", "", REQUIRED)
        else:
            self._add("vad", DEGRADED, "silero_vad", "", "",
                      "silero unavailable — energy-based VAD active",
                      "energy-based VAD", REQUIRED)
        # 3. Command STT — Whisper unavailable means command recognition
        #    is DISABLED: FAILED (required capability lost, no equivalent).
        whisper_ver = _import_version("faster_whisper")
        if whisper_ver is not None:
            self._add("stt", READY, "faster_whisper", whisper_ver, "",
                      "ready", "", REQUIRED)
        else:
            self._add("stt", FAILED, "faster_whisper", "", "",
                      "package not installed — command STT disabled",
                      "command STT disabled", REQUIRED)
        # 4. TTS
        tts_ver = _import_version("kokoro")
        if tts_ver is not None:
            self._add("tts", READY, "kokoro", tts_ver, "",
                      "ready", "", REQUIRED)
        else:
            # Kokoro requires Python <3.13. Check pyttsx3 fallback.
            pyttsx3_ver = _import_version("pyttsx3")
            if pyttsx3_ver is not None:
                self._add("tts", DEGRADED, "kokoro", "",
                          "", "kokoro requires Python<3.13 (runtime is "
                          f"{sys.version_info.major}.{sys.version_info.minor})",
                          "pyttsx3 (espeak)", REQUIRED)
            else:
                self._add("tts", FAILED, "kokoro", "", "",
                          "no TTS engine available", "none", REQUIRED)

        # ══ OPTIONAL — degraded features ═══════════════════════════
        # 5. Wake model
        wake_asset = base / "models" / "wake" / "verifier.pkl"
        if no_wake:
            # Intentionally disabled — BYPASSED, never READY.
            self._add("wake", BYPASSED, "openwakeword",
                      _import_version("openwakeword") or "",
                      str(wake_asset) if wake_asset.exists() else "",
                      "--no-wake active (wake detection skipped)", "", OPTIONAL)
        elif _check_import("openwakeword"):
            self._add("wake", READY, "openwakeword",
                      _import_version("openwakeword") or "?",
                      str(wake_asset) if wake_asset.exists() else "",
                      "ready", "", OPTIONAL)
        else:
            self._add("wake", MISSING, "openwakeword", "",
                      str(wake_asset) if wake_asset.exists() else "",
                      "package not installed", "run with --no-wake", OPTIONAL)

        # 6. Screen capture
        self._check_pkg("screen_capture", "mss", OPTIONAL,
                        fallback="pyautogui")

        # 7. OCR
        ocr_backends = []
        for mod in ("paddleocr", "easyocr", "pytesseract"):
            if _check_import(mod):
                ocr_backends.append(mod)
        if ocr_backends:
            self._add("ocr", OK, ",".join(ocr_backends), "",
                      str(base / "data" / "tessdata" / "eng.traineddata")
                      if (base / "data" / "tessdata" / "eng.traineddata").exists() else "",
                      "ready", "none", OPTIONAL)
        else:
            self._add("ocr", MISSING, "paddleocr/easyocr/pytesseract", "",
                      "", "no OCR backend installed", "none", OPTIONAL)

        # 8. Local LLM (Ollama) — fail-safe: an unreachable Ollama NEVER
        # blocks startup. Health simply reports UNAVAILABLE and Diego
        # keeps running with deterministic/local capabilities.
        try:
            import httpx
            r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                self._add("llm", READY, "ollama", "",
                          f"http://localhost:11434 ({len(models)} models)",
                          "ready", "", OPTIONAL)
            else:
                self._add("llm", UNAVAILABLE, "ollama", "",
                          "http://localhost:11434",
                          f"HTTP {r.status_code}", "", OPTIONAL)
        except Exception as e:
            self._add("llm", UNAVAILABLE, "ollama", "",
                      "http://localhost:11434",
                      f"unreachable: {type(e).__name__}", "", OPTIONAL)

        # 9. DuckDB persistence
        self._check_pkg("duckdb", "duckdb", OPTIONAL,
                        fallback="in-memory fallback",
                        asset=str(base / "data" / "Diego.duckdb"))

        # 10. Browser automation
        self._check_pkg("browser", "playwright", OPTIONAL,
                        fallback="selenium")

        # 11. Search
        search_ok = _check_import("trafilatura") or _check_import("bs4")
        if search_ok:
            self._add("search", OK, "trafilatura/bs4", "",
                      "", "ready", "none", OPTIONAL)
        else:
            self._add("search", MISSING, "trafilatura/bs4", "",
                      "", "no extraction backend", "none", OPTIONAL)

        # 12. Vision (OpenCV)
        self._check_pkg("vision", "cv2", OPTIONAL, fallback="none")

        # ══ DEVONLY — development/benchmark only ═══════════════════
        self._check_pkg("sherpa_onnx", "sherpa_onnx", DEVONLY,
                        fallback="faster-whisper (production STT)")
        self._check_pkg("nemo_toolkit", "nemo", DEVONLY,
                        fallback="faster-whisper (production STT)")

        return self._components

    # ── Report ────────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print the [HEALTH] report to stdout."""
        print()
        print("  Diego Runtime Health")
        print("  " + "=" * 60)
        for c in self._components:
            print("  " + c.to_line())
        print()

        required = [c for c in self._components if c.component_class == REQUIRED]
        required_ok = [c for c in required if c.status == READY]
        required_degraded = [c for c in required if c.status == DEGRADED]
        required_down = [c for c in required if c.status in (MISSING, FAILED)]

        print(f"  REQUIRED voice components: {len(required_ok)} READY, "
              f"{len(required_degraded)} DEGRADED, {len(required_down)} down")
        if required_down:
            print("  ✗ Normal voice operation BLOCKED — down: "
                  + ", ".join(f"{c.name}({c.status})" for c in required_down))
        elif required_degraded:
            print("  ✓ Voice operation available (degraded: "
                  + ", ".join(c.name for c in required_degraded) + ")")
        else:
            print("  ✓ Normal voice operation READY")
        print()

    @property
    def voice_ready(self) -> bool:
        """True if all REQUIRED components are READY or DEGRADED (operable
        with fallback). MISSING/FAILED required components block voice."""
        required = [c for c in self._components if c.component_class == REQUIRED]
        return all(c.status not in (MISSING, FAILED) for c in required)


# Global singleton
runtime_health = RuntimeHealth()


def run_runtime_health(no_wake: bool = False) -> List[ComponentHealth]:
    """Run the diagnostic and print the report. Returns components.

    `no_wake=True` reports the wake detector as BYPASSED instead of READY.
    """
    components = runtime_health.run(no_wake=no_wake)
    runtime_health.print_report()
    return components
