#!/usr/bin/env python3
"""
Diego container model bootstrap (first-run asset pre-fetch).

Contract (docs/DOCKER_PREFLIGHT.md §5, DOCKER_RUNTIME_MATRIX.md §6):

  - Uses an EXISTING cache when present (HF cache, Playwright cache,
    bundled openWakeWord models) — never re-downloads on every start.
  - Downloads ONLY what is missing, into the persistent cache volumes
    (HF_HOME=/app/.cache/huggingface, /app/.cache/ms-playwright,
    /app/models/wake/bundled).
  - FAILS CLEARLY (exit 1) when a REQUIRED asset cannot be obtained.
  - Best-effort assets (Kokoro TTS, browser engine) only warn: Diego's
    documented fallback chains (pyttsx3, non-browser actions) apply and
    the runtime health reports them as DEGRADED — never container failure.

Classification:
  REQUIRED (fail startup) : faster-whisper STT, all-MiniLM-L6-v2 embeddings
  BEST-EFFORT (warn)      : Kokoro-82M TTS, openWakeWord bundled ONNX,
                            Playwright Chromium
  NOTHING TO DOWNLOAD     : Silero VAD (model ships inside the
                            silero-vad pip package)

Run manually:  python /app/docker/bootstrap.py
Skip at start: DIEGO_SKIP_BOOTSTRAP=1
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BOOTSTRAP_START = __import__("time").time()

HF_HOME = Path(os.environ.get("HF_HOME", "/app/.cache/huggingface"))
BROWSERS_DIR = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH",
                                   "/app/.cache/ms-playwright"))
WAKE_BUNDLED_DIR = Path("/app/models/wake/bundled")

OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or \
          os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


def log(msg: str) -> None:
    print(f"[BOOTSTRAP] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[BOOTSTRAP] REQUIRED ASSET FAILED: {msg}", file=sys.stderr,
          flush=True)
    print("[BOOTSTRAP] Container startup aborted. Provide the asset via a "
          "mounted cache volume or network access, or run with "
          "DIEGO_SKIP_BOOTSTRAP=1 to bypass (Diego will then degrade "
          "instead of failing).", file=sys.stderr, flush=True)
    sys.exit(1)


def hf_repo_ready(repo_id: str) -> bool:
    """True if the repo is already present in the local HF cache."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def ensure_hf_model(repo_id: str, label: str, required: bool) -> None:
    """Ensure an HF model is in the cache: reuse → download → fail/warn."""
    if hf_repo_ready(repo_id):
        log(f"{label}: cache hit ({repo_id}) — no download")
        return

    if OFFLINE:
        msg = (f"{label}: not in HF cache ({repo_id}) and offline mode "
               f"is active (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1)")
        if required:
            fail(msg)
        log(f"{label}: {msg} — continuing (degraded)")
        return

    log(f"{label}: not cached — downloading {repo_id} …")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id)
        log(f"{label}: download complete ({repo_id})")
    except Exception as e:
        msg = f"{label}: download failed ({repo_id}): {type(e).__name__}: {e}"
        if required:
            fail(msg)
        log(f"{label}: {msg} — continuing (degraded)")


def ensure_silero_vad() -> None:
    """Silero VAD ships inside the silero-vad pip package — verify only."""
    try:
        import silero_vad  # noqa: F401
        log("Silero VAD: bundled with the silero-vad package — nothing "
            "to download")
    except Exception as e:
        log(f"Silero VAD: package verification failed ({e}) — the runtime "
            f"energy-based VAD fallback applies (DEGRADED)")


def ensure_openwakeword_models() -> None:
    """Bundled openWakeWord ONNX models → persistent /app/models/wake/bundled.

    The canonical resolver (voice/wake_resolver.py) searches this dir.
    Existing ONNX files are never re-downloaded.
    """
    try:
        from openwakeword.utils import download_models
    except Exception as e:
        log(f"openWakeWord: package unavailable ({e}) — wake falls back "
            f"to always-LISTEN (DEGRADED)")
        return

    if any(WAKE_BUNDLED_DIR.glob("*.onnx")):
        log(f"openWakeWord: models already present in {WAKE_BUNDLED_DIR} "
            f"— no download")
        return

    def _copy_from_package_resources() -> bool:
        """Copy any ONNX models shipped inside the openwakeword package."""
        import openwakeword
        import shutil
        pkg_dir = Path(openwakeword.__file__).parent
        copied = False
        for src in list((pkg_dir / "resources" / "models").glob("*.onnx")) + \
                   list((pkg_dir / "models").glob("*.onnx")):
            dst = WAKE_BUNDLED_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied = True
        return copied

    try:
        WAKE_BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
        download_models(target_dir=str(WAKE_BUNDLED_DIR))
        log(f"openWakeWord: bundled models downloaded to {WAKE_BUNDLED_DIR}")
    except TypeError:
        # Older download_models() without target_dir → default location
        try:
            download_models()
            log("openWakeWord: bundled models downloaded to the package "
                "resources directory")
        except Exception as e:
            if _copy_from_package_resources():
                log(f"openWakeWord: models recovered from package "
                    f"resources into {WAKE_BUNDLED_DIR}")
            else:
                log(f"openWakeWord: model download failed ({e}) — wake "
                    f"falls back to always-LISTEN (DEGRADED)")
    except Exception as e:
        try:
            if _copy_from_package_resources():
                log(f"openWakeWord: models recovered from package "
                    f"resources into {WAKE_BUNDLED_DIR} (download failed: "
                    f"{e})")
                return
        except Exception:
            pass
        log(f"openWakeWord: model download failed ({e}) — wake falls back "
            f"to always-LISTEN (DEGRADED)")


def ensure_playwright_chromium() -> None:
    """Playwright Chromium browser → /app/.cache/ms-playwright (best effort)."""
    if any(BROWSERS_DIR.glob("chromium-*")):
        log(f"Playwright Chromium: already present in {BROWSERS_DIR} — "
            f"no download")
        return
    log("Playwright Chromium: not cached — downloading …")
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install",
                        "chromium"], check=True)
        log("Playwright Chromium: download complete")
    except Exception as e:
        log(f"Playwright Chromium: download failed ({e}) — browser "
            f"automation unavailable (DEGRADED)")


def main() -> None:
    log(f"Starting model bootstrap (offline={OFFLINE}, "
        f"HF_HOME={HF_HOME})")

    # ── Verify cache volume is writable ────────────────────────
    for d in (HF_HOME, BROWSERS_DIR, WAKE_BUNDLED_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            fail(f"persistent cache dir not writable: {d}: {e}")

    # ── Silero VAD (bundled — verify only) ─────────────────────
    ensure_silero_vad()

    # ── REQUIRED: faster-whisper STT model (default: base) ─────
    whisper_model = os.environ.get("WHISPER_MODEL", "base").strip() or "base"
    ensure_hf_model(f"Systran/faster-whisper-{whisper_model}",
                    f"faster-whisper '{whisper_model}' (STT)",
                    required=True)

    # ── REQUIRED: sentence-transformers embeddings model ───────
    ensure_hf_model("sentence-transformers/all-MiniLM-L6-v2",
                    "all-MiniLM-L6-v2 (embeddings)", required=True)

    # ── BEST-EFFORT: Kokoro TTS ────────────────────────────────
    ensure_hf_model("hexgrad/kokoro-82M",
                    "kokoro-82M (TTS)", required=False)

    # ── BEST-EFFORT: openWakeWord bundled models ───────────────
    ensure_openwakeword_models()

    # ── BEST-EFFORT: Playwright Chromium ───────────────────────
    ensure_playwright_chromium()

    log(f"Model bootstrap complete in {__import__('time').time() - BOOTSTRAP_START:.1f}s")


if __name__ == "__main__":
    main()