# ═══════════════════════════════════════════════════════════════
# Diego Desktop Agent — production image
#
# Contract: docs/DOCKER_PREFLIGHT.md + docs/DOCKER_RUNTIME_MATRIX.md
#   - Python 3.11 (the only fully-resolvable lockfile target)
#   - CPU-only torch (lockfile pin: torch==2.13.0+cpu)
#   - REQUIRED-IN-IMAGE system libs + pip packages
#   - Static assets: tessdata, wake verifier+metadata, wake chime,
#     face-detection ONNX, intent classifier
#   - NOT baked: DuckDB state, embedding cache, device JSONs, face
#     encodings, HF caches, secrets, dev-only trees (see .dockerignore)
#   - First-run models (Whisper/MiniLM/Kokoro/OWW/Chromium) are
#     bootstrapped at container start into persistent volumes.
# ═══════════════════════════════════════════════════════════════

# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11

# Runtime user UID. MUST match the host uid that owns the PulseAudio
# socket (/run/user/<UID>/pulse/native) so the container process is
# allowed to connect. Override at build time for non-default hosts:
#   docker build --build-arg DIEGO_UID=$(id -u) -t diego:latest .
ARG DIEGO_UID=1000

# ─────────────────────────────────────────────────────────────
# Stage 1 — builder: compile native wheels (dlib etc.) into a venv
# ─────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

# Build-time system deps (docs §2):
#   build-essential/cmake/libopenblas-dev → dlib (face-recognition)
#   portaudio19-dev                        → sounddevice backend
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libopenblas-dev \
        portaudio19-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Runtime lockfile (exact pins). pytest* lines are DEV-ONLY and are
# stripped before install (docs/DOCKER_RUNTIME_MATRIX.md §3.3).
COPY requirements.runtime-lock.txt /tmp/lock-raw.txt
RUN grep -Ev '^pytest' /tmp/lock-raw.txt > /tmp/lock.txt

# torch/torchaudio are +cpu builds → need the PyTorch CPU index.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/lock.txt

# REQUIRED-IN-IMAGE packages missing from the lockfile (the lock
# resolves the core runtime; these are on the documented production
# import graph — docs/DOCKER_RUNTIME_MATRIX.md §3.1):
#   PySide6-Essentials (Qt for ui/), face-recognition (+dlib),
#   desktop automation (PyAutoGUI/pyperclip), psutil,
#   SpeechRecognition, sentencepiece.
RUN pip install \
        "PySide6-Essentials>=6.5.0" \
        "face-recognition>=1.3.0" \
        "PyAutoGUI>=0.9.54" \
        "pyperclip>=1.8.2" \
        "psutil>=5.9.0" \
        "SpeechRecognition>=3.10.0" \
        "sentencepiece>=0.2.0"

# ─────────────────────────────────────────────────────────────
# Stage 2 — runtime: slim image + shared libs + app + assets
# ─────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim

# Re-declare the global ARG so it is available inside this stage.
# Default 1000; override with --build-arg DIEGO_UID=$(id -u).
ARG DIEGO_UID=1000

# Persistent model caches (volume-backed — never baked in layers),
# bundled Tesseract data (seeded into /app/data by the entrypoint).
# Ollama: NEVER container-local localhost. Docker users override
# with -e OLLAMA_BASE_URL=... (host.docker.internal / host IP /
# --network host). Override freely; this is only a sane default.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface \
    PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TESSDATA_PREFIX=/app/data/tessdata \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

# Runtime system deps (docs/DOCKER_PREFLIGHT.md §2):
#   REQUIRED: libportaudio2/libasound2/libpulse0 (audio),
#             tesseract-ocr (OCR), libgl1/libglib2.0-0 (OpenCV),
#             libxcb*/libxkbcommon0/libx11-6/libdbus-1-3 (Qt runtime),
#             libopenblas0 (dlib/numpy), tini (PID1 signal forwarding)
#   OPTIONAL (fallback chain): espeak-ng (pyttsx3 TTS),
#             fonts-dejavu-core (Qt/OCR rendering)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tini \
        libopenblas0 \
        libportaudio2 \
        libasound2 \
        libpulse0 \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxkbcommon0 \
        libxcb1 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xfixes0 \
        libxcb-cursor0 \
        libdbus-1-3 \
        espeak-ng \
        fonts-dejavu-core \
        # Xvfb: virtual display shim — pyautogui/mouseinfo (imported by
        # services/screen_capture) require an X server at import time even
        # in --headless mode; the entrypoint starts Xvfb when no host
        # display is provided. DEV-ONLY in docs, but REQUIRED as the
        # container-level headless fallback mechanism.
        xvfb \
        # tcl/tk runtime libs: the official python image builds _tkinter
        # but slim lacks the shared libs — pyautogui→mouseinfo imports
        # tkinter at module level (desktop automation path).
        tcl8.6 \
        tk8.6 \
 && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. uid MUST match the host uid that owns the
# PulseAudio socket (/run/user/<UID>/pulse/native) — PulseAudio
# rejects connections from mismatched uids. audio/video groups for
# direct ALSA device node access (fallback when no PA server).
RUN groupadd -g "${DIEGO_UID}" diego \
 && useradd -m -u "${DIEGO_UID}" -g "${DIEGO_UID}" -G audio,video diego

# Python venv from the builder (pre-compiled wheels, no build tools)
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# ── Application code (context filtered by .dockerignore: no .venv,
#    .git, secrets, runtime state, caches, dev-only trees) ──
COPY --chown=diego:diego . /app

# ── Stash REQUIRED-IN-IMAGE static assets outside the volume mount
#    points. The entrypoint seeds them into freshly-initialised
#    volumes (never overwriting existing state).
#      data/tessdata/eng.traineddata        Tesseract English data
#      models/wake/verifier.pkl + metadata  trained wake verifier
#      models/intent_classifier.pkl (+meta) trained NLP classifier
COPY data/tessdata/eng.traineddata /opt/diego-assets/tessdata/eng.traineddata
COPY models/wake/verifier.pkl /opt/diego-assets/wake/verifier.pkl
COPY models/wake/metadata.json /opt/diego-assets/wake/metadata.json
COPY models/intent_classifier.pkl /opt/diego-assets/intent_classifier.pkl
COPY models/metadata.json /opt/diego-assets/metadata.json

# ── openWakeWord pretrained ONNX models — REQUIRED-IN-IMAGE
#    (docs/DOCKER_RUNTIME_MATRIX.md §4.1). The pip package does NOT
#    bundle them → download at build time into the asset stash (as
#    root; the runtime user cannot write site-packages).
RUN mkdir -p /opt/diego-assets/wake-bundled \
 && (python -c "from openwakeword.utils import download_models; \
                download_models(target_dir='/opt/diego-assets/wake-bundled')" \
      || python -c "from openwakeword.utils import download_models; \
                    download_models()") \
 && cp -n /opt/venv/lib/python3.11/site-packages/openwakeword/resources/models/*.onnx \
          /opt/diego-assets/wake-bundled/ 2>/dev/null || true \
 && ls -la /opt/diego-assets/wake-bundled/

# ── Persistent runtime state directories (VOLUME-backed) ──
#   /app/data       Diego.duckdb, knowledge_base.json, embedding_cache,
#                   device JSONs, TTS cache, tessdata (seeded)
#   /app/models     wake verifier, bundled wake ONNX, intent classifier
#   /app/auth       face encodings (mount as secret/volume — NOT baked)
#   /app/.cache     HF models + Playwright browsers (first-run downloads)
RUN mkdir -p /app/data /app/models/wake/bundled /app/auth \
             /app/.cache/huggingface /app/.cache/ms-playwright \
 && chown -R diego:diego /app /opt/diego-assets \
 && chmod +x /app/docker/entrypoint.sh \
 # Xlib/python-xlib reads ~/.Xauthority on connect even when the X
 # server uses no auth (Xvfb) — an empty file prevents XauthError.
 && touch /home/diego/.Xauthority \
 && chown diego:diego /home/diego/.Xauthority

VOLUME ["/app/data", "/app/models", "/app/auth", "/app/.cache"]

USER diego

# Health: distinguishes READY / DEGRADED / FAILED / UNAVAILABLE using
# Diego's own core.runtime_health component model + process liveness.
# Optional failures (Ollama unreachable, wake model missing, camera…)
# never mark the container unhealthy.
HEALTHCHECK --interval=30s --timeout=25s --start-period=300s --retries=3 \
    CMD ["python", "/app/docker/healthcheck.py"]

# tini = PID 1 → forwards SIGTERM/SIGINT to the Python process so
# Diego's documented ≤15 s graceful shutdown always runs. NOTE: no
# -g (process-group signalling) — the Xvfb display shim must survive
# the shutdown drain path; killing it concurrently hangs an X-linked
# task and prevents container exit.
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]

# Primary production command (documented container recommendation).
# Alternative documented entrypoint stays available:
#   docker/podman run <image> python Diego.py
CMD ["python", "main.py", "--headless"]