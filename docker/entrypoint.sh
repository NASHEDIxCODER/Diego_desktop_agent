#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Diego container entrypoint
#
# Runs as PID-1 child of tini (see Dockerfile ENTRYPOINT). It:
#   1. records the shell PID (→ becomes the exec'd Python PID) so the
#      container healthcheck can verify Diego process liveness,
#   2. seeds REQUIRED-IN-IMAGE static assets into freshly-initialised
#      volumes (never overwrites existing persistent state),
#   3. runs the first-run model bootstrap (downloads ONLY what is
#      missing; reuses any existing cache; fails clearly when a
#      REQUIRED asset cannot be obtained),
#   4. exec's the container CMD so SIGTERM reaches Python directly and
#      Diego's documented ≤15 s graceful shutdown runs.
#
# Environment:
#   DIEGO_SKIP_BOOTSTRAP=1   skip the model bootstrap entirely
#   OLLAMA_BASE_URL          override the Ollama endpoint (never
#                            container-local localhost by default)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

echo "[ENTRYPOINT] Diego container starting (pid $$)"
echo $$ > /tmp/diego.pid

# ── Ollama endpoint ────────────────────────────────────────────
# The Dockerfile sets a non-localhost default (host.docker.internal).
# Never silently fall back to container-local localhost.
if [ -z "${OLLAMA_BASE_URL:-}" ]; then
    export OLLAMA_BASE_URL="http://host.docker.internal:11434"
fi
echo "[ENTRYPOINT] OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"

# ── Seed REQUIRED-IN-IMAGE static assets into volumes ─────────
# A brand-new (anonymous or named) volume hides the image's bundled
# assets at the mount points, so re-seed missing files from the
# stash baked at /opt/diego-assets. Existing files are NEVER
# overwritten (persistent state wins).
seed() {
    src="$1"; dst="$2"
    if [ ! -e "$dst" ] && [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -r "$src" "$dst"
        echo "[ENTRYPOINT] Seeded $dst"
    fi
}

mkdir -p /app/data /app/models/wake/bundled /app/auth \
         /app/.cache/huggingface /app/.cache/ms-playwright

seed /opt/diego-assets/tessdata/eng.traineddata /app/data/tessdata/eng.traineddata
seed /opt/diego-assets/wake/verifier.pkl        /app/models/wake/verifier.pkl
seed /opt/diego-assets/wake/metadata.json       /app/models/wake/metadata.json
seed /opt/diego-assets/intent_classifier.pkl    /app/models/intent_classifier.pkl
seed /opt/diego-assets/metadata.json            /app/models/metadata.json
# Bundled openWakeWord ONNX models (REQUIRED-IN-IMAGE, baked at build)
shopt -s nullglob
for f in /opt/diego-assets/wake-bundled/*.onnx; do
    seed "$f" "/app/models/wake/bundled/$(basename "$f")"
done
shopt -u nullglob

# ── Audio: PulseAudio / PipeWire socket plumbing ───────────────
# The host PulseAudio/PipeWire server exposes a per-user Unix socket at
# /run/user/<UID>/pulse/native. When the operator mounts that socket
# into the container (the documented audio path), point the PulseAudio
# client at it via PULSE_SERVER so sounddevice/PortAudio open the host
# server instead of failing to find a device. Without this variable
# PortAudio sees NO input devices even though the socket is present.
#
# PORTABILITY: the entrypoint scans ALL /run/user/*/pulse/native paths,
# so the image works regardless of the container uid. The operator
# simply mounts -v /run/user/$UID/pulse:/run/user/$UID/pulse and the
# socket is found. No DIEGO_UID build arg required for audio.
if [ -z "${PULSE_SERVER:-}" ]; then
    # Auto-detect ANY mounted PulseAudio/PipeWire socket under /run/user.
    # This makes the image portable: the operator mounts the host socket
    # (-v /run/user/$UID/pulse:/run/user/$UID/pulse) and the entrypoint
    # finds it regardless of the container uid. No DIEGO_UID build arg
    # required for audio to work.
    _found=""
    for _sock in /run/user/*/pulse/native /run/user/*/pipewire-0; do
        if [ -S "$_sock" ]; then
            _found="$_sock"
            break
        fi
    done
    if [ -n "$_found" ]; then
        export PULSE_SERVER="unix:${_found}"
        echo "[ENTRYPOINT] Audio socket detected — PULSE_SERVER=${PULSE_SERVER}"
    else
        # Fall back to the current uid path for a clearer diagnostic.
        _uid="$(id -u)"
        echo "[ENTRYPOINT] WARNING: no PulseAudio/PipeWire socket found at " \
             "/run/user/*/ — audio input will be UNAVAILABLE unless " \
             "PULSE_SERVER is set and the socket is mounted"
    fi
else
    echo "[ENTRYPOINT] PULSE_SERVER=${PULSE_SERVER} (explicit)"
fi

# ── Display shim (headless containers) ─────────────────────────
# pyautogui/mouseinfo (imported via services/screen_capture even in
# --headless mode) require an X server at import time. If no host
# display is provided, start a virtual Xvfb display so the documented
# headless runtime works. With a mounted X11 socket + DISPLAY the
# shim is skipped entirely.
if [ "${DIEGO_XVFB:-1}" = "1" ]; then
    want_display="${DISPLAY:-:0}"
    sock="/tmp/.X11-unix/X${want_display#:}"
    if [ ! -S "$sock" ]; then
        if command -v Xvfb >/dev/null 2>&1; then
            Xvfb :99 -screen 0 1280x720x24 -nolisten tcp \
                >/tmp/xvfb.log 2>&1 &
            echo $! > /tmp/xvfb.pid
            export DISPLAY=:99
            echo "[ENTRYPOINT] No host display — started Xvfb on :99 " \
                 "(virtual 1280x720)"
        else
            echo "[ENTRYPOINT] WARNING: no display and no Xvfb — " \
                 "desktop automation imports will fail (DEGRADED)"
        fi
    fi
fi

# ── First-run model bootstrap ──────────────────────────────────
if [ "${DIEGO_SKIP_BOOTSTRAP:-0}" = "1" ]; then
    echo "[ENTRYPOINT] DIEGO_SKIP_BOOTSTRAP=1 — skipping model bootstrap"
else
    # REQUIRED assets that cannot be obtained → clear failure (exit 1).
    python /app/docker/bootstrap.py
fi

# ── Hand over to the container command (exec-form) ─────────────
# exec replaces this shell in-place: Python keeps PID $$ and tini
# forwards SIGTERM/SIGINT straight to it.
exec "$@"