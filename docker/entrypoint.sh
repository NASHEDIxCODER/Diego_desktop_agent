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