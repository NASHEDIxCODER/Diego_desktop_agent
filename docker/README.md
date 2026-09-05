# Diego Production Container

Image build contract: `docs/DOCKER_PREFLIGHT.md` + `docs/DOCKER_RUNTIME_MATRIX.md`.

## Files

| File | Role |
|---|---|
| `../Dockerfile` | Multi-stage build: Python 3.11-slim, lockfile wheels (CPU torch), runtime libs, static assets. |
| `../.dockerignore` | Excludes `.venv`, `.git`, secrets, runtime state, HF caches, dev-only trees. |
| `entrypoint.sh` | tini child; seeds REQUIRED-IN-IMAGE assets into fresh volumes, runs model bootstrap, `exec`s the CMD (SIGTERM reaches Python). |
| `bootstrap.py` | First-run model pre-fetch into persistent volumes. Reuses existing cache; downloads only what is missing; REQUIRED assets fail clearly. |
| `healthcheck.py` | Container probe on top of `core/runtime_health`. Distinguishes READY / DEGRADED / UNAVAILABLE (healthy) vs FAILED (unhealthy). |

## Build

```bash
docker build -t diego:latest .        # or: podman build -t diego:latest .
```

## Run (full host integration)

```bash
docker run -d --name diego \
  --init \
  --add-host=host.docker.internal:host-gateway \
  --device /dev/video0 \                       # camera (face auth only)
  -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
  -v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \   # PulseAudio/PipeWire socket
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 \ # X11 (optional)
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v diego-data:/app/data \
  -v diego-models:/app/models \
  -v diego-cache:/app/.cache \
  --shm-size=1g \
  diego:latest
```

> **Audio is PulseAudio/PipeWire socket-based, NOT `/dev/snd`.** The
> entrypoint auto-detects the socket at any `/run/user/*/pulse/native`
> path, so the image is portable — no `DIEGO_UID` build arg needed.
> The operator mounts `-v /run/user/$UID/pulse:/run/user/$UID/pulse`
> and `PULSE_SERVER` is set automatically. The explicit `-e` is
> optional but shown for clarity.

Primary command: `python main.py --headless` (image default).
Documented alternative: `docker run ... diego:latest python Diego.py`.

Optional GPU: replace torch wheels with `+cu*` builds and add `--gpus all`
(host driver + Container Toolkit). Ollama GPU acceleration is HOST-side.

## Volumes (persistent runtime state)

| Container path | Content |
|---|---|
| `/app/data` | `Diego.duckdb`, `knowledge_base.json`, `embedding_cache.pkl`, device JSONs, TTS cache, seeded `tessdata/` |
| `/app/models` | wake verifier + metadata, bundled wake ONNX, intent classifier |
| `/app/auth` | face encodings — mount as a secret/volume, never baked |
| `/app/.cache` | HF models (`huggingface/`), Playwright browsers (`ms-playwright/`) |

## Bootstrap behavior

- REQUIRED: faster-whisper (`WHISPER_MODEL`, default `base`),
  `all-MiniLM-L6-v2` — download failure aborts container start with a
  clear error.
- BEST-EFFORT: `kokoro-82M`, openWakeWord bundled ONNX
  (→ `/app/models/wake/bundled`), Playwright Chromium — failure degrades.
- Silero VAD ships inside the `silero-vad` package (nothing to download).
- `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` → cache-only; missing
  REQUIRED assets fail clearly.
- `DIEGO_SKIP_BOOTSTRAP=1` skips bootstrap entirely.

## Health

`HEALTHCHECK` (30 s interval, 300 s start-period) reports:

- `READY` — all REQUIRED components ready
- `DEGRADED` — operating with documented fallbacks (healthy)
- `UNAVAILABLE` — optional external service down, e.g. Ollama (healthy)
- `FAILED` — Diego process dead or REQUIRED component down (unhealthy)

## Shutdown

SIGTERM is forwarded by tini → `exec python` → Diego's signal handler
(`Diego._main_async`) → graceful shutdown within the documented hard
15 s timeout.