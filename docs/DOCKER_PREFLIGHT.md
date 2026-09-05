# Diego Docker Preflight

The exact runtime contract for a fresh Docker/container environment,
derived from `docs/DIEGO_ARCHITECTURE.md`, `docs/DOCKER_RUNTIME_MATRIX.md`,
`requirements.txt`, `requirements.runtime-lock.txt`, `main.py`, and the
current startup/model loaders. **Preflight document only — no image is
built, nothing is installed, no runtime behavior changes.**

---

## 1. Base OS / Python Version

| Item | Contract |
|---|---|
| Base image | Debian/Ubuntu-based Linux (glibc). No distro-specific code exists. |
| Python | **3.11** (pinned by `requirements.runtime-lock.txt`; the only fully-resolvable, fully-tested combination). |
| 3.14 note | `compat.py` allows booting on 3.14, but Coqui `TTS` (<3.12), `paddlepaddle` (<3.13), and `kokoro/misaki` 0.9.x are NOT installable there — 3.14 is a degraded runtime, not a supported target. Do not use 3.14 for the image. |
| venv | Install `requirements.txt` (bounds) or `requirements.runtime-lock.txt` (exact pins) into an image-local venv. |

---

## 2. Required apt / System Packages

| Package | Why | Classification |
|---|---|---|
| `portaudio19-dev` (+ `libasound2`, `libpulse0`) | `sounddevice` mic/speaker backend | REQUIRED (build + run) |
| `tesseract-ocr` | OCR backend for `pytesseract` | REQUIRED |
| `libgl1`, `libglib2.0-0` | OpenCV runtime | REQUIRED |
| `libxcb*`, `libxkbcommon0`, `libx11-6` | PySide6/Qt runtime libs | REQUIRED (DISPLAY still host-provided) |
| `espeak-ng` | pyttsx3 TTS fallback + wake calibration synthesis | OPTIONAL (fallback chain) |
| `libgtk-3-0`, `libatspi2.0-0`, `gir1.2-atspi-2.0` | AT-SPI a11y tree (PyGObject) | OPTIONAL |
| `ffmpeg` | audio decode for some TTS engines | OPTIONAL |
| `fonts-dejavu-core` | Qt/OCR text rendering | OPTIONAL |
| `xvfb` | headless virtual display (CI only) | DEV-ONLY |
| dlib build deps (`build-essential`, `cmake`, `libopenblas-dev`) | `face-recognition` wheel/build | REQUIRED (build) |

---

## 3. Required pip Packages

Install from `requirements.txt` (bounds) or `requirements.runtime-lock.txt`
(exact pins). Highlights of the REQUIRED set (full matrix in
`docs/DOCKER_RUNTIME_MATRIX.md` §3):

- Core: `numpy`, `sounddevice`, `faster-whisper`, `openwakeword`,
  `silero-vad`, `torch`(+`torchaudio`, `+cpu`), `onnxruntime`,
  `sentence-transformers`, `transformers`, `huggingface-hub`, `httpx`,
  `aiohttp`, `duckdb`, `pydantic-settings`, `python-dotenv`, `PySide6`,
  `opencv-python(-headless)`, `face-recognition`, `pillow`, `mss`,
  `rapidfuzz`, `jellyfish`, `scipy`, `scikit-learn`, `spacy` +
  `en_core_web_sm` (URL wheel in lockfile), `PyGObject`, `pytesseract`,
  `kokoro>=0.9.4`, `misaki[en]>=0.9.4`, `playwright`, `pyautogui`,
  `pyperclip`.
- Knowledge extraction: `PyPDF2`, `python-docx`, `python-pptx`, `openpyxl`.
- Post-install: `python -m playwright install chromium`.
- OPTIONAL (skip for minimal image): `sherpa-onnx`, `llama-cpp-python`,
  `nemo-toolkit[asr]`, `TTS` (<3.12), `piper-tts`, `easyocr`,
  `paddleocr`+`paddlepaddle`, `firebase-admin`, `audioop-lts` (≥3.13 only).
- DEV-ONLY (never in production image): `pytest*`, `debug/` script deps.

---

## 4. Required Bundled Assets (must be IN the image)

| Asset | Path | Used by |
|---|---|---|
| openWakeWord bundled ONNX models | inside the `openwakeword` package (`resources/models/`) — resolver also searches `models/wake/bundled/` | `wake_model_manager`, `wake_resolver` |
| Tesseract English data | `data/tessdata/eng.traineddata` (`TESSDATA_PREFIX` set by `Diego.py`) | `EnhancedOCREngine` |
| Wake verifier + metadata | `models/wake/verifier.pkl`, `models/wake/metadata.json` | wake transcript verification |
| Wake chime | `Diego.wav` (repo root) | wake chime playback |
| Face detection model | `face_detection_yunet_2023mar.onnx` (repo root) | `auth/face_detector` |
| Face encodings | `auth/Known_encodings.p` | face auth match |
| Intent classifier (optional) | `models/intent_classifier.pkl`, `models/metadata.json` | NLP classifier path (degrades if absent) |

Do NOT strip `models/` or `data/tessdata/` from the image.

---

## 5. First-Run Downloaded Models

| Model | Trigger | Cache location | Download source |
|---|---|---|---|
| faster-whisper `base` | first STT / wake verification | `HF_HOME` (default `~/.cache/huggingface`) | `Systran/faster-whisper-base` |
| `all-MiniLM-L6-v2` | first embedding query | `HF_HOME` | `sentence-transformers/all-MiniLM-L6-v2` |
| Silero VAD | `unified_vad.load()` at boot | package/HF cache | `silero-vad` |
| Kokoro `kokoro-82M` | first TTS synthesis | HF cache | `hexgrad/kokoro-82M` |
| Playwright Chromium | `python -m playwright install chromium` | `~/.cache/ms-playwright` | Playwright CDN |
| PaddleOCR/EasyOCR | first OCR use of those backends | `~/.paddleocr` etc. | vendor CDNs |

`main.py` auto-enables `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1` when the
embedding model is already cached — pre-baking `HF_HOME` into the image (or
a volume) makes the container fully offline-friendly.

---

## 6. Persistent Volumes

| Volume (container path) | Content | Why persist |
|---|---|---|
| `/app/data` | `Diego.duckdb`, `knowledge_base.json`, `embedding_cache.pkl`, `whisper_backend.json`, `audio_devices.json`, `mic_selection.json`, `tts_cache/` | Memory/knowledge/device state must survive restarts |
| `/app/models` | wake verifier + metadata, intent classifier | Trained assets |
| `/app/auth/Known_encodings.p` | Face encodings | Enrollment state |
| HF cache (e.g. `/app/.cache/huggingface`) | Whisper/embedding/Kokoro models | Avoid re-download on restart |

---

## 7. Host Devices

| Device | Purpose | docker flag |
|---|---|---|
| Microphone | Voice input | `--device /dev/snd` (ALSA) or PulseAudio/PipeWire socket (see §10) |
| Speakers | TTS/chime output | same as audio |
| Camera | Face auth | `--device /dev/video0` (+ v4l2) |
| GPU | CUDA offload | `--gpus all` (see §11) |

---

## 8. X11 / Wayland Requirements

- Qt UI mode requires an X server: mount `/tmp/.X11-unix`, set `DISPLAY`
  (e.g. `:0`), and `xhost +local:` on the host (Diego also runs
  `xhost +local:` itself, non-fatal).
- `XAUTHORITY` if the X server requires cookies.
- Wayland: only via XWayland (`DISPLAY` still required); no native Wayland
  path in the code.
- Without a display the runtime falls back to headless (`--headless` /
  `python Diego.py`); the engine still runs, but UI, screen capture
  (`mss`), and PyAutoGUI desktop actions are unavailable.

---

## 9. PipeWire / PulseAudio / ALSA Requirements

- Preferred: PulseAudio socket passthrough — mount
  `/run/user/1000/pulse` and set `PULSE_SERVER=unix:/run/user/1000/pulse/native`.
- PipeWire hosts: expose the PulseAudio-compatible socket (pipewire-pulse).
- Raw ALSA: `--device /dev/snd` (no mixing; Pulse/PipeWire preferred).
- Audio failure behavior: `AudioManager.start()` fails → engine retries
  every 5 s; if it never succeeds the engine stops. Runtime health reports
  `microphone` as REQUIRED.

---

## 10. Camera Access

- Device: `/dev/video0` (v4l2). Passed with `--device /dev/video0`.
- Camera opens ONLY in the `FACE_AUTH` state after a verified wake — never
  at startup.
- Failure behavior: auth failure is reported honestly; the auth provider
  stays active; the session continues unauthenticated. Only `--no-auth`
  disables auth.

---

## 11. GPU Requirements

- Lockfile is CPU-only (`torch==2.13.0+cpu`): the default image needs NO
  GPU.
- Optional CUDA image: replace torch wheels with `+cu*` builds, use
  `--gpus all` with host NVIDIA driver + Container Toolkit. STT prefers
  CUDA float16 when `torch.cuda.is_available()`.
- Ollama GPU acceleration is HOST-side (daemon outside the container).

---

## 12. Ollama Connectivity

- Default endpoint: `http://localhost:11434` (`OLLAMA_BASE_URL`).
- In a container, `localhost` is the container itself — point
  `OLLAMA_BASE_URL` at the host (e.g. `http://host.docker.internal:11434`
  or the host IP / `--network host`).
- Warm-up is non-blocking and fail-safe; if Ollama is unreachable the LLM
  loads on first use and failures degrade to canned responses.

---

## 13. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | LLM daemon endpoint |
| `OLLAMA_MODEL` | auto-detect | Ollama model name |
| `OLLAMA_KEEP_ALIVE` | `10m` | model residency |
| `LLM_WARMUP_ENABLED` / `LLM_WARMUP_TIMEOUT_S` / `LLM_WARMUP_VISION` | `true` / `120` / `false` | LLM warm-up |
| `WHISPER_MODEL` | `base` | STT model size |
| `WAKE_PHRASE` / `WAKE_WORD` / `WAKE_MODEL` | `hello diego` / `Diego` / auto | wake config |
| `WAKE_DEVICE_INDEX` | auto | input device index |
| `TTS_ENGINE` / `TTS_VOICE` / `TTS_DEVICE` / `TTS_RATE` / `TTS_VOLUME` / `TTS_CACHE` / `TTS_CACHE_DIR` / `TTS_SPEAKER_WAV` | auto | TTS config |
| `VOICE_RATE` / `VOICE_VOLUME` / `VOICE_ID` | — | legacy voice aliases |
| `DUCKDB_PATH` | `data/Diego.duckdb` | memory store location |
| `KNOWLEDGE_SCAN_ROOTS` | `~/Documents,~/Desktop,~/Downloads,~/Projects` | knowledge scan roots (mount host dirs) |
| `KNOWLEDGE_DENYLIST` | sensitive paths | never scanned |
| `HF_HOME` / `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | — | model cache/offline control |
| `DIEGO_RECORD_SESSION` | off | session recording |
| `LOG_LEVEL` | `INFO` | logging |
| `DISPLAY` / `XAUTHORITY` / `PULSE_SERVER` | — | host integration |
| `.env` file | — | all `config/settings.py` values can also come from a mounted `.env` |

---

## 14. Startup Command

```
python main.py            # UI mode (Qt on main thread; engine on DiegoPipeline background thread)
python main.py --headless # CLI runtime without Qt (recommended for containers)
python Diego.py           # same runtime, no UI
```

Container recommendation: `python main.py --headless` (or `python Diego.py`)
— no DISPLAY needed; the engine boots audio → wake → VAD → TTS → Whisper
and runs forever. Diego NEVER exits on its own.

---

## 15. Health Checks

- Boot-time: `core/runtime_health.run_runtime_health()` prints one
  `[HEALTH]` line per component with status `OK` / `DEGRADED` / `MISSING`
  and a `REQUIRED`/`OPTIONAL` class. Runs ONCE at startup; never retries a
  permanently missing component.
- REQUIRED components: microphone (sounddevice), VAD, STT, TTS.
- Runtime diagnostics: `conversation_engine.get_diagnostics()` exposes
  state, wake (READY/BYPASSED/DEGRADED), auth (ACTIVE/DISABLED), turn
  count, and response-guarantee status.
- Suggested container health probe: check the process is alive and (if
  logs are accessible) that a `STATE`/`[HEALTH]` line appeared after boot;
  there is no HTTP health endpoint in the current codebase.

---

## 16. Shutdown Requirements

- SIGINT/SIGTERM → `Diego.shutdown()` with a **hard 15 s timeout**; hung
  workers cannot prevent exit.
- Shutdown saves the session recorder, stops the engine, closes TTS, STT,
  audio, and the face-auth camera.
- The container must forward SIGTERM (no PID-1 shell wrapping that swallows
  signals; use `exec python main.py --headless` or `tini`).

---

## MUST NOT Be Baked Into the Image

- `data/Diego.duckdb` (+ `-wal`/`-shm`) — writable runtime state.
- `data/embedding_cache.pkl`, `data/knowledge_base.json` — runtime state.
- `data/audio_devices.json`, `data/mic_selection.json` — per-machine device selections.
- `data/whisper_backend.json`, `data/tts_cache/` — runtime caches.
- `auth/Known_encodings.p` — user biometric data (mount as a secret/volume).
- `.env` — credentials; mount at runtime, never in layers.
- HF model caches (unless intentionally pre-baking; prefer volumes).
- `debug/`, `tests/`, `scripts/`, `datasets/` — dev-only.
- `requirements.before-cuda-repair.txt` — historical snapshot.

## MUST Persist Across Restarts

- `data/Diego.duckdb` (memory/knowledge store)
- `data/knowledge_base.json`, `data/embedding_cache.pkl`
- `models/wake/verifier.pkl` + `metadata.json`, `models/intent_classifier.pkl`
- `auth/Known_encodings.p`
- `data/audio_devices.json`, `data/mic_selection.json` (or re-probe each start)
- HF model cache (avoid re-downloading multi-GB models)

## Must Be Downloaded at First Run (unless pre-baked)

- faster-whisper `base`, `all-MiniLM-L6-v2`, Silero VAD, Kokoro-82M,
  Playwright Chromium, PaddleOCR/EasyOCR models.

## Fail Startup vs Degrade Gracefully

| Condition | Behavior |
|---|---|
| Missing Python deps / broken import at boot | **FAIL STARTUP** (import error) |
| No microphone at all (after 5 s retries) | **FAIL** (engine stops) |
| Wake model unavailable | **DEGRADE** — always-LISTEN fallback; auth stays active |
| Silero VAD missing | **DEGRADE** — energy-based VAD |
| STT unavailable | **DEGRADE** — spoken error, return to IDLE |
| TTS engine failure | **DEGRADE** — next engine in chain; response_guarantee fallback |
| Ollama unreachable | **DEGRADE** — warm-up skipped; canned responses |
| No DISPLAY | **DEGRADE** — headless mode |
| Camera missing | **DEGRADE** — honest auth failure; session unauthenticated |
| DuckDB store failure | **DEGRADE** — memory features skip |
| OCR backends missing | **DEGRADE** — vision degraded |
| Knowledge scan roots empty | **DEGRADE** — local knowledge skipped |