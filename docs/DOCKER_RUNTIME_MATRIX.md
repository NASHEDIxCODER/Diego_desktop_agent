# Diego Docker Runtime Matrix

Companion to `docs/DIEGO_ARCHITECTURE.md`. Classifies every runtime
dependency/asset required to run Diego in a container, based on the actual
source tree: `requirements.txt`, `requirements.runtime-lock.txt`,
`config/settings.py`, model loaders, import graph, and startup code.

Not an instruction to build the image — this file prepares that decision.

---

## Classification Legend

| Code | Meaning |
|---|---|
| **REQUIRED-IN-IMAGE** | Must be present in the container image to boot the default production path. |
| **OPTIONAL-IN-IMAGE** | Not needed by the default path; enables extra functionality (provider/backends/features). |
| **DEV-ONLY** | Used only by tests / `debug/` / `scripts/`; must NOT ship in a production image. |
| **DOWNLOAD-ON-FIRST-RUN** | Package/model fetches on first use at runtime (network required at container start). |
| **HOST-PROVIDED** | Device, socket, or service from the host that is passed into the container (never baked in). |

---

## 1. Python Version

| Item | Classification | Notes |
|---|---|---|
| CPython runtime | REQUIRED-IN-IMAGE | Lockfile (`requirements.runtime-lock.txt`) resolves on **Python 3.11**. Runner also boots under **3.14** via `compat.py` shims (`audioop`, `imghdr`, `aifc`). Recommend 3.11-based image to match the lockfile; 3.12+ drops Coqui `TTS` (`requirements.txt` pins `TTS>=0.22.0; python_version < "3.12"`). |
| pip / venv | REQUIRED-IN-IMAGE | `requirements.txt` + `requirements.runtime-lock.txt` installed into image venv. |

---

## 2. System Packages (apt)

| Package | Purpose | Loaded/used by | Classification | Failure behavior |
|---|---|---|---|---|
| `libc6` + base toolchain | glibc, gcc, make, cmake, pkg-config | All C-ext wheels (dlib, sounddevice, onnxruntime) | REQUIRED-IN-IMAGE | Wheel import fails without matching glibc. |
| `portaudio19-dev` | sounddevice/PyAudio backend | `voice/audio_manager.py`, `voice/streaming_tts.py` | REQUIRED-IN-IMAGE (build+run) | `sounddevice` import or device open fails → audio probe fails, engine retries then stops. |
| `libasound2` (ALSA) + `libpulse0` | Microphone/speaker access | AudioManager/TTS | HOST-PROVIDED (host audio) + REQUIRED libs IN-IMAGE | No device → `[AUDIO]` failure, runtime health DEGRADED. |
| `tesseract-ocr` | OCR backend | `vision/ocr_pipeline.py` via `pytesseract` | REQUIRED-IN-IMAGE (or DOWNLOAD) | OCR unavailable → read_screen falls back to other OCR libs or fails honestly. |
| `espeak-ng` | pyttsx3 fallback + wake calibration synthesis | `voice/tts/...pyttsx3`, `voice/calibrate_wake.py` | OPTIONAL-IN-IMAGE (fallback chain) | pyttsx3 path fails → streaming_tts falls to next engine. |
| `libgtk-3-0` + `libatspi2.0-0` + `gir1.2-atspi-2.0` | AT-SPI accessibility tree (PyGObject `gi`) | `services/accessibility.py`, `services/ui_tree.py` | OPTIONAL-IN-IMAGE | a11y unavailable → perception uses window/OCR only. |
| `libgl1`, `libglib2.0` | OpenCV | `auth/*`, `vision/*` | REQUIRED-IN-IMAGE | cv2 import/init fails → face auth + vision unavailable. |
| `libxcb*`, `libxkbcommon*` | PySide6 Qt runtime | `ui/*` | REQUIRED-IN-IMAGE (library) but DISPLAY HOST-PROVIDED | Qt import fails without X libs; no DISPLAY → headless mode only. |
| `curl/wget` | Image build tooling | (build) | DEV-ONLY | — |
| `ffmpeg` libs | audio/video decode for TTS/WAVE media | `voice/streaming_tts.py` | OPTIONAL-IN-IMAGE | Only some engines need ffmpeg; pyttsx3/kokoro still work. |
| `fonts-dejavu*` | Qt font rendering | `ui/*`, OCR preprocessing | OPTIONAL-IN-IMAGE | Missing fonts → ugly text; Diego sets `QT_QPA_FONTDIR`. |
| `xvfb` | Headless virtual display for CI | tests / headless GUI | DEV-ONLY | — |

---

## 3. Python Packages

### 3.1 REQUIRED-IN-IMAGE (active production path)

| Package | Version source | Purpose | Imported by | Startup load | Failure behavior |
|---|---|---|---|---|---|
| `numpy` | requirements.txt / lock | Arrays, audio math, vision | All core/voice/vision | Import at boot | Crash if missing. |
| `sounddevice` | req | Mic/speaker devices | audio_manager, streaming_tts, engine | Start | Audio fails → engine retries 5s loop → stops. |
| `faster-whisper` | req | STT + wake verification | command_listener, wake verifier | Boot (model load) | STT unavailable → engine speaks "recognizer unavailable", returns IDLE. |
| `openwakeword` | req | Wake-word streaming predict | wake_model_manager, wake_listener | Boot | Wake DEGRADED → always-LISTEN fallback. |
| `silero-vad` | req | VAD | voice/vad.py | Boot | `unified_vad.ready=False` → energy fallback used. |
| `torch` (+`torchaudio`) | lock `+cpu` | VAD/whisper/embedding backend | voice/vad, command_listener, nlp/embeddings, vision/ocr | Boot | Heavy CPU dependency; CUDA variant is HOST/GPU concern (see §8). |
| `onnxruntime` | req | OpenWakeWord ONNX inference, Nemotron optional | wake_model_manager, providers | Boot | Wake load fails → DEGRADED. |
| `sentence-transformers` | req (`all-MiniLM-L6-v2`) | Embeddings | nlp/embeddings.py | First query / warm cache | Embeddings unavailable → classifier/knowledge degrade. |
| `transformers` | req | HF stack | embeddings, providers | First use | — |
| `huggingface-hub` | req | Model downloads | providers, embeddings | First use | Download failure → provider unavailable. |
| `httpx`, `aiohttp` | req | Ollama + search | streaming_llm, search_service | Boot/stream | LLM/Web ground → degrade. |
| `duckdb` | req | Persistent memory/knowledge store | memory/duckdb_store | Boot | Store unavailable → memory features skip. |
| `pydantic-settings` | req | `config/settings.py` | settings | Import | Config fails → boot fails. |
| `PySide6` | req | UI | ui/* | Boot (UI mode) | UI fails → headless mode only. |
| `python-dotenv` | lock | `.env` loading | pydantic-settings | Import | — |
| `opencv-python(-headless)` | req / lock (`headless`) | Face + vision | auth/*, vision/* | Import | Face/vision unavailable → auth fails, vision degraded. |
| `face-recognition` (dlib) | req | Face encoding match | auth/encode, live_auth | Auth gate | Auth unavailable → honest auth failure, session unauthenticated. |
| `pillow` | req | Image IO | auth/face_popup, action_dispatcher | Import | — |
| `mss` | req | Screen capture | screen_capture | First sight | Vision/perception degraded. |
| `rapidfuzz`, `jellyfish` | req | Text matching | command_normalizer, wake_word | Import | — |
| `scipy` | req | Audio filters/ports | audio_manager, audio_processing | Import | — |
| `scikit-learn` | req | Classifier verifier | wake calibrate, nlp | Import | Optional for default path, required for `--train`. |
| `spacy` + `en_core_web_sm` | req/lock (URL wheel) | Entity/tokenizer | nlp/entities, nlp/tokenizer | Import | Optional in pipeline; missing → deprecation path. |
| `PyGObject` | req | a11y | services/accessibility | Import | a11y disabled if absent. |
| `pytesseract` | req | OCR | ocr_pipeline, action_dispatcher | First OCR | OCR unavailable → vision degraded. |
| `kokoro` | req (default TTS) | Streaming TTS | streaming_tts | TTS init | Falls to next TTS engine. |
| `playwright` | req | Browser automation | agent/browser, search_service | First browser use | Browser actions fail with clear message. |
| `pyautogui`, `pyperclip` | req | Mouse/keyboard/clipboard | executor, tool_registry | Import | Desktop actions unavailable. |
| `rapidfuzz` | req | — | nlp | Import | — |

### 3.2 OPTIONAL-IN-IMAGE

| Package | Purpose | Used by | Failure behavior |
|---|---|---|---|
| `sherpa-onnx` (+ `sherpa-onnx-core`) | Alternative ASR providers | `voice/providers/sherpa_*` | Not on default path; skipped. |
| `llama-cpp-python` | Nemotron GGUF ASR | `voice/providers/nemotron_provider` | Provider unavailable; not default. |
| `nemo-toolkit[asr]` | Nemotron NeMo ASR | nemotron_provider | — |
| `TTS` (Coqui XTTS, `<3.12` only) | XTTS fallback | `_XTTSSynth` | On py≥3.12 unavailable; TTS chain degrades. |
| `piper-tts` | Piper TTS fallback | `_PiperSynth`, `voice/tts/manager` | Falls to pyttsx3. |
| `pyttsx3` | espeak fallback TTS | `_Pyttsx3Synth` | Requires `espeak-ng`; if absent, TTS fails. |
| `easyocr` | OCR alternative | `ocr_pipeline` | OCR degraded to Tesseract/Paddle. |
| `paddleocr` + `paddlepaddle` | OCR alternative | `ocr_pipeline` | — |
| `audioop-lts` (`python_version >= 3.13`) | audioop stdlib shim | `compat.py` | On 3.11 unneeded; on 3.14 required by compat. |
| `firebase-admin` | Face encoding upload | `auth/encode.py` | Required only for remote enroll; local auth works without. |
| `aiohttp` | Search async | `search_service` | Search degrades. |

### 3.3 DEV-ONLY (do NOT bake into production image)

`pytest`, `pytest-asyncio`, `pytest-base-url`, `pytest-playwright`,
`debug/` scripts deps (benchmark ASR etc.), and all `requirements.before-cuda-repair.txt`
transitive packages that are not on the production import graph.

---

## 4. Model / Asset Inventory

### 4.1 Wake models

| Item | Classification | Purpose | Loaded from | Startup | Download source | Fallback | Failure |
|---|---|---|---|---|---|---|---|
| Bundled openWakeWord .onnx (diego/hello diego) | REQUIRED-IN-IMAGE | Streaming wake predict | openWakeWord package `models/` or `models/wake/bundled/` (resolver searches); **`models/wake/` on disk currently contains only verifier/metadata, no .onnx** | Boot | pip `openwakeword` includes models; ensure they ship in image | Wake DEGRADED → always-LISTEN | Load error logged; engine continues |
| Custom trained verifier (`verifier.pkl` + `metadata.json`) | REQUIRED-IN-IMAGE (if shipped) | Whisper wake transcript verification | `models/wake/` + `models/wake/metadata.json` | Boot | Trained locally via `main.py --train-wake` | Detection-only (no verifier) | Verifier unavailable → wake accepts without transcript check |
| Positive/negative WAV corpora | DEV-ONLY | Retraining/calibration | `models/wake/positives|negatives/` | Retrain only | Generated locally | — | — |

### 4.2 STT models

| Item | Classification | Purpose | Loaded from | Download source | Fallback |
|---|---|---|---|---|---|
| faster-whisper `base` | DOWNLOAD-ON-FIRST-RUN (cache to HF_HOME) | STT + wake verification | HF cache (`~/.cache/huggingface`), `WHISPER_MODEL="base"` | HuggingFace `Systran/faster-whisper-base` | Backend cache in `data/whisper_backend.json`; on failure engine returns IDLE |
| sherpa/nemotron ASR models | OPTIONAL (not default) | Alternative ASR | `data/asr_models/<name>/` via HF snapshot | HuggingFace repos (see `sherpa_providers.py`) | Provider skipped |

### 4.3 TTS models

| Item | Classification | Purpose | Loaded from | Download source | Fallback |
|---|---|---|---|---|---|
| Kokoro `kokoro-82M` | DOWNLOAD-ON-FIRST-RUN | Primary TTS | HF cache via `KPipeline` | HuggingFace `hexgrad/kokoro-82M` | Piper → pyttsx3 |
| Piper voice .onnx | OPTIONAL | TTS | `voice_settings.piper_model` (path) | Manual model download | pyttsx3 |
| Coqui XTTS `xtts_v2` | OPTIONAL | TTS | `TTS("tts_models/.../xtts_v2")` | Coqui hub (py<3.12) | — |
| espeak (pyttsx3) | OPTIONAL (system) | Last TTS fallback | system `espeak-ng` | apt | TTS failure spoken via response_guarantee |

### 4.4 OCR data

| Item | Classification | Purpose | Loaded from | Fallback |
|---|---|---|---|---|
| `data/tessdata/eng.traineddata` | REQUIRED-IN-IMAGE (bundled) | Tesseract English data | repo `data/tessdata/`; `TESSDATA_PREFIX` set in `Diego.py` | system tessdata; PaddleOCR/EasyOCR |
| PaddleOCR/EasyOCR model dirs | DOWNLOAD-ON-FIRST-RUN | Worker OCR | `~/.paddleocr`, HF/.. | Tesseract |

### 4.5 Embedding models/cache

| Item | Classification | Purpose | Loaded from | Startup |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | DOWNLOAD-ON-FIRST-RUN then REQUIRED-IN-IMAGE or volume | Intent + knowledge embeddings | `HF_HOME` snapshots; `main.py` sets `TRANSFORMERS_OFFLINE=1` when cached locally | First query; warm on boot |
| `data/embedding_cache.pkl` | HOST-PROVIDED (runtime state) | Cached embeddings (33 MB local) | repo data dir, gitignored at runtime | — |

### 4.6 DuckDB / data

| Item | Classification | Purpose | Loaded from | Failure |
|---|---|---|---|---|
| DuckDB file `data/Diego.duckdb` | HOST-PROVIDED volume (runtime state, writable) | Unified memory / knowledge store | `settings.DUCKDB_PATH` creates on boot | Store failure logged; features skip. Never bake in image. |
| `data/knowledge_base.json` | HOST-PROVIDED (content state) | Knowledge/learning persistence | repo data dir | — |
| `data/audio_devices.json` + `data/mic_selection.json` | HOST-PROVIDED (per-machine devices) | Device persistence | written at runtime | First run probes/auto-selects |
| `models/intent_classifier.pkl` + `metadata.json` | OPTIONAL-IN-IMAGE (if `--train` used) | Trained NLP classifier | repo models dir | Missing → `--status` reports "not found"; classifier path disabled |

---

## 5. HOST-PROVIDED

| Item | Purpose | How passed in | Failure behavior |
|---|---|---|---|
| Microphone (PulseAudio/PipeWire/ALSA) | Voice input | Host audio socket / `--device /dev/snd`, `PULSE_SERVER` | Audio probe fails → retry → engine stops |
| Speakers | TTS/chime output | Same as audio | TTS may fail but engine continues |
| Camera (/dev/video0) | Face auth | `--device /dev/video*` + v4l2 | Auth unavailable → honest failure, session unauthenticated |
| X11 socket (`/tmp/.X11-unix`) | Qt GUI, screen capture, PyAutoGUI | `DISPLAY`, `XAUTHORITY`, `xhost +local:` | No display → headless mode |
| Wayland (via XWayland) | Qt/OCR | `WAYLAND_DISPLAY` or XWayland | Falls to headless |
| Ollama daemon (`http://localhost:11434`) | LLM + vision model | `OLLAMA_BASE_URL` env; bind to host port 11434 | Warm-up non-fatal; runtime LLM requests fall to canned responses |
| GPU/CUDA | CUDA offload | `--gpus all` + CUDA libs/driver host-side | Falls to CPU (lock `torch +cpu`) |
| Host file system (knowledge scan roots) | Knowledge indexing | `KNOWLEDGE_SCAN_ROOTS` volume mounts (`~/Documents` etc.) | Empty index → local knowledge skipped |
| `/dev/shm` | Browser/Qt shared memory | `--shm-size` | Playwright/Qt crashes if too small |

---

## 6. First-Run Downloads (network at container start)

| Asset | Trigger |
|---|---|
| faster-whisper `base` | First STT/wake verification |
| `all-MiniLM-L6-v2` | First embedding query |
| Silero VAD model | `load_silero_vad(onnx=True)` first load |
| Kokoro `kokoro-82M` | First TTS synthesis |
| Playwright Chromium | `python -m playwright install chromium` (post-install note in requirements.txt) |
| PaddleOCR/EasyOCR | First OCR use of those backends |

Recommendation: pre-bake caches (HF_HOME pipeline cache) or mount a
shared model volume to keep the container offline-friendly — `main.py`
auto-enables `TRANSFORMERS_OFFLINE=1` when the embedding model is cached.

---

## 7. Dependency Gaps (RESOLVED)

Cross-checked **actual production imports** (excluding stdlib/first-party
and package-name aliases) against `requirements.txt` + the lockfile.

**Fixed in this pass** (added to `requirements.txt` and pinned in
`requirements.runtime-lock.txt`):

| Package | Imported by | Note |
|---|---|---|
| `PyPDF2==3.0.1` | `knowledge/extractors.py` | The actively used PDF implementation (`pypdf` is tried first but is NOT installed; the code falls back to PyPDF2 — so PyPDF2 is the declared dependency) |
| `python-docx==1.2.0` | `knowledge/extractors.py` | DOCX extraction |
| `python-pptx==1.0.2` (+ `xlsxwriter==3.2.9` transitive) | `knowledge/extractors.py` | PPTX extraction |
| `openpyxl==3.1.5` (+ `et-xmlfile==2.0.0` transitive) | `knowledge/extractors.py` | XLSX extraction |

**Non-gaps / corrections:**

- `aifc` — NOT a gap: `compat.py` already handles the removed stdlib
  module correctly on 3.14 (`try: import aifc` → wave-delegating stub via
  `inject_aifc_stub()`, called from `inject_all()` on import). Python 3.14
  support is intentional.
- `piper` module — covered by the declared `piper-tts` distribution.

**Orphaned references (made inert, not faked):**

- `voice/tts/manager.py` referenced `voice.tts.piper_engine`,
  `voice.tts.xtts_engine`, and `voice.tts.pyttsx3_engine` — none of these
  modules exist in the tree. TTSManager is NOT reachable from any
  production entrypoint (only `tests/` and `debug/` import it); the active
  TTS path is `voice/streaming_tts.py`. The piper branch now logs a clear
  "legacy engine module missing" warning and returns None (clean
  fallback); the xtts/pyttsx3 branches are inert via the existing
  try/except with a clear log.

**Remaining notes:**
- `TTS` (Coqui XTTS) is `python_version < "3.12"`; on py≥3.12 `_XTTSSynth`
  import fails → TTS engine chain degrades to pyttsx3 (usable) — a real
  limitation for 3.14 images.
- The lockfile pins `torch==2.13.0+cpu` (CPU build) — a GPU image would
  need a separate torch wheel/tag (`+cu*`) — that's a build decision, not
  a code change.
- `data/Diego.duckdb` and `embedding_cache.pkl` are large runtime state
  files and must not be baked into the image (volume instead).
- `models/wake/` currently holds only the trained verifier; the ONNX
  wake model ships inside the `openwakeword` package, so the image must
  install `openwakeword` with its bundled resources (pip default) and NOT
  strip `models/`.
- openwakeword version compatibility: the runtime now supports BOTH
  openwakeword 0.4.0 (installed in the 3.14 user site-packages; no
  `inference_framework` parameter) and 0.6.0 (lockfile; requires explicit
  `inference_framework="onnx"` for .onnx models). `voice/wake_model_manager.py`
  and `voice/calibrate_wake.py` inspect the installed `Model.__init__`
  signature and pass `inference_framework` / choose
  `wakeword_model_paths` vs `wakeword_models` accordingly.

---

## 8. GPU / CUDA Note

- Production lockfile is CPU-only (`torch 2.13.0+cpu`).
- `command_listener`/`streaming_stt` prefer CUDA float16 when
  `torch.cuda.is_available()`; a CUDA image would replace the torch wheels
  and mount the host driver. No code change required.
- Ollama GPU is HOST-side (Ollama daemon outside container).

---

## 9. Summary Classification by Layer

| Layer | Classification |
|---|---|
| Python 3.11 runtime + lockfile wheels | REQUIRED-IN-IMAGE |
| System libs (ALSA, GTK, Tesseract, X libs, OpenCV libs) | REQUIRED-IN-IMAGE |
| `openwakeword` bundled models + `data/tessdata` | REQUIRED-IN-IMAGE |
| Whisper/embedding/Kokoro/VAD models | DOWNLOAD-ON-FIRST-RUN (pre-bake or volume) |
| ASR providers (sherpa/nemotron/XTTS/Piper/Paddle) | OPTIONAL-IN-IMAGE |
| Test tooling (`pytest*`), debug/ scripts | DEV-ONLY |
| Microphone, speakers, camera, X11/Wayland, Ollama, GPU, DuckDB data, knowledge scan roots, device JSONs | HOST-PROVIDED |