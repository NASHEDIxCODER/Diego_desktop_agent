<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/platform-Linux%20(Arch)-orange.svg" alt="Arch Linux">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

<h1 align="center">🦁 Diego — Desktop AI Assistant</h1>

<p align="center"><em>A local-first, voice-first desktop AI assistant for Linux.<br>
Wake word · face authentication · streaming speech · local knowledge · desktop and browser control.</em></p>

Diego is a **local-first desktop AI assistant** designed to run entirely on your machine, without mandatory cloud AI APIs. It listens for a wake word, verifies who is speaking with face authentication, transcribes your voice locally, thinks with a locally hosted LLM, and acts on your desktop — all on Linux.

- **Voice-first interaction** — full-duplex streaming conversation; interrupt Diego mid-sentence by speaking.
- **Wake-word activation** — openWakeWord detection with Whisper transcript verification.
- **Face authentication** — camera opens only after a verified wake; 10-minute sessions; fully local.
- **Speech recognition** — local streaming faster-whisper STT with confidence gating.
- **Local speech synthesis** — streaming TTS (Kokoro → Piper → pyttsx3 fallback chain).
- **Local knowledge / retrieval** — indexes your Documents/Desktop/Downloads/Projects and answers from your own files (DuckDB-backed, no cloud).
- **Desktop / system interaction** — open/close/focus apps, volume, brightness, windows, workspaces, screenshots, OCR, media playback, system info.
- **Browser interaction** — Chrome/Chromium automation via CDP, navigation, search, tabs, typing.
- **Verification-oriented task execution** — multi-step tasks run in a closed loop: plan → execute → verify (screen diffs + OS-level checks) → re-plan → confirm with you before sensitive actions.
- **No mandatory cloud AI** — LLM runs via local [Ollama](https://ollama.ai); optional cloud API keys exist in config but are never required.

---

## Current Status

⚠️ **Diego is under active development / production hardening.**

- **Native Linux/Arch execution is the primary development and runtime target.** Diego is developed, tested, and run directly on Arch Linux (Python 3.11).
- Some subsystems are still being improved — especially **audio input**, **wake-word detection**, **face authentication**, **vision**, and **autonomous task execution**. Expect rough edges and rapid change.
- **Docker support exists but is secondary/experimental** — it is *not* the primary installation path. Native execution on Arch is.

## Features

- Wake word ("hello diego") with openWakeWord + Whisper transcript verification; optional per-user verifier training (`--train-wake`).
- Face authentication with a live popup UI, quality gates (lighting, blur, pose, eyes-open), and liveness checks.
- Streaming conversation: VAD → streaming Whisper → intent routing → local LLM → streaming, interruptible TTS.
- Voice-first PySide6 HUD (transcript, response, state visualizer, latency metrics) plus a headless CLI mode.
- Desktop actions: app launch/close/focus, volume, brightness, window/workspace/tab management, lock screen, power, screenshots, screen reading (OCR), system info queries.
- Media: MPV, Spotify (playerctl), YouTube playback and search, local files.
- Browser control via Chrome DevTools Protocol (attaches to your running Chrome when possible; otherwise a dedicated persistent profile).
- Local knowledge base: read-only, incremental indexing of approved directories with hybrid semantic + keyword retrieval and spoken citations.
- Persistent memory (DuckDB): conversation history, facts, preferences, command history.
- Closed-loop task execution with plan validation, loop detection, re-planning from observed state, and user confirmation for sensitive actions.
- Boot-time runtime health report (`[HEALTH]` lines) for every component, with honest degraded-fallback behavior.
- Background self-learning during idle time (documentation crawling, workflow discovery, tool reliability tracking).

## Requirements

**Primary supported environment:**

- **Arch Linux** (other distros may work but are not the development target)
- **Python 3.11** (pinned by `requirements.runtime-lock.txt`; 3.14 boots via `compat.py` shims but is a degraded runtime)
- A working **microphone** (Diego refuses to run voice mode on a silent/virtual-only device)
- A working **audio output device** (speakers or headphones, for TTS)
- A **webcam/camera** — required for face authentication and where vision needs it (the camera opens only after wake, never at startup)
- **[Ollama](https://ollama.ai)** — required for LLM features (conversation, planning, reasoning, optional vision). Diego degrades gracefully if Ollama is unreachable, but these features need it.

**Optional:**

- **GPU (CUDA)** — optional. The pinned lockfile is CPU-only (`torch+cpu`); everything runs on CPU. For GPU acceleration, install CUDA-enabled torch wheels; Ollama GPU acceleration is host-side.
- `tesseract-ocr` + `espeak-ng` system packages (OCR backend and TTS fallback engine).

## Installation — Arch Linux

```bash
# 1. System packages (build deps for dlib/face-recognition, PortAudio for audio,
#    Tesseract for OCR; espeak-ng is the optional TTS fallback)
sudo pacman -S --needed base-devel cmake portaudio tesseract espeak-ng

# 2. Clone
git clone https://github.com/NASHEDIxCODER/Diego_desktop_agent.git
cd Diego_desktop_agent

# 3. Python environment
python -m venv .venv
source .venv/bin/activate

# 4. Dependencies
pip install -r requirements.txt
#    …or, for the exact tested pins:
#    pip install -r requirements.runtime-lock.txt

# 5. Playwright browser (browser automation)
python -m playwright install chromium

# 6. Ollama model (small chat model; this one is the tested default)
ollama pull qwen2.5:3b
```

Models (Whisper, embeddings, Silero VAD, Kokoro TTS) download automatically on first use and are cached locally afterward; Diego then runs fully offline except for explicit web search requests.

## Usage

```bash
# Full voice mode (wake word + face auth), with the Qt HUD
python main.py

# Headless CLI runtime (no Qt UI needed)
python main.py --headless

# Dev bypasses
python main.py --no-wake            # skip wake detection (auth still runs)
python main.py --no-auth            # skip face auth (development only)
python main.py --no-wake --no-auth  # full dev mode
```

Wake calibration (record wake phrases + train a personal verifier):

```bash
python main.py --train-wake
```

Other utility commands:

```bash
python main.py --select-mic     # interactively pick your microphone
python main.py --audio-debug    # live audio level visualizer
python main.py --train          # train the NLP intent model
python main.py --status         # NLP model status
python Diego.py --status        # subsystem status (Ollama, Whisper, TTS, encodings)
python Diego.py inspect screen  # full screen inspection report
python Diego.py debug vision    # live vision debug overlay
```

Example voice commands: "open firefox", "take a screenshot", "what's my CPU?", "turn volume up", "play some music", "what documents do I have about taxes?", "close that window".

## Face Enrollment

1. Put one clear face photo per user in `auth/images/`, named `<username>.jpg` (e.g. `auth/images/alice.jpg`).
2. From the `auth/` directory, encode them:

```bash
cd auth
python encode.py     # writes Known_encodings.p (local only — Firebase is optional and skipped without credentials)
cd ..
```

## Testing

```bash
source .venv/bin/activate
pytest -q
```

The suite covers the voice pipeline, wake resolution, UI, NLP/intent gating, knowledge indexing, runtime health, and production-readiness checks. (`pytest.ini` enables async mode; `conftest.py` excludes manual debug harnesses in `debug/`.)

## Docker (secondary / experimental)

Docker exists for containerized execution but is **not** the primary way to run Diego — native Arch execution is. The container image is Debian-based (`python:3.11-slim`), with host-side Arch audio (PipeWire/PulseAudio socket) passed through.

```bash
docker build -t diego:latest .
# see docker/README.md for the full run command (audio socket, camera, X11, volumes)
```

Details: [docker/README.md](docker/README.md), [docs/DOCKER_PREFLIGHT.md](docs/DOCKER_PREFLIGHT.md), [docs/DOCKER_RUNTIME_MATRIX.md](docs/DOCKER_RUNTIME_MATRIX.md).

## Documentation

- **Architecture (canonical)**: [docs/DIEGO_ARCHITECTURE.md](docs/DIEGO_ARCHITECTURE.md)
- **UI**: [docs/ui.md](docs/ui.md)
- **Database layer**: [docs/Database.md](docs/Database.md)
- **NLP training pipeline**: [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md)
- **Knowledge CLI**: `python -m knowledge.cli` (index/status/search/reindex)

## Contributing

Contributions are welcome. Please:

- Develop and test on **Arch Linux with Python 3.11** — that is the supported target.
- Run `pytest -q` before submitting.
- Keep subsystem behavior honest: degraded components must be reported, never silently disabled (see the architecture doc's failure-handling contract).
- Do not introduce mandatory cloud API dependencies — Diego must remain local-first.

## License

MIT — see [LICENSE](LICENSE).
