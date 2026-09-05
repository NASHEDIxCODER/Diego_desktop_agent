<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Linux-orange.svg" alt="Linux">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

<h1 align="center">🦁 Diego Desktop Agent</h1>

<p align="center"><em>A conversational AI desktop agent for Linux — wake-word activated, face-authenticated, full-duplex streaming voice, persistent memory, local knowledge, screen vision, and desktop control.</em></p>

<p align="center"><strong>Talk naturally. Interrupt any time. Diego listens, thinks, speaks, and acts.</strong></p>

---

## What Diego Does

- **Voice conversation** — wake word ("hello diego") activates, streaming Whisper STT, local LLM via Ollama, streaming Kokoro TTS. Full duplex: interrupt Diego mid-sentence by speaking.
- **Face authentication** — camera opens only after wake, matches enrolled faces, 10-minute sessions.
- **Desktop control** — open/close/focus apps, browser automation, mouse/keyboard, volume/brightness, screenshots, OCR.
- **Local knowledge** — indexes your Documents/Desktop/Downloads, answers questions from your files.
- **Persistent memory** — DuckDB-backed conversation and semantic memory across restarts.
- **Screen vision** — OCR, layout analysis, action verification via screen diffs.

## Requirements

- Linux (Debian/Ubuntu-based), Python 3.11
- [Ollama](https://ollama.ai) running locally (for LLM + vision)
- Microphone + speakers (audio)
- Camera (face auth, optional)

## Installation

```bash
# 1. System packages
sudo apt install portaudio19-dev tesseract-ocr libgl1 libglib2.0-0 espeak-ng \
                 libopenblas-dev libasound2 libpulse0

# 2. Python dependencies
pip install -r requirements.txt

# 3. Playwright browser
python -m playwright install chromium

# 4. Ollama model
ollama pull qwen2.5:3b   # or any chat model
```

## Usage

```bash
# Full voice mode (wake word + face auth)
python main.py

# Headless (no display needed)
python main.py --headless

# Dev bypass — skip wake and face auth
python main.py --no-wake --no-auth
```

Wake calibration (record your voice saying the wake phrase):
```bash
python main.py --train-wake
```

Face enrollment: place face images in `auth/images/<name>/`.

## Example Voice Commands

- "Open Firefox and go to wikipedia.org"
- "Take a screenshot"
- "What's my CPU temperature?"
- "Turn volume up"
- "Play some music"
- "What documents do I have about taxes?"
- "Close that window"

## Docker

```bash
docker build -t diego:latest .

docker run -d --name diego \
  --device /dev/snd --device /dev/video0 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 \
  -v diego-data:/app/data -v diego-models:/app/models -v diego-cache:/app/.cache \
  --shm-size=1g diego:latest
```

> Uses CPU-only PyTorch by default. For GPU: replace torch wheels with `+cu*` builds and add `--gpus all`. See [docker/README.md](docker/README.md) for volumes, health checks, and shutdown details.

## Documentation

- **Architecture**: [docs/DIEGO_ARCHITECTURE.md](docs/DIEGO_ARCHITECTURE.md)
- **Runtime matrix**: [docs/DOCKER_RUNTIME_MATRIX.md](docs/DOCKER_RUNTIME_MATRIX.md)
- **Docker preflight**: [docs/DOCKER_PREFLIGHT.md](docs/DOCKER_PREFLIGHT.md)

## License

MIT