# 🦁 Leo Desktop Assistant

Leo is a **modular, production-grade AI desktop assistant** for Linux with wake-word activation, face authentication, local NLP engine, plugin architecture, and optional LLM integration.

> Talk to your machine, let it talk back, send Telegrams, control YouTube, and tweak your desktop — all hands-free.

---

## Architecture

```
Speech → STT → Preprocessor → Intent Classifier → Entity Extractor
→ Context Manager → Plugin Router → Response → TTS
```

LLMs (Gemini/OpenAI/Ollama/etc.) are **optional** providers used only for:
- Unknown intent handling
- Reasoning and coding
- Summarization
- Long conversations

**Everything else executes locally.**

See [docs/architecture.md](docs/architecture.md) for full details, including Mermaid diagrams.

---

## ✨ Features

### 🧠 **Local NLP Engine**
- Semantic intent matching using `sentence-transformers` (all-MiniLM-L6-v2)
- 3-tier classification: embeddings → centroids → fuzzy matching
- Entity extraction via spaCy NER + regex
- Context-aware commands with slot-filling
- Confidence scoring with automatic LLM fallback
- Automatic training data generation (100+ examples per intent)

### 🎙 **Voice Control**
- Always-listening wake word (`hello leo`, `leo`, `hey leo`)
- Google Speech Recognition for STT
- Coqui TTS with "Friday-style" voice
- Single ambient-noise calibration at startup

### 🧑‍💻 **Face Authentication**
- Real-time face recognition via OpenCV + face_recognition
- New face enrollment on unknown detection
- Firebase backup of face encodings

### 🔌 **Plugin Architecture**
- Event-driven plugin system via async EventBus
- Each plugin is a standalone module with lifecycle hooks
- Current plugins:
  - **YouTube** — Hands-free video control (open, search, pause, skip, volume, speed, seek, close)
  - **Telegram** — Send, read, and reply to messages
  - **Brightness** — Screen brightness control
- Easy to extend: create a new file in `plugins/` inheriting from `BasePlugin`

### 💾 **Persistent Storage (DuckDB)**
- Stores intents, examples, embeddings, entities, synonyms
- Command history and user preferences
- Plugin registry and context memory with TTL

### 🌐 **Multi-Provider LLM Support**
- Google Gemini (primary)
- OpenAI
- Ollama (local, offline)
- Automatic fallback between providers
- Only used when local NLP confidence is low

### 📊 **Structured Logging & Telemetry**
- JSON-formatted logs with context fields
- Configurable log levels via `.env`
- Command history tracking

---

## Quick Start

### Prerequisites
- Python 3.11+
- Linux with PulseAudio/PipeWire
- Microphone
- Webcam (for face auth)
- Chrome/Chromium (for YouTube Selenium automation)

### Clone & Setup
```bash
git clone https://github.com/NASHEDIxCODER/leo_desktop_assistant.git
cd leo_desktop_assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configuration
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run
```bash
python main.py
```

---

## Project Structure

```
leo_desktop_assistant/
├── main.py                 # Application entry point
├── config/
│   ├── settings.py         # Centralized Pydantic Settings
│   └── .env.example        # Environment template
├── core/
│   ├── event_bus.py        # Async EventBus
│   ├── plugin_base.py      # BasePlugin interface
│   └── plugin_manager.py   # Plugin lifecycle manager
├── nlp/
│   ├── tokenizer.py        # Text tokenization
│   ├── normalizer.py       # Text normalization
│   ├── embeddings.py       # sentence-transformers
│   ├── classifier.py       # Intent classification
│   ├── entities.py         # Entity extraction
│   ├── context.py          # Conversation context
│   ├── confidence.py       # Confidence scoring
│   ├── parser.py           # NLP pipeline orchestrator
│   ├── trainer.py          # Intent training generator
│   └── evaluator.py        # Benchmarking
├── voice/
│   ├── stt.py              # Speech-to-Text
│   └── tts.py              # Text-to-Speech
├── ai/
│   └── llm_client.py       # Unified LLM client
├── plugins/
│   ├── youtube_plugin.py   # YouTube control
│   ├── telegram_plugin.py  # Telegram messaging
│   └── brightness_plugin.py# Screen brightness
├── memory/
│   └── duckdb_store.py     # DuckDB storage
├── telemetry/
│   └── logger.py           # Structured logging
├── auth/                   # Face authentication
├── scripts/                # Legacy scripts
├── tests/                  # Test suite
├── data/                   # Database files
├── docs/
│   └── architecture.md     # Full architecture docs
├── .env.example
├── requirements.txt
└── README.md
```

---

## Voice Commands

### YouTube
- "Play music on YouTube" / "Open YouTube"
- "Pause" / "Resume" / "Play"
- "Next song" / "Previous song"
- "Volume up" / "Volume down" / "Mute" / "Unmute"
- "Forward 10 seconds" / "Rewind"
- "Speed up" / "Slow down"
- "Close YouTube" / "Exit YouTube"

### Telegram
- "Send message to [contact] saying [message]"
- "Read messages from [contact]"
- "Reply saying [message]"

### Brightness
- "Set brightness to 50"
- "Brightness up" / "Brightness down"

### General
- "What time is it?" / "What's the date?"
- "Tell me a joke"
- "What can you do?"
- "Goodbye" / "Exit"

---

## Configuration

All configuration is in `config/settings.py`, read from `.env`:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OLLAMA_BASE_URL` | Local Ollama URL |
| `TELEGRAM_API_ID` | Telegram API ID |
| `TELEGRAM_API_HASH` | Telegram API hash |
| `LOG_LEVEL` | Log level (DEBUG/INFO/WARNING) |
| `WAKE_WORD` | Wake word phrase |
| `MODEL_NAME` | sentence-transformers model |
| `SIMILARITY_THRESHOLD` | Intent match threshold |

---

## Testing
```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ (asyncio) |
| Speech Recognition | SpeechRecognition + Google Web Speech |
| Text-to-Speech | Coqui TTS |
| Intent Classification | sentence-transformers (all-MiniLM-L6-v2) |
| Entity Extraction | spaCy en_core_web_sm |
| Fuzzy Matching | RapidFuzz |
| Database | DuckDB |
| LLM | Gemini / OpenAI / Ollama |
| Browser Automation | Selenium |
| Face Recognition | OpenCV + face_recognition |
| Configuration | Pydantic Settings + python-dotenv |
| Logging | JSON-structured |

---

## 🤝 Contributors

<table>
<tr>
<td align="center">
<a href="https://github.com/NASHEDIxCODER">
<img src="https://avatars.githubusercontent.com/NASHEDIxCODER" width="100px;" /><br />
<b>NASHEDIxCODER</b>
</a><br />
Owner & Lead Developer
</td>
<td align="center">
<a href="https://github.com/dem0000n">
<img src="https://avatars.githubusercontent.com/dem0000n" width="100px;" /><br />
<b>demo0000n</b>
</a><br />
Contributor
</td>
</tr>
</table>

---

## License
MIT License — see `LICENSE` file.

---

## 🔥 Author

**Leo Desktop Assistant** was crafted by **NASHEDI_X_CODER**.  
Updated by **demo0000n**.