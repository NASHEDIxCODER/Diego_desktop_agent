<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Linux-orange.svg" alt="Linux">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Active">
</p>

<h1 align="center">🦁 Diego Desktop Assistant</h1>

<p align="center">
  <em>A fully autonomous, conversational AI companion for Linux — wake-word activated, face-authenticated, full-duplex streaming voice, persistent memory, screen vision, and automatic desktop control.</em>
</p>

<p align="center">
  <strong>Talk naturally. Interrupt any time. Diego listens, thinks, speaks, and acts — all at once.</strong>
</p>

---

## 📖 Table of Contents

- [What is Diego?](#-what-is-Diego)
- [Why Diego Exists](#-why-Diego-exists)
- [Core Architecture](#-core-architecture)
- [Streaming Pipeline](#-streaming-pipeline)
- [Features](#-features)
- [Tech Stack & Rationale](#-tech-stack--rationale)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Voice Commands](#-voice-commands)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🧠 What is Diego?

Diego is a **production-grade conversational desktop agent** for Linux. Unlike traditional voice assistants that merely execute commands, Diego is a **companion** — it maintains context across conversations, remembers facts about you, sees your screen, controls your desktop, and speaks with a natural, varied personality.

Diego runs entirely on your machine. The core pipeline (wake word → speech recognition → intent understanding → action execution → speech synthesis) is **local-first**. Cloud LLMs (Gemini, OpenAI, Ollama) are used only as optional fallbacks for complex reasoning.

### The Philosophy

| Principle | What it means |
|-----------|---------------|
| **Conversation First** | Diego is a companion, not a command executor. Greetings, thanks, and small talk never hit the planner or LLM. |
| **Full Duplex** | You can interrupt Diego mid-sentence. Just start talking — TTS aborts instantly and Diego listens. |
| **Local First** | Everything critical runs offline. Wake word, STT, TTS, intent classification, face auth — all local. |
| **Autonomous** | Diego acts on its own. "Open VS Code", "search GitHub", "play Spotify" — no confirmation prompts. |
| **Always Alive** | Diego boots once and stays alive forever. It sleeps after a goodbye or ~60s of silence, then waits for the wake word again. |
| **Memory** | Rolling conversation context + long-term fact storage + automatic summarization. "Remember my project is Diego" … later "what was my project called?" → answered instantly. |

---

## 🎯 Why Diego Exists

Existing voice assistants (Siri, Alexa, Google Assistant) are:

- **Cloud-dependent** — your voice leaves your machine
- **Stateless** — no memory between sessions
- **Command-oriented** — not conversational
- **Closed** — can't control your desktop or see your screen
- **Generic** — same personality for everyone

Diego was built to be:

- **Private** — everything critical runs locally
- **Stateful** — remembers your projects, preferences, and conversations
- **Conversational** — natural back-and-forth, not "say a command, get a response"
- **Desktop-native** — sees your screen, controls your apps, automates your workflow
- **Personal** — varied, alive personality that feels like a companion

---

## 🏗 Core Architecture

Diego uses a **layered, event-driven architecture** with clear separation of concerns. Every spoken command flows through a single pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DIEGO ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  WAKE    │   │  FACE    │   │ STREAMING│   │ STREAMING│    │
│  │  WORD    │──▶│  AUTH    │──▶│   VAD    │──▶│  WHISPER │    │
│  │(openWake │   │(OpenCV + │   │ (Silero) │   │ (faster- │    │
│  │  Word)   │   │ face_rec)│   │          │   │ whisper) │    │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘    │
│                                                     │          │
│  ┌──────────────────────────────────────────────────┘          │
│  │                                                             │
│  │   ┌──────────────────────────────────────────────┐          │
│  │   │              AGENT BRAIN                      │          │
│  │   │  perceive → decide → plan → dispatch →       │          │
│  │   │  verify → learn → respond                    │          │
│  │   └──────────────────────────────────────────────┘          │
│  │         │           │           │           │               │
│  │         ▼           ▼           ▼           ▼               │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│  │   │PERCEPTION│ │DECISION │ │ PLANNER │ │DISPATCH │         │
│  │   │(screen + │ │ ENGINE  │ │ (LLM    │ │ (desktop│         │
│  │   │ a11y +   │ │(routing)│ │  plans) │ │ actions)│         │
│  │   │ OCR)     │ │         │ │         │ │         │         │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘         │
│  │                                                             │
│  │   ┌──────────────────────────────────────────────┐          │
│  │   │           STREAMING TTS (Kokoro → XTTS →     │          │
│  │   │           Piper → pyttsx3)                    │          │
│  │   └──────────────────────────────────────────────┘          │
│  │                                                             │
│  │   ┌──────────────────────────────────────────────┐          │
│  │   │  MEMORY LAYER                                │          │
│  │   │  ConversationMemory + UnifiedMemory +        │          │
│  │   │  ExperienceDB + ContextComposer              │          │
│  │   └──────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Subsystems

| Subsystem | Module(s) | Role |
|-----------|-----------|------|
| **Wake Word** | `voice/wake_word.py`, `voice/wake_listener.py` | Always-listening wake word detection ("Diego", "hey Diego", "hello Diego") using openWakeWord + ONNX |
| **Face Auth** | `auth/robust_auth.py`, `auth/live_auth.py` | Multi-frame voting, confidence averaging, head-pose detection, anti-spoofing. Camera opens ONLY after wake word. |
| **Streaming STT** | `voice/streaming_stt.py` | Silero VAD + faster-whisper with endpointing and interruption detection |
| **Streaming TTS** | `voice/streaming_tts.py` | Interruptible sentence-streamed TTS with provider chain: Kokoro → XTTS → Piper → pyttsx3 |
| **Agent Brain** | `agent/brain.py` | Central orchestrator: perceive → decide → plan → dispatch → verify → learn → respond |
| **Decision Engine** | `core/decision_engine.py` | Routes commands: conversation, cached, direct action, or LLM fallback |
| **Planner** | `agent/planner.py` | LLM-powered task decomposition for complex multi-step goals |
| **Action Dispatcher** | `agent/action_dispatcher.py` | Maps LLM actions → real desktop operations (open apps, browser control, system commands) |
| **Action Verifier** | `vision/action_verifier.py` | Vision-based verification: did the action actually work? Compares pre/post screen state. |
| **Perception** | `services/perception_pipeline.py`, `services/vision_service.py` | Screen capture (MSS) + OCR (Tesseract) + accessibility tree + UI element detection |
| **Conversation Memory** | `agent/conversation_memory.py` | Rolling context + long-term facts + auto-summarization |
| **Personality** | `agent/personality.py` | Varied, alive conversational phrasing — never robotic |
| **Streaming LLM** | `agent/streaming_llm.py` | Token-streaming LLM → sentence segmentation → ACTION extraction |
| **Context Composer** | `agent/context_composer.py` | Smart, ranked, compact memory injection for LLM prompts |
| **Learning Engine** | `learning/learning_engine.py` | Records action outcomes, builds experience DB, improves over time |
| **Search Service** | `services/search_service.py` | DuckDuckGo/Tavily web search + trafilatura content extraction |
| **Music Agent** | `services/music_agent.py` | Unified music control (MPV, Spotify, YouTube, local files) |
| **Screen Capture** | `services/screen_capture.py` | Ultra-fast screen capture via MSS (<20ms) |
| **NLP Pipeline** | `nlp/` | Local intent classification (sentence-transformers + fuzzy matching), entity extraction (spaCy), command normalization |
| **Plugin System** | `core/plugin_base.py`, `core/plugin_manager.py`, `plugins/` | Event-driven plugin architecture for extensibility |
| **Event Bus** | `core/event_bus.py` | Async pub/sub decoupling all components |
| **Persistent Storage** | `memory/duckdb_store.py`, `memory/unified_memory.py` | DuckDB for structured data, semantic memory for embeddings |

---

## 🔄 Streaming Pipeline

Everything streams concurrently — no blocking between stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FULL-DUPLEX STREAMING PIPELINE                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Wake ("Diego" / "hey Diego" / "hello Diego")                             │
│    │                                                                │
│    ├──▶ Transcript verification (fuzzy + phonetic)                  │
│    │                                                                │
│    ├──▶ Face auth (camera opens ONLY here, waits forever)           │
│    │                                                                │
│    ▼                                                                │
│  Streaming audio (shared ring buffer)                               │
│    │                                                                │
│    ├──▶ Streaming VAD (Silero) ─────────────────────┐               │
│    │                                                │               │
│    ├──▶ Streaming Whisper (faster-whisper)          │  FULL DUPLEX  │
│    │                                                │               │
│    ├──▶ Streaming LLM (Ollama, token stream)        │  speak while  │
│    │                                                │  Diego talks    │
│    ├──▶ Sentence-by-sentence generation             │  → instant    │
│    │                                                │  TTS abort    │
│    ├──▶ Streaming TTS (Kokoro → XTTS → Piper) ─────┘               │
│    │                                                                │
│    ▼                                                                │
│  Interruptible audio playback (sounddevice)                         │
│    │                                                                │
│    ├──▶ User interrupts → TTS aborts → resume listening             │
│    │                                                                │
│    └──▶ 60s silence / "goodbye" / "stop listening" → back to wake   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What Full Duplex Feels Like

- **Interrupt Diego mid-sentence** — just start talking. TTS aborts instantly and Diego listens.
- **Natural pauses** — pausing < 600ms doesn't cut you off.
- **Filler words** — "umm", "wait", "hold on", "actually" keep your turn open.
- **Stay awake** — after the wake word, Diego stays in conversation. No need to repeat "hey Diego" every time.
- **Alive personality** — varied greetings ("Hey.", "Welcome back.", "Good morning."), never "How may I assist you?".

---

## ✨ Features

### 🎙️ Voice & Audio

| Feature | Implementation |
|---------|---------------|
| Wake Word | openWakeWord + ONNX runtime — "Diego", "hey Diego", "hello Diego" |
| Wake Verification | Fuzzy + phonetic matching (RapidFuzz + Jellyfish metaphone) |
| Speech-to-Text | faster-whisper (streaming partials + finals) with Silero VAD endpointing |
| Text-to-Speech | Provider chain: Kokoro-82M → Coqui XTTS → Piper → pyttsx3 |
| Audio I/O | sounddevice full-duplex ring buffer |
| Audio Processing | Noise floor calibration, auto-gain, VAD pre-filtering |
| ASR Fallback | Sherpa-ONNX, Nemotron, Google Web Speech |

### 🔐 Security & Authentication

| Feature | Implementation |
|---------|---------------|
| Face Recognition | OpenCV + face_recognition + dlib |
| Robust Auth | Multi-frame voting, confidence averaging, head-pose detection |
| Anti-Spoofing | Liveness detection, blink detection |
| Deferred Auth | Camera opens ONLY after wake word, never at startup |
| Firebase Backup | Optional cloud backup of face encodings |

### 🧠 Intelligence

| Feature | Implementation |
|---------|---------------|
| Intent Classification | 3-tier: sentence-transformers embeddings → centroids → RapidFuzz fuzzy matching |
| Entity Extraction | spaCy NER + regex patterns |
| Command Normalization | App aliases, verb canonicalization, noise removal, follow-up detection |
| Conversation Memory | Rolling context window + long-term fact extraction + auto-summarization |
| Personality Engine | Varied, context-aware responses — never robotic |
| Context Composer | Smart, ranked memory injection for LLM prompts |
| Learning Engine | Records action outcomes, builds experience database |
| Goal Management | Multi-session goal tracking with task DAG decomposition |

### 🖥️ Desktop Automation

| Feature | Implementation |
|---------|---------------|
| App Control | Open, close, focus any desktop application |
| Browser Control | Navigate, search, click, type, scroll |
| System Control | Volume, brightness, lock screen, shutdown |
| Screen Vision | MSS capture + Tesseract OCR + accessibility tree + UI element detection |
| Action Verification | Pre/post screen comparison + OS-level process verification (pgrep) |
| Music Control | Unified MPV + Spotify + YouTube + local files |

### 🔌 Extensibility

| Feature | Implementation |
|---------|---------------|
| Plugin System | Event-driven, async EventBus, BasePlugin ABC |
| Built-in Plugins | YouTube, Telegram, Brightness |
| Tool Registry | Dynamic tool discovery and registration |
| LLM Providers | Gemini, OpenAI, Ollama (auto-fallback) |

### 💾 Storage & Memory

| Feature | Implementation |
|---------|---------------|
| Structured DB | DuckDB (intents, examples, embeddings, entities, synonyms, history, preferences) |
| Semantic Memory | Embedding-based similarity search |
| Unified Memory | Single API across DuckDB + semantic + conversation stores |
| Experience DB | Action outcome recording for learning |

---

## 🔧 Tech Stack & Rationale

Every technology choice in Diego is deliberate. Here's what we use and **why**:

### Core Runtime

| Technology | Why We Use It |
|-----------|---------------|
| **Python 3.11+** | Async/await maturity, extensive ML/AI ecosystem, rapid prototyping. Python's asyncio enables the full-duplex streaming pipeline without complex threading. |
| **asyncio** | First-class async I/O for concurrent audio streaming, LLM token processing, and TTS playback — all without blocking. |

### Voice Pipeline

| Technology | Why We Use It |
|-----------|---------------|
| **openWakeWord** | Offline, lightweight wake word detection. Runs entirely on CPU with ONNX. No cloud dependency. Models are ~1MB each. |
| **ONNX Runtime** | Cross-platform inference backend. Runs openWakeWord models efficiently on CPU without a GPU. |
| **faster-whisper** | CTranslate2-based Whisper implementation. 4x faster than original Whisper, supports streaming partial transcripts. Critical for real-time conversation. |
| **Silero VAD** | State-of-the-art voice activity detection. Tiny model (~1MB), runs in real-time, accurately detects speech boundaries for endpointing. |
| **Kokoro-82M** | Primary TTS engine. Only 82M parameters but produces natural, expressive speech. Fast enough for streaming sentence-by-sentence. |
| **Coqui XTTS** | Fallback TTS. High-quality voice cloning and multi-language support. |
| **Piper TTS** | Lightweight fallback. Runs on CPU, fast inference, multiple voices. |
| **pyttsx3** | Last-resort fallback. Works offline on any system, zero dependencies beyond espeak. |
| **sounddevice** | Full-duplex audio I/O. Enables simultaneous recording and playback — critical for interruption support. |
| **RapidFuzz** | Fast fuzzy string matching for wake word verification and intent classification fallback. |
| **Jellyfish** | Phonetic matching (metaphone) for wake word verification. Handles mispronunciations gracefully. |

### Intelligence & NLP

| Technology | Why We Use It |
|-----------|---------------|
| **sentence-transformers (all-MiniLM-L6-v2)** | 384-dimensional embeddings. Excellent semantic similarity at only 80MB. Runs locally, no API calls needed for intent matching. |
| **spaCy (en_core_web_sm)** | Fast, production-ready NER. Extracts person names, locations, organizations from commands. |
| **Ollama** | Local LLM inference. Runs models like Llama 3, Mistral, Gemma entirely offline. No API keys, no latency, no privacy concerns. |
| **Google Gemini** | Primary cloud LLM fallback. Strong reasoning, large context window, good tool-use capabilities. |
| **OpenAI** | Secondary cloud LLM fallback. GPT-4-level reasoning for complex tasks. |

### Vision & Desktop

| Technology | Why We Use It |
|-----------|---------------|
| **MSS** | Ultra-fast screen capture. C extension, <20ms per frame. Critical for real-time vision verification. |
| **Tesseract (pytesseract)** | Offline OCR. Extracts text from screen for context-aware responses. |
| **OpenCV** | Industry-standard computer vision. Face detection, image processing, frame differencing. |
| **face_recognition** | High-accuracy face recognition built on dlib. 99.38% accuracy on LFW benchmark. |
| **dlib** | C++ ML library. Provides the HOG face detector and face landmark detection used by face_recognition. |
| **PyAutoGUI** | Cross-platform GUI automation. Keyboard, mouse, screen control. |
| **Selenium** | Browser automation for YouTube and web-based plugins. |

### Storage & Data

| Technology | Why We Use It |
|-----------|---------------|
| **DuckDB** | Embedded analytical database. Zero-config, single-file, SQL-compatible. Perfect for local intent/entity/context storage. Faster than SQLite for analytical queries. |
| **NumPy** | Foundation for all numerical computation. Audio processing, embedding operations, signal analysis. |
| **scikit-learn** | Cosine similarity for embeddings, centroid computation for intent classification. |

### Infrastructure

| Technology | Why We Use It |
|-----------|---------------|
| **Pydantic Settings** | Type-safe configuration management. Validates all settings at startup, reads from .env. Prevents runtime config errors. |
| **aiohttp** | Async HTTP client. Used for web search, LLM API calls, and content fetching — all non-blocking. |
| **trafilatura** | High-quality HTML content extraction. Strips boilerplate from web pages for clean LLM context. |
| **BeautifulSoup4 + lxml** | HTML parsing for search results and content extraction. |
| **Firebase Admin** | Optional cloud backup for face encodings. |
| **Telethon** | Telegram client for the Telegram plugin. |

---

## 📁 Project Structure

```
Diego_desktop_agent/
│
├── Diego.py                          # 🚀 Conversational entry point (primary)
├── main.py                         # 🔧 Utility entry point (train, benchmark, debug)
├── compat.py                       # 🐍 Python 3.14 compatibility stubs
├── requirements.txt                # 📦 Dependencies
├── .gitignore                      # 🙈 Git ignore rules
├── LICENSE                         # 📄 MIT License
│
├── agent/                          # 🧠 Agent intelligence layer
│   ├── brain.py                    #    Central orchestrator (perceive→decide→plan→dispatch→verify→learn→respond)
│   ├── planner.py                  #    LLM-powered task decomposition
│   ├── action_dispatcher.py        #    Maps LLM actions → real desktop operations
│   ├── streaming_llm.py            #    Token-streaming LLM with sentence segmentation
│   ├── conversation_memory.py      #    Rolling context + long-term facts + auto-summarization
│   ├── personality.py              #    Varied, alive conversational phrasing
│   ├── context_composer.py         #    Smart, ranked memory injection for LLM prompts
│   ├── memory.py                   #    Memory abstractions
│   ├── executor.py                 #    Action execution helpers
│   ├── goal_manager.py             #    Multi-session goal tracking
│   ├── proactive_agent.py          #    Proactive suggestion engine
│   ├── project_mode.py             #    Project-aware context switching
│   ├── task_executor.py            #    Task execution helpers
│   ├── browser.py                  #    Browser automation
│   └── code_assistant.py           #    Code-aware assistance
│
├── voice/                          # 🎙️ Voice pipeline
│   ├── wake_word.py                #    openWakeWord detection
│   ├── wake_listener.py            #    Wake word listener loop
│   ├── wake_model_manager.py       #    Wake model download & management
│   ├── streaming_stt.py            #    Streaming Silero VAD + faster-whisper
│   ├── streaming_tts.py            #    Interruptible sentence-streamed TTS
│   ├── vad.py                      #    Voice activity detection
│   ├── audio_manager.py            #    Full-duplex audio ring buffer
│   ├── audio_processing.py         #    Noise floor, auto-gain, pre-filtering
│   ├── command_listener.py         #    Legacy command listener
│   ├── asr_provider.py             #    ASR provider abstraction
│   ├── asr_fallback.py             #    ASR fallback chain
│   ├── settings.py                 #    Voice-specific settings
│   ├── providers/                  #    ASR provider implementations
│   │   ├── whisper_provider.py     #       faster-whisper provider
│   │   ├── sherpa_base.py          #       Sherpa-ONNX base
│   │   ├── sherpa_providers.py     #       Sherpa-ONNX providers
│   │   └── nemotron_provider.py    #       Nemotron provider
│   └── tts/                        #    TTS engine implementations
│       ├── base.py                 #       TTS base class
│       ├── kokoro_engine.py        #       Kokoro-82M engine
│       └── manager.py              #       TTS provider manager
│
├── core/                           # ⚙️ Core infrastructure
│   ├── conversation_engine.py      #    Full-duplex orchestrator (wake/conversation lifecycle)
│   ├── decision_engine.py          #    Command routing (conversation/cached/direct/LLM)
│   ├── event_bus.py                #    Async pub/sub event system
│   ├── plugin_base.py              #    BasePlugin abstract class
│   ├── plugin_manager.py           #    Plugin discovery & lifecycle
│   ├── command_router.py           #    Command classification & routing
│   ├── tool_registry.py            #    Dynamic tool discovery
│   ├── tool_reliability.py         #    Tool reliability tracking
│   ├── state_machine.py            #    Application state machine
│   ├── service.py                  #    Service base class
│   ├── background_agent.py         #    Background task agent
│   ├── background_workers.py       #    Background worker pool
│   ├── background_learning.py      #    Idle-time self-improvement
│   ├── autonomous_reasoning.py     #    Autonomous reasoning engine
│   ├── cache_manager.py            #    Response caching
│   ├── metrics.py                  #    Performance metrics
│   ├── response_guarantee.py       #    Response delivery guarantee
│   ├── startup_health.py           #    Startup health checks
│   ├── gui_dispatcher.py           #    GUI event dispatching
│   ├── benchmark.py                #    Benchmarking utilities
│   └── manual_session_recorder.py  #    Session recording for debugging
│
├── vision/                         # 👁️ Computer vision
│   ├── action_verifier.py          #    Pre/post screen comparison for action verification
│   ├── perception.py               #    Visual perception
│   ├── ocr_pipeline.py             #    OCR pipeline (Tesseract)
│   ├── layout_analyzer.py          #    UI layout analysis
│   ├── screen_memory.py            #    Screen state memory
│   ├── frame_differencer.py        #    Frame difference detection
│   ├── debug_overlay.py            #    Live debug overlay (green/red/white boxes)
│   └── forensic_logger.py          #    Forensic vision logging
│
├── services/                       # 🛠️ Application services
│   ├── perception_pipeline.py      #    Unified perception (screen + a11y + OCR)
│   ├── vision_service.py           #    Structured UI tree + OCR service
│   ├── screen_capture.py           #    Ultra-fast MSS screen capture
│   ├── screen_reasoning.py         #    Screen content reasoning
│   ├── search_service.py           #    DuckDuckGo/Tavily web search
│   ├── music_agent.py              #    Unified music control (MPV/Spotify/YouTube)
│   ├── desktop_observer.py         #    Desktop state observation
│   ├── desktop_state.py            #    Desktop state tracking
│   ├── accessibility.py            #    Accessibility tree integration
│   └── ui_tree.py                  #    UI element tree
│
├── nlp/                            # 📝 Natural Language Processing
│   ├── parser.py                   #    NLP pipeline orchestrator
│   ├── classifier.py               #    Intent classification (3-tier)
│   ├── embeddings.py               #    sentence-transformers embeddings
│   ├── entities.py                 #    Entity extraction (spaCy + regex)
│   ├── tokenizer.py                #    Text tokenization
│   ├── normalizer.py               #    Text normalization + synonyms
│   ├── command_normalizer.py       #    Command canonicalization
│   ├── context.py                  #    Conversation context
│   ├── conversation_state.py       #    Conversation state machine
│   ├── confidence.py               #    Confidence scoring
│   ├── inference.py                #    Model inference
│   ├── trainer.py                  #    Intent training data generator
│   ├── evaluator.py                #    Benchmarking & evaluation
│   └── model_metadata.py           #    Model metadata management
│
├── memory/                         # 💾 Persistent storage
│   ├── duckdb_store.py             #    DuckDB structured storage
│   ├── unified_memory.py           #    Unified memory API
│   └── semantic_memory.py          #    Embedding-based semantic search
│
├── learning/                       # 📚 Learning & adaptation
│   ├── learning_engine.py          #    Action outcome recording
│   ├── experience_db.py            #    Experience database
│   ├── habits.py                   #    User habit learning
│   ├── preferences.py              #    User preference learning
│   ├── user_profile.py             #    User profile management
│   ├── skill_memory.py             #    Skill memory
│   └── desktop_layouts.py          #    Desktop layout learning
│
├── auth/                           # 🔐 Face authentication
│   ├── robust_auth.py              #    Multi-frame voting, anti-spoofing
│   ├── live_auth.py                #    Live camera authentication
│   ├── faceauth.py                 #    Face authentication core
│   ├── face_detector.py            #    Face detection
│   ├── face_popup.py               #    Auth popup UI
│   ├── auth_service.py             #    Auth service
│   └── encode.py                   #    Face encoding utilities
│
├── ai/                             # 🤖 LLM integration
│   └── llm_client.py               #    Unified multi-provider LLM client
│
├── plugins/                        # 🔌 Plugin system
│   ├── youtube_plugin.py           #    YouTube control
│   ├── telegram_plugin.py          #    Telegram messaging
│   └── brightness_plugin.py        #    Screen brightness
│
├── config/                         # ⚙️ Configuration
│   └── settings.py                 #    Pydantic Settings (all config)
│
├── telemetry/                      # 📊 Logging & telemetry
│   └── logger.py                   #    Structured JSON logging
│
├── runtime/                        # 🖥️ Runtime UI
│   └── status_popup.py             #    Status popup overlay
│
├── tests/                          # 🧪 Test suite
│   ├── test_command_listener.py    #    Voice command tests
│   ├── test_runtime_stability.py   #    Runtime stability tests
│   ├── test_event_bus.py           #    Event bus tests
│   ├── test_nlp.py                 #    NLP pipeline tests
│   ├── test_duckdb_store.py        #    Storage tests
│   ├── test_e2e.py                 #    End-to-end tests
│   ├── test_integration_pipeline.py#    Integration tests
│   ├── test_response_guarantee.py  #    Response guarantee tests
│   ├── test_trainer.py             #    Trainer tests
│   ├── test_runtime.py             #    Runtime tests
│   └── test_production_regression.py # Production regression tests
│
├── debug/                          # 🔍 Diagnostic & debugging tools
│   ├── voice_diagnostic_mode.py    #    Voice pipeline diagnostics
│   ├── benchmark_asr_alternatives.py#   ASR provider benchmarking
│   ├── benchmark_worker.py         #    Benchmark worker
│   ├── benchmark_asr.py            #    ASR benchmarks
│   ├── benchmark_stt.py            #    STT benchmarks
│   ├── benchmark_commands.py       #    Command benchmarks
│   ├── asr_dataset.py              #    ASR dataset tools
│   ├── record_asr_dataset.py       #    ASR dataset recording
│   ├── audit_system.py             #    System audit
│   ├── audit_execution_pipeline.py #    Execution pipeline audit
│   ├── voice_pipeline_validation.py#    Voice pipeline validation
│   ├── diag_channels.py            #    Audio channel diagnostics
│   ├── diag_live_mic.py            #    Live microphone diagnostics
│   ├── diag_vad_pipeline.py        #    VAD pipeline diagnostics
│   ├── audio_diagnostics.py        #    Audio system diagnostics
│   ├── analyze_wake_fail.py        #    Wake failure analysis
│   ├── retrain_wake_verifier.py    #    Wake verifier retraining
│   ├── test_all_mics.py            #    Multi-microphone testing
│   ├── test_audio.py               #    Audio tests
│   ├── test_camera.py              #    Camera tests
│   ├── test_detector.py            #    Detector tests
│   ├── test_faceauth.py            #    Face auth tests
│   ├── test_gain_pipeline.py       #    Gain pipeline tests
│   ├── test_gui_dispatcher.py      #    GUI dispatcher tests
│   ├── test_microphone.py          #    Microphone tests
│   ├── test_perception_pipeline.py #    Perception pipeline tests
│   ├── test_state_machine.py       #    State machine tests
│   ├── test_wake.py                #    Wake word tests
│   ├── test_wake_offline.py        #    Offline wake tests
│   ├── test_wake_pipeline_acceptance.py # Wake pipeline acceptance tests
│   ├── vision_test.py              #    Vision tests
│   └── vision/                     #    Vision debug screenshots
│
├── scripts/                        # 📜 Legacy scripts (wrapped by plugins)
│   ├── brightness.py               #    Brightness control
│   ├── volume.py                   #    Volume control
│   ├── youtube.py                  #    YouTube control
│   ├── telegram_bot.py             #    Telegram bot
│   ├── mail.py                     #    Email
│   ├── conversation_llm.py         #    Conversation LLM
│   ├── fill_datasets.py            #    Dataset generation
│   ├── generate_datasets.py        #    Dataset generation
│   └── nlp_controller.py           #    NLP controller
│
├── docs/                           # 📖 Documentation
│   └── architecture.md             #    Architecture documentation
│
├── datasets/                       # 📊 Training datasets
│   └── intents/                    #    Per-intent training examples (JSON)
│
└── data/                           # 💿 Local data
    ├── knowledge_base.json         #    Knowledge base
    └── mic_selection.json          #    Microphone selection
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Linux with PulseAudio or PipeWire |
| **Python** | 3.11 or higher |
| **Microphone** | Any working microphone |
| **Webcam** | Required for face authentication (can skip with `--no-auth`) |
| **Browser** | Chrome/Chromium (for YouTube Selenium automation) |
| **Disk Space** | ~4GB for models (Whisper, Kokoro, sentence-transformers, openWakeWord) |

### 1. Clone & Setup

```bash
git clone https://github.com/NASHEDIxCODER/Diego_desktop_agent.git
cd Diego_desktop_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev python3-pyaudio espeak tesseract-ocr

# Fedora
sudo dnf install portaudio-devel python3-pyaudio espeak tesseract

# Arch
sudo pacman -S portaudio espeak tesseract
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (optional — Diego works fully offline)
```

### 4. Run

```bash
python Diego.py              # Start conversational Diego (recommended)
python Diego.py --no-auth    # Skip face auth (development only)
python Diego.py --status     # Show subsystem status
```

---

## 📋 Usage

### Primary Entry Point: `Diego.py`

```bash
python Diego.py                 # Start conversational Diego
python Diego.py --no-auth       # Skip face authentication
python Diego.py --status        # Check subsystem health
python Diego.py debug vision    # Live vision debug overlay
python Diego.py inspect screen  # Comprehensive screen inspection report
```

### Utility Entry Point: `main.py`

```bash
python main.py                # Same as `python Diego.py`
python main.py --train        # Train NLP model with generated examples
python main.py --status       # Show NLP model status
python main.py --benchmark    # Run NLP benchmarks
python main.py --select-mic   # Interactive microphone selection
python main.py --audio-debug  # Real-time audio level visualizer
python main.py --train-wake   # Record wake phrases + train custom verifier
python main.py --record-session  # Record conversation turns to JSON
```

### What Happens When You Run Diego

1. **Boot** — Environment fixes, Python 3.14 compatibility shims, logging setup
2. **Model Loading** — Wake word model, Whisper, Kokoro TTS, Silero VAD, sentence-transformers
3. **Audio Init** — Microphone calibration, noise floor measurement, ring buffer setup
4. **Wait for Wake** — Diego listens silently for "Diego", "hey Diego", or "hello Diego"
5. **Wake Detected** → Transcript verification (fuzzy + phonetic)
6. **Face Auth** — Camera opens, verifies your face (skipped with `--no-auth`)
7. **Conversation** — Full-duplex streaming: you talk, Diego listens, thinks, speaks
8. **Sleep** — After 60s of silence or "goodbye", Diego goes back to wake listening
9. **Diego NEVER exits on its own** — it stays alive until you press Ctrl+C

---

## 🗣️ Voice Commands

### Desktop Control

| Command | Action |
|---------|--------|
| "Open Firefox" / "Open VS Code" / "Open Terminal" | Launch any application |
| "Close Firefox" / "Close VS Code" | Close an application |
| "Search for Python tutorials" | Web search |
| "Open github.com" | Navigate to URL |
| "Click on Settings" / "Type hello world" | UI interaction |
| "Scroll down" / "Scroll up" | Page scrolling |

### Media & System

| Command | Action |
|---------|--------|
| "Volume up" / "Volume down" / "Mute" | Audio control |
| "Set volume to 50 percent" | Precise volume |
| "Brightness up" / "Brightness down" | Screen brightness |
| "Set brightness to 70" | Precise brightness |
| "Play music" / "Play [song] by [artist]" | Music playback |
| "Pause" / "Resume" / "Next song" | Playback control |
| "Lock screen" / "Shutdown" | System control |

### YouTube

| Command | Action |
|---------|--------|
| "Play [video] on YouTube" | Search and play |
| "Pause" / "Resume" / "Play" | Playback control |
| "Next song" / "Previous song" | Playlist navigation |
| "Volume up" / "Volume down" / "Mute" | YouTube volume |
| "Forward 10 seconds" / "Rewind" | Seek |
| "Speed up" / "Slow down" | Playback speed |
| "Close YouTube" | Close YouTube |

### Telegram

| Command | Action |
|---------|--------|
| "Send message to [contact] saying [message]" | Send message |
| "Read messages from [contact]" | Read messages |
| "Reply saying [message]" | Reply to last message |

### Conversation & Memory

| Command | Action |
|---------|--------|
| "Hello" / "Hey Diego" / "Good morning" | Greeting (varied response) |
| "How are you?" | Status check |
| "What time is it?" / "What's the date?" | Time/date query |
| "Tell me a joke" | Entertainment |
| "What can you do?" | Capability listing |
| "Remember my project is called Diego" | Store fact in long-term memory |
| "What was my project called?" | Recall from memory |
| "Goodbye" / "Stop listening" / "Cancel" | End conversation, return to wake |

---

## ⚙️ Configuration

All configuration is centralized in `config/settings.py` using **Pydantic Settings**, read from `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OLLAMA_BASE_URL` | Local Ollama URL | `http://localhost:11434` |
| `TELEGRAM_API_ID` | Telegram API ID | — |
| `TELEGRAM_API_HASH` | Telegram API hash | — |
| `LOG_LEVEL` | Log level | `INFO` |
| `WAKE_WORD` | Wake word phrase | `hello Diego` |
| `MODEL_NAME` | sentence-transformers model | `all-MiniLM-L6-v2` |
| `SIMILARITY_THRESHOLD` | Intent match threshold | `0.75` |
| `KNOWN_ENCODINGS_PATH` | Face encodings file | `auth/Known_encodings.p` |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_event_bus.py -v
pytest tests/test_nlp.py -v
pytest tests/test_command_listener.py -v
pytest tests/test_runtime_stability.py -v
pytest tests/test_e2e.py -v
```

---

## 👥 Contributors

<table>
<tr>
<td align="center">
<a href="https://github.com/NASHEDIxCODER">
<img src="https://avatars.githubusercontent.com/NASHEDIxCODER" width="100px;" alt="NASHEDIxCODER"/><br />
<b>NASHEDIxCODER</b>
</a><br />
Owner & Lead Developer
</td>
<td align="center">
<a href="https://github.com/dem0000n">
<img src="https://avatars.githubusercontent.com/dem0000n" width="100px;" alt="dem0000n"/><br />
<b>dem0000n</b>
</a><br />
Contributor
</td>
</tr>
</table>

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>🦁 Diego Desktop Assistant</strong> — Crafted with ❤️ by <a href="https://github.com/NASHEDIxCODER">NASHEDI_X_CODER</a>
</p>