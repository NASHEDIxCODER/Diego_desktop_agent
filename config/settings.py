"""
Centralized configuration using Pydantic Settings + python-dotenv.

All credentials, API keys, and tunables are read from environment variables
or a .env file. NEVER hardcode secrets here.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM Providers ──────────────────────────────────────────────
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: Optional[str] = None  # Auto-detected if not set

    # ── LLM residency & warm-up ────────────────────────────────────
    # How long Ollama keeps the model in memory after the last request.
    # "0" unloads after EVERY request (cold start every turn — very slow).
    # "-1" keeps it forever. A duration like "10m" is the sane default.
    OLLAMA_KEEP_ALIVE: str = "10m"
    # Warm the LLM at startup (async, non-blocking, fail-safe) so the
    # model is loaded BEFORE the first user command.
    LLM_WARMUP_ENABLED: bool = True
    LLM_WARMUP_TIMEOUT_S: float = 120.0
    # Also pre-load the (heavy) vision model at startup. Off by default.
    LLM_WARMUP_VISION: bool = False

    # ── Telegram ───────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    TELEGRAM_API_ID: Optional[int] = None
    TELEGRAM_API_HASH: Optional[str] = None
    TELEGRAM_SESSION: str = "Diego_telegram"

    # ── Google / Firebase ──────────────────────────────────────────
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    FIREBASE_SERVICE_ACCOUNT_PATH: str = "auth/serviceAccountKey.json"

    # ── Weather ────────────────────────────────────────────────────
    OPENWEATHER_API_KEY: Optional[str] = None

    # ── Speech / TTS ───────────────────────────────────────────────
    WHISPER_MODEL: str = "base"
    COQUI_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    WAKE_WORD: str = "diego"
    WAKE_PHRASE: str = "hello diego"
    # Path to a local ONNX wake model. If set, this model is loaded.
    # If unset/empty, the bundled openWakeWord model is used.
    WAKE_MODEL: Optional[str] = None
    LANG_CODE: str = "en-IN"
    WAKE_DEVICE_INDEX: Optional[int] = None
    TTS_SPEED: float = 0.92
    TTS_VOLUME: float = 1.0

    # ── Logging ────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Database ───────────────────────────────────────────────────
    DUCKDB_PATH: str = "data/Diego.duckdb"
    DUCKDB_RETRY_MAX_ATTEMPTS: int = 3
    DUCKDB_RETRY_BASE_DELAY: float = 0.5

    # ── Training ─────────────────────────────────────────────────
    TRAIN_TIMEOUT_SECONDS: int = 600

    # ── NLP ────────────────────────────────────────────────────────
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.75
    EMBEDDING_DIM: int = 384
    MODEL_VERSION: str = "2.0.0"

    # ── Local Knowledge Index (read-only PC document indexing) ─────
    # Allowlist: user-approved scan roots. Extend via env/.env
    # (comma-separated). Never defaults to sensitive locations.
    KNOWLEDGE_SCAN_ROOTS: str = (
        "~/Documents,~/Desktop,~/Downloads,~/Projects"
    )
    # Denylist: sensitive/system locations that are NEVER scanned even
    # if they appear inside an approved root (e.g. ~/Projects/.venv).
    KNOWLEDGE_DENYLIST: str = (
        "~/.ssh,~/.gnupg,~/.config,~/.cache,~/.local/share/keyrings,"
        "~/.password-store,~/.aws,~/.kube,~/.docker,"
        ".git,.venv,venv,node_modules,__pycache__,.env,"
        "credentials,.mozilla,.thunderbird,.config/google-chrome,"
        ".config/chromium,/proc,/sys,/dev,/run,/var/run"
    )
    # File size limit for extraction (bytes). Larger files: metadata only.
    KNOWLEDGE_MAX_FILE_SIZE: int = 20 * 1024 * 1024
    # Per-file extraction timeout (seconds).
    KNOWLEDGE_EXTRACTION_TIMEOUT_S: float = 30.0
    # Bounded concurrency for background indexing.
    KNOWLEDGE_MAX_WORKERS: int = 2
    # Chunking parameters (deterministic).
    KNOWLEDGE_CHUNK_SIZE: int = 900
    KNOWLEDGE_CHUNK_OVERLAP: int = 120
    # Embedding backend override (defaults to the existing local
    # sentence-transformers model from MODEL_NAME — no cloud API).
    KNOWLEDGE_EMBEDDING_BACKEND: str = "local_sentence_transformers"
    # Periodic snapshot refresh (seconds; 0 disables).
    KNOWLEDGE_SNAPSHOT_REFRESH_S: int = 3600
    # Periodic incremental rescan (seconds; 0 disables). Keeps the local
    # index continuous: new/changed/deleted files under the approved
    # roots are picked up automatically without blocking anything.
    KNOWLEDGE_RESCAN_INTERVAL_S: float = 300.0

    # ── Paths ──────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    MODELS_WAKE_DIR: Path = BASE_DIR / "models" / "wake"
    VOICE_FILE: Path = BASE_DIR / "Diego.wav"
    IMAGES_DIR: Path = BASE_DIR / "auth" / "images"
    KNOWN_ENCODINGS_PATH: Path = BASE_DIR / "auth" / "Known_encodings.p"
    CLASSIFIER_PATH: Path = BASE_DIR / "models" / "intent_classifier.pkl"
    METADATA_PATH: Path = BASE_DIR / "models" / "metadata.json"
    EMBEDDING_CACHE_PATH: Path = BASE_DIR / "data" / "embedding_cache.pkl"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_WAKE_DIR.mkdir(parents=True, exist_ok=True)