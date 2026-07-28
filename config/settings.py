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

    # ── Telegram ───────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    TELEGRAM_API_ID: Optional[int] = None
    TELEGRAM_API_HASH: Optional[str] = None
    TELEGRAM_SESSION: str = "leo_telegram"

    # ── Google / Firebase ──────────────────────────────────────────
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    FIREBASE_SERVICE_ACCOUNT_PATH: str = "auth/serviceAccountKey.json"

    # ── Weather ────────────────────────────────────────────────────
    OPENWEATHER_API_KEY: Optional[str] = None

    # ── Speech / TTS ───────────────────────────────────────────────
    WHISPER_MODEL: str = "base"
    COQUI_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    WAKE_WORD: str = "leo"
    LANG_CODE: str = "en-IN"
    WAKE_DEVICE_INDEX: Optional[int] = None
    TTS_SPEED: float = 0.92
    TTS_VOLUME: float = 1.0

    # ── Logging ────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Database ───────────────────────────────────────────────────
    DUCKDB_PATH: str = "data/leo.duckdb"
    DUCKDB_RETRY_MAX_ATTEMPTS: int = 3
    DUCKDB_RETRY_BASE_DELAY: float = 0.5

    # ── Training ─────────────────────────────────────────────────
    TRAIN_TIMEOUT_SECONDS: int = 600

    # ── NLP ────────────────────────────────────────────────────────
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.75
    EMBEDDING_DIM: int = 384
    MODEL_VERSION: str = "2.0.0"

    # ── Paths ──────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    VOICE_FILE: Path = BASE_DIR / "leo.wav"
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