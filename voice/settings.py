"""
VoiceSettings — Runtime voice configuration for Leo.

Provides runtime-configurable voice parameters that can be
changed without restarting the assistant. All settings are
initialized from .env but can be modified at runtime.

Settings:
  TTS_ENGINE: TTS engine to use (kokoro, xtts, piper, pyttsx3, auto)
  TTS_VOICE: Voice identifier for the selected engine
  TTS_RATE: Speech rate (words per minute, default 145)
  TTS_VOLUME: Volume level (0.0 to 1.0, default 1.0)
  TTS_DEVICE: Compute device (auto, cpu, cuda)
  TTS_CACHE: Enable audio caching (true/false)
  TTS_STREAMING: Enable streaming playback (true/false)
  VOICE_PITCH: Voice pitch (0.5 to 2.0, default 1.0)
  WAKE_WORD: Wake word phrase
  WAKE_PHRASE: Phrase used for wake detection ("hello leo")
  WAKE_MODEL: Path to a local ONNX wake model (optional)
  WAKE_SENSITIVITY: Wake word detection sensitivity (0.0 to 1.0)
  LANG_CODE: Language code for STT
  STT_TIMEOUT: Max seconds to wait for speech
  STT_PHRASE_LIMIT: Max seconds per phrase
  TTS_BACKEND: Auto-detected or forced audio backend
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceSettings:
    """Runtime voice configuration. Thread-safe value object."""

    # TTS engine parameters
    tts_engine: str = "auto"       # kokoro, xtts, piper, pyttsx3, auto
    tts_voice: str = "default"     # Voice identifier for the selected engine
    tts_rate: int = 145            # Words per minute
    tts_volume: float = 0.7        # 0.0 to 1.0 (reduced to prevent echo feedback)
    tts_device: str = "auto"       # auto, cpu, cuda
    tts_cache: bool = True         # Cache generated audio
    tts_streaming: bool = True     # Enable streaming playback
    tts_cache_dir: str = "data/tts_cache"  # Cache directory
    tts_speaker_wav: str = ""      # Reference speaker WAV for voice cloning

    # Legacy (deprecated, kept for backward compatibility)
    voice_rate: int = 145          # Alias for tts_rate
    voice_volume: float = 1.0      # Alias for tts_volume
    voice_pitch: float = 1.0       # 0.5 to 2.0
    voice_id: str = "default"      # Legacy voice ID

    # Wake word
    wake_word: str = "leo"
    wake_phrase: str = "hello leo"
    wake_model: Optional[str] = None  # Path to local ONNX model
    wake_sensitivity: float = 0.5  # 0.0 to 1.0

    # STT
    lang_code: str = "en-IN"
    stt_timeout: float = 3.0       # Seconds to wait for speech start
    stt_phrase_limit: float = 7.0  # Max seconds per phrase

    # Audio backend (auto-detected)
    tts_backend: str = "auto"  # auto, alsa, pulseaudio, pipewire, jack

    # Streaming TTS engines
    kokoro_voice: str = "af_heart"   # Kokoro voice id
    piper_model: Optional[str] = None  # Path to Piper .onnx voice model

    # Microphone
    device_index: Optional[int] = None
    calibration_duration: float = 1.5

    # Conversational endpointing (milliseconds)
    conv_min_pause_ms: int = 600       # pauses shorter than this don't end a turn
    conv_endpoint_ms: int = 900        # trailing silence that finalizes a turn
    conv_timeout_s: float = 45.0       # idle time before returning to wake mode
    conv_interrupt_min_ms: int = 90    # sustained speech to trigger interruption

    def update_from_env(self) -> None:
        """Update settings from environment variables (called once at startup)."""
        import os

        if os.getenv("TTS_ENGINE"):
            self.tts_engine = os.getenv("TTS_ENGINE", "auto")
        if os.getenv("TTS_VOICE"):
            self.tts_voice = os.getenv("TTS_VOICE", "default")
        if os.getenv("TTS_RATE"):
            try:
                self.tts_rate = int(os.getenv("TTS_RATE", "145"))
                self.voice_rate = self.tts_rate  # Sync legacy field
            except ValueError:
                pass
        if os.getenv("TTS_VOLUME"):
            try:
                self.tts_volume = float(os.getenv("TTS_VOLUME", "1.0"))
                self.voice_volume = self.tts_volume
            except ValueError:
                pass
        if os.getenv("TTS_DEVICE"):
            self.tts_device = os.getenv("TTS_DEVICE", "auto")
        if os.getenv("TTS_CACHE"):
            self.tts_cache = os.getenv("TTS_CACHE", "true").lower() == "true"
        if os.getenv("TTS_STREAMING"):
            self.tts_streaming = os.getenv("TTS_STREAMING", "true").lower() == "true"
        if os.getenv("TTS_CACHE_DIR"):
            self.tts_cache_dir = os.getenv("TTS_CACHE_DIR", "data/tts_cache")
        if os.getenv("TTS_SPEAKER_WAV"):
            self.tts_speaker_wav = os.getenv("TTS_SPEAKER_WAV", "")

        # Legacy env vars (backward compatibility)
        if not os.getenv("TTS_RATE") and os.getenv("VOICE_RATE"):
            try:
                self.tts_rate = int(os.getenv("VOICE_RATE", "145"))
                self.voice_rate = self.tts_rate
            except ValueError:
                pass
        if not os.getenv("TTS_VOLUME") and os.getenv("VOICE_VOLUME"):
            try:
                self.tts_volume = float(os.getenv("VOICE_VOLUME", "1.0"))
                self.voice_volume = self.tts_volume
            except ValueError:
                pass
        if os.getenv("VOICE_ID"):
            self.voice_id = os.getenv("VOICE_ID", "default")
        if os.getenv("WAKE_WORD"):
            self.wake_word = os.getenv("WAKE_WORD", "leo")
        if os.getenv("WAKE_PHRASE"):
            self.wake_phrase = os.getenv("WAKE_PHRASE", "hello leo")
        if os.getenv("WAKE_MODEL"):
            self.wake_model = os.getenv("WAKE_MODEL")
        if os.getenv("LANG_CODE"):
            self.lang_code = os.getenv("LANG_CODE", "en-IN")
        if os.getenv("WAKE_DEVICE_INDEX"):
            try:
                self.device_index = int(os.getenv("WAKE_DEVICE_INDEX", ""))
            except (ValueError, TypeError):
                pass

    def to_dict(self) -> dict:
        """Return settings as a dictionary (for logging/serialization)."""
        return {
            "tts_engine": self.tts_engine,
            "tts_voice": self.tts_voice,
            "tts_rate": self.tts_rate,
            "tts_volume": self.tts_volume,
            "tts_device": self.tts_device,
            "tts_cache": self.tts_cache,
            "tts_streaming": self.tts_streaming,
            "voice_rate": self.voice_rate,
            "voice_volume": self.voice_volume,
            "voice_pitch": self.voice_pitch,
            "voice_id": self.voice_id,
            "wake_word": self.wake_word,
            "wake_phrase": self.wake_phrase,
            "wake_model": self.wake_model,
            "wake_sensitivity": self.wake_sensitivity,
            "lang_code": self.lang_code,
            "stt_timeout": self.stt_timeout,
            "stt_phrase_limit": self.stt_phrase_limit,
            "tts_backend": self.tts_backend,
            "device_index": self.device_index,
        }


# Global singleton
voice_settings = VoiceSettings()
