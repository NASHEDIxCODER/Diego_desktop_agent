"""
VoiceSettings — Runtime voice configuration for Leo.

Provides runtime-configurable voice parameters that can be
changed without restarting the assistant. All settings are
initialized from .env but can be modified at runtime.

Settings:
  VOICE_RATE: Speech rate (words per minute, default 150)
  VOICE_VOLUME: Volume level (0.0 to 1.0, default 1.0)
  VOICE_PITCH: Voice pitch (0.5 to 2.0, default 1.0)
  VOICE_ID: TTS voice identifier
  WAKE_WORD: Wake word phrase
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

    # TTS parameters
    voice_rate: int = 150        # Words per minute
    voice_volume: float = 1.0    # 0.0 to 1.0
    voice_pitch: float = 1.0     # 0.5 to 2.0
    voice_id: str = "default"

    # Wake word
    wake_word: str = "leo"
    wake_sensitivity: float = 0.5  # 0.0 to 1.0

    # STT
    lang_code: str = "en-IN"
    stt_timeout: float = 3.0       # Seconds to wait for speech start
    stt_phrase_limit: float = 7.0  # Max seconds per phrase

    # Audio backend (auto-detected)
    tts_backend: str = "auto"  # auto, alsa, pulseaudio, pipewire, jack

    # Microphone
    device_index: Optional[int] = None
    calibration_duration: float = 1.5

    def update_from_env(self) -> None:
        """Update settings from environment variables (called once at startup)."""
        import os

        if os.getenv("VOICE_RATE"):
            try:
                self.voice_rate = int(os.getenv("VOICE_RATE", "150"))
            except ValueError:
                pass
        if os.getenv("VOICE_VOLUME"):
            try:
                self.voice_volume = float(os.getenv("VOICE_VOLUME", "1.0"))
            except ValueError:
                pass
        if os.getenv("VOICE_PITCH"):
            try:
                self.voice_pitch = float(os.getenv("VOICE_PITCH", "1.0"))
            except ValueError:
                pass
        if os.getenv("VOICE_ID"):
            self.voice_id = os.getenv("VOICE_ID", "default")
        if os.getenv("WAKE_WORD"):
            self.wake_word = os.getenv("WAKE_WORD", "leo")
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
            "voice_rate": self.voice_rate,
            "voice_volume": self.voice_volume,
            "voice_pitch": self.voice_pitch,
            "voice_id": self.voice_id,
            "wake_word": self.wake_word,
            "wake_sensitivity": self.wake_sensitivity,
            "lang_code": self.lang_code,
            "stt_timeout": self.stt_timeout,
            "stt_phrase_limit": self.stt_phrase_limit,
            "tts_backend": self.tts_backend,
            "device_index": self.device_index,
        }


# Global singleton
voice_settings = VoiceSettings()