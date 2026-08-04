"""
Voice subsystem for Leo Desktop Assistant.

THE SINGLE VOICE PIPELINE (exactly one implementation of each stage):

    Microphone
      ↓
    AudioManager            voice/audio_manager.py   (ONE InputStream,
                                                      verified mic, ring buffer)
      ↓
    WakeListener            voice/wake_listener.py   (Silero VAD gate →
                                                      openWakeWord streaming)
      ↓
    Whisper verification    voice/wake_word.py       (verify_wake_transcript —
                                                      RapidFuzz + phonetic)
      ↓
    Conversation            core/conversation_engine.py

Supporting modules:
  - WakeModelManager  (voice/wake_model_manager.py) — openWakeWord lifecycle
  - StreamingSTT      (voice/streaming_stt.py)      — Whisper (verification +
                                                      conversation STT)
  - StreamingTTS      (voice/streaming_tts.py)      — speech output
  - VoiceSettings     (voice/settings.py)           — runtime configuration
"""

from voice.settings import VoiceSettings

__all__ = ["VoiceSettings"]
