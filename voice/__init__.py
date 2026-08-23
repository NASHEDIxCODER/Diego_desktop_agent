"""
Voice subsystem for Diego Desktop Assistant.

CLEAN ARCHITECTURE (2026-08-05):

    Microphone
      ↓
    AudioManager            voice/audio_manager.py   (ONE InputStream,
                                                     verified mic, ring buffer,
                                                     AGC + high-pass ONCE)
      ↓
    Unified VAD             voice/vad.py             (ONE Silero VAD instance,
                                                     shared by wake + command)
      ↓
    ┌─────────────────────┬──────────────────────────┐
    │ WakeListener        │ CommandListener          │
    │ voice/wake_listener │ voice/command_listener   │
    │ openWakeWord +      │ streaming Whisper with   │
    │ Whisper verification│ 800ms+ context windows   │
    └─────────────────────┴──────────────────────────┘
      ↓
    ConversationEngine      core/conversation_engine.py
      ↓
    StreamingTTS            voice/streaming_tts.py   (Kokoro TTS)

STATE MACHINE:
    IDLE → WAKE → FACE_AUTH → LISTEN → THINK → SPEAK → IDLE

SINGLE OWNERS:
  - Audio buffering:     AudioManager
  - VAD:                 voice/vad.py (unified_vad)
  - Endpoint detection:  CommandListener (silence-based)
  - Transcript:          CommandListener (Whisper, 800ms+ windows)
  - Interruption:        ConversationEngine._watch_interruption()

Supporting modules:
  - WakeModelManager  (voice/wake_model_manager.py) — openWakeWord lifecycle
  - WakeWord          (voice/wake_word.py)          — transcript verification
  - AudioProcessing   (voice/audio_processing.py)   — AGC, high-pass, stage tracing
  - VoiceSettings     (voice/settings.py)           — runtime configuration
"""

from voice.settings import VoiceSettings

__all__ = ["VoiceSettings"]