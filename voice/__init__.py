"""
Voice subsystem for Leo Desktop Assistant.

Production-grade voice pipeline with:
- VoiceSupervisor: State machine managing the entire voice lifecycle
- MicrophoneManager: Auto-recovery, hotplug support
- WakeWordEngine: Offline wake word detection
- SpeechRecognizer: STT with fallback chain
- SpeechSynthesizer: TTS with multi-backend (ALSA/PulseAudio/PipeWire/JACK)
- AudioDeviceManager: Auto-detect best audio backend
- NoiseCalibrator: Persist noise profile to disk
- VoiceSettings: Runtime configuration

Architecture:
  One crash never terminates the assistant.
  Every component is isolated with its own error handling.
  The VoiceSupervisor orchestrates the state machine.
"""

from voice.supervisor import VoiceSupervisor, VoiceState
from voice.settings import VoiceSettings

__all__ = [
    "VoiceSupervisor",
    "VoiceState",
    "VoiceSettings",
]