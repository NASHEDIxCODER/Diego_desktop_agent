"""
Pluggable TTS architecture for Diego Desktop Assistant.

Priority order:
1. Kokoro (lightweight, fast, natural)
2. XTTS v2 (best quality, GPU-accelerated)
3. Piper (fast, local)
4. pyttsx3 (espeak fallback)
"""

from voice.tts.manager import TTSManager, tts_manager

__all__ = ["TTSManager", "tts_manager"]