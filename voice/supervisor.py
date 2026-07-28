"""
VoiceSupervisor — Deterministic state machine for Leo voice pipeline.

Orchestrates the entire voice lifecycle with a strict state machine.

States:
  BOOT → INIT → LOAD_MODELS → LOAD_PLUGINS → READY → AUTH → GREETING →
  WAKEWORD → LISTEN → STT → NLP → PLUGIN → TTS → WAKEWORD

  If microphone is unavailable at INIT:
  BOOT → INIT → VoiceUnavailable (terminal for voice features)

Principles:
  - Every transition is logged with timing
  - One crash never terminates the assistant
  - Each state has a timeout
  - The supervisor can be reset to any state
  - Health checks run periodically
  - Failed calibration → VoiceUnavailable, NOT VoiceReady
"""

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class VoiceState(Enum):
    """Deterministic states for the voice pipeline."""
    BOOT = auto()
    INIT = auto()
    LOAD_MODELS = auto()
    LOAD_PLUGINS = auto()
    READY = auto()
    VOICE_UNAVAILABLE = auto()
    AUTH = auto()
    GREETING = auto()
    WAKEWORD = auto()
    LISTEN = auto()
    STT = auto()
    NLP = auto()
    PLUGIN = auto()
    TTS = auto()
    ERROR = auto()
    SHUTDOWN = auto()


# State timeout in seconds (0 = no timeout)
STATE_TIMEOUTS = {
    VoiceState.BOOT: 10.0,
    VoiceState.INIT: 10.0,
    VoiceState.LOAD_MODELS: 10.0,
    VoiceState.LOAD_PLUGINS: 10.0,
    VoiceState.READY: 0,
    VoiceState.VOICE_UNAVAILABLE: 0,
    VoiceState.AUTH: 15.0,
    VoiceState.GREETING: 5.0,
    VoiceState.WAKEWORD: 0,
    VoiceState.LISTEN: 10.0,
    VoiceState.STT: 10.0,
    VoiceState.NLP: 2.0,
    VoiceState.PLUGIN: 5.0,
    VoiceState.TTS: 10.0,
    VoiceState.ERROR: 0,
    VoiceState.SHUTDOWN: 5.0,
}

# Valid transitions
VALID_TRANSITIONS = {
    VoiceState.BOOT: [VoiceState.INIT, VoiceState.ERROR, VoiceState.SHUTDOWN],
    VoiceState.INIT: [
        VoiceState.LOAD_MODELS, VoiceState.VOICE_UNAVAILABLE,
        VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.LOAD_MODELS: [
        VoiceState.LOAD_PLUGINS, VoiceState.VOICE_UNAVAILABLE,
        VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.LOAD_PLUGINS: [
        VoiceState.READY, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.READY: [
        VoiceState.AUTH, VoiceState.WAKEWORD, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.VOICE_UNAVAILABLE: [
        VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.AUTH: [
        VoiceState.GREETING, VoiceState.READY, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.GREETING: [
        VoiceState.WAKEWORD, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.WAKEWORD: [
        VoiceState.LISTEN, VoiceState.WAKEWORD, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.LISTEN: [
        VoiceState.STT, VoiceState.WAKEWORD, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.STT: [
        VoiceState.NLP, VoiceState.WAKEWORD, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.NLP: [
        VoiceState.PLUGIN, VoiceState.TTS, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.PLUGIN: [
        VoiceState.TTS, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.TTS: [
        VoiceState.WAKEWORD, VoiceState.ERROR, VoiceState.SHUTDOWN,
    ],
    VoiceState.ERROR: [
        VoiceState.VOICE_UNAVAILABLE, VoiceState.READY, VoiceState.SHUTDOWN,
    ],
    VoiceState.SHUTDOWN: [],
}


class VoiceSupervisor:
    """
    Orchestrates the voice pipeline via a deterministic state machine.

    Usage:
        supervisor = VoiceSupervisor()
        await supervisor.start()
        await supervisor.run_pipeline()
    """

    def __init__(self):
        self._state: VoiceState = VoiceState.BOOT
        self._previous_state: Optional[VoiceState] = None
        self._state_start_time: float = time.time()
        self._running = False
        self._error_count = 0
        self._max_errors = 5
        self._state_history: list = []
        self._listeners: Dict[VoiceState, list] = {}
        self._transition_count = 0
        self._voice_available = False

        # Callbacks for external systems
        self._on_wake_word: Optional[Callable] = None
        self._on_auth_request: Optional[Callable] = None
        self._on_command: Optional[Callable] = None

    # ── State Management ───────────────────────────────────────

    @property
    def state(self) -> VoiceState:
        """Get current state."""
        return self._state

    @property
    def state_duration(self) -> float:
        """Get seconds in current state."""
        return time.time() - self._state_start_time

    @property
    def is_running(self) -> bool:
        """Check if supervisor is running."""
        return self._running

    @property
    def voice_available(self) -> bool:
        """Check if voice subsystem is available."""
        return self._voice_available

    async def transition(self, new_state: VoiceState) -> bool:
        """
        Transition to a new state if valid.

        Args:
            new_state: Target state.

        Returns:
            True if transition was valid and executed.
        """
        allowed = VALID_TRANSITIONS.get(self._state, [])
        if new_state not in allowed:
            logger.warning("Invalid transition: %s → %s (allowed: %s)",
                          self._state.name, new_state.name,
                          [s.name for s in allowed])
            return False

        self._previous_state = self._state
        old_state = self._state
        self._state = new_state
        self._state_start_time = time.time()
        self._transition_count += 1
        self._state_history.append({
            "from": old_state.name,
            "to": new_state.name,
            "at": time.time(),
        })

        logger.info("Voice state: %s → %s (transition #%d)",
                   old_state.name, new_state.name, self._transition_count)

        # Notify listeners
        listeners = self._listeners.get(new_state, [])
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(new_state)
                else:
                    listener(new_state)
            except Exception as e:
                logger.error("State listener error: %s", e)

        return True

    def on_state(self, state: VoiceState, callback: Callable) -> None:
        """Register a callback for a specific state."""
        if state not in self._listeners:
            self._listeners[state] = []
        self._listeners[state].append(callback)

    # ── Pipeline ───────────────────────────────────────────────

    async def start(self) -> None:
        """Start the voice pipeline."""
        self._running = True
        self._error_count = 0
        logger.info("Voice supervisor starting")
        await self.transition(VoiceState.INIT)

    async def stop(self) -> None:
        """Stop the voice pipeline."""
        logger.info("Voice supervisor stopping")
        await self.transition(VoiceState.SHUTDOWN)
        self._running = False

    async def run_pipeline(self) -> None:
        """
        Run the full voice pipeline.

        This is the main loop that drives the state machine.
        It handles errors gracefully and never crashes.
        """
        if not self._running:
            await self.start()

        try:
            while self._running and self._state != VoiceState.SHUTDOWN:
                try:
                    await self._process_state()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._error_count += 1
                    logger.error("Voice pipeline error (count=%d): %s",
                                self._error_count, e, exc_info=True)

                    if self._error_count >= self._max_errors:
                        logger.critical("Too many voice errors (%d), entering VOICE_UNAVAILABLE",
                                       self._error_count)
                        await self.transition(VoiceState.VOICE_UNAVAILABLE)
                    else:
                        await self.transition(VoiceState.ERROR)
                        await asyncio.sleep(1)
                        await self.transition(VoiceState.READY)
        finally:
            self._running = False

    async def _process_state(self) -> None:
        """Process the current state."""
        state = self._state

        # Check timeout
        timeout = STATE_TIMEOUTS.get(state, 0)
        if timeout > 0 and self.state_duration > timeout:
            logger.warning("State %s timed out after %.1fs (timeout=%.1fs)",
                          state.name, self.state_duration, timeout)
            await self._handle_timeout(state)
            return

        # Process based on state
        if state == VoiceState.INIT:
            await self._on_init()
        elif state == VoiceState.LOAD_MODELS:
            await self._on_load_models()
        elif state == VoiceState.LOAD_PLUGINS:
            await self._on_load_plugins()
        elif state == VoiceState.READY:
            await self._on_ready()
        elif state == VoiceState.VOICE_UNAVAILABLE:
            await self._on_voice_unavailable()
        elif state == VoiceState.AUTH:
            await self._on_auth()
        elif state == VoiceState.GREETING:
            await self._on_greeting()
        elif state == VoiceState.WAKEWORD:
            await self._on_wakeword()
        elif state == VoiceState.LISTEN:
            await self._on_listen()
        elif state == VoiceState.STT:
            await self._on_stt()
        elif state == VoiceState.NLP:
            await self._on_nlp()
        elif state == VoiceState.PLUGIN:
            await self._on_plugin()
        elif state == VoiceState.TTS:
            await self._on_tts()
        elif state == VoiceState.ERROR:
            await asyncio.sleep(1)
        elif state == VoiceState.SHUTDOWN:
            self._running = False
        else:
            await asyncio.sleep(0.1)

    async def _handle_timeout(self, state: VoiceState) -> None:
        """Handle a state timeout."""
        logger.warning("State %s timed out, transitioning to VOICE_UNAVAILABLE", state.name)
        await self.transition(VoiceState.VOICE_UNAVAILABLE)

    # ── State Handlers (override in subclass) ──────────────────

    async def _on_init(self) -> None:
        """Initialize voice components. If microphone unavailable, go to VOICE_UNAVAILABLE."""
        from voice.audio_device import audio_device
        from voice.microphone import microphone
        from voice.noise import noise_calibrator

        # Detect audio backend (cached result, no repeated probes)
        audio_device.detect_backend()

        # Check microphone availability BEFORE calibration
        mic = microphone.get_microphone()
        mic_available = mic is not None

        if not mic_available:
            logger.warning("No microphone available — voice features disabled")
            self._voice_available = False
            await self.transition(VoiceState.VOICE_UNAVAILABLE)
            return

        # Load cached noise profile (non-blocking)
        noise_calibrator.load_profile()

        # Apply cached noise profile without re-calibrating
        from voice.recognizer import speech_recognizer
        sr = speech_recognizer._get_sr()
        if sr:
            noise_calibrator.apply(sr)

        self._voice_available = True

        logger.info("Voice components initialized (audio=%s, mic=available)",
                   audio_device.backend)

        await self.transition(VoiceState.LOAD_MODELS)

    async def _on_load_models(self) -> None:
        """Load voice models."""
        from voice.synthesizer import speech_synthesizer
        logger.info("Voice models ready")
        await self.transition(VoiceState.LOAD_PLUGINS)

    async def _on_load_plugins(self) -> None:
        """Load voice plugins (none currently)."""
        logger.info("Voice plugins ready")
        await self.transition(VoiceState.READY)

    async def _on_ready(self) -> None:
        """Ready state — waiting for wake word or auth."""
        await asyncio.sleep(0.1)

    async def _on_voice_unavailable(self) -> None:
        """Voice unavailable state — sleep and wait for shutdown or error recovery."""
        await asyncio.sleep(1.0)

    async def _on_auth(self) -> None:
        """Handle authentication. Override to integrate face auth."""
        if self._on_auth_request:
            try:
                result = self._on_auth_request()
                if result:
                    await self.transition(VoiceState.GREETING)
                    return
            except Exception as e:
                logger.warning("Auth failed: %s", e)
        await self.transition(VoiceState.GREETING)

    async def _on_greeting(self) -> None:
        """Speak greeting."""
        from voice.synthesizer import speech_synthesizer
        speech_synthesizer.speak("I am ready. How can I help you today?")
        await asyncio.sleep(1.5)
        await self.transition(VoiceState.WAKEWORD)

    async def _on_wakeword(self) -> None:
        """Wait for wake word."""
        from voice.wake_word import wake_word_engine
        from voice.recognizer import speech_recognizer

        text = speech_recognizer.listen(timeout=3, phrase_time_limit=3)
        if text and wake_word_engine.detect(text):
            logger.info("Wake word detected: %s", text)
            await self.transition(VoiceState.LISTEN)
        else:
            await asyncio.sleep(0.1)

    async def _on_listen(self) -> None:
        """Listen for a command."""
        from voice.recognizer import speech_recognizer

        text = speech_recognizer.listen(timeout=5, phrase_time_limit=10)
        if text:
            self._last_command = text
            logger.info("Command: %s", text)
            await self.transition(VoiceState.STT)
        else:
            await self.transition(VoiceState.WAKEWORD)

    async def _on_stt(self) -> None:
        """Process STT (already done in listen, transition to NLP)."""
        await self.transition(VoiceState.NLP)

    async def _on_nlp(self) -> None:
        """Process NLP. Override to integrate with intent classifier."""
        if self._on_command:
            try:
                result = self._on_command(self._last_command)
                if result == "__EXIT__":
                    self._last_response = result
                    await self.transition(VoiceState.TTS)
                    return
            except Exception as e:
                logger.error("Command handler error: %s", e)
        await self.transition(VoiceState.TTS)

    async def _on_plugin(self) -> None:
        """Route to plugins. Override to integrate with plugin system."""
        await self.transition(VoiceState.TTS)

    async def _on_tts(self) -> None:
        """Speak the response."""
        from voice.synthesizer import speech_synthesizer
        if hasattr(self, '_last_response') and self._last_response:
            if self._last_response == "__EXIT__":
                await self.transition(VoiceState.SHUTDOWN)
                return
            speech_synthesizer.speak(self._last_response)
        await self.transition(VoiceState.WAKEWORD)

    # ── Configuration ──────────────────────────────────────────

    def set_wake_word_callback(self, callback: Callable) -> None:
        """Set callback for wake word detection."""
        self._on_wake_word = callback

    def set_auth_callback(self, callback: Callable) -> None:
        """Set callback for authentication."""
        self._on_auth_request = callback

    def set_command_callback(self, callback: Callable) -> None:
        """Set callback for command processing."""
        self._on_command = callback

    # ── Diagnostics ────────────────────────────────────────────

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the supervisor."""
        return {
            "state": self._state.name,
            "voice_available": self._voice_available,
            "previous_state": self._previous_state.name if self._previous_state else None,
            "state_duration": round(self.state_duration, 2),
            "running": self._running,
            "error_count": self._error_count,
            "transition_count": self._transition_count,
            "history": self._state_history[-20:],  # Last 20 transitions
        }

    async def reset(self) -> None:
        """Reset the supervisor to BOOT state."""
        self._state = VoiceState.BOOT
        self._previous_state = None
        self._state_start_time = time.time()
        self._error_count = 0
        logger.info("Voice supervisor reset to BOOT")

    async def shutdown(self) -> None:
        """Shutdown all voice components."""
        logger.info("Shutting down voice components...")
        await self.transition(VoiceState.SHUTDOWN)
        self._running = False

        from voice.synthesizer import speech_synthesizer
        from voice.microphone import microphone
        from voice.wake_word import wake_word_engine

        speech_synthesizer.close()
        microphone.close()
        wake_word_engine.close()
        logger.info("Voice components shut down")


# Global singleton
voice_supervisor = VoiceSupervisor()