"""
ConversationEngine — Leo's full-duplex conversational core.

This is the orchestrator that makes Leo feel like Siri / ChatGPT Voice /
Gemini Live instead of a command executor.

PIPELINE (all concurrent, all streaming):
    Wake
      ↓
    Streaming audio (ring buffer)
      ↓
    Streaming VAD ──────────────┐
      ↓                         │
    Streaming Whisper (partials)│
      ↓                         │  full duplex:
    Streaming LLM               │  user speech while Leo talks
      ↓                         │  → interrupt → stop TTS → listen
    Sentence generation         │
      ↓                         │
    Streaming TTS ──────────────┘
      ↓
    Audio playback (interruptible)

LIFECYCLE:
    WAKE mode: waiting for "leo" / "hey leo" / "hello leo"
      → on wake: greet, enter CONVERSATION mode
    CONVERSATION mode: continuous listening, no wake word needed
      → stays here across turns
      → returns to WAKE only on:
          * explicit goodbye ("bye", "goodbye", "that's all")
          * prolonged silence (CONVERSATION_TIMEOUT_S)

Everything is async and cancellable.
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional

from voice.audio_manager import audio_manager
from voice.streaming_stt import streaming_stt, is_filler, UtteranceEvent
from voice.streaming_tts import streaming_tts
from agent.streaming_llm import streaming_llm
from agent.conversation_memory import conv_memory
from agent.personality import personality

logger = logging.getLogger(__name__)

# ── Conversation lifecycle tuning ────────────────────────
CONVERSATION_TIMEOUT_S = 45.0   # silence this long → back to wake mode
GOODBYE_PHRASES = {
    "bye", "goodbye", "see you", "see ya", "later", "that's all",
    "thats all", "nothing else", "i'm done", "im done", "stop listening",
    "go to sleep", "good night", "goodnight",
}
WAKE_VARIANTS = ("leo", "hey leo", "hello leo", "hi leo", "okay leo", "ok leo")


class ConversationEngine:
    """
    Full-duplex conversational engine.

    Owns the wake/conversation state machine and coordinates
    streaming STT → streaming LLM → streaming TTS with interruption.
    """

    def __init__(self):
        self._mode = "wake"            # "wake" | "conversation"
        self._running = False
        self._authenticated = False
        self._user_name: Optional[str] = None

        # Per-turn cancellation
        self._tts_interrupt = asyncio.Event()      # set when user interrupts Leo
        self._current_response_task: Optional[asyncio.Task] = None
        self._interruption_monitor: Optional[asyncio.Task] = None

        # Conversation activity
        self._last_interaction = 0.0
        self._turn_count = 0

        # Vision context provider (set by main)
        self._vision_context_fn = None

        # Action executor (set by main) — maps ACTION dicts to desktop ops
        self._action_executor = None

    # ── Wiring ────────────────────────────────────────────

    def set_vision_context(self, fn) -> None:
        """Provide a callable() -> str that returns current screen context."""
        self._vision_context_fn = fn

    def set_action_executor(self, fn) -> None:
        """Provide an async callable(action_dict) -> str that runs desktop actions."""
        self._action_executor = fn

    def set_authenticated(self, name: Optional[str]) -> None:
        self._authenticated = True
        self._user_name = name
        if name:
            conv_memory.set_user_name(name)

    # ── Main run loop ─────────────────────────────────────

    async def run(self) -> None:
        """
        Run the conversational engine forever.

        Starts audio, streams utterances, and processes each turn.
        """
        logger.info("[ENGINE] Starting conversation engine")
        self._running = True

        # Start audio capture (shared ring buffer)
        if not audio_manager.is_running:
            ok = audio_manager.start()
            if not ok:
                logger.error("[ENGINE] AudioManager failed to start")
                return

        # Initialize streaming components
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, streaming_stt.initialize)
        await loop.run_in_executor(None, streaming_tts.initialize)

        self._last_interaction = time.time()

        # Wake: greet once authenticated
        await self._enter_conversation(greet=True)

        # Main streaming loop
        try:
            async for event in streaming_stt.stream_utterances():
                if not self._running:
                    break
                await self._handle_event(event)
        except asyncio.CancelledError:
            logger.info("[ENGINE] Conversation engine cancelled")
        finally:
            self._running = False

    # ── Event handling ────────────────────────────────────

    async def _handle_event(self, event: UtteranceEvent) -> None:
        """Route an STT event."""
        if event.kind == "speech_start":
            # User started speaking — if Leo is talking, this is an interruption
            if streaming_tts.is_speaking:
                await self._interrupt_leo()
            return

        if event.kind == "partial":
            # In wake mode, check partials for the wake word too — this makes
            # wake feel instant (partials arrive every ~400ms, long before the
            # final transcription).
            if self._mode == "wake" and self._contains_wake(event.text.lower()):
                await self._enter_conversation(greet=True)
            return

        if event.kind == "final":
            await self._handle_final_utterance(event.text)


    async def _handle_final_utterance(self, text: str) -> None:
        """Process a complete user utterance."""
        text = text.strip()
        if not text:
            return

        self._last_interaction = time.time()
        lower = text.lower()

        # ── Wake-mode gating ──────────────────────────────
        if self._mode == "wake":
            if self._contains_wake(lower):
                # Strip the wake phrase; if a command followed, run it.
                remainder = self._strip_wake(lower)
                await self._enter_conversation(greet=True)
                if remainder and not is_filler(remainder):
                    await self._handle_final_utterance(remainder)
            return  # ignore non-wake speech in wake mode

        # ── Conversation mode ─────────────────────────────

        # If the user re-says the wake word mid-conversation, just strip it.
        if self._contains_wake(lower):
            stripped = self._strip_wake(lower)
            if not stripped:
                return  # "leo" alone → already awake, ignore
            text = stripped
            lower = stripped

        # Filler-only: keep listening, don't respond
        if is_filler(text):
            logger.info("[ENGINE] Filler '%s' — continuing to listen", text)
            return


        # Goodbye → back to wake mode
        if self._is_goodbye(lower):
            await self._say_and_reset(personality.farewell())
            await self._exit_conversation()
            return

        self._turn_count += 1
        logger.info("[ENGINE] Turn #%d: '%s'", self._turn_count, text)

        # Cancel any in-flight response (user changed topic)
        await self._cancel_current_response()

        # Process this turn as a background task so we can keep listening
        self._current_response_task = asyncio.create_task(
            self._respond(text))

    # ── Response pipeline ─────────────────────────────────

    async def _respond(self, user_text: str) -> None:
        """
        Stream LLM → TTS for one user turn, with interruption support.
        """
        self._tts_interrupt.clear()
        interrupt = self._tts_interrupt

        # Start interruption monitor: if user speaks while Leo talks, abort
        self._interruption_monitor = asyncio.create_task(
            streaming_stt.detect_interruption(interrupt))

        actions_to_run = []

        try:
            # Build the sentence stream (LLM), injecting vision context if relevant
            sentence_stream = self._llm_sentences(user_text)

            # Tee the stream: speak sentences AND collect ACTION lines
            async def speakable():
                async for piece in sentence_stream:
                    if interrupt.is_set():
                        return
                    if piece.startswith("ACTION:"):
                        actions_to_run.append(piece[len("ACTION:"):].strip())
                        continue
                    yield piece

            await streaming_tts.speak_sentences(speakable(), interrupt)

        except asyncio.CancelledError:
            logger.info("[ENGINE] Response cancelled")
            raise
        except Exception as e:
            logger.warning("[ENGINE] Response error: %s", e)
        finally:
            if self._interruption_monitor is not None:
                self._interruption_monitor.cancel()
                self._interruption_monitor = None

        # Execute any desktop actions the LLM requested (after speaking the ack)
        for action_json in actions_to_run:
            await self._run_action(action_json)

    async def _llm_sentences(self, user_text: str):
        """Yield LLM sentences, adding vision context when useful."""
        # If the user references the screen, prepend fresh vision context
        if self._references_screen(user_text) and self._vision_context_fn:
            try:
                loop = asyncio.get_event_loop()
                ctx = await loop.run_in_executor(None, self._vision_context_fn)
                if ctx:
                    user_text = f"{user_text}\n[Screen context: {ctx}]"
            except Exception as e:
                logger.debug("[ENGINE] vision context failed: %s", e)

        async for sentence in streaming_llm.generate(user_text, self._tts_interrupt):
            yield sentence

    # ── Actions ───────────────────────────────────────────

    async def _run_action(self, action_json: str) -> None:
        """Parse and execute an ACTION line from the LLM."""
        if self._action_executor is None:
            return
        try:
            # Extract the JSON object
            start = action_json.find("{")
            end = action_json.rfind("}")
            if start < 0 or end <= start:
                return
            action = json.loads(action_json[start:end + 1])
            logger.info("[ENGINE] Executing action: %s", action)
            result = await self._action_executor(action)
            if result:
                logger.info("[ENGINE] Action result: %s", result)
        except Exception as e:
            logger.warning("[ENGINE] Action execution failed: %s", e)

    # ── Interruption ──────────────────────────────────────

    async def _interrupt_leo(self) -> None:
        """User interrupted Leo: stop TTS instantly and listen."""
        logger.info("[ENGINE] Interruption — stopping TTS")
        self._tts_interrupt.set()
        streaming_tts.stop()
        await self._cancel_current_response()

    async def _cancel_current_response(self) -> None:
        if self._current_response_task is not None and not self._current_response_task.done():
            self._current_response_task.cancel()
            try:
                await self._current_response_task
            except (asyncio.CancelledError, Exception):
                pass
        self._current_response_task = None

    async def _say_and_reset(self, text: str) -> None:
        """Speak a short line (no interruption monitor)."""
        self._tts_interrupt.clear()

        async def one():
            yield text
        await streaming_tts.speak_sentences(one(), self._tts_interrupt)

    # ── Conversation lifecycle ────────────────────────────

    async def _enter_conversation(self, greet: bool = False) -> None:
        """Enter conversation mode (optionally greet)."""
        self._mode = "conversation"
        self._last_interaction = time.time()
        if greet:
            returning = conv_memory.turn_count > 0
            name = conv_memory.user_name or self._user_name
            greeting = personality.greeting(returning=returning)
            if name and not returning:
                greeting = greeting.rstrip(".") + f", {name}."
            logger.info("[ENGINE] Entering conversation — greeting: %s", greeting)
            await self._say_and_reset(greeting)

    async def _exit_conversation(self) -> None:
        """Return to wake mode."""
        logger.info("[ENGINE] Exiting conversation → wake mode")
        self._mode = "wake"
        await self._cancel_current_response()

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _contains_wake(lower_text: str) -> bool:
        return any(v in lower_text for v in WAKE_VARIANTS)

    @staticmethod
    def _strip_wake(lower_text: str) -> str:
        """Remove the wake phrase and return the remaining command text."""
        # Try longest variants first so "hello leo" is stripped before "leo"
        for variant in sorted(WAKE_VARIANTS, key=len, reverse=True):
            idx = lower_text.find(variant)
            if idx >= 0:
                rest = (lower_text[:idx] + " " + lower_text[idx + len(variant):])
                # Clean up separators/extra spaces
                rest = rest.strip(" ,.\t")
                rest = " ".join(rest.split())
                return rest
        return ""

    @staticmethod
    def _is_goodbye(lower_text: str) -> bool:
        return any(p in lower_text for p in GOODBYE_PHRASES)

    @staticmethod
    def _references_screen(text: str) -> bool:
        t = text.lower()
        keys = [
            "screen", "looking at", "this page", "this window", "what am i",
            "read this", "what does this say", "on my screen", "this button",
            "click the", "click this", "what's open", "whats open",
        ]
        return any(k in t for k in keys)

    # ── Conversation timeout watchdog ─────────────────────

    async def timeout_watchdog(self) -> None:
        """Return to wake mode after prolonged silence."""
        while self._running:
            await asyncio.sleep(2.0)
            if self._mode != "conversation":
                continue
            # Don't time out while Leo is speaking or user is speaking
            if streaming_tts.is_speaking:
                continue
            idle = time.time() - self._last_interaction
            if idle >= CONVERSATION_TIMEOUT_S:
                logger.info("[ENGINE] Conversation timed out (%.0fs idle)", idle)
                await self._exit_conversation()

    # ── Shutdown ──────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully stop the engine."""
        logger.info("[ENGINE] Shutting down conversation engine")
        self._running = False
        streaming_stt.cancel()
        streaming_tts.stop()
        await self._cancel_current_response()
        if self._interruption_monitor is not None:
            self._interruption_monitor.cancel()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def running(self) -> bool:
        return self._running


# Global singleton
conversation_engine = ConversationEngine()
