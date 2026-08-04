"""
StreamingLLM — Token-streaming LLM client for conversational Leo.

Replaces the blocking `llm_client.chat()` with a streaming pipeline:
  - Streams tokens from Ollama as they are generated
  - Splits tokens into sentences as they arrive
  - Yields complete sentences immediately for TTS
  - Supports cancellation (for interruption)
  - Integrates conversation memory + personality

This enables the "begin speaking after the first meaningful sentence"
behavior that makes Leo feel like ChatGPT Voice / Gemini Live.

Usage:
    from agent.streaming_llm import streaming_llm

    async for sentence in streaming_llm.generate("open vs code"):
        tts.speak_streaming(sentence)
"""

import asyncio
import logging
import re
import time
from typing import AsyncIterator, Optional

import httpx

from config.settings import settings
from agent.conversation_memory import conv_memory
from agent.personality import personality

logger = logging.getLogger(__name__)


# Leo's conversational system prompt — makes Leo sound alive
LEO_SYSTEM_PROMPT = """You are Leo — a real desktop companion, not a command executor.

PERSONALITY:
- Speak casually like a friend sitting next to the user.
- Keep responses SHORT and natural. 1-3 sentences usually.
- Never say "How may I assist you?" or any robotic phrase.
- Never say "Task completed." or just "Done." — use natural confirmation instead.
- Never mention you are an AI.
- Never use emoji or markdown.
- Vary your language. Don't repeat phrases.
- Match the user's tone: if they're frustrated, acknowledge it; if happy, be upbeat.

CAPABILITIES:
- You can open apps, search the web, read the screen, control mouse/keyboard.
- You can play music, pause, skip, adjust volume, find songs.
- You know what window is focused, what project the user is on, battery level.
- When the user asks you to do something, say what you're doing briefly.
- If the request is an ACTION, respond with a SHORT spoken confirmation
  followed by a line starting with "ACTION:" describing the action.
- For follow-up references like "that", "this", "continue", "go on" — use
  conversation context, don't ask what they meant.

ACTION FORMAT (only when a desktop action is needed):
ACTION: {"action": "<action_name>", "params": {...}}

Available actions:
- desktop_open(app) — open an application (e.g. "code", "google-chrome", "firefox",
  "gnome-terminal", "nautilus"). Use for "open VS Code/Chrome/Terminal/Files".
- open_folder(path) — open a folder in the file manager (path optional, default home)
- browser_navigate(url) — open a website
- browser_search(query) — google search
- read_screen() — describe what is on screen
- click_text(text) — click visible text / a button
- scroll(direction) — scroll up/down
- key_press(key) — press a key
- type_text(text) — type text
- play_media(query) — play music/video (handles Spotify, MPV, YouTube, local)
- music_pause() — pause music
- music_resume() — resume music
- music_next() — skip to next track
- music_previous() — go to previous track
- music_stop() — stop music
- music_shuffle() — toggle shuffle
- music_repeat() — toggle repeat
- music_volume(percent) — set music volume (0-100)
- music_mute() — toggle mute
- music_status() — what's currently playing
- volume_up() / volume_down() / volume_set(percent) / volume_mute() — system volume
- brightness_up() / brightness_down() / brightness_set(percent) — screen brightness
- lock_screen() — lock the desktop session
- shutdown() — power off the computer (only when explicitly asked)
- restart() — reboot the computer (only when explicitly asked)

DESKTOP CONTEXT (you receive this automatically):
- [Desktop: ...] shows focused window, git branch, terminal path, browser tab
- [Music: ...] shows what's currently playing
- [Screen context: ...] shows what's visible on screen
- [Relevant context: ...] shows learned user facts, preferences, habits
- Use this context without mentioning it to the user unless asked.

RESPONSE STYLE:
- Simple chat: just answer naturally, no ACTION line.
- Actions: short confirmation + one ACTION line. Example:
    "Sure, opening VS Code now.
    ACTION: {"action": "desktop_open", "params": {"app": "code"}}"
- Music: "Playing lofi hip hop."
    ACTION: {"action": "play_media", "params": {"query": "lofi hip hop coding"}}
- For multi-step tasks, output one ACTION per step as separate ACTION lines.
- Never output code blocks or extra formatting around ACTION lines.
- You are created by Yeshu.
"""

# Sentence-ending punctuation
_SENTENCE_END = re.compile(r"^(.*?[.!?])(?:\s|$)(.*)$", re.DOTALL)

# Phrases that should NOT be treated as complete for TTS (abbreviations etc.)
_ABBREVS = ("mr.", "mrs.", "ms.", "dr.", "st.", "e.g.", "i.e.", "vs.", "etc.")


class StreamingLLM:
    """
    Streaming LLM client with sentence-level output.

    Streams tokens from Ollama, accumulates them, and yields complete
    sentences as soon as they're available. Also extracts ACTION lines
    so the conversation engine can execute desktop actions.
    """

    def __init__(self):
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self._model: Optional[str] = None
        self._available = False
        self._checked = False
        self._model_names: list = []

    async def ensure_initialized(self) -> bool:
        """Check if Ollama is reachable and auto-detect models."""
        if self._checked:
            return self._available
        self._checked = True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    self._model_names = [m["name"] for m in models]
                    logger.info("[STREAM-LLM] Ollama available. Models: %s", self._model_names)
                    self._available = True
                    self._select_model()
                else:
                    logger.warning("[STREAM-LLM] Ollama returned status %d", resp.status_code)
        except httpx.ConnectError:
            logger.warning("[STREAM-LLM] Ollama not reachable at %s", self._base_url)
        except Exception as e:
            logger.warning("[STREAM-LLM] Ollama check failed: %s", e)
        return self._available

    def _select_model(self) -> None:
        """Auto-select the best available model."""
        if not self._model_names:
            self._model = "llama3.2"
            return

        configured = getattr(settings, 'OLLAMA_MODEL', None) or getattr(settings, 'LLM_MODEL', None)
        if configured:
            for m in self._model_names:
                if m == configured or m.startswith(f"{configured}:"):
                    self._model = m
                    logger.info("[STREAM-LLM] Using configured model: %s", self._model)
                    return

        self._model = self._model_names[0]
        logger.info("[STREAM-LLM] Auto-selected model: %s", self._model)

    @staticmethod
    def _extract_complete_sentences(buffer: str):
        """Split buffer into (complete_sentences, remainder)."""
        sentences = []
        rest = buffer
        while True:
            m = _SENTENCE_END.match(rest)
            if not m:
                break
            sentence, rest = m.group(1), m.group(2)
            stripped = sentence.strip()
            if not stripped:
                continue
            # Avoid splitting on abbreviations
            lower = stripped.lower()
            if any(lower.endswith(a) for a in _ABBREVS):
                # Not a real sentence end; reattach and stop
                rest = sentence + " " + rest
                break
            sentences.append(stripped)
        return sentences, rest

    async def generate(
        self,
        user_text: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response as complete sentences.

        Args:
            user_text: What the user said.
            cancel_event: If set, generation stops immediately (interruption).

        Yields:
            Complete sentences as soon as they're available.
            ACTION lines are yielded as-is (the engine parses them).
        """
        if not await self.ensure_initialized():
            yield "I'm having trouble connecting to my brain right now."
            return

        # Try personality shortcut first for instant responses
        quick = personality.contextual_response(user_text)
        if quick is not None:
            conv_memory.add_user(user_text)
            conv_memory.add_assistant(quick)
            yield quick
            return

        # Try answering directly from long-term memory (instant)
        mem_answer = conv_memory.query_facts(user_text)
        if mem_answer and ("what" in user_text.lower() or "who" in user_text.lower()):
            answer = self._format_memory_answer(user_text, mem_answer)
            conv_memory.add_user(user_text)
            conv_memory.add_assistant(answer)
            yield answer
            return

        conv_memory.add_user(user_text)
        context = conv_memory.build_context()

        prompt_parts = [LEO_SYSTEM_PROMPT]
        if context:
            prompt_parts.append(f"\nContext:\n{context}")
        prompt_parts.append(f"\nUser: {user_text}\nLeo:")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 300,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        }

        sentence_buffer = ""
        full_response = ""
        first_token_time = None
        t0 = time.time()

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{self._base_url}/api/generate", json=payload
                ) as resp:
                    if resp.status_code != 200:
                        logger.warning("[STREAM-LLM] API error: %d", resp.status_code)
                        yield "I had trouble processing that."
                        return

                    async for line in resp.aiter_lines():
                        # Interruption check
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info("[STREAM-LLM] Generation cancelled (interruption)")
                            return

                        if not line:
                            continue
                        try:
                            import json as _json
                            data = _json.loads(line)
                        except Exception:
                            continue

                        token = data.get("response", "")
                        if token:
                            if first_token_time is None:
                                first_token_time = time.time()
                                logger.info(
                                    "[STREAM-LLM] First token in %.0fms",
                                    (first_token_time - t0) * 1000)
                            sentence_buffer += token
                            full_response += token

                            # If we hit an ACTION line, yield it separately
                            if "ACTION:" in sentence_buffer:
                                before, _, after = sentence_buffer.partition("ACTION:")
                                if before.strip():
                                    sents, rem = self._extract_complete_sentences(before)
                                    for s in sents:
                                        yield s
                                    sentence_buffer = "ACTION:" + after
                                # ACTION lines are single-line; emit once complete
                                if "\n" in after or data.get("done"):
                                    action_line = "ACTION:" + after.split("\n")[0]
                                    yield action_line.strip()
                                    sentence_buffer = after.split("\n", 1)[1] if "\n" in after else ""
                                continue

                            # Yield complete sentences as they form
                            sents, sentence_buffer = self._extract_complete_sentences(sentence_buffer)
                            for s in sents:
                                yield s

                        if data.get("done"):
                            break

        except httpx.ConnectError:
            logger.warning("[STREAM-LLM] Connection lost")
            self._available = False
            self._checked = False
            yield "I'm having trouble connecting to my brain right now."
            return
        except asyncio.CancelledError:
            logger.info("[STREAM-LLM] Generation task cancelled")
            raise
        except Exception as e:
            logger.warning("[STREAM-LLM] Generation error: %s", e)
            yield personality.error_response()
            return

        # Flush any remaining buffer
        remainder = sentence_buffer.strip()
        if remainder and (cancel_event is None or not cancel_event.is_set()):
            if remainder.startswith("ACTION:"):
                yield remainder
            elif not remainder.lower().endswith(_ABBREVS):
                yield remainder

        if full_response.strip():
            conv_memory.add_assistant(full_response.strip())
        logger.info(
            "[STREAM-LLM] Response complete: %d chars in %.2fs",
            len(full_response), time.time() - t0)

    @staticmethod
    def _format_memory_answer(question: str, fact: str) -> str:
        """Turn a stored fact into a natural spoken answer."""
        q = question.lower()
        # "my project is Leo" + "what was my project called?" → "Your project is Leo."
        if "called" in q or "name" in q:
            fact = re.sub(r"^my\s+", "your ", fact, flags=re.IGNORECASE)
            return f"Your {fact.split('your ', 1)[-1]}." if "your " in fact else f"{fact.capitalize()}."
        fact = re.sub(r"^my\s+", "your ", fact, flags=re.IGNORECASE)
        return f"{fact.capitalize()}."


# Global singleton
streaming_llm = StreamingLLM()
