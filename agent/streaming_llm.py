"""
StreamingLLM — Token-streaming LLM client for conversational Diego.

Replaces the blocking `llm_client.chat()` with a streaming pipeline:
  - Streams tokens from Ollama as they are generated
  - Splits tokens into sentences as they arrive
  - Yields complete sentences immediately for TTS
  - Supports cancellation (for interruption)
  - Integrates conversation memory + personality

This enables the "begin speaking after the first meaningful sentence"
behavior that makes Diego feel like ChatGPT Voice / Gemini Live.

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


# Diego's conversational system prompt — makes Diego an autonomous desktop agent
DIEGO_SYSTEM_PROMPT = """You are Diego — an autonomous desktop agent that lives on the user's computer.

CORE IDENTITY:
- You are NOT a chatbot. You are a desktop agent that sees, thinks, and acts.
- You continuously perceive the desktop, understand projects, and take initiative.
- You remember everything: projects, folders, repos, preferences, habits, people.
- You learn from every interaction and get better over time.
- You never give up. If something fails, you diagnose, retry, and find alternatives.
- You are proactive: you notice build failures, git conflicts, errors, and offer help.

PERSONALITY:
- Speak casually like a friend sitting next to the user.
- Keep responses SHORT and natural. 1-3 sentences usually.
- Never say "How may I assist you?" or any robotic phrase.
- Never say "Task completed." or just "Done." — use natural confirmation instead.
- Never mention you are an AI, a language model, or your training.
- Never use emoji or markdown in speech.
- Vary your language. Never repeat the same phrase twice in a conversation.
- Match the user's tone: if they're frustrated, acknowledge it; if happy, be upbeat.
- Use natural acknowledgments: "Sure.", "On it.", "One sec.", "Got it.", "Will do."
- When you already know something, say so: "I already know that." or "I remember."
- Ask follow-up questions ONLY when genuinely needed, not as filler.

AUTONOMOUS BEHAVIOR:
- When given a complex task, break it down and execute step by step.
- Verify each step before moving to the next.
- If a step fails, try an alternative approach automatically.
- Report progress naturally: "Step 1 done, moving to step 2."
- For long tasks, give brief status updates without being chatty.
- You can chain multiple ACTION lines for multi-step workflows.
- Use the desktop context to understand what the user is working on.
- Reference past conversations and learned facts naturally.

CAPABILITIES:
- Open any application, navigate the file system, control the browser.
- Read and understand what's on screen (not just OCR — you understand UI).
- Play music from any source (Spotify, YouTube, MPV, VLC, local files).
- Control system volume, brightness, lock screen, power.
- Search the web and summarize results.
- Execute terminal commands, read output, debug errors.
- Navigate codebases, understand project structure, run tests.
- Monitor for build failures, git conflicts, docker issues, low battery.

ACTION FORMAT (only when a desktop action is needed):
ACTION: {"action": "<action_name>", "params": {...}}

Available actions:
- desktop_open(app) — open an application
- open_folder(path) — open a folder in the file manager
- browser_navigate(url) — open a website
- browser_search(query) — google search
- read_screen() — describe what is on screen
- click_text(text) — click visible text / a button
- scroll(direction) — scroll up/down
- key_press(key) — press a key
- type_text(text) — type text
- play_media(query) — play music/video (auto-detects best provider)
- music_pause() / music_resume() / music_next() / music_previous() / music_stop()
- music_shuffle() / music_repeat() / music_volume(percent) / music_mute()
- music_status() — what's currently playing
- volume_up() / volume_down() / volume_set(percent) / volume_mute()
- brightness_up() / brightness_down() / brightness_set(percent)
- lock_screen() / shutdown() / restart()

DESKTOP CONTEXT (you receive this automatically every turn):
- [Desktop: ...] shows focused window, git branch, terminal path, browser tab
- [Music: ...] shows what's currently playing
- [Screen context: ...] shows what's visible on screen with UI element inventory
- [Relevant context: ...] shows learned user facts, preferences, habits, project info
- Use this context silently. Don't announce it unless asked.

RESPONSE STYLE:
- Simple chat: just answer naturally, no ACTION line.
- Single action: short confirmation + one ACTION line. Example:
    "Sure, opening VS Code now.
    ACTION: {"action": "desktop_open", "params": {"app": "code"}}"
- Multi-step: brief plan + one ACTION per step. Example:
    "I'll open the project, start docker, and run the tests.
    ACTION: {"action": "desktop_open", "params": {"app": "code"}}
    ACTION: {"action": "desktop_open", "params": {"app": "gnome-terminal"}}"
- Music: just play it. "Playing lofi hip hop."
    ACTION: {"action": "play_media", "params": {"query": "lofi hip hop coding"}}
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
        self._vision_model: Optional[str] = None
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
                    self._vision_model = self._select_vision_model()
                else:
                    logger.warning("[STREAM-LLM] Ollama returned status %d", resp.status_code)
        except httpx.ConnectError:
            logger.warning("[STREAM-LLM] Ollama not reachable at %s", self._base_url)
        except Exception as e:
            logger.warning("[STREAM-LLM] Ollama check failed: %s", e)
        return self._available

    # Vision model name patterns — these are heavy and should only be
    # loaded when screen context is explicitly requested.
    _VISION_MODEL_PATTERNS = ("qwen2.5vl", "qwen2-vl", "llava", "bakllava",
                               "minicpm-v", "cogvlm", "fuyu", "paligemma",
                               "moondream", "llama3.2-vision")

    # Preferred small conversational/routing models (ordered by preference).
    # These are fast, lightweight, and won't OOM alongside STT/TTS/Face Auth.
    _PREFERRED_SMALL_MODELS = (
        "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:0.5b",
        "qwen2:1.5b", "qwen2:0.5b",
        "llama3.2:1b", "llama3.2:3b",
        "phi3:mini", "phi3.5:mini",
        "gemma2:2b", "gemma2:9b",
        "mistral:7b", "tinyllama",
    )
    TEXT_MODEL = "qwen2.5:7b"
    VISION_MODEL = "qwen2.5vl:3b"

    @classmethod
    def _is_vision_model(cls, name: str) -> bool:
        """True if the model name matches a known vision-model pattern."""
        lower = name.lower()
        return any(pat in lower for pat in cls._VISION_MODEL_PATTERNS)

    def _needs_vision(self, user_text: str) -> bool:
        """Return True when the user is explicitly asking Diego to see the screen."""
        text = user_text.lower().strip()

        vision_phrases = (
            "read my screen",
            "read the screen",
            "what's on my screen",
            "what is on my screen",
            "what do you see",
            "look at my screen",
            "look at the screen",
            "what am i looking at",
            "what is this on my screen",
            "where is the button",
            "where is the run button",
            "find the button",
            "find this button",
            "read this",
            "read that",
            "what error is shown",
            "what error do you see",
            "what's wrong on my screen",
            "what is wrong on my screen",
        )

        return any(phrase in text for phrase in vision_phrases)


    def _select_model(self) -> None:
        """Auto-select the best available model.

        Priority:
          1. Configured model from settings (if installed).
          2. First preferred small model that is installed.
          3. First non-vision model that is installed.
          4. First installed model (even if vision — last resort).
          5. Fallback to 'llama3.2'.
        """
        if not self._model_names:
            self._model = "llama3.2"
            return

        # 1. Configured model
        configured = getattr(settings, 'OLLAMA_MODEL', None) or getattr(settings, 'LLM_MODEL', None)
        if configured:
            for m in self._model_names:
                if m == configured or m.startswith(f"{configured}:"):
                    self._model = m
                    logger.info("[STREAM-LLM] Using configured model: %s", self._model)
                    return

        # 2. First preferred small model that is installed
        for preferred in self._PREFERRED_SMALL_MODELS:
            for m in self._model_names:
                if m == preferred or m.startswith(f"{preferred}:"):
                    self._model = m
                    logger.info("[STREAM-LLM] Selected preferred small model: %s", self._model)
                    return

        # 3. First non-vision model
        for m in self._model_names:
            if not self._is_vision_model(m):
                self._model = m
                logger.info("[STREAM-LLM] Selected non-vision model: %s", self._model)
                return

        # 4. Last resort: first installed model (even if vision)
        self._model = self._model_names[0]
        logger.warning("[STREAM-LLM] No small/non-vision model found — "
                       "falling back to: %s (may be heavy)", self._model)

    def _select_vision_model(self) -> Optional[str]:
        """Select the best installed vision-capable Ollama model."""
        if not self._model_names:
            return None

        preferred = (
            "qwen2.5vl:3b",
            "qwen2.5vl:7b",
            "qwen2.5vl",
            "qwen2-vl",
            "llama3.2-vision",
            "minicpm-v",
            "llava",
            "bakllava",
            "moondream",
        )

        for wanted in preferred:
            for installed in self._model_names:
                if installed == wanted or installed.startswith(f"{wanted}:"):
                    logger.info(
                        "[STREAM-LLM] Vision model selected: %s",
                        installed,
                    )
                    return installed

        for installed in self._model_names:
            if self._is_vision_model(installed):
                logger.info(
                    "[STREAM-LLM] Vision model fallback selected: %s",
                    installed,
                )
                return installed

        logger.warning("[STREAM-LLM] No vision model installed")
        return None


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
                # Not a real sentence end; reattach remainder and stop
                rest = stripped + " " + rest
                break
            sentences.append(stripped)
        return sentences, rest

    async def generate(
        self,
        user_text: str,
        cancel_event: Optional[asyncio.Event] = None,
        screen_context: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response as complete sentences.

        Args:
            user_text: What the user said.
            cancel_event: If set, generation stops immediately (interruption).
            screen_context: Optional description of what's currently on screen.
                Injected into the prompt so Diego can answer "what is going on"
                or act on "click here" requests.

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

        prompt_parts = [DIEGO_SYSTEM_PROMPT]
        if screen_context:
            prompt_parts.append(f"\nScreen context:\n{screen_context}")
        if context:
            prompt_parts.append(f"\nContext:\n{context}")
        prompt_parts.append(f"\nUser: {user_text}\nDiego:")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": 0,
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
        # "my project is Diego" + "what was my project called?" → "Your project is Diego."
        if "called" in q or "name" in q:
            fact = re.sub(r"^my\s+", "your ", fact, flags=re.IGNORECASE)
            return f"Your {fact.split('your ', 1)[-1]}." if "your " in fact else f"{fact.capitalize()}."
        fact = re.sub(r"^my\s+", "your ", fact, flags=re.IGNORECASE)
        return f"{fact.capitalize()}."

    async def vision_generate(
            self,
            prompt: str,
            image_bytes: bytes,
            model: Optional[str] = None,
    ) -> str:
        """
        Send an actual screenshot to a vision-capable Ollama model.
        """

        if not await self.ensure_initialized():
            return ""

        vision_model = model or self._select_vision_model()

        if not vision_model:
            logger.warning("[VISION-LLM] No vision model available")
            return ""

        import base64

        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 220,
            },
        }

        logger.info(
            "[VISION-LLM] model=%s image_bytes=%d",
            vision_model,
            len(image_bytes),
        )

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                answer = (data.get("response") or "").strip()

                logger.info(
                    "[VISION-LLM] complete model=%s response_chars=%d",
                    vision_model,
                    len(answer),
                )

                return answer

        except Exception as exc:
            logger.warning(
                "[VISION-LLM] failed model=%s error=%s",
                vision_model,
                exc,
            )
            return ""


# Global singleton
streaming_llm = StreamingLLM()