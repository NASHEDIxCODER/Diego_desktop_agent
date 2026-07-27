"""
LLM client for Leo Desktop Assistant.

Provides a unified interface to multiple LLM providers:
- Google Gemini (primary)
- OpenAI
- Anthropic
- OpenRouter
- Groq
- Ollama (local)

Used only for:
- Unknown intent handling
- Reasoning
- Coding
- Summarization
- Long conversations
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are Leo — a friendly AI desktop assistant.
Give short answers (1–2 sentences).
Speak casually like a helpful friend.
Do NOT mention that you are an AI.
Do NOT use emoji.
You are created by Yeshu.
"""


class LLMClient:
    """
    Unified LLM client with automatic provider selection.
    Falls back through providers if one is unavailable.
    """

    def __init__(self):
        self._provider = None
        self._client = None
        self._model = None
        self._available_providers: List[str] = []
        self._detect_providers()

    def _detect_providers(self) -> None:
        """Detect which LLM providers are configured."""
        if settings.GEMINI_API_KEY:
            self._available_providers.append("gemini")
        if settings.OPENAI_API_KEY:
            self._available_providers.append("openai")
        if settings.ANTHROPIC_API_KEY:
            self._available_providers.append("anthropic")
        if settings.OPENROUTER_API_KEY:
            self._available_providers.append("openrouter")
        if settings.GROQ_API_KEY:
            self._available_providers.append("groq")
        # Ollama is always available if the URL is reachable
        self._available_providers.append("ollama")

        if not self._available_providers:
            logger.warning("No LLM providers configured. "
                           "Set at least one API key in .env")

    async def _init_gemini(self) -> bool:
        """Initialize Google Gemini."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._client = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=SYSTEM_PROMPT,
            )
            self._model = "gemini-2.0-flash"
            logger.info("Initialized Gemini provider")
            return True
        except Exception as e:
            logger.error("Gemini init failed: %s", e)
            return False

    async def _init_openai(self) -> bool:
        """Initialize OpenAI."""
        try:
            import openai
            self._client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self._model = "gpt-4o-mini"
            logger.info("Initialized OpenAI provider")
            return True
        except Exception as e:
            logger.error("OpenAI init failed: %s", e)
            return False

    async def _init_ollama(self) -> bool:
        """Initialize Ollama (local)."""
        try:
            import openai as ollama_client
            self._client = ollama_client.AsyncOpenAI(
                base_url=f"{settings.OLLAMA_BASE_URL}/v1",
                api_key="ollama",  # Required but not used
            )
            self._model = "llama3.2"
            logger.info("Initialized Ollama provider at %s",
                        settings.OLLAMA_BASE_URL)
            return True
        except Exception as e:
            logger.error("Ollama init failed: %s", e)
            return False

    async def ensure_initialized(self) -> bool:
        """Ensure at least one provider is initialized."""
        if self._client is not None:
            return True

        init_methods = {
            "gemini": self._init_gemini,
            "openai": self._init_openai,
            "ollama": self._init_ollama,
        }

        for provider in self._available_providers:
            method = init_methods.get(provider)
            if method and await method():
                self._provider = provider
                return True

        return False

    async def chat(self, message: str, context: str = "") -> str:
        """
        Send a message to the LLM and get a response.

        Args:
            message: User message
            context: Optional context string (e.g., last intent)

        Returns:
            Response text from the LLM.
        """
        if not await self.ensure_initialized():
            return "I'm having trouble connecting to my brain right now."

        try:
            if self._provider == "gemini":
                prompt = f"{context}\n\nUser: {message}" if context else message
                response = self._client.generate_content(prompt)
                if hasattr(response, "text"):
                    return response.text.strip()
                return "I didn't get that."

            elif self._provider == "openai":
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                if context:
                    messages.append({"role": "system", "content": context})
                messages.append({"role": "user", "content": message})

                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=200,
                )
                return response.choices[0].message.content.strip()

            elif self._provider == "ollama":
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                if context:
                    messages.append({"role": "system", "content": context})
                messages.append({"role": "user", "content": message})

                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=200,
                )
                return response.choices[0].message.content.strip()

            return "I'm not sure how to respond to that."

        except Exception as e:
            logger.error("LLM chat error (%s): %s", self._provider, e)
            # Try fallback to next provider
            self._client = None
            self._provider = None
            if await self.ensure_initialized():
                return await self.chat(message, context)
            return "I'm having trouble thinking right now."


# Global LLM client instance
llm_client = LLMClient()


async def llm_chat(message: str, context: str = "") -> str:
    """Convenience function for LLM chat."""
    return await llm_client.chat(message, context)