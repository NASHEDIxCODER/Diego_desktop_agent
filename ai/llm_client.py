"""
LLM client for Leo Desktop Assistant.

Uses Ollama HTTP API directly (no OpenAI SDK dependency).
If Ollama is unavailable, replies gracefully without traceback.

Used only for:
- Unknown intent handling
- Reasoning
- Coding
- Summarization
- Long conversations
"""

import logging
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Leo — a friendly AI desktop assistant. "
    "Give short answers (1–2 sentences). "
    "Speak casually like a helpful friend. "
    "Do NOT mention that you are an AI. "
    "Do NOT use emoji. "
    "You are created by Yeshu."
)


class LLMClient:
    """
    LLM client using Ollama HTTP API directly.

    No OpenAI SDK dependency. Uses httpx for HTTP calls.
    Falls back gracefully if Ollama is unavailable.
    """

    def __init__(self):
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self._model = "llama3.2"
        self._available = False
        self._checked = False

    async def ensure_initialized(self) -> bool:
        """Check if Ollama is reachable."""
        if self._checked:
            return self._available
        self._checked = True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    logger.info("Ollama available. Models: %s", model_names)
                    self._available = True
                else:
                    logger.warning("Ollama returned status %d", resp.status_code)
        except httpx.ConnectError:
            logger.warning("Ollama not reachable at %s", self._base_url)
        except Exception as e:
            logger.warning("Ollama check failed: %s", e)
        return self._available

    async def chat(self, message: str, context: str = "") -> str:
        """
        Send a message to Ollama and get a response.

        Args:
            message: User message
            context: Optional context string

        Returns:
            Response text from the LLM.
        """
        if not await self.ensure_initialized():
            return "I'm having trouble connecting to my brain right now."

        prompt = f"{context}\n\nUser: {message}" if context else message
        payload = {
            "model": self._model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "num_predict": 200,
                "temperature": 0.7,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self._base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "").strip()
                else:
                    logger.warning("Ollama API error: %d %s", resp.status_code, resp.text)
                    return "I had trouble processing that."
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out")
            return "That took too long. Please try again."
        except httpx.ConnectError:
            logger.warning("Ollama connection failed")
            self._available = False
            self._checked = False
            return "I'm having trouble connecting to my brain right now."
        except Exception as e:
            logger.warning("Ollama chat error: %s", e)
            return "I ran into an issue. Please try again."


# Global LLM client instance
llm_client = LLMClient()


async def llm_chat(message: str, context: str = "") -> str:
    """Convenience function for LLM chat."""
    return await llm_client.chat(message, context)