"""
LLM client for Diego Desktop Assistant.

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
    "You are Diego — a friendly AI desktop assistant. "
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
        self._model = None  # Will be auto-detected
        self._available = False
        self._checked = False
        self._model_names = []

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
                    logger.info("Ollama available. Models: %s", self._model_names)
                    self._available = True
                    # Auto-select model
                    self._select_model()
                else:
                    logger.warning("Ollama returned status %d", resp.status_code)
        except httpx.ConnectError:
            logger.warning("Ollama not reachable at %s", self._base_url)
        except Exception as e:
            logger.warning("Ollama check failed: %s", e)
        return self._available

    # Vision model name patterns — these are heavy and should only be
    # loaded when screen context is explicitly requested.
    _VISION_MODEL_PATTERNS = ("qwen2.5vl", "qwen2-vl", "llava", "bakllava",
                               "minicpm-v", "cogvlm", "fuyu", "paligemma",
                               "moondream", "llama3.2-vision")

    # Preferred small conversational/routing models (ordered by preference).
    _PREFERRED_SMALL_MODELS = (
        "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:0.5b",
        "qwen2:1.5b", "qwen2:0.5b",
        "llama3.2:1b", "llama3.2:3b",
        "phi3:mini", "phi3.5:mini",
        "gemma2:2b", "gemma2:9b",
        "mistral:7b", "tinyllama",
    )

    @classmethod
    def _is_vision_model(cls, name: str) -> bool:
        """True if the model name matches a known vision-model pattern."""
        lower = name.lower()
        return any(pat in lower for pat in cls._VISION_MODEL_PATTERNS)

    def _select_model(self) -> None:
        """
        Auto-select the best available model.

        Priority:
        1. Configured model from settings (if installed)
        2. First preferred small model that is installed
        3. First non-vision model that is installed
        4. First installed model (even if vision — last resort)
        5. Fallback to 'llama3.2'
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
                    logger.info("Using configured model: %s", self._model)
                    return

        # 2. First preferred small model that is installed
        for preferred in self._PREFERRED_SMALL_MODELS:
            for m in self._model_names:
                if m == preferred or m.startswith(f"{preferred}:"):
                    self._model = m
                    logger.info("Selected preferred small model: %s", self._model)
                    return

        # 3. First non-vision model
        for m in self._model_names:
            if not self._is_vision_model(m):
                self._model = m
                logger.info("Selected non-vision model: %s", self._model)
                return

        # 4. Last resort: first installed model (even if vision)
        self._model = self._model_names[0]
        logger.warning("No small/non-vision model found — "
                       "falling back to: %s (may be heavy)", self._model)

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
            "keep_alive": 0,
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