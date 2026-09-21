"""
GoalRuntime model layer — provider-agnostic LLM invocation with DETERMINISTIC
ROUTING BEFORE reasoning.

Roles (default models; each overridable via env):
    PLANNER  qwen2.5:7b             — goal decomposition, replanning
    VISION   qwen2.5-vl:3b          — visual perception (screenshots)
    CODER    deepseek-coder-v2:16b  — code generation / fixing
    LIGHT    qwen2.5:3b             — cheap classification decisions

Provider-agnostic: anything exposing ``complete(model, prompt, ...) -> str``
can be plugged in as a provider (Ollama ships as the default; OpenAI-style,
OpenRouter, or a test fake work identically).

Deterministic-first policy: ``ModelRouter.decide()`` never calls a provider
when a deterministic resolver produced a verdict — the LLM is only asked when
deterministic routing explicitly yields None.

Logging: [GOAL-LLM]
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class ModelRole(str, Enum):
    PLANNER = "planner"
    VISION = "vision"
    CODER = "coder"
    LIGHT = "light"


DEFAULT_MODELS: Dict[ModelRole, str] = {
    ModelRole.PLANNER: "qwen2.5:7b",
    ModelRole.VISION: "qwen2.5-vl:3b",
    ModelRole.CODER: "deepseek-coder-v2:16b",
    ModelRole.LIGHT: "qwen2.5:3b",
}

_ENV_KEYS = {
    ModelRole.PLANNER: "GOALRUNTIME_PLANNER_MODEL",
    ModelRole.VISION: "GOALRUNTIME_VISION_MODEL",
    ModelRole.CODER: "GOALRUNTIME_CODER_MODEL",
    ModelRole.LIGHT: "GOALRUNTIME_LIGHT_MODEL",
}


class LLMProvider(Protocol):
    """Minimal provider contract — provider-agnostic by design."""

    def complete(self, model: str, prompt: str, *,
                 system: str = "", images: Optional[list] = None,
                 max_tokens: int = 512, temperature: float = 0.2,
                 timeout: float = 60.0) -> str:
        ...


@dataclass
class ModelSpec:
    role: ModelRole
    model: str
    provider_name: str = "ollama"

    def __str__(self) -> str:  # pragma: no cover — display helper
        return f"{self.role.value}:{self.model}@{self.provider_name}"


class OllamaProvider:
    """Default local provider (Ollama HTTP API via httpx, fail-safe).

    Never raises: on any failure it returns "" and the router falls back to
    deterministic behaviour. Vision-capable models receive ``images`` as
    base64 data (no file paths cross the wire).
    """

    name = "ollama"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base_url = (base_url
                          or os.environ.get("OLLAMA_BASE_URL",
                                            "http://localhost:11434")
                          ).rstrip("/")

    def available(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def complete(self, model: str, prompt: str, *,
                 system: str = "", images: Optional[list] = None,
                 max_tokens: int = 512, temperature: float = 0.2,
                 timeout: float = 60.0) -> str:
        try:
            import httpx
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
                "options": {"num_predict": max_tokens,
                            "temperature": temperature},
            }
            if system:
                payload["system"] = system
            if images:
                payload["images"] = list(images)
            r = httpx.post(f"{self._base_url}/api/generate", json=payload,
                           timeout=timeout)
            if r.status_code == 200:
                return str(r.json().get("response", "")).strip()
            logger.warning("[GOAL-LLM] ollama status=%d model=%s",
                           r.status_code, model)
        except Exception as e:  # fail-safe: deterministic paths still work
            logger.warning("[GOAL-LLM] ollama unavailable (%s)", e)
        return ""


class ModelRouter:
    """role → (provider, model) with deterministic routing before reasoning.

    ``decide(prompt, resolver=...)``: ``resolver`` is a deterministic callable
    returning a decision or None. The LLM (role-selected) is invoked ONLY when
    the resolver returns None. This makes every cheap, unambiguous decision
    free of any model call at all.
    """

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider or OllamaProvider()
        self._models: Dict[ModelRole, str] = dict(DEFAULT_MODELS)
        for role, env_key in _ENV_KEYS.items():
            env_val = os.environ.get(env_key)
            if env_val:
                self._models[role] = env_val
        self._invocations: list = []

    # ── configuration ────────────────────────────────────────────

    def model_for(self, role: ModelRole) -> ModelSpec:
        return ModelSpec(role=role, model=self._models[role],
                         provider_name=getattr(self.provider, "name",
                                               "generic"))

    def set_model(self, role: ModelRole, model: str) -> None:
        self._models[role] = model

    def set_provider(self, provider: LLMProvider) -> None:
        self.provider = provider

    # ── deterministic routing BEFORE reasoning ───────────────────

    def decide(self, prompt: str, *,
               resolver: Optional[Callable[[], Optional[str]]] = None,
               role: ModelRole = ModelRole.LIGHT,
               system: str = "") -> str:
        """Deterministic resolver first; LIGHT-role LLM only as fallback."""
        if resolver is not None:
            deterministic = resolver()
            if deterministic:
                logger.debug("[GOAL-LLM] deterministic decision (no model call)")
                return deterministic
        return self.invoke(role, prompt, system=system, max_tokens=64)

    # ── reasoning ────────────────────────────────────────────────

    def invoke(self, role: ModelRole, prompt: str, *,
               system: str = "", images: Optional[list] = None,
               max_tokens: int = 512, temperature: float = 0.2,
               timeout: float = 60.0) -> str:
        spec = self.model_for(role)
        t0 = time.time()
        out = self.provider.complete(
            spec.model, prompt, system=system, images=images,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout)
        self._invocations.append({
            "role": role.value, "model": spec.model,
            "ms": round((time.time() - t0) * 1000, 1),
            "chars": len(out),
        })
        return out

    def invocations(self) -> list:
        """Model-call audit trail (deterministic tests assert on this)."""
        return list(self._invocations)


# ═══════════════════════════════════════════════════════════════════
# Deterministic reasoning helpers (no LLM needed for these)
# ═══════════════════════════════════════════════════════════════════

_CODING_HINT_RE = re.compile(
    r"\b(code|script|python|function|program|test|bug|fix|refactor|"
    r"calculator|class |def |import )\b", re.IGNORECASE)

_SECURITY_HINT_RE = re.compile(
    r"\b(scan|pentest|penetration|nmap|exploit|vulnerab|recon|"
    r"port scan|security test|audit)\b", re.IGNORECASE)

_VISION_HINT_RE = re.compile(
    r"\b(screenshot|screen|see|look|read the (?:screen|page)|what.?s on)\b",
    re.IGNORECASE)


def needs_coder(goal_text: str) -> bool:
    """Deterministic: does this goal need the CODER role?"""
    return bool(_CODING_HINT_RE.search(goal_text or ""))


def needs_security_scope(goal_text: str) -> bool:
    """Deterministic: does this goal look like a security-testing task?"""
    return bool(_SECURITY_HINT_RE.search(goal_text or ""))


def needs_vision(goal_text: str) -> bool:
    """Deterministic: does this goal reference visual perception?"""
    return bool(_VISION_HINT_RE.search(goal_text or ""))


__all__ = [
    "ModelRole", "ModelSpec", "LLMProvider", "OllamaProvider",
    "ModelRouter", "DEFAULT_MODELS",
    "needs_coder", "needs_security_scope", "needs_vision",
]
