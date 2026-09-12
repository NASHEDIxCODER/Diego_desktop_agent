"""
ReasoningModel — clean model-provider abstraction for Diego (Phase 21A).

The rest of Diego depends ONLY on this interface, never on one provider.

HARD CONTRACT:

  1. The model NEVER executes tools. It receives structured prompts /
     context and returns STRUCTURED JSON only. Every returned action is
     treated as a *proposal* that must pass the deterministic
     PlanValidator (action existence, authorization gate, parameter
     checks) before the runtime will even consider dispatching it.

  2. Every call is bounded: timeout, optional cancellation check, output
     reserve, and the existing ai.context_monitor guard (context-limit
     failure is an explicit status, never a silent truncation).

  3. Model failures fail SAFE: timeout / connection error / malformed
     JSON / context-limit are returned as explicit ReasoningResult
     statuses with data=None. Callers fall back to deterministic paths
     (PlanValidator + classify_failure + adjust_params_for_retry).

A model call may return one of:
    - plan               {"plan": [...], "assumptions": [...], "constraints": [...]}
    - reasoning update    {"keep_plan": bool, "revised_plan": [...], "reason": ...}
    - failure diagnosis   {"failure_kind": ..., "probable_cause": ...,
                           "retry_suitable": bool, "alternative": ...,
                           "next_strategy": "retry|repair_params|..."}
    - clarification req.  {"needs_input": true, "question": "..."}
    - final synthesis     {"answer": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT = 512


class ReasoningCallStatus(str, Enum):
    """Explicit call outcome. Anything other than OK means the caller
    must fall back to deterministic behavior."""
    OK = "ok"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    CONTEXT_LIMIT = "context_limit"
    INVALID_JSON = "invalid_json"
    CANCELLED = "cancelled"
    DISABLED = "disabled"


@dataclass
class ReasoningResult:
    """One bounded model call outcome (structured data only)."""
    status: ReasoningCallStatus = ReasoningCallStatus.DISABLED
    data: Optional[Dict[str, Any]] = None
    raw_text: str = ""
    error: str = ""
    model: str = ""
    estimated_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.status is ReasoningCallStatus.OK and self.data is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "error": self.error,
            "model": self.model,
            "has_data": self.data is not None,
        }


def extract_json(text: str) -> Any:
    """Robustly extract the first JSON object/array from model text.

    Handles code fences, leading prose, and trailing commentary. Raises
    ValueError when no parsable JSON value is present (callers convert
    that into the INVALID_JSON safe-failure status).
    """
    if not text:
        raise ValueError("empty model response")
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates: List[str] = [c.strip() for c in fenced if c.strip()]
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            if start < 0:
                continue
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        snippet = candidate[start:i + 1]
                        try:
                            return json.loads(snippet)
                        except (json.JSONDecodeError, ValueError):
                            break
    raise ValueError("no parsable JSON found in model response")


class BaseReasoningModel:
    """Provider-agnostic reasoning-model interface.

    Structured prompt + context in → structured JSON out. Bounded by
    timeout / cancellation / context guard. NEVER executes tools.
    """

    name: str = "base"

    async def reason(self, prompt: str, context: str = "", *,
                     timeout_s: float = DEFAULT_TIMEOUT_S,
                     cancel_check: Optional[Callable[[], bool]] = None,
                     max_output: int = DEFAULT_MAX_OUTPUT,
                     ) -> ReasoningResult:  # pragma: no cover - interface
        raise NotImplementedError

    # ── Structured helpers (thin prompt builders over `reason`) ──

    async def plan(self, goal: str, context: str = "", **kw) -> ReasoningResult:
        prompt = (
            "Create an executable step plan for the goal.\n"
            "Respond with ONLY JSON: "
            '{"plan": [{"action": "...", "params": {}, '
            '"description": "..."}], '
            '"assumptions": ["..."], "constraints": ["..."], '
            '"needs_input": false, "question": ""}\n'
            "Every action MUST come from the allowed actions listed in the "
            "context. You cannot execute anything yourself — you only propose "
            "steps that the runtime will validate and execute.\n\n"
            f"GOAL: {goal}\n\nCONTEXT:\n{context}"
        )
        return await self.reason(prompt, **kw)

    async def revise_plan(self, goal: str, observation: str,
                          completed: List[str],
                          remaining: List[Dict[str, Any]],
                          **kw) -> ReasoningResult:
        prompt = (
            "A plan step just finished and was VERIFIED. Given the new "
            "observation, decide whether the REMAINING plan is still valid.\n"
            "Respond with ONLY JSON: "
            '{"keep_plan": true|false, "revised_plan": [...], '
            '"reason": "..."}\n'
            "Revised steps must use the allowed actions from the context. "
            "Never repeat already-completed steps.\n\n"
            f"GOAL: {goal}\nLATEST OBSERVATION: {observation}\n"
            f"COMPLETED: {completed}\n"
            f"REMAINING PLAN: {json.dumps(remaining, ensure_ascii=False)}"
        )
        return await self.reason(prompt, **kw)

    async def diagnose(self, goal: str, action: str, observed_result: str,
                       error: str, attempt: int, **kw) -> ReasoningResult:
        prompt = (
            "A plan step failed. Diagnose it from the REAL observed result.\n"
            "Respond with ONLY JSON: "
            '{"failure_kind": "transient|wrong_params|wrong_tool|'
            'changed_state|unavailable_capability|impossible|unknown", '
            '"probable_cause": "...", "retry_suitable": true|false, '
            '"alternative": "...", '
            '"next_strategy": "retry|repair_params|observe_and_adapt|'
            'alternative_tool|replan|ask_user|stop"}\n'
            "Do NOT propose executing anything yourself.\n\n"
            f"GOAL: {goal}\nFAILED ACTION: {action} (attempt {attempt})\n"
            f"OBSERVED RESULT: {observed_result}\nERROR: {error}"
        )
        return await self.reason(prompt, **kw)

    async def reflect(self, goal: str, outcome: str, **kw) -> ReasoningResult:
        prompt = (
            "Reflect BRIEFLY on the finished task for reusable learning.\n"
            "Respond with ONLY JSON: "
            '{"goal_achieved": true|false, "what_worked": "...", '
            '"what_failed": "...", '
            '"lesson": {"type": "task_lesson|successful_strategy|'
            'failed_strategy|tool_preference|environment_state|'
            'user_preference|recurring_obstacle|verified_workaround", '
            '"task_pattern": "...", "lesson": "..."}, '
            '"confidence": 0.0-1.0}\n'
            "Only compact structured fields — no reasoning transcript.\n\n"
            f"GOAL: {goal}\nOUTCOME: {outcome}"
        )
        return await self.reason(prompt, **kw)


class OllamaReasoningModel(BaseReasoningModel):
    """Reasoning model over the local Ollama HTTP API (no cloud dependency).

    Uses the same base URL / keep-alive conventions as ai.llm_client and
    the same context_monitor guard + accounting. Falls back safely on
    every failure mode.
    """

    name = "ollama"

    # Preferred small reasoning-capable models (non-vision first).
    _PREFERRED = (
        "qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b",
        "llama3.2:3b", "llama3.2:1b",
        "phi3.5:mini", "phi3:mini", "gemma2:2b", "mistral:7b", "tinyllama",
    )

    def __init__(self, base_url: Optional[str] = None,
                 model: Optional[str] = None):
        self._base_url = (
            base_url or os.environ.get("DIEGO_OLLAMA_BASE_URL") or "").rstrip("/")
        if not self._base_url:
            try:
                from config.settings import settings
                self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
            except Exception:
                self._base_url = "http://localhost:11434"
        self._model = model or ""
        self._checked = False
        # Phase 21D: reusable HTTP client for connection pooling across the
        # plan/revise/reflect calls of a task. Lazily created on first use.
        self._client: Optional[Any] = None

    async def _get_client(self):
        """Return (and lazily create) the shared httpx client. Reuses the
        TCP/HTTP2 connection across calls within and across tasks."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT_S))
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client (call on shutdown)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_model(self) -> Optional[str]:
        """Auto-detect the best available local model (once)."""
        if self._checked:
            return self._model or None
        self._checked = True
        if self._model:
            return self._model
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code == 200:
                    names = [m.get("name", "")
                             for m in resp.json().get("models", [])]
                    for pref in self._PREFERRED:
                        if pref in names:
                            self._model = pref
                            break
                    if not self._model and names:
                        for n in names:
                            if not any(p in n.lower() for p in (
                                    "vl", "vision", "llava", "bakllava")):
                                self._model = n
                                break
        except Exception as e:
            logger.debug("[ReasoningModel] model detection failed: %s", e)
        return self._model or None

    async def reason(self, prompt: str, context: str = "", *,
                     timeout_s: float = DEFAULT_TIMEOUT_S,
                     cancel_check: Optional[Callable[[], bool]] = None,
                     max_output: int = DEFAULT_MAX_OUTPUT,
                     ) -> ReasoningResult:
        import httpx
        from ai.context_monitor import context_monitor

        if cancel_check is not None and cancel_check():
            return ReasoningResult(status=ReasoningCallStatus.CANCELLED)

        model = await self._ensure_model()
        if not model:
            return ReasoningResult(
                status=ReasoningCallStatus.DISABLED,
                error="no local reasoning model available")

        full_prompt = f"{prompt}\n\n{context}" if context else prompt

        # ── CONTEXT GUARD (existing monitor) ──────────────────
        ok, reason, est = context_monitor.pre_request_check(
            full_prompt, reserve_output=max_output)
        if not ok:
            logger.warning("[ReasoningModel] context guard refused: %s", reason)
            return ReasoningResult(
                status=ReasoningCallStatus.CONTEXT_LIMIT,
                error=reason, model=model, estimated_tokens=est)

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {"num_predict": max_output, "temperature": 0.2},
            "keep_alive": "10m",
        }
        try:
            from config.settings import settings
            payload["keep_alive"] = settings.OLLAMA_KEEP_ALIVE
        except Exception:
            pass

        try:
            # Phase 21D: reuse the shared client (connection pooling) instead
            # of creating a fresh httpx.AsyncClient per call.
            client = await self._get_client()
            resp = await client.post(
                f"{self._base_url}/api/generate", json=payload,
                timeout=timeout_s)
            if resp.status_code != 200:
                return ReasoningResult(
                    status=ReasoningCallStatus.MODEL_ERROR,
                    error=f"HTTP {resp.status_code}", model=model)
            data = resp.json()
            raw = (data.get("response") or "").strip()
            # ── CONTEXT MEASUREMENT (existing monitor) ────────
            context_monitor.record_request(
                model=model, prompt=full_prompt, max_output=max_output,
                provider_usage={
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                },
                response_text=raw,
            )
            if cancel_check is not None and cancel_check():
                return ReasoningResult(
                    status=ReasoningCallStatus.CANCELLED, model=model)
            try:
                parsed = extract_json(raw)
            except ValueError as e:
                return ReasoningResult(
                    status=ReasoningCallStatus.INVALID_JSON,
                    raw_text=raw[:400], error=str(e), model=model)
            if not isinstance(parsed, dict):
                return ReasoningResult(
                    status=ReasoningCallStatus.INVALID_JSON,
                    raw_text=raw[:400],
                    error="model JSON is not an object", model=model)
            return ReasoningResult(
                status=ReasoningCallStatus.OK, data=parsed,
                raw_text=raw[:400], model=model)
        except asyncio.TimeoutError:
            return ReasoningResult(
                status=ReasoningCallStatus.TIMEOUT,
                error=f"timed out after {timeout_s}s", model=model)
        except httpx.TimeoutException:
            return ReasoningResult(
                status=ReasoningCallStatus.TIMEOUT,
                error=f"HTTP timeout after {timeout_s}s", model=model)
        except httpx.ConnectError as e:
            return ReasoningResult(
                status=ReasoningCallStatus.MODEL_ERROR,
                error=f"connection failed: {e}", model=model)
        except Exception as e:
            return ReasoningResult(
                status=ReasoningCallStatus.MODEL_ERROR,
                error=str(e), model=model)


class DisabledReasoningModel(BaseReasoningModel):
    """Explicitly disabled model — every call returns DISABLED so callers
    take the deterministic path."""

    name = "disabled"

    async def reason(self, prompt: str, context: str = "", **kw
                     ) -> ReasoningResult:
        return ReasoningResult(
            status=ReasoningCallStatus.DISABLED,
            error="reasoning model disabled")


def _config_key() -> tuple:
    """Build a cache key from the current provider/model configuration.
    Changes to any of these invalidate the cached model instance."""
    setting = (os.environ.get("DIEGO_REASONING_MODEL", "") or
               "").strip().lower()
    base_url = (os.environ.get("DIEGO_OLLAMA_BASE_URL") or "").strip()
    try:
        from config.settings import settings
        base_url = base_url or settings.OLLAMA_BASE_URL.rstrip("/")
    except Exception:
        pass
    explicit = (os.environ.get("DIEGO_REASONING_MODEL_NAME") or "").strip()
    return (setting, base_url, explicit)


class _ModelCache:
    """Phase 21D: a single cached reasoning-model instance per distinct
    (provider/model/base_url) configuration. Avoids rebuilding the model
    and re-running `/api/tags` discovery on every task. Configuration
    changes invalidate the cache automatically."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._key: Optional[tuple] = None
        self._instance: Optional[BaseReasoningModel] = None

    async def get(self) -> BaseReasoningModel:
        key = _config_key()
        if self._instance is not None and self._key == key:
            return self._instance
        async with self._lock:
            # Re-check after acquiring the lock.
            if self._instance is not None and self._key == key:
                return self._instance
            await self._close_if_present()
            self._instance = self._create(key)
            self._key = key
            return self._instance

    def _create(self, key: tuple) -> BaseReasoningModel:
        setting = key[0]
        if setting in ("off", "none", "disabled", "0"):
            return DisabledReasoningModel()
        return OllamaReasoningModel()

    async def close(self) -> None:
        async with self._lock:
            await self._close_if_present()
            self._instance = None
            self._key = None

    async def _close_if_present(self) -> None:
        if self._instance is not None:
            closer = getattr(self._instance, "close", None)
            if callable(closer):
                try:
                    await closer()
                except Exception:
                    pass


_model_cache = _ModelCache()


def get_reasoning_model() -> BaseReasoningModel:
    """Factory: return the cached reasoning model for the current
    configuration. The instance is reused across tasks (one per distinct
    provider/model/base_url); configuration changes invalidate the cache.

    NOTE: this returns the cached instance synchronously when the loop is
    unavailable; the first call in an async context performs lazy creation.
    """
    key = _config_key()
    if _model_cache._instance is not None and _model_cache._key == key:
        return _model_cache._instance
    # First call or config changed: build synchronously (discovery is
    # deferred to the first model call, which caches the result).
    if key[0] in ("off", "none", "disabled", "0"):
        inst: BaseReasoningModel = DisabledReasoningModel()
    else:
        inst = OllamaReasoningModel()
    _model_cache._instance = inst
    _model_cache._key = key
    return inst


async def get_reasoning_model_async() -> BaseReasoningModel:
    """Async factory: returns the cached model, creating/closing under a
    lock so configuration changes never leak a stale client."""
    return await _model_cache.get()


async def close_reasoning_model() -> None:
    """Close the cached model client (call on shutdown)."""
    await _model_cache.close()
