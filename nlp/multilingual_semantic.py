"""
Phase 20C — Multilingual semantic prototype (ISOLATED, NOT wired to production).

ASR transcript -> semantic model -> StructuredIntent JSON -> (future:
existing intent authorization -> DecisionEngine/Brain -> autonomous pipeline).

SAFETY CONTRACT (hard rules):
  - This module ONLY interprets language into a strict structured
    representation. It NEVER executes actions, NEVER accesses tools,
    NEVER modifies task state, NEVER touches the dispatcher/planner,
    confirmation state, or routing.
  - Production routing (`core/command_router.py`, `core/decision_engine.py`,
    `agent/brain.py`, `nlp/command_normalizer.py`,
    `nlp/multilingual_lexicon.py`, `nlp/intent_authorizer.py`) is UNTOUCHED.
  - The model must NEVER return shell commands / Python / tool calls /
    executable instructions. Only {intent, entities, language}.
  - All model output is strictly validated; invalid output is rejected to
    an `unknown` intent (never executed, caller decides to ask/clarify).
  - Bounded timeout on every model call (default 15s, configurable).

Usage (prototype only):
    from nlp.multilingual_semantic import parse_command
    intent = parse_command("Firefox kholo")  # needs Ollama running
    # intent.intent == "open_application", intent.entities == {"application": "firefox"}
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ── Strict semantic schema ──────────────────────────────────────────

ALLOWED_INTENTS = frozenset({
    "open_application",
    "close_application",
    "play_media",
    "pause_media",
    "resume_media",
    "stop_media",
    "next_track",
    "previous_track",
    "search_web",
    "open_url",
    "open_file",
    "system_info",
    "volume_control",
    "brightness_control",
    "screenshot",
    "lock_screen",
    "conversation",
    "knowledge_question",
    "multi_step_task",
    "unknown",
})

ALLOWED_LANGUAGES = frozenset({"en", "hi", "mixed", "unknown"})

# Only these top-level keys may appear in model JSON.
ALLOWED_TOP_KEYS = frozenset({"intent", "entities", "language"})

# Entity keys we expect. Unknown string keys are tolerated ONLY if their
# values are plain strings without executable content (filenames, names,
# URLs, numbers-as-strings all pass through). Keys that look like code
# execution channels are always rejected.
FORBIDDEN_ENTITY_KEYS = frozenset({
    "shell_command", "shell", "command", "exec", "execute",
    "python", "code", "script", "tool_call", "tool", "action",
    "actions", "function", "function_call", "system",
})

# Substrings that must never appear in intent/entity values. Kept tight so
# legitimate entities ("test.py", "https://github.com/...", "Arijit Singh",
# "Believer") still pass.
FORBIDDEN_VALUE_PATTERNS = (
    "rm -rf", "rm ", "sudo ", "mkfs", "dd if=", ":(){", "chmod +x",
    "curl ", "wget ", "| sh", "| bash", "powershell", "cmd.exe",
    "import os", "import subprocess", "os.system", "subprocess.",
    "__import__", "eval(", "exec(",
)

_MAX_TEXT_CHARS = 500
_DEFAULT_TIMEOUT_S = 15.0
_DEFAULT_MODEL = "qwen2.5:3b"
_DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass
class StructuredIntent:
    """Strict structured representation of a user request. Data only."""

    intent: str = "unknown"
    entities: Dict[str, str] = field(default_factory=dict)
    language: str = "unknown"
    raw_text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    valid: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": dict(self.entities),
            "language": self.language,
        }


class SemanticParseError(Exception):
    """Raised internally when model output fails strict validation."""


def build_semantic_prompt(text: str) -> str:
    """Build the strict interpretation prompt. No tools, no actions."""
    intents = ", ".join(sorted(ALLOWED_INTENTS))
    return (
        "You interpret a desktop voice-assistant transcript into STRICT JSON only.\n"
        "Rules:\n"
        "- Output EXACTLY one JSON object with keys: intent, entities, language.\n"
        "- intent must be one of: " + intents + ".\n"
        "- entities must be an object of string->string (filenames, names, URLs,\n"
        "  queries preserved EXACTLY, never translated or transliterated away).\n"
        "- language must be one of: en, hi, mixed, unknown.\n"
        "  (en=English, hi=Hindi/Devanagari, mixed=Hinglish/code-switch, unknown=other/unclear)\n"
        "- NEVER output shell commands, Python, tool calls, ACTION lines, or\n"
        "  executable instructions. Only describe the request.\n"
        "- Ambiguous/unclear input -> {\"intent\": \"unknown\", \"entities\": {}, \"language\": \"unknown\"}.\n"
        "- No markdown, no comments, no extra text. JSON only.\n"
        f"Transcript: {text!r}\n"
        "JSON:"
    )


def _contains_forbidden_value(value: str) -> bool:
    low = value.lower()
    return any(pat in low for pat in FORBIDDEN_VALUE_PATTERNS)


def validate_structured_dict(data: Any) -> Dict[str, Any]:
    """Strictly validate raw model JSON. Returns normalized dict or raises."""
    if not isinstance(data, dict):
        raise SemanticParseError("top-level JSON must be an object")
    extra = set(data.keys()) - set(ALLOWED_TOP_KEYS)
    if extra:
        raise SemanticParseError(f"forbidden top-level keys: {sorted(extra)}")

    intent = data.get("intent", "unknown")
    entities = data.get("entities", {})
    language = data.get("language", "unknown")

    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise SemanticParseError(f"invalid intent: {intent!r}")
    if not isinstance(entities, dict):
        raise SemanticParseError("entities must be an object")
    if not isinstance(language, str) or language not in ALLOWED_LANGUAGES:
        raise SemanticParseError(f"invalid language: {language!r}")

    clean: Dict[str, str] = {}
    for k, v in entities.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise SemanticParseError("entity keys/values must be strings")
        kl = k.strip().lower()
        if kl in FORBIDDEN_ENTITY_KEYS:
            raise SemanticParseError(f"forbidden entity key: {k!r}")
        if _contains_forbidden_value(v):
            raise SemanticParseError(f"forbidden executable content in entity {k!r}")
        if len(k) > 64 or len(v) > 500:
            raise SemanticParseError("entity key/value too long")
        clean[k.strip()] = v.strip()
    if _contains_forbidden_value(intent):
        raise SemanticParseError("forbidden content in intent")
    return {"intent": intent, "entities": clean, "language": language}


def _extract_json_object(raw: str) -> str:
    """Extract first {...} JSON object from possibly noisy model output."""
    if not raw or not raw.strip():
        raise SemanticParseError("empty model output")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise SemanticParseError("no JSON object in model output")
    return raw[start:end + 1]


def _default_model_call(prompt: str, model: str, base_url: str,
                        timeout_s: float) -> str:
    """Call Ollama /api/generate with format=json. Sync; bounded timeout."""
    import httpx  # local import so unit tests never need network

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "5m",
        "options": {"temperature": 0.0, "num_predict": 220},
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{base_url.rstrip('/')}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()


def parse_command(
    text: str,
    model: str = _DEFAULT_MODEL,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    base_url: str = _DEFAULT_BASE_URL,
    model_call: Optional[Callable[[str], str]] = None,
) -> StructuredIntent:
    """Interpret `text` into a StructuredIntent. Never executes anything.

    Args:
        text: ASR transcript (any language).
        model: Ollama model name (prototype only; no production routing).
        timeout_s: bounded timeout for the model call.
        base_url: Ollama base URL.
        model_call: injectable stub for tests; takes prompt, returns raw text.
            When given, no network is used.

    Returns:
        StructuredIntent (valid=False + intent=unknown on any failure).
    """
    t0 = time.time()
    raw_text = text or ""
    clipped = raw_text.strip()[:_MAX_TEXT_CHARS]
    if not clipped:
        return StructuredIntent(intent="unknown", entities={},
                                language="unknown", raw_text=raw_text,
                                model=model, latency_ms=0.0, valid=False,
                                error="empty input")

    prompt = build_semantic_prompt(clipped)
    try:
        if model_call is not None:
            raw = model_call(prompt)
        else:
            raw = _default_model_call(prompt, model, base_url, timeout_s)
        obj_text = _extract_json_object(raw)
        try:
            data = json.loads(obj_text)
        except json.JSONDecodeError as e:
            raise SemanticParseError(f"invalid JSON: {e}") from e
        norm = validate_structured_dict(data)
        latency = (time.time() - t0) * 1000.0
        return StructuredIntent(intent=norm["intent"],
                                entities=norm["entities"],
                                language=norm["language"],
                                raw_text=raw_text, model=model,
                                latency_ms=latency, valid=True, error="")
    except SemanticParseError as e:
        latency = (time.time() - t0) * 1000.0
        logger.warning("[SEMANTIC] rejected model output for %r: %s", clipped, e)
        return StructuredIntent(intent="unknown", entities={},
                                language="unknown", raw_text=raw_text,
                                model=model, latency_ms=latency, valid=False,
                                error=str(e))
    except Exception as e:  # timeout / connection / unavailable
        latency = (time.time() - t0) * 1000.0
        err = f"{type(e).__name__}: {e}"
        # Distinguish timeout vs unavailable for tests/reporting.
        if "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower() \
                or "timed out" in str(e).lower():
            err = f"timeout: {e}"
        logger.warning("[SEMANTIC] model call failed for %r: %s", clipped, err)
        return StructuredIntent(intent="unknown", entities={},
                                language="unknown", raw_text=raw_text,
                                model=model, latency_ms=latency, valid=False,
                                error=err)
