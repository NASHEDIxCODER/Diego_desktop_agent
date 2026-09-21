"""GoalRuntime model layer — provider-agnostic routing tests (no network)."""

from __future__ import annotations

import pytest

from goalruntime.llm import (
    DEFAULT_MODELS, ModelRole, ModelRouter, needs_coder,
    needs_security_scope, needs_vision,
)


class RecordingProvider:
    """A provider-agnostic fake: records calls, returns fixed text."""

    name = "recording"

    def __init__(self, reply: str = "ok") -> None:
        self.calls: list = []
        self.reply = reply

    def complete(self, model, prompt, *, system="", images=None,
                 max_tokens=512, temperature=0.2, timeout=60.0) -> str:
        self.calls.append({"model": model, "prompt": prompt,
                           "images": images})
        return self.reply


def test_default_models_match_policy():
    assert DEFAULT_MODELS[ModelRole.PLANNER] == "qwen2.5:7b"
    assert DEFAULT_MODELS[ModelRole.VISION] == "qwen2.5-vl:3b"
    assert DEFAULT_MODELS[ModelRole.CODER] == "deepseek-coder-v2:16b"
    assert DEFAULT_MODELS[ModelRole.LIGHT] == "qwen2.5:3b"


def test_router_maps_roles_to_models():
    p = RecordingProvider()
    r = ModelRouter(provider=p)
    assert r.model_for(ModelRole.PLANNER).model == "qwen2.5:7b"
    assert r.model_for(ModelRole.VISION).model == "qwen2.5-vl:3b"
    assert r.model_for(ModelRole.CODER).model == "deepseek-coder-v2:16b"
    assert r.model_for(ModelRole.LIGHT).model == "qwen2.5:3b"


def test_router_is_provider_agnostic():
    p = RecordingProvider(reply="planned")
    r = ModelRouter(provider=p)
    out = r.invoke(ModelRole.PLANNER, "decompose this")
    assert out == "planned"
    assert p.calls[0]["model"] == "qwen2.5:7b"
    assert r.invocations()[0]["role"] == "planner"


def test_deterministic_routing_runs_BEFORE_the_model():
    p = RecordingProvider()
    r = ModelRouter(provider=p)
    out = r.decide("anything", resolver=lambda: "DETERMINISTIC",
                   role=ModelRole.LIGHT)
    assert out == "DETERMINISTIC"
    assert p.calls == []          # the model was NEVER invoked
    assert r.invocations() == []


def test_router_falls_back_to_model_when_resolver_returns_none():
    p = RecordingProvider(reply="llm-said")
    r = ModelRouter(provider=p)
    out = r.decide("hard question", resolver=lambda: None,
                   role=ModelRole.LIGHT)
    assert out == "llm-said"
    assert len(p.calls) == 1
    assert p.calls[0]["model"] == "qwen2.5:3b"


def test_model_override_per_role():
    p = RecordingProvider()
    r = ModelRouter(provider=p)
    r.set_model(ModelRole.PLANNER, "qwen2.5:14b-instruct")
    r.invoke(ModelRole.PLANNER, "x")
    assert p.calls[0]["model"] == "qwen2.5:14b-instruct"


def test_env_override(monkeypatch):
    monkeypatch.setenv("GOALRUNTIME_CODER_MODEL", "qwen2.5-coder:7b")
    r = ModelRouter(provider=RecordingProvider())
    assert r.model_for(ModelRole.CODER).model == "qwen2.5-coder:7b"


def test_deterministic_goal_kind_resolvers():
    assert needs_coder("generate a python calculator")
    assert needs_coder("write me a script")
    assert not needs_coder("open instagram")
    assert needs_security_scope("run a port scan on staging")
    assert not needs_security_scope("read my email")
    assert needs_vision("take a screenshot")
    assert not needs_vision("send a message")


def test_ollama_provider_fails_safe():
    """The default provider never raises when Ollama is absent."""
    from goalruntime.llm import OllamaProvider
    p = OllamaProvider(base_url="http://127.0.0.1:9")   # nothing listens
    out = p.complete("qwen2.5:3b", "hi", timeout=2.0)
    assert out == ""
