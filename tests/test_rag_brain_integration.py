"""M-I acceptance: brain wiring (M-G) + interaction guards.

Covers the grounded RAG layer inside ``AgentBrain._generate_response``:
hybrid activation fast path, live-state-over-RAG guard, grounded ANSWER /
CLARIFICATION / ACTION emission through the EXISTING ``ACTION: {json}``
channel, ABSTAIN fall-through, and finally-resume of the indexer pause
(M-H).  Everything runs on injected fakes — no real index, no real LLM.
"""

import asyncio
import json

# ── shared fakes ────────────────────────────────────────────────

ks_mod = __import__("knowledge.service", fromlist=["knowledge_service"])


class _FakeKS:
    """knowledge_service stand-in (search/context + pause hooks)."""

    def __init__(self, results=None, context=""):
        self.results = list(results or [])
        self.context = context
        self.paused_calls = 0
        self.resumed_calls = 0

    def search(self, q, top_k=5):
        return list(self.results)

    def context_for_llm(self, q, top_k=4):
        return self.context

    def pause_indexing(self):
        self.paused_calls += 1

    def resume_indexing(self):
        self.resumed_calls += 1


class _Result:
    response = ""
    actions_executed = 0
    actions_failed = 0


def _brain():
    return __import__("agent.brain", fromlist=["agent_brain"]).agent_brain


def _patch_llm(monkeypatch, sentences):
    """Replace streaming_llm.generate with a fresh fake generator per call."""
    import agent.streaming_llm as sl

    async def fake_generate(text, screen_context="", web_context=None,
                            local_context=None):
        for s in sentences:
            yield s

    monkeypatch.setattr(sl.streaming_llm, "generate", fake_generate)


def _patch_llm_boom(monkeypatch):
    """Any LLM call on this path is a test failure."""
    import agent.streaming_llm as sl

    async def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called on this path")
        yield  # pragma: no cover

    monkeypatch.setattr(sl.streaming_llm, "generate", boom)


def _hit(text, score=0.3, path="/home/user/notes/audio.md"):
    return {"score": score, "doc_path": path, "filename": "audio.md",
            "locator": "line 42", "text": text}


# ── M-G: hybrid activation gate ─────────────────────────────────

def test_fast_path_never_touches_llm_or_index(monkeypatch):
    monkeypatch.setattr(ks_mod, "knowledge_service",
                        _FakeKS(results=[_hit("spam", score=0.99)]))
    _patch_llm_boom(monkeypatch)
    out = asyncio.run(_brain()._rag_structured_response(
        "Click the notification button.", "", None))
    assert out is None  # fast path — orchestrator never entered


def test_live_state_question_never_reaches_rag_layer(monkeypatch):
    class _BoomKS:
        def search(self, *a, **k):
            raise AssertionError("index must never answer live questions")

        def context_for_llm(self, *a, **k):
            raise AssertionError("index must never answer live questions")

    monkeypatch.setattr(ks_mod, "knowledge_service", _BoomKS())
    _patch_llm_boom(monkeypatch)
    out = asyncio.run(_brain()._rag_structured_response(
        "What is my current browser tab?", "", None))
    assert out is None  # answered by live perception, not the index


# ── M-G: structured results ─────────────────────────────────────

def test_brain_rag_answer_grounds_and_speaks(monkeypatch):
    monkeypatch.setattr(ks_mod, "knowledge_service",
                        _FakeKS(results=[_hit(
                            "AudioBackend wraps portaudio for capture.",
                            score=0.3)]))
    _patch_llm(monkeypatch, [json.dumps({
        "type": "ANSWER",
        "answer": "AudioBackend wraps portaudio.",
        "evidence_ids": ["ev00"]})])
    resp = asyncio.run(_brain()._generate_response(
        "what does the AudioBackend use for capture?", None, _Result()))
    assert "AudioBackend wraps portaudio" in resp


def test_brain_rag_clarification_returned(monkeypatch):
    monkeypatch.setattr(ks_mod, "knowledge_service",
                        _FakeKS(results=[_hit("file notes", score=0.3)]))
    _patch_llm(monkeypatch, [json.dumps({
        "type": "CLARIFICATION",
        "question": "Which AudioBackend file do you mean?"})])
    resp = asyncio.run(_brain()._generate_response(
        "Open the AudioBackend implementation.", None, _Result()))
    assert resp == "Which AudioBackend file do you mean?"


def test_brain_rag_action_emits_validated_action_line(monkeypatch):
    monkeypatch.setattr("core.tool_registry.CAPABILITIES",
                        {"editor.open": {"description": "open a file"}})
    monkeypatch.setattr(ks_mod, "knowledge_service",
                        _FakeKS(results=[_hit(
                            "implementation lives in voice/audio_backend.py",
                            score=0.3)]))
    _patch_llm(monkeypatch, [json.dumps({
        "type": "ACTION", "goal": "Open AudioBackend",
        "capability": "editor.open",
        "arguments": {"path": "voice/audio_backend.py"},
        "evidence_ids": ["ev00"], "requires_confirmation": False})])
    resp = asyncio.run(_brain()._generate_response(
        "Where did we implement AudioBackend? Open it.", None, _Result()))
    assert "ACTION:" in resp
    payload = json.loads(resp.split("ACTION:", 1)[1].strip())
    assert payload["action"] == "editor.open"
    assert payload["params"]["path"] == "voice/audio_backend.py"
    assert payload["capability"] == "editor.open"


def test_brain_rag_unregistered_capability_never_emitted(monkeypatch):
    monkeypatch.setattr("core.tool_registry.CAPABILITIES",
                        {"editor.open": {"description": "open a file"}})
    monkeypatch.setattr(ks_mod, "knowledge_service",
                        _FakeKS(results=[_hit("notes", score=0.3)]))
    _patch_llm(monkeypatch, [
        json.dumps({"type": "ACTION", "goal": "x",
                    "capability": "evil.drop",
                    "arguments": {"path": "/etc/passwd"},
                    "evidence_ids": []}),
        "Here is a safe fallback instead.",
    ])
    resp = asyncio.run(_brain()._generate_response(
        "Where did we implement AudioBackend? Open it.", None, _Result()))
    assert "ACTION:" not in resp          # unregistered → never emitted
    assert "safe fallback" in resp        # fell through to plain LLM


def test_brain_rag_abstain_falls_through_to_streaming(monkeypatch):
    monkeypatch.setattr(ks_mod, "knowledge_service",
                        _FakeKS(results=[_hit("weak evidence", score=0.3)]))
    _patch_llm(monkeypatch, ["I don't have enough evidence for that."])
    resp = asyncio.run(_brain()._generate_response(
        "what does the AudioBackend use for capture?", None, _Result()))
    assert resp == "I don't have enough evidence for that."


# ── M-H: indexer pause is finally-resumed on every path ─────────

def test_generate_response_resumes_indexing_after_answer(monkeypatch):
    ks = _FakeKS(results=[_hit("grounded fact", score=0.3)])
    monkeypatch.setattr(ks_mod, "knowledge_service", ks)
    _patch_llm(monkeypatch, [json.dumps({
        "type": "ANSWER", "answer": "A grounded fact.",
        "evidence_ids": ["ev00"]})])
    asyncio.run(_brain()._generate_response(
        "what does the AudioBackend use?", None, _Result()))
    assert ks.paused_calls >= 1
    assert ks.resumed_calls >= ks.paused_calls


def test_generate_response_resumes_indexing_after_smalltalk_skip(
        monkeypatch):
    ks = _FakeKS(results=[])
    monkeypatch.setattr(ks_mod, "knowledge_service", ks)
    _patch_llm(monkeypatch, ["Tell me a joke…"])
    brain = _brain()
    import types
    monkeypatch.setattr(
        brain, "_last_intent_authorization",
        types.SimpleNamespace(
            category=types.SimpleNamespace(value="CONVERSATIONAL")),
        raising=False)
    asyncio.run(brain._generate_response(
        "tell me a joke", None, _Result()))
    # LookupError path inside the try — finally must still resume.
    assert ks.paused_calls >= 1
    assert ks.resumed_calls >= ks.paused_calls
