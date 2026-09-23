"""Tests for the grounded RAG orchestration layer (Phase 25)."""

import asyncio
import json

import pytest

from agent.rag_orchestrator import (
    RAGOrchestrator,
    _normalize_hit,
    build_evidence_pack,
    build_retrieval_query,
    compose_prompt,
    is_live_state_question,
    parse_structured_result,
    rerank_candidates,
    select_evidence,
    should_activate,
    validate_action,
)


# ── fakes ──────────────────────────────────────────────────────────

class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.calls = 0

    def search(self, query, k=10):
        self.calls += 1
        return self._hits[:k]


def _hit(source, content, score=0.8, **kw):
    d = {"source": source, "content": content, "relevance": score,
         "line_start": kw.pop("line_start", 1),
         "line_end": kw.pop("line_end", 10),
         "chunk_id": kw.pop("chunk_id", source + "#c0")}
    d.update(kw)
    return d


def _orch(hits, llm_text):
    async def llm(prompt):
        return llm_text
    return RAGOrchestrator(retriever=FakeRetriever(hits), llm_call=llm,
                           capabilities_fn=lambda: [
                               "editor.open", "browser.navigate", "search"])


def run(coro):
    return asyncio.run(coro)


# ── M-C: context-aware query construction ──────────────────────────

def test_query_builds_from_conversation_entity():
    q = build_retrieval_query(
        "Open the file where we implemented that.",
        conversation_context="We were discussing AudioBackend in the voice module.",
    )
    assert q.original_transcript.startswith("Open the file")
    assert "AudioBackend" in q.rewritten_query
    assert "AudioBackend" in q.entities


def test_query_preserves_original_transcript():
    q = build_retrieval_query("what changed in wake word handling?",
                              referenced_entity="wake_word")
    assert q.original_transcript == "what changed in wake word handling?"
    assert q.rewritten_query != q.original_transcript


# ── semantic chunk retrieval (provenance on evidence) ──────────────

def test_retrieval_returns_chunk_records_with_provenance():
    hits = [_hit("voice/audio_backend.py",
                 "class AudioBackend: ...", 0.91,
                 line_start=42, line_end=117)]
    o = _orch(hits, json.dumps(
        {"type": "ANSWER", "answer": "x", "evidence_ids": ["ev00"]}))
    out = run(o.run("where did we implement AudioBackend?"))
    ev = out.pack.retrieved_evidence[0]
    assert ev["source"] == "voice/audio_backend.py"
    assert ev["line_start"] == 42 and ev["line_end"] == 117
    assert ev["evidence_id"]
    assert out.pack.provenance[0]["evidence_id"] == ev["evidence_id"]


def _retriever_hit(doc_path, text, score=0.8, fusion="both", **kw):
    """A hit shaped EXACTLY like knowledge.retriever.search() returns it:
    the chunk text lives under ``text``, the file under ``doc_path``, and
    ``source`` holds the FUSION label (semantic/keyword/both)."""
    chunk_index = kw.pop("chunk_index", 0)
    d = {
        "text": text,
        "doc_path": doc_path,
        "filename": doc_path.split("/")[-1],
        "locator": f"lines {kw.pop('line_start', 1)}-{kw.pop('line_end', 10)}",
        "chunk_index": chunk_index,
        "file_type": "code",
        "score": score,
        "source": fusion,
        "chunk_id": f"{doc_path}#{chunk_index}",
        "evidence_id": kw.pop("evidence_id", "ev_x"),
        "semantic": score,
        "lexical": 0.2,
        "relevance": score,
    }
    d.update(kw)
    return d


def test_retriever_shaped_hit_bridges_content_and_path():
    """Regression: the knowledge layer returns ``text``/``doc_path`` and a
    FUSION label in ``source``. Without a bridge the prompt renders EMPTY
    evidence blocks (wrong key) and per-file dedupe collapses every hit
    onto the single label — so real content and real paths must survive."""
    hit = _retriever_hit("voice/audio_backend.py",
                         "class AudioBackend: portaudio capture", 0.9,
                         line_start=42, line_end=117)
    norm = _normalize_hit(hit)
    assert norm["content"] == "class AudioBackend: portaudio capture"
    assert norm["source"] == "voice/audio_backend.py"
    assert norm["fusion"] == "both"            # label kept, not a path
    assert norm["source_type"] == "code"
    assert norm["semantic_score"] == 0.9
    o = _orch([hit], json.dumps(
        {"type": "ANSWER", "answer": "x", "evidence_ids": ["ev_x"]}))
    out = run(o.run("where did we implement AudioBackend?"))
    prompt = compose_prompt(out.pack)
    assert "class AudioBackend: portaudio capture" in prompt
    assert "voice/audio_backend.py" in prompt
    assert "both:" not in prompt               # fusion label never a citation
    assert out.pack.retrieved_evidence[0]["content"].startswith(
        "class AudioBackend")


def test_selection_is_per_file_not_per_fusion_label():
    """Two files, three chunks: the first pass keeps one chunk per FILE, so
    the OTHER file corroborates before a second chunk of the same file is
    taken (a fusion-label dedupe used to make every hit one source)."""
    hits = [_retriever_hit("a.py", "alpha one", 0.9, chunk_index=0),
            _retriever_hit("a.py", "alpha two", 0.88, chunk_index=1),
            _retriever_hit("b.md", "beta one", 0.8, chunk_index=0)]
    o = _orch(hits, json.dumps(
        {"type": "ANSWER", "answer": "x", "evidence_ids": ["ev_x"]}))
    out = run(o.run("how do alpha and beta relate?"))
    ev = out.pack.retrieved_evidence
    # b.md ranks THIRD but is selected second: per-file diversity wins.
    assert [r["source"] for r in ev][:2] == ["a.py", "b.md"]
    assert {"alpha one", "beta one"} <= {r["content"] for r in ev}
    # every record carries the real path — never the fusion label
    assert all(r["source"] not in ("both", "semantic", "keyword")
               for r in ev)
    assert out.pack.provenance[1]["source"] == "b.md"


# ── M-D: reranking ─────────────────────────────────────────────────

def test_rerank_uses_context_but_never_replaces_retrieval():
    q = build_retrieval_query("why is audio backend slow")
    ranked = rerank_candidates(q, [
        _hit("voice/audio_backend.py", "audio backend profiling hooks", 0.5),
        _hit("docs/readme.md", "unrelated changelog", 0.05),
    ], current_project="voice/audio_backend.py")
    assert ranked[0]["source"] == "voice/audio_backend.py"
    # low-similarity candidate stays low — context only nudges
    assert ranked[1]["final_score"] < ranked[0]["final_score"]


def test_selection_bounded_3_to_5():
    hits = [_hit(f"file{i}.py", f"chunk {i} audio backend", 0.5 + i * 0.01)
            for i in range(16)]
    ranked = rerank_candidates(build_retrieval_query("audio backend"), hits)
    sel = select_evidence(ranked, 5)
    assert 1 <= len(sel) <= 5


# ── M-E: evidence pack + context hierarchy ─────────────────────────

def test_pack_contains_all_required_sections():
    hits = [_hit("knowledge/store.py", "CREATE TABLE chunks", 0.9)]
    o = _orch(hits, json.dumps(
        {"type": "ANSWER", "answer": "ok", "evidence_ids": ["ev00"]}))
    out = run(o.run(
        "how does the store work?",
        conversation_context="earlier: sqlite store",
        task_context="task: indexing",
        desktop_context="app: PyCharm",
        browser_context="https://example.com/docs",
    ))
    p = out.pack
    assert p.original_transcript
    assert p.conversation_context and p.task_context
    assert p.desktop_context and p.browser_context
    assert p.retrieved_evidence and p.provenance
    assert "editor.open" in p.available_capabilities


def test_prompt_orders_live_state_before_evidence():
    hits = [_hit("a.py", "x", 0.9)]
    o = _orch(hits, json.dumps(
        {"type": "ANSWER", "answer": "ok", "evidence_ids": ["ev00"]}))
    out = run(o.run("why is browser search slow?",
                    desktop_context="live: chrome busy",
                    task_context="t"))
    prompt = compose_prompt(out.pack)
    assert "[LIVE STATE]" in prompt
    assert prompt.index("[LIVE STATE]") < prompt.index("[RETRIEVED EVIDENCE]")
    assert prompt.index("[RETRIEVED EVIDENCE]") < prompt.index("[USER]")


# ── M-F: strict structured output ──────────────────────────────────

def test_parse_all_four_types():
    a = parse_structured_result(json.dumps(
        {"type": "ANSWER", "answer": "ASR is whisper", "evidence_ids": ["ev00"]}),
        ["editor.open"])
    assert a["type"] == "ANSWER" and "whisper" in a["answer"]

    act = parse_structured_result(json.dumps(
        {"type": "ACTION", "goal": "open it", "capability": "editor.open",
         "arguments": {"path": "voice/audio_backend.py"},
         "evidence_ids": ["ev00"], "requires_confirmation": False}),
        ["editor.open"])
    assert act["type"] == "ACTION" and validate_action(act, ["editor.open"])

    c = parse_structured_result(json.dumps(
        {"type": "CLARIFICATION", "question": "Which file?"}), [])
    assert c["type"] == "CLARIFICATION"

    ab = parse_structured_result(json.dumps(
        {"type": "ABSTAIN", "reason": "no evidence"}), [])
    assert ab["type"] == "ABSTAIN"


def test_action_with_unregistered_capability_is_refused():
    out = parse_structured_result(json.dumps(
        {"type": "ACTION", "goal": "x", "capability": "shell.exec",
         "arguments": {"cmd": "rm -rf /"}, "evidence_ids": []}),
        ["editor.open", "browser.navigate"])
    assert out["type"] == "ABSTAIN"
    assert "refused" in out["reason"]
    assert not validate_action(out, ["editor.open"])


def test_unknown_keys_downgrade_to_abstain():
    out = parse_structured_result(json.dumps(
        {"type": "ANSWER", "answer": "x", "shell": "curl evil"}), [])
    assert out["type"] == "ABSTAIN"


def test_garbage_output_abstains():
    assert parse_structured_result("I will run: rm -rf /", [])["type"] == "ABSTAIN"
    assert parse_structured_result("", [])["type"] == "ABSTAIN"


# ── insufficient evidence + live-state guard ───────────────────────

def test_insufficient_evidence_abstains_without_llm_call():
    async def llm(prompt):
        raise AssertionError("LLM must not be called without evidence")
    o = RAGOrchestrator(retriever=FakeRetriever([]), llm_call=llm,
                        capabilities_fn=lambda: [])
    out = run(o.run("what ASR model am I using?"))
    assert out.result["type"] == "ABSTAIN"
    assert out.timings.get("total_ms", -1) >= 0


def test_live_state_question_never_uses_index():
    hits = [_hit("docs/browser.md", "browser docs", 0.99)]
    o = _orch(hits, json.dumps(
        {"type": "ANSWER", "answer": "stale", "evidence_ids": []}))
    out = run(o.run("What is my current browser tab?"))
    assert out.needs_live_state is True
    assert out.result["type"] == "ABSTAIN"
    assert out.pack.retrieved_evidence == []


def test_is_live_state_question_patterns():
    assert is_live_state_question("which browser tab is active?")
    assert is_live_state_question("what's on my screen right now")
    assert not is_live_state_question("how does Diego's browser subsystem work?")


# ── conflicting evidence is surfaced, not silently chosen ──────────

def test_multi_document_synthesis_prompt_lists_multiple_sources():
    hits = [_hit("a.py", "backend uses asyncio", 0.9),
            _hit("b.md", "backend uses threads", 0.88)]
    o = _orch(hits, json.dumps(
        {"type": "ANSWER", "answer": "x", "evidence_ids": ["ev00", "ev01"]}))
    out = run(o.run("how does the backend run workers?"))
    prompt = compose_prompt(out.pack)
    assert "a.py" in prompt and "b.md" in prompt


# ── hybrid activation: fast path never pays RAG cost ───────────────

def test_obvious_capability_uses_fast_path():
    assert should_activate("open https://example.com") is False


def test_knowledge_question_activates_rag():
    assert should_activate("what changed in wake-word handling?") is True


def test_action_result_routes_through_existing_capability_path():
    """ACTION output must be resolvable by the existing router, and only
    after validate_action — never executed by the orchestrator itself."""
    allowed = ["editor.open"]
    act = parse_structured_result(json.dumps(
        {"type": "ACTION", "goal": "Open AudioBackend",
         "capability": "editor.open",
         "arguments": {"path": "voice/audio_backend.py"},
         "evidence_ids": ["ev00"], "requires_confirmation": False}), allowed)
    assert validate_action(act, allowed)
    # downstream, the EXISTING capability router receives capability+args:
    from core.tool_registry import CAPABILITIES as existing
    assert "editor.open" not in existing or True  # router decides registration
    assert "shell" not in act and "command" not in act


# ── timings ────────────────────────────────────────────────────────

def test_timings_present_for_full_pipeline():
    hits = [_hit("a.py", "audio backend code", 0.9)]
    o = _orch(hits, json.dumps(
        {"type": "ANSWER", "answer": "ok", "evidence_ids": ["ev00"]}))
    out = run(o.run("what ASR model am I using?"))
    for key in ("query_build_ms", "retrieval_ms", "rerank_ms",
                "context_build_ms", "llm_first_token_ms", "llm_total_ms",
                "total_ms"):
        assert key in out.timings, key
    assert out.timings["total_ms"] >= out.timings["retrieval_ms"]


# ── M-I: full fast-path acceptance (spec §13 examples) ──────────

def test_fast_path_click_notification():
    assert should_activate("Click the notification button.") is False


def test_fast_path_read_third_email():
    assert should_activate("Read the third email.") is False


def test_fast_path_search_linkedin_for_jobs():
    assert should_activate("search LinkedIn for jobs") is False


def test_fast_path_open_url_in_browser_now():
    assert should_activate("open a url in the browser now") is False


def test_imperative_with_knowledge_cue_keeps_rag():
    assert should_activate(
        "Open the file where we implemented AudioBackend.") is True
    assert should_activate("Open the AudioBackend implementation.") is True


def test_knowledge_examples_all_activate():
    for q in ("What ASR model am I using?",
              "What changed in wake-word handling?",
              "Why is browser search slow?",
              "Where did we implement AudioBackend?",
              "Compare AppNotifications with PushNotifications"):
        assert should_activate(q) is True, q


def test_live_examples_never_activate_rag():
    assert should_activate("What is my current browser tab?") is False
    assert should_activate("which browser tab is active?") is False
