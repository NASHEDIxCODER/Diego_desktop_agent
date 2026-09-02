"""
Tests for the local-knowledge RESPONSE UX contract.

When Diego answers from local knowledge, retrieval results stay
INTERNAL. The user only ever hears a concise synthesized answer with a
short friendly citation — never raw retrieval lists, filename dumps,
directory paths, chunk metadata, or embedding scores.

Covers:
  * normal knowledge question → concise synthesized answer
  * filenames are not dumped
  * directory paths are not dumped
  * retrieval scores/metadata are never spoken
  * explicit file-list request still returns filenames
  * explicit path request still returns paths
  * citations remain available (friendly: file name + locator)
  * strong local match bypasses the LLM
  * partial match uses bounded LLM context
  * live screen/app queries still use live context
  * response-size guard bounds spoken knowledge answers
"""

import asyncio

import pytest

from knowledge.presentation import (
    MAX_SPOKEN_CHARS,
    dedupe_evidence,
    friendly_source,
    is_explicit_listing_request,
    sanitize_spoken,
    synthesize_local_answer,
)


# ── Helpers ────────────────────────────────────────────────────────

def _result(doc="/home/user/Projects/Diego/README.md",
            filename="README.md", locator="", text="", score=0.9,
            chunk_index=0):
    return {
        "text": text,
        "doc_path": doc,
        "filename": filename,
        "locator": locator,
        "chunk_index": chunk_index,
        "file_type": "md",
        "score": score,
        "source": "both",
    }


class _Result:
    """Minimal CommandResult stand-in for brain._generate_response."""
    response = ""
    actions_executed = 0
    actions_failed = 0


def _patch_knowledge(monkeypatch, results, context=""):
    """Replace the knowledge_service singleton with a fake."""
    ks_mod = __import__("knowledge.service", fromlist=["knowledge_service"])
    monkeypatch.setattr(
        ks_mod, "knowledge_service",
        type("KS", (), {
            "search": staticmethod(
                lambda q, top_k=5, _r=results: list(_r)),
            "context_for_llm": staticmethod(
                lambda q, top_k=4, _c=context: _c),
        })())


def _patch_llm(monkeypatch, sentences, captured=None):
    """Replace streaming_llm.generate with a fake async generator."""
    import agent.streaming_llm as sl

    async def fake_generate(text, screen_context="", web_context=None,
                            local_context=None):
        if captured is not None:
            captured["text"] = text
            captured["screen"] = screen_context
            captured["local"] = local_context
        for s in sentences:
            yield s

    monkeypatch.setattr(sl.streaming_llm, "generate", fake_generate)


def _brain():
    return __import__("agent.brain", fromlist=["agent_brain"]).agent_brain


# ── Presentation layer: friendly citations ─────────────────────────

def test_friendly_source_never_contains_directory_path():
    r = _result(doc="/home/user/Projects/Diego/README.md",
                filename="README.md", locator="section 2")
    cite = friendly_source(r)
    assert cite == "README.md, section 2"
    assert "/" not in cite
    assert "home" not in cite


def test_friendly_source_falls_back_to_basename():
    r = _result(doc="/home/user/Downloads/report.pdf", filename="",
                locator="page 13")
    cite = friendly_source(r)
    assert cite == "report.pdf, page 13"
    assert "/home" not in cite


def test_friendly_source_virtual_document():
    r = _result(doc="diego://pc-snapshot", filename="diego://pc-snapshot",
                locator="")
    assert friendly_source(r) == "pc-snapshot"


# ── Presentation layer: dedupe ─────────────────────────────────────

def test_dedupe_collapses_overlapping_chunks_same_doc():
    a = _result(text="diego is a desktop agent with voice and vision",
                chunk_index=0)
    b = _result(text="diego is a desktop agent with voice and vision "
                     "systems", chunk_index=1)
    kept = dedupe_evidence([a, b])
    assert len(kept) == 1
    assert kept[0]["chunk_index"] == 0  # best-ranked kept


def test_dedupe_keeps_distinct_documents():
    a = _result(doc="/home/user/a.txt", filename="a.txt",
                text="alpha content about the project")
    b = _result(doc="/home/user/b.txt", filename="b.txt",
                text="beta content about the roadmap")
    kept = dedupe_evidence([a, b])
    assert len(kept) == 2


def test_dedupe_caps_evidence_blocks():
    results = [
        _result(doc=f"/home/user/f{i}.txt", filename=f"f{i}.txt",
                text=f"distinct evidence block number {i}")
        for i in range(10)
    ]
    kept = dedupe_evidence(results)
    assert len(kept) <= 3


# ── Presentation layer: synthesis ──────────────────────────────────

def test_synthesize_concise_answer_with_citation():
    r = _result(text="The Diego project includes the agent, voice, "
                     "vision, authentication, knowledge, plugins, and "
                     "testing systems.")
    answer = synthesize_local_answer("what do you know about my diego "
                                     "project?", [r])
    assert "README.md" in answer            # friendly citation
    assert "Diego project includes" in answer  # answered from evidence
    assert "/home/user" not in answer       # no raw path
    assert len(answer) <= MAX_SPOKEN_CHARS  # bounded for voice


def test_synthesize_empty_results_returns_empty():
    assert synthesize_local_answer("anything", []) == ""


def test_synthesize_bounds_huge_evidence():
    r = _result(text="word " * 2000)
    answer = synthesize_local_answer("query", [r])
    assert len(answer) <= MAX_SPOKEN_CHARS


# ── Presentation layer: explicit listing / path detection ──────────

@pytest.mark.parametrize("text", [
    "list the files in my Diego project",
    "show files in Downloads",
    "what files are in this folder?",
    "which files mention the budget?",
    "directory listing of Projects",
])
def test_explicit_listing_requests_detected(text):
    assert is_explicit_listing_request(text) is True


@pytest.mark.parametrize("text", [
    "what is the path to report.pdf?",
    "where is the file I saved yesterday?",
    "give me the full path of the config file",
    "where did i put the contract document?",
])
def test_explicit_path_requests_detected(text):
    assert is_explicit_listing_request(text) is True


@pytest.mark.parametrize("text", [
    "what do you know about my Diego project?",
    "tell me about the budget review",
    "what does diego use for its knowledge index?",
    "summarize my meeting notes",
])
def test_normal_knowledge_questions_not_listing(text):
    assert is_explicit_listing_request(text) is False


# ── Presentation layer: spoken sanitizer ───────────────────────────

def test_sanitize_scrubs_absolute_paths():
    leaky = ("I found /home/user/Documents/notes.txt and "
             "/home/user/Downloads/report.pdf in your files.")
    out = sanitize_spoken(leaky)
    assert "/home/user" not in out
    assert "notes.txt" in out        # short names survive as citations
    assert "report.pdf" in out


def test_sanitize_keeps_paths_when_explicitly_requested():
    leaky = "report.pdf is at /home/user/Documents/report.pdf."
    out = sanitize_spoken(leaky, allow_paths=True)
    assert "/home/user/Documents/report.pdf" in out


def test_sanitize_removes_scores_and_metadata():
    leaky = ("According to README.md (score=0.95, chunk_index: 3, "
             "embedding: local) the project has seven systems.")
    out = sanitize_spoken(leaky)
    assert "score" not in out.lower()
    assert "chunk_index" not in out.lower()
    assert "embedding" not in out.lower()
    assert "README.md" in out
    assert "seven systems" in out


def test_sanitize_removes_bracketed_citation_blocks():
    leaky = ("[/home/user/Projects/Diego/README.md (section 2)] "
             "The project has seven systems.")
    out = sanitize_spoken(leaky)
    assert "[" not in out and "]" not in out
    assert "/home/user" not in out
    assert "seven systems" in out


def test_sanitize_size_guard_bounds_response():
    huge = "Sentence about the project. " * 200
    out = sanitize_spoken(huge)
    assert len(out) <= MAX_SPOKEN_CHARS
    assert out  # not empty


def test_sanitize_does_not_mangle_urls():
    text = "See https://example.com/docs/page for details."
    out = sanitize_spoken(text)
    assert "https://example.com/docs/page" in out


# ── Brain integration: normal knowledge question ───────────────────

def test_brain_normal_knowledge_question_concise_synthesized(monkeypatch):
    """Normal knowledge question → concise synthesized answer with a
    friendly citation. No filename dump, no paths, no metadata."""
    _patch_knowledge(monkeypatch, results=[
        _result(text="The Diego project includes the agent, voice, "
                     "vision, authentication, knowledge, plugins, and "
                     "testing systems."),
        _result(doc="/home/user/Projects/Diego/docs/architecture.md",
                filename="architecture.md", locator="section 1",
                text="Diego architecture: perceive, decide, plan, "
                     "dispatch, verify, learn.", score=0.7),
    ])
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "what do you know about my diego project?", None, _Result()))

    # Synthesized locally — concise, cited, evidence-based
    assert "README.md" in resp
    assert "Diego project includes" in resp
    # No retrieval internals spoken
    assert "/home/user" not in resp
    assert "Projects/Diego" not in resp
    assert "architecture.md" not in resp or resp.count(".md") <= 2
    assert "score" not in resp.lower()
    assert "chunk" not in resp.lower()
    assert len(resp) <= MAX_SPOKEN_CHARS
    # LLM never invoked for a strong match
    assert "text" not in captured


def test_brain_no_filename_dump_for_multi_match(monkeypatch):
    """Multiple matches must never produce a dump of matched file
    names — only the smallest useful evidence is spoken."""
    _patch_knowledge(monkeypatch, results=[
        _result(doc=f"/home/user/Projects/Diego/file{i}.md",
                filename=f"file{i}.md",
                text=f"diego knowledge note number {i}",
                score=0.9 - i * 0.01)
        for i in range(5)
    ])
    resp = asyncio.run(_brain()._generate_response(
        "what do you know about diego?", None, _Result()))
    # At most ONE friendly citation — never a list of five files
    mentioned = sum(1 for i in range(5) if f"file{i}.md" in resp)
    assert mentioned <= 1
    assert "/home/user" not in resp


def test_brain_scores_and_metadata_never_spoken(monkeypatch):
    """Even if the evidence text itself contains metadata-like tokens,
    the spoken answer never carries retrieval scores/metadata."""
    _patch_knowledge(monkeypatch, results=[
        _result(text="diego uses duckdb for the knowledge index. "
                     "score=0.99 chunk_index=7 doc_path=/x/y.txt"),
    ])
    resp = asyncio.run(_brain()._generate_response(
        "what does diego use for its knowledge index?", None, _Result()))
    assert "duckdb" in resp
    assert "score=" not in resp.lower()
    assert "chunk_index" not in resp.lower()
    assert "doc_path" not in resp.lower()
    assert "/x/y.txt" not in resp


# ── Brain integration: explicit listing / path requests ────────────

def test_brain_explicit_file_list_still_returns_filenames(monkeypatch):
    """An explicit file-list request is answered through the bounded
    LLM context — filenames survive in the spoken response."""
    _patch_knowledge(
        monkeypatch,
        results=[_result(text="notes.txt budget report")],
        context="[/home/user/Documents/notes.txt]\nnotes.txt budget "
                "report\n\n[/home/user/Documents/budget.xlsx]\nbudget "
                "numbers")
    captured = {}
    _patch_llm(monkeypatch,
               ["You have notes.txt and budget.xlsx in Documents."],
               captured)

    resp = asyncio.run(_brain()._generate_response(
        "list the files in my documents folder", None, _Result()))

    # LLM used with bounded local context (not the strong-match bypass)
    assert captured.get("local")
    # Filenames are allowed for an explicit listing
    assert "notes.txt" in resp
    assert "budget.xlsx" in resp


def test_brain_explicit_path_request_still_returns_paths(monkeypatch):
    """An explicit path request may hear the full path."""
    _patch_knowledge(
        monkeypatch,
        results=[_result(doc="/home/user/Documents/report.pdf",
                         filename="report.pdf",
                         text="quarterly budget review")],
        context="[/home/user/Documents/report.pdf]\nquarterly budget "
                "review")
    _patch_llm(monkeypatch,
               ["report.pdf is at /home/user/Documents/report.pdf."])

    resp = asyncio.run(_brain()._generate_response(
        "what is the path to report.pdf?", None, _Result()))
    assert "/home/user/Documents/report.pdf" in resp


# ── Brain integration: partial match → bounded LLM context ─────────

def test_brain_partial_match_uses_bounded_llm_context(monkeypatch):
    """A weak/partial local match is NOT answered directly — bounded
    context is sent to the LLM, which answers the question (not the
    retrieval process). Any leaked path is scrubbed from speech."""
    _patch_knowledge(
        monkeypatch,
        results=[_result(text="partial note about giraffe feeding "
                              "habits in the savanna", score=0.3)],
        context="[/home/user/Documents/giraffe.txt]\npartial note "
                "about giraffe feeding habits")
    captured = {}
    # The LLM misbehaves and leaks the path — the guard must scrub it.
    _patch_llm(
        monkeypatch,
        ["According to /home/user/Documents/giraffe.txt, giraffes "
         "browse on acacia leaves in the savanna."],
        captured)

    resp = asyncio.run(_brain()._generate_response(
        "what do my notes say about giraffe feeding?", None, _Result()))

    # Bounded context reached the LLM (partial-match path)
    assert captured.get("local")
    assert len(captured["local"]) <= 2500 + 200
    # The answer itself survives ...
    assert "giraffes" in resp
    # ... but the leaked absolute path is scrubbed to the short name.
    assert "/home/user" not in resp
    assert "giraffe.txt" in resp


def test_brain_no_local_match_uses_normal_llm_path(monkeypatch):
    """No local match → the existing normal LLM path, no local
    context, no sanitization side effects."""
    _patch_knowledge(monkeypatch, results=[], context="")
    captured = {}
    _patch_llm(monkeypatch, ["The capital of France is Paris."],
               captured)

    resp = asyncio.run(_brain()._generate_response(
        "what is the capital of france?", None, _Result()))
    assert resp == "The capital of France is Paris."
    assert captured.get("local") in (None, "")


# ── Brain integration: live-state precedence ───────────────────────

def test_brain_live_screen_query_uses_live_context(monkeypatch):
    """A live screen query is never answered from stale local
    knowledge — the screen context goes to the LLM and the spoken
    answer carries no local-knowledge leakage."""
    _patch_knowledge(
        monkeypatch,
        results=[_result(doc="/tmp/screen.txt", filename="screen.txt",
                         text="a stale file mentioning the screen",
                         score=0.99)],
        context="[/tmp/screen.txt]\nstale screen content")
    captured = {}
    _patch_llm(monkeypatch, ["You have a pytest run on screen."],
               captured)

    class _Ctx:
        compact_summary = "Terminal: pytest output on screen"

    resp = asyncio.run(_brain()._generate_response(
        "what is on my screen?", _Ctx(), _Result()))

    assert captured["screen"] == "Terminal: pytest output on screen"
    assert "pytest" in resp
    # Strong-match bypass suppressed — no local synthesis
    assert "According to" not in resp
    assert "stale file" not in resp
    assert "/tmp/screen.txt" not in resp


def test_brain_live_app_query_not_answered_from_knowledge(monkeypatch):
    """Screen/app/process/current-state requests must use live tools —
    even WITHOUT perception context, the strong-match shortcut is
    suppressed for live desktop queries."""
    _patch_knowledge(monkeypatch, results=[
        _result(text="firefox is a web browser used daily", score=0.95),
    ])
    captured = {}
    _patch_llm(monkeypatch, ["Firefox and the terminal are running."],
               captured)

    resp = asyncio.run(_brain()._generate_response(
        "what apps are running right now?", None, _Result()))

    # Not answered from the stale local chunk
    assert "According to" not in resp
    assert "web browser used daily" not in resp
    # The LLM path handled it (live tools own this request)
    assert "text" in captured


# ── Brain integration: response-size guard ─────────────────────────

def test_brain_response_size_guard(monkeypatch):
    """Local knowledge must never produce a huge spoken response."""
    _patch_knowledge(
        monkeypatch,
        results=[_result(text="word " * 300, score=0.4)],
        context="[/home/user/Documents/big.txt]\n" + "word " * 300)
    _patch_llm(monkeypatch, ["The document says " + "word " * 400 + "."])

    resp = asyncio.run(_brain()._generate_response(
        "summarize my big document", None, _Result()))
    assert len(resp) <= MAX_SPOKEN_CHARS
    assert resp


def test_brain_action_lines_never_sanitized(monkeypatch):
    """ACTION lines in an LLM response are never touched by the
    spoken guard (they are parsed, not spoken)."""
    _patch_knowledge(
        monkeypatch,
        results=[_result(text="notes about the folder", score=0.4)],
        context="[/home/user/Documents/notes.txt]\nnotes about the "
                "folder")
    _patch_llm(monkeypatch, [
        "Opening that folder for you.",
        'ACTION: {"action": "open_folder", "params": '
        '{"path": "/home/user/Documents"}}',
    ])

    resp = asyncio.run(_brain()._generate_response(
        "open my documents folder", None, _Result()))
    assert "ACTION:" in resp
    assert "/home/user/Documents" in resp  # ACTION payload intact