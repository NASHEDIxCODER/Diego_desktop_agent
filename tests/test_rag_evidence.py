"""
RetrievedEvidence / semantic-chunk provenance tests (RAG phase, M-B).

Required behavior: retrieval returns meaningful semantic CHUNKS with full
provenance; matching lines are never the retrieval unit. Lines exist for
attribution only.

Covers:
  * locator parsing (line ranges, single lines, headings, empty)
  * symbol / source-type attribution
  * deterministic, collision-free evidence ids
  * retriever results carry evidence_id/chunk_id/line range/heading
  * component scores (semantic, lexical, relevance) exposed per chunk
  * full content block returned (not a single matching line)
  * keyword-only mode (no embedding model) still yields provenance
  * deterministic output across identical retrievals
"""

from knowledge.evidence import (
    first_symbol,
    make_evidence_fields,
    parse_locator,
    source_type_of,
)
from knowledge.retriever import KnowledgeRetriever

DOC = "/home/user/Projects/Diego/voice/audio_backend.py"

_CHUNK = {
    "text": (
        "class AudioBackend:\n"
        "    def __init__(self, device):\n"
        "        self.device = device\n"
        "\n"
        "    def start(self):\n"
        "        \"\"\"Start the audio backend stream.\"\"\"\n"
        "        return self.device.open()\n"
    ),
    "doc_path": DOC,
    "filename": "audio_backend.py",
    "locator": "lines 42-117",
    "chunk_index": 3,
    "file_type": "py",
    "embedding": [],
    "indexed_at": 0,
}


class _FakeStore:
    """Minimal store: one semantic chunk, no schema needed."""

    def all_chunks_with_embeddings(self):
        return [dict(_CHUNK)]


class _FakeEmbedder:
    """Keyword-only mode: query vector unavailable (model not loaded)."""

    name = "fake"

    def embed_query(self, query):
        return None


def _retriever():
    return KnowledgeRetriever(_FakeStore(), embedder=_FakeEmbedder())


# ── evidence.py unit tests ─────────────────────────────────────────

def test_parse_locator_line_range():
    info = parse_locator("lines 42-117")
    assert info["line_start"] == 42
    assert info["line_end"] == 117


def test_parse_locator_single_line_heading_and_empty():
    assert (parse_locator("lines 5")["line_start"],
            parse_locator("lines 5")["line_end"]) == (5, 5)
    assert parse_locator("AudioBackend setup")["heading"] == \
        "AudioBackend setup"
    empty = parse_locator("")
    assert empty["line_start"] is None and empty["heading"] == ""


def test_first_symbol_and_source_type():
    assert first_symbol(_CHUNK["text"]) == "AudioBackend"
    assert source_type_of(_CHUNK) == "py"
    assert source_type_of({"doc_path": "/tmp/notes.md"}) == "md"


def test_make_evidence_fields_deterministic_and_unique():
    a = make_evidence_fields(_CHUNK, semantic=0.9, lexical=0.8,
                             relevance=0.86, source="both")
    b = make_evidence_fields(dict(_CHUNK), semantic=0.1, lexical=0.2,
                             relevance=0.14, source="keyword")
    # Content-stable id: same chunk -> same id regardless of query scores.
    assert a["evidence_id"] == b["evidence_id"]
    assert a["evidence_id"].startswith("ev_")
    # Different chunk position -> different id.
    other = make_evidence_fields(dict(_CHUNK, chunk_index=4),
                                 semantic=0.0, lexical=0.0,
                                 relevance=0.0, source="keyword")
    assert other["evidence_id"] != a["evidence_id"]
    assert a["chunk_id"] == f"{DOC}#3"
    assert (a["line_start"], a["line_end"]) == (42, 117)
    assert a["semantic_score"] == 0.9
    assert a["lexical_score"] == 0.8
    assert a["symbol"] == "AudioBackend"
    assert DOC in a["provenance"]


# ── retriever integration ──────────────────────────────────────────

def test_search_returns_semantic_chunk_with_provenance():
    hits = _retriever().search("audio backend stream", top_k=5)
    assert hits, "expected the audio backend chunk to rank"
    h = hits[0]
    # Full content block (multi-line chunk), never a single matching line.
    assert "def start" in h["text"]
    assert h["text"].count("\n") >= 3
    assert h["evidence_id"].startswith("ev_")
    assert h["chunk_id"] == f"{DOC}#3"
    assert (h["line_start"], h["line_end"]) == (42, 117)
    assert h["lexical_score"] > 0
    assert abs(h["relevance_score"] - h["score"]) < 1e-3
    assert h["doc_path"] in h["provenance"]


def test_search_provenance_deterministic_across_calls():
    r1 = _retriever().search("audio backend stream", top_k=5)
    r2 = _retriever().search("audio backend stream", top_k=5)
    assert [h["evidence_id"] for h in r1] == \
        [h["evidence_id"] for h in r2]
    assert [h["score"] for h in r1] == [h["score"] for h in r2]


def test_search_unrelated_query_returns_no_hits():
    # Below the relevance floor — weak matches are never authoritative.
    assert _retriever().search("zzz qqq unrelated", top_k=5) == []
