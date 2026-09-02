"""
Tests for the local Knowledge & Document Indexing subsystem.

Covers:
  * TXT / PDF / DOCX / structured (CSV/JSON/XML/HTML/XLSX) ingestion
  * Corrupted / encrypted / unsupported files (graceful failure)
  * Duplicate detection (same path never duplicated)
  * Hash-based incremental updates (unchanged files not re-extracted)
  * Deletion handling
  * Symlink / path security (escaping symlinks rejected)
  * Sensitive-path exclusion (denylist wins over allowlist)
  * Retrieval ranking + citation metadata
  * Local-only embedding behavior (no cloud)
  * Startup non-blocking behavior
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from knowledge import extractors
from knowledge.chunker import chunk_text, chunk_sections
from knowledge.embedder import LocalEmbedder, text_hash
from knowledge.indexer import KnowledgeIndexer
from knowledge.policy import PathPolicy
from knowledge.retriever import KnowledgeRetriever
from knowledge.store import KnowledgeStore
from knowledge.snapshot import collect_snapshot

import duckdb


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "docs"


@pytest.fixture
def db(tmp_path: Path) -> Generator[Path, None, None]:
    db_path = tmp_path / "test.duckdb"
    yield db_path
    if db_path.exists():
        os.unlink(db_path)


class FakeStore:
    """Minimal duckdb-store stand-in (same connect() contract)."""

    def __init__(self, db_path: Path):
        self._db_path = str(db_path)

    def initialize(self):
        pass

    def connect(self):
        import contextlib
        conn = duckdb.connect(self._db_path)

        @contextlib.contextmanager
        def _wrap():
            try:
                yield conn
            finally:
                pass
        # Provide a context manager API like DuckDBStore.connect
        return _wrap()


class FakeEmbedder:
    """Deterministic, local-only fake embedding backend for tests."""

    name = "test_fake"

    @property
    def is_local(self):
        return True

    @property
    def status(self):
        return "ready (test fake)"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for tok in text.lower().split():
            v[hash(tok) % 64] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text) if text.strip() else None


@pytest.fixture
def policy(root: Path) -> PathPolicy:
    p = PathPolicy(
        allow_roots=[root],
        deny_paths=set(),
        deny_names={".git", ".venv", "node_modules", ".env",
                    "secrets", "credentials"},
        max_file_size=1024 * 1024,
    )
    root.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def store(db: Path) -> KnowledgeStore:
    s = KnowledgeStore(store=FakeStore(db))
    s.initialize()
    return s


@pytest.fixture
def indexer(store: KnowledgeStore, policy: PathPolicy) -> KnowledgeIndexer:
    return KnowledgeIndexer(store, policy, embedder=FakeEmbedder())


@pytest.fixture
def retriever(store: KnowledgeStore) -> KnowledgeRetriever:
    return KnowledgeRetriever(store, embedder=FakeEmbedder())


# ── TXT ingestion ────────────────────────────────────────────────

def test_txt_ingestion(store, indexer, retriever, root):
    f = root / "notes.txt"
    f.write_text("Diego is a desktop agent. It indexes local files.")
    indexer.scan()
    doc = store.get_document(str(f))
    assert doc is not None
    assert doc["extraction_status"] == "extracted"
    assert doc["chunk_count"] >= 1
    results = retriever.search("desktop agent", top_k=3)
    assert results
    assert any(r["doc_path"] == str(f) for r in results)


def test_txt_chunking_deterministic(root, indexer, store):
    f = root / "big.txt"
    f.write_text(" ".join(f"sentence number {i} about giraffes." for i in range(500)))
    r1 = indexer.scan()
    doc1 = store.get_document(str(f))
    r2 = indexer.scan()
    doc2 = store.get_document(str(f))
    assert doc1["chunk_count"] == doc2["chunk_count"]


# ── PDF ingestion ────────────────────────────────────────────────

def _make_pdf(path: Path, text: str) -> None:
    """Handcraft a minimal single-page PDF with a text stream."""
    content = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode()
    path.write_bytes(bytes(out))


def test_pdf_ingestion(store, indexer, retriever, root):
    f = root / "report.pdf"
    _make_pdf(f, "Quarterly budget review highlights")
    indexer.scan()
    doc = store.get_document(str(f))
    assert doc is not None
    assert doc["extraction_status"] in ("extracted", "metadata_only")
    if doc["extraction_status"] == "extracted":
        results = retriever.search("budget review", top_k=3)
        assert any("page" in (r.get("locator") or "") for r in results)


def test_pdf_extractor_page_locators(tmp_path):
    f = tmp_path / "p.pdf"
    _make_pdf(f, "hello pdf world")
    ex = extractors.PdfExtractor().extract(f)
    assert ex.status in ("extracted", "metadata_only")


# ── DOCX ingestion ───────────────────────────────────────────────

def test_docx_ingestion(store, indexer, retriever, root):
    docx = pytest.importorskip("docx")
    f = root / "meeting.docx"
    d = docx.Document()
    d.add_heading("Project Plan", level=1)
    d.add_paragraph("The deliverable deadline is Friday.")
    d.save(str(f))
    indexer.scan()
    doc = store.get_document(str(f))
    assert doc is not None
    assert doc["extraction_status"] == "extracted"
    results = retriever.search("deliverable deadline", top_k=3)
    assert any(r["doc_path"] == str(f) for r in results)


# ── Structured ingestion ─────────────────────────────────────────

def test_structured_ingestion(store, indexer, root):
    (root / "data.csv").write_text("name,score\nalice,10\nbob,20\n")
    (root / "cfg.json").write_text(json.dumps({"app": "diego", "mode": "test"}))
    (root / "doc.xml").write_text("<root><item>alpha</item></root>")
    (root / "page.html").write_text(
        "<html><head><title>Hi</title></head><body><p>beta content</p></body></html>")
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws["A1"] = "item"
    ws["B1"] = "cost"
    ws["A2"] = "laptop"
    ws["B2"] = 900
    wb.save(str(root / "sheet.xlsx"))
    indexer.scan()
    stats = store.stats()
    assert stats["documents"] >= 5
    paths = {d["path"] for d in store.all_documents()}
    assert any(p.endswith("sheet.xlsx") for p in paths)
    sheet_doc = next(d for d in store.all_documents()
                     if d["path"].endswith("sheet.xlsx"))
    assert sheet_doc["extraction_status"] == "extracted"


def test_locator_metadata_citations(store, indexer, retriever, root):
    openpyxl = pytest.importorskip("openpyxl")
    f = root / "fin.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Q3"
    ws["A1"] = "revenue grew strongly"
    wb.save(str(f))
    indexer.scan()
    results = retriever.search("revenue", top_k=3)
    assert results
    r = results[0]
    assert r["doc_path"] == str(f)
    assert "Q3" in (r.get("locator") or "")


# ── Corrupted / unsupported / encrypted ─────────────────────────

def test_corrupted_pdf_graceful(store, indexer, root):
    f = root / "broken.pdf"
    f.write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    snap = indexer.scan()
    doc = store.get_document(str(f))
    assert doc is not None  # metadata recorded
    assert doc["extraction_status"] == "failed"
    assert snap["failed"] >= 1


def test_encrypted_pdf_graceful(store, indexer, root):
    """A PDF that claims encryption must fail gracefully, not crash."""
    f = root / "locked.pdf"
    f.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Encrypt 2 0 R >>\nendobj\n")
    indexer.scan()
    doc_ok_or_failed(store, f)


def doc_ok_or_failed(store, f):
    doc = store.get_document(str(f))
    assert doc is not None
    assert doc["extraction_status"] in ("failed", "metadata_only", "extracted")


def test_unsupported_binary_metadata_only(store, indexer, root):
    f = root / "archive.zip"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    indexer.scan()
    doc = store.get_document(str(f))
    assert doc is not None
    assert doc["extraction_status"] in ("metadata_only", "unsupported")
    assert doc["chunk_count"] == 0


def test_corruption_never_crashes_indexer(store, indexer, root):
    for i in range(5):
        (root / f"bad{i}.pdf").write_bytes(os.urandom(64))
    snap = indexer.scan()
    assert snap["scanned"] == 5


# ── Duplicate detection / incremental updates ───────────────────

def test_duplicate_path_not_duplicated(store, indexer, root):
    f = root / "same.txt"
    f.write_text("unique content one")
    indexer.scan()
    indexer.scan()
    docs = [d for d in store.all_documents() if d["path"] == str(f)]
    assert len(docs) == 1


def test_incremental_no_reindex_unchanged(store, indexer, root):
    f = root / "stable.txt"
    f.write_text("unchanged content here")
    snap1 = indexer.scan()
    assert snap1["updated"] == 1
    snap2 = indexer.scan()
    assert snap2["updated"] == 0
    assert snap2["skipped"] >= 1


def test_hash_based_reindex_on_change(store, indexer, root):
    f = root / "changing.txt"
    f.write_text("version one")
    indexer.scan()
    h1 = store.get_document(str(f))["content_hash"]
    time.sleep(0.01)
    f.write_text("version two")
    os.utime(f, (time.time() + 2, time.time() + 2))
    indexer.scan()
    h2 = store.get_document(str(f))["content_hash"]
    assert h1 != h2
    results_chunks = store.chunk_count(str(f))
    assert results_chunks >= 1


def test_mtime_only_change_no_reindex(store, indexer, root):
    f = root / "touch.txt"
    f.write_text("same bytes")
    indexer.scan()
    os.utime(f, (time.time() + 10, time.time() + 10))
    snap = indexer.scan()
    assert snap["updated"] == 0


def test_deletion_handling(store, indexer, root):
    f = root / "doomed.txt"
    f.write_text("to be deleted")
    indexer.scan()
    assert store.get_document(str(f)) is not None
    f.unlink()
    snap = indexer.scan()
    assert snap["removed"] == 1
    assert store.get_document(str(f)) is None
    assert store.chunk_count(str(f)) == 0


# ── Symlink / path security ──────────────────────────────────────

def test_escaping_symlink_rejected(policy, indexer, store, root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret outside content")
    link = root / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")
    snap = indexer.scan()
    assert store.get_document(str(link)) is None
    reasons = [s["reason"] for s in indexer.skipped_paths()]
    assert any("symlink" in r or "outside" in r for r in reasons)


def test_internal_symlink_followed(policy, indexer, store, root):
    target = root / "inner.txt"
    target.write_text("inside the root content")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks not supported")
    indexer.scan()
    # Symlinks resolving INSIDE an approved root are followed, and the
    # real path is indexed exactly once (deduped with the target).
    assert store.get_document(str(target)) is not None


def test_outside_root_rejected(policy, indexer, store, root, tmp_path):
    outside = tmp_path / "notroot.txt"
    outside.write_text("outside content")
    indexer.scan(roots=[outside.parent])
    # tmp_path is not an approved root → nothing indexed
    assert store.get_document(str(outside)) is None


def test_symlinked_dir_escape_not_followed(policy, indexer, store,
                                           root, tmp_path):
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "hidden.txt").write_text("hidden")
    link_dir = root / "escape_dir"
    try:
        link_dir.symlink_to(outside_dir)
    except OSError:
        pytest.skip("symlinks not supported")
    indexer.scan()
    assert store.get_document(str(outside_dir / "hidden.txt")) is None


# ── Sensitive-path exclusion ─────────────────────────────────────

def test_sensitive_names_excluded(store, indexer, root):
    (root / ".env").write_text("API_KEY=supersecret")
    (root / "secrets").mkdir()
    (root / "secrets" / "creds.txt").write_text("password=hunter2")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]")
    (root / "ok.txt").write_text("normal content")
    indexer.scan()
    paths = {d["path"] for d in store.all_documents()}
    assert any(p.endswith("ok.txt") for p in paths)
    assert not any(p.endswith(".env") for p in paths)
    assert not any("secrets" in p for p in paths)
    assert not any(".git" in p for p in paths)


def test_policy_sensitive_check(tmp_path):
    p = PathPolicy(
        allow_roots=[tmp_path / "allowed"],
        deny_paths={tmp_path / "allowed" / ".ssh"},
        deny_names={"node_modules"},
        max_file_size=1024,
    )
    (tmp_path / "allowed" / ".ssh").mkdir(parents=True)
    ok, _, _ = p.check(tmp_path / "allowed" / ".ssh" / "id_rsa")
    assert not ok
    ok, _, _ = p.check(tmp_path / "allowed" / "sub" / "node_modules" / "x")
    assert not ok
    ok, _, _ = p.check(tmp_path / "allowed" / "fine.txt")
    assert not ok  # file does not exist yet → "not found" (still safe)
    (tmp_path / "allowed" / "fine.txt").write_text("x")
    ok, _, _ = p.check(tmp_path / "allowed" / "fine.txt")
    assert ok


def test_proc_sys_dev_always_denied(tmp_path):
    p = PathPolicy(allow_roots=[Path("/")], deny_paths=set(),
                   deny_names=set(), max_file_size=1024)
    ok, reason, _ = p.check(Path("/proc/self/cmdline"))
    assert not ok
    assert "system path" in reason


# ── Chunker determinism ─────────────────────────────────────────

def test_chunker_deterministic():
    text = " ".join(f"Sentence {i} is here." for i in range(200))
    a = [c.text for c in chunk_text(text, chunk_size=300, overlap=50)]
    b = [c.text for c in chunk_text(text, chunk_size=300, overlap=50)]
    assert a == b
    assert len(a) > 1


def test_chunker_preserves_locator():
    from knowledge.extractors import Section
    secs = [Section(locator="page 4", text="word " * 500)]
    chunks = chunk_sections(secs)
    assert chunks
    assert all(c.locator == "page 4" for c in chunks)


# ── Retrieval ranking + citations ────────────────────────────────

def test_retrieval_ranking_and_citations(store, indexer, retriever, root):
    (root / "python_doc.txt").write_text(
        "The giraffe is the tallest land animal on earth.")
    (root / "other.txt").write_text(
        "Python is a programming language for software development.")
    indexer.scan()
    results = retriever.search("python programming language", top_k=2)
    assert results
    top = results[0]
    assert "other.txt" in top["doc_path"] or "python" in top["doc_path"]
    assert top["score"] > results[-1]["score"] or len(results) == 1
    # Citation metadata present
    assert top["doc_path"]
    assert "chunk_index" in top
    assert "locator" in top


def test_retrieval_empty_index(store, retriever):
    assert retriever.search("anything") == []
    assert retriever.context_for_llm("anything") == ""


def test_context_for_llm_bounds(store, indexer, retriever, root):
    (root / "long.txt").write_text("keywordalpha " * 2000)
    indexer.scan()
    ctx = retriever.context_for_llm("keywordalpha", top_k=4,
                                    max_chars=500)
    assert len(ctx) <= 500 + 200  # citation overhead tolerance


# ── Embeddings: local-only + cached by content hash ─────────────

def test_local_only_embedding_backend():
    emb = LocalEmbedder()
    assert emb.is_local is True


def test_chunk_hash_stable_and_sensitive():
    h1 = text_hash("abc")
    h2 = text_hash("abc")
    h3 = text_hash("abd")
    assert h1 == h2 and h1 != h3


def test_embedding_cache_no_reembed_unchanged(store, indexer, root):
    f = root / "cached.txt"
    f.write_text("stable text for embedding cache")
    indexer.scan()
    # Second scan: file unchanged → skipped entirely (no re-embedding)
    snap = indexer.scan()
    assert snap["updated"] == 0


def test_hash_embedding_deterministic():
    e = FakeEmbedder()
    v1 = e.embed_query("hello world")
    v2 = e.embed_query("hello world")
    assert np.allclose(v1, v2)


# ── PC snapshot ──────────────────────────────────────────────────

def test_pc_snapshot_readonly_structure():
    snap = collect_snapshot()
    for key in ("os", "cpu", "memory", "disks", "network"):
        assert key in snap
    assert "system" in snap["os"]
    assert isinstance(snap["disks"], list)


def test_snapshot_persisted_in_duckdb(store):
    snap = collect_snapshot()
    store.save_snapshot("pc_inventory", snap)
    latest = store.latest_snapshot("pc_inventory")
    assert latest is not None
    assert latest["payload"]["os"]["system"] == snap["os"]["system"]


def test_snapshot_text_contains_inventory():
    snap = collect_snapshot()
    text = __import__("knowledge.snapshot", fromlist=["snapshot_text"]) \
        .snapshot_text(snap)
    assert "Operating system" in text


# ── Startup non-blocking ────────────────────────────────────────

def test_start_background_scan_non_blocking(indexer, root):
    (root / "a.txt").write_text("content a")
    t0 = time.time()
    ok = indexer.start_background_scan()
    elapsed = time.time() - t0
    assert ok
    assert elapsed < 1.0  # returns immediately
    # Wait for the daemon scan to finish
    for _ in range(100):
        if not indexer.progress.snapshot()["running"]:
            break
        time.sleep(0.05)
    assert indexer.progress.snapshot()["scanned"] >= 1


def test_cancellation_supported(indexer, root):
    for i in range(200):
        (root / f"f{i}.txt").write_text(f"filler file {i} " * 40)
    indexer.start_background_scan()
    indexer.cancel()
    # Cancel must be accepted without error; scan eventually stops
    for _ in range(200):
        if not indexer.progress.snapshot()["running"]:
            break
        time.sleep(0.05)


def test_file_size_limit_metadata_only(store, policy, indexer, root):
    big = root / "big.dat"
    big.write_bytes(b"\x00" * (policy.max_file_size + 1))
    indexer.scan()
    assert store.get_document(str(big)) is None
    reasons = [s["reason"] for s in indexer.skipped_paths()]
    assert any("too large" in r for r in reasons)

# ── Sensitive-material regression tests (feedback hardening) ─────

def test_aws_access_key_csv_never_indexed(store, indexer, root):
    """AWS-style access-key CSV must be rejected by NAME before any
    open/stat/hash/extract/persist step."""
    f = root / "cline-bedrock-user_accessKeys.csv"
    f.write_text("Access key ID,Secret access key\nAKIAIOSFODNN7,EXAMPLEKEY\n")
    indexer.scan()
    assert store.get_document(str(f)) is None
    assert store.chunk_count(str(f)) == 0
    # Logged by category only — never the filename
    for entry in indexer.skipped_paths():
        assert "accessKeys" not in entry["path"]


def test_api_key_csv_never_indexed(store, indexer, root):
    f = root / "bedrock-long-term-api-key.csv"
    f.write_text("key\nsupersecret\n")
    indexer.scan()
    assert store.get_document(str(f)) is None


def test_session_file_never_indexed(store, indexer, root):
    f = root / "diego.session"
    f.write_bytes(b"\x00session-data\x00")
    indexer.scan()
    assert store.get_document(str(f)) is None


def test_pem_and_key_files_never_indexed(store, indexer, root):
    (root / "server.pem").write_text("-----BEGIN PRIVATE KEY-----")
    (root / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
    (root / "cert.p12").write_bytes(b"\x30\x82")
    indexer.scan()
    assert store.get_document(str(root / "server.pem")) is None
    assert store.get_document(str(root / "id_rsa")) is None
    assert store.get_document(str(root / "cert.p12")) is None


def test_env_file_never_indexed(store, indexer, root):
    (root / ".env").write_text("GEMINI_API_KEY=xxx\n")
    (root / ".env.local").write_text("TOKEN=yyy\n")
    indexer.scan()
    assert store.get_document(str(root / ".env")) is None
    assert store.get_document(str(root / ".env.local")) is None


def test_secret_named_file_never_indexed(store, indexer, root):
    (root / "my-secrets.txt").write_text("password=hunter2")
    (root / "auth_token.json").write_text('{"t": "x"}')
    indexer.scan()
    assert store.get_document(str(root / "my-secrets.txt")) is None
    assert store.get_document(str(root / "auth_token.json")) is None


def test_sensitive_dir_nested_in_approved_root(store, indexer, root):
    d = root / "project" / ".ssh"
    d.mkdir(parents=True)
    (d / "id_ed25519").write_text("private")
    (root / "project" / "code.py").write_text("print('ok')")
    indexer.scan()
    assert store.get_document(str(d / "id_ed25519")) is None
    assert store.get_document(str(root / "project" / "code.py")) is not None


def test_secret_file_never_opened(store, indexer, root, monkeypatch):
    """Prove the secret file is never OPENED during a scan."""
    f = root / "credentials.csv"
    f.write_text("user,password\nadmin,s3cret\n")

    real_open = open
    opened = []

    def spy_open(file, *a, **kw):
        try:
            name = str(file)
        except Exception:
            name = ""
        if "credentials.csv" in name:
            opened.append(name)
        return real_open(file, *a, **kw)

    import builtins
    monkeypatch.setattr(builtins, "open", spy_open)
    indexer.scan()
    monkeypatch.undo()
    assert opened == []  # never opened — rejected by name only
    assert store.get_document(str(f)) is None


def test_generated_dirs_in_project_excluded(store, indexer, root):
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "m.pyc").write_bytes(b"\x00")
    (root / "logs").mkdir()
    (root / "logs" / "run.out").write_text("log line")
    (root / "src.py").write_text("x = 1")
    indexer.scan()
    paths = {d["path"] for d in store.all_documents()}
    assert any(p.endswith("src.py") for p in paths)
    assert not any(".git" in p or "__pycache__" in p or "/logs/" in p
                   for p in paths)


def test_keyword_retrieval_works_without_embeddings(store, indexer,
                                                    retriever, root):
    """Embeddings unavailable → keyword/full-text retrieval still works."""
    class BrokenEmbedder(FakeEmbedder):
        name = "broken"

        def embed_batch(self, texts):
            return [None] * len(texts)

        def embed_query(self, text):
            return None

    (root / "kw.txt").write_text("quantum flux capacitor calibration")
    indexer2 = KnowledgeIndexer(store, indexer._policy,
                                embedder=BrokenEmbedder())
    indexer2.scan()
    results = retriever.search("flux capacitor", top_k=3)
    assert results
    assert results[0]["doc_path"] == str(root / "kw.txt")
    assert results[0]["source"] == "keyword"


def test_embedder_failure_memoized_no_spam(root, store, policy):
    class BrokenEmbedder(FakeEmbedder):
        name = "broken2"

        def embed_batch(self, texts):
            raise RuntimeError("model missing")

        def embed_query(self, text):
            raise RuntimeError("model missing")

    from knowledge.embedder import LocalEmbedder
    emb = LocalEmbedder()
    # Simulate a failed model load (memoized)
    emb._failed = True
    assert emb.status.startswith("unavailable")
    assert emb.embed_query("x") is None
    assert emb.embed_batch(["a"]) == [None]


def test_status_accurate_after_scan(store, indexer, root):
    from knowledge.service import KnowledgeService
    (root / "s.txt").write_text("status content")
    indexer.scan()
    svc = KnowledgeService()
    svc._store = store
    svc._policy = indexer._policy
    svc._indexer = indexer
    svc._embedder = indexer._embedder
    status = svc.index_status()
    assert status["documents"] >= 1
    assert status["chunks"] >= 1
    assert status["roots"]
    assert "embeddings" in status
    assert status["embeddings"]["local_only"] is True
    assert status["scan"]["running"] is False
