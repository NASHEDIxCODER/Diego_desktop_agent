"""
DuckDB-backed storage for the local knowledge index.

Extends the EXISTING Diego DuckDB database (memory.duckdb_store.store) —
no parallel database. Adds three tables:

    knowledge_documents  — one row per indexed file (hash, mtime, status)
    knowledge_chunks     — one row per chunk (text, locator, embedding)
    knowledge_snapshots  — PC inventory snapshots (structured JSON)

Embeddings are stored as FLOAT[] like the rest of Diego's DuckDB data.
All writes go through short-lived connections from the shared store so
the existing lock/retry handling applies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """SHA-256 of file content (read-only streaming)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    except OSError as e:
        logger.debug("hash failed for %s: %s", path, e)
        return ""
    return h.hexdigest()


class KnowledgeStore:
    """DuckDB persistence for documents, chunks, and snapshots."""

    def __init__(self, store=None):
        # Reuse the global Diego DuckDB store (shared lock handling).
        if store is None:
            from memory.duckdb_store import store as _store
            store = _store
        self._store = store

    # ── Schema ────────────────────────────────────────────────

    def initialize(self) -> None:
        self._store.initialize()
        with self._store.connect() as conn:
            if conn is None:
                return
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    path VARCHAR PRIMARY KEY,
                    filename VARCHAR,
                    root VARCHAR,
                    size BIGINT,
                    mtime DOUBLE,
                    content_hash VARCHAR,
                    file_type VARCHAR,
                    mime_type VARCHAR,
                    extraction_status VARCHAR,
                    error VARCHAR,
                    chunk_count INTEGER,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY,
                    doc_path VARCHAR,
                    chunk_index INTEGER,
                    locator VARCHAR,
                    text VARCHAR,
                    text_hash VARCHAR,
                    embedding FLOAT[],
                    embedding_model VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_snapshots (
                    id INTEGER PRIMARY KEY,
                    kind VARCHAR,
                    payload JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS seq_knowledge_chunks START 1
            """)
            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS seq_knowledge_snapshots START 1
            """)

    # ── Documents ─────────────────────────────────────────────

    def get_document(self, path: str) -> Optional[Dict]:
        with self._store.connect() as conn:
            if conn is None:
                return None
            row = conn.execute(
                "SELECT path, size, mtime, content_hash, extraction_status, "
                "chunk_count, indexed_at FROM knowledge_documents "
                "WHERE path = ?", [path]).fetchone()
        if not row:
            return None
        return {"path": row[0], "size": row[1], "mtime": row[2],
                "content_hash": row[3], "extraction_status": row[4],
                "chunk_count": row[5], "indexed_at": str(row[6])}

    def upsert_document(self, path: str, filename: str, root: str,
                        size: int, mtime: float, content_hash: str,
                        file_type: str, mime_type: str,
                        extraction_status: str, error: str,
                        chunk_count: int) -> None:
        with self._store.connect() as conn:
            if conn is None:
                return
            conn.execute("""
                INSERT INTO knowledge_documents
                    (path, filename, root, size, mtime, content_hash,
                     file_type, mime_type, extraction_status, error,
                     chunk_count, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
                ON CONFLICT (path) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    root = EXCLUDED.root,
                    size = EXCLUDED.size,
                    mtime = EXCLUDED.mtime,
                    content_hash = EXCLUDED.content_hash,
                    file_type = EXCLUDED.file_type,
                    mime_type = EXCLUDED.mime_type,
                    extraction_status = EXCLUDED.extraction_status,
                    error = EXCLUDED.error,
                    chunk_count = EXCLUDED.chunk_count,
                    indexed_at = now()
            """, [path, filename, root, size, mtime, content_hash,
                  file_type, mime_type, extraction_status, error,
                  chunk_count])

    def all_documents(self) -> List[Dict]:
        with self._store.connect() as conn:
            if conn is None:
                return []
            rows = conn.execute(
                "SELECT path, filename, root, size, mtime, content_hash, "
                "extraction_status, chunk_count, indexed_at FROM knowledge_documents"
            ).fetchall()
        return [{"path": r[0], "filename": r[1], "root": r[2],
                 "size": r[3], "mtime": r[4], "content_hash": r[5],
                 "extraction_status": r[6], "chunk_count": r[7],
                 "indexed_at": str(r[8])}
                for r in rows]

    def delete_document(self, path: str) -> None:
        """Remove a document and its chunks (deleted/moved file)."""
        with self._store.connect() as conn:
            if conn is None:
                return
            conn.execute("DELETE FROM knowledge_chunks WHERE doc_path = ?",
                         [path])
            conn.execute("DELETE FROM knowledge_documents WHERE path = ?",
                         [path])

    def document_paths(self) -> set:
        with self._store.connect() as conn:
            if conn is None:
                return set()
            rows = conn.execute(
                "SELECT path FROM knowledge_documents").fetchall()
        return {r[0] for r in rows}

    def stats(self) -> Dict:
        with self._store.connect() as conn:
            if conn is None:
                return {"documents": 0, "chunks": 0, "by_status": {},
                        "roots": []}
            docs = conn.execute(
                "SELECT COUNT(*) FROM knowledge_documents").fetchone()
            chunks = conn.execute(
                "SELECT COUNT(*) FROM knowledge_chunks").fetchone()
            by_status = conn.execute(
                "SELECT extraction_status, COUNT(*) "
                "FROM knowledge_documents GROUP BY extraction_status"
            ).fetchall()
            roots = conn.execute(
                "SELECT DISTINCT root FROM knowledge_documents"
            ).fetchall()
        return {
            "documents": docs[0] if docs else 0,
            "chunks": chunks[0] if chunks else 0,
            "by_status": {r[0]: r[1] for r in by_status},
            "roots": [r[0] for r in roots],
        }

    # ── Chunks ────────────────────────────────────────────────

    def replace_chunks(self, doc_path: str, chunks: List[Dict]) -> None:
        """Atomically replace all chunks for a document.

        Each chunk dict: text, locator, index, embedding (optional
        list/ndarray), text_hash, embedding_model.
        """
        with self._store.connect() as conn:
            if conn is None:
                return
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    "DELETE FROM knowledge_chunks WHERE doc_path = ?",
                    [doc_path])
                for c in chunks:
                    emb = c.get("embedding")
                    emb_list = None
                    if emb is not None:
                        emb_list = [float(x) for x in emb]
                    conn.execute("""
                        INSERT INTO knowledge_chunks
                            (id, doc_path, chunk_index, locator, text,
                             text_hash, embedding, embedding_model)
                        SELECT nextval('seq_knowledge_chunks'),
                               ?, ?, ?, ?, ?, ?, ?
                    """, [doc_path, c.get("index", 0),
                          c.get("locator", ""), c.get("text", ""),
                          c.get("text_hash", ""), emb_list,
                          c.get("embedding_model", "")])
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def replace_document(self, path: str, filename: str, root: str,
                         size: int, mtime: float, content_hash: str,
                         file_type: str, mime_type: str,
                         extraction_status: str, error: str,
                         chunks: List[Dict]) -> None:
        """Atomically replace a document's metadata and all its chunks.

        Single transaction: either the document row AND all its chunks
        are updated, or neither is. This is the idempotent write path
        used by the indexer (new path → insert, changed hash → replace).
        The PRIMARY KEY on path is preserved — never dropped.
        """
        with self._store.connect() as conn:
            if conn is None:
                return
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    "DELETE FROM knowledge_chunks WHERE doc_path = ?",
                    [path])
                for c in chunks:
                    emb = c.get("embedding")
                    emb_list = None
                    if emb is not None:
                        emb_list = [float(x) for x in emb]
                    conn.execute("""
                        INSERT INTO knowledge_chunks
                            (id, doc_path, chunk_index, locator, text,
                             text_hash, embedding, embedding_model)
                        SELECT nextval('seq_knowledge_chunks'),
                               ?, ?, ?, ?, ?, ?, ?
                    """, [path, c.get("index", 0),
                          c.get("locator", ""), c.get("text", ""),
                          c.get("text_hash", ""), emb_list,
                          c.get("embedding_model", "")])
                conn.execute("""
                    INSERT INTO knowledge_documents
                        (path, filename, root, size, mtime, content_hash,
                         file_type, mime_type, extraction_status, error,
                         chunk_count, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
                    ON CONFLICT (path) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        root = EXCLUDED.root,
                        size = EXCLUDED.size,
                        mtime = EXCLUDED.mtime,
                        content_hash = EXCLUDED.content_hash,
                        file_type = EXCLUDED.file_type,
                        mime_type = EXCLUDED.mime_type,
                        extraction_status = EXCLUDED.extraction_status,
                        error = EXCLUDED.error,
                        chunk_count = EXCLUDED.chunk_count,
                        indexed_at = now()
                """, [path, filename, root, size, mtime, content_hash,
                      file_type, mime_type, extraction_status, error,
                      len(chunks)])
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def all_chunks_with_embeddings(self) -> List[Dict]:
        with self._store.connect() as conn:
            if conn is None:
                return []
            rows = conn.execute("""
                SELECT c.id, c.doc_path, c.chunk_index, c.locator, c.text,
                       c.embedding, d.filename, d.file_type, d.indexed_at
                FROM knowledge_chunks c
                LEFT JOIN knowledge_documents d ON d.path = c.doc_path
            """).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0], "doc_path": r[1], "chunk_index": r[2],
                "locator": r[3], "text": r[4],
                "embedding": r[5], "filename": r[6],
                "file_type": r[7], "indexed_at": r[8],
            })
        return out

    def chunk_count(self, doc_path: str) -> int:
        with self._store.connect() as conn:
            if conn is None:
                return 0
            row = conn.execute(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE doc_path = ?",
                [doc_path]).fetchone()
        return row[0] if row else 0

    # ── Snapshots ─────────────────────────────────────────────

    def save_snapshot(self, kind: str, payload: dict) -> None:
        with self._store.connect() as conn:
            if conn is None:
                return
            conn.execute("""
                INSERT INTO knowledge_snapshots (id, kind, payload)
                SELECT nextval('seq_knowledge_snapshots'), ?, ?
            """, [kind, json.dumps(payload, default=str)])

    def latest_snapshot(self, kind: str) -> Optional[Dict]:
        with self._store.connect() as conn:
            if conn is None:
                return None
            row = conn.execute("""
                SELECT payload, created_at FROM knowledge_snapshots
                WHERE kind = ? ORDER BY created_at DESC LIMIT 1
            """, [kind]).fetchone()
        if not row:
            return None
        try:
            return {"payload": json.loads(row[0]),
                    "created_at": str(row[1])}
        except Exception:
            return None