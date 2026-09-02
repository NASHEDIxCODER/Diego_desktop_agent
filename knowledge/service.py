"""
KnowledgeService — facade over the local knowledge subsystem.

Wires: policy → indexer → store → embedder → retriever → snapshot.

Startup contract:
  * start() NEVER blocks Diego startup — it only schedules background
    work (indexing thread + optional snapshot refresh).
  * All retrieval is local (no cloud API).

Developer/test commands (see knowledge/cli.py):
    index files / index status / reindex file / search local knowledge /
    rebuild embeddings / show indexed roots / show skipped paths
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional

from config.settings import settings
from knowledge.embedder import LocalEmbedder
from knowledge.indexer import KnowledgeIndexer
from knowledge.policy import PathPolicy
from knowledge.retriever import KnowledgeRetriever
from knowledge.snapshot import collect_snapshot, snapshot_text
from knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


class KnowledgeService:
    """Singleton facade for the local knowledge subsystem."""

    def __init__(self):
        self._policy = PathPolicy.from_settings()
        self._store = KnowledgeStore()
        self._embedder = LocalEmbedder()
        self._indexer = KnowledgeIndexer(self._store, self._policy,
                                         self._embedder)
        self._retriever = KnowledgeRetriever(self._store, self._embedder)
        self._snapshot_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Non-blocking startup: initialize schema + kick off a
        background incremental scan. Returns immediately."""
        if self._started:
            return
        self._started = True
        try:
            self._store.initialize()
        except Exception as e:
            logger.warning("[KNOWLEDGE] store init failed (non-fatal): %s", e)
            return
        # Background indexing — never blocks startup. The INITIAL scan
        # runs in a daemon thread; Diego never waits for it.
        try:
            self._indexer.start_background_scan()
        except Exception as e:
            logger.warning("[KNOWLEDGE] background scan failed to start: %s",
                           e)
        # CONTINUOUS INDEXING: periodic incremental rescan picks up
        # new/changed/deleted files automatically (daemon, serialized,
        # cancellable — no startup blocking, no user-facing latency).
        try:
            self._indexer.start_periodic_rescan()
        except Exception as e:
            logger.warning("[KNOWLEDGE] periodic rescan failed to start: %s",
                           e)
        # Periodic snapshot refresh (background, optional)
        if settings.KNOWLEDGE_SNAPSHOT_REFRESH_S > 0:
            self._stop.clear()
            self._snapshot_thread = threading.Thread(
                target=self._snapshot_loop, name="knowledge-snapshot",
                daemon=True)
            self._snapshot_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._indexer.stop_periodic_rescan()
        self._indexer.cancel()

    def ensure_ready(self) -> None:
        """Initialize the schema WITHOUT starting background work."""
        try:
            self._store.initialize()
        except Exception as e:
            logger.warning("[KNOWLEDGE] store init failed (non-fatal): %s", e)

    # ── Retrieval (local-first knowledge routing) ─────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid local search with citations."""
        try:
            return self._retriever.search(query, top_k=top_k)
        except Exception as e:
            logger.warning("[KNOWLEDGE] search failed: %s", e)
            return []

    def context_for_llm(self, query: str, top_k: int = 4) -> str:
        """Smallest relevant local context for the LLM ("" if none)."""
        try:
            return self._retriever.context_for_llm(query, top_k=top_k)
        except Exception as e:
            logger.warning("[KNOWLEDGE] context failed: %s", e)
            return ""

    def invalidate_retriever_cache(self) -> None:
        try:
            self._retriever.invalidate_cache()
        except Exception:
            pass

    # ── Indexing commands ─────────────────────────────────────

    def index_files(self) -> Dict:
        """Synchronous incremental scan (developer command)."""
        return self._indexer.scan()

    def index_status(self) -> Dict:
        """Accurate status: roots, docs, chunks, scan state, embedding
        availability. Always reads the LIVE database (never reports an
        empty state while a scan is populating it)."""
        self.ensure_ready()
        status = self._store.stats()
        status["scan"] = self._indexer.progress.snapshot()
        status["roots"] = self.indexed_roots()
        status["embeddings"] = {
            "backend": self._embedder.name,
            "local_only": True,
            "status": self._embedder.status,
        }
        return status

    def reindex_file(self, path: str) -> Dict:
        return self._indexer.reindex_file(path)

    def rebuild_embeddings(self) -> Dict:
        """Force re-embedding of all chunks (developer command)."""
        count = 0
        try:
            with self._store._store.connect() as conn:
                if conn is not None:
                    conn.execute(
                        "UPDATE knowledge_chunks SET embedding = NULL")
            docs = self._store.all_documents()
            for d in docs:
                self._indexer.reindex_file(d["path"])
                count += 1
        except Exception as e:
            logger.warning("[KNOWLEDGE] rebuild embeddings failed: %s", e)
        return {"reindexed": count}

    def indexed_roots(self) -> List[str]:
        return [str(r) for r in self._policy.allow_roots]

    def skipped_sensitive(self) -> List[Dict]:
        return self._policy.skipped_report()

    # ── PC snapshot ───────────────────────────────────────────

    def refresh_snapshot(self, kind: str = "pc_inventory") -> Dict:
        """Collect + persist a fresh PC snapshot (read-only)."""
        snap = collect_snapshot()
        try:
            self._store.save_snapshot(kind, snap)
        except Exception as e:
            logger.warning("[KNOWLEDGE] snapshot persist failed: %s", e)
        # Also index the snapshot text so it is retrievable
        try:
            text = snapshot_text(snap)
            if text:
                chunks = []
                from knowledge.chunker import chunk_text
                for i, c in enumerate(chunk_text(text)):
                    chunks.append({
                        "index": i, "locator": "pc snapshot",
                        "text": c.text,
                        "text_hash": __import__(
                            "knowledge.embedder", fromlist=["text_hash"]
                        ).text_hash(c.text),
                        "embedding": None,
                        "embedding_model": "",
                    })
                self._store.replace_chunks("diego://pc-snapshot", chunks)
                self._store.upsert_document(
                    path="diego://pc-snapshot", filename="pc-snapshot",
                    root="diego://", size=len(text),
                    mtime=time.time(), content_hash="",
                    file_type="snapshot", mime_type="text/plain",
                    extraction_status="extracted", error="",
                    chunk_count=len(chunks))
                self._retriever.invalidate_cache()
        except Exception as e:
            logger.debug("[KNOWLEDGE] snapshot indexing skipped: %s", e)
        return snap

    def latest_snapshot(self, kind: str = "pc_inventory") -> Optional[Dict]:
        try:
            return self._store.latest_snapshot(kind)
        except Exception:
            return None

    def _snapshot_loop(self) -> None:
        interval = max(60, settings.KNOWLEDGE_SNAPSHOT_REFRESH_S)
        while not self._stop.is_set():
            try:
                self.refresh_snapshot()
            except Exception as e:
                logger.debug("[KNOWLEDGE] snapshot refresh failed: %s", e)
            self._stop.wait(interval)


# Global singleton
knowledge_service = KnowledgeService()