"""
Local retrieval over the knowledge index: semantic + keyword, fused.

Ranking combines:
  * relevance      — cosine similarity (semantic) + keyword overlap score
  * source quality — extracted documents rank above metadata-only
  * recency        — recently indexed documents get a small boost

Results carry full citation metadata: absolute file path, filename,
and page/sheet/section locator. Nothing is sent to any LLM here.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from knowledge.embedder import LocalEmbedder
from knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text or "") if len(w) > 1]


class KnowledgeRetriever:
    """Hybrid (semantic + keyword) retrieval with citation metadata."""

    def __init__(self, store: KnowledgeStore,
                 embedder: Optional[LocalEmbedder] = None):
        self._store = store
        self._embedder = embedder or LocalEmbedder()
        self._cache: Optional[List[Dict]] = None
        self._cache_loaded_at: float = 0.0
        self._cache_ttl = 30.0  # seconds

    def invalidate_cache(self) -> None:
        self._cache = None

    def _load_chunks(self) -> List[Dict]:
        now = time.monotonic()
        if (self._cache is not None
                and now - self._cache_loaded_at < self._cache_ttl):
            return self._cache
        chunks = self._store.all_chunks_with_embeddings()
        # Pre-compute numpy embeddings once per cache cycle
        for c in chunks:
            emb = c.get("embedding")
            c["_vec"] = (np.asarray(emb, dtype=np.float32)
                         if emb else None)
        self._cache = chunks
        self._cache_loaded_at = now
        return chunks

    # ── Retrieval ─────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search. Returns ranked results with citations.

        Each result: text, doc_path, filename, locator, chunk_index,
        score, source (semantic/keyword/both), file_type.
        """
        t0 = time.time()
        query = (query or "").strip()
        if not query:
            return []
        chunks = self._load_chunks()
        if not chunks:
            logger.info("[KNOWLEDGE] retrieval: index empty, query skipped")
            return []

        q_tokens = set(_tokens(query))

        # Keyword scores (always computed — cheap, local)
        kw_scores: Dict[int, float] = {}
        for i, c in enumerate(chunks):
            c_tokens = set(_tokens(c.get("text", "")))
            if not c_tokens:
                continue
            overlap = q_tokens & c_tokens
            if overlap:
                kw_scores[i] = len(overlap) / max(1, len(q_tokens))

        # Semantic scores (local model)
        sem_scores: Dict[int, float] = {}
        q_vec = self._embedder.embed_query(query)
        if q_vec is not None:
            for i, c in enumerate(chunks):
                v = c.get("_vec")
                if v is None or v.size == 0:
                    continue
                denom = (np.linalg.norm(v) * np.linalg.norm(q_vec)) + 1e-10
                sem_scores[i] = float(np.dot(v, q_vec) / denom)

        # Fuse: 0.6 * semantic + 0.4 * keyword (either alone still ranks)
        candidates = set(kw_scores) | set(sem_scores)
        results: List[Dict] = []
        for i in candidates:
            sem = sem_scores.get(i, 0.0)
            kw = kw_scores.get(i, 0.0)
            score = 0.6 * sem + 0.4 * kw
            c = chunks[i]
            # Source quality: extracted > metadata-only docs
            if c.get("file_type"):
                score += 0.02
            # Recency boost (up to +0.05, decays over 7 days)
            try:
                age_days = (time.time() - _to_ts(c.get("indexed_at"))) / 86400
                score += max(0.0, 0.05 - 0.05 * (age_days / 7.0))
            except Exception:
                pass
            source = ("both" if i in sem_scores and i in kw_scores
                      else "semantic" if i in sem_scores else "keyword")
            results.append({
                "text": c.get("text", ""),
                "doc_path": c.get("doc_path", ""),
                "filename": c.get("filename", ""),
                "locator": c.get("locator", ""),
                "chunk_index": c.get("chunk_index", 0),
                "file_type": c.get("file_type", ""),
                "score": round(score, 4),
                "source": source,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:top_k]
        latency_ms = (time.time() - t0) * 1000
        logger.info(
            "[KNOWLEDGE] retrieval query=%r results=%d latency=%.0fms "
            "sources=%s", query[:60], len(results), latency_ms,
            [r["doc_path"] for r in results])
        return results

    def context_for_llm(self, query: str, top_k: int = 4,
                        max_chars: int = 2500) -> str:
        """Smallest relevant retrieved context, formatted for the LLM.

        Returns "" when the index has nothing relevant — the caller then
        falls back to the normal LLM path without local grounding.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        parts: List[str] = []
        total = 0
        for r in results:
            cite = f"{r['doc_path']}"
            if r.get("locator"):
                cite += f" ({r['locator']})"
            block = f"[{cite}]\n{r['text']}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)


def _to_ts(value) -> float:
    """Best-effort conversion of a DuckDB timestamp to epoch seconds."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime
        if isinstance(value, datetime):
            return value.timestamp()
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0