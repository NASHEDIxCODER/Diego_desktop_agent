"""
Local retrieval over the knowledge index: semantic + keyword, fused.

Ranking combines:
  * relevance      — cosine similarity (semantic) + keyword overlap score
  * source quality — extracted documents rank above metadata-only
  * recency        — recently indexed documents get a small boost

Guarantees:
  * Keyword-only retrieval works WITHOUT embeddings at FULL weight — a
    chunk covering the whole query ranks as high as a strong semantic
    match (the 0.6/0.4 fusion only applies when a query vector exists,
    otherwise keyword scores are never scaled down to 0.4x).
  * Deterministic ranking: results are ordered by score with ties broken
    by (doc_path, chunk_index, locator) — identical inputs always
    produce identical output regardless of set iteration order.
  * Relevance floor: results below MIN_RELEVANCE are dropped — weak
    matches are never surfaced as authoritative local knowledge.
  * PC snapshot facts (diego:// virtual documents) stay OUT of document
    retrieval results — they are served separately via the snapshot API.
  * Citations carry the absolute file path plus the page/sheet/section
    locator where applicable. Nothing is sent to any LLM here.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

import numpy as np

from knowledge.embedder import LocalEmbedder
from knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")

# Relevance floor: results scoring below this are dropped — weak matches
# must never be surfaced as authoritative local knowledge. The caller
# (brain) applies its own stricter threshold for direct answers.
MIN_RELEVANCE = 0.20

# Minimal stopword set — query filler words must NOT inflate keyword
# coverage ("what is the Diego architecture" must not score 0.75 on a
# chunk containing only the filler words).
_STOPWORDS = frozenset({
    "the", "is", "a", "an", "of", "to", "in", "on", "at", "for", "and",
    "or", "but", "it", "its", "this", "that", "these", "those", "what",
    "how", "why", "when", "where", "who", "which", "are", "was", "were",
    "be", "been", "being", "my", "your", "me", "you", "i", "we", "they",
    "he", "she", "his", "her", "with", "as", "by", "from", "about",
    "into", "tell", "show", "give", "get", "do", "does", "did", "can",
    "could", "will", "would", "please",
})

# Virtual document paths (PC snapshots) are kept out of document search.
_VIRTUAL_PREFIX = "diego://"


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text or "") if len(w) > 1]


def _query_tokens(text: str) -> set:
    """Content tokens of a query (stopwords removed).

    Coverage is measured against CONTENT tokens only so filler words
    never inflate a chunk's keyword score.
    """
    return {t for t in (w.lower() for w in _TOKEN_RE.findall(text or ""))
            if len(t) > 1 and t not in _STOPWORDS}


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
        # PC snapshot facts (diego:// virtual documents) stay OUT of
        # document retrieval — they are served separately.
        chunks = [c for c in self._store.all_chunks_with_embeddings()
                  if not str(c.get("doc_path", "")).startswith(
                      _VIRTUAL_PREFIX)]
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

        Deterministic: identical index + query always produce identical
        ranked output. Results below MIN_RELEVANCE are dropped.
        """
        t0 = time.time()
        query = (query or "").strip()
        if not query:
            return []
        chunks = self._load_chunks()
        if not chunks:
            logger.info("[KNOWLEDGE] retrieval: index empty, query skipped")
            return []

        q_tokens = _query_tokens(query)

        # Keyword scores (always computed — cheap, local). Coverage of
        # the CONTENT query tokens: a chunk containing the whole query
        # scores 1.0.
        kw_scores: Dict[int, float] = {}
        if q_tokens:
            for i, c in enumerate(chunks):
                c_tokens = set(_tokens(c.get("text", "")))
                if not c_tokens:
                    continue
                overlap = q_tokens & c_tokens
                if overlap:
                    kw_scores[i] = len(overlap) / len(q_tokens)

        # Semantic scores (local model). When the model is unavailable
        # the keyword score takes FULL weight so keyword-only retrieval
        # can still rank authoritative matches high.
        q_vec = self._embedder.embed_query(query)
        sem_available = q_vec is not None
        sem_scores: Dict[int, float] = {}
        if sem_available:
            for i, c in enumerate(chunks):
                v = c.get("_vec")
                if v is None or v.size == 0:
                    continue
                denom = (np.linalg.norm(v) * np.linalg.norm(q_vec)) + 1e-10
                sem_scores[i] = float(np.dot(v, q_vec) / denom)

        # Fuse: 0.6*semantic + 0.4*keyword when a query vector exists;
        # full keyword weight otherwise (both modes reach 1.0).
        kw_weight = 0.4 if sem_available else 1.0
        sem_weight = 0.6 if sem_available else 0.0

        candidates = set(kw_scores) | set(sem_scores)
        results: List[Dict] = []
        for i in candidates:
            sem = sem_scores.get(i, 0.0)
            kw = kw_scores.get(i, 0.0)
            score = sem_weight * sem + kw_weight * kw
            if score < MIN_RELEVANCE:
                continue  # low-quality match — never authoritative
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

        # DETERMINISTIC RANKING: primary by score, ties broken by
        # (doc_path, chunk_index, locator) — never by set iteration
        # order, so identical inputs always produce identical output.
        results.sort(key=lambda r: (-r["score"], r["doc_path"],
                                    r["chunk_index"], r["locator"]))
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

        * Only results above the relevance floor are included — weak
          matches are never presented as authoritative grounding.
        * Hard-bounded by max_chars. The first oversized block is
          truncated so a single large chunk cannot empty the context.
        * Every block carries a full citation: file path + locator.
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
            text = r["text"] or ""
            block = f"[{cite}]\n{text}"
            if total + len(block) > max_chars:
                if not parts and max_chars > 0:
                    # First block alone exceeds the budget — truncate it
                    # so relevant grounding is never lost entirely.
                    budget = max(0, max_chars - len(cite) - 4)
                    parts.append(f"[{cite}]\n{text[:budget]}")
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