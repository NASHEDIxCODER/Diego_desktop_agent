"""
Local-only embedding backend for the knowledge index.

Uses the EXISTING local sentence-transformers model (nlp.embeddings /
settings.MODEL_NAME) — no cloud API, no API key, fully offline after the
first model download. The backend is configurable via
KNOWLEDGE_EMBEDDING_BACKEND so it can be swapped later.

Embeddings are cached by chunk content hash: a chunk is re-embedded only
when its text changes (text_hash stored per chunk in DuckDB).
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

_MODEL_NAME = settings.MODEL_NAME  # local sentence-transformers model


def text_hash(text: str) -> str:
    """Deterministic hash of chunk text (embedding cache key)."""
    return hashlib.sha256(
        f"{_MODEL_NAME}::||::{text}".encode("utf-8")).hexdigest()


class LocalEmbedder:
    """Local sentence-transformers embedder (no cloud calls)."""

    name = "local_sentence_transformers"

    def __init__(self, backend: str = ""):
        self.backend = backend or settings.KNOWLEDGE_EMBEDDING_BACKEND
        self._ready = False
        self._failed = False  # memoized: never spam the log per file

    @property
    def is_local(self) -> bool:
        """True — this backend never performs network calls at query time."""
        return True

    @property
    def available(self) -> bool:
        """True when the embedding model is loaded (or can be)."""
        return self._ready or self._ensure_model()

    @property
    def status(self) -> str:
        """Non-probing status string for diagnostics/status reports."""
        if self._ready:
            return "ready"
        if self._failed:
            return "unavailable (keyword-only retrieval fallback)"
        return "not loaded yet"

    def _ensure_model(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False  # already reported — no per-file log spam
        try:
            from nlp import embeddings as _emb
            # Preload is safe: model is cached locally / offline mode.
            _emb.preload_embedding_model()
            self._ready = True
            return True
        except Exception as e:
            # One warning per PROCESS — indexing continues with
            # keyword-only retrieval (degraded, clearly reported).
            self._failed = True
            logger.warning(
                "[KNOWLEDGE] embedding model unavailable (%s) — falling "
                "back to keyword-only retrieval for this process", e)
            return False

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Embed texts; returns None for entries that fail (graceful)."""
        if not texts:
            return []
        if self._failed or not self._ensure_model():
            return [None] * len(texts)
        try:
            from nlp import embeddings as _emb
            vecs = _emb.embed_batch(texts, show_progress=False)
            return [np.asarray(v, dtype=np.float32) for v in vecs]
        except Exception as e:
            logger.warning("[KNOWLEDGE] batch embedding failed: %s", e)
            return [None] * len(texts)

    def embed_query(self, text: str) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None
        if self._failed or not self._ensure_model():
            return None
        try:
            from nlp import embeddings as _emb
            return np.asarray(_emb.embed(text), dtype=np.float32)
        except Exception as e:
            logger.warning("[KNOWLEDGE] query embedding failed: %s", e)
            return None