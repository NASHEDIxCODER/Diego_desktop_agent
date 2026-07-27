"""
Embedding generation for Leo NLP pipeline.

Uses sentence-transformers to generate semantic embeddings
for intent matching and similarity search.

Caches embeddings to disk to avoid recomputation on every startup.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

# Lazy-load the model
_model = None
_cache: Dict[str, np.ndarray] = {}
_cache_dirty = False


def _load_cache() -> None:
    """Load embedding cache from disk."""
    global _cache
    cache_path = settings.EMBEDDING_CACHE_PATH
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                _cache = pickle.load(f)
            logger.info("Loaded %d cached embeddings from %s", len(_cache), cache_path)
        except Exception as e:
            logger.warning("Failed to load embedding cache: %s", e)
            _cache = {}
    else:
        _cache = {}


def _save_cache() -> None:
    """Persist embedding cache to disk."""
    global _cache_dirty
    if not _cache_dirty:
        return
    cache_path = settings.EMBEDDING_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %d cached embeddings to %s", len(_cache), cache_path)
        _cache_dirty = False
    except Exception as e:
        logger.warning("Failed to save embedding cache: %s", e)


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.MODEL_NAME)
            logger.info("Loaded embedding model: %s (dim=%d)",
                        settings.MODEL_NAME, settings.EMBEDDING_DIM)
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise
    return _model


def embed(text: str) -> np.ndarray:
    """Generate embedding vector for a single text string, using cache."""
    if not text or not text.strip():
        return np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)

    # Check cache first
    global _cache, _cache_dirty
    if not _cache:
        _load_cache()

    cache_key = text.strip().lower()
    if cache_key in _cache:
        return _cache[cache_key]

    # Compute and cache
    model = _get_model()
    vec = model.encode(cache_key, normalize_embeddings=True)
    _cache[cache_key] = vec
    _cache_dirty = True
    return vec


def embed_batch(texts: List[str]) -> np.ndarray:
    """Generate embedding vectors for a batch of texts, using cache where possible."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return np.zeros((0, settings.EMBEDDING_DIM), dtype=np.float32)

    global _cache, _cache_dirty
    if not _cache:
        _load_cache()

    # Separate cached and uncached
    uncached: List[str] = []
    uncached_indices: List[int] = []
    results: List[Optional[np.ndarray]] = [None] * len(texts)

    for i, t in enumerate(texts):
        key = t.strip().lower()
        if key in _cache:
            results[i] = _cache[key]
        else:
            uncached.append(key)
            uncached_indices.append(i)

    # Compute uncached in batch
    if uncached:
        model = _get_model()
        new_vecs = model.encode(uncached, normalize_embeddings=True)
        for idx, key, vec in zip(uncached_indices, uncached, new_vecs):
            _cache[key] = vec
            results[idx] = vec
        _cache_dirty = True

    # Save cache periodically (if dirty)
    if _cache_dirty:
        _save_cache()

    return np.array(results, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def find_best_match(query_embedding: np.ndarray,
                    stored_embeddings: np.ndarray,
                    threshold: float = None) -> Tuple[int, float]:
    """
    Find the best matching stored embedding for a query.

    Returns (best_index, best_score).
    """
    if threshold is None:
        threshold = settings.SIMILARITY_THRESHOLD
    if stored_embeddings.shape[0] == 0:
        return -1, 0.0

    scores = np.dot(stored_embeddings, query_embedding)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < threshold:
        return -1, best_score
    return best_idx, best_score