
"""
Embedding generation for Leo NLP pipeline.

Uses sentence-transformers to generate semantic embeddings
for intent matching and similarity search.

Caches embeddings to disk to avoid recomputation on every startup.
Cache key is SHA256(embedding_model + dataset_hash + text).

Refactored to:
- Load SentenceTransformer once (singleton)
- Collect ALL examples, encode in batches (batch_size=64)
- normalize_embeddings=True, show_progress_bar=True
- Store embeddings after batch completion
- Heartbeat after every batch
"""

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Monkey-patch: prevent transformers from importing TF classes ──────────
import transformers.utils.import_utils as _transformers_import_utils
_transformers_import_utils._tf_available = False

# Lazy-loaded model singleton
_model = None
_cache: Dict[str, np.ndarray] = {}
_cache_dirty = False
_cache_loaded = False

# Heartbeat callback — set by trainer to update watchdog
_heartbeat_callback: Optional[callable] = None

# Dataset hash for cache key — set by trainer before encoding
_dataset_hash: str = ""


def set_heartbeat_callback(cb: Optional[callable]) -> None:
    """Set a callback that the watchdog uses to know we're alive."""
    global _heartbeat_callback
    _heartbeat_callback = cb


def _heartbeat() -> None:
    """Notify watchdog that work is progressing."""
    if _heartbeat_callback:
        _heartbeat_callback()


def set_dataset_hash(dataset_hash: str) -> None:
    """Set the dataset hash for cache key computation."""
    global _dataset_hash
    _dataset_hash = dataset_hash


def _compute_cache_key(text: str) -> str:
    """Compute a deterministic cache key using SHA256.

    Includes the embedding model name and dataset hash so that changing
    either the model or the dataset invalidates all caches automatically.
    """
    raw = f"{settings.MODEL_NAME}::||::{_dataset_hash}::||::{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cache() -> None:
    """Load embedding cache from disk."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
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
    _cache_loaded = True


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
        _cache_dirty = False
    except Exception as e:
        logger.warning("Failed to save embedding cache: %s", e)


def _get_model():
    """Lazy-load the sentence-transformers model (loaded once)."""
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


def preload_embedding_model() -> None:
    """
    Preload the SentenceTransformer model at startup.
    
    This ensures the model is loaded exactly once, before any
    command handling occurs. No HuggingFace HTTP requests will
    happen during runtime inference.
    
    Call this during startup_diagnostics() before entering the
    main loop.
    """
    logger.info("Preloading embedding model: %s", settings.MODEL_NAME)
    _get_model()
    logger.info("Embedding model preloaded successfully")


def embed(text: str) -> np.ndarray:
    """Generate embedding vector for a single text string, using cache."""
    if not text or not text.strip():
        return np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)

    global _cache, _cache_dirty
    _load_cache()

    cache_key = _compute_cache_key(text.strip().lower())
    if cache_key in _cache:
        return _cache[cache_key]

    # Compute and cache
    model = _get_model()
    vec = model.encode(text.strip().lower(), normalize_embeddings=True)
    _cache[cache_key] = vec
    _cache_dirty = True
    return vec


def embed_batch(texts: List[str],
                batch_size: int = 64,
                show_progress: bool = True) -> np.ndarray:
    """Generate embedding vectors for a batch of texts, using cache where possible.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts per encoding batch (default: 64).
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return np.zeros((0, settings.EMBEDDING_DIM), dtype=np.float32)

    global _cache, _cache_dirty
    _load_cache()

    # Separate cached and uncached
    uncached_texts: List[str] = []
    uncached_indices: List[int] = []
    results: List[Optional[np.ndarray]] = [None] * len(texts)
    cached_count = 0

    for i, t in enumerate(texts):
        key = _compute_cache_key(t.strip().lower())
        if key in _cache:
            results[i] = _cache[key]
            cached_count += 1
        else:
            uncached_texts.append(t.strip().lower())
            uncached_indices.append(i)

    if cached_count > 0:
        logger.debug("embed_batch: %d/%d cached", cached_count, len(texts))

    # Compute uncached in batches with progress bar
    if uncached_texts:
        model = _get_model()

        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(uncached_texts), desc="Encoding embeddings", unit="text")
            except ImportError:
                pbar = None
        else:
            pbar = None

        # Process in mini-batches
        for start_idx in range(0, len(uncached_texts), batch_size):
            end_idx = min(start_idx + batch_size, len(uncached_texts))
            batch = uncached_texts[start_idx:end_idx]

            try:
                new_vecs = model.encode(batch,
                                        normalize_embeddings=True,
                                        show_progress_bar=False)
            except Exception as e:
                logger.error("Batch encoding failed at indices %d-%d: %s",
                             start_idx, end_idx, e)
                raise

            for j, (idx, key) in enumerate(
                zip(uncached_indices[start_idx:end_idx],
                    [None] * (end_idx - start_idx))
            ):
                key = _compute_cache_key(batch[j])
                vec = new_vecs[j]
                _cache[key] = vec
                results[uncached_indices[start_idx + j]] = vec

            _cache_dirty = True

            if pbar:
                pbar.update(len(batch))

            # Heartbeat after every batch so watchdog knows we're alive
            _heartbeat()

        if pbar:
            pbar.close()

        # Save cache after all batches
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