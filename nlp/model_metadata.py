"""
Model metadata management for Diego NLP.

Tracks:
- model_version
- dataset_hash (SHA256 of all intent examples)
- embedding_model
- created_at
- num_intents
- num_examples
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


def compute_dataset_hash(intents: Dict[str, List[str]]) -> str:
    """Compute a SHA256 hash of all intent examples to detect changes."""
    # Sort for deterministic hashing
    lines = []
    for intent_name in sorted(intents.keys()):
        examples = sorted(intents[intent_name])
        for ex in examples:
            lines.append(f"{intent_name}:{ex}")
    payload = "\n".join(lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_metadata() -> Optional[Dict[str, Any]]:
    """Load model metadata from disk. Returns None if missing."""
    path = settings.METADATA_PATH
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load metadata: %s", e)
        return None


def save_metadata(intents: Dict[str, List[str]],
                  num_examples: int,
                  training_time: Optional[float] = None) -> Dict[str, Any]:
    """Save model metadata to disk."""
    metadata = {
        "model_version": settings.MODEL_VERSION,
        "dataset_hash": compute_dataset_hash(intents),
        "embedding_model": settings.MODEL_NAME,
        "embedding_dim": settings.EMBEDDING_DIM,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_intents": len(intents),
        "num_examples": num_examples,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    }
    if training_time is not None:
        metadata["training_time"] = round(training_time, 2)
    path = settings.METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", path)
    return metadata


def dataset_has_changed(intents: Dict[str, List[str]]) -> bool:
    """Check if the dataset has changed since the last training."""
    meta = load_metadata()
    if meta is None:
        return True
    old_hash = meta.get("dataset_hash")
    if not old_hash:
        return True
    new_hash = compute_dataset_hash(intents)
    return old_hash != new_hash


def is_model_ready() -> bool:
    """Check if a trained model exists on disk."""
    if not settings.CLASSIFIER_PATH.exists():
        return False
    meta = load_metadata()
    if meta is None:
        return False
    return True


def get_model_status() -> Dict[str, Any]:
    """Get a human-readable model status dict."""
    meta = load_metadata()
    if meta is None:
        return {
            "ready": False,
            "message": "No trained model found. Run: Diego train",
        }
    return {
        "ready": True,
        "version": meta.get("model_version", "unknown"),
        "embedding_model": meta.get("embedding_model", "unknown"),
        "intents": meta.get("num_intents", 0),
        "examples": meta.get("num_examples", 0),
        "trained_at": meta.get("created_at", "unknown"),
        "threshold": meta.get("similarity_threshold", 0.75),
    }