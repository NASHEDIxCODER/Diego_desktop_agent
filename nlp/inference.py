"""
Inference-only NLP engine for Leo.

- Loads pre-trained classifier from disk.
- Loads cached embeddings from disk.
- Only embeds incoming user queries.
- NEVER trains or recomputes training data.
- Returns top-k intents with confidence scores.
- Routes to Plugin Manager.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from nlp.model_metadata import is_model_ready, get_model_status
from nlp.classifier import classifier
from nlp.embeddings import embed, _load_cache
from nlp.normalizer import normalize

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Production inference engine.

    Loads on startup. Never trains. Fast path for every query.
    """

    def __init__(self):
        self._loaded = False

    def load(self) -> bool:
        """Load pre-trained classifier + cached embeddings. Returns True on success."""
        if self._loaded:
            return True

        if not is_model_ready():
            logger.warning("No trained model found. Run: leo train")
            return False

        # Load classifier
        if not classifier.load():
            logger.error("Failed to load classifier from %s", settings.CLASSIFIER_PATH)
            return False

        # Load embedding cache (silent if missing — embeddings recompute lazily)
        _load_cache()

        self._loaded = True
        logger.info("Inference engine loaded: %s", get_model_status().get("message", "ok"))
        return True

    def classify(self, text: str,
                 top_k: int = 3,
                 threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Classify text into top-k intents.

        Args:
            text: User input text.
            top_k: Number of top intents to return.
            threshold: Minimum confidence threshold.

        Returns:
            List of dicts: [{intent, confidence, metadata}, ...]
            Sorted by confidence descending.
        """
        if threshold is None:
            threshold = settings.SIMILARITY_THRESHOLD

        if not self._loaded:
            return [{"intent": "unknown", "confidence": 0.0, "metadata": {}}]

        # Normalize
        norm_text = normalize(text)
        if not norm_text:
            return [{"intent": "unknown", "confidence": 0.0, "metadata": {}}]

        # Embed query only
        query_emb = embed(norm_text)

        # Get all intent scores
        scores: List[Tuple[str, float, Dict]] = []

        # 1. Semantic similarity against all examples
        if classifier._all_examples:
            all_embs = classifier._example_embeddings.get("all")
            if all_embs is not None and all_embs.shape[0] > 0:
                dot_scores = np.dot(all_embs, query_emb)
                # Get top-k from examples
                top_indices = np.argsort(dot_scores)[-top_k:][::-1]
                for idx in top_indices:
                    intent = classifier._all_intent_labels[idx]
                    score = float(dot_scores[idx])
                    if score >= threshold:
                        scores.append((intent, score,
                                       {"matched_example": classifier._all_examples[idx]}))

        # 2. Centroid matching for coverage
        if not scores:
            best_intent = "unknown"
            best_score = 0.0
            for intent_name, centroid in classifier._intent_embeddings.items():
                if centroid.shape[0] > 0:
                    sim = float(np.dot(centroid, query_emb) /
                                (np.linalg.norm(centroid) * np.linalg.norm(query_emb) + 1e-10))
                    if sim > best_score:
                        best_score = sim
                        best_intent = intent_name

            if best_score >= threshold:
                scores.append((best_intent, best_score, {}))

        # 3. Fuzzy fallback (rapidfuzz preferred, difflib fallback)
        if not scores:
            best_intent = "unknown"
            best_score = 0.0
            best_example = ""
            try:
                from rapidfuzz import fuzz as _fuzz
                for intent_name, examples in classifier._intents.items():
                    for example in examples:
                        ratio = _fuzz.ratio(norm_text.lower(), example.lower()) / 100.0
                        if ratio > best_score:
                            best_score = ratio
                            best_intent = intent_name
                            best_example = example
            except ImportError:
                # Fallback to difflib if rapidfuzz is not installed
                import difflib
                for intent_name, examples in classifier._intents.items():
                    for example in examples:
                        ratio = difflib.SequenceMatcher(
                            None, norm_text.lower(), example.lower()
                        ).ratio()
                        if ratio > best_score:
                            best_score = ratio
                            best_intent = intent_name
                            best_example = example
            if best_score >= threshold:
                scores.append((best_intent, best_score, {"fuzzy_match": best_example}))

        # Format results
        if not scores:
            return [{"intent": "unknown", "confidence": 0.0, "metadata": {}}]

        # Sort by confidence descending
        scores.sort(key=lambda x: x[1], reverse=True)

        result = []
        for intent, conf, meta in scores[:top_k]:
            result.append({
                "intent": intent,
                "confidence": round(float(conf), 4),
                "metadata": meta,
            })

        return result

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        base = get_model_status()
        base["loaded"] = self._loaded
        return base


# Global singleton
inference = InferenceEngine()