"""
Confidence scoring for Diego NLP pipeline.

Computes confidence scores for intent classifications
and determines when to fall back to LLM.
"""

import logging
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Evaluates confidence of intent classification and
    determines if LLM fallback is needed.
    """

    def __init__(self, threshold: Optional[float] = None):
        self._threshold = threshold or settings.SIMILARITY_THRESHOLD

    def is_reliable(self, confidence: float) -> bool:
        """Check if confidence score is reliable enough."""
        return confidence >= self._threshold

    def needs_llm_fallback(self, confidence: float,
                           intent: str = "unknown") -> bool:
        """Determine if LLM fallback should be used."""
        if intent == "unknown":
            return True
        if confidence < self._threshold * 0.8:
            return True
        return False

    def should_confirm(self, confidence: float) -> bool:
        """Determine if we should ask the user to confirm."""
        return self._threshold * 0.6 <= confidence < self._threshold

    def get_confidence_label(self, confidence: float) -> str:
        """Get a human-readable confidence label."""
        if confidence >= 0.95:
            return "very_high"
        elif confidence >= 0.85:
            return "high"
        elif confidence >= self._threshold:
            return "medium"
        elif confidence >= self._threshold * 0.6:
            return "low"
        else:
            return "very_low"

    def adjust_threshold(self, intent: str) -> float:
        """
        Return an adjusted threshold for specific intents.
        Some intents (like 'exit', 'youtube_close') need higher
        thresholds to avoid false positives.
        """
        high_confidence_intents = {
            "exit", "youtube_close", "shutdown", "power_off",
            "youtube_pause", "youtube_resume",
        }
        if intent in high_confidence_intents:
            return min(self._threshold + 0.1, 0.95)
        return self._threshold


# Global scorer
confidence_scorer = ConfidenceScorer()


def evaluate_confidence(confidence: float, intent: str = "unknown") -> dict:
    """Evaluate confidence and return decision metadata."""
    scorer = ConfidenceScorer()
    return {
        "score": confidence,
        "label": scorer.get_confidence_label(confidence),
        "reliable": scorer.is_reliable(confidence),
        "needs_llm": scorer.needs_llm_fallback(confidence, intent),
        "should_confirm": scorer.should_confirm(confidence),
    }