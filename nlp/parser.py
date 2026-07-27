"""
Main NLP parser for Leo Desktop Assistant.

Orchestrates the full NLP pipeline:
1. Tokenize
2. Normalize
3. Classify intent
4. Extract entities
5. Score confidence
6. Route to plugin
7. Update context
"""

import logging
from typing import Any, Dict, Optional, Tuple

from config.settings import settings
from nlp.classifier import classifier, IntentClassifier
from nlp.confidence import confidence_scorer, evaluate_confidence
from nlp.context import context_manager, ContextManager
from nlp.entities import extract_entities, extract_target_name
from nlp.normalizer import normalize
from nlp.tokenizer import tokenize

logger = logging.getLogger(__name__)


class NLPParser:
    """
    Main NLP parser that orchestrates the pipeline.

    Usage:
        parser = NLPParser()
        result = await parser.parse("set brightness to 50")
        # result = {
        #     "intent": "brightness_set",
        #     "confidence": 0.92,
        #     "entities": {"brightness": 50},
        #     "text": "set brightness to 50",
        #     "response": "...",
        # }
    """

    def __init__(self):
        self._classifier = classifier
        self._context = context_manager
        self._scorer = confidence_scorer

    async def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse user text through the full NLP pipeline.

        Returns a dict with:
        - text: original text
        - normalized: normalized text
        - intent: classified intent name
        - confidence: confidence score
        - entities: extracted entities dict
        - confidence_meta: confidence evaluation metadata
        - needs_llm: whether LLM fallback is needed
        """
        if not text or not text.strip():
            return {
                "text": text or "",
                "normalized": "",
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "confidence_meta": evaluate_confidence(0.0),
                "needs_llm": True,
            }

        # Check context for active session
        active_session = self._context.active_session
        if active_session:
            # If we're in a session, prepend session context
            logger.debug("Active session: %s", active_session)

        # Step 1: Normalize
        normalized = normalize(text)
        logger.debug("Normalized: %s -> %s", text, normalized)

        # Step 2: Classify intent
        intent, confidence, meta = self._classifier.classify(normalized)
        logger.debug("Intent: %s (confidence=%.4f, meta=%s)", intent, confidence, meta)

        # Step 3: Extract entities
        entities = extract_entities(text)

        # Step 4: Extract target name for communication intents
        if intent.startswith("telegram"):
            target = extract_target_name(text, intent)
            if target:
                entities["target"] = target

        # Step 5: Evaluate confidence
        conf_meta = evaluate_confidence(confidence, intent)
        needs_llm = conf_meta["needs_llm"]

        # Step 6: Update context
        self._context.update(text, intent, entities, confidence)

        result = {
            "text": text,
            "normalized": normalized,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "classification_meta": meta,
            "confidence_meta": conf_meta,
            "needs_llm": needs_llm,
        }

        return result

    async def parse_with_llm_fallback(self, text: str,
                                      llm_func=None) -> Dict[str, Any]:
        """
        Parse text and fall back to LLM if confidence is low.

        Args:
            text: User input text
            llm_func: Optional async callable that takes (text, context)
                       and returns a response string.

        Returns parse result dict with optional 'llm_response' key.
        """
        result = await self.parse(text)

        if result["needs_llm"] and llm_func is not None:
            try:
                # Build context for LLM
                context_str = f"Last intent: {self._context.get_last_intent(2) or 'none'}"
                llm_response = await llm_func(text, context_str)
                result["llm_response"] = llm_response
                result["intent"] = "llm_fallback"
                logger.info("LLM fallback used for: %s", text)
            except Exception as e:
                logger.error("LLM fallback error: %s", e)
                result["llm_response"] = None

        return result

    def set_context(self, intent: str, session: Optional[str] = None) -> None:
        """Manually set context (used by plugins)."""
        self._context._current_intent = intent
        if session:
            self._context.set_active_session(session)

    def clear_context(self) -> None:
        """Clear conversation context."""
        self._context.clear()

    @property
    def context(self) -> ContextManager:
        return self._context


# Global parser instance
parser = NLPParser()


async def parse_text(text: str) -> Dict[str, Any]:
    """Convenience function to parse text."""
    return await parser.parse(text)