"""
Text tokenizer for Diego NLP pipeline.

Splits text into tokens using spaCy or a simple fallback.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try to load spaCy; fall back to simple split
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    HAS_SPACY = True
    logger.info("spaCy loaded for tokenization")
except Exception:
    HAS_SPACY = False
    logger.warning("spaCy not available, using simple tokenizer")


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    if HAS_SPACY:
        doc = _nlp(text)
        return [token.text.lower() for token in doc if not token.is_space]
    # Simple fallback
    return [w.strip(".,!?;:'\"()[]{}").lower() for w in text.split() if w.strip()]


def tokenize_with_pos(text: str) -> List[tuple]:
    """Tokenize and return (token, pos_tag) pairs."""
    if not text:
        return []
    if HAS_SPACY:
        doc = _nlp(text)
        return [(token.text.lower(), token.pos_) for token in doc if not token.is_space]
    return [(w.lower(), "UNKNOWN") for w in text.split() if w.strip()]


def sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []
    if HAS_SPACY:
        doc = _nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    # Simple fallback
    import re
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]