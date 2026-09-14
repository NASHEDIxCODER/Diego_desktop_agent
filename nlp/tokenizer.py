"""
Text tokenizer for Diego NLP pipeline.

Splits text into tokens using spaCy or a simple fallback.

NOTE (Phase 21D): spaCy is loaded LAZILY on first use (cached), NOT at
module import time. The import-time `spacy.load("en_core_web_sm")` call
was taking ~12s when the model was unavailable, blocking the first context
composition. Lazy loading means import is instant; the (optional) spaCy
load is triggered by the first tokenize() call or by warm_tokenizer().
"""

import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy state (loaded on first use, cached thereafter).
_nlp = None
_HAS_SPACY: Optional[bool] = None
_load_lock = threading.Lock()


def _load_spacy() -> None:
    """Load spaCy once (thread-safe). Sets _HAS_SPACY / _nlp."""
    global _nlp, _HAS_SPACY
    if _HAS_SPACY is not None:
        return
    with _load_lock:
        if _HAS_SPACY is not None:
            return
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm",
                               disable=["parser", "ner", "lemmatizer"])
            _HAS_SPACY = True
            logger.info("[Tokenizer] spaCy loaded for tokenization")
        except Exception:
            _HAS_SPACY = False
            logger.warning("[Tokenizer] spaCy not available, using simple "
                           "tokenizer")


def warm_tokenizer() -> None:
    """Eagerly trigger the (optional) spaCy load in a background thread so
    the first tokenize()/estimate_tokens() call does not pay the load cost.
    Safe to call during reasoning-subsystem initialization. Non-blocking."""
    _load_spacy()  # fast if already loaded; otherwise kicks off the load


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    _load_spacy()
    if _HAS_SPACY:
        doc = _nlp(text)
        return [token.text.lower() for token in doc if not token.is_space]
    # Simple fallback
    return [w.strip(".,!?;:'\"()[]{}").lower() for w in text.split() if w.strip()]


def tokenize_with_pos(text: str) -> List[tuple]:
    """Tokenize and return (token, pos_tag) pairs."""
    if not text:
        return []
    _load_spacy()
    if _HAS_SPACY:
        doc = _nlp(text)
        return [(token.text.lower(), token.pos_) for token in doc if not token.is_space]
    return [(w.lower(), "UNKNOWN") for w in text.split() if w.strip()]


def sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []
    _load_spacy()
    if _HAS_SPACY:
        doc = _nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    # Simple fallback
    import re
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]