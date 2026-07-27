"""
Entity extractor for Leo NLP pipeline.

Uses spaCy NER and regex patterns to extract entities
like names, numbers, dates, and custom slot values.
"""

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to load spaCy NER
try:
    import spacy
    _ner = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    HAS_NER = True
    logger.info("spaCy NER loaded for entity extraction")
except Exception:
    HAS_NER = False
    logger.warning("spaCy NER not available, using regex entity extraction")

# Patterns for common entities
PATTERNS = {
    "number": re.compile(r"\b(\d+)\b"),
    "percentage": re.compile(r"\b(\d{1,3})\s*%"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(\+?\d[\d\s\-()]{7,15}\d)\b"),
    "url": re.compile(r"https?://[^\s]+"),
    "time": re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\b"),
    "brightness": re.compile(r"(?:brightness|bright|dim|light)\s*(?:to\s*)?(\d{1,3})", re.I),
    "volume": re.compile(r"(?:volume|vol)\s*(?:to\s*)?(\d{1,3})", re.I),
    "speed": re.compile(r"(?:speed|rate)\s*(?:to\s*)?(\d+(?:\.\d+)?)", re.I),
    "seconds": re.compile(r"(\d+)\s*seconds?"),
}


def extract_numbers(text: str) -> List[int]:
    """Extract all integers from text."""
    return [int(m) for m in PATTERNS["number"].findall(text)]


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract entities from text.

    Returns dict with keys like:
    - numbers: list of integers
    - names: list of person names
    - places: list of location names
    - percentages: list of integers
    - emails: list of email strings
    - phone: phone number strings
    - urls: list of URLs
    - brightness: brightness value (0-100)
    - volume: volume value (0-100)
    - speed: playback speed float
    - seconds: time duration in seconds
    """
    entities: Dict[str, Any] = {
        "numbers": [],
        "names": [],
        "places": [],
        "percentages": [],
        "emails": [],
        "phones": [],
        "urls": [],
        "brightness": None,
        "volume": None,
        "speed": None,
        "seconds": None,
    }

    if HAS_NER and text.strip():
        doc = _ner(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["names"].append(ent.text)
            elif ent.label_ in ("GPE", "LOC"):
                entities["places"].append(ent.text)

    # Regex extractions
    nums = extract_numbers(text)
    entities["numbers"] = nums

    for match in PATTERNS["brightness"].finditer(text):
        entities["brightness"] = max(0, min(100, int(match.group(1))))

    for match in PATTERNS["volume"].finditer(text):
        entities["volume"] = max(0, min(100, int(match.group(1))))

    for match in PATTERNS["speed"].finditer(text):
        entities["speed"] = float(match.group(1))

    for match in PATTERNS["seconds"].finditer(text):
        entities["seconds"] = int(match.group(1))

    entities["emails"] = PATTERNS["email"].findall(text)
    entities["phones"] = PATTERNS["phone"].findall(text)
    entities["urls"] = PATTERNS["url"].findall(text)

    # Clean empty lists
    for key in ["names", "places", "emails", "phones", "urls"]:
        entities[key] = list(set(entities[key]))

    return entities


def extract_target_name(text: str, intent: str) -> Optional[str]:
    """
    Extract a target name from Telegram-related commands.
    E.g., "send message to John" -> "John"
    """
    # send message to X
    m = re.search(r"(?:to|for)\s+([A-Za-z\s]+?)(?:\s+(?:saying|that|and|about|the|a|an)\s|$)", text, re.I)
    if m:
        return m.group(1).strip()
    # read from X
    m = re.search(r"(?:from|for)\s+([A-Za-z\s]+?)(?:\s+(?:saying|that|and|about|the|a|an)\s|$)", text, re.I)
    if m:
        return m.group(1).strip()
    return None