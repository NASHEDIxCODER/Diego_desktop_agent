"""
Text normalizer for Leo NLP pipeline.

Handles:
- Lowercasing
- Punctuation removal
- Whitespace normalization
- Synonym replacement
- Contraction expansion
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Common contractions
CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "can't": "cannot",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "let's": "let us",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
}


def normalize(text: str, synonyms: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Normalize text: lowercase, remove extra whitespace,
    expand contractions, replace synonyms.
    """
    if not text:
        return ""

    text = text.lower().strip()

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Remove punctuation (keep apostrophes for contractions already expanded)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Synonym replacement
    if synonyms:
        words = text.split()
        for i, word in enumerate(words):
            for canonical, syn_list in synonyms.items():
                if word in syn_list:
                    words[i] = canonical
                    break
        text = " ".join(words)

    return text


def remove_stopwords(tokens: List[str],
                     stopwords: Optional[set] = None) -> List[str]:
    """Remove common stopwords from token list."""
    if stopwords is None:
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our",
            "their", "this", "that", "these", "those", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "up", "about",
            "into", "over", "after", "before", "between", "under",
            "and", "but", "or", "nor", "not", "so", "yet", "if",
            "because", "as", "until", "while", "then", "than", "too",
            "very", "just", "also", "now", "here", "there", "when",
            "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "some", "any", "no", "only", "own",
            "same", "such", "what", "which", "who", "whom",
        }
    return [t for t in tokens if t not in stopwords]