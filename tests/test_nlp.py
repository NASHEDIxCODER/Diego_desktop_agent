"""
Tests for the NLP pipeline.

Run with:
    python -m pytest tests/test_nlp.py -v
"""

import pytest

# We skip tests that require heavy models
pytestmark = pytest.mark.skipif(
    True,
    reason="NLP tests require sentence-transformers model download"
)


class TestTokenizer:
    def test_basic(self):
        """Placeholder for tokenizer tests."""
        pass


class TestNormalizer:
    def test_contractions(self):
        """Placeholder for normalizer tests."""
        pass


class TestClassifier:
    def test_builtin_intents(self):
        """Placeholder for classifier tests."""
        pass


class TestEntities:
    def test_extract_numbers(self):
        """Placeholder for entity tests."""
        pass


class TestContext:
    def test_history(self):
        """Placeholder for context tests."""
        pass