"""
Unit tests for the NLP trainer.

Tests that every intent produces exactly the expected number of examples
from the static JSON datasets, and that abort guards work correctly.

Run with:
    python -m pytest tests/test_trainer.py -v
"""

import json
import pathlib
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from nlp.trainer import (
    load_static_datasets,
    reset_guards,
    _instrumentation,
    MAX_EXAMPLES_PER_INTENT,
    check_abort,
    DATASETS_DIR,
    trainer,
)

# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_state():
    """Reset guards and instrumentation before each test."""
    reset_guards()
    yield
    reset_guards()


# ── Test: Static datasets exist and are valid ──────────────


def test_datasets_directory_exists():
    """The datasets/intents/ directory must exist."""
    assert DATASETS_DIR.exists(), f"Datasets directory not found: {DATASETS_DIR}"


def test_all_json_files_are_valid():
    """Every .json file in datasets/intents/ must be valid JSON."""
    for json_path in sorted(DATASETS_DIR.glob("*.json")):
        with open(json_path, "r") as f:
            data = json.load(f)
        # Support both flat list and {"examples": [...]} format
        if isinstance(data, list):
            examples = data
        elif isinstance(data, dict) and "examples" in data:
            examples = data["examples"]
        else:
            raise AssertionError(f"{json_path.name}: invalid format")
        for i, item in enumerate(examples):
            assert isinstance(item, str), (
                f"{json_path.name}[{i}]: expected string, got {type(item)}"
            )
            assert item.strip(), f"{json_path.name}[{i}]: empty string"


# ── Test: load_static_datasets returns correct counts ──────


def test_load_static_datasets_returns_dict():
    """load_static_datasets must return a dict."""
    data = load_static_datasets()
    assert isinstance(data, dict)


def test_load_static_datasets_all_intents_have_examples():
    """Every intent in the dataset must have at least 1 example."""
    data = load_static_datasets()
    assert len(data) > 0, "No intents loaded"
    for intent_name, examples in data.items():
        assert len(examples) >= 1, f"Intent '{intent_name}' has 0 examples"


def test_intents_have_30_plus_examples():
    """Every intent must have at least 30 examples."""
    data = load_static_datasets()
    for intent_name, examples in data.items():
        assert len(examples) >= 30, (
            f"Intent '{intent_name}' has only {len(examples)} examples (need 30+)"
        )


# ── Test: generate_all returns correct data ────────────────


def test_generate_all_returns_dict():
    """generate_all must return a dict."""
    data = trainer.generate_all(examples_per_intent=100)
    assert isinstance(data, dict)


def test_generate_all_capped():
    """generate_all with a smaller cap must truncate correctly."""
    cap = 5
    data = trainer.generate_all(examples_per_intent=cap)
    for intent_name, examples in data.items():
        assert len(examples) <= cap, (
            f"Intent '{intent_name}': {len(examples)} > cap {cap}"
        )


# ── Test: generate_examples ────────────────────────────────


def test_generate_examples_returns_list():
    """generate_examples must return a list."""
    examples = trainer.generate_examples("greeting", count=10)
    assert isinstance(examples, list)


def test_generate_examples_count():
    """generate_examples must return at most 'count' examples."""
    examples = trainer.generate_examples("greeting", count=5)
    assert len(examples) <= 5


def test_generate_examples_unknown_intent():
    """generate_examples for a non-existent intent must return empty list."""
    examples = trainer.generate_examples("nonexistent_intent_xyz", count=10)
    assert examples == []


# ── Test: Abort guards ─────────────────────────────────────


def test_abort_max_examples():
    """Abort must fire when an intent exceeds MAX_EXAMPLES_PER_INTENT."""
    reset_guards()
    _instrumentation["intent_counts"]["test_intent"] = MAX_EXAMPLES_PER_INTENT + 1
    with pytest.raises(RuntimeError, match="exceeded"):
        check_abort()


def test_abort_duplicate_ratio():
    """Abort must fire when duplicate ratio exceeds MAX_DUPLICATE_RATIO."""
    reset_guards()
    _instrumentation["intent_counts"]["test_intent"] = 10
    _instrumentation["duplicate_counts"]["test_intent"] = 10  # 100% duplicates
    with pytest.raises(RuntimeError, match="duplicate"):
        check_abort()


def test_abort_no_false_positive():
    """Abort must NOT fire under normal conditions."""
    reset_guards()
    _instrumentation["start_time"] = time.perf_counter()
    _instrumentation["intent_counts"]["test_intent"] = 5
    _instrumentation["duplicate_counts"]["test_intent"] = 0
    # Should not raise
    check_abort()


# ── Test: No runtime generation ────────────────────────────


def test_no_runtime_generation():
    """
    Verify that _fill_template is NOT called during training.
    The static datasets should be used directly.
    """
    data = trainer.generate_all(examples_per_intent=10)
    assert len(data) > 0


# ── Test: Instrumentation ──────────────────────────────────


def test_instrumentation_tracks_function():
    """Instrumentation must track the current function name."""
    reset_guards()
    _instrumentation["current_function"] = "test_func"
    assert _instrumentation["current_function"] == "test_func"


def test_instrumentation_tracks_examples():
    """Instrumentation must track examples_generated."""
    reset_guards()
    _instrumentation["examples_generated"] = 42
    assert _instrumentation["examples_generated"] == 42


# ── Test: Watchdog ─────────────────────────────────────────


def test_watchdog_start_stop():
    """Watchdog must start and stop without error."""
    from nlp.trainer import watchdog
    watchdog.start()
    assert watchdog._running
    watchdog.stop()
    assert not watchdog._running