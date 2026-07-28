"""
Intent trainer for Leo NLP pipeline.

Run manually:
    python main.py --train
    python -m nlp.trainer --train

Training flow:
    1. Load pre-computed static examples from datasets/intents/*.json
    2. Compute embeddings (cached via sentence-transformers, SHA256 keys)
    3. Store embeddings in DuckDB (single transaction, bulk insert)
    4. Build classifier index
    5. Save classifier to models/intent_classifier.pkl
    6. Save metadata to models/metadata.json

Startup NEVER retrains. Only loads pre-saved model.

Highlights:
    - NO runtime example generation (static JSON only)
    - NO recursive synonym generation
    - NO Cartesian-product template expansion
    - Heartbeat-based watchdog (aborts only if no heartbeat for timeout)
    - SHA256 cache keys (embedding_model + dataset_hash + text)
    - Batch embedding with progress bar
    - Performance timing for every stage
    - Single transaction, rollback only on actual exceptions
"""

import argparse
import hashlib
import json
import logging
import os
import pathlib
import sys
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config.settings import settings
from memory.duckdb_store import store
from nlp.classifier import classifier, BUILTIN_INTENTS
from nlp.embeddings import embed, embed_batch, set_heartbeat_callback, set_dataset_hash
from nlp.model_metadata import save_metadata, compute_dataset_hash

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────

MAX_EXAMPLES_PER_INTENT = 100       # cap per intent
DATASETS_DIR = pathlib.Path("datasets/intents")

# ── Profiling / Instrumentation ────────────────────────────

_instrumentation = {
    "call_depth": 0,
    "current_function": None,
    "start_time": None,
    "abort_flag": False,
    "examples_generated": 0,
    "current_intent": None,
    "intent_counts": {},
    "duplicate_counts": {},
    "timings": [],
}

_ABORT_REASON: Optional[str] = None


def instrument(func: Callable) -> Callable:
    """Decorator: log entry/exit, execution time, iteration counts."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        prev_func = _instrumentation["current_function"]
        _instrumentation["current_function"] = func.__name__
        _instrumentation["call_depth"] += 1

        t0 = time.perf_counter()
        logger.debug(">> ENTER %s (depth=%d)", func.__name__, _instrumentation["call_depth"])

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            _instrumentation["timings"].append((func.__name__, elapsed))
            logger.debug("<< EXIT  %s (elapsed=%.4fs)", func.__name__, elapsed)

            if func.__name__ in ("generate_examples", "_fill_template", "load_examples"):
                if result is not None:
                    if isinstance(result, dict):
                        count = sum(len(v) for v in result.values())
                    elif isinstance(result, (list, set)):
                        count = len(result)
                    else:
                        count = 1
                    _instrumentation["examples_generated"] += count

            return result

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("!! EXCEPTION in %s after %.4fs: %s", func.__name__, elapsed, e)
            raise
        finally:
            _instrumentation["call_depth"] -= 1
            _instrumentation["current_function"] = prev_func

    return wrapper


def instrument_async(func: Callable) -> Callable:
    """Decorator for async functions: same instrumentation."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        prev_func = _instrumentation["current_function"]
        _instrumentation["current_function"] = func.__name__
        _instrumentation["call_depth"] += 1

        t0 = time.perf_counter()
        logger.debug(">> ENTER %s (depth=%d)", func.__name__, _instrumentation["call_depth"])

        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            _instrumentation["timings"].append((func.__name__, elapsed))
            logger.debug("<< EXIT  %s (elapsed=%.4fs)", func.__name__, elapsed)
            return result
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("!! EXCEPTION in %s after %.4fs: %s", func.__name__, elapsed, e)
            raise
        finally:
            _instrumentation["call_depth"] -= 1
            _instrumentation["current_function"] = prev_func

    return wrapper


def check_abort() -> None:
    """Check abort conditions and raise RuntimeError if any fire."""
    global _ABORT_REASON
    if _ABORT_REASON:
        raise RuntimeError(f"Aborted: {_ABORT_REASON}")

    # Guard 1: max examples per intent
    for intent_name, cnt in _instrumentation["intent_counts"].items():
        if cnt > MAX_EXAMPLES_PER_INTENT:
            _ABORT_REASON = (f"intent '{intent_name}' exceeded "
                             f"{MAX_EXAMPLES_PER_INTENT} examples (got {cnt})")
            raise RuntimeError(f"Aborted: {_ABORT_REASON}")

    # Guard 2: duplicate ratio
    for intent_name in _instrumentation["duplicate_counts"]:
        total = _instrumentation["intent_counts"].get(intent_name, 0)
        dupes = _instrumentation["duplicate_counts"][intent_name]
        if total > 0 and (dupes / total) > 0.90:
            _ABORT_REASON = (f"intent '{intent_name}' duplicate ratio "
                             f"{dupes}/{total} > 0.90")
            raise RuntimeError(f"Aborted: {_ABORT_REASON}")


def reset_guards() -> None:
    """Reset all abort guards and instrumentation for a fresh run."""
    global _ABORT_REASON
    _ABORT_REASON = None
    _instrumentation["call_depth"] = 0
    _instrumentation["current_function"] = None
    _instrumentation["start_time"] = None
    _instrumentation["abort_flag"] = False
    _instrumentation["examples_generated"] = 0
    _instrumentation["current_intent"] = None
    _instrumentation["intent_counts"] = {}
    _instrumentation["duplicate_counts"] = {}
    _instrumentation["timings"] = []


# ── Heartbeat-based Watchdog ──────────────────────────────

class Watchdog:
    """
    Heartbeat-based watchdog.

    Never aborts while measurable progress is occurring.
    Aborts only if there has been NO heartbeat for the configured timeout.

    Default timeout: 600 seconds (configurable via TRAIN_TIMEOUT_SECONDS in .env)
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat = 0.0
        self._timeout = settings.TRAIN_TIMEOUT_SECONDS
        self._current_stage = "idle"
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_heartbeat = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Watchdog started (timeout=%ds)", self._timeout)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("Watchdog stopped")

    def heartbeat(self, stage: str = "") -> None:
        """Update the heartbeat timestamp and optionally set the current stage."""
        with self._lock:
            self._last_heartbeat = time.monotonic()
            if stage:
                self._current_stage = stage

    def _loop(self) -> None:
        while self._running:
            time.sleep(5)
            if not self._running:
                break

            with self._lock:
                elapsed_since_heartbeat = time.monotonic() - self._last_heartbeat
                stage = self._current_stage

            wall_elapsed = time.perf_counter() - (_instrumentation["start_time"] or time.perf_counter())

            logger.info(
                "[WATCHDOG] stage=%-25s heartbeat=%.1fs ago timeout=%ds elapsed=%.1fs",
                stage,
                elapsed_since_heartbeat,
                self._timeout,
                wall_elapsed,
            )

            if elapsed_since_heartbeat > self._timeout:
                global _ABORT_REASON
                _ABORT_REASON = (
                    f"No heartbeat for {elapsed_since_heartbeat:.1f}s "
                    f"(timeout={self._timeout}s) at stage '{stage}'"
                )
                logger.error("Watchdog abort: %s", _ABORT_REASON)
                # Stop so we don't keep printing abort messages
                self._running = False


watchdog = Watchdog()


def _heartbeat_callback():
    """Callback used by embeddings module to update watchdog."""
    watchdog.heartbeat("encoding_embeddings")


# ── Performance Timing ────────────────────────────────────

class Timer:
    """Simple context manager for timing blocks."""

    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._t0
        logger.info("TIMING [%-25s] %.4fs", self.name, self.elapsed)


def print_timings(timings: List[tuple]) -> None:
    """Print a sorted timing report."""
    if not timings:
        print("[TIMING] No timing data collected.")
        return

    agg: Dict[str, Dict[str, float]] = {}
    for func_name, elapsed in timings:
        if func_name not in agg:
            agg[func_name] = {"total": 0.0, "count": 0, "min": float("inf"),
                              "max": 0.0}
        agg[func_name]["total"] += elapsed
        agg[func_name]["count"] += 1
        agg[func_name]["min"] = min(agg[func_name]["min"], elapsed)
        agg[func_name]["max"] = max(agg[func_name]["max"], elapsed)

    print()
    print("=" * 70)
    print("PERFORMANCE TIMINGS")
    print("=" * 70)
    print(f"{'Stage':<30s} {'Calls':>6s} {'Total (s)':>10s} "
          f"{'Avg (s)':>10s} {'Min (s)':>10s} {'Max (s)':>10s}")
    print("-" * 70)
    sorted_funcs = sorted(agg.items(), key=lambda x: x[1]["total"], reverse=True)
    for func_name, stats in sorted_funcs:
        avg = stats["total"] / stats["count"] if stats["count"] else 0
        print(f"{func_name:<30s} {stats['count']:>6d} {stats['total']:>10.4f} "
              f"{avg:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f}")
    print("=" * 70)


# ── Static Dataset Loader ──────────────────────────────────

def load_static_datasets() -> Dict[str, List[str]]:
    """
    Load all intent examples from datasets/intents/*.json.

    Each JSON file must be a flat list of example strings.
    Returns a dict mapping intent_name -> list of examples.
    """
    datasets_dir = DATASETS_DIR
    if not datasets_dir.exists():
        logger.warning("Datasets directory not found: %s", datasets_dir)
        return {}

    all_examples: Dict[str, List[str]] = {}
    for json_path in sorted(datasets_dir.glob("*.json")):
        intent_name = json_path.stem
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            # Support both flat list and {"examples": [...]} format
            if isinstance(data, list):
                examples = data
            elif isinstance(data, dict) and "examples" in data:
                examples = data["examples"]
            else:
                logger.warning("Invalid format in %s: expected list or {examples: [...]}", json_path)
                continue
            examples = [str(e).strip() for e in examples if e and str(e).strip()]
            all_examples[intent_name] = examples
            logger.debug("Loaded %d examples for '%s' from %s",
                         len(examples), intent_name, json_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", json_path, e)

    logger.info("Loaded %d intents from %s (total examples=%d)",
                len(all_examples), datasets_dir,
                sum(len(v) for v in all_examples.values()))
    return all_examples


# ── Trainer ────────────────────────────────────────────────

class Trainer:
    """
    Loads pre-computed static datasets, trains classifier, persists to disk.

    NO runtime example generation.
    NO recursive synonym generation.
    NO Cartesian-product template expansion.
    """

    def __init__(self):
        self._trained = False

    # ── Dataset loading ─────────────────────────────────────

    @instrument
    def load_examples(self) -> Dict[str, List[str]]:
        """Load pre-computed examples from static JSON datasets."""
        all_examples = load_static_datasets()

        missing = [n for n in BUILTIN_INTENTS
                   if n != "unknown" and n not in all_examples]
        if missing:
            logger.warning("Missing static datasets for intents: %s", missing)

        return all_examples

    # ── Guarded per-intent loader with abort checks ─────────

    @instrument
    def get_examples_for_intent(self, intent_name: str,
                                max_count: int = 100) -> List[str]:
        """Return examples for a single intent from the static datasets."""
        check_abort()
        _instrumentation["current_intent"] = intent_name

        all_examples = load_static_datasets()
        examples = all_examples.get(intent_name, [])

        if not examples:
            logger.warning("No static examples for intent '%s'", intent_name)
            return []

        if len(examples) > max_count:
            logger.warning("Intent '%s' has %d examples, truncating to %d",
                           intent_name, len(examples), max_count)
            examples = examples[:max_count]

        _instrumentation["intent_counts"][intent_name] = len(examples)
        check_abort()
        return examples

    # ── Generation (kept for backward compat but loads from static) ─

    @instrument
    def generate_examples(self, intent_name: str, count: int = 100) -> List[str]:
        """Load examples from static dataset (NOT runtime generation)."""
        check_abort()
        _instrumentation["current_intent"] = intent_name
        examples = self.get_examples_for_intent(intent_name, max_count=count)
        _instrumentation["examples_generated"] += len(examples)
        return examples

    @instrument
    def _fill_template(self, template: str) -> str:
        """DEPRECATED: kept only for reference. Not used in training."""
        logger.warning("_fill_template called but runtime generation is DISABLED")
        return template

    @instrument
    def generate_all(self, examples_per_intent: int = 100) -> Dict[str, List[str]]:
        """Load all examples from static JSON datasets."""
        reset_guards()
        _instrumentation["start_time"] = time.perf_counter()

        all_examples = self.load_examples()

        result: Dict[str, List[str]] = {}
        total = 0
        for intent_name, examples in all_examples.items():
            _instrumentation["current_intent"] = intent_name
            check_abort()

            capped = examples[:examples_per_intent]
            result[intent_name] = capped
            _instrumentation["intent_counts"][intent_name] = len(capped)
            _instrumentation["examples_generated"] += len(capped)
            total += len(capped)

            unique = set(capped)
            duplicates = len(capped) - len(unique)
            _instrumentation["duplicate_counts"][intent_name] = duplicates
            if duplicates > 0:
                logger.warning("Intent '%s': %d duplicates in static dataset",
                               intent_name, duplicates)

            check_abort()

        logger.info("Loaded %d examples across %d intents from static datasets",
                    total, len(result))
        return result

    # ── Training ────────────────────────────────────────────

    @instrument_async
    async def train(self, examples_per_intent: int = 100) -> int:
        """
        Full training pipeline with heartbeat-based watchdog.

        1. Start watchdog (heartbeat-based, 600s timeout)
        2. Begin a single DuckDB transaction
        3. Delete derived tables (intent_embeddings, cached_embeddings, intent_examples)
        4. Load static datasets (JSON)
        5. Compute embeddings in batches (batch_size=64, heartbeat after each)
        6. Store in DuckDB (bulk insert)
        7. Build classifier index
        8. Save to models/intent_classifier.pkl
        9. Save metadata to models/metadata.json
        10. Commit transaction

        On failure, rolls back the transaction so no partial state persists.

        Returns total example count.
        """
        reset_guards()
        _instrumentation["start_time"] = time.perf_counter()

        t0 = time.perf_counter()
        logger.info("Training with %d examples per intent...", examples_per_intent)

        # Register heartbeat callback from embeddings module
        set_heartbeat_callback(_heartbeat_callback)

        # Start watchdog
        watchdog.start()
        watchdog.heartbeat("loading_dataset")

        # Begin a single transaction for the entire pipeline
        store.begin_transaction()

        try:
            # ── Stage 1: Clear derived tables ──
            with Timer("delete_derived_tables"):
                store.delete_derived_tables()
            watchdog.heartbeat("loading_dataset")

            # ── Stage 2: Load static datasets ──
            with Timer("load_datasets"):
                all_examples = self.generate_all(examples_per_intent)
            watchdog.heartbeat("loading_dataset")

            if not all_examples:
                logger.warning("No examples loaded, nothing to train")
                store.commit()
                return 0

            total = 0
            intent_names = sorted(all_examples.keys())

            # Compute dataset hash once and set it for embedding cache keys
            dataset_hash = compute_dataset_hash(all_examples)
            set_dataset_hash(dataset_hash)

            # ── Stage 3: Collect ALL examples and encode in batches ──
            watchdog.heartbeat("encoding_embeddings")

            # Collect all examples with their intent names
            all_texts: List[str] = []
            all_intent_labels: List[str] = []
            for intent_name in intent_names:
                examples = all_examples[intent_name]
                for ex in examples:
                    all_texts.append(ex)
                    all_intent_labels.append(intent_name)

            logger.info("Encoding %d examples in batches...", len(all_texts))

            with Timer("embedding_generation"):
                # Encode all in one batch call (handles caching, batching, heartbeat)
                all_embeddings = embed_batch(
                    all_texts,
                    batch_size=64,
                    show_progress=True
                )
            watchdog.heartbeat("writing_database")

            # ── Stage 4: Store in DuckDB ──
            with Timer("database_writes"):
                for i, intent_name in enumerate(intent_names):
                    _instrumentation["current_intent"] = intent_name
                    examples = all_examples[intent_name]
                    intent_id = store.add_intent(intent_name)

                    # Find the indices for this intent
                    start_idx = total
                    end_idx = total + len(examples)
                    for j, example in enumerate(examples):
                        idx = start_idx + j
                        emb = all_embeddings[idx] if idx < len(all_embeddings) else None
                        store.add_example(intent_id, example, emb)

                    # Add to classifier
                    classifier.add_intent(intent_name, examples)
                    total += len(examples)

                    watchdog.heartbeat("writing_database")

            # ── Stage 5: Build classifier index ──
            watchdog.heartbeat("training_classifier")
            with Timer("classifier_training"):
                classifier.build_index()

            # ── Stage 6: Persist to models/ ──
            watchdog.heartbeat("saving_model")
            with Timer("model_serialization"):
                classifier.save()
                training_time = time.perf_counter() - t0
                save_metadata(all_examples, total, training_time=training_time)

            # ── Commit the transaction ──
            store.commit()

            self._trained = True
            elapsed = time.perf_counter() - t0
            logger.info("Training complete: %d examples across %d intents in %.1fs",
                        total, len(all_examples), elapsed)

            # Print timing report
            print_timings(_instrumentation["timings"])

            return total

        except Exception as e:
            logger.error("Training failed, rolling back transaction: %s", e)
            store.rollback()
            raise

        finally:
            watchdog.stop()
            set_heartbeat_callback(None)

    def is_trained(self) -> bool:
        return self._trained


# Global singleton
trainer = Trainer()


# ── CLI entry point ────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Leo NLP Trainer")
    print("=" * 60)
    print()
    print("This trainer loads pre-computed static datasets from:")
    print(f"  {DATASETS_DIR.resolve()}")
    print()
    print("To regenerate the static datasets:")
    print("  python scripts/generate_datasets.py")
    print()
    print("To run training:")
    print("  python main.py --train")
    print()

    # Quick validation
    print("Validating datasets...")
    examples = load_static_datasets()
    print(f"Loaded {len(examples)} intents")
    total_ex = sum(len(v) for v in examples.values())
    print(f"Total examples: {total_ex}")

    # Validate each intent has at least 30 examples
    for intent_name, ex_list in examples.items():
        if len(ex_list) < 30:
            print(f"  WARNING: '{intent_name}' has only {len(ex_list)} examples (need 30+)")
        else:
            print(f"  OK: '{intent_name}' has {len(ex_list)} examples")