"""
Evaluator for Leo NLP pipeline.

Benchmarks intent classification accuracy, measures
latency, and provides performance reports.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from nlp.classifier import classifier
from nlp.embeddings import embed, cosine_similarity

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates NLP pipeline performance.

    Measures:
    - Classification accuracy
    - Latency per query
    - Confusion matrix
    """

    def __init__(self):
        self._results: List[Dict] = []

    def evaluate_classification(self,
                                test_cases: Dict[str, List[str]]) -> Dict:
        """
        Evaluate classification accuracy.

        Args:
            test_cases: Dict mapping expected_intent -> [test phrases]

        Returns accuracy metrics.
        """
        correct = 0
        total = 0
        confusion: Dict[str, Dict[str, int]] = {}
        latencies = []

        for expected_intent, phrases in test_cases.items():
            if expected_intent not in confusion:
                confusion[expected_intent] = {}

            for phrase in phrases:
                start = time.perf_counter()
                predicted, confidence, meta = classifier.classify(phrase)
                elapsed = time.perf_counter() - start

                latencies.append(elapsed)
                total += 1

                if predicted == expected_intent:
                    correct += 1
                else:
                    if predicted not in confusion[expected_intent]:
                        confusion[expected_intent][predicted] = 0
                    confusion[expected_intent][predicted] += 1

                self._results.append({
                    "phrase": phrase,
                    "expected": expected_intent,
                    "predicted": predicted,
                    "confidence": confidence,
                    "latency": elapsed,
                })

        accuracy = correct / total if total > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_latency_ms": avg_latency * 1000,
            "confusion_matrix": confusion,
        }

    def benchmark_embedding(self, texts: List[str],
                            iterations: int = 10) -> Dict:
        """Benchmark embedding generation speed."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            for text in texts:
                embed(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = sum(times) / len(times)
        return {
            "texts_count": len(texts),
            "iterations": iterations,
            "total_time_avg": avg,
            "avg_per_text_ms": (avg / len(texts)) * 1000 if texts else 0,
        }

    def report(self) -> str:
        """Generate a human-readable evaluation report."""
        if not self._results:
            return "No evaluation results available."

        lines = ["=" * 60, "NLP EVALUATION REPORT", "=" * 60, ""]

        # Accuracy
        correct = sum(1 for r in self._results
                      if r["predicted"] == r["expected"])
        total = len(self._results)
        accuracy = correct / total * 100 if total else 0
        lines.append(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")

        # Latency
        latencies = [r["latency"] for r in self._results]
        avg_lat = sum(latencies) / len(latencies) * 1000 if latencies else 0
        max_lat = max(latencies) * 1000 if latencies else 0
        lines.append(f"Avg latency: {avg_lat:.2f}ms")
        lines.append(f"Max latency: {max_lat:.2f}ms")

        # Per-intent breakdown
        lines.append("")
        lines.append("Per-Intent Results:")
        by_intent: Dict[str, Dict] = {}
        for r in self._results:
            intent = r["expected"]
            if intent not in by_intent:
                by_intent[intent] = {"correct": 0, "total": 0}
            by_intent[intent]["total"] += 1
            if r["predicted"] == intent:
                by_intent[intent]["correct"] += 1

        for intent, stats in sorted(by_intent.items()):
            acc = stats["correct"] / stats["total"] * 100
            lines.append(f"  {intent:25s}: {acc:5.1f}% "
                         f"({stats['correct']}/{stats['total']})")

        return "\n".join(lines)


# Global evaluator
evaluator = Evaluator()