"""Profile knowledge.diagnostics.answer_diagnostic_query (real runtime)."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

from knowledge.diagnostics import (
    detect_diagnostic_query, answer_diagnostic_query,
    collect_runtime_health, collect_latency_metrics,
    collect_system_resources,
)

text = "how are you doing today"
q = detect_diagnostic_query(text)
print("topic:", q.topic)

t = time.perf_counter(); rh = collect_runtime_health()
print(f"collect_runtime_health: {(time.perf_counter()-t)*1000:.1f}ms -> {len(rh)} subsystems")

t = time.perf_counter(); lm = collect_latency_metrics()
print(f"collect_latency_metrics: {(time.perf_counter()-t)*1000:.1f}ms")

t = time.perf_counter(); sr = collect_system_resources()
print(f"collect_system_resources: {(time.perf_counter()-t)*1000:.1f}ms")

t = time.perf_counter(); r = answer_diagnostic_query(text)
print(f"answer_diagnostic_query total: {(time.perf_counter()-t)*1000:.1f}ms")
print("response:", r)

t = time.perf_counter(); r2 = answer_diagnostic_query(text)
print(f"answer_diagnostic_query second call: {(time.perf_counter()-t)*1000:.1f}ms")