"""
MetricsRegistry — Central metrics collection for the runtime dashboard.

Collects:
  * System metrics   — CPU %, RAM MB, GPU util/VRAM (optional)
  * Audio metrics    — microphone RMS, camera FPS
  * Model metrics    — wake score, VAD prob, Whisper latency, LLM latency
  * Service metrics  — per-service health, restarts
  * State metrics    — current runtime state, state history
  * Event metrics    — events/sec (from the bus)

Thread-safe: metrics are written from executor threads and read from the
async dashboard. Uses a lock-free snapshot via periodic sampling.

Usage:
    from core.metrics import metrics

    metrics.set("wake.score", 0.93)
    metrics.set("llm.first_token_ms", 450)
    snapshot = metrics.snapshot()
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Rolling window length for time-series metrics
WINDOW = 120  # samples (~2 min at 1 Hz)

# Historical lengths
HISTORY_MAX = 3600  # ~1 hour of 1 Hz samples


class MetricsRegistry:
    """
    Thread-safe metrics registry.

    Values are plain scalars or time-series (list-like). The dashboard
    reads snapshots at 1 Hz.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._values: Dict[str, Any] = {}
        self._series: Dict[str, Deque[float]] = {}
        self._timestamps: Dict[str, Deque[float]] = {}
        self._start_time = time.time()

    # ── Core API ───────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Set a scalar metric."""
        with self._lock:
            self._values[key] = value

    def record(self, key: str, value: float, ts: Optional[float] = None) -> None:
        """Record a time-series data point (appends to a rolling window)."""
        if value is None:
            return
        now = ts if ts is not None else time.time()
        with self._lock:
            series = self._series.setdefault(key, deque(maxlen=WINDOW))
            tseries = self._timestamps.setdefault(key, deque(maxlen=WINDOW))
            series.append(float(value))
            tseries.append(now)

    def record_event(self, key: str, value: float = 1.0) -> None:
        """Record a discrete event (e.g. a wake detection)."""
        self.record(key, value)

    def increment(self, key: str, delta: float = 1.0) -> None:
        """Atomically increment a counter."""
        with self._lock:
            self._values[key] = float(self._values.get(key, 0.0)) + delta

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def series(self, key: str) -> List[float]:
        with self._lock:
            return list(self._series.get(key, []))

    def series_recent(self, key: str, n: int = 10) -> List[float]:
        with self._lock:
            s = self._series.get(key, [])
            return list(s)[-n:]

    def latest(self, key: str, default: Any = None) -> Any:
        """Most recent value of a series, or a scalar, or default."""
        with self._lock:
            if key in self._values:
                return self._values[key]
            s = self._series.get(key)
            if s:
                return s[-1]
            return default

    def average(self, key: str, default: float = 0.0) -> float:
        """Average of the recent series window."""
        with self._lock:
            s = self._series.get(key)
            if not s:
                return default
            try:
                return float(np.mean(list(s)))
            except Exception:
                vals = list(s)
                return float(sum(vals) / len(vals)) if vals else default

    def count(self, key: str) -> int:
        """Number of events recorded in the current window."""
        with self._lock:
            return len(self._series.get(key, []))

    def rate(self, key: str, window_s: float = 60.0) -> float:
        """Events per second over the recent window."""
        with self._lock:
            tseries = self._timestamps.get(key)
            if not tseries:
                return 0.0
            now = time.time()
            cutoff = now - window_s
            return sum(1 for t in tseries if t >= cutoff) / window_s

    def reset(self, key: Optional[str] = None) -> None:
        """Reset a metric (or all metrics when key is None)."""
        with self._lock:
            if key is None:
                self._values.clear()
                self._series.clear()
                self._timestamps.clear()
            else:
                self._values.pop(key, None)
                self._series.pop(key, None)
                self._timestamps.pop(key, None)

    # ── Snapshot for the dashboard ─────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Takes a thread-safe snapshot of all metrics."""
        with self._lock:
            values = dict(self._values)
            latest_series = {
                k: (list(v)[-1] if v else None)
                for k, v in self._series.items()
            }
            rates = {
                f"{k}.rate": self.rate(k)
                for k in self._series
            }
            averages = {
                f"{k}.avg": self.average(k)
                for k in self._series
            }
        snap = {**values, **latest_series, **rates, **averages}
        snap["_uptime_s"] = time.time() - self._start_time
        snap["_n_metrics"] = len(values) + len(self._series)
        return snap

    def clear_series(self) -> None:
        with self._lock:
            self._series.clear()
            self._timestamps.clear()


# Global singleton
metrics = MetricsRegistry()


# ═══════════════════════════════════════════════════════════
# System metrics sampler
# ═══════════════════════════════════════════════════════════

class SystemMetricsSampler:
    """
    Samples CPU / RAM / GPU at a fixed interval (default 1s).

    Runs as an asyncio task. Uses psutil when available; falls back to
    /proc on Linux. The sampler NEVER blocks boot and survives exceptions.
    """

    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._task: Optional[asyncio.Task] = None
        self._psutil = None
        self._running = False

    def _init_psutil(self):
        if self._psutil is not None:
            return
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None

    def _sample_cpu_ram(self) -> Dict[str, float]:
        """Returns {cpu_percent, ram_used_mb, ram_total_mb, ram_percent}."""
        if self._psutil:
            try:
                cpu = self._psutil.cpu_percent(interval=None)
                vm = self._psutil.virtual_memory()
                return {
                    "cpu_percent": float(cpu),
                    "ram_used_mb": float(vm.used) / (1024 * 1024),
                    "ram_total_mb": float(vm.total) / (1024 * 1024),
                    "ram_percent": float(vm.percent),
                }
            except Exception:
                pass
        # Fallback: /proc
        try:
            with open("/proc/stat", "r") as f:
                fields = f.readline().split()[1:]
            idle = int(fields[3])
            total = sum(int(x) for x in fields)
            # Diff from last sample
            prev = getattr(self, "_prev_stat", None)
            self._prev_stat = (idle, total)
            if prev:
                dtotal = max(total - prev[1], 1)
                didle = idle - prev[0]
                cpu = (1.0 - didle / dtotal) * 100.0
            else:
                cpu = 0.0
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem_total = int(lines[0].split()[1]) / 1024  # kB → MB
            mem_avail = int(lines[2].split()[1]) / 1024 if len(lines) > 2 else mem_total
            mem_used = mem_total - mem_avail
            return {
                "cpu_percent": float(cpu),
                "ram_used_mb": float(mem_used),
                "ram_total_mb": float(mem_total),
                "ram_percent": (mem_used / mem_total * 100.0) if mem_total else 0.0,
            }
        except Exception:
            return {}

    def _sample_gpu(self) -> Dict[str, float]:
        """Returns {gpu_util_percent, gpu_vram_mb} when nvidia-smi works."""
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                line = out.stdout.strip().splitlines()[0]
                util, vram = line.split(",")
                return {
                    "gpu_util_percent": float(util.strip()),
                    "gpu_vram_mb": float(vram.strip()),
                }
        except Exception:
            pass
        return {}

    async def run(self) -> None:
        """Sample loop. NEVER raises — survives all exceptions."""
        self._running = True
        self._init_psutil()
        try:
            while self._running:
                try:
                    sys_m = self._sample_cpu_ram()
                    for k, v in sys_m.items():
                        metrics.record(f"system.{k}", v)
                    gpu_m = self._sample_gpu()
                    for k, v in gpu_m.items():
                        metrics.record(f"system.{k}", v)
                    metrics.set("system._last_sample", time.time())
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("[METRICS] sample error: %s", e)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    def start(self) -> None:
        """Start the sampler. Safe to call from any context — if no event
        loop is running, the sampler is started lazily on the next loop."""
        if self._running:
            return
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self.run())
        except RuntimeError:
            # No running loop — schedule on the default loop when available.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._task = loop.create_task(self.run())
                else:
                    # Loop exists but not running yet — start when it runs.
                    self._task = loop.create_task(self.run())
            except Exception:
                logger.warning("[METRICS] No event loop available — "
                               "sampler will start on next boot")

    def stop(self) -> None:
        """Stop the sampler (idempotent)."""
        self._running = False
        if self._task is not None:
            try:
                self._task.cancel()
            except Exception:
                pass
            self._task = None


# Global sampler
system_metrics = SystemMetricsSampler()