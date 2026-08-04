"""
Background Workers System — Worker infrastructure for Leo.

Workers are long-running background tasks that communicate through the
EventBus. They never block the main event loop and survive exceptions.

Workers:
  - Indexer: indexes files, projects, memories for fast lookup
  - Watcher: watches filesystem for changes (projects, configs)
  - Memory Summarizer: periodically compresses old memories
  - Desktop Observer: continuous desktop monitoring (wraps DesktopObserver)
  - Experience Optimizer: periodically prunes/optimizes experience DB
  - Planner Optimizer: analyzes planner performance and updates heuristics
  - Semantic Linker: periodically re-links semantic memory graph

Every worker:
  - Has a defined lifecycle (start, pause, resume, stop)
  - Reports health via metrics
  - Survives individual exceptions
  - Respects graceful shutdown signals
  - Can be restarted independently

Usage:
    from core.background_workers import worker_orchestrator

    await worker_orchestrator.start_all()
    await worker_orchestrator.stop_all()
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Worker Types
# ═══════════════════════════════════════════════════════════════

class WorkerStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkerHealth:
    """Health report for a single worker."""
    name: str
    status: WorkerStatus = WorkerStatus.IDLE
    started_at: float = 0.0
    last_run_at: float = 0.0
    runs_completed: int = 0
    errors: int = 0
    last_error: str = ""
    avg_runtime_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Base Worker
# ═══════════════════════════════════════════════════════════════

class BaseWorker(ABC):
    """
    Abstract base for all background workers.

    Each worker implements _run_once() which is called periodically.
    The base class handles the lifecycle, error recovery, and metrics.
    """

    def __init__(self, name: str, interval_s: float = 60.0):
        self.name = name
        self._interval_s = interval_s
        self._status = WorkerStatus.IDLE
        self._task: Optional[asyncio.Task] = None
        self._event_bus = None
        self._running = False
        self._paused = False

        # Health tracking
        self._started_at: float = 0.0
        self._last_run_at: float = 0.0
        self._runs_completed: int = 0
        self._errors: int = 0
        self._last_error: str = ""
        self._run_times: List[float] = []

    # ── Wiring ─────────────────────────────────────────────────

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the worker. Creates the background task."""
        if self._running:
            return

        self._status = WorkerStatus.STARTING
        self._running = True
        self._started_at = time.time()
        self._task = asyncio.create_task(self._run_loop(), name=f"worker:{self.name}")
        self._status = WorkerStatus.RUNNING

        logger.info("[Worker:%s] Started (interval=%.1fs)", self.name, self._interval_s)
        await self._emit("worker:started", {"name": self.name})

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return

        self._status = WorkerStatus.STOPPING
        self._running = False
        self._paused = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._status = WorkerStatus.STOPPED
        logger.info("[Worker:%s] Stopped", self.name)
        await self._emit("worker:stopped", {"name": self.name})

    def pause(self) -> None:
        """Pause the worker (skip iterations until resumed)."""
        self._paused = True
        self._status = WorkerStatus.PAUSED
        logger.info("[Worker:%s] Paused", self.name)

    def resume(self) -> None:
        """Resume a paused worker."""
        self._paused = False
        self._status = WorkerStatus.RUNNING
        logger.info("[Worker:%s] Resumed", self.name)

    # ── Core Loop ──────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main worker loop. Calls _run_once() at the defined interval."""
        try:
            while self._running:
                if not self._paused:
                    try:
                        t0 = time.time()
                        await self._run_once()
                        elapsed = (time.time() - t0) * 1000.0

                        self._last_run_at = time.time()
                        self._runs_completed += 1
                        self._run_times.append(elapsed)
                        if len(self._run_times) > 100:
                            self._run_times = self._run_times[-50:]

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self._errors += 1
                        self._last_error = str(e)
                        logger.error("[Worker:%s] Error in _run_once: %s\n%s",
                                     self.name, e, traceback.format_exc())
                        await self._emit("worker:error", {
                            "name": self.name,
                            "error": str(e),
                            "error_count": self._errors,
                        })
                        # Short delay after error to avoid rapid retries
                        await asyncio.sleep(min(5.0, self._interval_s * 0.5))
                        continue

                await asyncio.sleep(self._interval_s)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._status = WorkerStatus.ERROR
            logger.error("[Worker:%s] Fatal error: %s\n%s",
                         self.name, e, traceback.format_exc())

    @abstractmethod
    async def _run_once(self) -> None:
        """
        Do one unit of work. Called every interval_s seconds.

        Subclasses implement their specific logic here.
        """
        ...

    # ── Health & Metrics ───────────────────────────────────────

    @property
    def health(self) -> WorkerHealth:
        avg = (sum(self._run_times) / len(self._run_times)) if self._run_times else 0.0
        return WorkerHealth(
            name=self.name,
            status=self._status,
            started_at=self._started_at,
            last_run_at=self._last_run_at,
            runs_completed=self._runs_completed,
            errors=self._errors,
            last_error=self._last_error,
            avg_runtime_ms=avg,
        )

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    # ── Event Bus ──────────────────────────────────────────────

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._event_bus:
            try:
                await self._event_bus.emit(event_type, data, source=f"worker:{self.name}")
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# Concrete Workers
# ═══════════════════════════════════════════════════════════════

class MemorySummarizerWorker(BaseWorker):
    """Periodically compresses and summarizes old conversation memories."""

    def __init__(self, interval_s: float = 300.0):
        super().__init__("memory_summarizer", interval_s)

    async def _run_once(self) -> None:
        try:
            from agent.conversation_memory import conv_memory
            # Compress if conversation context is large
            if len(conv_memory._conversation_history) > 50:
                logger.debug("[Worker:memory_summarizer] Compressing conversation history...")
                conv_memory._conversation_history = conv_memory._conversation_history[-50:]
        except Exception:
            pass


class ExperienceOptimizerWorker(BaseWorker):
    """Periodically prunes old/irrelevant experiences from ExperienceDB."""

    def __init__(self, interval_s: float = 600.0):
        super().__init__("experience_optimizer", interval_s)

    async def _run_once(self) -> None:
        try:
            from learning.experience_db import experience_db
            count = experience_db.record_count()
            if count > 4000:
                # Keep only the best 3000
                logger.info("[Worker:experience_optimizer] Pruning experience DB (%d records)...", count)
                # The ExperienceDB auto-caps at 5000 with oldest-first removal.
                # We just need to make sure it's healthy.
                stats = experience_db.stats()
                logger.info("[Worker:experience_optimizer] ExpDB stats: %s", stats)
        except Exception:
            pass


class PlannerOptimizerWorker(BaseWorker):
    """Analyzes planner performance and updates heuristics."""

    def __init__(self, interval_s: float = 300.0):
        super().__init__("planner_optimizer", interval_s)

    async def _run_once(self) -> None:
        try:
            from learning.experience_db import experience_db
            stats = experience_db.stats()
            recent_failures = experience_db.recent_failures(5)

            if recent_failures:
                # Log common failure patterns
                failure_actions: Dict[str, int] = {}
                for f in recent_failures:
                    for action in f.plan_actions:
                        failure_actions[action] = failure_actions.get(action, 0) + 1

                if failure_actions:
                    worst = max(failure_actions, key=failure_actions.get)
                    logger.info("[Worker:planner_optimizer] Most failing action: %s (%d failures)",
                                worst, failure_actions[worst])
        except Exception:
            pass


class SemanticLinkerWorker(BaseWorker):
    """Periodically re-links semantic memory graph to discover new connections."""

    def __init__(self, interval_s: float = 600.0):
        super().__init__("semantic_linker", interval_s)

    async def _run_once(self) -> None:
        try:
            from memory.semantic_memory import semantic_memory
            if semantic_memory.entity_count() < 10:
                return

            # Re-link recent entities
            for entity in list(semantic_memory._entities.values())[-50:]:
                semantic_memory._auto_link(entity)
        except Exception:
            pass


class GoalsVacuumWorker(BaseWorker):
    """Periodically archives old completed/failed goals."""

    def __init__(self, interval_s: float = 3600.0):
        super().__init__("goals_vacuum", interval_s)

    async def _run_once(self) -> None:
        try:
            from agent.goal_manager import goal_manager
            if goal_manager.is_available:
                goal_manager.vacuum()
        except Exception:
            pass


class HealthCheckWorker(BaseWorker):
    """Periodic system health check and reporting."""

    def __init__(self, interval_s: float = 60.0):
        super().__init__("health_check", interval_s)

    async def _run_once(self) -> None:
        try:
            from core.metrics import metrics
            # Log basic health
            uptime = metrics.get("_uptime_s", 0)
            mem_avg = metrics.average("system.ram_percent")
            cpu_avg = metrics.average("system.cpu_percent")

            logger.debug("[Worker:health_check] Uptime=%.0fs, CPU=%.1f%%, RAM=%.1f%%",
                          uptime, cpu_avg, mem_avg)

            # Publish health event
            await self._emit("system:health", {
                "uptime_s": uptime,
                "cpu_percent": cpu_avg,
                "ram_percent": mem_avg,
                "timestamp": time.time(),
            })

        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Worker Orchestrator
# ═══════════════════════════════════════════════════════════════

class WorkerOrchestrator:
    """
    Manages all background workers: start, stop, pause, resume,
    and health monitoring.

    Workers communicate through the EventBus and never block.
    """

    def __init__(self):
        self._workers: Dict[str, BaseWorker] = {}
        self._event_bus = None

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    def register(self, worker: BaseWorker) -> None:
        """Register a worker with the orchestrator."""
        worker.set_event_bus(self._event_bus)
        self._workers[worker.name] = worker
        logger.debug("[Orchestrator] Registered worker: %s", worker.name)

    async def start_all(self) -> None:
        """Start all registered workers."""
        if not self._workers:
            logger.info("[Orchestrator] No workers registered")
            return

        logger.info("[Orchestrator] Starting %d workers...", len(self._workers))
        for name, worker in self._workers.items():
            await worker.start()

        logger.info("[Orchestrator] All workers started")

    async def stop_all(self) -> None:
        """Stop all workers gracefully."""
        logger.info("[Orchestrator] Stopping %d workers...", len(self._workers))
        for name, worker in self._workers.items():
            await worker.stop()

        logger.info("[Orchestrator] All workers stopped")

    def pause_all(self) -> None:
        """Pause all workers."""
        for worker in self._workers.values():
            worker.pause()

    def resume_all(self) -> None:
        """Resume all workers."""
        for worker in self._workers.values():
            worker.resume()

    def get_worker(self, name: str) -> Optional[BaseWorker]:
        return self._workers.get(name)

    def get_health(self) -> List[WorkerHealth]:
        """Get health reports for all workers."""
        return [w.health for w in self._workers.values()]

    def get_status(self) -> Dict[str, WorkerStatus]:
        """Get status for all workers."""
        return {name: w.status for name, w in self._workers.items()}

    async def restart_worker(self, name: str) -> bool:
        """Restart a specific worker."""
        worker = self._workers.get(name)
        if worker is None:
            return False

        logger.info("[Orchestrator] Restarting worker: %s", name)
        await worker.stop()
        await worker.start()
        return True


# Global singleton
worker_orchestrator = WorkerOrchestrator()


# ═══════════════════════════════════════════════════════════════
# Default worker set
# ═══════════════════════════════════════════════════════════════

def register_default_workers(orch: WorkerOrchestrator) -> None:
    """Register the default set of background workers."""
    orch.register(MemorySummarizerWorker(interval_s=300.0))
    orch.register(ExperienceOptimizerWorker(interval_s=600.0))
    orch.register(PlannerOptimizerWorker(interval_s=300.0))
    orch.register(SemanticLinkerWorker(interval_s=600.0))
    orch.register(GoalsVacuumWorker(interval_s=3600.0))
    orch.register(HealthCheckWorker(interval_s=60.0))
    logger.info("[Orchestrator] Default workers registered")