"""
Service Layer — Diego's layered architecture backbone.

Every subsystem in Diego is a Service. Services:

  * Start/stop through a uniform lifecycle (async start/stop)
  * Report health (health() -> dict, is_ready property)
  * Communicate ONLY through the event bus — never direct imports
  * Are registered in the global ServiceRegistry
  * Can be restarted independently without killing the runtime

Layered architecture:
    User
      ↓
    Conversation Runtime  (orchestrator)
      ↓
    Planner              (task decomposition)
      ↓
    Reasoning            (LLM)
      ↓
    Memory               (duckdb + facts + embeddings)
      ↓
    Tools                (browser, terminal, desktop, ...)
      ↓
    Operating System

No module directly controls another unrelated subsystem. Everything
communicates through events and services.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.event_bus import bus, Event

logger = logging.getLogger(__name__)


class ServiceState(str, Enum):
    """Lifecycle states for a service."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"


@dataclass
class ServiceHealth:
    """Health report for a service."""
    state: ServiceState = ServiceState.STOPPED
    message: str = ""
    last_heartbeat: float = 0.0
    uptime: float = 0.0
    restart_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "message": self.message,
            "last_heartbeat": self.last_heartbeat,
            "uptime": self.uptime,
            "restart_count": self.restart_count,
            "details": self.details,
        }


class BaseService:
    """
    Base class for all Diego services.

    Subclasses implement `_start()` and `_stop()`. The public `start()`
    / `stop()` wrappers manage lifecycle safety (idempotency, state
    transitions, event emissions).

    Services emit lifecycle events on the bus:
        service.started      {name, details}
        service.stopped      {name}
        service.failed       {name, error}
        service.degraded     {name, message}
        service.heartbeat    {name, state}
    """

    name: str = "base"
    dependencies: List[str] = []  # service names that must RUN first

    def __init__(self):
        self._state = ServiceState.STOPPED
        self._start_time: float = 0.0
        self._last_heartbeat: float = 0.0
        self._restart_count: int = 0
        self._health_message: str = ""
        self._health_details: Dict[str, Any] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: float = 10.0

    # ── Public lifecycle ────────────────────────────────────

    async def start(self) -> bool:
        """Start the service (idempotent). Returns True when running."""
        if self._state == ServiceState.RUNNING:
            return True
        if self._state == ServiceState.STARTING:
            return False

        self._state = ServiceState.STARTING
        self._start_time = time.time()
        try:
            logger.info("[SERVICE] %s starting", self.name)
            ok = await self._start()
            if ok is False:
                self._state = ServiceState.FAILED
                await self._emit("service.failed", error="start returned False")
                return False
            self._state = ServiceState.RUNNING
            self._last_heartbeat = time.time()
            logger.info("[SERVICE] %s running", self.name)
            await self._emit("service.started", uptime=self._start_time)
            self._start_heartbeat()
            return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._state = ServiceState.FAILED
            logger.exception("[SERVICE] %s failed to start: %s", self.name, e)
            await self._emit("service.failed", error=str(e))
            return False

    async def stop(self) -> None:
        """Stop the service (idempotent)."""
        if self._state in (ServiceState.STOPPED, ServiceState.STOPPING):
            return
        self._state = ServiceState.STOPPING
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None
        try:
            await self._stop()
        except Exception as e:
            logger.warning("[SERVICE] %s stop error: %s", self.name, e)
        finally:
            self._state = ServiceState.STOPPED
            await self._emit("service.stopped")

    async def restart(self) -> bool:
        """Restart the service after a failure (self-healing)."""
        self._restart_count += 1
        logger.info("[SERVICE] %s restarting (attempt #%d)",
                    self.name, self._restart_count)
        await self.stop()
        return await self.start()

    # ── Health / introspection ──────────────────────────────

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == ServiceState.RUNNING

    @property
    def health(self) -> ServiceHealth:
        now = time.time()
        uptime = now - self._start_time if self._start_time else 0.0
        hb = self._last_heartbeat
        # Stale heartbeat (>3× interval) → degraded
        if (self._state == ServiceState.RUNNING and hb
                and now - hb > self._heartbeat_interval * 3):
            return ServiceHealth(
                state=ServiceState.DEGRADED,
                message="stale heartbeat",
                last_heartbeat=hb,
                uptime=uptime,
                restart_count=self._restart_count,
                details=self._health_details,
            )
        return ServiceHealth(
            state=self._state,
            message=self._health_message,
            last_heartbeat=hb,
            uptime=uptime,
            restart_count=self._restart_count,
            details=self._health_details,
        )

    def set_health(self, message: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        """Update the service's health message/details."""
        self._health_message = message
        if details is not None:
            self._health_details.update(details)
        self._last_heartbeat = time.time()

    # ── Subclass contract ───────────────────────────────────

    async def _start(self) -> bool:
        """Implemented by subclasses. Return True when ready."""
        return True

    async def _stop(self) -> None:
        """Implemented by subclasses."""

    # ── Internals ───────────────────────────────────────────

    def _start_heartbeat(self) -> None:
        async def _hb():
            try:
                while True:
                    await asyncio.sleep(self._heartbeat_interval)
                    self._last_heartbeat = time.time()
                    await self._emit(
                        "service.heartbeat", state=self._state.value)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        try:
            loop = asyncio.get_running_loop()
            self._heartbeat_task = loop.create_task(_hb())
        except RuntimeError:
            # No running loop — try the default loop if it exists.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._heartbeat_task = loop.create_task(_hb())
            except Exception:
                pass  # no loop available — heartbeat skipped (non-fatal)

    async def _emit(self, event_type: str, **extra: Any) -> None:
        try:
            await bus.emit(
                event_type,
                data={"name": self.name, **extra},
                source=f"service:{self.name}",
            )
        except Exception as e:
            logger.debug("[SERVICE] %s event emit failed: %s", self.name, e)


# ═══════════════════════════════════════════════════════════
# ServiceRegistry — global registry + startup orchestration
# ═══════════════════════════════════════════════════════════

class ServiceRegistry:
    """
    Global registry of services with dependency-aware startup.

    start_all() starts services in dependency order (topological sort).
    A failing service does NOT stop the others — it is marked FAILED
    and the runtime self-healing can restart it later.
    """

    def __init__(self):
        self._services: Dict[str, BaseService] = {}
        self._boot_order: List[str] = []

    def register(self, service: BaseService) -> None:
        """Register a service. Duplicate names replace the old one."""
        if service.name in self._services:
            logger.warning("[SERVICE] %s already registered — replacing",
                           service.name)
        self._services[service.name] = service
        logger.info("[SERVICE] Registered: %s", service.name)

    def get(self, name: str) -> Optional[BaseService]:
        return self._services.get(name)

    def all(self) -> Dict[str, BaseService]:
        return dict(self._services)

    def unregister(self, name: str) -> None:
        self._services.pop(name, None)

    def _resolve_order(self) -> List[BaseService]:
        """Topological sort by dependency declarations."""
        services = list(self._services.values())
        resolved: List[BaseService] = []
        visited: set = set()
        visiting: set = set()

        def visit(s: BaseService) -> None:
            if s.name in visited:
                return
            if s.name in visiting:
                logger.warning("[SERVICE] dependency cycle at %s — "
                               "breaking at current order", s.name)
                return
            visiting.add(s.name)
            for dep in s.dependencies:
                dep_svc = self._services.get(dep)
                if dep_svc is not None:
                    visit(dep_svc)
                else:
                    logger.warning("[SERVICE] %s depends on missing "
                                   "service '%s'", s.name, dep)
            visiting.discard(s.name)
            visited.add(s.name)
            resolved.append(s)

        for s in services:
            visit(s)
        return resolved

    async def start_all(self) -> Dict[str, bool]:
        """Start all services in dependency order.

        Returns {name: success} for each service. Services that fail
        do not block others (failures are recorded and recoverable).
        """
        order = self._resolve_order()
        self._boot_order = [s.name for s in order]
        results: Dict[str, bool] = {}
        for svc in order:
            ok = await svc.start()
            results[svc.name] = ok
        return results

    async def stop_all(self) -> None:
        """Stop all services in reverse boot order."""
        for name in reversed(self._boot_order):
            svc = self._services.get(name)
            if svc is not None:
                try:
                    await svc.stop()
                except Exception as e:
                    logger.warning("[SERVICE] %s stop error: %s", name, e)

    def healthy_services(self) -> List[str]:
        return [n for n, s in self._services.items() if s.is_ready]

    def failed_services(self) -> List[str]:
        return [n for n, s in self._services.items()
                if s.state == ServiceState.FAILED]

    def health_report(self) -> Dict[str, Dict[str, Any]]:
        return {n: s.health.to_dict() for n, s in self._services.items()}


# Global singleton registry
registry = ServiceRegistry()