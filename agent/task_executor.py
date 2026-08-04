"""
TaskExecutor — DAG-based task execution engine with full lifecycle management.

Supports:
  - Sequential tasks (one after another)
  - Parallel tasks (concurrent execution, respecting dependencies)
  - Conditional tasks (only run if a condition is met)
  - Retries with exponential backoff
  - Rollback on failure
  - Pre/post verification hooks
  - Per-task timeouts
  - Cancellation (graceful)
  - Dependency graph resolution

Every task returns one of: SUCCESS, FAILED, RETRY, SKIPPED

The TaskExecutor feeds into the Brain, which feeds into the Planner.
"Simple tasks are PLANIFIED. Complex goal graphs are EXECUTED."

Usage:
    from agent.task_executor import TaskExecutor, TaskNode, ExecutionResult

    executor = TaskExecutor(planner=agent_planner)
    graph = executor.build_graph(tasks_data)
    results = await executor.execute(graph)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Core Types
# ═══════════════════════════════════════════════════════════════

class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRY = "RETRY"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


@dataclass
class ExecutionResult:
    """Result of executing a single task node."""
    node_id: str
    status: ExecutionStatus
    output: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    retry_attempt: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class TaskNode:
    """A node in the task execution DAG."""
    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    parallel_group: Optional[str] = None  # Tasks in same group run in parallel
    condition: Optional[Callable[[], Awaitable[bool]]] = None  # Conditional gate
    timeout_s: float = 300.0
    max_retries: int = 3
    retry_delay_s: float = 1.0
    rollback: Optional[Callable[[], Awaitable[bool]]] = None  # Rollback handler
    verify: Optional[Callable[[], Awaitable[bool]]] = None     # Post-exec verify
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    status: ExecutionStatus = ExecutionStatus.SKIPPED
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0


class TaskGraph:
    """A directed acyclic graph of tasks with dependency resolution."""

    def __init__(self, nodes: Optional[List[TaskNode]] = None):
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: Dict[str, List[str]] = {}  # node_id → list of dependent node_ids
        self._in_degree: Dict[str, int] = {}

        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []
        if node.id not in self._in_degree:
            self._in_degree[node.id] = 0

        for dep_id in node.depends_on:
            if dep_id not in self.nodes:
                self._in_degree[dep_id] = 0
                self.edges[dep_id] = []
            self.edges[dep_id].append(node.id)
            self._in_degree[node.id] = self._in_degree.get(node.id, 0) + 1

    def topological_order(self) -> List[List[TaskNode]]:
        """
        Return tasks grouped by topological level.
        Each level's tasks can run in parallel.
        """
        # Kahn's algorithm
        in_degree = dict(self._in_degree)
        queue: List[str] = [nid for nid, deg in in_degree.items() if deg == 0]
        levels: List[List[TaskNode]] = []

        while queue:
            level_nodes = [self.nodes[nid] for nid in queue if nid in self.nodes]
            if level_nodes:
                levels.append(level_nodes)
            next_queue: List[str] = []
            for nid in queue:
                for dependent in self.edges.get(nid, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            queue = next_queue

        # Check for cycles
        if len(levels) == 0 and len(self.nodes) > 0:
            # Fallback: return all nodes as one level
            return [[node for node in self.nodes.values()]]

        return levels

    def find_parallel_groups(self) -> Dict[str, List[TaskNode]]:
        """Group nodes by their parallel_group attribute."""
        groups: Dict[str, List[TaskNode]] = {}
        for node in self.nodes.values():
            group = node.parallel_group or "__default__"
            if group not in groups:
                groups[group] = []
            groups[group].append(node)
        return groups


# ═══════════════════════════════════════════════════════════════
# Task Executor
# ═══════════════════════════════════════════════════════════════

class TaskExecutor:
    """
    DAG-based task execution engine.

    Executes tasks respecting their dependency graph, with support
    for parallelism, retries, rollback, verification, timeouts,
    and cancellation.

    The executor itself does NOT perform desktop actions — it delegates
    each task's actual work to the Planner via an execution callback.
    """

    def __init__(
        self,
        max_parallel: int = 4,
        default_timeout_s: float = 300.0,
    ):
        self._max_parallel = max_parallel
        self._default_timeout_s = default_timeout_s
        self._planner = None
        self._execution_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[Tuple[bool, str]]]] = None
        self._cancelled: Set[str] = set()
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._event_bus = None

    # ── Wiring ─────────────────────────────────────────────────

    def set_execution_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], Awaitable[Tuple[bool, str]]],
    ) -> None:
        """
        Wire the actual execution function.

        Args:
            callback: async(task_description, context) -> (success_bool, result_string)
        """
        self._execution_callback = callback

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    @property
    def is_available(self) -> bool:
        return self._execution_callback is not None

    # ── Graph Construction ─────────────────────────────────────

    def build_graph(self, tasks_data: List[Dict[str, Any]]) -> TaskGraph:
        """
        Build a TaskGraph from raw task definitions.

        Args:
            tasks_data: List of task dicts with keys:
                id, description, depends_on, parallel_group, timeout_s, max_retries, condition_fn, rollback_fn, verify_fn

        Returns:
            A TaskGraph ready for execution.
        """
        nodes = []
        for td in tasks_data:
            node = TaskNode(
                id=str(td.get("id", f"task_{len(nodes)}")),
                description=str(td.get("description", "")),
                depends_on=[str(d) for d in td.get("depends_on", [])],
                parallel_group=td.get("parallel_group"),
                condition=td.get("condition_fn"),
                timeout_s=float(td.get("timeout_s", self._default_timeout_s)),
                max_retries=int(td.get("max_retries", 3)),
                retry_delay_s=float(td.get("retry_delay_s", 1.0)),
                rollback=td.get("rollback_fn"),
                verify=td.get("verify_fn"),
                metadata=td.get("metadata", {}),
            )
            nodes.append(node)

        return TaskGraph(nodes)

    # ── Graph Execution ────────────────────────────────────────

    async def execute(
        self,
        graph: TaskGraph,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ExecutionResult]:
        """
        Execute a complete task graph.

        Args:
            graph: The TaskGraph to execute.
            context: Optional context dict for the execution callback.

        Returns:
            Dict mapping node_id → ExecutionResult.
        """
        if not self.is_available:
            return {nid: ExecutionResult(
                node_id=nid, status=ExecutionStatus.FAILED,
                error="No execution callback wired"
            ) for nid in graph.nodes}

        self._cancelled.clear()
        results: Dict[str, ExecutionResult] = {}
        ctx = context or {}

        levels = graph.topological_order()
        logger.info("[TaskExecutor] Executing DAG with %d nodes across %d levels",
                     len(graph.nodes), len(levels))

        for level_idx, level_nodes in enumerate(levels):
            if self._cancelled:
                logger.info("[TaskExecutor] Execution cancelled at level %d", level_idx)
                break

            logger.info("[TaskExecutor] Level %d: %d tasks", level_idx, len(level_nodes))

            # Parallel group handling within this level
            parallel_groups = self._group_by_parallel(level_nodes)

            tasks = []
            for group_name, group_nodes in parallel_groups.items():
                if group_name == "__sequential__":
                    # Execute one at a time
                    for node in group_nodes:
                        result = await self._execute_node(node, ctx, results)
                        results[node.id] = result
                        await self._emit_event("task:result", {
                            "node_id": node.id,
                            "status": result.status.value,
                            "output": result.output,
                        })
                        # If a critical node fails, cancel downstream
                        if result.status == ExecutionStatus.FAILED and node.metadata.get("critical", True):
                            self._cancel_downstream(graph, node.id)
                            break
                else:
                    # Parallel group — execute concurrently
                    parallel_tasks = [
                        self._execute_node(node, ctx, results)
                        for node in group_nodes
                    ]
                    group_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                    for node, result in zip(group_nodes, group_results):
                        if isinstance(result, Exception):
                            results[node.id] = ExecutionResult(
                                node_id=node.id,
                                status=ExecutionStatus.FAILED,
                                error=str(result),
                            )
                        else:
                            results[node.id] = result
                        await self._emit_event("task:result", {
                            "node_id": node.id,
                            "status": results[node.id].status.value,
                            "output": results[node.id].output,
                        })

            # Abort if all tasks in this level failed and they were critical
            if all(
                results.get(n.id, ExecutionResult(node_id=n.id, status=ExecutionStatus.FAILED)).status
                in (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.TIMEOUT)
                for n in level_nodes
                if getattr(n, "metadata", {}).get("critical", True)
            ):
                logger.warning("[TaskExecutor] All critical tasks in level %d failed — aborting", level_idx)
                break

        return results

    @staticmethod
    def _group_by_parallel(nodes: List[TaskNode]) -> Dict[str, List[TaskNode]]:
        """Group nodes by parallel_group for concurrent execution."""
        groups: Dict[str, List[TaskNode]] = {}
        for node in nodes:
            group = node.parallel_group or "__sequential__"
            if group not in groups:
                groups[group] = []
            groups[group].append(node)
        return groups

    async def _execute_node(
        self,
        node: TaskNode,
        context: Dict[str, Any],
        prior_results: Dict[str, ExecutionResult],
    ) -> ExecutionResult:
        """Execute a single task node with full lifecycle."""
        # Check cancellation
        if node.id in self._cancelled:
            return ExecutionResult(
                node_id=node.id,
                status=ExecutionStatus.CANCELLED,
                error="Cancelled",
            )

        # Check condition
        if node.condition:
            try:
                should_run = await node.condition()
                if not should_run:
                    logger.info("[TaskExecutor] Node '%s' skipped by condition", node.id)
                    return ExecutionResult(
                        node_id=node.id,
                        status=ExecutionStatus.SKIPPED,
                        output="Skipped by condition",
                    )
            except Exception as e:
                logger.warning("[TaskExecutor] Condition check failed for '%s': %s", node.id, e)

        # Execute with retries
        last_result: Optional[ExecutionResult] = None
        for attempt in range(node.max_retries + 1):
            if node.id in self._cancelled:
                return ExecutionResult(
                    node_id=node.id,
                    status=ExecutionStatus.CANCELLED,
                    error="Cancelled during retry",
                )

            node.retry_count = attempt
            result = await self._execute_with_timeout(node, context)
            node.started_at = result.started_at
            node.completed_at = result.completed_at
            node.status = result.status
            node.result = result.output
            node.error = result.error
            last_result = result

            if result.status == ExecutionStatus.SUCCESS:
                # Post-verification
                if node.verify:
                    try:
                        verified = await node.verify()
                        if not verified:
                            logger.warning("[TaskExecutor] Verification failed for '%s'", node.id)
                            result = ExecutionResult(
                                node_id=node.id,
                                status=ExecutionStatus.FAILED,
                                error="Post-execution verification failed",
                                latency_ms=result.latency_ms,
                                retry_attempt=attempt,
                            )
                            node.status = ExecutionStatus.FAILED
                            node.error = result.error
                            last_result = result
                            # Continue to retry
                        else:
                            return result
                    except Exception as e:
                        logger.warning("[TaskExecutor] Verify error for '%s': %s", node.id, e)
                        result = ExecutionResult(
                            node_id=node.id,
                            status=ExecutionStatus.FAILED,
                            error=f"Verify error: {e}",
                            latency_ms=result.latency_ms,
                            retry_attempt=attempt,
                        )
                        last_result = result
                else:
                    return result

            elif result.status == ExecutionStatus.SKIPPED:
                return result

            # Prepare retry
            if attempt < node.max_retries:
                delay = node.retry_delay_s * (2 ** attempt)  # Exponential backoff
                logger.info("[TaskExecutor] Retrying '%s' in %.1fs (attempt %d/%d)",
                            node.id, delay, attempt + 1, node.max_retries)
                await self._emit_event("task:retry", {
                    "node_id": node.id,
                    "attempt": attempt + 1,
                    "max": node.max_retries,
                    "error": result.error,
                })
                await asyncio.sleep(delay)

        # All retries exhausted — rollback if handler exists
        if node.rollback and last_result and last_result.status == ExecutionStatus.FAILED:
            try:
                logger.info("[TaskExecutor] Rolling back '%s'", node.id)
                rolled_back = await node.rollback()
                if rolled_back:
                    logger.info("[TaskExecutor] Rollback of '%s' succeeded", node.id)
            except Exception as e:
                logger.error("[TaskExecutor] Rollback of '%s' failed: %s", node.id, e)

        return last_result or ExecutionResult(
            node_id=node.id,
            status=ExecutionStatus.FAILED,
            error="No result after all retries",
        )

    async def _execute_with_timeout(
        self, node: TaskNode, context: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute with a hard timeout."""
        async with self._semaphore:
            started = time.time()
            try:
                output = await asyncio.wait_for(
                    self._execution_callback(node.description, context),
                    timeout=node.timeout_s,
                )
                elapsed = (time.time() - started) * 1000.0

                success, message = output if isinstance(output, tuple) else (True, str(output))

                return ExecutionResult(
                    node_id=node.id,
                    status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
                    output=message if success else None,
                    error=None if success else message,
                    latency_ms=elapsed,
                    retry_attempt=node.retry_count,
                    started_at=started,
                    completed_at=time.time(),
                )

            except asyncio.TimeoutError:
                elapsed = (time.time() - started) * 1000.0
                logger.warning("[TaskExecutor] Node '%s' timed out after %.0fs",
                               node.id, node.timeout_s)
                return ExecutionResult(
                    node_id=node.id,
                    status=ExecutionStatus.TIMEOUT,
                    error=f"Timed out after {node.timeout_s}s",
                    latency_ms=elapsed,
                    retry_attempt=node.retry_count,
                    started_at=started,
                    completed_at=time.time(),
                )

            except asyncio.CancelledError:
                elapsed = (time.time() - started) * 1000.0
                return ExecutionResult(
                    node_id=node.id,
                    status=ExecutionStatus.CANCELLED,
                    error="Cancelled",
                    latency_ms=elapsed,
                    retry_attempt=node.retry_count,
                    started_at=started,
                    completed_at=time.time(),
                )

            except Exception as e:
                elapsed = (time.time() - started) * 1000.0
                logger.error("[TaskExecutor] Node '%s' error: %s", node.id, e)
                return ExecutionResult(
                    node_id=node.id,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    latency_ms=elapsed,
                    retry_attempt=node.retry_count,
                    started_at=started,
                    completed_at=time.time(),
                )

    def _cancel_downstream(self, graph: TaskGraph, failed_node_id: str) -> None:
        """Mark all downstream nodes as cancelled."""
        self._cancelled.add(failed_node_id)
        visited: Set[str] = {failed_node_id}
        queue = [failed_node_id]

        while queue:
            current = queue.pop(0)
            for dependent in graph.edges.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    self._cancelled.add(dependent)
                    queue.append(dependent)

        logger.info("[TaskExecutor] Cancelled %d downstream nodes after '%s' failure",
                     len(visited) - 1, failed_node_id)

    async def cancel_all(self) -> None:
        """Cancel all in-flight and pending tasks."""
        for node_id in list(self._cancelled):
            pass  # already cancelled
        self._cancelled.add("__global_cancel__")
        logger.info("[TaskExecutor] Global cancellation requested")

    # ── Event Emission ────────────────────────────────────────

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._event_bus:
            try:
                await self._event_bus.emit(event_type, data, source="task_executor")
            except Exception:
                pass

    # ── Result Statistics ─────────────────────────────────────

    @staticmethod
    def summarize(results: Dict[str, ExecutionResult]) -> Dict[str, Any]:
        """Summarize execution results."""
        total = len(results)
        if total == 0:
            return {"total": 0, "success": 0, "failed": 0}

        success = sum(1 for r in results.values() if r.status == ExecutionStatus.SUCCESS)
        failed = sum(1 for r in results.values() if r.status == ExecutionStatus.FAILED)
        skipped = sum(1 for r in results.values() if r.status == ExecutionStatus.SKIPPED)
        cancelled = sum(1 for r in results.values() if r.status == ExecutionStatus.CANCELLED)
        timeout = sum(1 for r in results.values() if r.status == ExecutionStatus.TIMEOUT)
        latencies = [r.latency_ms for r in results.values() if r.latency_ms > 0]

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "cancelled": cancelled,
            "timeout": timeout,
            "success_rate": round(success / total, 3) if total > 0 else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "total_latency_ms": round(sum(latencies), 1),
        }

    def close(self) -> None:
        logger.info("[TaskExecutor] Shut down")


# Global singleton
task_executor = TaskExecutor()