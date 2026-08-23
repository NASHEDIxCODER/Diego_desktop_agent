"""
GoalManager — Persistent goal and task storage with DuckDB.

Goals survive reboots. The GoalManager stores:
  - Goal metadata (id, description, status, context, timestamps)
  - Task definitions with dependencies, retries, timeouts
  - Task execution results and errors
  - Goal-task relationships

The Brain delegates persistence to the GoalManager, which writes to
DuckDB tables goals, goal_tasks, and goal_events.

Usage:
    from agent.goal_manager import goal_manager

    manager = goal_manager
    manager.initialize()
    manager.save_goal(goal)
    goals = manager.list_goals()
    goal = manager.load_goal("goal_1234567890")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Reuse Brain types (avoiding circular import via local copy)
# ═══════════════════════════════════════════════════════════════

class GoalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class GoalRecord:
    """Serializable goal record for DuckDB storage."""
    id: str
    description: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    context_json: str = "{}"
    result_summary: str = ""
    tags: str = ""  # comma-separated


@dataclass
class TaskRecord:
    """Serializable task record for DuckDB storage."""
    id: str
    goal_id: str
    description: str
    status: str = "pending"
    depends_on: str = "[]"  # JSON array of task IDs
    retry_count: int = 0
    max_retries: int = 3
    timeout_s: float = 300.0
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: float = 0.0
    metadata_json: str = "{}"


@dataclass
class GoalEvent:
    """An event in a goal's lifecycle."""
    id: str
    goal_id: str
    event_type: str  # started, completed, failed, paused, cancelled, task_started, etc.
    data_json: str = "{}"
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════
# GoalManager
# ═══════════════════════════════════════════════════════════════

class GoalManager:
    """
    Persistent goal storage backed by DuckDB.

    Goals, their tasks, and lifecycle events are stored in DuckDB
    and survive Diego restarts. Long-running goals like "Finish GhostLine"
    persist across sessions.
    """

    def __init__(self):
        self._store = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize DuckDB tables for goals."""
        try:
            from memory.duckdb_store import store
            self._store = store
            self._store.initialize()

            conn = self._store._get_conn()
            if conn is None:
                logger.warning("[GoalManager] DuckDB unavailable — goals will not persist")
                self._initialized = True
                return False

            # Create goals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id VARCHAR PRIMARY KEY,
                    description VARCHAR NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    created_at DOUBLE NOT NULL,
                    updated_at DOUBLE NOT NULL,
                    context_json VARCHAR DEFAULT '{}',
                    result_summary VARCHAR DEFAULT '',
                    tags VARCHAR DEFAULT '',
                    archived BOOLEAN DEFAULT FALSE
                )
            """)

            # Create goal_tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_tasks (
                    id VARCHAR NOT NULL,
                    goal_id VARCHAR NOT NULL,
                    description VARCHAR NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    depends_on VARCHAR DEFAULT '[]',
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    timeout_s DOUBLE DEFAULT 300.0,
                    result VARCHAR,
                    error VARCHAR,
                    started_at DOUBLE,
                    completed_at DOUBLE,
                    latency_ms DOUBLE DEFAULT 0.0,
                    metadata_json VARCHAR DEFAULT '{}',
                    PRIMARY KEY (goal_id, id),
                    FOREIGN KEY (goal_id) REFERENCES goals(id)
                )
            """)

            # Create goal_events table for lifecycle tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_events (
                    id VARCHAR PRIMARY KEY,
                    goal_id VARCHAR NOT NULL,
                    event_type VARCHAR NOT NULL,
                    data_json VARCHAR DEFAULT '{}',
                    timestamp DOUBLE NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals(id)
                )
            """)

            # Create index for fast goal listing
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goals_status
                ON goals(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goals_updated
                ON goals(updated_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goal_tasks_goal
                ON goal_tasks(goal_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goal_events_goal
                ON goal_events(goal_id)
            """)

            self._initialized = True
            logger.info("[GoalManager] DuckDB goals tables initialized")
            return True

        except Exception as e:
            logger.warning("[GoalManager] Initialization failed: %s — goals will not persist", e)
            self._initialized = True
            return False

    @property
    def is_available(self) -> bool:
        return self._initialized and self._store is not None

    # ── Save / Load Goals ──────────────────────────────────────

    def save_goal(self, goal: Any) -> bool:
        """
        Save (upsert) a complete goal with all its tasks to DuckDB.

        Args:
            goal: A Goal dataclass from agent.brain (or compatible).

        Returns:
            True if the goal was saved successfully.
        """
        if not self.is_available:
            return False

        try:
            conn = self._store._get_conn()
            if conn is None:
                return False

            # Upsert goal
            context_json = json.dumps(getattr(goal, "context", {}) or {})
            conn.execute("""
                INSERT INTO goals (id, description, status, created_at, updated_at,
                                   context_json, result_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    description = EXCLUDED.description,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at,
                    context_json = EXCLUDED.context_json,
                    result_summary = EXCLUDED.result_summary
            """, [
                goal.id,
                goal.description,
                goal.status.value if hasattr(goal.status, 'value') else str(goal.status),
                getattr(goal, "created_at", time.time()),
                getattr(goal, "updated_at", time.time()),
                context_json,
                getattr(goal, "result_summary", ""),
            ])

            # Upsert tasks
            tasks = getattr(goal, "tasks", [])
            for task in tasks:
                depends_json = json.dumps(getattr(task, "depends_on", []) or [])
                metadata_json = json.dumps(getattr(task, "metadata", {}) or {})

                conn.execute("""
                    INSERT INTO goal_tasks (goal_id, id, description, status, depends_on,
                                            retry_count, max_retries, timeout_s,
                                            result, error, started_at, completed_at,
                                            latency_ms, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (goal_id, id) DO UPDATE SET
                        description = EXCLUDED.description,
                        status = EXCLUDED.status,
                        depends_on = EXCLUDED.depends_on,
                        retry_count = EXCLUDED.retry_count,
                        max_retries = EXCLUDED.max_retries,
                        timeout_s = EXCLUDED.timeout_s,
                        result = EXCLUDED.result,
                        error = EXCLUDED.error,
                        started_at = EXCLUDED.started_at,
                        completed_at = EXCLUDED.completed_at,
                        latency_ms = EXCLUDED.latency_ms,
                        metadata_json = EXCLUDED.metadata_json
                """, [
                    goal.id,
                    task.id,
                    task.description,
                    task.status.value if hasattr(task.status, 'value') else str(task.status),
                    depends_json,
                    getattr(task, "retry_count", 0),
                    getattr(task, "max_retries", 3),
                    getattr(task, "timeout_s", 300.0),
                    getattr(task, "result", None),
                    getattr(task, "error", None),
                    getattr(task, "started_at", None),
                    getattr(task, "completed_at", None),
                    getattr(task, "latency_ms", 0.0),
                    metadata_json,
                ])

            logger.debug("[GoalManager] Goal '%s' saved with %d tasks", goal.id, len(tasks))
            return True

        except Exception as e:
            logger.error("[GoalManager] Failed to save goal '%s': %s",
                         getattr(goal, "id", "unknown"), e)
            return False

    def load_goal(self, goal_id: str) -> Optional[GoalRecord]:
        """
        Load a goal from DuckDB by ID.

        Returns a GoalRecord or None if not found.
        """
        if not self.is_available:
            return None

        try:
            conn = self._store._get_conn()
            if conn is None:
                return None

            row = conn.execute(
                "SELECT id, description, status, created_at, updated_at, "
                "context_json, result_summary, tags FROM goals WHERE id = ?",
                [goal_id]
            ).fetchone()

            if row is None:
                return None

            return GoalRecord(
                id=row[0],
                description=row[1],
                status=row[2],
                created_at=row[3],
                updated_at=row[4],
                context_json=row[5],
                result_summary=row[6],
                tags=row[7] or "",
            )

        except Exception as e:
            logger.error("[GoalManager] Failed to load goal '%s': %s", goal_id, e)
            return None

    def load_tasks(self, goal_id: str) -> List[TaskRecord]:
        """Load all tasks for a goal from DuckDB."""
        if not self.is_available:
            return []

        try:
            conn = self._store._get_conn()
            if conn is None:
                return []

            rows = conn.execute("""
                SELECT id, goal_id, description, status, depends_on,
                       retry_count, max_retries, timeout_s,
                       result, error, started_at, completed_at,
                       latency_ms, metadata_json
                FROM goal_tasks
                WHERE goal_id = ?
                ORDER BY id
            """, [goal_id]).fetchall()

            tasks = []
            for row in rows:
                tasks.append(TaskRecord(
                    id=row[0],
                    goal_id=row[1],
                    description=row[2],
                    status=row[3],
                    depends_on=row[4],
                    retry_count=row[5],
                    max_retries=row[6],
                    timeout_s=row[7],
                    result=row[8],
                    error=row[9],
                    started_at=row[10],
                    completed_at=row[11],
                    latency_ms=row[12] or 0.0,
                    metadata_json=row[13],
                ))

            return tasks

        except Exception as e:
            logger.error("[GoalManager] Failed to load tasks for goal '%s': %s", goal_id, e)
            return []

    # ── Goal Queries ───────────────────────────────────────────

    def list_goals(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> List[GoalRecord]:
        """
        List goals, optionally filtered by status.

        Args:
            status: Filter by status (pending, running, completed, failed, cancelled)
            limit: Max goals to return
            offset: Pagination offset
            include_archived: Whether to include archived goals
        """
        if not self.is_available:
            return []

        try:
            conn = self._store._get_conn()
            if conn is None:
                return []

            query = """
                SELECT id, description, status, created_at, updated_at,
                       context_json, result_summary, tags
                FROM goals
                WHERE 1=1
            """
            params: list = []

            if status:
                query += " AND status = ?"
                params.append(status)

            if not include_archived:
                query += " AND (archived IS NULL OR archived = FALSE)"

            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()

            goals = []
            for row in rows:
                goals.append(GoalRecord(
                    id=row[0],
                    description=row[1],
                    status=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                    context_json=row[5],
                    result_summary=row[6],
                    tags=row[7] or "",
                ))

            return goals

        except Exception as e:
            logger.error("[GoalManager] Failed to list goals: %s", e)
            return []

    def list_active_goals(self) -> List[GoalRecord]:
        """Return goals that are still pending or running."""
        pending = self.list_goals(status="pending", limit=100)
        running = self.list_goals(status="running", limit=100)
        paused = self.list_goals(status="paused", limit=100)
        return pending + running + paused

    def list_completed_goals(self, limit: int = 50) -> List[GoalRecord]:
        """Return recently completed goals."""
        return self.list_goals(status="completed", limit=limit)

    def list_failed_goals(self, limit: int = 50) -> List[GoalRecord]:
        """Return recently failed goals."""
        return self.list_goals(status="failed", limit=limit)

    # ── Goal Search ────────────────────────────────────────────

    def search_goals(self, query: str, limit: int = 20) -> List[GoalRecord]:
        """Search goals by description (case-insensitive substring match)."""
        if not self.is_available:
            return []

        try:
            conn = self._store._get_conn()
            if conn is None:
                return []

            rows = conn.execute("""
                SELECT id, description, status, created_at, updated_at,
                       context_json, result_summary, tags
                FROM goals
                WHERE LOWER(description) LIKE ?
                   OR LOWER(result_summary) LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, [f"%{query.lower()}%", f"%{query.lower()}%", limit]).fetchall()

            goals = []
            for row in rows:
                goals.append(GoalRecord(
                    id=row[0],
                    description=row[1],
                    status=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                    context_json=row[5],
                    result_summary=row[6],
                    tags=row[7] or "",
                ))
            return goals

        except Exception as e:
            logger.error("[GoalManager] Search failed: %s", e)
            return []

    # ── Goal Events ────────────────────────────────────────────

    def record_event(self, goal_id: str, event_type: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Record a lifecycle event for a goal."""
        if not self.is_available:
            return False

        try:
            conn = self._store._get_conn()
            if conn is None:
                return False

            event_id = f"evt_{goal_id}_{event_type}_{int(time.time() * 1000)}"
            data_json = json.dumps(data or {})

            conn.execute("""
                INSERT INTO goal_events (id, goal_id, event_type, data_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, [event_id, goal_id, event_type, data_json, time.time()])

            return True

        except Exception as e:
            logger.debug("[GoalManager] Event recording failed: %s", e)
            return False

    def get_events(self, goal_id: str, limit: int = 50) -> List[GoalEvent]:
        """Get lifecycle events for a goal."""
        if not self.is_available:
            return []

        try:
            conn = self._store._get_conn()
            if conn is None:
                return []

            rows = conn.execute("""
                SELECT id, goal_id, event_type, data_json, timestamp
                FROM goal_events
                WHERE goal_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, [goal_id, limit]).fetchall()

            events = []
            for row in rows:
                events.append(GoalEvent(
                    id=row[0],
                    goal_id=row[1],
                    event_type=row[2],
                    data_json=row[3],
                    timestamp=row[4],
                ))
            return events

        except Exception as e:
            logger.error("[GoalManager] Event query failed: %s", e)
            return []

    # ── Goal Management ────────────────────────────────────────

    def update_status(self, goal_id: str, status: str) -> bool:
        """Update a goal's status."""
        if not self.is_available:
            return False

        try:
            conn = self._store._get_conn()
            if conn is None:
                return False

            conn.execute(
                "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
                [status, time.time(), goal_id]
            )
            return True

        except Exception as e:
            logger.error("[GoalManager] Status update failed: %s", e)
            return False

    def archive_goal(self, goal_id: str) -> bool:
        """Archive a completed/failed goal."""
        if not self.is_available:
            return False

        try:
            conn = self._store._get_conn()
            if conn is None:
                return False

            conn.execute(
                "UPDATE goals SET archived = TRUE, updated_at = ? WHERE id = ?",
                [time.time(), goal_id]
            )
            logger.info("[GoalManager] Goal '%s' archived", goal_id)
            return True

        except Exception as e:
            logger.error("[GoalManager] Archive failed: %s", e)
            return False

    def delete_goal(self, goal_id: str) -> bool:
        """Permanently delete a goal and its tasks/events."""
        if not self.is_available:
            return False

        try:
            conn = self._store._get_conn()
            if conn is None:
                return False

            # Delete in order: events → tasks → goal
            conn.execute("DELETE FROM goal_events WHERE goal_id = ?", [goal_id])
            conn.execute("DELETE FROM goal_tasks WHERE goal_id = ?", [goal_id])
            conn.execute("DELETE FROM goals WHERE id = ?", [goal_id])

            logger.info("[GoalManager] Goal '%s' deleted", goal_id)
            return True

        except Exception as e:
            logger.error("[GoalManager] Delete failed: %s", e)
            return False

    # ── Statistics ─────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return aggregate goal statistics."""
        if not self.is_available:
            return {"total": 0, "by_status": {}}

        try:
            conn = self._store._get_conn()
            if conn is None:
                return {"total": 0, "by_status": {}}

            total = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]

            by_status_rows = conn.execute(
                "SELECT status, COUNT(*) FROM goals GROUP BY status"
            ).fetchall()

            by_status = {row[0]: row[1] for row in by_status_rows}

            active = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status IN ('pending', 'running', 'paused')"
            ).fetchone()[0]

            return {
                "total": total,
                "active": active,
                "by_status": by_status,
            }

        except Exception as e:
            logger.error("[GoalManager] Stats failed: %s", e)
            return {"total": 0, "by_status": {}}

    # ── Cleanup ────────────────────────────────────────────────

    def vacuum(self) -> None:
        """Clean up old archived goals (older than 30 days)."""
        if not self.is_available:
            return

        try:
            conn = self._store._get_conn()
            if conn is None:
                return

            cutoff = time.time() - (30 * 86400)
            old_goals = conn.execute(
                "SELECT id FROM goals WHERE archived = TRUE AND updated_at < ?",
                [cutoff]
            ).fetchall()

            for (goal_id,) in old_goals:
                self.delete_goal(goal_id)

            logger.info("[GoalManager] Vacuumed %d old goals", len(old_goals))

        except Exception as e:
            logger.warning("[GoalManager] Vacuum failed: %s", e)

    def close(self) -> None:
        self._initialized = False
        logger.info("[GoalManager] Shut down")


# Global singleton
goal_manager = GoalManager()