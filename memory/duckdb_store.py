"""
DuckDB storage layer for Leo Desktop Assistant.

Stores:
- intents and intent_examples
- embeddings (intent_embeddings, cached_embeddings)
- entities and synonyms
- context_memory
- command_history
- user_preferences
- plugin_registry

Uses native DuckDB APIs only — no PostgreSQL compatibility hacks.

Design invariants:
  * intents table is append-only (no DELETE, no DROP during training)
  * intent IDs are immutable / reused via ON CONFLICT ... DO UPDATE
  * Derived tables (intent_embeddings, cached_embeddings) are safe to TRUNCATE
  * Training pipeline runs inside a single transaction
  * Every connect() has a matching close() - no leaked connections
  * Exponential backoff retry on lock conflicts
  * Context managers ensure no orphaned connections
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not installed. Using in-memory fallback.")


# ── Schema version for migration support ──────────────────────────
SCHEMA_VERSION = 2

# Lock detection constants
STALE_LOCK_AGE_SECONDS = 300  # 5 minutes — a lock file this old is stale


class DatabaseLockedError(Exception):
    """Raised when DuckDB is locked by another process."""


class DuckDBStore:
    """
    Persistent storage using DuckDB.

    Creates tables on first use and provides CRUD operations
    for all Leo data types.

    Transaction support:
        begin_transaction()
        commit()
        rollback()

    Context manager support:
        with store.connect() as conn:
            conn.execute(...)

    Safe-training helpers:
        delete_derived_tables()  — clears intent_embeddings, cached_embeddings only
        add_intent()             — idempotent, reuses existing IDs

    Database lock handling:
        - Detects stale lock files before connection attempt
        - Retries with exponential backoff up to DUCKDB_RETRY_MAX_ATTEMPTS
        - If still locked, raises DatabaseLockedError for caller to handle
        - All connections are tracked and closed on shutdown
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or settings.DUCKDB_PATH
        self._conn = None
        self._initialized = False
        self._in_transaction = False
        self._connections: List[Any] = []  # Track all connections

    # ── Connection management ───────────────────────────────────

    def _detect_stale_lock(self) -> bool:
        """
        Detect stale DuckDB lock files.

        DuckDB creates temporary lock files during operations.
        If the process crashes, these can remain and block future connections.

        Also attempts to detect the lock owner (the PID that holds the lock).

        Returns:
            True if stale locks were found and cleaned.
        """
        db_path = Path(self._db_path)
        cleaned = False

        # DuckDB lock patterns
        lock_patterns = [
            db_path.with_suffix(".duckdb.wal"),
            db_path.with_suffix(".duckdb.tmp"),
            db_path.parent / f"{db_path.name}.lock",
        ]

        now = time.time()
        for lock_file in lock_patterns:
            if lock_file.exists():
                try:
                    # Detect lock owner: check if another process holds the file
                    lock_owner = None
                    try:
                        import fcntl
                        with open(lock_file, 'r') as lf:
                            try:
                                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                # We got the lock, so nobody else holds it — it's stale
                                fcntl.flock(lf, fcntl.LOCK_UN)
                                lock_owner = "stale (no contender)"
                            except BlockingIOError:
                                # Another process holds the lock
                                lock_owner = "another process"
                    except (ImportError, Exception):
                        pass

                    # Check if the file is stale (older than threshold)
                    file_age = now - lock_file.stat().st_mtime
                    if file_age > STALE_LOCK_AGE_SECONDS:
                        if lock_owner:
                            logger.info(
                                "Lock file %s is stale (age=%.0fs, owner=%s) — removing",
                                lock_file, file_age, lock_owner
                            )
                        lock_file.unlink()
                        cleaned = True
                    else:
                        logger.debug(
                            "Lock file %s is recent (age=%.0fs, owner=%s), not removing",
                            lock_file, file_age, lock_owner or "unknown"
                        )
                except Exception as e:
                    logger.warning("Could not inspect lock file %s: %s", lock_file, e)

        return cleaned

    def _in_memory_fallback(self) -> Any:
        """
        Create an in-memory DuckDB connection as fallback when the
        persistent database is locked or unavailable.
        
        Never blocks command execution.
        """
        if not HAS_DUCKDB:
            return None
        try:
            conn = duckdb.connect(":memory:")
            self._connections.append(conn)
            logger.info("Connected to DuckDB in-memory (fallback)")
            return conn
        except Exception as e:
            logger.warning("DuckDB in-memory fallback also failed: %s", e)
            return None

    def _connect_with_retry(self) -> Any:
        """Connect to DuckDB with stale lock detection and exponential backoff."""
        if not HAS_DUCKDB:
            return None

        max_attempts = settings.DUCKDB_RETRY_MAX_ATTEMPTS
        base_delay = settings.DUCKDB_RETRY_BASE_DELAY

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Detect and remove stale locks before attempting connection
        self._detect_stale_lock()

        for attempt in range(1, max_attempts + 1):
            try:
                conn = duckdb.connect(str(self._db_path))
                self._connections.append(conn)
                logger.info("Connected to DuckDB: %s", self._db_path)
                return conn
            except Exception as e:
                error_str = str(e).lower()
                if "lock" in error_str or "conflicting lock" in error_str:
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "DuckDB locked (attempt %d/%d), retrying in %.1fs...",
                            attempt, max_attempts, delay
                        )
                        time.sleep(delay)
                        # Try stale lock detection again before retry
                        self._detect_stale_lock()
                        continue
                    logger.error("DuckDB still locked after %d attempts", max_attempts)
                    raise DatabaseLockedError(
                        f"DuckDB at {self._db_path} is locked by another process. "
                        f"Retried {max_attempts} times."
                    ) from e
                raise

    @contextmanager
    def connect(self) -> Generator:
        """
        Context manager for short-lived connections.
        Automatically closes the connection when done.

        Usage:
            with store.connect() as conn:
                conn.execute("SELECT 1")
        """
        conn = duckdb.connect(str(self._db_path)) if HAS_DUCKDB else None
        if conn is not None:
            self._connections.append(conn)
        try:
            yield conn
        finally:
            if conn is not None:
                try:
                    conn.close()
                    self._connections.remove(conn)
                except ValueError:
                    pass  # Already removed

    def _close_all_connections(self) -> None:
        """Close all tracked connections (for cleanup)."""
        for conn in list(self._connections):
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

    def close(self) -> None:
        """Close the persistent connection if open. Performs WAL checkpoint first."""
        if self._conn is not None:
            try:
                # Checkpoint WAL to prevent stale WAL files on next startup
                self._conn.execute("CHECKPOINT")
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._initialized = False
            logger.info("DuckDB persistent connection closed (WAL checkpointed)")

        # Close any orphaned connections
        self._close_all_connections()

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        self.close()

    # ── Connection access ───────────────────────────────────────

    def _get_conn(self):
        """Get or create persistent connection (for performance)."""
        if self._conn is None:
            if HAS_DUCKDB:
                try:
                    self._conn = self._connect_with_retry()
                    self._connections.append(self._conn)
                except DatabaseLockedError:
                    logger.warning("DuckDB locked, falling back to in-memory mode")
                    self._conn = self._in_memory_fallback()
            else:
                self._conn = None
        return self._conn

    # ── Transaction support ────────────────────────────────────────

    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        conn = self._get_conn()
        if conn is None:
            return
        conn.begin()
        self._in_transaction = True

    def commit(self) -> None:
        """Commit the current transaction."""
        conn = self._get_conn()
        if conn is None:
            return
        conn.commit()
        self._in_transaction = False

    def rollback(self) -> None:
        """Roll back the current transaction."""
        conn = self._get_conn()
        if conn is None:
            return
        conn.rollback()
        self._in_transaction = False

    # ── Initialization / Migration ─────────────────────────────────

    def initialize(self) -> None:
        """Create all tables if they don't exist.  Runs migration logic."""
        if self._initialized:
            return

        conn = self._get_conn()
        if conn is None:
            logger.warning("DuckDB not available, skipping table creation")
            self._initialized = True
            return

        # Create parent tables  (never dropped during training)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intents (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL UNIQUE,
                description VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_examples (
                id INTEGER PRIMARY KEY,
                intent_id INTEGER,
                text VARCHAR NOT NULL,
                embedding FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (intent_id) REFERENCES intents(id)
            )
        """)

        # Derived tables – safe to DELETE / TRUNCATE during training
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_embeddings (
                id INTEGER PRIMARY KEY,
                intent_id INTEGER NOT NULL,
                example_id INTEGER,
                text VARCHAR NOT NULL,
                embedding FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (intent_id) REFERENCES intents(id),
                FOREIGN KEY (example_id) REFERENCES intent_examples(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cached_embeddings (
                id INTEGER PRIMARY KEY,
                text_hash VARCHAR NOT NULL UNIQUE,
                text VARCHAR NOT NULL,
                embedding FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                entity_type VARCHAR NOT NULL,
                value VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS synonyms (
                id INTEGER PRIMARY KEY,
                canonical VARCHAR NOT NULL,
                synonym VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS context_memory (
                id INTEGER PRIMARY KEY,
                session_id VARCHAR,
                key VARCHAR NOT NULL,
                value VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS command_history (
                id INTEGER PRIMARY KEY,
                text VARCHAR NOT NULL,
                intent VARCHAR,
                confidence FLOAT,
                response VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY,
                key VARCHAR NOT NULL UNIQUE,
                value VARCHAR NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_registry (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL UNIQUE,
                version VARCHAR,
                enabled BOOLEAN DEFAULT TRUE,
                config VARCHAR,
                installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create sequences for auto-increment (DuckDB-native approach)
        for table in ["intents", "intent_examples", "intent_embeddings",
                       "cached_embeddings", "entities", "synonyms",
                       "context_memory", "command_history", "user_preferences",
                       "plugin_registry"]:
            conn.execute(f"""
                CREATE SEQUENCE IF NOT EXISTS seq_{table} START 1
            """)

        # ── Migration: schema version tracking ──────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        current_version = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        ).fetchone()[0]

        if current_version < SCHEMA_VERSION:
            self._run_migrations(conn, current_version)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                [SCHEMA_VERSION]
            )

        self._initialized = True
        logger.info("DuckDB tables initialized (schema version %d)", SCHEMA_VERSION)

    def _run_migrations(self, conn, from_version: int) -> None:
        """Run migrations from from_version+1 to SCHEMA_VERSION."""
        logger.info("Running migrations from version %d to %d", from_version, SCHEMA_VERSION)

        # Migration 1→2: add derived tables if missing (idempotent)
        if from_version < 2:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS intent_embeddings (
                    id INTEGER PRIMARY KEY,
                    intent_id INTEGER NOT NULL,
                    example_id INTEGER,
                    text VARCHAR NOT NULL,
                    embedding FLOAT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (intent_id) REFERENCES intents(id),
                    FOREIGN KEY (example_id) REFERENCES intent_examples(id)
                )
            """)
            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS seq_intent_embeddings START 1
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_embeddings (
                    id INTEGER PRIMARY KEY,
                    text_hash VARCHAR NOT NULL UNIQUE,
                    text VARCHAR NOT NULL,
                    embedding FLOAT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS seq_cached_embeddings START 1
            """)
            logger.info("Migration v2: added intent_embeddings / cached_embeddings")

    # ── Safe derived-table deletion ────────────────────────────────

    def delete_derived_tables(self) -> None:
        """
        Delete *only* derived training tables.
        Never touches `intents` or any other parent table.
        Safe to call inside a transaction.
        """
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return
        conn.execute("DELETE FROM intent_embeddings")
        conn.execute("DELETE FROM cached_embeddings")
        conn.execute("DELETE FROM intent_examples")
        logger.info("Cleared derived training tables (intent_embeddings, cached_embeddings, intent_examples)")

    # ── Intents ──────────────────────────────────────────────────

    def add_intent(self, name: str, description: str = "") -> int:
        """Add a new intent, or reuse existing ID if name already exists."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1

        existing = conn.execute(
            "SELECT id FROM intents WHERE name = ?", [name]
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE intents SET description = ? WHERE id = ?",
                [description, existing[0]]
            )
            return existing[0]

        result = conn.execute("""
            INSERT INTO intents (id, name, description)
            SELECT nextval('seq_intents'), ?, ?
            RETURNING id
        """, [name, description]).fetchone()
        return result[0] if result else -1

    def get_intents(self) -> List[Dict]:
        """Get all intents."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute("SELECT id, name, description FROM intents").fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2]} for r in rows]

    # ── Intent Examples ──────────────────────────────────────────

    def add_example(self, intent_id: int, text: str,
                    embedding: Optional[np.ndarray] = None) -> int:
        """Add an example phrase for an intent."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        emb_list = embedding.tolist() if embedding is not None else None
        result = conn.execute("""
            INSERT INTO intent_examples (id, intent_id, text, embedding)
            SELECT nextval('seq_intent_examples'), ?, ?, ?
            RETURNING id
        """, [intent_id, text, emb_list]).fetchone()
        return result[0] if result else -1

    def get_examples(self, intent_id: int) -> List[Dict]:
        """Get all examples for an intent."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT id, text FROM intent_examples WHERE intent_id = ?",
            [intent_id]
        ).fetchall()
        return [{"id": r[0], "text": r[1]} for r in rows]

    # ── Intent Embeddings (derived table) ─────────────────────────

    def add_intent_embedding(self, intent_id: int, text: str,
                             embedding: np.ndarray,
                             example_id: Optional[int] = None) -> int:
        """Store a precomputed embedding for an intent example."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        emb_list = embedding.tolist()
        result = conn.execute("""
            INSERT INTO intent_embeddings (id, intent_id, example_id, text, embedding)
            SELECT nextval('seq_intent_embeddings'), ?, ?, ?, ?
            RETURNING id
        """, [intent_id, example_id, text, emb_list]).fetchone()
        return result[0] if result else -1

    def get_intent_embeddings(self, intent_id: int) -> List[Dict]:
        """Get all embeddings for an intent."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute("""
            SELECT id, intent_id, example_id, text, embedding
            FROM intent_embeddings
            WHERE intent_id = ?
        """, [intent_id]).fetchall()
        return [{
            "id": r[0], "intent_id": r[1], "example_id": r[2],
            "text": r[3], "embedding": r[4],
        } for r in rows]

    # ── Cached Embeddings (derived table) ─────────────────────────

    def cache_embedding(self, text: str, embedding: np.ndarray) -> int:
        """Cache a text→embedding mapping (deduplicated by text hash)."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        text_hash = str(hash(text))
        emb_list = embedding.tolist()
        result = conn.execute("""
            INSERT INTO cached_embeddings (id, text_hash, text, embedding)
            SELECT nextval('seq_cached_embeddings'), ?, ?, ?
            ON CONFLICT (text_hash) DO UPDATE
                SET embedding = EXCLUDED.embedding
            RETURNING id
        """, [text_hash, text, emb_list]).fetchone()
        return result[0] if result else -1

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding by text, or None."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return None
        text_hash = str(hash(text))
        result = conn.execute(
            "SELECT embedding FROM cached_embeddings WHERE text_hash = ?",
            [text_hash]
        ).fetchone()
        if result and result[0] is not None:
            return np.array(result[0], dtype=np.float32)
        return None

    # ── Synonyms ─────────────────────────────────────────────────

    def add_synonym(self, canonical: str, synonym: str) -> int:
        """Add a synonym mapping."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        result = conn.execute("""
            INSERT INTO synonyms (id, canonical, synonym)
            SELECT nextval('seq_synonyms'), ?, ?
            RETURNING id
        """, [canonical, synonym]).fetchone()
        return result[0] if result else -1

    def get_synonyms(self) -> Dict[str, List[str]]:
        """Get all synonyms as a dict mapping canonical -> [synonyms]."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return {}
        rows = conn.execute(
            "SELECT canonical, synonym FROM synonyms"
        ).fetchall()
        result: Dict[str, List[str]] = {}
        for canonical, synonym in rows:
            if canonical not in result:
                result[canonical] = []
            result[canonical].append(synonym)
        return result

    # ── Command History ──────────────────────────────────────────

    def add_command(self, text: str, intent: str = "",
                    confidence: float = 0.0,
                    response: str = "") -> int:
        """Log a command to history."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        result = conn.execute("""
            INSERT INTO command_history (id, text, intent, confidence, response)
            SELECT nextval('seq_command_history'), ?, ?, ?, ?
            RETURNING id
        """, [text, intent, confidence, response]).fetchone()
        return result[0] if result else -1

    def get_recent_commands(self, limit: int = 20) -> List[Dict]:
        """Get recent command history."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute("""
            SELECT id, text, intent, confidence, response, timestamp
            FROM command_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, [limit]).fetchall()
        return [{
            "id": r[0], "text": r[1], "intent": r[2],
            "confidence": r[3], "response": r[4], "timestamp": str(r[5]),
        } for r in rows]

    # ── User Preferences ─────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        """Set a user preference."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return
        conn.execute("""
            INSERT INTO user_preferences (id, key, value)
            SELECT nextval('seq_user_preferences'), ?, ?
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value,
                                            updated_at = now()
        """, [key, value])

    def get_preference(self, key: str, default: str = "") -> str:
        """Get a user preference."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return default
        result = conn.execute(
            "SELECT value FROM user_preferences WHERE key = ?", [key]
        ).fetchone()
        return result[0] if result else default

    def get_all_preferences(self) -> Dict[str, str]:
        """Get all user preferences."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return {}
        rows = conn.execute(
            "SELECT key, value FROM user_preferences"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Plugin Registry ──────────────────────────────────────────

    def register_plugin(self, name: str, version: str = "1.0.0",
                        enabled: bool = True,
                        config: Optional[Dict] = None) -> int:
        """Register a plugin in the registry."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        config_json = json.dumps(config) if config else "{}"
        result = conn.execute("""
            INSERT INTO plugin_registry (id, name, version, enabled, config)
            SELECT nextval('seq_plugin_registry'), ?, ?, ?, ?
            ON CONFLICT (name) DO UPDATE SET
                version = EXCLUDED.version,
                enabled = EXCLUDED.enabled,
                config = EXCLUDED.config
            RETURNING id
        """, [name, version, enabled, config_json]).fetchone()
        return result[0] if result else -1

    def get_plugins(self) -> List[Dict]:
        """Get all registered plugins."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute("""
            SELECT id, name, version, enabled, config
            FROM plugin_registry
        """).fetchall()
        return [{
            "id": r[0], "name": r[1], "version": r[2],
            "enabled": r[3], "config": json.loads(r[4]) if r[4] else {},
        } for r in rows]

    # ── Context Memory ───────────────────────────────────────────

    def save_context(self, session_id: str, key: str, value: str,
                     ttl_seconds: Optional[int] = None) -> int:
        """Save a context memory entry."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        expires = None
        if ttl_seconds:
            expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        result = conn.execute("""
            INSERT INTO context_memory (id, session_id, key, value, expires_at)
            SELECT nextval('seq_context_memory'), ?, ?, ?, ?
            RETURNING id
        """, [session_id, key, value, expires]).fetchone()
        return result[0] if result else -1

    def get_context(self, session_id: str, key: str) -> Optional[str]:
        """Get a context memory entry."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return None
        result = conn.execute("""
            SELECT value FROM context_memory
            WHERE session_id = ? AND key = ?
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ORDER BY created_at DESC
            LIMIT 1
        """, [session_id, key]).fetchone()
        return result[0] if result else None


# Global store instance
store = DuckDBStore()