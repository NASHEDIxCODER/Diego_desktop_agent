"""
DuckDB storage layer for Leo Desktop Assistant.

Stores:
- intents and intent_examples
- embeddings
- entities and synonyms
- context_memory
- command_history
- user_preferences
- plugin_registry
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not installed. Using in-memory fallback.")


class DuckDBStore:
    """
    Persistent storage using DuckDB.

    Creates tables on first use and provides CRUD operations
    for all Leo data types.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or settings.DUCKDB_PATH
        self._conn = None
        self._initialized = False

    def _get_conn(self):
        """Get or create DuckDB connection."""
        if self._conn is None:
            if HAS_DUCKDB:
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
                self._conn = duckdb.connect(str(self._db_path))
                logger.info("Connected to DuckDB: %s", self._db_path)
            else:
                # In-memory fallback
                self._conn = None
        return self._conn

    def initialize(self) -> None:
        """Create all tables if they don't exist."""
        if self._initialized:
            return

        conn = self._get_conn()
        if conn is None:
            logger.warning("DuckDB not available, skipping table creation")
            self._initialized = True
            return

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

        # Create sequences for auto-increment
        for table in ["intents", "intent_examples", "entities", "synonyms",
                       "context_memory", "command_history", "user_preferences",
                       "plugin_registry"]:
            conn.execute(f"""
                CREATE SEQUENCE IF NOT EXISTS seq_{table} START 1
            """)

        self._initialized = True
        logger.info("DuckDB tables initialized")

    # ── Intents ──────────────────────────────────────────────────

    def add_intent(self, name: str, description: str = "") -> int:
        """Add a new intent. Returns intent ID."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        conn.execute("""
            INSERT INTO intents (id, name, description)
            VALUES (nextval('seq_intents'), ?, ?)
            ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description
        """, [name, description])
        result = conn.execute("SELECT id FROM intents WHERE name = ?", [name]).fetchone()
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
        conn.execute("""
            INSERT INTO intent_examples (id, intent_id, text, embedding)
            VALUES (nextval('seq_intent_examples'), ?, ?, ?)
        """, [intent_id, text, emb_list])
        result = conn.execute("SELECT lastval()").fetchone()
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

    # ── Synonyms ─────────────────────────────────────────────────

    def add_synonym(self, canonical: str, synonym: str) -> int:
        """Add a synonym mapping."""
        self.initialize()
        conn = self._get_conn()
        if conn is None:
            return -1
        conn.execute("""
            INSERT INTO synonyms (id, canonical, synonym)
            VALUES (nextval('seq_synonyms'), ?, ?)
        """, [canonical, synonym])
        result = conn.execute("SELECT lastval()").fetchone()
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
        conn.execute("""
            INSERT INTO command_history (id, text, intent, confidence, response)
            VALUES (nextval('seq_command_history'), ?, ?, ?, ?)
        """, [text, intent, confidence, response])
        result = conn.execute("SELECT lastval()").fetchone()
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
            VALUES (nextval('seq_user_preferences'), ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value,
                                            updated_at = CURRENT_TIMESTAMP
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
        conn.execute("""
            INSERT INTO plugin_registry (id, name, version, enabled, config)
            VALUES (nextval('seq_plugin_registry'), ?, ?, ?, ?)
            ON CONFLICT (name) DO UPDATE SET
                version = EXCLUDED.version,
                enabled = EXCLUDED.enabled,
                config = EXCLUDED.config
        """, [name, version, enabled, config_json])
        result = conn.execute(
            "SELECT id FROM plugin_registry WHERE name = ?", [name]
        ).fetchone()
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
            from datetime import timedelta
            expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        conn.execute("""
            INSERT INTO context_memory (id, session_id, key, value, expires_at)
            VALUES (nextval('seq_context_memory'), ?, ?, ?, ?)
        """, [session_id, key, value, expires])
        result = conn.execute("SELECT lastval()").fetchone()
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

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("DuckDB connection closed")


# Global store instance
store = DuckDBStore()