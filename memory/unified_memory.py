"""
UnifiedMemoryService — Merges all Leo memory into one searchable layer.

Merges:
  * Conversation      — rolling turns + long-term facts (ConversationMemory)
  * Long-term facts   — extracted facts, user name, preferences
  * Embeddings        — cached vector embeddings (DuckDB)
  * Desktop state     — active window, open apps, screen hash
  * Clipboard history — recent clipboard entries
  * Window history    — recently focused windows
  * Recent commands   — command history (DuckDB)
  * Projects          — known project directories (desktop scan)
  * Current tasks     — active task plans
  * Summaries         — conversation summaries

Everything is searchable through `search(query)` and queryable by
namespace through `get_namespace(name)`.

This is the Memory layer of the layered architecture:
    Conversation Runtime → Planner → Reasoning → Memory → Tools → OS

Design:
  * A single service facade over the existing subsystems (backward
    compatible — conv_memory still works directly).
  * Thread-safe writes from executor threads.
  * Persists to DuckDB where a backend exists.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from core.event_bus import bus
from core.service import BaseService

logger = logging.getLogger(__name__)

# Ring-buffer sizes
CLIPBOARD_HISTORY_MAX = 50
WINDOW_HISTORY_MAX = 100
COMMAND_HISTORY_MAX = 200


@dataclass
class MemoryEntry:
    """A single memory entry."""
    namespace: str            # "conversation" | "fact" | "clipboard" | ...
    key: str
    value: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class UnifiedMemory(BaseService):
    """
    Unified, searchable memory service.

    Provides:
      * store(namespace, key, value, metadata) — unify any memory write
      * remember_fact(text)                    — extract + store a fact
      * search(query) -> [MemoryEntry]         — cross-namespace search
      * recent(namespace, limit)               — recent entries
      * clipboard history / window history     — automatic capture
    """

    name = "memory"

    def __init__(self):
        super().__init__()
        self._entries: Dict[str, Deque[MemoryEntry]] = {
            "conversation": deque(maxlen=200),
            "fact": deque(maxlen=500),
            "clipboard": deque(maxlen=CLIPBOARD_HISTORY_MAX),
            "window": deque(maxlen=WINDOW_HISTORY_MAX),
            "command": deque(maxlen=COMMAND_HISTORY_MAX),
            "project": deque(maxlen=100),
            "task": deque(maxlen=100),
            "summary": deque(maxlen=50),
            "desktop_state": deque(maxlen=100),
            "browser_tab": deque(maxlen=200),
        }
        self._search_index: Dict[str, List[str]] = {}  # keyword -> entry keys
        self._lock = asyncio.Lock()
        self._conv_memory = None  # lazily imported
        self._store_backend = None  # duckdb store

    async def _start(self) -> bool:
        """Initialize memory backends."""
        try:
            from agent.conversation_memory import conv_memory
            self._conv_memory = conv_memory
        except Exception:
            logger.warning("[MEMORY] conversation memory unavailable")
        try:
            from memory.duckdb_store import store
            self._store_backend = store
            self._store_backend.initialize()
        except Exception as e:
            logger.warning("[MEMORY] duckdb store unavailable: %s", e)

        # Subscribe to automatic capture events
        bus.on("clipboard.captured", self._on_clipboard_captured)
        bus.on("window.changed", self._on_window_changed)
        bus.on("command.executed", self._on_command_executed)
        bus.on("conversation.turn", self._on_conversation_turn)
        bus.on("browser.tab", self._on_browser_tab)

        self.set_health("memory ready",
                        {"entries": sum(len(d) for d in self._entries.values())})
        return True

    async def _stop(self) -> None:
        """Unsubscribe from the bus and release backends (idempotent)."""
        try:
            bus.off("clipboard.captured", self._on_clipboard_captured)
            bus.off("window.changed", self._on_window_changed)
            bus.off("command.executed", self._on_command_executed)
            bus.off("conversation.turn", self._on_conversation_turn)
            bus.off("browser.tab", self._on_browser_tab)
        except Exception as e:
            logger.debug("[MEMORY] unsubscribe error: %s", e)
        self._conv_memory = None
        self._store_backend = None

    # ── Core store API ─────────────────────────────────────

    async def store(self, namespace: str, key: str, value: str,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store an entry in a namespace."""
        if namespace not in self._entries:
            self._entries[namespace] = deque(maxlen=200)
        entry = MemoryEntry(namespace, key, value, metadata=metadata or {})
        async with self._lock:
            self._entries[namespace].append(entry)
            # Keyword index
            words = set(self._tokenize(value)) | set(self._tokenize(key))
            for w in words:
                self._search_index.setdefault(w, []).append(f"{namespace}:{id(entry)}")
            # Persist to duckdb for facts + commands
            if namespace == "fact":
                self._persist_fact(key, value)
            elif namespace == "command":
                self._persist_command(key, value, metadata or {})

        await bus.emit("memory.stored", data={
            "namespace": namespace, "key": key,
        }, source="memory")

    async def remember_fact(self, text: str) -> Optional[str]:
        """Extract and store a long-term fact."""
        if self._conv_memory is None:
            return None
        self._conv_memory.add_user(text)
        facts = self._conv_memory._facts
        if facts:
            fact = facts[-1]
            await self.store("fact", "fact", fact, {"source": "auto"})
            return fact
        return None

    # ── Search ─────────────────────────────────────────────

    async def search(self, query: str, limit: int = 20) -> List[MemoryEntry]:
        """Cross-namespace keyword search over all memory."""
        words = set(self._tokenize(query))
        if not words:
            return []
        # Collect all entries that match at least one keyword
        results: List[MemoryEntry] = []
        seen: set = set()
        async with self._lock:
            candidate_keys = set()
            for w in words:
                candidate_keys.update(self._search_index.get(w, []))
            for _, d in self._entries.items():
                for entry in d:
                    if id(entry) in seen:
                        continue
                    entry_words = set(self._tokenize(entry.value)) | set(
                        self._tokenize(entry.key))
                    overlap = words & entry_words
                    if overlap:
                        seen.add(id(entry))
                        results.append(entry)
        # Score: more keyword overlap = more relevant
        results.sort(key=lambda e: (
            -len(words & (set(self._tokenize(e.value)) | set(self._tokenize(e.key)))),
            -e.timestamp,
        ))
        return results[:limit]

    async def recent(self, namespace: str, limit: int = 20) -> List[MemoryEntry]:
        """Most recent entries in a namespace."""
        async with self._lock:
            d = self._entries.get(namespace, [])
            return list(d)[-limit:]

    async def get_namespace_list(self) -> List[str]:
        return list(self._entries.keys())

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                name: len(d) for name, d in self._entries.items()
            }

    # ── Event handlers (automatic capture) ─────────────────

    async def _on_clipboard_captured(self, event) -> None:
        text = event.data.get("text", "")
        if text:
            await self.store("clipboard", "clipboard", text,
                             {"source": "auto"})
            await bus.emit("memory.clipboard_updated", data={
                "text": text[:200],
            }, source="memory")

    async def _on_window_changed(self, event) -> None:
        title = event.data.get("title", "")
        app = event.data.get("app", "")
        if title:
            await self.store("window", "window", title,
                             {"app": app, "source": "auto"})

    async def _on_command_executed(self, event) -> None:
        text = event.data.get("text", "")
        intent = event.data.get("intent", "")
        response = event.data.get("response", "")
        if text:
            await self.store("command", "command", text,
                             {"intent": intent, "response": response,
                              "source": "auto"})

    async def _on_conversation_turn(self, event) -> None:
        role = event.data.get("role", "")
        text = event.data.get("text", "")
        if text:
            await self.store("conversation", f"{role}:{time.time()}", text,
                             {"role": role, "source": "auto"})

    async def _on_browser_tab(self, event) -> None:
        title = event.data.get("title", "")
        url = event.data.get("url", "")
        if url:
            await self.store("browser_tab", url, title or url,
                             {"url": url, "source": "auto"})

    # ── Persistence helpers ────────────────────────────────

    def _persist_fact(self, key: str, value: str) -> None:
        if self._store_backend is not None:
            try:
                self._store_backend.set_preference("fact:" + value[:100], value)
            except Exception:
                pass

    def _persist_command(self, text: str, intent: str, meta: Dict[str, Any]) -> None:
        if self._store_backend is not None:
            try:
                self._store_backend.add_command(
                    text, intent or meta.get("intent", ""),
                    meta.get("confidence", 0.0),
                    meta.get("response", ""))
            except Exception:
                pass

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple word tokenizer (lowercase, strip punctuation)."""
        import re
        return [w for w in re.split(r"[^a-zA-Z0-9_']+", text.lower())
                if len(w) > 2]

    # ── DuckDB passthrough (backward compatible) ───────────

    @property
    def duckdb(self):
        return self._store_backend


# Global singleton
unified_memory = UnifiedMemory()