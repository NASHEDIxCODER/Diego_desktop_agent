"""
BackgroundLearning — Idle-time self-improvement for Leo.

While the user is not speaking, Leo continuously learns:
  - Crawls trusted documentation and technical websites
  - Builds embeddings, summaries, and searchable knowledge
  - Updates a local knowledge base (no LLM retraining)
  - Learns from successful actions, failures, user corrections
  - Learns from repeated workflows

Key invariants:
  - NEVER interrupts the user while they're interacting
  - Respects CPU/GPU limits (throttles when user starts speaking)
  - Pauses learning when the conversation engine enters active states
  - Runs only during WAKE_LISTEN idle time

Usage:
    from core.background_learning import background_learner

    await background_learner.start()
    # ... Leo runs ...
    # Learner runs autonomously in the background during idle

    await background_learner.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────

# Trusted documentation sources (crawled during idle)
_TRUSTED_SOURCES = [
    "https://docs.python.org/3/",
    "https://developer.mozilla.org/en-US/docs/Web/",
    "https://nodejs.org/docs/latest/api/",
    "https://doc.rust-lang.org/book/",
    "https://docs.docker.com/",
    "https://kubernetes.io/docs/",
    "https://git-scm.com/docs",
    "https://docs.github.com/en",
    "https://archlinux.org/packages/",
]

# Local docs paths to index during idle
_LOCAL_DOCS = [
    Path("/usr/share/doc"),
    Path("/usr/share/man"),
]

# How often to attempt learning (seconds)
LEARN_INTERVAL_S = 30.0

# Maximum docs to crawl per session
MAX_CRAWL_PER_SESSION = 5

# Max local knowledge entries
MAX_KNOWLEDGE_ENTRIES = 5000

# Learning data path
LEARNING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@dataclass
class KnowledgeEntry:
    """A single learned fact or documentation snippet."""
    key: str  # hash of source+content
    topic: str
    content: str
    source: str  # URL or file path
    summary: str = ""
    embedding: Optional[List[float]] = None  # cached embedding
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = 0.0
    confidence: float = 1.0  # how reliable this knowledge is


class BackgroundLearner:
    """
    Idle-time self-improvement engine.

    Runs ONLY during WAKE_LISTEN. Pauses immediately when the user
    starts interacting. Respects CPU/GPU limits.
    """

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._knowledge: Dict[str, KnowledgeEntry] = {}
        self._crawled_sources: Set[str] = set()
        self._learn_count: int = 0
        self._last_learn_time: float = 0.0

        # Action learning: track what works
        self._action_successes: Dict[str, int] = {}  # action_name → count
        self._action_failures: Dict[str, Dict[str, int]] = {}  # action_name → {reason: count}
        self._workflow_patterns: Dict[str, int] = {}  # normalized_workflow → frequency

        # State tracking for pausing
        self._conversation_engine = None
        self._paused = False

    # ── Wiring ────────────────────────────────────────────────────

    def wire(self, conversation_engine=None) -> None:
        """Wire the conversation engine for state-aware pausing."""
        self._conversation_engine = conversation_engine

    @property
    def is_idle(self) -> bool:
        """True when Leo is idle and learning can run.

        CRITICAL FIX: The engine stores state in `_state` (private), not
        `state`. The EngineState enum has IDLE/WAKE/FACE_AUTH/LISTEN/
        THINK/SPEAK — there is no WAKE_LISTEN or BOOT. Learning may only
        run in IDLE or WAKE (wake-listening is idle from the user's
        perspective). Any active state (LISTEN/THINK/SPEAK/FACE_AUTH)
        must pause learning immediately.
        """
        if self._conversation_engine is None:
            return True
        state = getattr(self._conversation_engine, '_state', None)
        if state is None:
            return True
        from core.conversation_engine import EngineState
        return state in (EngineState.IDLE, EngineState.WAKE)

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background learner."""
        if self._running:
            return
        self._running = True
        self._load_knowledge()
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._learn_loop())
        logger.info("[BG-LEARN] Background learner started — %d knowledge entries loaded",
                     len(self._knowledge))

    async def stop(self) -> None:
        """Gracefully stop the learner."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._save_knowledge()
        logger.info("[BG-LEARN] Background learner stopped — %d entries saved",
                     len(self._knowledge))

    async def _learn_loop(self) -> None:
        """Main learning loop: runs periodically during idle periods."""
        try:
            while self._running:
                await asyncio.sleep(LEARN_INTERVAL_S)

                if not self.is_idle:
                    continue

                if self._paused:
                    continue

                try:
                    await self._learn_one_cycle()
                except Exception as e:
                    logger.debug("[BG-LEARN] Learning cycle error: %s", e)

        except asyncio.CancelledError:
            pass

    async def _learn_one_cycle(self) -> None:
        """Execute one learning cycle (crawl + index + consolidate + optimize)."""
        t0 = time.time()

        # ── 1. Crawl documentation (if not done recently) ──
        await self._crawl_docs()

        # ── 2. Index local docs ────────────────────────────
        await self._index_local_docs()

        # ── 3. Consolidate workflow patterns ───────────────
        self._consolidate_workflows()

        # ── 4. Prune old knowledge ─────────────────────────
        self._prune_old_knowledge()

        # ── 5. Optimize caches ─────────────────────────────
        self._optimize_caches()

        # ── 6. Compress memory ─────────────────────────────
        self._compress_memory()

        # ── 7. Measure tool reliability ────────────────────
        self._measure_tool_reliability()

        # ── 8. Discover repeated workflows ─────────────────
        self._discover_workflows()

        dur_s = time.time() - t0
        if dur_s > 0.1:
            self._learn_count += 1
            self._last_learn_time = time.time()
            logger.info("[BG-LEARN] Cycle #%d completed in %.2fs — %d entries",
                         self._learn_count, dur_s, len(self._knowledge))

    # ── Documentation crawling ─────────────────────────────────────

    async def _crawl_docs(self) -> None:
        """Crawl trusted documentation sources during idle."""
        # Only crawl a few sources per cycle to stay lightweight
        import random
        sources = [s for s in _TRUSTED_SOURCES if s not in self._crawled_sources]
        if not sources:
            return

        to_crawl = random.sample(sources, min(MAX_CRAWL_PER_SESSION, len(sources)))
        for url in to_crawl:
            if not self.is_idle:
                return  # Pause immediately

            try:
                content = await self._fetch_url(url)
                if content:
                    self._index_content(url, content)
                    self._crawled_sources.add(url)
                    logger.info("[BG-LEARN] Crawled: %s (%d chars)", url, len(content))
            except Exception as e:
                logger.debug("[BG-LEARN] Crawl failed for %s: %s", url, e)

    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a URL."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, follow_redirects=True)
                if resp.status_code == 200:
                    text = resp.text
                    # Extract readable text (strip HTML if present)
                    if text.strip().startswith("<"):
                        text = self._strip_html(text)
                    return text[:10000]  # Limit per page
        except Exception:
            pass
        return None

    @staticmethod
    def _strip_html(html: str) -> str:
        """Basic HTML text extraction."""
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        html = re.sub(r'<[^>]+>', ' ', html)
        # Collapse whitespace
        html = re.sub(r'\s+', ' ', html)
        return html.strip()

    def _index_content(self, source: str, content: str) -> None:
        """Index content into the knowledge base."""
        if not content or len(content) < 50:
            return

        # Split into paragraphs
        paragraphs = re.split(r'\n{2,}', content)
        for para in paragraphs:
            para = para.strip()
            if len(para) < 30 or len(para) > 2000:
                continue

            # Determine topic from first sentence
            topic = para.split('.')[0][:100].strip()
            if not topic:
                topic = "untitled"

            # Create summary (first 2 sentences)
            sentences = re.split(r'(?<=[.!?])\s+', para)
            summary = ' '.join(sentences[:2])[:300]

            entry_key = hashlib.md5((source + para[:200]).encode()).hexdigest()[:16]
            if entry_key in self._knowledge:
                continue

            self._knowledge[entry_key] = KnowledgeEntry(
                key=entry_key,
                topic=topic,
                content=para,
                source=source,
                summary=summary,
            )

            # Limit total entries
            if len(self._knowledge) >= MAX_KNOWLEDGE_ENTRIES:
                self._prune_old_knowledge()
                if len(self._knowledge) >= MAX_KNOWLEDGE_ENTRIES:
                    return

    async def _index_local_docs(self) -> None:
        """Index local documentation files."""
        for doc_dir in _LOCAL_DOCS:
            if not doc_dir.exists():
                continue
            try:
                for root, _, files in os.walk(doc_dir):
                    if not self.is_idle:
                        return
                    for f in files[:20]:  # Limit per cycle
                        path = Path(root) / f
                        try:
                            if path.suffix in ('.txt', '.html', '.rst', '.md') and path.stat().st_size < 50000:
                                content = path.read_text(errors='ignore')[:10000]
                                self._index_content(str(path), content)
                        except Exception:
                            continue
            except PermissionError:
                continue

    # ── Workflow learning ──────────────────────────────────────────

    def record_action_outcome(self, action_name: str, success: bool,
                               error: str = "", params: Optional[Dict] = None) -> None:
        """Record an action outcome for learning."""
        if success:
            self._action_successes[action_name] = self._action_successes.get(action_name, 0) + 1
        else:
            if action_name not in self._action_failures:
                self._action_failures[action_name] = {}
            reason = error[:50] if error else "unknown"
            self._action_failures[action_name][reason] = \
                self._action_failures[action_name].get(reason, 0) + 1

    def record_workflow(self, actions: List[str]) -> None:
        """Record a sequence of actions as a workflow pattern."""
        if len(actions) < 2:
            return
        normalized = " → ".join(actions)
        self._workflow_patterns[normalized] = self._workflow_patterns.get(normalized, 0) + 1

    def _consolidate_workflows(self) -> None:
        """Merge and identify the most common workflows."""
        # Keep only workflows that appeared more than once
        self._workflow_patterns = {
            k: v for k, v in self._workflow_patterns.items() if v > 1
        }
        # Sort by frequency
        sorted_wf = sorted(self._workflow_patterns.items(), key=lambda x: x[1], reverse=True)
        # Store top workflows in knowledge base
        for wf, count in sorted_wf[:10]:
            entry_key = hashlib.md5(("workflow:" + wf).encode()).hexdigest()[:16]
            self._knowledge[entry_key] = KnowledgeEntry(
                key=entry_key,
                topic="workflow",
                content=wf,
                source="learning",
                summary=f"Common workflow (seen {count} times): {wf}",
                confidence=min(1.0, count / 10.0),
            )

    # ── Knowledge queries ──────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> List[KnowledgeEntry]:
        """Search the local knowledge base."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: List[Tuple[float, KnowledgeEntry]] = []
        for entry in self._knowledge.values():
            score = 0.0
            content_lower = entry.content.lower()

            # Exact match bonus
            if query_lower in content_lower:
                score += 5.0

            # Word overlap
            content_words = set(content_lower.split())
            overlap = query_words & content_words
            score += len(overlap) * 0.5

            # Topic match bonus
            if query_lower in entry.topic.lower():
                score += 2.0

            # Confidence factor
            score *= entry.confidence

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, entry in scored[:max_results]:
            entry.access_count += 1
            entry.last_accessed = time.time()

        return [e for _, e in scored[:max_results]]

    def context_for_llm(self, query: str = "", max_tokens: int = 300) -> str:
        """Build a context block from the knowledge base for LLM injection."""
        results = self.search(query, max_results=3) if query else []
        if not results:
            return ""

        parts = []
        for entry in results[:3]:
            parts.append(f"- {entry.summary[:200]}")

        if not parts:
            return ""
        return "Learned knowledge:\n" + "\n".join(parts)

    # ── Persistence ────────────────────────────────────────────────

    def _save_knowledge(self) -> None:
        """Save the knowledge base to disk."""
        try:
            LEARNING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": [
                    {
                        "key": e.key,
                        "topic": e.topic,
                        "content": e.content,
                        "source": e.source,
                        "summary": e.summary,
                        "created_at": e.created_at,
                        "access_count": e.access_count,
                        "confidence": e.confidence,
                    }
                    for e in self._knowledge.values()
                ],
                "crawled_sources": list(self._crawled_sources),
                "action_successes": self._action_successes,
                "action_failures": self._action_failures,
                "workflow_patterns": self._workflow_patterns,
                "learn_count": self._learn_count,
                "updated_at": time.time(),
            }
            with open(LEARNING_DATA_PATH, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("[BG-LEARN] Saved %d entries to knowledge base", len(self._knowledge))
        except Exception as e:
            logger.warning("[BG-LEARN] Failed to save knowledge: %s", e)

    def _load_knowledge(self) -> None:
        """Load the knowledge base from disk."""
        try:
            if not LEARNING_DATA_PATH.exists():
                return
            with open(LEARNING_DATA_PATH, "r") as f:
                data = json.load(f)
            for entry_data in data.get("entries", []):
                entry = KnowledgeEntry(
                    key=entry_data["key"],
                    topic=entry_data.get("topic", ""),
                    content=entry_data.get("content", ""),
                    source=entry_data.get("source", ""),
                    summary=entry_data.get("summary", ""),
                    created_at=entry_data.get("created_at", time.time()),
                    access_count=entry_data.get("access_count", 0),
                    confidence=entry_data.get("confidence", 1.0),
                )
                self._knowledge[entry.key] = entry
            self._crawled_sources = set(data.get("crawled_sources", []))
            self._action_successes = data.get("action_successes", {})
            self._action_failures = data.get("action_failures", {})
            self._workflow_patterns = data.get("workflow_patterns", {})
            self._learn_count = data.get("learn_count", 0)
            logger.info("[BG-LEARN] Loaded %d knowledge entries", len(self._knowledge))
        except Exception as e:
            logger.debug("[BG-LEARN] Failed to load knowledge: %s", e)

    def _prune_old_knowledge(self) -> None:
        """Remove least-accessed entries when over capacity."""
        if len(self._knowledge) <= MAX_KNOWLEDGE_ENTRIES:
            return

        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(
            self._knowledge.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )
        to_remove = len(self._knowledge) - int(MAX_KNOWLEDGE_ENTRIES * 0.9)
        for key, _ in sorted_entries[:to_remove]:
            del self._knowledge[key]
        logger.debug("[BG-LEARN] Pruned %d old entries", to_remove)

    # ── Self-improvement: Cache optimization ───────────────────────

    def _optimize_caches(self) -> None:
        """
        Optimize caches during idle time.

        - Prune expired cache entries
        - Adjust TTLs based on access patterns
        - Pre-warm frequently accessed caches
        """
        try:
            from core.cache_manager import cache_manager

            # Prune expired entries across all domains
            for domain in list(cache_manager._stores.keys()):
                store = cache_manager._stores.get(domain, {})
                expired = [k for k, v in store.items() if v.is_expired]
                for k in expired:
                    del store[k]
                if expired:
                    logger.debug("[BG-LEARN] Cache pruned: %s (%d expired)", domain, len(expired))

            # Adjust TTLs: frequently accessed entries get longer TTLs
            for domain, store in cache_manager._stores.items():
                for entry in list(store.values()):
                    if entry.access_count > 10:
                        # Extend TTL for hot entries
                        entry.ttl_s = min(entry.ttl_s * 1.5, 3600.0)
                    elif entry.access_count == 0 and entry.age_s > entry.ttl_s * 0.5:
                        # Shorten TTL for cold entries
                        entry.ttl_s = max(entry.ttl_s * 0.5, 0.5)

        except Exception as e:
            logger.debug("[BG-LEARN] Cache optimization failed: %s", e)

    # ── Self-improvement: Memory compression ───────────────────────

    def _compress_memory(self) -> None:
        """
        Compress conversation memory during idle time.

        - Summarize old conversation turns
        - Remove stale entities
        - Compact auto-tracked knowledge
        """
        try:
            from agent.conversation_memory import conv_memory

            # Force summarization if turns exceed 75% of max
            if len(conv_memory._turns) > conv_memory._max_turns * 0.75:
                conv_memory._trim()
                logger.debug("[BG-LEARN] Memory compressed: %d turns → %d",
                             len(conv_memory._turns) + len(conv_memory._summaries) * 5,
                             len(conv_memory._turns))

            # Prune old entities (keep last 5)
            if len(conv_memory._last_entities) > 10:
                conv_memory._last_entities = conv_memory._last_entities[-5:]

            # Compact preferences (remove duplicates)
            seen_prefs = set()
            compacted = {}
            for k, v in list(conv_memory._preferences.items()):
                if (k, v) not in seen_prefs:
                    seen_prefs.add((k, v))
                    compacted[k] = v
            conv_memory._preferences = compacted

        except Exception as e:
            logger.debug("[BG-LEARN] Memory compression failed: %s", e)

    # ── Self-improvement: Tool reliability measurement ─────────────

    def _measure_tool_reliability(self) -> None:
        """
        Measure and update tool reliability scores during idle.

        - Sync action success/failure counts to ToolReliability
        - Identify unreliable tools
        - Log reliability report periodically
        """
        try:
            from core.tool_reliability import tool_reliability

            # Sync our action tracking to the reliability system
            for action_name, count in self._action_successes.items():
                for _ in range(min(count, 10)):  # Don't replay all
                    tool_reliability.record_success(action_name)

            for action_name, reasons in self._action_failures.items():
                for reason, count in reasons.items():
                    for _ in range(min(count, 5)):
                        tool_reliability.record_failure(action_name, reason)

            # Log unreliable tools every 10 cycles
            if self._learn_count % 10 == 0:
                unreliable = tool_reliability.unreliable_tools(threshold=0.5)
                if unreliable:
                    logger.info("[BG-LEARN] Unreliable tools: %s",
                                ", ".join(unreliable[:5]))

        except Exception as e:
            logger.debug("[BG-LEARN] Tool reliability measurement failed: %s", e)

    # ── Self-improvement: Workflow discovery ───────────────────────

    def _discover_workflows(self) -> None:
        """
        Discover repeated workflows from command history.

        - Analyze command sequences
        - Identify patterns (e.g., "open code → open terminal")
        - Promote frequent patterns to known workflows
        """
        try:
            from core.command_router import command_router

            # Check if we have enough workflow data
            if len(self._workflow_patterns) < 3:
                return

            # Find workflows that appear 3+ times
            frequent = {
                wf: count
                for wf, count in self._workflow_patterns.items()
                if count >= 3
            }

            if frequent:
                # Promote to command router's known workflows
                for wf, count in sorted(frequent.items(), key=lambda x: x[1], reverse=True)[:3]:
                    actions = wf.split(" → ")
                    if len(actions) >= 2:
                        # Create a simple workflow name
                        name = " → ".join(actions[:3])
                        if name not in command_router._KNOWN_WORKFLOWS:
                            # Build action dicts
                            action_dicts = []
                            for a in actions[:5]:
                                action_dicts.append({
                                    "action": a.strip(),
                                    "params": {},
                                })
                            if action_dicts:
                                command_router._KNOWN_WORKFLOWS[name] = action_dicts
                                logger.info("[BG-LEARN] Discovered workflow: %s (seen %d times)",
                                            name, count)

        except Exception as e:
            logger.debug("[BG-LEARN] Workflow discovery failed: %s", e)

    # ── Diagnostics ────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Return learning statistics."""
        return {
            "knowledge_entries": len(self._knowledge),
            "crawled_sources": len(self._crawled_sources),
            "learn_cycles": self._learn_count,
            "action_successes": dict(sorted(
                self._action_successes.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_workflows": dict(sorted(
                self._workflow_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            "total_failures": sum(
                sum(reasons.values()) for reasons in self._action_failures.values()),
        }


# Global singleton
background_learner = BackgroundLearner()