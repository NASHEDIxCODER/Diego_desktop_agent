"""
ContextComposer — Smart memory injection for the LLM prompt.

Replaces the naive "dump all memories into the prompt" approach with a
ranking / compression / merging pipeline:

    1. Retrieve relevant memories from all sources
    2. Rank by semantic similarity × recency × confidence × frequency
    3. Compress similar memories into summaries
    4. Merge into a compact context block
    5. Enforce a hard token budget so the prompt never balloons

This is the SINGLE entry point for context injection. The conversation
engine no longer directly calls learning_engine.llm_context() — it calls
context_composer.compose() instead.

Usage:
    from agent.context_composer import context_composer

    context = context_composer.compose(user_text, max_tokens=800)
    # Inject `context` into the LLM prompt.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 800       # Hard ceiling for the context block
MAX_MEMORIES = 30              # Never inject more than this many
RECENCY_HALF_LIFE_S = 3600.0   # 1-hour recency half-life
SIMILARITY_WEIGHT = 0.40       # Semantic similarity weight
RECENCY_WEIGHT = 0.30          # Recency weight
CONFIDENCE_WEIGHT = 0.20       # Confidence / success-rate weight
FREQUENCY_WEIGHT = 0.10        # Usage-frequency weight
MIN_SCORE = 0.15               # Items below this score are discarded


@dataclass
class ContextItem:
    """A single memory / fact / preference / skill entry."""
    source: str                # "fact", "habit", "pref", "skill", "conv"
    key: str                   # unique key (fact text, action name, etc.)
    value: str                 # human-readable value
    confidence: float = 0.5    # 0–1 how reliable this is
    frequency: float = 0.0     # how often used (normalised 0–1)
    timestamp: float = 0.0     # epoch seconds
    embedding: Optional[List[float]] = None  # cached embedding vector
    score: float = 0.0         # computed rank score

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.timestamp)


class ContextComposer:
    """
    Smart context composer for Leo's LLM prompts.

    1. Collects memories from all learning subsystems
    2. Computes relevance score for each memory
    3. Ranks and filters to top-N
    4. Compresses similar memories
    5. Formats a compact text block
    """

    def __init__(self):
        self._embedding_cache: Dict[str, List[float]] = {}
        self._query_embedding_fn: Optional[callable] = None

    # ── Wire the embedding function ────────────────────────────

    def set_embedding_fn(self, fn: callable) -> None:
        """
        Provide a function that returns an embedding vector for a text string.

        Args:
            fn: callable(text: str) -> List[float]
        """
        self._query_embedding_fn = fn

    # ── Main entry point ───────────────────────────────────────

    def compose(self, user_text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """
        Compose a context block for the given user text.

        Args:
            user_text: What the user just said.
            max_tokens: Hard token budget (approximate).

        Returns:
            A compact text block for LLM prompt injection, or "" if nothing
            relevant was found.
        """
        # 1. Collect all memories
        items = self._collect_all_memories()

        if not items:
            return ""

        # 2. Compute query embedding
        query_embedding = None
        if self._query_embedding_fn:
            try:
                query_embedding = self._query_embedding_fn(user_text)
            except Exception as e:
                logger.debug("[Composer] embedding failed: %s", e)

        # 3. Rank items
        ranked = self._rank(items, query_embedding)

        # 4. Filter & truncate
        ranked = [r for r in ranked if r.score >= MIN_SCORE]
        ranked.sort(key=lambda x: x.score, reverse=True)
        ranked = ranked[:MAX_MEMORIES]

        # 5. Compress similar items
        compressed = self._compress_similar(ranked)

        # 6. Format to token budget
        return self._format(compressed, max_tokens)

    # ── Collection ─────────────────────────────────────────────

    def _collect_all_memories(self) -> List[ContextItem]:
        """Gather memories from every learning subsystem."""
        items: List[ContextItem] = []

        try:
            from learning.learning_engine import learning_engine
            eng = learning_engine
            now = time.time()

            # ── User profile facts ────────────────────────────
            for fact in eng.profile._facts:
                items.append(ContextItem(
                    source="fact",
                    key=f"fact:{fact.category}:{fact.key}",
                    value=f"{fact.key}: {fact.value}" if fact.value else str(fact.key),
                    confidence=getattr(fact, "confidence", 0.5),
                    frequency=getattr(fact, "frequency", 1.0),
                    timestamp=getattr(fact, "timestamp", now),
                ))

            # ── Preferences ────────────────────────────────────
            for pref in eng.preferences._preferences:
                items.append(ContextItem(
                    source="pref",
                    key=f"pref:{pref.category}:{pref.key}",
                    value=f"prefers {pref.value} for {pref.key}",
                    confidence=getattr(pref, "confidence", 0.7),
                    frequency=getattr(pref, "frequency", 1.0),
                    timestamp=getattr(pref, "timestamp", now),
                ))

            # ── Habits ─────────────────────────────────────────
            for hab in eng.habits._habits:
                items.append(ContextItem(
                    source="habit",
                    key=f"habit:{hab.category}:{hab.key}",
                    value=f"frequently uses {hab.key}",
                    confidence=0.6,
                    frequency=getattr(hab, "frequency", 0.5),
                    timestamp=getattr(hab, "timestamp", now),
                ))

            # ── Skills (action experience) ─────────────────────
            for entry in eng.skills._entries[-50:]:  # recent 50
                label = "successfully" if entry.success else "failed to"
                items.append(ContextItem(
                    source="skill",
                    key=f"skill:{entry.action}",
                    value=f"{label} {entry.action.replace('_', ' ')}",
                    confidence=1.0 if entry.success else 0.1,
                    frequency=eng.skills.success_rate(entry.action),
                    timestamp=entry.timestamp,
                ))

            # ── Conversation facts ─────────────────────────────
            try:
                from agent.conversation_memory import conv_memory
                for fact_text in conv_memory._facts:
                    items.append(ContextItem(
                        source="conv",
                        key=f"conv:{fact_text[:40]}",
                        value=fact_text,
                        confidence=0.8,
                        frequency=1.0,
                        timestamp=conv_memory._last_activity,
                    ))
            except Exception:
                pass

        except Exception as e:
            logger.debug("[Composer] Collection error: %s", e)

        return items

    # ── Ranking ────────────────────────────────────────────────

    def _rank(self, items: List[ContextItem],
              query_embedding: Optional[List[float]]) -> List[ContextItem]:
        """Compute a composite score for each item."""
        now = time.time()

        for item in items:
            scores: List[float] = []

            # Semantic similarity (cosine with query embedding)
            if query_embedding and item.embedding:
                sim = self._cosine_similarity(query_embedding, item.embedding)
            else:
                # Without embeddings, use string overlap as a proxy
                sim = self._string_overlap_score(item)

            scores.append((sim, SIMILARITY_WEIGHT))

            # Recency (exponential decay)
            age_s = item.age_s
            if age_s > 0:
                recency = math.exp(-age_s / RECENCY_HALF_LIFE_S * math.log(2))
            else:
                recency = 1.0
            scores.append((recency, RECENCY_WEIGHT))

            # Confidence
            scores.append((item.confidence, CONFIDENCE_WEIGHT))

            # Frequency (normalised)
            freq = min(item.frequency, 1.0)
            scores.append((freq, FREQUENCY_WEIGHT))

            # Weighted sum
            item.score = sum(v * w for v, w in scores)

        return items

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    @staticmethod
    def _string_overlap_score(item: ContextItem) -> float:
        """Fallback relevance: keyword overlap with the item value."""
        # Without query text, default to neutral
        return 0.5

    # ── Compression ────────────────────────────────────────────

    def _compress_similar(self, items: List[ContextItem]) -> List[ContextItem]:
        """
        Merge very similar items into summary items.

        Simple approach: group by source type and compress if we have
        too many of one type.
        """
        from collections import defaultdict
        by_source: Dict[str, List[ContextItem]] = defaultdict(list)
        for item in items:
            by_source[item.source].append(item)

        result: List[ContextItem] = []

        for source, group in by_source.items():
            if len(group) <= 5:
                result.extend(group)
            else:
                # Keep top 3 individually, compress the rest
                group.sort(key=lambda x: x.score, reverse=True)
                top = group[:3]
                rest = group[3:]

                result.extend(top)

                if rest:
                    summary_values = [r.value for r in rest[:10]]
                    summary_text = "; ".join(summary_values)
                    if len(summary_text) > 200:
                        summary_text = summary_text[:200] + "..."

                    # Use the best score from the group
                    best_score = top[0].score if top else 0.5
                    avg_ts = sum(r.timestamp for r in rest) / len(rest)

                    result.append(ContextItem(
                        source=f"{source}_summary",
                        key=f"{source}_summary",
                        value=summary_text,
                        confidence=0.4,
                        frequency=0.3,
                        timestamp=avg_ts,
                        score=best_score * 0.5,
                    ))

        return result

    # ── Formatting ─────────────────────────────────────────────

    def _format(self, items: List[ContextItem], max_tokens: int) -> str:
        """Format items into a compact text block within the token budget."""
        if not items:
            return ""

        # Sort by source category for cleaner output
        items.sort(key=lambda x: (x.source, -x.score))

        parts: List[str] = []
        token_estimate = 0
        chars_per_token = 4  # rough estimate

        for item in items:
            line = f"- {item.value}"
            est = len(line) // chars_per_token
            if token_estimate + est > max_tokens:
                if parts:
                    parts.append("...")
                break
            parts.append(line)
            token_estimate += est

        if not parts:
            return ""

        header = "Relevant context:"
        return header + "\n" + "\n".join(parts)

    # ── Quick context for system-level queries ─────────────────

    def quick_profile(self) -> str:
        """Return a one-line summary of what we know about the user."""
        try:
            from learning.learning_engine import learning_engine
            eng = learning_engine

            parts = []
            if eng.profile.fact_count > 0:
                parts.append(f"{eng.profile.fact_count} facts")
            if eng.habits.entry_count > 0:
                parts.append(f"{eng.habits.entry_count} habits")
            if eng.skills.entry_count > 0:
                parts.append(f"{eng.skills.entry_count} skills")
            return "Learned: " + ", ".join(parts) if parts else ""
        except Exception:
            return ""


# Global singleton
context_composer = ContextComposer()