"""
ConversationMemory — Rolling context + long-term memory for Leo.

Architecture:
  - Short-term: rolling window of recent turns (configurable, default 20)
  - Long-term: facts extracted from conversation (e.g. "user's project is Leo")
  - Summaries: old history automatically summarized when window overflows

This makes Leo remember things across the session without sending the
entire history to the LLM every time.

Usage:
    from agent.conversation_memory import conv_memory

    conv_memory.add_user("remember my project is Leo")
    conv_memory.add_assistant("Got it, your project is Leo.")

    # Later...
    context = conv_memory.build_context()  # recent turns + long-term facts
    conv_memory.add_user("what was my project called?")
    # The context will include the long-term fact so the LLM can answer.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    role: str  # "user" or "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Rolling conversation memory with long-term fact extraction.

    - Keeps a rolling window of recent turns for immediate context.
    - Extracts "remember" statements into long-term facts.
    - Summarizes old turns when the rolling window overflows.
    - Provides a compact context string for the LLM prompt.
    """

    def __init__(self, max_turns: int = 20, max_facts: int = 50):
        self._turns: List[ConversationTurn] = []
        self._facts: List[str] = []  # long-term facts
        self._summaries: List[str] = []  # summaries of old conversation
        self._max_turns = max_turns
        self._max_facts = max_facts
        self._user_name: Optional[str] = None
        self._conversation_start = time.time()
        self._last_activity = time.time()

        # Patterns for fact extraction
        self._remember_patterns = [
            r"(?:remember|note|keep in mind|don't forget)\s+(?:that\s+)?(.+)",
            r"my\s+(\w+)\s+is\s+(.+)",
            r"i\s+(?:like|prefer|use|work on|am working on)\s+(.+)",
            r"i\s+am\s+(?:a|an)\s+(.+)",
            r"i\s+live\s+(?:in|at)\s+(.+)",
            r"i\s+work\s+(?:at|for|in)\s+(.+)",
            r"i\s+study\s+(.+)",
            r"i\s+am\s+called\s+(.+)",
            r"my\s+name\s+is\s+(.+)",
            r"call\s+me\s+(.+)",
        ]

    # ── Adding turns ──────────────────────────────────────

    def add_user(self, text: str) -> None:
        """Add a user turn and extract any facts."""
        self._turns.append(ConversationTurn("user", text))
        self._last_activity = time.time()
        self._extract_facts(text)
        self._trim()

    def add_assistant(self, text: str) -> None:
        """Add an assistant turn."""
        self._turns.append(ConversationTurn("assistant", text))
        self._last_activity = time.time()
        self._trim()

    # ── Fact extraction ───────────────────────────────────

    def _extract_facts(self, text: str) -> None:
        """Extract long-term facts from user text."""
        text_lower = text.lower().strip()

        # "remember my project is Leo" → "user's project is Leo"
        for pattern in self._remember_patterns:
            match = re.search(pattern, text_lower)
            if match:
                fact = match.group(0).strip()
                # Normalize: "remember that X" → "X"
                fact = re.sub(r"^(?:remember|note|keep in mind|don't forget)\s+(?:that\s+)?", "", fact)
                fact = fact.strip().rstrip(".!?")

                if fact and len(fact) > 3 and fact not in self._facts:
                    self._facts.append(fact)
                    logger.info("[MEMORY] Extracted fact: %s", fact)
                    if len(self._facts) > self._max_facts:
                        self._facts = self._facts[-self._max_facts:]

        # Extract name
        name_match = re.search(r"(?:my name is|call me|i am called)\s+([a-zA-Z]+)", text_lower)
        if name_match:
            self._user_name = name_match.group(1).capitalize()
            logger.info("[MEMORY] User name set: %s", self._user_name)

    # ── Context building ──────────────────────────────────

    def build_context(self) -> str:
        """
        Build a context string for the LLM prompt.

        Includes:
        - Long-term facts (if any)
        - Summaries of old conversation (if any)
        - Recent turns (rolling window)
        """
        parts: List[str] = []

        # Long-term facts
        if self._facts:
            facts_str = "\n".join(f"  - {f}" for f in self._facts[-10:])
            parts.append(f"Things you know about the user:\n{facts_str}")

        # Summaries
        if self._summaries:
            parts.append(f"Earlier conversation summary: {self._summaries[-1]}")

        # Recent turns
        if self._turns:
            recent = self._turns[-self._max_turns:]
            turns_str = "\n".join(
                f"{'User' if t.role == 'user' else 'Leo'}: {t.text}"
                for t in recent
            )
            parts.append(f"Recent conversation:\n{turns_str}")

        return "\n\n".join(parts) if parts else ""

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the N most recent turns."""
        return self._turns[-n:] if self._turns else []

    # ── Summarization ─────────────────────────────────────

    def _trim(self) -> None:
        """Trim the rolling window and summarize overflow."""
        if len(self._turns) <= self._max_turns:
            return

        # Keep the most recent turns, summarize the rest
        overflow = self._turns[:-self._max_turns]
        self._turns = self._turns[-self._max_turns:]

        # Create a simple summary of overflowed turns
        summary = self._create_summary(overflow)
        if summary:
            self._summaries.append(summary)
            # Keep only last 3 summaries
            self._summaries = self._summaries[-3:]
            logger.info("[MEMORY] Summarized %d old turns", len(overflow))

    def _create_summary(self, turns: List[ConversationTurn]) -> str:
        """Create a brief summary of a set of turns."""
        if not turns:
            return ""

        topics: List[str] = []
        for turn in turns:
            if turn.role == "user":
                # Extract key words (simple approach)
                words = turn.text.lower().split()
                # Look for action keywords
                for keyword in ["open", "search", "play", "send", "write", "run",
                                "close", "read", "find", "create", "delete", "remember"]:
                    if keyword in words:
                        topics.append(f"User asked to {keyword} something")
                        break
                else:
                    # General topic
                    if len(turn.text) > 10:
                        topics.append(f"User discussed: {turn.text[:50]}...")

        if not topics:
            return ""

        # Deduplicate
        seen = set()
        unique = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return "; ".join(unique[:5])

    # ── Querying facts ────────────────────────────────────

    def query_facts(self, question: str) -> Optional[str]:
        """
        Try to answer a question from long-term facts.

        Returns the relevant fact if found, None otherwise.
        """
        question_lower = question.lower()

        # "what was my project called?" → look for "project" in facts
        if "what" in question_lower or "what's" in question_lower:
            # Extract the key noun from the question
            for fact in reversed(self._facts):
                # Simple keyword matching
                fact_words = set(fact.lower().split())
                question_words = set(question_lower.split())
                overlap = fact_words & question_words
                # Remove common words
                common = {"what", "was", "is", "my", "the", "called", "name",
                          "i", "am", "a", "an", "do", "did", "you", "know",
                          "tell", "me", "about"}
                meaningful = overlap - common
                if len(meaningful) >= 1:
                    return fact

        return None

    # ── State management ──────────────────────────────────

    @property
    def user_name(self) -> Optional[str]:
        return self._user_name

    def set_user_name(self, name: str) -> None:
        self._user_name = name

    @property
    def is_active(self) -> bool:
        """Check if there's recent activity."""
        return (time.time() - self._last_activity) < 300.0  # 5 minutes

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    def clear(self) -> None:
        """Clear all memory (e.g., on new session)."""
        self._turns.clear()
        self._facts.clear()
        self._summaries.clear()
        self._user_name = None
        self._conversation_start = time.time()
        self._last_activity = time.time()
        logger.info("[MEMORY] Cleared all conversation memory")

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "turns": len(self._turns),
            "facts": len(self._facts),
            "summaries": len(self._summaries),
            "user_name": self._user_name,
            "active": self.is_active,
        }


# Global singleton
conv_memory = ConversationMemory()