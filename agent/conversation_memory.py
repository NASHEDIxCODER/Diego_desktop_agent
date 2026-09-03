"""
ConversationMemory — Rolling context + long-term memory for Diego.

Architecture:
  - Short-term: rolling window of recent turns (configurable, default 20)
  - Long-term: facts extracted from conversation (e.g. "user's project is Diego")
  - Auto-tracking: projects, folders, repos, apps, websites, commands
  - Summaries: old history automatically summarized when window overflows

This makes Diego remember things across the session without sending the
entire history to the LLM every time.

Usage:
    from agent.conversation_memory import conv_memory

    conv_memory.add_user("open my GhostLine project")
    conv_memory.add_assistant("Got it, opening GhostLine.")

    # Later...
    context = conv_memory.build_context()  # recent turns + long-term facts
    conv_memory.add_user("what was my project called?")
    # The context will include the long-term fact so the LLM can answer.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Pre-compiled regex patterns (avoids re-compilation on every turn) ──
_RE_PROJECT_1 = re.compile(r"(?:open|start|launch|work on|working on|my)\s+([A-Z][a-zA-Z0-9_-]+)(?:\s+project)?")
_RE_PROJECT_2 = re.compile(r"project\s+(?:is\s+)?(?:called\s+)?([A-Z][a-zA-Z0-9_-]+)")
_RE_FOLDER_1 = re.compile(r"(?:open|go to|cd to|navigate to|show)\s+(~?/[^\s,]+)")
_RE_FOLDER_2 = re.compile(r"(?:in|at)\s+(~?/[^\s,]+)\s+(?:folder|directory)")
_RE_REPO_1 = re.compile(r"(?:clone|pull|push|repo|repository)\s+(?:from\s+)?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)")
_RE_REPO_2 = re.compile(r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)")
_RE_APP = re.compile(r"(?:open|start|launch|run)\s+(VS Code|VSCode|Firefox|Chrome|Terminal|Spotify|Slack|Discord|Telegram|Notion|Obsidian|Docker|Postman|Insomnia|GIMP|Blender|Inkscape)", re.IGNORECASE)
_RE_URL = re.compile(r"(?:open|go to|navigate to|browse)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s,]*)?)")
_RE_CMD = re.compile(r"(?:run|execute|do)\s+(pytest|docker(?:\s+\w+)*|git\s+\w+|npm\s+\w+|pip\s+\w+|python\s+\S+|make\s+\w+|cargo\s+\w+)", re.IGNORECASE)
_RE_PERSON = re.compile(r"(?:message|email|call|text|tell|ask)\s+([A-Z][a-z]+)")
_RE_PREF_1 = re.compile(r"i\s+(?:prefer|like|love|enjoy|want)\s+(.+)")
_RE_PREF_2 = re.compile(r"my\s+(?:favorite|preferred)\s+(\w+)\s+is\s+(.+)")
_RE_REMEMBER = re.compile(r"(?:remember|note|keep in mind|don't forget)\s+(?:that\s+)?(.+)")
_RE_MY_IS = re.compile(r"my\s+(\w+)\s+is\s+(.+)")
_RE_I_AM = re.compile(r"i\s+(?:like|prefer|use|work on|am working on)\s+(.+)")
_RE_I_AM_A = re.compile(r"i\s+am\s+(?:a|an)\s+(.+)")
_RE_I_LIVE = re.compile(r"i\s+live\s+(?:in|at)\s+(.+)")
_RE_I_WORK = re.compile(r"i\s+work\s+(?:at|for|in)\s+(.+)")
_RE_I_STUDY = re.compile(r"i\s+study\s+(.+)")
_RE_I_CALLED = re.compile(r"i\s+am\s+called\s+(.+)")
_RE_MY_NAME = re.compile(r"my\s+name\s+is\s+(.+)")
_RE_CALL_ME = re.compile(r"call\s+me\s+(.+)")
_RE_NAME_EXTRACT = re.compile(r"(?:my name is|call me|i am called)\s+([a-zA-Z]+)")


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
    - Auto-tracks: projects, folders, repos, apps, websites, commands.
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

        # ── Auto-tracked knowledge ────────────────────────
        self._known_projects: Dict[str, str] = {}  # name → path
        self._known_folders: List[str] = []  # recently accessed folders
        self._known_repos: Dict[str, str] = {}  # name → path
        self._known_apps: List[str] = []  # frequently used apps
        self._known_websites: List[str] = []  # frequently visited sites
        self._known_commands: List[str] = []  # frequently used commands
        self._known_people: List[str] = []  # people mentioned
        self._preferences: Dict[str, str] = {}  # key → value

        # ── Pronoun resolution: track last N entities for "it", "that", etc. ──
        self._last_entities: List[str] = []  # most recent entities mentioned
        self._last_action: Optional[str] = None  # last action performed
        self._last_goal: Optional[str] = None  # last incomplete goal

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
        """Add a user turn, resolve pronouns, and extract any facts."""
        resolved = self._resolve_pronouns(text)
        self._turns.append(ConversationTurn("user", resolved))
        self._last_activity = time.time()
        self._extract_facts(resolved)
        self._auto_track(resolved)
        self._trim()

    def add_assistant(self, text: str) -> None:
        """Add an assistant turn."""
        self._turns.append(ConversationTurn("assistant", text))
        self._last_activity = time.time()
        self._auto_track(text)
        self._trim()

    # ── Auto-tracking: learn without explicit "remember" ──

    def _auto_track(self, text: str) -> None:
        """
        Automatically extract projects, folders, repos, apps, websites,
        commands, and people from conversation without explicit "remember".
        """
        text_lower = text.lower()

        # ── Project detection ─────────────────────────────
        # "open GhostLine" / "my GhostLine project" / "working on Diego"
        project_patterns = [
            r"(?:open|start|launch|work on|working on|my)\s+([A-Z][a-zA-Z0-9_-]+)(?:\s+project)?",
            r"project\s+(?:is\s+)?(?:called\s+)?([A-Z][a-zA-Z0-9_-]+)",
        ]
        for pat in project_patterns:
            for match in re.finditer(pat, text):
                name = match.group(1)
                if name.lower() not in ("the", "this", "that", "my", "your", "our"):
                    if name not in self._known_projects:
                        # Try to find the project path
                        path = self._find_project_path(name)
                        self._known_projects[name] = path or "unknown"
                        logger.info("[MEMORY] Auto-tracked project: %s → %s", name, path or "unknown")

        # ── Folder detection ──────────────────────────────
        # "open ~/dev" / "go to /home/user/projects"
        folder_patterns = [
            r"(?:open|go to|cd to|navigate to|show)\s+(~?/[^\s,]+)",
            r"(?:in|at)\s+(~?/[^\s,]+)\s+(?:folder|directory)",
        ]
        for pat in folder_patterns:
            for match in re.finditer(pat, text):
                folder = match.group(1)
                expanded = os.path.expanduser(folder)
                if expanded not in self._known_folders:
                    self._known_folders.append(expanded)
                    self._known_folders = self._known_folders[-20:]  # keep last 20
                    logger.info("[MEMORY] Auto-tracked folder: %s", expanded)

        # ── Repo detection ────────────────────────────────
        # "clone X" / "repo X" / "git repo X"
        repo_patterns = [
            r"(?:clone|pull|push|repo|repository)\s+(?:from\s+)?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)",
            r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)",
        ]
        for pat in repo_patterns:
            for match in re.finditer(pat, text):
                repo = match.group(1)
                if repo not in self._known_repos:
                    self._known_repos[repo] = f"github.com/{repo}"
                    logger.info("[MEMORY] Auto-tracked repo: %s", repo)

        # ── App detection ─────────────────────────────────
        # "open VS Code" / "start Firefox"
        app_patterns = [
            r"(?:open|start|launch|run)\s+(VS Code|VSCode|Firefox|Chrome|Terminal|Spotify|Slack|Discord|Telegram|Notion|Obsidian|Docker|Postman|Insomnia|GIMP|Blender|Inkscape)",
        ]
        for pat in app_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                app = match.group(1)
                if app not in self._known_apps:
                    self._known_apps.append(app)
                    self._known_apps = self._known_apps[-15:]
                    logger.info("[MEMORY] Auto-tracked app: %s", app)

        # ── Website detection ─────────────────────────────
        # "open github.com" / "go to stackoverflow.com"
        url_patterns = [
            r"(?:open|go to|navigate to|browse)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s,]*)?)",
        ]
        for pat in url_patterns:
            for match in re.finditer(pat, text):
                site = match.group(1)
                if site not in self._known_websites and not site.startswith(("http", "www")):
                    self._known_websites.append(site)
                    self._known_websites = self._known_websites[-20:]
                    logger.info("[MEMORY] Auto-tracked website: %s", site)

        # ── Command detection ─────────────────────────────
        # "run pytest" / "docker compose up" / "git status"
        cmd_patterns = [
            r"(?:run|execute|do)\s+(pytest|docker(?:\s+\w+)*|git\s+\w+|npm\s+\w+|pip\s+\w+|python\s+\S+|make\s+\w+|cargo\s+\w+)",
        ]
        for pat in cmd_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                cmd = match.group(1).strip()
                if cmd not in self._known_commands:
                    self._known_commands.append(cmd)
                    self._known_commands = self._known_commands[-30:]
                    logger.info("[MEMORY] Auto-tracked command: %s", cmd)

        # ── People detection ──────────────────────────────
        # "message John" / "email Sarah"
        people_patterns = [
            r"(?:message|email|call|text|tell|ask)\s+([A-Z][a-z]+)",
        ]
        for pat in people_patterns:
            for match in re.finditer(pat, text):
                person = match.group(1)
                if person not in self._known_people and person.lower() not in ("the", "me", "him", "her"):
                    self._known_people.append(person)
                    self._known_people = self._known_people[-20:]
                    logger.info("[MEMORY] Auto-tracked person: %s", person)

        # ── Preference detection ──────────────────────────
        # "I prefer dark mode" / "I like Python"
        pref_patterns = [
            r"i\s+(?:prefer|like|love|enjoy|want)\s+(.+)",
            r"my\s+(?:favorite|preferred)\s+(\w+)\s+is\s+(.+)",
        ]
        for pat in pref_patterns:
            for match in re.finditer(pat, text_lower):
                if match.lastindex and match.lastindex >= 1:
                    key = match.group(1).strip()[:40]
                    value = match.group(2).strip()[:60] if match.lastindex >= 2 else "yes"
                    if key not in self._preferences:
                        self._preferences[key] = value
                        logger.info("[MEMORY] Auto-tracked preference: %s = %s", key, value)

    @staticmethod
    def _find_project_path(name: str) -> Optional[str]:
        """Try to find a project directory by name in common locations."""
        search_dirs = [
            os.path.expanduser("~/PycharmProjects"),
            os.path.expanduser("~/projects"),
            os.path.expanduser("~/dev"),
            os.path.expanduser("~/Development"),
            os.path.expanduser("~/code"),
            os.path.expanduser("~/Documents"),
        ]
        for base in search_dirs:
            if not os.path.isdir(base):
                continue
            try:
                for entry in os.listdir(base):
                    full = os.path.join(base, entry)
                    if os.path.isdir(full) and entry.lower() == name.lower():
                        return full
            except PermissionError:
                continue
        return None

    # ── Fact extraction ───────────────────────────────────

    def _extract_facts(self, text: str) -> None:
        """Extract long-term facts from user text."""
        text_lower = text.lower().strip()

        # "remember my project is Diego" → "user's project is Diego"
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
        - Auto-tracked knowledge (projects, folders, repos, apps, preferences)
        - Summaries of old conversation (if any)
        - Recent turns (rolling window)
        """
        parts: List[str] = []

        # Auto-tracked knowledge
        knowledge_parts = []
        if self._known_projects:
            proj_str = ", ".join(
                f"{name} ({path})" if path != "unknown" else name
                for name, path in list(self._known_projects.items())[-5:]
            )
            knowledge_parts.append(f"Projects: {proj_str}")
        if self._known_folders:
            folder_str = ", ".join(self._known_folders[-5:])
            knowledge_parts.append(f"Folders: {folder_str}")
        if self._known_repos:
            repo_str = ", ".join(list(self._known_repos.keys())[-5:])
            knowledge_parts.append(f"Repos: {repo_str}")
        if self._known_apps:
            app_str = ", ".join(self._known_apps[-5:])
            knowledge_parts.append(f"Apps: {app_str}")
        if self._known_websites:
            site_str = ", ".join(self._known_websites[-5:])
            knowledge_parts.append(f"Sites: {site_str}")
        if self._known_commands:
            cmd_str = ", ".join(self._known_commands[-5:])
            knowledge_parts.append(f"Commands: {cmd_str}")
        if self._preferences:
            pref_str = ", ".join(f"{k}={v}" for k, v in list(self._preferences.items())[-5:])
            knowledge_parts.append(f"Preferences: {pref_str}")
        if knowledge_parts:
            parts.append("Known context:\n  " + "\n  ".join(knowledge_parts))

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
                f"{'User' if t.role == 'user' else 'Diego'}: {t.text}"
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

        # Also check auto-tracked knowledge
        if "project" in question_lower and self._known_projects:
            proj_names = list(self._known_projects.keys())
            return f"your projects: {', '.join(proj_names)}"
        if "folder" in question_lower and self._known_folders:
            return f"your folders: {', '.join(self._known_folders[-5:])}"
        if "repo" in question_lower and self._known_repos:
            return f"your repos: {', '.join(list(self._known_repos.keys())[-5:])}"

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
        # Don't clear auto-tracked knowledge — it persists across sessions
        logger.info("[MEMORY] Cleared conversation memory (auto-tracked knowledge preserved)")

    # ── Pronoun resolution ───────────────────────────────

    def _resolve_pronouns(self, text: str) -> str:
        """
        Resolve ambiguous pronouns using recent context.

        "open it" → "open GhostLine" (if last entity was GhostLine)
        "close that" → "close Firefox" (if last entity was Firefox)
        "go back" → references last action
        "continue" / "finish what you started" → references last goal
        """
        text_lower = text.lower().strip()

        # "it" / "that" / "this" / "there" → last entity
        if text_lower in ("open it", "close it", "start it", "run it", "show it",
                          "open that", "close that", "start that",
                          "what is it", "what's it", "tell me about it",
                          "go there", "navigate there", "open this"):
            if self._last_entities:
                entity = self._last_entities[-1]
                resolved = text_lower.replace(" it", f" {entity}").replace(
                    " that", f" {entity}").replace(
                    " this", f" {entity}").replace(
                    " there", f" {entity}")
                logger.info("[MEMORY] Pronoun resolved: '%s' → '%s'", text, resolved)
                return resolved

        # "go back" / "undo" → reference last action
        if text_lower in ("go back", "undo", "undo that", "revert"):
            if self._last_action:
                logger.info("[MEMORY] Pronoun resolved: '%s' → undo '%s'", text, self._last_action)
                return f"undo {self._last_action}"

        # "continue" / "finish what you started" → reference last goal
        # BUG-FIX (2026-09-03, runtime pass): "resume" was hijacked into
        # "continue <last goal>". Standalone "resume" is a MEDIA command
        # (resume playback) with a deterministic music_resume route — it
        # must never be rewritten as a task continuation. Users who mean
        # the task say "continue" or "resume the task" (the task-state
        # FollowUpResolver matches "resume the task" explicitly).
        if text_lower in ("continue", "finish what you started", "finish that",
                          "keep going", "carry on"):
            if self._last_goal:
                logger.info("[MEMORY] Pronoun resolved: '%s' → continue '%s'", text, self._last_goal)
                return f"continue {self._last_goal}"

        return text

    def track_entity(self, entity: str) -> None:
        """Track an entity for pronoun resolution."""
        if entity and entity.lower() not in ("it", "that", "this", "the", "a", "an"):
            self._last_entities.append(entity)
            self._last_entities = self._last_entities[-10:]

    def track_action(self, action: str) -> None:
        """Track the last action for 'go back' resolution."""
        self._last_action = action

    def track_goal(self, goal: str) -> None:
        """Track the last goal for 'continue' resolution."""
        self._last_goal = goal

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "turns": len(self._turns),
            "facts": len(self._facts),
            "summaries": len(self._summaries),
            "user_name": self._user_name,
            "active": self.is_active,
            "projects": len(self._known_projects),
            "folders": len(self._known_folders),
            "repos": len(self._known_repos),
            "apps": len(self._known_apps),
            "websites": len(self._known_websites),
            "commands": len(self._known_commands),
            "preferences": len(self._preferences),
        }


# Global singleton
conv_memory = ConversationMemory()