"""
CommandNormalizer — Dedicated command normalization layer.

Normalizes spoken commands before they reach the Brain. Handles:
  - Common Whisper mishearings ("fire fox" → "firefox")
  - App name aliases ("vs code" → "vscode", "visual studio code" → "vscode")
  - Verb normalization ("play some music" → "play music")
  - Noise word removal ("please", "can you", "would you")
  - Contraction expansion ("what's" → "what is")
  - Follow-up command resolution ("open it", "continue", "go back")

This is the SINGLE place where raw transcripts become canonical commands.
The Brain calls normalize() before planning or dispatching.

Usage:
    from nlp.command_normalizer import command_normalizer

    cmd = command_normalizer.normalize("open fire fox")
    # → "open firefox"
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── App name aliases ───────────────────────────────────────────
# Maps spoken variants → canonical app name
_APP_ALIASES: Dict[str, str] = {
    # VS Code
    "vs code": "vscode",
    "v s code": "vscode",
    "visual studio code": "vscode",
    "visual studio": "vscode",
    "vscode": "vscode",
    "vs code editor": "vscode",
    "code editor": "vscode",
    "code": "vscode",

    # Firefox
    "fire fox": "firefox",
    "firefox": "firefox",
    "fire fox browser": "firefox",
    "mozilla firefox": "firefox",
    "mozilla": "firefox",

    # Chrome
    "chrome": "chrome",
    "crome": "chrome",
    "crom": "chrome",
    "google chrome": "chrome",
    "chrome browser": "chrome",
    "google chrome browser": "chrome",

    # Terminal
    "terminal": "terminal",
    "gnome terminal": "terminal",
    "command line": "terminal",
    "command prompt": "terminal",
    "shell": "terminal",
    "console": "terminal",

    # Spotify
    "spotify": "spotify",
    "spot if i": "spotify",
    "spot a fire": "spotify",
    "spotify music": "spotify",

    # YouTube
    "youtube": "youtube",
    "you tube": "youtube",
    "you too": "youtube",
    "you too fo me": "youtube",

    # PyCharm
    "pycharm": "pycharm",
    "pie charm": "pycharm",
    "pie chum": "pycharm",
    "python charm": "pycharm",

    # GitHub
    "github": "github",
    "get hub": "github",
    "git hub": "github",

    # Slack
    "slack": "slack",
    "slack app": "slack",

    # Discord
    "discord": "discord",
    "discord app": "discord",

    # Telegram
    "telegram": "telegram",
    "telegram desktop": "telegram",

    # Notion
    "notion": "notion",
    "notion app": "notion",

    # Calculator
    "calculator": "calculator",
    "calc": "calculator",
    "calculator app": "calculator",

    # Settings
    "settings": "settings",
    "preferences": "settings",
    "system settings": "settings",
    "control panel": "settings",

    # Files
    "files": "files",
    "file manager": "files",
    "nautilus": "files",
    "file explorer": "files",

    # Obsidian
    "obsidian": "obsidian",
    "obsidian app": "obsidian",
}

# ── Verb normalization ─────────────────────────────────────────
# Maps spoken verb phrases → canonical verb
_VERB_ALIASES: Dict[str, str] = {
    "launch": "open",
    "start": "open",
    "run": "open",
    "open up": "open",
    "open the": "open",
    "open a": "open",
    "open an": "open",
    "bring up": "open",
    "pull up": "open",
    "load": "open",
    "switch to": "open",
    "go to": "open",
    "navigate to": "open",
    "take me to": "open",
    "show me": "open",
    "display": "open",

    "turn up": "volume up",
    "increase volume": "volume up",
    "volume up": "volume up",
    "louder": "volume up",
    "make it louder": "volume up",
    "raise volume": "volume up",

    "turn down": "volume down",
    "decrease volume": "volume down",
    "volume down": "volume down",
    "quieter": "volume down",
    "make it quieter": "volume down",
    "lower volume": "volume down",

    "turn off": "shutdown",
    "shut down": "shutdown",
    "power off": "shutdown",
    "shutdown": "shutdown",
    "restart": "restart",
    "reboot": "restart",

    "increase brightness": "brightness up",
    "brightness up": "brightness up",
    "make it brighter": "brightness up",
    "brighter": "brightness up",
    "decrease brightness": "brightness down",
    "brightness down": "brightness down",
    "make it dimmer": "brightness down",
    "dimmer": "brightness down",

    "play some": "play",
    "play a": "play",
    "play the": "play",
    "play me": "play",
    "play": "play",
    "start playing": "play",
    "put on": "play",

    "pause the": "pause",
    "pause": "pause",
    "stop the": "stop",
    "stop": "stop",
    "resume": "resume",
    "continue playing": "resume",

    "next song": "next",
    "next track": "next",
    "next": "next",
    "skip": "next",
    "skip this": "next",

    "previous song": "previous",
    "previous track": "previous",
    "previous": "previous",
    "go back": "previous",
    "back": "previous",

    "lock the": "lock",
    "lock": "lock",
    "lock screen": "lock",
    "lock my": "lock",

    # Screenshot
    "take a screenshot": "screenshot",
    "take screenshot": "screenshot",
    "take a picture of the screen": "screenshot",
    "capture the screen": "screenshot",
    "capture screen": "screenshot",
    "screenshot": "screenshot",
}

# ── Noise words to remove ──────────────────────────────────────
_NOISE_WORDS = {
    "please", "can you", "could you", "would you", "will you",
    "hey", "hey leo", "ok leo", "okay leo", "leo", "listen",
    "i want to", "i need to", "i'd like to", "i would like to",
    "can i", "could i", "let's", "lets", "go ahead and",
    "just", "maybe", "perhaps", "kind of", "sort of",
    # Trailing phrases
    "for me", "right now", "right away", "now", "quickly",
    "fast", "immediately", "asap", "in a second", "in a moment",
    "in a bit", "in a while", "after a bit", "after a while",
    "later", "soon", "eventually", "finally", "at last", "at once",
}

# ── Contraction expansion ──────────────────────────────────────
_CONTRACTIONS = {
    "what's": "what is",
    "whats": "what is",
    "who's": "who is",
    "whos": "who is",
    "where's": "where is",
    "wheres": "where is",
    "when's": "when is",
    "whens": "when is",
    "how's": "how is",
    "hows": "how is",
    "it's": "it is",
    "its": "it is",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "don't": "do not",
    "dont": "do not",
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "i'm": "i am",
    "im": "i am",
    "you're": "you are",
    "youre": "you are",
    "we're": "we are",
    "were": "we are",
    "they're": "they are",
    "theyre": "they are",
}

# ── Follow-up command patterns ─────────────────────────────────
# CRITICAL FIX: "pause music", "stop music", "resume music",
# "open it", "close it" etc. must NOT be swallowed here — they are
# real commands that need the full router / pronoun-resolution path.
# Removing them lets the DecisionEngine resolve pronouns against
# conv_memory ("open it" → "open firefox") and the CommandRouter
# match music-control patterns directly.
_FOLLOW_UP_PATTERNS = {
    "continue": "continue",
    "go on": "continue",
    "keep going": "continue",
    "carry on": "continue",
    "proceed": "continue",
    "go back": "previous",
    "back": "previous",
    "again": "again",
    "do it again": "again",
    "repeat": "repeat",
    "say again": "again",
    "cancel": "cancel",
    "never mind": "cancel",
    "forget it": "cancel",
    "yes": "yes",
    "yeah": "yes",
    "yep": "yes",
    "sure": "yes",
    "ok": "yes",
    "okay": "yes",
    "go ahead": "yes",
    "do it": "yes",
    "no": "no",
    "nope": "no",
    "nah": "no",
}

# ── Music query patterns ───────────────────────────────────────
# "start X" must NOT match music — "start terminal" = open terminal.
# Only "start playing X" and "play X" are music commands.
_MUSIC_PATTERNS = [
    (r"^play\s+(?:some\s+)?(.+)$", "play_music"),
    (r"^play\s+(?:me\s+)?(.+)$", "play_music"),
    (r"^put\s+on\s+(.+)$", "play_music"),
    (r"^start\s+playing\s+(.+)$", "play_music"),
    (r"^start\s+(?:the\s+)?(?:music|song|track|playback)$", "play_music:music"),
]

# ── Screen reading patterns (must run BEFORE search) ───────────
_SCREEN_PATTERNS = [
    r"^what\s+is\s+(?:on|being\s+shown|displayed)\s+(?:my\s+|the\s+)?(?:screen|display|monitor|window|page|tab|browser|editor|terminal)",
    r"^what's\s+(?:on|being\s+shown|displayed)\s+(?:my\s+|the\s+)?(?:screen|display|monitor|window|page|tab|browser|editor|terminal)",
    r"^read\s+(?:the\s+)?(?:screen|display|monitor|window|page|tab|browser|editor|terminal)",
    r"^what\s+do\s+you\s+see",
    r"^what\s+are\s+you\s+looking\s+at",
]

# ── Search patterns ────────────────────────────────────────────
_SEARCH_PATTERNS = [
    (r"^search\s+the\s+web\s+(?:for\s+)?(.+)$", "search"),
    (r"^search\s+(?:for\s+)?(.+)$", "search"),
    (r"^look\s+up\s+(.+)$", "search"),
    (r"^find\s+(.+)$", "search"),
    (r"^google\s+(.+)$", "search"),
    (r"^what\s+is\s+(.+)$", "search"),
    (r"^who\s+is\s+(.+)$", "search"),
    (r"^how\s+to\s+(.+)$", "search"),
    (r"^weather\s+in\s+(.+)$", "search"),
    (r"^news\s+about\s+(.+)$", "search"),
    (r"^news$", "search:news"),
]


class CommandNormalizer:
    """Normalizes spoken commands into canonical form."""

    def __init__(self):
        self._stats = {
            "normalized": 0,
            "app_aliases": 0,
            "verb_aliases": 0,
            "noise_removed": 0,
            "follow_ups": 0,
            "music": 0,
            "search": 0,
            "screen": 0,
        }

    def normalize(self, text: str) -> str:
        """
        Normalize a spoken command into canonical form.

        Pipeline:
          1. Clean whitespace/punctuation
          2. Expand contractions
          3. Remove noise words
          4. Normalize verbs
          5. Resolve app aliases
          6. Detect follow-up commands
          7. Detect music/search patterns

        Args:
            text: Raw transcript from Whisper.

        Returns:
            Normalized command string.
        """
        if not text:
            return text

        self._stats["normalized"] += 1
        original = text

        # 1. Clean
        text = text.strip()
        text = re.sub(r'^[,.!?;:\s]+', '', text)
        text = re.sub(r'[,.!?;:\s]+$', '', text)
        text = re.sub(r'\s+', ' ', text)
        text_lower = text.lower()

        # 2. Expand contractions
        for wrong, correct in _CONTRACTIONS.items():
            if wrong in text_lower:
                text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
                text_lower = text.lower()

        # 3. Remove noise words
        for noise in sorted(_NOISE_WORDS, key=len, reverse=True):
            if noise in text_lower:
                text = re.sub(r'\b' + re.escape(noise) + r'\b', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s+', ' ', text).strip()
                text_lower = text.lower()
                self._stats["noise_removed"] += 1

        # 3b. Strip Hindi verb suffixes and prefix "open" when needed
        # "open firefox karo" → "open firefox"
        # "firefox kholo" → "open firefox" (kholo = open)
        # "volume badhao" → "volume up" (badhao = increase)
        # "volume kam karo" → "volume down" (kam = decrease)
        # "music chalao" → "play music" (chalao = play)
        # CRITICAL FIX: More specific patterns MUST come first.
        # `\s+karo$` would match "volume kam karo" and strip only "karo",
        # leaving "volume kam". The specific `\s+kam\s+karo$` must be
        # checked before the generic `\s+karo$`.
        _HINDI_SUFFIXES = [
            r"\s+kam\s+karo$", r"\s+kam\s+kar\s+do$",
            r"\s+khol\s+do$", r"\s+badha\s+do$", r"\s+chala\s+do$",
            r"\s+karo$", r"\s+kar\s+do$", r"\s+kholo$",
            r"\s+badhao$", r"\s+chalao$",
        ]
        for suffix in _HINDI_SUFFIXES:
            if re.search(suffix, text_lower):
                stripped = re.sub(suffix, '', text)
                stripped = re.sub(r'\s+', ' ', stripped).strip()
                stripped_lower = stripped.lower()
                # "kholo" means "open" — if the stripped text is just an app name,
                # prefix with "open "
                if "khol" in suffix and not stripped_lower.startswith(("open", "start", "launch")):
                    text = "open " + stripped
                # "badhao"/"badha do" = increase → "volume up"
                elif "badha" in suffix and stripped_lower.startswith("volume"):
                    text = "volume up"
                # "kam karo"/"kam kar do" = decrease → "volume down"
                # CRITICAL FIX: The suffix `\s+kam\s+karo$` strips "karo"
                # but leaves "kam" — we must detect the full "kam karo"
                # pattern and map it to "volume down".
                elif "kam" in suffix and stripped_lower.startswith("volume"):
                    text = "volume down"
                # "chalao"/"chala do" = play → "play X"
                # CRITICAL FIX: "lofi chalao" → "play lofi" (chalao = play)
                elif "chala" in suffix and stripped_lower.startswith("music"):
                    text = "play music"
                elif "chala" in suffix and stripped_lower.startswith("song"):
                    text = "play song"
                elif "chala" in suffix:
                    text = "play " + stripped
                else:
                    text = stripped
                text = re.sub(r'\s+', ' ', text).strip()
                text_lower = text.lower()

        # 4. Detect follow-up commands (before verb normalization)
        follow_up = self._detect_follow_up(text_lower)
        if follow_up:
            self._stats["follow_ups"] += 1
            return follow_up

        # 5. Detect music patterns
        music = self._detect_music(text_lower)
        if music:
            self._stats["music"] += 1
            return music

        # 5b. Detect screen reading patterns (BEFORE search — screen
        # questions must not be classified as searches)
        if self._detect_screen(text_lower):
            self._stats["screen"] += 1
            return text

        # 6. Detect search patterns
        search = self._detect_search(text_lower)
        if search:
            self._stats["search"] += 1
            return search

        # 7. Normalize verbs
        for wrong, correct in sorted(_VERB_ALIASES.items(), key=lambda x: -len(x[0])):
            if text_lower.startswith(wrong):
                text = correct + text[len(wrong):]
                text_lower = text.lower()
                self._stats["verb_aliases"] += 1
                break

        # 7b. Strip redundant object words after verb normalization
        # "volume up the volume" → "volume up"
        # "shutdown the computer" → "shutdown"
        # "lock the screen" → "lock"
        # "scroll down the page" → "scroll down"
        _REDUNDANT_OBJECTS = [
            r"\s+the\s+volume$", r"\s+volume$",
            r"\s+the\s+computer$", r"\s+computer$",
            r"\s+the\s+pc$", r"\s+pc$",
            r"\s+the\s+laptop$", r"\s+laptop$",
            r"\s+the\s+screen$", r"\s+screen$",
            r"\s+the\s+page$", r"\s+page$",
            # CRITICAL FIX: "music", "song", "track", "playback" are NOT
            # redundant — they are the OBJECT of music-control commands.
            # "pause music" must stay "pause music" so the CommandRouter's
            # ^(?:pause|stop)\s*(?:the\s+)?(?:music|song|track|playback)$
            # pattern can match it. Stripping them turned "pause music"
            # into "pause" which fell through to the LLM.
            r"\s+the\s+brightness$", r"\s+brightness$",
            r"\s+percent$",
        ]
        for pattern in _REDUNDANT_OBJECTS:
            if re.search(pattern, text_lower):
                text = re.sub(pattern, '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                text_lower = text.lower()

        # 8. Resolve app aliases
        for alias, canonical in sorted(_APP_ALIASES.items(), key=lambda x: -len(x[0])):
            if alias in text_lower:
                text = re.sub(r'\b' + re.escape(alias) + r'\b', canonical, text, flags=re.IGNORECASE)
                text_lower = text.lower()
                self._stats["app_aliases"] += 1

        # 9. Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

        if text != original:
            logger.debug("[NORMALIZE] '%s' → '%s'", original, text)
        return text

    def _detect_follow_up(self, text: str) -> Optional[str]:
        """Detect follow-up commands that reference prior context."""
        for pattern, action in _FOLLOW_UP_PATTERNS.items():
            if text == pattern or text.startswith(pattern + " "):
                return action
        return None

    def _detect_music(self, text: str) -> Optional[str]:
        """Detect music playback commands."""
        for pattern, action in _MUSIC_PATTERNS:
            m = re.match(pattern, text)
            if m:
                try:
                    query = m.group(1).strip()
                except (IndexError, AttributeError):
                    query = "music"
                if query:
                    # CRITICAL FIX: return a NATURAL command that the
                    # CommandRouter can match — NOT a tag format like
                    # "play_music:lofi" which the router's regex patterns
                    # cannot match and would fall through to the LLM.
                    if action == "play_music":
                        return f"play {query}"
                    if action == "play_music:music":
                        return "play music"
        return None

    def _detect_screen(self, text: str) -> bool:
        """Detect screen-reading commands (what's on screen, read screen)."""
        for pattern in _SCREEN_PATTERNS:
            if re.match(pattern, text):
                return True
        return False

    def _detect_search(self, text: str) -> Optional[str]:
        """Detect web search commands."""
        for pattern, action in _SEARCH_PATTERNS:
            m = re.match(pattern, text)
            if m:
                try:
                    query = m.group(1).strip()
                except (IndexError, AttributeError):
                    query = text
                if query:
                    # CRITICAL FIX: return a NATURAL command that the
                    # CommandRouter can match — NOT a tag format like
                    # "search:cats" which the router's regex patterns
                    # cannot match and would fall through to the LLM.
                    return f"search {query}"
        return None

    def report(self) -> Dict[str, int]:
        """Return normalization statistics."""
        return dict(self._stats)


# Global singleton
command_normalizer = CommandNormalizer()