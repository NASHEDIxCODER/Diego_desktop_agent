"""
Leo ASR command dataset — shared ground truth for benchmark + recording.

Defines 100+ real Leo utterances across:
  - English commands
  - Natural speech (wake-word prefixed, corrections, follow-ups)
  - Hindi / Hinglish
  - Short commands (critical: stop/pause/resume/yes/no/cancel/back/again)
  - Noise-condition variants (same text, tagged for noise testing)

Each entry: (spoken, expected, language, category)
  - spoken: what is synthesized/recorded
  - expected: ground-truth transcript for scoring
  - language: en | hi | hinglish
  - category: english | natural | hindi | hinglish | short | noise

The benchmark MUST report English / Hindi / Hinglish accuracy separately.
"""

from __future__ import annotations

from typing import List, Tuple

# (spoken, expected, language, category)
COMMANDS: List[Tuple[str, str, str, str]] = [
    # ── English commands ────────────────────────────────────────
    ("open firefox", "open firefox", "en", "english"),
    ("open youtube", "open youtube", "en", "english"),
    ("open vscode", "open vscode", "en", "english"),
    ("open chrome", "open chrome", "en", "english"),
    ("open terminal", "open terminal", "en", "english"),
    ("open spotify", "open spotify", "en", "english"),
    ("play music", "play music", "en", "english"),
    ("pause music", "pause music", "en", "english"),
    ("stop", "stop", "en", "english"),
    ("continue", "continue", "en", "english"),
    ("go back", "go back", "en", "english"),
    ("search youtube", "search youtube", "en", "english"),
    ("search google", "search google", "en", "english"),
    ("close firefox", "close firefox", "en", "english"),
    ("open ghostline", "open ghostline", "en", "english"),
    ("what time is it", "what time is it", "en", "english"),
    ("what am I working on", "what am i working on", "en", "english"),
    ("summarize this screen", "summarize this screen", "en", "english"),
    ("volume up", "volume up", "en", "english"),
    ("volume down", "volume down", "en", "english"),
    ("mute", "mute", "en", "english"),
    ("take a screenshot", "take a screenshot", "en", "english"),
    ("lock the screen", "lock the screen", "en", "english"),
    ("open settings", "open settings", "en", "english"),
    ("open files", "open files", "en", "english"),
    ("next song", "next song", "en", "english"),
    ("previous song", "previous song", "en", "english"),
    ("shuffle", "shuffle", "en", "english"),
    ("repeat", "repeat", "en", "english"),
    ("go to github", "go to github", "en", "english"),
    ("check my email", "check my email", "en", "english"),
    ("open telegram", "open telegram", "en", "english"),
    ("open calendar", "open calendar", "en", "english"),
    ("open calculator", "open calculator", "en", "english"),
    ("show desktop", "show desktop", "en", "english"),
    ("minimize this window", "minimize this window", "en", "english"),
    ("maximize this window", "maximize this window", "en", "english"),
    ("close this window", "close this window", "en", "english"),
    ("scroll down", "scroll down", "en", "english"),
    ("scroll up", "scroll up", "en", "english"),
    ("open a new tab", "open a new tab", "en", "english"),
    ("close this tab", "close this tab", "en", "english"),
    ("refresh the page", "refresh the page", "en", "english"),
    ("what is the weather like", "what is the weather like", "en", "english"),
    ("set a timer for five minutes", "set a timer for five minutes", "en", "english"),
    ("tell me a joke", "tell me a joke", "en", "english"),
    ("thank you", "thank you", "en", "english"),
    ("good morning", "good morning", "en", "english"),
    ("what can you do", "what can you do", "en", "english"),
    ("how are you", "how are you", "en", "english"),

    # ── Natural speech (wake-word prefixed, corrections, follow-ups) ──
    ("hey leo open youtube", "open youtube", "en", "natural"),
    ("leo can you open firefox", "open firefox", "en", "natural"),
    ("actually open vscode instead", "open vscode", "en", "natural"),
    ("no cancel that", "cancel", "en", "natural"),
    ("continue", "continue", "en", "natural"),
    ("open it again", "open it again", "en", "natural"),
    ("hey leo play some music", "play music", "en", "natural"),
    ("leo please open chrome", "open chrome", "en", "natural"),
    ("can you search youtube", "search youtube", "en", "natural"),
    ("i want to open spotify", "open spotify", "en", "natural"),
    ("go ahead and open terminal", "open terminal", "en", "natural"),
    ("just open settings please", "open settings", "en", "natural"),
    ("hey leo what time is it", "what time is it", "en", "natural"),
    ("actually pause the music", "pause music", "en", "natural"),
    ("no stop that", "stop", "en", "natural"),
    ("never mind", "cancel", "en", "natural"),
    ("do it again", "again", "en", "natural"),
    ("go on", "continue", "en", "natural"),
    ("open it", "open it", "en", "natural"),
    ("close that", "close that", "en", "natural"),

    # ── Hindi / Hinglish ────────────────────────────────────────
    ("youtube kholo", "youtube kholo", "hi", "hindi"),
    ("firefox kholo", "firefox kholo", "hi", "hindi"),
    ("gaana chalao", "gaana chalao", "hi", "hindi"),
    ("volume kam karo", "volume kam karo", "hi", "hindi"),
    ("isko band karo", "isko band karo", "hi", "hindi"),
    ("youtube par music chalao", "youtube par music chalao", "hi", "hindi"),
    ("open firefox karo", "open firefox", "hinglish", "hinglish"),
    ("chrome kholo", "chrome kholo", "hinglish", "hinglish"),
    ("volume badhao", "volume badhao", "hinglish", "hinglish"),
    ("music chalao", "music chalao", "hinglish", "hinglish"),
    ("pause karo", "pause karo", "hinglish", "hinglish"),
    ("stop karo", "stop karo", "hinglish", "hinglish"),
    ("next karo", "next karo", "hinglish", "hinglish"),
    ("search karo", "search karo", "hinglish", "hinglish"),
    ("lock karo", "lock karo", "hinglish", "hinglish"),
    ("shutdown karo", "shutdown karo", "hinglish", "hinglish"),
    ("open spotify karo", "open spotify", "hinglish", "hinglish"),
    ("terminal kholo", "terminal kholo", "hinglish", "hinglish"),

    # ── Short commands (critical — must not be discarded) ────────
    ("stop", "stop", "en", "short"),
    ("pause", "pause", "en", "short"),
    ("resume", "resume", "en", "short"),
    ("yes", "yes", "en", "short"),
    ("no", "no", "en", "short"),
    ("cancel", "cancel", "en", "short"),
    ("back", "back", "en", "short"),
    ("again", "again", "en", "short"),
    ("next", "next", "en", "short"),
    ("go", "go", "en", "short"),
    ("on", "on", "en", "short"),
    ("off", "off", "en", "short"),
    ("up", "up", "en", "short"),
    ("down", "down", "en", "short"),
    ("open", "open", "en", "short"),
    ("play", "play", "en", "short"),
    ("quit", "quit", "en", "short"),
    ("exit", "exit", "en", "short"),
    ("help", "help", "en", "short"),
    ("home", "home", "en", "short"),
    ("menu", "menu", "en", "short"),
    ("mute", "mute", "en", "short"),
    ("skip", "skip", "en", "short"),
    ("repeat", "repeat", "en", "short"),

    # ── Noise-condition variants (same text, tagged for noise testing) ──
    ("open firefox", "open firefox", "en", "noise"),
    ("play music", "play music", "en", "noise"),
    ("stop", "stop", "en", "noise"),
    ("volume up", "volume up", "en", "noise"),
    ("what time is it", "what time is it", "en", "noise"),
    ("open youtube", "open youtube", "en", "noise"),
    ("pause music", "pause music", "en", "noise"),
    ("continue", "continue", "en", "noise"),
    ("search google", "search google", "en", "noise"),
    ("close firefox", "close firefox", "en", "noise"),
]


def by_language() -> dict:
    """Group commands by language (en / hi / hinglish)."""
    out = {"en": [], "hi": [], "hinglish": []}
    for spoken, expected, lang, cat in COMMANDS:
        out[lang].append((spoken, expected, lang, cat))
    return out


def by_category() -> dict:
    """Group commands by category."""
    out = {}
    for spoken, expected, lang, cat in COMMANDS:
        out.setdefault(cat, []).append((spoken, expected, lang, cat))
    return out


def total() -> int:
    return len(COMMANDS)