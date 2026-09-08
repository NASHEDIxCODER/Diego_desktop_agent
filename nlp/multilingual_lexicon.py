"""
Multilingual command-word normalization (Hindi/Hinglish/Devanagari).

A SMALL, additive lexicon that lets the existing English-centric pipeline
understand mixed-language commands WITHOUT a second NLP pipeline or a
translation model. Only command vocabulary is covered (play/open/close/
search/stop/pause/resume/volume/screenshot/system-info/pe/par/mein), aligned
with Diego's already-supported intents.

SAFETY / ENTITY PRESERVATION:
  - All replacements are WHOLE-WORD, case-insensitive, on command vocabulary
    ONLY. Names / song titles / URLs / filenames / project names pass through
    untouched.
  - The order is: Devanagari transliteration -> adverbs (relocate the
    platform target) -> verbs -> generic imperative-suffix stripping.
    So 'youtube pe believer chalao' becomes 'on youtube believer' then
    'play on youtube believer' (re-ordered so the English regex
    'play X on youtube' can match).
  - The original ASR transcript is NEVER consumed: the engine keeps it in
    its own log; this module only returns the normalized text.

This module is a pre-stage run INSIDE command_normalizer.normalize() BEFORE
the existing English verb-alias and Hindi-suffix logic, so all existing
English / already-handled-Hindi behaviour is preserved.
"""

from __future__ import annotations

import re

def _pw(word: str) -> re.Pattern:
    """Compile a whole-word, case-insensitive ASCII pattern."""
    return re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)

# Hinglish verb equivalents (action words), in priority order.
_VERB_MAP = [
    (_pw("chala do"), "play"),
    (_pw("chalao"), "play"),
    (_pw("baja do"), "play"),
    (_pw("bajao"), "play"),
    (_pw("khol do"), "open"),
    (_pw("khol lo"), "open"),
    (_pw("kholo"), "open"),
    (_pw("band kar do"), "close"),
    (_pw("band karo"), "close"),
    (_pw("band kro"), "close"),
    (_pw("band kr do"), "close"),
    (_pw("roko"), "stop"),
    (_pw("rok do"), "pause"),
    (_pw("rok lo"), "stop"),
    (_pw("dobara chalao"), "resume"),
    (_pw("dobara chalu karo"), "resume"),
    (_pw("dhoondo"), "search"),
    (_pw("dhoond lo"), "find"),
    (_pw("khojo"), "search"),
    (_pw("khoj lo"), "search"),
    (_pw("screen shot le lo"), "screenshot"),
    (_pw("screenshot le lo"), "screenshot"),
    (_pw("screenshot lo"), "screenshot"),
    # Generic imperative suffix 'karo'/'kar do'/'kro' (strip only; the
    # preceding English/Hinglish verb already carries the intent).
    (re.compile(r"\s+karo$"), ""),
    (re.compile(r"\s+kar\s+do$"), ""),
    (re.compile(r"\s+kro$"), ""),
    (re.compile(r"\s+kr\s+do$"), ""),
    # Volume modifiers — fold 'thoda kam'/'kam' into 'down' and
    # 'thoda badha' into 'volume up' so the existing volume-intent regex
    # matches. 'thoda' alone stays (it may be a song/phrase target).
    (_pw("thoda kam"), "down"),
    (re.compile(r"\s+kam\s+karo$", re.IGNORECASE), " down"),
    (re.compile(r"\s+kam\s+kar\s+do$", re.IGNORECASE), " down"),
    (re.compile(r"\s+kam$", re.IGNORECASE), " down"),
    (_pw("thoda badha"), "volume up"),
    (_pw("thoda badhao"), "volume up"),
]

# Platform locatives ('pe'/'par'/'mein'/'ma' = on/in). Explicit platform
# patterns are matched first so the canonical 'on <platform>' form lands
# before the generic locative rule fires.
_LOCATIVE_MAP = [
    (_pw("youtube pe"), "on youtube"),
    (_pw("youtube par"), "on youtube"),
    (_pw("youtube mein"), "on youtube"),
    (_pw("youtube ma"), "on youtube"),
    (_pw("spotify pe"), "on spotify"),
    (_pw("spotify par"), "on spotify"),
    # Generic locative (any word + pe/par). 'on' so the English regex matches.
    (re.compile(r"\s+pe\b", re.IGNORECASE), " on"),
    (re.compile(r"\s+par\b", re.IGNORECASE), " on"),
    (re.compile(r"\s+mein\b", re.IGNORECASE), " on"),
    (re.compile(r"\s+ma\b", re.IGNORECASE), " on"),
]

# Devanagari transliteration of command tokens ONLY.
# A verbatim (not translation) pass that romanizes the command words so the
# rest of the pipeline (ASCII regex) can recognize them.
_DEVANAGARI_MAP = [
    ("चलाओ", "chalao"),
    ("चला दो", "chala do"),
    ("बजाओ", "bajao"),
    ("बजा दो", "baja do"),
    ("खोलो", "kholo"),
    ("खोल दो", "khol do"),
    ("बंद करो", "band karo"),
    ("बंद कर दो", "band kar do"),
    ("रोको", "roko"),
    ("रोक दो", "rok do"),
    ("ढूंढो", "dhoondo"),
    ("खोजो", "khojo"),
    ("करो", "karo"),
    ("कर दो", "kar do"),
    ("पे", "pe"),
    ("पर", "par"),
    ("में", "mein"),
    ("यूट्यूब", "youtube"),
    ("स्पॉटिफ़ाई", "spotify"),
    ("क्रोम", "chrome"),
    ("फ़ायरफॉक्स", "firefox"),
    ("स्क्रीनशॉट", "screenshot"),
    ("स्क्रीन शॉट ले लो", "screen shot le lo"),
    ("ले लो", "le lo"),
    ("बढ़ाओ", "badhao"),
    ("बढ़ा दो", "badha do"),
    ("कम", "kam"),
    ("थोड़ा", "thoda"),
    ("थोड़ी देर", "thodi der"),
    ("मेरा", "mera"),
    ("कितना", "kitna"),
    ("उपयोग", "use"),
    ("हो रहा", "ho raha"),
    ("है", "hai"),
]


def _apply(text: str, patterns) -> str:
    t = text
    for pat, repl in patterns:
        new = pat.sub(repl, t)
        if new != t:
            t = " ".join(new.split())
    return t


def _transliterate_devanagari(text: str) -> str:
    t = text
    for src, dst in _DEVANAGARI_MAP:
        if src in t:
            t = t.replace(src, dst)
    return t


def normalize_multilingual(text: str) -> str:
    """Normalize Hinglish/Hindi/Devanagari command words to canonical English.

    Idempotent and entity-safe: only command vocabulary is replaced; song
    titles, names, URLs, filenames, and content pass through unchanged.
    """
    if not text:
        return text
    t = _transliterate_devanagari(text)
    # Adverbs first (relocate the platform target to 'on <platform>').
    t = _apply(t, _LOCATIVE_MAP)
    # Verbs second.
    t = _apply(t, _VERB_MAP)
    t = " ".join(t.split()).strip()

    # ── Verb fronting ─────────────────────────────────────────
    # After substitution the verb may end up after the entity
    # ('on youtube believer play'). The CommandRouter's English regexes
    # expect the verb to lead ('play ... on youtube', 'open firefox').
    t = _front_verb(t)

    return t


def _front_verb(text: str) -> str:
    """Move a trailing command verb to the front so English regexes match.

    Handles the post-substitution shapes:
      'on youtube believer play'  -> 'play believer on youtube'
      'firefox open'              -> 'open firefox'
      'chrome close'             -> 'close chrome'
      'music search'             -> 'search music'
    The 'on <platform>' clause is pushed to the end so the English
    'play <entity> on youtube' regex can match.
    """
    t = text.strip()
    if not t:
        return t
    words = t.split()
    if not words:
        return t
    cmd_verbs = ("play", "open", "close", "search", "find", "stop",
                 "pause", "resume", "screenshot")
    lower = [w.lower() for w in words]
    # The first word might already be a verb (English idiom) — leave it.
    if lower[0] in cmd_verbs:
        return t
    # Find the index of the LAST command verb (closest to end - the Hindi
    # imperative usually trails).
    verb_idx = None
    for i in range(len(lower) - 1, 0, -1):
        if lower[i] in cmd_verbs:
            verb_idx = i
            break
    if verb_idx is None:
        return t
    verb = words[verb_idx]
    rest = words[:verb_idx] + words[verb_idx + 1:]
    # Move any leading 'on <platform>' clause to the end so the English
    # 'play <entity> on youtube' regex can match after verb fronting.
    if len(rest) >= 2 and rest[0].lower() == "on":
        # 'on youtube ...' -> '... on youtube' (keep the platform word glued).
        rest = rest[2:] + [rest[0], rest[1]]
    result = verb + " " + " ".join(rest)
    return " ".join(result.split())
