"""
Wake transcript verification — THE single wake-verification implementation.

This module is the SECOND (and final) wake authority. It runs ONLY after
openWakeWord triggers (see voice/wake_listener.py) and decides whether the
Whisper transcript of the trigger window actually contains the wake phrase.

Matching: RapidFuzz fuzzy ratio + metaphone phonetic equality, blended into
a per-word confidence against the DISTINCTIVE wake words. Whole-word
containment of a known variant accepts immediately.

Accept:  "hello diego"  "hey diego"  "hello dego"  "hello diego"  "hello digo"
         "hi diego"  "ok diego"  "hello diego!"  "hello, diego"  (phonetic near-misses)
Reject:  "hello"  "hello everyone"  "thank you"  "good morning"
         "hello video"  "yellow meow"  "<no speech>"

Both backends are OPTIONAL and degrade gracefully (difflib fallback).
"""

import difflib
import logging
import re
from typing import List, Optional, Tuple

from voice.settings import voice_settings

logger = logging.getLogger(__name__)

# ── Matching backends ──────────────────────────────────────────
# RapidFuzz: fast Levenshtein-based fuzzy ratios (preferred over difflib).
# jellyfish: metaphone phonetic encoding (offline).
try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _rf_fuzz = None
    _HAS_RAPIDFUZZ = False
    logger.debug("[WAKE-VERIFY] rapidfuzz not installed — using difflib fallback")

try:
    from jellyfish import metaphone as _metaphone
    _HAS_METAPHONE = True
except ImportError:
    _metaphone = None
    _HAS_METAPHONE = False
    logger.debug("[WAKE-VERIFY] jellyfish not installed — phonetic matching off")


# Accepted wake variants (whole-word containment). The distinctive words
# are derived from these: every word that is not a generic greeting.
DEFAULT_WAKE_VARIANTS = [
    "hello diego",
    "hey diego",
    "hi diego",
    "ok diego",
    "okay diego",
    "hello dego",
    "hello digo",
    "hello diego",
    "diego",
    "dego",
]

# Minimum combined confidence for a distinctive-word match. Deliberately
# strict: the openWakeWord trigger already fired, so this check only needs
# to reject transcripts that clearly do NOT contain the wake phrase.
WAKE_VERIFY_MIN_RATIO = 0.80

# Generic greeting words — never distinctive enough to verify on their own.
_GENERIC_WAKE_WORDS = {"hello", "hey", "ok", "okay", "hi", "ho"}


def _normalize_wake_text(text: str) -> str:
    """Normalize a transcript for wake verification."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _distinctive_wake_words() -> List[str]:
    """Return the distinctive (non-generic) words across all wake variants
    AND the configured WAKE_PHRASE — e.g. 'leo', 'lio', 'leyo', 'lido'."""
    distinctive = set()
    sources = list(DEFAULT_WAKE_VARIANTS) + [voice_settings.wake_phrase or ""]
    for variant in sources:
        for w in _normalize_wake_text(variant).split():
            if w and w not in _GENERIC_WAKE_WORDS:
                distinctive.add(w)
    return sorted(distinctive)


def _fuzzy_ratio(a: str, b: str) -> float:
    """Fuzzy similarity 0..1 (RapidFuzz when available, difflib fallback)."""
    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _phonetic_match(a: str, b: str) -> bool:
    """True when both words share a metaphone phonetic encoding.

    Catches Whisper's phonetically-plausible mishearings of the wake word
    ('leo', 'lea', 'lio' → metaphone 'L') that edit-distance alone would
    miss, while staying immune to unrelated words ('please' → 'PLS').
    """
    if not _HAS_METAPHONE or len(a) < 2 or len(b) < 2:
        return False
    try:
        ma, mb = _metaphone(a), _metaphone(b)
        return bool(ma) and ma == mb
    except Exception:
        return False


def _wake_word_confidence(word: str, distinctive: str) -> Tuple[float, str]:
    """Combined confidence that `word` IS the distinctive wake word.

    Returns (confidence 0..1, evidence tag). Blends fuzzy edit-distance
    similarity with phonetic equality; a phonetic hit is strong evidence:
    0.85 base + 0.15 × fuzzy.
    """
    fuzzy = _fuzzy_ratio(word, distinctive)
    if _phonetic_match(word, distinctive):
        return max(fuzzy, 0.85 + 0.15 * fuzzy), "phonetic+fuzzy"
    return fuzzy, "fuzzy"


def verify_wake_transcript(text: Optional[str], wake_score: float = 0.0,
                           whisper_confidence: float = 0.0) -> bool:
    """
    THE wake transcript verifier (RapidFuzz + phonetic matching).

    Passes ONLY when the normalized transcript EITHER:
      (a) contains a full wake variant as whole words (containment), OR
      (b) contains a word whose combined phonetic+fuzzy CONFIDENCE ≥ 0.80
          against one of the DISTINCTIVE wake words, OR
      (c) the openWakeWord score is ≥ 0.995 AND the transcript contains
          wake-word evidence (phonetic near-match), OR
      (d) the openWakeWord score is ≥ 0.95 AND Whisper confidence < -1.5
          (Whisper is clearly hallucinating on silence-heavy audio —
          trust the acoustic model).

    Exact transcript equality is NEVER required — Whisper's phonetically
    plausible mishearings of the wake word pass, while unrelated speech
    fails even when it shares the greeting.

    The openWakeWord model must ALSO have fired (enforced by the caller —
    this function only runs after the model crossed its threshold), so a
    phonetic near-match cannot wake Leo on its own.
    """
    if not text:
        return False

    # ── Uncertainty bypass: Whisper is clearly hallucinating ──
    # When the acoustic model is confident (≥ 0.95) but Whisper produces
    # a very low-confidence transcript (< -1.5), Whisper is hallucinating
    # on silence-heavy audio. Trust the acoustic model.
    if wake_score >= 0.98 and whisper_confidence < -0.7:
        logger.info(
            "[WAKE-VERIFY] ACCEPTED (uncertainty bypass): "
            "wake_score=%.3f ≥ 0.98 whisper_confidence=%.3f < -0.7 "
            "— Whisper highly uncertain, trusting acoustic model",
            wake_score, whisper_confidence)
        return True

    # ── High-confidence model path (CONSISTENCY REQUIRED) ──
    # A wake_score >= 0.995 is strong evidence the model fired, but it
    # MUST NOT override a transcript that clearly does not contain the
    # wake phrase. A score of 1.000 with transcript "I'll try to wrap it."
    # is evidence of a false positive, not a confirmation.
    #
    # The high-confidence shortcut is kept ONLY when the normalized
    # transcript contains at least one distinctive wake word (or a
    # phonetically-close match). If the transcript has no wake-word
    # evidence at all, reject and keep listening.
    if wake_score >= 0.995:
        norm_high = _normalize_wake_text(text)
        high_words = set(norm_high.split())
        high_distinctive = set(_distinctive_wake_words())
        if any(
            w in high_distinctive or
            (len(w) >= 2 and any(
                _wake_word_confidence(w, d)[0] >= WAKE_VERIFY_MIN_RATIO
                for d in high_distinctive
            ))
            for w in high_words
        ):
            logger.info(
                "[WAKE-VERIFY] ACCEPTED (high-confidence + transcript "
                "evidence): wake_score=%.3f ≥ 0.995 transcript='%s'",
                wake_score, text.strip())
            return True
        logger.info(
            "[WAKE-VERIFY] REJECTED (high-confidence but transcript "
            "disagrees): wake_score=%.3f ≥ 0.995 transcript='%s' — "
            "no wake-word evidence",
            wake_score, text.strip())
        return False

    norm = _normalize_wake_text(text)
    if not norm:
        return False
    norm_words = norm.split()
    norm_word_set = set(norm_words)

    # ── (a) Full-variant containment (whole-word) ──
    for variant in DEFAULT_WAKE_VARIANTS:
        v = _normalize_wake_text(variant)
        if not v:
            continue
        v_words = v.split()
        if all(w in norm_word_set for w in v_words):
            logger.debug("[WAKE-VERIFY] ACCEPTED (containment): "
                         "variant='%s' text='%s'", v, norm)
            return True

    # ── (b) Distinctive-word confidence (phonetic + fuzzy) ──
    # At least one transcript word must confidently match a distinctive
    # wake word. This prevents generic-phrase false accepts like
    # "hello video" that happen to ratio-match a full variant.
    distinctive = _distinctive_wake_words()
    best_conf = 0.0
    best_pair = ("", "")
    best_evidence = ""
    for tword in norm_words:
        if tword in _GENERIC_WAKE_WORDS:
            continue
        for dword in distinctive:
            conf, evidence = _wake_word_confidence(tword, dword)
            if conf > best_conf:
                best_conf = conf
                best_pair = (tword, dword)
                best_evidence = evidence
            if conf >= WAKE_VERIFY_MIN_RATIO:
                logger.info(
                    "[WAKE-VERIFY] ACCEPTED (%s): '%s' ≈ '%s' "
                    "confidence=%.3f ≥ %.2f text='%s'",
                    evidence, tword, dword, conf,
                    WAKE_VERIFY_MIN_RATIO, norm)
                return True

    logger.info("[WAKE-VERIFY] REJECTED: text='%s' best='%s'≈'%s' "
                "confidence=%.3f (< %.2f) evidence=%s",
                norm, best_pair[0], best_pair[1], best_conf,
                WAKE_VERIFY_MIN_RATIO, best_evidence or "none")
    return False
