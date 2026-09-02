"""
Presentation layer for local knowledge answers — the ONLY boundary
between INTERNAL retrieval results and what Diego is allowed to speak.

Retrieval results (knowledge.retriever) are INTERNAL evidence. They
carry absolute paths, chunk indexes, embedding scores, and raw chunk
text. NONE of that may reach the user's ears verbatim.

This module provides:
  * friendly_source()             — short spoken citation ("README.md",
                                    "report.pdf, page 13") — never a
                                    directory path.
  * dedupe_evidence()             — collapse overlapping chunks into the
                                    smallest useful evidence set.
  * synthesize_local_answer()     — concise deterministic answer for
                                    strong local matches (no LLM).
  * is_explicit_listing_request() — detect explicit file-list / path
                                    requests ("list the files", "what
                                    files are in this folder?", "what
                                    is the path to X?"). Only those may
                                    hear filenames/paths enumerated.
  * sanitize_spoken()             — final spoken-response guard: scrubs
                                    leaked absolute paths, retrieval
                                    metadata, and bounds the response
                                    length for voice UX.

Rules enforced here (voice UX contract):
  1. Retrieval lists, matched filenames, directory paths, chunk
     metadata, embedding scores, database rows, and filesystem scan
     output are NEVER spoken for normal knowledge questions.
  2. A source citation is exposed only when useful, and only as a
     short file name (+ page/sheet/section locator).
  3. Directory contents are never enumerated unless the user
     explicitly asked for a file list or a path.
"""

from __future__ import annotations

import re
from typing import Dict, List

# ── Voice UX limits ────────────────────────────────────────────────

# Hard ceiling for a spoken knowledge answer (characters). Local
# knowledge must never produce a multi-paragraph spoken response.
MAX_SPOKEN_CHARS = 500

# Evidence snippet ceiling inside a synthesized answer.
MAX_EVIDENCE_CHARS = 300

# Maximum distinct evidence blocks kept after dedupe.
MAX_EVIDENCE_BLOCKS = 3

# ── Explicit listing / path request detection ─────────────────────

# The user explicitly asked to ENUMERATE files/directories. Only then
# may the response list file names or directory contents.
_LISTING_PHRASES = (
    "list the files", "list files", "list all files", "list my files",
    "list the folder", "list folder", "list the directory",
    "list directory", "directory listing",
    "show files", "show me the files", "show the files",
    "show me what files", "show all files",
    "what files", "which files", "how many files",
    "files in this folder", "files in the folder", "files in my",
    "files in downloads", "files in documents", "files in desktop",
    "what's in this folder", "whats in this folder",
    "what is in this folder", "what's in the folder",
    "contents of this folder", "contents of the folder",
    "enumerate the files", "enumerate files",
)

# The user explicitly asked for a PATH / location. Only then may the
# response contain full paths.
_PATH_PHRASES = (
    "full path", "the path to", "path to the", "path of",
    "what's the path", "whats the path", "what is the path",
    "where is the file", "where's the file", "where is the document",
    "where's the document", "where is my file", "where's my file",
    "where did i save", "where did i put", "where does it live",
    "which folder is", "which directory is",
    "locate the file", "locate the document", "locate my",
    "file location", "file's location", "folder location",
)


def is_explicit_listing_request(text: str) -> bool:
    """True when the user explicitly asked for a file list, directory
    enumeration, or a file path. Only such requests may hear file
    names/paths enumerated — normal knowledge questions never do."""
    t = " ".join((text or "").lower().split())
    if not t:
        return False
    return (any(p in t for p in _LISTING_PHRASES)
            or any(p in t for p in _PATH_PHRASES))


# ── Friendly citations ─────────────────────────────────────────────

def friendly_source(result: Dict) -> str:
    """Short spoken citation for a retrieval result.

    Returns e.g. "README.md" or "report.pdf, page 13" — NEVER a
    directory path. Falls back to the basename of doc_path when the
    filename field is missing. Returns "" when no usable name exists.
    """
    filename = str(result.get("filename") or "").strip()
    doc_path = str(result.get("doc_path") or "").strip()
    if not filename and doc_path:
        filename = doc_path.rstrip("/").rsplit("/", 1)[-1]
    # Virtual documents (diego://pc-snapshot) → readable name
    if filename.startswith("diego://"):
        filename = filename.split("diego://", 1)[-1] or ""
    locator = str(result.get("locator") or "").strip()
    if filename and locator:
        return f"{filename}, {locator}"
    return filename


# ── Evidence dedupe / selection ────────────────────────────────────

def _norm_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard-ish overlap between two normalized texts (0..1)."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = min(len(ta), len(tb))  # containment-biased
    return inter / denom if denom else 0.0


def dedupe_evidence(results: List[Dict],
                    max_blocks: int = MAX_EVIDENCE_BLOCKS) -> List[Dict]:
    """Collapse overlapping chunks into the smallest useful evidence.

    Chunks from the same document whose text is contained in (or
    largely overlaps) an already-kept chunk are dropped. Keeps at most
    `max_blocks` distinct evidence blocks, best score first (input is
    expected pre-ranked by the retriever).
    """
    kept: List[Dict] = []
    kept_norms: List[str] = []
    for r in results or []:
        norm = _norm_text(r.get("text", ""))
        if not norm:
            continue
        duplicate = False
        for i, k in enumerate(kept):
            if k.get("doc_path") != r.get("doc_path"):
                continue
            prev = kept_norms[i]
            if norm in prev or prev in norm:
                duplicate = True
                break
            if _token_overlap_ratio(norm, prev) >= 0.7:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(r)
        kept_norms.append(norm)
        if len(kept) >= max_blocks:
            break
    return kept


# ── Deterministic synthesis (strong local match — no LLM) ─────────

def _clean_evidence(text: str, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    """Collapse whitespace and bound the evidence snippet for speech."""
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    # Prefer ending on a sentence boundary.
    for punct in (". ", "! ", "? "):
        idx = cut.rfind(punct)
        if idx > max_chars // 2:
            return cut[: idx + 1].strip()
    # Otherwise end on a word boundary.
    return cut[: cut.rfind(" ")].rstrip() if " " in cut else cut


def synthesize_local_answer(query: str,
                            results: List[Dict],
                            max_evidence_chars: int = MAX_EVIDENCE_CHARS
                            ) -> str:
    """Concise spoken answer for a STRONG local match.

    Answers directly from the local evidence with a friendly citation.
    Never exposes raw paths, retrieval lists, scores, or metadata.
    Returns "" when there is no usable evidence.
    """
    evidence = dedupe_evidence(results)
    if not evidence:
        return ""
    top = evidence[0]
    snippet = _clean_evidence(top.get("text", ""), max_evidence_chars)
    cite = friendly_source(top)
    if not snippet:
        return f"I found a note in {cite}." if cite else ""
    if cite:
        return f"According to {cite} — {snippet}"
    return snippet


# ── Spoken-response guard ──────────────────────────────────────────

# Absolute filesystem paths (home/tmp/etc. rooted, or ~/...). URLs are
# excluded via the negative look-behind for ":" and the "//" check.
_PATH_RE = re.compile(
    r"(?<![:\w])(?:~|/home|/root|/tmp|/var|/etc|/opt|/usr|/mnt|/media|"
    r"/srv|/data)(?:/[\w.\-@]+)+")

# Retrieval-metadata leaks (score=0.95, chunk_index: 3, ...).
_META_RE = re.compile(
    r"\b(?:score|chunk_index|chunk index|embedding|doc_path|text_hash|"
    r"indexed_at|file_type|embedding_model)\s*[:=]\s*\S+",
    re.IGNORECASE)

# Bracketed citation blocks like "[/home/x/doc.txt (page 2)]" that an
# LLM might copy verbatim from its grounding context.
_CITATION_BLOCK_RE = re.compile(r"\[(?:[^\]\[]*/[\w.\-]+[^\]\[]*)\]")


def _scrub_path(match: "re.Match") -> str:
    """Replace a leaked absolute path with its short file name."""
    path = match.group(0)
    base = path.rstrip("/").rsplit("/", 1)[-1]
    return base if base else "a file"


def sanitize_spoken(response: str,
                    allow_paths: bool = False,
                    max_chars: int = MAX_SPOKEN_CHARS) -> str:
    """Final guard before a knowledge-influenced response is spoken.

    * Scrubs leaked absolute paths (replaces with the short file name)
      unless the user explicitly asked for file lists / paths.
    * Removes retrieval metadata (scores, chunk indexes, ...).
    * Bounds the response length for voice UX (sentence-boundary cut).

    Never applied to ACTION lines or non-knowledge responses.
    """
    text = (response or "").strip()
    if not text:
        return text

    # Metadata is NEVER spoken, regardless of the request.
    text = _META_RE.sub("", text)

    if not allow_paths:
        text = _CITATION_BLOCK_RE.sub("", text)
        text = _PATH_RE.sub(_scrub_path, text)

    # Collapse whitespace left behind by scrubbing.
    text = " ".join(text.split()).strip(" -–—:|")

    # Response-size guard for voice UX.
    if len(text) > max_chars:
        cut = text[:max_chars]
        boundary = -1
        for punct in (". ", "! ", "? "):
            boundary = max(boundary, cut.rfind(punct))
        if boundary > max_chars // 2:
            text = cut[: boundary + 1].strip()
        else:
            text = cut[: cut.rfind(" ")].rstrip() if " " in cut else cut
    return text