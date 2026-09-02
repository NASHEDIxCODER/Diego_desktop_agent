"""
Deterministic chunking for the local knowledge index.

Same input text + same settings → same chunks, every time. Chunks carry
the document path, filename, and section locator so retrieval results
include precise citations (file path + page/sheet/section).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from config.settings import settings

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A deterministic chunk of extracted content."""
    text: str
    locator: str        # page/sheet/section locator from the extractor
    index: int          # chunk index within the document


def chunk_text(text: str,
               chunk_size: int = None,
               overlap: int = None) -> List[Chunk]:
    """Split text into overlapping, sentence-aware chunks.

    Deterministic: no randomness, stable ordering. Empty/whitespace
    input yields no chunks.
    """
    chunk_size = chunk_size or settings.KNOWLEDGE_CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.KNOWLEDGE_CHUNK_OVERLAP
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(text=text, locator="", index=0)]

    # Sentence-aware split, then pack sentences into chunks.
    sentences: List[str] = []
    for s in _SENTENCE_SPLIT.split(text):
        s = s.strip()
        if not s:
            continue
        if len(s) > chunk_size:
            # Very long "sentence" (e.g. minified code): hard-split.
            for i in range(0, len(s), chunk_size - overlap):
                sentences.append(s[i:i + chunk_size])
        else:
            sentences.append(s)

    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_len = 0
    for s in sentences:
        if buf and buf_len + len(s) + 1 > chunk_size:
            chunks.append("\n".join(buf))
            # Keep the tail for overlap
            tail: List[str] = []
            tail_len = 0
            for prev in reversed(buf):
                if tail_len + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            buf = tail
            buf_len = tail_len
        buf.append(s)
        buf_len += len(s) + 1
    if buf:
        chunks.append("\n".join(buf))

    return [Chunk(text=c, locator="", index=i)
            for i, c in enumerate(chunks)]


def chunk_sections(sections) -> List[Chunk]:
    """Chunk extracted sections, preserving the section locator."""
    out: List[Chunk] = []
    for sec in sections:
        for c in chunk_text(sec.text):
            c.locator = sec.locator
            out.append(c)
    # Renumber globally for stable ordering
    for i, c in enumerate(out):
        c.index = i
    return out