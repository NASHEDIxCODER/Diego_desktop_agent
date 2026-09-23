"""
RetrievedEvidence — provenance-carrying evidence value object (RAG phase).

The retriever keeps its existing dict-shaped results (the presentation
layer, brain's LOCAL KNOWLEDGE path, and the UX tests all consume those
keys). This module adds, and standardizes, CHUNK-level provenance:

    evidence_id      stable id for one retrieved chunk (attribution)
    chunk_id         doc_path#chunk_index
    source_type      md / py / pdf / ...
    line_start/end   parsed from the extractor locator ("lines 10-40")
    heading/symbol   markdown heading or def/class found in the chunk
    semantic_score   embedding cosine component
    lexical_score    keyword-overlap component
    relevance        final fused ranking score

A chunk is a MEANINGFUL CONTENT BLOCK (extractor section / line window),
never a single matching line. Lines exist for provenance only.

Pure / side-effect free — unit-testable without DuckDB or embeddings.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ── Locator parsing ────────────────────────────────────────────────
# Extractor locators look like: "lines 10-40", "lines 5", "page 3",
# "sheet Budget", "rows 1-40", "slide 2", a markdown/code heading title,
# or "" (whole file).

_LINE_RANGE_RE = re.compile(r"lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?", re.I)
_PAGE_RE = re.compile(r"^page\s+(\d+)$", re.I)
_ROWS_RE = re.compile(r"^rows?\s+(\d+)", re.I)
_SLIDE_RE = re.compile(r"^slide\s+(\d+)$", re.I)

# Technical locators that are NOT human headings.
_KIND_LOCATORS = {"xml", "json", "html", "preamble", "section"}

# First def/class (or JS function/const) near the top of a chunk.
_SYMBOL_RE = re.compile(
    r"^[ \t]*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)"
    r"|^[ \t]*(?:export\s+)?(?:function|const|let|var)\s+([A-Za-z_]\w*)\s*[=(]",
    re.M,
)


def parse_locator(locator: str) -> Dict[str, Any]:
    """Parse an extractor locator into structured provenance fields.

    Returns keys: line_start, line_end, page, heading.
    Unknown/empty locators yield Nones/"" — never a guess.
    """
    loc = (locator or "").strip()
    out: Dict[str, Any] = {
        "line_start": None, "line_end": None,
        "page": None, "heading": "",
    }
    if not loc:
        return out
    m = _LINE_RANGE_RE.search(loc)
    if m:
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        out["line_start"], out["line_end"] = start, end
        return out
    for rx, key in ((_PAGE_RE, "page"), (_ROWS_RE, "page"),
                    (_SLIDE_RE, "page")):
        m = rx.match(loc)
        if m:
            out[key] = int(m.group(1))
            return out
    low = loc.lower()
    if low not in _KIND_LOCATORS and not low.startswith("sheet json"):
        if low.startswith("sheet "):
            out["heading"] = loc[6:].strip()
        else:
            out["heading"] = loc
    return out


def first_symbol(text: str, max_chars: int = 1500) -> str:
    """First def/class/function name found near the top of a chunk."""
    m = _SYMBOL_RE.search((text or "")[:max_chars])
    if not m:
        return ""
    return m.group(1) or m.group(2) or ""


def source_type_of(chunk: Dict[str, Any]) -> str:
    """File type for a chunk: explicit file_type, else extension."""
    ft = str(chunk.get("file_type") or "").strip().lower().lstrip(".")
    if ft:
        return ft
    doc = str(chunk.get("doc_path") or "")
    if "." in doc.rsplit("/", 1)[-1]:
        return doc.rsplit(".", 1)[-1].lower()
    return ""


def make_evidence_fields(chunk: Dict[str, Any], *, semantic: float,
                         lexical: float, relevance: float,
                         source: str) -> Dict[str, Any]:
    """Standard provenance fields added to every retriever result.

    evidence_id is deterministic (same chunk content + locator → same
    id) so attribution is stable across calls, and unique per chunk.
    """
    doc = str(chunk.get("doc_path") or "")
    idx = int(chunk.get("chunk_index") or 0)
    loc = str(chunk.get("locator") or "")
    text = str(chunk.get("text") or "")
    chunk_id = f"{doc}#{idx}"
    seed = f"{chunk_id}|{loc}|{len(text)}|{text[:160]}"
    evidence_id = "ev_" + hashlib.sha1(
        seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    loc_info = parse_locator(loc)
    symbol = first_symbol(text)
    return {
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "source_type": source_type_of(chunk),
        "semantic_score": round(float(semantic), 4),
        "lexical_score": round(float(lexical), 4),
        "relevance_score": round(float(relevance), 4),
        "line_start": loc_info["line_start"],
        "line_end": loc_info["line_end"],
        "heading": loc_info["heading"],
        "symbol": symbol,
        "provenance": provenance_text(doc, loc_info, symbol),
    }


def provenance_text(doc_path: str, loc_info: Dict[str, Any],
                    symbol: str = "") -> str:
    """Compact human/LLM citation: 'voice/audio_backend.py:42-117'."""
    if not doc_path:
        return ""
    base = doc_path
    if loc_info.get("line_start") is not None:
        end = loc_info["line_end"]
        base += (f":{loc_info['line_start']}"
                 if end == loc_info["line_start"]
                 else f":{loc_info['line_start']}-{end}")
    elif loc_info.get("page"):
        base += f":p{loc_info['page']}"
    elif loc_info.get("heading"):
        base += f" ({loc_info['heading']})"
    elif symbol:
        base += f"#{symbol}"
    return base


# ── The value object (spec §3) ─────────────────────────────────────

@dataclass
class RetrievedEvidence:
    """One semantic chunk handed to the reasoning layer.

    content is the whole meaningful block; line_start/line_end (or
    heading/symbol) exist for PROVENANCE, never as the content itself.
    """

    source: str
    content: str = ""
    evidence_id: str = ""
    chunk_id: str = ""
    source_type: str = ""
    filename: str = ""
    locator: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    heading: str = ""
    symbol: str = ""
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    relevance: float = 0.0
    provenance: str = ""

    @classmethod
    def from_result(cls, r: Dict[str, Any]) -> "RetrievedEvidence":
        """Build from a retriever result dict (enriched or legacy)."""
        doc = str(r.get("doc_path") or r.get("source") or "")
        loc_info = parse_locator(str(r.get("locator") or ""))
        symbol = str(r.get("symbol") or "") or first_symbol(
            str(r.get("text") or ""))
        provenance = str(r.get("provenance") or "") or provenance_text(
            doc, loc_info, symbol)
        return cls(
            source=doc,
            content=str(r.get("text") or r.get("content") or ""),
            evidence_id=str(r.get("evidence_id") or ""),
            chunk_id=str(r.get("chunk_id") or f"{doc}#{r.get('chunk_index', 0)}"),
            source_type=str(r.get("source_type") or ""),
            filename=str(r.get("filename") or ""),
            locator=str(r.get("locator") or ""),
            line_start=r.get("line_start", loc_info["line_start"]),
            line_end=r.get("line_end", loc_info["line_end"]),
            heading=str(r.get("heading") or loc_info["heading"] or ""),
            symbol=symbol,
            semantic_score=float(r.get("semantic_score") or 0.0),
            lexical_score=float(r.get("lexical_score") or 0.0),
            relevance=float(r.get("relevance_score") or r.get("score") or 0.0),
            provenance=provenance,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "source_type": self.source_type,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "heading": self.heading,
            "symbol": self.symbol,
            "locator": self.locator,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "relevance": self.relevance,
            "provenance": self.provenance,
        }
