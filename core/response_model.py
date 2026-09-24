"""
Single Response object for Diego's reply pipeline (2026-09-24 refactor).

One ``Response`` drives BOTH the UI text and the TTS speech for a turn:

- ``text``         → full on-screen reply (may contain ``ACTION:`` control
                      lines consumed by the dispatcher/UI);
- ``speak_text``   → TTS-safe speech string (defaults to ``text``; control
                      responses can override it, e.g. the goal line only);
- ``response_mode`` / ``source`` / ``grounded`` / ``evidence`` /
  ``metadata``     → observability: how the reply was produced (guard,
                      RAG verdict, RAG-evidence-grounded generation,
                      freeform), the relevance-ranked evidence behind it,
                      the cheapest-path route and per-stage timings.

This module also hosts the shared response-policy helpers so Brain, the
streaming LLM and the guarantee layer stay decoupled:

- ``classify_response_length`` / ``response_length_budget`` /
  ``response_length_instruction`` → dynamic response length (short/medium/
  long) derived from transcript complexity;
- ``format_evidence_context``     → relevance-driven context assembly
  (score-ranked, count- and char-bounded evidence block for the LLM);
- ``route_knowledge_path``        → cheapest-path routing decision table.

No production imports live here — the module must stay importable from
anywhere (including tests) without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ═══════════════════════════════════════════════════════════════════
# The single Response object
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Response:
    """One turn's reply — the single source for UI text and TTS speech."""

    text: str = ""
    speak_text: str = ""
    response_mode: str = "direct"      # direct | clarification | action | ...
    source: str = "pipeline"           # pipeline | local_quick_answer | rag | llm | error
    grounded: bool = False             # True when local/RAG evidence backed it
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TTS always has something to say; ``text`` remains the full
        # (possibly control-line-bearing) on-screen reply.
        if not self.speak_text:
            self.speak_text = self.text


# ═══════════════════════════════════════════════════════════════════
# Dynamic response length
# ═══════════════════════════════════════════════════════════════════

# Markers of questions that deserve a thorough (not clipped) answer.
_LONG_MARKERS: tuple = (
    "explain", "describe", "why ", "why?", "how does", "how do ",
    "how can", "walk me through", "step by step", "difference between",
    "compare", "pros and cons", "summarize", "summary", "overview",
    "in detail", "reasons", "story",
)

_LENGTH_BUDGETS: Dict[str, int] = {
    "short": 320,     # ~1-2 spoken sentences
    "medium": 720,    # ~2-4 spoken sentences
    "long": 1600,     # a few short paragraphs
}

_LENGTH_INSTRUCTIONS: Dict[str, str] = {
    "short": "Keep the reply to one or two short sentences.",
    "medium": "Keep the reply to about two to four sentences.",
    "long": ("Give a complete, well-structured reply in a few short "
             "paragraphs — cover the key details without padding."),
}


def classify_response_length(text: str) -> str:
    """Profile a transcript → ``short`` | ``medium`` | ``long``.

    Deterministic and cheap (word count + explain-style markers); used to
    pick the response budget/instruction so voice replies are dynamic
    instead of a fixed canned length.
    """
    t = (text or "").strip().lower()
    if not t:
        return "short"
    wc = len(t.split())
    if wc <= 10:
        return "short"
    if any(marker in t for marker in _LONG_MARKERS):
        return "long"
    if wc >= 45:
        return "long"
    if wc <= 22:
        return "short"
    return "medium"


def response_length_budget(profile: str) -> int:
    """Character budget for the spoken reply of a profile."""
    return _LENGTH_BUDGETS.get(profile, _LENGTH_BUDGETS["medium"])


def response_length_instruction(profile: str) -> str:
    """Prompt instruction for the generation LLM (``""`` for unknown)."""
    return _LENGTH_INSTRUCTIONS.get(profile, "")


# ═══════════════════════════════════════════════════════════════════
# Relevance-driven context assembly
# ═══════════════════════════════════════════════════════════════════

def _evidence_score(rec: Mapping[str, Any]) -> float:
    """Best available relevance score on an evidence record."""
    for key in ("final_score", "score", "relevance", "semantic_score"):
        try:
            val = rec.get(key)
            if val is not None:
                return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def format_evidence_context(
        evidence: Iterable[Mapping[str, Any]],
        *,
        top_k: int = 5,
        max_chars: int = 4000,
        max_block_chars: int = 1200,
) -> str:
    """Score-ranked, bounded ``[RETRIEVED EVIDENCE]`` block for the LLM.

    Reuses the orchestrator's evidence labels (``evidence_id | source``
    + line range) so the generation prompt reads like ``compose_prompt``.
    Returns ``""`` when nothing usable remains after filtering.
    """
    if not evidence:
        return ""
    ranked = sorted(evidence, key=_evidence_score, reverse=True)
    blocks: List[str] = []
    used = 0
    for rec in ranked[: max(0, top_k)]:
        if not isinstance(rec, Mapping):
            continue
        body = str(rec.get("content") or rec.get("text") or "").strip()
        if not body:
            continue
        loc = ""
        if rec.get("line_start") is not None:
            loc = f":{rec.get('line_start')}-{rec.get('line_end')}"
        source = (rec.get("source") or rec.get("doc_path")
                  or rec.get("filename") or "document")
        eid = rec.get("evidence_id") or rec.get("filename") or "ev"
        block = f"--- {eid} | {source}{loc}\n{body[:max_block_chars]}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    if not blocks:
        return ""
    return "[RETRIEVED EVIDENCE]\n" + "\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════
# Cheapest-path routing decision table
# ═══════════════════════════════════════════════════════════════════

def route_knowledge_path(
        *,
        guard_hit: bool = False,
        rag_verdict: str = "",
        evidence_count: int = 0,
) -> str:
    """Decide (and name) the cheapest path that can answer the turn.

    - ``guard``            → deterministic shortcut answered it (0 LLM calls);
    - ``rag_verdict``      → CLARIFICATION / ACTION from the structured RAG
                              call (1 LLM call, no generation call);
    - ``rag_evidence_gen`` → RAG evidence feeds the generation LLM
                              (structured call + grounded generation);
    - ``freeform_llm``     → plain grounded-free generation (1 LLM call).

    Returned on ``Response.metadata["route"]`` for observability.
    """
    if guard_hit:
        return "guard"
    if rag_verdict in ("CLARIFICATION", "ACTION"):
        return "rag_verdict"
    if evidence_count > 0:
        return "rag_evidence_gen"
    return "freeform_llm"
