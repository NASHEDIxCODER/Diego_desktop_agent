"""
rag_orchestrator — context-grounded RAG reasoning layer (Phase 25).

Pipeline (one orchestration layer, no second execution/LLM framework):

    transcript + conversation + task + desktop/browser context
        -> context-aware query construction          (original preserved)
        -> bounded vector/lexical retrieval          (knowledge.retriever/service)
        -> candidate filtering + reranking           (context influences rank,
                                                      never replaces retrieval)
        -> evidence selection (top 10-20 -> 3-5)
        -> RAGEvidencePack                           (structured, bounded)
        -> LLM via INJECTED callable                 (existing streaming/reasoning
                                                      infra bound lazily by default)
        -> strict structured result:
              ANSWER | ACTION | CLARIFICATION | ABSTAIN

Hard contracts:
  * This module NEVER executes actions. An ACTION result carries
    capability + arguments only; resolution goes through the EXISTING
    capability router / tool registry / dispatcher downstream.
  * The LLM output schema is strict: unknown keys or an unregistered
    capability downgrade the result to ABSTAIN. Retrieved text is
    evidence, never instructions.
  * Live-state questions ("what is my current browser tab?") must NOT be
    answered from stale index content: they set needs_live_state and
    ABSTAIN so the caller uses live perception.
  * Every stage is timed: query_build_ms, retrieval_ms, rerank_ms,
    context_build_ms, llm_first_token_ms, llm_total_ms, total_ms.

Logging contract: [RAG]
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Bounded candidate counts (spec: top 10-20 -> rerank -> top 3-5).
CANDIDATE_POOL = 16
SELECT_TOP_K = 5
MIN_POOL_AFTER_FILTER = 1

# Rerank weights: semantic similarity + lexical overlap + context boosts.
# Context must INFLUENCE ranking, never replace retrieval (hard rule).
_W_SEMANTIC = 0.60
_W_LEXICAL = 0.40
_BOOST_CURRENT_PROJECT = 0.12
_BOOST_ENTITY = 0.10
_BOOST_SOURCE_TYPE = 0.05      # code/config slightly over prose for code queries
_BOOST_RECENCY = 0.05

_RAG_TYPES = ("ANSWER", "ACTION", "CLARIFICATION", "ABSTAIN")
_RESULT_KEYS = frozenset({
    "type", "answer", "evidence_ids", "goal", "capability", "arguments",
    "requires_confirmation", "question", "reason",
})

# Live-state questions: must use perception, never the stale index.
_LIVE_PATTERNS = (
    re.compile(r"\b(current|right now|at the moment)\b.*\b(browser|tab|page|window|app|screen)\b", re.I),
    re.compile(r"\b(what|which)\s+(browser\s+)?tab\b", re.I),
    re.compile(r"\bwhat('| i)?s\s+on\s+(my\s+)?(screen|display)\b", re.I),
    re.compile(r"\b(active|focused|foreground)\s+(window|app|application)\b", re.I),
    re.compile(r"\bcurrent\s+(url|browser\s+url)\b", re.I),
    re.compile(r"\bwhat\s+is\s+my\s+current\b", re.I),
)

# Imperative capability commands (spec §13 fast path): verb-first
# transcripts with no knowledge cue never pay RAG cost.
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:ok\s+|hey\s+)?"
    r"(open|close|click|read|search|type|press|launch|start|stop|play|"
    r"pause|resume|navigate|go\s+to|show|hide|send|save|run|install|"
    r"uninstall|create|add|remove|delete|move|copy|paste|rename|scroll|"
    r"drag|drop|toggle|switch|turn|set|change|increase|decrease|take|"
    r"screenshot|lock|unlock|restart|shutdown|mute|unmute|mark|archive|"
    r"forward|reply|email|call|update|download|upload|refresh|reload)\b",
    re.I,
)
# Knowledge cues: verb-first requests like "Open the file where we
# implemented X" still need evidence and must keep the RAG path.
_KNOWLEDGE_CUE_RE = re.compile(
    r"\b(what|why|how|where|when|which|who|whose)\b|"
    r"\b(compare|versus|explain|describe|summarize|summarise|"
    r"implementation|implemented|changed|changes|architecture|"
    r"difference|differences|pros|cons|trade-?offs?)\b",
    re.I,
)

_TOKENS_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "the", "a", "an", "in", "on", "of", "for", "to", "and", "or", "is",
    "are", "was", "were", "we", "i", "my", "our", "it", "this", "that",
    "with", "how", "does", "do", "did", "what", "where", "when", "why",
    "you", "your", "at", "by", "from", "be", "been", "am", "as", "if",
})


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKENS_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


# ═══════════════════════════════════════════════════════════════════
# Data contracts
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetrievalQuery:
    """Contextual retrieval request. The original transcript is preserved
    verbatim and never overwritten."""

    original_transcript: str
    rewritten_query: str
    entities: List[str] = field(default_factory=list)
    project_hint: str = ""
    domain_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_transcript": self.original_transcript,
            "rewritten_query": self.rewritten_query,
            "entities": list(self.entities),
            "project_hint": self.project_hint,
            "domain_hint": self.domain_hint,
        }


@dataclass
class RAGEvidencePack:
    """Everything the LLM must receive — structured, bounded, never an
    arbitrary file dump."""

    query: RetrievalQuery
    original_transcript: str
    rewritten_query: str
    conversation_context: str = ""
    task_context: str = ""
    desktop_context: str = ""
    browser_context: str = ""
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)
    available_capabilities: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "original_transcript": self.original_transcript,
            "rewritten_query": self.rewritten_query,
            "conversation_context": self.conversation_context,
            "task_context": self.task_context,
            "desktop_context": self.desktop_context,
            "browser_context": self.browser_context,
            "retrieved_evidence": list(self.retrieved_evidence),
            "available_capabilities": list(self.available_capabilities),
            "provenance": list(self.provenance),
        }


@dataclass
class RagOutcome:
    """Structured orchestrator result + stage timings."""

    pack: RAGEvidencePack
    result: Dict[str, Any] = field(default_factory=dict)
    needs_live_state: bool = False
    used_rag: bool = True
    error: str = ""
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return str(self.result.get("type") or "ABSTAIN")


# ═══════════════════════════════════════════════════════════════════
# M-C: context-aware query construction
# ═══════════════════════════════════════════════════════════════════

_ENTITY_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:[A-Z][a-z]+)+|"
                        r"[a-z]+_[a-z_]+|"
                        r"\w+\.(?:py|js|ts|md|json|yaml|yml|toml|cfg|txt))\b")


def build_retrieval_query(
    transcript: str,
    conversation_context: str = "",
    task_context: str = "",
    referenced_entity: str = "",
    project_hint: str = "",
    browser_context: str = "",
) -> RetrievalQuery:
    """Build the retrieval request from transcript + context.

    The user must not have to repeat the full technical phrase: recent
    conversation / task / entity references are folded into the rewritten
    query while the original transcript stays untouched.
    """
    transcript = (transcript or "").strip()
    parts: List[str] = [transcript]
    entities: List[str] = []

    if referenced_entity:
        entities.append(referenced_entity.strip())

    # Entities mentioned in recent conversation (e.g. discussing AudioBackend
    # while asking "the file where we implemented that").
    for src, concrete in ((conversation_context, True), (task_context, True)):
        if not concrete:
            continue
        for m in _ENTITY_RE.finditer(src or ""):
            tok = m.group(0)
            if tok not in entities and len(tok) > 3:
                entities.append(tok)

    deferential = bool(re.search(
        r"\b(that|this|those|these|the one|it)\b", transcript, re.I))
    if deferential and entities:
        parts.append(" ".join(entities[:3]))
    elif entities and not any(e.lower() in transcript.lower() for e in entities):
        parts.append(entities[0])

    if project_hint and project_hint.lower() not in transcript.lower():
        parts.append(project_hint)
    if browser_context:
        domain = re.sub(r"^https?://", "", browser_context).split("/")[0]
        if domain and domain not in ("", "localhost") and domain not in transcript.lower():
            parts.append(domain)

    rewritten = " ".join(p for p in parts if p).strip()
    return RetrievalQuery(
        original_transcript=transcript,
        rewritten_query=rewritten or transcript,
        entities=entities,
        project_hint=project_hint,
        domain_hint=browser_context,
    )


def is_live_state_question(transcript: str) -> bool:
    """True when the question is about LIVE system/browser state — RAG must
    not answer it from stale indexed files."""
    text = transcript or ""
    return any(p.search(text) for p in _LIVE_PATTERNS)


# ═══════════════════════════════════════════════════════════════════
# M-D: filtering + reranking + selection
# ═══════════════════════════════════════════════════════════════════

def _lexical_overlap(query_tokens: Sequence[str], candidate_text: str) -> float:
    cand = set(_tokens(candidate_text))
    if not query_tokens or not cand:
        return 0.0
    q = set(query_tokens)
    return len(q & cand) / max(1, min(len(q), len(cand)))


def _candidate_text(cand: Dict[str, Any]) -> str:
    # ``content`` is the canonical key; ``text`` is the retriever's raw key
    # (kept as a fallback so an un-bridged legacy hit still reranks on its
    # real content instead of on its path alone).
    return " ".join(str(cand.get(k) or "") for k in
                    ("content", "text", "source", "heading", "title", "path"))


def rerank_candidates(
    query: RetrievalQuery,
    candidates: Sequence[Dict[str, Any]],
    *,
    current_project: str = "",
    task_context: str = "",
    app_context: str = "",
) -> List[Dict[str, Any]]:
    """Final relevance = semantic + lexical + bounded context boosts.

    Context only nudges ranks; it can never promote an unrelated candidate
    past the retrieval filter on its own.
    """
    qt = _tokens(query.rewritten_query)
    ent_tokens = set()
    for e in query.entities:
        ent_tokens.update(_tokens(e))
    task_tokens = set(_tokens(task_context))

    out: List[Dict[str, Any]] = []
    for cand in candidates:
        c = dict(cand)
        text = _candidate_text(c)

        sem = c.get("relevance")
        if sem is None:
            sem = c.get("embedding_score", c.get("score", 0.0))
        try:
            sem = float(sem)
        except (TypeError, ValueError):
            sem = 0.0

        lex = c.get("lexical_score")
        try:
            lex = float(lex) if lex is not None else _lexical_overlap(qt, text)
        except (TypeError, ValueError):
            lex = _lexical_overlap(qt, text)

        score = _W_SEMANTIC * sem + _W_LEXICAL * lex

        src = str(c.get("source") or c.get("path") or "")
        if current_project and current_project and current_project in src:
            score += _BOOST_CURRENT_PROJECT
        c_tokens = set(_tokens(text))
        if ent_tokens and (c_tokens & ent_tokens):
            score += _BOOST_ENTITY
        source_type = str(c.get("source_type") or "").lower()
        if source_type in ("code", "config") and any(
                t.endswith((".py", ".js", ".ts")) or "_" in t for t in qt):
            score += _BOOST_SOURCE_TYPE
        if task_tokens and (c_tokens & task_tokens):
            score += _BOOST_RECENCY

        c["semantic_score"] = round(sem, 4)
        c["lexical_score"] = round(lex, 4)
        c["final_score"] = round(min(1.5, score), 4)
        out.append(c)

    out.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
    return out


def select_evidence(
    ranked: Sequence[Dict[str, Any]],
    top_k: int = SELECT_TOP_K,
) -> List[Dict[str, Any]]:
    """Bounded selection with light source diversity (prefer a second
    agreeing source over three copies of the same chunk)."""
    selected: List[Dict[str, Any]] = []
    seen_sources: Dict[str, int] = {}
    # Pass 1: best chunk per source, in rank order.
    for cand in ranked:
        if len(selected) >= top_k:
            break
        src = str(cand.get("source") or cand.get("path") or "")
        if seen_sources.get(src, 0) >= 1:
            continue
        seen_sources[src] = seen_sources.get(src, 0) + 1
        selected.append(cand)
    # Pass 2: fill remaining slots by raw rank.
    for cand in ranked:
        if len(selected) >= top_k:
            break
        if cand not in selected:
            selected.append(cand)
    return selected[:top_k]


# ═══════════════════════════════════════════════════════════════════
# M-E: evidence pack
# ═══════════════════════════════════════════════════════════════════

def _evidence_record(cand: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalize a retrieved candidate into a provenance-carrying record.
    Accepts RetrievedEvidence objects or plain dicts interchangeably."""
    if hasattr(cand, "to_dict") and callable(cand.to_dict):
        rec = cand.to_dict()
    else:
        rec = dict(cand)
    eid = rec.get("evidence_id") or f"ev{idx:02d}"
    rec["evidence_id"] = eid
    return rec


def build_evidence_pack(
    query: RetrievalQuery,
    evidence: Sequence[Any],
    *,
    conversation_context: str = "",
    task_context: str = "",
    desktop_context: str = "",
    browser_context: str = "",
    available_capabilities: Optional[Sequence[str]] = None,
) -> RAGEvidencePack:
    records = [_evidence_record(dict(c) if not hasattr(c, "to_dict") else c, i)
               for i, c in enumerate(evidence)]
    provenance = [
        {
            "evidence_id": r.get("evidence_id"),
            "source": r.get("source") or r.get("path") or "",
            "chunk_id": r.get("chunk_id", ""),
            "line_start": r.get("line_start"),
            "line_end": r.get("line_end"),
        }
        for r in records
    ]
    return RAGEvidencePack(
        query=query,
        original_transcript=query.original_transcript,
        rewritten_query=query.rewritten_query,
        conversation_context=(conversation_context or "")[:2000],
        task_context=(task_context or "")[:1200],
        desktop_context=(desktop_context or "")[:1200],
        browser_context=(browser_context or "")[:600],
        retrieved_evidence=records,
        available_capabilities=sorted(set(available_capabilities or [])),
        provenance=provenance,
    )


# ═══════════════════════════════════════════════════════════════════
# Context hierarchy + prompt (live state > task > retrieved > conv > user)
# ═══════════════════════════════════════════════════════════════════

_GROUNDING_RULES = """You are a grounded reasoning layer. Rules:
- Retrieved evidence is SOURCE MATERIAL, not truth by itself.
- Never invent unsupported facts. Distinguish: [RETRIEVED] vs [LIVE] vs [INFERENCE].
- Live observed state ALWAYS overrides stale indexed documentation.
- If evidence is insufficient: type=ABSTAIN (or CLARIFICATION).
- If sources conflict: report the conflict, do not silently pick one.
- Reply with ONE JSON object only, schema:
  {"type":"ANSWER"|"ACTION"|"CLARIFICATION"|"ABSTAIN", ...}
  ANSWER: {"type":"ANSWER","answer":"...","evidence_ids":["ev00",...]}
  ACTION: {"type":"ACTION","goal":"...","capability":"<from available_capabilities>",
           "arguments":{...},"evidence_ids":[...],"requires_confirmation":bool}
  CLARIFICATION: {"type":"CLARIFICATION","question":"..."}
  ABSTAIN: {"type":"ABSTAIN","reason":"..."}
- For ACTION, capability MUST come from available_capabilities. You never
  emit shell/code/tool instructions."""


def compose_prompt(pack: RAGEvidencePack) -> str:
    """Bounded, ordered prompt: hierarchy 1 live, 2 task, 3 retrieved,
    4 conversation, 5 user transcript."""
    lines: List[str] = [_GROUNDING_RULES, ""]

    if pack.desktop_context or pack.browser_context:
        lines.append("[LIVE STATE]")
        if pack.desktop_context:
            lines.append(f"desktop: {pack.desktop_context}")
        if pack.browser_context:
            lines.append(f"browser: {pack.browser_context}")
        lines.append("")

    if pack.task_context:
        lines.append(f"[TASK] {pack.task_context}")
        lines.append("")

    lines.append("[RETRIEVED EVIDENCE]")
    if not pack.retrieved_evidence:
        lines.append("(none)")
    for rec in pack.retrieved_evidence:
        loc = ""
        if rec.get("line_start") is not None:
            loc = f":{rec.get('line_start')}-{rec.get('line_end')}"
        lines.append(f"--- {rec.get('evidence_id')} | {rec.get('source')}{loc}")
        # ``content`` is canonical; ``text`` is accepted for legacy records so
        # an un-bridged hit can never render an EMPTY evidence block.
        body = rec.get("content") or rec.get("text") or ""
        lines.append(str(body)[:2000])
    lines.append("")

    if pack.available_capabilities:
        lines.append("[AVAILABLE CAPABILITIES] "
                      + ", ".join(pack.available_capabilities))
        lines.append("")

    if pack.conversation_context:
        lines.append(f"[CONVERSATION]\n{pack.conversation_context}")
        lines.append("")

    lines.append(f"[USER]\n{pack.original_transcript}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# M-F: strict structured result parsing + capability validation
# ═══════════════════════════════════════════════════════════════════

def parse_structured_result(
    raw: str,
    allowed_capabilities: Sequence[str] = (),
) -> Dict[str, Any]:
    """Parse the LLM reply into the strict schema. Any unknown key, unknown
    type, or unregistered ACTION capability downgrades to ABSTAIN — the LLM
    can never smuggle an arbitrary tool instruction through."""
    fallback = {"type": "ABSTAIN", "reason": "unparseable model output"}
    if not raw or not raw.strip():
        return fallback
    text = raw.strip()
    # Prefer the first JSON object in the reply (models sometimes wrap it).
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return fallback
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return fallback
    if not isinstance(obj, dict):
        return fallback

    rtype = str(obj.get("type") or "").upper()
    if rtype not in _RAG_TYPES:
        return {"type": "ABSTAIN", "reason": "invalid result type"}
    if not (_RESULT_KEYS >= set(obj.keys())):
        return {"type": "ABSTAIN", "reason": "unknown result keys"}

    out = {k: obj[k] for k in obj if k in _RESULT_KEYS}
    out["type"] = rtype

    if rtype == "ACTION":
        cap = str(out.get("capability") or "")
        allowed = set(allowed_capabilities or ())
        if not cap or cap not in allowed:
            return {"type": "ABSTAIN",
                    "reason": "capability not registered — refused"}
        if not isinstance(out.get("arguments"), dict):
            out["arguments"] = {}
        out.setdefault("goal", "")
        out.setdefault("evidence_ids", [])
        out.setdefault("requires_confirmation", False)
    elif rtype == "ANSWER":
        if not out.get("answer"):
            return {"type": "ABSTAIN", "reason": "empty answer"}
        out.setdefault("evidence_ids", [])
    elif rtype == "CLARIFICATION":
        if not out.get("question"):
            return {"type": "ABSTAIN", "reason": "missing question"}
    else:
        if not out.get("reason"):
            out["reason"] = "insufficient evidence"
    return out


def validate_action(result: Dict[str, Any],
                    allowed_capabilities: Sequence[str]) -> bool:
    """Final gate before anything is handed to the capability router.
    Retrieved text or the LLM alone can never authorize execution."""
    if not isinstance(result, dict) or result.get("type") != "ACTION":
        return False
    cap = str(result.get("capability") or "")
    return bool(cap) and cap in set(allowed_capabilities) and isinstance(
        result.get("arguments"), dict)


# ═══════════════════════════════════════════════════════════════════
# Hybrid activation gate (deterministic -> semantic -> RAG -> GoalRuntime)
# ═══════════════════════════════════════════════════════════════════

def should_activate(transcript: str) -> bool:
    """RAG activates only when the task needs knowledge/evidence. Obvious
    registered capabilities keep their fast path (no LLM, no retrieval)."""
    if not transcript or not transcript.strip():
        return False
    if is_live_state_question(transcript):
        return False          # live perception answers this, not RAG
    try:
        from core.semantic_router import semantic_router
        outcome = semantic_router.route(transcript)
        if getattr(outcome, "resolved", False):
            return False      # capability fast path wins
    except Exception:
        pass
    # Imperative fast path (spec §13): a verb-first command with no
    # knowledge cue is a capability request, not a knowledge request —
    # "Click the notification button." / "Read the third email." /
    # "search LinkedIn for jobs" never pay RAG cost, while "Open the
    # file where we implemented X" keeps RAG (knowledge cue present).
    if (_IMPERATIVE_RE.match(transcript)
            and not _KNOWLEDGE_CUE_RE.search(transcript)):
        return False
    return True


def _normalize_hit(hit: Any) -> Dict[str, Any]:
    """Bridge a knowledge-layer hit to the pipeline's canonical keys.

    The retriever returns *text* / *doc_path* / *semantic* / *lexical*
    and its ``source`` field is the FUSION label (semantic/keyword/both)
    — not a file path. The orchestrator's stages (per-file diversity
    dedupe, context rerank, prompt rendering, provenance) speak
    *content* / *source* / *source_type* / *semantic_score* /
    *lexical_score*. Bridging HERE (one place) keeps every downstream
    stage working on real content and real paths:

      text           -> content      (else the prompt renders empty blocks)
      doc_path       -> source       (else dedupe collapses to one source)
      fusion label   -> fusion       (kept, no longer mistaken for a path)

    Score components (semantic/lexical) and file_type are bridged only as
    a fallback: real retriever hits already carry ``semantic_score`` /
    ``lexical_score`` / ``source_type`` from make_evidence_fields().
    """
    d = (dict(hit) if hasattr(hit, "items") or isinstance(hit, dict)
         else _obj_to_dict(hit))

    # 1. content: the retrieval unit (full semantic chunk), never a path.
    if not d.get("content") and d.get("text"):
        d["content"] = d["text"]

    # 2. source: the real file path. The retriever's `source` is a fusion
    #    label, so derive the path from provenance fields instead.
    label = str(d.get("source") or "")
    chunk = str(d.get("chunk_id") or "")
    path = (d.get("doc_path") or d.get("path")
            or (chunk.rsplit("#", 1)[0] if "#" in chunk else "")
            or d.get("filename") or "")
    if label.lower() in ("both", "semantic", "keyword"):
        d["fusion"] = label
        if path:
            d["source"] = str(path)
    elif not label and path:
        d["source"] = str(path)
    if path:
        d.setdefault("path", str(path))

    # 3. score components / type, so reranking and provenance see them.
    if d.get("semantic_score") is None and d.get("semantic") is not None:
        d["semantic_score"] = d["semantic"]
    if d.get("lexical_score") is None and d.get("lexical") is not None:
        d["lexical_score"] = d["lexical"]
    if not d.get("source_type") and d.get("file_type"):
        d["source_type"] = d["file_type"]
    return d


# ═══════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════

class RAGOrchestrator:
    """Single RAG orchestration layer. Retrievers, context composer and the
    LLM callable are injectable; the defaults bind lazily to the EXISTING
    knowledge/service and streaming/reasoning infrastructure."""

    def __init__(
        self,
        *,
        retriever: Any = None,
        llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
        capabilities_fn: Optional[Callable[[], Sequence[str]]] = None,
        candidate_pool: int = CANDIDATE_POOL,
        top_k: int = SELECT_TOP_K,
    ) -> None:
        self._retriever = retriever
        self._llm_call = llm_call
        self._capabilities_fn = capabilities_fn
        self.candidate_pool = candidate_pool
        self.top_k = top_k

    # ── default bindings (lazy, exception-safe) ──────────────────

    def _get_retriever(self) -> Any:
        if self._retriever is not None:
            return self._retriever
        try:
            from knowledge.service import knowledge_service
            self._retriever = knowledge_service
        except Exception:
            try:
                from knowledge.retriever import KnowledgeRetriever
                self._retriever = KnowledgeRetriever()
            except Exception as e:
                logger.warning("[RAG] no retriever available: %s", e)
                self._retriever = None
        return self._retriever

    async def _get_llm(self, prompt: str) -> str:
        if self._llm_call is not None:
            return await self._llm_call(prompt)
        # Reuse the EXISTING streaming/reasoning infrastructure — never a
        # second LLM framework.
        try:
            from agent.reasoning_agent import reasoning_agent
            for meth in ("complete", "answer", "generate", "run"):
                fn = getattr(reasoning_agent, meth, None)
                if callable(fn):
                    res = fn(prompt)
                    if asyncio.iscoroutine(res):
                        res = await res
                    return str(res)
        except Exception as e:
            logger.debug("[RAG] reasoning_agent unavailable: %s", e)
        try:
            from agent.streaming_llm import streaming_llm
            for meth in ("complete", "generate", "ask"):
                fn = getattr(streaming_llm, meth, None)
                if callable(fn):
                    res = fn(prompt)
                    if asyncio.iscoroutine(res):
                        res = await res
                    return str(res)
        except Exception as e:
            logger.debug("[RAG] streaming_llm unavailable: %s", e)
        return ""

    def _capabilities(self) -> List[str]:
        if self._capabilities_fn is not None:
            return list(self._capabilities_fn())
        try:
            from core.tool_registry import CAPABILITIES
            return sorted(CAPABILITIES.keys())
        except Exception:
            return []

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Bounded retrieval through the existing knowledge layer
        (service.search preferred, retriever.search as fallback)."""
        ret = self._get_retriever()
        if ret is None:
            return []
        for meth, karg in (("search", "k"), ("search", "limit"),
                           ("retrieve", "k"), ("query", "k")):
            fn = getattr(ret, meth, None)
            if not callable(fn):
                continue
            for kwargs in ({karg: self.candidate_pool}, {"k": self.candidate_pool}, {}):
                try:
                    res = fn(query, **kwargs)
                    break
                except TypeError:
                    continue
            else:
                continue
            if asyncio.iscoroutine(res):
                return []   # sync path only; async callers inject a fake
            return [_normalize_hit(h) for h in (res or [])]
        return []

    # ── main entry ───────────────────────────────────────────────

    async def run(
        self,
        transcript: str,
        *,
        conversation_context: str = "",
        task_context: str = "",
        desktop_context: str = "",
        browser_context: str = "",
        referenced_entity: str = "",
        project_hint: str = "",
        force: bool = False,
    ) -> RagOutcome:
        t_all = time.perf_counter()
        timings: Dict[str, float] = {}

        # 0. activation gate: obvious capabilities never pay RAG/LLM cost
        if not force and not should_activate(transcript):
            q = build_retrieval_query(transcript, conversation_context,
                                      task_context, referenced_entity,
                                      project_hint, browser_context)
            pack = build_evidence_pack(q, [], conversation_context=conversation_context,
                                       task_context=task_context,
                                       desktop_context=desktop_context,
                                       browser_context=browser_context,
                                       available_capabilities=self._capabilities())
            return RagOutcome(pack=pack,
                              result={"type": "ABSTAIN",
                                      "reason": "fast path / live state"},
                              used_rag=False,
                              needs_live_state=is_live_state_question(transcript),
                              timings={"total_ms": 0.0})

        # 1. query construction
        t0 = time.perf_counter()
        query = build_retrieval_query(transcript, conversation_context,
                                      task_context, referenced_entity,
                                      project_hint, browser_context)
        timings["query_build_ms"] = (time.perf_counter() - t0) * 1000.0

        # Live-state questions never consult the stale index.
        if is_live_state_question(transcript):
            pack = build_evidence_pack(query, [], conversation_context=conversation_context,
                                       task_context=task_context,
                                       desktop_context=desktop_context,
                                       browser_context=browser_context,
                                       available_capabilities=self._capabilities())
            timings["total_ms"] = (time.perf_counter() - t_all) * 1000.0
            return RagOutcome(pack=pack,
                              result={"type": "ABSTAIN",
                                      "reason": "requires live perception"},
                              needs_live_state=True, timings=timings)

        # 2. retrieval (bounded)
        t0 = time.perf_counter()
        try:
            candidates = self._search(query.rewritten_query)
        except Exception as e:
            logger.warning("[RAG] retrieval failed: %s", e)
            candidates = []
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000.0
        candidates = candidates[: self.candidate_pool]

        # 3. filter + rerank + select
        t0 = time.perf_counter()
        ranked = rerank_candidates(query, candidates,
                                   current_project=project_hint,
                                   task_context=task_context)
        ranked = [c for c in ranked
                  if float(c.get("final_score", 0.0)) > 0.05][: self.candidate_pool]
        evidence = select_evidence(ranked, self.top_k) if ranked else []
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. evidence pack
        t0 = time.perf_counter()
        pack = build_evidence_pack(
            query, evidence,
            conversation_context=conversation_context,
            task_context=task_context,
            desktop_context=desktop_context,
            browser_context=browser_context,
            available_capabilities=self._capabilities(),
        )
        timings["context_build_ms"] = (time.perf_counter() - t0) * 1000.0

        # Insufficient evidence -> abstain WITHOUT an LLM call (cheap, safe).
        if not pack.retrieved_evidence:
            timings["total_ms"] = (time.perf_counter() - t_all) * 1000.0
            logger.info("[RAG] abstain: no evidence (%.1fms)", timings["total_ms"])
            return RagOutcome(pack=pack,
                              result={"type": "ABSTAIN",
                                      "reason": "retrieved evidence is insufficient"},
                              timings=timings)

        # 5. LLM (injected / existing infra) with timing
        prompt = compose_prompt(pack)
        t0 = time.perf_counter()
        timings["llm_first_token_ms"] = -1.0
        try:
            raw = await self._get_llm(prompt)
            timings["llm_first_token_ms"] = (time.perf_counter() - t0) * 1000.0
            timings["llm_total_ms"] = timings["llm_first_token_ms"]
        except Exception as e:
            logger.warning("[RAG] LLM failed: %s", e)
            timings["llm_total_ms"] = (time.perf_counter() - t0) * 1000.0
            raw = ""

        result = parse_structured_result(raw, pack.available_capabilities)
        timings["total_ms"] = (time.perf_counter() - t_all) * 1000.0
        logger.info("[RAG] %s type=%s evidence=%d total=%.1fms",
                    query.rewritten_query[:60], result.get("type"),
                    len(pack.retrieved_evidence), timings["total_ms"])
        return RagOutcome(pack=pack, result=result, timings=timings)


def _obj_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"content": str(obj)}


# Global singleton (matches codebase style: module-level service objects).
rag_orchestrator = RAGOrchestrator()
