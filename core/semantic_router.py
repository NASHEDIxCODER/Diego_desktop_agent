"""
SemanticRouter — generic capability ROUTING CONTRACTS (Phase 25).

Architecture contract (explicitly NOT a second execution framework):

  * This module DESCRIBES and ROUTES capabilities. It NEVER executes
    anything: it has no execute()/dispatch() path, imports no
    ComputerController/BrowserGoalEngine/DesktopGoalEngine/GoalRuntime,
    and never calls the LLM or planner (Decision Budget tier 2 —
    cheap; tiers 3-5 stay with the layers that already own them).

  * Capability truth lives in core.tool_registry.CAPABILITIES (single
    registry). Every contract derived here MUST match an existing
    CAPABILITIES entry; a routing spec with no CAPABILITIES entry is
    ignored at contract-build time (correction: descriptions are
    derived, never duplicated).

  * A resolved route is a plain action dict
        {"action": <existing dispatcher/registry action>, "params": {...}}
    returned to DecisionEngine as Decision(action=...). Execution flows
    through the EXISTING pipeline:
        Decision.action → Brain._dispatch_and_verify → ActionDispatcher
        / tool_registry.execute → ActionVerifier (unchanged).

  * Precedence (DecisionEngine owns it, this module only slot-fills):
        deterministic direct routes (L0-L3)
        -> SEMANTIC capability router (this module, L3.5)
        -> existing L4-L8 fallback

  * semantic similarity ALONE never executes. A route resolves only
    when ALL gates pass together:
        similarity >= gate AND margin over runner-up >= MARGIN
        AND required params extracted from the utterance
        AND the target action is available (KNOWN_ACTIONS/registry)
        AND permission/authorize_intent permits it
    Otherwise the outcome is unresolved and the engine falls through
    to the next (cheaper-to-trust / already-existing) layer.

  * No site-specific logic anywhere: no LinkedIn/Gmail/Instagram/
    Telegram names, no phrase tables per site. Contracts are built
    from generic purpose/verbs/examples per CAPABILITY; scoring is
    token-overlap lexical now, hybridized with sentence-transformer
    embeddings when the existing nlp.embeddings model is ready
    (warmed in a background thread — routing never blocks on model
    load; under pytest the warmup is skipped so the suite stays
    deterministic and green).

  * Every route records timing (route_ms) for the latency budget:
    transcript→route / route→execute comparisons downstream.

Logging contract: [SEMROUTE]
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Decision Budget tiers (escalation policy — correction 6)
# ═══════════════════════════════════════════════════════════════

TIER_DIRECT = "cached/direct"          # L0-L3 (existing)
TIER_SEMANTIC = "semantic"             # this module (L3.5)
TIER_CHEAP_CONTEXT = "cheap_context"   # L4-L6 (existing)
TIER_PLANNER = "planner/llm"           # L7 (existing)
TIER_GOAL_RUNTIME = "goal_runtime"     # goal engines (existing)


# ═══════════════════════════════════════════════════════════════
# Gates — similarity alone is never sufficient (correction 4)
# ═══════════════════════════════════════════════════════════════

# Lexical-only scoring gates.
LEX_GATE = 0.45          # min overlap for top candidate
LEX_MARGIN = 0.12        # top must beat runner-up by this much
# Hybrid (lexical + embedding) scoring gates.
HYBRID_GATE = 0.50
HYBRID_MARGIN = 0.10

# Actions handled explicitly by the EXISTING ActionDispatcher (verified
# handlers in agent/action_dispatcher.py) beyond task_state.KNOWN_ACTIONS.
# These are references to existing handlers — not a new execution path.
_DISPATCHER_EXTRA_ACTIONS = frozenset({"browser_get_url", "browser_get_text"})

_STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "of", "to", "for", "from", "at", "by",
    "and", "or", "is", "are", "it", "this", "that", "these", "those", "be",
    "me", "my", "your", "you", "we", "us", "our", "please", "could", "would",
    "can", "do", "does", "did", "with", "into", "onto", "some", "any",
    "i", "then", "so", "if", "not", "no", "yes", "get", "got", "make",
    # Immediacy/time fillers — carry no capability intent. Without them a
    # contract whose EXAMPLE merely mentions "now" could outrank the
    # capability whose PURPOSE answers the utterance. Generic by design:
    # no site names, no phrase tables.
    "now", "right", "just", "quickly", "quick", "soon", "asap",
    "immediately", "today", "tonight", "already",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;\"'<>]+")
_DOMAIN_RE = re.compile(
    r"(?<![\w./-])([a-z0-9][a-z0-9-]{1,62}\.(?:com|org|net|io|dev|ai|gov|edu)"
    r"(?:/[^\s,;\"'<>]*)?)", re.IGNORECASE)
_PATH_RE = re.compile(
    r"(?<![\w/])(~?(?:/[\w.\-]+(?:/[\w.\-]+)*)|(?:[A-Za-z]:\\[\w.\-\\]+))"
)
_QUOTED_RE = re.compile(r"[\"']([^\"']{1,160})[\"']")
_EXT_RE = re.compile(r"\*\.([A-Za-z0-9]{1,6})\b")
# Plural "X files" only: singular "python file" is a NAME cue, not a
# type cue — never guess an extension from it (missing params → the
# route falls through to the existing layers instead).
_EXT_BEFORE_NOUN_RE = re.compile(r"\b([A-Za-z0-9]{1,6})\s+files\b")


def _tokens(text: str) -> set:
    """Lowercased word tokens minus stopwords (pure, no I/O)."""
    return {
        t for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) >= 2 and t not in _STOPWORDS
    }


def _overlap(a: set, b: set) -> float:
    """Overlap coefficient in [0, 1]; robust for short utterances."""
    if not a or not b:
        return 0.0
    return len(a & b) / float(min(len(a), len(b)))


# ═══════════════════════════════════════════════════════════════
# Generic param binders — utterance -> params or None (never guess)
# ═══════════════════════════════════════════════════════════════

def _build_url(text: str) -> Optional[Dict[str, Any]]:
    m = _URL_RE.search(text)
    if m:
        return {"url": m.group(0).rstrip(".,;:!?)\"'")}
    m = _DOMAIN_RE.search(text)
    if m:
        return {"url": m.group(1).rstrip(".,;:!?)\"'")}
    return None


def _build_no_params(text: str) -> Optional[Dict[str, Any]]:
    return {}


def _build_path(text: str) -> Optional[Dict[str, Any]]:
    m = _QUOTED_RE.search(text) or _PATH_RE.search(text)
    if not m:
        return None
    return {"path": m.group(1)}


def _build_search(text: str) -> Optional[Dict[str, Any]]:
    """filesystem.search: needs at least a path or a file-type cue."""
    params: Dict[str, Any] = {}
    exts: List[str] = []
    for m in _EXT_RE.finditer(text):
        exts.append(m.group(1).lower())
    if not exts:
        for m in _EXT_BEFORE_NOUN_RE.finditer(text):
            ext = m.group(1).lower()
            if ext not in _STOPWORDS and ext != "file":
                exts.append(ext)
    path_m = _QUOTED_RE.search(text) or _PATH_RE.search(text)
    if path_m:
        params["path"] = path_m.group(1)
    if exts:
        # de-dup, preserve order
        seen: Dict[str, bool] = {}
        for e in exts:
            seen.setdefault(e, True)
        params["extensions"] = list(seen.keys())
    if not params:
        return None
    return params


# ═══════════════════════════════════════════════════════════════
# Routing specs — how an EXISTING capability maps to an EXISTING
# dispatcher/registry action. Capability truth stays in CAPABILITIES;
# this map is stripped at build time for entries that do not exist
# there (no parallel registry, correction 1).
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _RouteSpec:
    action: str
    build_params: Callable[[str], Optional[Dict[str, Any]]]
    verbs: Tuple[str, ...]
    examples: Tuple[str, ...]
    gate: float = LEX_GATE


_ROUTE_SPECS: Dict[str, _RouteSpec] = {
    "browser.navigate": _RouteSpec(
        action="browser_navigate",
        build_params=_build_url,
        verbs=("navigate", "open", "visit", "browse", "go", "load", "url"),
        examples=(
            "open https://example.com",
            "navigate to www.example.com",
            "visit https://site.org/page",
            "go to example.com",
        ),
    ),
    "browser.extract": _RouteSpec(
        action="browser_get_url",
        build_params=_build_no_params,
        verbs=("url", "address", "page", "link", "read", "current"),
        examples=(
            "what is the current page url",
            "read the browser address",
            "which url is open right now",
        ),
    ),
    "filesystem.search": _RouteSpec(
        action="filesystem",
        build_params=_build_search,
        verbs=("find", "search", "locate", "lookup", "files", "folder"),
        examples=(
            "find pdf files in documents",
            "find the largest python file",
            "locate recent screenshots in home",
        ),
    ),
    "filesystem.list": _RouteSpec(
        action="filesystem",
        build_params=_build_path,
        verbs=("list", "show", "contents", "directory", "folder"),
        examples=(
            "list the contents of home",
            "show files in /var/log",
            "list directory ~/projects",
        ),
    ),
    "filesystem.stat": _RouteSpec(
        action="filesystem",
        build_params=_build_path,
        verbs=("size", "stat", "metadata", "details", "file"),
        examples=(
            "get the size of a file",
            "show details for /etc/hosts",
            "what is the size of ~/notes.txt",
        ),
    ),
    "vision.inspect": _RouteSpec(
        action="read_screen",
        build_params=_build_no_params,
        verbs=("screen", "display", "monitor", "see", "inspect", "visible"),
        examples=(
            "inspect the screen",
            "what is visible on the display",
            "read the monitor",
        ),
    ),
}


# Note: filesystem.search/list/stat/count all map to the SAME registry
# tool ("filesystem") with different params; the dispatcher/registry
# already differentiates by action=params["action"]. We bind it here so
# the action dict is complete and canonical:
_FS_PARAM_ACTIONS = {
    "filesystem.search": "search",
    "filesystem.list": "list",
    "filesystem.stat": "stat",
}


@dataclass(frozen=True)
class RouterContract:
    """A capability DESCRIPTION for routing — no executable handle."""

    capability: str
    purpose: str
    side_effects: str
    verification: str
    source_tool: str
    action: str
    build_params: Callable[[str], Optional[Dict[str, Any]]]
    verbs: Tuple[str, ...]
    examples: Tuple[str, ...]
    gate: float = LEX_GATE

    @property
    def text(self) -> str:
        """Full contract text used for similarity scoring."""
        return " ".join((
            self.capability.replace(".", " "),
            self.purpose,
            " ".join(self.verbs),
            " ".join(self.examples),
        ))


@dataclass
class RouterOutcome:
    """Result of ONE semantic routing decision (never an execution)."""

    resolved: bool = False
    budget: str = TIER_SEMANTIC
    capability: str = ""
    action: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    similarity: float = 0.0
    scoring: str = "lexical"          # lexical | hybrid
    side_effects: str = "READ_ONLY"
    verification: str = ""
    route_ms: float = 0.0
    reason: str = ""
    ranked: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolved": self.resolved,
            "budget": self.budget,
            "capability": self.capability,
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "similarity": round(self.similarity, 3),
            "scoring": self.scoring,
            "side_effects": self.side_effects,
            "route_ms": round(self.route_ms, 3),
            "reason": self.reason,
            "ranked": [(c, round(s, 3)) for c, s in self.ranked],
        }


class SemanticRouter:
    """Ranks capabilities and emits action dicts for the EXISTING engine.

    Pure + thread-safe. No execution, no LLM, no site-specific logic.
    """

    def __init__(self,
                 contracts: Optional[Sequence[RouterContract]] = None,
                 embeddings: str = "auto") -> None:
        """contracts: override for tests; None derives from CAPABILITIES.

        embeddings: "auto" (background warmup outside pytest, disabled
        under PYTEST_CURRENT_TEST), "1"/"on" force, "0"/"off" disable.
        """
        self._override = list(contracts) if contracts is not None else None
        self._contracts: Optional[List[RouterContract]] = None
        self._lock = threading.Lock()

        self._embeddings_mode = (embeddings or "auto").lower()
        self._emb_model_ready = False
        self._emb_vectors: Dict[str, Any] = {}
        self._emb_query = None
        self._emb_cosine = None
        self._warmup_started = False

    # ── Contract derivation (CAPABILITIES is the single registry) ──

    @staticmethod
    def derive_contracts() -> List[RouterContract]:
        """Build contracts ONLY for CAPABILITIES entries we can route to
        an existing dispatcher/registry action with deterministic param
        binding. Unknown/unsupported capabilities stay unrouted and fall
        through to the existing L4-L8 + planner layers."""
        try:
            from core.tool_registry import capabilities
            caps = capabilities()
        except Exception as e:  # registry unavailable → no semantic tier
            logger.debug("[SEMROUTE] capabilities unavailable: %s", e)
            return []

        contracts: List[RouterContract] = []
        for name, spec in _ROUTE_SPECS.items():
            entry = caps.get(name)
            if not entry:
                # Routing spec with no capability truth → ignored.
                logger.debug("[SEMROUTE] spec without CAPABILITIES entry: %s",
                             name)
                continue
            build = spec.build_params
            fs_action = _FS_PARAM_ACTIONS.get(name)
            if fs_action:
                inner = build

                def build(t: str, _inner=build,
                          _fa=fs_action) -> Optional[Dict[str, Any]]:
                    p = _inner(t)
                    if p is None:
                        return None
                    p["action"] = _fa
                    return p
            contracts.append(RouterContract(
                capability=name,
                purpose=str(entry.get("purpose", "")),
                side_effects=str(entry.get("side_effects", "READ_ONLY")),
                verification=str(entry.get("verification", "")),
                source_tool=str(entry.get("tool", "")),
                action=spec.action,
                build_params=build,
                verbs=spec.verbs,
                examples=spec.examples,
                gate=spec.gate,
            ))
        return contracts

    def contracts(self) -> List[RouterContract]:
        with self._lock:
            if self._contracts is None:
                self._contracts = (list(self._override)
                                   if self._override is not None
                                   else self.derive_contracts())
            return list(self._contracts)

    # ── Availability of the EXISTING execution target ─────────────

    @staticmethod
    def _action_available(action: str) -> bool:
        if not action:
            return False
        if action in _DISPATCHER_EXTRA_ACTIONS:
            return True
        try:
            from agent.task_state import registry_tool_available
            if registry_tool_available(action):
                return True
        except Exception:
            pass
        try:
            from agent.task_state import KNOWN_ACTIONS
            return action in KNOWN_ACTIONS
        except Exception:
            return False

    # ── Permission gate (authorize_intent — never bypassed) ───────

    @staticmethod
    def _permission_allows(text: str) -> bool:
        try:
            from nlp.intent_authorizer import authorize_intent
        except Exception:
            # Mirror Brain semantics: authorizer unavailable → don't block
            # here; brain's own gate still runs before dispatch.
            return True
        try:
            auth = authorize_intent(text)
        except Exception:
            return True
        if auth is None:
            return True
        return bool(getattr(auth, "actionable", True))

    # ── Scoring ───────────────────────────────────────────────────

    @staticmethod
    def _score_lexical(query_tokens: set,
                       contracts: List[RouterContract]) -> List[
                           Tuple[RouterContract, float]]:
        """Blend FULL contract-text overlap with PURPOSE overlap.

        Intent alignment (the capability's stated purpose) must dominate
        incidental token overlap inside examples: a filler word that
        happens to appear in some example must not let a non-matching
        capability outrank the one whose purpose answers the utterance.
        Similarity is still only the FIRST gate — params/availability/
        permission gates in route() decide whether anything resolves.
        """
        scored: List[Tuple[RouterContract, float]] = []
        for c in contracts:
            full = _overlap(query_tokens, _tokens(c.text))
            purpose_tokens = _tokens(c.purpose)
            purpose = (_overlap(query_tokens, purpose_tokens)
                       if purpose_tokens else full)
            scored.append((c, 0.5 * purpose + 0.5 * full))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored

    def _maybe_start_embedding_warmup(
            self, contracts: List[RouterContract]) -> None:
        if self._warmup_started or self._embeddings_mode in ("0", "off"):
            return
        if (self._embeddings_mode == "auto"
                and os.environ.get("PYTEST_CURRENT_TEST")):
            # Deterministic, model-free test runs (suite stays green).
            self._warmup_started = True
            return
        forced = self._embeddings_mode in ("1", "on") or bool(
            os.environ.get("DIEGO_ROUTER_EMBEDDINGS"))
        if not forced and self._embeddings_mode == "auto":
            pass  # background warmup — never blocks routing
        self._warmup_started = True
        texts = [c.text for c in contracts]
        names = [c.capability for c in contracts]

        def _warm() -> None:
            try:
                import numpy as np
                from nlp.embeddings import embed_batch, cosine_similarity
                vecs = embed_batch(texts, show_progress=False)
                self._emb_query = embed_batch
                self._emb_cosine = cosine_similarity
                self._emb_vectors = dict(zip(names, vecs))
                self._emb_model_ready = True
                logger.info(
                    "[SEMROUTE] embeddings warm for %d capabilities",
                    len(names))
            except BaseException as e:  # never take routing down
                logger.debug("[SEMROUTE] embedding warmup skipped: %s", e)

        threading.Thread(target=_warm, daemon=True,
                         name="semroute-embed-warm").start()

    def _score_hybrid(self, query: str, query_tokens: set,
                      contracts: List[RouterContract]) -> Tuple[
                          List[Tuple[RouterContract, float]], str]:
        lexical = self._score_lexical(query_tokens, contracts)
        if not (self._emb_model_ready and self._emb_query
                and self._emb_cosine):
            return lexical, "lexical"
        try:
            qv = self._emb_query([query])[0]
            combined: List[Tuple[RouterContract, float]] = []
            for c, lex in lexical:
                ev = float(self._emb_cosine(qv, self._emb_vectors[c.capability]))
                # Calibrated blend: cosine in [-1,1] → [0,1] first.
                ev01 = max(0.0, ev)
                combined.append((c, 0.6 * ev01 + 0.4 * lex))
            combined.sort(key=lambda p: p[1], reverse=True)
            return combined, "hybrid"
        except Exception as e:
            logger.debug("[SEMROUTE] hybrid scoring fell back: %s", e)
            return lexical, "lexical"

    # ── Main entry (sync, fast, no I/O beyond lazy imports) ───────

    def route(self, text: str) -> RouterOutcome:
        t0 = time.perf_counter_ns()
        outcome = RouterOutcome()
        q = (text or "").strip()
        if not q:
            outcome.reason = "empty"
            outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
            return outcome

        contracts = self.contracts()
        if not contracts:
            outcome.reason = "no_contracts"
            outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
            return outcome

        self._maybe_start_embedding_warmup(contracts)

        q_tokens = _tokens(q)
        ranked, scoring = self._score_hybrid(q, q_tokens, contracts)
        outcome.scoring = scoring
        outcome.ranked = [(c.capability, s) for c, s in ranked[:3]]

        top_c, top_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        gate = (HYBRID_GATE if scoring == "hybrid" else top_c.gate)
        margin_need = (HYBRID_MARGIN if scoring == "hybrid" else LEX_MARGIN)

        # Gate 1: similarity threshold (necessary, NOT sufficient).
        if top_s < gate:
            outcome.reason = f"below_gate:{top_s:.2f}<{gate:.2f}"
            outcome.similarity = top_s
            outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
            return outcome

        # Gate 2: margin over runner-up (ambiguity → fall through).
        # A near-perfect score may rescue a SMALL margin, but only when
        # it STRICTLY leads the runner-up — an exact tie at the top is
        # ambiguous by definition and must fall through.
        if ((top_s - second_s) < margin_need
                and (top_s < 0.999 or top_s <= second_s)):
            outcome.reason = (f"ambiguous:margin={top_s - second_s:.2f}"
                              f"<{margin_need:.2f}")
            outcome.similarity = top_s
            outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
            return outcome

        # Gate 3: required params must bind deterministically.
        params = top_c.build_params(q)
        if params is None:
            outcome.reason = "missing_params"
            outcome.similarity = top_s
            outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
            return outcome

        # Gate 4: execution target must already exist.
        if not self._action_available(top_c.action):
            outcome.reason = f"unavailable:{top_c.action}"
            outcome.similarity = top_s
            outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
            return outcome

        # Gate 5: permission / authorize_intent.
        if not self._permission_allows(q):
            outcome.reason = "permission_denied"
            outcome.similarity = top_s
            outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
            return outcome

        outcome.resolved = True
        outcome.capability = top_c.capability
        outcome.action = {"action": top_c.action, "params": params}
        outcome.confidence = float(top_s)
        outcome.similarity = float(top_s)
        outcome.side_effects = top_c.side_effects
        outcome.verification = top_c.verification
        outcome.reason = "resolved"
        outcome.route_ms = (time.perf_counter_ns() - t0) / 1e6
        logger.info(
            "[SEMROUTE] %s -> %s score=%.2f scoring=%s route_ms=%.2f",
            q[:90], top_c.capability, top_s, scoring, outcome.route_ms)
        return outcome


# Module singleton (wired lazily by DecisionEngine L3.5).
semantic_router = SemanticRouter()
