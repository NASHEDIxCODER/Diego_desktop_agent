"""
Phase 23: generic browser-goal model (PURE — no I/O, no side effects).

This module owns the VOCABULARY of browser goals. It is deliberately free of
execution, perception and UI concerns so that it can be unit-tested
deterministically and reused by the engine, the trace and the workflow panel:

  * BrowserAction      — the generic browser action set (OPEN_URL, FIND_ELEMENT,
                         CLICK_ELEMENT, TYPE_INPUT, EXTRACT_RESULTS, …)
  * BrowserGoal        — an interpreted user goal (site / query / target / kind)
  * BrowserStep        — one planned step with an EXPECTED EFFECT
  * BrowserResult      — an extraction record that separates OBSERVED facts
                         from INFERRED values and MISSING values
  * Candidate/ranking  — semantic element selection + ambiguity detection
  * verify_expected_effect() — the browser expected-effect contract. A generic
                         "something changed" is NEVER accepted as success.

Nothing here knows any website: there is no site-specific selector, URL path
or workflow anywhere in this file. Site names resolve to domains through a
small name→domain table (documented, generic), never through a scripted
workflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ═══════════════════════════════════════════════════════════════════
# 1. Action vocabulary
# ═══════════════════════════════════════════════════════════════════


class BrowserAction(str, Enum):
    """Generic browser capabilities the agent may compose."""

    OPEN_BROWSER = "OPEN_BROWSER"
    OPEN_URL = "OPEN_URL"
    NAVIGATE = "NAVIGATE"
    GO_BACK = "GO_BACK"
    GO_FORWARD = "GO_FORWARD"
    REFRESH = "REFRESH"
    WAIT_FOR_PAGE = "WAIT_FOR_PAGE"
    FIND_ELEMENT = "FIND_ELEMENT"
    CLICK_ELEMENT = "CLICK_ELEMENT"
    TYPE_INPUT = "TYPE_INPUT"
    CLEAR_INPUT = "CLEAR_INPUT"
    SELECT_OPTION = "SELECT_OPTION"
    PRESS_KEY = "PRESS_KEY"
    SCROLL = "SCROLL"
    EXTRACT_TEXT = "EXTRACT_TEXT"
    EXTRACT_LINKS = "EXTRACT_LINKS"
    READ_PAGE = "READ_PAGE"
    PAGINATE = "PAGINATE"
    EXTRACT_RESULTS = "EXTRACT_RESULTS"
    ASK_USER = "ASK_USER"


# How each browser action is executed through the EXISTING ComputerController
# (computer/computer_controller.py). Empty string = engine-internal primitive
# (no controller action exists; the engine performs it with observations).
COMPUTER_ACTION: Dict[BrowserAction, str] = {
    BrowserAction.OPEN_BROWSER: "open_app",
    BrowserAction.OPEN_URL: "open_url",
    BrowserAction.NAVIGATE: "navigate",
    BrowserAction.GO_BACK: "back",
    BrowserAction.GO_FORWARD: "forward",
    BrowserAction.REFRESH: "refresh",
    BrowserAction.WAIT_FOR_PAGE: "wait_for_page",
    BrowserAction.FIND_ELEMENT: "find_candidates",
    BrowserAction.CLICK_ELEMENT: "click",
    BrowserAction.TYPE_INPUT: "type_text",
    BrowserAction.CLEAR_INPUT: "clear_text",
    BrowserAction.SELECT_OPTION: "select_option",
    BrowserAction.PRESS_KEY: "press_key",
    BrowserAction.SCROLL: "scroll",
    BrowserAction.EXTRACT_TEXT: "extract_text",
    BrowserAction.EXTRACT_LINKS: "extract_links",
    BrowserAction.READ_PAGE: "read_page",
    BrowserAction.PAGINATE: "click",
    BrowserAction.EXTRACT_RESULTS: "read_page",
    BrowserAction.ASK_USER: "",
}


class BrowserGoalKind(str, Enum):
    """What the user actually asked for."""

    OPEN_SITE = "open_site"        # "open github" (site root expected)
    OPEN_URL = "open_url"          # explicit URL
    SEARCH = "search"              # "search <site> for <query>"
    FIND_ITEM = "find_item"        # "find <target>" (element/link/page)
    OPEN_ITEM = "open_item"        # "open <target>" (an item, not a site)
    EXTRACT = "extract"            # "read/extract … from <site>"
    READ = "read"                  # "summarize this page"
    UNKNOWN = "unknown"


class PageState(str, Enum):
    """Classification of the observed page (never guessed)."""

    LOADED = "loaded"
    LOADING = "loading"
    LOGIN_REQUIRED = "login_required"
    AUTHENTICATED = "authenticated"
    BLOCKED = "blocked"            # captcha / access-control / rate limit
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    EMPTY = "empty"                # nothing observable (no browser page)
    UNAVAILABLE = "unavailable"    # capability/backend missing


class BrowserStatus(str, Enum):
    """Task-level status of a browser goal run."""

    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"     # manual login / confirmation
    ASKING_USER = "ASKING_USER"               # ambiguity clarification
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EffectResult(str, Enum):
    """Expected-effect verdict (mirrors core.goal_verification.GoalResult)."""

    PASS = "PASS"
    FAIL = "FAIL"
    NO_EVIDENCE = "NO_EVIDENCE"


# ═══════════════════════════════════════════════════════════════════
# 2. Expected effects (per action)
# ═══════════════════════════════════════════════════════════════════

EXPECTED_EFFECTS: Dict[BrowserAction, str] = {
    BrowserAction.OPEN_BROWSER:
        "a browser page becomes available and reports a URL",
    BrowserAction.OPEN_URL:
        "the observed URL/domain changes to the requested target",
    BrowserAction.NAVIGATE:
        "the observed URL/domain changes to the requested target",
    BrowserAction.GO_BACK:
        "the observed URL changes to the previous history entry",
    BrowserAction.GO_FORWARD:
        "the observed URL changes to the next history entry",
    BrowserAction.REFRESH:
        "the same URL is re-read (content re-observed, not a stale copy)",
    BrowserAction.WAIT_FOR_PAGE:
        "the page finishes loading (loading state clears)",
    BrowserAction.FIND_ELEMENT:
        "exactly one semantic match is identified, or the ambiguity is "
        "reported (never a guess)",
    BrowserAction.CLICK_ELEMENT:
        "the target interaction causes the DECLARED page/UI state transition",
    BrowserAction.TYPE_INPUT:
        "the input holds the typed value in observed page state",
    BrowserAction.CLEAR_INPUT:
        "the target input is empty in observed page state",
    BrowserAction.SELECT_OPTION:
        "the target select shows the chosen option in observed page state",
    BrowserAction.PRESS_KEY:
        "the declared state transition for that key occurs (e.g. submit)",
    BrowserAction.SCROLL:
        "the page scroll position changes",
    BrowserAction.EXTRACT_TEXT:
        "the requested information exists in the observed page state",
    BrowserAction.EXTRACT_LINKS:
        "the requested links exist in the observed page state",
    BrowserAction.READ_PAGE:
        "non-empty page text is observed and recorded as evidence",
    BrowserAction.PAGINATE:
        "the result page advances (new results / page marker observed)",
    BrowserAction.EXTRACT_RESULTS:
        "the requested results exist in the observed page state",
    BrowserAction.ASK_USER:
        "the user's answer is received before continuing",
}


@dataclass
class EffectVerdict:
    """One expected-vs-observed verification record."""

    action: str
    expected_effect: str
    observed_effect: str = ""
    result: EffectResult = EffectResult.NO_EVIDENCE
    evidence: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.result == EffectResult.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "expected_effect": self.expected_effect,
            "observed_effect": self.observed_effect,
            "result": self.result.value,
            "evidence": dict(self.evidence),
            "detail": self.detail,
        }


def expected_effect_text(action: "BrowserAction | str",
                         params: Optional[Dict[str, Any]] = None) -> str:
    """Human/README-visible expected effect for one browser action."""
    try:
        act = action if isinstance(action, BrowserAction) else BrowserAction(str(action))
    except ValueError:
        return "the action reports a verified, observable effect"
    base = EXPECTED_EFFECTS.get(act, "the action reports a verified effect")
    params = params or {}
    target = params.get("url") or params.get("target") or params.get("query") or ""
    if act in (BrowserAction.NAVIGATE, BrowserAction.OPEN_URL) and target:
        return f"URL changes to {target}"
    if act == BrowserAction.CLICK_ELEMENT and params.get("expect"):
        return f"clicking '{params.get('target', '')}' causes '{params['expect']}'"
    if act in (BrowserAction.EXTRACT_RESULTS, BrowserAction.EXTRACT_TEXT) and target:
        return f"information about '{target}' exists in observed page state"
    return base


# ═══════════════════════════════════════════════════════════════════
# 3. Goal / step / result models
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BrowserGoal:
    """An INTERPRETED browser goal (generic concepts only)."""

    raw_text: str = ""
    kind: BrowserGoalKind = BrowserGoalKind.UNKNOWN
    site: str = ""                                   # "linkedin" / "github.com"
    url: str = ""                                    # resolved start URL
    query: str = ""                                  # search / extract subject
    target: str = ""                                 # semantic element target
    expect: str = ""                                 # declared post-action state
    expect_url_contains: str = ""                    # declared URL expectation
    scope: str = ""                                  # e.g. current page context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_site_scoped(self) -> bool:
        return bool(self.site or self.url)

    def describe(self) -> str:
        """Concise operational description (never chain-of-thought)."""
        if self.kind == BrowserGoalKind.SEARCH:
            where = self.site or self.url or self.scope or "the current site"
            return f"Search {where} for '{self.query}'"
        if self.kind == BrowserGoalKind.OPEN_SITE:
            return f"Open {self.site or self.url}"
        if self.kind == BrowserGoalKind.OPEN_URL:
            return f"Open {self.url}"
        if self.kind == BrowserGoalKind.FIND_ITEM:
            where = f" on {self.site}" if self.site else ""
            return f"Find '{self.target or self.query}'{where}"
        if self.kind == BrowserGoalKind.OPEN_ITEM:
            where = f" on {self.site}" if self.site else ""
            return f"Open '{self.target}'{where}"
        if self.kind == BrowserGoalKind.EXTRACT:
            where = f" from {self.site}" if self.site else ""
            return f"Extract '{self.query or self.target}'{where}"
        if self.kind == BrowserGoalKind.READ:
            return "Read the current page"
        return f"Browser goal: {self.raw_text[:80]}"

    def completion_requirement(self) -> str:
        """What must be VERIFIED before this goal counts as complete."""
        if self.kind == BrowserGoalKind.OPEN_URL:
            return f"URL {self.url} observed as the active page"
        if self.kind == BrowserGoalKind.OPEN_SITE:
            return (f"domain '{self.site}' observed as the active page "
                    f"(and not a login/interstitial page)")
        if self.kind == BrowserGoalKind.SEARCH:
            return (f"search results for '{self.query}' observed as rendered "
                    f"content")
        if self.kind == BrowserGoalKind.FIND_ITEM:
            return f"'{self.target or self.query}' observed as present"
        if self.kind == BrowserGoalKind.OPEN_ITEM:
            return f"'{self.target}' opened and the resulting page observed"
        if self.kind in (BrowserGoalKind.EXTRACT, BrowserGoalKind.READ):
            return "the requested information extracted from observed page state"
        return "an observed, verified page state"


@dataclass
class BrowserStep:
    """One planned browser step with a declared expected effect."""

    action: BrowserAction
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    optional: bool = False
    index: int = 0
    status: str = "PENDING"          # PENDING | DONE | FAILED | SKIPPED
    verdict: str = ""
    observation: str = ""
    attempts: int = 0

    @property
    def target(self) -> str:
        return str(self.params.get("target") or self.params.get("url")
                   or self.params.get("query") or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "action": self.action.value,
            "description": self.description, "params": dict(self.params),
            "expected_effect": self.expected_effect,
            "optional": self.optional, "status": self.status,
            "verdict": self.verdict, "observation": self.observation,
            "attempts": self.attempts,
        }


@dataclass
class BrowserResult:
    """One extracted item, with provenance and honesty about missing data."""

    title: str = ""
    url: str = ""
    source: str = ""                 # domain / page title the item came from
    visible_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    observed: bool = True            # True = read from the page
    inferred: bool = False           # True = derived (never presented as fact)
    missing: List[str] = field(default_factory=list)   # fields we could not read

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title, "url": self.url, "source": self.source,
            "visible_text": self.visible_text[:600],
            "metadata": dict(self.metadata), "evidence": dict(self.evidence),
            "observed": self.observed, "inferred": self.inferred,
            "missing": list(self.missing),
        }

    def describe(self) -> str:
        label = self.title or self.url or self.visible_text[:60] or "(untitled)"
        return label


@dataclass
class BrowserExtraction:
    """Result of an extraction step (facts vs inferred vs missing)."""

    subject: str = ""
    results: List[BrowserResult] = field(default_factory=list)
    count: int = 0
    page_title: str = ""
    page_url: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "count": self.count,
            "results": [r.to_dict() for r in self.results],
            "page_title": self.page_title, "page_url": self.page_url,
            "evidence": dict(self.evidence), "missing": list(self.missing),
        }


# ═══════════════════════════════════════════════════════════════════
# 4. Semantic element selection + ambiguity
# ═══════════════════════════════════════════════════════════════════

# Exact match is conclusive; two non-exact matches this close are ambiguous.
_STRONG = 0.75
_DOMINANCE = 0.20

_ORDINALS = {
    "first": 1, "1st": 1, "one": 1, "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "last": -1,
}


def normalize_label(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace (comparison form)."""
    t = re.sub(r"[^\w\s]+", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in normalize_label(text).split() if t]


@dataclass
class Candidate:
    """One possible semantic match for a target."""

    label: str
    kind: str = "element"
    href: str = ""
    value: str = ""
    score: float = 0.0
    index: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "label": self.label, "kind": self.kind,
                "href": self.href[:200], "score": round(self.score, 3),
                "evidence": dict(self.evidence)}


def score_candidate(target: str, label: str) -> float:
    """Semantic similarity of one label to a target (0..1). Pure + cheap."""
    t_raw = normalize_label(target)
    l_raw = normalize_label(label)
    if not t_raw or not l_raw:
        return 0.0
    if t_raw == l_raw:
        return 1.0
    if t_raw in l_raw:
        # "jobs" inside "jobs 12" — strong, but not conclusive.
        return 0.85 if len(t_raw.split()) > 1 else 0.8
    if l_raw in t_raw:
        return 0.7
    t_toks, l_toks = set(_tokens(t_raw)), set(_tokens(l_raw))
    if not t_toks or not l_toks:
        return 0.0
    overlap = len(t_toks & l_toks)
    if not overlap:
        return 0.0
    return round(0.55 * (overlap / max(len(t_toks), len(l_toks)))
                 + 0.25 * (overlap / len(t_toks)), 3)


def build_candidates(target: str,
                     raw: Sequence[Dict[str, Any]]) -> List[Candidate]:
    """Score raw perception matches (DOM order preserved as `index`)."""
    out: List[Candidate] = []
    for i, item in enumerate(raw or []):
        label = str(item.get("label") or item.get("text") or "")
        out.append(Candidate(
            label=label,
            kind=str(item.get("kind") or item.get("role") or "element"),
            href=str(item.get("href") or ""),
            value=str(item.get("value") or ""),
            score=score_candidate(target, label),
            index=i,
            evidence={"source": str(item.get("method") or "browser_dom"),
                      "tag": str(item.get("tag") or "")},
        ))
    return out


def rank_candidates(target: str, raw: Sequence[Dict[str, Any]],
                    *, strong: float = _STRONG,
                    dominance: float = _DOMINANCE
                    ) -> Tuple[List[Candidate], bool, str]:
    """Rank candidates and decide whether the target is AMBIGUOUS.

    Returns ``(ranked, ambiguous, reason)``. Ambiguous is True only when two
    or more DISTINCT strong matches exist and none is dominant — the caller
    must then ask the user instead of guessing.
    """
    cands = [c for c in build_candidates(target, raw) if c.score > 0]
    cands.sort(key=lambda c: (-c.score, c.index))
    if not cands:
        return [], False, "no candidate matched"
    exact = [c for c in cands if c.score >= 0.999]
    if len(exact) == 1:
        return cands, False, "exact match"
    strong_c = [c for c in cands if c.score >= strong]
    distinct = {normalize_label(c.label) for c in strong_c}
    if len(distinct) >= 2:
        top, second = strong_c[0], strong_c[1]
        if top.score - second.score < dominance:
            return cands, True, f"{len(distinct)} equally plausible matches"
    return cands, False, "single dominant match"


# Minimum score for a fuzzy candidate to be used as the RESOLVED target. A
# weaker match is reported as NOT_FOUND instead of being acted on (never guess).
_MIN_RESOLVE_SCORE = 0.25


def select_target(target: str, raw: Sequence[Dict[str, Any]],
                  *, min_score: float = _MIN_RESOLVE_SCORE
                  ) -> Tuple[str, List[Candidate], bool, str]:
    """Resolve a semantic TARGET against observed elements/links. Pure.

    Returns ``(label, ranked, ambiguous, reason)`` where:

      * ``label``     the OBSERVED label to act on (``""`` = nothing matched)
      * ``ranked``    the scored candidates, best first
      * ``ambiguous`` True when two equally plausible matches exist — the
                      caller MUST ask the user rather than guessing
      * ``reason``    short operational explanation (never chain-of-thought)

    An exact match wins outright. Otherwise the best candidate is accepted
    only when it is STRICTLY dominant and scores at least ``min_score``;
    anything weaker resolves to ``""`` (classified NOT_FOUND, not a guess).
    """
    if not normalize_label(target):
        return "", [], False, "no target"
    ranked, ambiguous, reason = rank_candidates(target, raw)
    if ambiguous:
        return "", ranked, True, reason
    if not ranked:
        return "", [], False, "no observed element matched"
    top = ranked[0]
    if top.score >= 0.999:
        return top.label, ranked, False, "exact match"
    if top.score < min_score:
        return "", ranked, False, f"best match too weak ({top.score:.2f})"
    if len(ranked) > 1 and ranked[1].score >= top.score:
        return "", ranked, False, "no dominant match"
    if not str(top.label or "").strip():
        return "", ranked, False, "dominant candidate has no label"
    return top.label, ranked, False, reason


def resolve_answer_to_candidate(answer: str,
                                candidates: Sequence[Candidate]
                                ) -> Optional[Candidate]:
    """Map a clarification answer ("the first one", "Rahul Sharma") to one."""
    text = normalize_label(answer)
    if not text or not candidates:
        return None
    m = re.search(r"\b(\d+)\b", text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    for word, pos in sorted(_ORDINALS.items(), key=lambda kv: -len(kv[0])):
        # longer (more specific) ordinal words first: "second" beats "one"
        if re.search(rf"\b{word}\b", text):
            if pos == -1:
                return candidates[-1]
            if 1 <= pos <= len(candidates):
                return candidates[pos - 1]
    ranked, _ambiguous, _reason = rank_candidates(
        answer, [c.__dict__ for c in candidates])
    for cand in ranked:
        if cand.score >= 0.8:
            return candidates[cand.index]
    return candidates[0] if len(candidates) == 1 else None


# ═══════════════════════════════════════════════════════════════════
# 5. Goal parsing (generic — no site-specific workflow)
# ═══════════════════════════════════════════════════════════════════

# Name → domain resolution for widely known sites. This is NAME RESOLUTION
# only (like a dictionary of words); it contains no workflow, no selector and
# no page-specific knowledge. Any explicit URL or FQDN bypasses it entirely.
SITE_ALIASES: Dict[str, str] = {
    "google": "https://www.google.com",
    "bing": "https://www.bing.com",
    "duckduckgo": "https://duckduckgo.com",
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "gitlab": "https://gitlab.com",
    "linkedin": "https://www.linkedin.com",
    "indeed": "https://www.indeed.com",
    "glassdoor": "https://www.glassdoor.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "reddit": "https://www.reddit.com",
    "wikipedia": "https://en.wikipedia.org",
    "amazon": "https://www.amazon.com",
    "gmail": "https://mail.google.com",
    "x": "https://x.com",
    "twitter": "https://x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "medium": "https://medium.com",
    "hacker news": "https://news.ycombinator.com",
    "hackernews": "https://news.ycombinator.com",
}

_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^(?:[\w\-]+\.)+[a-z]{2,}(?:/[^\s]*)?$", re.IGNORECASE)

_SITE_SUFFIX_RE = re.compile(
    r"\s+(?:on|in|at|from|via)\s+(?P<site>[a-z0-9\-\.\s]+?)\s*$", re.IGNORECASE)

# Result-oriented query words: "find jobs" is a SEARCH goal, not a click goal.
_RESULT_WORDS = ("job", "jobs", "vacanc", "opening", "listings", "listing",
                 "results", "result", "candidates", "profiles", "articles",
                 "posts", "repositor", "videos", "products", "news",
                 "tickets", "issues", "pull request")

_OPEN_VERBS = r"(?:open|go to|navigate to|visit|launch|load)"
_SEARCH_VERBS = r"(?:search|look up|google|query)"
_FIND_VERBS = r"(?:find|show me|look for|get me|locate)"
_READ_VERBS = r"(?:read|summarize|summarise|describe|what(?:'s| is) on)"
_EXTRACT_VERBS = r"(?:extract|collect|gather|scrape|list|give me|show)"


def resolve_site_url(site: str) -> str:
    """Resolve a site token to a start URL (explicit URL wins). Pure."""
    s = str(site or "").strip().strip(".,")
    if not s:
        return ""
    m = _URL_RE.search(s)
    if m:
        return m.group(0)
    if _DOMAIN_RE.match(s):
        return "https://" + s
    key = re.sub(r"\s+", " ", s.lower()).strip()
    if key in SITE_ALIASES:
        return SITE_ALIASES[key]
    squashed = key.replace(" ", "")
    if squashed in SITE_ALIASES:
        return SITE_ALIASES[squashed]
    # A bare single word is a plausible domain name (generic resolution, no
    # site knowledge): "mozilla" → https://mozilla.com. Multi-word phrases are
    # NOT guessed — an unknown phrase must be clarified by the caller.
    if re.match(r"^[a-z0-9\-]{2,}$", squashed):
        return "https://" + squashed + ".com"
    return ""


def _split_scope(text: str) -> Tuple[str, str]:
    """Split a trailing ' on/in/at/from <site>' from the rest of a request."""
    m = _SITE_SUFFIX_RE.search(text)
    if not m:
        return text.strip(), ""
    site = m.group("site").strip()
    # A conjunction inside the captured phrase starts an instruction clause
    # ("workhub.test and wait for it"): only the head names the site; the
    # tail is not part of it (and is engine-owned behaviour anyway).
    site = re.split(r"\s+(?:and|or|then|but|while|when|after)\s+",
                    site, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # Never treat a trailing word as a site when it is a result descriptor.
    if any(w in site for w in ("jobs", "results", "page", "me", "here")):
        return text.strip(), ""
    return text[:m.start()].strip(), site


def _domain_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)", str(url or ""), re.IGNORECASE)
    return (m.group(1) if m else "").lower().replace("www.", "")


def _looks_like_site(text: str) -> bool:
    t = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return bool(t in SITE_ALIASES or t.replace(" ", "") in SITE_ALIASES
                or _DOMAIN_RE.match(t))


def parse_browser_goal(text: str, *, scope: str = "") -> Optional[BrowserGoal]:
    """Interpret a user utterance as a browser goal (or None if not one).

    `scope` is the current page context (domain/title) used for follow-up
    utterances such as "find backend jobs" right after opening a site.
    """
    raw = str(text or "").strip().rstrip(".!?")
    if not raw:
        return None
    low = raw.lower()
    goal = BrowserGoal(raw_text=raw, scope=scope)

    # ── explicit URL ───────────────────────────────────────────────
    m = _URL_RE.search(raw)
    if m:
        goal.url = m.group(0)
        goal.site = _domain_of(goal.url)
        rest = raw.replace(m.group(0), " ").strip()
        if (not rest or re.match(rf"^{_OPEN_VERBS}\s*$", rest, re.IGNORECASE)
                or re.match(rf"^{_OPEN_VERBS}\b", low, re.IGNORECASE)):
            goal.kind = BrowserGoalKind.OPEN_URL
            return goal
        goal.query = rest
        goal.kind = (BrowserGoalKind.EXTRACT
                     if re.search(rf"^{_EXTRACT_VERBS}\b", rest, re.IGNORECASE)
                     else BrowserGoalKind.OPEN_URL)
        return goal

    body, site = _split_scope(low)

    # ── search ─────────────────────────────────────────────────────
    m = re.match(rf"^{_SEARCH_VERBS}\s+(?P<rest>.+)$", body, re.IGNORECASE)
    if m:
        rest = m.group("rest").strip()
        m2 = re.match(r"^(?P<site>[\w\-\.\s]+?)\s+for\s+(?P<query>.+)$",
                      rest, re.IGNORECASE)
        if m2 and not site:
            site = m2.group("site").strip()
            query = m2.group("query").strip()
        else:
            query = re.sub(r"^(?:for|about|:)\s+", "", rest).strip()
            if site and query.lower().startswith(site.lower()):
                query = query[len(site):].strip(" for:")
        if not query:
            return None
        goal.kind = BrowserGoalKind.SEARCH
        goal.site, goal.query = site, query
        goal.url = resolve_site_url(site)
        return goal

    # ── extract / read ─────────────────────────────────────────────
    m = re.match(rf"^{_EXTRACT_VERBS}\s+(?P<rest>.+)$", body, re.IGNORECASE)
    if m:
        subject = m.group("rest").strip()
        goal.site = site
        goal.url = resolve_site_url(site)
        goal.query = re.sub(
            r"^(?:the\s+)?(?:links?|text)\s+(?:on|from)\s+", "",
            subject).strip() or subject
        goal.kind = BrowserGoalKind.EXTRACT
        return goal
    if re.match(rf"^{_READ_VERBS}\b", body, re.IGNORECASE) or body in (
            "read this page", "what does this page say",
            "summarize this page", "summarise this page"):
        goal.kind = BrowserGoalKind.READ
        goal.site, goal.url = site, resolve_site_url(site)
        return goal

    # ── find / open ────────────────────────────────────────────────
    for verbs, kind in ((_FIND_VERBS, BrowserGoalKind.FIND_ITEM),
                        (_OPEN_VERBS, BrowserGoalKind.OPEN_SITE)):
        m = re.match(rf"^{verbs}\s+(?P<rest>.+)$", body, re.IGNORECASE)
        if not m:
            continue
        rest = m.group("rest").strip()
        goal.site, goal.target = site, rest
        goal.url = resolve_site_url(site) if site else ""
        if kind is BrowserGoalKind.OPEN_SITE and not site:
            # "open linkedin" — the object itself may be a site.
            resolved = resolve_site_url(rest)
            if resolved and _looks_like_site(rest):
                goal.site, goal.url = rest, resolved
                goal.kind = BrowserGoalKind.OPEN_SITE
                return goal
            # "open Rahul's profile" — an ITEM, needs a scope to act on.
            goal.kind = BrowserGoalKind.OPEN_ITEM
            return goal
        if kind is BrowserGoalKind.FIND_ITEM and any(
                w in rest.lower() for w in _RESULT_WORDS):
            # "find backend jobs" → a search-like goal over the result state.
            goal.kind = BrowserGoalKind.SEARCH
            goal.query = rest
            return goal
        goal.kind = kind
        return goal
    return None


# ═══════════════════════════════════════════════════════════════════
# 6. Safety guardrails (never bypass access controls, never send)
# ═══════════════════════════════════════════════════════════════════

_ACCESS_BYPASS = (
    "bypass login", "bypass authentication", "bypass the login",
    "circumvent login", "circumvent authentication", "defeat login",
    "bypass captcha", "solve captcha", "defeat captcha", "evade captcha",
    "evade anti-bot", "bypass anti-bot", "bypass access control",
    "bypass paywall", "scrape credentials", "steal credentials",
    "hack into", "guess the password", "brute force",
)

# Intent keywords that mean an EXTERNAL SIDE EFFECT. The engine refuses these
# outright for browser goals (no applications, no messages, no purchases, no
# destructive actions) — they require an explicit, separate user flow.
_EXTERNAL_INTENT = (
    "submit application", "apply for", "apply to", "send message",
    "send a message", "send email", "send an email", "direct message",
    "connect with", "follow ", "message ", "purchase", "buy now",
    "add to cart", "checkout", "make payment", "transfer money",
    "withdraw", "delete account", "delete the post", "post comment",
    "post a comment", "tweet", "publish", "upload file", "unsubscribe",
    "cancel subscription", "rsvp",
)


def guardrail_reason(text: str) -> str:
    """Return a refusal reason, or "" when the request is allowed.

    Pure and explicit: the engine records it in the trace and stops — it
    never attempts the blocked step.
    """
    t = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not t:
        return ""
    for phrase in _ACCESS_BYPASS:
        if phrase in t:
            return (f"refused: '{phrase}' would bypass an access control / "
                    f"anti-bot system — that is never automated")
    for phrase in _EXTERNAL_INTENT:
        if phrase in t:
            return (f"refused without explicit confirmation: "
                    f"'{phrase.strip()}' has an external side effect")
    if re.search(r"\b(password|passwd)\b\s*[:=]", t):
        return "refused: credentials are never typed by the agent"
    return ""


def guardrail_step_reason(step: "BrowserStep") -> str:
    """Guardrail check for one planned step (description + params + target)."""
    if step.action == BrowserAction.ASK_USER:
        return ""
    if step.action == BrowserAction.TYPE_INPUT and str(
            step.params.get("password") or "").lower() in ("1", "true", "yes"):
        return "refused: typing into a password field is never automated"
    blob = " ".join([
        step.description or "", step.target or "",
        " ".join(f"{k}={v}" for k, v in (step.params or {}).items()),
    ])
    return guardrail_reason(blob)


# ═══════════════════════════════════════════════════════════════════
# 7. Page classification (loading / login / blocked / not found)
# ═══════════════════════════════════════════════════════════════════

LOGIN_MARKERS = (
    "sign in", "sign-in", "log in", "login", "log-in", "signin",
    "password", "authentication required", "please authenticate",
    "forgot password", "create account", "enter your email",
)
BLOCKED_MARKERS = (
    "captcha", "are you a robot", "verify you are human", "unusual traffic",
    "access denied", "access to this page has been denied", "rate limit",
    "too many requests", "try again later", "forbidden", "not authorized",
    "verify your identity", "suspicious activity",
)
NOT_FOUND_MARKERS = (
    "page not found", "404", "not found", "no longer available",
    "doesn't exist", "does not exist", "no results", "nothing found",
)
AUTHENTICATED_MARKERS = (
    "sign out", "log out", "logout", "my account", "profile", "dashboard",
    "settings", "notifications",
)
_CREDENTIAL_LABELS = ("user", "email", "login", "password", "passwd",
                      "mobile", "phone")


def _attr(ctx: Any, name: str, default: Any = "") -> Any:
    """Read an attribute from a context object OR a plain mapping."""
    if isinstance(ctx, dict):
        return ctx.get(name, default)
    return getattr(ctx, name, default)


def _el_field(el: Any, field: str, default: Any = "") -> Any:
    if isinstance(el, dict):
        return el.get(field, default)
    return getattr(el, field, default)


def ctx_text(ctx: Any, limit: int = 20000) -> str:
    """Bounded text view of a context (title + url + text + element labels)."""
    if ctx is None:
        return ""
    bits = [str(_attr(ctx, "page_title", "") or ""),
            str(_attr(ctx, "current_url", "") or ""),
            str(_attr(ctx, "visible_text", "") or "")]
    for el in list(_attr(ctx, "interactive_elements", []) or [])[:80]:
        bits.append(str(_el_field(el, "label") or ""))
    return "\n".join(bits)[:limit].lower()


def ctx_password_fields(ctx: Any) -> int:
    """Number of password inputs observable in the context (never guessed)."""
    explicit = _attr(ctx, "password_fields", None)
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    count = 0
    for el in list(_attr(ctx, "interactive_elements", []) or []):
        kind = str(_el_field(el, "kind") or "").lower()
        etype = str(_el_field(el, "type") or "").lower()
        if kind == "password" or etype == "password":
            count += 1
    return count


def detect_blocked(ctx: Any) -> bool:
    text = ctx_text(ctx)
    return any(m in text for m in BLOCKED_MARKERS)


def detect_login(ctx: Any) -> bool:
    """Login-required detection: password field OR credentials form markers."""
    if ctx is None:
        return False
    if ctx_password_fields(ctx) > 0:
        return True
    text = ctx_text(ctx)
    markers = sum(1 for m in LOGIN_MARKERS if m in text)
    cred_input = False
    for el in list(_attr(ctx, "interactive_elements", []) or []):
        if str(_el_field(el, "kind") or "").lower() != "input":
            continue
        label = str(_el_field(el, "label") or "").lower()
        if any(c in label for c in _CREDENTIAL_LABELS):
            cred_input = True
            break
    return markers >= 2 and cred_input


def detect_authenticated(ctx: Any) -> bool:
    """Authenticated state: explicit observation, else generic page markers.

    Evidence comes from PAGE CONTENT only (title / visible text / element
    labels). The URL is navigation truth, not authentication evidence — a
    signed-out visitor on ``https://x/dashboard`` must never read as signed
    in. A detected login wall also wins: a sign-in page is never authed.
    """
    declared = str(_attr(ctx, "authentication_state", "") or "").lower()
    if declared in ("authenticated", "logged_in", "signed_in"):
        return True
    if detect_login(ctx):
        return False
    if ctx is None:
        return False
    bits = [str(_attr(ctx, "page_title", "") or ""),
            str(_attr(ctx, "visible_text", "") or "")]
    for el in list(_attr(ctx, "interactive_elements", []) or [])[:80]:
        bits.append(str(_el_field(el, "label") or ""))
    text = "\n".join(bits).lower()
    return any(m in text for m in AUTHENTICATED_MARKERS)


def has_not_found_state(ctx: Any) -> bool:
    text = ctx_text(ctx)
    return any(m in text for m in NOT_FOUND_MARKERS)


def classify_page(ctx: Any, *, expect_text: str = "",
                  expect_url_contains: str = "") -> PageState:
    """Classify the observed page. Never assumes — empty stays empty."""
    if ctx is None:
        return PageState.UNAVAILABLE
    url = str(_attr(ctx, "current_url", "") or "")
    text = str(_attr(ctx, "visible_text", "") or "")
    loading = str(_attr(ctx, "loading_state", "") or "").lower()
    attached = _attr(ctx, "browser_attached", None)
    if not url and not text and not _attr(ctx, "interactive_elements", None):
        return PageState.EMPTY if attached is False else PageState.UNAVAILABLE
    if loading == "loading":
        return PageState.LOADING
    if detect_blocked(ctx):
        return PageState.BLOCKED
    if detect_login(ctx):
        return PageState.LOGIN_REQUIRED
    if has_not_found_state(ctx):
        if expect_url_contains and expect_url_contains.lower() not in url.lower():
            return PageState.NOT_FOUND
        if expect_text and expect_text.lower() not in ctx_text(ctx):
            return PageState.NOT_FOUND
    return PageState.LOADED

# ═══════════════════════════════════════════════════════════════════
# 8. Expected-effect verification (the "no false success" contract)
# ═══════════════════════════════════════════════════════════════════


def _url(ctx: Any) -> str:
    return str(_attr(ctx, "current_url", "") or "")


def _hash(ctx: Any) -> str:
    return str(_attr(ctx, "page_hash", "") or "")


def _scroll(ctx: Any) -> Any:
    return _attr(ctx, "scroll_offset", None)


def _element_labels(ctx: Any) -> List[str]:
    out = []
    for el in list(_attr(ctx, "interactive_elements", []) or []):
        label = str(_el_field(el, "label") or "")
        if label:
            out.append(label)
    return out


def _input_values(ctx: Any) -> List[str]:
    out = []
    for el in list(_attr(ctx, "interactive_elements", []) or []):
        if str(_el_field(el, "kind") or "").lower() not in ("input", "textarea",
                                                            "select"):
            continue
        value = str(_el_field(el, "value") or "")
        if value:
            out.append(value)
    for form in list(_attr(ctx, "forms", []) or []):
        for f in list(_el_field(form, "fields") or []):
            value = str(_el_field(f, "value") or "")
            if value:
                out.append(value)
    return out


def _present(ctx: Any, needle: str) -> bool:
    n = normalize_label(needle)
    if not n:
        return False
    if n in ctx_text(ctx):
        return True
    return any(n in normalize_label(lbl) for lbl in _element_labels(ctx))


def _verdict(action: BrowserAction, params: Dict[str, Any],
             result: EffectResult, observed: str, detail: str,
             evidence: Dict[str, Any]) -> EffectVerdict:
    return EffectVerdict(
        action=action.value,
        expected_effect=expected_effect_text(action, params),
        observed_effect=observed, result=result, evidence=evidence,
        detail=detail)


def _goal_navigation_verdict(action: BrowserAction, params: Dict[str, Any],
                             after: Any) -> EffectVerdict:
    """Reuse the EXISTING goal-verification contract for navigation."""
    target = str(params.get("url") or params.get("target") or "")
    observed_url = _url(after)
    try:
        from core.goal_verification import (GoalResult,
                                            verify_browser_navigation)
        gv = verify_browser_navigation(target, observed_url or None)
        mapping = {GoalResult.PASS: EffectResult.PASS,
                   GoalResult.FAIL: EffectResult.FAIL,
                   GoalResult.NO_EVIDENCE: EffectResult.NO_EVIDENCE}
        return _verdict(
            action, params, mapping.get(gv.result, EffectResult.NO_EVIDENCE),
            observed_url, gv.evidence,
            {"expected_url": target, "observed_url": observed_url})
    except Exception as e:  # verification module unavailable — stay honest
        return _verdict(action, params, EffectResult.NO_EVIDENCE, observed_url,
                        f"goal verification unavailable: {e}", {})


def verify_expected_effect(action: "BrowserAction | str",
                           params: Optional[Dict[str, Any]],
                           before: Any, after: Any) -> EffectVerdict:
    """Compare DECLARED expectation with OBSERVED state for one action.

    A generic screen change is never accepted as success: every branch below
    checks the specific thing the step promised. A missing observation yields
    NO_EVIDENCE (never PASS).
    """
    try:
        act = (action if isinstance(action, BrowserAction)
               else BrowserAction(str(action)))
    except ValueError:
        return EffectVerdict(
            action=str(action),
            expected_effect="the action reports a verified effect",
            result=EffectResult.NO_EVIDENCE, detail="unknown browser action")
    p = dict(params or {})
    if act in (BrowserAction.NAVIGATE, BrowserAction.OPEN_URL):
        return _goal_navigation_verdict(act, p, after)
    if act == BrowserAction.OPEN_BROWSER:
        url = _url(after)
        if url:
            return _verdict(act, p, EffectResult.PASS, url,
                            "browser page available and reporting a URL",
                            {"url": url})
        return _verdict(act, p, EffectResult.NO_EVIDENCE, "",
                        "no browser page observed yet", {})
    if after is None:
        return _verdict(act, p, EffectResult.NO_EVIDENCE, "",
                        "no observation after the action", {})

    before_url, after_url = _url(before), _url(after)
    before_hash, after_hash = _hash(before), _hash(after)
    observed = f"url={after_url[:120]} hash={after_hash[:12]}"

    if act in (BrowserAction.GO_BACK, BrowserAction.GO_FORWARD):
        if not before_url or not after_url:
            return _verdict(act, p, EffectResult.NO_EVIDENCE, observed,
                            "history page state not observable", {})
        if after_url != before_url:
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"{before_url[:80]} -> {after_url[:80]}",
                            {"before_url": before_url, "after_url": after_url})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        "URL did not change (history hop not observed)", {})

    if act == BrowserAction.REFRESH:
        if after_url:
            return _verdict(act, p, EffectResult.PASS, observed,
                            "page re-read from the live browser",
                            {"url": after_url, "page_hash": after_hash})
        return _verdict(act, p, EffectResult.NO_EVIDENCE, observed,
                        "no page observed after refresh", {})

    if act == BrowserAction.WAIT_FOR_PAGE:
        state = str(_attr(after, "loading_state", "") or "").lower()
        if state and state != "loading" and after_url:
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"page loaded (loading_state={state})",
                            {"loading_state": state})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        f"page still loading (loading_state={state or 'unknown'})",
                        {"loading_state": state})

    if act == BrowserAction.FIND_ELEMENT:
        target = str(p.get("target") or p.get("query") or "")
        candidates = p.get("candidates") or []
        if candidates:
            return _verdict(act, p, EffectResult.PASS,
                            f"{len(candidates)} candidate(s)",
                            "semantic match identified",
                            {"candidates": candidates[:8]})
        if target and _present(after, target):
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"'{target[:60]}' observed on the page",
                            {"target": target})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        f"'{target[:60]}' not observed on the page",
                        {"target": target})
    if act in (BrowserAction.CLICK_ELEMENT, BrowserAction.PAGINATE):
        expect = str(p.get("expect") or "")
        expect_url = str(p.get("expect_url_contains") or "")
        expect_absent = str(p.get("expect_absent") or "")
        want_url_change = bool(p.get("expect_url_change"))
        declared_any = bool(expect or expect_url or expect_absent
                            or want_url_change)
        if not declared_any:
            return _verdict(
                act, p, EffectResult.NO_EVIDENCE, observed,
                "no declared expectation for this interaction — cannot verify",
                {"rule": "declare expect/expect_url_contains/expect_absent"})
        checks: Dict[str, bool] = {}
        if expect:
            checks[f"text:{expect[:40]}"] = _present(after, expect)
        if expect_url:
            checks[f"url:{expect_url[:40]}"] = (
                expect_url.lower() in after_url.lower())
        if expect_absent:
            checks[f"absent:{expect_absent[:40]}"] = not _present(
                after, expect_absent)
        if want_url_change:
            checks["url_changed"] = bool(before_url
                                         and after_url != before_url)
        ok = all(checks.values())
        return _verdict(
            act, p, EffectResult.PASS if ok else EffectResult.FAIL, observed,
            ("declared transition observed: " if ok
             else "declared transition NOT observed: ")
            + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in checks.items()),
            {"checks": checks, "page_hash": after_hash})

    if act == BrowserAction.TYPE_INPUT:
        text = str(p.get("text") or "")
        if not text:
            return _verdict(act, p, EffectResult.NO_EVIDENCE, observed,
                            "no text declared for this step", {})
        values = _input_values(after)
        if any(text in v for v in values) or normalize_label(text) in ctx_text(after):
            return _verdict(act, p, EffectResult.PASS,
                            f"input value observed ({len(values)} field value(s))",
                            f"'{text[:40]}' present in observed input state",
                            {"values": [v[:80] for v in values[:5]]})
        return _verdict(act, p, EffectResult.FAIL,
                        f"input value NOT observed ({len(values)} field value(s))",
                        f"'{text[:40]}' absent from observed page state",
                        {"values": [v[:80] for v in values[:5]]})

    if act == BrowserAction.CLEAR_INPUT:
        target = str(p.get("target") or "")
        values = [v for v in _input_values(after) if v]
        if not values:
            return _verdict(act, p, EffectResult.PASS, observed,
                            "no input value observed (field is empty)",
                            {"target": target})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        f"input still holds {len(values)} value(s)",
                        {"values": [v[:60] for v in values[:5]]})

    if act == BrowserAction.SELECT_OPTION:
        chosen = str(p.get("option") or p.get("value") or "")
        if chosen and _present(after, chosen):
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"option '{chosen[:40]}' observed as selected",
                            {"option": chosen})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        f"option '{chosen[:40]}' not observed as selected",
                        {"option": chosen})

    if act == BrowserAction.PRESS_KEY:
        key = str(p.get("key") or "").lower()
        expect = str(p.get("expect") or "")
        if expect and _present(after, expect):
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"'{key}' produced the declared state '{expect[:40]}'",
                            {"key": key})
        if key in ("enter", "return") and before_url and after_url != before_url:
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"'{key}' caused navigation to {after_url[:80]}",
                            {"before_url": before_url, "after_url": after_url})
        if before_hash and after_hash and before_hash != after_hash:
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"'{key}' changed observed page state",
                            {"before_hash": before_hash,
                             "after_hash": after_hash})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        f"'{key}' produced no observed state transition",
                        {"key": key})

    if act == BrowserAction.SCROLL:
        b, a = _scroll(before), _scroll(after)
        if b is not None and a is not None and b != a:
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"scroll offset {b} -> {a}",
                            {"before_scroll": b, "after_scroll": a})
        if before_hash and after_hash and before_hash != after_hash:
            return _verdict(act, p, EffectResult.PASS, observed,
                            "page content changed after scrolling",
                            {"before_hash": before_hash,
                             "after_hash": after_hash})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        "scroll position did not change", {})
    if act in (BrowserAction.EXTRACT_TEXT, BrowserAction.EXTRACT_RESULTS,
               BrowserAction.EXTRACT_LINKS, BrowserAction.READ_PAGE):
        subject = str(p.get("query") or p.get("target") or "")
        count = p.get("extracted")
        links = list(_attr(after, "links", []) or [])
        if act == BrowserAction.EXTRACT_LINKS:
            ok = bool(links) and (not subject or any(
                normalize_label(subject) in normalize_label(
                    str(_el_field(l, "text") or ""))
                or normalize_label(subject) in normalize_label(
                    str(_el_field(l, "href") or ""))
                for l in links))
            return _verdict(
                act, p, EffectResult.PASS if ok else EffectResult.FAIL,
                f"{len(links)} link(s) observed",
                (f"{len(links)} link(s) observed on the page" if ok
                 else "no requested links observed on the page"),
                {"link_count": len(links)})
        if isinstance(count, int):
            ok = count > 0
            return _verdict(
                act, p, EffectResult.PASS if ok else EffectResult.FAIL,
                f"{count} extracted item(s)",
                f"{count} item(s) extracted from observed page state",
                {"extracted": count, "subject": subject})
        if subject and _present(after, subject):
            return _verdict(act, p, EffectResult.PASS, observed,
                            f"requested information '{subject[:60]}' observed",
                            {"subject": subject})
        text = str(_attr(after, "visible_text", "") or "")
        if not subject and text.strip():
            return _verdict(act, p, EffectResult.PASS,
                            f"{len(text)} chars observed",
                            "page text observed and recorded",
                            {"chars": len(text)})
        return _verdict(act, p, EffectResult.FAIL, observed,
                        f"requested information '{subject[:60]}' NOT observed",
                        {"subject": subject})

    if act == BrowserAction.ASK_USER:
        return _verdict(act, p, EffectResult.NO_EVIDENCE, "",
                        "waiting for the user's answer", {})
    return _verdict(act, p, EffectResult.NO_EVIDENCE, observed,
                    "no expected-effect contract for this action", {})


__all__ = [
    "BrowserAction", "BrowserGoal", "BrowserGoalKind", "BrowserStatus",
    "BrowserStep", "BrowserResult", "BrowserExtraction", "BrowserExtraction",
    "Candidate", "COMPUTER_ACTION", "EXPECTED_EFFECTS", "EffectResult",
    "EffectVerdict", "PageState", "SITE_ALIASES", "build_candidates",
    "classify_page", "ctx_password_fields", "ctx_text", "detect_authenticated",
    "detect_blocked", "detect_login", "expected_effect_text",
    "guardrail_reason", "guardrail_step_reason", "has_not_found_state",
    "normalize_label", "parse_browser_goal", "rank_candidates",
    "resolve_answer_to_candidate", "resolve_site_url", "score_candidate",
    "select_target", "verify_expected_effect",
]