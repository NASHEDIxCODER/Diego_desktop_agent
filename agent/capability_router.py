"""
Phase 24.6 §3/§5: GENERIC AGENT GOAL ROUTER (PURE — no I/O, no side effects).

A capability resolver: goal → required capabilities → execution engine.

The observed production failure was:

    "Open telegram and send message to Delhi"
    → classified MULTI_STEP_TASK
    → routed into BrowserGoalEngine
    → failed there.

That is WRONG. Routing must use APPLICATION/CAPABILITY IDENTITY, not verbs.
The words "send", "message", "search", "open" do NOT select the engine — the
named application does:

    "search LinkedIn for jobs"            → browser.search    → BrowserGoalEngine
    "open Telegram"                       → desktop.open_app  → existing app resolver
    "open Telegram and send a message"    → desktop.*         → DesktopGoalEngine
    "search Google, then send the result on Telegram"
                                          → cross-app plan:
                                              browser subgoal (search)
                                              desktop subgoal (telegram)

This module NEVER executes anything and contains NO per-app if-chains: app
identity is resolved through the declared app tables (the messaging registry
in agent/desktop_goal.py and the generic desktop-app table here), and browser
identity through agent/browser_goal.py's site resolver. New applications are
added by REGISTERING an alias, not by writing a new branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# 1. Route model
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RouteSubgoal:
    """One capability-scoped piece of the user's goal."""

    engine: str                 # "browser" | "desktop" | "perception" | "none"
    capability: str             # e.g. "browser.search", "desktop.open_app",
                                # "telegram.send_message", "perception.screen_state"
    text: str = ""              # the clause this subgoal was derived from
    app: str = ""               # desktop app identity (canonical id)
    site: str = ""              # browser site identity

    def describe(self) -> str:
        return f"{self.engine}:{self.capability}"


@dataclass
class RouteDecision:
    """The routing verdict for one utterance."""

    kind: str                   # "browser" | "desktop" | "cross_app" |
                                # "perception" | "none"
    subgoals: List[RouteSubgoal] = field(default_factory=list)
    reason: str = ""

    @property
    def engines(self) -> List[str]:
        seen: List[str] = []
        for s in self.subgoals:
            if s.engine not in seen:
                seen.append(s.engine)
        return seen

    def is_desktop_identity(self) -> bool:
        """True when the goal REQUIRES the desktop engine (whole or part).

        A desktop-identity goal must NEVER be handed to BrowserGoalEngine
        merely because it contains words like "send" or "message".
        """
        return any(s.engine == "desktop" for s in self.subgoals)


# ═══════════════════════════════════════════════════════════════════
# 2. App identity tables (extensible by REGISTRATION, not if-chains)
# ═══════════════════════════════════════════════════════════════════

# Generic desktop applications (opened through the existing app resolver).
# Messaging apps live in agent/desktop_goal.MESSAGING_APPS and are merged in.
_DESKTOP_APP_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("vscode", ("vscode", "vs code", "visual studio code", "code editor")),
    ("terminal", ("terminal", "console", "kitty", "konsole")),
    ("files", ("files", "file manager", "nautilus", "dolphin", "thunar")),
    ("settings", ("settings", "system settings", "control center")),
    ("calculator", ("calculator",)),
    ("spotify", ("spotify",)),
    ("vlc", ("vlc",)),
    ("gimp", ("gimp",)),
    ("libreoffice", ("libreoffice", "writer", "calc", "impress")),
    ("firefox", ("firefox", "firefox-esr")),
    ("chrome", ("chrome", "chromium", "google chrome")),
)

_MESSAGING_APPS_FALLBACK: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("telegram", ("telegram", "tg")),
    ("whatsapp", ("whatsapp",)),
    ("discord", ("discord",)),
    ("slack", ("slack",)),
    ("signal", ("signal",)),
)


def _aliases() -> List[Tuple[str, Tuple[str, ...]]]:
    """Merged app table: the messaging registry first (authoritative), then
    the generic desktop table. Imported lazily so this module stays
    import-safe without the agent package initialized."""
    try:
        from agent.desktop_goal import MESSAGING_APPS  # type: ignore
        messaging = [(canon, tuple(aliases))
                     for canon, aliases in MESSAGING_APPS.items()]
    except Exception:
        messaging = list(_MESSAGING_APPS_FALLBACK)
    return list(messaging) + list(_DESKTOP_APP_ALIASES)


_APP_TABLE_CACHE: Optional[List[Tuple[str, List[str]]]] = None


def _app_match(text_low: str) -> Tuple[str, str]:
    """Return ``(canonical, display)`` for the first app named in `text_low`.

    Longest alias wins ("visual studio code" beats "code"). Empty canonical
    means no app is named.
    """
    global _APP_TABLE_CACHE
    if _APP_TABLE_CACHE is None:
        _APP_TABLE_CACHE = [
            (canon, sorted({a for a in aliases if a}, key=len, reverse=True))
            for canon, aliases in _aliases()
        ]
    best: Tuple[str, str, int] = ("", "", -1)
    for canon, aliases in _APP_TABLE_CACHE:
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text_low):
                if len(alias) > best[2]:
                    best = (canon, canon.title(), len(alias))
    return best[0], best[1]


# ── browser identity ────────────────────────────────────────────────
_SEARCH_VERBS_RE = re.compile(
    r"\b(?:search|look\s+up|google|find|query|browse)\b", re.IGNORECASE)
_BROWSER_HINT_RE = re.compile(
    r"\b(?:website|web|page|site|online|internet|url|link)\b", re.IGNORECASE)


def _site_match(clause_low: str) -> str:
    """Return the known site named in the clause (via the browser goal's
    site resolver), or ""."""
    try:
        from agent.browser_goal import _looks_like_site, resolve_site_url
    except Exception:
        return ""
    for token in re.split(r"[^a-z0-9\-\.\+]+", clause_low):
        token = token.strip(".,!?")
        if len(token) < 3:
            continue
        try:
            if _looks_like_site(token) and resolve_site_url(token):
                return token
        except Exception:
            continue
    return ""


# ── screen-state / perception identity ──────────────────────────────
_SCREEN_QUESTION_RE = re.compile(
    r"\b(?:what(?:'s| is)?|which)\s+(?:app|window|application)s?\s+"
    r"(?:is|are|am)\s+(?:open|active|focused|running)\b"
    r"|\bwhat(?:'s| is)?\s+(?:open|running)\b"
    r"|\b(?:do|did|can|could)\s+you\s+see\b.+\b(?:screen|display|monitor)\b"
    r"|\bwhat(?:'s| is)?\s+(?:on|shown|displayed)\b.+\b(?:screen|display|monitor)\b"
    r"|\bclick\b.+\bon\s+(?:my\s+|the\s+)?screen\b",
    re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════
# 3. Clause splitting + classification
# ═══════════════════════════════════════════════════════════════════

_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,\s*|\band\s+then\b|\bthen\b|\band after that\b|\band\b|;"
    r"|\bafter that\b)\s*", re.IGNORECASE)


def split_goal_clauses(text: str) -> List[str]:
    """Split a compound utterance into goal clauses (pure, conservative).

    "open telegram and send message to Delhi"
        → ["open telegram", "send message to Delhi"]
    """
    raw = str(text or "").strip().rstrip(".!?")
    if not raw:
        return []
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(raw) if p.strip()]
    return parts or [raw]


def _message_verb(clause_low: str) -> str:
    for v in ("send", "message", "text", "dm", "tell", "write to", "msg"):
        if re.search(rf"(^|\s){re.escape(v)}(\s|$)", clause_low):
            return v
    return ""


def _read_verb(clause_low: str) -> bool:
    return bool(re.search(r"\b(?:read|check|show me|open my)\b", clause_low,
                          re.IGNORECASE))


def _open_verb(clause_low: str) -> bool:
    return bool(re.match(r"^\s*(?:please\s+)?(?:open|launch|start|run|use)\b",
                         clause_low, re.IGNORECASE))


def classify_clause(clause: str) -> RouteSubgoal:
    """Classify ONE clause by application/capability identity."""
    low = re.sub(r"\s+", " ", str(clause or "").lower()).strip()

    # 1. Screen-state / perception questions are computer-use tasks.
    if _SCREEN_QUESTION_RE.search(low):
        return RouteSubgoal(engine="perception",
                            capability="perception.screen_state",
                            text=clause)

    app, _display = _app_match(low)
    site = _site_match(low)

    # 2. Desktop-app identity wins: the named application selects the engine.
    if app:
        if _message_verb(low):
            return RouteSubgoal(engine="desktop",
                                capability=f"{app}.send_message",
                                text=clause, app=app)
        if _read_verb(low):
            return RouteSubgoal(engine="desktop",
                                capability=f"{app}.read_messages",
                                text=clause, app=app)
        if _open_verb(low):
            return RouteSubgoal(engine="desktop", capability="desktop.open_app",
                                text=clause, app=app)
        return RouteSubgoal(engine="desktop",
                            capability=f"desktop.{app}", text=clause, app=app)

    # 3. Browser identity: a known site, or explicit search phrasing.
    if site:
        if _message_verb(low):
            # A send clause with no app word stays desktop (it consumes a
            # desktop skill; the app is resolved by the desktop engine).
            return RouteSubgoal(engine="desktop",
                                capability="desktop.send_message",
                                text=clause, app="")
        if _SEARCH_VERBS_RE.search(low):
            return RouteSubgoal(engine="browser", capability="browser.search",
                                text=clause, site=site)
        return RouteSubgoal(engine="browser", capability="browser.open_site",
                            text=clause, site=site)
    if _SEARCH_VERBS_RE.search(low) or _BROWSER_HINT_RE.search(low):
        return RouteSubgoal(engine="browser", capability="browser.search",
                            text=clause, site=site)

    # 4. Bare messaging verbs ("send Rahul a message") are desktop identity —
    #    the app is resolved by the desktop engine / asked of the user.
    if _message_verb(low):
        return RouteSubgoal(engine="desktop", capability="desktop.send_message",
                            text=clause, app="")

    # 5. Unknown — no engine identity.
    return RouteSubgoal(engine="none", capability="", text=clause)


def resolve_route(text: str) -> RouteDecision:
    """goal → required capabilities → execution engine (PURE).

    Single-engine goals stay single-engine. Mixed browser+desktop goals are a
    CROSS_APP plan: browser subgoals first (they produce artifacts), desktop
    subgoals after (they consume them).
    """
    clauses = split_goal_clauses(text)
    subgoals = [classify_clause(c) for c in clauses]
    subgoals = [s for s in subgoals if s.engine != "none"]
    if not subgoals:
        return RouteDecision(kind="none", subgoals=[],
                             reason="no application/capability identity found")
    engines = [s.engine for s in subgoals]
    if len(set(engines)) == 1:
        kind = engines[0]
        reason = f"single-engine {kind} goal: " + ", ".join(
            s.describe() for s in subgoals)
    else:
        kind = "cross_app"
        reason = ("cross-app plan (browser artifacts → desktop consumption): "
                  + " | ".join(s.describe() for s in subgoals))
    return RouteDecision(kind=kind, subgoals=subgoals, reason=reason)


__all__ = [
    "RouteSubgoal", "RouteDecision", "split_goal_clauses", "classify_clause",
    "resolve_route",
]
