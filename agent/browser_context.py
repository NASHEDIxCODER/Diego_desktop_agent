"""
Phase 23: structured BrowserContext — the browser half of ComputerState.

The agent reasons about a PAGE, never about a screenshot. This module turns
the existing browser perception tier (computer/browser_controller.py, tier 1
of the Phase 22 hierarchy) into one normalized, JSON-able record:

    current_url, domain, page_title, navigation_state, visible_text,
    interactive_elements, links, forms, focused_element, page_hash,
    last_navigation, loading_state, authentication_state

Design rules:
  * Tier 1 (browser/DOM) is used whenever a page is attached; tier 2
    (accessibility) and the existing desktop-state observer are used only as
    honest, clearly-labelled fallbacks (perception_method records which one
    answered — OCR/vision stay reserved for non-browser surfaces).
  * Nothing is invented: an unavailable browser yields an EMPTY context with
    ``browser_attached=False`` instead of fabricated fields.
  * No website knowledge whatsoever: only generic roles/labels/text.

Logging: [PERCEPTION]
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.browser_goal import (
    Candidate,
    PageState,
    classify_page,
    detect_authenticated,
    detect_login,
    normalize_label,
    rank_candidates,
)

logger = logging.getLogger(__name__)

LOADING = "loading"
LOADED = "loaded"


@dataclass
class InteractiveElement:
    """One generic interactive element observed on the page."""

    label: str = ""
    kind: str = "element"          # link | button | input | select | role…
    tag: str = ""
    type: str = ""
    name: str = ""
    value: str = ""
    href: str = ""
    placeholder: str = ""
    enabled: bool = True
    confidence: float = 0.9
    method: str = "browser_dom"

    @property
    def is_input(self) -> bool:
        return self.kind in ("input", "textarea", "select") or self.tag in (
            "input", "textarea", "select")

    @property
    def is_link(self) -> bool:
        return self.kind == "link" or self.tag == "a"

    @property
    def is_password(self) -> bool:
        return self.type.lower() == "password" or self.kind == "password"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "kind": self.kind, "tag": self.tag,
            "type": self.type, "name": self.name, "value": self.value[:200],
            "href": self.href[:300], "placeholder": self.placeholder,
            "enabled": self.enabled, "confidence": self.confidence,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractiveElement":
        return cls(
            label=str(data.get("label") or ""),
            kind=str(data.get("kind") or "element"),
            tag=str(data.get("tag") or ""),
            type=str(data.get("type") or ""),
            name=str(data.get("name") or ""),
            value=str(data.get("value") or ""),
            href=str(data.get("href") or ""),
            placeholder=str(data.get("placeholder") or ""),
            enabled=bool(data.get("enabled", True)),
            confidence=float(data.get("confidence", 0.9) or 0.9),
            method=str(data.get("method") or "browser_dom"),
        )


@dataclass
class BrowserForm:
    """One generic form observed on the page."""

    action: str = ""
    method: str = "get"
    fields: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_password_field(self) -> bool:
        return any(str(f.get("type", "")).lower() == "password"
                   for f in self.fields)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "method": self.method,
                "fields": [dict(f) for f in self.fields[:20]]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserForm":
        return cls(action=str(data.get("action") or ""),
                   method=str(data.get("method") or "get"),
                   fields=[dict(f) for f in (data.get("fields") or [])])


@dataclass
class BrowserContext:
    """Structured browser truth for one observation (never screenshot-only)."""

    current_url: str = ""
    domain: str = ""
    page_title: str = ""
    navigation_state: str = ""          # navigated | same_page | redirected
    visible_text: str = ""
    interactive_elements: List[InteractiveElement] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    forms: List[BrowserForm] = field(default_factory=list)
    focused_element: Dict[str, Any] = field(default_factory=dict)
    page_hash: str = ""
    last_navigation: str = ""
    loading_state: str = LOADED
    authentication_state: str = "unknown"   # authenticated | anonymous | unknown
    password_fields: int = 0
    scroll_offset: int = 0
    perception_method: str = "browser_dom"
    browser_attached: bool = False
    error: str = ""
    observed_at: float = field(default_factory=time.time)

    # ── Derived views ─────────────────────────────────────────────

    @property
    def page_state(self) -> PageState:
        return classify_page(self)

    @property
    def is_loading(self) -> bool:
        return self.loading_state == LOADING

    @property
    def login_required(self) -> bool:
        return detect_login(self)

    @property
    def authenticated(self) -> bool:
        return detect_authenticated(self)

    @property
    def is_empty(self) -> bool:
        return not (self.current_url or self.visible_text
                    or self.interactive_elements)

    def text_blob(self, limit: int = 8000) -> str:
        """Bounded searchable text: title + url + text + element labels."""
        parts = [self.page_title, self.current_url, self.visible_text]
        parts.extend(e.label for e in self.interactive_elements if e.label)
        parts.extend(str(l.get("text", "")) for l in self.links)
        return "\n".join(p for p in parts if p)[:limit]

    def contains(self, needle: str) -> bool:
        n = normalize_label(needle)
        return bool(n) and n in self.text_blob().lower()

    def elements_by_kind(self, *kinds: str) -> List[InteractiveElement]:
        wanted = {k.lower() for k in kinds}
        return [e for e in self.interactive_elements
                if e.kind.lower() in wanted or e.tag.lower() in wanted]

    def inputs(self) -> List[InteractiveElement]:
        return [e for e in self.interactive_elements if e.is_input]

    def buttons(self) -> List[InteractiveElement]:
        return [e for e in self.interactive_elements
                if e.kind.lower() in ("button", "submit") or e.tag == "button"]

    def candidates(self, target: str, limit: int = 12) -> List[Candidate]:
        """Semantic candidates for `target` from the observed elements."""
        raw = [{"label": e.label, "kind": e.kind, "href": e.href,
                "value": e.value, "method": e.method, "tag": e.tag}
               for e in self.interactive_elements
               if e.label and normalize_label(target) in normalize_label(e.label)]
        for link in self.links:
            text = str(link.get("text") or "")
            if text and normalize_label(target) in normalize_label(text):
                raw.append({"label": text, "kind": "link",
                            "href": str(link.get("href") or ""),
                            "method": "browser_dom"})
        return rank_candidates(target, raw[:max(limit * 3, 12)])[0][:limit]

    def summary(self) -> str:
        bits = [f"domain={self.domain or '?'}"]
        if self.page_title:
            bits.append(f"title='{self.page_title[:60]}'")
        bits.append(f"elements={len(self.interactive_elements)}")
        bits.append(f"links={len(self.links)}")
        bits.append(f"loading={self.loading_state}")
        bits.append(f"auth={self.authentication_state}")
        bits.append(f"method={self.perception_method}")
        return " ".join(bits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_url": self.current_url, "domain": self.domain,
            "page_title": self.page_title,
            "navigation_state": self.navigation_state,
            "visible_text": self.visible_text[:4000],
            "interactive_elements": [e.to_dict()
                                     for e in self.interactive_elements[:80]],
            "links": [dict(l) for l in self.links[:60]],
            "forms": [f.to_dict() for f in self.forms[:6]],
            "focused_element": dict(self.focused_element),
            "page_hash": self.page_hash,
            "last_navigation": self.last_navigation,
            "loading_state": self.loading_state,
            "authentication_state": self.authentication_state,
            "password_fields": self.password_fields,
            "scroll_offset": self.scroll_offset,
            "perception_method": self.perception_method,
            "browser_attached": self.browser_attached,
            "error": self.error,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserContext":
        return cls(
            current_url=str(data.get("current_url") or ""),
            domain=str(data.get("domain") or ""),
            page_title=str(data.get("page_title") or ""),
            navigation_state=str(data.get("navigation_state") or ""),
            visible_text=str(data.get("visible_text") or ""),
            interactive_elements=[InteractiveElement.from_dict(e) for e in
                                  (data.get("interactive_elements") or [])],
            links=[dict(l) for l in (data.get("links") or [])],
            forms=[BrowserForm.from_dict(f) for f in (data.get("forms") or [])],
            focused_element=dict(data.get("focused_element") or {}),
            page_hash=str(data.get("page_hash") or ""),
            last_navigation=str(data.get("last_navigation") or ""),
            loading_state=str(data.get("loading_state") or LOADED),
            authentication_state=str(
                data.get("authentication_state") or "unknown"),
            password_fields=int(data.get("password_fields") or 0),
            scroll_offset=int(data.get("scroll_offset") or 0),
            perception_method=str(data.get("perception_method") or "browser_dom"),
            browser_attached=bool(data.get("browser_attached", False)),
            error=str(data.get("error") or ""),
            observed_at=float(data.get("observed_at") or time.time()),
        )


# ══════════════════════════════════════════════════════════════════
# Observer — builds a BrowserContext through the perception hierarchy
# ══════════════════════════════════════════════════════════════════


def domain_of(url: str) -> str:
    m = None
    try:
        import re
        m = re.match(r"https?://([^/]+)", str(url or ""), re.IGNORECASE)
    except Exception:
        return ""
    return (m.group(1) if m else "").lower().replace("www.", "")


def _page_hash(material: str, url: str, title: str, text: str = "") -> str:
    base = material or f"{url}|{title}|{len(text)}"
    return hashlib.md5(base.encode("utf-8", "ignore")).hexdigest()[:16]


def _auth_state(ctx: "BrowserContext") -> str:
    """Authentication state from OBSERVED structure (`unknown` otherwise)."""
    if detect_login(ctx):
        return "anonymous"
    if detect_authenticated(ctx):
        return "authenticated"
    return "unknown"


def build_context(structure: Dict[str, Any]) -> BrowserContext:
    """Build a BrowserContext from computer.browser_controller structure."""
    elements = [InteractiveElement.from_dict({
        "label": str(e.get("label") or ""),
        "kind": str(e.get("kind") or "element"),
        "tag": str(e.get("tag") or ""),
        "type": str(e.get("type") or ""),
        "name": str(e.get("name") or ""),
        "value": str(e.get("value") or ""),
        "href": str(e.get("href") or ""),
        "placeholder": str(e.get("placeholder") or ""),
        "enabled": bool(e.get("enabled", True)),
    }) for e in (structure.get("elements") or [])]
    url = str(structure.get("url") or "")
    title = str(structure.get("title") or "")
    text = str(structure.get("text") or "")
    ctx = BrowserContext(
        current_url=url,
        domain=domain_of(url),
        page_title=title,
        visible_text=text,
        interactive_elements=elements,
        links=[{"text": str(l.get("text") or ""),
                "href": str(l.get("href") or "")}
               for l in (structure.get("links") or [])],
        forms=[BrowserForm.from_dict(f) for f in (structure.get("forms") or [])],
        focused_element=dict(structure.get("focused") or {}),
        page_hash=_page_hash(str(structure.get("hash_material") or ""),
                             url, title, text),
        loading_state=LOADING if structure.get("loading") else LOADED,
        password_fields=int(structure.get("password_fields") or 0),
        scroll_offset=int(structure.get("scroll") or 0),
        perception_method="browser_dom",
        browser_attached=True,
        error=str(structure.get("error") or ""),
    )
    if not ctx.password_fields:
        ctx.password_fields = sum(1 for e in elements if e.is_password)
    ctx.authentication_state = _auth_state(ctx)
    return ctx


# ══════════════════════════════════════════════════════════════════
# Perception hierarchy for browser observation
#
#   1. browser/DOM        → structured page truth (preferred)
#   2. accessibility      → generic a11y elements + desktop-state url
#   3. nothing observable → EMPTY context (never fabricated)
#
# OCR/vision are NOT used for browser pages: structured data answers first,
# and the trace records which tier answered every single observation.
# ══════════════════════════════════════════════════════════════════


class BrowserContextObserver:
    """Builds BrowserContext objects from the live perception hierarchy."""

    def __init__(self, *, max_text: int = 6000, max_elements: int = 120) -> None:
        self._max_text = max_text
        self._max_elements = max_elements
        self._last: Optional[BrowserContext] = None

    # ── public ────────────────────────────────────────────────────

    def observe(self, note: str = "") -> BrowserContext:
        """One observation. Never raises; degrades honestly."""
        ctx = self._observe_dom()
        if ctx is None:
            ctx = self._observe_accessibility()
        if ctx is None:
            ctx = BrowserContext(
                perception_method="unavailable", browser_attached=False,
                error="no browser page observable")
        self._annotate_navigation(ctx)
        logger.info("[PERCEPTION] observe: %s", ctx.summary())
        return ctx

    def last(self) -> Optional[BrowserContext]:
        return self._last

    def note_navigation(self, url: str, state: str = "navigated") -> None:
        """Record an intentional navigation (the engine knows the target)."""
        if self._last is not None:
            self._last.last_navigation = url
            self._last.navigation_state = state

    # ─ tiers ─────────────────────────────────────────────────────

    def _observe_dom(self) -> Optional[BrowserContext]:
        """Tier 1: structured DOM via computer.browser_controller."""
        try:
            from computer import browser_controller as bctl
            if not bctl.attached():
                return None
            structure = bctl.get_page_structure(
                max_text=self._max_text, max_elements=self._max_elements)
            if not structure.get("attached"):
                return None
            return build_context(structure)
        except Exception as e:
            logger.debug("[PERCEPTION] tier=dom observe failed: %s", e)
            return None

    def _observe_accessibility(self) -> Optional[BrowserContext]:
        """Tier 2: accessibility elements (+ desktop-state url/title)."""
        elements: List[InteractiveElement] = []
        try:
            from computer import accessibility as a11y
            for el in a11y.clickable(limit=self._max_elements):
                data = el.to_dict()
                elements.append(InteractiveElement(
                    label=str(data.get("name") or ""),
                    kind="input" if data.get("text_input") else "button",
                    tag="",
                    enabled=bool(data.get("enabled", True)),
                    confidence=float(data.get("confidence", 0.9) or 0.9),
                    method=str(data.get("method") or "accessibility"),
                ))
        except Exception as e:
            logger.debug("[PERCEPTION] tier=accessibility observe failed: %s", e)
        url = title = ""
        try:
            from services.desktop_state import desktop_state
            snap = desktop_state.snapshot()
            url = str(getattr(snap, "browser_url", "") or "")
            title = str(getattr(snap, "browser_tab", "") or "")
        except Exception:
            pass
        if not elements and not url and not title:
            return None
        ctx = BrowserContext(
            current_url=url, domain=domain_of(url), page_title=title,
            interactive_elements=elements,
            page_hash=_page_hash("", url, title),
            perception_method="accessibility",
            browser_attached=bool(url),
        )
        ctx.authentication_state = _auth_state(ctx)
        return ctx

    # ── bookkeeping ──────────────────────────────────────────────

    def _annotate_navigation(self, ctx: BrowserContext) -> None:
        prev = self._last
        if prev is not None:
            if (ctx.current_url and prev.current_url
                    and ctx.current_url != prev.current_url):
                ctx.navigation_state = "navigated"
                ctx.last_navigation = f"{prev.current_url[:120]} -> {ctx.current_url[:120]}"
            elif ctx.page_hash and prev.page_hash and ctx.page_hash != prev.page_hash:
                ctx.navigation_state = "same_page"
            else:
                ctx.navigation_state = "unchanged"
        elif ctx.current_url:
            ctx.navigation_state = "initial"
        self._last = ctx


def observer() -> BrowserContextObserver:
    """A fresh observer bound to the live perception hierarchy."""
    return BrowserContextObserver()


__all__ = [
    "BrowserContext", "BrowserContextObserver", "BrowserForm",
    "InteractiveElement", "LOADED", "LOADING", "build_context", "domain_of",
    "observer",
]