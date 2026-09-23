"""
Phase 22: browser perception/action tier (tier 1 of the hierarchy).

Wraps the existing agent.browser manager (CDP-attached Chrome). This module
never launches its own browser or invents selectors — everything goes through
the already-running integration. Site-specific behavior is FORBIDDEN here:
only generic DOM operations (find by visible text, click, type, scroll).

The browser tier is the HIGHEST-confidence element source: when an attached
browser can answer "where is this element", no accessibility/OCR/vision work
is needed.

Logging: [PERCEPTION]
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from computer.action_result import ActionOutcome, ActionResult

logger = logging.getLogger(__name__)


def _browser():
    """Duck-load the existing browser singleton (name may evolve).

    The canonical singleton is ``agent.browser.browser_controller``; the
    historical names ("browser", "browser_manager") stay supported so this
    tier keeps working if that module is reorganised again.
    """
    try:
        import agent.browser as _m
        for name in ("browser_controller", "browser", "browser_manager"):
            obj = getattr(_m, name, None)
            if obj is not None and hasattr(obj, "navigate"):
                return obj
    except Exception as e:
        logger.debug("[PERCEPTION] browser tier unavailable: %s", e)
    return None


def _flag(obj: Any, name: str) -> bool:
    """Read an availability flag that may be a plain attribute OR a method."""
    try:
        value = getattr(obj, name, None)
        if value is None:
            return False
        return bool(value() if callable(value) else value)
    except Exception:
        return False


# ── Externally managed page (validation harnesses / tests) ─────────
# Allows a page driven by another owner (e.g. a Playwright context created
# by a validation script) to be used by this tier WITHOUT reimplementing
# any browser logic here. Production never needs it: the CDP-attached
# agent.browser singleton is used automatically.
_external_page: Any = None


def bind_page(page: Any) -> None:
    """Use `page` (Playwright-like: evaluate/url/title/wait_for_load_state)."""
    global _external_page
    _external_page = page
    logger.info("[PERCEPTION] browser tier bound to externally managed page")


def unbind_page() -> None:
    global _external_page
    _external_page = None


def _page() -> Any:
    """The page this tier should read: external binding first, else attach."""
    if _external_page is not None:
        return _external_page
    b = _browser()
    if b is None:
        return None
    for attr in ("_page", "_page_obj"):
        page = getattr(b, attr, None)
        if page is not None:
            return page
    return None


def attached() -> bool:
    if _external_page is not None:
        return True
    b = _browser()
    if b is None:
        return False
    if not _flag(b, "is_attached"):
        return False
    return _flag(b, "is_available") or _page() is not None


def _jsonify(raw: Any) -> Any:
    """`evaluate` may return a JSON string or an already-decoded value."""
    if raw is None or isinstance(raw, (dict, list)):
        return raw
    try:
        import json
        return json.loads(raw)
    except Exception:
        return None


def evaluate(script: str, *, default: Any = None) -> Any:
    """Run one script on the attached/bound page (duck-typed, decoded).

    Works for BOTH the CDP-attached browser manager and an externally bound
    Playwright-like page, so the DOM tier never needs a second code path and
    no backend object is accessed directly.
    """
    page = _page()
    if page is None:
        return default
    try:
        value = _jsonify(page.evaluate(script))
    except Exception as e:
        logger.debug("[PERCEPTION] page evaluate failed: %s", e)
        return default
    return default if value is None else value


def _wait_loaded(page: Any, timeout_ms: int = 8000) -> None:
    """Best-effort wait for the bound page to settle (never raises)."""
    try:
        wait = getattr(page, "wait_for_load_state", None)
        if callable(wait):
            wait("load", timeout=timeout_ms)
    except Exception as e:
        logger.debug("[PERCEPTION] wait_for_load_state: %s", e)


def _page_url_title(page: Any) -> Dict[str, str]:
    """url/title of a bound page without assuming a specific backend."""
    url = title = ""
    try:
        url = str(getattr(page, "url", "") or "")
    except Exception:
        url = ""
    try:
        title = str(page.title() or "")
    except Exception:
        title = ""
    return {"url": url, "title": title}


def open_url(url: str, *, action: str = "open_url",
             target: str = "") -> ActionResult:
    """Navigate to `url` and wait for load.

    Uses the CDP-attached browser manager when available; otherwise falls back
    to the externally bound page (validation harnesses / tests) so navigation
    is still real, never simulated.
    """
    b = _browser()
    page = _page()
    if b is None and page is None:
        return ActionResult(action=action, target=target, method="browser_dom",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="browser integration unavailable")
    t0 = time.time()
    try:
        # The externally bound page (validation harness) is the CURRENT owner
        # of the tab: when it exists it wins, even if the CDP singleton is
        # present but not attached (its navigate() would honestly refuse).
        if page is not None:
            page.goto(url)
            _wait_loaded(page)
            evidence = _page_url_title(page)
        else:
            if not b.navigate(url):
                return ActionResult(
                    action=action, target=target, method="browser_dom",
                    outcome=ActionOutcome.FAILED,
                    error=f"navigate failed: {url}", latency_ms=_ms(t0))
            b.wait_for_load()
            evidence = {"url": str(b.get_current_url() or ""),
                        "title": str(b.get_page_title() or "")}
        return ActionResult(
            action=action, target=target or url, method="browser_dom",
            success=True, outcome=ActionOutcome.SUCCESS, evidence=evidence,
            verification=f"url={evidence['url'][:120]}",
            latency_ms=_ms(t0))
    except Exception as e:
        return ActionResult(action=action, target=target, method="browser_dom",
                            outcome=ActionOutcome.FAILED, error=str(e),
                            latency_ms=_ms(t0))


def get_page_state(max_text: int = 2000) -> Dict[str, Any]:
    """Structured page state (url, title, bounded visible text).

    Answers for the CDP-attached browser AND for a bound page, so the
    Phase 22 ComputerState/delta verification works on a real browser page
    instead of degrading to an empty screenshot-only delta.
    """
    b = _browser()
    page = _page()
    if b is None and page is None:
        return {"attached": False}
    if b is None:
        try:
            info = _page_url_title(page)
            text = evaluate("() => (document.body ? document.body.innerText : '')")
            return {"attached": True, "url": info["url"], "title": info["title"],
                    "visible_text": str(text or "")[:max_text],
                    "perception_method": "browser_dom"}
        except Exception as e:
            logger.debug("[PERCEPTION] bound page state failed: %s", e)
            return {"attached": True, "error": str(e)}
    try:
        text = ""
        if attached():
            text = str(b.get_text("body") or "")[:max_text]
        return {
            "attached": True,
            "url": str(b.get_current_url() or ""),
            "title": str(b.get_page_title() or ""),
            "visible_text": text,
            "perception_method": "browser_dom",
        }
    except Exception as e:
        logger.debug("[PERCEPTION] page state failed: %s", e)
        return {"attached": True, "error": str(e)}


def click(selector: str, *, target: str = "") -> ActionResult:
    return _dom_op("click", target or selector,
                   lambda b: b.click(selector))


_CLICK_TEXT_SCRIPT = """
() => {
  const want = __TEXT__.toLowerCase();
  const sel = 'a, button, input[type=submit], input[type=button], ' +
              '[role=button], [role=link], [role=tab], [role=menuitem], summary';
  const els = [...document.querySelectorAll(sel)];
  const el = els.find(e => (((e.innerText || e.value
      || e.getAttribute('aria-label') || '') + '').toLowerCase().includes(want)));
  if (!el) return JSON.stringify({ok: false, reason: 'no element with that text'});
  try { el.scrollIntoView({block: 'center'}); } catch (e) {}
  el.click();
  return JSON.stringify({ok: true,
    label: ((el.innerText || el.value || '') + '').trim().slice(0, 120),
    href: ((el.href || '') + '').slice(0, 400)});
}
"""

_TYPE_TEXT_SCRIPT = """
() => {
  const want = __TARGET__.toLowerCase();
  const value = __VALUE__;
  const txt = (e) => ((e.getAttribute('aria-label') || e.getAttribute('placeholder')
      || e.getAttribute('name') || e.getAttribute('type') || '') + '').toLowerCase();
  const els = [...document.querySelectorAll('input, textarea')];
  if (!want && els.length > 1)
    return JSON.stringify({ok: false, reason: 'no target given for a page with several inputs'});
  let el = els.find(e => txt(e).includes(want));
  if (!el && els.length === 1) el = els[0];
  if (!el) return JSON.stringify({ok: false, reason: 'no input matched the target'});
  try { el.focus(); } catch (e) {}
  el.value = value;
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return JSON.stringify({ok: true, value: ((el.value || '') + '').slice(0, 200),
                         label: txt(el)});
}
"""

_PRESS_KEY_SCRIPT = """
() => {
  const key = __KEY__.toLowerCase();
  const el = document.activeElement;
  if (key === 'enter' || key === 'return') {
    const form = (el && el.form) ? el.form
        : (el && el.closest ? el.closest('form') : null);
    if (form) {
      try {
        if (form.requestSubmit) { form.requestSubmit(); }
        else { form.submit(); }
        return JSON.stringify({ok: true, submitted: true});
      } catch (e) {
        return JSON.stringify({ok: false, reason: 'submit failed: ' + e});
      }
    }
  }
  if (!el) return JSON.stringify({ok: false, reason: 'nothing focused in the page'});
  const norm = {enter: 'Enter', return: 'Enter', escape: 'Escape', esc: 'Escape'}[key] || key;
  const opts = {key: norm, bubbles: true, cancelable: true};
  el.dispatchEvent(new KeyboardEvent('keydown', opts));
  el.dispatchEvent(new KeyboardEvent('keypress', opts));
  el.dispatchEvent(new KeyboardEvent('keyup', opts));
  return JSON.stringify({ok: true, key: norm});
}
"""


def _bound_result(action: str, target: str, data: Any, page: Any,
                  t0: float, *, verb: str) -> ActionResult:
    """Uniform ActionResult for a DOM operation performed on a bound page."""
    ok = bool(data.get("ok")) if isinstance(data, dict) else False
    evidence = dict(data) if isinstance(data, dict) else {}
    try:
        evidence["url"] = _page_url_title(page)["url"]
    except Exception:
        pass
    return ActionResult(
        action=action, target=target, method="browser_dom", success=ok,
        outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
        evidence=evidence,
        verification=(f"{verb} '{target[:60]}'" if ok else ""),
        error="" if ok else str(evidence.get("reason") or f"{verb} failed"),
        latency_ms=_ms(t0))


def click_text(text: str, *, target: str = "") -> ActionResult:
    """Click an element by its visible text (tier 1 DOM, no coordinates).

    Runs the DOM script for both the CDP-attached browser and a bound page;
    the CSS/text click of the browser manager stays as a second chance.
    """
    b = _browser()
    page = _page()
    if b is None and page is None:
        return ActionResult(action="click", target=target or text,
                            method="browser_dom",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="browser integration unavailable")
    t0 = time.time()
    data = evaluate(_CLICK_TEXT_SCRIPT.replace("__TEXT__", repr(str(text).lower())))
    result = _bound_result("click", target or text, data, page, t0,
                           verb="clicked")
    if result.success:
        _wait_loaded(page)
        return result
    if b is not None:
        legacy = _dom_op("click", target or text, lambda b: b.click_text(text))
        if legacy.success:
            return legacy
    return result


def type_text(text: str, selector: Optional[str] = None, *,
              target: str = "") -> ActionResult:
    """Type into an input (tier 1 DOM, by semantic label — never blind typing)."""
    b = _browser()
    page = _page()
    label = str(target or selector or "")
    if b is None and page is None:
        return ActionResult(action="type_text", target=label or text,
                            method="browser_dom",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="browser integration unavailable")
    t0 = time.time()
    script = (_TYPE_TEXT_SCRIPT
              .replace("__TARGET__", repr(label.lower()))
              .replace("__VALUE__", repr(str(text))))
    result = _bound_result("type_text", label or text, evaluate(script), page,
                           t0, verb="typed into")
    if result.success:
        return result
    if b is not None:
        legacy = _dom_op("type_text", label or text,
                         lambda b: b.type_text(text, selector))
        if legacy.success:
            return legacy
    return result


def press_key(key: str) -> ActionResult:
    """Press a key (tier 1 DOM for the page, else the browser manager).

    Inside a form, ``enter`` submits that form through the DOM — the generic
    behaviour every search box relies on, with no site-specific knowledge.
    """
    b = _browser()
    page = _page()
    if b is None and page is None:
        return ActionResult(action="press_key", target=key,
                            method="browser_dom",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="browser integration unavailable")
    t0 = time.time()
    data = evaluate(_PRESS_KEY_SCRIPT.replace("__KEY__", repr(str(key).lower())))
    result = _bound_result("press_key", key, data, page, t0, verb="pressed key")
    if result.success:
        _wait_loaded(page)
        return result
    if b is not None:
        legacy = _dom_op("press_key", key, lambda b: b.press_key(key))
        if legacy.success:
            return legacy
    return result


def _history_op(action: str, expr: str) -> ActionResult:
    """back/forward/refresh via page history (generic, no site specifics)."""
    b = _browser()
    page = _page()
    if b is None and page is None:
        return ActionResult(action=action, method="browser_dom",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="browser integration unavailable")
    t0 = time.time()
    try:
        # A bound page is the current owner of the tab (see open_url).
        if page is not None:
            before = _page_url_title(page)["url"]
            evaluate(f"() => {expr}")
            _wait_loaded(page)
            after = _page_url_title(page)["url"]
        else:
            before = str(b.get_current_url() or "")
            b.evaluate(expr)
            b.wait_for_load()
            after = str(b.get_current_url() or "")
        return ActionResult(
            action=action, target=after[:120], method="browser_dom",
            success=True, outcome=ActionOutcome.SUCCESS,
            evidence={"before_url": before, "after_url": after},
            latency_ms=_ms(t0))
    except Exception as e:
        return ActionResult(action=action, method="browser_dom",
                            outcome=ActionOutcome.FAILED, error=str(e),
                            latency_ms=_ms(t0))


def back() -> ActionResult:
    return _history_op("back", "history.back()")


def forward() -> ActionResult:
    return _history_op("forward", "history.forward()")


def refresh() -> ActionResult:
    return _history_op("refresh", "location.reload()")


def scroll(delta: int) -> ActionResult:
    return _history_op("scroll", f"window.scrollBy(0, {int(delta)})")


def _dom_op(action: str, target: str, fn) -> ActionResult:
    """Run one DOM operation through the browser manager, honestly."""
    b = _browser()
    if b is None:
        return ActionResult(action=action, target=target, method="browser_dom",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="browser integration unavailable")
    t0 = time.time()
    try:
        ok = bool(fn(b))
        return ActionResult(
            action=action, target=target, method="browser_dom",
            success=ok,
            outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
            error="" if ok else f"dom {action} failed for '{target[:80]}'",
            latency_ms=_ms(t0))
    except Exception as e:
        return ActionResult(action=action, target=target, method="browser_dom",
                            outcome=ActionOutcome.FAILED, error=str(e),
                            latency_ms=_ms(t0))


def _ms(t0: float) -> float:
    return round((time.time() - t0) * 1000, 1)


# ═══════════════════════════════════════════════════════════════════
# Phase 23: STRUCTURED page reading (tier 1 — generic DOM only)
#
# These helpers answer "what is on the page" with STRUCTURED data
# (elements / links / forms / loading / focus / text), so the browser
# goal engine never has to fall back to screenshots for ordinary work.
#
# Site-specific behavior is FORBIDDEN here: no selector, no label and no
# URL of any particular website appears in this file.
# ══════════════════════════════════════════════════════════════════════

_STRUCTURE_SCRIPT = """
() => {
  const MAXE = __MAXE__, MAXT = __MAXT__;
  const txt = (el) => ((el.innerText || el.value || el.getAttribute('aria-label')
      || el.getAttribute('placeholder') || el.getAttribute('name') || '') + '')
      .replace(/\\s+/g, ' ').trim().slice(0, 140);
  const visible = (el) => {
    try {
      const r = el.getBoundingClientRect();
      if (r.width < 1 && r.height < 1) return false;
      const st = window.getComputedStyle(el);
      return st.display !== 'none' && st.visibility !== 'hidden';
    } catch (e) { return true; }
  };
  const sel = 'a, button, input, select, textarea, [role=button], ' +
              '[role=link], [role=tab], [role=checkbox], [role=menuitem], ' +
              '[role=searchbox], [type=submit]';
  const elements = [];
  for (const el of document.querySelectorAll(sel)) {
    if (!visible(el)) continue;
    const tag = el.tagName.toLowerCase();
    let kind = 'element';
    if (tag === 'a') kind = 'link';
    else if (tag === 'input' || tag === 'textarea') kind = 'input';
    else if (tag === 'select') kind = 'select';
    else if (tag === 'button') kind = 'button';
    else if (el.getAttribute('role')) kind = el.getAttribute('role');
    elements.push({
      label: txt(el), kind: kind, tag: tag,
      type: (el.getAttribute('type') || ''), name: (el.getAttribute('name') || ''),
      value: ((el.value || '') + '').slice(0, 300),
      href: ((el.href || '') + '').slice(0, 400),
      placeholder: (el.getAttribute('placeholder') || ''),
      enabled: !el.disabled
    });
    if (elements.length >= MAXE) break;
  }
  const links = elements.filter(e => e.kind === 'link' && e.href && e.label)
                        .map(e => ({text: e.label, href: e.href}));
  const forms = [];
  for (const f of document.querySelectorAll('form')) {
    if (forms.length >= 6) break;
    const fields = [];
    for (const i of f.querySelectorAll('input, textarea, select')) {
      if (fields.length >= 20) break;
      fields.push({name: (i.getAttribute('name') || ''),
                   type: ((i.getAttribute('type') || i.tagName) + '').toLowerCase(),
                   label: txt(i)});
    }
    forms.push({action: (f.getAttribute('action') || ''),
                method: (f.getAttribute('method') || 'get'), fields: fields});
  }
  const ae = document.activeElement;
  const focused = (ae && ae !== document.body)
      ? {label: txt(ae), tag: ae.tagName.toLowerCase(),
         value: ((ae.value || '') + '').slice(0, 200)}
      : null;
  const bodyText = (document.body && document.body.innerText)
      ? document.body.innerText : '';
  return {
    url: ((location && location.href) || ''),
    title: document.title || '',
    ready_state: document.readyState || '',
    focused: focused, elements: elements, links: links, forms: forms,
    password_fields: document.querySelectorAll('input[type=password]').length,
    text: bodyText.replace(/\\s+\\n/g, '\\n').slice(0, MAXT),
    text_length: bodyText.length,
    scroll: (window.scrollY || 0),
    hash_material: (((location && location.href) || '') + '|' +
                    (document.title || '') + '|' + bodyText.length)
  };
}
"""

_CANDIDATES_SCRIPT = """
() => {
  const q = __QUERY__, LIMIT = __LIMIT__;
  const txt = (el) => ((el.innerText || el.value || el.getAttribute('aria-label')
      || el.getAttribute('placeholder') || el.getAttribute('name') || '') + '')
      .replace(/\\s+/g, ' ').trim().slice(0, 140);
  const sel = 'a, button, input, select, [role=button], [role=link], ' +
              '[role=tab], [role=menuitem], label';
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const label = txt(el);
    if (!label || label.toLowerCase().indexOf(q) === -1) continue;
    const tag = el.tagName.toLowerCase();
    let kind = 'element';
    if (tag === 'a') kind = 'link';
    else if (tag === 'button') kind = 'button';
    else if (tag === 'input' || tag === 'select') kind = 'input';
    out.push({label: label, kind: kind, tag: tag,
              href: ((el.href || '') + '').slice(0, 400),
              value: ((el.value || '') + '').slice(0, 300)});
    if (out.length >= LIMIT) break;
  }
  return JSON.stringify(out);
}
"""


def _json_value(raw: Any) -> Any:
    """`evaluate` may return a JSON string or an already-decoded object."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        import json
        return json.loads(raw)
    except Exception:
        return None


def get_page_structure(max_text: int = 6000,
                       max_elements: int = 120) -> Dict[str, Any]:
    """Structured page truth (generic DOM): elements, links, forms, focus.

    Returns ``{"attached": False}`` when no browser page is available — the
    caller must degrade to the next perception tier, never invent values.
    """
    page = _page()
    if page is None:
        return {"attached": False, "error": "no attached browser page"}
    t0 = time.time()
    try:
        script = (_STRUCTURE_SCRIPT
                  .replace("__MAXE__", str(int(max_elements)))
                  .replace("__MAXT__", str(int(max_text))))
        data = _json_value(page.evaluate(script))
        if not isinstance(data, dict):
            return {"attached": True,
                    "error": "structure extraction returned nothing"}
        data["text"] = str(data.get("text") or "")
        data["elements"] = list(data.get("elements") or [])
        data["links"] = list(data.get("links") or [])
        data["forms"] = list(data.get("forms") or [])
        data["loading"] = str(data.get("ready_state") or "") != "complete"
        data["search_ms"] = _ms(t0)
        data["perception_method"] = "browser_dom"
        data["attached"] = True
        return data
    except Exception as e:
        logger.debug("[PERCEPTION] structured page read failed: %s", e)
        return {"attached": True, "error": str(e)}


def find_candidates(query: str, limit: int = 12) -> list:
    """ALL DOM elements whose visible label contains `query` (DOM order).

    Generic on purpose: the caller (agent.browser_goal) scores and ranks
    them, so ambiguity is surfaced instead of silently guessed.
    """
    page = _page()
    q = str(query or "").strip().lower()
    if page is None or not q:
        return []
    try:
        script = (_CANDIDATES_SCRIPT.replace("__QUERY__", repr(q))
                  .replace("__LIMIT__", str(int(limit))))
        data = _json_value(page.evaluate(script))
        return list(data) if isinstance(data, list) else []
    except Exception as e:
        logger.debug("[PERCEPTION] candidate enumeration failed: %s", e)
        return []


def wait_for_ready(timeout: float = 8.0,
                   interval: float = 0.25) -> Dict[str, Any]:
    """Bounded wait for ``document.readyState == complete``.

    Uses observe → wait → observe (never one long arbitrary sleep).
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(timeout))
    state = ""
    while True:
        page = _page()
        if page is None:
            return {"ready": False, "ready_state": "",
                    "elapsed_ms": _ms(t0),
                    "error": "no attached browser page"}
        try:
            state = str(page.evaluate("() => document.readyState") or "")
        except Exception as e:
            return {"ready": False, "ready_state": state,
                    "elapsed_ms": _ms(t0), "error": str(e)}
        if state == "complete":
            return {"ready": True, "ready_state": state, "elapsed_ms": _ms(t0)}
        if time.time() >= deadline:
            return {"ready": False, "ready_state": state,
                    "elapsed_ms": _ms(t0), "error": "page still loading"}
        time.sleep(min(interval, max(0.0, deadline - time.time())))


def select_option(label: str, value: str = "", option: str = "", *,
                  target: str = "") -> ActionResult:
    """Select an option by visible text in a generic ``<select>``."""
    page = _page()
    if page is None:
        return ActionResult(action="select_option", target=target or label,
                            method="browser_dom",
                            outcome=ActionOutcome.UNAVAILABLE,
                            error="browser integration unavailable")
    t0 = time.time()
    script = """
    () => {
      const want = __LABEL__, opt = __OPTION__, val = __VALUE__;
      const txt = (el) => ((el.innerText || el.getAttribute('aria-label')
          || el.getAttribute('name') || '') + '').toLowerCase();
      const selects = [...document.querySelectorAll('select')];
      let el = selects.find(s => txt(s).indexOf(want) !== -1);
      if (!el && selects.length === 1) el = selects[0];
      if (!el) return JSON.stringify({ok: false, reason: 'select not found'});
      const opts = [...el.options];
      let chosen = opts.find(o => (o.text || '').toLowerCase().indexOf(opt) !== -1);
      if (!chosen && val) chosen = opts.find(o => (o.value || '') === val);
      if (!chosen) return JSON.stringify({ok: false, reason: 'option not found'});
      el.value = chosen.value;
      el.dispatchEvent(new Event('input', {bubbles: true}));
      el.dispatchEvent(new Event('change', {bubbles: true}));
      return JSON.stringify({ok: true, selected: chosen.text || chosen.value});
    }
    """
    try:
        raw = page.evaluate(
            script.replace("__LABEL__", repr(str(label).lower()))
                  .replace("__OPTION__", repr(str(option).lower()))
                  .replace("__VALUE__", repr(str(value))))
        data = _json_value(raw) or {}
        ok = bool(data.get("ok"))
        return ActionResult(
            action="select_option", target=target or label, method="browser_dom",
            success=ok,
            outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
            evidence={"selected": data.get("selected", ""),
                      "reason": data.get("reason", "")},
            verification=(f"selected '{data.get('selected')}'" if ok else ""),
            error="" if ok else str(data.get("reason") or "selection failed"),
            latency_ms=_ms(t0))
    except Exception as e:
        return ActionResult(action="select_option", target=target or label,
                            method="browser_dom", outcome=ActionOutcome.FAILED,
                            error=str(e), latency_ms=_ms(t0))
