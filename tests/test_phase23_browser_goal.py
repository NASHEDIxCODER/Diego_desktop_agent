"""Phase 23: browser goal-directed agent tests.

Deterministic unit tests (no external websites):

  - browser goal parsing (search / open / find / extract / URL / guardrails)
  - structured BrowserContext (build_context, page classification,
    navigation annotation)
  - expected-effect verification (PASS / FAIL / NO_EVIDENCE) and
    false-positive prevention (generic change is never success)
  - semantic element selection + ambiguity handling
  - login-required pause + resume-from-observed-state
  - loading state bounded waiting
  - navigation failure, recovery, replan
  - trace events (PLAN/STEP/OBSERVATION/VERIFY/RECOVERY/COMPLETE)
  - task completion only when the goal outcome is verified
  - result extraction (observed vs missing, never invented)

Integration tests (real Chromium via Playwright + a local HTTP server —
no external websites) run in TestLiveBrowser at the bottom.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest

from agent.browser_context import (
    BrowserContext,
    BrowserContextObserver,
    InteractiveElement,
    build_context,
)
from agent.browser_goal import (
    BrowserAction,
    BrowserGoal,
    BrowserGoalKind,
    BrowserStatus,
    EffectResult,
    PageState,
    expected_effect_text,
    guardrail_reason,
    parse_browser_goal,
    rank_candidates,
    resolve_answer_to_candidate,
    verify_expected_effect,
)
from agent.browser_goal_engine import (
    BrowserGoalEngine,
    BrowserLimits,
    BrowserTaskRun,
)
from agent.trace import AgentTrace


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def structure(url: str = "", title: str = "", text: str = "",
              elements: tuple = (), links: tuple = (), forms: tuple = (),
              loading: bool = False, password_fields: Optional[int] = None,
              focused: Any = None) -> Dict[str, Any]:
    return {
        "url": url, "title": title, "text": text,
        "elements": list(elements), "links": list(links),
        "forms": list(forms), "loading": loading,
        "password_fields": (0 if password_fields is None else password_fields),
        "focused": focused, "scroll": 0,
        "hash_material": f"{url}|{title}|{len(text)}",
    }


def el(label: str, kind: str = "element", **kw: Any) -> Dict[str, Any]:
    d = {"label": label, "kind": kind}
    d.update(kw)
    return d


def link(text: str, href: str = "https://x.test/") -> Dict[str, str]:
    return {"text": text, "href": href}


def fast_limits() -> BrowserLimits:
    return BrowserLimits(wait_timeout_s=0.4, wait_interval_s=0.05,
                         max_waits=2)


# ═══════════════════════════════════════════════════════════════
# 1. Goal parsing
# ═══════════════════════════════════════════════════════════════

class TestGoalParsing:
    def test_search_goal_with_site(self):
        g = parse_browser_goal("Search LinkedIn for Go backend jobs")
        assert g is not None
        assert g.kind == BrowserGoalKind.SEARCH
        assert g.site == "linkedin"
        assert "go backend jobs" in g.query.lower()
        assert g.url  # resolved start URL

    def test_open_site_goal(self):
        g = parse_browser_goal("Open LinkedIn")
        assert g.kind == BrowserGoalKind.OPEN_SITE
        assert "linkedin.com" in g.url

    def test_open_url_goal(self):
        g = parse_browser_goal(
            "open https://github.com/NASHEDIxCODER/Diego_desktop_agent")
        assert g.kind == BrowserGoalKind.OPEN_URL
        assert "github.com" in g.url

    def test_find_results_becomes_search(self):
        g = parse_browser_goal("Find backend jobs on linkedin")
        assert g.kind == BrowserGoalKind.SEARCH
        assert g.site == "linkedin"

    def test_find_item_goal(self):
        g = parse_browser_goal("Open Rahul's profile on linkedin")
        assert g is not None
        assert g.site == "linkedin"
        assert "rahul" in g.target.lower()

    def test_non_goal_returns_none(self):
        assert parse_browser_goal("what's my cpu usage?") is None
        assert parse_browser_goal("") is None

    def test_completion_requirements_differ(self):
        nav = parse_browser_goal("Open LinkedIn")
        search = parse_browser_goal("Search LinkedIn for Go backend jobs")
        assert "search results" in search.completion_requirement()
        assert "search results" not in nav.completion_requirement()

    def test_guardrails(self):
        assert guardrail_reason("bypass login on the site")
        assert guardrail_reason("solve captcha for me")
        assert not guardrail_reason("Search LinkedIn for Go backend jobs")


# ═══════════════════════════════════════════════════════════════
# 2. Structured BrowserContext
# ═══════════════════════════════════════════════════════════════

class TestBrowserContext:
    def test_build_context_structured_fields(self):
        ctx = build_context(structure(
            url="https://www.linkedin.com/jobs",
            title="Jobs",
            text="Go backend jobs are listed below",
            elements=(el("Search jobs", "input", type="search", name="q"),
                      el("Search", "button", type="submit")),
            links=(link("Go Backend Engineer at Acme",
                        "https://www.linkedin.com/jobs/1"),),
            forms=({"action": "/jobs", "method": "get",
                    "fields": [{"name": "q", "type": "search"}]},),
        ))
        assert ctx.domain == "linkedin.com"
        assert ctx.page_title == "Jobs"
        assert len(ctx.interactive_elements) == 2
        assert len(ctx.links) == 1
        assert ctx.forms[0].action == "/jobs"
        assert ctx.page_state == PageState.LOADED
        assert ctx.page_hash
        assert ctx.browser_attached is True
        assert ctx.perception_method == "browser_dom"

    def test_login_required_classification(self):
        ctx = build_context(structure(
            url="https://www.linkedin.com/login",
            title="Sign in",
            text="Enter your email and password to sign in",
            elements=(el("Email", "input", type="text"),
                      el("", "input", type="password"),),
            password_fields=1,
        ))
        assert ctx.page_state == PageState.LOGIN_REQUIRED
        assert ctx.login_required is True
        assert ctx.authentication_state == "anonymous"

    def test_blocked_classification(self):
        ctx = build_context(structure(
            url="https://site.test/search",
            text="Please verify you are human — captcha required",
        ))
        assert ctx.page_state == PageState.BLOCKED

    def test_loading_classification(self):
        ctx = build_context(structure(
            url="https://site.test/results", text="loading…", loading=True))
        assert ctx.page_state == PageState.LOADING
        assert ctx.is_loading

    def test_empty_context_when_no_browser(self):
        ctx = BrowserContext()   # no browser attached: nothing fabricated
        assert ctx.is_empty
        assert ctx.page_state == PageState.EMPTY

    def test_navigation_annotation(self):
        obs = BrowserContextObserver()
        first = build_context(structure(url="https://a.test/"))
        second = build_context(structure(url="https://a.test/results?q=go"))
        obs._annotate_navigation(first)
        obs._annotate_navigation(second)
        assert second.navigation_state == "navigated"
        assert "https://a.test/" in second.last_navigation

    def test_contains_uses_structured_text(self):
        ctx = build_context(structure(
            url="https://a.test/jobs",
            text="Go Backend Engineer at Acme — remote",
            links=(link("Go Backend Engineer at Acme"),),
        ))
        assert ctx.contains("Go Backend Engineer")
        assert not ctx.contains("Python Senior Staff")


# ═══════════════════════════════════════════════════════════════
# 3. Expected-effect verification (no false positives)
# ═══════════════════════════════════════════════════════════════

class TestExpectedEffects:
    def test_navigate_pass_requires_url_match(self):
        before = build_context(structure(url="https://old.test/"))
        after = build_context(structure(url="https://www.linkedin.com/jobs"))
        v = verify_expected_effect(
            BrowserAction.NAVIGATE, {"url": "https://www.linkedin.com"},
            before, after)
        assert v.result == EffectResult.PASS

    def test_navigate_fail_on_wrong_domain(self):
        before = build_context(structure(url="https://old.test/"))
        after = build_context(structure(url="https://unrelated.test/"))
        v = verify_expected_effect(
            BrowserAction.NAVIGATE, {"url": "https://www.linkedin.com"},
            before, after)
        assert v.result == EffectResult.FAIL

    def test_navigate_no_evidence_without_observation(self):
        before = build_context(structure(url="https://old.test/"))
        empty = BrowserContext()  # nothing observable
        v = verify_expected_effect(
            BrowserAction.NAVIGATE, {"url": "https://www.linkedin.com"},
            before, empty)
        assert v.result in (EffectResult.NO_EVIDENCE, EffectResult.FAIL)

    def test_type_input_verified_from_observed_value(self):
        before = build_context(structure(url="https://s.test/"))
        after = build_context(structure(
            url="https://s.test/",
            elements=(el("Search jobs", "input", type="search",
                         value="go backend jobs"),)))
        v = verify_expected_effect(
            BrowserAction.TYPE_INPUT,
            {"target": "Search jobs", "text": "go backend jobs"},
            before, after)
        assert v.result == EffectResult.PASS

    def test_type_input_fails_when_value_absent(self):
        before = build_context(structure(url="https://s.test/"))
        after = build_context(structure(url="https://s.test/"))
        v = verify_expected_effect(
            BrowserAction.TYPE_INPUT,
            {"target": "Search jobs", "text": "go backend jobs"},
            before, after)
        assert v.result == EffectResult.FAIL

    def test_click_requires_declared_expectation(self):
        before = build_context(structure(url="https://s.test/"))
        after = build_context(structure(url="https://s.test/results"))
        # No declared expectation: NO_EVIDENCE (never "something changed").
        v = verify_expected_effect(BrowserAction.CLICK_ELEMENT,
                                   {"target": "Jobs"}, before, after)
        assert v.result == EffectResult.NO_EVIDENCE

    def test_click_url_change_verified(self):
        before = build_context(structure(url="https://s.test/"))
        after = build_context(structure(url="https://s.test/results?q=go"))
        v = verify_expected_effect(
            BrowserAction.CLICK_ELEMENT,
            {"target": "Jobs", "expect_url_change": True}, before, after)
        assert v.result == EffectResult.PASS

    def test_extract_requires_observed_information(self):
        before = build_context(structure(url="https://s.test/"))
        after = build_context(structure(
            url="https://s.test/jobs",
            links=(link("Go Backend Engineer at Acme",
                        "https://s.test/jobs/1"),
                   link("Backend Engineer at Beta",
                        "https://s.test/jobs/2")),
            text="Go Backend Engineer at Acme | Backend Engineer at Beta"))
        v = verify_expected_effect(
            BrowserAction.EXTRACT_RESULTS,
            {"query": "go backend", "extracted": 2}, before, after)
        assert v.result == EffectResult.PASS
        # Same page WITHOUT results: extraction cannot pass.
        empty = build_context(structure(url="https://s.test/"))
        v2 = verify_expected_effect(
            BrowserAction.EXTRACT_RESULTS,
            {"query": "go backend", "extracted": 0}, before, empty)
        assert v2.result != EffectResult.PASS

    def test_wait_for_page(self):
        loading = build_context(structure(url="https://s.test/",
                                          loading=True))
        loaded = build_context(structure(url="https://s.test/",
                                         text="done"))
        assert verify_expected_effect(
            BrowserAction.WAIT_FOR_PAGE, {}, loading, loaded
        ).result == EffectResult.PASS
        assert verify_expected_effect(
            BrowserAction.WAIT_FOR_PAGE, {}, loading, loading
        ).result == EffectResult.FAIL

    def test_expected_effect_text_declared_for_actions(self):
        for action in (BrowserAction.NAVIGATE, BrowserAction.CLICK_ELEMENT,
                       BrowserAction.TYPE_INPUT, BrowserAction.EXTRACT_RESULTS,
                       BrowserAction.READ_PAGE):
            assert expected_effect_text(action)

    def test_click_generic_screen_change_is_not_success(self):
        """Unrelated change (same URL, only clock text) must NOT verify."""
        before = build_context(structure(url="https://s.test/",
                                         text="12:00 results pending"))
        after = build_context(structure(url="https://s.test/",
                                        text="12:01 results pending"))
        v = verify_expected_effect(
            BrowserAction.CLICK_ELEMENT,
            {"target": "Jobs", "expect": "job results"},
            before, after)
        assert v.result != EffectResult.PASS


# ═══════════════════════════════════════════════════════════════
# 4. Semantic element selection + ambiguity
# ═══════════════════════════════════════════════════════════════

from agent.browser_goal import Candidate as Candidate_m  # noqa: E402


class TestSemanticSelection:
    def test_rank_unique_match(self):
        raw = [{"label": "Go Backend Engineer at Acme", "kind": "link",
                "href": "https://s.test/1"}]
        ranked, ambiguous, reason = rank_candidates("go backend", raw)
        assert not ambiguous
        assert ranked and ranked[0].score > 0.5

    def test_rank_two_equal_matches_is_ambiguous(self):
        raw = [{"label": "Rahul Kumar", "kind": "link"},
               {"label": "Rahul Sharma", "kind": "link"}]
        ranked, ambiguous, reason = rank_candidates("rahul", raw)
        assert ambiguous
        assert "plausible" in reason.lower() or "ambiguous" in reason.lower()

    def test_exact_match_wins_over_partial(self):
        raw = [{"label": "Jobs", "kind": "link"},
               {"label": "Saved Jobs", "kind": "link"}]
        ranked, ambiguous, _ = rank_candidates("jobs", raw)
        assert not ambiguous
        assert ranked[0].label == "Jobs"

    def test_resolve_answer_by_ordinal(self):
        cands = [Candidate_m(label="Rahul Kumar"),
                 Candidate_m(label="Rahul Sharma")]
        assert resolve_answer_to_candidate(
            "the second one", cands).label == "Rahul Sharma"
        assert resolve_answer_to_candidate("1", cands).label == "Rahul Kumar"

    def test_search_input_found_semantically(self):
        eng = BrowserGoalEngine(controller=object(), trace=AgentTrace())
        ctx = build_context(structure(
            url="https://s.test/",
            elements=(el("Main menu", "button"),
                      el("Search jobs", "input", type="search"),
                      el("Notifications", "button"))))
        found = eng.search_input(ctx)
        assert found is not None and "search" in found.label.lower()

    def test_search_input_none_when_ambiguous(self):
        eng = BrowserGoalEngine(controller=object(), trace=AgentTrace())
        ctx = build_context(structure(
            url="https://s.test/",
            elements=(el("First name", "input"),
                      el("Last name", "input"),
                      el("City", "input"))))
        assert eng.search_input(ctx) is None

    def test_password_field_never_used_as_search_input(self):
        eng = BrowserGoalEngine(controller=object(), trace=AgentTrace())
        ctx = build_context(structure(
            url="https://s.test/login",
            elements=(el("Email", "input", type="text"),
                      el("", "input", type="password")),
            password_fields=1))
        found = eng.search_input(ctx)
        assert found is None or not found.is_password


# ═══════════════════════════════════════════════════════════════
# 5. Engine flows on a deterministic MiniBrowser simulation
# ═══════════════════════════════════════════════════════════════

class MiniPage:
    """One deterministic page: structured truth the way tier 1 sees it."""

    def __init__(self, url: str, title: str = "", text: str = "",
                 elements: tuple = (), links: tuple = (),
                 forms: tuple = (), password_fields: int = 0):
        self.url, self.title, self.text = url, title, text
        self.elements = [dict(e) for e in elements]
        self.links = [dict(l) for l in links]
        self.forms = [dict(f) for f in forms]
        self.password_fields = password_fields


class MiniBrowser:
    """The smallest honest browser simulation: pages mutate on actions."""

    def __init__(self) -> None:
        self.pages: Dict[str, MiniPage] = {}
        self.url = ""
        self.typed: Dict[str, str] = {}       # target label -> typed value
        self.submitted_query = ""
        self.loading_until = 0.0              # wall clock until "loaded"
        self.history: List[str] = []

    def add(self, page: MiniPage) -> MiniPage:
        self.pages[page.url] = page
        return page

    def page(self) -> Optional[MiniPage]:
        """Page for the current URL (trailing-slash / query tolerant)."""
        page = self.pages.get(self.url)
        if page is None:
            base = self.url.split("?")[0].rstrip("/")
            page = self.pages.get(base + "/") or self.pages.get(base)
        return page

    def structure(self) -> Dict[str, Any]:
        page = self.page()
        if page is None:
            return {"attached": True, "url": self.url, "title": "",
                    "text": "", "elements": [], "links": [], "forms": [],
                    "loading": False, "password_fields": 0}
        elements = []
        for e in page.elements:
            e = dict(e)
            if e.get("kind") == "input" and e["label"] in self.typed:
                e["value"] = self.typed[e["label"]]
            elements.append(e)
        return {
            "url": self.url, "title": page.title, "text": page.text,
            "elements": elements, "links": list(page.links),
            "forms": list(page.forms),
            "loading": time.time() < self.loading_until,
            "password_fields": page.password_fields,
            "hash_material": f"{self.url}|{page.title}|{len(page.text)}"
                             f"|{sorted(self.typed.items())}",
        }

    def navigate(self, url: str) -> None:
        self.history.append(self.url)
        self.url = url

    def type_into(self, target: str, text: str) -> bool:
        page = self.page()
        if page is None:
            return False
        labels = [e["label"] for e in page.elements
                  if e.get("kind") == "input"]
        hits = [l for l in labels if target.lower() in l.lower()]
        if len(hits) != 1:
            return False
        self.typed[hits[0]] = text
        return True

    def press_enter(self) -> None:
        """Submit the page's form → results URL (generic form semantics)."""
        page = self.page()
        if page is None or not page.forms:
            return
        q = next(iter(self.typed.values()), "")
        self.submitted_query = q
        action = page.forms[0].get("action") or "/results"
        parts = self.url.split("/")
        base = f"{parts[0]}//{parts[2]}" if len(parts) > 2 else ""
        if action.startswith("/") and base:
            action = base + action
        self.navigate(f"{action}?q={q.replace(' ', '+')}")

    def click_target(self, target: str) -> bool:
        page = self.page()
        if page is None:
            return False
        cands = [e for e in page.elements
                 if target.lower() in str(e.get("label", "")).lower()]
        cands += [{"label": l.get("text", ""), "href": l.get("href", "")}
                  for l in page.links
                  if target.lower() in str(l.get("text", "")).lower()]
        exact = [c for c in cands
                 if str(c.get("label", "")).lower() == target.lower()]
        if len(cands) > 1 and not exact:
            return False  # ambiguous — never guess
        chosen = (exact or cands)[:1]


class FakeController:
    """Same interface as ComputerController: execute(action, params)."""

    def __init__(self, mini: MiniBrowser, mutate=None) -> None:
        self.mini = mini
        self.mutate = mutate or (lambda action, params: None)
        self.calls: List[tuple] = []

    def execute(self, action: str, params: Dict[str, Any]):
        from computer.action_result import ActionOutcome, ActionResult
        self.calls.append((action, dict(params)))
        p = dict(params or {})
        ok, error, evidence = True, "", {}
        if action in ("navigate", "open_url"):
            url = str(p.get("url") or "")
            if url.startswith("http"):
                self.mini.navigate(url)
                evidence = {"url": self.mini.url}
            else:
                ok, error = False, f"unknown url: {url}"
        elif action in ("find_candidates", "find_element"):
            target = str(p.get("target") or "")
            page = self.mini.page()
            cands: List[Dict[str, Any]] = []
            if page is not None:
                for e in page.elements:
                    if target.lower() in str(e.get("label", "")).lower():
                        cands.append({"label": e["label"],
                                      "kind": e.get("kind", "element")})
                for l in page.links:
                    if target.lower() in str(l.get("text", "")).lower():
                        cands.append({"label": l.get("text", ""),
                                      "kind": "link",
                                      "href": l.get("href", "")})
            evidence = {"candidates": cands}
            if not cands:
                ok = False
                error = f"no element matched '{target[:40]}'"
        elif action in ("type_text", "type"):
            ok = self.mini.type_into(str(p.get("target") or ""),
                                     str(p.get("text") or ""))
            error = "" if ok else "input not found"
        elif action == "clear_text":
            ok = self.mini.type_into(str(p.get("target") or ""), "")
        elif action == "press_key":
            if str(p.get("key") or "").lower() == "enter":
                self.mini.press_enter()
        elif action == "click":
            ok = self.mini.click_target(str(p.get("target") or ""))
            error = "" if ok else "click target not found"
        elif action in ("scroll", "wait_for_page", "read_page",
                        "extract_text", "get_page_state", "extract_links",
                        "select_option", "back", "forward", "refresh",
                        "open_app"):
            evidence = {"url": self.mini.url, "ready_state": "complete"}
        else:
            ok, error = False, f"unsupported action: {action}"
        self.mutate(action, dict(p))
        return ActionResult(
            action=action, method="browser_dom", success=ok,
            outcome=(ActionOutcome.SUCCESS if ok else ActionOutcome.FAILED),
            evidence=evidence, error=error)


class FakeObserver:
    """Observes the MiniBrowser through the REAL build_context path."""

    def __init__(self, mini: MiniBrowser) -> None:
        self.mini = mini

    def observe(self, note: str = "") -> BrowserContext:
        return build_context(self.mini.structure())


def make_engine(mini: MiniBrowser, mutate=None,
                limits: Optional[BrowserLimits] = None) -> BrowserGoalEngine:
    return BrowserGoalEngine(
        controller=FakeController(mini, mutate), observer=FakeObserver(mini),
        trace=AgentTrace(), limits=limits or fast_limits())


def jobsite() -> MiniBrowser:
    mini = MiniBrowser()
    mini.add(MiniPage(
        "https://workhub.test/", title="Jobsite",
        text="Welcome to Jobsite. Find your next role.",
        elements=(el("Search jobs", "input", type="search", name="q"),
                  el("Search", "button", type="submit"),
                  el("Sign in", "button")),
        forms=({"action": "/jobs", "method": "get",
                "fields": [{"name": "q", "type": "search"}]},)))
    mini.add(MiniPage(
        "https://workhub.test/login", title="Sign in",
        text="Enter your email and password to sign in",
        elements=(el("Email", "input", type="text"),
                  el("", "input", type="password")),
        password_fields=1))
    return mini


def add_results(mini: MiniBrowser) -> MiniPage:
    return mini.add(MiniPage(
        "https://workhub.test/jobs", title="Job results",
        text=("12 results for go backend jobs — "
              "Go Backend Engineer at Acme · Backend Engineer at Beta · "
              "Remote Go Developer at Gamma"),
        elements=(el("Search jobs", "input", type="search", name="q"),),
        links=(link("Go Backend Engineer at Acme",
                    "https://workhub.test/jobs/1"),
               link("Backend Engineer at Beta",
                    "https://workhub.test/jobs/2"),
               link("Remote Go Developer at Gamma",
                    "https://workhub.test/jobs/3")),
        forms=({"action": "/jobs", "method": "get",
                "fields": [{"name": "q", "type": "search"}]},)))


# ── search / open flows ──────────────────────────────────────────

class TestEngineSearchFlow:
    def test_search_goal_completes_with_extracted_results(self):
        mini = jobsite()
        add_results(mini)
        eng = make_engine(mini)
        msg = eng.handle_command("Search workhub.test for Go backend jobs")
        run = eng.pending_run()
        assert run.status == BrowserStatus.COMPLETED
        assert run.verified
        assert run.results_verified and run.navigation_verified
        assert run.extraction.count >= 2
        # evidence-backed: results carry observed provenance, never invented
        r0 = run.extraction.results[0]
        assert r0.observed and not r0.inferred
        assert "completed" in msg.lower() or "extracted" in msg.lower()

    def test_search_without_results_never_completes(self):
        mini = jobsite()  # no results page registered: results never appear
        eng = make_engine(mini)
        eng.handle_command("Search workhub.test for Go backend jobs")
        run = eng.pending_run()
        assert not run.verified
        assert run.status != BrowserStatus.COMPLETED

    def test_open_url_goal_completes_on_verified_navigation(self):
        mini = jobsite()
        add_results(mini)
        eng = make_engine(mini)
        eng.handle_command("open https://workhub.test/jobs")
        run = eng.pending_run()
        assert run.status == BrowserStatus.COMPLETED
        assert run.navigation_verified
        assert "workhub.test" in run.last_context.current_url

    def test_wrong_landing_url_fails_honestly(self):
        """Navigation 'succeeds' but lands elsewhere → FAIL, never PASS."""
        mini = jobsite()

        def land_elsewhere(action, params):
            if action in ("navigate", "open_url"):
                mini.url = "https://unrelated.test/"
        eng = make_engine(mini, mutate=land_elsewhere)
        eng.handle_command("open https://workhub.test/jobs")
        run = eng.pending_run()
        assert run.status == BrowserStatus.FAILED
        assert not run.verified

    def test_goal_completion_requires_more_than_navigation(self):
        """For a SEARCH goal, navigation alone must never complete the task."""
        mini = jobsite()
        add_results(mini)
        eng = make_engine(mini)
        run = eng.start("Search workhub.test for Go backend jobs")
        eng._run_step(run, run.steps[0])          # the NAVIGATE step only
        run.navigation_verified = True
        assert not eng._goal_outcome_verified(run)  # results still missing


# ── authentication pause / resume ────────────────────────────────

class TestEngineAuthFlow:
    def test_login_pause_then_resume_completes(self):
        mini = jobsite()
        add_results(mini)
        feed = MiniPage(
            "https://workhub.test/", title="Jobsite",
            text="Welcome back — signed in",
            elements=(el("Search jobs", "input", type="search", name="q"),
                      el("Sign out", "button"), el("My profile", "link")),
            forms=({"action": "/jobs", "method": "get",
                    "fields": [{"name": "q", "type": "search"}]},))
        mini.pages["https://workhub.test/"] = feed
        state = {"authed": False}

        def gate(action, params):
            if action in ("navigate", "open_url") and not state["authed"]:
                mini.url = "https://workhub.test/login"  # login wall
        eng = make_engine(mini, mutate=gate)
        msg = eng.handle_command("Search workhub.test for Go backend jobs")
        run = eng.pending_run()
        assert run.status == BrowserStatus.WAITING_FOR_USER
        assert eng.has_pending()
        assert "sign in" in (run.question or "").lower()

        # the user signs in manually; the task resumes from the SAME page
        state["authed"] = True
        mini.url = "https://workhub.test/"
        msg2 = eng.handle_command("continue")
        assert run.status == BrowserStatus.COMPLETED
        assert run.verified and run.results_verified

    def test_resume_still_on_login_page_stays_paused(self):
        mini = jobsite()
        state = {"authed": False}

        def gate(action, params):
            if action in ("navigate", "open_url") and not state["authed"]:
                mini.url = "https://workhub.test/login"
        eng = make_engine(mini, mutate=gate)
        eng.handle_command("Search workhub.test for Go backend jobs")
        run = eng.pending_run()
        assert run.status == BrowserStatus.WAITING_FOR_USER
        eng.handle_command("continue")  # user NOT signed in yet
        assert run.status == BrowserStatus.WAITING_FOR_USER
        assert eng.has_pending()

    def test_cancel_while_paused_fails_the_task(self):
        mini = jobsite()
        eng = make_engine(mini)
        run = eng.start("Search workhub.test for Go backend jobs")
        mini.url = "https://workhub.test/login"
        run.status = BrowserStatus.WAITING_FOR_USER
        eng.resume("cancel")
        assert run.status == BrowserStatus.FAILED


# ── ambiguity ────────────────────────────────────────────────────

class TestEngineAmbiguity:
    def test_ambiguous_target_asks_and_resumes(self):
        mini = jobsite()
        # The ambiguity must exist on the CURRENT page for FIND to hit it.
        mini.pages["https://workhub.test/"].links += (
            link("Rahul Kumar", "https://workhub.test/p/1"),
            link("Rahul Sharma", "https://workhub.test/p/2"))
        mini.add(MiniPage(
            "https://workhub.test/people", title="People",
            text="People named Rahul",
            links=(link("Rahul Kumar", "https://workhub.test/p/1"),
                   link("Rahul Sharma", "https://workhub.test/p/2"))))
        eng = make_engine(mini)
        msg = eng.handle_command("find Rahul on workhub.test")
        run = eng.pending_run()
        assert run.status == BrowserStatus.ASKING_USER
        assert eng.has_pending()
        low = (run.question or "").lower()
        assert "which" in low or "matches" in low
        labels = [c.label for c in run.candidates]
        assert "Rahul Kumar" in labels and "Rahul Sharma" in labels

        msg2 = eng.handle_command("Rahul Kumar")
        assert run.status == BrowserStatus.COMPLETED


# ── loading / blocked states ─────────────────────────────────────

class TestEngineLoadingAndBlocked:
    def test_loading_page_bounded_wait(self):
        mini = jobsite()
        add_results(mini)
        mini.loading_until = time.time() + 30.0   # stuck "loading"
        eng = make_engine(mini)
        eng.handle_command("Search workhub.test for Go backend jobs")
        run = eng.pending_run()
        assert 0 < run.waits <= fast_limits().max_waits
        # no arbitrary long sleeps: total run time stays bounded
        assert (run.ended_at or time.time()) - run.started_at < 15

    def test_blocked_page_pauses_and_never_bypasses(self):
        mini = jobsite()
        mini.add(MiniPage(
            "https://workhub.test/captcha", title="Access check",
            text="Please verify you are human — captcha challenge",
            elements=(el("Reload", "button"),)))

        def to_captcha(action, params):
            if action in ("navigate", "open_url"):
                mini.url = "https://workhub.test/captcha"
        eng = make_engine(mini, mutate=to_captcha)
        eng.handle_command("open https://workhub.test/")
        run = eng.pending_run()
        assert run.status == BrowserStatus.WAITING_FOR_USER
        assert eng.has_pending()
        low = (run.question or "").lower()
        assert "captcha" in low or "access-control" in low
        # nothing tried to interact with the challenge
        assert not [c for c in eng.controller().calls if c[0] == "click"]


# ── failure, recovery, replan, budgets ───────────────────────────

class TestEngineFailureAndRecovery:
    def test_missing_target_fails_after_bounded_recovery(self):
        mini = jobsite()   # /nonexistent serves nothing
        eng = make_engine(mini)
        eng.handle_command("open Missing Widget on workhub.test")
        run = eng.pending_run()
        assert run.status == BrowserStatus.FAILED
        assert not run.verified
        for step in run.steps:
            assert step.attempts <= (fast_limits().max_retries_per_step
                                     + fast_limits().max_replans + 1)
        assert run.replans <= fast_limits().max_replans

    def test_no_infinite_loop_when_target_never_appears(self):
        mini = jobsite()
        eng = make_engine(mini)
        eng.handle_command("Search workhub.test for something nonexistent")
        run = eng.pending_run()
        assert run.actions <= fast_limits().max_actions
        assert run.observation_count <= fast_limits().max_observations
        assert not run.verified  # no false success

    def test_handle_command_fallthrough_for_non_goals(self):
        mini = jobsite()
        eng = make_engine(mini)
        assert eng.handle_command("what's the weather?") is None
        assert eng.pending_run() is None

    def test_guardrailed_goal_refused(self):
        mini = jobsite()
        eng = make_engine(mini)
        msg = eng.handle_command(
                   "open https://workhub.test/login and bypass login")
        run = eng.pending_run()
        assert run is not None and run.status == BrowserStatus.FAILED
        low = (msg or "").lower()
        assert "bypass" in low or "refused" in low
        assert [c for c in eng.controller().calls
                if c[0] in ("type_text", "click")] == []


# ═══════════════════════════════════════════════════════════════
# 6. Trace integration
# ═══════════════════════════════════════════════════════════════

class TestTraceIntegration:
    def test_full_trace_lifecycle(self):
        mini = jobsite()
        add_results(mini)
        eng = make_engine(mini)
        eng.handle_command("Search workhub.test for Go backend jobs")
        tr = eng.trace()
        types = [e.event_type.value for e in tr.events(limit=300)]
        assert "GOAL_RECEIVED" in types
        assert "PLAN_CREATED" in types
        assert "OBSERVATION" in types
        assert "ACTION_STARTED" in types and "ACTION_COMPLETED" in types
        assert "VERIFICATION_COMPLETED" in types
        assert "TASK_COMPLETED" in types

    def test_phases_are_operational_not_private(self):
        mini = jobsite()
        add_results(mini)
        eng = make_engine(mini)
        eng.handle_command("Search workhub.test for Go backend jobs")
        tr = eng.trace()
        phases = [e.detail for e in tr.events(limit=300)
                  if e.event_type.value == "PHASE_CHANGED"]
        allowed = {"THINKING", "PLANNING", "OBSERVING", "ACTING", "WAITING",
                   "VERIFYING", "RECOVERING", "ASKING_USER", "COMPLETED",
                   "FAILED"}
        assert phases and set(phases) <= allowed
        snap = tr.snapshot()
        assert snap.phase in allowed
        # each observation records the perception method used
        obs = [e for e in tr.events(limit=300)
               if e.event_type.value == "OBSERVATION"]
        assert obs and all(e.method == "browser_dom" for e in obs)

    def test_snapshot_shows_goal_plan_and_step(self):
        mini = jobsite()
        add_results(mini)
        eng = make_engine(mini)
        eng.handle_command("Search workhub.test for Go backend jobs")
        snap = eng.trace().snapshot()
        assert "workhub.test" in snap.goal
        assert snap.plan
        assert snap.total_steps == len(snap.plan)

    def test_failure_recorded_in_trace(self):
        mini = jobsite()
        eng = make_engine(mini)
        eng.handle_command("open Missing Widget on workhub.test")
        types = [e.event_type.value for e in eng.trace().events(limit=300)]
        assert "TASK_FAILED" in types

    def test_open_site_scoped_item_ambiguity(self):
        mini = jobsite()
        mini.pages["https://workhub.test/"].links += (
            link("Rahul Kumar", "https://workhub.test/p/1"),
            link("Rahul Sharma", "https://workhub.test/p/2"))
        mini.add(MiniPage(
            "https://workhub.test/people", title="People",
            text="People named Rahul",
            links=(link("Rahul Kumar", "https://workhub.test/p/1"),
                   link("Rahul Sharma", "https://workhub.test/p/2"))))
        eng = make_engine(mini)
        eng.handle_command("open Rahul on workhub.test")
        run = eng.pending_run()
        # Navigation alone is NOT completion; the ambiguous item forces a
        # clarification instead of a guess.
        assert run.status == BrowserStatus.ASKING_USER

