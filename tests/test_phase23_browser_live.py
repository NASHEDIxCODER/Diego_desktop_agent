"""Phase 23 integration tests: REAL Chromium + local HTTP server.

No external websites: a controlled local site (search interface, results,
people pages, login wall, captcha page, slow page) is served from a
threaded HTTP server and rendered by headless Chromium (Playwright).

What stays REAL:
  - DOM observation (BrowserContextObserver -> computer.browser_controller
    structure extraction over the live page)
  - page classification (login / blocked / loading / not-found)
  - expected-effect verification
  - the engine's full control loop, recovery and trace

What is harness-owned: navigation/typing/clicking go through semantic
Playwright locators (never screen coordinates, never host-level input).
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import pytest

from agent.browser_context import observer
from agent.browser_goal import BrowserAction, BrowserStatus
from agent.browser_goal_engine import (
    BrowserGoalEngine,
    BrowserLimits,
    BrowserTaskRun,
)
from agent.trace import AgentTrace

try:
    from playwright.sync_api import sync_playwright
    _HAS_PLAYWRIGHT = True
except Exception:  # pragma: no cover
    _HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(
    not _HAS_PLAYWRIGHT, reason="playwright not installed")


# ═══════════════════════════════════════════════════════════════
# Local site (controlled; generic pages only)
# ═══════════════════════════════════════════════════════════════

def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><title>{title}</title></head>"
            f"<body>{body}</body></html>")


HOME = _page("Workhub", """
<h1>Workhub</h1>
<form action="/results" method="get">
  <input type="search" name="q" placeholder="Search jobs"
         aria-label="Search jobs">
  <button type="submit">Search</button>
</form>
<nav><a href="/people">People</a></nav>
<nav><a href="/slow">Slow page</a></nav>
<nav><a href="/dashboard">Dashboard</a></nav>
<nav><a href="/person/rahul-kumar">Rahul Kumar</a></nav>
<nav><a href="/person/rahul-sharma">Rahul Sharma</a></nav>
""")

AUTHED_HOME = _page("Workhub — signed in", """
<h1>Welcome back</h1>
<a href="/signout">Sign out</a>
<form action="/results" method="get">
  <input type="search" name="q" placeholder="Search jobs"
         aria-label="Search jobs">
  <button type="submit">Search</button>
</form>
""")

LOGIN = _page("Sign in", """
<h1>Sign in to Workhub</h1>
<form action="/auth" method="get">
  <input type="text" name="email" placeholder="Email" aria-label="Email">
  <input type="password" name="password" aria-label="Password">
  <button type="submit">Sign in</button>
</form>
""")

BLOCKED = _page("Access check", """
<h1>Please verify you are human</h1>
<p>Captcha challenge — access denied until resolved.</p>
""")

PEOPLE = _page("People", """
<h1>People</h1>
<a href="/person/rahul-kumar">Rahul Kumar</a>
<a href="/person/rahul-sharma">Rahul Sharma</a>
""")

RAHUL_KUMAR = _page("Rahul Kumar — profile",
                    "<h1>Rahul Kumar</h1><p>Go backend engineer.</p>")

DASHBOARD = _page("Dashboard", """
<h1>Dashboard</h1>
<p>Signed in — welcome back.</p>
<a href="/signout">Sign out</a>
""")


def _results_html(query: str) -> str:
    tokens = [t for t in query.lower().split() if t]
    relevant = any(t in "go backend" for t in tokens) if tokens else True
    if relevant:
        cards = "".join(
            f'<a class="job" href="/job/{i}">Go Backend Engineer at '
            f"Acme {i}</a>" for i in (1, 2, 3))
    else:
        cards = "<p>No results found.</p>"
    return _page("Job results",
                 f"<h1>Results for {query or 'all'}</h1>{cards}")


# Harness-owned session flag: only the USER's manual sign-in sets it.
_STATE = {"signed_in": False}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, html: str) -> None:
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        from urllib.parse import urlparse, parse_qs, unquote
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path == "/authed":
            self._send(AUTHED_HOME)
        elif path == "/login":
            self._send(LOGIN)
        elif path == "/blocked":
            self._send(BLOCKED)
        elif path == "/people":
            self._send(PEOPLE)
        elif path.startswith("/person/rahul-kumar"):
            self._send(RAHUL_KUMAR)
        elif path == "/dashboard":
            # Protected: a signed-out visitor sees the sign-in wall.
            self._send(DASHBOARD if _STATE["signed_in"] else LOGIN)
        elif path == "/auth":
            # The USER's manual sign-in (the agent never fills these).
            _STATE["signed_in"] = True
            # Redirect so the signed-in page — not the credential query — is
            # what the browser (and the resuming task) observes.
            self.send_response(302)
            self.send_header("Location", "/dashboard")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/signout":
            _STATE["signed_in"] = False
            self._send(HOME)
        elif path == "/results":
            self._send(_results_html(unquote(qs.get("q", [""])[0])))
        elif path.startswith("/slow"):
            time.sleep(1.2)
            self._send(HOME)
        else:
            self._send(HOME)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture(scope="module")
def live():
    _STATE["signed_in"] = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base + "/", wait_until="domcontentloaded")
        from computer import browser_controller as bctl
        bctl.bind_page(page)
        try:
            yield base, page
        finally:
            bctl.unbind_page()
            browser.close()
    server.shutdown()

# ═══════════════════════════════════════════════════════════════
# Integration tests (real Chromium + local HTTP server)
# ═══════════════════════════════════════════════════════════════

def _engine(base: str, page: Any) -> BrowserGoalEngine:
    return BrowserGoalEngine(
        target_site="workhub.test",
        local_base=base,
        limits=BrowserLimits(),)


def _go(base: str, page: Any, goal: str,
        *, expect: str = "TASK_COMPLETED") -> BrowserTaskRun:
    engine = _engine(base, page)
    run = engine.start(goal)
    engine.run_to_terminal(run)
    if expect == "TASK_COMPLETED":
        assert run.status is BrowserStatus.COMPLETED, (
            f"goal={goal!r} status={run.status} error={run.error}")
    elif expect == "ASKING_USER":
        assert run.status is BrowserStatus.ASKING_USER, (
            f"goal={goal!r} status={run.status}")
    elif expect == "FAILED":
        assert run.status is BrowserStatus.FAILED, (
            f"goal={goal!r} status={run.status}")
    return run


def _trace_events(run: BrowserTaskRun) -> list:
    out = []
    for seq in range(1, (getattr(run, "_trace_seq", 0) or 0) + 1):
        ev = getattr(run, "_trace_by_seq", {}).get(seq)
        if ev is not None:
            out.append(ev)
    return out


# ── A. Open a public website ─────────────────────────────────────

def test_live_open_site(live):
    base, page = live
    run = _go(base, page, "Open workhub.test")
    ctx = observer().observe("test")
    assert ctx.current_url.startswith(base)
    assert ctx.page_title == "Workhub"
    ev = _trace_events(run)[-1]
    assert ev.event_type.value == "TASK_COMPLETED"


# ── B. Navigate between pages ─────────────────────────────────────

def test_live_navigate_people(live):
    base, page = live
    run = _go(base, page, "Go to the People page on workhub.test")
    ctx = observer().observe("test")
    assert "/people" in ctx.current_url, ctx.current_url
    assert ctx.page_title == "People"


# ── C. Search a public site ───────────────────────────────────────

def test_live_search_jobs(live):
    base, page = live
    run = _go(base, page,
              "Search workhub.test for Go backend jobs")
    ctx = observer().observe("test")
    assert "/results" in ctx.current_url, ctx.current_url
    assert "Go Backend Engineer" in ctx.visible_text
    assert run.status is BrowserStatus.COMPLETED
    evs = _trace_events(run)
    v = [e for e in evs
         if e.event_type.value == "VERIFICATION_COMPLETED"]
    assert v, "no verification event"
    assert v[-1].verification_result in ("PASS", "PASS")


# ── D/E/F. Find element by semantic meaning, click it, verify state ─

def test_live_click_people_link(live):
    base, page = live
    run = _go(base, page, "Open the People page on workhub.test")
    ctx = observer().observe("test")
    assert ctx.page_title == "People"


# ── G. Handle loading state ───────────────────────────────────────

def test_live_slow_page(live):
    base, page = live
    run = _go(base, page,
              "Open the slow page on workhub.test and wait for it")
    ctx = observer().observe("test")
    assert ctx.page_title == "Workhub"
    assert run.status is BrowserStatus.COMPLETED


# ── H/M. Missing target: bounded recovery then HONEST failure ────
# (No false-positive completion: navigation alone never completes the task.)

def test_live_missing_target_fails_honestly(live):
    base, page = live
    page.goto(base + "/", wait_until="domcontentloaded")
    run = _go(base, page,
              "Open the Missing Widget page on workhub.test",
              expect="FAILED")
    assert run.error, "the failure must carry an honest reason"
    assert not run.verified
    # Bounded: the recovery/replan budgets were respected (no endless loop).
    limits = _engine(base, page).limits
    assert run.recoveries <= limits.max_actions
    assert all(s.attempts <= limits.max_retries_per_step + limits.max_replans + 1
               for s in run.steps)
    # The page really is unchanged — no unrelated change was spun as success.
    ctx = observer().observe("test")
    assert ctx.page_title == "Workhub"
    evs = _trace_events(run)
    assert any(e.event_type.value == "RECOVERY_STARTED" for e in evs)
    assert any(e.event_type.value == "TASK_FAILED" for e in evs)


# ── I. Ambiguous target: ASK, never guess; resume the same task ──

def test_live_ambiguous_target_asks_and_resumes(live):
    base, page = live
    page.goto(base + "/", wait_until="domcontentloaded")
    eng = _engine(base, page)
    run = eng.start("Open Rahul on workhub.test")
    eng.run_to_terminal(run)
    assert run.status is BrowserStatus.ASKING_USER, (run.status, run.error)
    assert eng.has_pending()
    q = (run.question or "").lower()
    assert "rahul" in q and "which one" in q, run.question
    # Both observed candidates are offered — the agent invented neither.
    labels = {c.label for c in run.candidates}
    assert {"Rahul Kumar", "Rahul Sharma"} <= labels, labels
    # The clarification continues the SAME task (no restart).
    eng.resume("Rahul Kumar", task_id=run.task_id)
    assert run.status is BrowserStatus.COMPLETED, (run.status, run.error)
    ctx = observer().observe("test")
    assert "/person/rahul-kumar" in ctx.current_url, ctx.current_url
    evs = _trace_events(run)
    assert any(e.event_type.value == "CONFIRMATION_REQUIRED"
               and e.verification_result == "AMBIGUOUS" for e in evs)


# ── J/K. Login required: pause, let the USER sign in, resume ─────

def test_live_login_pause_then_resume(live):
    base, page = live
    eng = _engine(base, page)
    page.goto(base + "/", wait_until="domcontentloaded")
    run = eng.start("Open the Dashboard page on workhub.test")
    eng.run_to_terminal(run)
    # The sign-in wall is observed: the task pauses for the USER.
    assert run.status is BrowserStatus.WAITING_FOR_USER, (run.status, run.error)
    assert eng.has_pending()
    assert "sign in" in (run.question or "").lower(), run.question
    evs = _trace_events(run)
    assert any(e.event_type.value == "OBSERVATION" and "login" in
               e.observation.lower() for e in evs), \
        [e.observation for e in evs if e.event_type.value == "OBSERVATION"]
    # The agent NEVER touches the credentials — the user signs in manually.
    page.fill("input[name=email]", "user@example.com")
    page.fill("input[name=password]", "secret")
    page.click("button[type=submit]")          # GET /auth → Dashboard
    eng.resume("continue", task_id=run.task_id)
    assert run.status is BrowserStatus.COMPLETED, (run.status, run.error)
    ctx = observer().observe("test")
    assert "/dashboard" in ctx.current_url, ctx.current_url
    assert "Dashboard" in ctx.page_title


# ── L. Extract visible structured information (evidence-backed) ──

def test_live_extract_results_are_observed_facts(live):
    base, page = live
    page.goto(base + "/", wait_until="domcontentloaded")
    run = _go(base, page, "Search workhub.test for Go backend jobs")
    # Extraction carries OBSERVED provenance only (title/url/source).
    ex = run.extraction
    assert ex is not None and ex.count >= 1, run.evidence.get("results")
    r0 = ex.results[0]
    assert r0.observed and not r0.inferred
    assert "Go Backend Engineer" in r0.title
    assert r0.url.startswith("/job/") or r0.url.startswith(base)
    # Evidence records where the fact came from.
    assert r0.evidence.get("page_url"), r0.evidence
    # And the trace recorded the verification, not just navigation.
    evs = _trace_events(run)
    verifs = [e for e in evs if e.event_type.value == "VERIFICATION_COMPLETED"]
    assert verifs and verifs[-1].verification_result == "PASS"

