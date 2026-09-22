"""
GoalRuntime backend layer — the single world-touching boundary.

``RuntimeBackend`` is the protocol every skill talks to. Skills NEVER touch
pyautogui/xdotool/httpx directly; they call the backend. Two implementations:

    DesktopBackend — the REAL one. It composes (never duplicates) the
    existing Phase 22+ capabilities:
        desktop actions  → computer.computer_controller (ACT boundary)
        browser DOM      → computer.browser_controller
        windows/focus    → computer.window_manager
        OCR              → vision.ocr_pipeline
        vision model     → goalruntime.llm.ModelRouter (VISION role)
        filesystem/term  → pathlib + bounded subprocess

Every method returns a plain dict: {"success": bool, ...evidence}. No
exceptions escape; failures are honest dicts.

Logging: [GOAL-BACKEND]
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RuntimeBackend(Protocol):
    """The world boundary. All methods are fail-safe dict-returning."""

    # perception
    def active_window(self) -> Dict[str, Any]: ...
    def list_windows(self) -> List[Dict[str, Any]]: ...
    def focus_window(self, target: str) -> Dict[str, Any]: ...
    def screenshot(self) -> Dict[str, Any]: ...
    def accessibility_tree(self) -> Dict[str, Any]: ...
    def ocr(self) -> Dict[str, Any]: ...
    def vision_query(self, prompt: str) -> Dict[str, Any]: ...

    # browser
    def browser_navigate(self, url: str) -> Dict[str, Any]: ...
    def browser_read(self) -> Dict[str, Any]: ...
    def browser_search(self, query: str) -> Dict[str, Any]: ...

    # visual UI interaction
    def search_ui(self, target: str) -> Dict[str, Any]: ...
    def click(self, target: str, *, evidence: Optional[dict] = None) -> Dict[str, Any]: ...
    def type_text(self, text: str, *, evidence: Optional[dict] = None) -> Dict[str, Any]: ...
    def press_key(self, key: str) -> Dict[str, Any]: ...

    # applications
    def open_app(self, name: str) -> Dict[str, Any]: ...

    # filesystem / coding / terminal
    def fs_create_folder(self, path: str) -> Dict[str, Any]: ...
    def fs_write_file(self, path: str, content: str) -> Dict[str, Any]: ...
    def fs_read_file(self, path: str) -> Dict[str, Any]: ...
    def fs_delete(self, path: str) -> Dict[str, Any]: ...
    def run_command(self, cmd: str, *, cwd: str = "") -> Dict[str, Any]: ...
    def run_tests(self, path: str) -> Dict[str, Any]: ...

    # messaging (desktop-app automation; send requires prior permission)
    def telegram_search_contact(self, name: str) -> Dict[str, Any]: ...
    def telegram_read_latest(self, n: int = 1) -> Dict[str, Any]: ...
    def telegram_send(self, contact: str, text: str) -> Dict[str, Any]: ...
    def gmail_read_email(self, n: int = 1) -> Dict[str, Any]: ...

    # security (only reachable with an authorized scope)
    def security_scan(self, target: str, operations: List[str]) -> Dict[str, Any]: ...


# ═══════════════════════════════════════════════════════════════════
# The real desktop backend
# ═══════════════════════════════════════════════════════════════════

def _ok(success: bool, **kw: Any) -> Dict[str, Any]:
    d = {"success": bool(success)}
    d.update(kw)
    return d


class DesktopBackend:
    """Real implementation — delegates to the EXISTING Diego capabilities."""

    name = "desktop"

    def __init__(self, llm_router=None) -> None:
        from goalruntime.llm import ModelRouter
        self.router = llm_router

    # ── perception ───────────────────────────────────────────────

    def active_window(self) -> Dict[str, Any]:
        try:
            from computer import window_manager as wm
            win = wm.active_window()
            if win is None:
                return _ok(False, reason="no focused window")
            return _ok(True, **win.to_dict())
        except Exception as e:
            return _ok(False, error=str(e))

    def list_windows(self) -> List[Dict[str, Any]]:
        try:
            from computer import window_manager as wm
            return [w.to_dict() for w in wm.list_windows()]
        except Exception as e:
            logger.warning("[GOAL-BACKEND] list_windows failed: %s", e)
            return []

    def focus_window(self, target: str) -> Dict[str, Any]:
        try:
            from computer import window_manager as wm
            r = wm.focus_window(str(target))
            return _ok(bool(r.success), target=str(target),
                       evidence=dict(r.evidence or {}), error=r.error)
        except Exception as e:
            return _ok(False, error=str(e))

    def screenshot(self) -> Dict[str, Any]:
        try:
            import pyautogui  # Diego already depends on it
            img = pyautogui.screenshot()
            return _ok(True, size=list(img.size))
        except Exception as e:
            return _ok(False, error=str(e))

    def accessibility_tree(self) -> Dict[str, Any]:
        try:
            from computer import accessibility as acc
            return _ok(True, **(acc.current_tree() or {}))
        except ImportError:
            return _ok(False, reason="accessibility tier unavailable")
        except Exception as e:
            return _ok(False, error=str(e))

    def ocr(self) -> Dict[str, Any]:
        try:
            from vision.ocr_pipeline import ocr_pipeline
            text = ocr_pipeline.extract_text()
            return _ok(bool(text), text=text or "")
        except Exception as e:
            return _ok(False, error=str(e))

    def vision_query(self, prompt: str) -> Dict[str, Any]:
        try:
            from computer.perception import perception
            answer = perception.describe_screen(str(prompt))
            return _ok(bool(answer), text=answer or "")
        except Exception as e:
            return _ok(False, error=str(e))

    # ── browser ──────────────────────────────────────────────────

    def browser_navigate(self, url: str) -> Dict[str, Any]:
        """Navigate the real browser, ensuring the browser tier is live.

        The DOM tier (``computer.browser_controller``) can only act on a page
        it owns, so a fresh desktop session must first attach to (or launch)
        the persistent CDP Chrome. That attach step is the existing Diego
        capability; this method simply sequences it, then re-runs the action
        and reports which path produced the evidence.
        """
        url = str(url)
        first = self._computer("browser_navigate", {"url": url})
        if first.get("success"):
            return self._flatten_nav(first, url)

        ensured = self.ensure_browser()
        if not ensured.get("success"):
            # Honest unavailability: surface BROWSER_SESSION_UNAVAILABLE (with
            # the exact reason + remediation) on the action result so the
            # runtime reports WHY instead of pretending the task succeeded.
            first["browser_ensure"] = ensured
            if ensured.get("unavailable"):
                first.setdefault("unavailable", True)
                first["error"] = str(ensured.get("error") or first.get("error")
                                     or ensured.get("error"))
            return first

        retry = self._computer("browser_navigate", {"url": url})
        if retry.get("success"):
            retry["via"] = "cdp_attach_retry"
            return self._flatten_nav(retry, url)
        retry["browser_ensure"] = ensured
        retry["first_attempt"] = first
        return retry

    @staticmethod
    def _flatten_nav(result: Dict[str, Any], requested: str) -> Dict[str, Any]:
        """Expose the requested URL at top level for landing verification.

        The skills' verifier compares the host in ``result["url"]`` (what was
        ASKED for) against the host the browser reports afterwards (what was
        actually shown). Keeping ``url`` = requested preserves the contract the
        deterministic fakes implement, and ``observed_url`` is added as extra
        evidence so a redirect (mail.google.com → a marketing page when not
        signed in) is reported as a FAILED navigation, not a silent success.
        """
        ev = result.get("evidence") or {}
        observed = str(ev.get("url") or "")
        out = dict(result)
        out["url"] = requested
        out["observed_url"] = observed
        return out

    def ensure_browser(self) -> Dict[str, Any]:
        """Attach to the user's EXISTING Chrome session (never fake success).

        Delegates to agent.chrome_session.CurrentChromeSession via the
        BrowserController singleton. When the running Chrome cannot be
        attached this returns success=False with `unavailable=True` and the
        EXACT `BROWSER_SESSION_UNAVAILABLE` reason + remediation — never a
        fresh/temporary profile fallback and never a pretend success.
        """
        try:
            from agent.browser import browser_controller as bc
        except Exception as e:
            return _ok(False, unavailable=True,
                       error=f"browser controller import failed: {e}")
        try:
            if not bc.is_available:
                if not bc.initialize():
                    # Honest BROWSER_SESSION_UNAVAILABLE (reason + remediation)
                    # plus the discovered-session evidence that IS known.
                    return _ok(False, unavailable=True,
                               session=bc.unavailable_evidence(),
                               error=(bc.unavailable_reason
                                      or "browser session unavailable"))
            return _ok(True, attached=bool(bc.is_attached),
                       session=bc.session_evidence(),
                       url=str(bc.get_current_url() or ""))
        except Exception as e:
            return _ok(False, unavailable=True,
                       error=f"browser session attach failed: {e}")

    def browser_read(self) -> Dict[str, Any]:
        """Structured page observation in the GoalRuntime contract.

        ``computer_controller.get_page_state`` nests the real page data under
        ``evidence.page`` (url/title/visible_text). The skills and verifiers
        consume a flat ``url`` / ``page_title`` / ``text`` shape, so normalize
        here rather than leaking the ActionResult envelope into the skills.
        """
        r = self._computer("get_page_state", {})
        page = (r.get("evidence") or {}).get("page") or {}
        if not r.get("success") or not page.get("attached"):
            return _ok(False, url="", page_title="", text="",
                       error=r.get("error") or page.get("error")
                       or "browser not attached")
        return _ok(True,
                   url=str(page.get("url") or ""),
                   page_title=str(page.get("title") or ""),
                   text=str(page.get("visible_text") or ""),
                   perception_method=str(page.get("perception_method") or ""))

    def browser_search(self, query: str) -> Dict[str, Any]:
        return self._computer("browser_search", {"query": str(query)})

    # ── visual UI ────────────────────────────────────────────────

    def search_ui(self, target: str) -> Dict[str, Any]:
        return self._computer("find_candidates", {"target": str(target)})

    def click(self, target: str, *, evidence: Optional[dict] = None) -> Dict[str, Any]:
        return self._computer("click_text", {"target": str(target),
                                             "evidence": dict(evidence or {})})

    def type_text(self, text: str, *, evidence: Optional[dict] = None) -> Dict[str, Any]:
        return self._computer("type_text", {"text": str(text),
                                            "evidence": dict(evidence or {})})

    def press_key(self, key: str) -> Dict[str, Any]:
        return self._computer("press_key", {"key": str(key)})

    # ── applications ─────────────────────────────────────────────

    def open_app(self, name: str) -> Dict[str, Any]:
        return self._computer("open_app", {"app": str(name)})

    # ── filesystem / coding / terminal ───────────────────────────

    def fs_create_folder(self, path: str) -> Dict[str, Any]:
        try:
            p = Path(path).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            return _ok(p.is_dir(), path=str(p))
        except Exception as e:
            return _ok(False, error=str(e))

    def fs_write_file(self, path: str, content: str) -> Dict[str, Any]:
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return _ok(True, path=str(p), bytes=len(content))
        except Exception as e:
            return _ok(False, error=str(e))

    def fs_read_file(self, path: str) -> Dict[str, Any]:
        try:
            p = Path(path).expanduser()
            return _ok(p.exists(), path=str(p),
                       content=p.read_text(encoding="utf-8") if p.exists() else "")
        except Exception as e:
            return _ok(False, error=str(e))

    def fs_delete(self, path: str) -> Dict[str, Any]:
        """DESTRUCTIVE — the permission layer gates calls to this."""
        import shutil
        try:
            p = Path(path).expanduser()
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            else:
                return _ok(True, path=str(p), already_absent=True)
            return _ok(not p.exists(), path=str(p))
        except Exception as e:
            return _ok(False, error=str(e))

    def run_command(self, cmd: str, *, cwd: str = "") -> Dict[str, Any]:
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=60.0, cwd=(cwd or None))
            return _ok(r.returncode == 0, cmd=cmd,
                       stdout=(r.stdout or "")[-4000:],
                       stderr=(r.stderr or "")[-2000:],
                       returncode=r.returncode)
        except subprocess.TimeoutExpired:
            return _ok(False, cmd=cmd, error="command timed out (60s)")
        except Exception as e:
            return _ok(False, cmd=cmd, error=str(e))

    def run_tests(self, path: str) -> Dict[str, Any]:
        """Run pytest in the given project dir (bounded, fail-safe)."""
        return self.run_command(f"{os.environ.get('PYTEST_BIN', 'pytest')} -x -q {path}",
                                cwd=path if Path(path).is_dir() else "")

    # ── messaging via desktop automation (no APIs, no tokens) ────

    def telegram_search_contact(self, name: str) -> Dict[str, Any]:
        r1 = self._computer("open_app", {"app": "telegram"})
        if not r1.get("success"):
            return _ok(False, stage="open_app", error=r1.get("error", ""))
        r2 = self._computer("click_text", {"target": "search"})
        if not r2.get("success"):
            return _ok(False, stage="open_search", error=r2.get("error", ""))
        r3 = self._computer("type_text", {"text": str(name)})
        if not r3.get("success"):
            return _ok(False, stage="type_query", error=r3.get("error", ""))
        found = self._computer("click_text", {"target": str(name)})
        if not found.get("success"):
            return _ok(False, stage="select_contact",
                       error=found.get("error", "contact not found"),
                       contact=str(name))
        return _ok(True, contact=str(name), evidence=found.get("evidence", {}))

    def telegram_read_latest(self, n: int = 1) -> Dict[str, Any]:
        page = self._computer("get_page_state", {})
        if not page.get("success"):
            return _ok(False, error=page.get("error", "telegram not visible"))
        return _ok(True, messages=[str(page.get("text", ""))[:2000]], count=n)

    def telegram_send(self, contact: str, text: str) -> Dict[str, Any]:
        """EXTERNAL_SIDE_EFFECT — PermissionManager gates before this runs."""
        r = self._computer("type_text", {"text": str(text)})
        if not r.get("success"):
            return _ok(False, stage="type", error=r.get("error", ""))
        sent = self._computer("press_key", {"key": "enter"})
        if not sent.get("success"):
            return _ok(False, stage="submit", error=sent.get("error", ""))
        # Post-send verification: message text must be visible in the
        # conversation afterwards (mirrors DesktopGoalEngine hard rule).
        after = self._computer("get_page_state", {})
        visible = str(text) in str(after.get("text", ""))
        return _ok(visible, contact=str(contact), text=str(text),
                   verification="message observed in conversation"
                   if visible else "sent text not observed in conversation")

    def gmail_session_state(self) -> Dict[str, Any]:
        """Deterministic "is the browser profile signed into Gmail?" probe.

        Gmail reading is only meaningful with an authenticated session; the
        evidence is the post-navigation URL (the inbox lives under
        mail.google.com/mail, while a signed-out visitor is redirected to the
        Google Workspace / accounts sign-in pages).
        """
        try:
            from agent.browser import browser_controller as bc
            if not bc.is_available:
                self.ensure_browser()
            url = str(bc.get_current_url() or "")
            title = str(bc.get_page_title() or "")
        except Exception as e:
            return _ok(False, signed_in=False, error=str(e))
        signed_in = ("mail.google.com/mail" in url) and \
            ("sign in" not in title.lower())
        return _ok(True, signed_in=signed_in, url=url, title=title)

    def gmail_read_email(self, n: int = 1) -> Dict[str, Any]:
        nav = self.browser_navigate("https://mail.google.com")
        if not nav.get("success"):
            return _ok(False, stage="navigate", error=nav.get("error", ""))
        state = self._computer("get_page_state", {})
        text = str(state.get("text", ""))
        emails = _parse_email_list(text)
        if n <= len(emails):
            return _ok(True, index=n, email=emails[n - 1], count=len(emails))
        # Distinguish "no session" from "session but parse failure" so callers
        # can report an environment gap instead of a silent capability gap.
        sess = self.gmail_session_state()
        signed_in = bool(sess.get("signed_in"))
        return _ok(
            False, count=len(emails), signed_in=signed_in,
            stage="parse" if signed_in else "session",
            url=str(nav.get("observed_url") or sess.get("url") or ""),
            error=(f"only {len(emails)} email(s) visible" if signed_in
                   else "gmail not signed in for the Diego browser profile "
                        "(navigated to the signed-out Google page)"))

    # ── security ─────────────────────────────────────────────────

    def security_scan(self, target: str, operations: List[str]) -> Dict[str, Any]:
        """Authorized-only probing. The PermissionManager has verified the
        target/scope BEFORE this runs; this method re-checks defensively."""
        results = []
        for op in operations:
            if op == "port_scan":
                r = self.run_command(
                    f"timeout 20 nc -zv {target} 1-1024 2>&1 | tail -50")
                if not r.get("success"):
                    r = self._python_port_scan(target)
            elif op == "recon":
                r = self.run_command(
                    f"timeout 15 host {target} 2>&1 || timeout 15 nslookup {target} 2>&1")
            elif op == "service_enum":
                r = self.run_command(f"timeout 20 nmap -sV --top-ports 100 {target} 2>&1")
            else:
                r = _ok(False, error=f"unsupported operation '{op}'")
            results.append(r)
        return _ok(True, target=str(target), operations=list(operations),
                   results=results)

    def _python_port_scan(self, target: str) -> Dict[str, Any]:
        script = (
            "import socket\n"
            f"t={target!r}\n"
            "open_ports=[]\n"
            "for p in (21,22,25,80,110,143,443,445,3389,5432,8080):\n"
            "    s=socket.socket(); s.settimeout(1.5)\n"
            "    if s.connect_ex((t,p))==0: open_ports.append(p)\n"
            "    s.close()\n"
            "print('open_ports=', open_ports)\n"
        )
        return self.run_command(f"timeout 25 python3 - <<'PYEOF'\n{script}PYEOF")

    # ── internal ─────────────────────────────────────────────────

    def _computer(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route through the existing ComputerController (the ACT boundary)."""
        try:
            from computer.computer_controller import computer_controller
            r = computer_controller.execute(action, params)
            return _ok(bool(r.success), action=action,
                       evidence=dict(r.evidence or {}),
                       method=r.method, error=r.error)
        except Exception as e:
            logger.warning("[GOAL-BACKEND] computer action %s failed: %s",
                           action, e)
            return _ok(False, action=action, error=str(e))


def _parse_email_list(text: str) -> List[Dict[str, Any]]:
    """Deterministic extraction of visible email entries from page text."""
    emails: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        if "@" in ln or "Re:" in ln or "Fwd" in ln or len(ln) > 20:
            emails.append({"summary": ln[:200]})
        if len(emails) >= 10:
            break
    return emails


__all__ = ["RuntimeBackend", "DesktopBackend"]
