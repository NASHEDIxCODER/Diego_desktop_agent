"""
Deterministic simulated backend for GoalRuntime tests.

NO real I/O: telegram/gmail/browser/windows/security are state machines;
filesystem/terminal/code execution use a temp sandbox so the calculator
generation + test/fix loop is REAL execution, still fully deterministic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ok(success: bool, **kw: Any) -> Dict[str, Any]:
    d = {"success": bool(success)}
    d.update(kw)
    return d


class SimulatedBackend:
    """State-machine backend mirroring the DesktopBackend contract."""

    name = "simulated"

    def __init__(self, sandbox: Path) -> None:
        self.sandbox = sandbox
        # windows: focused app + window inventory
        self._focused = "desktop"
        self.windows: Dict[str, str] = {}   # app -> title
        # telegram state
        self.telegram_open = False
        self.contacts: Dict[str, List[Dict[str, str]]] = {}
        self.current_contact: Optional[str] = None
        # browser state
        self.current_url = ""
        self.page_title = ""
        self.page_text = ""
        # gmail
        self.gmail_emails: List[Dict[str, str]] = []
        # perception knobs (tests control what the ladder sees)
        self.ocr_text = ""
        self.vision_answer = ""
        self.ui_search_hits: List[str] = []
        # failure injection
        self.ui_search_fail_until = 0     # search_ui fails n times
        self.test_fail_until_run = 0      # run_tests fails first n runs
        self._test_runs = 0
        # security
        self.security_calls: List[Dict[str, Any]] = []

    # ── perception ───────────────────────────────────────────────

    def active_window(self) -> Dict[str, Any]:
        title = self.windows.get(self._focused, self._focused.title())
        return _ok(True, title=title, window_class=self._focused)

    def list_windows(self) -> List[Dict[str, Any]]:
        out = []
        for app, title in self.windows.items():
            out.append({"id": f"win-{app}", "title": title, "class": app})
        return out

    def focus_window(self, target: str) -> Dict[str, Any]:
        if target in self.windows:
            self._focused = target
            return _ok(True, target=target)
        return _ok(False, error=f"no window for {target}")

    def screenshot(self) -> Dict[str, Any]:
        return _ok(True, size=[1920, 1080])

    def accessibility_tree(self) -> Dict[str, Any]:
        return _ok(False, reason="accessibility tier unavailable")

    def ocr(self) -> Dict[str, Any]:
        return _ok(bool(self.ocr_text), text=self.ocr_text)

    def vision_query(self, prompt: str) -> Dict[str, Any]:
        return _ok(bool(self.vision_answer), text=self.vision_answer)

    # ── browser ──────────────────────────────────────────────────

    def browser_navigate(self, url: str) -> Dict[str, Any]:
        self.current_url = url
        self._focused = "chrome"
        self.windows.setdefault("chrome", "Google Chrome")
        if "instagram.com" in url:
            handle = url.rstrip("/").split("/")[-1]
            if handle == "instagram.com":
                self.page_title = "Instagram"
                self.page_text = "Login • Instagram"
            else:
                self.page_title = f"@{handle} • Instagram photos and videos"
                self.page_text = (f"{handle} profile page with posts and "
                                  f"followers listed")
        elif "mail.google.com" in url:
            self.page_title = "Inbox - Gmail"
            self.page_text = "\n".join(
                f"{e['from']} - {e['subject']}" for e in self.gmail_emails)
        else:
            self.page_title = url
            self.page_text = f"content of {url}"
        return _ok(True, url=url)

    def browser_read(self) -> Dict[str, Any]:
        return _ok(True, url=self.current_url, page_title=self.page_title,
                   text=self.page_text)

    def browser_search(self, query: str) -> Dict[str, Any]:
        self.current_url = f"google.com/search?q={query.replace(' ', '+')}"
        self.page_title = f"{query} - Google Search"
        self.page_text = (f"Results for {query}: "
                          f"https://result-1.example.com and "
                          f"https://result-2.example.com")
        return _ok(True, url=self.current_url)

    # ── visual UI ────────────────────────────────────────────────

    def search_ui(self, target: str) -> Dict[str, Any]:
        if self.ui_search_fail_until > 0:
            self.ui_search_fail_until -= 1
            return _ok(False, error=f"no element matched {target!r}")
        if any(target.lower() in h.lower() for h in self.ui_search_hits) \
                or target.lower() in self.page_text.lower() \
                or target.lower() in self._focused:
            return _ok(True, evidence={"candidates": [
                {"label": target, "method": "accessibility"}]})
        return _ok(False, error=f"no element matched {target!r}")

    def click(self, target: str, *, evidence: Optional[dict] = None) -> Dict[str, Any]:
        if target.lower() in self._focused or evidence:
            return _ok(True, evidence={"clicked": target})
        return _ok(False, error=f"cannot click {target!r} (no evidence)")

    def type_text(self, text: str, *, evidence: Optional[dict] = None) -> Dict[str, Any]:
        if self._focused == "telegram" and self.current_contact:
            return _ok(True, evidence={"typed": text})
        return _ok(False, error="no text input focused")

    def press_key(self, key: str) -> Dict[str, Any]:
        return _ok(True, evidence={"key": key})

    # ── applications ─────────────────────────────────────────────

    def open_app(self, name: str) -> Dict[str, Any]:
        app = name.lower()
        self._focused = app
        self.windows[app] = f"{app.title()} window"
        if app == "telegram":
            self.telegram_open = True
        return _ok(True, app=app)

    # ── filesystem / coding / terminal ───────────────────────────

    def fs_create_folder(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return _ok(p.is_dir(), path=str(p))

    def fs_write_file(self, path: str, content: str) -> Dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return _ok(True, path=str(p), bytes=len(content))

    def fs_read_file(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return _ok(False, error="not found", content="")
        return _ok(True, path=str(p), content=p.read_text(encoding="utf-8"))

    def fs_delete(self, path: str) -> Dict[str, Any]:
        import shutil
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        return _ok(not p.exists(), path=str(p))

    def run_command(self, cmd: str, *, cwd: str = "") -> Dict[str, Any]:
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
                cwd=cwd or str(self.sandbox))
            return _ok(r.returncode == 0, cmd=cmd, returncode=r.returncode,
                       stdout=(r.stdout or "")[-4000:],
                       stderr=(r.stderr or "")[-2000:])
        except Exception as e:
            return _ok(False, cmd=cmd, error=str(e), returncode=-1)

    def run_tests(self, path: str) -> Dict[str, Any]:
        self._test_runs += 1
        if self._test_runs <= self.test_fail_until_run:
            return _ok(False, returncode=1, cmd=f"pytest {path}",
                       stdout="", stderr="1 failed — injected test failure")
        r = self.run_command(
            f"{Path(sys.executable).parent / 'pytest'} -x -q {path}",
            cwd=path if Path(path).is_dir() else "")
        return r

    # ── messaging ────────────────────────────────────────────────

    def telegram_search_contact(self, name: str) -> Dict[str, Any]:
        if not self.telegram_open:
            if not self.open_app("telegram").get("success"):
                return _ok(False, stage="open_app", error="telegram failed")
        if name not in self.contacts:
            return _ok(False, stage="select_contact",
                       error=f"contact {name!r} not found", contact=name)
        self.current_contact = name
        return _ok(True, contact=name,
                   evidence={"conversation": name, "unread": 1})

    def telegram_read_latest(self, n: int = 1) -> Dict[str, Any]:
        if not self.telegram_open or not self.current_contact:
            return _ok(False, error="no telegram conversation open")
        msgs = self.contacts.get(self.current_contact, [])
        incoming = [m for m in msgs if m["dir"] == "in"]
        if not incoming:
            return _ok(False, error="no incoming messages", messages=[])
        latest = incoming[-1]["text"]
        # The latest incoming text is what a real read-back would observe.
        self.page_text = latest
        return _ok(True, messages=[latest], count=n)

    def telegram_send(self, contact: str, text: str) -> Dict[str, Any]:
        if not self.telegram_open or self.current_contact != contact:
            return _ok(False, stage="type", error="wrong conversation open",
                       contact=contact)
        # type → enter → the message lands in the conversation
        self.contacts[contact].append({"dir": "out", "text": text})
        visible = any(m["text"] == text and m["dir"] == "out"
                      for m in self.contacts[contact])
        return _ok(
            visible, contact=contact, text=text,
            verification=("message observed in conversation" if visible
                          else "sent text not observed in conversation"))

    def gmail_read_email(self, n: int = 1) -> Dict[str, Any]:
        self.open_app("chrome")
        self.browser_navigate("https://mail.google.com")
        if 1 <= n <= len(self.gmail_emails):
            return _ok(True, index=n, email=self.gmail_emails[n - 1],
                       count=len(self.gmail_emails))
        return _ok(False, error=f"only {len(self.gmail_emails)} email(s)",
                   count=len(self.gmail_emails))

    # ── security ─────────────────────────────────────────────────

    def security_scan(self, target: str,
                      operations: List[str]) -> Dict[str, Any]:
        self.security_calls.append({"target": target,
                                    "operations": list(operations)})
        return _ok(True, target=target, operations=list(operations),
                   results=[{"op": op, "open_ports": [22, 443]}
                            for op in operations])

    # ── test helpers ─────────────────────────────────────────────

    def seed_telegram(self, contact: str, incoming: List[str]) -> None:
        self.contacts[contact] = [{"dir": "in", "text": t} for t in incoming]

    def seed_gmail(self, emails: List[Dict[str, str]]) -> None:
        self.gmail_emails = list(emails)
