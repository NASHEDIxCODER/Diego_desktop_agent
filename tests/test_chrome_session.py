"""
CurrentChromeSession — the existing-Chrome attach contract (unit tests).

Every test is offline/deterministic: /proc is faked with temp dirs, CDP
probes and connections are stubbed, and NOTHING here touches the user's
real Chrome or profile.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.chrome_session as cs
from agent.chrome_session import (
    ChromeSessionInfo,
    ChromeSessionUnavailable,
    CurrentChromeSession,
    UNAVAILABLE,
    candidate_debug_ports,
    default_debug_port,
    discover_chrome_session,
    launch_real_chrome,
    major_version,
    parse_chrome_args,
    read_profile_directory,
    singleton_lock_pid,
)
from agent.browser import BrowserController

EVIDENCE_KEYS = {
    "browser_process", "user_data_dir", "profile_directory",
    "connection_method", "authenticated_state", "active_tab",
    "current_url",
}


# ── fakes ────────────────────────────────────────────────────────────

def _fake_proc(tmp_path: Path, *processes: tuple) -> Path:
    """Create a fake /proc: processes = (pid, [args...]), ..."""
    for pid, args in processes:
        d = tmp_path / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "cmdline").write_bytes(b"\0".join(
            a.encode() for a in args) + b"\0")
    return tmp_path


class _FakePage:
    def __init__(self, url="https://www.instagram.com/", title="Instagram"):
        self.url = url
        self.title_calls = 0
        self.fronted = False
        self._title = title

    def title(self):
        self.title_calls += 1
        return self._title

    def bring_to_front(self):
        self.fronted = True


class _FakeContext:
    def __init__(self, page):
        self.pages = [page]


class _FakeBrowser:
    def __init__(self, context):
        self.contexts = [context]


class _FakePlaywright:
    def stop(self):
        pass


# ── switch parsing ───────────────────────────────────────────────────

def test_parse_chrome_args_extracts_switches():
    out = parse_chrome_args([
        "/opt/google/chrome/chrome",
        "--user-data-dir=/home/u/.config/google-chrome",
        "--profile-directory=Profile 2",
        "--remote-debugging-port=9333",
    ])
    assert out == {"user_data_dir": "/home/u/.config/google-chrome",
                   "profile_directory": "Profile 2", "debug_port": 9333}


def test_parse_chrome_args_without_switches_is_clean():
    out = parse_chrome_args(["/opt/google/chrome/chrome"])
    assert out == {"user_data_dir": "", "profile_directory": "",
                   "debug_port": None}


def test_parse_chrome_args_space_form():
    out = parse_chrome_args(["chrome", "--user-data-dir", "/tmp/udd"])
    assert out["user_data_dir"] == "/tmp/udd"


# ── discovery (dynamic, never hardcoded "Default") ───────────────────

def test_discover_running_chrome_with_switches(tmp_path):
    # The parsed profile must exist on disk to be trusted (Chrome can rewrite
    # its argv and truncate values containing spaces) — mirror real Chrome.
    udd = tmp_path / "udd"
    (udd / "Profile 2").mkdir(parents=True, exist_ok=True)
    proc = _fake_proc(tmp_path,
                      (87038, ["/opt/google/chrome/chrome",
                               f"--user-data-dir={udd}",
                               "--profile-directory=Profile 2",
                               "--remote-debugging-port=9333"]),
                      (87078, ["/opt/google/chrome/chrome",
                               "--type=renderer"]))
    info = discover_chrome_session(
        proc_dir=proc,
        chrome_version_fn=lambda **kw: "Chrome/149.0.7827.102")
    assert info.running and info.pid == 87038
    assert info.user_data_dir == str(udd)
    assert info.profile_directory == "Profile 2"   # NOT hardcoded Default
    assert info.debug_port == 9333
    assert "pid=87038" in info.browser_process
    assert major_version(info.version) == 149


def test_discover_uses_default_dir_and_local_state_when_no_switches(
        tmp_path, monkeypatch):
    udd = tmp_path / "udd"
    udd.mkdir()
    (udd / "Local State").write_text(json.dumps({
        "profile": {"last_used": "Profile 5",
                    "info_cache": {"Profile 5": {}, "Default": {}}}}))
    proc = _fake_proc(tmp_path, (500, ["/usr/bin/chromium"]))
    monkeypatch.setenv(cs.ENV_USER_DATA_DIR, str(udd))
    info = discover_chrome_session(
        proc_dir=proc, chrome_version_fn=lambda **kw: "Chromium/153.0.1")
    assert info.user_data_dir == str(udd)
    assert info.profile_directory == "Profile 5"
    assert info.debug_port is None


def test_local_state_last_used_unknown_profile_falls_back_to_default(
        tmp_path):
    udd = tmp_path / "udd2"
    udd.mkdir()
    (udd / "Local State").write_text(json.dumps({
        "profile": {"last_used": "Gone",
                    "info_cache": {"Default": {}}}}))
    assert read_profile_directory(str(udd)) == "Default"


def test_local_state_absent_returns_default(tmp_path):
    assert read_profile_directory(str(tmp_path)) == "Default"


# ── Chrome 144+ existing-session switch (profile Preferences) ────────

def _write_prefs(udd: Path, profile: str, prefs: dict) -> None:
    d = udd / profile
    d.mkdir(parents=True, exist_ok=True)
    (d / "Preferences").write_text(json.dumps(prefs))


def test_read_profile_pref_reads_nested_value(tmp_path):
    _write_prefs(tmp_path, "Profile 1",
                 {"devtools": {"remote_debugging": {"user-enabled": True}}})
    assert cs.read_profile_pref(str(tmp_path), "Profile 1",
                                cs.REMOTE_DEBUGGING_PREF) is True


def test_read_profile_pref_absent_is_none(tmp_path):
    _write_prefs(tmp_path, "Default", {"devtools": {}})
    assert cs.read_profile_pref(str(tmp_path), "Default",
                                cs.REMOTE_DEBUGGING_PREF) is None
    assert cs.read_profile_pref(str(tmp_path), "Gone",
                                cs.REMOTE_DEBUGGING_PREF) is None


def test_read_profile_pref_unreadable_file_is_none(tmp_path):
    d = tmp_path / "Default"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Preferences").write_text("{ this is not json")
    assert cs.read_profile_pref(str(tmp_path), "Default",
                                cs.REMOTE_DEBUGGING_PREF) is None


def test_remote_debugging_state_enabled_and_disabled(tmp_path):
    _write_prefs(tmp_path, "Default",
                 {"devtools": {"remote_debugging": {"user-enabled": True}}})
    info = ChromeSessionInfo(user_data_dir=str(tmp_path),
                             profile_directory="Default")
    assert cs.remote_debugging_state(info) == "enabled"
    _write_prefs(tmp_path, "Default",
                 {"devtools": {"remote_debugging": {"user-enabled": False}}})
    assert cs.remote_debugging_state(info) == "disabled"


def test_remote_debugging_state_policy_blocked(tmp_path):
    _write_prefs(tmp_path, "Default", {"devtools": {"remote_debugging": {
        "user-enabled": True, "allowed": False}}})
    info = ChromeSessionInfo(user_data_dir=str(tmp_path),
                             profile_directory="Default")
    assert cs.remote_debugging_state(info) == "policy-blocked"


def test_remote_debugging_state_unknown_without_profile(tmp_path):
    _write_prefs(tmp_path, "Default", {"devtools": {}})
    assert cs.remote_debugging_state(
        ChromeSessionInfo(user_data_dir=str(tmp_path),
                          profile_directory="Default")) == "unknown"
    assert cs.remote_debugging_state(ChromeSessionInfo()) == "unknown"


def test_discovery_records_remote_debugging_state(tmp_path, monkeypatch):
    udd = tmp_path / "udd"
    _write_prefs(udd, "Profile 7", {"devtools": {
        "remote_debugging": {"user-enabled": True}}})
    (udd / "Local State").write_text(json.dumps(
        {"profile": {"last_used": "Profile 7",
                     "info_cache": {"Profile 7": {}}}}))
    proc = _fake_proc(tmp_path, (900, ["/usr/bin/google-chrome"]))
    monkeypatch.setenv(cs.ENV_USER_DATA_DIR, str(udd))
    info = discover_chrome_session(proc_dir=proc,
                                  chrome_version_fn=lambda **kw: "Chrome/149")
    assert info.profile_directory == "Profile 7"
    assert info.extra["remote_debugging"] == "enabled"


def test_unavailable_reason_and_remediation_report_switch_state(
        tmp_path, monkeypatch):
    udd = tmp_path / "udd"
    _write_prefs(udd, "Profile 2", {"devtools": {"remote_debugging": {
        "user-enabled": False}}})                      # switch explicitly OFF
    info = ChromeSessionInfo(running=True, pid=4242,
                             user_data_dir=str(udd),
                             profile_directory="Profile 2",
                             browser_process="chrome pid=4242",
                             version="Chrome/150.0.1")
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    monkeypatch.setattr(cs, "probe_cdp", lambda *a, **k: None)
    monkeypatch.setattr(CurrentChromeSession, "_is_default_dir",
                        staticmethod(lambda udd: False))
    with pytest.raises(ChromeSessionUnavailable) as exc:
        CurrentChromeSession().attach(allow_launch=False)
    assert "disabled" in exc.value.reason
    assert cs.REMOTE_DEBUGGING_PREF in exc.value.reason
    assert cs.REMOTE_DEBUGGING_TOGGLE_URL in exc.value.remediation
    # launching is refused: a second instance must never be started here
    assert not info.launched_by_diego


def test_unavailable_remediation_when_switch_already_enabled(monkeypatch):
    info = _running_no_cdp_info()
    info.extra["remote_debugging"] = "enabled"
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    monkeypatch.setattr(cs, "probe_cdp", lambda *a, **k: None)
    with pytest.raises(ChromeSessionUnavailable) as exc:
        CurrentChromeSession().attach(allow_launch=False)
    assert "ALREADY ON" in exc.value.remediation
    assert "APPROVE" in exc.value.remediation


def test_unavailable_remediation_when_policy_blocked(monkeypatch):
    info = _running_no_cdp_info()
    info.extra["remote_debugging"] = "policy-blocked"
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    monkeypatch.setattr(cs, "probe_cdp", lambda *a, **k: None)
    with pytest.raises(ChromeSessionUnavailable) as exc:
        CurrentChromeSession().attach(allow_launch=False)
    assert "ADMIN POLICY" in exc.value.remediation


def test_singleton_lock_pid(tmp_path):
    lock = tmp_path / "SingletonLock"
    os.symlink("myhost-1234", str(lock))
    assert singleton_lock_pid(str(tmp_path)) == 1234


def test_singleton_lock_missing(tmp_path):
    assert singleton_lock_pid(str(tmp_path)) is None


def test_candidate_ports_order():
    info = ChromeSessionInfo(debug_port=9333)
    info.extra["devtools_active_port"] = 9444
    ports = candidate_debug_ports(info)
    assert ports[0] == 9333
    assert 9444 in ports
    assert ports[-1] == default_debug_port()
    assert len(ports) == len(set(ports))


# ── listener discovery: only ports the browser process OWNS ──────────

def test_listening_ports_owned_by_pid_only(tmp_path):
    """/proc/<pid>/net/tcp is namespace-wide: a foreign LISTEN row (e.g. a
    database or dns service) must NEVER enter the Chrome attach probe."""
    pid_dir = tmp_path / "77"
    (pid_dir / "fd").mkdir(parents=True)
    (pid_dir / "net").mkdir()
    os.symlink("socket:[12345]", str(pid_dir / "fd" / "3"))
    # 1F90 hex = 8080 (owned), 1F91 hex = 8081 (foreign inode 99999).
    (pid_dir / "net" / "tcp").write_text(
        "   sl local rem st tx:rx tr:when retrnsmt uid timeout inode\n"
        "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 1000 0 12345 1\n"
        "   1: 0100007F:1F91 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 0 0 99999 1\n",
        encoding="utf-8")
    assert cs._listening_ports_of_pid(77, proc_dir=tmp_path) == [8080]


def test_listening_ports_without_socket_fds_is_empty(tmp_path):
    pid_dir = tmp_path / "78"
    (pid_dir / "net").mkdir(parents=True)
    (pid_dir / "net" / "tcp").write_text(
        "   sl local rem st tx:rx tr:when retrnsmt uid timeout inode\n"
        "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 0 0 55 1\n", encoding="utf-8")
    # The process owns no socket fd → nothing is a candidate, even though
    # the namespace table shows a listener.
    assert cs._listening_ports_of_pid(78, proc_dir=tmp_path) == []


# ── Chrome 144+ existing-session (approval-mode) endpoint ────────────

def test_devtools_active_ws_parses_the_dap_file(tmp_path):
    dap = tmp_path / "DevToolsActivePort"
    dap.write_text("9222\n/devtools/browser/abc-123\n", encoding="utf-8")
    assert cs.devtools_active_ws(str(tmp_path)) == \
        "ws://127.0.0.1:9222/devtools/browser/abc-123"


def test_devtools_active_ws_absent_is_none(tmp_path):
    assert cs.devtools_active_ws(str(tmp_path)) is None


def test_attach_prefers_approval_mode_ws_endpoint(monkeypatch, tmp_path):
    """When the DAP WebSocket exists it is tried FIRST — the approval-mode
    handshake is the Chrome 144+ existing-session attach."""
    udd = tmp_path / "udd"
    udd.mkdir()
    (udd / "DevToolsActivePort").write_text(
        "9222\n/devtools/browser/abc-123\n", encoding="utf-8")
    info = ChromeSessionInfo(running=True, pid=42, user_data_dir=str(udd),
                             profile_directory="Profile 9",
                             browser_process="chrome pid=42 (chrome)")
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    urls = []
    _page, _ctx, browser = _patch_connect(monkeypatch)

    def connect(url, timeout=10000.0):
        urls.append((url, timeout))
        return ("PW", browser)

    monkeypatch.setattr(cs, "connect_cdp", connect)
    out, _pw, _b, _ctx, _page = CurrentChromeSession().attach(
        allow_launch=False)
    assert urls and urls[0][0] == \
        "ws://127.0.0.1:9222/devtools/browser/abc-123"
    # The approval window covers the user's time to click Allow.
    assert urls[0][1] >= 30_000
    assert "existing_session_approval_mode" in out.connection_method


def test_attach_approval_mode_timeout_is_honest(monkeypatch, tmp_path):
    """A DAP endpoint that never completes the handshake reports exactly
    what the user must do — it is never swapped for a fresh browser."""
    udd = tmp_path / "udd"
    udd.mkdir()
    (udd / "DevToolsActivePort").write_text(
        "9222\n/devtools/browser/abc-123\n", encoding="utf-8")
    info = ChromeSessionInfo(running=True, pid=42, user_data_dir=str(udd),
                             profile_directory="Profile 9",
                             browser_process="chrome pid=42 (chrome)")
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)

    def connect(url, timeout=10000.0):
        raise RuntimeError("Timeout 90000ms exceeded")

    monkeypatch.setattr(cs, "connect_cdp", connect)
    with pytest.raises(ChromeSessionUnavailable) as exc:
        CurrentChromeSession().attach(allow_launch=False)
    assert "handshake" in exc.value.reason
    assert "chrome://inspect" in exc.value.remediation
    assert exc.value.info.profile_directory == "Profile 9"


# ── attach: running chrome with a live endpoint ──────────────────────

def _patch_connect(monkeypatch, page=None, captured=None):
    page = page or _FakePage()
    context = _FakeContext(page)
    browser = _FakeBrowser(context)

    def fake_connect(url):
        if captured is not None:
            captured.append(url)
        return _FakePlaywright(), browser

    monkeypatch.setattr(cs, "connect_cdp", fake_connect)
    return page, context, browser


def test_attach_to_running_chrome_reuses_existing_tab(monkeypatch):
    info = ChromeSessionInfo(pid=100, running=True,
                             user_data_dir="/home/u/udd",
                             profile_directory="Default",
                             debug_port=9333,
                             browser_process="chrome pid=100 (chrome)")
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    monkeypatch.setattr(cs, "probe_cdp",
                        lambda port, timeout=cs._PROBE_TIMEOUT_S:
                        {"Browser": "Chrome/149.0"} if port == 9333 else None)
    captured: list = []
    page, context, browser = _patch_connect(monkeypatch, captured=captured)
    session = CurrentChromeSession()
    out, pw, browser2, context2, page2 = session.attach(allow_launch=False)
    assert captured == ["http://127.0.0.1:9333"]
    assert page2 is page                       # the user's EXISTING tab
    assert page.fronted
    assert out.connection_method == "cdp:running_chrome_debug_port"
    ev = out.evidence()
    assert set(ev) == EVIDENCE_KEYS
    assert ev["connection_method"] == "cdp:running_chrome_debug_port"
    assert ev["current_url"] == page.url


def test_attach_uses_devtools_active_port_fallback(monkeypatch):
    info = ChromeSessionInfo(pid=100, running=True,
                             user_data_dir="/home/u/udd",
                             profile_directory="Default",
                             browser_process="chrome pid=100")
    info.extra["devtools_active_port"] = 9444
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    monkeypatch.setattr(cs, "probe_cdp",
                        lambda port, timeout=cs._PROBE_TIMEOUT_S:
                        {"Browser": "Chrome/149.0"} if port == 9444 else None)
    captured: list = []
    page, context, browser = _patch_connect(monkeypatch, captured=captured)
    out, *_ = CurrentChromeSession().attach(allow_launch=False)
    assert captured == ["http://127.0.0.1:9444"]
    assert "auto_connect_approval_mode" in out.connection_method


# ── attach: running chrome WITHOUT an endpoint → honest failure ─────

def _running_no_cdp_info() -> ChromeSessionInfo:
    return ChromeSessionInfo(pid=87038, running=True,
                             user_data_dir="/home/u/.config/google-chrome",
                             profile_directory="Default",
                             browser_process="chrome pid=87038",
                             version="Chrome/149.0.7827.102")


def test_running_chrome_without_cdp_fails_honestly(monkeypatch):
    info = _running_no_cdp_info()
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    monkeypatch.setattr(cs, "probe_cdp", lambda *a, **k: None)

    def _no_launch(*a, **k):                   # must NEVER be reached
        raise AssertionError("launched a browser while Chrome is running")

    monkeypatch.setattr(cs, "launch_real_chrome", _no_launch)
    with pytest.raises(ChromeSessionUnavailable) as exc:
        CurrentChromeSession().attach(allow_launch=True)  # even if allowed
    assert "Chrome is running" in exc.value.reason
    assert "9222" in exc.value.reason          # probed ports are reported
    assert "auto-connect" in exc.value.remediation


def test_running_default_dir_chrome_136_explains_the_restriction(
        monkeypatch):
    info = _running_no_cdp_info()
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    monkeypatch.setattr(cs, "probe_cdp", lambda *a, **k: None)
    monkeypatch.setattr(CurrentChromeSession, "_is_default_dir",
                        staticmethod(lambda udd: True))
    with pytest.raises(ChromeSessionUnavailable) as exc:
        CurrentChromeSession().attach(allow_launch=False)
    assert "136" in exc.value.reason
    assert "DEFAULT user data dir" in exc.value.reason


def test_unavailable_message_carries_marker():
    exc = ChromeSessionUnavailable("reason", "fix")
    assert str(exc).startswith(UNAVAILABLE)


# ── attach: no chrome running → launch the REAL profile ─────────────

def test_no_chrome_running_launches_real_profile(monkeypatch):
    info = ChromeSessionInfo(running=False, pid=None,
                             user_data_dir="/home/u/.config/google-chrome",
                             profile_directory="Default",
                             browser_process="")
    monkeypatch.setattr(cs, "discover_chrome_session", lambda **kw: info)
    launches = []
    endpoint = {"up": False}

    def fake_probe(port, timeout=cs._PROBE_TIMEOUT_S):
        return {"Browser": "Chrome/149.0"} if endpoint["up"] else None

    def fake_launch(i, port):
        launches.append((i.user_data_dir, i.profile_directory, port))
        i.debug_port = port
        i.launched_by_diego = True
        endpoint["up"] = True               # Chrome exposes CDP after launch
        return True

    monkeypatch.setattr(cs, "launch_real_chrome", fake_launch)
    monkeypatch.setattr(cs, "probe_cdp", fake_probe)
    page, context, browser = _patch_connect(monkeypatch)
    out, *_ = CurrentChromeSession().attach(allow_launch=True)
    assert launches == [("/home/u/.config/google-chrome", "Default", 9222)]
    assert "relaunched_real_profile" in out.connection_method


def test_launch_real_chrome_uses_real_profile_never_temp(
        monkeypatch, tmp_path):
    real_dir = tmp_path / "real-profile"
    real_dir.mkdir()
    info = ChromeSessionInfo(user_data_dir=str(real_dir),
                             profile_directory="Profile 1")
    monkeypatch.setattr(cs, "find_chrome_binary",
                        lambda: "/usr/bin/google-chrome")
    monkeypatch.setenv("DISPLAY", ":0")
    cmds = []

    class _FakePopen:
        def __init__(self, cmd, **kw):
            cmds.append(cmd)

    monkeypatch.setattr(cs.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(cs, "probe_cdp", lambda *a, **k: {"Browser": "x"})
    assert launch_real_chrome(info, 9222) is True
    cmd = cmds[0]
    assert cmd[0] == "/usr/bin/google-chrome"
    assert f"--user-data-dir={real_dir}" in cmd      # the REAL profile
    assert "--profile-directory=Profile 1" in cmd
    assert "--remote-debugging-port=9222" in cmd
    assert "--headless=new" not in cmd               # visible desktop Chrome
    assert info.launched_by_diego


# ── BrowserController honesty ────────────────────────────────────────

class _UnavailableSession:
    def attach(self, **kw):
        raise ChromeSessionUnavailable(
            "Chrome is running (pid 1) but exposes no DevTools endpoint",
            "approve the auto-connect prompt or relaunch with debugging")


def test_browser_controller_unavailable_is_honest(monkeypatch):
    monkeypatch.setattr("agent.browser.chrome_session", _UnavailableSession)
    bc = BrowserController()
    assert bc.initialize() is False
    assert bc.unavailable_reason.startswith(UNAVAILABLE)
    assert "no DevTools endpoint" in bc.unavailable_reason
    assert "Remediation" in bc.unavailable_reason
    assert bc.is_available is False
    ev = bc.unavailable_evidence()
    assert set(ev) == EVIDENCE_KEYS | {"unavailable"}
    assert ev["connection_method"] == "none"


def test_browser_controller_attach_populates_session_evidence(monkeypatch):
    info = ChromeSessionInfo(
        pid=100, running=True, user_data_dir="/home/u/udd",
        profile_directory="Profile 3",
        connection_method="cdp:running_chrome_debug_port",
        browser_process="chrome pid=100 (chrome)")
    page = _FakePage(url="https://mail.google.com/", title="Inbox")

    class _OkSession:
        def attach(self, **kw):
            return (info, _FakePlaywright(), object(), object(), page)

    monkeypatch.setattr("agent.browser.chrome_session", _OkSession)
    bc = BrowserController()
    assert bc.initialize() is True
    ev = bc.session_evidence()
    assert set(ev) == EVIDENCE_KEYS
    assert ev["profile_directory"] == "Profile 3"
    assert ev["current_url"] == "https://mail.google.com/"
    assert ev["active_tab"] == "Inbox"

def test_browser_controller_unavailable_keeps_discovered_identity(monkeypatch):
    """The honest failure still records WHICH Chrome could not be attached."""
    monkeypatch.setattr("agent.browser.chrome_session", _UnavailableSession)
    info = ChromeSessionInfo(
        pid=87038, running=True,
        user_data_dir="/home/u/.config/google-chrome",
        profile_directory="Profile 7",
        browser_process="chrome pid=87038 (/opt/google/chrome/chrome)")
    monkeypatch.setattr("agent.chrome_session.discover_chrome_session",
                        lambda **kw: info)
    bc = BrowserController()
    assert bc.initialize() is False
    assert bc.unavailable_reason.startswith(UNAVAILABLE)
    ev = bc.unavailable_evidence()
    # discovered identity is reported (never silently "unknown") ...
    assert ev["profile_directory"] == "Profile 7"
    assert ev["user_data_dir"] == "/home/u/.config/google-chrome"
    assert "pid=87038" in ev["browser_process"]
    # ... but NOTHING claims a live connection while nothing is attached.
    assert ev["connection_method"] == "none"
    assert ev["unavailable"].startswith(UNAVAILABLE)




# ── BrowserGoalEngine honesty (production path) ──────────────────────

def _make_run():
    from agent.browser_goal import parse_browser_goal
    from agent.browser_goal_engine import BrowserTaskRun
    return BrowserTaskRun(task_id="t-session",
                          goal=parse_browser_goal("Open instagram"))


class _StubBrowserController:
    is_attached = False
    is_available = False

    def initialize(self):
        self.initialize_called = True
        return False

    def session_evidence(self):
        return {k: "" for k in EVIDENCE_KEYS}

    def unavailable_evidence(self):
        ev = {k: "" for k in EVIDENCE_KEYS}
        ev["unavailable"] = (f"{UNAVAILABLE}: Chrome is running but has no "
                             f"DevTools endpoint")
        return ev


def test_engine_fails_honestly_when_session_unavailable(monkeypatch):
    stub = _StubBrowserController()
    monkeypatch.setattr("agent.browser.browser_controller", stub)
    from agent.browser_goal_engine import BrowserGoalEngine
    engine = BrowserGoalEngine(trace=__import__(
        "agent.trace", fromlist=["AgentTrace"]).AgentTrace())
    run = engine.start("Open instagram")
    assert run is not None
    assert run.status.value == "FAILED"
    assert "BROWSER_SESSION_UNAVAILABLE" in (run.error or "")
    assert stub.initialize_called


def test_engine_bound_page_owns_its_browser(monkeypatch):
    from computer import browser_controller as bctl
    bctl.bind_page(object())
    try:
        def _boom(*a, **k):
            raise AssertionError("production attach must not run for a "
                                 "harness-bound page")
        monkeypatch.setattr("agent.browser.browser_controller",
                            type("X", (), {"initialize": _boom})())
        from agent.browser_goal_engine import BrowserGoalEngine
        engine = BrowserGoalEngine()
        run = _make_run()
        assert engine._ensure_browser_session(run) is None
    finally:
        bctl.unbind_page()


def test_engine_injected_harness_owns_its_browser(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("production attach must not run for an "
                             "injected harness")
    monkeypatch.setattr("agent.browser.browser_controller",
                        type("X", (), {"initialize": _boom})())
    from agent.browser_goal_engine import BrowserGoalEngine
    engine = BrowserGoalEngine(controller=object(), observer=object())
    run = _make_run()
    assert engine._ensure_browser_session(run) is None


def test_engine_refreshes_authenticated_state_from_observation(monkeypatch):
    from agent.browser_goal_engine import BrowserGoalEngine
    from agent.browser_context import BrowserContext
    engine = BrowserGoalEngine(controller=object(), observer=object())
    run = _make_run()
    engine._record_session_evidence(run, {"connection_method": "cdp"})
    ctx = BrowserContext(
        current_url="https://www.instagram.com/nashedi/",
        page_title="nashedi",
        authentication_state="authenticated")
    assert ctx.authenticated            # derived view (detect_authenticated)
    engine._refresh_session_evidence(run, ctx)
    ev = engine._session_evidence[run.task_id]
    assert ev["authenticated_state"] == "authenticated"
    assert ev["active_tab"] == "nashedi"
    assert "instagram.com/nashedi" in ev["current_url"]


def test_engine_paused_resume_fails_honestly_without_session(monkeypatch):
    """A paused task must not resume against a dead/different browser."""
    stub = _StubBrowserController()
    monkeypatch.setattr("agent.browser.browser_controller", stub)
    from agent.browser_goal import BrowserStatus
    from agent.browser_goal_engine import BrowserGoalEngine
    engine = BrowserGoalEngine(trace=__import__(
        "agent.trace", fromlist=["AgentTrace"]).AgentTrace())
    run = _make_run()
    run.status = BrowserStatus.WAITING_FOR_USER
    engine._runs[run.task_id] = run
    result = engine.resume("continue", task_id=run.task_id)
    assert result is not None
    assert result.status.value == "FAILED"
    assert "BROWSER_SESSION_UNAVAILABLE" in (result.error or "")
    assert stub.initialize_called


def test_engine_resume_skips_attach_for_injected_harness(monkeypatch):
    """Injected harnesses own their browser: resume never attaches."""
    def _boom(*a, **k):
        raise AssertionError("resume must not attach for an injected harness")
    monkeypatch.setattr("agent.browser.browser_controller",
                        type("X", (), {"initialize": _boom})())
    from agent.browser_goal import BrowserStatus
    from agent.browser_goal_engine import BrowserGoalEngine
    engine = BrowserGoalEngine(controller=object(), observer=object())
    run = _make_run()
    run.status = BrowserStatus.WAITING_FOR_USER
    engine._runs[run.task_id] = run
    assert engine._ensure_browser_session(run) is None


def test_trace_browser_session_carries_the_seven_evidence_fields():
    """AgentTrace.browser_session emits the exact session evidence."""
    from agent.trace import AgentTrace
    from agent.trace_event import TraceEventType
    trace = AgentTrace()
    evidence = {
        "browser_process": "chrome pid=4242 (chrome)",
        "user_data_dir": "/home/u/.config/google-chrome",
        "profile_directory": "Profile 2",
        "connection_method": "cdp:running_chrome_debug_port",
        "authenticated_state": "authenticated",
        "active_tab": "Instagram",
        "current_url": "https://www.instagram.com/",
    }
    event = trace.browser_session(evidence, task_id="t-evidence")
    assert event.event_type == TraceEventType.BROWSER_SESSION
    assert event.task_id == "t-evidence"
    assert EVIDENCE_KEYS <= set(event.evidence)
    for key in EVIDENCE_KEYS:
        assert event.evidence[key] == evidence[key]
    assert event.method == "cdp:running_chrome_debug_port"
