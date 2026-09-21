"""GoalRuntime skills — structured contracts & execution (simulated backend)."""

from __future__ import annotations

import pytest

from goalruntime.models import PermissionClass
from goalruntime.skills import (
    ActionSpec, SkillExecutor, SkillRegistry, terminal_permission,
    BROWSER_SKILL, TELEGRAM_SKILL, GMAIL_SKILL, FILESYSTEM_SKILL,
    CODING_SKILL, TERMINAL_SKILL, SECURITY_SKILL, VISION_SKILL,
    VISUAL_UI_SKILL,
)
from tests.fakes_goal_backend import SimulatedBackend


@pytest.fixture
def sandbox(tmp_path):
    return tmp_path


@pytest.fixture
def backend(sandbox):
    return SimulatedBackend(sandbox)


@pytest.fixture
def registry():
    return SkillRegistry()


@pytest.fixture
def executor(registry):
    return SkillExecutor(registry)


def _run(executor, backend, skill_id, action, params=None):
    from goalruntime.models import Subgoal
    sub = Subgoal(skill_id=skill_id, action=action,
                  params=dict(params or {}))
    return executor.execute(sub, backend, {})


# ═══════════════════════════════════════════════════════════════════
# Registry: nine generic skills, no per-app branching
# ═══════════════════════════════════════════════ capabilities ═════

def test_nine_generic_skills_registered(registry):
    assert set(registry.ids()) == {
        "browser", "visual_ui", "telegram", "gmail", "filesystem",
        "coding", "terminal", "security", "vision"}


def test_every_action_exposes_full_contract(registry):
    """preconditions / expected effect / verification / permission/evidence."""
    for sid in registry.ids():
        skill = registry.get(sid)
        assert skill.capabilities, sid
        for name, spec in skill.actions.items():
            assert isinstance(spec, ActionSpec), (sid, name)
            assert isinstance(spec.permission_class, PermissionClass)
            assert spec.expected_effect, (sid, name)
            assert callable(spec.executor), (sid, name)
            assert spec.verifier is None or callable(spec.verifier)


def test_find_capable_resolves_capability_names(registry):
    assert registry.find_capable("telegram.send_message") is TELEGRAM_SKILL
    assert registry.find_capable("browser.search") is BROWSER_SKILL
    assert registry.find_capable("send_message") is TELEGRAM_SKILL
    assert registry.find_capable("nope.nothing") is None


# ═══════════════════════════════════════════════════════════════════
# Browser
# ═══════════════════════════════════════════════════════════════════

def test_browser_navigate_instagram_profile(backend, executor):
    r = _run(executor, backend, "browser", "navigate",
             {"url": "instagram.com/nashedi"})
    assert r.success and r.verified
    assert backend.page_title == "@nashedi • Instagram photos and videos"
    assert r.observation.data["page_title"]


def test_browser_navigate_adds_scheme(backend, executor):
    r = _run(executor, backend, "browser", "navigate", {"url": "example.com"})
    assert backend.current_url == "https://example.com"


def test_browser_navigate_verification_catches_wrong_landing(backend, executor):
    """If the browser lands somewhere else than requested, NOT verified."""
    class _Redirect(SimulatedBackend):
        def browser_navigate(self, url):
            r = super().browser_navigate(url)
            self.current_url = "google.com/search?q=wrong"   # landed elsewhere
            return r

    red = _Redirect(backend.sandbox)
    r = _run(executor, red, "browser", "navigate",
             {"url": "instagram.com/nashedi"})
    assert not r.verified


# ═════════════════════════════ DesktopGoalEngine-parity flows ═════

def test_telegram_contact_search_and_send_confirmation(backend, executor):
    backend.seed_telegram("Rahul", ["hi there"])
    r = _run(executor, backend, "telegram", "search_contact",
             {"contact": "Rahul"})
    assert r.success and r.verified
    assert backend.current_contact == "Rahul"

    # Send requires EXTERNAL permission → its result carries the class.
    r2 = _run(executor, backend, "telegram", "send_message",
              {"contact": "Rahul", "text": "hello Rahul"})
    assert r2.permission_class is PermissionClass.EXTERNAL_SIDE_EFFECT
    assert backend.current_contact == "Rahul"
    out = [m for m in backend.contacts["Rahul"]
           if m["dir"] == "out" and m["text"] == "hello Rahul"]
    assert out  # actually delivered into the conversation


def test_telegram_send_verification_catches_silent_failure(backend, executor):
    """When the text never lands in the conversation, send is NOT verified."""
    backend.seed_telegram("Rahul", [])
    backend.telegram_open = True
    backend.current_contact = "Rahul"
    # Corrupt the delivery path: the message never becomes visible.
    original = backend.contacts["Rhul" if False else "Rahul"]
    backend.contacts["Rahul"] = original

    class _Ghost(SimulatedBackend):
        def telegram_send(self, contact, text):
            # pretend to type/enter but the conversation never shows it
            return {"success": True, "contact": contact, "text": text,
                    "verification": "sent text not observed in conversation"}

    ghost = _Ghost(backend.sandbox)
    ghost.telegram_open = True
    ghost.current_contact = "Rhul" if False else "Rahul"
    r = _run(executor, ghost, "telegram", "send_message",
             {"contact": "Rahul", "text": "invisible"})
    assert r.success is False or r.verified is False


def test_telegram_read_latest(backend, executor):
    backend.seed_telegram("Priya", ["first", "second message"])
    backend.open_app("telegram")
    backend.current_contact = "Priya"
    r = _run(executor, backend, "telegram", "read_latest", {"n": 1})
    assert r.success and r.verified
    assert r.artifacts[0].name == "latest_messages"
    assert "second message" in str(r.artifacts[0].value)


def test_gmail_first_second_third(backend, executor):
    backend.seed_gmail([
        {"from": "a@x.com", "subject": "First"},
        {"from": "b@x.com", "subject": "Second"},
        {"from": "c@x.com", "subject": "Third"},
    ])
    for n, subject in ((1, "First"), (2, "Second"), (3, "Third")):
        r = _run(executor, backend, "gmail", "read_email", {"n": n})
        assert r.success and r.verified
        assert r.observation.data or r.evidence
        assert r.evidence.get("email", {}).get("subject") == subject


# ═══════════════════════════════════════════════════════════════════
# Filesystem + coding: REAL execution in a sandbox
# ═════════ from tests.fakes_goal_backend import SimulatedBackend ═══

def test_fs_create_folder(backend, executor):
    path = str(backend.sandbox / "new_folder")
    r = _run(executor, backend, "filesystem", "create_folder", {"path": path})
    assert r.success and r.verified
    from pathlib import Path
    assert Path(path).is_dir()


def test_calculator_generate_execute_and_test_fix_loop(backend, executor):
    from goalruntime.templates import (CALCULATOR_TEMPLATE,
                                       CALCULATOR_TEST_TEMPLATE)
    ws = backend.sandbox / "calc"
    # 1. generate the calculator + its tests
    r = _run(executor, backend, "coding", "generate_code",
             {"path": str(ws / "calculator.py"), "content": CALCULATOR_TEMPLATE})
    assert r.success
    r = _run(executor, backend, "coding", "generate_code",
             {"path": str(ws / "test_calculator.py"),
              "content": CALCULATOR_TEST_TEMPLATE})
    assert r.success
    # 2. execute — output lands in evidence
    r = _run(executor, backend, "coding", "run_command",
             {"cmd": f"python3 {ws / 'calculator.py'}"})
    assert r.success and r.verified
    assert "Calculator ready" in r.evidence.get("stdout", "")
    # 3. tests pass
    r = _run(executor, backend, "coding", "run_tests", {"path": str(ws)})
    assert r.success and r.verified
    # 4. test/fix loop: break it → tests fail → regenerate → tests pass
    (ws / "calculator.py").write_text("def add(a, b):\n    return a - b\n")
    r = _run(executor, backend, "coding", "run_tests", {"path": str(ws)})
    assert not r.success
    r = _run(executor, backend, "coding", "generate_code",
             {"path": str(ws / "calculator.py"), "content": CALCULATOR_TEMPLATE})
    assert r.success
    r = _run(executor, backend, "coding", "run_tests", {"path": str(ws)})
    assert r.success and r.verified


# ═══════════════════════════════════════════════════════════════════
# Terminal classification
# ═════ READ-ONLY commands run automatically (permission layer) ═════

@pytest.mark.parametrize("cmd,expected", [
    ("ls -la", PermissionClass.READ_ONLY),
    ("cat file.txt", PermissionClass.READ_ONLY),
    ("whoami", PermissionClass.READ_ONLY),
    ("mkdir -p a/b", PermissionClass.REVERSIBLE_LOCAL),
    ("python3 script.py", PermissionClass.REVERSIBLE_LOCAL),
    ("pytest -q", PermissionClass.REVERSIBLE_LOCAL),
    ("rm -rf /", PermissionClass.DESTRUCTIVE),
    ("shutdown now", PermissionClass.DESTRUCTIVE),
    ("curl http://evil.example.com", PermissionClass.EXTERNAL_SIDE_EFFECT),
])
def test_terminal_permission(cmd, expected):
    assert terminal_permission(cmd) is expected


def test_terminal_run(backend, executor):
    r = _run(executor, backend, "terminal", "run_command",
             {"cmd": "echo hello"})
    assert r.success and r.verified
    assert "hello" in r.evidence.get("stdout", "")


# ═══════════════════════════════════════════════════════════════════
# Security (permission-gated at the runtime; skill executes the probe)
# ═══════════════════════════════════════════════════════════════════

def test_security_scan_collects_findings(backend, executor):
    r = _run(executor, backend, "security", "scan",
             {"target": "staging.example.com", "operations": ["recon"]})
    assert r.success and r.verified
    assert r.evidence.get("target") == "staging.example.com"
    assert r.evidence.get("results")


# ═══════════════════════════════════════════════════════════════════
# Vision / OCR
# ═══════════════════════════════════════════════════════════════════

def test_vision_ocr(backend, executor):
    backend.ocr_text = "Hello Instagram profile header"
    r = _run(executor, backend, "vision", "ocr", {})
    assert r.success and r.verified
    assert r.artifacts[0].name == "screen_text"


def test_vision_describe(backend, executor):
    backend.vision_answer = "A browser window with the Instagram login page"
    r = _run(executor, backend, "vision", "describe", {"prompt": "describe"})
    assert r.success and r.verified


def test_unknown_skill_or_action_fails_honestly(backend, executor):
    r = _run(executor, backend, "nosuch", "action", {})
    assert not r.success and "unknown skill" in r.error
    r = _run(executor, backend, "browser", "nosuch", {})
    assert not r.success and "no action" in r.error
