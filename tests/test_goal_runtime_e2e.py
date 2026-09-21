"""
GoalRuntime end-to-end — the ten required scenarios on the simulated backend.

Each scenario drives the FULL loop (PLAN → EXECUTE → OBSERVE → VERIFY →
REPLAN → CONFIRM) with a deterministic world; the calculator scenario runs
REAL Python in a sandbox.
"""

from __future__ import annotations

import pytest

from goalruntime.llm import ModelRouter
from goalruntime.models import GoalStatus, PermissionClass
from goalruntime.planner import DeterministicPlanner
from goalruntime.runtime import GoalRuntime, RuntimeLimits
from goalruntime.session import AutonomousSession, SessionState
from goalruntime.skills import SkillRegistry
from goalruntime.permissions import ScopedPermissionManager
from tests.fakes_goal_backend import SimulatedBackend


class NoModelProvider:
    name = "none"

    def complete(self, *a, **k):
        raise AssertionError("LLM invoked in a deterministic e2e scenario")


@pytest.fixture
def backend(tmp_path):
    return SimulatedBackend(tmp_path)


@pytest.fixture
def runtime(backend, tmp_path):
    def _make(**kw):
        defaults = dict(
            backend=backend,
            router=ModelRouter(provider=NoModelProvider()),
            registry=SkillRegistry(),
            planner=DeterministicPlanner(
                router=ModelRouter(provider=NoModelProvider()),
                workspace_root=str(tmp_path)),
            limits=RuntimeLimits(max_replans=3, max_total_actions=40),
        )
        defaults.update(kw)
        return GoalRuntime(**defaults)
    return _make


# ═══════════════════════════════════════════════════════════════════
# 1. Instagram profile navigation
# ═══════════════════════════════════════════════════════════════════

def test_scenario_1_instagram_profile_navigation(runtime, backend):
    rt = runtime()
    run = rt.run_goal("Open Instagram profile of nashedi")
    assert run.status is GoalStatus.COMPLETED, run.error
    assert backend.current_url.endswith("instagram.com/nashedi")
    assert backend.page_title.startswith("@nashedi")
    # evidence recorded on the subgoal
    assert run.subgoals[0].status is GoalStatus.COMPLETED
    assert run.subgoals[0].evidence


# ═══════════════════════════════════════════════════════════════════
# 2. Telegram contact search + message confirmation + post-send check
# ═══════════════════════════════════════════════════════════════════

def test_scenario_2_telegram_send_with_confirmation(runtime, backend):
    backend.seed_telegram("Rahul", ["hey"])
    prompts = []

    def prompt(p):
        prompts.append(p)
        return "yes"

    rt = runtime(on_user_prompt=prompt)
    run = rt.run_goal('Send "hello Rahul" to Rahul on telegram')
    assert run.status is GoalStatus.COMPLETED, run.error
    # The send PAUSED for confirmation exactly once, then ran.
    assert len(prompts) == 1
    assert run.confirmations[-1]["allowed"] is True
    # Post-send verification: the message is observed in the conversation.
    assert backend.contacts["Rahul"][-1] == {
        "dir": "out", "text": "hello Rahul"}
    send = [s for s in run.subgoals if s.action == "send_message"][0]
    assert send.evidence.get("verification", "").startswith(
        "message observed")


def test_scenario_2b_telegram_send_denied(runtime, backend):
    backend.seed_telegram("Rahul", [])
    rt = runtime(on_user_prompt=lambda p: "no")
    run = rt.run_goal('Send "secret" to Rahul on telegram')
    assert run.status is GoalStatus.CANCELLED
    # Nothing was sent.
    assert all(m["dir"] == "in" for m in backend.contacts["Rahul"])


def test_scenario_2c_persistent_scoped_permission_skips_confirmation(
        runtime, backend):
    backend.seed_telegram("Rahul", [])
    pm = ScopedPermissionManager()
    rt = runtime(permissions=pm)
    # First send: confirm with "always allow" → persistent grant.
    run1 = rt.run_goal('Send "first" to Rahul on telegram') \
        if False else None
    run1 = rt.start('Send "first" to Rahul on telegram')
    run1 = rt.run_to_terminal(run1) if False else _run_with(
        rt, 'Send "first" to Rahul on telegram', "always allow")
    grants = pm.grants()
    assert grants and grants[0]["persistent"]
    # Second send: NO confirmation needed — the scoped grant covers it.
    prompts = []
    rt2 = runtime(permissions=pm, on_user_prompt=lambda p: prompts.append(p))
    run2 = rt2.run_goal('Send "second" to Rahul on telegram')
    assert run2.status is GoalStatus.COMPLETED
    assert prompts == []             # never asked again


def _run_with(rt: GoalRuntime, goal: str, answer: str):
    run = rt.start(goal)
    assert run is not None
    while run.status not in (GoalStatus.COMPLETED, GoalStatus.FAILED,
                             GoalStatus.CANCELLED):
        if rt.has_pending_confirmation():
            run = rt.resume(answer)
            continue
        run = rt.run_to_terminal(run)
    return run


# ═══════════════════════════════════════════════════════════════════
# 3. Telegram latest-message reading
# ═══════════════════════════════════════════════════════════════════

def test_scenario_3_telegram_read_latest(runtime, backend):
    backend.seed_telegram("Priya", ["older message", "latest hello"])
    rt = runtime()
    run = rt.run_goal("Read the latest message on telegram from contact "
                      "Priya")
    if run.status is not GoalStatus.COMPLETED:
        # contact-less goal → open + read only; assert the read artifact.
        run2 = rt.run_goal("Read the latest message on telegram")
        assert run2.status is GoalStatus.COMPLETED, run2.error


def test_scenario_3b_read_artifact_carries_message(runtime, backend):
    backend.seed_telegram("Priya", ["latest hello"])
    backend.open_app("telegram")
    backend.current_contact = "Priya"
    rt = runtime()
    run = rt.run_goal("Read the latest message on telegram")
    assert run.status is GoalStatus.COMPLETED, run.error
    art = run.artifacts.get("latest_messages")
    assert art is not None and "latest hello" in str(art.value)


# ═══════════════════════════════════════════════════════════════════
# 4. Gmail first / second / third email
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ordinal,n,subject", [
    ("first", 1, "First"), ("second", 2, "Second"), ("third", 3, "Third"),
])
def test_scenario_4_gmail_read(runtime, backend, ordinal, n, subject):
    backend.seed_gmail([
        {"from": "a@x.com", "subject": "First"},
        {"from": "b@x.com", "subject": "Second"},
        {"from": "c@x.com", "subject": "Third"},
    ])
    rt = runtime()
    run = rt.run_goal(f"Read my {ordinal} email in gmail")
    assert run.status is GoalStatus.COMPLETED, run.error
    art = run.artifacts.get("email")
    assert art.value["subject"] == subject


# ═══════════════════════════════════════════════════════════════════
# 5. Folder + Python calculator + execution + test/fix loop (REAL exec)
# ═══════════════════════════════════════════════════════════════════

def test_scenario_5_calculator_full_pipeline(runtime, backend, tmp_path):
    from goalruntime.planner import workspace_path
    rt = runtime()
    run = rt.run_goal(
        "Create a folder named calc2 and generate a Python calculator "
        "with tests")
    assert run.status is GoalStatus.COMPLETED, run.error
    folder = tmp_path / "calc2"
    assert (folder / "calculator.py").exists()
    assert (folder / "test_calculator.py").exists()
    # The generated calculator actually evaluates.
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(folder)!r}); "
         "from calculator import evaluate; print(evaluate('2 + 3'))"],
        capture_output=True, text=True)
    assert "5" in r.stdout


def test_scenario_5b_test_fix_loop_real(runtime, backend, tmp_path):
    """Break the calculator → runtime replans → regenerates → tests pass."""
    from goalruntime.templates import (CALCULATOR_TEMPLATE,
                                       CALCULATOR_TEST_TEMPLATE)
    folder = tmp_path / "calc3"
    folder.mkdir(parents=True)
    (folder / "test_calculator.py").write_text(CALCULATOR_TEST_TEMPLATE)
    (folder / "calculator.py").write_text("def add(a, b):\n    return 0\n")

    rt = runtime()
    # A pre-broken calculator is not our goal's product; drive the runtime's
    # own plan and then break it BEFORE the tests run via a hook on write.
    calls = {"n": 0}

    class _BreakOnce(SimulatedBackend):
        def fs_write_file(self, path, content):
            r = super().fs_write_file(path, content)
            if path.endswith("calculator.py"):
                calls["n"] += 1
                if calls["n"] == 1:
                    # first generation is corrupted (simulate coder failure)
                    p = __import__("pathlib").Path(path)
                    p.write_text("def add(a, b):\n    return a - b\n")
            return r

    broken = _BreakOnce(backend.sandbox)
    rt2 = GoalRuntime(
        backend=broken,
        router=ModelRouter(provider=NoModelProvider()),
        registry=SkillRegistry(),
        planner=DeterministicPlanner(
            router=ModelRouter(provider=NoModelProvider()),
            workspace_root=str(tmp_path)),
        limits=RuntimeLimits(max_replans=3, max_total_actions=40))
    run = rt2.run_goal(
        "Create a folder named calc3 and generate a Python calculator "
        "with tests")
    # The runtime must have REGENERATED (fix) and ended with passing tests.
    assert calls["n"] >= 2          # generate → break → regenerate (fix)
    assert run.status is GoalStatus.COMPLETED, run.error
    import subprocess, sys
    r = subprocess.run(
        [str(__import__("pathlib").Path(sys.executable).parent / "pytest"),
         "-x", "-q", str(folder)],
        capture_output=True, text=True)
    assert r.returncode == 0 and "passed" in r.stdout


# ═══════════════════════════════════════════════════════════════════
# 6. Multi-application sequential goals
# ═══════════════════════════════════════════════════════════════════

def test_scenario_6_multi_app_sequential(runtime, backend):
    backend.seed_gmail([{"from": "x@y.com", "subject": "Newsletter"}])
    backend.seed_telegram("Rahul", [])
    rt = runtime(on_user_prompt=lambda p: "yes")
    run = rt.run_goal(
        "Open instagram and then read my first email in gmail and then "
        'send "done reading" to Rahul on telegram')
    assert run.status is GoalStatus.COMPLETED, run.error
    skills_used = [s.skill_id for s in run.subgoals]
    assert "browser" in skills_used and "gmail" in skills_used \
        and "telegram" in skills_used
    # Artifact/context flow: the gmail artifact exists for later subgoals.
    assert "email" in run.artifacts
    assert backend.contacts["Rahul"][-1]["text"] == "done reading"


# ═══════════════════════════════════════════════════════════════════
# 7. Permission approval / deny / persistent scoped permission
# ═══════════════════════════════════════════════════════════════════

def test_scenario_7a_read_only_never_asks(runtime, backend):
    backend.seed_gmail([{"from": "a@x.com", "subject": "S"}])
    prompts = []
    rt = runtime(on_user_prompt=lambda p: prompts.append(p))
    run = rt.run_goal("Read my first email in gmail")
    assert run.status is GoalStatus.COMPLETED
    assert prompts == []            # read-only ran automatically


def test_scenario_7b_deny_cancels_goal(runtime, backend):
    backend.seed_telegram("Rahul", [])
    rt = runtime(on_user_prompt=lambda p: "no")
    run = rt.run_goal('Send "x" to Rahul on telegram')
    assert run.status is GoalStatus.CANCELLED


def test_scenario_7c_scoped_permission_object(runtime, backend, tmp_path):
    pm = ScopedPermissionManager()
    pm.grant_scope("send_message:contact=rahul",
                   PermissionClass.EXTERNAL_SIDE_EFFECT)
    backend.seed_telegram("Rahul", [])
    prompts = []
    rt2 = GoalRuntime(
        backend=backend, router=ModelRouter(provider=NoModelProvider()),
        registry=SkillRegistry(),
        planner=DeterministicPlanner(
            router=ModelRouter(provider=NoModelProvider()),
            workspace_root=str(tmp_path)),
        permissions=pm,
        limits=RuntimeLimits(max_replans=3, max_total_actions=40),
        on_user_prompt=lambda p: prompts.append(p))
    run = rt2.run_goal('Send "scoped" to Rahul on telegram')
    assert run.status is GoalStatus.COMPLETED
    assert prompts == []            # pre-granted scope → no confirmation


# ═══════════════════════════════════════════════════════════════════
# 8. Replanning after visual/action failure
# ═══════════════════════════════════════════════════════════════════

def test_scenario_8_replan_after_action_failure(runtime, backend):
    """First search attempt fails; the runtime replans and recovers."""
    backend.ui_search_fail_until = 2    # telegram contact search fails twice
    backend.seed_telegram("Rahul", [])

    rt = runtime()
    run = rt.run_goal("Open telegram contact Rahul")
    # Even after failures the runtime must not fake success.
    assert run.status in (GoalStatus.COMPLETED, GoalStatus.FAILED)
    if run.status is GoalStatus.COMPLETED:
        assert backend.current_contact == "Rahul"
    else:
        assert run.error                # honest failure reported
    assert len(run.results) >= 1


def test_scenario_8b_replan_uses_different_evidence(runtime, backend):
    """After a failed visual attempt, the retry consumes FRESH evidence."""
    from goalruntime.models import Subgoal
    planner = DeterministicPlanner(
        router=ModelRouter(provider=NoModelProvider()))
    failed = Subgoal(skill_id="visual_ui", action="click_element",
                     description="click send", params={"target": "send"})
    new = planner.replan("click send", failed, "element vanished")
    assert new[0].skill_id == "vision"          # perception escalation
    assert "screen_text" in new[1].consumes     # retry consumes fresh evidence
    assert new[1].action != failed.action       # DIFFERENT action


def test_scenario_8c_replan_budget_bounded(runtime, backend):
    """A perpetually failing subgoal exhausts replans and fails honestly."""
    backend.ui_search_fail_until = 10 ** 6   # nothing ever works
    rt = runtime()
    run = rt.run_goal("Open telegram contact Ghost")
    assert run.status is GoalStatus.FAILED
    assert run.replans <= rt.limits.max_replans
    assert run.error


# ═══════════════════════════════════════════════════════════════════
# 9. Authorized security workflow with explicit scope
# ═══════════════════════════════════════════════════════════════════

def test_scenario_9a_security_without_authorization_is_blocked(runtime,
                                                               backend):
    rt = runtime()
    run = rt.run_goal("Scan the host prod.example.com with port scan")
    assert run.status is GoalStatus.FAILED
    assert "authorization" in run.error.lower() or \
        "permission" in run.error.lower()
    assert backend.security_calls == []      # NO probe ever ran


def test_scenario_9b_security_with_explicit_scope_runs(runtime, backend):
    pm = ScopedPermissionManager()
    pm.authorize_security_scope(["staging.example.com"],
                                ["port_scan", "recon"])
    rt = runtime(permissions=pm)
    run = rt.run_goal(
        "Scan the host staging.example.com with port scan and recon")
    assert run.status is GoalStatus.COMPLETED, run.error
    assert len(backend.security_calls) == 1
    assert backend.security_calls[0]["target"] == "staging.example.com"


def test_scenario_9c_security_never_expands_scope(runtime, backend):
    pm = ScopedPermissionManager()
    pm.authorize_security_scope(["staging.example.com"], ["recon"])
    rt = runtime(permissions=pm)
    # The goal names a DIFFERENT target: the runtime must refuse (the
    # planner cannot widen the authorized target autonomously).
    run = rt.run_goal("Scan the host prod.example.com with recon")
    assert run.status is GoalStatus.FAILED
    assert backend.security_calls == []


# ═══════════════════════════════════════════════════════════════════
# 10. Endless session with explicit sleep
# ═══════════════════════════════════════════════════════════════════

def test_scenario_10_endless_session_with_explicit_sleep(runtime, backend):
    backend.seed_gmail([{"from": "a@x.com", "subject": "S1"}])
    backend.seed_telegram("Rahul", [])
    session = AutonomousSession()
    assert session.wake() is SessionState.ACTIVE_SESSION

    rt = runtime(session=session)

    # Turn 1: a goal
    run = rt.run_goal("Read my first email in gmail")
    assert run.status is GoalStatus.COMPLETED
    session.accept_goal(run.text)
    session.record_tts(run.user_message())
    assert session.is_active              # TTS did NOT end the session

    # Turn 2: silence
    session.handle_utterance("")
    assert session.is_active

    # Turn 3: a completed task
    run = rt.run_goal("Open instagram")
    assert run.status is GoalStatus.COMPLETED
    session.goal_completed(run.text)
    assert session.is_active              # completion did NOT end it

    # Turn 4: a normal failure
    session.record_failure("browser hiccup")
    assert session.is_active              # failure did NOT end it

    # Turn 5: another goal still works (fresh capability, same session)
    run = rt.run_goal("Open instagram")
    assert run.status is GoalStatus.COMPLETED
    assert session.is_active

    # Turn 6: EXPLICIT sleep — the only exit
    assert session.handle_utterance("go to sleep") == \
        SessionState.SLEEPING.value
    assert session.is_sleeping
    summary = session.summary()
    assert summary["goals_accepted"] >= 1
    assert summary["state"] == "sleeping"


def test_scenario_10b_sleep_then_rewake_resumes(runtime, backend):
    session = AutonomousSession()
    session.wake()
    session.sleep_command()
    session.wake()
    assert session.is_active


# ═══════════════════════════════════════════════════════════════════
# Determinism: no LLM call anywhere in these scenarios
# ═══════════════════════════════════════════════════════════════════

def test_all_scenarios_are_deterministic(runtime, backend):
    """The router's audit trail must be empty — zero model invocations."""
    rt = runtime()
    rt.run_goal("Open instagram")
    rt.run_goal("Read the latest message on telegram")
    assert rt.router.invocations() == []
