"""GoalRuntime planner — deterministic decomposition tests."""

from __future__ import annotations

import pytest

from goalruntime.llm import ModelRole, ModelRouter
from goalruntime.models import PermissionClass
from goalruntime.planner import DeterministicPlanner, workspace_path


class NoModelProvider:
    name = "none"

    def complete(self, *a, **k):  # any LLM call in a deterministic test = fail
        raise AssertionError("LLM invoked during deterministic planning")


@pytest.fixture
def planner():
    return DeterministicPlanner(router=ModelRouter(
        provider=NoModelProvider()))


# ═══════════════════════════════════════════════════════════════════
# Single-domain decomposition
# ═══════════════════════════════════════════════════════════════════

def test_instagram_profile_navigation(planner):
    subs = planner.plan("Open Instagram profile of nashedi")
    assert len(subs) == 1
    s = subs[0]
    assert (s.skill_id, s.action) == ("browser", "navigate")
    assert s.params["url"] == "instagram.com/nashedi"
    assert s.permission_class is PermissionClass.REVERSIBLE_LOCAL


def test_instagram_bare(planner):
    subs = planner.plan("open instagram")
    assert subs[0].params["url"] == "instagram.com"


def test_telegram_send_with_quoted_message(planner):
    subs = planner.plan('Send "hello there" to Rahul on telegram')
    actions = [(s.skill_id, s.action) for s in subs]
    assert actions == [("telegram", "search_contact"),
                       ("telegram", "send_message")]
    send = subs[1]
    assert send.params == {"contact": "Rahul", "text": "hello there"}
    # The send is declared EXTERNAL: it will hit the confirmation flow.
    assert send.permission_class is PermissionClass.EXTERNAL_SIDE_EFFECT


def test_telegram_send_unquoted_message(planner):
    subs = planner.plan("send good morning to Priya on telegram")
    send = [s for s in subs if s.action == "send_message"][0]
    assert send.params["contact"] == "Priya"
    assert send.params["text"] == "good morning"


def test_telegram_read_latest(planner):
    subs = planner.plan("Read the latest message on telegram")
    actions = [(s.skill_id, s.action) for s in subs]
    assert actions == [("telegram", "open"), ("telegram", "read_latest")]
    assert subs[1].permission_class is PermissionClass.READ_ONLY


def test_gmail_first_second_third(planner):
    for ordinal, n in (("first", 1), ("second", 2), ("third", 3)):
        subs = planner.plan(f"Read my {ordinal} email in gmail")
        assert [(s.skill_id, s.action) for s in subs] == \
            [("gmail", "read_email")], ordinal
        assert subs[0].params["n"] == n


def test_folder_and_calculator(planner):
    subs = planner.plan(
        "Create a folder named calc and generate a Python calculator "
        "with tests")
    actions = [(s.skill_id, s.action) for s in subs]
    assert actions == [
        ("filesystem", "create_folder"),
        ("coding", "generate_code"),
        ("coding", "generate_code"),
        ("coding", "run_tests"),
    ]
    assert subs[0].params["path"] == workspace_path("calc")
    assert "def add(" in subs[1].params["content"]
    assert "def test_add" in subs[2].params["content"]
    # run_tests consumes the generated file artifact (artifact passing)
    assert "file_path" in subs[3].consumes


def test_plain_web_search(planner):
    subs = planner.plan("Search google for python tutorial")
    s = subs[0]
    assert (s.skill_id, s.action) == ("browser", "search")
    assert s.params["query"] == "python tutorial"


def test_security_goal(planner):
    subs = planner.plan(
        "Scan the host staging.example.com with port scan and recon")
    s = subs[0]
    assert (s.skill_id, s.action) == ("security", "scan")
    assert s.params["target"] == "staging.example.com"
    assert set(s.params["operations"]) == {"port_scan", "recon"}
    assert s.permission_class is PermissionClass.SECURITY_TESTING


# ═══════════════════════════════════════════════════════════════════
# Cross-application decomposition
# ═══════════════════════════════════════════════════════════════════

def test_cross_app_browser_then_telegram(planner):
    subs = planner.plan(
        "Search google for python calculator tutorial and send the "
        'result to Rahul on telegram saying "here is what I found"')
    actions = [(s.skill_id, s.action) for s in subs]
    assert actions == [
        ("browser", "search"),
        ("telegram", "search_contact"),
        ("telegram", "send_message"),
    ]
    # Artifact passing: search produces, telegram consumes.
    search = subs[0]
    assert "search_results" in search.produces
    send = subs[2]
    assert send.params["contact"] == "Rahul"
    assert "here is what I found" in send.params["text"]


def test_multi_app_sequential_clause_order(planner):
    subs = planner.plan(
        "Open instagram and then read my first email in gmail")
    actions = [(s.skill_id, s.action) for s in subs]
    assert actions[0] == ("browser", "navigate")
    assert actions[1] == ("gmail", "read_email")


# ═══════════════════════════════════════════════════════════════════
# Replanning — the next attempt is ALWAYS different
# ═══════════════════════════════════════════════════════════════════

def test_replan_visual_failure_escalates_to_perception(planner):
    from goalruntime.models import Subgoal
    failed = Subgoal(skill_id="visual_ui", action="click_element",
                     description="click the send button",
                     params={"target": "send"})
    new = planner.replan("click the send button", failed, "element gone")
    assert [s.skill_id for s in new] == ["vision", "visual_ui"]
    assert new[0].action == "ocr"          # perception escalation first
    assert new[0].consumes == []
    assert "screen_text" in new[1].consumes  # fresh evidence feeds the retry
    assert new[1].action == "find_element"   # DIFFERENT action, not a repeat


def test_replan_test_failure_regenerates_then_reruns(planner):
    from goalruntime.models import Subgoal
    ws = workspace_path("calc")
    failed = Subgoal(skill_id="coding", action="run_tests",
                     description="run tests", params={"path": ws})
    new = planner.replan("generate calculator", failed, "1 failed")
    actions = [(s.skill_id, s.action) for s in new]
    assert actions == [("coding", "generate_code"), ("coding", "run_tests")]
    assert new[0].params["path"] == f"{ws}/calculator.py"
    assert "def add(" in new[0].params["content"]  # known-good fix


def test_planner_budget(planner):
    subs = planner.plan(
        "open instagram and then send a message to a on telegram and "
        "read email and create folder named z")
    assert len(subs) <= DeterministicPlanner().limits.max_subgoals


def test_llm_planner_fallback_parses_json_lines(planner):
    from goalruntime.planner import _parse_llm_plan
    raw = ('{"skill": "browser", "action": "navigate", '
           '"params": {"url": "example.com"}}\n'
           'not json\n'
           '{"skill": "vision", "action": "ocr", "params": {}}')
    subs = _parse_llm_plan(raw)
    assert [(s.skill_id, s.action) for s in subs] == [
        ("browser", "navigate"), ("vision", "ocr")]


def test_novel_goal_without_router_still_returns_a_plan():
    p = DeterministicPlanner(router=None)
    subs = p.plan("something completely unprecedented")
    assert subs and subs[0].skill_id == "vision"


# ═══════════════════════════════════════════════════════════════════
# Explicit destination navigation (URL / domain after a nav verb)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("goal,url", [
    ("Navigate to https://example.com", "https://example.com"),
    ("navigate to https://example.com/a/b?c=1", "https://example.com/a/b?c=1"),
    ("go to github.com", "https://github.com"),
    ("open the website mozilla.org", "https://mozilla.org"),
    ("visit news.ycombinator.com", "https://news.ycombinator.com"),
    ("browse to http://localhost:8000", "http://localhost:8000"),
])
def test_explicit_navigation_plans_deterministically(planner, goal, url):
    subs = planner.plan(goal)
    assert len(subs) == 1
    assert (subs[0].skill_id, subs[0].action) == ("browser", "navigate")
    assert subs[0].params["url"] == url
    assert subs[0].permission_class is PermissionClass.REVERSIBLE_LOCAL


def test_prose_mentioning_a_domain_is_not_a_navigation(planner):
    """A domain in prose (no nav verb) must NOT become a browser goal."""
    subs = planner.plan("explain how python.org works")
    assert not any(s.action == "navigate" for s in subs)


# ═══════════════════════════════════════════════════════════════════
# Destructive goals — verbatim target, DESTRUCTIVE class, never widened
# ═══════════════════════════════════════════════════════════════════

def test_delete_file_by_absolute_path_is_destructive(planner):
    subs = planner.plan("delete the file /tmp/xyz.txt")
    assert len(subs) == 1
    assert (subs[0].skill_id, subs[0].action) == ("filesystem", "delete")
    assert subs[0].params["path"] == "/tmp/xyz.txt"
    assert subs[0].permission_class is PermissionClass.DESTRUCTIVE


def test_delete_folder_name_resolves_inside_workspace(planner):
    subs = planner.plan("remove the folder calc9")
    assert subs[0].action == "delete"
    assert subs[0].params["path"].endswith("/calc9")
    assert subs[0].permission_class is PermissionClass.DESTRUCTIVE


def test_delete_path_is_not_widened_or_normalized(planner):
    """The planner plans the EXACT path; no expansion, no traversal fixup."""
    subs = planner.plan("delete the directory /tmp/../tmp/keep.me")
    assert subs[0].params["path"] == "/tmp/../tmp/keep.me"


def test_destructive_prose_without_target_is_not_hijacked(planner):
    """'remove duplicates from my list' has no target → not a delete goal.

    The deterministic layer must decline (the LLM fallback owns novelty), and
    no DESTRUCTIVE subgoal may be produced for it.
    """
    assert planner._deterministic_plan("remove duplicates from my list") is None


# ═══════════════════════════════════════════════════════════════════
# Explicit URL navigation evidence
# ═══════════════════════════════════════════════════════════════════

def test_explicit_https_url_navigates(planner):
    subs = planner.plan("Navigate to https://nonexistent-smoke.invalid")
    assert (subs[0].skill_id, subs[0].action) == ("browser", "navigate")
    assert subs[0].params["url"] == "https://nonexistent-smoke.invalid"
    assert subs[0].permission_class is PermissionClass.REVERSIBLE_LOCAL


@pytest.mark.parametrize("goal,url", [
    ("go to github.com", "https://github.com"),
    ("open the website mozilla.org", "https://mozilla.org"),
    ("visit example.com/docs", "https://example.com/docs"),
])
def test_bare_host_with_nav_verb_gains_scheme(planner, goal, url):
    subs = planner.plan(goal)
    assert (subs[0].skill_id, subs[0].action) == ("browser", "navigate")
    assert subs[0].params["url"] == url


def test_bare_site_identity_is_not_a_url(planner):
    """'open instagram' stays a site-identity goal (never 'instagram' as host)."""
    subs = planner.plan("open instagram")
    assert subs[0].params["url"] == "instagram.com"


@pytest.mark.parametrize("goal", [
    "open my resume.pdf", "read the notes.txt", "show report.xlsx",
])
def test_dotted_filenames_are_not_navigated(planner, goal):
    """File names must never be mistaken for web hosts."""
    subs = planner._deterministic_plan(goal) or []
    assert not [s for s in subs if s.action == "navigate"], \
        f"{goal!r} was treated as a navigation"


def test_profile_navigation_beats_explicit_host_rule(planner):
    subs = planner.plan("Open Instagram profile of nashedi")
    assert subs[0].params["url"] == "instagram.com/nashedi"


def test_url_clause_inside_composed_goal(planner):
    subs = planner.plan(
        "Navigate to https://example.com and then create a folder named "
        "smoke_x")
    kinds = [(s.skill_id, s.action) for s in subs]
    assert ("browser", "navigate") in kinds
    assert any(k[0] == "filesystem" for k in kinds)
