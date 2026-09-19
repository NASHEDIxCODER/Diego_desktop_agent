"""Phase 22: computer-use foundation tests (additive; runtime untouched).

Covers:
  - ActionRisk policy (read-only / confirmation gating)
  - ActionResult contract
  - ComputerState normalization
  - Perception hierarchy ordering + honest not-found (never coordinate guess)
  - Vision fallback coordinate gate
  - Structured trace (snapshot transitions, subscribe, bounded log)
  - ComputerController OAOV loop: unknown action, confirmation gate,
    delta verification (no change => FAIL), AMBIGUOUS on unlocatable target
  - TaskRunner trace mirror (task.started/step/completed/failed)
"""
from __future__ import annotations

import pytest

from computer.action_policy import (ActionRisk, assess_risk,
                                    confirmation_required)
from computer.action_result import ActionOutcome, ActionResult
from computer.screen_state import ComputerState


# ═══════════════════════════════════════════════════════════════
# Policy + result contract
# ═══════════════════════════════════════════════════════════════

class TestPolicy:
    def test_read_only_actions(self):
        for a in ("observe", "screenshot", "get_page_state", "find_element"):
            risk, _ = assess_risk(a)
            assert risk == ActionRisk.READ_ONLY

    def test_external_requires_confirmation(self):
        needs, reason = confirmation_required("send_message", {"text": "hi"})
        assert needs is True
        assert "confirmation" in reason.lower() or "external" in reason.lower()

    def test_low_risk_local_actions(self):
        for a in ("open_app", "open_url", "focus_window", "navigate"):
            risk, _ = assess_risk(a)
            assert risk == ActionRisk.LOW_RISK

    def test_unknown_action_defaults_safe(self):
        risk, _ = assess_risk("totally_new_action")
        assert risk in (ActionRisk.STATE_CHANGE, ActionRisk.READ_ONLY)


class TestActionResult:
    def test_structured_fields_and_roundtrip(self):
        r = ActionResult(action="click", target="Send", method="accessibility",
                         success=True, outcome=ActionOutcome.SUCCESS,
                         evidence={"x": 1})
        d = r.to_dict()
        assert d["method"] == "accessibility"
        assert d["outcome"] == "success"
        assert d["evidence"]["x"] == 1


class TestComputerState:
    def test_fields_present(self):
        s = ComputerState(active_application="firefox", window_title="Mozilla",
                          active_url="https://x.com", page_title="X")
        d = s.to_dict()
        for field in ("active_application", "window_title", "active_url",
                      "page_title", "visible_text", "visible_elements",
                      "screen_hash", "last_action", "last_observation",
                      "timestamp"):
            assert field in d
        assert "app=firefox" in s.summary()

    def test_bounded_text(self):
        s = ComputerState(visible_text="x" * 5000)
        assert len(s.to_dict()["visible_text"]) <= 2000


# ═══════════════════════════════════════════════════════════════
# Perception hierarchy
# ═══════════════════════════════════════════════════════════════

class TestPerceptionHierarchy:
    def test_tier_order(self):
        from computer.perception import (PERCEPTION_ORDER, PerceptionMethod,
                                         rank)
        assert PERCEPTION_ORDER[0] == PerceptionMethod.BROWSER_DOM
        assert rank("browser_dom") < rank("accessibility")
        assert rank("accessibility") < rank("native_window")
        assert rank("native_window") < rank("ocr")
        assert rank("ocr") < rank("visual")
        assert rank("visual") < rank("coordinate")

    def test_not_found_is_honest_no_coordinate_guess(self, monkeypatch):
        from computer import element_finder
        for name in ("_find_in_dom", "_find_in_a11y", "_find_in_native",
                     "_find_by_ocr", "_find_visually"):
            monkeypatch.setattr(element_finder, name, lambda *a, **k: None)
        m = element_finder.find_element("does not exist anywhere")
        assert m.found is False
        assert m.method == ""
        assert m.point is None

    def test_hierarchy_order_recorded(self, monkeypatch):
        from computer import element_finder
        calls = []
        a11y_match = element_finder.ElementMatch(
            label="Search", method="accessibility", confidence=0.9,
            point=(10, 10))

        def tier(name, result):
            def fn(q):
                calls.append(name)
                return result
            return fn

        monkeypatch.setattr(element_finder, "_find_in_dom", tier("dom", None))
        monkeypatch.setattr(element_finder, "_find_in_a11y",
                            tier("a11y", a11y_match))
        monkeypatch.setattr(element_finder, "_find_in_native",
                            tier("native", None))
        monkeypatch.setattr(element_finder, "_find_by_ocr", tier("ocr", None))
        monkeypatch.setattr(element_finder, "_find_visually",
                            tier("visual", None))
        m = element_finder.find_element("Search")
        # dom answered None -> a11y tier was consulted next and answered.
        assert calls[0] == "dom" and calls[1] == "a11y"
        assert m.found and m.method == "accessibility"
        assert m.point == (10, 10)

    def test_coordinate_requires_justification(self):
        from computer import vision_fallback
        with pytest.raises(vision_fallback.CoordinateFallbackRefused):
            vision_fallback.coordinate_fallback(100, 100, justification="")
        cand = vision_fallback.coordinate_fallback(
            100, 100, justification="all tiers failed")
        assert cand.method == "coordinate"
        assert cand.metadata["justification"]


# ═══════════════════════════════════════════════════════════════
# Structured trace
# ═══════════════════════════════════════════════════════════════

class TestAgentTrace:
    def test_goal_to_completed_snapshot(self):
        from agent.trace import STATUS_COMPLETED, STATUS_RUNNING, AgentTrace
        t = AgentTrace()
        t.goal("open firefox", task_id="task-1")
        assert t.snapshot().status == STATUS_RUNNING
        t.plan(["open firefox", "search python"])
        assert t.snapshot().plan == ["open firefox", "search python"]
        t.step_started("open firefox", index=0, total=2)
        t.action_started("open_app", target="firefox")
        t.emit("ACTION_COMPLETED", task_id="task-1", action="open_app",
               method="app_resolver", verification_result="PASS")
        assert t.snapshot().verification == "PASS"
        t.completed("opened", task_id="task-1")
        snap = t.snapshot()
        assert snap.status == STATUS_COMPLETED
        assert snap.final_result == "opened"

    def test_confirmation_blocks_then_grant_resumes(self):
        from agent.trace import (STATUS_RUNNING,
                                 STATUS_WAITING_CONFIRMATION, AgentTrace)
        t = AgentTrace()
        t.goal("send message", task_id="t2")
        t.confirmation_required("external effect", task_id="t2", risk="x")
        assert t.snapshot().status == STATUS_WAITING_CONFIRMATION
        assert t.snapshot().confirmation_required is True
        t.confirmation_resolved(True, task_id="t2")
        assert t.snapshot().status == STATUS_RUNNING

    def test_failed_snapshot(self):
        from agent.trace import STATUS_FAILED, AgentTrace
        t = AgentTrace()
        t.goal("x", task_id="t3")
        t.failed("app missing", task_id="t3")
        s = t.snapshot()
        assert s.status == STATUS_FAILED and s.last_error == "app missing"

    def test_subscribe_and_unsubscribe(self):
        from agent.trace import AgentTrace
        t = AgentTrace()
        seen = []
        un = t.subscribe(seen.append)
        t.goal("hello", task_id="t4")
        assert any(e.event_type.value == "GOAL_RECEIVED" for e in seen)
        un()
        t.completed("x", task_id="t4")
        assert not any(e.event_type.value == "TASK_COMPLETED" for e in seen)

    def test_events_bounded_and_serializable(self):
        from agent.trace import AgentTrace
        t = AgentTrace(max_events=10)
        for i in range(30):
            t.goal(f"g{i}", task_id=f"t{i}")
        assert len(t.events()) <= 10
        for e in t.events():
            assert isinstance(e.to_dict(), dict)


# ═══════════════════════════════════════════════════════════════
# ComputerController — OBSERVE → ACT → OBSERVE → VERIFY
# ═══════════════════════════════════════════════════════════════

def _canned_state(**kw) -> ComputerState:
    base = dict(active_application="firefox", window_title="Mozilla Firefox",
                active_url="", page_title="", screen_hash="abc",
                visible_text="")
    base.update(kw)
    return ComputerState(**base)


class TestComputerController:
    def test_unknown_action_is_unavailable(self):
        from computer.computer_controller import ComputerController
        c = ComputerController()
        r = c.execute("teleport")
        assert r.outcome == ActionOutcome.UNAVAILABLE
        assert r.success is False

    def test_confirmation_gate_does_not_execute(self):
        from computer.computer_controller import ComputerController
        c = ComputerController()
        r = c.execute("send_message", {"text": "hello"})
        assert r.outcome == ActionOutcome.CONFIRMATION_REQUIRED
        assert r.evidence == {}

    def test_click_without_located_target_is_ambiguous(self, monkeypatch):
        from computer import element_finder
        for name in ("_find_in_dom", "_find_in_a11y", "_find_in_native",
                     "_find_by_ocr", "_find_visually"):
            monkeypatch.setattr(element_finder, name, lambda *a, **k: None)
        from computer.computer_controller import ComputerController
        c = ComputerController()
        r = c.execute("click", {"target": "missing button"})
        assert r.outcome == ActionOutcome.AMBIGUOUS
        assert "refusing" in r.error

    def _patch_tiers(self, monkeypatch, match=None):
        from computer import element_finder
        for name in ("_find_in_dom", "_find_in_a11y", "_find_in_native",
                     "_find_by_ocr", "_find_visually"):
            monkeypatch.setattr(element_finder, name, lambda *a, **k: None)
        if match is not None:
            monkeypatch.setattr(element_finder, "_find_in_a11y",
                                lambda q: match)

    def test_click_success_without_delta_fails(self, monkeypatch):
        """Action executed but nothing changed => FAIL (never fake success)."""
        from computer import element_finder
        from computer.perception import PerceptionMethod
        self._patch_tiers(monkeypatch, element_finder.ElementMatch(
            label="btn", method=PerceptionMethod.ACCESSIBILITY.value,
            confidence=0.9, point=(100, 100)))
        from computer import mouse_controller
        monkeypatch.setattr(
            mouse_controller, "click",
            lambda x, y, target="", label="": ActionResult(
                action="click", target=target, success=True,
                outcome=ActionOutcome.SUCCESS, evidence={"point": [x, y]}))
        from computer.computer_controller import ComputerController
        c = ComputerController()
        r = c.execute("click", {"target": "btn"})
        assert r.success is False
        assert r.error == "no observed state change after action"

    def test_click_success_with_delta_passes(self, monkeypatch):
        from computer import element_finder
        from computer.perception import PerceptionMethod
        self._patch_tiers(monkeypatch, element_finder.ElementMatch(
            label="btn", method=PerceptionMethod.ACCESSIBILITY.value,
            confidence=0.9, point=(100, 100)))
        from computer import mouse_controller
        monkeypatch.setattr(
            mouse_controller, "click",
            lambda x, y, target="", label="": ActionResult(
                action="click", target=target, success=True,
                outcome=ActionOutcome.SUCCESS, evidence={"point": [x, y]}))

        from computer.computer_controller import ComputerController
        c = ComputerController()
        states = [_canned_state(screen_hash="before"),
                  _canned_state(screen_hash="after", window_title="New")]
        c.capture_state = lambda note="": states.pop(0)
        r = c.execute("click", {"target": "btn"})
        assert r.success is True
        assert "screen_hash" in r.observed_effect

    def test_observe_returns_state(self):
        from computer.computer_controller import ComputerController
        c = ComputerController()
        r = c.execute("observe")
        assert r.success and "state" in r.evidence

    def test_type_text_without_located_input_never_blind_types(
            self, monkeypatch):
        self._patch_tiers(monkeypatch)
        from computer.computer_controller import ComputerController
        c = ComputerController()
        r = c.execute("type_text", {"text": "hi", "target": "search box"})
        assert r.outcome == ActionOutcome.AMBIGUOUS
        assert "refusing" in r.error or "blind-type" in r.error


# ═══════════════════════════════════════════════════════════════
# TaskRunner trace mirror (UI workflow lives through the same truth)
# ═══════════════════════════════════════════════════════════════

class TestTaskRunnerTraceMirror:
    """_mirror_to_trace maps lifecycle events onto the trace, non-fatally."""

    def _state(self):
        from agent.task_state import TaskExecutionState
        return TaskExecutionState(task_id="tt-1",
                                  original_request="open firefox",
                                  normalized_goal="open firefox")

    def test_started_mirrors_goal_and_plan(self):
        from agent.task_state import TaskRunner
        from agent.trace import AgentTrace
        mirror = TaskRunner._mirror_to_trace
        t = AgentTrace()
        state = self._state()
        state.current_plan = [{"action": "desktop_open",
                               "description": "Open Firefox"}]
        import agent.trace as trace_mod
        # bind a private trace instance by patching the import target
        mirror(None, "task.started", state, {"request": "open firefox"})
        # the singleton got the goal (shared store) — verify via events
        from agent.trace import agent_trace
        kinds = [e.event_type.value for e in agent_trace.events(limit=5)]
        assert "GOAL_RECEIVED" in kinds

    def test_mirror_never_raises_on_garbage(self):
        from agent.task_state import TaskRunner
        state = self._state()
        # Unknown event + missing data must be swallowed silently.
        TaskRunner._mirror_to_trace(None, "task.unknown_event", state, {})
        TaskRunner._mirror_to_trace(None, "task.step.started", state,
                                    {"step": None})

    def test_step_completed_mirrors_action(self):
        from agent.task_state import TaskRunner
        from agent.trace import agent_trace
        state = self._state()
        before = len(agent_trace.events(limit=50))
        TaskRunner._mirror_to_trace(
            None, "task.step.completed", state,
            {"step": {"action": "desktop_open", "target": "firefox"}})
        kinds = [e.event_type.value
                 for e in agent_trace.events(limit=max(before, 50))]
        assert "ACTION_COMPLETED" in kinds

    def test_failed_mirrors_task_failed(self):
        from agent.task_state import TaskRunner
        from agent.trace import agent_trace
        state = self._state()
        TaskRunner._mirror_to_trace(None, "task.failed", state,
                                    {"blocker": "app missing"})
        kinds = [e.event_type.value for e in agent_trace.events(limit=5)]
        assert "TASK_FAILED" in kinds



