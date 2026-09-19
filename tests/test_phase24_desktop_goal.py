"""
Phase 24: desktop goal-directed agent tests.

Deterministic unit tests (no real desktop, no Telegram):

  - goal parsing (send/read intents, app aliases, no app = fallthrough)
  - desktop context construction (observed state only, never invented)
  - skill registry (generic registration + telegram semantic hints)
  - contact resolution + ambiguity (Rahul Kumar / Sharma / Singh → ASK)
  - message-send confirmation (external side effect pauses; resume SAME run)
  - post-effect verification: message INSIDE conversation (never frame/window
    change as proof)
  - recovery (missing target / launch failure / no effect — bounded)
  - completion only when the message was verified inside the conversation
  - trace emission (GOAL/PLAN/STEP/OBSERVE/ACTION/VERIFY/CONFIRMATION/COMPLETE)

The engine is driven with a FAKE controller + FAKE observer so no desktop I/O
happens, and the send-confirmation gate is exercised deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from agent.desktop_context import DesktopContext, InteractiveElement
from agent.desktop_goal import (
    DesktopAction,
    DesktopGoalKind,
    DesktopStatus,
    DesktopStep,
    EffectResult,
    parse_desktop_goal,
    verify_expected_effect,
)
from agent.desktop_goal_engine import (
    DesktopGoalEngine,
    DesktopLimits,
    DesktopTaskRun,
)
from agent.desktop_skill_registry import (
    DesktopSkill,
    DesktopSkillRegistry,
)
from agent.trace import AgentTrace


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def fast_limits() -> DesktopLimits:
    return DesktopLimits(wait_timeout_s=0.2, wait_interval_s=0.02,
                         max_waits=2, max_observations=40, max_actions=20)


def el(label: str, kind: str = "button", value: str = "") -> InteractiveElement:
    return InteractiveElement(label=label, kind=kind, value=value)


def contact_ctx(labels: List[str]) -> DesktopContext:
    return DesktopContext(
        active_application="telegram", window_title="Telegram",
        window_class="TelegramDesktop", application_state="present",
        observation_method="accessibility",
        interactive_elements=[el(l) for l in labels],
        visible_text="\n".join(labels),
    )


def conversation_ctx(recipient: str, message: str) -> DesktopContext:
    return DesktopContext(
        active_application="telegram",
        window_title=f"Telegram — {recipient}",
        window_class="TelegramDesktop", application_state="present",
        observation_method="accessibility",
        interactive_elements=[el(recipient), el(message, kind="text")],
        visible_text=message,
    )


# ═══════════════════════════════════════════════════════════════
# 1. Goal parsing
# ═══════════════════════════════════════════════════════════════

class TestGoalParsing:
    def test_send_message_with_recipient_colon(self):
        g = parse_desktop_goal("Open Telegram and send Rahul: I'll call you after 6")
        assert g is not None
        assert g.kind == DesktopGoalKind.SEND_MESSAGE
        assert g.app == "telegram"
        assert g.recipient == "Rahul"
        assert g.message == "I'll call you after 6"

    def test_send_message_with_app(self):
        g = parse_desktop_goal("OPEN TELEGRAM and send Rahul: call me tonight")
        assert g is not None
        assert g.kind == DesktopGoalKind.SEND_MESSAGE
        assert g.app == "telegram"
        assert g.recipient == "Rahul"
        assert g.message == "call me tonight"

    def test_no_app_message_falls_through(self):
        assert parse_desktop_goal("send Rahul: hello") is None

    def test_open_telegram_alone_is_not_a_message_goal(self):
        assert parse_desktop_goal("Open Telegram") is None

    def test_read_messages_intent(self):
        g = parse_desktop_goal("read my messages on telegram")
        assert g is not None
        assert g.kind == DesktopGoalKind.READ_MESSAGES
        assert g.app == "telegram"


# ═══════════════════════════════════════════════════════════════
# 2. Desktop context
# ═══════════════════════════════════════════════════════════════

class TestDesktopContext:
    def test_structured_fields(self):
        c = DesktopContext(active_application="telegram",
                           window_title="Telegram — Rahul",
                           observation_method="accessibility",
                           application_state="present")
        d = c.to_dict()
        for f in ("active_application", "focused_window", "window_title",
                  "window_class", "process", "visible_text",
                  "interactive_elements", "focused_element", "screen_hash",
                  "application_state", "observation_method", "timestamp"):
            assert f in d

    def test_empty_is_honest_not_invented(self):
        c = DesktopContext()
        assert c.application_state == "unavailable"
        assert c.observation_method == ""


# ═══════════════════════════════════════════════════════════════
# 3. Skill registry
# ═══════════════════════════════════════════════════════════════

class TestSkillRegistry:
    def test_telegram_registered_and_generic(self):
        reg = DesktopSkillRegistry()
        assert reg.get("telegram") is not None
        reg.register(DesktopSkill(app="signal", display_name="Signal"))
        assert reg.get("signal") is not None
        assert reg.skill_for("unknown-app") is not None

    def test_telegram_semantic_hints_are_generic(self):
        skill = DesktopSkillRegistry().get("telegram")
        assert skill is not None
        assert all(isinstance(w, str) for w in skill.search_words)
        assert "telegram" in skill.window_markers


# ═══════════════════════════════════════════════════════════════
# 4. Expected-effect verification (the no-false-success contract)
# ═══════════════════════════════════════════════════════════════

class TestVerification:
    def test_send_requires_message_inside_conversation(self):
        before = conversation_ctx("Rahul", "")
        after = conversation_ctx("Rahul", "I'll call you after 6")
        v = verify_expected_effect(
            DesktopAction.SEND_MESSAGE,
            {"recipient": "Rahul", "text": "I'll call you after 6"},
            before, after)
        assert v.result == EffectResult.PASS
        assert v.evidence["proof"] == "message-inside-conversation"

    def test_frame_change_alone_is_never_success(self):
        before = DesktopContext(active_application="telegram", screen_hash="aaaa")
        after = DesktopContext(active_application="telegram", screen_hash="bbbb")
        v = verify_expected_effect(
            DesktopAction.SEND_MESSAGE,
            {"recipient": "Rahul", "text": "I'll call you after 6"},
            before, after)
        assert v.result in (EffectResult.FAIL, EffectResult.NO_EVIDENCE)

    def test_message_seen_but_wrong_conversation_is_fail(self):
        after = DesktopContext(active_application="telegram",
                               window_title="Telegram — Someone Else",
                               visible_text="I'll call you after 6",
                               observation_method="ocr")
        v = verify_expected_effect(
            DesktopAction.SEND_MESSAGE,
            {"recipient": "Rahul", "text": "I'll call you after 6"},
            None, after)
        assert v.result == EffectResult.FAIL

    def test_draft_verification_needs_input_value(self):
        after = DesktopContext(interactive_elements=[
            InteractiveElement(label="Message", kind="input",
                               value="I'll call you after 6")])
        v = verify_expected_effect(DesktopAction.VERIFY_DRAFT,
                                   {"text": "I'll call you after 6"},
                                   None, after)
        assert v.result == EffectResult.PASS


# ═══════════════════════════════════════════════════════════════
# 5. Engine (fake controller + fake observer)
# ═══════════════════════════════════════════════════════════════

class FakeController:
    """Record calls and simulate a desktop app UI driven by observation."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self._app_open = False

    def execute(self, action: str, params: Optional[Dict[str, Any]] = None):
        # Import here to avoid a module-level computer import in tests.
        from computer.action_result import ActionOutcome, ActionResult
        params = dict(params or {})
        self.calls.append((action, params))
        if action == "open_app":
            self._app_open = True
            return ActionResult(action=action, method="app_resolver",
                                success=True, outcome=ActionOutcome.SUCCESS)
        if action in ("click", "find_element", "type_text", "press_key",
                      "clear_text", "focus_window"):
            return ActionResult(action=action, method="native_input",
                                success=True, outcome=ActionOutcome.SUCCESS)
        return ActionResult(action=action, method="perception", success=True,
                            outcome=ActionOutcome.SUCCESS)


class FakeObserver:
    """A scripted observer that returns contexts in order, then holds."""

    def __init__(self, contexts: List[DesktopContext]) -> None:
        self._contexts = list(contexts)
        self._index = 0
        self.notes: List[str] = []

    def observe(self, note: str = "") -> DesktopContext:
        self.notes.append(note)
        if self._contexts:
            ctx = self._contexts[min(self._index, len(self._contexts) - 1)]
            if self._index < len(self._contexts) - 1:
                self._index += 1
            return ctx
        return DesktopContext(application_state="unavailable")


def make_engine(ctxs: List[DesktopContext]) -> DesktopGoalEngine:
    return DesktopGoalEngine(
        controller=FakeController(),
        observer=FakeObserver(ctxs),
        trace=AgentTrace(),
        limits=fast_limits(),
    )


def full_success_contexts() -> List[DesktopContext]:
    """Telegram → contact search → conversation → input → draft → sent."""
    msg = "I'll call you after 6"
    return [
        # OPEN_APPLICATION before
        DesktopContext(application_state="unavailable"),
        # OPEN_APPLICATION after → app present
        DesktopContext(active_application="telegram", window_title="Telegram",
                       window_class="TelegramDesktop", application_state="present",
                       observation_method="native_window"),
        # FIND_CONTACT after → contact observed
        contact_ctx(["Rahul"]),
        # VERIFY_CONTACT / OPEN_CONVERSATION after → conversation identity
        conversation_ctx("Rahul", ""),
        # COMPOSE / DRAFT after → input holds draft
        DesktopContext(active_application="telegram",
                       window_title="Telegram — Rahul",
                       interactive_elements=[
                           InteractiveElement(label="Message", kind="input",
                                              value=msg)],
                       application_state="present",
                       observation_method="accessibility"),
        # SEND / VERIFY after → message inside conversation
        conversation_ctx("Rahul", msg),
    ]


class TestEngineFlow:
    def test_send_goal_pauses_for_confirmation_then_completes(self):
        eng = make_engine(full_success_contexts())
        run = eng.start("Send Rahul: I'll call you after 6 on Telegram")
        assert run is not None
        eng.run_to_terminal(run)
        # Sending is external: it MUST pause for confirmation first.
        assert run.status == DesktopStatus.WAITING_CONFIRMATION, \
            (run.status, run.error)
        assert "Rahul" in (run.question or "")
        assert "I'll call you after 6" in (run.question or "")
        assert eng.has_pending()
        # Confirm → resume the SAME run (no restart).
        eng.resume("yes", task_id=run.task_id)
        assert run.status == DesktopStatus.COMPLETED, (run.status, run.error)
        assert run.sent_verified

    def test_confirm_pause_cancels(self):
        eng = make_engine(full_success_contexts())
        run = eng.start("Send Rahul: I'll call you after 6 on Telegram")
        eng.run_to_terminal(run)
        assert run.status == DesktopStatus.WAITING_CONFIRMATION
        eng.resume("cancel", task_id=run.task_id)
        assert run.status == DesktopStatus.FAILED
        assert run.error

    def test_trace_emits_full_lifecycle(self):
        eng = make_engine(full_success_contexts())
        run = eng.start("Send Rahul: I'll call you after 6 on Telegram")
        eng.run_to_terminal(run)
        eng.resume("yes", task_id=run.task_id)
        tr = eng.trace()
        types = [e.event_type.value for e in tr.events(limit=300)]
        for want in ("GOAL_RECEIVED", "PLAN_CREATED", "STEP_STARTED",
                     "OBSERVATION", "ACTION_STARTED", "ACTION_COMPLETED",
                     "VERIFICATION_COMPLETED", "CONFIRMATION_REQUIRED",
                     "CONFIRMATION_GRANTED", "TASK_COMPLETED"):
            assert want in types, want

    def test_completion_requires_sent_verification(self):
        # If the message never appears inside the conversation, the goal can
        # never complete (no false success from "send executed").
        ctxs = full_success_contexts()[:-1]  # drop the "message inside chat"
        eng = make_engine(ctxs)
        run = eng.start("Send Rahul: I'll call you after 6 on Telegram")
        eng.run_to_terminal(run)
        eng.resume("yes", task_id=run.task_id)
        # May recover then fail — but never COMPLETED without sent_verified.
        assert run.status != DesktopStatus.COMPLETED
        assert not run.sent_verified


class TestEngineAmbiguity:
    def test_ambiguous_contact_asks(self):
        ctxs = [
            DesktopContext(application_state="unavailable"),
            DesktopContext(active_application="telegram", window_title="Telegram",
                           application_state="present",
                           observation_method="native_window"),
            contact_ctx(["Rahul Kumar", "Rahul Sharma", "Rahul Singh"]),
        ]
        eng = make_engine(ctxs)
        run = eng.start("Send Rahul: hello on Telegram")
        eng.run_to_terminal(run)
        assert run.status == DesktopStatus.ASKING_USER, (run.status, run.error)
        q = (run.question or "").lower()
        assert "rahul" in q and "which one" in q
        labels = {c.label for c in run.candidates}
        assert "Rahul Kumar" in labels and "Rahul Sharma" in labels

    def test_ambiguous_contact_resume_uses_choice(self):
        ctxs = [
            DesktopContext(application_state="unavailable"),
            DesktopContext(active_application="telegram", window_title="Telegram",
                           application_state="present",
                           observation_method="native_window"),
            contact_ctx(["Rahul Kumar", "Rahul Sharma", "Rahul Singh"]),
            conversation_ctx("Rahul Sharma", ""),
            DesktopContext(active_application="telegram",
                           window_title="Telegram — Rahul Sharma",
                           interactive_elements=[
                               InteractiveElement(label="Message", kind="input",
                                                  value="hello")],
                           observation_method="accessibility"),
            conversation_ctx("Rahul Sharma", "hello"),
        ]
        eng = make_engine(ctxs)
        run = eng.start("Send Rahul: hello on Telegram")
        eng.run_to_terminal(run)
        assert run.status == DesktopStatus.ASKING_USER
        # User picks "Rahul Sharma".
        eng.resume("Rahul Sharma", task_id=run.task_id)
        assert run.status == DesktopStatus.WAITING_CONFIRMATION
        eng.resume("yes", task_id=run.task_id)
        assert run.status == DesktopStatus.COMPLETED
        # The send used the chosen contact.
        assert run.resolved_contact == "Rahul Sharma"


class TestEngineRecovery:
    def test_app_open_failure_is_bounded(self):
        ctxs = [DesktopContext(application_state="unavailable")] * 10
        eng = make_engine(ctxs)
        eng._controller.execute = lambda a, p: _fail_result(a)
        run = eng.start("Send Rahul: hello on Telegram")
        eng.run_to_terminal(run)
        assert run.status == DesktopStatus.FAILED
        assert run.recoveries <= fast_limits().max_retries_per_step + 1


def _fail_result(action: str):
    from computer.action_result import ActionOutcome, ActionResult
    return ActionResult(action=action, method="perception", success=False,
                        outcome=ActionOutcome.FAILED, error="unavailable")





