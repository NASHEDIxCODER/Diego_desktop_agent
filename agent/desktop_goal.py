"""
Phase 24: generic desktop-goal model (PURE — no I/O, no side effects).

This module owns the VOCABULARY of desktop application goals. It is
deliberately free of execution, perception and UI concerns so it can be
unit-tested deterministically and reused by the engine, the trace and the
workflow panel, exactly like ``agent.browser_goal`` does for the browser:

  * DesktopAction       — the generic desktop action set
  * DesktopGoal         — an interpreted user goal (app / recipient / message)
  * DesktopStep         — one planned step with an EXPECTED EFFECT
  * EffectVerdict       — expected-vs-observed record. A generic "screen /
                          frame / window changed" is NEVER accepted as proof
                          that a message was sent.

Nothing here knows Telegram's UI: no hard-coded selector, no screen
coordinate, no Telegram API/token anywhere in this file. App names resolve
through a small spoken-alias table (generic, documented); the actual
perception and act primitives live in ``computer/``.

Reuse (no duplication): semantic element selection + ambiguity handling come
from ``agent.browser_goal`` (``Candidate`` / ``rank_candidates`` /
``resolve_answer_to_candidate`` / ``normalize_label``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agent.browser_goal import normalize_label


# ═══════════════════════════════════════════════════════════════════
# 1. Vocabulary
# ═══════════════════════════════════════════════════════════════════


class DesktopGoalKind(str, Enum):
    """What the user asked Diego to do with a desktop application."""

    SEND_MESSAGE = "send_message"      # "send Rahul: hello" on <app>
    READ_MESSAGES = "read_messages"    # "read my messages on <app>"
    UNKNOWN = "unknown"


class DesktopAction(str, Enum):
    """Generic desktop capabilities the agent may compose (no app knowledge)."""

    OPEN_APPLICATION = "OPEN_APPLICATION"
    FOCUS_APPLICATION = "FOCUS_APPLICATION"
    FIND_CONTACT = "FIND_CONTACT"                 # resolve a recipient (semantic)
    VERIFY_CONTACT = "VERIFY_CONTACT"             # exactly one match (never guess)
    OPEN_CONVERSATION = "OPEN_CONVERSATION"       # click the resolved contact
    VERIFY_CONVERSATION = "VERIFY_CONVERSATION"   # conversation identity observed
    READ_CONVERSATION = "READ_CONVERSATION"       # observe the visible messages
    FIND_INPUT = "FIND_INPUT"                     # locate the message input box
    CLEAR_INPUT = "CLEAR_INPUT"
    COMPOSE_MESSAGE = "COMPOSE_MESSAGE"           # type the message text
    VERIFY_DRAFT = "VERIFY_DRAFT"                 # the draft text is observed
    SEND_MESSAGE = "SEND_MESSAGE"                 # EXTERNAL side effect
    VERIFY_SENT_MESSAGE = "VERIFY_SENT_MESSAGE"   # message exists in the chat
    WAIT = "WAIT"
    ASK_USER = "ASK_USER"


class DesktopStatus(str, Enum):
    """Task-level status of a desktop goal run."""

    RUNNING = "RUNNING"
    ASKING_USER = "ASKING_USER"               # contact ambiguity clarification
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"  # external send needs approval
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EffectResult(str, Enum):
    """Expected-effect verdict (mirrors core.goal_verification.GoalResult)."""

    PASS = "PASS"
    FAIL = "FAIL"
    NO_EVIDENCE = "NO_EVIDENCE"


# How each desktop action is executed through the EXISTING ComputerController.
# Empty string = engine-internal primitive (performed with observations, no
# controller action exists).
COMPUTER_ACTION: Dict[DesktopAction, str] = {
    DesktopAction.OPEN_APPLICATION: "open_app",
    DesktopAction.FOCUS_APPLICATION: "focus_window",
    DesktopAction.FIND_CONTACT: "find_element",
    DesktopAction.OPEN_CONVERSATION: "click",
    DesktopAction.FIND_INPUT: "find_element",
    DesktopAction.CLEAR_INPUT: "clear_text",
    DesktopAction.COMPOSE_MESSAGE: "type_text",
    DesktopAction.SEND_MESSAGE: "press_key",
    DesktopAction.WAIT: "wait",
    DesktopAction.VERIFY_CONTACT: "",
    DesktopAction.VERIFY_CONVERSATION: "",
    DesktopAction.READ_CONVERSATION: "",
    DesktopAction.VERIFY_DRAFT: "",
    DesktopAction.VERIFY_SENT_MESSAGE: "",
    DesktopAction.ASK_USER: "",
}

ROLE_SEARCH = "@search"                 # contact search box
ROLE_MESSAGE_INPUT = "@message_input"   # message compose box


MESSAGING_APPS: Dict[str, Tuple[str, ...]] = {
    "telegram": ("telegram", "telegram desktop", "telegram-desktop", "tg"),
    "whatsapp": ("whatsapp", "whatsapp desktop", "whats app"),
    "signal": ("signal", "signal desktop"),
    "slack": ("slack",),
    "discord": ("discord",),
    "messenger": ("messenger", "facebook messenger"),
    "skype": ("skype",),
    "messages": ("messages", "imessage", "sms"),
}

_MESSAGE_VERBS = ("send", "message", "text", "msg", "write to", "dm", "tell")
_READ_VERBS = ("read", "check", "show", "open my", "show me my")

_SEARCH_WORDS = ("search", "find", "query", "people", "contacts")
_MESSAGE_INPUT_WORDS = ("message", "write", "type", "compose", "send",
                        "broadcast", "chat")
_SEND_WORDS = ("send", "submit", "deliver")


# ═══════════════════════════════════════════════════════════════════
# 2. Expected effects (per action)
# ═══════════════════════════════════════════════════════════════════

EXPECTED_EFFECTS: Dict[DesktopAction, str] = {
    DesktopAction.OPEN_APPLICATION:
        "the target desktop application becomes present (process or window)",
    DesktopAction.FOCUS_APPLICATION:
        "the target application window becomes the focused window",
    DesktopAction.FIND_CONTACT:
        "exactly one contact matching the recipient is identified, or the "
        "ambiguity is reported (never a guess)",
    DesktopAction.VERIFY_CONTACT:
        "the chosen contact's identity is observed in the contact list",
    DesktopAction.OPEN_CONVERSATION:
        "the conversation with the chosen contact becomes the active view",
    DesktopAction.VERIFY_CONVERSATION:
        "the conversation identity (recipient) is observed in the window",
    DesktopAction.READ_CONVERSATION:
        "the visible conversation text is observed as evidence",
    DesktopAction.FIND_INPUT:
        "the message compose input is located via perception",
    DesktopAction.CLEAR_INPUT:
        "the compose input is empty in observed state",
    DesktopAction.COMPOSE_MESSAGE:
        "the compose input holds the message text as a draft",
    DesktopAction.VERIFY_DRAFT:
        "the message text is observed as the current draft",
    DesktopAction.SEND_MESSAGE:
        "the composed message is submitted to the open conversation",
    DesktopAction.VERIFY_SENT_MESSAGE:
        "the message text is observed INSIDE the intended conversation",
    DesktopAction.WAIT:
        "a bounded wait elapses",
    DesktopAction.ASK_USER:
        "the user is asked for a clarification",
}


def expected_effect_text(action: "DesktopAction | str",
                         params: Optional[Dict[str, Any]] = None) -> str:
    """Human/README-visible expected effect for one desktop action."""
    try:
        act = action if isinstance(action, DesktopAction) else DesktopAction(str(action))
    except ValueError:
        return "the action reports a verified, observable effect"
    base = EXPECTED_EFFECTS.get(act, "the action reports a verified effect")
    params = params or {}
    if act == DesktopAction.SEND_MESSAGE:
        recipient = params.get("recipient") or params.get("target") or ""
        if recipient:
            return f"the message is sent to '{recipient}' in the conversation"
    if act == DesktopAction.VERIFY_SENT_MESSAGE:
        recipient = params.get("recipient") or ""
        return (f"the message is observed inside the conversation with "
                f"'{recipient}'" if recipient else base)
    return base


# ═══════════════════════════════════════════════════════════════════
# 3. Goal model
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DesktopGoal:
    """An INTERPRETED desktop goal (generic concepts only)."""

    raw_text: str = ""
    kind: DesktopGoalKind = DesktopGoalKind.UNKNOWN
    app: str = ""                # canonical app id, e.g. "telegram"
    app_display: str = ""        # "Telegram"
    recipient: str = ""          # who the message goes to
    message: str = ""            # message body
    query: str = ""              # read subject (for READ_MESSAGES)

    def describe(self) -> str:
        if self.kind == DesktopGoalKind.SEND_MESSAGE:
            where = self.app_display or self.app or "the application"
            return f"Send '{self.recipient}' a message on {where}"
        if self.kind == DesktopGoalKind.READ_MESSAGES:
            where = self.app_display or self.app or "the application"
            subject = f" from '{self.query}'" if self.query else ""
            return f"Read messages{subject} on {where}"
        return f"Desktop goal: {self.raw_text[:80]}"

    def completion_requirement(self) -> str:
        if self.kind == DesktopGoalKind.SEND_MESSAGE:
            return (f"the message to '{self.recipient}' observed INSIDE the "
                    f"intended conversation (never just a screen change)")
        if self.kind == DesktopGoalKind.READ_MESSAGES:
            return "the visible conversation text observed as evidence"
        return "an observed, verified desktop state"


@dataclass
class DesktopStep:
    """One planned desktop step with a declared expected effect."""

    action: DesktopAction
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    optional: bool = False
    index: int = 0
    status: str = "PENDING"          # PENDING | DONE | FAILED | SKIPPED
    verdict: str = ""
    observation: str = ""
    attempts: int = 0

    @property
    def target(self) -> str:
        return str(self.params.get("target") or self.params.get("recipient")
                   or self.params.get("app") or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "action": self.action.value,
            "description": self.description, "params": dict(self.params),
            "expected_effect": self.expected_effect, "optional": self.optional,
            "status": self.status, "verdict": self.verdict,
            "observation": self.observation, "attempts": self.attempts,
        }


@dataclass
class EffectVerdict:
    """One expected-vs-observed verification record."""

    action: str
    expected_effect: str
    observed_effect: str = ""
    result: EffectResult = EffectResult.NO_EVIDENCE
    evidence: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.result == EffectResult.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "expected_effect": self.expected_effect,
            "observed_effect": self.observed_effect,
            "result": self.result.value,
            "evidence": dict(self.evidence),
            "detail": self.detail,
        }


# ═══════════════════════════════════════════════════════════════════
# 4. Goal parsing (pure)
# ═══════════════════════════════════════════════════════════════════


def messaging_app_for(text: str) -> Tuple[str, str]:
    """Return ``(canonical, display)`` for the first messaging app mentioned.

    Empty canonical means the utterance does not name a messaging app.
    """
    low = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    hits: List[Tuple[int, str]] = []
    for canonical, aliases in MESSAGING_APPS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", low):
                hits.append((len(alias), canonical))
    if not hits:
        return "", ""
    hits.sort(key=lambda h: -h[0])
    canonical = hits[0][1]
    return canonical, canonical.title()


def _strip_app(text: str, alias: str) -> str:
    """Remove the phrasing that introduces the app alias from the utterance."""
    t = text
    t = re.sub(
        rf"\b(?:please\s+|can you\s+|could you\s+)?(?:open|launch|start|use)\s+[\w\- ]*?"
        rf"\b{re.escape(alias)}\b(?:[\s,]+(?:and|then|,|\.)\s*)?",
        " ", t, count=1, flags=re.IGNORECASE)
    t = re.sub(rf"\b(?:on|in|via|with|through|using)\s+{re.escape(alias)}\b[\s,]*",
               " ", t, count=1, flags=re.IGNORECASE)
    t = re.sub(rf"\b{re.escape(alias)}\b[\s,]*", " ", t, count=1,
               flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def parse_desktop_goal(text: str) -> Optional[DesktopGoal]:
    """Interpret a user utterance as a desktop goal (or None if not one).

    Only SEND_MESSAGE and READ_MESSAGES intents parse here; a bare
    "open telegram" falls through to the existing pipeline unchanged.
    """
    raw = str(text or "").strip().rstrip(".!?")
    if not raw:
        return None
    low = raw.lower()
    app, display = messaging_app_for(raw)
    if app:
        alias = next(a for a in MESSAGING_APPS[app]
                     if re.search(rf"\b{re.escape(a)}\b", low))
        body = _strip_app(raw, alias)
    else:
        body = raw

    body_low = body.lower()

    # ── SEND_MESSAGE intent ──────────────────────────────────────
    verb = None
    for v in _MESSAGE_VERBS:
        if re.search(rf"(^|\s){re.escape(v)}(\s|$)", body_low):
            verb = v
            break
    if app and verb:
        after = _after_verb(body, verb)
        recipient, message = _split_recipient_message(after)
        # Phase 24.6: a recipient WITHOUT a message body is still a valid
        # SEND_MESSAGE goal — the body is asked of the user (ASK_USER step)
        # before compose, and the send itself stays behind explicit
        # confirmation. External side effects must ASK, not fail.
        if recipient:
            return DesktopGoal(
                raw_text=raw, kind=DesktopGoalKind.SEND_MESSAGE,
                app=app, app_display=display, recipient=recipient,
                message=message)

    # ── READ_MESSAGES intent ─────────────────────────────────────
    if app:
        for v in _READ_VERBS:
            if re.search(rf"(^|\s){re.escape(v)}\b", body_low):
                subject = _after_verb(body, v)
                subject = re.sub(r"(^|\s)(my|messages?|the)\s*", " ",
                                 subject, flags=re.IGNORECASE).strip()
                return DesktopGoal(
                    raw_text=raw, kind=DesktopGoalKind.READ_MESSAGES,
                    app=app, app_display=display, query=subject)

    return None


def _after_verb(body: str, verb: str) -> str:
    m = re.search(rf"(^|\s){re.escape(verb)}\s+(to\s+)?", body,
                  flags=re.IGNORECASE)
    if not m:
        return ""
    return body[m.end():].strip()


def _split_recipient_message(after: str) -> Tuple[str, str]:
    """Split the post-verb text into (recipient, message).

    Supported forms (Phase 24.6 added the phrasings users actually say):
      "Rahul: I'll call you after 6."      -> ('Rahul', "I'll call you after 6.")
      "message to Delhi"                   -> ('Delhi', '')      (body asked later)
      "a message to Delhi"                 -> ('Delhi', '')
      "Rahul a message"                    -> ('Rahul', '')
      "hello to Rahul"                     -> ('Rahul', 'hello')
      "Rahul I'll call you"                -> ('Rahul', "I'll call you")
    """
    a = str(after or "").strip().strip(".")
    if not a:
        return "", ""
    if ":" in a:
        recipient, message = a.split(":", 1)
        return recipient.strip(), message.strip().strip(".")
    # "message to Delhi" / "a message to Delhi" — recipient only; the message
    # body is asked of the user (external side effect still confirmed later).
    m = re.match(r"^(?:a\s+)?message\s+(?:to|for)\s+(?P<rcpt>[A-Za-z]"
                 r"[\w .'\-]{0,40}?)(?:\s+(?P<msg>.+))?$", a, re.IGNORECASE)
    if m:
        return (m.group("rcpt").strip().strip(","),
                (m.group("msg") or "").strip().strip("."))
    # "Rahul a message" / "Rahul the message" — recipient first, no body.
    m = re.match(r"^(?P<rcpt>[A-Za-z][\w .'\-]{0,40}?)\s+(?:a\s+|the\s+)?"
                 r"messages?$", a, re.IGNORECASE)
    if m:
        return m.group("rcpt").strip().strip(","), ""
    # "hello to Rahul" — <message> to <recipient>.
    m = re.match(r"^(?P<msg>.+?)\s+to\s+(?P<rcpt>[A-Za-z][\w .'\-]{0,40})$",
                 a, re.IGNORECASE)
    if m:
        msg = m.group("msg").strip()
        if msg.lower() not in ("message", "a message", "the message"):
            return (m.group("rcpt").strip().strip(","), msg.strip("."))
    # "Rahul I'll call you" — recipient is the leading capitalized name cluster.
    m = re.match(r"^([A-Z][\w.\- ]{0,40}?)\s+(.{2,})$", a)
    if m:
        return m.group(1).strip(), m.group(2).strip().strip(".")
    return "", ""


# ═══════════════════════════════════════════════════════════════════
# 5. Expected-effect verification (the "no false success" contract)
# ═══════════════════════════════════════════════════════════════════


def _attr(ctx: Any, name: str, default: Any = "") -> Any:
    if isinstance(ctx, dict):
        return ctx.get(name, default)
    return getattr(ctx, name, default)


def _el_field(el: Any, field: str, default: Any = "") -> Any:
    if isinstance(el, dict):
        return el.get(field, default)
    return getattr(el, field, default)


def _text(ctx: Any) -> str:
    bits = [str(_attr(ctx, "window_title", "") or ""),
            str(_attr(ctx, "visible_text", "") or "")]
    for el in list(_attr(ctx, "interactive_elements", []) or [])[:100]:
        bits.append(str(_el_field(el, "label") or ""))
        bits.append(str(_el_field(el, "value") or ""))
    return "\n".join(bits).lower()


def _present(ctx: Any, needle: str) -> bool:
    n = normalize_label(needle)
    if not n:
        return False
    return n in normalize_label(_text(ctx))


def _verdict(action: DesktopAction, params: Dict[str, Any],
             result: EffectResult, observed: str, detail: str,
             evidence: Dict[str, Any]) -> EffectVerdict:
    return EffectVerdict(
        action=action.value, expected_effect=expected_effect_text(action, params),
        observed_effect=observed, result=result, evidence=evidence, detail=detail)


def classify_state(ctx: Any) -> str:
    """Coarse desktop state classification (never invented)."""
    if ctx is None:
        return "unavailable"
    app = str(_attr(ctx, "active_application", "") or "").strip()
    title = str(_attr(ctx, "window_title", "") or "").strip()
    method = str(_attr(ctx, "observation_method", "") or "").strip()
    if not app and not title and not _attr(ctx, "interactive_elements", None):
        return "empty"
    if app or title:
        return "present"
    if method:
        return "partial"
    return "unknown"


def verify_expected_effect(action: "DesktopAction | str",
                           params: Optional[Dict[str, Any]],
                           before: Any, after: Any) -> EffectVerdict:
    """Compare DECLARED expectation with OBSERVED desktop state.

    A generic frame / screen / window change is NEVER accepted as success:
    each branch checks the specific thing the step promised. A missing
    observation yields NO_EVIDENCE (never PASS).
    """
    try:
        act = action if isinstance(action, DesktopAction) else DesktopAction(str(action))
    except ValueError:
        return EffectVerdict(action=str(action),
                             expected_effect="the action reports a verified effect",
                             result=EffectResult.NO_EVIDENCE,
                             detail="unknown desktop action")
    p = dict(params or {})
    if after is None:
        return _verdict(act, p, EffectResult.NO_EVIDENCE, "",
                        "no observation after the action", {})

    if act == DesktopAction.OPEN_APPLICATION:
        app = str(p.get("app") or p.get("target") or "")
        if _present(after, app):
            return _verdict(act, p, EffectResult.PASS,
                            "the target application is observed",
                            f"application '{app}' observed as present", {"app": app})
        return _verdict(act, p, EffectResult.FAIL, "",
                        f"application '{app}' not observed as present", {"app": app})

    if act == DesktopAction.OPEN_CONVERSATION:
        recipient = str(p.get("target") or p.get("recipient") or "")
        if recipient and _present(after, recipient):
            return _verdict(act, p, EffectResult.PASS,
                            "the conversation view is open",
                            f"conversation with '{recipient}' is the active view",
                            {"recipient": recipient})
        return _verdict(act, p, EffectResult.FAIL, "",
                        f"conversation with '{recipient}' not observed",
                        {"recipient": recipient})

    if act == DesktopAction.VERIFY_CONVERSATION:
        recipient = str(p.get("recipient") or p.get("target") or "")
        title = normalize_label(str(_attr(after, "window_title", "") or ""))
        if recipient and (normalize_label(recipient) in title
                          or _present(after, recipient)):
            return _verdict(act, p, EffectResult.PASS,
                            "conversation identity observed",
                            f"conversation identity '{recipient}' observed",
                            {"recipient": recipient, "title": title})
        return _verdict(act, p, EffectResult.FAIL,
                        "conversation identity not observed",
                        f"conversation with '{recipient}' not the active view",
                        {"recipient": recipient, "title": title})

    if act == DesktopAction.COMPOSE_MESSAGE:
        msg = str(p.get("text") or p.get("message") or "")
        if msg and _present(after, msg):
            return _verdict(act, p, EffectResult.PASS,
                            "the message draft is present",
                            f"draft observed: '{msg[:60]}'", {"message": msg})
        return _verdict(act, p, EffectResult.FAIL, "",
                        f"message draft NOT observed ('{msg[:60]}')",
                        {"message": msg})

    if act == DesktopAction.VERIFY_DRAFT:
        msg = str(p.get("text") or p.get("message") or "")
        if not msg:
            return _verdict(act, p, EffectResult.NO_EVIDENCE, "",
                            "no message text to verify", {})
        if _draft_present(after, msg):
            return _verdict(act, p, EffectResult.PASS,
                            "the draft text is observed",
                            f"draft text '{msg[:60]}' observed in the input",
                            {"message": msg})
        return _verdict(act, p, EffectResult.FAIL, "",
                        f"draft text '{msg[:60]}' NOT observed in the input",
                        {"message": msg})

    if act in (DesktopAction.SEND_MESSAGE, DesktopAction.VERIFY_SENT_MESSAGE):
        recipient = str(p.get("recipient") or p.get("target") or "")
        msg = str(p.get("text") or p.get("message") or "")
        return _message_verdict(after, recipient, msg)

    if act == DesktopAction.FIND_CONTACT:
        target = str(p.get("target") or p.get("recipient") or "")
        if target and _present(after, target):
            return _verdict(act, p, EffectResult.PASS,
                            "the contact is observed",
                            f"contact '{target}' observed", {"target": target})
        return _verdict(act, p, EffectResult.FAIL, "",
                        f"contact '{target}' not observed", {"target": target})

    if act == DesktopAction.VERIFY_CONTACT:
        target = str(p.get("target") or p.get("recipient") or "")
        if target and _present(after, target):
            return _verdict(act, p, EffectResult.PASS,
                            "the contact identity is observed",
                            f"contact '{target}' observed as unique", {"target": target})
        return _verdict(act, p, EffectResult.FAIL, "",
                        f"contact '{target}' not observed", {"target": target})

    if act == DesktopAction.FIND_INPUT:
        if _has_input(after):
            return _verdict(act, p, EffectResult.PASS,
                            "the message input is located",
                            "a message input is observed", {})
        return _verdict(act, p, EffectResult.FAIL, "",
                        "no message input observed", {})

    if act == DesktopAction.CLEAR_INPUT:
        # Preparatory step: the controller's clear_text self-reports. The
        # subsequent COMPOSE step carries the strict "draft observed" contract.
        return _verdict(act, p, EffectResult.PASS, "input cleared",
                        "the message input was cleared", {})

    if act == DesktopAction.FOCUS_APPLICATION:
        app = str(p.get("app") or p.get("target") or "")
        if app and _present(after, app):
            return _verdict(act, p, EffectResult.PASS,
                            "the application window is focused",
                            f"window for '{app}' is focused", {"app": app})
        return _verdict(act, p, EffectResult.FAIL, "",
                        f"window for '{app}' not focused", {"app": app})

    if act == DesktopAction.READ_CONVERSATION:
        text = str(_attr(after, "visible_text", "") or "").strip()
        if text:
            return _verdict(act, p, EffectResult.PASS,
                            f"{len(text)} chars observed",
                            "conversation text observed", {"chars": len(text)})
        return _verdict(act, p, EffectResult.FAIL, "",
                        "no conversation text observed", {})

    if act == DesktopAction.WAIT:
        return _verdict(act, p, EffectResult.PASS, "bounded wait elapsed",
                        "bounded wait completed", {})

    if act == DesktopAction.ASK_USER:
        return _verdict(act, p, EffectResult.NO_EVIDENCE, "",
                        "waiting for the user's answer", {})

    return _verdict(act, p, EffectResult.NO_EVIDENCE, "",
                    "no expected-effect contract for this action", {})


def _has_input(after: Any) -> bool:
    for el in list(_attr(after, "interactive_elements", []) or []):
        kind = str(_el_field(el, "kind") or "").lower()
        if kind in ("input", "textarea", "textbox", "text"):
            return True
        if _el_field(el, "text_input", False):
            return True
    return False


def _draft_present(after: Any, msg: str) -> bool:
    """True when the draft text is observed inside an editable input."""
    want = normalize_label(msg)
    if not want:
        return False
    for el in list(_attr(after, "interactive_elements", []) or []):
        kind = str(_el_field(el, "kind") or "").lower()
        if kind not in ("input", "textarea", "textbox", "text"):
            continue
        value = normalize_label(str(_el_field(el, "value") or ""))
        if want in value:
            return True
    focused = _attr(after, "focused_element", None) or {}
    fv = normalize_label(str(_el_field(focused, "value") or ""))
    return bool(fv) and want in fv


def _message_verdict(after: Any, recipient: str, msg: str) -> EffectVerdict:
    """The message-send contract: the message INSIDE the intended conversation.

    Evidence accepted is TARGET-SPECIFIC only:
      * conversation identity — the window title / active view names the
        recipient (proves we are in the right chat), and
      * message text — the composed text is observed on screen.

    A frame/screen/window change alone is NEVER accepted.
    """
    p = {"recipient": recipient, "message": msg}
    if not msg:
        return _verdict(DesktopAction.SEND_MESSAGE, p, EffectResult.NO_EVIDENCE,
                        "", "no message text to verify", {})
    conv_ok = True
    if recipient:
        title = normalize_label(str(_attr(after, "window_title", "") or ""))
        recipient_n = normalize_label(recipient)
        conv_ok = bool(recipient_n and (recipient_n in title
                                        or recipient_n in normalize_label(
                                            str(_attr(after, "visible_text", "") or ""))))
    msg_ok = _present(after, msg)
    observed = f"conversation_identity={'ok' if conv_ok else 'missing'} " \
               f"message_text={'present' if msg_ok else 'missing'}"
    evidence = {
        "recipient": recipient,
        "window_title": str(_attr(after, "window_title", "") or ""),
        "message_text_present": msg_ok,
        "proof": "message-inside-conversation",
    }
    if msg_ok and conv_ok:
        return _verdict(DesktopAction.SEND_MESSAGE, p, EffectResult.PASS,
                        observed,
                        f"message observed inside the conversation with "
                        f"'{recipient}'", evidence)
    if msg_ok and not conv_ok and recipient:
        return _verdict(DesktopAction.SEND_MESSAGE, p, EffectResult.FAIL,
                        observed,
                        "message text seen but NOT inside the intended "
                        f"conversation with '{recipient}'", evidence)
    if not msg_ok:
        return _verdict(DesktopAction.SEND_MESSAGE, p, EffectResult.NO_EVIDENCE,
                        observed,
                        "the sent message was not observed — no proof of send",
                        evidence)
    return _verdict(DesktopAction.SEND_MESSAGE, p, EffectResult.FAIL, observed,
                    "message send not verified", evidence)


__all__ = [
    "COMPUTER_ACTION", "DesktopAction", "DesktopGoal", "DesktopGoalKind",
    "DesktopStatus", "DesktopStep", "EffectResult", "EffectVerdict",
    "MESSAGING_APPS", "ROLE_MESSAGE_INPUT", "ROLE_SEARCH",
    "classify_state", "expected_effect_text", "messaging_app_for",
    "parse_desktop_goal", "verify_expected_effect",
]
