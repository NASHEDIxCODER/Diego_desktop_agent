"""
GoalRuntime permission layer — SCOPED PermissionManager.

Policy matrix:
    READ_ONLY             → AUTO_RUN (never blocks, never confirms)
    REVERSIBLE_LOCAL      → AUTO_RUN (local, undoable: create folder, edit file,
                            open app, navigate)
    EXTERNAL_SIDE_EFFECT  → CONFIRM required unless a scoped grant exists for
                            the exact scope
    DESTRUCTIVE           → CONFIRM required unless a scoped grant exists
    SECURITY_TESTING      → requires an explicitly authorized TARGET+SCOPE
                            (granted per target); scope NEVER expands
                            autonomously — a probe outside the authorized
                            scope is denied, not widened.

Scoped grants:
    grant(scope_key, permission_class, persistent=False)
    A grant for a coarse scope ("telegram.send") covers every narrower
    parameterization ("telegram.send:contact=rahul") but not vice versa.
    Non-persistent grants are one-shot: consumed by the single approval.
    Scope can also be revoked at any time.

Confirmation flow mirrors agent.task_continuation vocabulary so Diego's
existing "yes / no / do it / cancel" handling stays authoritative.

Logging: [GOAL-PERM]
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from goalruntime.models import PermissionClass

logger = logging.getLogger(__name__)

# Natural confirm/cancel vocabulary — kept IDENTICAL to
# agent.task_continuation so Diego's existing confirmation UX is unchanged.
try:
    from agent.task_continuation import (
        CONFIRM_PHRASES, CANCEL_PHRASES,
    )
except Exception:  # pragma: no cover — standalone fallback copy
    CONFIRM_PHRASES = frozenset({
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "k",
        "do it", "go ahead", "play it", "go for it", "yes please",
        "yeah sure", "sure thing", "absolutely", "definitely", "confirm",
        "proceed", "continue",
    })
    CANCEL_PHRASES = frozenset({
        "no", "nope", "nah", "cancel", "cancel it", "stop", "stop it",
        "don't", "dont", "do not", "don't do it", "dont do it",
        "do not do it", "no thanks", "never mind", "nevermind",
        "forget it", "abort",
    })

_SCOPE_RE = re.compile(r"^[a-z0-9_.\-]+(?::[a-z0-9_.\-=]+)*$")


def classify_action(action: str, params: Optional[dict] = None
                    ) -> Tuple[PermissionClass, str]:
    """Deterministic action → (PermissionClass, reason).

    Pure vocabulary classification — no app-specific if-chains; unknown
    actions default to the SAFE side (confirmation), never silent auto-run.
    Accepts either a bare action ('read_email') or a skill-qualified one
    ('gmail.read_email') — classification uses the bare action verb.
    """
    full = (action or "").strip().lower()
    a = full.split(".")[-1]          # strip skill prefix: 'gmail.read_email'
    p = str(params or {}).lower()

    if a in _SECURITY_ACTIONS:
        return PermissionClass.SECURITY_TESTING, "security probe action"
    if a in _DESTRUCTIVE_ACTIONS or "rm -rf" in p or "format " in p:
        return PermissionClass.DESTRUCTIVE, "irreversible deletion"
    if a in _EXTERNAL_ACTIONS:
        return PermissionClass.EXTERNAL_SIDE_EFFECT, "external side effect"
    # Compose-intent heuristic: a type/click whose params declare a send.
    if a in ("type_text", "click", "press_key", "hotkey") and (
            ("send" in p and "message" in p) or "send" in p and "email" in p):
        return (PermissionClass.EXTERNAL_SIDE_EFFECT,
                "params declare an external send intent")
    if a in _LOCAL_ACTIONS:
        return PermissionClass.REVERSIBLE_LOCAL, "reversible local change"
    if a in _READONLY_ACTIONS:
        return PermissionClass.READ_ONLY, "read-only observation"
    # Unknown action — NEVER silent: safe side is to require confirmation
    # (mirrors computer.action_policy's "unknown → treat as state change").
    return (PermissionClass.EXTERNAL_SIDE_EFFECT,
            "unknown action — safe side requires confirmation")


_EXTERNAL_ACTIONS = frozenset({
    "send_message", "send_email", "post_message", "submit_form",
    "submit_order", "make_payment", "transfer_money", "purchase",
    "publish", "tweet", "send", "submit",
})
_DESTRUCTIVE_ACTIONS = frozenset({
    "delete_file", "delete_folder", "delete_data", "delete", "format_disk",
    "shutdown", "restart", "remove",
})
_LOCAL_ACTIONS = frozenset({
    "open_app", "close_app", "focus_window", "navigate", "open_url",
    "browser_navigate", "browser_search", "back", "forward", "refresh",
    "scroll", "click", "type_text", "press_key", "hotkey", "clear_text",
    "create_folder", "create_file", "write_file", "edit_file", "move_file",
    "copy_file", "run_tests", "generate_code", "search_contact", "open",
    "run_command", "type_into_focus", "click_element", "press_key",
    "focus_app", "find_element",
})
_READONLY_ACTIONS = frozenset({
    "observe", "screenshot", "read_screen", "ocr", "vision_query",
    "get_page_state", "get_window_state", "list_windows", "active_window",
    "read_page", "extract_text", "extract_links", "read_latest_message",
    "read_email", "read_file", "list_dir", "search_ui", "find_candidates",
    "get_time", "get_date", "wait_for_page", "get_url", "whoami", "read",
    "read_latest", "describe", "browser_read", "search",
})
_SECURITY_ACTIONS = frozenset({
    "security_scan", "port_scan", "vuln_probe", "recon", "service_enum",
})


# ═══════════════════════════════════════════════════════════════════
# Decisions & grants
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PermissionDecision:
    allowed: bool
    mode: str                      # auto | confirmed_by_grant | confirmation | deny
    reason: str = ""
    permission_class: PermissionClass = PermissionClass.READ_ONLY
    scope_key: str = ""

    def __str__(self) -> str:  # pragma: no cover — display helper
        return f"PermissionDecision({self.mode}, allowed={self.allowed}, {self.reason})"


@dataclass
class _Grant:
    scope_key: str
    permission_class: PermissionClass
    persistent: bool = False
    used: bool = False
    created_at: float = field(default_factory=time.time)

    def covers(self, scope_key: str) -> bool:
        """Coarse covers narrow: 'telegram.send' ⊇ 'telegram.send:contact=x'."""
        if self.scope_key == scope_key:
            return True
        return scope_key.startswith(self.scope_key + ":")


@dataclass
class SecurityScope:
    """Explicit, user-granted security-testing authorization.

    A probe is legal ONLY when its target matches an authorized target AND
    its operation is inside the authorized scope list. Scope is never
    inferred, guessed, or widened.
    """

    targets: Tuple[str, ...]
    scope: Tuple[str, ...]        # e.g. ("port_scan", "service_enum")
    created_at: float = field(default_factory=time.time)

    def covers(self, target: str, operation: str) -> bool:
        t = (target or "").strip().lower()
        for allowed in self.targets:
            if t == allowed or t.endswith("." + allowed):
                break
        else:
            return False
        return (operation or "").strip().lower() in self.scope


@dataclass
class PendingConfirmation:
    id: str
    permission_class: PermissionClass
    action: str
    params: Dict[str, Any]
    scope_key: str
    prompt: str
    created_at: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════
# The manager
# ═══════════════════════════════════════════════════════════════════

class ScopedPermissionManager:
    """Scoped permission gate for every runtime action."""

    def __init__(self) -> None:
        self._grants: List[_Grant] = []
        self._security_scopes: List[SecurityScope] = []
        self._pending: Optional[PendingConfirmation] = None
        self._audit: List[Dict[str, Any]] = []

    # ── core gate ────────────────────────────────────────────────

    def decide(self, permission_class: PermissionClass, action: str,
               params: Optional[Dict[str, Any]] = None,
               *, target: str = "", operation: str = "") -> PermissionDecision:
        """The single gate every runtime action passes through."""
        params = params or {}
        scope_key = self._scope_key(action, params)
        self._audit.append({
            "t": time.time(), "cls": permission_class.value,
            "action": action, "scope": scope_key,
        })

        if permission_class is PermissionClass.READ_ONLY:
            return PermissionDecision(True, "auto", "read-only observation",
                                      permission_class, scope_key)
        if permission_class is PermissionClass.REVERSIBLE_LOCAL:
            return PermissionDecision(True, "auto", "reversible local change",
                                      permission_class, scope_key)

        if permission_class is PermissionClass.SECURITY_TESTING:
            if not target:
                target = str(params.get("target") or "")
            if not operation:
                operation = str(params.get("operation") or action)
            scope = self._matching_security_scope(target, operation)
            if scope is not None:
                return PermissionDecision(
                    True, "confirmed_by_grant",
                    f"within authorized scope {scope.scope} on {target}",
                    permission_class, scope_key)
            # NO scope → DENY (not a confirmation): no verbal yes can make
            # an unauthorized probe legal; explicit scope authorization is
            # required first, and scope is never widened by asking.
            return PermissionDecision(
                False, "deny",
                "security testing requires explicit target/scope "
                "authorization (use authorize_security_scope)",
                permission_class, scope_key)

        # EXTERNAL_SIDE_EFFECT / DESTRUCTIVE
        for g in self._grants:
            if g.permission_class is permission_class and g.covers(scope_key) \
                    and not g.used:
                if not g.persistent:
                    g.used = True
                return PermissionDecision(
                    True, "confirmed_by_grant",
                    f"scoped grant '{g.scope_key}'"
                    + (" (persistent)" if g.persistent else ""),
                    permission_class, scope_key)
        return PermissionDecision(
            False, "confirmation",
            f"{permission_class.value} requires user confirmation",
            permission_class, scope_key)

    # ── confirmation lifecycle ───────────────────────────────────

    def request_confirmation(self, permission_class: PermissionClass,
                             action: str,
                             params: Optional[Dict[str, Any]] = None
                             ) -> PendingConfirmation:
        """Create (or return) the pending confirmation for this action."""
        params = params or {}
        scope_key = self._scope_key(action, params)
        if self._pending is None or self._pending.action != action \
                or self._pending.scope_key != scope_key:
            self._pending = PendingConfirmation(
                id=uuid.uuid4().hex[:10],
                permission_class=permission_class, action=action,
                params=dict(params), scope_key=scope_key,
                prompt=(f"Allow {action} ({permission_class.value})?"
                        f" This has a {permission_class.value} effect."))
        return self._pending

    def pending(self) -> Optional[PendingConfirmation]:
        return self._pending

    def resolve_confirmation(self, user_text: str, *,
                             persist_scope: bool = False) -> PermissionDecision:
        """Resolve the pending confirmation from a user utterance.

        Returns (allowed=False, mode='deny') when denied/cancelled, or
        (allowed=True, mode='confirmed_by_grant') when approved; a persistent
        approval stores a scoped grant for future identical scopes.
        """
        pending = self._pending
        if pending is None:
            return PermissionDecision(False, "deny", "no pending confirmation")
        text = (user_text or "").strip().lower()
        words = " ".join(text.split())
        if any(p in words for p in CANCEL_PHRASES):
            self._pending = None
            logger.info("[GOAL-PERM] DENIED %s (%s)", pending.action,
                        pending.scope_key)
            return PermissionDecision(False, "deny", "user denied",
                                      pending.permission_class,
                                      pending.scope_key)
        if any(p in words for p in CONFIRM_PHRASES) \
                or "always" in words or "remember" in words \
                or "allow" in words:
            self._pending = None
            if persist_scope or "always" in words or "remember" in words \
                    or "always allow" in words:
                self._grants.append(_Grant(
                    scope_key=pending.scope_key,
                    permission_class=pending.permission_class,
                    persistent=True))
                logger.info("[GOAL-PERM] persistent grant %s [%s]",
                            pending.scope_key, pending.permission_class.value)
            logger.info("[GOAL-PERM] approved %s (%s)", pending.action,
                        pending.scope_key)
            return PermissionDecision(True, "confirmed_by_grant",
                                      "user approved",
                                      pending.permission_class,
                                      pending.scope_key)
        # Not a confirmation phrase — keep pending; caller treats as a
        # non-answer (e.g. a changed goal).
        return PermissionDecision(False, "confirmation",
                                  "awaiting explicit confirmation",
                                  pending.permission_class,
                                  pending.scope_key)

    # ── scoped grants & security scope ───────────────────────────

    def scope_key_for(self, action: str,
                      params: Optional[Dict[str, Any]] = None) -> str:
        """Public builder for the CANONICAL scope key of an action.

        Scope keys are lowercase-normalized (``contact=Rahul`` becomes
        ``contact=rahul``), so callers must not hand-build them: use this so a
        grant matches exactly what :meth:`decide` will compute.
        """
        return self._scope_key(action, params or {})

    def grant_scope(self, scope_key: str,
                    permission_class: PermissionClass, *,
                    persistent: bool = True) -> None:
        if not _SCOPE_RE.match(scope_key):
            raise ValueError(f"invalid scope key: {scope_key!r}")
        self._grants.append(_Grant(scope_key=scope_key,
                                   permission_class=permission_class,
                                   persistent=persistent))

    def revoke_scope(self, scope_key: str) -> int:
        before = len(self._grants)
        self._grants = [g for g in self._grants if g.scope_key != scope_key]
        return before - len(self._grants)

    def grants(self) -> List[Dict[str, Any]]:
        return [{"scope": g.scope_key, "cls": g.permission_class.value,
                 "persistent": g.persistent, "used": g.used}
                for g in self._grants]

    def authorize_security_scope(self, targets: List[str],
                                 operations: List[str]) -> SecurityScope:
        """Register an EXPLICIT security scope (never derived from a goal)."""
        scope = SecurityScope(
            targets=tuple(t.strip().lower() for t in targets),
            scope=tuple(o.strip().lower() for o in operations))
        self._security_scopes.append(scope)
        logger.info("[GOAL-PERM] security scope authorized targets=%s ops=%s",
                    scope.targets, scope.scope)
        return scope

    def security_scopes(self) -> List[SecurityScope]:
        return list(self._security_scopes)

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _scope_key(action: str, params: Dict[str, Any]) -> str:
        """Deterministic scope key: 'skill.action:k=v;k=v' (sorted)."""
        base = f"{action}".strip().lower()
        extras = []
        for k in sorted(params):
            if k in ("target", "contact", "path", "recipient", "email",
                     "host", "url", "app"):
                extras.append(f"{k}={str(params[k]).strip().lower()}")
        return base + (":" + ";".join(extras) if extras else "")

    def _matching_security_scope(self, target: str,
                                 operation: str) -> Optional[SecurityScope]:
        for s in self._security_scopes:
            if s.covers(target, operation):
                return s
        return None

    def audit(self) -> List[Dict[str, Any]]:
        return list(self._audit)


__all__ = [
    "PermissionDecision", "ScopedPermissionManager", "PendingConfirmation",
    "SecurityScope", "classify_action", "CONFIRM_PHRASES", "CANCEL_PHRASES",
]
