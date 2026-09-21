"""GoalRuntime permission layer — deterministic tests (no I/O)."""

from __future__ import annotations

import pytest

from goalruntime.models import PermissionClass
from goalruntime.permissions import (
    CANCEL_PHRASES, CONFIRM_PHRASES, PermissionDecision,
    ScopedPermissionManager, classify_action,
)


# ═══════════════════════════════════════════════════════════════════
# Deterministic classification
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("action,expected", [
    ("observe", PermissionClass.READ_ONLY),
    ("screenshot", PermissionClass.READ_ONLY),
    ("read_latest_message", PermissionClass.READ_ONLY),
    ("read_email", PermissionClass.READ_ONLY),
    ("open_app", PermissionClass.REVERSIBLE_LOCAL),
    ("navigate", PermissionClass.REVERSIBLE_LOCAL),
    ("create_folder", PermissionClass.REVERSIBLE_LOCAL),
    ("type_text", PermissionClass.REVERSIBLE_LOCAL),
    ("send_message", PermissionClass.EXTERNAL_SIDE_EFFECT),
    ("send_email", PermissionClass.EXTERNAL_SIDE_EFFECT),
    ("delete_file", PermissionClass.DESTRUCTIVE),
    ("format_disk", PermissionClass.DESTRUCTIVE),
    ("security_scan", PermissionClass.SECURITY_TESTING),
    ("port_scan", PermissionClass.SECURITY_TESTING),
])
def test_classify_action(action, expected):
    cls, reason = classify_action(action)
    assert cls is expected
    assert reason


def test_compose_intent_is_external():
    """Typing that declares a send intent is EXTERNAL, not local."""
    cls, _ = classify_action("type_text", {"text": "send message: hi"})
    assert cls is PermissionClass.EXTERNAL_SIDE_EFFECT


def test_unknown_side_effect_verb_is_external():
    cls, _ = classify_action("blast_notification")
    assert cls is PermissionClass.EXTERNAL_SIDE_EFFECT


# ═══════════════════════════════════════════════════════════════════
# The gate
# ═══════════════════════════════════════════════════════════════════

def test_read_only_runs_automatically():
    pm = ScopedPermissionManager()
    d = pm.decide(PermissionClass.READ_ONLY, "observe", {})
    assert d.allowed and d.mode == "auto"


def test_reversible_local_runs_automatically():
    pm = ScopedPermissionManager()
    d = pm.decide(PermissionClass.REVERSIBLE_LOCAL, "create_folder",
                  {"path": "/tmp/x"})
    assert d.allowed and d.mode == "auto"


def test_external_requires_confirmation():
    pm = ScopedPermissionManager()
    d = pm.decide(PermissionClass.EXTERNAL_SIDE_EFFECT, "send_message",
                  {"contact": "Rahul"})
    assert not d.allowed
    assert d.mode == "confirmation"


def test_destructive_requires_confirmation():
    pm = ScopedPermissionManager()
    d = pm.decide(PermissionClass.DESTRUCTIVE, "delete_file",
                  {"path": "/tmp/x"})
    assert not d.allowed and d.mode == "confirmation"


# ═══════════════════════════════════════════════════════════════════
# Confirmation flow: approve / deny / persistent scoped grant
# ═══════════════════════════════════════════════════════════════════

def test_confirmation_approve_then_one_shot():
    pm = ScopedPermissionManager()
    pm.request_confirmation(PermissionClass.EXTERNAL_SIDE_EFFECT,
                            "send_message", {"contact": "Rahul"})
    d = pm.resolve_confirmation("yes")
    assert d.allowed and d.mode == "confirmed_by_grant"
    # One-shot: the SAME action needs confirmation again (no persistent grant).
    d2 = pm.decide(PermissionClass.EXTERNAL_SIDE_EFFECT, "send_message",
                   {"contact": "Rahul"})
    assert not d2.allowed and d2.mode == "confirmation"


def test_confirmation_deny():
    pm = ScopedPermissionManager()
    pm.request_confirmation(PermissionClass.DESTRUCTIVE, "delete_file",
                            {"path": "/tmp/x"})
    d = pm.resolve_confirmation("no")
    assert not d.allowed and d.mode == "deny"
    assert pm.pending() is None


def test_persistent_scoped_grant_covers_narrower_scope():
    pm = ScopedPermissionManager()
    pm.request_confirmation(PermissionClass.EXTERNAL_SIDE_EFFECT,
                            "send_message", {"contact": "Rahul"})
    d = pm.resolve_confirmation("always allow")
    assert d.allowed
    grants = pm.grants()
    assert grants and grants[0]["persistent"] is True
    # The persistent grant now auto-approves the same action (and narrower
    # parameterizations) without any further confirmation.
    d2 = pm.decide(PermissionClass.EXTERNAL_SIDE_EFFECT, "send_message",
                   {"contact": "Rahul"})
    assert d2.allowed and d2.mode == "confirmed_by_grant"


def test_scoped_grant_is_scoped():
    """A grant for contact=Rahul does NOT cover contact=Priya."""
    pm = ScopedPermissionManager()
    pm.grant_scope("send_message:contact=rahul",
                   PermissionClass.EXTERNAL_SIDE_EFFECT)
    d_rahul = pm.decide(PermissionClass.EXTERNAL_SIDE_EFFECT, "send_message",
                        {"contact": "Rahul"})
    assert d_rahul.allowed
    d_priya = pm.decide(PermissionClass.EXTERNAL_SIDE_EFFECT, "send_message",
                        {"contact": "Priya"})
    assert not d_priya.allowed and d_priya.mode == "confirmation"


def test_coarse_grant_covers_narrow():
    pm = ScopedPermissionManager()
    pm.grant_scope("send_message", PermissionClass.EXTERNAL_SIDE_EFFECT)
    d = pm.decide(PermissionClass.EXTERNAL_SIDE_EFFECT, "send_message",
                  {"contact": "Anyone"})
    assert d.allowed


def test_revoke_scope():
    pm = ScopedPermissionManager()
    pm.grant_scope("send_message", PermissionClass.EXTERNAL_SIDE_EFFECT)
    assert pm.revoke_scope("send_message") == 1
    d = pm.decide(PermissionClass.EXTERNAL_SIDE_EFFECT, "send_message", {})
    assert not d.allowed


def test_invalid_scope_key_rejected():
    pm = ScopedPermissionManager()
    with pytest.raises(ValueError):
        pm.grant_scope("BAD SCOPE!!", PermissionClass.EXTERNAL_SIDE_EFFECT)


# ═══════════════════════════════════════════════════════════════════
# Security scope: explicit target/scope, NEVER auto-expanding
# ═══════════════════════════════════════════════════════════════════

def test_security_without_scope_is_denied():
    pm = ScopedPermissionManager()
    d = pm.decide(PermissionClass.SECURITY_TESTING, "security_scan",
                  {"target": "staging.example.com", "operation": "port_scan"})
    assert not d.allowed
    assert "authorization" in d.reason


def test_security_with_explicit_scope_allowed():
    pm = ScopedPermissionManager()
    pm.authorize_security_scope(["staging.example.com"],
                                ["port_scan", "recon"])
    d = pm.decide(PermissionClass.SECURITY_TESTING, "security_scan",
                  {"target": "staging.example.com", "operation": "port_scan"})
    assert d.allowed and d.mode == "confirmed_by_grant"


def test_security_never_expands_target_scope():
    pm = ScopedPermissionManager()
    pm.authorize_security_scope(["staging.example.com"], ["port_scan"])
    d = pm.decide(PermissionClass.SECURITY_TESTING, "security_scan",
                  {"target": "prod.example.com", "operation": "port_scan"})
    assert not d.allowed


def test_security_never_expands_operation_scope():
    pm = ScopedPermissionManager()
    pm.authorize_security_scope(["staging.example.com"], ["port_scan"])
    d = pm.decide(PermissionClass.SECURITY_TESTING, "security_scan",
                  {"target": "staging.example.com", "operation": "vuln_probe"})
    assert not d.allowed


# ═══════════════════════════════════════════════════════════════════
# Vocabulary parity with agent.task_continuation
# ═══════════════════════════════════════════════════════════════════

def test_confirmation_vocabulary_matches_task_continuation():
    from agent.task_continuation import CONFIRM_PHRASES as TC_CONFIRM, \
        CANCEL_PHRASES as TC_CANCEL
    assert CONFIRM_PHRASES == TC_CONFIRM
    assert CANCEL_PHRASES == TC_CANCEL


def test_ambiguous_utterance_keeps_pending():
    pm = ScopedPermissionManager()
    pm.request_confirmation(PermissionClass.EXTERNAL_SIDE_EFFECT,
                            "send_message", {"contact": "Rahul"})
    d = pm.resolve_confirmation("what message are we sending?")
    assert not d.allowed and d.mode == "confirmation"
    assert pm.pending() is not None  # still waiting for a real answer
