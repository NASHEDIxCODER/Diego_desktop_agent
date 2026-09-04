"""
Startup integration regression tests: wake and face auth are INDEPENDENT.

Bug fixed (2026-09-04): normal `python main.py` could unintentionally
bypass face authentication when the wake model was unavailable. The
engine logged "Wake unavailable: bypassing wake detection and face auth"
and entered LISTEN with the auth provider skipped — even though
no_wake=False and no_auth=False.

Guarantees verified here:

  1. Normal startup            → wake enabled  + auth enabled
  2. --no-wake                 → wake disabled + auth STILL enabled
  3. --no-auth                 → auth disabled + wake STILL enabled
  4. --no-wake --no-auth       → both disabled
  5. Wake model unavailable    → does NOT disable auth
  6. Auth unavailable/failing  → does NOT disable wake,
                                 reported honestly, NOT treated as --no-auth
  7. Both unavailable          → handled independently

State-transition invariant: WAKE → FACE_AUTH → LISTEN are verified
independently. WAKE failure → bypass FACE_AUTH is FORBIDDEN.

The forbidden log line "bypassing wake detection and face auth" may only
appear when BOTH --no-wake and --no-auth were explicitly requested.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

FORBIDDEN_BYPASS_LOG = "bypassing wake detection and face auth"


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _stub_engine(monkeypatch, wake_ok: bool = True):
    """Build a ConversationEngine with all heavy subsystems stubbed.

    Mirrors the stubbing pattern of
    tests/test_production_readiness.py::
    test_engine_degrades_to_listen_when_wake_unavailable.
    """
    import core.conversation_engine as ce

    monkeypatch.setattr(
        ce, "audio_manager",
        SimpleNamespace(is_running=True, total_samples=0,
                        read_since=lambda *a, **k: (None, 0)))
    monkeypatch.setattr(ce, "streaming_tts", MagicMock())
    monkeypatch.setattr(ce, "command_listener", MagicMock())
    monkeypatch.setattr(
        ce, "wake_model_manager",
        MagicMock(load_error=None if wake_ok else "wake model missing",
                  model_name="hey_jarvis" if wake_ok else None,
                  threshold=0.5, loaded=wake_ok))
    monkeypatch.setattr(ce, "session_recorder", MagicMock())
    monkeypatch.setattr(ce, "conv_memory", MagicMock())

    import voice.vad as vad_mod
    monkeypatch.setattr(vad_mod, "unified_vad", MagicMock())

    monkeypatch.setitem(sys.modules, "knowledge.service",
                        SimpleNamespace(knowledge_service=MagicMock()))

    eng = ce.ConversationEngine()

    # Deterministic wake-model boot result
    monkeypatch.setattr(eng, "_ensure_wake_model", lambda: wake_ok)
    # No chime hardware in tests
    monkeypatch.setattr(eng, "_play_wake_chime", lambda: None)

    async def _noop_speak(text):
        return True
    monkeypatch.setattr(eng, "_speak_guarded", _noop_speak)

    return ce, eng


def _spy_transitions(monkeypatch, eng):
    """Record every state the engine enters."""
    transitions = []
    orig = eng._set_state

    def spy(new_state, **diag):
        transitions.append(new_state)
        return orig(new_state, **diag)

    monkeypatch.setattr(eng, "_set_state", spy)
    return transitions


def _install_session_stop(monkeypatch, eng, ce, visited):
    """Fake conversation session: record entry state, stop the loop.

    NOTE: the REAL `_conversation_session()` enters LISTEN itself after
    stream setup. On the wake-bypass/degraded path the engine ALSO sets
    LISTEN explicitly before calling the session (documented always-LISTEN
    fallback entry), so session-entry state there is LISTEN. On the normal
    wake path the session-entry state is whatever preceded it (WAKE or
    FACE_AUTH) — the real session would then set LISTEN.
    """

    async def fake_session():
        visited.append(eng._state)
        eng._running = False

    monkeypatch.setattr(eng, "_conversation_session", fake_session)


def _fake_wake_event(ce, once):
    """Return one accepted WakeEvent, then shut down."""

    async def fake_wake_loop():
        if once:
            once.clear()
            return ce.WakeEvent(model="hey_jarvis", score=0.9,
                                transcript="hey diego")
        # Second pass: end the forever loop cleanly.
        return None

    return fake_wake_loop


# ═══════════════════════════════════════════════════════════════
# 1. Normal startup: wake enabled + auth enabled
# ═══════════════════════════════════════════════════════════════

async def test_normal_startup_wake_and_auth_enabled(monkeypatch):
    """`python main.py` (no flags): WAKE → FACE_AUTH → LISTEN."""
    ce, eng = _stub_engine(monkeypatch, wake_ok=True)
    transitions = _spy_transitions(monkeypatch, eng)

    auth_calls = []

    async def provider():
        auth_calls.append("called")
        return "Alice"

    eng.set_auth_provider(provider)

    monkeypatch.setattr(eng, "_wake_listen_loop",
                        _fake_wake_event(ce, once=[True]))
    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    await asyncio.wait_for(eng.run(no_wake=False), timeout=10)

    # Wake enabled and used
    assert eng._wake_active is True
    assert eng._no_wake is False
    assert ce.EngineState.WAKE in transitions

    # Auth enabled: provider active, gate ran, authenticated
    assert eng._auth_provider is not None
    assert auth_calls == ["called"]
    assert ce.EngineState.FACE_AUTH in transitions
    assert eng._auth_user == "Alice"

    # Independent ordering: WAKE before FACE_AUTH before LISTEN entry
    assert transitions.index(ce.EngineState.WAKE) \
        < transitions.index(ce.EngineState.FACE_AUTH)
    # Session starts right after the auth gate (the real session would
    # now enter LISTEN itself).
    assert visited == [ce.EngineState.FACE_AUTH]

    # Diagnostics report both subsystems active
    diag = eng.get_diagnostics()
    assert diag["wake"] == "READY"
    assert diag["auth"] == "ACTIVE"


# ═══════════════════════════════════════════════════════════════
# 2. --no-wake: wake disabled + auth STILL enabled
# ═══════════════════════════════════════════════════════════════

async def test_no_wake_keeps_auth_enabled(monkeypatch):
    """--no-wake bypasses ONLY wake detection; face auth still runs."""
    ce, eng = _stub_engine(monkeypatch, wake_ok=True)
    transitions = _spy_transitions(monkeypatch, eng)

    auth_calls = []

    async def provider():
        auth_calls.append("called")
        return "Alice"

    eng.set_auth_provider(provider)

    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    await asyncio.wait_for(eng.run(no_wake=True), timeout=10)

    # Wake bypassed
    assert eng._no_wake is True
    assert ce.EngineState.WAKE not in transitions

    # Auth UNCHANGED by --no-wake: provider active and gate ran
    assert eng._auth_provider is not None
    assert eng._needs_auth() is False          # authenticated session now
    assert auth_calls == ["called"]
    assert ce.EngineState.FACE_AUTH in transitions
    assert eng._auth_user == "Alice"
    assert visited == [ce.EngineState.LISTEN]

    diag = eng.get_diagnostics()
    assert diag["wake"] == "BYPASSED"
    assert diag["auth"] == "ACTIVE"


# ═══════════════════════════════════════════════════════════════
# 3. --no-auth: auth disabled + wake STILL enabled
# ═══════════════════════════════════════════════════════════════

async def test_no_auth_keeps_wake_enabled(monkeypatch):
    """--no-auth disables ONLY face auth; wake detection still runs."""
    ce, eng = _stub_engine(monkeypatch, wake_ok=True)
    transitions = _spy_transitions(monkeypatch, eng)

    eng.set_auth_disabled()                    # the --no-auth wiring

    monkeypatch.setattr(eng, "_wake_listen_loop",
                        _fake_wake_event(ce, once=[True]))
    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    await asyncio.wait_for(eng.run(no_wake=False), timeout=10)

    # Wake UNCHANGED by --no-auth: still active and used
    assert eng._wake_active is True
    assert eng._no_wake is False
    assert ce.EngineState.WAKE in transitions

    # Auth genuinely disabled, no FACE_AUTH state entered
    assert eng._auth_provider is None
    assert eng._needs_auth() is False
    assert ce.EngineState.FACE_AUTH not in transitions
    # Wake path: session starts right after WAKE (no auth gate), and the
    # real session would now enter LISTEN itself.
    assert visited == [ce.EngineState.WAKE]

    diag = eng.get_diagnostics()
    assert diag["wake"] == "READY"
    assert diag["auth"] == "DISABLED"


# ═══════════════════════════════════════════════════════════════
# 4. --no-wake --no-auth: both disabled
# ═══════════════════════════════════════════════════════════════

async def test_no_wake_and_no_auth_both_disabled(monkeypatch, caplog):
    """Full dev mode: straight to LISTEN, no wake, no auth."""
    ce, eng = _stub_engine(monkeypatch, wake_ok=True)
    transitions = _spy_transitions(monkeypatch, eng)

    eng.set_auth_disabled()

    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    import logging
    with caplog.at_level(logging.INFO, logger="core.conversation_engine"):
        await asyncio.wait_for(eng.run(no_wake=True), timeout=10)

    assert eng._no_wake is True
    assert eng._auth_provider is None
    assert ce.EngineState.WAKE not in transitions
    assert ce.EngineState.FACE_AUTH not in transitions
    assert visited == [ce.EngineState.LISTEN]

    # ONLY in this configuration is the combined bypass log permitted.
    assert any(FORBIDDEN_BYPASS_LOG in r.message for r in caplog.records)

    diag = eng.get_diagnostics()
    assert diag["wake"] == "BYPASSED"
    assert diag["auth"] == "DISABLED"


# ═══════════════════════════════════════════════════════════════
# 5. Wake model unavailable does NOT disable auth
# ═══════════════════════════════════════════════════════════════

async def test_wake_unavailable_does_not_disable_auth(monkeypatch, caplog):
    """THE production bug: wake model fails at boot in normal mode.

    Expected: wake marked DEGRADED, documented always-LISTEN fallback
    applies, but the auth provider stays ACTIVE and the FACE_AUTH gate
    still runs before LISTEN. Never equivalent to --no-auth.
    """
    ce, eng = _stub_engine(monkeypatch, wake_ok=False)
    transitions = _spy_transitions(monkeypatch, eng)

    auth_calls = []

    async def provider():
        auth_calls.append("called")
        return "Alice"

    eng.set_auth_provider(provider)

    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    import logging
    with caplog.at_level(logging.INFO, logger="core.conversation_engine"):
        await asyncio.wait_for(eng.run(no_wake=False), timeout=10)

    # Wake degraded, documented fallback followed (no WAKE spin)
    assert eng._wake_active is False
    assert eng._no_wake is False
    assert ce.EngineState.WAKE not in transitions

    # AUTH MUST SURVIVE THE WAKE FAILURE
    assert eng._auth_provider is not None, \
        "wake failure must not clear the auth provider"
    assert auth_calls == ["called"], \
        "face auth gate must still run when wake is degraded"
    assert ce.EngineState.FACE_AUTH in transitions
    assert eng._auth_user == "Alice"
    assert visited == [ce.EngineState.LISTEN]

    # Wake clearly marked DEGRADED/UNAVAILABLE, auth untouched
    assert any("DEGRADED" in r.message for r in caplog.records)
    diag = eng.get_diagnostics()
    assert diag["wake"] == "DEGRADED"
    assert diag["auth"] == "ACTIVE"

    # The forbidden combined-bypass log must NOT appear (no flags given)
    assert not any(FORBIDDEN_BYPASS_LOG in r.message for r in caplog.records), \
        "wake failure must not log a face-auth bypass"


# ═══════════════════════════════════════════════════════════════
# 6. Auth unavailable does NOT disable wake
# ═══════════════════════════════════════════════════════════════

async def test_auth_failure_does_not_disable_wake(monkeypatch, caplog):
    """Auth provider fails → reported honestly; wake stays enabled and
    the provider is NOT cleared (never silently treated as --no-auth)."""
    ce, eng = _stub_engine(monkeypatch, wake_ok=True)
    transitions = _spy_transitions(monkeypatch, eng)

    async def failing_provider():
        raise RuntimeError("camera unavailable")

    eng.set_auth_provider(failing_provider)

    monkeypatch.setattr(eng, "_wake_listen_loop",
                        _fake_wake_event(ce, once=[True]))
    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    import logging
    with caplog.at_level(logging.INFO, logger="core.conversation_engine"):
        await asyncio.wait_for(eng.run(no_wake=False), timeout=10)

    # Wake UNTOUCHED by the auth failure
    assert eng._wake_active is True
    assert eng._no_wake is False
    assert ce.EngineState.WAKE in transitions

    # Auth failure reported honestly at the gate
    assert ce.EngineState.FACE_AUTH in transitions
    assert any("FAILED" in r.message and "FACE_AUTH" in r.message
               for r in caplog.records)

    # NOT treated as --no-auth: provider remains active for the next gate
    assert eng._auth_provider is not None, \
        "auth failure must not clear the provider (that would be --no-auth)"
    assert eng._auth_user is None

    # Session proceeds (existing documented behaviour), wake path intact:
    # session starts right after the failed auth gate.
    assert visited == [ce.EngineState.FACE_AUTH]

    diag = eng.get_diagnostics()
    assert diag["wake"] == "READY"
    assert diag["auth"] == "ACTIVE"


# ═══════════════════════════════════════════════════════════════
# 7. Both unavailable — handled independently
# ═══════════════════════════════════════════════════════════════

async def test_wake_and_auth_both_unavailable_independent(monkeypatch, caplog):
    """Wake model fails AND auth fails: each subsystem degrades on its
    own; neither failure flips the other's configured state."""
    ce, eng = _stub_engine(monkeypatch, wake_ok=False)
    transitions = _spy_transitions(monkeypatch, eng)

    async def failing_provider():
        return None                      # e.g. denied / camera missing

    eng.set_auth_provider(failing_provider)

    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    import logging
    with caplog.at_level(logging.INFO, logger="core.conversation_engine"):
        await asyncio.wait_for(eng.run(no_wake=False), timeout=10)

    # Wake degraded → documented always-LISTEN fallback
    assert eng._wake_active is False
    assert ce.EngineState.WAKE not in transitions

    # Auth gate STILL ran despite the wake failure
    assert ce.EngineState.FACE_AUTH in transitions
    # Auth failure reported honestly; provider remains ACTIVE
    assert eng._auth_provider is not None
    assert eng._auth_user is None
    assert any("FAILED" in r.message and "FACE_AUTH" in r.message
               for r in caplog.records)

    assert visited == [ce.EngineState.LISTEN]

    # Neither flag was implicitly set
    assert eng._no_wake is False

    diag = eng.get_diagnostics()
    assert diag["wake"] == "DEGRADED"
    assert diag["auth"] == "ACTIVE"

    # No combined bypass log without explicit flags
    assert not any(FORBIDDEN_BYPASS_LOG in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════
# Log-policy regression: the forbidden line
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("no_wake,auth_disabled", [
    (False, False),   # normal
    (True, False),    # --no-wake only
    (False, True),    # --no-auth only
])
async def test_combined_bypass_log_only_with_both_flags(
        monkeypatch, caplog, no_wake, auth_disabled):
    """"bypassing wake detection and face auth" may be logged ONLY when
    both --no-wake and --no-auth were explicitly requested."""
    ce, eng = _stub_engine(monkeypatch, wake_ok=True)

    if auth_disabled:
        eng.set_auth_disabled()
    else:
        async def provider():
            return "Alice"
        eng.set_auth_provider(provider)

    if not no_wake:
        monkeypatch.setattr(eng, "_wake_listen_loop",
                            _fake_wake_event(ce, once=[True]))

    visited = []
    _install_session_stop(monkeypatch, eng, ce, visited)

    import logging
    with caplog.at_level(logging.INFO, logger="core.conversation_engine"):
        await asyncio.wait_for(eng.run(no_wake=no_wake), timeout=10)

    assert not any(FORBIDDEN_BYPASS_LOG in r.message for r in caplog.records), \
        (f"forbidden bypass log emitted with no_wake={no_wake}, "
         f"auth_disabled={auth_disabled}")


# ═══════════════════════════════════════════════════════════════
# Wiring-level independence (entry points)
# ═══════════════════════════════════════════════════════════════

def test_no_wake_flag_does_not_touch_auth_wiring():
    """Diego.run_Diego wires auth from no_auth ONLY — no_wake must not
    appear in the auth decision."""
    import inspect
    import Diego

    src = inspect.getsource(Diego.run_Diego)
    # The auth branch is keyed on no_auth alone.
    assert "if no_auth:" in src
    assert "conversation_engine.set_auth_disabled()" in src
    assert "conversation_engine.set_auth_provider(authenticate_on_wake)" in src
    # no_wake is forwarded ONLY to the engine's run() / health report,
    # never to the auth wiring block.
    auth_block = src.split("if no_auth:", 1)[1].split("# ── Start background", 1)[0]
    assert "no_wake" not in auth_block


def test_engine_run_signature_has_no_auth_param_removed():
    """ConversationEngine.run() takes no auth flag: auth state is owned
    exclusively by the provider wiring (set_auth_provider/set_auth_disabled).
    A no_auth parameter on run() would allow wake-path code to mutate it."""
    import inspect
    from core.conversation_engine import ConversationEngine

    params = inspect.signature(ConversationEngine.run).parameters
    assert "no_auth" not in params
    assert "no_wake" in params


def test_wake_degraded_path_calls_face_auth_gate():
    """Static guarantee: the wake-bypass/degraded branch of run() invokes
    the same _face_auth_gate as the normal post-wake path."""
    import inspect
    from core.conversation_engine import ConversationEngine

    src = inspect.getsource(ConversationEngine.run)
    assert src.count("_face_auth_gate") >= 2, \
        "both the wake path and the wake-bypass/degraded path must run the auth gate"