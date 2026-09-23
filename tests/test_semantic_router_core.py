"""
Phase 25 M2 — semantic capability router core + DecisionEngine L3.5 layer.

Verifies the architecture corrections:
  * Capability objects are DESCRIPTIONS/ROUTING CONTRACTS only — derived
    from the EXISTING tool_registry CAPABILITIES registry; there is no
    execute callable and resolution yields a plain action dict for the
    existing dispatcher (no parallel execution framework).
  * Precedence stays: deterministic direct routes (L0-L3) ->
    semantic capability router (L3.5) -> existing L4-L8 fallback.
  * Semantic similarity alone NEVER executes: gate, ambiguity margin,
    permission (authorize_intent) and precondition/availability checks
    must all pass or the route falls through unchanged.
  * No site/phrase-specific routing logic; generic verb/text/param
    extraction only.
  * Timing instrumentation (semantic_route_ms) is recorded per decision.
"""
import asyncio
import types

import nlp.intent_authorizer as intent_authorizer
from core.decision_engine import DecisionEngine, DecisionPath
from core.semantic_router import RouterContract, SemanticRouter


def _decide(engine: DecisionEngine, text: str):
    return asyncio.run(engine.decide(text))


class _FakeRouter:
    """Minimal command_router stand-in for L3 precedence tests."""

    def __init__(self, kind, action=None, response=None, actions=None,
                 confidence=0.0):
        from core.command_router import RouteKind
        if isinstance(kind, str):
            kind = RouteKind(kind)
        self.calls = 0
        self._result = types.SimpleNamespace(
            kind=kind, action=action, response=response, actions=actions,
            confidence=confidence)

    async def route(self, text):
        self.calls += 1
        return self._result


def _engine(fake_router) -> DecisionEngine:
    eng = DecisionEngine()
    eng._wired = True          # isolated layer test: skip subsystem wiring
    eng._command_router = fake_router
    return eng


# ═══════════════════════════════════════════════════════════
# Contract derivation (description-only, single source of truth)
# ═══════════════════════════════════════════════════════════

def test_contracts_derived_from_capabilities_registry():
    from core.tool_registry import capabilities
    router = SemanticRouter()
    contracts = router.contracts()
    assert contracts, "routable subset of CAPABILITIES must exist"
    known = capabilities()
    for c in contracts:
        assert c.capability in known, (
            f"{c.capability} must be declared in tool_registry.CAPABILITIES")
        # Description/routing contract only — no execution surface.
        assert not hasattr(c, "execute")
        assert c.side_effects in ("READ_ONLY", "MUTATES_SYSTEM",
                                  "EXTERNAL_EFFECT")
        assert c.verification


# ═══════════════════════════════════════════════════════════
# Resolution — canonical action dict for the EXISTING dispatcher
# ═══════════════════════════════════════════════════════════

def test_route_resolves_to_existing_action_dict():
    out = SemanticRouter().route("open https://example.com/dashboard now")
    assert out.resolved, out.reason
    assert out.action is not None
    assert set(out.action.keys()) == {"action", "params"}
    assert out.action["action"] == "browser_navigate"
    assert out.action["params"]["url"].startswith("https://example.com")
    assert out.budget == "semantic"
    assert out.similarity >= 0.42
    assert out.route_ms >= 0.0


def test_route_filesystem_search_binds_path_and_extension():
    out = SemanticRouter().route('find *.pdf files in "/home/user/docs"')
    assert out.resolved, out.reason
    assert out.action["action"] == "filesystem"
    assert out.action["params"]["action"] == "search"
    assert out.action["params"]["path"] == "/home/user/docs"
    assert out.action["params"]["extensions"] == ["pdf"]


# ═══════════════════════════════════════════════════════════
# Gates — similarity alone must never execute
# ═══════════════════════════════════════════════════════════

def test_route_below_gate_unresolved():
    out = SemanticRouter().route(
        "quantum banana pancakes for breakfast please")
    assert not out.resolved
    assert out.reason.startswith("below_gate")


def test_route_high_similarity_without_params_falls_through():
    # The text mirrors a capability description (high similarity) but
    # the required URL cannot be extracted → never guess, fall through.
    out = SemanticRouter().route("open a url in the browser now")
    assert not out.resolved
    assert out.reason.startswith("missing")


def test_route_ambiguous_margin_falls_through():
    def _c(name):
        return RouterContract(
            capability=name, purpose="find things fast now",
            side_effects="READ_ONLY", verification="observed",
            source_tool="t", action="browser_get_url",
            build_params=lambda t: {}, verbs=("locate",),
            examples=("find things fast now",))
    router = SemanticRouter(contracts=[_c("x.alpha"), _c("y.beta")])
    out = router.route("find things fast now")
    assert not out.resolved
    assert out.reason.startswith("ambiguous")


def test_route_blocked_by_authorization(monkeypatch):
    monkeypatch.setattr(
        intent_authorizer, "authorize_intent",
        lambda t: types.SimpleNamespace(actionable=False,
                                        category="BLOCKED"))
    out = SemanticRouter().route("open https://example.com")
    assert not out.resolved
    assert out.reason.startswith("permission")


def test_route_unavailable_action_falls_through(monkeypatch):
    monkeypatch.setattr(
        intent_authorizer, "authorize_intent",
        lambda t: types.SimpleNamespace(actionable=True,
                                        category="DETERMINISTIC_COMMAND"))
    contract = RouterContract(
        capability="ghost.do", purpose="do ghost things softly now",
        side_effects="READ_ONLY", verification="observed",
        source_tool="t", action="definitely_not_a_real_action",
        build_params=lambda t: {}, verbs=("ghost",),
        examples=("do ghost things softly now",))
    out = SemanticRouter(contracts=[contract]).route(
        "do ghost things softly now")
    assert not out.resolved
    assert out.reason.startswith("unavailable")


# ═══════════════════════════════════════════════════════════
# DecisionEngine precedence (correction: keep EXACTLY as before
# + semantic layer between L3 and L4-L8)
# ═══════════════════════════════════════════════════════════

def test_decision_path_has_semantic_member():
    assert DecisionPath.SEMANTIC.value == "SEMANTIC"
    eng = DecisionEngine()
    assert DecisionPath.SEMANTIC in eng._path_counts


def test_l3_direct_execution_wins_over_semantic(monkeypatch):
    calls = []
    orig = SemanticRouter.route

    def spy(self, text):
        calls.append(text)
        return orig(self, text)

    monkeypatch.setattr(SemanticRouter, "route", spy)
    fake = _FakeRouter(
        kind="SIMPLE_DESKTOP",
        action={"action": "desktop_open", "params": {"app": "firefox"}})
    d = _decide(_engine(fake), "open firefox")
    assert fake.calls == 1
    assert d.path == DecisionPath.DIRECT_EXECUTION
    assert calls == [], "semantic layer must not run after a direct route"


def test_semantic_resolves_after_l3_composite(monkeypatch):
    fake = _FakeRouter(kind="COMPLEX")  # L3: not directly executable
    d = _decide(_engine(fake), "open https://example.com/route-a")
    assert d.path == DecisionPath.SEMANTIC, d
    assert d.needs_llm is False
    assert d.action["action"] == "browser_navigate"
    assert "semantic_route_ms" in d.debug
    assert d.debug["semantic_route_ms"] >= 0.0
    assert d.debug["decision_budget"] == "semantic"
    assert d.latency_us > 0.0


def test_semantic_never_preempts_web_search(monkeypatch):
    def always_resolve(self, text):
        return types.SimpleNamespace(
            resolved=True, budget="semantic", capability="x",
            action={"action": "browser_navigate",
                    "params": {"url": "https://x.example"}},
            confidence=0.99, similarity=0.99, scoring="lexical",
            side_effects="READ_ONLY", verification="", route_ms=0.1,
            reason="ok")
    monkeypatch.setattr(SemanticRouter, "route", always_resolve)
    fake = _FakeRouter(kind="COMPLEX")
    d = _decide(_engine(fake), "search the web for python tutorials")
    assert d.path != DecisionPath.SEMANTIC


def test_semantic_never_preempts_local_knowledge(monkeypatch):
    def always_resolve(self, text):
        return types.SimpleNamespace(
            resolved=True, budget="semantic", capability="x",
            action={"action": "filesystem",
                    "params": {"action": "search"}},
            confidence=0.99, similarity=0.99, scoring="lexical",
            side_effects="READ_ONLY", verification="", route_ms=0.1,
            reason="ok")
    monkeypatch.setattr(SemanticRouter, "route", always_resolve)
    monkeypatch.setattr(intent_authorizer, "_match_local_knowledge",
                        lambda t: True)
    d = _decide(_engine(_FakeRouter(kind="COMPLEX")),
                "find my project files")
    assert d.path != DecisionPath.SEMANTIC


def test_semantic_fallthrough_keeps_llm_fallback(monkeypatch):
    fake = _FakeRouter(kind="COMPLEX")
    d = _decide(_engine(fake),
                "quantum banana pancakes for breakfast please")
    assert d.path != DecisionPath.SEMANTIC
    assert d.needs_llm is True
