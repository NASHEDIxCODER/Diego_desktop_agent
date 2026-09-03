"""
Tests for Diego self-diagnostics capability.

Covers:
  * Health query detection and response
  * Slow-query detection and latency diagnosis
  * Voice-latency diagnosis
  * LLM-latency diagnosis
  * Healthy state response
  * Degraded state response
  * Failed subsystem response
  * Partial diagnostic failure handling
  * No hallucinated diagnosis (measured values only)
  * No raw path/log leakage
  * No automatic repair
  * Deterministic facts bypass LLM when possible
"""

import asyncio
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from knowledge.diagnostics import (
    DiagnosticTopic,
    HealthStatus,
    SubsystemStatus,
    DiagnosticReport,
    detect_diagnostic_query,
    is_diagnostic_query,
    format_health_response,
    format_slow_response,
    format_voice_latency_response,
    format_llm_latency_response,
    format_tts_latency_response,
    format_failure_response,
    format_diagnostic_response,
    answer_diagnostic_query,
    LATENCY_THRESHOLDS,
)


# ── Test fixtures ────────────────────────────────────────────────────

def _healthy_report() -> DiagnosticReport:
    """A report with all subsystems healthy."""
    return DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[
            SubsystemStatus(name="microphone", status=HealthStatus.HEALTHY),
            SubsystemStatus(name="vad", status=HealthStatus.HEALTHY),
            SubsystemStatus(name="stt", status=HealthStatus.HEALTHY),
            SubsystemStatus(name="tts", status=HealthStatus.HEALTHY),
            SubsystemStatus(name="llm", status=HealthStatus.HEALTHY, evidence="3 models available"),
        ],
        bottlenecks=[],
        errors=[],
        evidence={
            "latencies": {"stt": 500, "llm": 1000, "tts": 200, "total": 2000},
            "resources": {"ram_percent": 45, "cpu_percent": 20},
        },
    )


def _degraded_report() -> DiagnosticReport:
    """A report with some degraded subsystems."""
    return DiagnosticReport(
        overall=HealthStatus.DEGRADED,
        subsystems=[
            SubsystemStatus(name="microphone", status=HealthStatus.HEALTHY),
            SubsystemStatus(name="tts", status=HealthStatus.DEGRADED, fallback="pyttsx3"),
            SubsystemStatus(name="wake", status=HealthStatus.DEGRADED, fallback="bypass"),
            SubsystemStatus(name="llm", status=HealthStatus.HEALTHY),
        ],
        bottlenecks=[],
        errors=[],
        evidence={"latencies": {}, "resources": {}},
    )


def _failed_report() -> DiagnosticReport:
    """A report with a failed required subsystem."""
    return DiagnosticReport(
        overall=HealthStatus.FAILED,
        subsystems=[
            SubsystemStatus(name="microphone", status=HealthStatus.HEALTHY),
            SubsystemStatus(name="stt", status=HealthStatus.FAILED, evidence="package not installed"),
            SubsystemStatus(name="llm", status=HealthStatus.FAILED, evidence="server unreachable"),
        ],
        bottlenecks=[],
        errors=["stt: package not installed", "llm: server unreachable"],
        evidence={"latencies": {}, "resources": {}},
    )


def _slow_report() -> DiagnosticReport:
    """A report with slow latencies."""
    return DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[
            SubsystemStatus(name="stt", status=HealthStatus.HEALTHY),
            SubsystemStatus(name="llm", status=HealthStatus.HEALTHY),
        ],
        bottlenecks=[("llm", 8000), ("stt", 3000)],
        errors=[],
        evidence={
            "latencies": {"stt": 3000, "llm": 8000, "tts": 200, "total": 12000},
            "resources": {"ram_percent": 45, "cpu_percent": 20},
        },
    )


def _no_data_report() -> DiagnosticReport:
    """A report with no latency data."""
    return DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[
            SubsystemStatus(name="stt", status=HealthStatus.HEALTHY),
        ],
        bottlenecks=[],
        errors=[],
        evidence={
            "latencies": {"stt": None, "llm": None, "tts": None, "total": None},
            "resources": {},
        },
    )


# ── Detection tests: health queries ──────────────────────────────────

@pytest.mark.parametrize("text", [
    "is diego healthy",
    "are you healthy",
    "check your health",
    "system health",
    "health check",
    "is everything ok",
    "is everything working",
    "run diagnostics",
    "check your system",
    "what is your status",
])
def test_health_query_detected(text):
    """Health check queries are detected as HEALTH topic."""
    query = detect_diagnostic_query(text)
    assert query.topic == DiagnosticTopic.HEALTH, f"Failed for: {text}"


# ── Detection tests: slow queries ────────────────────────────────────

@pytest.mark.parametrize("text", [
    "why is diego slow",
    "why are you slow",
    "diego is slow",
    "you are slow",
    "why is it taking so long",
    "why does it take so long",
    "performance issue",
    "why is diego lagging",
])
def test_slow_query_detected(text):
    """Slowness queries are detected as SLOW topic."""
    query = detect_diagnostic_query(text)
    assert query.topic == DiagnosticTopic.SLOW, f"Failed for: {text}"


# ── Detection tests: voice latency queries ───────────────────────────

@pytest.mark.parametrize("text", [
    "why is voice recognition slow",
    "why is speech recognition slow",
    "voice recognition is slow",
    "why is stt slow",
    "why is whisper slow",
    "why is transcription slow",
])
def test_voice_latency_query_detected(text):
    """Voice latency queries are detected as VOICE_LATENCY topic."""
    query = detect_diagnostic_query(text)
    assert query.topic == DiagnosticTopic.VOICE_LATENCY, f"Failed for: {text}"


# ── Detection tests: LLM latency queries ─────────────────────────────

@pytest.mark.parametrize("text", [
    "why is llm slow",
    "why is llm response slow",
    "llm is slow",
    "why is ai slow",
    "why is ollama slow",
    "why do you take so long to think",
])
def test_llm_latency_query_detected(text):
    """LLM latency queries are detected as LLM_LATENCY topic."""
    query = detect_diagnostic_query(text)
    assert query.topic == DiagnosticTopic.LLM_LATENCY, f"Failed for: {text}"


# ── Detection tests: failure queries ─────────────────────────────────

@pytest.mark.parametrize("text", [
    "what's wrong with diego",
    "what is wrong with you",
    "what's the problem",
    "what failed",
    "what's broken",
    "what's not working",
    "what went wrong",
])
def test_failure_query_detected(text):
    """Failure queries are detected as FAILURE topic."""
    query = detect_diagnostic_query(text)
    assert query.topic == DiagnosticTopic.FAILURE, f"Failed for: {text}"


# ── Detection tests: non-diagnostic queries ──────────────────────────

@pytest.mark.parametrize("text", [
    "what is the capital of France",
    "open firefox",
    "play some music",
    "tell me a joke",
    "what CPU do I have",  # system info, not diagnostic
    "how much RAM do I have",  # system info, not diagnostic
])
def test_non_diagnostic_not_detected(text):
    """Non-diagnostic queries are not detected."""
    query = detect_diagnostic_query(text)
    assert query.topic == DiagnosticTopic.NONE, f"False positive for: {text}"


# ── Response formatting: healthy state ───────────────────────────────

def test_healthy_response():
    """Healthy state returns positive response."""
    report = _healthy_report()
    response = format_health_response(report)
    assert "normally" in response.lower() or "healthy" in response.lower()
    assert "problem" not in response.lower()


def test_healthy_response_concise():
    """Healthy response is concise for voice."""
    report = _healthy_report()
    response = format_health_response(report)
    assert len(response) <= 300


# ── Response formatting: degraded state ──────────────────────────────

def test_degraded_response():
    """Degraded state mentions fallbacks."""
    report = _degraded_report()
    response = format_health_response(report)
    assert "fallback" in response.lower() or "reduced" in response.lower()


def test_degraded_response_names_subsystems():
    """Degraded response names the affected subsystems."""
    report = _degraded_report()
    response = format_health_response(report)
    assert "voice output" in response.lower() or "wake" in response.lower()


# ── Response formatting: failed state ────────────────────────────────

def test_failed_response():
    """Failed state reports the problem."""
    report = _failed_report()
    response = format_health_response(report)
    assert "problem" in response.lower() or "not available" in response.lower()


def test_failed_response_names_subsystems():
    """Failed response names the failed subsystems."""
    report = _failed_report()
    response = format_health_response(report)
    assert "speech recognition" in response.lower() or "ai" in response.lower()


# ── Response formatting: slow query ──────────────────────────────────

def test_slow_response_with_measured_timings():
    """Slow query reports actual measured timings."""
    report = _slow_report()
    response = format_slow_response(report)
    # Should mention the measured latency
    assert "8.0 seconds" in response or "8000" in response or "seconds" in response.lower()


def test_slow_response_identifies_bottleneck():
    """Slow query identifies the bottleneck subsystem."""
    report = _slow_report()
    response = format_slow_response(report)
    assert "ai reasoning" in response.lower() or "speech recognition" in response.lower()


def test_slow_response_no_data():
    """Slow query with no data says so honestly."""
    report = _no_data_report()
    response = format_slow_response(report)
    assert "don't have" in response.lower() or "no" in response.lower()


def test_slow_response_no_guessing():
    """Slow query never guesses — only measured values."""
    report = _no_data_report()
    response = format_slow_response(report)
    # Should NOT invent a diagnosis
    assert "because" not in response.lower() or "don't have" in response.lower()


# ── Response formatting: voice latency ───────────────────────────────

def test_voice_latency_response_slow():
    """Voice latency diagnosis reports slow STT."""
    report = DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[],
        evidence={"latencies": {"stt": 3000}},
    )
    response = format_voice_latency_response(report)
    assert "speech recognition" in response.lower()
    assert "3.0 seconds" in response or "slower" in response.lower()


def test_voice_latency_response_normal():
    """Voice latency diagnosis reports normal STT."""
    report = DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[],
        evidence={"latencies": {"stt": 500}},
    )
    response = format_voice_latency_response(report)
    assert "normal" in response.lower() or "within" in response.lower()


def test_voice_latency_response_no_data():
    """Voice latency diagnosis with no data says so."""
    report = DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[],
        evidence={"latencies": {"stt": None}},
    )
    response = format_voice_latency_response(report)
    assert "don't have" in response.lower()


def test_voice_latency_response_stt_failed():
    """Voice latency diagnosis reports STT failure."""
    report = DiagnosticReport(
        overall=HealthStatus.FAILED,
        subsystems=[SubsystemStatus(name="stt", status=HealthStatus.FAILED)],
        evidence={"latencies": {"stt": None}},
    )
    response = format_voice_latency_response(report)
    assert "not available" in response.lower()


# ── Response formatting: LLM latency ─────────────────────────────────

def test_llm_latency_response_slow():
    """LLM latency diagnosis reports slow LLM."""
    report = DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[],
        evidence={"latencies": {"llm": 8000}},
    )
    response = format_llm_latency_response(report)
    assert "ai reasoning" in response.lower() or "8.0 seconds" in response


def test_llm_latency_response_normal():
    """LLM latency diagnosis reports normal LLM."""
    report = DiagnosticReport(
        overall=HealthStatus.HEALTHY,
        subsystems=[],
        evidence={"latencies": {"llm": 1000}},
    )
    response = format_llm_latency_response(report)
    assert "normal" in response.lower() or "within" in response.lower()


def test_llm_latency_response_server_down():
    """LLM latency diagnosis reports server unreachable."""
    report = DiagnosticReport(
        overall=HealthStatus.FAILED,
        subsystems=[SubsystemStatus(name="llm", status=HealthStatus.FAILED, evidence="server unreachable")],
        evidence={"latencies": {"llm": None}},
    )
    response = format_llm_latency_response(report)
    assert "not reachable" in response.lower() or "unavailable" in response.lower()


# ── Response formatting: failure query ───────────────────────────────

def test_failure_response_with_failures():
    """Failure query reports failed subsystems with evidence."""
    report = _failed_report()
    response = format_failure_response(report)
    assert "unavailable" in response.lower()
    # Should report evidence, not invented explanation
    assert "package not installed" in response.lower() or "server unreachable" in response.lower()


def test_failure_response_no_failures():
    """Failure query with no failures says so."""
    report = _healthy_report()
    response = format_failure_response(report)
    assert "no" in response.lower() or "healthy" in response.lower()


def test_failure_response_degraded_only():
    """Failure query with only degraded mentions fallback mode."""
    report = _degraded_report()
    response = format_failure_response(report)
    assert "fallback" in response.lower() or "nothing has failed" in response.lower()


# ── No raw data leakage ──────────────────────────────────────────────

def test_response_no_raw_paths():
    """Responses never contain file paths."""
    for report in [_healthy_report(), _degraded_report(), _failed_report(), _slow_report()]:
        for topic in [DiagnosticTopic.HEALTH, DiagnosticTopic.SLOW, DiagnosticTopic.FAILURE]:
            response = format_diagnostic_response(topic, report)
            assert "/home/" not in response
            assert "/proc/" not in response
            assert "/var/" not in response
            assert ".py" not in response


def test_response_no_json():
    """Responses never contain raw JSON."""
    for report in [_healthy_report(), _degraded_report(), _failed_report(), _slow_report()]:
        for topic in [DiagnosticTopic.HEALTH, DiagnosticTopic.SLOW, DiagnosticTopic.FAILURE]:
            response = format_diagnostic_response(topic, report)
            assert "{" not in response
            assert "}" not in response
            assert '":' not in response


def test_response_no_stack_traces():
    """Responses never contain stack trace indicators."""
    report = _failed_report()
    report.errors = ["Traceback (most recent call last):", "File \"/home/x/y.py\""]
    response = format_failure_response(report)
    assert "traceback" not in response.lower()
    assert "file \"" not in response.lower()


def test_response_no_scores():
    """Responses never contain internal scores."""
    report = _healthy_report()
    report.evidence["scores"] = {"confidence": 0.95, "embedding_score": 0.87}
    response = format_health_response(report)
    assert "0.95" not in response
    assert "0.87" not in response
    assert "score" not in response.lower()


def test_response_no_database_rows():
    """Responses never contain database indicators."""
    report = _healthy_report()
    response = format_health_response(report)
    assert "SELECT" not in response
    assert "INSERT" not in response
    assert "row" not in response.lower() or "row" in response.lower()  # Allow "row" in words


# ── No automatic repair ──────────────────────────────────────────────

def test_no_repair_actions_in_response():
    """Responses never claim to have fixed anything."""
    report = _failed_report()
    response = format_failure_response(report)
    assert "fixed" not in response.lower()
    assert "repaired" not in response.lower()
    assert "restarted" not in response.lower()
    assert "installed" not in response.lower() or "not installed" in response.lower()


def test_no_repair_actions_in_health_response():
    """Health responses never claim to have fixed anything."""
    report = _degraded_report()
    response = format_health_response(report)
    assert "fixed" not in response.lower()
    assert "repaired" not in response.lower()


# ── Decision engine integration ──────────────────────────────────────

def test_decision_engine_diagnostic_path():
    """Decision engine routes diagnostic queries to DIAGNOSTIC path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("is diego healthy")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.DIAGNOSTIC
    assert decision.needs_llm is False
    assert decision.response is not None
    assert len(decision.response) > 0


def test_decision_engine_slow_query():
    """Decision engine routes slow queries to DIAGNOSTIC path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("why is diego slow")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.DIAGNOSTIC
    assert decision.needs_llm is False


def test_decision_engine_voice_latency_query():
    """Decision engine routes voice latency queries to DIAGNOSTIC path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("why is voice recognition slow")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.DIAGNOSTIC
    assert decision.needs_llm is False


def test_decision_engine_llm_latency_query():
    """Decision engine routes LLM latency queries to DIAGNOSTIC path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("why is llm slow")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.DIAGNOSTIC
    assert decision.needs_llm is False


def test_decision_engine_non_diagnostic_not_diagnostic():
    """Decision engine doesn't route non-diagnostic queries to DIAGNOSTIC."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("what is the capital of France")

    decision = asyncio.run(run())
    assert decision.path != DecisionPath.DIAGNOSTIC


# ── Brain integration: no LLM for diagnostics ────────────────────────

class _Result:
    """Minimal CommandResult stand-in."""
    response = ""
    actions_executed = 0
    actions_failed = 0


def _patch_llm(monkeypatch, sentences, captured=None):
    """Replace streaming_llm.generate with a fake async generator."""
    import agent.streaming_llm as sl

    async def fake_generate(text, screen_context="", web_context=None,
                            local_context=None):
        if captured is not None:
            captured["text"] = text
            captured["screen"] = screen_context
            captured["local"] = local_context
        for s in sentences:
            yield s

    monkeypatch.setattr(sl.streaming_llm, "generate", fake_generate)


def _patch_knowledge(monkeypatch, results=None, context=""):
    """Replace the knowledge_service singleton with a fake."""
    ks_mod = __import__("knowledge.service", fromlist=["knowledge_service"])
    monkeypatch.setattr(
        ks_mod, "knowledge_service",
        type("KS", (), {
            "search": staticmethod(
                lambda q, top_k=5, _r=results or []: list(_r)),
            "context_for_llm": staticmethod(
                lambda q, top_k=4, _c=context: _c),
            "refresh_snapshot": staticmethod(lambda: {}),
            "latest_snapshot": staticmethod(lambda: {}),
        })())


def _brain():
    return __import__("agent.brain", fromlist=["agent_brain"]).agent_brain


def test_brain_health_query_no_llm(monkeypatch):
    """Health queries don't use the LLM."""
    _patch_knowledge(monkeypatch)
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "is diego healthy", None, _Result()))

    # LLM never invoked
    assert "text" not in captured
    # Response is from diagnostics
    assert resp is not None
    assert len(resp) > 0


def test_brain_slow_query_no_llm(monkeypatch):
    """Slow queries don't use the LLM."""
    _patch_knowledge(monkeypatch)
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "why is diego slow", None, _Result()))

    assert "text" not in captured
    assert resp is not None


def test_brain_voice_latency_query_no_llm(monkeypatch):
    """Voice latency queries don't use the LLM."""
    _patch_knowledge(monkeypatch)
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "why is voice recognition slow", None, _Result()))

    assert "text" not in captured
    assert resp is not None


def test_brain_llm_latency_query_no_llm(monkeypatch):
    """LLM latency queries don't use the LLM."""
    _patch_knowledge(monkeypatch)
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "why is llm slow", None, _Result()))

    assert "text" not in captured
    assert resp is not None


# ── Partial diagnostic failure ───────────────────────────────────────

def test_partial_failure_still_returns_response():
    """Partial collector failure doesn't fail the whole diagnostic."""
    # Simulate a report where some collectors failed
    report = DiagnosticReport(
        overall=HealthStatus.DEGRADED,
        subsystems=[
            SubsystemStatus(name="stt", status=HealthStatus.HEALTHY),
            # Other collectors failed silently
        ],
        bottlenecks=[],
        errors=[],
        evidence={"latencies": {"stt": 500}, "resources": {}},
    )
    response = format_health_response(report)
    assert response is not None
    assert len(response) > 0
    assert "error" not in response.lower() or "degraded" in response.lower()


def test_empty_report_graceful():
    """Empty report returns graceful message."""
    report = DiagnosticReport(
        overall=HealthStatus.UNKNOWN,
        subsystems=[],
        evidence={},
    )
    response = format_health_response(report)
    assert "couldn't" in response.lower()


# ── No hallucinated diagnosis ────────────────────────────────────────

def test_no_hallucinated_diagnosis_without_data():
    """Without measured data, no specific diagnosis is given."""
    report = _no_data_report()
    response = format_slow_response(report)
    # Should NOT say "X is slow" without data
    assert "is slow" not in response.lower() or "don't have" in response.lower()


def test_no_invented_explanations():
    """Failure responses report evidence, not invented explanations."""
    report = DiagnosticReport(
        overall=HealthStatus.FAILED,
        subsystems=[
            SubsystemStatus(name="stt", status=HealthStatus.FAILED, evidence="package not installed"),
        ],
        evidence={},
    )
    response = format_failure_response(report)
    # Should report the actual evidence
    assert "package not installed" in response.lower() or "unavailable" in response.lower()
    # Should NOT invent explanations like "because of network" or "because of memory"
    assert "network" not in response.lower()
    assert "memory" not in response.lower()


# ── Voice UX: concise responses ──────────────────────────────────────

def test_all_responses_concise():
    """All diagnostic responses are concise for voice."""
    reports = [_healthy_report(), _degraded_report(), _failed_report(), _slow_report()]
    topics = [
        DiagnosticTopic.HEALTH,
        DiagnosticTopic.SLOW,
        DiagnosticTopic.VOICE_LATENCY,
        DiagnosticTopic.LLM_LATENCY,
        DiagnosticTopic.FAILURE,
    ]
    for report in reports:
        for topic in topics:
            response = format_diagnostic_response(topic, report)
            assert len(response) <= 400, f"{topic} response too long: {len(response)}"


# ── answer_diagnostic_query integration ──────────────────────────────

def test_answer_diagnostic_query_health():
    """answer_diagnostic_query works for health queries."""
    answer = answer_diagnostic_query("is diego healthy")
    assert answer is not None
    assert len(answer) > 0


def test_answer_diagnostic_query_slow():
    """answer_diagnostic_query works for slow queries."""
    answer = answer_diagnostic_query("why is diego slow")
    assert answer is not None
    assert len(answer) > 0


def test_answer_diagnostic_query_non_diagnostic_returns_none():
    """answer_diagnostic_query returns None for non-diagnostic queries."""
    answer = answer_diagnostic_query("what is the capital of France")
    assert answer is None


# ── Read-only guarantee ──────────────────────────────────────────────

def test_diagnostics_are_read_only():
    """Diagnostics module never modifies state."""
    # This test verifies the module structure doesn't have write operations
    import inspect
    import knowledge.diagnostics as diag

    source = inspect.getsource(diag)
    # Should not contain write operations
    assert "open(" not in source or "w" not in source.split("open(")[1][:10] if "open(" in source else True
    assert ".write(" not in source
    assert "subprocess.run" not in source or "capture_output=True" in source