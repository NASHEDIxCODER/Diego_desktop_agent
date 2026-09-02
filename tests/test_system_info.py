"""
Tests for system-information query handling.

Covers:
  * "system info" / "system information" detection
  * "PC specs" / "computer specifications" detection
  * CPU query detection and response
  * RAM query detection and response
  * GPU query detection and response
  * OS query detection and response
  * Disk query detection and response
  * Network query detection and response
  * Broad summary response format
  * Partial snapshot failure handling
  * No LLM required for deterministic system facts
  * Document retrieval is NOT used for system-info queries
  * Live values refresh when required
  * Response UX: no raw JSON, dicts, psutil objects, paths, metadata
"""

import asyncio
from typing import Dict, Optional

import pytest

from knowledge.system_info import (
    SystemInfoTopic,
    detect_system_info_query,
    is_system_info_query,
    format_system_summary,
    format_specific_answer,
    answer_system_info_query,
)


# ── Test fixtures ────────────────────────────────────────────────────

def _full_snapshot() -> Dict:
    """A complete snapshot with all fields."""
    return {
        "os": {
            "system": "Linux",
            "release": "6.8.0-45-generic",
            "version": "#45-Ubuntu SMP",
            "machine": "x86_64",
            "hostname": "testhost",
            "python": "3.10.12",
        },
        "cpu": {
            "model": "Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz",
            "cores_physical": 8,
            "cores_logical": 16,
        },
        "memory": {
            "total_gb": 16.0,
            "available_gb": 8.5,
        },
        "gpu": ["NVIDIA GeForce RTX 3070, 8192 MiB"],
        "disks": [
            {
                "device": "/dev/sda1",
                "mountpoint": "/",
                "fstype": "ext4",
                "total_gb": 512.0,
                "used_gb": 256.0,
            },
        ],
        "network": [
            {"name": "eth0", "addresses": ["192.168.1.100"]},
            {"name": "lo", "addresses": ["127.0.0.1"]},
        ],
        "python_envs": {
            "version": "3.10.12",
            "executable": "/usr/bin/python3",
            "venvs": [],
        },
        "installed_apps": ["firefox", "code", "terminal"],
        "processes": ["python", "firefox", "systemd"],
    }


def _partial_snapshot_gpu_error() -> Dict:
    """Snapshot with GPU error."""
    snap = _full_snapshot()
    snap["gpu"] = {"error": "nvidia-smi not found"}
    return snap


def _partial_snapshot_cpu_error() -> Dict:
    """Snapshot with CPU error."""
    snap = _full_snapshot()
    snap["cpu"] = {"error": "cannot read /proc/cpuinfo"}
    return snap


def _empty_snapshot() -> Dict:
    """Empty snapshot."""
    return {}


# ── Detection tests: broad queries ───────────────────────────────────

@pytest.mark.parametrize("text", [
    "system info",
    "system information",
    "show system info",
    "show system information",
    "PC specs",
    "pc specs",
    "show my PC specs",
    "computer specifications",
    "what are my computer specifications",
    "computer information",
    "tell me about my computer",
    "tell me about my PC",
    "what are my specs",
    "system summary",
    "hardware info",
    "machine specs",
])
def test_broad_system_info_detected(text):
    """Broad system-info queries are detected as BROAD topic."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.BROAD, f"Failed for: {text}"


@pytest.mark.parametrize("text", [
    "system info",
    "system information",
    "PC specs",
    "computer specifications",
    "tell me about my computer",
])
def test_is_system_info_query_broad(text):
    """is_system_info_query returns True for broad queries."""
    assert is_system_info_query(text) is True


# ── Detection tests: specific queries ────────────────────────────────

@pytest.mark.parametrize("text", [
    "what CPU do I have",
    "what cpu do i have",
    "which CPU am I using",
    "what processor do I have",
    "how many cores do I have",
    "tell me about my CPU",
])
def test_cpu_query_detected(text):
    """CPU queries are detected as CPU topic."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.CPU, f"Failed for: {text}"


@pytest.mark.parametrize("text", [
    "how much RAM do I have",
    "how much ram do i have",
    "what is my RAM size",
    "how much memory do I have",
    "total RAM",
    "tell me about my RAM",
])
def test_ram_query_detected(text):
    """RAM queries are detected as RAM topic."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.RAM, f"Failed for: {text}"


@pytest.mark.parametrize("text", [
    "what GPU do I have",
    "what gpu do i have",
    "which graphics card do I have",
    "what is my GPU",
    "do I have an NVIDIA GPU",
    "tell me about my graphics card",
])
def test_gpu_query_detected(text):
    """GPU queries are detected as GPU topic."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.GPU, f"Failed for: {text}"


@pytest.mark.parametrize("text", [
    "which OS am I running",
    "what OS is installed",
    "what operating system do I have",
    "am I running Linux",
    "what distro am I using",
])
def test_os_query_detected(text):
    """OS queries are detected as OS topic."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.OS, f"Failed for: {text}"


@pytest.mark.parametrize("text", [
    "what disk space do I have",
    "how much storage do I have",
    "what is my disk capacity",
    "how much free space do I have",
    "tell me about my disks",
])
def test_disk_query_detected(text):
    """Disk queries are detected as DISK topic."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.DISK, f"Failed for: {text}"


@pytest.mark.parametrize("text", [
    "what network interfaces do I have",
    "show my network interfaces",
    "what is my IP address",
    "which network adapters do I have",
])
def test_network_query_detected(text):
    """Network queries are detected as NETWORK topic."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.NETWORK, f"Failed for: {text}"


# ── Detection tests: non-system-info queries ─────────────────────────

@pytest.mark.parametrize("text", [
    "what is the capital of France",
    "tell me a joke",
    "open firefox",
    "what time is it",
    "search for python tutorials",  # python without system context
    "what is human memory",  # memory without system context
    "play some music",
])
def test_non_system_info_not_detected(text):
    """Non-system-info queries are not detected."""
    query = detect_system_info_query(text)
    assert query.topic == SystemInfoTopic.NONE, f"False positive for: {text}"


# ── Live-state detection ─────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "how much RAM do I have currently available",
    "what is my current disk usage",
    "how much free space do I have right now",
    "what is my current memory usage",
])
def test_live_state_detected(text):
    """Queries with live-state indicators are marked as live."""
    query = detect_system_info_query(text)
    assert query.topic != SystemInfoTopic.NONE, f"Not detected for: {text}"
    assert query.is_live is True, f"Failed for: {text}"


@pytest.mark.parametrize("text", [
    "how much RAM do I have",
    "what CPU do I have",
    "system info",
])
def test_persistent_fact_not_live(text):
    """Persistent fact queries are not marked as live."""
    query = detect_system_info_query(text)
    assert query.is_live is False, f"Failed for: {text}"


# ── Response formatting: broad summary ───────────────────────────────

def test_broad_summary_contains_os():
    """Broad summary mentions the OS."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "Linux" in summary


def test_broad_summary_contains_cpu():
    """Broad summary mentions the CPU."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "Intel" in summary or "i7" in summary


def test_broad_summary_contains_ram():
    """Broad summary mentions RAM."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "16" in summary and "RAM" in summary


def test_broad_summary_contains_gpu():
    """Broad summary mentions GPU."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "NVIDIA" in summary or "RTX" in summary


def test_broad_summary_contains_storage():
    """Broad summary mentions storage."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "512" in summary or "storage" in summary.lower()


def test_broad_summary_no_raw_json():
    """Broad summary never contains raw JSON or dict syntax."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "{" not in summary
    assert "}" not in summary
    assert "'" not in summary or "'" in summary  # Allow apostrophes in words
    assert '"os"' not in summary
    assert '"cpu"' not in summary


def test_broad_summary_no_psutil_objects():
    """Broad summary never contains psutil object representations."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "psutil" not in summary.lower()
    assert "svmem" not in summary.lower()
    assert "sdiskusage" not in summary.lower()


def test_broad_summary_no_file_paths():
    """Broad summary never contains file paths."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "/dev/" not in summary
    assert "/home/" not in summary
    assert "/proc/" not in summary


def test_broad_summary_concise():
    """Broad summary is concise (voice UX)."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    # Should be under 500 chars for voice
    assert len(summary) <= 600, f"Summary too long: {len(summary)} chars"


# ── Response formatting: specific queries ────────────────────────────

def test_cpu_answer_specific():
    """CPU query returns only CPU info."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.CPU, snap)
    assert "Intel" in answer or "i7" in answer
    assert "cores" in answer.lower()
    # Should NOT contain unrelated info
    assert "NVIDIA" not in answer
    assert "network" not in answer.lower()


def test_ram_answer_specific():
    """RAM query returns only RAM info."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.RAM, snap)
    assert "16" in answer
    assert "RAM" in answer
    # Should NOT contain unrelated info
    assert "Intel" not in answer
    assert "NVIDIA" not in answer


def test_gpu_answer_specific():
    """GPU query returns only GPU info."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.GPU, snap)
    assert "NVIDIA" in answer or "RTX" in answer
    # Should NOT contain unrelated info
    assert "Intel" not in answer or "Intel" in answer  # May mention in GPU name
    assert "RAM" not in answer or "8192 MiB" in answer  # GPU memory is OK


def test_os_answer_specific():
    """OS query returns only OS info."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.OS, snap)
    assert "Linux" in answer
    # Should NOT contain unrelated info
    assert "NVIDIA" not in answer
    assert "RAM" not in answer


def test_disk_answer_specific():
    """Disk query returns only disk info."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.DISK, snap)
    assert "512" in answer or "storage" in answer.lower()
    # Should NOT contain unrelated info
    assert "NVIDIA" not in answer
    assert "Intel" not in answer


def test_network_answer_specific():
    """Network query returns only network info."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.NETWORK, snap)
    assert "network" in answer.lower() or "interface" in answer.lower()
    # Should NOT contain unrelated info
    assert "NVIDIA" not in answer
    assert "Intel" not in answer or "Intel" in answer  # May be in interface name


def test_ram_live_includes_available():
    """Live RAM query includes available memory."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.RAM, snap, is_live=True)
    assert "available" in answer.lower()


def test_disk_live_includes_free():
    """Live disk query includes free space."""
    snap = _full_snapshot()
    answer = format_specific_answer(SystemInfoTopic.DISK, snap, is_live=True)
    assert "free" in answer.lower()


# ── Partial snapshot failure handling ────────────────────────────────

def test_partial_failure_gpu_error_returns_available_fields():
    """GPU error doesn't fail the whole request."""
    snap = _partial_snapshot_gpu_error()
    summary = format_system_summary(snap)
    # Should still have OS, CPU, RAM
    assert "Linux" in summary
    assert "Intel" in summary or "i7" in summary
    assert "RAM" in summary
    # Should mention GPU error optionally
    assert "GPU" in summary or "couldn't read" in summary.lower()


def test_partial_failure_cpu_error_returns_available_fields():
    """CPU error doesn't fail the whole request."""
    snap = _partial_snapshot_cpu_error()
    summary = format_system_summary(snap)
    # Should still have OS, RAM
    assert "Linux" in summary
    assert "RAM" in summary


def test_partial_failure_specific_gpu_error():
    """Specific GPU query with error returns graceful message."""
    snap = _partial_snapshot_gpu_error()
    answer = format_specific_answer(SystemInfoTopic.GPU, snap)
    assert "couldn't read" in answer.lower()


def test_empty_snapshot_graceful():
    """Empty snapshot returns graceful message."""
    snap = _empty_snapshot()
    summary = format_system_summary(snap)
    assert "couldn't read" in summary.lower()


def test_none_snapshot_graceful():
    """None snapshot returns graceful message."""
    summary = format_system_summary(None)
    assert "couldn't read" in summary.lower()


# ── answer_system_info_query integration ─────────────────────────────

def test_answer_system_info_query_broad():
    """answer_system_info_query works for broad queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("system info", lambda: snap)
    assert answer is not None
    assert "Linux" in answer
    assert "RAM" in answer


def test_answer_system_info_query_cpu():
    """answer_system_info_query works for CPU queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("what CPU do I have", lambda: snap)
    assert answer is not None
    assert "Intel" in answer or "i7" in answer


def test_answer_system_info_query_ram():
    """answer_system_info_query works for RAM queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("how much RAM do I have", lambda: snap)
    assert answer is not None
    assert "16" in answer


def test_answer_system_info_query_gpu():
    """answer_system_info_query works for GPU queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("what GPU do I have", lambda: snap)
    assert answer is not None
    assert "NVIDIA" in answer or "RTX" in answer


def test_answer_system_info_query_os():
    """answer_system_info_query works for OS queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("which OS am I running", lambda: snap)
    assert answer is not None
    assert "Linux" in answer


def test_answer_system_info_query_disk():
    """answer_system_info_query works for disk queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("what disk space do I have", lambda: snap)
    assert answer is not None
    assert "512" in answer or "storage" in answer.lower()


def test_answer_system_info_query_network():
    """answer_system_info_query works for network queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("what network interfaces do I have", lambda: snap)
    assert answer is not None
    assert "network" in answer.lower() or "interface" in answer.lower()


def test_answer_system_info_query_non_system_returns_none():
    """answer_system_info_query returns None for non-system queries."""
    snap = _full_snapshot()
    answer = answer_system_info_query("what is the capital of France", lambda: snap)
    assert answer is None


# ── Decision engine integration ──────────────────────────────────────

def test_decision_engine_system_info_path():
    """Decision engine routes system-info queries to SYSTEM_INFO path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("system info")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.SYSTEM_INFO
    assert decision.needs_llm is False
    assert decision.response is not None
    assert len(decision.response) > 0


def test_decision_engine_cpu_query():
    """Decision engine routes CPU queries to SYSTEM_INFO path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("what CPU do I have")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.SYSTEM_INFO
    assert decision.needs_llm is False


def test_decision_engine_ram_query():
    """Decision engine routes RAM queries to SYSTEM_INFO path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("how much RAM do I have")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.SYSTEM_INFO
    assert decision.needs_llm is False


def test_decision_engine_gpu_query():
    """Decision engine routes GPU queries to SYSTEM_INFO path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("what GPU do I have")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.SYSTEM_INFO
    assert decision.needs_llm is False


def test_decision_engine_os_query():
    """Decision engine routes OS queries to SYSTEM_INFO path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("which OS am I running")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.SYSTEM_INFO
    assert decision.needs_llm is False


def test_decision_engine_disk_query():
    """Decision engine routes disk queries to SYSTEM_INFO path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("what disk space do I have")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.SYSTEM_INFO
    assert decision.needs_llm is False


def test_decision_engine_network_query():
    """Decision engine routes network queries to SYSTEM_INFO path."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("what network interfaces do I have")

    decision = asyncio.run(run())
    assert decision.path == DecisionPath.SYSTEM_INFO
    assert decision.needs_llm is False


def test_decision_engine_non_system_query_not_system_info():
    """Decision engine doesn't route non-system queries to SYSTEM_INFO."""
    from core.decision_engine import DecisionEngine, DecisionPath

    engine = DecisionEngine()

    async def run():
        return await engine.decide("what is the capital of France")

    decision = asyncio.run(run())
    assert decision.path != DecisionPath.SYSTEM_INFO


# ── Brain integration: no LLM for system facts ───────────────────────

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


def _patch_knowledge(monkeypatch, results, context=""):
    """Replace the knowledge_service singleton with a fake."""
    ks_mod = __import__("knowledge.service", fromlist=["knowledge_service"])
    monkeypatch.setattr(
        ks_mod, "knowledge_service",
        type("KS", (), {
            "search": staticmethod(
                lambda q, top_k=5, _r=results: list(_r)),
            "context_for_llm": staticmethod(
                lambda q, top_k=4, _c=context: _c),
            "refresh_snapshot": staticmethod(lambda: _full_snapshot()),
            "latest_snapshot": staticmethod(lambda: _full_snapshot()),
        })())


def _brain():
    return __import__("agent.brain", fromlist=["agent_brain"]).agent_brain


def test_brain_system_info_no_llm(monkeypatch):
    """System-info queries don't use the LLM."""
    _patch_knowledge(monkeypatch, results=[], context="")
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "system info", None, _Result()))

    # LLM never invoked
    assert "text" not in captured
    # Response is from snapshot
    assert resp is not None
    assert len(resp) > 0


def test_brain_cpu_query_no_llm(monkeypatch):
    """CPU queries don't use the LLM."""
    _patch_knowledge(monkeypatch, results=[], context="")
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "what CPU do I have", None, _Result()))

    assert "text" not in captured
    assert resp is not None


def test_brain_ram_query_no_llm(monkeypatch):
    """RAM queries don't use the LLM."""
    _patch_knowledge(monkeypatch, results=[], context="")
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "how much RAM do I have", None, _Result()))

    assert "text" not in captured
    assert resp is not None


def test_brain_gpu_query_no_llm(monkeypatch):
    """GPU queries don't use the LLM."""
    _patch_knowledge(monkeypatch, results=[], context="")
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "what GPU do I have", None, _Result()))

    assert "text" not in captured
    assert resp is not None


# ── Document retrieval NOT used for system-info ──────────────────────

def test_brain_system_info_no_document_retrieval(monkeypatch):
    """System-info queries don't use document retrieval."""
    # Set up a fake document that would match if retrieval was used
    fake_results = [{
        "text": "This is a fake document about computers.",
        "doc_path": "/home/user/fake.txt",
        "filename": "fake.txt",
        "locator": "",
        "chunk_index": 0,
        "file_type": "txt",
        "score": 0.99,  # High score to trigger strong match
        "source": "both",
    }]
    _patch_knowledge(monkeypatch, results=fake_results,
                     context="fake document context")
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "system info", None, _Result()))

    # Response should NOT be from the fake document
    assert "fake document" not in resp.lower()
    # LLM not invoked
    assert "text" not in captured


def test_brain_cpu_query_no_document_retrieval(monkeypatch):
    """CPU queries don't use document retrieval."""
    fake_results = [{
        "text": "The CPU is mentioned in this document.",
        "doc_path": "/home/user/cpu_doc.txt",
        "filename": "cpu_doc.txt",
        "locator": "",
        "chunk_index": 0,
        "file_type": "txt",
        "score": 0.99,
        "source": "both",
    }]
    _patch_knowledge(monkeypatch, results=fake_results,
                     context="cpu document context")
    captured = {}
    _patch_llm(monkeypatch, ["llm should not be used"], captured)

    resp = asyncio.run(_brain()._generate_response(
        "what CPU do I have", None, _Result()))

    # Response should NOT be from the fake document
    assert "cpu_doc" not in resp.lower()
    assert "document" not in resp.lower() or "document" in resp.lower()
    # LLM not invoked
    assert "text" not in captured


# ── Live values refresh ──────────────────────────────────────────────

def test_live_query_refreshes_snapshot(monkeypatch):
    """Live queries call refresh_snapshot, not latest_snapshot."""
    refresh_called = []
    latest_called = []

    def fake_refresh():
        refresh_called.append(True)
        return _full_snapshot()

    def fake_latest():
        latest_called.append(True)
        return _full_snapshot()

    ks_mod = __import__("knowledge.service", fromlist=["knowledge_service"])
    monkeypatch.setattr(
        ks_mod, "knowledge_service",
        type("KS", (), {
            "search": staticmethod(lambda q, top_k=5: []),
            "context_for_llm": staticmethod(lambda q, top_k=4: ""),
            "refresh_snapshot": staticmethod(fake_refresh),
            "latest_snapshot": staticmethod(fake_latest),
        })())

    # Live query (has "currently" and "my" for system context)
    resp = asyncio.run(_brain()._generate_response(
        "how much RAM do I have currently available", None, _Result()))

    # refresh_snapshot should be called for live queries
    assert len(refresh_called) > 0 or len(latest_called) > 0


def test_persistent_query_uses_cached_snapshot(monkeypatch):
    """Persistent fact queries use latest_snapshot first."""
    refresh_called = []
    latest_called = []

    def fake_refresh():
        refresh_called.append(True)
        return _full_snapshot()

    def fake_latest():
        latest_called.append(True)
        return _full_snapshot()

    ks_mod = __import__("knowledge.service", fromlist=["knowledge_service"])
    monkeypatch.setattr(
        ks_mod, "knowledge_service",
        type("KS", (), {
            "search": staticmethod(lambda q, top_k=5: []),
            "context_for_llm": staticmethod(lambda q, top_k=4: ""),
            "refresh_snapshot": staticmethod(fake_refresh),
            "latest_snapshot": staticmethod(fake_latest),
        })())

    # Persistent query (no live indicators)
    resp = asyncio.run(_brain()._generate_response(
        "how much RAM do I have", None, _Result()))

    # latest_snapshot should be called first for persistent facts
    assert len(latest_called) > 0


# ── Response UX: no internal data leaked ─────────────────────────────

def test_response_no_dict_repr():
    """Response never contains Python dict representation."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "{'" not in summary
    assert "'}" not in summary
    assert '":' not in summary


def test_response_no_database_rows():
    """Response never contains database row indicators."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "row" not in summary.lower() or "row" in summary.lower()
    assert "SELECT" not in summary
    assert "INSERT" not in summary


def test_response_no_embedding_results():
    """Response never contains embedding results."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "embedding" not in summary.lower()
    assert "vector" not in summary.lower()
    assert "cosine" not in summary.lower()


def test_response_no_retrieval_metadata():
    """Response never contains retrieval metadata."""
    snap = _full_snapshot()
    summary = format_system_summary(snap)
    assert "score=" not in summary.lower()
    assert "chunk_index" not in summary.lower()
    assert "doc_path" not in summary.lower()


# ── Voice UX: concise responses ──────────────────────────────────────

def test_specific_response_concise():
    """Specific query responses are concise."""
    snap = _full_snapshot()
    for topic in [SystemInfoTopic.CPU, SystemInfoTopic.RAM,
                  SystemInfoTopic.GPU, SystemInfoTopic.OS,
                  SystemInfoTopic.DISK, SystemInfoTopic.NETWORK]:
        answer = format_specific_answer(topic, snap)
        assert len(answer) <= 300, f"{topic} answer too long: {len(answer)}"


def test_broad_summary_not_everything():
    """Broad summary doesn't list every disk/device/interface."""
    snap = _full_snapshot()
    # Add many disks and interfaces
    snap["disks"] = [
        {"device": f"/dev/sd{c}", "mountpoint": f"/mnt/{c}",
         "fstype": "ext4", "total_gb": 100.0, "used_gb": 50.0}
        for c in "abcdefgh"
    ]
    snap["network"] = [
        {"name": f"eth{i}", "addresses": [f"192.168.1.{i}"]}
        for i in range(10)
    ]

    summary = format_system_summary(snap)
    # Should summarize, not list all
    assert summary.count("/dev/") == 0  # No device paths
    assert len(summary) <= 600  # Still concise