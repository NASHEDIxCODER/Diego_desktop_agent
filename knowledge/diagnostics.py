"""
Diego self-diagnostics — read-only health and latency analysis.

This module provides deterministic detection and response generation for
self-diagnostic queries such as:
  - "Why is Diego slow?"
  - "Is Diego healthy?"
  - "What's wrong with Diego?"
  - "Check your system health."
  - "Why is voice recognition slow?"
  - "Why is LLM response slow?"

Routing priority (per task requirements):
    LIVE STATE request
        ↓
    SYSTEM INFO / MACHINE FACT request
        ↓
    DIAGNOSTIC / HEALTH request  ← this module
        ↓
    LOCAL DOCUMENT KNOWLEDGE
        ↓
    NORMAL LLM/tool path

The diagnostic flow:
    User question
    → detect DIAGNOSTIC/HEALTH intent
    → gather relevant live read-only diagnostics
    → gather relevant local knowledge/log evidence
    → summarize evidence
    → use LLM only when reasoning is required
    → concise spoken response

STRICTLY READ-ONLY: Never executes repair actions. Never modifies state.
Never exposes raw logs, paths, JSON, stack traces, database rows, scores,
or internal retrieval metadata in speech.

Status classification:
  - healthy: all required subsystems operational
  - degraded: some subsystems using fallbacks or slower than expected
  - failed: required subsystem unavailable
  - unknown: diagnostic data unavailable
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DiagnosticTopic(str, Enum):
    """Which diagnostic topic a query targets."""
    HEALTH = "health"           # General health check
    SLOW = "slow"               # Why is Diego slow?
    VOICE_LATENCY = "voice"     # Voice/STT/speech recognition latency
    LLM_LATENCY = "llm"         # LLM response latency
    TTS_LATENCY = "tts"         # TTS/speech output latency
    VISION_LATENCY = "vision"   # Vision/OCR/perception latency
    RETRIEVAL_LATENCY = "retrieval"  # Knowledge retrieval latency
    FAILURE = "failure"         # What's wrong / what failed
    NONE = "none"               # Not a diagnostic query


class HealthStatus(str, Enum):
    """Overall health classification."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class SubsystemStatus:
    """Status of a single subsystem."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: Optional[float] = None
    evidence: str = ""
    fallback: str = ""


@dataclass
class DiagnosticQuery:
    """Parsed diagnostic query."""
    topic: DiagnosticTopic
    raw_text: str = ""


@dataclass
class DiagnosticReport:
    """Complete diagnostic report (internal, never spoken directly)."""
    overall: HealthStatus = HealthStatus.UNKNOWN
    subsystems: List[SubsystemStatus] = field(default_factory=list)
    bottlenecks: List[Tuple[str, float]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


# ── Query detection ──────────────────────────────────────────────────

# Health check phrases
_HEALTH_PHRASES = (
    "is diego healthy", "are you healthy", "how healthy are you",
    "check your health", "check my health", "check system health",
    "system health", "health check", "health status",
    "are you ok", "are you okay", "how are you doing",
    "is everything ok", "is everything okay", "is everything working",
    "is everything fine", "are all systems working",
    "status check", "system status", "diego status",
    "check yourself", "check your system", "check your systems",
    "run diagnostics", "run a diagnostic", "run diagnostic",
    "self diagnostic", "self diagnostics", "self check",
    "what is your status", "what's your status",
)

# Slowness/latency phrases (general)
_SLOW_PHRASES = (
    "why is diego slow", "why are you slow", "why so slow",
    "diego is slow", "you are slow", "you're slow", "youre slow",
    "why is this slow", "why is it slow", "why is everything slow",
    "what is making you slow", "what's making you slow",
    "why does it take so long", "why is it taking so long",
    "why the delay", "what is the delay", "what's the delay",
    "why is response slow", "why are responses slow",
    "why is diego lagging", "why are you lagging",
    "performance issue", "performance problem",
    "latency issue", "latency problem",
    "why is diego so slow", "why are you so slow",
)

# Voice/STT latency phrases
_VOICE_LATENCY_PHRASES = (
    "why is voice recognition slow", "why is speech recognition slow",
    "why is voice slow", "why is speech slow", "why is stt slow",
    "voice recognition is slow", "speech recognition is slow",
    "voice is slow", "speech is slow", "stt is slow",
    "why do you take so long to hear", "why do you take so long to listen",
    "why is transcription slow", "transcription is slow",
    "why is whisper slow", "whisper is slow",
    "voice latency", "speech latency", "stt latency",
    "why is my voice not recognized", "why is my speech not recognized",
    "why don't you hear me", "why can't you hear me",
    "why is voice detection slow", "voice detection is slow",
)

# LLM latency phrases
_LLM_LATENCY_PHRASES = (
    "why is llm slow", "why is the llm slow", "llm is slow",
    "why is llm response slow", "llm response is slow",
    "why is ai slow", "why is the ai slow", "ai is slow",
    "why is thinking slow", "why is your thinking slow",
    "why do you take so long to think", "why do you take so long to respond",
    "why is response generation slow", "response generation is slow",
    "llm latency", "ai latency", "thinking latency",
    "why is ollama slow", "ollama is slow",
    "why is model slow", "why is the model slow", "model is slow",
    "why is inference slow", "inference is slow",
)

# TTS latency phrases
_TTS_LATENCY_PHRASES = (
    "why is tts slow", "why is speech output slow", "tts is slow",
    "why is your voice slow", "why do you speak slowly",
    "why is talking slow", "why is speaking slow",
    "speech output is slow", "voice output is slow",
    "tts latency", "speech synthesis latency",
    "why is kokoro slow", "kokoro is slow",
)

# Vision/OCR latency phrases
_VISION_LATENCY_PHRASES = (
    "why is vision slow", "why is ocr slow", "why is screen reading slow",
    "vision is slow", "ocr is slow", "screen reading is slow",
    "why is perception slow", "perception is slow",
    "why does reading the screen take so long",
    "vision latency", "ocr latency", "perception latency",
)

# Retrieval latency phrases
_RETRIEVAL_LATENCY_PHRASES = (
    "why is search slow", "why is knowledge search slow",
    "why is retrieval slow", "why is lookup slow",
    "search is slow", "retrieval is slow", "lookup is slow",
    "why is finding information slow",
    "retrieval latency", "search latency",
)

# Failure/problem phrases
_FAILURE_PHRASES = (
    "what's wrong with diego", "whats wrong with diego",
    "what is wrong with diego", "what's wrong with you",
    "whats wrong with you", "what is wrong with you",
    "what's the problem", "whats the problem", "what is the problem",
    "what failed", "what has failed", "what is failing",
    "what's broken", "whats broken", "what is broken",
    "what's not working", "whats not working", "what is not working",
    "what error", "what errors", "any errors",
    "why isn't it working", "why is it not working",
    "why doesn't it work", "why did it fail",
    "diagnose the problem", "diagnose the issue",
    "what went wrong", "what happened",
)

# Subsystem context words for targeted diagnostics
_SUBSYSTEM_CONTEXT = {
    "voice": ("voice", "speech", "stt", "whisper", "transcription", "hear", "listen"),
    "llm": ("llm", "ai", "thinking", "think", "ollama", "model", "inference", "respond"),
    "tts": ("tts", "speak", "speaking", "talking", "voice output", "kokoro"),
    "vision": ("vision", "ocr", "screen", "perception", "see", "camera"),
    "retrieval": ("search", "retrieval", "lookup", "knowledge", "find"),
}


def _normalize(text: str) -> str:
    """Normalize query text for matching."""
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s'-]", " ", t)
    t = " ".join(t.split())
    return t


def detect_diagnostic_query(text: str) -> DiagnosticQuery:
    """
    Detect if a query is asking for self-diagnostics.

    Returns a DiagnosticQuery with the detected topic, or topic=NONE
    if this is not a diagnostic query.

    Detection priority:
      1. Specific latency topics (voice, LLM, TTS, vision, retrieval)
      2. General slowness
      3. Failure/problem queries
      4. General health checks
    """
    norm = _normalize(text)
    if not norm:
        return DiagnosticQuery(topic=DiagnosticTopic.NONE, raw_text=text)

    # 1. Specific latency topics (most specific first)
    for phrase in _VOICE_LATENCY_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.VOICE_LATENCY, raw_text=text)

    for phrase in _LLM_LATENCY_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.LLM_LATENCY, raw_text=text)

    for phrase in _TTS_LATENCY_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.TTS_LATENCY, raw_text=text)

    for phrase in _VISION_LATENCY_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.VISION_LATENCY, raw_text=text)

    for phrase in _RETRIEVAL_LATENCY_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.RETRIEVAL_LATENCY, raw_text=text)

    # 2. General slowness
    for phrase in _SLOW_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.SLOW, raw_text=text)

    # 3. Failure/problem queries
    for phrase in _FAILURE_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.FAILURE, raw_text=text)

    # 4. General health checks
    for phrase in _HEALTH_PHRASES:
        if phrase in norm:
            return DiagnosticQuery(topic=DiagnosticTopic.HEALTH, raw_text=text)

    return DiagnosticQuery(topic=DiagnosticTopic.NONE, raw_text=text)


def is_diagnostic_query(text: str) -> bool:
    """Convenience: True if the text is a diagnostic query."""
    return detect_diagnostic_query(text).topic != DiagnosticTopic.NONE


# ── Live diagnostic collectors (READ-ONLY) ───────────────────────────

def collect_runtime_health() -> List[SubsystemStatus]:
    """Collect component health from the existing runtime_health module."""
    subsystems: List[SubsystemStatus] = []
    try:
        from core.runtime_health import runtime_health, OK, DEGRADED, MISSING, REQUIRED

        # Run the health check (read-only import checks)
        components = runtime_health.run()

        for comp in components:
            if comp.status == OK:
                status = HealthStatus.HEALTHY
            elif comp.status == DEGRADED:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.FAILED if comp.component_class == REQUIRED else HealthStatus.DEGRADED

            subsystems.append(SubsystemStatus(
                name=comp.name,
                status=status,
                evidence=comp.reason,
                fallback=comp.fallback,
            ))
    except Exception as e:
        logger.debug("[DIAG] runtime_health collection failed: %s", e)
    return subsystems


def collect_latency_metrics() -> Dict[str, Optional[float]]:
    """Collect measured latencies from existing metrics/benchmark systems.

    Returns a dict of subsystem → average latency in ms (or None if unknown).
    NEVER guesses — only returns actually measured values.
    """
    latencies: Dict[str, Optional[float]] = {
        "stt": None,
        "llm": None,
        "tts": None,
        "vision": None,
        "retrieval": None,
        "total": None,
    }

    # From benchmark (turn-level latencies)
    try:
        from core.benchmark import benchmark
        if benchmark._total_turns > 0:
            latencies["total"] = benchmark.avg_latency_ms
            if benchmark.avg_stt_latency_ms > 0:
                latencies["stt"] = benchmark.avg_stt_latency_ms
            if benchmark.avg_llm_latency_ms > 0:
                latencies["llm"] = benchmark.avg_llm_latency_ms
            if benchmark.avg_tts_latency_ms > 0:
                latencies["tts"] = benchmark.avg_tts_latency_ms
    except Exception as e:
        logger.debug("[DIAG] benchmark metrics failed: %s", e)

    # From metrics registry (more granular)
    try:
        from core.metrics import metrics
        snap = metrics.snapshot()

        # STT/Whisper latency
        if "stt.latency_ms.avg" in snap and snap["stt.latency_ms.avg"]:
            latencies["stt"] = snap["stt.latency_ms.avg"]
        elif "whisper.latency_ms.avg" in snap and snap["whisper.latency_ms.avg"]:
            latencies["stt"] = snap["whisper.latency_ms.avg"]

        # LLM latency
        if "llm.latency_ms.avg" in snap and snap["llm.latency_ms.avg"]:
            latencies["llm"] = snap["llm.latency_ms.avg"]
        elif "llm.first_token_ms.avg" in snap and snap["llm.first_token_ms.avg"]:
            latencies["llm"] = snap["llm.first_token_ms.avg"]

        # TTS latency
        if "tts.latency_ms.avg" in snap and snap["tts.latency_ms.avg"]:
            latencies["tts"] = snap["tts.latency_ms.avg"]

        # Vision/perception latency
        if "perception.latency_ms.avg" in snap and snap["perception.latency_ms.avg"]:
            latencies["vision"] = snap["perception.latency_ms.avg"]
        elif "ocr.latency_ms.avg" in snap and snap["ocr.latency_ms.avg"]:
            latencies["vision"] = snap["ocr.latency_ms.avg"]

        # Retrieval latency
        if "retrieval.latency_ms.avg" in snap and snap["retrieval.latency_ms.avg"]:
            latencies["retrieval"] = snap["retrieval.latency_ms.avg"]
        elif "knowledge.search_ms.avg" in snap and snap["knowledge.search_ms.avg"]:
            latencies["retrieval"] = snap["knowledge.search_ms.avg"]

    except Exception as e:
        logger.debug("[DIAG] metrics registry failed: %s", e)

    return latencies


def collect_system_resources() -> Dict[str, Any]:
    """Collect current system resource usage (read-only)."""
    resources: Dict[str, Any] = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        resources["ram_percent"] = vm.percent
        resources["ram_available_gb"] = round(vm.available / (1024 ** 3), 1)
        resources["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except Exception as e:
        logger.debug("[DIAG] system resources failed: %s", e)
    return resources


def collect_llm_status() -> SubsystemStatus:
    """Check LLM (Ollama) availability (read-only HTTP check)."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return SubsystemStatus(
                name="llm",
                status=HealthStatus.HEALTHY,
                evidence=f"{len(models)} models available",
            )
        return SubsystemStatus(
            name="llm",
            status=HealthStatus.FAILED,
            evidence=f"server returned error {r.status_code}",
        )
    except Exception as e:
        return SubsystemStatus(
            name="llm",
            status=HealthStatus.FAILED,
            evidence="server unreachable",
        )


def gather_diagnostics(topic: DiagnosticTopic) -> DiagnosticReport:
    """
    Gather all relevant diagnostics for the given topic.

    STRICTLY READ-ONLY: only queries existing systems, never modifies.
    """
    report = DiagnosticReport()

    # Collect runtime health (component status)
    health_subsystems = collect_runtime_health()
    report.subsystems.extend(health_subsystems)

    # Collect latency metrics
    latencies = collect_latency_metrics()
    report.evidence["latencies"] = latencies

    # Collect system resources
    resources = collect_system_resources()
    report.evidence["resources"] = resources

    # Check LLM specifically
    llm_status = collect_llm_status()
    # Replace or add LLM status
    report.subsystems = [s for s in report.subsystems if s.name != "llm"]
    report.subsystems.append(llm_status)

    # Identify bottlenecks (sorted by latency, highest first)
    bottleneck_candidates = [
        (name, ms) for name, ms in latencies.items()
        if ms is not None and ms > 0
    ]
    report.bottlenecks = sorted(bottleneck_candidates, key=lambda x: x[1], reverse=True)

    # Determine overall status
    failed_required = [
        s for s in report.subsystems
        if s.status == HealthStatus.FAILED
    ]
    degraded = [
        s for s in report.subsystems
        if s.status == HealthStatus.DEGRADED
    ]

    if failed_required:
        report.overall = HealthStatus.FAILED
        report.errors = [f"{s.name}: {s.evidence}" for s in failed_required]
    elif degraded:
        report.overall = HealthStatus.DEGRADED
    elif report.subsystems:
        report.overall = HealthStatus.HEALTHY
    else:
        report.overall = HealthStatus.UNKNOWN

    return report


# ── Response formatting (voice-friendly, no raw data) ────────────────

# Latency thresholds for "slow" classification (ms)
LATENCY_THRESHOLDS = {
    "stt": 2000,      # STT should be < 2s
    "llm": 5000,      # LLM should be < 5s
    "tts": 1000,      # TTS should be < 1s
    "vision": 5000,   # Vision should be < 5s
    "retrieval": 1000,  # Retrieval should be < 1s
    "total": 8000,    # Total turn should be < 8s
}

# Human-friendly names for subsystems
_SUBSYSTEM_NAMES = {
    "stt": "speech recognition",
    "llm": "AI reasoning",
    "tts": "voice output",
    "vision": "screen reading",
    "retrieval": "information lookup",
    "total": "overall response",
    "microphone": "microphone",
    "vad": "voice detection",
    "wake": "wake word detection",
    "ocr": "screen reading",
    "screen_capture": "screen capture",
    "duckdb": "memory storage",
    "browser": "browser automation",
    "search": "web search",
}


def _friendly_name(name: str) -> str:
    """Convert subsystem name to human-friendly term."""
    return _SUBSYSTEM_NAMES.get(name.lower(), name.replace("_", " "))


def _format_latency(ms: float) -> str:
    """Format latency for speech."""
    if ms < 1000:
        return f"{ms:.0f} milliseconds"
    return f"{ms / 1000:.1f} seconds"


def format_health_response(report: DiagnosticReport) -> str:
    """Format a general health check response."""
    if report.overall == HealthStatus.HEALTHY:
        parts = ["All my systems are running normally."]
        # Mention any degraded optional components
        degraded = [s for s in report.subsystems if s.status == HealthStatus.DEGRADED]
        if degraded:
            names = ", ".join(_friendly_name(s.name) for s in degraded[:2])
            parts.append(f"Some features are using fallbacks: {names}.")
        return " ".join(parts)

    if report.overall == HealthStatus.DEGRADED:
        degraded = [s for s in report.subsystems if s.status == HealthStatus.DEGRADED]
        names = ", ".join(_friendly_name(s.name) for s in degraded[:3])
        return f"I'm running with reduced capability. {names} are using fallbacks."

    if report.overall == HealthStatus.FAILED:
        failed = [s for s in report.subsystems if s.status == HealthStatus.FAILED]
        names = ", ".join(_friendly_name(s.name) for s in failed[:2])
        return f"I have a problem. {names} are not available."

    return "I couldn't complete the health check right now."


def format_slow_response(report: DiagnosticReport) -> str:
    """Format a 'why is Diego slow' response with measured timings."""
    latencies = report.evidence.get("latencies", {})

    # Find slow subsystems based on measured latencies
    slow_parts = []
    for name, ms in report.bottlenecks[:3]:
        threshold = LATENCY_THRESHOLDS.get(name, 3000)
        if ms > threshold:
            friendly = _friendly_name(name)
            slow_parts.append(f"{friendly} is taking about {_format_latency(ms)}")

    if slow_parts:
        response = "Based on my measurements, " + ", ".join(slow_parts[:2]) + "."
        # Add resource context if relevant
        resources = report.evidence.get("resources", {})
        if resources.get("ram_percent", 0) > 90:
            response += " Your memory usage is also very high."
        elif resources.get("cpu_percent", 0) > 90:
            response += " Your CPU usage is also very high."
        return response

    # No measured bottlenecks — check if we have any data at all
    if not any(ms is not None and ms > 0 for ms in latencies.values()):
        return "I don't have enough performance data yet. Try using me for a few commands first."

    # Everything within thresholds
    total = latencies.get("total")
    if total and total > LATENCY_THRESHOLDS["total"]:
        return f"Your overall response time is about {_format_latency(total)}, which is slower than expected."

    return "My measured latencies look normal. The slowness might be from network or system load."


def format_voice_latency_response(report: DiagnosticReport) -> str:
    """Format a voice/STT latency diagnosis."""
    latencies = report.evidence.get("latencies", {})
    stt_ms = latencies.get("stt")

    if stt_ms is not None and stt_ms > 0:
        if stt_ms > LATENCY_THRESHOLDS["stt"]:
            return f"Speech recognition is taking about {_format_latency(stt_ms)}, which is slower than expected."
        return f"Speech recognition is taking about {_format_latency(stt_ms)}, which is within normal range."

    # Check if STT is even available
    stt_subsystem = next((s for s in report.subsystems if s.name == "stt"), None)
    if stt_subsystem and stt_subsystem.status == HealthStatus.FAILED:
        return "Speech recognition is not available right now."

    return "I don't have speech recognition timing data yet."


def format_llm_latency_response(report: DiagnosticReport) -> str:
    """Format an LLM latency diagnosis."""
    latencies = report.evidence.get("latencies", {})
    llm_ms = latencies.get("llm")

    # Check LLM availability first
    llm_subsystem = next((s for s in report.subsystems if s.name == "llm"), None)
    if llm_subsystem and llm_subsystem.status == HealthStatus.FAILED:
        return "The AI model server is not reachable. That's why responses are slow or failing."

    if llm_ms is not None and llm_ms > 0:
        if llm_ms > LATENCY_THRESHOLDS["llm"]:
            return f"AI reasoning is taking about {_format_latency(llm_ms)}, which is slower than expected."
        return f"AI reasoning is taking about {_format_latency(llm_ms)}, which is within normal range."

    return "I don't have AI reasoning timing data yet."


def format_tts_latency_response(report: DiagnosticReport) -> str:
    """Format a TTS latency diagnosis."""
    latencies = report.evidence.get("latencies", {})
    tts_ms = latencies.get("tts")

    if tts_ms is not None and tts_ms > 0:
        if tts_ms > LATENCY_THRESHOLDS["tts"]:
            return f"Voice output is taking about {_format_latency(tts_ms)}, which is slower than expected."
        return f"Voice output is taking about {_format_latency(tts_ms)}, which is within normal range."

    # Check TTS availability
    tts_subsystem = next((s for s in report.subsystems if s.name == "tts"), None)
    if tts_subsystem and tts_subsystem.status == HealthStatus.FAILED:
        return "Voice output is not available right now."
    if tts_subsystem and tts_subsystem.status == HealthStatus.DEGRADED:
        return "Voice output is using a fallback engine, which may be slower."

    return "I don't have voice output timing data yet."


def format_vision_latency_response(report: DiagnosticReport) -> str:
    """Format a vision/OCR latency diagnosis."""
    latencies = report.evidence.get("latencies", {})
    vision_ms = latencies.get("vision")

    if vision_ms is not None and vision_ms > 0:
        if vision_ms > LATENCY_THRESHOLDS["vision"]:
            return f"Screen reading is taking about {_format_latency(vision_ms)}, which is slower than expected."
        return f"Screen reading is taking about {_format_latency(vision_ms)}, which is within normal range."

    # Check OCR availability
    ocr_subsystem = next((s for s in report.subsystems if s.name == "ocr"), None)
    if ocr_subsystem and ocr_subsystem.status == HealthStatus.FAILED:
        return "Screen reading is not available right now."

    return "I don't have screen reading timing data yet."


def format_retrieval_latency_response(report: DiagnosticReport) -> str:
    """Format a retrieval latency diagnosis."""
    latencies = report.evidence.get("latencies", {})
    retrieval_ms = latencies.get("retrieval")

    if retrieval_ms is not None and retrieval_ms > 0:
        if retrieval_ms > LATENCY_THRESHOLDS["retrieval"]:
            return f"Information lookup is taking about {_format_latency(retrieval_ms)}, which is slower than expected."
        return f"Information lookup is taking about {_format_latency(retrieval_ms)}, which is within normal range."

    return "I don't have information lookup timing data yet."


def format_failure_response(report: DiagnosticReport) -> str:
    """Format a 'what's wrong' response."""
    failed = [s for s in report.subsystems if s.status == HealthStatus.FAILED]

    if not failed:
        degraded = [s for s in report.subsystems if s.status == HealthStatus.DEGRADED]
        if degraded:
            names = ", ".join(_friendly_name(s.name) for s in degraded[:2])
            return f"Nothing has failed, but {names} are running in fallback mode."
        return "I don't see any failures right now. Everything looks healthy."

    # Report the subsystem and evidence, not an invented explanation
    parts = []
    for s in failed[:2]:
        name = _friendly_name(s.name)
        if s.evidence:
            parts.append(f"{name} is unavailable ({s.evidence})")
        else:
            parts.append(f"{name} is unavailable")

    return "I found problems. " + ". ".join(parts) + "."


def format_diagnostic_response(topic: DiagnosticTopic, report: DiagnosticReport) -> str:
    """Format the appropriate response for the diagnostic topic."""
    if topic == DiagnosticTopic.HEALTH:
        return format_health_response(report)
    if topic == DiagnosticTopic.SLOW:
        return format_slow_response(report)
    if topic == DiagnosticTopic.VOICE_LATENCY:
        return format_voice_latency_response(report)
    if topic == DiagnosticTopic.LLM_LATENCY:
        return format_llm_latency_response(report)
    if topic == DiagnosticTopic.TTS_LATENCY:
        return format_tts_latency_response(report)
    if topic == DiagnosticTopic.VISION_LATENCY:
        return format_vision_latency_response(report)
    if topic == DiagnosticTopic.RETRIEVAL_LATENCY:
        return format_retrieval_latency_response(report)
    if topic == DiagnosticTopic.FAILURE:
        return format_failure_response(report)
    return "I couldn't run the diagnostic right now."


# ── Main entry point ─────────────────────────────────────────────────

def answer_diagnostic_query(text: str) -> Optional[str]:
    """
    Answer a diagnostic query using live read-only diagnostics.

    Args:
        text: The user's query.

    Returns:
        A voice-friendly response string, or None if this is not a
        diagnostic query.
    """
    query = detect_diagnostic_query(text)
    if query.topic == DiagnosticTopic.NONE:
        return None

    try:
        report = gather_diagnostics(query.topic)
        response = format_diagnostic_response(query.topic, report)

        logger.info(
            "[DIAG] Answered diagnostic: topic=%s overall=%s response_len=%d",
            query.topic.value, report.overall.value, len(response),
        )
        return response

    except Exception as e:
        logger.warning("[DIAG] diagnostic failed: %s", e)
        return "I couldn't complete the diagnostic check right now."