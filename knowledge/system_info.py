"""
System-information query handling for Diego.

This module provides deterministic detection and response generation for
system-information queries such as:
  - "system info" / "system information"
  - "PC specs" / "computer specifications"
  - "what CPU do I have?"
  - "how much RAM do I have?"
  - "what GPU do I have?"
  - "which OS am I running?"
  - "what disk space do I have?"
  - "what network interfaces do I have?"

Routing priority (per task requirements):
    LIVE STATE request
        ↓
    SYSTEM INFO / MACHINE FACT request  ← this module
        ↓
    LOCAL DOCUMENT KNOWLEDGE
        ↓
    NORMAL LLM/tool path

The authoritative source is the existing read-only PC snapshot collector
(knowledge/snapshot.py). Document retrieval is NEVER used for basic
machine facts.

Response UX contract:
  * Never dump raw JSON, Python dicts, psutil objects, DB rows,
    embedding results, file paths, or retrieval metadata.
  * Broad queries return a concise structured summary.
  * Specific queries return only the requested fact(s).
  * Voice responses are concise.
  * Partial failures return available fields without failing the
    whole request.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SystemInfoTopic(str, Enum):
    """Which system-information topic a query targets."""
    BROAD = "broad"           # Full system summary
    OS = "os"                 # Operating system
    CPU = "cpu"               # Processor
    RAM = "ram"               # Memory
    GPU = "gpu"               # Graphics
    DISK = "disk"             # Storage/disks
    NETWORK = "network"       # Network interfaces
    PYTHON = "python"         # Python environment
    NONE = "none"             # Not a system-info query


@dataclass
class SystemInfoQuery:
    """Parsed system-info query."""
    topic: SystemInfoTopic
    is_live: bool = False     # True if volatile data needs refresh
    raw_text: str = ""


# ── Query detection ──────────────────────────────────────────────────

# Broad system-info phrases (full summary)
_BROAD_PHRASES = (
    "system info", "system information", "system spec", "system specs",
    "pc spec", "pc specs", "pc info", "pc information",
    "computer spec", "computer specs", "computer information",
    "computer info", "machine spec", "machine specs", "machine info",
    "hardware info", "hardware information", "hardware specs",
    "tell me about my computer", "tell me about my pc",
    "tell me about my machine", "tell me about my system",
    "show my pc specs", "show my computer specs", "show system info",
    "show system information", "show me my specs",
    "what are my computer specifications",
    "what are my pc specifications",
    "what are my system specifications",
    "what are my specs", "what is my system",
    "my computer specs", "my pc specs", "my system specs",
    "about my computer", "about my pc", "about my system",
    "describe my computer", "describe my pc", "describe my system",
    "system summary", "computer summary", "pc summary",
    "hardware summary", "machine summary",
)

# Topic-specific patterns (regex for flexibility)
_TOPIC_PATTERNS: Dict[SystemInfoTopic, List[str]] = {
    SystemInfoTopic.OS: [
        r"\b(os|operating system)\b",
        r"which os", r"what os", r"what.*operating system",
        r"what.*distro", r"which.*distro", r"what.*linux",
        r"what.*windows", r"what.*version.*os",
        r"am i running (linux|windows|mac)",
        r"is this (linux|windows|mac)",
    ],
    SystemInfoTopic.CPU: [
        r"\bcpu\b", r"\bprocessor\b", r"\bcore(s)?\b",
        r"what cpu", r"which cpu", r"what processor",
        r"what.*cpu.*do i have", r"what.*processor.*do i have",
        r"how many cores", r"cpu model", r"cpu speed",
        r"intel|amd|ryzen|core i\d",
    ],
    SystemInfoTopic.RAM: [
        r"\bram\b", r"\bmemory\b", r"\bmem\b",
        r"how much ram", r"how much memory",
        r"what ram", r"ram size", r"memory size",
        r"ram.*do i have", r"memory.*do i have",
        r"total ram", r"total memory", r"available ram",
        r"free ram", r"free memory",
        r"ram.*available", r"memory.*available",
        r"current.*memory", r"memory.*usage", r"ram.*usage",
    ],
    SystemInfoTopic.GPU: [
        r"\bgpu\b", r"\bgraphics\b", r"\bvideo card\b",
        r"\bgraphics card\b", r"\bdisplay adapter\b",
        r"what gpu", r"which gpu", r"what graphics",
        r"what.*gpu.*do i have", r"what.*graphics card",
        r"nvidia|amd radeon|intel.*graphics",
        r"gpu.*do i have", r"graphics.*do i have",
    ],
    SystemInfoTopic.DISK: [
        r"\bdisk(s)?\b", r"\bstorage\b", r"\bdrive(s)?\b", r"\bhdd\b",
        r"\bssd\b", r"\bhard drive\b", r"\bfile ?system\b",
        r"disk space", r"storage space", r"free space",
        r"how much (disk|storage|space)",
        r"what disk", r"which disk", r"disk.*do i have",
        r"storage.*do i have", r"disk usage", r"disk capacity",
        r"total (disk|storage|space)",
        r"about my disk", r"about my disks", r"tell me about.*disk",
    ],
    SystemInfoTopic.NETWORK: [
        r"\bnetwork\b", r"\bnic\b", r"\binterface(s)?\b",
        r"\bip address\b", r"\bmac address\b", r"\bethernet\b",
        r"\bwifi\b", r"\bwi-?fi\b", r"\bwireless\b",
        r"network interface", r"what network", r"which network",
        r"network.*do i have", r"interface.*do i have",
        r"what.*ip.*address", r"my ip",
    ],
    SystemInfoTopic.PYTHON: [
        r"\bpython\b.*\b(version|env|environment)\b",
        r"what python", r"which python", r"python version",
        r"python.*do i have", r"what.*venv", r"virtual environment",
    ],
}

# Live-state indicators (volatile data that needs refresh)
_LIVE_INDICATORS = (
    "current", "right now", "now", "currently", "at the moment",
    "running", "active", "in use", "being used", "usage",
    "free", "available", "used",
)

# Words that indicate a system-info context (vs. general questions)
_SYSTEM_CONTEXT_WORDS = {
    "my", "i", "have", "am", "running", "installed", "using",
    "computer", "pc", "machine", "system", "laptop", "desktop",
    "total", "available", "free", "current", "usage", "much",
}


def _normalize(text: str) -> str:
    """Normalize query text for matching."""
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s'-]", " ", t)  # Remove punctuation except apostrophes/hyphens
    t = " ".join(t.split())
    return t


def _has_system_context(text: str) -> bool:
    """Check if the query has system/computer context."""
    words = set(text.split())
    return bool(words & _SYSTEM_CONTEXT_WORDS)


def detect_system_info_query(text: str) -> SystemInfoQuery:
    """
    Detect if a query is asking for system information.

    Returns a SystemInfoQuery with the detected topic, or topic=NONE
    if this is not a system-info query.

    Detection priority:
      1. Broad phrases (full system summary)
      2. Topic-specific patterns (CPU, RAM, GPU, etc.)
      3. Context-aware fallback (must have system context words)
    """
    norm = _normalize(text)
    if not norm:
        return SystemInfoQuery(topic=SystemInfoTopic.NONE, raw_text=text)

    # Check for live-state indicators
    is_live = any(ind in norm for ind in _LIVE_INDICATORS)

    # 1. Broad phrases — exact substring match
    for phrase in _BROAD_PHRASES:
        if phrase in norm:
            return SystemInfoQuery(
                topic=SystemInfoTopic.BROAD,
                is_live=is_live,
                raw_text=text,
            )

    # 2. Topic-specific patterns
    for topic, patterns in _TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, norm, re.IGNORECASE):
                # Require system context for ambiguous single-word matches
                # like "memory" (could be about human memory) or "python"
                # (could be about the snake or programming language)
                if topic in (SystemInfoTopic.RAM, SystemInfoTopic.PYTHON):
                    if not _has_system_context(norm):
                        continue
                return SystemInfoQuery(
                    topic=topic,
                    is_live=is_live,
                    raw_text=text,
                )

    return SystemInfoQuery(topic=SystemInfoTopic.NONE, raw_text=text)


def is_system_info_query(text: str) -> bool:
    """Convenience: True if the text is a system-info query."""
    return detect_system_info_query(text).topic != SystemInfoTopic.NONE


# ── Response formatting ──────────────────────────────────────────────

def _format_os(os_info: Dict) -> str:
    """Format OS info for speech."""
    if not os_info or "error" in os_info:
        return ""
    system = os_info.get("system", "")
    release = os_info.get("release", "")
    machine = os_info.get("machine", "")
    hostname = os_info.get("hostname", "")

    parts = []
    if system:
        parts.append(system)
    if release:
        parts.append(release)
    if machine:
        parts.append(f"({machine})")

    result = " ".join(parts) if parts else ""
    if hostname and result:
        result = f"{result} on {hostname}"
    return result


def _format_cpu(cpu_info: Dict) -> str:
    """Format CPU info for speech."""
    if not cpu_info or "error" in cpu_info:
        return ""
    model = cpu_info.get("model", "")
    physical = cpu_info.get("cores_physical")
    logical = cpu_info.get("cores_logical")

    if not model or model == "unknown":
        model = "an unknown CPU"

    if physical and logical:
        return f"{model} ({physical} physical / {logical} logical cores)"
    if physical:
        return f"{model} ({physical} cores)"
    return model


def _format_ram(mem_info: Dict, include_available: bool = False) -> str:
    """Format RAM info for speech."""
    if not mem_info or "error" in mem_info:
        return ""
    total = mem_info.get("total_gb")
    available = mem_info.get("available_gb")

    if not total:
        return ""

    result = f"{total} GB of RAM"
    if include_available and available:
        result += f" ({available} GB available)"
    return result


def _format_gpu(gpu_info: List) -> str:
    """Format GPU info for speech."""
    if not gpu_info:
        return ""
    if isinstance(gpu_info, dict) and "error" in gpu_info:
        return ""
    if not isinstance(gpu_info, list):
        return ""

    # Filter out "unknown" entries
    gpus = [g for g in gpu_info if g and g != "unknown"]
    if not gpus:
        return ""

    if len(gpus) == 1:
        return gpus[0]
    return f"{len(gpus)} GPUs: " + ", ".join(gpus[:2])


def _format_disks(disk_info: List, include_usage: bool = False) -> str:
    """Format disk info for speech (concise)."""
    if not disk_info:
        return ""
    if isinstance(disk_info, dict) and "error" in disk_info:
        return ""
    if not isinstance(disk_info, list):
        return ""

    valid = [d for d in disk_info if isinstance(d, dict) and "error" not in d]
    if not valid:
        return ""

    # Summarize: total storage across all disks
    total_gb = sum(d.get("total_gb", 0) or 0 for d in valid)
    used_gb = sum(d.get("used_gb", 0) or 0 for d in valid)

    if total_gb <= 0:
        return ""

    # Round to sensible precision
    if total_gb >= 1000:
        total_str = f"{total_gb / 1000:.1f} TB"
    else:
        total_str = f"{total_gb:.0f} GB"

    if len(valid) == 1:
        result = f"{total_str} of storage"
    else:
        result = f"{total_str} of storage across {len(valid)} drives"

    if include_usage and used_gb > 0:
        free_gb = total_gb - used_gb
        if free_gb >= 1000:
            free_str = f"{free_gb / 1000:.1f} TB"
        else:
            free_str = f"{free_gb:.0f} GB"
        result += f" ({free_str} free)"

    return result


def _format_network(net_info: List) -> str:
    """Format network info for speech (concise)."""
    if not net_info:
        return ""
    if isinstance(net_info, dict) and "error" in net_info:
        return ""
    if not isinstance(net_info, list):
        return ""

    valid = [n for n in net_info if isinstance(n, dict) and n.get("name")]
    if not valid:
        return ""

    # List interface names only (concise for voice)
    names = [n["name"] for n in valid[:4]]  # Max 4 for voice UX
    if len(valid) > 4:
        return f"{len(valid)} network interfaces including " + ", ".join(names)
    return f"{len(valid)} network interfaces: " + ", ".join(names)


def _format_python(py_info: Dict) -> str:
    """Format Python environment info for speech."""
    if not py_info or "error" in py_info:
        return ""
    version = py_info.get("version", "")
    if not version:
        return ""
    return f"Python {version}"


def format_system_summary(snapshot: Dict) -> str:
    """
    Format a full system summary for broad queries.

    Returns a concise, voice-friendly summary. Never dumps raw JSON,
    dicts, or internal structures.
    """
    if not snapshot:
        return "I couldn't read your system information right now."

    parts = []
    errors = []

    # OS
    os_str = _format_os(snapshot.get("os", {}))
    if os_str:
        parts.append(f"Your system is running {os_str}")
    elif "error" in snapshot.get("os", {}):
        errors.append("OS information")

    # CPU
    cpu_str = _format_cpu(snapshot.get("cpu", {}))
    if cpu_str:
        parts.append(f"with {cpu_str}")
    elif "error" in snapshot.get("cpu", {}):
        errors.append("CPU information")

    # RAM
    ram_str = _format_ram(snapshot.get("memory", {}))
    if ram_str:
        parts.append(f"You have {ram_str}")
    elif "error" in snapshot.get("memory", {}):
        errors.append("RAM information")

    # GPU
    gpu_str = _format_gpu(snapshot.get("gpu", []))
    if gpu_str:
        parts.append(f"and a {gpu_str}")
    elif "error" in snapshot.get("gpu", {}):
        errors.append("GPU information")

    # Storage
    disk_str = _format_disks(snapshot.get("disks", []))
    if disk_str:
        parts.append(f"You have {disk_str}")
    elif "error" in snapshot.get("disks", {}):
        errors.append("disk information")

    # Network (brief mention only)
    net_str = _format_network(snapshot.get("network", []))
    if net_str:
        parts.append(f"and {net_str}")

    # Python environment (brief)
    py_str = _format_python(snapshot.get("python_envs", {}))
    if py_str:
        parts.append(f"running {py_str}")

    if not parts:
        return "I couldn't read your system information right now."

    # Join into natural sentences
    # First part is the main sentence, others are continuations
    result = parts[0]
    for i, part in enumerate(parts[1:], 1):
        if part.startswith(("with ", "and ", "running ")):
            result += f" {part}"
        else:
            result += f". {part}"

    # Add error notes (optional, non-fatal)
    if errors:
        result += f". I couldn't read your {', '.join(errors[:2])} right now."

    return result.strip()


def format_specific_answer(topic: SystemInfoTopic, snapshot: Dict,
                           is_live: bool = False) -> str:
    """
    Format an answer for a specific system-info query.

    Returns only the requested fact(s), not the entire snapshot.
    """
    if not snapshot:
        return f"I couldn't read that information right now."

    if topic == SystemInfoTopic.OS:
        os_str = _format_os(snapshot.get("os", {}))
        if os_str:
            return f"You're running {os_str}."
        return "I couldn't read your OS information right now."

    if topic == SystemInfoTopic.CPU:
        cpu_str = _format_cpu(snapshot.get("cpu", {}))
        if cpu_str:
            return f"You have {cpu_str}."
        return "I couldn't read your CPU information right now."

    if topic == SystemInfoTopic.RAM:
        ram_str = _format_ram(snapshot.get("memory", {}),
                              include_available=is_live)
        if ram_str:
            return f"You have {ram_str}."
        return "I couldn't read your RAM information right now."

    if topic == SystemInfoTopic.GPU:
        gpu_str = _format_gpu(snapshot.get("gpu", []))
        if gpu_str:
            return f"You have a {gpu_str}."
        return "I couldn't read your GPU information right now."

    if topic == SystemInfoTopic.DISK:
        disk_str = _format_disks(snapshot.get("disks", []),
                                 include_usage=is_live)
        if disk_str:
            return f"You have {disk_str}."
        return "I couldn't read your disk information right now."

    if topic == SystemInfoTopic.NETWORK:
        net_str = _format_network(snapshot.get("network", []))
        if net_str:
            return f"You have {net_str}."
        return "I couldn't read your network information right now."

    if topic == SystemInfoTopic.PYTHON:
        py_str = _format_python(snapshot.get("python_envs", {}))
        if py_str:
            return f"You're running {py_str}."
        return "I couldn't read your Python environment right now."

    return "I couldn't read that information right now."


# ── Main entry point ─────────────────────────────────────────────────

def answer_system_info_query(text: str,
                             snapshot_provider=None) -> Optional[str]:
    """
    Answer a system-info query using the snapshot collector.

    Args:
        text: The user's query.
        snapshot_provider: Optional callable that returns a snapshot dict.
            Defaults to knowledge_service.refresh_snapshot() for live
            queries or latest_snapshot() for persistent facts.

    Returns:
        A voice-friendly response string, or None if this is not a
        system-info query.
    """
    query = detect_system_info_query(text)
    if query.topic == SystemInfoTopic.NONE:
        return None

    # Get the snapshot
    if snapshot_provider is not None:
        snapshot = snapshot_provider()
    else:
        snapshot = None
        try:
            from knowledge.service import knowledge_service
            if query.is_live:
                # Live/volatile data: refresh the snapshot
                snapshot = knowledge_service.refresh_snapshot()
            else:
                # Persistent facts: use cached snapshot if available
                try:
                    stored = knowledge_service.latest_snapshot()
                    if stored:
                        # latest_snapshot returns {"payload": ..., "created_at": ...}
                        # or the snapshot dict directly depending on the store
                        if isinstance(stored, dict) and "payload" in stored:
                            snapshot = stored.get("payload")
                        else:
                            snapshot = stored
                except Exception:
                    snapshot = None
                if not snapshot:
                    # No cached snapshot — collect a fresh one
                    snapshot = knowledge_service.refresh_snapshot()
        except Exception as e:
            logger.warning("[SYSTEM-INFO] snapshot retrieval failed: %s", e)
            # Last resort: collect directly from the snapshot module
            try:
                from knowledge.snapshot import collect_snapshot
                snapshot = collect_snapshot()
            except Exception:
                snapshot = None

    if not snapshot:
        return "I couldn't read your system information right now."

    # Generate response
    if query.topic == SystemInfoTopic.BROAD:
        return format_system_summary(snapshot)
    return format_specific_answer(query.topic, snapshot, query.is_live)