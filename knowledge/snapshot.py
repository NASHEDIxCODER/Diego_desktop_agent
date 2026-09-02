"""
Read-only PC snapshot (system inventory) collector.

Collects structured, safe system facts:
  * OS / version, CPU, RAM, GPU, disks/filesystems
  * Network interfaces
  * Installed applications (safely discoverable only)
  * Python / virtual environments
  * Relevant Diego configuration
  * Running processes (names only — no command lines, no secrets)

STRICTLY READ-ONLY: uses psutil / platform / shutil queries only.
Never executes files from scanned directories, never writes user files.
Stored as structured local knowledge in the existing DuckDB.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def collect_snapshot() -> Dict:
    """Collect a structured PC inventory. Never raises."""
    snap: Dict = {}
    for name, fn in (
        ("os", _os_info),
        ("cpu", _cpu_info),
        ("memory", _memory_info),
        ("gpu", _gpu_info),
        ("disks", _disk_info),
        ("network", _network_info),
        ("installed_apps", _installed_apps),
        ("python_envs", _python_envs),
        ("diego_config", _diego_config),
        ("processes", _processes),
    ):
        try:
            snap[name] = fn()
        except Exception as e:
            snap[name] = {"error": str(e)}
    return snap


def snapshot_text(snap: Dict) -> str:
    """Flatten a snapshot into compact searchable text chunks."""
    lines: List[str] = []
    os_i = snap.get("os", {})
    if isinstance(os_i, dict):
        lines.append(f"Operating system: {os_i.get('system', '')} "
                     f"{os_i.get('release', '')} "
                     f"({os_i.get('version', '')}) "
                     f"machine={os_i.get('machine', '')}")
    cpu = snap.get("cpu", {})
    if isinstance(cpu, dict):
        lines.append(f"CPU: {cpu.get('model', 'unknown')} "
                     f"cores={cpu.get('cores_physical', '?')} physical / "
                     f"{cpu.get('cores_logical', '?')} logical")
    mem = snap.get("memory", {})
    if isinstance(mem, dict) and mem.get("total_gb"):
        lines.append(f"RAM: {mem.get('total_gb')} GB total, "
                     f"{mem.get('available_gb')} GB available")
    gpu = snap.get("gpu", [])
    if isinstance(gpu, list):
        for g in gpu:
            lines.append(f"GPU: {g}")
    disks = snap.get("disks", [])
    if isinstance(disks, list):
        for d in disks:
            if isinstance(d, dict):
                lines.append(
                    f"Disk {d.get('device', '')} mounted at "
                    f"{d.get('mountpoint', '')} filesystem={d.get('fstype', '')} "
                    f"total={d.get('total_gb', '?')}GB "
                    f"used={d.get('used_gb', '?')}GB")
    nets = snap.get("network", [])
    if isinstance(nets, list):
        for n in nets:
            lines.append(f"Network interface {n.get('name', '')}: "
                         f"{', '.join(n.get('addresses', []))}")
    apps = snap.get("installed_apps", [])
    if isinstance(apps, list) and apps:
        lines.append("Installed applications: " + ", ".join(apps[:200]))
    py = snap.get("python_envs", {})
    if isinstance(py, dict):
        lines.append(f"Python: {py.get('version', '')} at "
                     f"{py.get('executable', '')}")
        for venv in py.get("venvs", []):
            lines.append(f"Virtual environment: {venv}")
    proc = snap.get("processes", [])
    if isinstance(proc, list):
        lines.append("Running processes: " + ", ".join(proc[:150]))
    return "\n".join(lines)


# ── Collectors ───────────────────────────────────────────────────

def _os_info() -> Dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "python": platform.python_version(),
    }


def _cpu_info() -> Dict:
    model = ""
    try:
        # Linux: read /proc/cpuinfo (read-only)
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        model = platform.processor() or "unknown"
    import psutil
    return {
        "model": model,
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
    }


def _memory_info() -> Dict:
    import psutil
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024 ** 3), 1),
        "available_gb": round(vm.available / (1024 ** 3), 1),
    }


def _gpu_info() -> List[str]:
    gpus: List[str] = []
    try:
        # NVIDIA via nvidia-smi binary (read-only query, no file exec
        # from scanned dirs — a fixed system tool).
        import subprocess
        if shutil.which("nvidia-smi"):
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            for line in out.stdout.strip().splitlines():
                if line.strip():
                    gpus.append(line.strip())
    except Exception:
        pass
    if not gpus:
        try:
            # lspci fallback (read-only)
            if shutil.which("lspci"):
                import subprocess
                out = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=5)
                for line in out.stdout.splitlines():
                    low = line.lower()
                    if ("vga" in low or "3d controller" in low
                            or "display controller" in low):
                        gpus.append(line.split(":", 2)[-1].strip())
        except Exception:
            pass
    return gpus or ["unknown"]


def _disk_info() -> List[Dict]:
    import psutil
    out = []
    for part in psutil.disk_partitions(all=False):
        if any(part.mountpoint.startswith(p)
               for p in ("/proc", "/sys", "/dev", "/run", "/snap")):
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
            out.append({
                "device": part.device,
                "mountpoint": part.mountpoint,
                "fstype": part.fstype,
                "total_gb": round(usage.total / (1024 ** 3), 1),
                "used_gb": round(usage.used / (1024 ** 3), 1),
            })
        except (PermissionError, OSError):
            continue
    return out


def _network_info() -> List[Dict]:
    import psutil
    out = []
    addrs = psutil.net_if_addrs()
    import socket
    for name, addr_list in addrs.items():
        addresses = []
        for a in addr_list:
            try:
                if a.family == socket.AF_INET:
                    addresses.append(a.address)
                elif a.family == socket.AF_INET6:
                    addresses.append(a.address.split("%")[0])
            except Exception:
                continue
        if addresses:
            out.append({"name": name, "addresses": addresses})
    return out


def _installed_apps() -> List[str]:
    """Safely discoverable applications (PATH executables + .desktop)."""
    apps: set = set()
    # PATH executables (safe discovery — names only)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            if not d or not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                p = os.path.join(d, name)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    apps.add(name)
        except OSError:
            continue
    # .desktop entries (Linux app launchers)
    for d in ("/usr/share/applications",
              os.path.expanduser("~/.local/share/applications")):
        try:
            if os.path.isdir(d):
                for name in os.listdir(d):
                    if name.endswith(".desktop"):
                        apps.add(name[:-8])
        except OSError:
            continue
    return sorted(apps)[:500]


def _python_envs() -> Dict:
    envs: Dict = {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "venvs": [],
    }
    # Discover venvs in approved project locations (names only)
    home = os.path.expanduser("~")
    for base in ("~/Projects", "~/Documents", "~/Desktop"):
        root = os.path.expanduser(base)
        if not os.path.isdir(root):
            continue
        try:
            for dirpath, dirnames, _files in os.walk(root):
                dirpath = os.path.basename(dirpath)
                if dirpath in (".venv", "venv"):
                    envs["venvs"].append(dirpath)
                    dirnames[:] = []  # don't descend into venvs
                if dirpath in (".git", "node_modules", "__pycache__"):
                    dirnames[:] = []
                if len(envs["venvs"]) > 50:
                    break
        except OSError:
            continue
    return envs


def _diego_config() -> Dict:
    from config.settings import settings
    return {
        "model_name": settings.MODEL_NAME,
        "embedding_dim": settings.EMBEDDING_DIM,
        "duckdb_path": settings.DUCKDB_PATH,
        "knowledge_scan_roots": settings.KNOWLEDGE_SCAN_ROOTS,
        "knowledge_max_file_size": settings.KNOWLEDGE_MAX_FILE_SIZE,
        "llm_warmup_enabled": settings.LLM_WARMUP_ENABLED,
    }


def _processes() -> List[str]:
    """Running process NAMES only (no command lines — may hold secrets)."""
    import psutil
    names: set = set()
    for p in psutil.process_iter(["name"]):
        try:
            n = p.info.get("name")
            if n:
                names.add(n)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(names)[:300]