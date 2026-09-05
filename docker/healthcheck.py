#!/usr/bin/env python3
"""
Diego container healthcheck.

Reuses Diego's OWN runtime-health mechanism (core/runtime_health.py —
the same component model the engine prints at boot) plus process
liveness, and reports exactly one container status:

  READY        all REQUIRED components READY and the Diego process is
               alive (optional services like Ollama do not affect this)
  DEGRADED     Diego is operational WITH fallbacks (e.g. energy-based
               VAD, pyttsx3 TTS) — container stays healthy (exit 0)
  UNAVAILABLE  an OPTIONAL external service (Ollama) is unreachable —
               Diego keeps running with deterministic/local
               capabilities, so the container stays healthy (exit 0)
  FAILED       Diego process dead OR a REQUIRED component is
               MISSING/FAILED — container unhealthy (exit 1)

Exit codes follow the container healthcheck contract:
  0 = healthy, 1 = unhealthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/app")

PID_FILE = Path("/tmp/diego.pid")
STATUS_FILE = Path("/tmp/diego_health")


def diego_process_alive() -> tuple[bool, bool]:
    """(alive, is_python) from the entrypoint PID file.

    is_python is False while the entrypoint is still in the bootstrap
    phase (shell running, Python not yet exec'd).
    """
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return False, False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
            errors="replace")
    except Exception:
        return False, False
    is_python = "main.py" in cmdline or "Diego.py" in cmdline
    return True, is_python


def main() -> int:
    from core.runtime_health import (READY, DEGRADED, MISSING, FAILED,
                                     UNAVAILABLE, REQUIRED, runtime_health)

    alive, is_python = diego_process_alive()
    if not alive:
        print("[CONTAINER-HEALTH] status=FAILED reason=Diego process not "
              "running")
        STATUS_FILE.write_text("FAILED")
        return 1

    components = runtime_health.run()
    required = [c for c in components if c.component_class == REQUIRED]
    required_down = [c for c in required if c.status in (MISSING, FAILED)]
    required_degraded = [c for c in required if c.status == DEGRADED]
    optional_unavailable = [c for c in components
                            if c.status == UNAVAILABLE]

    if required_down:
        status = FAILED
    elif not is_python:
        # Entrypoint still bootstrapping; Diego Python not started yet.
        status = "STARTING"
    elif required_degraded:
        status = DEGRADED
    elif optional_unavailable:
        status = UNAVAILABLE
    else:
        status = READY

    for c in components:
        print("  " + c.to_line())
    detail = ""
    if required_down:
        detail = (" down: " + ", ".join(
            f"{c.name}({c.status})" for c in required_down))
    elif required_degraded:
        detail = (" degraded: " + ", ".join(
            c.name for c in required_degraded))
    elif optional_unavailable:
        detail = (" optional-unavailable: " + ", ".join(
            c.name for c in optional_unavailable))
    print(f"[CONTAINER-HEALTH] status={status}{detail}")

    STATUS_FILE.write_text(status)

    # FAILED is the only unhealthy state. DEGRADED / UNAVAILABLE are
    # documented Diego operating modes, NOT container failures.
    return 0 if status != FAILED else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never crash the probe silently
        print(f"[CONTAINER-HEALTH] status=FAILED reason=healthcheck "
              f"error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)