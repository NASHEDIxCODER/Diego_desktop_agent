"""
Full live pipeline run — drives agent_brain.process_command() (the real
production entry point used after STT) for every live voice-session
utterance, with real action execution, verification, and latency.

Usage:
    python debug/live_pipeline_run.py [--safe]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

from telemetry.logger import setup_logging
setup_logging()

import logging
logging.getLogger().setLevel(logging.INFO)

SAFE = "--safe" in sys.argv

COMMANDS = [
    "system info",
    "how much RAM do I have?",
    "what is on my screen?",
    "find my Diego project",
    "what do you know about my project?",
    "what project was I using recently?",
    "tell me a joke",
    # actions (skipped with --safe)
    ("open Firefox", False),
    ("play something on YouTube", False),
    ("pause", False),
    ("resume", False),
]


async def main() -> None:
    from agent.brain import agent_brain
    await agent_brain.initialize()

    rows = []
    for item in COMMANDS:
        text, do_exec = (item if isinstance(item, tuple) else (item, True))
        if SAFE and not do_exec:
            print(f"\n=== SKIP (safe): {text}")
            continue
        print("\n" + "=" * 72, flush=True)
        print(f"TRANSCRIPT : {text}", flush=True)
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                agent_brain.process_command(text), timeout=120.0)
        except asyncio.TimeoutError:
            print("  X TIMED OUT after 120s", flush=True)
            rows.append({"utterance": text, "timeout": True})
            continue
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"  path      : {getattr(result, 'path', None)}", flush=True)
        print(f"  used_llm  : {getattr(result, 'used_llm', None)}", flush=True)
        print(f"  actions   : {getattr(result, 'actions_executed', '?')} "
              f"(ok={getattr(result, 'actions_succeeded', '?')}, "
              f"failed={getattr(result, 'actions_failed', '?')})", flush=True)
        print(f"  verified  : {getattr(result, 'verified', None)}", flush=True)
        print(f"  brain_ms  : {getattr(result, 'latency_ms', -1):.0f} ms", flush=True)
        print(f"  wall_ms   : {total_ms:.0f} ms", flush=True)
        resp = getattr(result, "response", "") or ""
        print(f"  RESPONSE  : {resp[:400]}", flush=True)
        rows.append({
            "utterance": text,
            "path": str(getattr(result, "path", None)),
            "used_llm": getattr(result, "used_llm", None),
            "actions_executed": getattr(result, "actions_executed", None),
            "actions_succeeded": getattr(result, "actions_succeeded", None),
            "actions_failed": getattr(result, "actions_failed", None),
            "verified": getattr(result, "verified", None),
            "brain_ms": getattr(result, "latency_ms", None),
            "wall_ms": round(total_ms, 1),
            "response": resp[:600],
        })

    out = Path(PROJECT_ROOT) / "debug" / "live_pipeline_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())