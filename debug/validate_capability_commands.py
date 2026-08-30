"""
Live production-path validation for the 2026-08-30 hardening.

Drives agent_brain.process_command() — the SAME entry point the voice
pipeline uses after STT — for every validation command, and prints the
full trace: transcript -> routing path -> actions -> verification -> response.

Usage:
    python debug/validate_capability_commands.py
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

from telemetry.logger import setup_logging
setup_logging()

import logging
logging.getLogger().setLevel(logging.WARNING)


COMMANDS = [
    "what apps are running?",
    "switch window",
    "switch to Firefox",
    "open Firefox",
    "what is on my screen?",
    "search GitHub for Python websocket examples",
    "search the web for current information about Rust programming language",
    "search Rust programming language and open the best result",
    "tell me about animal DNA",
]


async def main() -> None:
    from agent.brain import agent_brain
    await agent_brain.initialize()

    for text in COMMANDS:
        print("\n" + "=" * 72, flush=True)
        print(f"TRANSCRIPT : {text}", flush=True)
        try:
            result = await asyncio.wait_for(
                agent_brain.process_command(text), timeout=90.0)
        except asyncio.TimeoutError:
            print("  X TIMED OUT after 90s", flush=True)
            continue
        print(f"  path      : {result.path}", flush=True)
        print(f"  used_llm  : {result.used_llm}", flush=True)
        print(f"  actions   : {result.actions_executed} "
              f"(ok={result.actions_succeeded}, failed={result.actions_failed})",
              flush=True)
        print(f"  verified  : {result.verified}", flush=True)
        print(f"  latency   : {result.latency_ms:.0f} ms", flush=True)
        print(f"  RESPONSE  : {result.response[:400]}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("Validation complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())