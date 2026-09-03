"""Reproduce 'resume' routing through the full process_command pipeline."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

from telemetry.logger import setup_logging
setup_logging()


async def main() -> None:
    from agent.brain import agent_brain
    await agent_brain.initialize()
    r = await agent_brain.process_command("resume")
    print("\nRESULT path=", r.path, "used_llm=", r.used_llm,
          "actions=", r.actions_executed, "ok=", r.actions_succeeded,
          "failed=", r.actions_failed, "verified=", r.verified)
    print("RESPONSE:", r.response[:200])


if __name__ == "__main__":
    asyncio.run(main())