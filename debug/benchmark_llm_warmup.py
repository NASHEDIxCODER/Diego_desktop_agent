"""
LLM warm-up benchmark — measure first-request (cold) vs warm-request latency.

Compares:
  1. Cold request  — model NOT resident in Ollama (keep_alive=0 unload first)
  2. Warm request  — model resident (after warm_up / keep_alive window)

Usage:
    python debug/benchmark_llm_warmup.py
    python debug/benchmark_llm_warmup.py --model qwen2.5:1.5b

Requires a running Ollama server (config/settings.OLLAMA_BASE_URL).
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import compat  # noqa: F401

import httpx

from config.settings import settings


async def _generate(client: httpx.AsyncClient, model: str, keep_alive: str,
                    num_predict: int = 24) -> float:
    """Run one generate call; return wall-clock latency in ms."""
    t0 = time.time()
    resp = await client.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": "Say hello in one short sentence.",
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_predict": num_predict},
        },
    )
    resp.raise_for_status()
    return (time.time() - t0) * 1000


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM cold vs warm benchmark")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-selected by StreamingLLM)")
    args = parser.parse_args()

    from agent.streaming_llm import streaming_llm
    if not await streaming_llm.ensure_initialized():
        print(f"ERROR: Ollama not reachable at {settings.OLLAMA_BASE_URL}")
        return

    model = args.model or streaming_llm._model
    print(f"Model: {model}")
    print(f"Ollama: {settings.OLLAMA_BASE_URL}")
    print(f"Configured keep_alive: {settings.OLLAMA_KEEP_ALIVE}")
    print()

    async with httpx.AsyncClient(timeout=300.0) as client:
        # ── 1. COLD: force-unload the model, then request ──
        try:
            await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0,
                      "options": {"num_predict": 0}},
            )
        except Exception:
            pass
        await asyncio.sleep(1.0)

        cold_ms = await _generate(client, model, keep_alive=0)
        print(f"COLD first request : {cold_ms:8.0f} ms  (model was unloaded)")

        # ── 2. WARM: model now resident ──
        warm1_ms = await _generate(client, model, keep_alive=settings.OLLAMA_KEEP_ALIVE)
        warm2_ms = await _generate(client, model, keep_alive=settings.OLLAMA_KEEP_ALIVE)
        print(f"WARM request #1    : {warm1_ms:8.0f} ms")
        print(f"WARM request #2    : {warm2_ms:8.0f} ms")

    print()
    speedup = cold_ms / max(warm2_ms, 1)
    print(f"Speed-up (cold / warm): {speedup:.1f}x")
    print()
    print("Interpretation:")
    print("  - With keep_alive=0 (the OLD default) EVERY request paid the cold cost.")
    print(f"  - With keep_alive={settings.OLLAMA_KEEP_ALIVE} only the first request is cold;")
    print("    warm-up at startup moves that cost off the user's first command.")


if __name__ == "__main__":
    asyncio.run(main())