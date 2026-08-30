"""
Runtime audit harness — End-to-end execution pipeline tracer.

Traces every command through:
  Normalize → Perceive → Decide → Plan → Dispatch+Verify → Respond

CRITICAL FIX (audit B4): dispatch now goes through
Brain._dispatch_and_verify() — the SAME production path used by
Brain.process_command() — so the trace includes pre-action capture,
post-action verification, and the retry-with-adjusted-params logic.
Previously the harness called ActionDispatcher.execute() +
Brain._verify() directly, bypassing retries, which made the trace
diverge from production (e.g. "open terminal" failed without the
gnome-terminal → xterm retry ever being attempted).

For EVERY stage logs: INPUT, OUTPUT, latency, return value.
Stops immediately if any stage returns None or an empty action list.

Usage:
    python debug/audit_execution_pipeline.py
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compat  # noqa: F401  (Python 3.14 stubs)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
# Quiet the noisiest loggers
for noisy in ("httpx", "urllib3", "PIL", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("AUDIT")


# ═══════════════════════════════════════════════════════════════
# Stage tracer
# ═══════════════════════════════════════════════════════════════

class StageTracer:
    """Logs INPUT/OUTPUT/latency for each pipeline stage."""

    def __init__(self):
        self._stages: List[Dict[str, Any]] = []
        self._cmd_active = False

    def reset(self):
        self._stages = []

    def stage(self, name: str, input_repr: Any, output: Any,
              latency_ms: float, note: str = "") -> Any:
        """Record a stage. Returns output unchanged."""
        stage = {
            "stage": name,
            "input": str(input_repr)[:200],
            "output_type": type(output).__name__,
            "output": str(output)[:200],
            "latency_ms": round(latency_ms, 1),
            "note": note,
        }
        self._stages.append(stage)

        arrow = "─" * 3
        print(f"\n  {name} {arrow} INPUT: {stage['input']}")
        print(f"  {name} {arrow} OUTPUT ({stage['output_type']}): "
              f"{stage['output']}")
        print(f"  {name} {arrow} LATENCY: {stage['latency_ms']}ms")
        if note:
            print(f"  {name} {arrow} NOTE: {note}")

        # STOP immediately if None
        if output is None:
            raise PipelineStop(
                f"[STOP] {name} returned None — pipeline halted before response."
            )
        # STOP immediately if an empty action list
        if (name == "PLAN" and isinstance(output, list) and len(output) == 0):
            raise PipelineStop(
                f"[STOP] {name} returned empty action list — nothing to dispatch."
            )
        return output

    def dump(self, cmd: str, result: Any) -> None:
        """Write the full trace for a command to a JSON file."""
        out = {
            "command": cmd,
            "stages": self._stages,
            "final_response": str(getattr(result, "response", result)),
            "actions_executed": getattr(result, "actions_executed", 0),
            "actions_succeeded": getattr(result, "actions_succeeded", 0),
            "actions_failed": getattr(result, "actions_failed", 0),
            "verified": getattr(result, "verified", None),
            "path": getattr(result, "path", ""),
            "latency_ms": getattr(result, "latency_ms", 0),
        }
        out_path = ROOT / "logs" / "audit_trace.json"
        existing = []
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
            except Exception:
                existing = []
        existing.append(out)
        out_path.write_text(json.dumps(existing, indent=2, default=str))
        print(f"\n  [DUMP] Trace appended to {out_path}")


class PipelineStop(Exception):
    """Raised when a stage returns None or the pipeline cannot proceed."""


tracer = StageTracer()


# ═══════════════════════════════════════════════════════════════
# Manual command tests
# ═══════════════════════════════════════════════════════════════

TEST_COMMANDS = [
    "open youtube",
    "open firefox",
    "open vs code",
    "open terminal",
    "search youtube for lo-fi",
    "play music",
    "pause music",
    "resume music",
    "close firefox",
]


def summarize_pipeline(cmd: str) -> str:
    """Quick one-line summary of where the pipeline went."""
    parts = []
    for s in tracer._stages:
        parts.append(f"{s['stage']}→{s['output_type']}")
    return " → ".join(parts)


async def run_single_command(brain, cmd: str) -> None:
    """Run ONE command through the full pipeline with trace."""
    print("\n" + "═" * 70)
    print(f"  COMMAND: '{cmd}'")
    print("═" * 70)
    tracer.reset()
    t0_cmd = time.time()

    # ── Stage 0: Normalize ──────────────────────────────────
    from nlp.command_normalizer import command_normalizer
    t0 = time.time()
    normalized = command_normalizer.normalize(cmd)
    tracer.stage("NORMALIZE", cmd, normalized, (time.time() - t0) * 1000,
                 "would-be input to Brain")

    # ── Stage 1: Perceive ───────────────────────────────────
    t0 = time.time()
    try:
        perception_ctx = await brain._perceive()
        tracer.stage("PERCEIVE", normalized, perception_ctx,
                     (time.time() - t0) * 1000,
                     "desktop context snapshot")
    except Exception as e:
        tracer.stage("PERCEIVE", normalized, f"EXC: {e}",
                     (time.time() - t0) * 1000)
        raise PipelineStop(f"[STOP] PERCEIVE raised {e}")

    # ── Stage 2: Decide ─────────────────────────────────────
    t0 = time.time()
    try:
        decision = await brain._decide(normalized, perception_ctx)
        tracer.stage(
            "DECIDE", normalized, decision,
            (time.time() - t0) * 1000,
            f"path={getattr(decision, 'path', '?').value if hasattr(getattr(decision, 'path', None), 'value') else getattr(decision, 'path', '?')} "
            f"needs_llm={getattr(decision, 'needs_llm', '?')} "
            f"action={getattr(decision, 'action', None) is not None} "
            f"actions={len(getattr(decision, 'actions', []) or [])}",
        )
    except Exception as e:
        tracer.stage("DECIDE", normalized, f"EXC: {e}",
                     (time.time() - t0) * 1000)
        raise PipelineStop(f"[STOP] DECIDE raised {e}")

    resolved = bool(getattr(decision, "resolved", False))

    # ── Stage 3: Plan (only for LLM path) ───────────────────
    plan = None
    if not resolved:
        t0 = time.time()
        try:
            plan = await brain._plan(normalized, perception_ctx)
            tracer.stage(
                "PLAN", normalized, plan, (time.time() - t0) * 1000,
                f"{len(plan)} executable action(s)" if plan else "None — no plan",
            )
        except Exception as e:
            tracer.stage("PLAN", normalized, f"EXC: {e}",
                         (time.time() - t0) * 1000)
            raise PipelineStop(f"[STOP] PLAN raised {e}")

    # ── Stage 4: Dispatch + Verify (per action) ─────────────
    actions: List[Dict[str, Any]] = []
    if resolved and getattr(decision, "action", None):
        actions.append(decision.action)
    if resolved and getattr(decision, "actions", None):
        actions.extend(decision.actions)
    if not resolved and plan:
        actions = [brain._step_to_action(s) for s in plan]
        actions = [a for a in actions if a]

    verifications = []
    if not actions:
        raise PipelineStop(
            f"[STOP] Stage 4: dispatch got EMPTY action list — "
            f"nothing to execute. (resolved={resolved}, "
            f"decision.action={bool(getattr(decision, 'action', None))}, "
            f"decision.actions={len(getattr(decision, 'actions', []) or [])}, "
            f"plan={plan!r})"
        )

    for i, action in enumerate(actions, 1):
        aname = action.get("action", "?")
        aparams = action.get("params", {})
        print(f"\n  ── DISPATCH[{i}/{len(actions)}] {aname} {aparams}")

        # ── Production path (audit B4) ──
        # Brain._dispatch_and_verify() is the ONLY place production
        # dispatches actions: it captures pre-action state, executes,
        # verifies AFTER execution, and retries with adjusted params
        # (up to MAX_ACTION_RETRIES) on verification failure. Using it
        # here keeps the audit trace faithful to the live pipeline
        # without duplicating any business logic.
        t0 = time.time()
        try:
            ok, action_result = await brain._dispatch_and_verify(action)
            tracer.stage(
                f"DISPATCH_VERIFY {aname}", action,
                action_result if action_result else ("OK" if ok else "FAILED"),
                (time.time() - t0) * 1000,
                f"Brain._dispatch_and_verify() ok={ok} "
                f"(dispatch → verify → retry, production path)",
            )
            verifications.append(ok)
        except Exception as e:
            tracer.stage(f"DISPATCH_VERIFY {aname}", action, f"EXC: {e}",
                         (time.time() - t0) * 1000)
            raise PipelineStop(
                f"[STOP] DISPATCH_VERIFY {aname} raised {e} — "
                f"action did not execute."
            )

    # ── Stage 5: Response (generated AFTER execution, from traced result) ──
    # CRITICAL: Do NOT call brain.process_command() here — it would re-run
    # the ENTIRE pipeline and execute every action a SECOND time.
    # We construct the CommandResult equivalent from the traced stages.
    from agent.brain import CommandResult
    result = CommandResult()
    result.actions_executed = len(actions)
    result.actions_succeeded = sum(1 for v in verifications if v)
    result.actions_failed = len(actions) - result.actions_succeeded
    result.verified = result.actions_failed == 0
    result.path = (
        decision.path.value if hasattr(getattr(decision, "path", None), "value")
        else str(getattr(decision, "path", "?"))
    )
    result.used_llm = not resolved

    # Response: only after execution, never the canned promise on failure
    if resolved:
        if result.actions_failed > 0:
            result.response = brain._default_response(result)
        else:
            result.response = decision.response or brain._default_response(result)
    else:
        result.response = await brain._generate_response(normalized, perception_ctx, result)
    result.latency_ms = (time.time() - t0_cmd) * 1000

    t0 = time.time()
    tracer.stage(
        "RESPONSE", cmd, result.response, (time.time() - t0) * 1000,
        f"actions={result.actions_executed} ok={result.actions_succeeded} "
        f"failed={result.actions_failed} verified={result.verified}",
    )

    print("\n  ── SUMMARY ──")
    print(f"  Path: {summarize_pipeline(cmd)}")
    print(f"  Final response: '{result.response}'")
    print(f"  Actions: {result.actions_executed} | "
          f"OK: {result.actions_succeeded} | "
          f"Failed: {result.actions_failed} | "
          f"Verified: {result.verified}")
    print(f"  Total pipeline latency: {result.latency_ms:.0f}ms")

    tracer.dump(cmd, result)


async def main() -> None:
    print("═" * 70)
    print("  DIEGO DESKTOP ASSISTANT — RUNTIME EXECUTION PIPELINE AUDIT")
    print("═" * 70)

    from agent.brain import agent_brain

    # Initialize the brain (wires planner, dispatcher, verifier, etc.)
    init_ok = await agent_brain.initialize()
    print(f"\n[Brain.initialize()] → {init_ok}")
    print(f"[Brain.is_available] → {agent_brain.is_available}")
    print(f"[Brain._dispatcher]  → {'ready' if agent_brain._dispatcher else 'MISSING — cannot execute!'}")
    print(f"[Brain._planner]     → {'ready' if agent_brain._planner else 'MISSING — cannot plan!'}")
    print(f"[Brain._verifier]    → {'ready' if agent_brain._verifier else 'MISSING — verification will trust dispatch'}")
    print(f"[Brain._decision]    → {'ready' if agent_brain._decision_engine else 'MISSING — ALWAYS LLM path'}")

    results = []
    for cmd in TEST_COMMANDS:
        try:
            await run_single_command(agent_brain, cmd)
            results.append({"command": cmd, "status": "OK"})
        except PipelineStop as stop:
            print(f"\n  ❌ {cmd}: {stop}")
            traceback.print_exc()
            results.append({"command": cmd, "status": "STOPPED", "why": str(stop)})
        except Exception as e:
            print(f"\n  ❌ {cmd}: UNEXPECTED {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"command": cmd, "status": "ERROR", "why": str(e)})

    print("\n" + "═" * 70)
    print("  FINAL RESULTS")
    print("═" * 70)
    for r in results:
        status = "✅" if r["status"] == "OK" else "❌"
        detail = f" — {r['why']}" if r["status"] != "OK" else ""
        print(f"  {status} {r['command']}: {r['status']}{detail}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n  {ok_count}/{len(results)} commands executed successfully.")


if __name__ == "__main__":
    asyncio.run(main())