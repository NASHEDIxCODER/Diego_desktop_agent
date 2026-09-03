"""
Runtime routing probe — drives the REAL production routing chain
(normalizer → intent authorization → decision engine) for the
live voice-session utterances, WITHOUT executing actions.

Usage:
    python debug/probe_routing_runtime.py
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
logging.getLogger().setLevel(logging.WARNING)

# The exact utterances from the live voice pass (direct + natural phrasing)
UTTERANCES = [
    ("system info",                          "SYSTEM_INFO"),
    ("how much RAM do I have?",              "SYSTEM_INFO"),
    ("open Firefox",                         "DESKTOP/ACTION"),
    ("what is on my screen?",                "LIVE/VISION"),
    ("find my Diego project",                "LOCAL_KNOWLEDGE"),
    ("what do you know about my project?",   "LOCAL_KNOWLEDGE"),
    ("play something on YouTube",            "MEDIA"),
    ("pause",                                "MEDIA"),
    ("resume",                               "MEDIA"),
    ("hey Diego, open Firefox",              "DESKTOP/ACTION"),
    ("Diego, find the project I was working on", "LOCAL_KNOWLEDGE"),
    ("could you play some music on YouTube?", "MEDIA"),
    ("what project was I using recently?",   "LOCAL_KNOWLEDGE"),
    # conversational / false-acceptance probes
    ("what's the weather like outside?",     "WEB_SEARCH or CONVERSATION"),
    ("tell me a joke",                       "CONVERSATION"),
    ("how are you doing today?",             "CONVERSATION"),
]


async def main() -> None:
    from nlp.command_normalizer import command_normalizer
    from nlp.intent_authorizer import authorize_intent, IntentCategory

    rows = []
    for text, expected in UTTERANCES:
        t0 = time.perf_counter()
        normalized = command_normalizer.normalize(text)
        t_norm = (time.perf_counter() - t0) * 1000

        verdict = None
        category = None
        try:
            auth = authorize_intent(normalized, transcript_quality=1.0)
            verdict = getattr(auth, "verdict", None)
            verdict = getattr(verdict, "value", verdict)
            category = getattr(auth, "category", None)
            category = getattr(category, "value", category)
        except Exception as e:
            category = f"AUTH-ERROR: {e}"

        # decision engine (no actions executed — decide() only routes)
        from core.decision_engine import decision_engine
        t1 = time.perf_counter()
        try:
            decision = await asyncio.wait_for(decision_engine.decide(normalized), timeout=30.0)
            path = decision.path.value if hasattr(decision.path, "value") else str(decision.path)
            needs_llm = getattr(decision, "needs_llm", None)
        except Exception as e:
            path = f"DECIDE-ERROR: {e}"
            needs_llm = None
        t_decide = (time.perf_counter() - t1) * 1000

        rows.append({
            "utterance": text,
            "expected": expected,
            "normalized": normalized,
            "normalized_changed": normalized != text,
            "auth_category": str(category),
            "auth_verdict": str(verdict),
            "decision_path": str(path),
            "needs_llm": needs_llm,
            "norm_ms": round(t_norm, 2),
            "decide_ms": round(t_decide, 2),
        })

        print(f"\n--- '{text}'")
        print(f"    expected   : {expected}")
        print(f"    normalized : '{normalized}'" + ("  (CHANGED)" if normalized != text else ""))
        print(f"    auth       : category={category} verdict={verdict}")
        print(f"    decision   : path={path} needs_llm={needs_llm}")
        print(f"    latency    : norm={t_norm:.2f}ms decide={t_decide:.2f}ms")

    out = Path(PROJECT_ROOT) / "debug" / "probe_routing_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    asyncio.run(main())