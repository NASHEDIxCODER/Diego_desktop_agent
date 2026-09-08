#!/usr/bin/env python3
"""
Phase 19B - ASR accent-robustness audit (READ-ONLY over the production stack).

Drives the REAL production path, unchanged:

    microphone -> audio_manager -> command_listener (Silero VAD + hysteresis
    + silence endpoint) -> faster-whisper (base/int8, greedy) -> _postprocess
    -> _validate_transcript -> nlp.command_normalizer -> nlp.intent_authorizer
    -> core.decision_engine -> (safe execution via agent_brain.process_command)

For every live utterance this records: audio duration, VAD speech evidence,
STT latency, raw transcript, Whisper confidence (avg_logprob), normalization,
intent category + authorization, decision path/action, execution result, and
per-stage latencies. Nothing in production ASR/VAD/NLP code is modified.

Results -> debug/asr_accent_audit_results.json
Usage:  python debug/asr_accent_audit.py [--limit N]
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ALSA_CONFIG_PATH", "")
os.environ["ALSA_DEBUG"] = "0"
os.environ["PULSE_LOG"] = "0"
os.environ["JACK_NO_AUDIO"] = "1"

import compat  # noqa: F401,E402

from telemetry.logger import setup_logging  # noqa: E402
setup_logging()

import logging  # noqa: E402
logging.getLogger().setLevel(logging.INFO)

from voice.audio_manager import audio_manager, FRAME_SAMPLES  # noqa: E402
from voice.command_listener import (  # noqa: E402
    command_listener, command_config, _postprocess,
    _validate_transcript, is_low_quality_transcript,
)
from nlp.command_normalizer import command_normalizer  # noqa: E402
from nlp.intent_authorizer import authorize_intent  # noqa: E402
from core.decision_engine import decision_engine  # noqa: E402

# Actions the audit must never execute (safety gate).
NEVER_EXECUTE = {
    "shutdown", "restart", "terminal", "python", "filesystem", "git",
    "git_status", "docker", "type_text", "key_press", "mouse", "click_text",
    "delete_file", "delete_folder", "clipboard_write",
}

RESULTS_PATH = PROJECT_ROOT / "debug" / "asr_accent_audit_results.json"

# Prompt script (natural speech; NOT required to be spoken verbatim).
PROMPTS = [
    {"say": "Hey Diego, how are you today?", "cat": "conversational"},
    {"say": "Thank you", "cat": "conversational"},
    {"say": "Open Firefox", "cat": "app-command", "rep": "firefox-1"},
    {"say": "Open Firefox", "cat": "app-command", "rep": "firefox-2"},
    {"say": "Open Firefox", "cat": "app-command", "rep": "firefox-3"},
    {"say": "Open the file manager", "cat": "app-command"},
    {"say": "Open PyCharm", "cat": "app-command/proper-noun"},
    {"say": "Open GitHub", "cat": "app-command/proper-noun"},
    {"say": "Play Believer on YouTube", "cat": "media", "rep": "yt-1"},
    {"say": "Play Believer on YouTube", "cat": "media", "rep": "yt-2"},
    {"say": "What's my RAM?", "cat": "system-info", "rep": "ram-1"},
    {"say": "What's my RAM?", "cat": "system-info", "rep": "ram-2"},
    {"say": "What time is it?", "cat": "deterministic"},
    {"say": "What's the weather in Chennai?", "cat": "search/proper-noun"},
    {"say": "Volume up", "cat": "short-command"},
    {"say": "Stop", "cat": "short-command"},
    {"say": "Can you open the terminal and check the system information for me?",
     "cat": "long-sentence"},
    {"say": "What is my CPU usage right now?", "cat": "system-info"},
]


def frames_from_pcm(pcm: bytes):
    import numpy as np
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return [audio[i:i + FRAME_SAMPLES]
            for i in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES)]


def now_ms() -> float:
    return time.time() * 1000.0


def announce(idx: int, total: int, prompt: dict) -> None:
    print("", flush=True)
    print("=" * 64, flush=True)
    print(f">>> AUDIT {idx}/{total} - SPEAK NOW: \"{prompt['say']}\"", flush=True)
    print("    (natural speech - no need to match exactly)", flush=True)
    print("\a" * 3, flush=True)  # terminal bell cue


async def run_prompt(idx: int, total: int, prompt: dict, brain) -> dict:
    record = {"id": idx, "category": prompt["cat"],
              "repeat_tag": prompt.get("rep", ""), "prompt": prompt["say"]}
    announce(idx, total, prompt)

    t_wait_start = time.time()
    event = None
    try:
        agen = command_listener.stream_utterances()
        deadline = t_wait_start + 40.0
        async for ev in agen:
            if ev.kind in ("final", "failure"):
                event = ev
                break
            if time.time() > deadline:
                break
        try:
            await agen.aclose()
        except Exception:
            pass
    except Exception as e:
        record["capture_error"] = f"{type(e).__name__}: {e}"
        return record

    if event is None:
        record["error"] = "no utterance captured within timeout"
        record["wait_s"] = round(time.time() - t_wait_start, 2)
        return record

    record.update({
        "event_kind": event.kind,
        "failure_reason": event.failure_reason or "",
        "endpoint_reason": event.endpoint_reason or "",
        "audio_duration_ms": round(event.audio_duration_ms, 1),
        "stt_latency_ms": round(event.whisper_latency_ms, 1),
        "capture_to_transcript_ms": round((time.time() - t_wait_start) * 1000.0, 1),
        "stt_confidence_avg_logprob": round(event.confidence, 3),
        "transcript_raw": event.text,
    })

    # VAD speech evidence from the captured audio (real per-frame signals).
    try:
        frames = frames_from_pcm(event.audio or b"")
        ev = command_listener._measure_speech_evidence(frames)
        voiced = round(ev.get("silero_voiced_ms", 0.0), 1)
        total_ms = max(record["audio_duration_ms"], 1.0)
        record["vad_evidence"] = {
            "silero_voiced_ms": voiced,
            "strong_rms_ms": round(ev.get("strong_ms", 0.0), 1),
            "loud_rms_ms": round(ev.get("loud_ms", 0.0), 1),
            "peak_rms": round(ev.get("peak_rms", 0.0), 1),
            "voiced_ratio": round(voiced / total_ms, 3),
        }
    except Exception as e:
        record["vad_evidence_error"] = str(e)

    if event.kind == "failure":
        record["normalized"] = None
        record["intent"] = None
        record["decision"] = None
        record["executed"] = False
        record["exec_note"] = f"STT failure: {event.failure_reason}"
        return record

    # Stage: command normalization
    t0 = now_ms()
    normalized = command_normalizer.normalize(event.text)
    record["normalize_ms"] = round(now_ms() - t0, 2)
    record["normalized"] = normalized
    record["low_quality_guard"] = is_low_quality_transcript(normalized)

    # Stage: intent authorization (production arguments)
    t0 = now_ms()
    auth = authorize_intent(event.text,
                            stt_confidence=event.confidence,
                            audio_duration_ms=event.audio_duration_ms)
    record["intent_ms"] = round(now_ms() - t0, 2)
    record["intent"] = {
        "category": auth.category.value,
        "actionable": auth.actionable,
        "llm_allowed": auth.llm_allowed,
        "confidence": round(auth.confidence, 3),
        "reason": auth.reason,
    }

    # Stage: decision routing
    t0 = now_ms()
    decision = await decision_engine.decide(normalized)
    record["decision_ms"] = round(now_ms() - t0, 2)
    record["decision"] = {
        "path": decision.path.value,
        "needs_llm": decision.needs_llm,
        "action": decision.action,
        "response": (decision.response or "")[:80],
    }

    # Execution through the production Brain (safe subset).
    action_name = (decision.action or {}).get("action", "") if decision.action else ""
    if action_name in NEVER_EXECUTE:
        record["executed"] = False
        record["exec_note"] = f"skipped: '{action_name}' is audit-forbidden"
        return record

    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            brain.process_command(event.text,
                                  stt_confidence=event.confidence,
                                  audio_duration_ms=event.audio_duration_ms),
            timeout=90.0)
        record["executed"] = True
        record["exec_ms"] = round((time.time() - t0) * 1000.0, 1)
        record["execution"] = {
            "path": result.path,
            "response": (result.response or "")[:120],
            "actions_executed": result.actions_executed,
            "actions_succeeded": result.actions_succeeded,
            "actions_failed": result.actions_failed,
            "verified": result.verified,
            "task_status": result.task_status,
            "used_llm": result.used_llm,
        }
        try:
            record["brain_stage_timings_ms"] = {
                k: round(v, 1) for k, v in brain._pipeline_timings.items()}
        except Exception:
            pass
    except asyncio.TimeoutError:
        record["executed"] = True
        record["exec_note"] = "process_command TIMEOUT (>90s)"
        record["exec_ms"] = round((time.time() - t0) * 1000.0, 1)
    except Exception as e:
        record["executed"] = False
        record["exec_note"] = f"execution error: {type(e).__name__}: {e}"
    return record


async def main() -> None:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    prompts = PROMPTS[:limit] if limit else PROMPTS

    print("[AUDIT] starting audio backend...", flush=True)
    if not audio_manager.start():
        print("[AUDIT] FATAL: audio backend failed to start", flush=True)
        return
    print("[AUDIT] loading production Whisper (base) ...", flush=True)
    if not command_listener.initialize():
        print("[AUDIT] FATAL: command listener failed to initialize", flush=True)
        return
    print("[AUDIT] production ASR ready "
          f"(model=base, device={command_listener._whisper._device}, "
          f"compute={command_listener._whisper._compute})", flush=True)

    from agent.brain import agent_brain
    try:
        await asyncio.wait_for(agent_brain.initialize(), timeout=90.0)
        print("[AUDIT] brain initialized", flush=True)
    except Exception as e:
        print(f"[AUDIT] brain init failed ({e}) - decisions still recorded, "
              "execution may fail", flush=True)

    results = []
    try:
        total = len(prompts)
        for i, prompt in enumerate(prompts, 1):
            rec = await run_prompt(i, total, prompt, agent_brain)
            results.append(rec)
            print(f"[AUDIT] {i}/{total} done -> "
                  f"kind={rec.get('event_kind')} "
                  f"text={rec.get('transcript_raw')!r} "
                  f"conf={rec.get('stt_confidence_avg_logprob')} "
                  f"intent={(rec.get('intent') or {}).get('category')} "
                  f"decision={(rec.get('decision') or {}).get('path')}",
                  flush=True)
            with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
                json.dump(results, fh, ensure_ascii=False, indent=2)
            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        print("[AUDIT] interrupted", flush=True)
    finally:
        with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        audio_manager.stop()
        print(f"[AUDIT] saved {len(results)} records -> {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
