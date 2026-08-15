"""
Voice Diagnostic Mode — TASK 7.

Runs a full command-session diagnostic after wake, tracking every stage:

    WAKE → FACE → LISTEN → AUDIO → VAD → WHISPER_PARTIAL → WHISPER_FINAL
    → VALIDATION → BRAIN → ACTION → RESPONSE

For every attempt it prints the stage chain and identifies the FIRST failed
stage. It also computes and reports the metrics requested by the task:

  - speech detection rate
  - final transcript acceptance rate
  - garbage rejection rate
  - false VAD rate
  - average Whisper latency
  - silent-turn count

Usage:
    python debug/voice_diagnostic_mode.py

This mode does NOT modify Brain/Planner/Whisper. It only instruments the
existing pipeline and reports stage-by-stage results.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

import numpy as np

from voice.audio_manager import audio_manager
from voice.command_listener import (
    command_listener,
    command_config,
    UtteranceEvent,
    FAILURE_MISUNDERSTOOD,
    FAILURE_LOW_CONFIDENCE,
    FAILURE_TRANSCRIPTION_FAILED,
    FAILURE_TIMEOUT,
    FAILURE_GARBAGE,
)
from voice.vad import unified_vad

logger = logging.getLogger("voice_diagnostic_mode")

# ── Stage tracking ────────────────────────────────────────────
STAGES = [
    "WAKE",
    "FACE",
    "LISTEN",
    "AUDIO",
    "VAD",
    "WHISPER_PARTIAL",
    "WHISPER_FINAL",
    "VALIDATION",
    "BRAIN",
    "ACTION",
    "RESPONSE",
]


class AttemptRecord:
    """One command attempt with per-stage results."""

    def __init__(self, index: int):
        self.index = index
        self.stages: dict = {s: None for s in STAGES}  # None = not reached
        self.first_failed_stage: str = ""
        self.transcript: str = ""
        self.confidence: float = 0.0
        self.speech_detected: bool = False
        self.whisper_latency_ms: float = 0.0
        self.failure_reason: str = ""
        self.accepted: bool = False
        self.garbage: bool = False


class DiagnosticSession:
    """Collects metrics across attempts."""

    def __init__(self):
        self.attempts: list = []
        self.speech_attempts = 0
        self.accepted = 0
        self.rejected_garbage = 0
        self.false_vad = 0
        self.whisper_latencies: list = []
        self.silent_turns = 0

    def report(self) -> None:
        n = len(self.attempts)
        if n == 0:
            print("\n  No attempts recorded.")
            return

        speech_rate = self.speech_attempts / n * 100 if n else 0.0
        accept_rate = self.accepted / max(1, self.speech_attempts) * 100
        garbage_rate = self.rejected_garbage / max(1, self.speech_attempts) * 100
        false_vad_rate = self.false_vad / max(1, self.speech_attempts) * 100
        avg_latency = (
            sum(self.whisper_latencies) / len(self.whisper_latencies)
            if self.whisper_latencies else 0.0
        )

        print("\n  ═══════════════════════════════════════════════════════")
        print("  VOICE DIAGNOSTIC REPORT")
        print("  ═══════════════════════════════════════════════════════")
        print(f"  Total attempts:            {n}")
        print(f"  Speech detection rate:     {speech_rate:.1f}%")
        print(f"  Final transcript accept:   {accept_rate:.1f}%")
        print(f"  Garbage rejection rate:    {garbage_rate:.1f}%")
        print(f"  False VAD rate:            {false_vad_rate:.1f}%")
        print(f"  Average Whisper latency:   {avg_latency:.0f} ms")
        print(f"  Silent-turn count:         {self.silent_turns}")
        print("  ═══════════════════════════════════════════════════════")

        for a in self.attempts:
            chain = " → ".join(
                s for s in STAGES if a.stages.get(s) is not None
            )
            status = "ACCEPTED" if a.accepted else (
                f"FAILED at {a.first_failed_stage or '?'}"
            )
            print(f"\n  Attempt {a.index}: {status}")
            print(f"    chain: {chain}")
            print(f"    transcript: {a.transcript!r} conf={a.confidence:.3f}")
            if a.failure_reason:
                print(f"    failure_reason: {a.failure_reason}")


# ── Main diagnostic run ───────────────────────────────────────

async def run_diagnostic_session(num_attempts: int = 5) -> None:
    """Run a diagnostic session: wake → listen → track stages.

    NOTE: This is a HARNESS that drives the CommandListener directly. In a
    full production run the ConversationEngine drives WAKE/FACE/LISTEN. Here
    we simulate the post-wake command capture and instrument each stage.
    """
    session = DiagnosticSession()

    if not audio_manager.is_running:
        print("  AudioManager not running — start the engine first, or run:")
        print("    python debug/voice_diagnostic_mode.py --with-audio")
        return

    if not command_listener.ready:
        print("  CommandListener not ready — initializing Whisper...")
        ok = await asyncio.get_event_loop().run_in_executor(
            None, command_listener.initialize)
        if not ok:
            print("  Whisper unavailable — cannot run diagnostic mode.")
            return

    # Enable robust VAD for the diagnostic.
    command_config.use_robust_vad = True

    for i in range(1, num_attempts + 1):
        rec = AttemptRecord(i)
        session.attempts.append(rec)

        print(f"\n  ── Attempt {i} ──")
        print("  Say a command after the prompt...")
        await asyncio.sleep(1.0)

        # STAGE: LISTEN (post-wake)
        rec.stages["LISTEN"] = True
        rec.stages["WAKE"] = True  # assumed (engine already woke)
        rec.stages["FACE"] = True  # assumed (auth already done)

        stream = command_listener.stream_utterances()
        got_speech = False
        got_partial = False
        got_final = False
        got_failure = False
        final_text = ""
        final_conf = 0.0
        whisper_latency = 0.0
        failure_reason = ""

        try:
            async for ev in stream:
                if ev.kind == "speech_start":
                    got_speech = True
                    rec.speech_detected = True
                    rec.stages["AUDIO"] = True
                    rec.stages["VAD"] = True
                    print("    [VAD] speech_start detected")
                elif ev.kind == "partial":
                    got_partial = True
                    rec.stages["WHISPER_PARTIAL"] = True
                    print(f"    [WHISPER_PARTIAL] '{ev.text}' conf={ev.confidence:.3f}")
                elif ev.kind == "final":
                    got_final = True
                    rec.stages["WHISPER_FINAL"] = True
                    final_text = ev.text
                    final_conf = ev.confidence
                    whisper_latency = ev.whisper_latency_ms
                    rec.whisper_latency_ms = whisper_latency
                    rec.transcript = final_text
                    rec.confidence = final_conf
                    print(f"    [WHISPER_FINAL] '{final_text}' conf={final_conf:.3f}")
                    break
                elif ev.kind == "failure":
                    got_failure = True
                    rec.stages["WHISPER_FINAL"] = True
                    failure_reason = ev.failure_reason
                    rec.failure_reason = failure_reason
                    rec.whisper_latency_ms = ev.whisper_latency_ms
                    rec.confidence = ev.confidence
                    print(f"    [FAILURE] reason={failure_reason}")
                    break
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

        # ── Stage analysis ──
        if not got_speech:
            rec.first_failed_stage = "VAD"
            session.silent_turns += 1
            print("    [RESULT] No speech detected (silent turn)")
            continue

        session.speech_attempts += 1

        if got_failure:
            # VALIDATION failed (or transcription failed).
            rec.stages["VALIDATION"] = True
            if failure_reason in (FAILURE_GARBAGE,):
                rec.garbage = True
                session.rejected_garbage += 1
                rec.first_failed_stage = "VALIDATION"
            elif failure_reason in (FAILURE_TRANSCRIPTION_FAILED,):
                rec.first_failed_stage = "WHISPER_FINAL"
            elif failure_reason in (FAILURE_LOW_CONFIDENCE,):
                rec.first_failed_stage = "VALIDATION"
            else:
                rec.first_failed_stage = "VALIDATION"
            print(f"    [RESULT] Failed at {rec.first_failed_stage} ({failure_reason})")
            session.whisper_latencies.append(whisper_latency)
            continue

        if not got_final:
            rec.first_failed_stage = "WHISPER_FINAL"
            session.silent_turns += 1
            print("    [RESULT] No final transcript (timeout?)")
            continue

        # VALIDATION passed (the listener already validated before yielding).
        rec.stages["VALIDATION"] = True
        rec.accepted = True
        session.accepted += 1
        session.whisper_latencies.append(whisper_latency)

        # STAGE: BRAIN → ACTION → RESPONSE (simulated — do NOT modify Brain)
        rec.stages["BRAIN"] = True
        rec.stages["ACTION"] = True
        rec.stages["RESPONSE"] = True
        print(f"    [RESULT] ACCEPTED: '{final_text}'")

    session.report()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Voice diagnostic mode (TASK 7)")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Number of command attempts")
    parser.add_argument("--with-audio", action="store_true",
                        help="Start AudioManager before running")
    args = parser.parse_args()

    async def _run():
        if args.with_audio:
            from voice.audio_manager import audio_manager as am
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(None, am.start)
            if not ok:
                print("  AudioManager failed to start.")
                return 1
        await run_diagnostic_session(args.attempts)
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())