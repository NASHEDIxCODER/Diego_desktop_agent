"""
Benchmark — Runtime validation and performance tracking for Leo.

Tracks:
  - LLM usage before vs after routing
  - Average response latency
  - Cache hit rates
  - Files modified
  - Remaining bottlenecks
  - Wake latency, face latency, STT first token latency
  - GPU status, VAD acceptance rate, wake acceptance rate
  - Face verification overlap

Usage:
    from core.benchmark import benchmark

    # After each turn:
    benchmark.record_turn(llm_used=True, latency_ms=450.0)

    # Print report:
    print(benchmark.report())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark.json"


@dataclass
class TurnRecord:
    """A single conversation turn record."""
    timestamp: float = field(default_factory=time.time)
    text: str = ""
    llm_used: bool = False
    router_kind: str = ""  # SIMPLE_DESKTOP, CONVERSATION, COMPLEX, etc.
    latency_ms: float = 0.0
    stt_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    cache_hit: bool = False
    action_executed: bool = False
    action_success: bool = False


class Benchmark:
    """
    Runtime performance tracker for Leo.

    Collects per-turn metrics and provides aggregate reports
    showing LLM usage reduction, latency improvements, and bottlenecks.
    """

    def __init__(self):
        self._turns: List[TurnRecord] = []
        self._start_time: float = time.time()
        self._session_id: str = str(int(self._start_time))

        # Aggregate counters
        self._total_turns: int = 0
        self._llm_turns: int = 0
        self._bypassed_turns: int = 0
        self._cache_hits: int = 0
        self._actions_executed: int = 0
        self._actions_succeeded: int = 0
        self._total_latency_ms: float = 0.0
        self._total_stt_ms: float = 0.0
        self._total_llm_ms: float = 0.0
        self._total_tts_ms: float = 0.0

        # Files modified tracking
        self._files_modified: List[str] = []

        # Bottleneck tracking
        self._bottlenecks: Dict[str, float] = {}  # stage → cumulative ms

        # ── ISSUE-1/2/3/4: Production validation metrics ──
        # Wake pipeline
        self._wake_latencies_ms: List[float] = []       # time from trigger to verification complete
        self._wake_scores: List[float] = []              # openWakeWord scores
        self._wake_accept_count: int = 0                 # verified wakes
        self._wake_reject_count: int = 0                 # rejected wakes
        self._wake_total_triggers: int = 0               # total triggers (accepted + rejected)
        # VAD
        self._vad_accept_count: int = 0                  # VAD gate opened
        self._vad_reject_count: int = 0                  # VAD gate closed on trigger
        # Face auth
        self._face_latencies_ms: List[float] = []        # face verification duration
        self._face_overlap_count: int = 0                # face auth overlapped with STT
        self._face_success_count: int = 0
        self._face_fail_count: int = 0
        # STT
        self._stt_first_token_ms: List[float] = []       # time from speech_start to first partial
        self._stt_partial_count: int = 0
        # GPU
        self._gpu_available: Optional[bool] = None
        self._gpu_device: str = ""
        self._gpu_compute: str = ""

    # ── Recording ──────────────────────────────────────────────────

    def record_turn(self, text: str = "", llm_used: bool = False,
                    router_kind: str = "", latency_ms: float = 0.0,
                    stt_latency_ms: float = 0.0, llm_latency_ms: float = 0.0,
                    tts_latency_ms: float = 0.0, cache_hit: bool = False,
                    action_executed: bool = False, action_success: bool = False) -> None:
        """Record a single conversation turn."""
        turn = TurnRecord(
            text=text,
            llm_used=llm_used,
            router_kind=router_kind,
            latency_ms=latency_ms,
            stt_latency_ms=stt_latency_ms,
            llm_latency_ms=llm_latency_ms,
            tts_latency_ms=tts_latency_ms,
            cache_hit=cache_hit,
            action_executed=action_executed,
            action_success=action_success,
        )
        self._turns.append(turn)

        # Update aggregates
        self._total_turns += 1
        if llm_used:
            self._llm_turns += 1
        else:
            self._bypassed_turns += 1
        if cache_hit:
            self._cache_hits += 1
        if action_executed:
            self._actions_executed += 1
            if action_success:
                self._actions_succeeded += 1

        self._total_latency_ms += latency_ms
        self._total_stt_ms += stt_latency_ms
        self._total_llm_ms += llm_latency_ms
        self._total_tts_ms += tts_latency_ms

        # Track bottlenecks
        if stt_latency_ms > 0:
            self._bottlenecks["stt"] = self._bottlenecks.get("stt", 0) + stt_latency_ms
        if llm_latency_ms > 0:
            self._bottlenecks["llm"] = self._bottlenecks.get("llm", 0) + llm_latency_ms
        if tts_latency_ms > 0:
            self._bottlenecks["tts"] = self._bottlenecks.get("tts", 0) + tts_latency_ms

        # Keep only last 1000 turns in memory
        if len(self._turns) > 1000:
            self._turns = self._turns[-500:]

    def record_file_modified(self, path: str) -> None:
        """Track a file that was modified."""
        if path not in self._files_modified:
            self._files_modified.append(path)

    # ── ISSUE-1/2/3/4: Production validation recording ─────────────

    def record_wake(self, score: float, latency_ms: float, accepted: bool) -> None:
        """Record a wake trigger event."""
        self._wake_total_triggers += 1
        self._wake_scores.append(score)
        self._wake_latencies_ms.append(latency_ms)
        if accepted:
            self._wake_accept_count += 1
        else:
            self._wake_reject_count += 1

    def record_vad_gate(self, gate_open: bool) -> None:
        """Record whether the VAD gate was open at trigger onset."""
        if gate_open:
            self._vad_accept_count += 1
        else:
            self._vad_reject_count += 1

    def record_face_auth(self, latency_ms: float, success: bool, overlapped: bool) -> None:
        """Record a face authentication event."""
        self._face_latencies_ms.append(latency_ms)
        if success:
            self._face_success_count += 1
        else:
            self._face_fail_count += 1
        if overlapped:
            self._face_overlap_count += 1

    def record_stt_first_token(self, latency_ms: float) -> None:
        """Record time from speech_start to first partial transcript."""
        self._stt_first_token_ms.append(latency_ms)
        self._stt_partial_count += 1

    def record_gpu_status(self, available: bool, device: str = "", compute: str = "") -> None:
        """Record GPU backend status (called once at startup)."""
        self._gpu_available = available
        self._gpu_device = device
        self._gpu_compute = compute

    # ── Queries ────────────────────────────────────────────────────

    @property
    def llm_usage_rate(self) -> float:
        """Fraction of turns that used the LLM."""
        if self._total_turns == 0:
            return 0.0
        return self._llm_turns / self._total_turns

    @property
    def llm_bypass_rate(self) -> float:
        """Fraction of turns that bypassed the LLM."""
        if self._total_turns == 0:
            return 0.0
        return self._bypassed_turns / self._total_turns

    @property
    def avg_latency_ms(self) -> float:
        """Average total turn latency."""
        if self._total_turns == 0:
            return 0.0
        return self._total_latency_ms / self._total_turns

    @property
    def avg_stt_latency_ms(self) -> float:
        """Average STT latency."""
        if self._total_turns == 0:
            return 0.0
        return self._total_stt_ms / self._total_turns

    @property
    def avg_llm_latency_ms(self) -> float:
        """Average LLM latency (only for LLM turns)."""
        if self._llm_turns == 0:
            return 0.0
        return self._total_llm_ms / self._llm_turns

    @property
    def avg_tts_latency_ms(self) -> float:
        """Average TTS latency."""
        if self._total_turns == 0:
            return 0.0
        return self._total_tts_ms / self._total_turns

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate."""
        if self._total_turns == 0:
            return 0.0
        return self._cache_hits / self._total_turns

    @property
    def action_success_rate(self) -> float:
        """Action success rate."""
        if self._actions_executed == 0:
            return 1.0
        return self._actions_succeeded / self._actions_executed

    @property
    def uptime_s(self) -> float:
        """Session uptime in seconds."""
        return time.time() - self._start_time

    # ── Reports ────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        # Identify bottlenecks
        bottleneck_stages = sorted(
            self._bottlenecks.items(), key=lambda x: x[1], reverse=True)

        report = {
            "session": {
                "id": self._session_id,
                "uptime_s": f"{self.uptime_s:.0f}",
                "total_turns": self._total_turns,
            },
            "llm_usage": {
                "llm_turns": self._llm_turns,
                "bypassed_turns": self._bypassed_turns,
                "llm_usage_rate": f"{self.llm_usage_rate:.1%}",
                "llm_bypass_rate": f"{self.llm_bypass_rate:.1%}",
                "reduction_vs_baseline": f"{(1.0 - self.llm_usage_rate) * 100:.0f}%",
            },
            "latency": {
                "avg_total_ms": f"{self.avg_latency_ms:.1f}",
                "avg_stt_ms": f"{self.avg_stt_latency_ms:.1f}",
                "avg_llm_ms": f"{self.avg_llm_latency_ms:.1f}",
                "avg_tts_ms": f"{self.avg_tts_latency_ms:.1f}",
            },
            "caching": {
                "cache_hits": self._cache_hits,
                "cache_hit_rate": f"{self.cache_hit_rate:.1%}",
            },
            "actions": {
                "executed": self._actions_executed,
                "succeeded": self._actions_succeeded,
                "success_rate": f"{self.action_success_rate:.1%}",
            },
            "bottlenecks": [
                {"stage": stage, "cumulative_ms": f"{ms:.0f}"}
                for stage, ms in bottleneck_stages[:5]
            ],
            "files_modified": self._files_modified[-20:],
        }

        # ── Add decision engine stats ──────────────────────
        try:
            from core.decision_engine import decision_engine
            report["decision_engine"] = decision_engine.report()
        except Exception:
            pass

        # ── Add cache manager stats ────────────────────────
        try:
            from core.cache_manager import cache_manager
            report["cache_manager"] = cache_manager.report()
        except Exception:
            pass

        # ── Add tool reliability stats ─────────────────────
        try:
            from core.tool_reliability import tool_reliability
            report["tool_reliability"] = tool_reliability.report()
        except Exception:
            pass

        # ── Add autonomous reasoning stats ─────────────────
        try:
            from core.autonomous_reasoning import auto_context
            report["autonomous_reasoning"] = auto_context.report()
        except Exception:
            pass

        # ── Add command router stats ───────────────────────
        try:
            from core.command_router import command_router
            report["command_router"] = command_router.report()
        except Exception:
            pass

        # ── Add background learner stats ───────────────────
        try:
            from core.background_learning import background_learner
            report["background_learner"] = background_learner.report()
        except Exception:
            pass

        return report

    def print_report(self) -> None:
        """Print a human-readable benchmark report."""
        r = self.report()
        print("\n" + "=" * 60)
        print("  LEO BENCHMARK REPORT")
        print("=" * 60)
        print(f"  Session:     {r['session']['id']}")
        print(f"  Uptime:      {r['session']['uptime_s']}s")
        print(f"  Total turns: {r['session']['total_turns']}")
        print()
        print("  ── LLM Usage ──")
        print(f"  LLM calls:   {r['llm_usage']['llm_turns']}")
        print(f"  Bypassed:    {r['llm_usage']['bypassed_turns']}")
        print(f"  LLM rate:    {r['llm_usage']['llm_usage_rate']}")
        print(f"  Bypass rate: {r['llm_usage']['llm_bypass_rate']}")
        print(f"  Reduction:   {r['llm_usage']['reduction_vs_baseline']}")
        print()
        print("  ── Latency ──")
        print(f"  Avg total:   {r['latency']['avg_total_ms']}ms")
        print(f"  Avg STT:     {r['latency']['avg_stt_ms']}ms")
        print(f"  Avg LLM:     {r['latency']['avg_llm_ms']}ms")
        print(f"  Avg TTS:     {r['latency']['avg_tts_ms']}ms")
        print()
        print("  ── Caching ──")
        print(f"  Cache hits:  {r['caching']['cache_hits']}")
        print(f"  Hit rate:    {r['caching']['cache_hit_rate']}")
        print()
        print("  ── Actions ──")
        print(f"  Executed:    {r['actions']['executed']}")
        print(f"  Succeeded:   {r['actions']['succeeded']}")
        print(f"  Success rate:{r['actions']['success_rate']}")
        print()
        if r['bottlenecks']:
            print("  ── Bottlenecks ──")
            for b in r['bottlenecks']:
                print(f"  {b['stage']}: {b['cumulative_ms']}ms cumulative")
        print()
        if r['files_modified']:
            print("  ── Files Modified ──")
            for f in r['files_modified']:
                print(f"  {f}")
        print("=" * 60)

    def save(self) -> None:
        """Persist benchmark data to disk."""
        try:
            BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "session_id": self._session_id,
                "start_time": self._start_time,
                "total_turns": self._total_turns,
                "llm_turns": self._llm_turns,
                "bypassed_turns": self._bypassed_turns,
                "cache_hits": self._cache_hits,
                "actions_executed": self._actions_executed,
                "actions_succeeded": self._actions_succeeded,
                "avg_latency_ms": self.avg_latency_ms,
                "avg_llm_latency_ms": self.avg_llm_latency_ms,
                "llm_bypass_rate": self.llm_bypass_rate,
                "files_modified": self._files_modified,
                "updated_at": time.time(),
            }
            with open(BENCHMARK_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug("[BENCH] Failed to save: %s", e)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load previous benchmark data."""
        try:
            if BENCHMARK_PATH.exists():
                with open(BENCHMARK_PATH, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return None


# Global singleton
benchmark = Benchmark()