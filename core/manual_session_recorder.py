"""
ManualSessionRecorder — Real-world validation recording for every conversation turn.

Enabled via:
    DIEGO_RECORD_SESSION=1 python main.py
    python main.py --record-session

Records per-interaction metrics:
    - wake latency (ms)
    - wake confidence (openWakeWord score)
    - face auth latency (ms)
    - speech duration (ms)
    - endpoint reason (silence / timeout / interruption)
    - whisper transcript
    - transcript confidence (avg_logprob)
    - command classification (router path)
    - action selected
    - action verification result
    - total turn latency (ms)

Saves to: logs/manual_voice_session.json

Usage:
    from core.manual_session_recorder import session_recorder

    # At wake:
    session_recorder.record_wake(latency_ms=320, confidence=0.87)

    # At face auth:
    session_recorder.record_face_auth(latency_ms=450, success=True)

    # At utterance final:
    session_recorder.record_utterance(
        transcript="open firefox",
        confidence=0.95,
        speech_duration_ms=1200,
        endpoint_reason="silence",
    )

    # At decision:
    session_recorder.record_decision(
        classification="DIRECT_EXECUTION",
        action={"type": "desktop_open", "app": "firefox"},
    )

    # At action verification:
    session_recorder.record_verification(success=True, detail="Window appeared")

    # At turn end:
    session_recorder.record_turn_end(total_latency_ms=1800)

    # On shutdown:
    session_recorder.save()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
SESSION_PATH = LOGS_DIR / "manual_voice_session.json"


@dataclass
class TurnRecord:
    """A single complete interaction turn."""
    turn_id: int = 0
    timestamp_iso: str = ""

    # Wake
    wake_latency_ms: float = 0.0
    wake_confidence: float = 0.0
    wake_model: str = ""
    wake_transcript: str = ""

    # Face auth
    face_auth_latency_ms: float = 0.0
    face_auth_success: bool = False
    face_auth_user: str = ""

    # Speech / STT
    speech_duration_ms: float = 0.0
    endpoint_reason: str = ""
    whisper_transcript: str = ""
    transcript_confidence: float = 0.0
    whisper_latency_ms: float = 0.0

    # Decision / routing
    command_classification: str = ""       # DecisionPath value
    decision_confidence: float = 0.0
    decision_latency_us: float = 0.0
    llm_used: bool = False

    # Action
    action_selected: Optional[Dict[str, Any]] = None
    actions_selected: Optional[List[Dict[str, Any]]] = None

    # Verification
    action_verification_result: str = ""   # "verified" | "no_change" | "error" | ""
    action_verification_detail: str = ""

    # Response
    response_text: str = ""
    tts_latency_ms: float = 0.0

    # Total
    total_turn_latency_ms: float = 0.0

    # Post-turn checklist (filled by human or automated check)
    checklist: Dict[str, Optional[bool]] = field(default_factory=lambda: {
        "transcript_correct": None,
        "planner_correct": None,
        "action_executed": None,
        "verification_succeeded": None,
        "response_natural": None,
        "latency_acceptable": None,
    })

    # Raw diagnostics
    extra: Dict[str, Any] = field(default_factory=dict)


class ManualSessionRecorder:
    """Records every conversation turn for real-world validation."""

    def __init__(self):
        self._enabled: bool = False
        self._lock = threading.Lock()
        self._turns: List[TurnRecord] = []
        self._current_turn: Optional[TurnRecord] = None
        self._turn_counter: int = 0
        self._session_started: float = 0.0
        self._turn_started: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────

    def enable(self) -> None:
        """Enable session recording."""
        self._enabled = True
        self._session_started = time.time()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[RECORDER] Manual session recording ENABLED → %s", SESSION_PATH)

    def disable(self) -> None:
        """Disable session recording."""
        self._enabled = False
        logger.info("[RECORDER] Manual session recording DISABLED")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    # ── Per-turn recording ─────────────────────────────────────

    def new_turn(self) -> None:
        """Start a new turn record."""
        if not self._enabled:
            return
        with self._lock:
            self._turn_counter += 1
            self._turn_started = time.time()
            self._current_turn = TurnRecord(
                turn_id=self._turn_counter,
                timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            )

    def record_wake(self, latency_ms: float, confidence: float,
                    model: str = "", transcript: str = "") -> None:
        """Record wake word detection metrics."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            self._current_turn.wake_latency_ms = latency_ms
            self._current_turn.wake_confidence = confidence
            self._current_turn.wake_model = model
            self._current_turn.wake_transcript = transcript

    def record_face_auth(self, latency_ms: float, success: bool,
                         user: str = "") -> None:
        """Record face authentication metrics."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            self._current_turn.face_auth_latency_ms = latency_ms
            self._current_turn.face_auth_success = success
            self._current_turn.face_auth_user = user

    def record_utterance(self, transcript: str, confidence: float,
                         speech_duration_ms: float, endpoint_reason: str,
                         whisper_latency_ms: float = 0.0) -> None:
        """Record STT / utterance metrics."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            self._current_turn.whisper_transcript = transcript
            self._current_turn.transcript_confidence = confidence
            self._current_turn.speech_duration_ms = speech_duration_ms
            self._current_turn.endpoint_reason = endpoint_reason
            self._current_turn.whisper_latency_ms = whisper_latency_ms

    def record_decision(self, classification: str, confidence: float = 1.0,
                        latency_us: float = 0.0, llm_used: bool = False,
                        action: Optional[Dict[str, Any]] = None,
                        actions: Optional[List[Dict[str, Any]]] = None) -> None:
        """Record decision / routing metrics."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            self._current_turn.command_classification = classification
            self._current_turn.decision_confidence = confidence
            self._current_turn.decision_latency_us = latency_us
            self._current_turn.llm_used = llm_used
            self._current_turn.action_selected = action
            self._current_turn.actions_selected = actions

    def record_verification(self, success: bool, detail: str = "") -> None:
        """Record action verification result."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            self._current_turn.action_verification_result = (
                "verified" if success else "no_change"
            )
            self._current_turn.action_verification_detail = detail

    def record_response(self, text: str, tts_latency_ms: float = 0.0) -> None:
        """Record TTS response metrics."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            self._current_turn.response_text = text
            self._current_turn.tts_latency_ms = tts_latency_ms

    def record_turn_end(self, total_latency_ms: float = 0.0) -> None:
        """Finalize the current turn and append to session."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            if total_latency_ms == 0.0:
                total_latency_ms = (time.time() - self._turn_started) * 1000.0
            self._current_turn.total_turn_latency_ms = total_latency_ms
            self._turns.append(self._current_turn)
            logger.info(
                "[RECORDER] Turn #%d recorded: '%s' → %s (%.0fms)",
                self._current_turn.turn_id,
                self._current_turn.whisper_transcript,
                self._current_turn.command_classification,
                total_latency_ms,
            )
            self._current_turn = None

    def record_extra(self, key: str, value: Any) -> None:
        """Record arbitrary diagnostic data for the current turn."""
        if not self._enabled or self._current_turn is None:
            return
        with self._lock:
            self._current_turn.extra[key] = value

    # ── Persistence ────────────────────────────────────────────

    def save(self) -> Optional[Path]:
        """Save the session to logs/manual_voice_session.json."""
        if not self._enabled:
            return None
        with self._lock:
            data = {
                "session_id": str(int(self._session_started)),
                "session_started_iso": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(self._session_started)
                ),
                "session_ended_iso": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime()
                ),
                "total_turns": len(self._turns),
                "turns": [asdict(t) for t in self._turns],
            }
        try:
            SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
            SESSION_PATH.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info("[RECORDER] Session saved: %d turns → %s",
                        len(self._turns), SESSION_PATH)
            return SESSION_PATH
        except Exception as e:
            logger.error("[RECORDER] Failed to save session: %s", e)
            return None

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the recorded session."""
        with self._lock:
            turns = list(self._turns)
        if not turns:
            return {"total_turns": 0}

        wake_latencies = [t.wake_latency_ms for t in turns if t.wake_latency_ms > 0]
        total_latencies = [t.total_turn_latency_ms for t in turns if t.total_turn_latency_ms > 0]
        classifications = {}
        for t in turns:
            c = t.command_classification or "unknown"
            classifications[c] = classifications.get(c, 0) + 1

        return {
            "total_turns": len(turns),
            "avg_wake_latency_ms": sum(wake_latencies) / len(wake_latencies) if wake_latencies else 0,
            "avg_total_latency_ms": sum(total_latencies) / len(total_latencies) if total_latencies else 0,
            "classifications": classifications,
            "llm_turns": sum(1 for t in turns if t.llm_used),
            "bypassed_turns": sum(1 for t in turns if not t.llm_used),
        }


# Global singleton
session_recorder = ManualSessionRecorder()