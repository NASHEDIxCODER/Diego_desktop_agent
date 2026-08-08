"""
ResponseGuarantee — Leo's "never silent" layer.

Guarantees that every completed user utterance receives a spoken
response. No exceptions.

PRIORITY ORDER (per turn):
  1. Real response (from Brain/LLM)
  2. Recovery response (action failed, verification failed, LLM error)
  3. Generic fallback ("I didn't catch that.")

WATCHDOG LOGGING:
  For every turn, logs:
    turn_id, transcript, brain_result, planner_result, dispatcher_result,
    verification_result, llm_result, tts_started, tts_finished, response_length

  If response_length == 0 OR tts never started:
    [FATAL] SILENT TURN with complete pipeline diagnostics.

USAGE:
    from core.response_guarantee import response_guarantee

    # Wrap the whole turn:
    await response_guarantee.run_turn(
        transcript=text,
        process_fn=lambda: agent_brain.process_command(text),
        speak_fn=lambda response: engine._think_and_speak(response, events),
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ── Generic fallbacks (priority 3) ─────────────────────────────
GENERIC_FALLBACKS = [
    "I didn't catch that.",
    "Could you repeat that?",
    "I'm not sure I understood.",
    "Something went wrong.",
    "Let me try another way.",
    "I'm still working on it.",
    "I lost the conversation context.",
]

# ── Recovery responses (priority 2) ────────────────────────────
RECOVERY_RESPONSES = [
    "I couldn't do that.",
    "I couldn't verify that.",
    "Something went wrong with that.",
    "I ran into a problem there.",
    "That didn't work as expected.",
]


@dataclass
class TurnDiagnostics:
    """Per-turn pipeline diagnostics for the watchdog."""
    turn_id: str = ""
    transcript: str = ""
    brain_result: str = ""
    planner_result: str = ""
    dispatcher_result: str = ""
    verification_result: str = ""
    llm_result: str = ""
    tts_started: bool = False
    tts_finished: bool = False
    response_length: int = 0
    error: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "brain_result": self.brain_result,
            "planner_result": self.planner_result,
            "dispatcher_result": self.dispatcher_result,
            "verification_result": self.verification_result,
            "llm_result": self.llm_result,
            "tts_started": self.tts_started,
            "tts_finished": self.tts_finished,
            "response_length": self.response_length,
            "error": self.error,
            "duration_ms": round((self.ended_at - self.started_at) * 1000, 1)
            if self.started_at and self.ended_at else 0.0,
        }


class ResponseGuarantee:
    """
    The "never silent" layer.

    Wraps the full turn (process → speak) and guarantees a spoken
    response. If any stage fails, it falls back to a recovery or
    generic response. If TTS itself fails, it retries with a shorter
    fallback. If EVERYTHING fails, it logs [FATAL] SILENT TURN.
    """

    def __init__(self):
        self._last_diag: Optional[TurnDiagnostics] = None
        self._silent_turns: int = 0
        self._total_turns: int = 0

    # ── Public API ─────────────────────────────────────────

    async def run_turn(
        self,
        transcript: str,
        process_fn: Callable[[], Awaitable[Any]],
        speak_fn: Callable[[str], Awaitable[bool]],
    ) -> bool:
        """
        Run one complete turn with a guaranteed spoken response.

        Args:
            transcript: The user's spoken text.
            process_fn: Async callable that processes the command and
                        returns a CommandResult (or raises).
            speak_fn: Async callable that speaks a response string and
                      returns True if speech actually started.

        Returns:
            True if a response was spoken, False if even the fallback
            failed (should be impossible — logs FATAL).
        """
        self._total_turns += 1
        diag = TurnDiagnostics(
            turn_id=str(uuid.uuid4())[:8],
            transcript=transcript,
            started_at=time.time(),
        )
        self._last_diag = diag

        # ── Stage 1: Process (Brain) ──────────────────────
        result = None
        try:
            result = await process_fn()
            if result is not None:
                diag.brain_result = getattr(result, "path", "") or "BRAIN"
                diag.planner_result = (
                    "ok" if getattr(result, "actions_executed", 0) > 0 else "none"
                )
                diag.dispatcher_result = (
                    f"{getattr(result, 'actions_executed', 0)} executed / "
                    f"{getattr(result, 'actions_failed', 0)} failed"
                )
                diag.verification_result = (
                    "verified" if getattr(result, "verified", False) else "unverified"
                )
                diag.llm_result = "used" if getattr(result, "used_llm", False) else "not_used"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            diag.error = f"process: {e}"
            logger.error("[GUARANTEE] Brain process failed: %s", e)
            result = None

        # ── Stage 2: Select response (priority order) ─────
        response = self._select_response(result)

        # ── Stage 3: Speak with guarantee ─────────────────
        spoken = await self._speak_with_guarantee(response, speak_fn, diag)

        diag.ended_at = time.time()
        self._log_watchdog(diag)
        return spoken

    # ── Response selection ────────────────────────────────

    def _select_response(self, result: Any) -> str:
        """Priority 1: real response. Priority 2: recovery. Priority 3: generic."""
        # Priority 1: real response from the Brain
        if result is not None:
            response = getattr(result, "response", "") or ""
            if response and response.strip():
                return response.strip()
            # Check followup
            followup = getattr(result, "followup_response", "") or ""
            if followup and followup.strip():
                return followup.strip()
            # Action failed → recovery
            if getattr(result, "actions_failed", 0) > 0:
                return self._recovery_response()
            # No response at all → generic
            return self._generic_fallback()

        # Process failed → recovery
        return self._recovery_response()

    def _recovery_response(self) -> str:
        """Priority 2: recovery response."""
        return random.choice(RECOVERY_RESPONSES)

    def _generic_fallback(self) -> str:
        """Priority 3: generic fallback."""
        return random.choice(GENERIC_FALLBACKS)

    # ── Speech with guarantee ─────────────────────────────

    async def _speak_with_guarantee(
        self,
        response: str,
        speak_fn: Callable[[str], Awaitable[bool]],
        diag: TurnDiagnostics,
    ) -> bool:
        """Speak the response, retrying with fallbacks if TTS fails."""
        # Ensure we have a non-empty response
        if not response or not response.strip():
            response = self._generic_fallback()

        diag.response_length = len(response)

        # Attempt 1: speak the real response
        try:
            diag.tts_started = True
            ok = await speak_fn(response)
            diag.tts_finished = ok
            if ok:
                return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            diag.error = f"speak: {e}"
            logger.error("[GUARANTEE] Speak failed: %s", e)

        # Attempt 2: recovery response (shorter, more likely to work)
        recovery = self._recovery_response()
        diag.response_length = len(recovery)
        try:
            ok = await speak_fn(recovery)
            diag.tts_finished = ok
            if ok:
                return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            diag.error = f"speak recovery: {e}"
            logger.error("[GUARANTEE] Recovery speak failed: %s", e)

        # Attempt 3: generic fallback (shortest)
        fallback = self._generic_fallback()
        diag.response_length = len(fallback)
        try:
            ok = await speak_fn(fallback)
            diag.tts_finished = ok
            if ok:
                return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            diag.error = f"speak fallback: {e}"
            logger.error("[GUARANTEE] Fallback speak failed: %s", e)

        # Everything failed — log FATAL
        self._silent_turns += 1
        diag.tts_started = False
        diag.tts_finished = False
        logger.critical(
            "[FATAL] SILENT TURN — all speech attempts failed. "
            "turn_id=%s transcript=%r error=%s",
            diag.turn_id, diag.transcript, diag.error,
        )
        return False

    # ── Watchdog logging ──────────────────────────────────

    def _log_watchdog(self, diag: TurnDiagnostics) -> None:
        """Log the full pipeline diagnostics for this turn."""
        if diag.response_length == 0 or not diag.tts_started:
            self._silent_turns += 1
            logger.critical(
                "[FATAL] SILENT TURN — turn_id=%s transcript=%r "
                "brain=%s planner=%s dispatcher=%s verification=%s llm=%s "
                "tts_started=%s tts_finished=%s response_length=%d error=%s",
                diag.turn_id, diag.transcript,
                diag.brain_result, diag.planner_result, diag.dispatcher_result,
                diag.verification_result, diag.llm_result,
                diag.tts_started, diag.tts_finished, diag.response_length,
                diag.error,
            )
        else:
            logger.info(
                "[GUARANTEE] Turn %s: transcript=%r brain=%s planner=%s "
                "dispatcher=%s verification=%s llm=%s tts_started=%s "
                "tts_finished=%s response_length=%d",
                diag.turn_id, diag.transcript,
                diag.brain_result, diag.planner_result, diag.dispatcher_result,
                diag.verification_result, diag.llm_result,
                diag.tts_started, diag.tts_finished, diag.response_length,
            )

    # ── Stats ─────────────────────────────────────────────

    @property
    def silent_turns(self) -> int:
        return self._silent_turns

    @property
    def total_turns(self) -> int:
        return self._total_turns

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "total_turns": self._total_turns,
            "silent_turns": self._silent_turns,
            "last_turn": self._last_diag.to_dict() if self._last_diag else None,
        }


# Global singleton
response_guarantee = ResponseGuarantee()