"""
RuntimeStateMachine — Diego's production-grade runtime state machine.

Siri-style lifecycle (Problem 8):

    BOOT
      ↓
    LOAD MODELS
      ↓
    INITIALIZE SERVICES
      ↓
    IDLE
      ↓
    WAIT WAKE
      ↓
    WAKE DETECTED
      ↓
    FACE AUTH
      ↓
    GREETING
      ↓
    CONVERSATION
      ↓
    LISTENING
      ↓
    THINKING
      ↓
    EXECUTING
      ↓
    SPEAKING
      ↓
    FOLLOW-UP
      ↓
    LISTENING
      ↓
    … (repeat)
      ↓
    TIMEOUT
      ↓
    WAIT WAKE

HARD INVARIANT (Problem 6): the wake detector runs ONLY in WAIT_WAKE.
During conversation, thinking, speaking, face auth, or tool execution
the wake detector is NEVER active.

Design goals:
  * Every transition is validated against a legal transition table.
  * Any state can RECOVER after a subsystem failure — a single failure
    NEVER terminates Diego.
  * State changes are broadcast on the event bus (`runtime.state`).
  * A bounded history is retained for the runtime dashboard.
  * The machine is thread-safe (used from async event loop only).
  * Failure handlers per-state can be registered by the orchestrator.

The state machine itself does NOT own subsystems. It is the nervous
system — the services are the organs. The orchestrator registers
transition callbacks and failure handlers.
"""

import asyncio
import logging
import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from core.event_bus import bus

logger = logging.getLogger(__name__)


class RuntimeState(str, Enum):
    """The full runtime state vocabulary (Siri-style lifecycle)."""
    BOOT = "BOOT"
    LOAD_MODELS = "LOAD_MODELS"
    INITIALIZE_SERVICES = "INITIALIZE_SERVICES"
    FACE_AUTH = "FACE_AUTH"
    IDLE = "IDLE"
    WAIT_WAKE = "WAIT_WAKE"
    WAKE_DETECTED = "WAKE_DETECTED"
    GREETING = "GREETING"
    CONVERSATION = "CONVERSATION"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    FOLLOW_UP = "FOLLOW_UP"
    TIMEOUT = "TIMEOUT"
    RECOVERING = "RECOVERING"
    SHUTDOWN = "SHUTDOWN"


# ── Legal transitions ────────────────────────────────────────────
# The complete transition table. Any transition not listed here is a
# state-machine violation and is logged (but never crashes Leo).

ALLOWED_TRANSITIONS: Dict[RuntimeState, set] = {
    RuntimeState.BOOT: {
        RuntimeState.LOAD_MODELS,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.LOAD_MODELS: {
        RuntimeState.INITIALIZE_SERVICES,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.INITIALIZE_SERVICES: {
        RuntimeState.FACE_AUTH,
        RuntimeState.IDLE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.FACE_AUTH: {
        RuntimeState.IDLE,
        RuntimeState.GREETING,
        RuntimeState.WAIT_WAKE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.IDLE: {
        RuntimeState.WAIT_WAKE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.WAIT_WAKE: {
        RuntimeState.WAKE_DETECTED,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.WAKE_DETECTED: {
        RuntimeState.FACE_AUTH,
        RuntimeState.GREETING,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.GREETING: {
        RuntimeState.CONVERSATION,
        RuntimeState.IDLE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.CONVERSATION: {
        RuntimeState.LISTENING,
        RuntimeState.THINKING,
        RuntimeState.EXECUTING,
        RuntimeState.SPEAKING,
        RuntimeState.FOLLOW_UP,
        RuntimeState.IDLE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.LISTENING: {
        RuntimeState.THINKING,
        RuntimeState.EXECUTING,
        RuntimeState.SPEAKING,
        RuntimeState.TIMEOUT,
        RuntimeState.CONVERSATION,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.THINKING: {
        RuntimeState.SPEAKING,
        RuntimeState.EXECUTING,
        RuntimeState.CONVERSATION,
        RuntimeState.LISTENING,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.EXECUTING: {
        RuntimeState.SPEAKING,
        RuntimeState.CONVERSATION,
        RuntimeState.LISTENING,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.SPEAKING: {
        RuntimeState.FOLLOW_UP,
        RuntimeState.CONVERSATION,
        RuntimeState.LISTENING,
        RuntimeState.IDLE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.FOLLOW_UP: {
        RuntimeState.LISTENING,
        RuntimeState.CONVERSATION,
        RuntimeState.IDLE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.TIMEOUT: {
        RuntimeState.WAIT_WAKE,
        RuntimeState.RECOVERING,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.RECOVERING: {
        # A recovering state can return to ANY operational state.
        RuntimeState.BOOT,
        RuntimeState.LOAD_MODELS,
        RuntimeState.INITIALIZE_SERVICES,
        RuntimeState.FACE_AUTH,
        RuntimeState.IDLE,
        RuntimeState.WAIT_WAKE,
        RuntimeState.WAKE_DETECTED,
        RuntimeState.GREETING,
        RuntimeState.CONVERSATION,
        RuntimeState.LISTENING,
        RuntimeState.THINKING,
        RuntimeState.EXECUTING,
        RuntimeState.SPEAKING,
        RuntimeState.FOLLOW_UP,
        RuntimeState.TIMEOUT,
        RuntimeState.SHUTDOWN,
    },
    RuntimeState.SHUTDOWN: set(),
}


# A transition callback receives (old_state, new_state) and may be async.
StateCallback = Callable[..., Any]


class RuntimeStateMachine:
    """
    Thread-safe runtime state machine with recovery.

    Usage:
        sm = RuntimeStateMachine()
        sm.set_transition_callback("on_enter_WAIT_WAKE", handler)
        await sm.transition_to(RuntimeState.WAIT_WAKE)
        sm.on_failure(RuntimeState.WAIT_WAKE, recovery_handler)
    """

    def __init__(self):
        self._state: Optional[RuntimeState] = None
        self._state_entered: float = time.monotonic()
        self._lock = asyncio.Lock()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._transition_callbacks: Dict[str, List[StateCallback]] = {}
        self._failure_handlers: Dict[RuntimeState, List[StateCallback]] = {}
        self._state_timeouts: Dict[RuntimeState, float] = {}
        self._recovery_count: int = 0
        self._running = False

    # ── Properties ──────────────────────────────────────────

    @property
    def state(self) -> Optional[RuntimeState]:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.value if self._state else "NONE"

    @property
    def state_duration(self) -> float:
        return time.monotonic() - self._state_entered

    @property
    def running(self) -> bool:
        return self._running

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    # ── Registration ────────────────────────────────────────

    def on_transition(self, name: str, cb: StateCallback) -> None:
        """
        Register a transition callback.

        name is one of:
          "on_enter_<STATE>"  — when entering a state
          "on_exit_<STATE>"   — when leaving a state
          "on_any"            — every transition (old, new)
        """
        self._transition_callbacks.setdefault(name, []).append(cb)

    def on_failure(self, state: RuntimeState, cb: StateCallback) -> None:
        """Register a recovery handler for failures occurring in `state`."""
        self._failure_handlers.setdefault(state, []).append(cb)

    def set_state_timeout(self, state: RuntimeState, timeout_s: float) -> None:
        """Set a maximum duration for a state. Exceeding it triggers recovery."""
        self._state_timeouts[state] = timeout_s

    # ── Transitions ─────────────────────────────────────────

    async def transition_to(self, new_state: RuntimeState, reason: str = "") -> bool:
        """
        Transition to a new state. Returns True when the transition is valid
        and completed. Illegal transitions are logged and rejected without
        crashing the runtime.
        """
        async with self._lock:
            return await self._do_transition(new_state, reason)

    async def _do_transition(self, new_state: RuntimeState, reason: str) -> bool:
        old = self._state
        if old == new_state:
            return True

        if old is not None:
            allowed = ALLOWED_TRANSITIONS.get(old, set())
            if new_state not in allowed:
                logger.error(
                    "STATE VIOLATION: %s → %s is not legal — rejected.%s",
                    old.value, new_state.value,
                    f" reason='{reason}'" if reason else "")
                await self._emit_violation(old, new_state, reason)
                return False

        elapsed = time.monotonic() - self._state_entered
        if old is None:
            logger.info("STATE %s%s", new_state.value,
                        f" ({reason})" if reason else "")
        else:
            logger.info("STATE %s → %s (%.1fs)%s",
                        old.value, new_state.value, elapsed,
                        f" ({reason})" if reason else "")

        # Exit callbacks
        if old is not None:
            await self._run_callbacks(f"on_exit_{old.value}", old, new_state)

        self._state = new_state
        self._state_entered = time.monotonic()

        # History record
        self._history.append({
            "from": old.value if old else None,
            "to": new_state.value,
            "entered": self._state_entered,
            "duration": elapsed,
            "reason": reason,
        })

        # Emit on the bus (dashboard/supervision)
        try:
            await bus.emit("runtime.state", data={
                "from": old.value if old else None,
                "to": new_state.value,
                "reason": reason,
                "elapsed_s": elapsed,
                "recovery_count": self._recovery_count,
            }, source="state_machine")
        except Exception as e:
            logger.debug("[STATE] event emit error: %s", e)

        # Enter callbacks
        await self._run_callbacks(f"on_enter_{new_state.value}", old, new_state)
        try:
            await self._run_callbacks("on_any", old, new_state)
        except Exception as e:
            logger.debug("[STATE] on_any callback error: %s", e)
        return True

    async def _run_callbacks(self, name: str, old: RuntimeState, new: RuntimeState) -> None:
        for cb in self._transition_callbacks.get(name, []):
            try:
                result = cb(old, new)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("[STATE] callback %s failed: %s", name, e)

    # ── Recovery ────────────────────────────────────────────

    async def recover(self, from_state: RuntimeState, error: Exception, detail: str = "") -> RuntimeState:
        """
        Recover from a failure in `from_state`.

        1. Move to RECOVERING.
        2. Run registered failure handlers for `from_state`.
        3. The handlers (orchestrator) decide the return state.
        4. Returns the chosen recovery target state.

        Diego NEVER terminates because of one subsystem — this is the
        guarantee this machine enforces.
        """
        self._recovery_count += 1
        logger.warning("[STATE] Recovery #%d from %s: %s%s",
                       self._recovery_count, from_state.value, error,
                       f" ({detail})" if detail else "")

        await self._do_transition(RuntimeState.RECOVERING,
                                  f"recovery#{self._recovery_count} from {from_state.value}")

        target = from_state
        for cb in self._failure_handlers.get(from_state, []):
            try:
                result = cb(from_state, error)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    target = result  # handler chose the recovery target
            except Exception as e:
                logger.error("[STATE] recovery handler failed: %s", e)

        # The natural recovery target is the state we came from.
        await self._do_transition(target, f"recovered#{self._recovery_count}")
        return target

    # ── Timeout watchdog ────────────────────────────────────

    async def timeout_watchdog(self, on_timeout: Optional[StateCallback] = None) -> None:
        """
        Watch for states exceeding their configured timeout.

        If a timeout triggers and `on_timeout` is given, it is called with
        (old_state, current_state). Otherwise the machine transitions to
        RECOVERING automatically.
        """
        while self._running:
            await asyncio.sleep(0.5)
            if self._state is None:
                continue
            timeout = self._state_timeouts.get(self._state)
            if timeout is None:
                continue
            if self.state_duration > timeout:
                logger.warning(
                    "[STATE] %s exceeded timeout %.1fs — recovering",
                    self._state.value, timeout)
                if on_timeout is not None:
                    try:
                        result = on_timeout(self._state, self._state)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error("[STATE] timeout handler error: %s", e)
                else:
                    await self._do_transition(
                        RuntimeState.RECOVERING, f"timeout in {self._state.value}")

    # ── Lifecycle ───────────────────────────────────────────

    async def run(self, start_state: RuntimeState = RuntimeState.BOOT) -> None:
        """Enter the machine and keep the state loop alive."""
        self._running = True
        await self.transition_to(start_state, "runtime start")
        try:
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    async def shutdown(self) -> None:
        """Transition to SHUTDOWN and stop."""
        self._running = False
        await self.transition_to(RuntimeState.SHUTDOWN, "runtime shutdown")

    # ── Internals ───────────────────────────────────────────

    async def _emit_violation(self, old: RuntimeState, new: RuntimeState, reason: str) -> None:
        try:
            await bus.emit("runtime.state_violation", data={
                "from": old.value, "to": new.value, "reason": reason,
            }, source="state_machine")
        except Exception:
            pass


# Global singleton
runtime_state_machine = RuntimeStateMachine()
