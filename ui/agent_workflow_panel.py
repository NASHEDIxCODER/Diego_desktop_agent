"""
AgentWorkflowPanel — live view of Diego's computer-use workflow (Phase 22).

Renders the SAME structured truth the runtime records:

    GOAL → PLAN → STEP → OBSERVE → ACT → VERIFY → RECOVERY → COMPLETION

Data source: agent.trace.agent_trace (the single structured store). The
panel NEVER parses logs and NEVER imports the pipeline — it subscribes to
trace events and renders the derived WorkflowSnapshot.

Threading: subscribe() callbacks run on the RECORDING thread, so events are
pushed into a thread-safe queue and drained on a Qt QTimer (the same
pattern EventBridge uses). The Qt event loop is never blocked.

Additive module: nothing here executes actions.
"""
from __future__ import annotations

import queue
import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from agent.trace import (
    STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING,
    STATUS_WAITING_CONFIRMATION, agent_trace,
)
from agent.trace_event import TraceEvent

try:  # UI tokens exist; degrade gracefully if they change
    from ui.tokens import SPACING_SMALL, SPACING_MEDIUM
except Exception:  # pragma: no cover
    SPACING_SMALL = SPACING_MEDIUM = 8

_STATUS_COLOR = {
    STATUS_RUNNING: "#7aa2f7",
    STATUS_COMPLETED: "#9ece6a",
    STATUS_FAILED: "#f7768e",
    STATUS_WAITING_CONFIRMATION: "#e0af68",
}

# Phase 23: operational phase colors (concise states, never chain-of-thought).
_PHASE_COLOR = {
    "THINKING": "#7aa2f7",
    "PLANNING": "#7aa2f7",
    "OBSERVING": "#7dcfff",
    "ACTING": "#7aa2f7",
    "WAITING": "#e0af68",
    "VERIFYING": "#9ece6a",
    "RECOVERING": "#e0af68",
    "ASKING_USER": "#e0af68",
    "COMPLETED": "#9ece6a",
    "FAILED": "#f7768e",
}

_MAX_LOG_ROWS = 120


class AgentWorkflowPanel(QWidget):
    """Right-column panel: the live GOAL→...→COMPLETION workflow."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._queue: "queue.Queue[TraceEvent]" = queue.Queue()
        self._unsubscribe = agent_trace.subscribe(self._on_trace_event)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_SMALL)

        self._title = QLabel("AGENT WORKFLOW")
        self._title.setStyleSheet(
            f"color: #a9b1d6; font-size: 10px; letter-spacing: 1px; "
            f"padding-left: {SPACING_SMALL}px;")

        self._status = QLabel("IDLE")
        self._status.setStyleSheet("color: #565f89; font-size: 11px;")

        self._goal = QLabel("—")
        self._goal.setWordWrap(True)
        self._goal.setStyleSheet(
            "color: #c0caf5; font-size: 12px; font-weight: 600;")

        self._stage = QLabel("")            # current step / action line
        self._stage.setWordWrap(True)
        self._stage.setStyleSheet("color: #7aa2f7; font-size: 11px;")

        self._verify = QLabel("")           # observation + verification line
        self._verify.setWordWrap(True)
        self._verify.setStyleSheet("color: #9ece6a; font-size: 11px;")

        self._plan_list = QListWidget()
        self._plan_list.setSelectionMode(
            QListWidget.SelectionMode.NoSelection)
        self._plan_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._plan_list.setStyleSheet(
            "QListWidget { background: transparent; border: none; "
            "color: #565f89; font-size: 11px; }"
            "QListWidget::item { padding: 1px 4px; }")

        self._log = QListWidget()
        self._log.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._log.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._log.setStyleSheet(
            "QListWidget { background: transparent; border: none; "
            "color: #565f89; font-size: 10px; }"
            "QListWidget::item { padding: 0px 4px; }")

        layout.addWidget(self._title)
        layout.addWidget(self._status)
        layout.addWidget(self._goal)
        layout.addWidget(self._stage)
        layout.addWidget(self._verify)
        layout.addWidget(self._plan_list, 1)
        layout.addWidget(self._log, 2)

        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    # ── Trace subscription (recording thread) ────────────────────

    def _on_trace_event(self, event: TraceEvent) -> None:
        self._queue.put(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        try:
            self._unsubscribe()
        except Exception:
            pass
        super().closeEvent(event)

    # ── GUI-thread update ────────────────────────────────────────

    def _drain(self) -> None:
        changed = False
        try:
            while True:
                event = self._queue.get_nowait()
                self._append_log(event)
                changed = True
        except queue.Empty:
            pass
        if changed or self._log.count() == 0:
            self._refresh_snapshot()

    def _refresh_snapshot(self) -> None:
        try:
            snap = agent_trace.snapshot()
        except Exception:
            return

        color = _STATUS_COLOR.get(snap.status, "#565f89")
        # Phase 23: show the coarse OPERATIONAL phase (THINKING…COMPLETED)
        # next to the task status — concise operational reasoning only.
        if snap.phase and snap.phase not in ("IDLE", snap.status):
            phase_color = _PHASE_COLOR.get(snap.phase, "#565f89")
            self._status.setText(f"{snap.status} · {snap.phase}")
            self._status.setStyleSheet(
                f"color: {phase_color}; font-size: 11px; font-weight: 600;")
        else:
            self._status.setText(snap.status)
            self._status.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: 600;")
        if snap.phase_detail:
            self._goal.setToolTip(f"phase: {snap.phase_detail[:160]}")

        goal = snap.goal or "—"
        self._goal.setText(goal[:160])

        stage_bits: List[str] = []
        if snap.phase_detail:
            stage_bits.append(snap.phase_detail[:90])
        if snap.current_step:
            stage_bits.append(
                f"step {snap.current_step_index + 1}/"
                f"{max(snap.total_steps, 1)}: {snap.current_step}")
        if snap.action:
            stage_bits.append(
                f"act {snap.action}"
                + (f" '{snap.target[:40]}'" if snap.target else "")
                + (f" [{snap.method}]" if snap.method else ""))
        if snap.recovery:
            stage_bits.append(f"recover: {snap.recovery}")
        self._stage.setText("  ·  ".join(stage_bits))
        self._stage.setStyleSheet(
            f"color: {'#e0af68' if snap.recovery else '#7aa2f7'}; "
            f"font-size: 11px;")

        verify_bits: List[str] = []
        if snap.observation:
            verify_bits.append(f"observed: {snap.observation[:80]}")
        if snap.verification:
            verify_bits.append(f"verify: {snap.verification}")
        if snap.confirmation_required:
            verify_bits.append("awaiting your confirmation")
        if snap.final_result:
            verify_bits.append(snap.final_result[:120])
        self._verify.setText("  |  ".join(verify_bits))

        if snap.plan and self._plan_list.count() != len(snap.plan):
            self._plan_list.clear()
            for step in snap.plan[:20]:
                QListWidgetItem(str(step)[:90], self._plan_list)
        for i in range(self._plan_list.count()):
            item = self._plan_list.item(i)
            done = i < snap.current_step_index
            current = i == snap.current_step_index and snap.status == STATUS_RUNNING
            if current:
                item.setForeground(Qt.GlobalColor.white)
            elif done:
                item.setForeground(Qt.GlobalColor.darkGreen)
            else:
                item.setForeground(Qt.GlobalColor.gray)

    def _append_log(self, event: TraceEvent) -> None:
        text = self._format(event)
        if not text:
            return
        QListWidgetItem(text, self._log)
        while self._log.count() > _MAX_LOG_ROWS:
            self._log.takeItem(0)
        self._log.scrollToBottom()

    @staticmethod
    def _format(event: TraceEvent) -> str:
        et = event.event_type.value
        bits = [time.strftime("%H:%M:%S", time.localtime(
            event.timestamp or time.time())), et.replace("TASK_", "")]
        if event.action:
            bits.append(f"{event.action}"
                        + (f" '{event.target[:36]}'" if event.target else ""))
        if event.method:
            bits.append(f"[{event.method}]")
        if event.verification_result:
            bits.append(f"→ {event.verification_result}")
        if event.detail:
            bits.append(str(event.detail)[:70])
        return " ".join(bits)

