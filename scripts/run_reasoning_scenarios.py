#!/usr/bin/env python3
"""
Phase 21A — REAL autonomous scenarios (Task 17).

Runs four safe, real scenarios through the ReasoningAgent + TaskRunner
with a REAL executor (subprocess / stdlib — no mocks for execution) and
the real reasoning model when a local Ollama is reachable (otherwise the
agent falls back to deterministic planning, which is itself a validated
Phase 21A behavior).

    A  Open Firefox and then open the file manager.
    B  Find a file in my workspace and report its size.
    C  Search for a project file, inspect it, and tell me what it does.
    D  A small task that intentionally causes a recoverable failure.

No destructive filesystem operations are performed (read-only file access
+ application launches only). For every scenario the report records:
goal → plan → steps → observations → verification → recovery/replan →
final result → lessons generated.

Usage:  .venv/bin/python scripts/run_reasoning_scenarios.py [A B C D]
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.reasoning_agent import ReasoningAgent  # noqa: E402
from agent.reasoning_context import ReasoningContextComposer  # noqa: E402
from agent.lessons import TaskLessonStore  # noqa: E402
from agent.task_state import TaskLimits  # noqa: E402
from ai.reasoning_model import get_reasoning_model  # noqa: E402

WORKSPACE = str(ROOT)
REPORT_JSON = ROOT / "data" / "scenarios.json"
REPORT_MD = ROOT / "docs" / "PHASE21A_SCENARIOS.md"


class RealExecutor:
    """REAL execution of safe operations only (no mocks).

    desktop_open / open_folder launch real applications (Popen with a
    session detach). find_file / file_info / read_file_head are REAL
    read-only filesystem operations.
    """

    def __init__(self, workspace: str = WORKSPACE):
        self.workspace = workspace
        self.last_find: Optional[str] = None

    async def __call__(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        name = action.get("action", "")
        params = action.get("params") or {}
        try:
            if name == "desktop_open":
                return self._desktop_open(str(params.get("app", "")))
            if name == "open_folder":
                return await self._open_folder(str(params.get("path", "")))
            if name == "find_file":
                return self._find_file(str(params.get("pattern", "")))
            if name == "file_info":
                return self._file_info(str(params.get("path", "")))
            if name == "read_file_head":
                return self._read_file_head(str(params.get("path", "")))
            return False, f"action '{name}' not executable in this scenario " \
                          "runner (read-only/safe set only)"
        except Exception as e:  # honest failure — never a fake success
            return False, f"execution error: {e}"

    # ── Real implementations ──────────────────────────────────

    def _desktop_open(self, app: str) -> Tuple[bool, str]:
        if not app:
            return False, "no app specified"
        path = shutil.which(app)
        if path is None:
            return False, f"'{app}' is not installed on this system"
        try:
            subprocess.Popen(
                [path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            return True, f"launched {app} (pid group started)"
        except OSError as e:
            return False, f"failed to launch {app}: {e}"

    async def _open_folder(self, path: str) -> Tuple[bool, str]:
        target = os.path.expanduser(path or self.workspace)
        if not os.path.isdir(target):
            return False, f"folder not found: {target}"
        opener = shutil.which("xdg-open") or shutil.which("nautilus")
        if opener is None:
            return False, "no file manager available to open the folder"
        try:
            subprocess.Popen(
                [opener, target],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            return True, f"opened folder {target} in the file manager"
        except OSError as e:
            return False, f"failed to open folder: {e}"

    def _find_file(self, pattern: str) -> Tuple[bool, str]:
        if not pattern:
            return False, "no file pattern given"
        skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules",
                     "data", ".pytest_cache"}
        matches: List[str] = []
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                if f == pattern or f.endswith(pattern):
                    matches.append(os.path.join(root, f))
                if len(matches) >= 5:
                    break
            if len(matches) >= 5:
                break
        if not matches:
            return False, f"no file matching '{pattern}' in the workspace"
        self.last_find = matches[0]
        sizes = ", ".join(
            f"{m.replace(self.workspace + os.sep, '')} "
            f"({os.path.getsize(m)} bytes)" for m in matches[:3])
        return True, f"found: {sizes}"

    def _file_info(self, path: str) -> Tuple[bool, str]:
        target = path if os.path.isabs(path) else os.path.join(
            self.workspace, path)
        if not os.path.exists(target):
            return False, f"file not found: {target}"
        st = os.stat(target)
        return True, (f"{target}: {st.st_size} bytes, modified "
                      f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")

    def _read_file_head(self, path: str) -> Tuple[bool, str]:
        target = path if os.path.isabs(path) else os.path.join(
            self.workspace, path)
        if not os.path.isfile(target):
            return False, f"file not found: {target}"
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            lines = [fh.readline().rstrip() for _ in range(12)]
        head = " | ".join(l for l in lines if l.strip())[:400]
        return True, f"{target} begins with: {head}"


class RealObserver:
    """REAL observation: which known desktop apps are actually running."""

    _APPS = ("firefox", "chromium", "chrome", "nautilus", "code", "gedit")

    async def __call__(self) -> str:
        running = []
        for app in self._APPS:
            try:
                r = subprocess.run(["pgrep", "-x", app],
                                   capture_output=True, timeout=5)
                if r.returncode == 0:
                    running.append(app)
            except Exception:
                pass
        return ("running apps: " + ", ".join(running)) if running \
            else "no known desktop apps running"


def register_scenario_tools() -> None:
    """Register the REAL read-only filesystem tools used by scenarios
    B/C/D in the canonical ToolRegistry (PlanValidator's source of
    truth). Handlers are real stdlib implementations — no mocks."""
    from core.tool_registry import Tool, tool_registry
    skip = {".git", ".venv", "venv", "__pycache__", "node_modules",
            "data", ".pytest_cache"}

    def find_file(params):
        pattern = str((params or {}).get("pattern", ""))
        if not pattern:
            return "no file pattern given"
        matches: List[str] = []
        for root, dirs, files in os.walk(WORKSPACE):
            dirs[:] = [d for d in dirs if d not in skip]
            for f in files:
                if f == pattern or f.endswith(pattern):
                    matches.append(os.path.join(root, f))
                if len(matches) >= 5:
                    break
            if len(matches) >= 5:
                break
        if not matches:
            return f"no file matching '{pattern}' in the workspace"
        listing = ", ".join(
            f"{m.replace(WORKSPACE + os.sep, '')} "
            f"({os.path.getsize(m)} bytes)" for m in matches[:3])
        return f"found: {listing}"

    def file_info(params):
        path = str((params or {}).get("path", ""))
        target = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
        if not os.path.exists(target):
            return f"file not found: {target}"
        st = os.stat(target)
        return (f"{target}: {st.st_size} bytes, modified "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")

    def read_file_head(params):
        path = str((params or {}).get("path", ""))
        target = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
        if not os.path.isfile(target):
            return f"file not found: {target}"
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            lines = [fh.readline().rstrip() for _ in range(12)]
        head = " | ".join(l for l in lines if l.strip())[:400]
        return f"{target} begins with: {head}"

    if not tool_registry.is_available("find_file"):
        tool_registry.register(Tool(
            name="find_file",
            description="Find a file by name in the Diego workspace "
                        "(read-only).",
            parameters={"pattern": {"type": "string"}},
            handler=find_file))
    if not tool_registry.is_available("file_info"):
        tool_registry.register(Tool(
            name="file_info",
            description="Report real size/mtime of a file (read-only).",
            parameters={"path": {"type": "string"}},
            handler=file_info))
    if not tool_registry.is_available("read_file_head"):
        tool_registry.register(Tool(
            name="read_file_head",
            description="Read the first lines of a file (read-only).",
            parameters={"path": {"type": "string"}},
            handler=read_file_head))


# ═══════════════════════════════════════════════════════════════
# Scenario plans (deterministic; the model may diagnose/revise when live)
# ═══════════════════════════════════════════════════════════════

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "A": {
        "goal": "Open Firefox and then open the file manager.",
        "plans": [
            [{"action": "desktop_open", "params": {"app": "firefox"},
              "description": "launch the Firefox browser"},
             {"action": "open_folder", "params": {"path": "~/Documents"},
              "description": "open the file manager on Documents"}],
            # recovery: ESR/alternative build if firefox is missing
            [{"action": "desktop_open", "params": {"app": "firefox-esr"},
              "description": "try the ESR build"},
             {"action": "open_folder", "params": {"path": "~/Documents"}}],
        ],
    },
    "B": {
        "goal": "Find a file in my workspace and report its size.",
        "plans": [
            [{"action": "find_file", "params": {"pattern": "context_monitor.py"},
              "description": "locate context_monitor.py in the workspace"},
             {"action": "file_info", "params": {"path": "ai/context_monitor.py"},
              "description": "report its real size on disk"}],
        ],
    },
    "C": {
        "goal": "Search for a project file, inspect it, and tell me what it does.",
        "plans": [
            [{"action": "find_file", "params": {"pattern": "event_bus.py"},
              "description": "locate the event bus module"},
             {"action": "read_file_head", "params": {"path": "core/event_bus.py"},
              "description": "inspect the first lines to describe it"}],
        ],
    },
    "D": {
        "goal": "Report the size of the event bus file (start from a wrong path "
                "on purpose so the agent must recover).",
        "plans": [
            [{"action": "file_info", "params": {"path": "core/does_not_exist_eventbus.py"},
              "description": "INTENTIONAL recoverable failure: wrong path"}],
            # recovery: locate the real file first, then report its size
            [{"action": "find_file", "params": {"pattern": "event_bus.py"},
              "description": "recover: locate the real file"},
             {"action": "file_info", "params": {"path": "core/event_bus.py"},
              "description": "report its size"}],
        ],
    },
}


class ScenarioPlanner:
    """Deterministic scripted planner (plans pop per call)."""

    def __init__(self, plans: List[Optional[List[Dict[str, Any]]]]):
        self._plans = list(plans)

    async def __call__(self, request: str, context: Dict[str, Any]
                       ) -> Optional[List[Dict[str, Any]]]:
        if self._plans:
            return self._plans.pop(0)
        return None


async def run_scenario(key: str, store: TaskLessonStore) -> Dict[str, Any]:
    spec = SCENARIOS[key]
    executor = RealExecutor()
    planner = ScenarioPlanner(spec["plans"])
    agent = ReasoningAgent(
        executor=executor,
        observer=RealObserver(),
        planner=planner,
        reasoning_model=get_reasoning_model(),  # real model when Ollama up
        limits=TaskLimits(max_task_steps=6, max_retries_per_step=1,
                          max_replans=2, max_total_execution_time=90),
        lesson_store=store,
        composer=ReasoningContextComposer(),
        transcript=spec["goal"],
    )
    t0 = time.time()
    result = await agent.run(spec["goal"])
    state = result.task_state
    report = {
        "scenario": key,
        "goal": spec["goal"],
        "mode": result.mode.value,
        "plan": [dict(s) for s in state.current_plan],
        "steps": [
            {"action": s.action, "params": s.params,
             "verified": s.verified, "status": s.status.value,
             "result": (s.result or s.error or "")[:200]}
            for s in state.completed_steps + state.failed_steps
        ],
        "observations": state.observed_state,
        "verification": state.verification_results,
        "recovery": {
            "replans": state.replan_count,
            "retries": state.retry_count,
            "diagnosis": (agent.last_diagnosis.to_dict()
                          if agent.last_diagnosis else None),
        },
        "final_status": (state.final_status.value
                         if state.final_status else None),
        "final_result": state.summary(),
        "reflection": result.reflection.to_dict() if result.reflection else None,
        "lessons": [l.to_dict() for l in result.lessons],
        "latency_s": round(time.time() - t0, 2),
    }
    return report


def write_markdown(reports: List[Dict[str, Any]]) -> None:
    lines = ["# Phase 21A — Real Autonomous Scenario Results",
             "", "(generated by scripts/run_reasoning_scenarios.py; "
              "real execution, no mocks for the executed steps)", ""]
    for r in reports:
        lines += [f"## Scenario {r['scenario']} — {r['goal']}", "",
                  f"- mode: `{r['mode']}`", f"- final status: "
                  f"**{r['final_status']}**", f"- final result: {r['final_result']}",
                  f"- replans: {r['recovery']['replans']}, "
                  f"retries: {r['recovery']['retries']}", "",
                  "### Steps", ""]
        for s in r["steps"]:
            lines.append(f"- `{s['action']}` {s['params']} → "
                         f"{'VERIFIED' if s['verified'] else 'FAILED'}: "
                         f"{s['result'][:120]}")
        diag = (r.get("recovery") or {}).get("diagnosis")
        if diag:
            lines += ["", f"- failure diagnosis: `{diag.get('failure_kind')}` "
                      f"→ strategy `{diag.get('next_strategy')}`"]
        lines += ["", "### Lessons", ""]
        for l in r["lessons"]:
            lines.append(f"- [{l['type']}] {l['lesson']} "
                         f"(evidence={l['evidence']}, "
                         f"conf={l['confidence']})")
        if not r["lessons"]:
            lines.append("- (none — outcome not lesson-worthy)")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


async def main() -> int:
    register_scenario_tools()
    keys = [k.upper() for k in (sys.argv[1:] or ["A", "B", "C", "D"])]
    store = TaskLessonStore()  # the REAL lesson store (data/task_lessons.json)
    reports = []
    for key in keys:
        if key not in SCENARIOS:
            print(f"unknown scenario {key}")
            continue
        print(f"\n=== Scenario {key}: {SCENARIOS[key]['goal']} ===")
        report = await run_scenario(key, store)
        reports.append(report)
        print(f"  status: {report['final_status']}  "
              f"({report['latency_s']}s)")
        print(f"  result: {report['final_result']}")
        for s in report["steps"]:
            print(f"   - {s['action']} {'OK' if s['verified'] else 'FAIL'}: "
                  f"{s['result'][:100]}")
        for l in report["lessons"]:
            print(f"   lesson[{l['type']}]: {l['lesson'][:80]}")
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(reports)
    print(f"\nreports: {REPORT_JSON} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
