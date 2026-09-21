"""
GoalRuntime skill layer — generic, structured capability units.

Every skill action exposes:
    preconditions      — what must hold before the action may run
    permission_class   — one of the five PermissionClass values
    executor           — backend call (the ONLY world-touching step)
    observation        — post-action state read-back
    expected_effect    — what success must look like
    verifier           — deterministic check that the effect happened
    evidence           — structured proof recorded on the SkillResult

The registry is app-agnostic: telegram/gmail/etc. are handled by generic
messaging/mail skills driving a backend, never by hard-coded coordinates or
per-app if-chains in the runtime.

Logging: [GOAL-SKILL]
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from goalruntime.models import (
    Artifact, Observation, PermissionClass, SkillResult, Subgoal,
)
from goalruntime.permissions import classify_action

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Contracts
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ActionSpec:
    """One executable action of a skill."""

    name: str
    permission_class: PermissionClass
    expected_effect: str
    executor: Callable[..., Dict[str, Any]]     # (backend, params) -> dict
    observer: Optional[Callable[..., Dict[str, Any]]] = None  # (backend) -> dict
    verifier: Optional[Callable[..., bool]] = None  # (result_dict, obs) -> bool
    preconditions: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
    # preconditions(context) -> None (ok) | reason string (blocked)
    produces: List[str] = field(default_factory=list)


@dataclass
class SkillSpec:
    id: str
    name: str
    capabilities: List[str]
    actions: Dict[str, ActionSpec] = field(default_factory=dict)

    def has_action(self, action: str) -> bool:
        return action in self.actions


# ═══════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════

class SkillExecutor:
    """Runs one Subgoal through its skill action with full structure."""

    def __init__(self, registry: "SkillRegistry") -> None:
        self.registry = registry

    def execute(self, subgoal: Subgoal, backend: Any,
                context: Dict[str, Any]) -> SkillResult:
        skill = self.registry.get(subgoal.skill_id)
        t0 = time.time()
        if skill is None:
            return _fail(subgoal, f"unknown skill '{subgoal.skill_id}'", t0)
        spec = skill.actions.get(subgoal.action)
        if spec is None:
            return _fail(subgoal,
                         f"skill '{subgoal.skill_id}' has no action "
                         f"'{subgoal.action}'", t0)

        # 1. Preconditions
        if spec.preconditions is not None:
            blocked = spec.preconditions(context)
            if blocked:
                return _fail(subgoal, f"precondition failed: {blocked}", t0)

        # 2. Execute through the backend (the only world-touching step)
        try:
            raw = spec.executor(backend, subgoal.params)
        except Exception as e:  # fail-safe: honest failure, never raise
            logger.exception("[GOAL-SKILL] executor crashed")
            return _fail(subgoal, str(e), t0)
        success = bool(raw.get("success"))
        error = str(raw.get("error") or "")

        # 3. Observation (post-state read-back)
        obs_data: Dict[str, Any] = {}
        obs_text = ""
        evidence_level = str(raw.get("evidence_level") or "")
        if spec.observer is not None:
            try:
                o = spec.observer(backend) or {}
                obs_data = o
                obs_text = str(o.get("text") or "")
                evidence_level = evidence_level or str(o.get("evidence_level") or "")
            except Exception as e:
                obs_data = {"observer_error": str(e)}
        observation = Observation(success=success, text=obs_text,
                                  data=obs_data, evidence_level=evidence_level)

        # 4. Deterministic verification of the expected effect
        verified = False
        verdict = ""
        if success and spec.verifier is not None:
            try:
                verified = bool(spec.verifier(raw, observation))
                verdict = ("verified: " + spec.expected_effect) if verified \
                    else ("effect NOT observed: " + spec.expected_effect)
            except Exception as e:
                verified = False
                verdict = f"verifier error: {e}"
        elif success:
            verified = True
            verdict = "verified: " + spec.expected_effect

        result = SkillResult(
            skill_id=subgoal.skill_id, action=subgoal.action,
            success=success, observation=observation,
            expected_effect=spec.expected_effect, verification=verdict,
            verified=verified,
            permission_class=spec.permission_class,
            evidence=_evidence(raw), error=error,
            artifacts=_artifacts(spec, raw, subgoal),
            duration_ms=round((time.time() - t0) * 1000, 1))
        logger.info("[GOAL-SKILL] %s.%s success=%s verified=%s (%s)",
                    subgoal.skill_id, subgoal.action, success, verified,
                    result.duration_ms)
        return result


def _fail(subgoal: Subgoal, error: str, t0: float) -> SkillResult:
    return SkillResult(
        skill_id=subgoal.skill_id, action=subgoal.action, success=False,
        expected_effect="", verification="", error=error,
        duration_ms=round((time.time() - t0) * 1000, 1))


def _evidence(raw: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k in ("evidence", "contact", "verification", "method", "url", "path",
              "messages", "email", "count", "returncode", "open_ports",
              "operations", "results", "results_summary", "target",
              "stdout", "stderr", "bytes"):
        if k in raw and raw[k] not in (None, ""):
            keep[k] = raw[k]
    return keep


# Skill artifact name → backend result key.
_ARTIFACT_SOURCE = {
    "latest_messages": "messages",
    "program_output": "stdout",
    "terminal_output": "stdout",
    "security_findings": "results",
    "test_result": "returncode",
    "screen_text": "text",
    "screen_description": "text",
    "page": "text",
    "search_results": "evidence",
    "email": "email",
    "contact": "contact",
    "folder_path": "path",
    "file_path": "path",
    "file_content": "content",
}


def _artifacts(spec: ActionSpec, raw: Dict[str, Any],
               subgoal: Subgoal) -> List[Artifact]:
    out = []
    for name in spec.produces:
        src = _ARTIFACT_SOURCE.get(name, name)
        value = raw.get(src)
        if value is None:
            value = raw.get("evidence", {}).get(name)
        kind = "json"
        if isinstance(value, str):
            kind = "text"
        elif isinstance(value, list):
            kind = "list"
        out.append(Artifact(name=name, kind=kind, value=value,
                            producer=subgoal.id))
    return out


# ═══════════════════════════════════════════════════════════════════
# The nine generic skills
# ═══════════════════════════════════════════════════════════════════

def _need(cond: bool, reason: str):
    def check(context: Dict[str, Any]) -> Optional[str]:
        if not cond:
            return reason
        return None
    return check


# ── 1. Browser ───────────────────────────────────────────────────

def _browser_url(params: Dict[str, Any]) -> str:
    url = str(params.get("url") or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-z][a-z0-9+.\-]*://", url):
        url = "https://" + url
    return url


def _browser_verify(result: Dict[str, Any], obs: Observation) -> bool:
    want = str(result.get("url") or "").lower()
    got = str(obs.data.get("url") or obs.data.get("page_url")
              or result.get("url") or "").lower()
    if got and want:
        return want.split("//")[-1].split("/")[0] in got
    return bool(obs.data.get("page_title") or obs.data.get("text")
                or result.get("evidence"))


BROWSER_SKILL = SkillSpec(
    id="browser", name="Browser Navigation & Reading",
    capabilities=["browser.navigate", "browser.search", "browser.read"],
    actions={
        "navigate": ActionSpec(
            name="navigate",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the browser shows the requested URL",
            executor=lambda be, p: be.browser_navigate(_browser_url(p)),
            observer=lambda be: be.browser_read(),
            verifier=_browser_verify,
            produces=["page"]),
        "search": ActionSpec(
            name="search",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="search results for the query are visible",
            executor=lambda be, p: be.browser_search(str(p.get("query") or "")),
            observer=lambda be: be.browser_read(),
            verifier=lambda r, o: bool(o.data.get("text") or o.data.get("title")),
            produces=["page", "search_results"]),
        "read": ActionSpec(
            name="read",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="page text is captured as evidence",
            executor=lambda be, p: be.browser_read(),
            verifier=lambda r, o: bool(o.data.get("text") or o.data.get("title")),
            produces=["page"]),
    },
)


# ── 2. Visual UI interaction ─────────────────────────────────────

VISUAL_UI_SKILL = SkillSpec(
    id="visual_ui", name="Visual UI Interaction",
    capabilities=["ui.click", "ui.type", "ui.focus", "ui.find"],
    actions={
        "focus_app": ActionSpec(
            name="focus_app",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the named application window is focused",
            executor=lambda be, p: be.focus_window(str(p.get("app") or "")),
            observer=lambda be: be.active_window(),
            verifier=lambda r, o: bool(o.data.get("title") or o.data.get("window_id"))),
        "click_element": ActionSpec(
            name="click_element",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the located element was activated",
            executor=lambda be, p: be.click(
                str(p.get("target") or ""),
                evidence=p.get("evidence") or p.get("location_evidence")),
            observer=lambda be: be.active_window(),
            verifier=lambda r, o: bool(r.get("evidence")) or bool(o.data),
            preconditions=_need(
                bool, "click requires located evidence (no blind coordinates)")),
        "type_into_focus": ActionSpec(
            name="type_into_focus",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the text was typed into the focused control",
            executor=lambda be, p: be.type_text(str(p.get("text") or "")),
            verifier=lambda r, o: bool(r.get("evidence")) or o.success),
        "press_key": ActionSpec(
            name="press_key",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the key press reached the focused window",
            executor=lambda be, p: be.press_key(str(p.get("key") or "enter")),
            verifier=lambda r, o: o.success),
        "find_element": ActionSpec(
            name="find_element",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="matching UI element candidates were observed",
            executor=lambda be, p: be.search_ui(str(p.get("target") or "")),
            verifier=lambda r, o: bool(r.get("evidence"))),
    },
)


# ── 3. Telegram (desktop automation; no API tokens) ──────────────

def _telegram_verify_contact(r: Dict[str, Any], obs: Observation) -> bool:
    return bool(r.get("contact")) and r.get("stage") != "select_contact"


def _telegram_verify_send(r: Dict[str, Any], obs: Observation) -> bool:
    # Post-send verification: the message must be observed INSIDE the
    # conversation afterwards (never from a window change alone).
    return bool(r.get("verification", "").startswith("message observed"))


TELEGRAM_SKILL = SkillSpec(
    id="telegram", name="Telegram Desktop",
    capabilities=["telegram.search_contact", "telegram.read_latest",
                  "telegram.send_message"],
    actions={
        "open": ActionSpec(
            name="open",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the Telegram window is open and focused",
            executor=lambda be, p: be.open_app("telegram"),
            observer=lambda be: be.active_window(),
            verifier=lambda r, o: "telegram" in str(
                o.data.get("window_class") or o.data.get("title") or "").lower()),
        "search_contact": ActionSpec(
            name="search_contact",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the named contact's conversation is open",
            executor=lambda be, p: be.telegram_search_contact(
                str(p.get("contact") or "")),
            verifier=_telegram_verify_contact,
            produces=["contact"]),
        "read_latest": ActionSpec(
            name="read_latest",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="the latest incoming message(s) were read",
            executor=lambda be, p: be.telegram_read_latest(
                int(p.get("n") or 1)),
            verifier=lambda r, o: bool(r.get("messages")),
            produces=["latest_messages"]),
        "send_message": ActionSpec(
            name="send_message",
            permission_class=PermissionClass.EXTERNAL_SIDE_EFFECT,
            expected_effect=("the message text is observed inside the "
                             "intended conversation after sending"),
            executor=lambda be, p: be.telegram_send(
                str(p.get("contact") or ""), str(p.get("text") or "")),
            verifier=_telegram_verify_send,
            produces=["sent_message"]),
    },
)


# ── 4. Gmail (browser-based reading) ─────────────────────────────

GMAIL_SKILL = SkillSpec(
    id="gmail", name="Gmail Reading",
    capabilities=["gmail.read_email"],
    actions={
        "read_email": ActionSpec(
            name="read_email",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="the requested email was read",
            executor=lambda be, p: be.gmail_read_email(int(p.get("n") or 1)),
            verifier=lambda r, o: bool(r.get("email")),
            produces=["email"]),
    },
)


# ── 5. Filesystem ────────────────────────────────────────────────

def _fs_safe_path(p: Dict[str, Any]) -> str:
    path = str(p.get("path") or "").strip()
    return path


FILESYSTEM_SKILL = SkillSpec(
    id="filesystem", name="Filesystem",
    capabilities=["fs.create_folder", "fs.write_file", "fs.read_file",
                  "fs.delete"],
    actions={
        "create_folder": ActionSpec(
            name="create_folder",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the folder exists on disk",
            executor=lambda be, p: be.fs_create_folder(_fs_safe_path(p)),
            verifier=lambda r, o: bool(r.get("path")),
            produces=["folder_path"]),
        "write_file": ActionSpec(
            name="write_file",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the file exists with the given content",
            executor=lambda be, p: be.fs_write_file(
                _fs_safe_path(p), str(p.get("content") or "")),
            verifier=lambda r, o: bool(r.get("bytes") is not None),
            produces=["file_path"]),
        "read_file": ActionSpec(
            name="read_file",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="the file content was read",
            executor=lambda be, p: be.fs_read_file(_fs_safe_path(p)),
            verifier=lambda r, o: bool(r.get("content") is not None),
            produces=["file_content"]),
        "delete": ActionSpec(
            name="delete",
            permission_class=PermissionClass.DESTRUCTIVE,
            expected_effect="the path no longer exists",
            executor=lambda be, p: be.fs_delete(_fs_safe_path(p)),
            verifier=lambda r, o: bool(r.get("success"))),
    },
)


# ── 6. Coding & test execution ───────────────────────────────────

CODING_SKILL = SkillSpec(
    id="coding", name="Coding & Test Execution",
    capabilities=["coding.generate", "coding.run_tests", "coding.fix"],
    actions={
        "generate_code": ActionSpec(
            name="generate_code",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the requested code file exists and parses",
            executor=lambda be, p: be.fs_write_file(
                str(p.get("path") or ""), str(p.get("content") or "")),
            verifier=lambda r, o: bool(r.get("bytes") is not None),
            produces=["file_path"]),
        "run_tests": ActionSpec(
            name="run_tests",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the test suite ran and its result is known",
            executor=lambda be, p: be.run_tests(str(p.get("path") or ".")),
            verifier=lambda r, o: r.get("returncode") is not None,
            produces=["test_result"]),
        "run_command": ActionSpec(
            name="run_command",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            expected_effect="the program executed and its output is known",
            executor=lambda be, p: be.run_command(
                str(p.get("cmd") or ""),
                cwd=str(p.get("cwd") or "")),
            verifier=lambda r, o: r.get("returncode") is not None,
            produces=["program_output"]),
    },
)


# ── 7. Terminal ──────────────────────────────────────────────────

_TERMINAL_DESTRUCTIVE_RE = re.compile(
    r"\b(rm\s+-rf|rm\s+-fr|mkfs|dd\s+if=|>\s*/dev/|:(){|shutdown|reboot)\b")
_TERMINAL_READONLY_RE = re.compile(
    r"^\s*(ls|cat|pwd|whoami|date|echo|head|tail|wc|which|file|stat|df|du|"
    r"uname|hostname|ip a|ip addr|ifconfig|ps\b|top -bn1|env|printenv)\b")
_TERMINAL_STATE_RE = re.compile(
    r"^\s*(mkdir|touch|cp|mv|cd|pip install|python3? |pytest|git status|"
    r"git add|git commit|npm|node)\b")


def terminal_permission(cmd: str) -> PermissionClass:
    """Deterministic command classification (safe side on unknown)."""
    c = (cmd or "").strip()
    low = c.lower()
    if _TERMINAL_DESTRUCTIVE_RE.search(low):
        return PermissionClass.DESTRUCTIVE
    if _TERMINAL_READONLY_RE.match(low):
        return PermissionClass.READ_ONLY
    if _TERMINAL_STATE_RE.match(low):
        return PermissionClass.REVERSIBLE_LOCAL
    return PermissionClass.EXTERNAL_SIDE_EFFECT  # unknown → confirm


TERMINAL_SKILL = SkillSpec(
    id="terminal", name="Terminal",
    capabilities=["terminal.run_command"],
    actions={
        "run_command": ActionSpec(
            name="run_command",
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            # NOTE: the runtime re-classifies via terminal_permission()
            # before dispatch; the declared class here is the ceiling.
            expected_effect="the command ran and produced output/exit code",
            executor=lambda be, p: be.run_command(
                str(p.get("cmd") or ""), cwd=str(p.get("cwd") or "")),
            verifier=lambda r, o: r.get("returncode") is not None,
            produces=["terminal_output"]),
    },
)


# ── 8. Security (authorized workflows only) ──────────────────────

SECURITY_SKILL = SkillSpec(
    id="security", name="Authorized Security Testing",
    capabilities=["security.scan"],
    actions={
        "scan": ActionSpec(
            name="scan",
            permission_class=PermissionClass.SECURITY_TESTING,
            expected_effect=("probe results for the explicitly authorized "
                             "target/scope were collected"),
            executor=lambda be, p: be.security_scan(
                str(p.get("target") or ""),
                list(p.get("operations") or ["recon"])),
            verifier=lambda r, o: bool(r.get("results")),
            produces=["security_findings"]),
    },
)


# ── 9. Vision / OCR ──────────────────────────────────────────────

VISION_SKILL = SkillSpec(
    id="vision", name="Vision & OCR Perception",
    capabilities=["vision.ocr", "vision.describe"],
    actions={
        "ocr": ActionSpec(
            name="ocr",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="screen text was extracted",
            executor=lambda be, p: be.ocr(),
            verifier=lambda r, o: bool(r.get("text")),
            produces=["screen_text"]),
        "describe": ActionSpec(
            name="describe",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="a vision-model description of the screen exists",
            executor=lambda be, p: be.vision_query(
                str(p.get("prompt") or "Describe the current screen.")),
            verifier=lambda r, o: bool(r.get("text")),
            produces=["screen_description"]),
        "screenshot": ActionSpec(
            name="screenshot",
            permission_class=PermissionClass.READ_ONLY,
            expected_effect="a screenshot was captured",
            executor=lambda be, p: be.screenshot(),
            verifier=lambda r, o: bool(r.get("size"))),
    },
)


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

class SkillRegistry:
    """App-agnostic registry; new skills are REGISTERED, not branched."""

    def __init__(self, skills: Optional[List[SkillSpec]] = None) -> None:
        self._skills: Dict[str, SkillSpec] = {}
        for s in (skills or DEFAULT_SKILLS):
            self.register(s)

    def register(self, skill: SkillSpec) -> None:
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Optional[SkillSpec]:
        return self._skills.get(skill_id)

    def ids(self) -> List[str]:
        return sorted(self._skills)

    def find_capable(self, capability: str) -> Optional[SkillSpec]:
        """capability like 'telegram.send_message' or 'browser.search'."""
        for s in self._skills.values():
            if capability in s.capabilities:
                return s
        # last segment match: 'send_message' → telegram.send_message
        tail = capability.split(".")[-1]
        for s in self._skills.values():
            if any(c.split(".")[-1] == tail for c in s.capabilities):
                return s
        return None


DEFAULT_SKILLS = [
    BROWSER_SKILL, VISUAL_UI_SKILL, TELEGRAM_SKILL, GMAIL_SKILL,
    FILESYSTEM_SKILL, CODING_SKILL, TERMINAL_SKILL, SECURITY_SKILL,
    VISION_SKILL,
]


__all__ = [
    "ActionSpec", "SkillSpec", "SkillExecutor", "SkillRegistry",
    "DEFAULT_SKILLS", "terminal_permission", "classify_action",
]
