"""
GoalRuntime planner — deterministic-first goal decomposition.

The planner NEVER calls a model when a deterministic resolver can produce a
plan. Decomposition is vocabulary/table-driven (mirroring the existing
agent.capability_router conventions: new apps are REGISTERED, not branched),
with the PLANNER-role LLM (Qwen2.5:7B) as fallback for genuinely novel goals.

Cross-application workflows: browser/telegram/gmail/filesystem/coding/
terminal subgoals chain, and every produced artifact is recorded so later
subgoals can consume it (artifact passing between subgoals).

Logging: [GOAL-PLAN]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from goalruntime.models import Artifact, GoalStatus, PermissionClass, Subgoal
from goalruntime.llm import ModelRouter, ModelRole, needs_coder, \
    needs_security_scope
from goalruntime.templates import CALCULATOR_TEMPLATE, \
    CALCULATOR_TEST_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class PlannerLimits:
    max_subgoals: int = 8
    max_replans: int = 3


# ═══════════════════════════════════════════════════════════════════
# Deterministic vocabulary (tables + regexes, not if-chains)
# ═══════════════════════════════════════════════════════════════════

_QUOTED_RE = re.compile(r'"([^"]{1,300})"')
_QUOTED_SQ_RE = re.compile(r"'([^']{1,300})'")
_SAYING_RE = re.compile(
    r"\b(?:saying|that says|with (?:the )?(?:text|message))\s+(.{1,300})$",
    re.I)

_CONTACT_AFTER_TO_RE = re.compile(r"\bto\s+([A-Za-z][\w .'-]{1,30})")
_CONTACT_NAMED_RE = re.compile(
    r"\b(?:contact|chat with)\s+([A-Za-z][\w'-]{1,30})", re.I)

_CONTACT_STOPWORDS = frozenset({
    "on", "in", "via", "the", "a", "an", "to", "and", "latest", "last",
    "telegram", "gmail", "whatsapp", "diego",
})


def _named_contact(text: str) -> str:
    """Explicit 'contact X' / 'chat with X' — single token, no stopwords."""
    m = _CONTACT_NAMED_RE.search(text)
    if m and m.group(1).lower() not in _CONTACT_STOPWORDS:
        return m.group(1)
    return ""

_FOLDER_RE = re.compile(
    r"\b(?:folder|directory)\s+(?:named\s+|called\s+)?"
    r"[\"']?([\w.\-/]+)[\"']?", re.I)
_CALCULATOR_RE = re.compile(r"\b(calculator|calc)\b", re.I)

# Destructive goals: the path is taken VERBATIM from the goal and never
# widened. The subgoal carries DESTRUCTIVE so the permission layer must
# obtain confirmation (or a previously granted scope) before anything runs.
_DESTRUCTIVE_RE = re.compile(r"\b(?:delete|remove|erase|wipe|rm)\b", re.I)
# Path-like targets win over bare names; `delete the file /tmp/x` must yield
# /tmp/x, not the verb "delete".
_DESTRUCTIVE_PATH_RE = re.compile(r"~?/[\w./\-]*[\w]")
_DESTRUCTIVE_TOKEN_RE = re.compile(r"\b[\w.\-]{2,}\b")
_DESTRUCTIVE_NOISE = frozenset({
    "delete", "remove", "erase", "wipe", "rm", "the", "a", "an", "my",
    "this", "that", "it", "please", "and", "then", "file", "folder",
    "directory", "path", "workspace", "diegos", "diego",
})

_SITE_RESOLVER = {
    "instagram": "instagram.com",
    "youtube": "youtube.com",
    "google": "google.com",
    "github": "github.com",
    "linkedin": "linkedin.com",
    "twitter": "twitter.com",
    "x.com": "x.com",
    "gmail": "mail.google.com",
}

_PROFILE_PATH_RE = re.compile(r"instagram(?:\.com)?/([A-Za-z0-9_.]{2,30})")
_BARE_PROFILE_RE = re.compile(
    r"\b(?:profile|page)\s+(?:of\s+)?@?([A-Za-z0-9_.]{2,30})")

_SEARCH_RE = re.compile(r"\b(search|google|look up)\b", re.I)

_HOST_RE = re.compile(
    r"\b(?:scan|probe|test|audit|target(?:ing)?)\s+(?:the\s+)?"
    r"(?:host\s+|target\s+)?"
    r"([a-z0-9.\-]+\.[a-z]{2,}|localhost|\d{1,3}(?:\.\d{1,3}){3})", re.I)
_SECURITY_OPS_RE = re.compile(
    r"\b(port[- ]?scan|service[- ]?enum|recon)\b", re.I)

_CLAUSE_SPLIT_RE = re.compile(r"\s+(?:and then|then|and also|and)\s+", re.I)

# Explicit URL / host navigation evidence
_URL_RE = re.compile(
    r"(?:https?://[^\s\"']+|"
    r"\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+(?:/[^\s\"']*)?)",
    re.I)
_NAV_VERB_RE = re.compile(
    r"\b(?:navigate|go|open|visit|browse|load|launch|head)\b", re.I)
# Dotted tokens that are FILES, not hosts ("open my resume.pdf").
_FILE_EXT_BLOCK = frozenset({
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv", "tsv",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "png", "jpg", "jpeg", "gif",
    "bmp", "svg", "webp", "mp3", "wav", "ogg", "mp4", "mkv", "mov", "avi",
    "py", "js", "ts", "json", "yaml", "yml", "md", "log", "sql", "db",
    "iso", "img", "deb", "rpm", "apk", "exe", "msi", "whl", "cfg", "ini",
})


def _explicit_url(text: str) -> str:
    """Deterministic explicit-URL extraction for navigation goals.

    Returns the URL only when the evidence is unambiguous: an ``http(s)``
    scheme, or a navigation verb plus a dotted host that is neither a bare
    site identity owned by ``_SITE_RESOLVER`` (so "open instagram" stays a
    site-identity goal) nor a file name (so "open resume.pdf" is not
    mistaken for a web host).
    """
    m = _URL_RE.search(text or "")
    if not m:
        return ""
    candidate = m.group(0).strip().strip("\"'.,;)")
    low = candidate.lower()
    if low.startswith(("http://", "https://")):
        return candidate
    host = low.split("/", 1)[0]
    if host in _SITE_RESOLVER or "." not in host:
        return ""
    tld = host.rsplit(".", 1)[-1]
    if tld in _FILE_EXT_BLOCK or not (2 <= len(tld) <= 24) or not tld.isalpha():
        return ""
    if not _NAV_VERB_RE.search(text or ""):
        return ""
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    return candidate

def workspace_path(name: str = "", *, base: Optional[str] = None) -> str:
    """Deterministic sandbox path for generated folders/files.

    ``base`` overrides the root (tests inject tmp_path); default is the
    repo's runtime_workspace directory.
    """
    root = Path(base) if base else (
        Path(__file__).resolve().parent.parent / "runtime_workspace")
    return str(root / name.strip().strip("/")) if name else str(root)


class DeterministicPlanner:
    """Deterministic decomposition. LLM only for genuinely novel goals."""

    def __init__(self, router: Optional[ModelRouter] = None,
                 limits: Optional[PlannerLimits] = None,
                 workspace_root: Optional[str] = None) -> None:
        self.router = router
        self.limits = limits or PlannerLimits()
        self.workspace_root = workspace_root

    # ── public API ───────────────────────────────────────────────

    def plan(self, goal_text: str) -> List[Subgoal]:
        goal_text = (goal_text or "").strip()
        subgoals = self._deterministic_plan(goal_text)
        if subgoals is None:
            subgoals = self._llm_plan(goal_text)
        return subgoals[: self.limits.max_subgoals]

    def replan(self, goal_text: str, failed: Subgoal,
               observation_text: str) -> List[Subgoal]:
        """A failed subgoal → a DIFFERENT strategy (never repeat as-is).

        Deterministic rules first: try the same skill with a different
        action/evidence; then escalate to the perception ladder; the
        PLANNER-role LLM only as the final fallback.
        """
        attempts = failed.attempts
        low = goal_text.lower()

        # Rule 1: visual/UI failure → perception escalation subgoal. The
        # escalation is ADVISORY: failing to re-perceive must not fail the
        # goal — the retry subgoal itself is what matters.
        if failed.skill_id in ("visual_ui", "telegram", "browser"):
            escalation = {
                "visual_ui": "find_element",
                "telegram": "search_contact",
                "browser": "read",
            }.get(failed.skill_id, "read")
            return [Subgoal(
                skill_id="vision", action="ocr",
                description=f"Re-perceive the screen after failed "
                            f"{failed.skill_id}.{failed.action}",
                permission_class=PermissionClass.READ_ONLY,
                consumes=[], produces=["screen_text"],
                max_attempts=1, required=False),
                Subgoal(
                skill_id=failed.skill_id, action=escalation,
                description=f"Retry {failed.description} with fresh evidence",
                params=dict(failed.params),
                permission_class=failed.permission_class,
                consumes=["screen_text"],
                max_attempts=max(1, failed.max_attempts - attempts),
                required=True)]

        # Rule 2: coding test failure → REGENERATE the file (deterministic
        # fix from the template), then re-run the tests (the test/fix loop).
        if failed.skill_id == "coding":
            if failed.action == "run_tests":
                ws = str(failed.params.get("path") or self._ws(""))
                return [Subgoal(
                    skill_id="coding", action="generate_code",
                    description="Fix: regenerate calculator.py from the "
                                "known-good template",
                    params={"path": f"{ws}/calculator.py",
                            "content": CALCULATOR_TEMPLATE},
                    permission_class=PermissionClass.REVERSIBLE_LOCAL,
                    produces=["file_path"]),
                    Subgoal(
                    skill_id="coding", action="run_tests",
                    description="Re-run the tests after the fix",
                    params={"path": ws},
                    permission_class=PermissionClass.REVERSIBLE_LOCAL,
                    consumes=["file_path"], produces=["test_result"])]
            return [Subgoal(
                skill_id="coding", action="run_tests",
                description="Run tests to gather failure evidence",
                params={"path": str(failed.params.get("path", ""))},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                produces=["test_result"])]

        # Rule 3: general fallback — one retry with a DIFFERENT action id
        # recorded as evidence (the runtime enforces the anti-repeat rule).
        desc = re.sub(r"^(?:Replan: |Retry )+", "", failed.description)
        return [Subgoal(
            skill_id=failed.skill_id,
            action=failed.action,
            description=f"Retry {desc} (attempt {attempts + 1}, new evidence)",
            params=dict(failed.params),
            permission_class=failed.permission_class,
            consumes=list(failed.produces),
            max_attempts=max(1, failed.max_attempts))]

    # ── deterministic decomposition ──────────────────────────────

    def _deterministic_plan(self, text: str) -> Optional[List[Subgoal]]:
        text = text or ""
        low = text.lower()

        # 1. Security goal (classified before generic verbs).
        if needs_security_scope(low):
            return self._plan_security(text)

        # 1b. Destructive goal — classified before folder/messaging so the
        #     DESTRUCTIVE permission class reaches the gate (never auto-run).
        #     Requires a concrete target (a path, or a file-ish noun) so prose
        #     like "remove duplicates from my list" is not hijacked.
        if _DESTRUCTIVE_RE.search(low) and (
                _DESTRUCTIVE_PATH_RE.search(text)
                or re.search(r"\b(?:file|folder|directory|path|workspace)\b",
                             low)):
            return self._plan_destructive(text)

        # 2. Folder+coding compound goal — owned by the folder planner
        #    BEFORE clause splitting (folder context must reach the coder).
        #    But a COMPOSED workflow ("Open instagram and then create a folder
        #    ... and then generate a calculator ...") must not lose its
        #    non-folder clauses: folder-ish clauses are grouped into one
        #    _plan_folder (keeping the folder context for the coder) while
        #    every other clause keeps its own deterministic route.
        if re.search(r"\b(folder|directory)\b", low) and needs_coder(low):
            clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text)
                       if c.strip()]
            if len(clauses) > 1:
                subs: List[Subgoal] = []
                folderish: List[str] = []
                for c in clauses:
                    cl = c.lower()
                    if re.search(r"\b(folder|directory)\b", cl) or (
                            needs_coder(cl) and folderish):
                        folderish.append(c)
                    else:
                        part = self._deterministic_plan(c)
                        if part:
                            subs.extend(part)
                if folderish:
                    joined = " and ".join(folderish)
                    subs.extend(self._plan_folder(joined, joined.lower()))
                if subs:
                    return subs
            return self._plan_folder(text, low)

        # 3. Composed workflows (sequential clauses).
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text)
                   if c.strip()]
        if len(clauses) > 1:
            subs: List[Subgoal] = []
            for c in clauses:
                part = self._deterministic_plan(c)
                if part:
                    subs.extend(part)
            if subs:
                return subs

        # 3b. Folder-only goal (no coding).
        if re.search(r"\b(folder|directory)\b", low):
            return self._plan_folder(text, low)

        # 3b-bis. Explicit URL navigation ("navigate to https://x/y") — the
        #     most specific browser evidence, so it precedes the site-identity
        #     and messaging routes ("navigate to mail.google.com" is a
        #     navigation, not an email-read goal, and outranks telegram/gmail
        #     keyword matching).
        target = _explicit_url(text)
        if target:
            return [Subgoal(
                skill_id="browser", action="navigate",
                description=f"Navigate to {target}",
                params={"url": target},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                produces=["page", "page_url"])]

        # 3c. Messaging (telegram / gmail) — checked BEFORE browser because
        #    "gmail" is also a site identity but read-email goals are
        #    messaging goals.
        telegram = bool(re.search(r"\btelegram\b", low))
        gmail = bool(re.search(r"\bgmail\b|\be-?mail\b|\bemail\b", low))
        if telegram or gmail:
            return self._plan_messaging(text, low, telegram, gmail)

        # 3d. Browser / search identity routes BEFORE coding identity
        #     ("search for python calculator tutorial" is a BROWSER goal).
        #     An EXPLICIT destination (url/domain after a nav verb) is an even
        #     stronger signal and routes first.
        if _extract_nav_target(text):
            return self._plan_browser(text, low)
        if re.search(r"\b(browser|chrome|firefox|website|web page|web)\b",
                     low) or _SEARCH_RE.search(low) \
                or any(k in low for k in _SITE_RESOLVER):
            return self._plan_browser(text, low)

        # 4. Coding goal.
        if needs_coder(low):
            return self._plan_coding(text, low)

        # 7. Unknown → PLANNER LLM.
        return None

    # ── specific planners ────────────────────────────────────────

    def _plan_browser(self, text: str, low: str) -> List[Subgoal]:
        # Search intent routes FIRST ("search google for python tutorial"
        # is a search, not a navigation to google.com).
        if _SEARCH_RE.search(low):
            # Drop site names and prepositions from the search query.
            query = _strip_verb_prefix(text)
            query = re.sub(r"\b(google|youtube|instagram|github|linkedin|"
                           r"twitter)\b", "", query, flags=re.I)
            query = re.sub(r"\b(on|for|in)\b\s*$", "", query,
                           flags=re.I).strip(" .")
            query = re.sub(r"^(?:for|on|in)\s+", "", query)
            return [Subgoal(
                skill_id="browser", action="search",
                description=f"Search the web for: {query}",
                params={"query": query},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                produces=["page", "search_results"])]

        # Explicit destination (URL or domain in navigation context).
        target = _extract_nav_target(text)
        if target:
            if "://" not in target:
                target = f"https://{_strip_nav_prefix(target)}"
            return [Subgoal(
                skill_id="browser", action="navigate",
                description=f"Navigate to {target}",
                params={"url": target},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                produces=["page", "page_url"])]

        # Instagram profile navigation ("open instagram profile of x")
        pm = _PROFILE_PATH_RE.search(low)
        if not pm:
            pm = _BARE_PROFILE_RE.search(low)
        for site_key, site in _SITE_RESOLVER.items():
            if site_key in low:
                params: Dict[str, Any] = {"url": site}
                if site_key == "instagram":
                    profile = (pm.group(1) if pm else "").strip("@")
                    if profile and profile not in ("instagram",):
                        params["url"] = f"instagram.com/{profile}"
                desc = f"Navigate to {params['url']}"
                return [Subgoal(
                    skill_id="browser", action="navigate", description=desc,
                    params=params,
                    permission_class=PermissionClass.REVERSIBLE_LOCAL,
                    produces=["page", "page_url"])]
        if _SEARCH_RE.search(low):
            # Drop site names and prepositions from the search query.
            query = _strip_verb_prefix(text)
            query = re.sub(r"\b(google|youtube|instagram|github|linkedin|"
                           r"twitter)\b", "", query, flags=re.I)
            query = re.sub(r"\b(on|for|in)\b\s*$", "", query,
                           flags=re.I).strip(" .")
            query = re.sub(r"^(?:for|on|in)\s+", "", query)
            return [Subgoal(
                skill_id="browser", action="search",
                description=f"Search the web for: {query}",
                params={"query": query},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                produces=["page", "search_results"])]
        # Explicit destination (URL or domain in navigation context).
        # (Handled above, before the search branch, by _extract_nav_target.)
        return [Subgoal(
            skill_id="browser", action="navigate",
            description=f"Navigate to {text}",
            params={"url": text},
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            produces=["page", "page_url"])]

    def _plan_messaging(self, text: str, low: str,
                        telegram: bool, gmail: bool) -> List[Subgoal]:
        subs: List[Subgoal] = []

        if telegram:
            contact = ""
            m = _CONTACT_AFTER_TO_RE.search(text)
            if m:
                contact = m.group(1).strip()
            else:
                contact = _named_contact(text)
            contact = _clean_contact(contact or _quoted_or_empty(text))
            message = _extract_message(text)

            if contact:
                subs.append(Subgoal(
                    skill_id="telegram", action="search_contact",
                    description=f"Open Telegram conversation with {contact}",
                    params={"contact": contact},
                    permission_class=PermissionClass.REVERSIBLE_LOCAL,
                    produces=["contact"]))
            else:
                subs.append(Subgoal(
                    skill_id="telegram", action="open",
                    description="Open Telegram",
                    params={"app": "telegram"},
                    permission_class=PermissionClass.REVERSIBLE_LOCAL,
                    produces=["contact"]))

            if re.search(r"\bread\b|\blast message\b|\blatest\b|\binbox\b",
                         low) and not message:
                subs.append(Subgoal(
                    skill_id="telegram", action="read_latest",
                    description="Read the latest Telegram message(s)",
                    params={"n": 1},
                    permission_class=PermissionClass.READ_ONLY,
                    consumes=["contact"], produces=["latest_messages"]))
            elif message:
                subs.append(Subgoal(
                    skill_id="telegram", action="send_message",
                    description=f"Send the message to {contact}",
                    params={"contact": contact, "text": message},
                    permission_class=PermissionClass.EXTERNAL_SIDE_EFFECT,
                    consumes=["contact"], produces=["sent_message"]))

        if gmail:
            n = 1
            m = re.search(r"\b(second|third|fourth)\b", low)
            if m:
                n = {"second": 2, "third": 3, "fourth": 4}[m.group(1)]
            subs.append(Subgoal(
                skill_id="gmail", action="read_email",
                description=f"Read email #{n} from Gmail",
                params={"n": n},
                permission_class=PermissionClass.READ_ONLY,
                produces=["email"]))
        return subs

    def _plan_folder(self, text: str, low: str) -> List[Subgoal]:
        fm = _FOLDER_RE.search(text)
        folder = fm.group(1).strip() if fm else "diego_folder"
        calc = bool(_CALCULATOR_RE.search(low))
        ws = self._ws(folder)
        subs = [Subgoal(
            skill_id="filesystem", action="create_folder",
            description=f"Create folder '{folder}'",
            params={"path": ws},
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            produces=["folder_path"])]
        if calc:
            subs.append(Subgoal(
                skill_id="coding", action="generate_code",
                description="Generate a Python calculator with tests",
                params={"path": f"{ws}/calculator.py",
                        "content": CALCULATOR_TEMPLATE},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                consumes=["folder_path"], produces=["file_path"]))
            subs.append(Subgoal(
                skill_id="coding", action="generate_code",
                description="Generate the calculator's test suite",
                params={"path": f"{ws}/test_calculator.py",
                        "content": CALCULATOR_TEST_TEMPLATE},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                consumes=["file_path"], produces=["file_path"]))
            subs.append(Subgoal(
                skill_id="coding", action="run_tests",
                description="Run the calculator's tests; fix on failure",
                params={"path": ws},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                consumes=["file_path"], produces=["test_result"]))
        return subs

    def _plan_coding(self, text: str, low: str) -> List[Subgoal]:
        # Coding requests without a folder context: generate + run.
        content = CALCULATOR_TEMPLATE if _CALCULATOR_RE.search(low) else ""
        gen_path = self._ws("generated.py")
        return [
            Subgoal(
                skill_id="coding", action="generate_code",
                description="Generate the requested code",
                params={"path": gen_path, "content": content},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                produces=["file_path"]),
            Subgoal(
                skill_id="coding", action="run_command",
                description="Execute the generated code",
                params={"cmd": f"python3 {gen_path}",
                        "cwd": self._ws("")},
                permission_class=PermissionClass.REVERSIBLE_LOCAL,
                consumes=["file_path"], produces=["program_output"]),
        ]

    def _ws(self, name: str) -> str:
        """Workspace path honoring this planner's injected root."""
        return workspace_path(name, base=self.workspace_root)

    def _plan_destructive(self, text: str) -> List[Subgoal]:
        """Destructive goals: verbatim target, DESTRUCTIVE class, never widened.

        The planner deliberately does NOT infer, normalize, or expand the
        target — it plans exactly the path the goal names, and the
        PermissionManager decides whether that exact action may run.
        """
        raw = ""
        m = _DESTRUCTIVE_PATH_RE.search(text or "")
        if m:
            raw = m.group(0).strip().strip("'\"")
        else:
            for tok in _DESTRUCTIVE_TOKEN_RE.findall(text or ""):
                if tok.lower() not in _DESTRUCTIVE_NOISE:
                    raw = tok.strip().strip("'\"")
                    break
        if raw.startswith(("~", "/")):
            path = str(Path(raw).expanduser())
        else:
            path = self._ws(raw) if raw else self._ws("")
        return [Subgoal(
            skill_id="filesystem", action="delete",
            description=f"Delete the path {path}",
            params={"path": path},
            permission_class=PermissionClass.DESTRUCTIVE,
            produces=["deletion"])]

    def _plan_security(self, text: str) -> List[Subgoal]:
        host = ""
        hm = _HOST_RE.search(text)
        if hm:
            host = hm.group(1)
        ops = [o.replace(" ", "_").replace("-", "_")
               for o in _SECURITY_OPS_RE.findall(text)]
        if not ops:
            ops = ["recon"]
        return [Subgoal(
            skill_id="security", action="scan",
            description=f"Authorized security scan of {host or '?'} "
                        f"({', '.join(ops)})",
            params={"target": host, "operations": ops},
            permission_class=PermissionClass.SECURITY_TESTING,
            produces=["security_findings"])]

    # ── LLM fallback (PLANNER role: qwen2.5:7b) ──────────────────

    def _llm_plan(self, goal_text: str) -> List[Subgoal]:
        if self.router is None:
            logger.warning("[GOAL-PLAN] no router; novel goal cannot be "
                           "decomposed → single unknown subgoal")
            return [Subgoal(
                skill_id="vision", action="describe",
                description=f"Observe the screen for goal: {goal_text}",
                permission_class=PermissionClass.READ_ONLY,
                produces=["screen_description"])]
        prompt = (
            "Decompose this desktop-agent goal into at most "
            f"{self.limits.max_subgoals} steps.\n"
            f"GOAL: {goal_text}\n"
            "Available skills/actions:\n"
            "browser.navigate(url), browser.search(query), "
            "visual_ui.focus_app(app), visual_ui.click_element(target), "
            "visual_ui.type_into_focus(text), telegram.search_contact("
            "contact), telegram.send_message(contact,text), "
            "telegram.read_latest(n), gmail.read_email(n), "
            "filesystem.create_folder(path), filesystem.write_file("
            "path,content), coding.run_command(cmd), terminal.run_command("
            "cmd), vision.ocr, vision.describe\n"
            "Reply with one JSON line per step: "
            '{"skill": "...", "action": "...", "params": {...}}')
        raw = self.router.invoke(ModelRole.PLANNER, prompt,
                                 system="You output ONLY JSON lines.",
                                 max_tokens=400, temperature=0.1)
        return _parse_llm_plan(raw)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _clean_contact(name: str) -> str:
    name = (name or "").strip().strip(",.?!").strip()
    # Strip trailing application suffixes ("rahul on telegram" → "rahul").
    name = re.sub(
        r"\s+(?:on|in|via|through)\s+(?:telegram|whatsapp|gmail|"
        r"e-?mail|signal|slack|instagram)\b.*$", "", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip()


def _quoted_or_empty(text: str) -> str:
    for rx in (_QUOTED_RE, _QUOTED_SQ_RE):
        m = rx.search(text)
        if m:
            return m.group(1)
    return ""


def _extract_message(text: str) -> str:
    m = _QUOTED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _QUOTED_SQ_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _SAYING_RE.search(text)
    if m:
        tail = m.group(1).strip()
        tail = re.sub(r"\s+to\s+[A-Za-z][\w .'-]{0,30}$", "", tail,
                      flags=re.I)
        return tail.strip().strip(",.?!")
    # Unquoted imperative: "send hello to Rahul" → "hello".
    m = re.search(r"\bsend\s+(?:the\s+)?(?:message\s+)?(.+?)\s+to\s+", text,
                  re.I)
    if m:
        return m.group(1).strip().strip("'\"")
    return ""


def _strip_verb_prefix(text: str) -> str:
    return re.sub(
        r"^\s*(?:please\s+)?(?:can you\s+)?"
        r"(?:search(?: the web)?(?: for)?|google|look up)\s*",
        "", text, flags=re.I).strip() or text


def _looks_like_domain(token: str) -> bool:
    """A bare host like `github.com` / `news.ycombinator.com` (no spaces)."""
    return bool(re.fullmatch(
        r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?:/\S*)?",
        token.strip().lower()))


def _extract_nav_target(text: str) -> str:
    """Explicit navigation target: a full URL, or a domain in nav context.

    Returns "" when the text carries no explicit destination. Bare domains are
    only accepted after a navigation verb so prose like "explain how
    python.org works" is not mistaken for a navigation goal.

    Thin compatibility alias over the hardened :func:`_explicit_url`.
    """
    return _explicit_url(text or "")


def _strip_nav_prefix(text: str) -> str:
    """Drop a leading navigation verb/phrase, leaving the destination."""
    return re.sub(
        r"^\s*(?:please\s+)?(?:can you\s+)?"
        r"(?:navigate\s+to|go\s+to|open(?:\s+the)?(?:\s+website|\s+url|"
        r"\s+page)?|visit|browse\s+to|load)\s+", "", text, flags=re.I).strip()


def _parse_llm_plan(raw: str) -> List[Subgoal]:
    import json
    subs: List[Subgoal] = []
    for line in (raw or "").splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        skill = str(obj.get("skill") or "")
        action = str(obj.get("action") or "")
        if not skill or not action:
            continue
        subs.append(Subgoal(
            skill_id=skill, action=action,
            description=str(obj.get("description") or f"{skill}.{action}"),
            params=dict(obj.get("params") or {}),
            permission_class=PermissionClass.REVERSIBLE_LOCAL,
            produces=[f"{skill}_artifact"]))
    return subs


__all__ = ["DeterministicPlanner", "PlannerLimits", "workspace_path",
           "CALCULATOR_TEMPLATE", "CALCULATOR_TEST_TEMPLATE"]
