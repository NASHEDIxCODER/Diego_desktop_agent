"""
LocalComputerIntent — Deterministic local-filesystem intent detection + resolution.

Problem fixed (reliability layer, Phase 2):
    Requests like "find the largest Python file in my Diego project",
    "how many documents are on my PC?", "show me 10 PDFs in Downloads"
    were previously routed to semantic knowledge retrieval / session
    memory / web search because nothing recognised them as LOCAL
    filesystem operations.

This module is PURE DETERMINISTIC (regex + os.walk):
    - No LLM. No embeddings. No network.
    - detect_local_intent(text)  → Optional[LocalIntent]
    - resolve_local_intent(intent, roots) → str (evidence-backed answer)

Every answer is computed from ACTUAL filesystem observations, so the
returned string is evidence, not model-generated reasoning.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── Scan bounds (safety on real hardware) ─────────────────────────
MAX_FILES_SCANNED = 50_000     # hard cap on entries walked
MAX_DEPTH = 12                 # directory depth cap
SCAN_BUDGET_S = 8.0            # wall-clock cap per resolution


class LocalIntentKind(str, Enum):
    FILE_SEARCH = "FILE_SEARCH"        # find/list files matching criteria
    FILE_COUNT = "FILE_COUNT"          # "how many documents…?"
    SYSTEM_INFO = "SYSTEM_INFO"        # disk/memory/process/CPU facts


class LocalSort(str, Enum):
    LARGEST = "largest"
    SMALLEST = "smallest"
    NEWEST = "newest"
    OLDEST = "oldest"
    NONE = "none"


# Common document / file-type vocabulary the user can speak.
_EXTENSION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "documents": (".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt",
                  ".xlsx", ".xls", ".csv", ".pptx", ".ppt"),
    "document": (".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt",
                 ".xlsx", ".xls", ".csv", ".pptx", ".ppt"),
    "pdfs": (".pdf",),
    "pdf": (".pdf",),
    "python": (".py",),
    "python files": (".py",),
    "images": (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"),
    "photos": (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"),
    "videos": (".mp4", ".mkv", ".avi", ".mov", ".webm"),
    "music": (".mp3", ".flac", ".wav", ".ogg", ".m4a"),
    "audio": (".mp3", ".flac", ".wav", ".ogg", ".m4a"),
    "files": (),   # any file
    "spreadsheets": (".xlsx", ".xls", ".csv", ".ods"),
    "presentations": (".pptx", ".ppt", ".odp"),
    "archives": (".zip", ".tar", ".gz", ".rar", ".7z"),
    "code": (".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go",
             ".rs", ".rb", ".sh", ".json", ".yaml", ".yml", ".toml"),
}

# Direct regex so "py" / "pdf" style types are understood too.
_TYPE_TOKEN = (
    r"(?:documents?|pdfs?|python(?:\s+files?)?|images?|photos?|videos?|"
    r"music|audio|spreadsheets?|presentations?|archives?|files?|code)"
)


@dataclass
class LocalIntent:
    """A fully-resolved deterministic local-computer intent."""
    kind: LocalIntentKind
    extensions: Tuple[str, ...] = ()
    folder: str = ""                 # spoken folder hint ("" = whole PC)
    limit: int = 0                   # 0 = all / asker-specified count
    sort: LocalSort = LocalSort.NONE
    modified: str = ""               # "" | "today" | "yesterday" | "week"
    confidence: float = 1.0
    source_text: str = ""

    def describe(self) -> str:
        return (f"{self.kind.value} ext={self.extensions or '*'} "
                f"folder={self.folder or '~'} limit={self.limit} "
                f"sort={self.sort.value} modified={self.modified or 'any'}")


# ═══════════════════════════════════════════════════════════════
# Detection (deterministic regex — the ONLY router for this class)
# ═══════════════════════════════════════════════════════════════

_COUNT_RE = re.compile(
    rf"\b(?:how many|count (?:of |the )?|number of)\s+(?P<type>{_TYPE_TOKEN})\b",
    re.IGNORECASE,
)
_SUPERLATIVE_RE = re.compile(
    rf"\b(?:largest|biggest|smallest|newest|most\s+recent|oldest)\s+"
    rf"(?P<type>{_TYPE_TOKEN})\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    rf"\b(?:show|list|find|open|display)\s+(?:me\s+)?"
    r"(?:(?P<limit>\d+)\s+)?(?:of\s+the\s+)?"
    rf"(?P<type>{_TYPE_TOKEN})\b",
    re.IGNORECASE,
)
_FOLDER_RE = re.compile(
    r"\b(?:in|from)\s+(?:my\s+|the\s+)?(?P<folder>[a-z0-9 _\-/]+?)(?:\s+(?:folder|directory|project))?"
    r"(?=\s*(?:\?|modified|from|created|\.|$))",
    re.IGNORECASE,
)
_MODIFIED_RE = re.compile(
    r"\b(?:modified|changed|created)\s+(today|yesterday|this week|recently)\b",
    re.IGNORECASE,
)
_SYSTEM_RE = re.compile(
    r"\b(?:disk|storage|memory|ram|cpu|processes|system information|"
    r"system info|free space)\b",
    re.IGNORECASE,
)
_ON_MY_PC = re.compile(r"\bon my (pc|computer|laptop|machine)\b|\bmy pc\b|\bmy computer\b",
                       re.IGNORECASE)


def _extract_extensions(type_text: str) -> Tuple[str, ...]:
    key = " ".join(type_text.lower().split())
    candidates = [key]
    # "python file" / "python files" → "python"; "pdf file" → "pdf"
    if key.endswith((" file", " files")):
        candidates.insert(0, key.rsplit(" ", 1)[0])
    for cand in candidates:
        if cand in _EXTENSION_ALIASES:
            return _EXTENSION_ALIASES[cand]
        singular = cand.rstrip("s")
        if singular in _EXTENSION_ALIASES:
            return _EXTENSION_ALIASES[singular]
    return ()


def detect_local_intent(text: str) -> Optional[LocalIntent]:
    """Deterministically detect a local-filesystem intent.

    Returns None for anything that is not clearly a local computer
    request — the caller then falls through to the normal routing.
    """
    if not text:
        return None
    t = " ".join(text.lower().strip(" .!?").split())

    kind: Optional[LocalIntentKind] = None
    type_match: Optional[re.Match] = None
    limit = 0

    m = _COUNT_RE.search(t)
    if m:
        kind = LocalIntentKind.FILE_COUNT
        type_match = m
    else:
        m = _SUPERLATIVE_RE.search(t)
        if m:
            kind = LocalIntentKind.FILE_SEARCH
            type_match = m
            sort_word = m.group(0).strip().split()[0]
            sort = (LocalSort.SMALLEST if sort_word == "smallest"
                    else LocalSort.NEWEST if sort_word in ("newest", "recent")
                    else LocalSort.OLDEST if sort_word == "oldest"
                    else LocalSort.LARGEST)
        else:
            m = _LIST_RE.search(t)
            if m:
                kind = LocalIntentKind.FILE_SEARCH
                type_match = m
                if m.group("limit"):
                    limit = int(m.group("limit"))
                sort = (LocalSort.NEWEST if _MODIFIED_RE.search(t)
                        else LocalSort.NONE)
            else:
                return None

    extensions = _extract_extensions(type_match.group("type"))

    folder = ""
    fm = _FOLDER_RE.search(t)
    if fm:
        folder = fm.group("folder").strip()

    modified = ""
    mm = _MODIFIED_RE.search(t)
    if mm:
        word = mm.group(1).lower()
        modified = {"this week": "week", "recently": "week"}.get(word, word)

    return LocalIntent(
        kind=kind,
        extensions=extensions,
        folder=folder,
        limit=limit,
        sort=(locals().get("sort", LocalSort.NONE)),
        modified=modified,
        confidence=0.99,
        source_text=text,
    )


def is_local_system_query(text: str) -> bool:
    """True for system-information questions handled by local tooling."""
    return bool(text) and bool(_SYSTEM_RE.search(text))


# ═══════════════════════════════════════════════════════════════
# Resolution (real filesystem observation — bounded)
# ═══════════════════════════════════════════════════════════════

def default_roots() -> List[Path]:
    """Candidate roots for folder hints and full-PC scans."""
    home = Path.home()
    roots = [home]
    for sub in ("Desktop", "Documents", "Downloads", "Projects", "Code",
                "dev", "workspace", "Workspace", "PycharmProjects"):
        p = home / sub
        if p.is_dir():
            roots.append(p)
    return roots


def resolve_folder(hint: str, roots: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """Resolve a spoken folder hint to an actual directory.

    Handles: "Downloads", "my Diego project", "Projects",
    "home/nashedi_x_coder/Workspace/...", and project names that live
    under ~/Projects or ~/Workspace.
    """
    if not hint:
        return None
    hint_l = hint.strip().strip("'\"").lower()
    hint_l = re.sub(r"\b(my|the|a)\b", " ", hint_l)
    hint_l = " ".join(hint_l.split())

    candidates = list(roots) if roots else default_roots()

    # Absolute or ~-relative path spoken directly
    raw = Path(hint.strip().strip("'\"")).expanduser()
    if raw.is_dir():
        return raw

    # Direct child of home ("Downloads", "Documents")
    direct = Path.home() / hint_l.replace(" ", "_")
    if direct.is_dir():
        return direct
    direct2 = Path.home() / hint_l
    if direct2.is_dir():
        return direct2

    # Project-style: search known roots (depth ≤ 3) for a matching dir name
    token = hint_l.split()[-1] if hint_l.split() else hint_l
    for root in candidates:
        try:
            for dirpath, dirnames, _ in os.walk(root):
                depth = dirpath[len(str(root)):].count(os.sep)
                if depth >= 3:
                    dirnames[:] = []
                    continue
                dirnames[:] = [d for d in dirnames
                               if not d.startswith((".", "__"))]
                for d in dirnames:
                    dl = d.lower()
                    if dl == hint_l or token in dl or hint_l in dl:
                        return Path(dirpath) / d
        except OSError:
            continue
    return None


def _iter_files(root: Path, budget_s: float = SCAN_BUDGET_S,
                ) -> List[Path]:
    """Bounded recursive file walk."""
    files: List[Path] = []
    deadline = time.time() + budget_s
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith((".", "__"))
                           and d not in ("node_modules", ".venv", "venv",
                                         "site-packages", "__pycache__")]
            if dirpath[len(str(root)):].count(os.sep) >= MAX_DEPTH:
                dirnames[:] = []
            for f in filenames:
                files.append(Path(dirpath) / f)
                if len(files) >= MAX_FILES_SCANNED or time.time() > deadline:
                    return files
    except OSError as e:
        logger.debug("[LOCAL] walk error under %s: %s", root, e)
    return files


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0  # type: ignore[assignment]
    return f"{n} B"


def resolve_local_intent(intent: LocalIntent,
                         roots: Optional[Sequence[Path]] = None) -> str:
    """Execute the intent against the REAL filesystem and return an
    evidence-backed, speakable answer. Never calls an LLM."""
    root: Optional[Path] = resolve_folder(intent.folder, roots)
    if intent.folder and root is None:
        return (f"I couldn't find a folder called '{intent.folder}' on "
                f"your computer.")

    scan_root = root or Path.home()
    files = _iter_files(scan_root)

    if intent.extensions:
        files = [f for f in files if f.suffix.lower() in intent.extensions]

    if intent.modified == "today":
        cutoff = time.time() - 86400
        files = [f for f in files if _mtime(f) >= cutoff]
    elif intent.modified == "yesterday":
        lo, hi = time.time() - 2 * 86400, time.time() - 86400
        files = [f for f in files if lo <= _mtime(f) < hi]
    elif intent.modified == "week":
        cutoff = time.time() - 7 * 86400
        files = [f for f in files if _mtime(f) >= cutoff]

    ext_desc = (f" {intent.extensions[0].lstrip('.').upper()} files"
                if len(intent.extensions) == 1
                else (" document files" if ".pdf" in intent.extensions
                      or ".docx" in intent.extensions else " files"))
    ext_desc = ext_desc.strip()

    if intent.kind == LocalIntentKind.FILE_COUNT:
        scope = (f"in {root}" if root else "on your PC")
        return (f"You have {len(files)} {ext_desc.strip()} {scope} "
                f"(scanned under {scan_root}).")

    # FILE_SEARCH — sort and take top N
    if intent.sort == LocalSort.LARGEST:
        ranked = sorted(files, key=lambda f: _size(f), reverse=True)
        desc = "largest"
    elif intent.sort == LocalSort.SMALLEST:
        ranked = sorted(files, key=lambda f: _size(f))
        desc = "smallest"
    elif intent.sort in (LocalSort.NEWEST, LocalSort.OLDEST):
        reverse = intent.sort == LocalSort.NEWEST
        ranked = sorted(files, key=_mtime, reverse=reverse)
        desc = intent.sort.value
    else:
        ranked = files
        desc = "matching"

    limit = intent.limit or 5
    top = ranked[:limit]
    if not top:
        return (f"I found no {ext_desc.strip()} "
                f"{'in ' + str(root) if root else 'on your PC'}.")
    lines = []
    for f in top:
        lines.append(f"{f.name} — {_human_size(_size(f))} ({f})")
    scope = (f"in {root}" if root else "on your PC")
    return (f"The {desc} {ext_desc.strip()} {scope}: " + "; ".join(lines))


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0
