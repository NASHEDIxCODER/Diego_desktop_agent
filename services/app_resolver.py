"""AppResolver — canonical application identity resolution for desktop_open.

Problem fixed (agentic execution flow, 2026-09-17):
    The dispatcher resolved app names to executables with a narrow,
    executable-name-only map ("telegram" -> "telegram-desktop"). When the
    guess was wrong the action failed with "Couldn't find X" even though
    the app WAS installed (Telegram: /usr/local/bin/telegram,
    org.telegram.desktop.desktop, StartupWMClass=TelegramDesktop).
    Separately, verification of desktop_open relied on a process-name map
    plus a vision fallthrough that could VERIFY a failed launch from a
    generic screen change (false positive).

This module is the single runtime boundary for "which app did the user
mean, and how do I launch / recognise it?". Resolution is deterministic
and ordered:

    1. Alias table (spoken name -> canonical id).  O(1), no guessing.
    2. .desktop entry scan (XDG dirs): match entry id / Name /
       GenericName / StartupWMClass / Exec basename.  Entries that exist
       but whose launcher is missing are reported, not hidden.
    3. PATH lookup of the canonical id and common variants.
    4. Bounded fuzzy match (difflib >= 0.80) against installed entry
       names/ids only — never against arbitrary strings.
    5. Otherwise: NOT_INSTALLED (honest failure, never a guessed launch).

A resolved identity carries everything dispatch and verification need:
executable, .desktop entry id, process-name patterns, and window identity
(WM_CLASS / window-name patterns).

Verification helpers are OS-evidence only (process table via /proc,
windows via xdotool). Generic "screen changed" evidence is explicitly
NOT accepted here — see invariant 2 in tests/test_app_open_flow.py.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Canonical identities + spoken aliases
# ═══════════════════════════════════════════════════════════════

# canonical id -> metadata.  Process/window patterns are derived from the
# entry scan at runtime; these are the static seeds every resolver run
# starts from.
_CANONICAL: Dict[str, Dict[str, object]] = {
    "code": {
        "display": "VS Code",
        "aliases": ("vs code", "vscode", "visual studio code", "code",
                    "visual studio"),
        "execs": ("code", "code-insiders", "vscodium", "codium"),
        "wmclasses": ("Code",),
    },
    "firefox": {
        "display": "Firefox",
        "aliases": ("firefox", "firefox browser", "mozilla firefox",
                    "browser", "web browser"),
        "execs": ("firefox", "firefox-esr"),
        "wmclasses": ("firefox", "Firefox"),
    },
    "google-chrome": {
        "display": "Google Chrome",
        "aliases": ("chrome", "google chrome"),
        "execs": ("google-chrome", "google-chrome-stable", "chrome"),
        "wmclasses": ("google-chrome", "Google-chrome"),
    },
    "chromium": {
        "display": "Chromium",
        "aliases": ("chromium", "chromium browser"),
        "execs": ("chromium", "chromium-browser"),
        "wmclasses": ("chromium", "Chromium"),
    },
    "telegram": {
        "display": "Telegram",
        "aliases": ("telegram", "telegram desktop", "telegram-desktop",
                    "tg"),
        "execs": ("telegram", "Telegram", "telegram-desktop"),
        "wmclasses": ("TelegramDesktop", "Telegram", "telegram"),
    },
    "pycharm": {
        "display": "PyCharm",
        "aliases": ("pycharm", "jetbrains pycharm", "pycharm community",
                    "pycharm professional"),
        "execs": ("pycharm", "pycharm-community", "pycharm-professional"),
        "wmclasses": ("jetbrains-pycharm", "jetbrains-pycharm-ce"),
    },
    "gnome-terminal": {
        "display": "Terminal",
        "aliases": ("terminal", "console", "gnome terminal", "command line",
                    "shell"),
        "execs": ("gnome-terminal", "gnome-terminal-server"),
        "wmclasses": ("Gnome-terminal", "gnome-terminal-server"),
    },
    "nautilus": {
        "display": "Files",
        "aliases": ("files", "file manager", "nautilus", "file explorer",
                    "file browser"),
        "execs": ("nautilus",),
        "wmclasses": ("org.gnome.Nautilus", "Nautilus"),
    },
    "gnome-calculator": {
        "display": "Calculator",
        "aliases": ("calculator", "calc", "gnome calculator"),
        "execs": ("gnome-calculator",),
        "wmclasses": ("org.gnome.Calculator",),
    },
    "gnome-control-center": {
        "display": "Settings",
        "aliases": ("settings", "preferences", "control center"),
        "execs": ("gnome-control-center",),
        "wmclasses": ("gnome-control-center",),
    },
    "spotify": {
        "display": "Spotify",
        "aliases": ("spotify", "music", "music player", "music_player"),
        "execs": ("spotify",),
        "wmclasses": ("spotify", "Spotify"),
    },
    "slack": {
        "display": "Slack",
        "aliases": ("slack",),
        "execs": ("slack",),
        "wmclasses": ("Slack", "slack"),
    },
    "discord": {
        "display": "Discord",
        "aliases": ("discord",),
        "execs": ("discord", "Discord"),
        "wmclasses": ("discord", "Discord"),
    },
    "notion": {
        "display": "Notion",
        "aliases": ("notion", "notion-app"),
        "execs": ("notion-app", "notion"),
        "wmclasses": ("notion-app", "Notion"),
    },
    "obsidian": {
        "display": "Obsidian",
        "aliases": ("obsidian",),
        "execs": ("obsidian",),
        "wmclasses": ("obsidian", "Obsidian"),
    },
}

# spoken/alias (normalized) -> canonical id
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canon, _meta in _CANONICAL.items():
    _ALIAS_TO_CANONICAL[_canon] = _canon
    for _alias in _meta["aliases"]:  # type: ignore[union-attr]
        _ALIAS_TO_CANONICAL[_alias] = _canon

_FUZZY_THRESHOLD = 0.80
_FUZZY_MAX_CANDIDATES = 3


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    NOT_INSTALLED = "not_installed"
    AMBIGUOUS = "ambiguous"
    INVALID_REQUEST = "invalid_request"


@dataclass(frozen=True)
class AppIdentity:
    """Canonical, evidence-backed identity of one desktop application."""
    canonical: str              # e.g. "telegram"
    display_name: str           # e.g. "Telegram"
    requested: str              # raw user/planner string, e.g. "Telegram"
    executable: str             # absolute path of the launch binary
    desktop_entry: str = ""     # desktop file id, e.g. "org.telegram.desktop"
    process_patterns: Tuple[str, ...] = ()   # pgrep -x candidates
    window_patterns: Tuple[str, ...] = ()    # WM_CLASS / title candidates
    evidence: Tuple[str, ...] = ()           # human-readable resolution trail

    def matches_process(self, proc_name: str) -> bool:
        name = (proc_name or "").strip().lower()
        return any(name == p.lower() for p in self.process_patterns)

    def matches_window(self, wm_class: str, title: str = "") -> bool:
        blob = f"{wm_class or ''} {title or ''}".lower()
        return any(p.lower() in blob for p in self.window_patterns)


@dataclass
class Resolution:
    status: ResolutionStatus
    identity: Optional[AppIdentity] = None
    candidates: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    message: str = ""


# ═══════════════════════════════════════════════════════════════
# .desktop entry scanning
# ═══════════════════════════════════════════════════════════════

def _xdg_app_dirs() -> List[Path]:
    dirs: List[Path] = []
    home = Path.home() / ".local" / "share" / "applications"
    dirs.append(home)
    data_dirs = os.environ.get(
        "XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    for base in data_dirs:
        base = base.strip()
        if base:
            dirs.append(Path(base) / "applications")
    # de-dup, keep order
    seen, unique = set(), []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


_DESKTOP_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)=(.*)$")


def _parse_desktop_file(path: Path) -> Optional[Dict[str, str]]:
    """Minimal tolerant .desktop parser (Desktop Entry section only)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    in_entry = False
    fields: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_entry = line.strip().lower() == "[desktop entry]"
            continue
        if not in_entry:
            continue
        m = _DESKTOP_KEY_RE.match(line)
        if m:
            fields.setdefault(m.group(1), m.group(2).strip())
    return fields or None


@dataclass
class _DesktopEntry:
    entry_id: str          # filename without .desktop
    path: Path
    name: str
    generic: str
    exec_line: str
    try_exec: str
    wm_class: str
    no_display: bool
    terminal: bool


def _scan_desktop_entries() -> List[_DesktopEntry]:
    entries: List[_DesktopEntry] = []
    for appdir in _xdg_app_dirs():
        try:
            files = sorted(appdir.glob("*.desktop"))
        except OSError:
            continue
        for f in files:
            fields = _parse_desktop_file(f)
            if not fields:
                continue
            entries.append(_DesktopEntry(
                entry_id=f.stem,
                path=f,
                name=fields.get("Name", ""),
                generic=fields.get("GenericName", ""),
                exec_line=fields.get("Exec", ""),
                try_exec=fields.get("TryExec", ""),
                wm_class=fields.get("StartupWMClass", ""),
                no_display=fields.get("NoDisplay", "").lower() == "true",
                terminal=fields.get("Terminal", "").lower() == "true",
            ))
    return entries


def _exec_basename(exec_line: str) -> str:
    """First token of an Exec line, de-quoted, basename only."""
    if not exec_line:
        return ""
    token = exec_line.strip().split()[0].strip("\"'")
    token = token.split("/")[-1]
    # strip field codes that leaked into the first token position
    return token.strip()


def _which_or_path(candidate: str) -> Optional[str]:
    if not candidate:
        return None
    candidate = candidate.strip("\"'")
    if "/" in candidate:
        p = Path(candidate)
        try:
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
        except OSError:
            return None
        return None
    return shutil.which(candidate)


def _entry_executable(entry: _DesktopEntry) -> Optional[str]:
    """Best-effort executable for a .desktop entry (TryExec first)."""
    if entry.try_exec:
        hit = _which_or_path(entry.try_exec)
        if hit:
            return hit
    base = _exec_basename(entry.exec_line)
    if base and base not in ("%f", "%F", "%u", "%U", "%d", "%D"):
        hit = _which_or_path(base)
        if hit:
            return hit
    return None


# ═══════════════════════════════════════════════════════════════
# Resolution
# ═══════════════════════════════════════════════════════════════

def _normalize_requested(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _entry_score(entry: _DesktopEntry, canon: str,
                 meta: Dict[str, object], norm: str) -> int:
    """Deterministic match score of an entry against a canonical id.

    Higher wins. 0 = no match. Exact-name and exec-basename matches
    dominate; substring/fuzzy matches score low.
    """
    name_l = entry.name.lower()
    id_l = entry.entry_id.lower()
    exec_base = _exec_basename(entry.exec_line).lower()
    try_l = entry.try_exec.lower()
    execs = [str(e).lower() for e in meta.get("execs", ())]  # type: ignore[union-attr]
    wm_l = entry.wm_class.lower()

    if name_l == norm or id_l == norm:
        return 100
    if exec_base and (exec_base == norm or exec_base in execs):
        return 90
    if try_l and (try_l == norm or try_l in execs):
        return 90
    if norm == canon and (canon in id_l or (exec_base and canon in exec_base)):
        return 80
    if wm_l and wm_l == norm:
        return 70
    for alias in list(meta.get("aliases", ())) + [canon]:  # type: ignore[union-attr]
        a = str(alias).lower()
        if a and (a == name_l or a in id_l.split("-") or a in id_l.split("_")):
            return 60
    if norm and (norm in name_l or norm in id_l or
                 (exec_base and norm in exec_base)):
        return 30
    return 0


def resolve_app(requested: str) -> Resolution:
    """Resolve a spoken/planner app name to a canonical AppIdentity.

    Never launches anything. Pure + deterministic given host state.
    """
    t0 = time.time()
    norm = _normalize_requested(requested)
    if not norm:
        return Resolution(ResolutionStatus.INVALID_REQUEST,
                          message="No application name was provided.")

    canon = _ALIAS_TO_CANONICAL.get(norm)
    if canon is None and norm.endswith(".desktop"):
        canon = _ALIAS_TO_CANONICAL.get(norm[:-len(".desktop")])

    entries = _scan_desktop_entries()
    evidence: List[str] = [f"requested='{requested}' normalized='{norm}'",
                           f"scanned {len(entries)} desktop entries"]

    # ── Unknown alias: bounded fuzzy match over known names only ──
    if canon is None:
        known = sorted(_ALIAS_TO_CANONICAL.keys())
        close = difflib.get_close_matches(
            norm, known, n=_FUZZY_MAX_CANDIDATES, cutoff=_FUZZY_THRESHOLD)
        if len(close) == 1:
            canon = _ALIAS_TO_CANONICAL[close[0]]
            evidence.append(f"fuzzy alias match '{norm}' -> '{close[0]}'")
        elif len(close) > 1:
            cands = sorted({_ALIAS_TO_CANONICAL[c] for c in close})
            if len(cands) == 1:
                canon = cands[0]
                evidence.append(f"fuzzy alias match '{norm}' -> '{close[0]}'")
            else:
                logger.info("[APP-RESOLVE] requested='%s' status=ambiguous "
                           "candidates=%s (%.0fms)",
                           requested, cands,
                           (time.time() - t0) * 1000)
                return Resolution(
                    ResolutionStatus.AMBIGUOUS,
                    candidates=cands, evidence=evidence,
                    message=(f"'{requested}' could refer to several apps "
                             f"({', '.join(cands)}). Which one did you mean?"))
        else:
            # Last resort: fuzzy over installed entry names/ids (installed
            # apps only — never invents a target).
            pool: Dict[str, _DesktopEntry] = {}
            for e in entries:
                if e.no_display or not e.name:
                    continue
                pool.setdefault(e.name.lower(), e)
                pool.setdefault(e.entry_id.lower(), e)
            close_entries = difflib.get_close_matches(
                norm, sorted(pool.keys()),
                n=_FUZZY_MAX_CANDIDATES, cutoff=_FUZZY_THRESHOLD)
            if close_entries:
                picked = pool[close_entries[0]]
                canon = _canon_from_entry(picked)
                evidence.append(
                    f"fuzzy entry match '{norm}' -> '{picked.name}' "
                    f"({picked.entry_id})")
            else:
                logger.info("[APP-RESOLVE] requested='%s' status=not_installed "
                           "(%.0fms)", requested, (time.time() - t0) * 1000)
                return Resolution(
                    ResolutionStatus.NOT_INSTALLED,
                    evidence=evidence,
                    message=(f"'{requested}' isn't installed or isn't "
                             f"available to launch."))

    meta = _CANONICAL[canon]
    display = str(meta["display"])

    # ── Pick the best .desktop entry for this canonical id ──
    visible = [e for e in entries if not e.no_display] or entries
    scored = [(_entry_score(e, canon, meta, norm), e) for e in visible]
    scored.sort(key=lambda t: (-t[0], t[1].entry_id))
    best_entry: Optional[_DesktopEntry] = None
    best_score = scored[0][0] if scored else 0
    if scored and scored[0][0] > 0:
        # Ambiguity: two different entries tie at the top score.
        top = [e for s, e in scored if s == scored[0][0]]
        ids = sorted({e.entry_id for e in top})
        if len(ids) > 1 and scored[0][0] < 80:
            logger.info("[APP-RESOLVE] requested='%s' status=ambiguous "
                       "entries=%s (%.0fms)",
                       requested, ids, (time.time() - t0) * 1000)
            return Resolution(
                ResolutionStatus.AMBIGUOUS,
                candidates=ids, evidence=evidence,
                message=(f"'{requested}' matches several installed apps "
                         f"({', '.join(ids)}). Which one did you mean?"))
        best_entry = top[0]
        evidence.append(f"desktop entry '{best_entry.entry_id}' "
                        f"(Name='{best_entry.name}', score={best_score})")

    # ── Executable: entry first, then PATH variants ──
    exe: Optional[str] = None
    if best_entry is not None:
        exe = _entry_executable(best_entry)
        if exe:
            evidence.append(f"launcher from entry: {exe}")
    if exe is None:
        for variant in list(meta.get("execs", ())) + [canon]:  # type: ignore[union-attr]
            hit = _which_or_path(str(variant))
            if hit:
                exe = hit
                evidence.append(f"launcher from PATH: {exe}")
                break

    if exe is None:
        detail = ""
        if best_entry is not None:
            detail = (f" A desktop entry exists ('{best_entry.entry_id}') "
                      f"but its launcher is missing.")
        logger.info("[APP-RESOLVE] requested='%s' canonical='%s' "
                    "status=not_installed (%.0fms)",
                    requested, canon, (time.time() - t0) * 1000)
        return Resolution(
            ResolutionStatus.NOT_INSTALLED,
            evidence=evidence,
            message=(f"'{display}' isn't installed or isn't available to "
                     f"launch.{detail}"))

    # ── Build identity: process + window patterns ──
    procs = {canon.lower()}
    procs.update(str(e).lower() for e in meta.get("execs", ()))  # type: ignore[union-attr]
    wins = {display.lower()}
    wins.update(str(w).lower() for w in meta.get("wmclasses", ()))  # type: ignore[union-attr]
    entry_id = ""
    if best_entry is not None:
        entry_id = best_entry.entry_id
        base = _exec_basename(best_entry.exec_line)
        if base:
            procs.add(base.lower())
        if best_entry.try_exec:
            procs.add(best_entry.try_exec.lower())
        if best_entry.wm_class:
            wins.add(best_entry.wm_class.lower())
        if best_entry.name:
            wins.add(best_entry.name.lower())

    identity = AppIdentity(
        canonical=canon,
        display_name=display,
        requested=requested,
        executable=exe,
        desktop_entry=entry_id,
        process_patterns=tuple(sorted(procs)),
        window_patterns=tuple(sorted(w for w in wins if w)),
        evidence=tuple(evidence),
    )
    logger.info("[APP-RESOLVE] requested='%s' canonical='%s' exe='%s' "
                "entry='%s' status=resolved (%.0fms)",
                requested, canon, exe, entry_id or "-",
                (time.time() - t0) * 1000)
    return Resolution(ResolutionStatus.RESOLVED, identity=identity,
                      evidence=evidence,
                      message=f"Resolved '{requested}' to {display} ({exe}).")


def _canon_from_entry(entry: _DesktopEntry) -> str:
    """Best canonical id for an entry found via fuzzy matching."""
    name_l = entry.name.lower()
    id_l = entry.entry_id.lower()
    blob = f"{name_l} {id_l}"
    for canon in _CANONICAL:
        if canon in blob:
            return canon
    # Fall back to the entry id stem as an ad-hoc canonical id.
    return id_l.split("-")[0].split("_")[0]


# ═══════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════

@dataclass
class LaunchResult:
    ok: bool
    method: str = ""      # "executable" | "gtk-launch" | "xdg-open"
    command: Tuple[str, ...] = ()
    message: str = ""


def launch(identity: AppIdentity) -> LaunchResult:
    """Launch a resolved app. Returns honest ok/method — never claims
    success for a spawn that failed. Callers must still VERIFY the
    effect (process/window appeared); launching is not proof."""
    attempts: List[Tuple[str, Tuple[str, ...]]] = [
        ("executable", (identity.executable,)),
    ]
    if identity.desktop_entry and shutil.which("gtk-launch"):
        attempts.append(("gtk-launch", ("gtk-launch", identity.desktop_entry)))
    if shutil.which("xdg-open"):
        entry_path = _entry_path(identity.desktop_entry)
        if entry_path:
            attempts.append(("xdg-open", ("xdg-open", entry_path)))

    last_error = ""
    for method, cmd in attempts:
        if not cmd[0]:
            continue
        try:
            subprocess.Popen(
                list(cmd), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            logger.info("[APP-DISPATCH] launched '%s' via %s: %s",
                        identity.canonical, method, " ".join(cmd))
            return LaunchResult(True, method, cmd,
                                f"Launched {identity.display_name} "
                                f"via {method}.")
        except Exception as e:
            last_error = str(e)
            logger.debug("[APP-DISPATCH] launch via %s failed for '%s': %s",
                         method, identity.canonical, e)
    msg = (f"Failed to launch {identity.display_name}."
           + (f" ({last_error})" if last_error else ""))
    logger.warning("[APP-DISPATCH] %s", msg)
    return LaunchResult(False, "", (), msg)


def _entry_path(entry_id: str) -> str:
    if not entry_id:
        return ""
    for appdir in _xdg_app_dirs():
        candidate = appdir / f"{entry_id}.desktop"
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return ""


# ═══════════════════════════════════════════════════════════════
# OS-evidence verification helpers
# ═══════════════════════════════════════════════════════════════

def _read_comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8",
                  errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def _proc_state(pid: int) -> str:
    """Single-letter state from /proc/<pid>/stat ('Z' = zombie)."""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8",
                  errors="replace") as f:
            parts = f.read().rsplit(")", 1)
            rest = parts[1].split() if len(parts) > 1 else f.read().split()
            return rest[0] if rest else ""
    except (OSError, IndexError):
        return ""


def find_processes(patterns: Tuple[str, ...],
                   include_zombies: bool = False) -> List[int]:
    """PIDs whose /proc comm matches any pattern (case-insensitive exact).

    Exact-name match only (the pgrep -x discipline): never matches
    wrapper shells or Diego's own process tree via cmdline substrings.
    """
    wanted = {p.strip().lower() for p in patterns if p and p.strip()}
    if not wanted:
        return []
    hits: List[int] = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return []
    for pid_s in pids:
        pid = int(pid_s)
        comm = _read_comm(pid).lower()
        if comm in wanted:
            if not include_zombies and _proc_state(pid) == "Z":
                continue
            hits.append(pid)
    return hits


def _xdotool(args: List[str], timeout: int = 3) -> Tuple[int, str]:
    if not shutil.which("xdotool"):
        return 127, ""
    try:
        out = subprocess.run(["xdotool"] + args, capture_output=True,
                             text=True, timeout=timeout)
        return out.returncode, (out.stdout or "").strip()
    except Exception:
        return 1, ""


def find_windows(patterns: Tuple[str, ...]) -> List[str]:
    """Window ids whose WM_CLASS or title matches any pattern.

    Case-insensitive substring on class/title — windows are identified
    by the TARGET app's own class/name, never by "any window changed".
    """
    pats = [p.strip().lower() for p in patterns if p and p.strip()]
    if not pats:
        return []
    found: Dict[str, None] = {}
    for pat in pats:
        for mode in ("--class", "--name"):
            code, out = _xdotool(["search", "--onlyvisible", mode, pat])
            if code == 0 and out:
                for wid in out.split():
                    found[wid.strip()] = None
    # Fallback: scan all visible windows and match class+title blobs.
    if not found:
        code, out = _xdotool(["search", "--onlyvisible", ".*"])
        if code == 0 and out:
            for wid in out.split():
                wid = wid.strip()
                _, cls = _xdotool(["getwindowclassname", wid])
                _, title = _xdotool(["getwindowname", wid])
                blob = f"{cls} {title}".lower()
                if any(p in blob for p in pats):
                    found[wid] = None
    return list(found.keys())


def focused_window_identity() -> Tuple[str, str]:
    """(wm_class, title) of the currently focused window (best effort)."""
    code, out = _xdotool(["getactivewindow"])
    if code != 0 or not out:
        return "", ""
    wid = out.split()[0]
    _, cls = _xdotool(["getwindowclassname", wid])
    _, title = _xdotool(["getwindowname", wid])
    return cls, title


@dataclass
class PresenceEvidence:
    present: bool
    via: str = ""              # "process" | "window" | ""
    pids: Tuple[int, ...] = ()
    windows: Tuple[str, ...] = ()
    focused_matches: bool = False
    detail: str = ""


def check_presence(identity: AppIdentity) -> PresenceEvidence:
    """Single poll: is the TARGET app (and only the target) present?

    Process match is authoritative; window match is corroborating.
    Never consults generic screen deltas.
    """
    pids = find_processes(identity.process_patterns)
    if pids:
        return PresenceEvidence(
            True, "process", tuple(pids), (),
            detail=(f"process {identity.canonical} running "
                    f"(pids={sorted(pids)[:5]})"))
    wins = find_windows(identity.window_patterns)
    if wins:
        cls, title = focused_window_identity()
        focused = identity.matches_window(cls, title)
        return PresenceEvidence(
            True, "window", (), tuple(wins), focused,
            detail=(f"window for {identity.canonical} visible "
                    f"(ids={wins[:3]}, focused_match={focused})"))
    return PresenceEvidence(
        False, "", (),
        detail=f"no process {identity.process_patterns} and no window "
               f"{identity.window_patterns}")


def wait_for_presence(identity: AppIdentity,
                      timeout_s: float = 4.0,
                      poll_s: float = 0.3) -> PresenceEvidence:
    """Poll check_presence until timeout. Bounded; total wait <= timeout."""
    deadline = time.time() + max(0.1, timeout_s)
    last = check_presence(identity)
    while not last.present and time.time() < deadline:
        time.sleep(min(poll_s, max(0.05, deadline - time.time())))
        last = check_presence(identity)
    return last


def log_verify(action: str, identity: AppIdentity,
               evidence: PresenceEvidence, ok: bool) -> None:
    level = logger.info if ok else logger.warning
    level("[APP-VERIFY] action=%s app=%s verified=%s via=%s %s",
          action, identity.canonical, ok,
          evidence.via or "none", evidence.detail)
