"""
CurrentChromeSession — the browser-session adapter (the user's REAL Chrome).

ONE production contract for every browser goal:

    existing Chrome session
        ↓  attach / reuse the current browser (never a second instance)
    existing profile + cookies + logged-in accounts + tabs
        ↓
    BrowserGoalEngine / browser tier

What this module enforces:

  * DISCOVER, never guess: the actual running Chrome process is found via
    /proc; its real `--user-data-dir` and `--profile-directory` are read
    from the process command line (falling back to `Local State`'s
    `profile.last_used`), so nothing is hardcoded to "Default".
  * VALIDATE: the discovered profile is checked against the RUNNING Chrome
    process (SingletonLock owner pid) before it is trusted.
  * ATTACH beats launch: if a Chrome process exists, its DevTools endpoint
    is probed (running `--remote-debugging-port`, the `DevToolsActivePort`
    file of the discovered user-data dir, then the default 9222) and the
    browser is attached over CDP. No second browser instance is started
    while Chrome is already open.
  * Chrome 136+: `--remote-debugging-port` / `--remote-debugging-pipe` are
    silently IGNORED when Chrome runs on its DEFAULT user data dir
    (RemoteDebuggingServer::NotStartedReason::kDisabledByDefaultUserDataDir).
  * Chrome 144+: the existing-session auto-connect workflow is supported —
    remote debugging is switched on in the RUNNING browser via
    `chrome://inspect/#remote-debugging` and Chrome then listens on the
    DevTools port in APPROVAL MODE (even for the default user data dir); the
    user approves each incoming connection. This is the supported way to
    drive an ALREADY-OPEN session without touching its profile.
  * NO fresh profile, NO profile copy, NO temporary profile, NO silent
    fallback to a clean browser for production tasks. If Chrome is running
    but cannot be attached, the caller receives
    `BROWSER_SESSION_UNAVAILABLE` with the EXACT reason and what to enable.
  * The user's profile is read (Local State, SingletonLock) — never written,
    copied or modified. A browser is launched only when NO Chrome runs at
    all, and then with the user's REAL user-data dir + profile so every
    login is preserved.

Logging: [CHROME-SESSION]
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default Chrome DevTools / remote-debugging port (also Chrome 144+'s
# approval-mode default: RemoteDebuggingServer.kDefaultDevToolsPort).
DEFAULT_CDP_PORT = 9222

# Environment overrides (documented, explicit — never silent).
ENV_CDP_PORT = "DIEGO_BROWSER_CDP_PORT"
ENV_ALLOW_LAUNCH = "DIEGO_BROWSER_ALLOW_LAUNCH"   # default: enabled
ENV_USER_DATA_DIR = "DIEGO_BROWSER_USER_DATA_DIR"
ENV_PROFILE_DIR = "DIEGO_BROWSER_PROFILE_DIR"

# Marker embedded in every honest failure (callers and logs can grep it).
UNAVAILABLE = "BROWSER_SESSION_UNAVAILABLE"

# First Chrome major with the existing-session auto-connect workflow
# (`chrome://inspect/#remote-debugging` → approval-mode remote debugging,
# which is the ONLY way to attach to a running default-dir session).
_AUTO_CONNECT_MAJOR = 144

# The Chrome 144+ existing-session switch, as it appears in the ACTIVE
# profile's `Preferences` file. The toggle on
# `chrome://inspect#remote-debugging` (exact string from the Chrome binary's
# `chrome/browser/devtools/remote_debugging_server.cc`) writes
# `devtools.remote_debugging.user-enabled`, and
# `devtools.remote_debugging.allowed` records the admin-policy gate.
# Verified against the installed Chrome 149 binary strings.
REMOTE_DEBUGGING_PREF = "devtools.remote_debugging.user-enabled"
REMOTE_DEBUGGING_ALLOWED_PREF = "devtools.remote_debugging.allowed"
REMOTE_DEBUGGING_TOGGLE_URL = "chrome://inspect#remote-debugging"

# Bounded wait for a freshly launched Chrome to expose CDP.
_LAUNCH_WAIT_S = 20.0
_PROBE_TIMEOUT_S = 0.8
# The Chrome 144+ approval-mode handshake: the WebSocket stays pending until
# the user clicks "Allow" on Chrome's debugging-connection prompt. This budget
# is consumed by `connect_over_cdp(timeout=...)`, which takes MILLISECONDS.
_APPROVAL_WAIT_MS = 90_000.0
# The same budget expressed in seconds — for user-facing messages only.
_APPROVAL_WAIT_S = _APPROVAL_WAIT_MS / 1000.0

_CHROME_EXE_TOKENS = ("chrome", "chromium")

# Substrings that mark a Chrome CHILD process rather than the main browser
# process. Matched against the JOINED command line, because Chrome rewrites
# argv in place and often collapses it into one space-joined blob.
_CHILD_MARKERS = (
    "--type=",             # renderer / gpu-process / utility / zygote / ppapi
    "crashpad_handler",    # the standalone crash handler binary
    "--zygote",            # the zygote helper
)

# Where Chrome keeps its default user data dir per platform (checked in
# order; the first existing directory wins). Nothing here is copied.
_DEFAULT_USER_DATA_DIRS = (
    "~/.config/google-chrome",
    "~/.config/google-chrome-stable",
    "~/.config/google-chrome-beta",
    "~/.config/chromium",
    "~/snap/chromium/common/chromium",
    "~/Library/Application Support/Google/Chrome",
    "~/AppData/Local/Google/Chrome/User Data",
)

_CHROME_BINARIES = (
    "google-chrome", "google-chrome-stable", "google-chrome-beta",
    "chromium-browser", "chromium", "chrome",
)


class ChromeSessionUnavailable(RuntimeError):
    """Chrome cannot be attached — with the exact reason + remediation.

    `info` carries the discovered-session evidence (browser_process,
    user_data_dir, profile_directory, ...) even though the attach failed, so
    the AgentTrace can still report WHAT Chrome was found and why it could
    not be used — never an anonymous failure.
    """

    def __init__(self, reason: str, remediation: str,
                 info: Optional["ChromeSessionInfo"] = None) -> None:
        super().__init__(f"{UNAVAILABLE}: {reason} Remediation: {remediation}")
        self.reason = reason
        self.remediation = remediation
        self.info = info


@dataclass
class ChromeSessionInfo:
    """Everything known about the user's CURRENT Chrome session.

    `evidence()` is the exact browser-session evidence recorded on the
    AgentTrace (browser_process, user_data_dir, profile_directory,
    connection_method, authenticated_state, active_tab, current_url).
    """

    browser_process: str = ""
    version: str = ""
    user_data_dir: str = ""
    profile_directory: str = ""
    connection_method: str = "none"
    authenticated_state: str = "unknown"   # unknown|authenticated|unauthenticated
    active_tab: str = ""
    current_url: str = ""
    debug_port: Optional[int] = None
    pid: Optional[int] = None
    running: bool = False
    launched_by_diego: bool = False
    profile_locked: bool = False           # SingletonLock present (in use)
    extra: Dict[str, Any] = field(default_factory=dict)

    def evidence(self) -> Dict[str, str]:
        """The 7 trace fields — nothing more, nothing less."""
        return {
            "browser_process": self.browser_process or "unknown",
            "user_data_dir": self.user_data_dir or "unknown",
            "profile_directory": self.profile_directory or "unknown",
            "connection_method": self.connection_method or "none",
            "authenticated_state": self.authenticated_state or "unknown",
            "active_tab": self.active_tab or "",
            "current_url": self.current_url or "",
        }


# ── discovery (pure, injectable for tests) ───────────────────────────

def parse_chrome_args(args: List[str]) -> Dict[str, Any]:
    """Extract session-defining switches from a Chrome command line."""
    out: Dict[str, Any] = {"user_data_dir": "", "profile_directory": "",
                           "debug_port": None}
    for i, arg in enumerate(args):
        if arg.startswith("--user-data-dir="):
            out["user_data_dir"] = arg.split("=", 1)[1]
        elif arg == "--user-data-dir" and i + 1 < len(args):
            out["user_data_dir"] = args[i + 1]
        elif arg.startswith("--profile-directory="):
            out["profile_directory"] = arg.split("=", 1)[1]
        elif arg.startswith("--remote-debugging-port="):
            try:
                out["debug_port"] = int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return out


def _cmdline_tokens(args: List[str]) -> List[str]:
    """Recover argv from a REWRITTEN cmdline.

    Chrome overwrites the argv memory of its own process and of every child
    in place, so ``/proc/<pid>/cmdline`` frequently collapses into a SINGLE
    NUL-free, space-joined blob (verified on Chrome 149/Linux: renderer,
    GPU, utility, zygote and crashpad entries). The main browser process is
    often reduced to just its exe path, so its session switches can be gone
    too — which is exactly why the profile must also be discovered from the
    on-disk Local State and validated against the running process.
    """
    if len(args) == 1 and " " in args[0]:
        return args[0].split()
    return list(args)


def is_chrome_child(tokens: List[str]) -> bool:
    """True for renderer/GPU/utility/zygote/crashpad child processes.

    Child detection is marker-based on the JOINED command line, so it still
    works when the argv has been flattened into one blob.
    """
    joined = " ".join(tokens)
    return any(marker in joined for marker in _CHILD_MARKERS)


def iter_chrome_processes(proc_dir: Path = Path("/proc")):
    """Yield (pid, exe, args) for MAIN Chrome browser processes only.

    Child processes (`--type=renderer` etc.) and the crashpad handler are
    excluded: only the main browser process carries the session switches.
    """
    try:
        entries = sorted(proc_dir.iterdir(), key=lambda p: p.name)
    except Exception:
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except Exception:
            continue                      # not ours / already gone
        tokens = _cmdline_tokens([a.decode("utf-8", "replace")
                                  for a in raw.split(b"\0") if a])
        if not tokens:
            continue
        exe = tokens[0]
        if not any(tok in os.path.basename(exe).lower()
                   for tok in _CHROME_EXE_TOKENS):
            continue
        if is_chrome_child(tokens):
            continue                      # child process, not the browser
        yield int(entry.name), exe, tokens


def _select_main_process(candidates: List[tuple],
                         proc_dir: Path) -> Optional[tuple]:
    """Pick the process that IS the user's live browser.

    Preference order (never a guess about which profile is in use):
      1. the pid owning the SingletonLock of the platform-default user-data
         dir — THE source of truth for "the Chrome that is really running";
      2. the first candidate that exposes session switches (--user-data-dir /
         --remote-debugging-port), i.e. a fully readable cmdline;
      3. the first candidate found (lowest pid) as a last resort.
    """
    if not candidates:
        return None
    if proc_dir == Path("/proc"):
        # The lock is only meaningful for the REAL process table.
        lock_pid = singleton_lock_pid(resolve_default_user_data_dir())
        if lock_pid is not None:
            for cand in candidates:
                if cand[0] == lock_pid:
                    return cand
    for cand in candidates:
        switches = parse_chrome_args(cand[2])
        if switches["user_data_dir"] or switches["debug_port"] is not None:
            return cand
    return candidates[0]


def default_user_data_dir_candidates() -> List[Path]:
    home = Path.home()
    return [Path(p).expanduser() for p in _DEFAULT_USER_DATA_DIRS]


def resolve_default_user_data_dir() -> str:
    """The platform-default Chrome user data dir (first existing)."""
    override = os.environ.get(ENV_USER_DATA_DIR)
    if override:
        return str(Path(override).expanduser())
    for candidate in default_user_data_dir_candidates():
        if candidate.is_dir():
            return str(candidate)
    return str(default_user_data_dir_candidates()[0])


def read_profile_directory(user_data_dir: str) -> str:
    """The profile Chrome ACTUALLY uses (Local State `profile.last_used`).

    Read-only. Falls back to "Default" only when the file is absent or the
    recorded profile is not one of the known ones — never invented.
    """
    override = os.environ.get(ENV_PROFILE_DIR)
    if override:
        return override
    try:
        local_state = json.loads(
            (Path(user_data_dir) / "Local State").read_text(
                encoding="utf-8", errors="replace"))
        profile = local_state.get("profile") or {}
        last_used = str(profile.get("last_used") or "").strip()
        known = set((profile.get("info_cache") or {}).keys())
        if last_used:
            if not known or last_used in known:
                return last_used
        elif known:
            return sorted(known)[0]
    except Exception as e:
        logger.debug("[CHROME-SESSION] Local State unreadable: %s", e)
    return "Default"


def read_profile_pref(user_data_dir: str, profile_directory: str,
                      pref: str) -> Any:
    """Read ONE pref from the ACTIVE profile's `Preferences` (read-only).

    Chrome keeps this file open for writing, so the read can race a partial
    flush — an unreadable file simply means "unknown", never a guess.
    """
    path = (Path(user_data_dir) / profile_directory / "Preferences")
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        logger.debug("[CHROME-SESSION] Preferences unreadable (%s): %s",
                     path, e)
        return None
    node: Any = data
    for part in pref.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def remote_debugging_state(info: "ChromeSessionInfo") -> str:
    """Whether the ACTIVE profile allows the Chrome 144+ existing-session flow.

    Returns one of:
      * ``enabled``      — the user toggle is on: Chrome publishes the
                           endpoint when a client requests it (no restart).
      * ``policy-blocked`` — an admin policy forbids it.
      * ``disabled``     — the toggle is off (the default).
      * ``unknown``      — the pref could not be read (Chrome < 144, profile
                           not on disk, or a racing/unreadable file).
    """
    if not (info.user_data_dir and info.profile_directory):
        return "unknown"
    enabled = read_profile_pref(info.user_data_dir, info.profile_directory,
                                REMOTE_DEBUGGING_PREF)
    allowed = read_profile_pref(info.user_data_dir, info.profile_directory,
                                REMOTE_DEBUGGING_ALLOWED_PREF)
    info.extra["remote_debugging_pref"] = enabled
    info.extra["remote_debugging_allowed_pref"] = allowed
    if allowed is False:
        return "policy-blocked"
    if enabled is True:
        return "enabled"
    if enabled is False:
        return "disabled"
    return "unknown"


def remote_debugging_pref_state(user_data_dir: str,
                                profile_directory: str,
                                ) -> Optional[bool]:
    """Chrome 144+ existing-session switch state from the ACTIVE profile.

    Read-only probe of `<profile>/Preferences` →
    `devtools.remote_debugging.user-enabled`. Returns True (the user already
    turned the toggle on), False (explicitly off), or None when the key/file
    is absent (older Chrome, or never touched) — never a guess.
    """
    if not user_data_dir or not profile_directory:
        return None
    path = (Path(user_data_dir) / profile_directory / "Preferences")
    try:
        prefs = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    for key in (REMOTE_DEBUGGING_PREF, REMOTE_DEBUGGING_ALLOWED_PREF):
        node: Any = prefs
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, bool):
            return node
    return None


def singleton_lock_pid(user_data_dir: str) -> Optional[int]:
    """Owner pid encoded in the profile's SingletonLock symlink."""
    try:
        target = os.readlink(str(Path(user_data_dir) / "SingletonLock"))
        tail = target.rsplit("-", 1)[-1]
        if tail.isdigit():
            return int(tail)
    except Exception:
        pass
    return None


def chrome_version(proc_dir: Path = Path("/proc"),
                   binary: Optional[str] = None) -> str:
    """Best-effort Chrome version: crashpad `ver=` annotation, else --version."""
    try:
        for entry in proc_dir.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except Exception:
                continue
            for arg in raw.decode("utf-8", "replace").split("\0"):
                if arg.startswith("--annotation=ver="):
                    # `--annotation=ver=149.0.7827.102` → "149.0.7827.102"
                    return arg.split("ver=", 1)[1]
    except Exception:
        pass
    if binary:
        try:
            out = subprocess.run([binary, "--version"], capture_output=True,
                                 text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except Exception:
            pass
    return ""


def default_debug_port() -> int:
    try:
        return int(os.environ.get(ENV_CDP_PORT, DEFAULT_CDP_PORT))
    except ValueError:
        return DEFAULT_CDP_PORT


def devtools_active_port(user_data_dir: str) -> Optional[int]:
    """Port Chrome itself recorded in `<user-data-dir>/DevToolsActivePort`."""
    try:
        first = (Path(user_data_dir) / "DevToolsActivePort").read_text(
            encoding="utf-8", errors="replace").splitlines()[0].strip()
        return int(first) if first.isdigit() else None
    except Exception:
        return None


def devtools_active_ws(user_data_dir: str) -> Optional[str]:
    """The browser WebSocket path Chrome records in DevToolsActivePort.

    Line 1 is the port, line 2 the `/devtools/browser/<uuid>` path. This is
    the Chrome 144+ existing-session (approval-mode) endpoint: the server
    exists only when the user switched remote debugging ON, and the WebSocket
    handshake stays PENDING until the user approves the connection prompt.
    """
    try:
        lines = (Path(user_data_dir) / "DevToolsActivePort").read_text(
            encoding="utf-8", errors="replace").split()
        if len(lines) >= 2 and lines[0].isdigit() and lines[1].startswith("/"):
            return f"ws://127.0.0.1:{lines[0]}{lines[1]}"
    except Exception:
        pass
    return None


def remote_debugging_enabled(user_data_dir: str,
                             profile_directory: str) -> Optional[bool]:
    """Is remote debugging ON for this EXACT Chrome profile?

    Thin boolean view over :func:`read_profile_pref` for the Chrome 144+
    existing-session toggle (`devtools.remote_debugging.user-enabled`).

    Returns True / False, or None when genuinely unknown (no Preferences
    file, or an older Chrome that has no such key at all — an older Chrome
    exposes CDP only via `--remote-debugging-port`, so its absence is NOT
    evidence of a disabled toggle). An admin-policy block reports False.
    """
    if read_profile_pref(user_data_dir, profile_directory,
                         REMOTE_DEBUGGING_ALLOWED_PREF) is False:
        return False                    # blocked by enterprise policy
    value = read_profile_pref(user_data_dir, profile_directory,
                              REMOTE_DEBUGGING_PREF)
    return None if value is None else bool(value)




def chrome_processes_owned_by_diego(proc_dir: Path = Path("/proc")
                                    ) -> set:
    """Pids of Chrome main processes started with Diego's launch marker.

    `launch_real_chrome` tags the processes it spawns with a distinctive
    per-launch window title (`--window-name`), so a later discovery pass can
    tell "Diego launched this on the user's real profile" from "the user's
    own Chrome" — honest connection_method reporting.
    """
    marker = _diego_launch_marker()
    pids: set = set()
    try:
        for pid, _exe, args in iter_chrome_processes(proc_dir):
            if marker in args:
                pids.add(pid)
    except Exception:
        pass
    return pids


def _diego_launch_marker() -> str:
    """The unique marker string for THIS Diego process' launches."""
    global _LAUNCH_MARKER
    if not _LAUNCH_MARKER:
        _LAUNCH_MARKER = f"{_LAUNCH_MARKER_PREFIX}{os.getpid()}"
    return _LAUNCH_MARKER


_LAUNCH_MARKER = ""
_LAUNCH_MARKER_PREFIX = "diego-browser-session-"


def profile_path(user_data_dir: str, profile_directory: str) -> str:
    """Absolute path of the profile Chrome ACTUALLY uses. Read-only."""
    if not user_data_dir or not profile_directory:
        return ""
    return str(Path(user_data_dir) / profile_directory)


def validate_profile_against_process(info: "ChromeSessionInfo"
                                     ) -> Dict[str, Any]:
    """Prove the DISCOVERED profile belongs to the RUNNING Chrome process.

    Three independent, strictly READ-ONLY checks:

    1. ``profile_path_exists``      — the profile dir is really on disk
    2. ``profile_in_user_data_dir`` — it lives INSIDE the discovered data dir
    3. ``profile_is_live_profile``  — `Local State` (the file Chrome itself
       maintains) lists it, i.e. Chrome considers it a real profile
    4. ``singleton_lock_matches_pid`` — the data dir's ``SingletonLock``
       owner pid IS the discovered running pid

    Nothing is created, copied or written: a validation failure is reported
    honestly so the caller can refuse instead of attaching to the wrong
    profile. Results land in ``ChromeSessionInfo.extra`` as session
    evidence.
    """
    result: Dict[str, Any] = {
        "profile_path": "",
        "profile_path_exists": False,
        "profile_in_user_data_dir": False,
        "profile_is_live_profile": None,
        "singleton_lock_matches_pid": None,
    }
    udd = str(info.user_data_dir or "")
    profile = str(info.profile_directory or "")

    if udd and profile:
        try:
            result["profile_path"] = profile_path(udd, profile)
            resolved = Path(result["profile_path"]).resolve()
            result["profile_path_exists"] = resolved.is_dir()
            result["profile_in_user_data_dir"] = (
                resolved == Path(udd).resolve()
                or Path(udd).resolve() in resolved.parents)
        except Exception:
            pass
        try:
            local_state = json.loads(
                (Path(udd) / "Local State").read_text(
                    encoding="utf-8", errors="replace"))
            known = set((local_state.get("profile") or {})
                        .get("info_cache", {}) or {})
            if known:
                result["profile_is_live_profile"] = profile in known
        except Exception:
            pass

    lock_pid = singleton_lock_pid(udd) if udd else None
    if lock_pid is not None and info.pid is not None:
        result["singleton_lock_matches_pid"] = (lock_pid == info.pid)
    return result


def discover_chrome_session(proc_dir: Path = Path("/proc"),
                            chrome_version_fn=chrome_version) -> ChromeSessionInfo:
    """Find the user's CURRENT Chrome session — dynamically, never 'Default'.

    The main browser process is SELECTED (not merely the first found): the
    pid owning the platform-default SingletonLock — the source of truth for
    "the Chrome that is really running" — wins when it is among the
    candidates. The discovered data dir + profile are then validated against
    that process by :func:`validate_profile_against_process`.
    """
    info = ChromeSessionInfo()

    candidates = list(iter_chrome_processes(proc_dir))
    chosen = _select_main_process(candidates, proc_dir)
    info.extra["candidate_processes"] = len(candidates)

    if chosen is not None:
        pid, exe, args = chosen
        switches = parse_chrome_args(args)
        info.pid = pid
        info.running = True
        info.browser_process = f"chrome pid={pid} ({exe})"
        # The launcher marker survives in the cmdline across Diego restarts:
        # a browser carrying it was started by Diego's existing-session
        # launcher (on the REAL profile) — honest connection_method evidence.
        info.launched_by_diego = any(
            a.startswith(f"--window-name={_LAUNCH_MARKER_PREFIX}")
            for a in args)
        info.user_data_dir = (switches["user_data_dir"]
                              or resolve_default_user_data_dir())
        info.debug_port = switches["debug_port"]
        parsed_profile = switches["profile_directory"]
        if not parsed_profile:
            info.profile_directory = read_profile_directory(info.user_data_dir)
        elif (Path(info.user_data_dir) / parsed_profile).is_dir():
            info.profile_directory = parsed_profile
        else:
            # Chrome rewrites its argv in place: a profile value containing a
            # space ("Profile 3") can be truncated ("Profile"). The parsed
            # profile is then a directory that does not exist — the on-disk
            # Local State is the truth and wins.
            info.profile_directory = read_profile_directory(info.user_data_dir)
        info.version = chrome_version_fn(proc_dir=proc_dir, binary=exe)
    else:
        # No browser process found: fall back to the on-disk default session
        # so a launch (only when nothing is running) still uses the REAL
        # profile with every login preserved.
        info.user_data_dir = resolve_default_user_data_dir()
        info.profile_directory = read_profile_directory(info.user_data_dir)
        info.version = chrome_version_fn(proc_dir=proc_dir)
        info.browser_process = ""

    if info.user_data_dir:
        lock_pid = singleton_lock_pid(info.user_data_dir)
        info.profile_locked = lock_pid is not None
        if info.profile_locked and info.pid is None:
            # A profile in use that our /proc scan could not see (another
            # user / namespace): still running — never launch over it.
            info.running = True
        # VALIDATION: the discovered user-data dir + profile must belong to
        # the discovered RUNNING process — not merely coincide with it.
        info.extra.update(validate_profile_against_process(info))

    # DevToolsActivePort of the DISCOVERED data dir is a connection candidate.
    if info.debug_port is None and info.user_data_dir:
        info.extra["devtools_active_port"] = devtools_active_port(
            info.user_data_dir)

    # A DevTools listener opened by the browser process itself is the
    # STRONGEST candidate: Chrome may not show --remote-debugging-port on the
    # (rewritten) command line, yet the running session still serves CDP.
    if info.running and info.pid is not None:
        listening = _listening_ports_of_pid(info.pid, proc_dir)
        if listening:
            info.extra["debug_listen_ports"] = listening
    # Chrome 144+ existing-session switch, read from the ACTIVE profile so an
    # unattachable browser is reported with the exact enablement state.
    if info.user_data_dir and info.profile_directory:
        info.extra["remote_debugging"] = remote_debugging_state(info)
    return info


def _listening_ports_of_pid(pid: int, proc_dir: Path = Path("/proc")
                            ) -> List[int]:
    """Listen ports OWNED by `pid` — fd `socket:[inode]`s matched to tcp.

    `/proc/<pid>/net/tcp` is the WHOLE net-namespace table (every process in
    the namespace reads the same file), so its ports are only meaningful for
    this process after they are restricted to the process' own socket fds.
    That match is what keeps unrelated system listeners (dns, postgres,
    kube-apiserver, …) out of the Chrome attach probe.
    """
    inodes: set = set()
    try:
        for fd in (proc_dir / str(pid) / "fd").iterdir():
            try:
                target = os.readlink(str(fd))
            except Exception:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[len("socket:["):-1])
    except Exception:
        pass
    if not inodes:
        return []
    ports: List[int] = []
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc_dir / str(pid) / "net" / name).read_text(
                encoding="utf-8", errors="replace").splitlines()[1:]
        except Exception:
            continue
        for line in lines:
            fields = line.split()
            # fields[9] is the socket inode; fields[3] == "0A" is LISTEN.
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
            except Exception:
                continue
            if port and port not in ports:
                ports.append(port)
    return ports


def major_version(version: str) -> int:
    """First version number in a string like 'Chrome/149.0.7827.102'."""
    try:
        import re
        match = re.search(r"\d+", str(version or ""))
        return int(match.group(0)) if match else 0
    except Exception:
        return 0


# ── connection helpers ───────────────────────────────────────────────

def probe_cdp(port: int, timeout: float = _PROBE_TIMEOUT_S
              ) -> Optional[Dict[str, Any]]:
    """`/json/version` answers → the endpoint is a REAL CDP browser."""
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version",
                timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def connect_cdp(url: str, timeout: float = 10000.0):
    """Playwright connect_over_cdp — attaches WITHOUT disturbing the browser.

    `timeout` is the WebSocket handshake budget in milliseconds. For the
    Chrome 144+ existing-session endpoint this window covers the time the
    user needs to approve the connection prompt (the handshake stays pending
    until they do).
    """
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(url, timeout=timeout)
    except Exception:
        try:
            pw.stop()
        except Exception:
            pass
        raise
    return pw, browser


def candidate_debug_ports(info: ChromeSessionInfo) -> List[int]:
    """Every plausible DevTools endpoint of the CURRENT session, in order.

    1. the running process' own --remote-debugging-port
    2. a listener opened by the running browser process itself
    3. the port recorded in the discovered user-data dir (DevToolsActivePort)
    4. the default DevTools port (9222) — Chrome 144+ approval mode
    """
    ports: List[int] = []
    if info.debug_port:
        ports.append(info.debug_port)
    for port in info.extra.get("debug_listen_ports") or []:
        if isinstance(port, int) and port not in ports:
            ports.append(port)
    active = info.extra.get("devtools_active_port")
    if isinstance(active, int) and active not in ports:
        ports.append(active)
    default = default_debug_port()
    if default not in ports:
        ports.append(default)
    return ports


def find_chrome_binary() -> Optional[str]:
    for name in _CHROME_BINARIES:
        path = shutil.which(name)
        if path:
            return path
    return None


def launch_real_chrome(info: ChromeSessionInfo, port: int) -> bool:
    """Start the user's REAL Chrome (real data dir + profile) with CDP.

    Only called when NO Chrome is running at all. This preserves every
    cookie/login — it is NOT a fresh or temporary profile.
    """
    binary = find_chrome_binary()
    if not binary:
        logger.warning("[CHROME-SESSION] no Chrome binary found")
        return False
    cmd = [
        binary,
        f"--user-data-dir={info.user_data_dir}",
        f"--profile-directory={info.profile_directory}",
        f"--remote-debugging-port={port}",
        # A distinctive marker lets a later discovery pass attribute this
        # browser to Diego's existing-session launcher (honest evidence).
        f"--window-name={_diego_launch_marker()}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        cmd.append("--headless=new")
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as e:
        logger.warning("[CHROME-SESSION] launch failed: %s", e)
        return False
    info.launched_by_diego = True
    info.debug_port = port
    deadline = time.time() + _LAUNCH_WAIT_S
    while time.time() < deadline:
        if probe_cdp(port):
            return True
        time.sleep(0.5)
    return False


# ── the adapter ──────────────────────────────────────────────────────

class CurrentChromeSession:
    """Attach to (or, only when nothing runs, launch) the REAL Chrome."""

    def __init__(self, *, allow_launch: Optional[bool] = None) -> None:
        if allow_launch is None:
            allow_launch = os.environ.get(ENV_ALLOW_LAUNCH, "1") != "0"
        self._allow_launch = allow_launch
        self._pw: Any = None
        self._browser: Any = None

    # -- attach -------------------------------------------------------

    def attach(self, *, allow_launch: Optional[bool] = None
               ) -> Tuple[ChromeSessionInfo, Any, Any, Any, Any]:
        """Attach to the current Chrome session.

        Returns (info, playwright, browser, context, page) where `page` is
        the user's EXISTING active tab. Raises ChromeSessionUnavailable —
        with the exact reason and remediation — on any failure. It never
        falls back to a fresh/clean browser.
        """
        launch = (self._allow_launch if allow_launch is None
                  else allow_launch)
        info = discover_chrome_session()

        # ── 1. Chrome 144+ existing-session (approval-mode) endpoint ──
        # The DAP WebSocket only exists when the user switched remote
        # debugging ON; the handshake then stays PENDING until they approve
        # the connection prompt Chrome shows. This is the ONLY way to attach
        # to a default-dir Chrome 136+ session without touching the profile.
        ws_url = devtools_active_ws(info.user_data_dir)
        if ws_url:
            try:
                pw, browser = connect_cdp(ws_url, timeout=_APPROVAL_WAIT_MS)
            except Exception as e:
                raise ChromeSessionUnavailable(
                    reason=(f"the existing-session debugging endpoint "
                            f"({ws_url.split('/devtools')[0]}…) did not "
                            f"complete the WebSocket handshake within "
                            f"{_APPROVAL_WAIT_S:.0f}s: {e}"),
                    remediation=(
                        "approve the debugging-connection prompt Chrome "
                        "shows for this connection (make sure 'Remote "
                        "debugging' is switched on at "
                        "chrome://inspect/#remote-debugging in the running "
                        "browser), then retry"),
                    info=info,
                ) from e
            self._pw, self._browser = pw, browser
            context = browser.contexts[0] if browser.contexts else \
                browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.bring_to_front()
            except Exception:
                pass
            info.connection_method = \
                f"cdp:existing_session_approval_mode:{info.debug_port or ''}"
            try:
                info.active_tab = str(page.title() or "")
                info.current_url = str(page.url or "")
            except Exception:
                pass
            logger.info("[CHROME-SESSION] attached via approval-mode: %s",
                        info.evidence())
            return info, pw, browser, context, page

        # ── 2. classic debug-port endpoints ──────────────────────────
        for port in candidate_debug_ports(info):
            version = probe_cdp(port)
            if not version:
                continue
            try:
                pw, browser = connect_cdp(f"http://127.0.0.1:{port}")
            except Exception as e:
                raise ChromeSessionUnavailable(
                    reason=(f"the DevTools endpoint on port {port} answered "
                            f"(/json/version) but the WebSocket attach was "
                            f"refused: {e}"),
                    remediation=(
                        f"Chrome {_AUTO_CONNECT_MAJOR}+ existing-session "
                        f"auto-connect runs in APPROVAL mode: approve the "
                        f"debugging-connection prompt Chrome shows for this "
                        f"connection (make sure 'Remote debugging' is switched "
                        f"on at chrome://inspect/#remote-debugging in the "
                        f"running browser), then retry"),
                    info=info,
                ) from e
            self._pw, self._browser = pw, browser
            context = browser.contexts[0] if browser.contexts else \
                browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.bring_to_front()
            except Exception:
                pass
            if info.launched_by_diego:
                # Diego started this browser on the user's real profile.
                info.connection_method = f"cdp:relaunched_real_profile:{port}"
            elif port == info.debug_port:
                info.connection_method = \
                    "cdp:running_chrome_debug_port"
            elif info.running:
                info.connection_method = \
                    f"cdp:auto_connect_approval_mode(port {port})"
            else:
                info.connection_method = f"cdp:relaunched_real_profile:{port}"
            info.version = str(version.get("Browser") or info.version)
            try:
                info.active_tab = str(page.title() or "")
                info.current_url = str(page.url or "")
            except Exception:
                pass
            logger.info("[CHROME-SESSION] attached: %s", info.evidence())
            return info, pw, browser, context, page

        if info.running:
            # Chrome IS running but exposes no attachable endpoint.
            raise self._unavailable_for_running(info)
        if launch:
            if launch_real_chrome(info, default_debug_port()):
                return self.attach(allow_launch=False)
            raise ChromeSessionUnavailable(
                reason=(f"no Chrome is running and launching it on the real "
                        f"profile ({info.user_data_dir} / "
                        f"{info.profile_directory}) did not expose a DevTools "
                        f"endpoint within {_LAUNCH_WAIT_S:.0f}s"),
                remediation=(
                    "start Chrome yourself with "
                    "--remote-debugging-port=9222, or check "
                    f"{ENV_ALLOW_LAUNCH} is not set to 0"),
                info=info,
            )
        raise ChromeSessionUnavailable(
            reason="no Chrome session is running and launching is disabled",
            remediation=f"start Chrome, or set {ENV_ALLOW_LAUNCH}=1",
            info=info,
        )

    # -- honest failure reasons ---------------------------------------

    def _unavailable_for_running(self, info: ChromeSessionInfo
                                 ) -> ChromeSessionUnavailable:
        probed = ", ".join(str(p) for p in candidate_debug_ports(info))
        major = major_version(info.version)
        default_dir = self._is_default_dir(info.user_data_dir)
        sw_state = str(info.extra.get("remote_debugging")
                       or remote_debugging_state(info))
        reason = (f"Chrome is running ({info.browser_process or 'pid unknown'},"
                  f" user-data-dir {info.user_data_dir}, profile "
                  f"{info.profile_directory}) but exposes no reachable "
                  f"DevTools/remote-debugging endpoint (probed ports: {probed})")
        if sw_state != "unknown":
            reason += (f"; the existing-session remote-debugging switch in "
                       f"this profile is '{sw_state}' "
                       f"({REMOTE_DEBUGGING_PREF})")
        if major >= 136 and default_dir:
            reason += (
                f"; Chrome {major} silently IGNORES --remote-debugging-port/"
                f"-pipe on the DEFAULT user data dir (Chrome 136+ security "
                f"restriction, 'Changes to remote debugging switches to "
                f"improve security')")
        # The remediation is version- and profile-aware: the existing-session
        # auto-connect workflow only exists from Chrome 144 (and is the ONLY
        # way to attach to a default-dir session without touching the profile).
        if major >= _AUTO_CONNECT_MAJOR:
            if sw_state == "enabled":
                open_line = (
                    f"the 'Remote debugging' switch is ALREADY ON in this "
                    f"profile, so Chrome {major} publishes the endpoint only "
                    f"when Diego asks: APPROVE the debugging-connection prompt "
                    f"Chrome shows (port {default_debug_port()}, approval "
                    f"mode)")
            elif sw_state == "policy-blocked":
                open_line = (
                    f"remote debugging is BLOCKED BY ADMIN POLICY for this "
                    f"profile ({REMOTE_DEBUGGING_ALLOWED_PREF}=false), so the "
                    f"existing-session workflow cannot be enabled here — ask "
                    f"your administrator, or use a Chrome profile/install "
                    f"that policy does not manage")
            else:
                open_line = (
                    f"enable the existing-session auto-connect workflow in the "
                    f"RUNNING Chrome {major}: open {REMOTE_DEBUGGING_TOGGLE_URL} "
                    f"in that browser window, turn on 'Remote debugging' "
                    f"(writes {REMOTE_DEBUGGING_PREF}), then APPROVE the "
                    f"debugging-connection prompt Chrome shows when Diego "
                    f"attaches (port {default_debug_port()}, approval mode)")
            remediation = (
                f"{open_line}. Your current tabs, cookies and logged-in "
                f"accounts stay exactly as they are — Diego attaches to THIS "
                f"browser and never starts a second instance")
            if not default_dir:
                remediation += (
                    f". This profile is already non-default, so the other "
                    f"supported option also applies: quit Chrome and let Diego "
                    f"relaunch it on the SAME user-data-dir "
                    f"({info.user_data_dir}) with "
                    f"--remote-debugging-port={default_debug_port()}")
        else:
            remediation = (
                f"Chrome {major or 'this version'} cannot be attached while it "
                f"keeps running on the default user data dir: "
                f"--remote-debugging-port is ignored there (Chrome 136+ "
                f"restriction) and the approval-mode auto-connect workflow "
                f"only exists from Chrome {_AUTO_CONNECT_MAJOR}. Update Chrome "
                f"to {_AUTO_CONNECT_MAJOR}+, then enable remote debugging via "
                f"{REMOTE_DEBUGGING_TOGGLE_URL} and approve the "
                f"connection prompt. Diego will NOT copy your profile or "
                f"create a temporary/clean one to work around this")
        return ChromeSessionUnavailable(reason=reason, remediation=remediation,
                                        info=info)

    @staticmethod
    def _is_default_dir(user_data_dir: str) -> bool:
        try:
            return Path(user_data_dir).resolve() in [
                Path(p).resolve()
                for p in default_user_data_dir_candidates()
                if Path(p).exists()]
        except Exception:
            return False


# Module-level singleton (BrowserController + the smoke script use this).
_session: Optional[CurrentChromeSession] = None


def chrome_session() -> CurrentChromeSession:
    global _session
    if _session is None:
        _session = CurrentChromeSession()
    return _session
