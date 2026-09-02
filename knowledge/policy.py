"""
Path security policy for the local knowledge indexer.

Responsibilities:
  * Allowlist of user-approved scan roots (configurable).
  * Denylist of sensitive/system locations that are NEVER scanned.
  * Sensitive filename/extension denylist (credentials, keys, tokens,
    sessions, .env, keystores, browser artifacts, ...) — applied by NAME
    ONLY, before any file is opened, stat'ed, hashed, or extracted.
  * Symlink safety: a path is only accepted if its REAL path stays inside
    an approved root and never enters a denylisted location.
  * Read-only guarantee: this module never opens, writes, or executes
    anything — it only inspects path metadata (lstat).

Design invariants:
  * Denylist always wins over allowlist.
  * Symlinks that resolve OUTSIDE every approved root are rejected.
  * /proc, /sys, /dev, /run and similar runtime paths are always denied.
  * Sensitive skips are logged by CATEGORY ONLY — never the filename.
  * No mutation of the filesystem, ever.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import stat as _stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

# Runtime/virtual paths that are always denied regardless of config.
_ALWAYS_DENIED = (
    "/proc", "/sys", "/dev", "/run", "/var/run",
    "/boot", "/efi",
)


def _split_csv(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def _expand(raw: str) -> Path:
    """Expand ~ and env vars, then normalize (no symlink resolution yet)."""
    p = os.path.expandvars(os.path.expanduser(raw.strip()))
    return Path(os.path.normpath(p))


# ── Sensitive filename material (checked by NAME ONLY) ────────────
# Extensions that are secret material, always denied.
SENSITIVE_EXTENSIONS = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".pfx", ".jks", ".kdbx", ".keystore",
    ".session", ".crt.pem", ".gpg", ".asc", ".sigstore",
})

# fnmatch patterns matched against the lowercased basename.
# Covers credentials/keys/tokens/sessions/auth encodings/cookies.
SENSITIVE_NAME_PATTERNS: Tuple[str, ...] = (
    "*accesskey*", "*access_key*", "*access-key*", "*accesskeys*",
    "*access-keys*", "*secret*", "*credential*", "*password*", "*passwd*",
    "*token*", "*_keys.*", "*keys.csv", "*key*.csv", "*apikey*",
    "*api_key*", "*api-key*", "*.session", "*.sessionstore*",
    "id_rsa*", "id_dsa*", "id_ed25519*", "id_ecdsa*",
    "authorized_keys*", "known_encodings*", "*cookies*", "*cookie*",
    "*keyring*", "*keystore*", "*.kdbx", ".env", ".env.*",
    "serviceaccountkey*", "service_account*", "*firebase*",
    "*authservice*", "*faceauth*", "*.pem", "*.p12", "*.pfx", "*.jks",
    "*private*", "*_private.*", "credentials*", "*creds*",
)

# Directory names that are never descended into (inside approved roots).
SENSITIVE_DIR_NAMES = frozenset({
    ".ssh", ".gnupg", ".gpg", ".password-store", ".aws", ".kube",
    ".docker", ".mozilla", ".thunderbird", ".config/google-chrome",
    ".config/chromium", "keyrings", "keychain", "keychains",
    "Cookies", "Login Data", ".env",
})

# Generated / non-useful directory names (inside approved roots).
GENERATED_DIR_NAMES = frozenset({
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    "caches", "logs", "log", "dist", "build", "coverage", ".idea",
    ".vscode-server", ".tox", ".eggs", "htmlcov", ".gradle",
})

# Generated / non-useful file extensions (metadata only, never indexed).
GENERATED_EXTENSIONS = frozenset({
    ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe", ".bin",
    ".whl", ".duckdb", ".duckdb.wal", ".sqlite", ".db", ".pkl",
    ".pickle", ".log.tmp",
})

# Categories for skip logging (never the filename itself).
CAT_SENSITIVE = "sensitive-material"
CAT_GENERATED = "generated-artifact"
CAT_SYSTEM = "system-path"
CAT_DenyPath = "denylisted-path"
CAT_DenyName = "denylisted-name"


def sensitive_name_reason(filename: str) -> Optional[str]:
    """Return a CATEGORY (not the filename) when the name is sensitive."""
    base = os.path.basename(filename)
    lower = base.lower()
    ext = os.path.splitext(lower)[1]
    if ext in SENSITIVE_EXTENSIONS:
        return "sensitive extension"
    for pat in SENSITIVE_NAME_PATTERNS:
        if fnmatch.fnmatch(lower, pat):
            return "sensitive filename pattern"
    return None


def generated_name_reason(filename: str) -> Optional[str]:
    """Return a CATEGORY when the name is generated/non-useful."""
    lower = os.path.basename(filename).lower()
    ext = os.path.splitext(lower)[1]
    if ext in GENERATED_EXTENSIONS:
        return "generated file extension"
    return None


@dataclass
class PathPolicy:
    """Allowlist + denylist path policy with symlink safety."""

    allow_roots: List[Path] = field(default_factory=list)
    deny_paths: Set[Path] = field(default_factory=set)
    deny_names: Set[str] = field(default_factory=set)
    max_file_size: int = 20 * 1024 * 1024
    # When True (default), sensitive/generated name rules are enforced.
    enforce_name_rules: bool = True

    @classmethod
    def from_settings(cls) -> "PathPolicy":
        roots = [_expand(r) for r in _split_csv(settings.KNOWLEDGE_SCAN_ROOTS)]
        # Diego project root is always approved (read-only).
        roots.append(settings.BASE_DIR)
        deny_raw = _split_csv(settings.KNOWLEDGE_DENYLIST)
        deny_paths: Set[Path] = set()
        deny_names: Set[str] = set()
        for d in deny_raw:
            if "/" in d:
                deny_paths.add(_expand(d))
            else:
                deny_names.add(d)
        return cls(
            allow_roots=roots,
            deny_paths=deny_paths,
            deny_names=deny_names,
            max_file_size=settings.KNOWLEDGE_MAX_FILE_SIZE,
        )

    # ── Checks ────────────────────────────────────────────────

    def _realpath_in_allow(self, real: Path) -> bool:
        for root in self.allow_roots:
            try:
                real.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _realpath_in_deny(self, real: Path) -> Optional[str]:
        """Return the deny reason if the real path hits the denylist."""
        # Absolute deny paths (e.g. /proc, ~/.ssh)
        for d in self.deny_paths:
            try:
                real.relative_to(d)
                return f"denylisted path: {d}"
            except ValueError:
                continue
        # Always-denied runtime paths
        for d in _ALWAYS_DENIED:
            try:
                real.relative_to(d)
                return f"system path: {d}"
            except ValueError:
                continue
        return None

    def _name_in_deny(self, real: Path) -> Optional[str]:
        """Component-name denylist + sensitive/generated name rules.

        Checked by NAME ONLY — no file is opened.
        """
        parts = real.parts
        # 1. Per-segment checks (directories AND the final component).
        #    Skipping the first segment ("/") is irrelevant — it never
        #    matches any rule.
        for part in parts:
            lower = part.lower()
            # Configured deny names (.git, .venv, node_modules, ...)
            if part in self.deny_names:
                return f"denylisted name ({CAT_DenyName})"
            # Sensitive directory/file segment names
            if lower in {d.lower() for d in SENSITIVE_DIR_NAMES}:
                return f"sensitive directory ({CAT_SENSITIVE})"
            # Generated / non-useful directory segments (project root
            # included: .git, .venv, __pycache__, logs, caches, ...)
            if lower in {d.lower() for d in GENERATED_DIR_NAMES}:
                return f"generated directory ({CAT_GENERATED})"
            # Keyring-style segment substrings (e.g. .../keyrings/...)
            if lower.startswith("keyring") or lower.startswith("keychain"):
                return f"sensitive directory ({CAT_SENSITIVE})"
        # 2. Compound sensitive segments (e.g. ".../google-chrome" under
        #    .config, "Login Data", "Network Cookies").
        plow = str(real).lower()
        for frag in ("/.ssh/", "/.gnupg/", "/.password-store/", "/.aws/",
                     "/.kube/", "/.docker/", "/.mozilla/", "/.thunderbird/",
                     "/google-chrome/", "/chromium/", "/keyring",
                     "/keychain", "/login data", "/cookies"):
            if frag in plow:
                return f"sensitive directory ({CAT_SENSITIVE})"
        if not self.enforce_name_rules:
            return None
        # 3. Sensitive / generated FILE names (final component only).
        reason = sensitive_name_reason(real.name)
        if reason:
            return reason
        return generated_name_reason(real.name)

    def check(self, path: Path) -> tuple:
        """Validate a path for READ-ONLY scanning.

        IMPORTANT: sensitive/generated name rules run FIRST (before any
        stat/open), so secret material is never even touched.

        Returns (allowed: bool, reason: str, real_path: Path).
        Never raises. Never mutates anything.
        """
        try:
            p = Path(os.path.normpath(os.path.abspath(str(path))))
            # lstat: do NOT follow the final symlink for existence check
            if not os.path.lexists(p):
                return False, "not found", p
            real = Path(os.path.realpath(p))
            # Symlink escaping all approved roots
            if real != p and not self._realpath_in_allow(real):
                return False, "symlink escapes approved roots", real
            if not self._realpath_in_allow(real):
                return False, "outside approved roots", real
            # Absolute denylist (system paths, ~/.ssh, ...)
            reason = self._realpath_in_deny(real)
            if reason:
                return False, reason, real
            # Name-based rules (sensitive + generated + deny names).
            # NOTE: checked against BOTH the symlink path and the real
            # path so a symlink cannot smuggle a secret name.
            for candidate in (p, real):
                reason = self._name_in_deny(candidate)
                if reason:
                    return False, reason, real
            # Size limit (metadata check only)
            try:
                st = os.lstat(real)
                if _stat.S_ISDIR(st.st_mode):
                    return True, "", real
                if st.st_size > self.max_file_size:
                    return False, "file too large", real
            except OSError:
                pass
            return True, "", real
        except Exception as e:  # never crash the scanner on a bad path
            return False, f"policy error: {e}", path

    def is_sensitive(self, path: Path) -> bool:
        """True when a path is denylisted/sensitive (for reporting)."""
        allowed, reason, _ = self.check(path)
        return (not allowed and reason != "not found"
                and reason != "file too large")

    def skipped_report(self) -> List[dict]:
        """Report of configured sensitive/system exclusions (no contents)."""
        items = []
        for d in sorted(self.deny_paths, key=str):
            items.append({"path": str(d), "kind": "denylisted path"})
        for n in sorted(self.deny_names):
            items.append({"path": n, "kind": "denylisted name"})
        for d in _ALWAYS_DENIED:
            items.append({"path": d, "kind": "system path"})
        items.append({"path": "<sensitive filename patterns>",
                      "kind": f"{len(SENSITIVE_NAME_PATTERNS)} patterns"})
        return items


def is_dir_stat(st: os.stat_result) -> bool:
    return _stat.S_ISDIR(st.st_mode)