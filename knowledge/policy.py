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
    ".pem", ".key", ".p12", ".pfx", ".jks", ".kdbx", ".keystore",
    ".session", ".crt.pem", ".gpg", ".asc", ".sigstore",
    ".p8", ".jceks", ".bks",
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
    # Auth / session / token material (name-only, never opened).
    # Precise patterns — "*auth*" would falsely match "author_notes.txt"
    # and "*session*" would falsely match "session_notes.md".
    "auth", "auth.*", "auth_*", "auth-*", "*_auth.*", "*-auth.*",
    "*.auth", "*_auth_*", "*-auth-*", "*_auth", "*-auth",
    "*.session*", "session.*", "*session.*", "*jwt*", "*bearer*",
    # Backups/derivatives of secret material (multi-dot names).
    "*.pem.*", "*.key.*", "*.p12.*", "*.pfx.*", "*.jks.*",
    "*.kdbx.*", "*.keystore.*", "*.gpg.*", "*.asc.*", "*.session.*",
    "*.p8.*", "*.jceks.*", "*.bks.*",
)

# Directory names that are never descended into (inside approved roots).
SENSITIVE_DIR_NAMES = frozenset({
    ".ssh", ".gnupg", ".gpg", ".password-store", ".aws", ".kube",
    ".docker", ".mozilla", ".thunderbird", ".config/google-chrome",
    ".config/chromium", "keyrings", "keychain", "keychains",
    "Cookies", "Login Data", ".env",
    # Additional sensitive directory components (name-only, no open).
    "secrets", "credentials", "private", "auth", "session", "sessions",
    "tokens", "keys", "passwords", "secret", "credential", "token",
    "password", "keyring", "keystore",
})

# Generated / non-useful directory names (inside approved roots).
GENERATED_DIR_NAMES = frozenset({
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    "caches", "cache", "logs", "log", "dist", "build", "coverage",
    ".idea", ".vscode-server", ".tox", ".eggs", "htmlcov", ".gradle",
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
    # Multi-dot sensitive extensions (e.g. ".crt.pem") are NOT captured
    # by os.path.splitext (it splits on the LAST dot only). Check the
    # full basename so secret material is never missed.
    for s_ext in SENSITIVE_EXTENSIONS:
        if s_ext.count(".") > 1 and lower.endswith(s_ext):
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
    # Multi-part extensions (e.g. ".duckdb.wal", ".log.tmp") are NOT
    # captured by os.path.splitext (it splits on the LAST dot only).
    # Check the full basename against every multi-dot extension so
    # database WAL/tmp files never reach persistence.
    for gen_ext in GENERATED_EXTENSIONS:
        if gen_ext.count(".") > 1 and lower.endswith(gen_ext):
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
            # Sensitive directory/file segment names FIRST so sensitive
            # material is always reported as "sensitive" (and therefore
            # sanitized in logs) even when it also appears in the
            # configured deny_names (e.g. "credentials").
            if lower in {d.lower() for d in SENSITIVE_DIR_NAMES}:
                return f"sensitive directory ({CAT_SENSITIVE})"
            # Configured deny names (.git, .venv, node_modules, ...)
            if part in self.deny_names:
                return f"denylisted name ({CAT_DenyName})"
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
            # 1. NAME-BASED RULES RUN FIRST — before ANY filesystem
            #    resolution (lexists/realpath/lstat). A denied name is
            #    rejected without the file ever being touched, so secret
            #    material is never opened, hashed, extracted, embedded,
            #    or persisted. Denylist always wins over allowlist.
            reason = self._name_in_deny(p)
            if reason:
                return False, reason, p
            # 2. lstat: do NOT follow the final symlink for existence
            #    check (metadata only — never opens file contents).
            if not os.path.lexists(p):
                return False, "not found", p
            real = Path(os.path.realpath(p))
            # 3. Symlink escaping all approved roots
            if real != p and not self._realpath_in_allow(real):
                return False, "symlink escapes approved roots", real
            if not self._realpath_in_allow(real):
                return False, "outside approved roots", real
            # 4. Absolute denylist (system paths, ~/.ssh, ...)
            reason = self._realpath_in_deny(real)
            if reason:
                return False, reason, real
            # 5. Name-based rules re-checked against the REAL path so a
            #    symlink cannot smuggle a secret name past step 1.
            reason = self._name_in_deny(real)
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

    def name_deny_reason(self, path: Path) -> Optional[str]:
        """Name-only deny check — NO filesystem access at all.

        Returns the deny reason when the path's NAME is sensitive,
        generated, or denylisted. Used for sanitized logging and for
        purging previously-indexed material without ever touching the
        file (a deleted secret must not be re-opened to be classified).
        """
        p = Path(os.path.normpath(os.path.abspath(str(path))))
        return self._name_in_deny(p)

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