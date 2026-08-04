"""
Autonomous Project Mode — Leo remembers project context across sessions.

When the user says "Work on GhostLine", Leo enters project mode and
automatically remembers:
  - Repository path
  - Terminal working directory
  - Dependencies (requirements.txt, package.json, go.mod, etc.)
  - Build/test commands
  - Recent files
  - Current branch
  - Previous TODO items
  - Last known state of the project

This context persists in DuckDB and is restored on restart.
The Brain and Planner automatically inject project context into goals.

Usage:
    from agent.project_mode import project_mode

    await project_mode.activate("GhostLine")
    ctx = project_mode.get_context()  # Returns dict for Brain injection
    await project_mode.record_file_edit("src/main.py")
    todo = project_mode.get_todos()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Project Context
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProjectContext:
    """Complete context for an active project."""
    name: str
    repo_path: str = ""
    repo_url: str = ""
    branch: str = ""
    terminal_cwd: str = ""
    ide: str = ""
    language: str = ""
    build_command: str = ""
    test_command: str = ""
    run_command: str = ""
    dependencies: List[str] = field(default_factory=list)
    recent_files: List[str] = field(default_factory=list)
    todos: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    last_active: float = field(default_factory=time.time)
    session_count: int = 0
    tags: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "repo_path": self.repo_path,
            "repo_url": self.repo_url,
            "branch": self.branch,
            "terminal_cwd": self.terminal_cwd,
            "ide": self.ide,
            "language": self.language,
            "build_command": self.build_command,
            "test_command": self.test_command,
            "run_command": self.run_command,
            "dependencies": self.dependencies,
            "recent_files": self.recent_files,
            "todos": self.todos,
            "notes": self.notes,
            "last_active": self.last_active,
            "session_count": self.session_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectContext":
        return cls(
            name=data.get("name", ""),
            repo_path=data.get("repo_path", ""),
            repo_url=data.get("repo_url", ""),
            branch=data.get("branch", ""),
            terminal_cwd=data.get("terminal_cwd", ""),
            ide=data.get("ide", ""),
            language=data.get("language", ""),
            build_command=data.get("build_command", ""),
            test_command=data.get("test_command", ""),
            run_command=data.get("run_command", ""),
            dependencies=data.get("dependencies", []),
            recent_files=data.get("recent_files", []),
            todos=data.get("todos", []),
            notes=data.get("notes", ""),
            last_active=data.get("last_active", time.time()),
            session_count=data.get("session_count", 0),
            tags=data.get("tags", ""),
        )


# ═══════════════════════════════════════════════════════════════
# ProjectMode
# ═══════════════════════════════════════════════════════════════

class ProjectMode:
    """
    Autonomous project context manager.

    Detects and remembers everything about the current project:
    repository, dependencies, build commands, recent files, TODOs, etc.

    Context survives reboots by persisting to DuckDB via GoalManager tables.
    The Brain and Planner automatically inject project context into goals.
    """

    def __init__(self):
        self._active_project: Optional[ProjectContext] = None
        self._projects: Dict[str, ProjectContext] = {}
        self._store = None
        self._initialized = False
        self._goal_manager = None

        # Auto-detection state
        self._detected_repo: str = ""
        self._detected_language: str = ""
        self._detected_build_cmd: str = ""
        self._detected_test_cmd: str = ""

    # ── Wiring ─────────────────────────────────────────────────

    def set_goal_manager(self, manager) -> None:
        """Wire the GoalManager for DuckDB persistence."""
        self._goal_manager = manager

    async def initialize(self) -> bool:
        """Initialize project mode — load all known projects from DB."""
        try:
            from memory.duckdb_store import store
            self._store = store

            conn = self._store._get_conn()
            if conn is None:
                logger.warning("[ProjectMode] DuckDB unavailable")
                return False

            # Create project_context table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_context (
                    name VARCHAR PRIMARY KEY,
                    data_json VARCHAR NOT NULL,
                    updated_at DOUBLE NOT NULL,
                    created_at DOUBLE NOT NULL DEFAULT 0
                )
            """)

            # Load all projects
            rows = conn.execute(
                "SELECT name, data_json, updated_at FROM project_context ORDER BY updated_at DESC"
            ).fetchall()

            for row in rows:
                name, data_json, updated_at = row
                try:
                    data = json.loads(data_json)
                    ctx = ProjectContext.from_dict(data)
                    ctx.last_active = updated_at
                    self._projects[name] = ctx
                except Exception as e:
                    logger.debug("[ProjectMode] Failed to load project '%s': %s", name, e)

            self._initialized = True
            logger.info("[ProjectMode] Loaded %d project(s): %s",
                         len(self._projects),
                         ", ".join(list(self._projects.keys())[:10]))
            return True

        except Exception as e:
            logger.warning("[ProjectMode] Initialization failed: %s", e)
            return False

    # ── Activation ─────────────────────────────────────────────

    async def activate(self, project_name: str, path: Optional[str] = None) -> ProjectContext:
        """
        Activate a project. If it exists, restore context.
        If new, auto-detect everything about it.

        Args:
            project_name: "GhostLine", "leo_desktop_assistant", etc.
            path: Optional explicit path to the repository.

        Returns:
            The ProjectContext (existing or newly detected).
        """
        if not self._initialized:
            await self.initialize()

        # Check if known
        if project_name in self._projects:
            ctx = self._projects[project_name]
            ctx.session_count += 1
            ctx.last_active = time.time()
            self._active_project = ctx
            logger.info("[ProjectMode] Activated known project '%s' (session #%d)",
                         project_name, ctx.session_count)
            await self._persist(ctx)
            return ctx

        # New project — auto-detect context
        logger.info("[ProjectMode] Auto-detecting new project '%s'...", project_name)

        ctx = ProjectContext(name=project_name)
        repo_path = path or await self._find_repo(project_name)
        ctx.repo_path = repo_path

        if repo_path:
            await self._detect_context(ctx)

        ctx.session_count = 1
        ctx.last_active = time.time()

        self._projects[project_name] = ctx
        self._active_project = ctx
        await self._persist(ctx)

        logger.info("[ProjectMode] Project '%s' activated: repo=%s, lang=%s, branch=%s",
                     project_name, ctx.repo_path, ctx.language, ctx.branch)
        return ctx

    async def deactivate(self) -> None:
        """Deactivate the current project."""
        if self._active_project:
            await self._persist(self._active_project)
            logger.info("[ProjectMode] Deactivated project '%s'", self._active_project.name)
        self._active_project = None

    # ── Auto-Detection ─────────────────────────────────────────

    async def _find_repo(self, project_name: str) -> str:
        """Find the repository directory for a project."""
        loop = __import__('asyncio').get_event_loop()

        # Check common locations
        common_dirs = [
            Path.home() / "PycharmProjects" / project_name,
            Path.home() / "projects" / project_name,
            Path.home() / "Development" / project_name,
            Path.home() / "dev" / project_name,
            Path.home() / "src" / project_name,
            Path.home() / "code" / project_name,
            Path.home() / project_name,
        ]

        for d in common_dirs:
            if d.exists():
                return str(d)

        # Try to find via focused window CWD
        try:
            info = await loop.run_in_executor(None, self._get_focused_window_info)
            if info and info.get("pid", 0) > 0:
                try:
                    cwd = os.readlink(f"/proc/{info['pid']}/cwd")
                    if cwd and project_name.lower() in cwd.lower():
                        return cwd
                except Exception:
                    pass
        except Exception:
            pass

        return ""

    async def _detect_context(self, ctx: ProjectContext) -> None:
        """Auto-detect project context from the repository."""
        loop = __import__('asyncio').get_event_loop()

        if not ctx.repo_path:
            return

        repo = Path(ctx.repo_path)

        # Detect language
        if (repo / "requirements.txt").exists() or (repo / "setup.py").exists() or list(repo.glob("*.py")):
            ctx.language = "Python"
        elif (repo / "go.mod").exists():
            ctx.language = "Go"
        elif (repo / "package.json").exists():
            ctx.language = "JavaScript/TypeScript"
        elif (repo / "Cargo.toml").exists():
            ctx.language = "Rust"
        elif (repo / "pom.xml").exists() or (repo / "build.gradle").exists() or (repo / "build.gradle.kts").exists():
            ctx.language = "Java"
        elif (repo / "CMakeLists.txt").exists():
            ctx.language = "C++"
        elif (repo / "Cargo.toml").exists():
            ctx.language = "Rust"
        elif (repo / "Makefile").exists():
            ctx.language = "C/C++"

        # Detect git branch
        try:
            result = await loop.run_in_executor(None, lambda: subprocess.run(
                ["git", "-C", str(repo), "branch", "--show-current"],
                capture_output=True, text=True, timeout=5
            ))
            if result.returncode == 0:
                ctx.branch = result.stdout.strip()
        except Exception:
            pass

        # Detect git remote URL
        try:
            result = await loop.run_in_executor(None, lambda: subprocess.run(
                ["git", "-C", str(repo), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5
            ))
            if result.returncode == 0:
                ctx.repo_url = result.stdout.strip()
        except Exception:
            pass

        # Detect build/test commands
        if ctx.language == "Python":
            if (repo / "Makefile").exists():
                ctx.build_command = "make"
                ctx.test_command = "make test"
            elif (repo / "pyproject.toml").exists():
                ctx.build_command = "pip install -e ."
                ctx.test_command = "pytest"
            else:
                ctx.test_command = "pytest"
                ctx.run_command = f"python {ctx.name.lower().replace(' ', '_')}"
        elif ctx.language == "Go":
            ctx.build_command = "go build ./..."
            ctx.test_command = "go test ./..."
            ctx.run_command = "go run ."
        elif ctx.language == "JavaScript/TypeScript":
            if (repo / "package.json").exists():
                try:
                    pkg = json.loads((repo / "package.json").read_text())
                    scripts = pkg.get("scripts", {})
                    ctx.build_command = "npm run build" if "build" in scripts else ""
                    ctx.test_command = "npm test" if "test" in scripts else ""
                    ctx.run_command = "npm start" if "start" in scripts else "npm run dev" if "dev" in scripts else ""
                except Exception:
                    pass
        elif ctx.language == "Rust":
            ctx.build_command = "cargo build"
            ctx.test_command = "cargo test"
            ctx.run_command = "cargo run"
        elif ctx.language == "Java":
            ctx.build_command = "mvn package" if (repo / "pom.xml").exists() else "gradle build"
            ctx.test_command = "mvn test" if (repo / "pom.xml").exists() else "gradle test"
        elif ctx.language == "C++":
            ctx.build_command = "cmake --build build"
            ctx.test_command = "ctest --test-dir build"

        # Detect dependencies
        ctx.dependencies = await loop.run_in_executor(None, lambda: self._detect_dependencies(repo))

        # Detect recent files
        ctx.recent_files = await loop.run_in_executor(None, lambda: self._detect_recent_files(repo))

        # Detect TODOs
        ctx.todos = await loop.run_in_executor(None, lambda: self._scan_todos(repo))

        # Detect IDE
        ctx.ide = await loop.run_in_executor(None, self._detect_ide)

    @staticmethod
    def _get_focused_window_info() -> Dict[str, Any]:
        """Get the currently focused window info (runs in executor)."""
        import shutil
        info = {"title": "", "application": "", "pid": 0}
        if shutil.which("xdotool"):
            try:
                wid = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True, text=True, timeout=1
                ).stdout.strip()
                if wid:
                    pid_str = subprocess.run(
                        ["xdotool", "getwindowpid", wid],
                        capture_output=True, text=True, timeout=1
                    ).stdout.strip()
                    if pid_str:
                        info["pid"] = int(pid_str)
            except Exception:
                pass
        return info

    @staticmethod
    def _detect_dependencies(repo: Path) -> List[str]:
        """Detect project dependencies (runs in executor)."""
        deps: List[str] = []

        if (repo / "requirements.txt").exists():
            try:
                with open(repo / "requirements.txt") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                            deps.append(pkg)
            except Exception:
                pass
        elif (repo / "pyproject.toml").exists():
            deps.append("pyproject.toml (see file)")

        if (repo / "package.json").exists():
            try:
                pkg = json.loads((repo / "package.json").read_text())
                for dep_type in ("dependencies", "devDependencies"):
                    for dep_name in pkg.get(dep_type, {}):
                        deps.append(f"npm:{dep_name}")
            except Exception:
                pass

        if (repo / "go.mod").exists():
            deps.append("go.mod (see file)")

        if (repo / "Cargo.toml").exists():
            deps.append("Cargo.toml (see file)")

        return deps[:20]

    @staticmethod
    def _detect_recent_files(repo: Path, max_files: int = 10) -> List[str]:
        """Detect recently modified files (runs in executor)."""
        recent = []
        try:
            src_exts = {".py", ".go", ".java", ".js", ".ts", ".rs", ".cpp", ".h", ".c", ".rb", ".php"}
            for root, dirs, filenames in os.walk(repo):
                # Skip hidden and common ignore dirs
                dirs[:] = [d for d in dirs if not d.startswith(".")
                           and d not in ("node_modules", "target", "build", "dist", "__pycache__", ".git")]
                for filename in filenames:
                    if any(filename.endswith(ext) for ext in src_exts):
                        filepath = os.path.join(root, filename)
                        try:
                            mtime = os.path.getmtime(filepath)
                            recent.append((mtime, str(Path(filepath).relative_to(repo))))
                        except OSError:
                            pass
        except Exception:
            pass

        recent.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in recent[:max_files]]

    @staticmethod
    def _scan_todos(repo: Path) -> List[Dict[str, Any]]:
        """Scan source files for TODO/FIXME/HACK comments (runs in executor)."""
        todos = []
        import re
        todo_pattern = re.compile(r"(?:TODO|FIXME|HACK|XXX|NOTE):\s*(.+)", re.IGNORECASE)

        src_exts = {".py", ".go", ".java", ".js", ".ts", ".rs", ".cpp", ".h", ".c"}
        for root, dirs, filenames in os.walk(repo):
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in ("node_modules", "target", "build", "dist", "__pycache__", ".git")]
            for filename in filenames:
                if any(filename.endswith(ext) for ext in src_exts):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, "r", errors="ignore") as f:
                            for lineno, line in enumerate(f, 1):
                                match = todo_pattern.search(line)
                                if match:
                                    todos.append({
                                        "file": str(Path(filepath).relative_to(repo)),
                                        "line": lineno,
                                        "text": match.group(1).strip(),
                                    })
                    except Exception:
                        pass
        return todos[:50]

    @staticmethod
    def _detect_ide() -> str:
        """Detect the current IDE (runs in executor)."""
        import shutil
        if shutil.which("xdotool"):
            try:
                wid = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True, text=True, timeout=1
                ).stdout.strip()
                if wid:
                    title = subprocess.run(
                        ["xdotool", "getwindowname", wid],
                        capture_output=True, text=True, timeout=1
                    ).stdout.strip()
                    title_lower = title.lower()
                    if "pycharm" in title_lower:
                        return "PyCharm"
                    elif "visual studio code" in title_lower:
                        return "VS Code"
                    elif "vim" in title_lower:
                        return "Vim"
                    elif "neovim" in title_lower:
                        return "Neovim"
                    elif "sublime" in title_lower:
                        return "Sublime Text"
                    elif "intellij" in title_lower:
                        return "IntelliJ"
                    elif "rider" in title_lower:
                        return "Rider"
                    elif "android studio" in title_lower:
                        return "Android Studio"
            except Exception:
                pass
        return ""

    # ── Context Injection ──────────────────────────────────────

    def get_context(self) -> Dict[str, Any]:
        """Return context dict for Brain/Planner injection."""
        if not self._active_project:
            # Try to auto-detect from desktop
            return self._auto_context()

        ctx = self._active_project
        return {
            "project_name": ctx.name,
            "repo": ctx.repo_path,
            "branch": ctx.branch,
            "language": ctx.language,
            "ide": ctx.ide,
            "build_command": ctx.build_command,
            "test_command": ctx.test_command,
            "run_command": ctx.run_command,
            "recent_files": ctx.recent_files,
            "terminal_cwd": ctx.terminal_cwd,
        }

    def _auto_context(self) -> Dict[str, Any]:
        """Auto-generate context from current desktop state."""
        context: Dict[str, Any] = {}
        try:
            from services.desktop_state import desktop_state
            snap = desktop_state.snapshot()
            context["ide"] = snap.focused_window.application
            context["branch"] = snap.terminal.git_branch
            context["terminal_cwd"] = snap.terminal.cwd
        except Exception:
            pass
        return context

    def context_for_llm(self) -> str:
        """Return a compact text block for LLM prompt injection."""
        ctx = self.get_context()
        if not ctx or not ctx.get("project_name"):
            return ""

        parts = [f"Project: {ctx.get('project_name', '')}"]
        if ctx.get("language"):
            parts.append(f"Language: {ctx['language']}")
        if ctx.get("branch"):
            parts.append(f"Branch: {ctx['branch']}")
        if ctx.get("build_command"):
            parts.append(f"Build: {ctx['build_command']}")
        if ctx.get("test_command"):
            parts.append(f"Test: {ctx['test_command']}")
        if ctx.get("recent_files"):
            parts.append(f"Recent files: {', '.join(ctx['recent_files'][:5])}")

        return " | ".join(parts)

    # ── Record Activity ────────────────────────────────────────

    async def record_file_edit(self, filepath: str) -> None:
        """Record that a file was edited."""
        if not self._active_project:
            return
        # Move to front of recent_files
        relpath = filepath
        repo = Path(self._active_project.repo_path)
        if repo.exists() and filepath.startswith(str(repo)):
            relpath = str(Path(filepath).relative_to(repo))
        if relpath in self._active_project.recent_files:
            self._active_project.recent_files.remove(relpath)
        self._active_project.recent_files.insert(0, relpath)
        self._active_project.recent_files = self._active_project.recent_files[:20]
        await self._persist(self._active_project)

    async def record_build(self, success: bool, output: str = "") -> None:
        """Record a build attempt."""
        if not self._active_project:
            return
        self._active_project.last_active = time.time()
        await self._persist(self._active_project)

    async def record_test_run(self, passed: int, failed: int) -> None:
        """Record a test run."""
        if not self._active_project:
            return
        self._active_project.last_active = time.time()
        await self._persist(self._active_project)

    async def add_todo(self, text: str, filepath: str = "", line: int = 0) -> None:
        """Add a TODO item."""
        if not self._active_project:
            return
        self._active_project.todos.append({
            "file": filepath,
            "line": line,
            "text": text,
            "added_at": time.time(),
        })
        await self._persist(self._active_project)

    async def add_note(self, note: str) -> None:
        """Add a note to the project."""
        if not self._active_project:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        self._active_project.notes += f"\n[{timestamp}] {note}"
        await self._persist(self._active_project)

    # ── Queries ────────────────────────────────────────────────

    def get_todos(self) -> List[Dict[str, Any]]:
        """Return all TODO items for the active project."""
        if not self._active_project:
            return []
        return list(self._active_project.todos)

    def get_recent_files(self, limit: int = 10) -> List[str]:
        """Return recently edited files."""
        if not self._active_project:
            return []
        return self._active_project.recent_files[:limit]

    def list_projects(self) -> List[str]:
        """List all known project names."""
        return sorted(self._projects.keys())

    def is_active(self) -> bool:
        """Check if a project is currently active."""
        return self._active_project is not None

    @property
    def active_project(self) -> Optional[ProjectContext]:
        return self._active_project

    @property
    def active_project_name(self) -> str:
        return self._active_project.name if self._active_project else ""

    # ── Persistence ────────────────────────────────────────────

    async def _persist(self, ctx: ProjectContext) -> None:
        """Persist a project context to DuckDB."""
        if not self._store:
            return

        try:
            conn = self._store._get_conn()
            if conn is None:
                return

            conn.execute("""
                INSERT INTO project_context (name, data_json, updated_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (name) DO UPDATE SET
                    data_json = EXCLUDED.data_json,
                    updated_at = EXCLUDED.updated_at
            """, [
                ctx.name,
                json.dumps(ctx.to_dict()),
                ctx.last_active,
                ctx.last_active if ctx.session_count == 1 else None,
            ])

        except Exception as e:
            logger.debug("[ProjectMode] Persist failed: %s", e)

    async def delete_project(self, name: str) -> bool:
        """Remove a project from the database."""
        if not self._store:
            return False

        try:
            conn = self._store._get_conn()
            if conn is None:
                return False

            conn.execute("DELETE FROM project_context WHERE name = ?", [name])
            self._projects.pop(name, None)
            if self._active_project and self._active_project.name == name:
                self._active_project = None
            logger.info("[ProjectMode] Deleted project '%s'", name)
            return True

        except Exception as e:
            logger.error("[ProjectMode] Delete failed: %s", e)
            return False

    def close(self) -> None:
        logger.info("[ProjectMode] ProjectMode shut down")


# Global singleton
project_mode = ProjectMode()