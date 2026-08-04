"""
CodeAssistant — Multi-language code understanding for Leo.

Leo understands code visible on screen or in clipboard content:
  - Errors and warnings (compiler output, linters, type checkers)
  - Tracebacks (Python, Java, Go, Rust, JavaScript, C++)
  - Test output (pytest, go test, cargo test, jest, etc.)
  - Git diffs (what changed, what was added/removed)
  - Coverage reports
  - Import/module errors
  - Syntax errors

The CodeAssistant extracts structured information from raw output
and generates human-readable explanations.

Usage:
    from agent.code_assistant import code_assistant

    explanation = code_assistant.explain_error(traceback_text)
    lang = code_assistant.detect_language(code_text)
    summary = code_assistant.summarize_diff(diff_text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Error extraction types
# ═══════════════════════════════════════════════════════════════

@dataclass
class CodeError:
    """A structured code error extracted from output."""
    language: str
    error_type: str  # syntax, runtime, import, type, linker, test_fail
    file: str = ""
    line: int = 0
    column: int = 0
    message: str = ""
    context: List[str] = field(default_factory=list)  # Surrounding lines
    suggestion: str = ""


@dataclass
class TestResult:
    """Structured test run results."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_s: float = 0.0
    failures: List[CodeError] = field(default_factory=list)


@dataclass
class DiffSummary:
    """Summary of a git diff."""
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    new_files: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    hunks_by_file: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Language detection
# ═══════════════════════════════════════════════════════════════

LANGUAGE_SIGNATURES = {
    "Python": [
        (re.compile(r"def\s+\w+\s*\(.*\)\s*:"), 10),
        (re.compile(r"import\s+\w+"), 8),
        (re.compile(r"from\s+\w+\s+import"), 9),
        (re.compile(r"class\s+\w+.*:"), 8),
        (re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]"), 10),
        (re.compile(r"Traceback\s+\(most\s+recent\s+call\s+last\)"), 10),
        (re.compile(r"File\s+\".*\",\s+line\s+\d+"), 10),
        (re.compile(r"ModuleNotFoundError|ImportError|AttributeError|TypeError|ValueError"), 9),
    ],
    "Go": [
        (re.compile(r"func\s+\w+\(.*\)\s*\w*\{"), 9),
        (re.compile(r"import\s+\(?.*\"\w+\""), 8),
        (re.compile(r"package\s+\w+"), 10),
        (re.compile(r"var\s+\w+\s+\w+"), 7),
        (re.compile(r"\.go:\d+"), 9),
        (re.compile(r"go\s+run|go\s+build|go\s+test"), 8),
    ],
    "Java": [
        (re.compile(r"public\s+(static\s+)?(void|class|int|String)"), 10),
        (re.compile(r"import\s+java\.\w+"), 10),
        (re.compile(r"throws\s+\w+Exception"), 9),
        (re.compile(r"Exception\s+in\s+thread"), 10),
        (re.compile(r"at\s+[\w.$]+\([\w.]+:\d+\)"), 10),
    ],
    "JavaScript": [
        (re.compile(r"function\s+\w+\s*\("), 8),
        (re.compile(r"const\s+\w+\s*="), 8),
        (re.compile(r"let\s+\w+\s*="), 8),
        (re.compile(r"import\s+\{.*\}\s+from"), 10),
        (re.compile(r"export\s+(default\s+)?(function|class|const)"), 10),
        (re.compile(r"Uncaught\s+(TypeError|ReferenceError|SyntaxError)"), 9),
        (re.compile(r"at\s+.+\s+\(\S+:\d+:\d+\)"), 9),
        (re.compile(r"TypeError:\s+.+"), 8),
    ],
    "TypeScript": [
        (re.compile(r"interface\s+\w+\s*\{" ), 10),
        (re.compile(r"type\s+\w+\s*="), 9),
        (re.compile(r"as\s+\w+"), 7),
        (re.compile(r":\s*(string|number|boolean|void)\b"), 8),
    ],
    "Rust": [
        (re.compile(r"fn\s+\w+\s*\("), 10),
        (re.compile(r"let\s+mut\s+\w+"), 10),
        (re.compile(r"impl\s+\w+\s+"), 9),
        (re.compile(r"use\s+\w+::"), 10),
        (re.compile(r"error\[E\d+\]"), 10),
        (re.compile(r"-->\s+\S+:\d+:\d+"), 10),
    ],
    "C++": [
        (re.compile(r"#include\s+<\w+>"), 10),
        (re.compile(r"int\s+main\s*\("), 9),
        (re.compile(r"std::\w+"), 9),
        (re.compile(r"template\s*<"), 10),
        (re.compile(r"error:\s+.+\.cpp"), 8),
    ],
}


# ═══════════════════════════════════════════════════════════════
# CodeAssistant
# ═══════════════════════════════════════════════════════════════

class CodeAssistant:
    """
    Multi-language code understanding engine.

    Extracts errors, parses test results, summarizes diffs,
    and detects programming languages from raw text output.
    """

    def __init__(self):
        self._supported_languages = list(LANGUAGE_SIGNATURES.keys())

    # ── Language Detection ─────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """
        Detect the programming language of a code snippet or error output.

        Returns the highest-scoring language name, or "unknown".
        """
        scores: Dict[str, int] = {lang: 0 for lang in LANGUAGE_SIGNATURES}

        for lang, patterns in LANGUAGE_SIGNATURES.items():
            for pattern, weight in patterns:
                if pattern.search(text):
                    scores[lang] += weight

        best = max(scores, key=scores.get)
        if scores[best] >= 10:
            return best
        return "unknown"

    # ── Error Extraction ───────────────────────────────────────

    def extract_errors(self, text: str) -> List[CodeError]:
        """Extract all code errors from raw output text."""
        lang = self.detect_language(text)
        errors: List[CodeError] = []

        if lang == "Python":
            errors = self._extract_python_errors(text)
        elif lang == "Go":
            errors = self._extract_go_errors(text)
        elif lang == "Rust":
            errors = self._extract_rust_errors(text)
        elif lang == "Java":
            errors = self._extract_java_errors(text)
        elif lang in ("JavaScript", "TypeScript"):
            errors = self._extract_js_errors(text)
        elif lang == "C++":
            errors = self._extract_cpp_errors(text)
        else:
            errors = self._extract_generic_errors(text, lang)

        return errors

    def explain_error(self, error_text: str) -> str:
        """
        Generate a human-readable explanation of a code error.

        Args:
            error_text: The raw error/traceback text.

        Returns:
            A natural language explanation.
        """
        errors = self.extract_errors(error_text)
        if not errors:
            return "I couldn't identify a specific error in that output."

        parts = []
        for err in errors:
            location = ""
            if err.file:
                location = f" in **{err.file}**"
                if err.line > 0:
                    location += f" (line {err.line})"

            parts.append(f"- **{err.error_type}**{location}: {err.message}")
            if err.suggestion:
                parts.append(f"  → {err.suggestion}")

        return "\n".join(parts)

    # ── Python Error Extraction ────────────────────────────────

    def _extract_python_errors(self, text: str) -> List[CodeError]:
        """Extract errors from Python traceback / output."""
        errors: List[CodeError] = []

        # Traceback format: File "path", line N, in func
        file_pattern = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')
        error_pattern = re.compile(
            r'(ModuleNotFoundError|ImportError|AttributeError|TypeError'
            r'|ValueError|KeyError|IndexError|SyntaxError|NameError'
            r'|RuntimeError|FileNotFoundError|PermissionError|AssertionError)'
            r':\s*(.+)'
        )
        line_pattern = re.compile(r'^\s*(.+)$', re.MULTILINE)

        # Find error blocks
        blocks = text.split("\n\n")
        for block in blocks:
            file_match = file_pattern.search(block)
            err_match = error_pattern.search(block)

            if err_match:
                err = CodeError(
                    language="Python",
                    error_type=self._classify_error(err_match.group(1)),
                    message=err_match.group(2).strip(),
                )
                if file_match:
                    err.file = file_match.group(1)
                    err.line = int(file_match.group(2))

                # Extract context lines
                context_lines = []
                for m in line_pattern.finditer(block):
                    line = m.group(1).strip()
                    if line and not line.startswith("File "):
                        context_lines.append(line)
                err.context = context_lines[:5]

                err.suggestion = self._suggest_fix(err)
                errors.append(err)

        return errors

    # ── Go Error Extraction ────────────────────────────────────

    def _extract_go_errors(self, text: str) -> List[CodeError]:
        """Extract errors from Go compiler output."""
        errors: List[CodeError] = []

        go_pattern = re.compile(r'(\.go):(\d+)(?::(\d+))?:\s+(.+)')

        for match in go_pattern.finditer(text):
            err = CodeError(
                language="Go",
                error_type="compilation",
                file=match.group(0).split(":")[0] if match.lastindex else "",
                line=int(match.group(2)) if match.lastindex and match.group(2) else 0,
                message=match.group(4).strip() if match.lastindex and match.group(4) else "",
            )
            err.suggestion = self._suggest_fix(err)
            errors.append(err)

        return errors

    # ── Rust Error Extraction ──────────────────────────────────

    def _extract_rust_errors(self, text: str) -> List[CodeError]:
        """Extract errors from Rust compiler output."""
        errors: List[CodeError] = []

        rust_pattern = re.compile(r'error\[(E\d+)\]:\s+(.+)')
        location_pattern = re.compile(r'-->\s+(\S+):(\d+):(\d+)')

        blocks = text.split("error[")
        for block in blocks[1:]:
            err_match = re.match(r'(E\d+)\]:\s+(.+)', block)
            loc_match = location_pattern.search(block)

            if err_match:
                err = CodeError(
                    language="Rust",
                    error_type="compilation",
                    message=err_match.group(2).strip() if err_match.lastindex else "",
                )
                if loc_match:
                    err.file = loc_match.group(1)
                    err.line = int(loc_match.group(2)) if loc_match.group(2) else 0
                err.suggestion = self._suggest_fix(err)
                errors.append(err)

        return errors

    # ── JavaScript/TypeScript Error Extraction ─────────────────

    def _extract_js_errors(self, text: str) -> List[CodeError]:
        """Extract errors from JS/TS output."""
        errors: List[CodeError] = []

        js_pattern = re.compile(
            r'(TypeError|ReferenceError|SyntaxError|RangeError|URIError|EvalError)'
            r':\s*(.+)'
        )
        loc_pattern = re.compile(r'at\s+.+\s+\((\S+):(\d+):(\d+)\)')

        for match in js_pattern.finditer(text):
            err = CodeError(
                language="JavaScript",
                error_type=self._classify_error(match.group(1)),
                message=match.group(2).strip() if match.lastindex else "",
            )

            loc = loc_pattern.search(text)
            if loc:
                err.file = loc.group(1)
                err.line = int(loc.group(2)) if loc.group(2) else 0

            err.suggestion = self._suggest_fix(err)
            errors.append(err)

        return errors

    # ── Java Error Extraction ──────────────────────────────────

    def _extract_java_errors(self, text: str) -> List[CodeError]:
        """Extract errors from Java output."""
        errors: List[CodeError] = []

        java_pattern = re.compile(
            r'(Exception|Error):\s+(.+)'
        )
        stack_pattern = re.compile(r'at\s+([\w.$]+)\(([\w.]+):(\d+)\)')

        for match in java_pattern.finditer(text):
            err = CodeError(
                language="Java",
                error_type="runtime",
                message=match.group(2).strip() if match.lastindex and match.group(2) else "",
            )

            stack = stack_pattern.search(text)
            if stack:
                err.file = stack.group(2) if stack.lastindex and stack.group(2) else ""
                err.line = int(stack.group(3)) if stack.lastindex and stack.group(3) else 0

            err.suggestion = self._suggest_fix(err)
            errors.append(err)

        return errors

    # ── C++ Error Extraction ───────────────────────────────────

    def _extract_cpp_errors(self, text: str) -> List[CodeError]:
        """Extract C++ compilation errors."""
        errors: List[CodeError] = []

        cpp_pattern = re.compile(r'([\w./]+\.(?:cpp|h|hpp|c|cc)):(\d+):(\d+)?:\s*error:\s*(.+)')

        for match in cpp_pattern.finditer(text):
            err = CodeError(
                language="C++",
                error_type="compilation",
                file=match.group(1) if match.lastindex and match.group(1) else "",
                line=int(match.group(2)) if match.lastindex and match.group(2) else 0,
                message=match.group(4).strip() if match.lastindex and match.group(4) else "",
            )
            err.suggestion = self._suggest_fix(err)
            errors.append(err)

        return errors

    # ── Generic Error Extraction ───────────────────────────────

    def _extract_generic_errors(self, text: str, lang: str) -> List[CodeError]:
        """Fallback error extraction for unrecognized languages."""
        errors: List[CodeError] = []

        # Look for common error patterns
        generic_patterns = [
            (re.compile(r'error[:\s]+(.+)', re.IGNORECASE), "error"),
            (re.compile(r'fail(?:ed|ure)?[:\s]+(.+)', re.IGNORECASE), "failure"),
            (re.compile(r'panic[:\s]+(.+)', re.IGNORECASE), "panic"),
            (re.compile(r'fatal[:\s]+(.+)', re.IGNORECASE), "fatal"),
        ]

        for pattern, err_type in generic_patterns:
            for match in pattern.finditer(text):
                err = CodeError(
                    language=lang,
                    error_type=err_type,
                    message=match.group(1).strip() if match.lastindex else match.group(0),
                )
                errors.append(err)

        return errors

    # ── Test Result Parsing ────────────────────────────────────

    def parse_test_results(self, text: str) -> TestResult:
        """Parse test output from various frameworks."""
        result = TestResult()

        # pytest: "X passed, Y failed, Z skipped"
        pytest_pattern = re.compile(
            r'(\d+)\s*passed.*?(\d+)\s*failed.*?(\d+)\s*skipped'
        )
        pytest_dur = re.compile(r'in\s+(\d+\.?\d*)s')

        if pytest_pattern.search(text):
            m = pytest_pattern.search(text)
            if m and m.lastindex >= 3:
                result.passed = int(m.group(1))
                result.failed = int(m.group(2))
                result.skipped = int(m.group(3))
                result.total = result.passed + result.failed + result.skipped

            d = pytest_dur.search(text)
            if d and d.lastindex:
                result.duration_s = float(d.group(1))

            return result

        # Go test: "ok/FAIL pkg ... Xs" or "--- FAIL:" / "--- PASS:"
        go_pattern = re.compile(r'(ok|FAIL)\s+\S+\s+(\d+\.?\d*)s')
        go_pass = re.compile(r'--- PASS:\s+(\S+)')
        go_fail = re.compile(r'--- FAIL:\s+(\S+)')

        go_matches = go_pattern.findall(text)
        if go_matches:
            result.passed = len(go_pass.findall(text))
            result.failed = len(go_fail.findall(text))
            result.total = result.passed + result.failed

        # cargo test: "test result: ok. X passed; Y failed"
        cargo_pattern = re.compile(r'test result:\s+(\w+)\.\s+(\d+)\s+passed;\s+(\d+)\s+failed')
        cargo_match = cargo_pattern.search(text)
        if cargo_match:
            result.passed = int(cargo_match.group(2))
            result.failed = int(cargo_match.group(3))
            result.total = result.passed + result.failed

        # Jest: "Tests: X passed, Y failed, Z total"
        jest_pattern = re.compile(r'Tests:\s+(\d+)\s+passed,\s+(\d+)\s+failed,\s+(\d+)\s+total')
        jest_match = jest_pattern.search(text)
        if jest_match:
            result.passed = int(jest_match.group(1))
            result.failed = int(jest_match.group(2))
            result.total = int(jest_match.group(3))

        return result

    # ── Diff Summarization ─────────────────────────────────────

    def summarize_diff(self, diff_text: str) -> DiffSummary:
        """Summarize a git diff output."""
        summary = DiffSummary()

        diff_file_pattern = re.compile(r'^diff --git a/(.+) b/(.+)', re.MULTILINE)
        new_file_pattern = re.compile(r'^new file mode', re.MULTILINE)
        deleted_file_pattern = re.compile(r'^deleted file mode', re.MULTILINE)
        insert_pattern = re.compile(r'^\+\s*[^+]', re.MULTILINE)
        delete_pattern = re.compile(r'^-\s*[^-]', re.MULTILINE)
        hunk_pattern = re.compile(r'^@@ .+ @@', re.MULTILINE)

        files = diff_file_pattern.findall(diff_text)
        summary.files_changed = len(files)

        for old_path, new_path in files:
            summary.modified_files.append(new_path)

        summary.new_files = new_file_pattern.findall(diff_text)
        summary.deleted_files = deleted_file_pattern.findall(diff_text)
        summary.insertions = len(insert_pattern.findall(diff_text))
        summary.deletions = len(delete_pattern.findall(diff_text))

        # Count hunks per file
        hunks = hunk_pattern.findall(diff_text)
        current_file = ""
        for i, line in enumerate(diff_text.split("\n")):
            file_match = diff_file_pattern.match(line) if i == 0 else None
            if line.startswith("diff --git"):
                m = re.match(r'diff --git a/(.+) b/(.+)', line)
                if m:
                    current_file = m.group(2)
                    summary.hunks_by_file[current_file] = 0
            elif line.startswith("@@ ") and current_file:
                summary.hunks_by_file[current_file] = \
                    summary.hunks_by_file.get(current_file, 0) + 1

        return summary

    # ── Coverage Parsing ───────────────────────────────────────

    def parse_coverage(self, text: str) -> dict:
        """Parse coverage report output."""
        coverage: Dict[str, Any] = {
            "total_percent": 0,
            "files": [],
        }

        # coverage.py: "TOTAL ... XX%"
        total_pattern = re.compile(r'TOTAL\s+.*?(\d+)%')
        total_match = total_pattern.search(text)
        if total_match:
            coverage["total_percent"] = int(total_match.group(1))

        # Per-file: "path/to/file.py ... X%"
        file_pattern = re.compile(r'(\S+\.\w+)\s+.*?(\d+)%')
        for match in file_pattern.finditer(text):
            fname = match.group(1)
            pct = int(match.group(2))
            if fname != "TOTAL":
                coverage["files"].append({"file": fname, "coverage": pct})

        return coverage

    # ── Error Classification ───────────────────────────────────

    @staticmethod
    def _classify_error(error_name: str) -> str:
        """Classify an error name into a category."""
        error_name = error_name.lower()
        if "import" in error_name or "module" in error_name:
            return "import_error"
        elif "syntax" in error_name:
            return "syntax_error"
        elif "type" in error_name:
            return "type_error"
        elif "attribute" in error_name:
            return "attribute_error"
        elif "key" in error_name:
            return "key_error"
        elif "index" in error_name:
            return "index_error"
        elif "value" in error_name:
            return "value_error"
        elif "name" in error_name:
            return "name_error"
        elif "file" in error_name:
            return "io_error"
        elif "permission" in error_name:
            return "permission_error"
        elif "assert" in error_name:
            return "assertion_error"
        elif "runtime" in error_name:
            return "runtime_error"
        elif "reference" in error_name:
            return "reference_error"
        elif "range" in error_name:
            return "range_error"
        return "error"

    @staticmethod
    def _suggest_fix(err: CodeError) -> str:
        """Generate a fix suggestion based on error type."""
        suggestions = {
            "import_error": "Check that the module is installed (`pip install X`) or the import path is correct.",
            "module_not_found": "Check that the module is installed (`pip install X`) or the import path is correct.",
            "syntax_error": "Check for missing colons, brackets, or indentation issues on the indicated line.",
            "type_error": "Verify that you are passing the correct type — check the expected argument types.",
            "attribute_error": "The object doesn't have this attribute. Check for typos or None values.",
            "name_error": "This variable or function is not defined. Check for typos or scope issues.",
            "index_error": "You are accessing an index that doesn't exist. Check the list/array length.",
            "key_error": "The key doesn't exist in the dictionary. Check key names or use .get().",
            "value_error": "The value passed is invalid for this operation. Check the accepted value range.",
            "io_error": "Check that the file exists at the specified path and you have read permissions.",
            "compilation": "Review the error message and check for type mismatches, missing imports, or syntax errors.",
            "linker_error": "A library or function is not being found. Check your link flags and dependencies.",
        }
        return suggestions.get(err.error_type, "Review the error message and check the indicated line.")

    # ── Quick Analysis ─────────────────────────────────────────

    def quick_analysis(self, text: str) -> str:
        """
        Quick all-in-one analysis of any code-related text.

        Returns a concise natural language summary.
        """
        lang = self.detect_language(text)

        # Try as error output
        errors = self.extract_errors(text)
        if errors:
            parts = [f"[{lang}] Found {len(errors)} error(s):"]
            for err in errors[:3]:
                loc = f" ({err.file}:{err.line})" if err.file else ""
                parts.append(f"  - {err.error_type}{loc}: {err.message[:120]}")
            return "\n".join(parts)

        # Try as test output
        tests = self.parse_test_results(text)
        if tests.total > 0:
            status = "✅ All passed" if tests.failed == 0 else f"⚠️ {tests.failed} failed"
            return f"[{lang}] Tests: {tests.passed}/{tests.total} passed, {tests.failed} failed. {status}"

        # Try as diff
        if text.startswith("diff ") or "@@" in text:
            diff = self.summarize_diff(text)
            return f"Diff: {diff.files_changed} files changed ({diff.insertions}+, {diff.deletions}-)"

        return f"[{lang}] Could not identify structured content in this text."


# Global singleton
code_assistant = CodeAssistant()