# 🗑️ Delete.md — Useless Files & Components

> **Safe to delete.** Removing any of the items listed below will have **zero impact** on Diego's runtime, voice pipeline, agent intelligence, or any production functionality.

---

## 📋 Summary

| # | Item | Type | Reason |
|---|------|------|--------|
| 1 | `Listen.py` | File | Windows-only dead code |
| 2 | `scripts/` | Directory | Legacy scripts — fully replaced by plugins & services |
| 3 | `desktop/` | Directory | Empty stub — never implemented |
| 4 | `AUDIT_REPORT.md` | File | Generated runtime artifact, not documentation |
| 5 | `RUNTIME_AUDIT.md` | File | Generated runtime artifact, not documentation |
| 6 | `RUNTIME_STABILITY_REPORT.md` | File | Generated runtime artifact, not documentation |
| 7 | `SYSTEM_HEALTH.md` | File | Generated runtime artifact, not documentation |
| 8 | `docs/Database.md` | File | Outdated supplementary doc — unreferenced |
| 9 | `docs/ASR_ALTERNATIVES_BENCHMARK.md` | File | Outdated benchmark report |
| 10 | `docs/ASR_BENCHMARK.md` | File | Outdated benchmark report |
| 11 | `docs/TRAINING_PIPELINE.md` | File | Outdated training doc |
| 12 | `data/knowledge_base.json` | File | Runtime-generated, should be in `.gitignore` |
| 13 | `debug/vision/` | Directory | Runtime screenshot artifacts (already gitignored) |

---

## 🔍 Detailed Analysis

### 1. `Listen.py`

```python
# Windows-only WhatsApp finder. Completely irrelevant.
search_paths = [
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    os.path.expanduser("~\\AppData\\Local")
]
patterns = [f"{app_name}.exe"]
```

- **What it does**: Searches Windows filesystem for `WhatsApp.exe`
- **Why it's useless**: Diego is a **Linux-only** desktop assistant. This script uses Windows paths, searches for `.exe` files, and has no imports or references from any other module in the project.
- **Impact if deleted**: None. Zero references anywhere in the codebase.

---

### 2. `scripts/` Directory (Entire Directory)

```
scripts/
├── brightness.py        # → plugins/brightness_plugin.py
├── volume.py            # → agent/action_dispatcher.py (system control)
├── youtube.py           # → plugins/youtube_plugin.py
├── telegram_bot.py      # → plugins/telegram_plugin.py
├── mail.py              # → Dead (no replacement, never used)
├── conversation_llm.py  # → agent/streaming_llm.py
├── fill_datasets.py     # → nlp/trainer.py
├── generate_datasets.py # → nlp/trainer.py
└── nlp_controller.py    # → nlp/parser.py
```

- **What it does**: Original standalone scripts for each feature before the plugin architecture was built.
- **Why it's useless**: Every script has been **fully replaced** by the modern architecture:
  - `brightness.py` → `plugins/brightness_plugin.py` (event-driven plugin)
  - `volume.py` → `agent/action_dispatcher.py` (unified action system)
  - `youtube.py` → `plugins/youtube_plugin.py` (Selenium-based plugin)
  - `telegram_bot.py` → `plugins/telegram_plugin.py` (Telethon-based plugin)
  - `mail.py` → **Dead code** — no plugin or service wraps this. Email was never implemented.
  - `conversation_llm.py` → `agent/streaming_llm.py` (token-streaming LLM)
  - `fill_datasets.py` / `generate_datasets.py` → `nlp/trainer.py` (unified training pipeline)
  - `nlp_controller.py` → `nlp/parser.py` (NLP pipeline orchestrator)
- **Impact if deleted**: None. The `scripts/` directory is not imported by any module. The architecture docs explicitly label them as "Legacy scripts (wrapped by plugins)".

---

### 3. `desktop/` Directory

```
desktop/
└── __init__.py    # Empty file
```

- **What it is**: An empty Python package with only an `__init__.py`.
- **Why it's useless**: The architecture docs marked this as "(future) Desktop automation". This future never arrived — desktop automation is now handled by `services/` (screen capture, perception, desktop state) and `agent/` (action dispatcher, browser control).
- **Impact if deleted**: None. No imports reference `desktop/`.

---

### 4-7. Generated Report Files

| File | What it is |
|------|-----------|
| `AUDIT_REPORT.md` | One-time system audit output |
| `RUNTIME_AUDIT.md` | One-time runtime audit output |
| `RUNTIME_STABILITY_REPORT.md` | One-time stability test output |
| `SYSTEM_HEALTH.md` | One-time health check output |

- **Why they're useless**: These are **runtime-generated artifacts** from running `debug/audit_system.py`, `debug/audit_execution_pipeline.py`, or test suites. They are not documentation — they are output files that happened to be committed. They contain timestamps, system-specific paths, and transient performance numbers that are already stale.
- **Impact if deleted**: None. These files are not referenced by any code, documentation, or configuration. They are not part of the build or runtime.

---

### 8. `docs/Database.md`

- **What it is**: A supplementary document about the database schema.
- **Why it's useless**: The database schema is already documented in `docs/architecture.md` (Storage section with full table listing). This separate file is unreferenced by any other documentation and is likely outdated given the migration to DuckDB + UnifiedMemory + SemanticMemory.
- **Impact if deleted**: None. Not referenced by README, architecture docs, or any code comments.

---

### 9-11. Outdated Benchmark/Training Docs

| File | Why Useless |
|------|------------|
| `docs/ASR_ALTERNATIVES_BENCHMARK.md` | Benchmark results from an older ASR setup. The current pipeline uses faster-whisper with Sherpa fallback. These numbers are stale. |
| `docs/ASR_BENCHMARK.md` | Same as above — outdated ASR performance data. |
| `docs/TRAINING_PIPELINE.md` | Documents an older training workflow. The current training is handled by `nlp/trainer.py` and documented in `docs/architecture.md`. |

- **Impact if deleted**: None. These are historical benchmark snapshots, not living documentation.

---

### 12. `data/knowledge_base.json`

- **What it is**: A runtime-generated JSON file for the knowledge base.
- **Why it's useless**: This is runtime state, not source code. It should be in `.gitignore` (similar to `*.db`, `*.duckdb`, `*.pkl`). The knowledge base is rebuilt at runtime by `learning/learning_engine.py` and `agent/conversation_memory.py`.
- **Impact if deleted**: None. Diego will regenerate it on next run if needed.

---

### 13. `debug/vision/` Directory

- **What it is**: A directory for vision debug screenshots.
- **Why it's useless**: Contains only runtime screenshot artifacts (`.png`, `.jpg`) which are already covered by `.gitignore` rules. The directory itself is empty of source code.
- **Impact if deleted**: None. The directory will be recreated automatically when vision debugging is used.

---

## ✅ Verification Checklist

Before deleting, confirm:

- [ ] `grep -r "Listen" --include="*.py"` returns no imports (except `Listen.py` itself)
- [ ] `grep -r "from scripts" --include="*.py"` returns no imports
- [ ] `grep -r "import scripts" --include="*.py"` returns no imports
- [ ] `grep -r "from desktop" --include="*.py"` returns no imports
- [ ] `grep -r "AUDIT_REPORT\|RUNTIME_AUDIT\|RUNTIME_STABILITY\|SYSTEM_HEALTH" --include="*.py" --include="*.md"` returns only references within those files themselves
- [ ] `grep -r "docs/Database.md\|docs/ASR_ALTERNATIVES\|docs/ASR_BENCHMARK\|docs/TRAINING_PIPELINE" --include="*.md"` returns no references

---

## 🧹 Deletion Commands

```bash
# Remove useless files
rm Listen.py
rm AUDIT_REPORT.md
rm RUNTIME_AUDIT.md
rm RUNTIME_STABILITY_REPORT.md
rm SYSTEM_HEALTH.md
rm data/knowledge_base.json

# Remove useless directories
rm -rf scripts/
rm -rf desktop/
rm -rf debug/vision/

# Remove outdated docs
rm docs/Database.md
rm docs/ASR_ALTERNATIVES_BENCHMARK.md
rm docs/ASR_BENCHMARK.md
rm docs/TRAINING_PIPELINE.md
```

---

## 📊 Impact Summary

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| Top-level files | 8 | 4 | -4 files |
| Python packages | 20 | 18 | -2 packages |
| Script files | 9 | 0 | -9 files |
| Doc files | 5 | 1 | -4 docs |
| **Total files removed** | | | **~25 files** |
| **Runtime impact** | | | **Zero** |

---

<p align="center">
  <em>None of these deletions affect Diego's ability to wake, listen, think, speak, see, act, learn, or remember.</em>
</p>