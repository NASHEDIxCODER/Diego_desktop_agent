"""
Incremental, resumable, READ-ONLY file indexer.

Incremental guarantees:
  * A file is re-extracted ONLY when it is new or its content hash changed
    (size+mtime pre-filter, SHA-256 confirmation).
  * Deleted/moved files are removed from the index cleanly.
  * Identical paths are never duplicated (PRIMARY KEY upsert).
  * Per-file work is committed immediately → resumable after interruption.

Performance / safety:
  * Bounded concurrency (ThreadPoolExecutor, KNOWLEDGE_MAX_WORKERS).
  * Per-file extraction timeout (KNOWLEDGE_EXTRACTION_TIMEOUT_S).
  * Cancellation support via threading.Event.
  * One corrupted file can never crash the indexer.
  * Never writes to user files — opens them read-only only.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

from config.settings import settings
from knowledge import extractors
from knowledge.chunker import chunk_sections
from knowledge.embedder import LocalEmbedder, text_hash
from knowledge.policy import PathPolicy
from knowledge.store import KnowledgeStore, file_hash

logger = logging.getLogger(__name__)

# Extensions that are recognized but never text-extracted (metadata only)
_BINARY_META_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico",
    ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".avi", ".mkv", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".deb", ".rpm",
    ".exe", ".dll", ".so", ".bin", ".iso", ".img", ".apk", ".AppImage",
    ".db", ".sqlite", ".duckdb", ".pkl", ".whl", ".pyc", ".class",
    ".doc", ".xls", ".ppt",   # legacy binary office formats
}


class ScanProgress:
    """Thread-safe progress reporting for a scan."""

    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.scanned = 0
        self.updated = 0
        self.skipped = 0
        self.failed = 0
        self.removed = 0
        self.current_file = ""
        self.started_at = 0.0
        self.finished_at = 0.0

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "running": self.running,
                "scanned": self.scanned,
                "updated": self.updated,
                "skipped": self.skipped,
                "failed": self.failed,
                "removed": self.removed,
                "current_file": self.current_file,
                "elapsed_s": round(
                    (self.finished_at or time.time()) - self.started_at, 1)
                if self.started_at else 0.0,
            }

    def update(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)


class KnowledgeIndexer:
    """Incremental read-only indexer over approved scan roots."""

    def __init__(self, store: KnowledgeStore, policy: PathPolicy,
                 embedder: Optional["LocalEmbedder"] = None):
        self._store = store
        self._policy = policy
        self._embedder = embedder or LocalEmbedder()
        self.progress = ScanProgress()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._skipped_log: List[Dict] = []
        # Serializes scans: two concurrent scans (background + explicit
        # CLI call) must never index the same file simultaneously (that
        # is what produced duplicate PRIMARY KEY races before).
        self._scan_lock = threading.Lock()
        # Continuous (periodic) incremental rescan loop state.
        self._rescan_thread: Optional[threading.Thread] = None
        self._rescan_stop = threading.Event()
        self._rescan_interval = 300.0

    # ── Public API ────────────────────────────────────────────

    def start_background_scan(self) -> bool:
        """Start indexing in a daemon thread. Never blocks the caller
        (used at startup — Diego must never wait for indexing)."""
        if self.progress.snapshot()["running"]:
            logger.info("[KNOWLEDGE] scan already running")
            return False
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._safe_scan, name="knowledge-indexer", daemon=True)
        self._thread.start()
        return True

    def start_periodic_rescan(self, interval_s: Optional[float] = None
                              ) -> bool:
        """Start the continuous incremental rescan loop (daemon thread).

        Every interval an incremental scan runs: new/changed (size+mtime,
        SHA-256 confirmed) and deleted files are picked up automatically.
        The loop is serialized with explicit scans via _scan_lock, is
        cancellable, and never blocks the caller. Safe to call twice —
        the second call is a no-op.
        """
        if self._rescan_thread is not None and self._rescan_thread.is_alive():
            return False
        interval = (interval_s if interval_s is not None
                    else max(1.0, settings.KNOWLEDGE_RESCAN_INTERVAL_S))
        if interval <= 0:
            return False
        self._rescan_interval = interval
        self._rescan_stop.clear()
        self._rescan_thread = threading.Thread(
            target=self._rescan_loop, name="knowledge-rescan", daemon=True)
        self._rescan_thread.start()
        logger.info("[KNOWLEDGE] periodic rescan started (every %.0fs)",
                    interval)
        return True

    def stop_periodic_rescan(self) -> None:
        """Stop the periodic rescan loop (idempotent)."""
        self._rescan_stop.set()

    def _rescan_loop(self) -> None:
        """Continuous incremental indexing: scan, then wait. Each cycle
        is a full incremental pass (hash+mtime change detection, deletion
        detection, purge of denied material). Cancellation-aware."""
        while not self._rescan_stop.is_set():
            try:
                self._cancel.clear()  # a prior cancel must not leak in
                self._scan()
            except Exception as e:
                logger.error("[KNOWLEDGE] rescan cycle failed (recovered): %s",
                             e)
            # Wait for the interval; stop wakes immediately.
            self._rescan_stop.wait(self._rescan_interval)

    def cancel(self) -> None:
        """Request cancellation of the running scan."""
        self._cancel.set()

    def scan(self, roots: Optional[List[Path]] = None) -> Dict:
        """Synchronous incremental scan (used by tests / CLI)."""
        self._cancel.clear()
        return self._scan(roots)

    def skipped_paths(self) -> List[Dict]:
        return list(self._skipped_log)

    # ── Scanning ──────────────────────────────────────────────

    def _safe_scan(self) -> None:
        try:
            self._scan()
        except Exception as e:
            logger.error("[KNOWLEDGE] scan crashed (recovered): %s", e)
        finally:
            self.progress.update(running=False, finished_at=time.time())

    def _scan(self, roots: Optional[List[Path]] = None) -> Dict:
        # Concurrent scans are serialized (idempotent, race-free).
        with self._scan_lock:
            return self._scan_locked(roots)

    def _scan_locked(self, roots: Optional[List[Path]] = None) -> Dict:
        self.progress.update(running=True, scanned=0, updated=0,
                             skipped=0, failed=0, removed=0,
                             current_file="", started_at=time.time(),
                             finished_at=0.0)
        self._skipped_log = []
        roots = roots or self._policy.allow_roots
        logger.info("[KNOWLEDGE] scan start roots=%s",
                    [str(r) for r in roots])

        # 1. Discover files (policy-checked, read-only walk)
        discovered: List[Path] = []
        for root in roots:
            self._discover(root, discovered)
        # Dedupe: a symlink and its target resolve to the same real path —
        # index each real path exactly once (avoids duplicate upserts).
        discovered = list(dict.fromkeys(discovered))
        logger.info("[KNOWLEDGE] discovered %d candidate files",
                    len(discovered))

        # 2. Load existing index state
        known = {d["path"]: d for d in self._store.all_documents()}

        # 3. Decide what needs (re)indexing (incremental)
        to_index: List[Path] = []
        seen: Set[str] = set()
        for p in discovered:
            sp = str(p)
            seen.add(sp)
            if self._cancel.is_set():
                break
            try:
                st = os.lstat(p)
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                continue
            prev = known.get(sp)
            if prev:
                same_size = prev.get("size") == size
                same_mtime = abs((prev.get("mtime") or 0) - mtime) < 1e-6
                if same_size and same_mtime:
                    self.progress.update(
                        skipped=self.progress.snapshot()["skipped"] + 1)
                    continue  # unchanged — skip (incremental)
                if same_size and not same_mtime and prev.get("content_hash"):
                    # mtime-only change: confirm via content hash. If the
                    # content is identical, just refresh mtime — no
                    # re-extraction, no re-embedding.
                    # DEFENSE IN DEPTH: re-check the policy BEFORE the
                    # content hash opens the file — a denied path must
                    # never be opened just to hash it.
                    allowed, why, _ = self._policy.check(p)
                    if not allowed:
                        self._record_skip(p, why)
                        continue
                    if file_hash(p) == prev["content_hash"]:
                        self._store.upsert_document(
                            path=sp, filename=p.name,
                            root=self._root_of(p), size=size, mtime=mtime,
                            content_hash=prev["content_hash"],
                            file_type=p.suffix.lstrip(".") or "unknown",
                            mime_type=mimetypes.guess_type(sp)[0]
                            or "application/octet-stream",
                            extraction_status=prev.get("extraction_status")
                            or "extracted", error="",
                            chunk_count=prev.get("chunk_count") or 0)
                        self.progress.update(
                            skipped=self.progress.snapshot()["skipped"] + 1)
                        continue
            to_index.append(p)

        # 4. Handle deleted / moved files AND purge previously-indexed
        #    material that is now denied by policy. A credential/session/
        #    auth file that reached persistence before the policy existed
        #    must be removed — it must never remain retrievable.
        removed = 0
        for path_str in list(known.keys()):
            # Virtual paths (e.g. "diego://pc-snapshot") are not real
            # files on disk — never treat them as deleted.
            if path_str.startswith("diego://"):
                continue
            # Purge denied material by NAME ONLY first — the file is
            # never opened, hashed, or inspected to classify it (a
            # deleted secret must not be re-opened either).
            deny_reason = self._policy.name_deny_reason(Path(path_str))
            if deny_reason:
                self._store.delete_document(path_str)
                removed += 1
                self._record_skip(Path(path_str), deny_reason)
                continue
            if path_str not in seen:
                # Full policy check (metadata-only: lexists/realpath/
                # lstat — never opens file contents). Purges files that
                # are now outside approved roots or inside a deny path,
                # and deleted files.
                allowed, why, _ = self._policy.check(Path(path_str))
                if not allowed and why != "not found":
                    self._store.delete_document(path_str)
                    removed += 1
                    self._record_skip(Path(path_str), why)
                    continue
                if not os.path.lexists(path_str):
                    self._store.delete_document(path_str)
                    removed += 1
                    # Log by category only when the path is sensitive —
                    # never expose the secret filename.
                    if self._policy.name_deny_reason(Path(path_str)):
                        logger.info(
                            "[KNOWLEDGE] removed denied file from index")
                    else:
                        logger.info(
                            "[KNOWLEDGE] removed deleted file from index: %s",
                            path_str)
        self.progress.update(removed=removed)

        # 5. Index with bounded concurrency
        workers = max(1, settings.KNOWLEDGE_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="knowledge") as pool:
            futures = {pool.submit(self._index_file, p): p
                       for p in to_index}
            try:
                for fut in as_completed(futures):
                    if self._cancel.is_set():
                        for f in futures:
                            f.cancel()
                        break
                    _ = fut.result()  # exceptions handled inside
            except Exception as e:
                logger.error("[KNOWLEDGE] scan worker error (recovered): %s", e)

        self.progress.update(running=False, finished_at=time.time())
        snap = self.progress.snapshot()
        logger.info("[KNOWLEDGE] scan end: %s", snap)
        return snap

    def _discover(self, root: Path, out: List[Path], depth: int = 0) -> None:
        """Recursive, policy-checked, read-only directory walk."""
        if depth > 12 or self._cancel.is_set():
            return
        allowed, reason, real = self._policy.check(root)
        if not allowed:
            self._record_skip(root, reason)
            return
        try:
            entries = list(os.scandir(real))
        except OSError as e:
            self._record_skip(root, f"unreadable: {e}")
            return
        for entry in entries:
            if self._cancel.is_set():
                return
            p = Path(entry.path)
            is_link = entry.is_symlink()
            # Policy check ALWAYS runs (symlinks only accepted when their
            # real path stays inside approved roots and off the denylist).
            ok, why, real_p = self._policy.check(p)
            if not ok:
                self._record_skip(p, why)
                continue
            # NOTE: follow_symlinks=is_link — a plain is_file/is_dir with
            # follow_symlinks=False would silently skip symlinked files.
            if entry.is_dir(follow_symlinks=is_link):
                self._discover(p, out, depth + 1)
            elif entry.is_file(follow_symlinks=is_link):
                self.progress.update(
                    scanned=self.progress.snapshot()["scanned"] + 1)
                out.append(real_p)

    # Reasons that must NEVER be logged with the real path (the path
    # itself can leak a secret's existence/name).
    _SENSITIVE_MARKERS = ("sensitive", "generated")

    def _record_skip(self, path: Path, reason: str) -> None:
        sensitive = any(m in reason.lower() for m in self._SENSITIVE_MARKERS)
        if not sensitive:
            # A configured deny name can be a secret filename even when
            # the reason string is generic ("denylisted name"). Sanitize
            # by NAME ONLY — never open the file to decide.
            name_reason = self._policy.name_deny_reason(path)
            if name_reason and any(
                    m in name_reason.lower()
                    for m in self._SENSITIVE_MARKERS):
                sensitive = True
        if sensitive:
            # Category only — never the filename for secret material.
            self._skipped_log.append({"path": "<sensitive>",
                                      "reason": reason})
            logger.info("[KNOWLEDGE] skipped sensitive path (%s)", reason)
        else:
            self._skipped_log.append({"path": str(path), "reason": reason})
            logger.info("[KNOWLEDGE] skipped %s (%s)", path, reason)

    # ── Per-file indexing ─────────────────────────────────────

    def _index_file(self, path: Path) -> None:
        """Extract + chunk + embed + persist one file. Never raises."""
        # DEFENSE IN DEPTH: the policy is re-checked HERE, before the
        # file is stat'ed, hashed, opened, extracted, chunked, embedded,
        # or persisted. Secret material never reaches any of those steps.
        allowed, reason, real = self._policy.check(path)
        if not allowed:
            self._record_skip(path, reason)
            return
        path = real
        self.progress.update(current_file=str(path))
        try:
            st = os.lstat(path)
            size, mtime = st.st_size, st.st_mtime
        except OSError as e:
            self._record_skip(path, f"stat failed: {e}")
            return

        ext = path.suffix.lower()
        file_type = ext.lstrip(".") or "unknown"
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        # Content hash (incremental correctness)
        content_hash = file_hash(path)
        if not content_hash:
            self.progress.update(
                failed=self.progress.snapshot()["failed"] + 1)
            return

        # Extraction with timeout
        extractor = extractors.get_extractor(ext)
        if extractor is None:
            status = (extractors.STATUS_METADATA_ONLY
                      if ext in _BINARY_META_EXT
                      else extractors.STATUS_UNSUPPORTED)
            extracted = extractors.Extracted(status=status,
                                             error="no extractor")
        else:
            extracted = _extract_with_timeout(
                extractor, path, settings.KNOWLEDGE_EXTRACTION_TIMEOUT_S)

        chunks = []
        if extracted.status == extractors.STATUS_EXTRACTED:
            chunks = chunk_sections(extracted.sections)

        # Embed only chunks whose text is not already embedded for this
        # document (hash-based cache — regenerate only on content change).
        prev_hashes = self._prev_chunk_hashes(str(path))
        embeddings: List[Optional[object]] = [None] * len(chunks)
        need = [i for i, c in enumerate(chunks)
                if text_hash(c.text) not in prev_hashes]
        if need:
            texts = [chunks[i].text for i in need]
            vecs = self._embedder.embed_batch(texts)
            for i, v in zip(need, vecs):
                embeddings[i] = v
            logger.info("[KNOWLEDGE] embeddings: %d new / %d total for %s",
                        len(need), len(chunks), path.name)

        chunk_dicts = []
        for c, emb in zip(chunks, embeddings):
            chunk_dicts.append({
                "index": c.index,
                "locator": c.locator,
                "text": c.text,
                "text_hash": text_hash(c.text),
                "embedding": emb,
                "embedding_model": self._embedder.name if emb is not None else "",
            })

        try:
            self._store.replace_document(
                path=str(path), filename=path.name, root=self._root_of(path),
                size=size, mtime=mtime, content_hash=content_hash,
                file_type=file_type, mime_type=mime,
                extraction_status=extracted.status,
                error=extracted.error[:300],
                chunks=chunk_dicts)
        except Exception as e:
            logger.error("[KNOWLEDGE] persist failed for %s: %s", path, e)
            self.progress.update(
                failed=self.progress.snapshot()["failed"] + 1)
            return

        if extracted.status == extractors.STATUS_EXTRACTED:
            self.progress.update(
                updated=self.progress.snapshot()["updated"] + 1)
            logger.info("[KNOWLEDGE] indexed %s (%d chunks)", path,
                        len(chunk_dicts))
        elif extracted.status == extractors.STATUS_FAILED:
            self.progress.update(
                failed=self.progress.snapshot()["failed"] + 1)
            logger.info("[KNOWLEDGE] extraction failed %s (%s)",
                        path, extracted.error[:120])
        else:
            self.progress.update(
                skipped=self.progress.snapshot()["skipped"] + 1)
            logger.info("[KNOWLEDGE] metadata-only/unsupported %s (%s)",
                        path, extracted.error[:120])
        self._retriever_invalidate()

    def _prev_chunk_hashes(self, doc_path: str) -> Set[str]:
        """Text hashes of chunks already stored for a document."""
        try:
            with self._store._store.connect() as conn:
                if conn is None:
                    return set()
                rows = conn.execute(
                    "SELECT text_hash FROM knowledge_chunks "
                    "WHERE doc_path = ?", [doc_path]).fetchall()
            return {r[0] for r in rows if r[0]}
        except Exception:
            return set()

    def _root_of(self, path: Path) -> str:
        for root in self._policy.allow_roots:
            try:
                path.relative_to(root)
                return str(root)
            except ValueError:
                continue
        return str(path.parent)

    def _retriever_invalidate(self) -> None:
        # Late import avoids a circular dependency at module load.
        try:
            from knowledge.service import knowledge_service
            knowledge_service.invalidate_retriever_cache()
        except Exception:
            pass

    def reindex_file(self, path: Path) -> Dict:
        """Force re-index of a single file (developer command).

        Denied files are purged — never re-persisted.
        """
        p = Path(path)
        allowed, reason, _ = self._policy.check(p)
        if not allowed:
            self._record_skip(p, reason)
            self._store.delete_document(str(p))
            return {"path": str(p), "status": "denied", "chunks": 0}
        self._index_file(p)
        return {"path": str(p), "status": "done",
                "chunks": self._store.chunk_count(str(p))}


# ── Helpers ──────────────────────────────────────────────────────

def _extract_with_timeout(extractor, path: Path, timeout_s: float):
    """Run an extractor with a hard timeout (daemon thread + join)."""
    result: Dict = {}

    def _run():
        try:
            result["value"] = extractor.extract(path)
        except Exception as e:
            result["value"] = extractors.Extracted(
                status=extractors.STATUS_FAILED, error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=max(1.0, timeout_s))
    if t.is_alive():
        logger.warning("[KNOWLEDGE] extraction timeout (%.0fs): %s",
                       timeout_s, path)
        return extractors.Extracted(status=extractors.STATUS_FAILED,
                                    error="extraction timeout")
    return result.get("value") or extractors.Extracted(
        status=extractors.STATUS_FAILED, error="extractor returned nothing")