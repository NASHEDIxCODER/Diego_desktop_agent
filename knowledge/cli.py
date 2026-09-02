"""
Developer/test CLI for the local knowledge subsystem.

    python -m knowledge.cli index                 # index files
    python -m knowledge.cli status                # index status
    python -m knowledge.cli reindex <file>        # reindex file
    python -m knowledge.cli search "<query>"      # search local knowledge
    python -m knowledge.cli rebuild-embeddings    # rebuild embeddings
    python -m knowledge.cli roots                 # show indexed roots
    python -m knowledge.cli skipped               # show skipped/sensitive
    python -m knowledge.cli snapshot              # refresh PC snapshot
"""

from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    cmd = args[0].lower()

    from knowledge.service import knowledge_service

    if cmd in ("status", "index-status", "index status"):
        # Status must NOT start a background scan — it reports the
        # current persisted state (accurate counts, no side effects).
        knowledge_service.ensure_ready()
        print(json.dumps(knowledge_service.index_status(), indent=2))
        return 0

    knowledge_service.start()  # idempotent, non-blocking schema init

    if cmd in ("index", "index-files", "index files"):
        print(json.dumps(knowledge_service.index_files(), indent=2))
    elif cmd in ("reindex", "reindex-file", "reindex file"):
        if len(args) < 2:
            print("usage: reindex <path>")
            return 1
        print(json.dumps(knowledge_service.reindex_file(args[1]),
                         indent=2))
    elif cmd in ("search", "search-local-knowledge", "search local knowledge"):
        query = " ".join(args[1:])
        results = knowledge_service.search(query, top_k=8)
        for r in results:
            print(f"[{r['score']:.3f}] {r['doc_path']}"
                  + (f" ({r['locator']})" if r.get("locator") else ""))
            print("    " + r["text"][:160].replace("\n", " "))
        if not results:
            print("No results.")
    elif cmd in ("rebuild-embeddings", "rebuild embeddings"):
        print(json.dumps(knowledge_service.rebuild_embeddings(), indent=2))
    elif cmd in ("roots", "show-indexed-roots", "show indexed roots"):
        for r in knowledge_service.indexed_roots():
            print(r)
    elif cmd in ("skipped", "show-skipped", "show skipped paths",
                 "show skipped/sensitive paths"):
        print(json.dumps(knowledge_service.skipped_sensitive(), indent=2))
    elif cmd in ("snapshot",):
        snap = knowledge_service.refresh_snapshot()
        print(json.dumps(snap, indent=2, default=str)[:8000])
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())