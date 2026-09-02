"""
Diego Local Knowledge & Document Indexing subsystem.

READ-ONLY: this subsystem never modifies, deletes, renames, executes, or
writes user files. It only opens files for reading inside user-approved
scan roots (allowlist) and never follows symlinks that escape those roots
or point into denylisted (sensitive/system) locations.

Public API:
    knowledge.service.knowledge_service  — background indexer + retrieval
    knowledge.retriever                  — semantic + keyword retrieval
    knowledge.snapshot                   — read-only PC inventory collector
"""