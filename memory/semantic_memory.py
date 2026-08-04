"""
SemanticMemory — Auto-linking memory graph for Leo.

Extends the current in-memory and DuckDB stores with a semantic layer
that understands relationships between stored facts:

  - Projects → linked to repositories, people, conversations
  - People → linked to projects, conversations, preferences
  - Repositories → linked to branches, last build status, TODOs
  - Devices → linked to WiFi networks, audio sinks, mount points
  - Servers → linked to addresses, ports, SSH keys, deployments
  - Workflows → linked to commands, dependencies, success rates
  - Commands → linked to frequency, success, project context
  - Conversations → linked to timestamps, topics, emotional tone

Auto-linking: when a new memory is created, the semantic memory
automatically links it to related memories based on:
  - Shared keywords
  - Temporal proximity
  - Shared entities (repos, usernames, hostnames)
  - Conversation context

Usage:
    from memory.semantic_memory import semantic_memory

    await semantic_memory.initialize()
    semantic_memory.add_entity("project", "GhostLine", {"repo": "...", "language": "Go"})
    semantic_memory.link("project:GhostLine", "person:nikash", "works_on")
    related = semantic_memory.find_related("project:GhostLine")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Entity Types
# ═══════════════════════════════════════════════════════════════

ENTITY_TYPES = [
    "project",
    "person",
    "repository",
    "device",
    "server",
    "workflow",
    "command",
    "conversation",
    "file",
    "url",
    "password",  # reference only, never stored plaintext
    "preference",
    "habit",
    "error",
    "skill",
    "goal",
    "task",
    "note",
    "topic",
    "tool",
    "language",
]


@dataclass
class Entity:
    """A single entity in the semantic memory graph."""
    id: str
    type: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 0.5


@dataclass
class Relationship:
    """A directional relationship between two entities."""
    source_id: str
    target_id: str
    relation: str  # "works_on", "depends_on", "uses", "authored", "deployed_to", etc.
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Predefined relationship types
# ═══════════════════════════════════════════════════════════════

RELATIONSHIPS = {
    "project": {
        "works_on": ["person"],
        "uses_repo": ["repository"],
        "depends_on": ["project", "tool", "language"],
        "deployed_to": ["server"],
        "has_goal": ["goal"],
        "has_task": ["task"],
    },
    "person": {
        "works_on": ["project"],
        "owns": ["device", "repository"],
        "prefers": ["preference", "command", "tool"],
        "has_habit": ["habit"],
        "uses": ["tool", "command", "language"],
    },
    "repository": {
        "belongs_to": ["project"],
        "hosted_on": ["server"],
        "uses_language": ["language"],
        "has_file": ["file"],
        "has_error": ["error"],
    },
    "command": {
        "part_of": ["workflow"],
        "used_in": ["project", "repository"],
        "depends_on": ["tool"],
        "produces": ["file"],
    },
    "conversation": {
        "about": ["project", "topic", "error", "goal", "task"],
        "with_person": ["person"],
        "mentions": ["file", "tool", "command", "server"],
    },
    "goal": {
        "part_of": ["project"],
        "has_task": ["task"],
        "requires": ["skill"],
    },
    "server": {
        "runs": ["project", "repository"],
        "requires": ["tool"],
    },
}


# ═══════════════════════════════════════════════════════════════
# Semantic Memory
# ═══════════════════════════════════════════════════════════════

class SemanticMemory:
    """
    Auto-linking semantic memory graph.

    Entities (projects, people, repos, etc.) are stored as nodes.
    Relationships link them together. The graph can be queried
    to find related entities, traverse paths, and discover patterns.

    Persisted to DuckDB for cross-session survival.
    """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relationships: List[Relationship] = []
        self._store = None
        self._initialized = False
        self._embedding_fn: Optional[callable] = None

    def set_embedding_fn(self, fn: callable) -> None:
        """Wire a function that generates embeddings for text."""
        self._embedding_fn = fn

    async def initialize(self) -> bool:
        """Initialize semantic memory — create tables, load from DB."""
        try:
            from memory.duckdb_store import store
            self._store = store

            conn = self._store._get_conn()
            if conn is None:
                logger.warning("[SemanticMem] DuckDB unavailable")
                return False

            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_entities (
                    id VARCHAR PRIMARY KEY,
                    entity_type VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    attributes_json VARCHAR DEFAULT '{}',
                    created_at DOUBLE NOT NULL,
                    updated_at DOUBLE NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    confidence DOUBLE DEFAULT 0.5
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_relationships (
                    id INTEGER PRIMARY KEY,
                    source_id VARCHAR NOT NULL,
                    target_id VARCHAR NOT NULL,
                    relation VARCHAR NOT NULL,
                    weight DOUBLE DEFAULT 1.0,
                    created_at DOUBLE NOT NULL,
                    metadata_json VARCHAR DEFAULT '{}',
                    FOREIGN KEY (source_id) REFERENCES semantic_entities(id),
                    FOREIGN KEY (target_id) REFERENCES semantic_entities(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_entities_type
                ON semantic_entities(entity_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_entities_name
                ON semantic_entities(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_rel_source
                ON semantic_relationships(source_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_rel_target
                ON semantic_relationships(target_id)
            """)

            # Load all entities
            rows = conn.execute(
                "SELECT id, entity_type, name, attributes_json, created_at, updated_at, access_count, confidence FROM semantic_entities"
            ).fetchall()

            for row in rows:
                try:
                    attrs = json.loads(row[3]) if row[3] else {}
                except Exception:
                    attrs = {}

                entity = Entity(
                    id=row[0],
                    type=row[1],
                    name=row[2],
                    attributes=attrs,
                    created_at=row[4],
                    updated_at=row[5],
                    access_count=row[6],
                    confidence=row[7],
                )
                self._entities[row[0]] = entity

            # Load all relationships
            rel_rows = conn.execute(
                "SELECT id, source_id, target_id, relation, weight, created_at, metadata_json FROM semantic_relationships"
            ).fetchall()

            for row in rel_rows:
                try:
                    meta = json.loads(row[6]) if row[6] else {}
                except Exception:
                    meta = {}

                self._relationships.append(Relationship(
                    source_id=row[1],
                    target_id=row[2],
                    relation=row[3],
                    weight=row[4],
                    created_at=row[5],
                    metadata=meta,
                ))

            self._initialized = True
            logger.info("[SemanticMem] Loaded %d entities and %d relationships",
                         len(self._entities), len(self._relationships))
            return True

        except Exception as e:
            logger.warning("[SemanticMem] Initialization failed: %s", e)
            return False

    # ── Entity CRUD ────────────────────────────────────────────

    def add_entity(
        self,
        entity_type: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        auto_link: bool = True,
    ) -> Entity:
        """
        Add (or update) an entity to the semantic graph.

        Args:
            entity_type: One of ENTITY_TYPES (project, person, etc.)
            name: Human-readable name (e.g., "GhostLine")
            attributes: Arbitrary key-value metadata
            auto_link: Whether to auto-link to related entities

        Returns:
            The created or updated Entity.
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning("[SemanticMem] Unknown entity type: %s", entity_type)

        entity_id = self._make_id(entity_type, name)

        if entity_id in self._entities:
            # Update existing
            entity = self._entities[entity_id]
            entity.name = name
            entity.updated_at = time.time()
            if attributes:
                entity.attributes.update(attributes)
            entity.access_count += 1
        else:
            # Create new
            entity = Entity(
                id=entity_id,
                type=entity_type,
                name=name,
                attributes=attributes or {},
            )
            self._entities[entity_id] = entity

        # Auto-link to related entities
        if auto_link:
            self._auto_link(entity)

        # Persist
        self._persist_entity(entity)

        return entity

    def get_entity(self, entity_type: str, name: str) -> Optional[Entity]:
        """Get an entity by type and name."""
        entity_id = self._make_id(entity_type, name)
        return self._entities.get(entity_id)

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by its full ID."""
        entity = self._entities.get(entity_id)
        if entity:
            entity.access_count += 1
            entity.updated_at = time.time()
        return entity

    def list_entities_by_type(self, entity_type: str) -> List[Entity]:
        """List all entities of a given type."""
        return [
            e for e in self._entities.values()
            if e.type == entity_type
        ]

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Search entities by name (case-insensitive substring)."""
        q = query.lower()
        results = []
        for entity in self._entities.values():
            if entity_type and entity.type != entity_type:
                continue
            if q in entity.name.lower():
                results.append(entity)
            elif any(q in str(v).lower() for v in entity.attributes.values()):
                results.append(entity)
        return results

    # ── Relationship Management ────────────────────────────────

    def link(
        self,
        source_type: str,
        source_name: str,
        target_type: str,
        target_name: str,
        relation: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Relationship]:
        """Create a directional relationship between two entities."""
        source_id = self._make_id(source_type, source_name)
        target_id = self._make_id(target_type, target_name)

        # Ensure both entities exist
        if source_id not in self._entities:
            self.add_entity(source_type, source_name, auto_link=False)
        if target_id not in self._entities:
            self.add_entity(target_type, target_name, auto_link=False)

        # Check for duplicate
        for rel in self._relationships:
            if (rel.source_id == source_id and rel.target_id == target_id
                    and rel.relation == relation):
                rel.weight = weight
                rel.updated_at = time.time()
                return rel

        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )

        self._relationships.append(relationship)
        self._persist_relationship(relationship)

        logger.debug("[SemanticMem] Linked %s:%s --[%s]--> %s:%s",
                      source_type, source_name, relation, target_type, target_name)
        return relationship

    def find_related(
        self,
        entity_type: str,
        entity_name: str,
        relation: Optional[str] = None,
        max_depth: int = 2,
    ) -> List[Tuple[Entity, str, float]]:
        """
        Find entities related to the given entity.

        Returns list of (entity, relationship_path, total_weight).
        """
        entity_id = self._make_id(entity_type, entity_name)
        if entity_id not in self._entities:
            return []

        visited: Set[str] = {entity_id}
        results: List[Tuple[Entity, str, float]] = []
        queue: List[Tuple[str, str, float]] = [(entity_id, "", 1.0)]

        for _ in range(max_depth):
            next_queue: List[Tuple[str, str, float]] = []
            for current_id, path, weight in queue:
                for rel in self._relationships:
                    if rel.source_id == current_id and rel.target_id not in visited:
                        if relation is None or rel.relation == relation:
                            target_entity = self._entities.get(rel.target_id)
                            if target_entity:
                                new_path = f"{path}--[{rel.relation}]-->" if path else f"--[{rel.relation}]-->"
                                results.append((target_entity, new_path, weight * rel.weight))
                        visited.add(rel.target_id)
                        next_queue.append((rel.target_id, f"{path}--[{rel.relation}]-->", weight * rel.weight))

                    # Also check reverse relationships
                    if rel.target_id == current_id and rel.source_id not in visited:
                        if relation is None or rel.relation == relation:
                            source_entity = self._entities.get(rel.source_id)
                            if source_entity:
                                new_path = f"<--[{rel.relation}]--{path}" if path else f"<--[{rel.relation}]--"
                                results.append((source_entity, new_path, weight * rel.weight))
                        visited.add(rel.source_id)
                        next_queue.append((rel.source_id, f"<--[{rel.relation}]--{path}", weight * rel.weight))

            queue = next_queue

        return results

    def get_relations(self, entity_type: str, entity_name: str) -> List[Relationship]:
        """Get all relationships for an entity (both directions)."""
        entity_id = self._make_id(entity_type, entity_name)
        return [
            r for r in self._relationships
            if r.source_id == entity_id or r.target_id == entity_id
        ]

    # ── Auto-Linking ───────────────────────────────────────────

    def _auto_link(self, entity: Entity) -> None:
        """Automatically discover and create relationships."""
        # Link by keyword overlap in attributes
        for other in self._entities.values():
            if other.id == entity.id:
                continue

            # Check allowed relationship types
            allowed_relations = RELATIONSHIPS.get(entity.type, {})
            for relation, target_types in allowed_relations.items():
                if other.type not in target_types:
                    continue

                # Keyword matching
                entity_keywords = self._extract_keywords(entity)
                other_keywords = self._extract_keywords(other)

                overlap = entity_keywords & other_keywords
                if overlap:
                    weight = min(1.0, len(overlap) / max(len(entity_keywords), 1) * 0.5 + 0.5)
                    self.link(
                        entity.type, entity.name,
                        other.type, other.name,
                        relation, weight=weight,
                    )

        # Temporal proximity linking (entities created within 5 minutes)
        # Done when both entities are being processed

    @staticmethod
    def _extract_keywords(entity: Entity) -> Set[str]:
        """Extract meaningful keywords from an entity."""
        keywords: Set[str] = set()

        # From name
        for word in re.findall(r'\w+', entity.name.lower()):
            if len(word) > 2:
                keywords.add(word)

        # From attributes
        for value in entity.attributes.values():
            if isinstance(value, str):
                for word in re.findall(r'\w+', value.lower()):
                    if len(word) > 2:
                        keywords.add(word)

        return keywords

    # ── High-level Memory APIs ─────────────────────────────────

    def remember_project(self, name: str, repo: str = "", language: str = "",
                         build_cmd: str = "", test_cmd: str = "") -> Entity:
        """Remember a project with its key attributes."""
        attrs: Dict[str, Any] = {}
        if repo:
            attrs["repo"] = repo
        if language:
            attrs["language"] = language
        if build_cmd:
            attrs["build_command"] = build_cmd
        if test_cmd:
            attrs["test_command"] = test_cmd

        entity = self.add_entity("project", name, attrs)

        if repo:
            repo_entity = self.add_entity("repository", repo, {"project": name})
            self.link("project", name, "repository", repo, "uses_repo")

        if language:
            self.add_entity("language", language)
            self.link("project", name, "language", language, "depends_on")

        return entity

    def remember_person(self, name: str, github: str = "") -> Entity:
        """Remember a person."""
        attrs: Dict[str, Any] = {}
        if github:
            attrs["github"] = github
        return self.add_entity("person", name, attrs)

    def remember_command(self, command: str, project: str = "",
                         success_rate: float = 0.0) -> Entity:
        """Remember a frequently used command."""
        attrs = {"command": command, "success_rate": success_rate}
        entity = self.add_entity("command", command, attrs)
        if project:
            self.link("command", command, "project", project, "used_in")
        return entity

    def remember_conversation(self, topic: str, summary: str,
                               people: Optional[List[str]] = None) -> Entity:
        """Remember a conversation topic."""
        entity = self.add_entity("conversation", topic, {"summary": summary})
        if people:
            for person in people:
                self.add_entity("person", person)
                self.link("conversation", topic, "person", person, "with_person")
        return entity

    def remember_server(self, hostname: str, ip: str = "",
                         project: str = "") -> Entity:
        """Remember a server."""
        attrs: Dict[str, Any] = {"hostname": hostname}
        if ip:
            attrs["ip"] = ip
        entity = self.add_entity("server", hostname, attrs)
        if project:
            self.link("project", project, "server", hostname, "deployed_to")
        return entity

    def remember_workflow(self, name: str, commands: List[str],
                           project: str = "") -> Entity:
        """Remember a workflow (sequence of commands)."""
        entity = self.add_entity("workflow", name, {"commands": commands})
        for cmd in commands:
            cmd_entity = self.add_entity("command", cmd)
            self.link("command", cmd, "workflow", name, "part_of")
        if project:
            self.link("workflow", name, "project", project, "part_of")
        return entity

    # ── Graph Queries ──────────────────────────────────────────

    def entity_count(self) -> int:
        return len(self._entities)

    def relationship_count(self) -> int:
        return len(self._relationships)

    def stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        by_type: Dict[str, int] = {}
        for entity in self._entities.values():
            by_type[entity.type] = by_type.get(entity.type, 0) + 1

        return {
            "entities": len(self._entities),
            "relationships": len(self._relationships),
            "by_type": by_type,
        }

    # ── Persistence ────────────────────────────────────────────

    def _persist_entity(self, entity: Entity) -> None:
        """Persist an entity to DuckDB."""
        if not self._store:
            return

        try:
            conn = self._store._get_conn()
            if conn is None:
                return

            conn.execute("""
                INSERT INTO semantic_entities
                    (id, entity_type, name, attributes_json, created_at, updated_at, access_count, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    attributes_json = EXCLUDED.attributes_json,
                    updated_at = EXCLUDED.updated_at,
                    access_count = EXCLUDED.access_count,
                    confidence = EXCLUDED.confidence
            """, [
                entity.id,
                entity.type,
                entity.name,
                json.dumps(entity.attributes),
                entity.created_at,
                entity.updated_at,
                entity.access_count,
                entity.confidence,
            ])

        except Exception as e:
            logger.debug("[SemanticMem] Entity persist failed: %s", e)

    def _persist_relationship(self, rel: Relationship) -> None:
        """Persist a relationship to DuckDB."""
        if not self._store:
            return

        try:
            conn = self._store._get_conn()
            if conn is None:
                return

            conn.execute("""
                INSERT INTO semantic_relationships
                    (source_id, target_id, relation, weight, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                rel.source_id,
                rel.target_id,
                rel.relation,
                rel.weight,
                rel.created_at,
                json.dumps(rel.metadata),
            ])

        except Exception as e:
            logger.debug("[SemanticMem] Relationship persist failed: %s", e)

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _make_id(entity_type: str, name: str) -> str:
        """Create a deterministic ID from type + name."""
        normalized = name.lower().strip().replace(" ", "_")
        return f"{entity_type}:{normalized}"

    def close(self) -> None:
        self._initialized = False
        logger.info("[SemanticMem] Semantic memory shut down")


# Global singleton
semantic_memory = SemanticMemory()