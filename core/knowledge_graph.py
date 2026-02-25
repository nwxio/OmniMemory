"""Knowledge Graph with Semantic Triples.

Provides a better Knowledge Graph than Memori:
- Semantic triples (subject, predicate, object)
- Automatic extraction from text
- Graph traversal and inference
- Batch processing for performance
- Multi-level attribution (entity/process/session)
- Optional Neo4j backend
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .db import db
from .ids import new_id
from .config import settings
from .search_match import build_like_clause, query_tokens, score_fields

_neo4j_backend: Any = None


def _get_neo4j() -> Any:
    """Get or init Neo4j backend."""
    global _neo4j_backend
    if _neo4j_backend is None:
        if getattr(settings, "neo4j_enabled", False):
            try:
                from .graph_db.neo4j_backend import get_neo4j_backend

                _neo4j_backend = get_neo4j_backend()
            except Exception:
                _neo4j_backend = None
    return _neo4j_backend


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_name(name: str) -> str:
    """Generate ID from name."""
    return hashlib.sha256(name.lower().encode()).hexdigest()[:16]


# Common predicates for better extraction
COMMON_PREDICATES = [
    "is",
    "has",
    "knows",
    "works_for",
    "likes",
    "prefers",
    "uses",
    "created",
    "located_in",
    "part_of",
    "related_to",
    "manages",
    "owns",
    "belongs_to",
    "connected_to",
    "depends_on",
    "wants",
    "needs",
    "avoids",
    "fears",
    "loves",
    "hates",
    "said",
    "told",
    "asked",
    "answered",
    "explained",
]

# Entity types for classification
ENTITY_TYPES = {
    "person": ["user", "customer", "developer", "manager", "agent"],
    "organization": ["company", "team", "department", "organization"],
    "location": ["city", "country", "address", "place"],
    "product": ["app", "software", "tool", "service", "product"],
    "concept": ["idea", "topic", "subject", "concept"],
    "event": ["meeting", "call", "session", "event"],
    "skill": ["ability", "skill", "capability"],
    "preference": ["preference", "setting", "config"],
}


@dataclass
class Triple:
    """A semantic triple."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_type: str = "text"
    source_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphPath:
    """A path in the knowledge graph."""

    nodes: list[str]
    edges: list[str]
    length: int
    confidence: float


class KnowledgeGraph:
    """Knowledge Graph with semantic triples.

    Features:
    - Add/query triples
    - Graph traversal (BFS, DFS)
    - Inference (transitive relations)
    - Batch processing
    - Entity resolution
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._use_neo4j = getattr(settings, "neo4j_enabled", False)
        if self._use_neo4j:
            self._neo4j = _get_neo4j()
            if not self._neo4j or not self._neo4j.is_available():
                self._use_neo4j = False
                self._neo4j = None

    def get_backend(self) -> str:
        """Get current backend name."""
        return "neo4j" if self._use_neo4j else "sqlite"

    async def set_backend(self, backend: str) -> dict[str, Any]:
        """Switch between backends.

        Args:
            backend: 'neo4j' or 'sqlite'
        """
        if backend == "neo4j":
            neo4j = _get_neo4j()
            if neo4j and neo4j.is_available():
                self._use_neo4j = True
                self._neo4j = neo4j
                return {"ok": True, "backend": "neo4j"}
            return {"ok": False, "error": "Neo4j not available"}
        else:
            self._use_neo4j = False
            self._neo4j = None
            return {"ok": True, "backend": "sqlite"}

    def _neo4j_backend(self) -> Any:
        if self._use_neo4j and self._neo4j:
            return self._neo4j
        return None

    async def _add_triple_neo4j(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        confidence: float,
        source_type: str,
        source_id: Optional[str],
        session_id: Optional[str],
        metadata: Optional[dict],
    ) -> dict[str, Any]:
        neo4j = self._neo4j_backend()
        if neo4j is None:
            raise RuntimeError("neo4j backend is not active")

        now = _utc_now()
        subj_id = _hash_name(subject)
        obj_id = _hash_name(object_name)
        triple_key = f"{subj_id}|{predicate}|{obj_id}"
        triple_id = _hash_name(triple_key)
        meta_json = json.dumps(metadata or {})

        rows = await neo4j.query(
            """
            MERGE (s:Entity {name: $subject})
            ON CREATE SET
                s.id = $subj_id,
                s.entity_type = $subject_type,
                s.created_at = $now,
                s.mention_count = 1
            ON MATCH SET s.mention_count = coalesce(s.mention_count, 0) + 1
            MERGE (o:Entity {name: $object_name})
            ON CREATE SET
                o.id = $obj_id,
                o.entity_type = $object_type,
                o.created_at = $now,
                o.mention_count = 1
            ON MATCH SET o.mention_count = coalesce(o.mention_count, 0) + 1
            MERGE (s)-[r:RELATES_TO {triple_key: $triple_key}]->(o)
            ON CREATE SET
                r.id = $triple_id,
                r.predicate = $predicate,
                r.confidence = $confidence,
                r.source_type = $source_type,
                r.source_id = $source_id,
                r.session_id = $session_id,
                r.created_at = $now,
                r.metadata_json = $metadata_json
            RETURN r.id AS triple_id
            """,
            {
                "subject": subject,
                "subj_id": subj_id,
                "subject_type": self._classify_entity(subject),
                "object_name": object_name,
                "obj_id": obj_id,
                "object_type": self._classify_entity(object_name),
                "triple_key": triple_key,
                "triple_id": triple_id,
                "predicate": predicate,
                "confidence": confidence,
                "source_type": source_type,
                "source_id": source_id,
                "session_id": session_id,
                "metadata_json": meta_json,
                "now": now,
            },
        )
        if rows and rows[0].get("triple_id"):
            triple_id = str(rows[0]["triple_id"])

        return {
            "triple_id": triple_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_name,
        }

    async def _get_triples_neo4j(
        self,
        subject: Optional[str],
        predicate: Optional[str],
        object_name: Optional[str],
        session_id: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        neo4j = self._neo4j_backend()
        if neo4j is None:
            raise RuntimeError("neo4j backend is not active")

        rows = await neo4j.query(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE ($subject IS NULL OR s.name = $subject)
              AND ($predicate IS NULL OR r.predicate = $predicate)
              AND ($object_name IS NULL OR o.name = $object_name)
              AND ($session_id IS NULL OR r.session_id = $session_id)
            RETURN
                r.id AS id,
                s.name AS subject,
                r.predicate AS predicate,
                o.name AS object,
                coalesce(r.confidence, 1.0) AS confidence,
                r.source_type AS source_type,
                r.source_id AS source_id,
                r.session_id AS session_id,
                r.created_at AS created_at,
                r.metadata_json AS metadata_json
            ORDER BY confidence DESC, created_at DESC
            LIMIT $limit
            """,
            {
                "subject": subject,
                "predicate": predicate,
                "object_name": object_name,
                "session_id": session_id,
                "limit": max(1, int(limit)),
            },
        )

        out: list[dict[str, Any]] = []
        for row in rows:
            meta: dict[str, Any] = {}
            meta_json = row.get("metadata_json")
            try:
                if isinstance(meta_json, str) and meta_json:
                    loaded = json.loads(meta_json)
                    if isinstance(loaded, dict):
                        meta = loaded
            except Exception:
                pass
            out.append(
                {
                    "id": row.get("id"),
                    "subject": row.get("subject"),
                    "predicate": row.get("predicate"),
                    "object": row.get("object"),
                    "confidence": row.get("confidence", 1.0),
                    "source_type": row.get("source_type"),
                    "source_id": row.get("source_id"),
                    "session_id": row.get("session_id"),
                    "created_at": row.get("created_at"),
                    "metadata": meta,
                }
            )
        return out

    async def _get_neighbors_neo4j(
        self,
        entity: str,
        direction: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        neo4j = self._neo4j_backend()
        if neo4j is None:
            raise RuntimeError("neo4j backend is not active")

        results: list[dict[str, Any]] = []
        safe_limit = max(1, int(limit))

        if direction in ("out", "both"):
            out_rows = await neo4j.query(
                """
                MATCH (s:Entity {name: $entity})-[r:RELATES_TO]->(o:Entity)
                RETURN s.name AS from_name, r.predicate AS predicate, o.name AS to_name,
                       'out' AS direction, coalesce(r.confidence, 1.0) AS confidence
                ORDER BY confidence DESC
                LIMIT $limit
                """,
                {"entity": entity, "limit": safe_limit},
            )
            for row in out_rows:
                results.append(
                    {
                        "from": row.get("from_name"),
                        "predicate": row.get("predicate"),
                        "to": row.get("to_name"),
                        "direction": "out",
                        "confidence": row.get("confidence", 1.0),
                    }
                )

        if direction in ("in", "both"):
            in_rows = await neo4j.query(
                """
                MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity {name: $entity})
                RETURN s.name AS from_name, r.predicate AS predicate, o.name AS to_name,
                       'in' AS direction, coalesce(r.confidence, 1.0) AS confidence
                ORDER BY confidence DESC
                LIMIT $limit
                """,
                {"entity": entity, "limit": safe_limit},
            )
            for row in in_rows:
                results.append(
                    {
                        "from": row.get("from_name"),
                        "predicate": row.get("predicate"),
                        "to": row.get("to_name"),
                        "direction": "in",
                        "confidence": row.get("confidence", 1.0),
                    }
                )
        return results

    async def _search_entities_neo4j(
        self,
        query: str,
        entity_type: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        neo4j = self._neo4j_backend()
        if neo4j is None:
            raise RuntimeError("neo4j backend is not active")

        q = (query or "").strip()
        if not q:
            return []
        tokens = query_tokens(q, max_terms=10)
        if not tokens:
            tokens = [q.casefold()]

        pool = max(1, int(limit) * 6)

        rows = await neo4j.query(
            """
            MATCH (n:Entity)
            WHERE any(tok IN $tokens WHERE toLower(n.name) CONTAINS tok)
              AND ($entity_type IS NULL OR n.entity_type = $entity_type)
            RETURN n.name AS name, n.entity_type AS type, coalesce(n.mention_count, 0) AS mention_count
            ORDER BY mention_count DESC
            LIMIT $limit
            """,
            {
                "tokens": tokens,
                "entity_type": entity_type,
                "limit": pool,
            },
        )
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            name = str(row.get("name") or "")
            if not name:
                continue
            etype = str(row.get("type") or "")
            mention_count = int(row.get("mention_count", 0) or 0)
            score = score_fields(q, [name, etype], tokens=tokens) + float(mention_count) * 0.01
            if score <= 0.0:
                continue
            scored.append(
                (
                    score,
                    {
                        "name": name,
                        "type": etype,
                        "mention_count": mention_count,
                        "role": "entity",
                        "search_score": float(score),
                    },
                )
            )

        scored.sort(key=lambda it: it[0], reverse=True)
        return [item for _, item in scored[: int(limit)]]

    async def _get_stats_neo4j(self) -> dict[str, Any]:
        neo4j = self._neo4j_backend()
        if neo4j is None:
            raise RuntimeError("neo4j backend is not active")

        node_rows = await neo4j.query("MATCH (n:Entity) RETURN count(n) AS entities")
        rel_rows = await neo4j.query(
            """
            MATCH (:Entity)-[r:RELATES_TO]->(:Entity)
            RETURN count(r) AS triples, count(DISTINCT r.predicate) AS predicates
            """
        )
        entities = int(node_rows[0].get("entities", 0)) if node_rows else 0
        triples = int(rel_rows[0].get("triples", 0)) if rel_rows else 0
        predicates = int(rel_rows[0].get("predicates", 0)) if rel_rows else 0
        return {
            "subjects": entities,
            "predicates": predicates,
            "objects": entities,
            "triples": triples,
        }

    async def _clear_neo4j(self, session_id: Optional[str]) -> dict[str, Any]:
        neo4j = self._neo4j_backend()
        if neo4j is None:
            raise RuntimeError("neo4j backend is not active")

        if session_id:
            count_rows = await neo4j.query(
                """
                MATCH (:Entity)-[r:RELATES_TO]->(:Entity)
                WHERE r.session_id = $session_id
                RETURN count(r) AS deleted
                """,
                {"session_id": session_id},
            )
            deleted = int(count_rows[0].get("deleted", 0)) if count_rows else 0
            await neo4j.query(
                """
                MATCH (:Entity)-[r:RELATES_TO]->(:Entity)
                WHERE r.session_id = $session_id
                DELETE r
                """,
                {"session_id": session_id},
            )
            await neo4j.query(
                """
                MATCH (n:Entity)
                WHERE NOT (n)--()
                DELETE n
                """
            )
            return {"deleted": deleted}

        count_rows = await neo4j.query(
            """
            MATCH (:Entity)-[r:RELATES_TO]->(:Entity)
            RETURN count(r) AS deleted
            """
        )
        deleted = int(count_rows[0].get("deleted", 0)) if count_rows else 0
        await neo4j.clear()
        return {"deleted": deleted}

    # --- Triple Operations ---

    async def add_triple(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        confidence: float = 1.0,
        source_type: str = "text",
        source_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Add a semantic triple to the knowledge graph.

        Args:
            subject: Subject entity (e.g., "John")
            predicate: Relationship (e.g., "works_for")
            object_name: Object entity (e.g., "Google")
            confidence: Confidence score (0.0 - 1.0)
            source_type: Source type ("text", "conversation", "document")
            source_id: Source identifier
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            dict with triple_id and status
        """
        if self._neo4j_backend() is not None:
            return await self._add_triple_neo4j(
                subject=subject,
                predicate=predicate,
                object_name=object_name,
                confidence=confidence,
                source_type=source_type,
                source_id=source_id,
                session_id=session_id,
                metadata=metadata,
            )

        now = _utc_now()

        async with db.connect() as conn:
            # Ensure subject exists
            subj_id = _hash_name(subject)
            await conn.execute(
                """INSERT INTO kg_subjects(id, name, entity_type, created_at, mention_count)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(name) DO UPDATE SET mention_count = kg_subjects.mention_count + 1""",
                (subj_id, subject, self._classify_entity(subject), now),
            )

            # Ensure predicate exists
            pred_id = _hash_name(predicate)
            await conn.execute(
                """INSERT INTO kg_predicates(id, name, created_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(name) DO NOTHING""",
                (pred_id, predicate, now),
            )

            # Ensure object exists
            obj_id = _hash_name(object_name)
            await conn.execute(
                """INSERT INTO kg_objects(id, name, entity_type, created_at, mention_count)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(name) DO UPDATE SET mention_count = kg_objects.mention_count + 1""",
                (obj_id, object_name, self._classify_entity(object_name), now),
            )

            # Add triple (upsert)
            triple_id = new_id("triple")
            meta_json = json.dumps(metadata or {})

            await conn.execute(
                """INSERT OR IGNORE INTO kg_triples(id, subject_id, predicate_id, object_id, confidence, source_type, source_id, session_id, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    triple_id,
                    subj_id,
                    pred_id,
                    obj_id,
                    confidence,
                    source_type,
                    source_id,
                    session_id,
                    now,
                    meta_json,
                ),
            )
            cur = await conn.execute(
                """SELECT id FROM kg_triples
                   WHERE subject_id = ? AND predicate_id = ? AND object_id = ?""",
                (subj_id, pred_id, obj_id),
            )
            row = await cur.fetchone()
            if row and row["id"]:
                triple_id = str(row["id"])

            await conn.commit()

        return {
            "triple_id": triple_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_name,
        }

    async def add_triples_batch(
        self,
        triples: list[Triple],
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add multiple triples efficiently.

        Args:
            triples: List of Triple objects
            session_id: Session identifier

        Returns:
            dict with count of added triples
        """
        added = 0
        for triple in triples:
            try:
                await self.add_triple(
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object_name=triple.object,
                    confidence=triple.confidence,
                    source_type=triple.source_type,
                    source_id=triple.source_id,
                    session_id=session_id or triple.session_id,
                    metadata=triple.metadata,
                )
                added += 1
            except Exception:
                pass

        return {"added": added, "total": len(triples)}

    async def get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query triples from the knowledge graph.

        Args:
            subject: Filter by subject (optional)
            predicate: Filter by predicate (optional)
            object_name: Filter by object (optional)
            session_id: Filter by session (optional)
            limit: Max results

        Returns:
            List of matching triples
        """
        if self._neo4j_backend() is not None:
            return await self._get_triples_neo4j(subject, predicate, object_name, session_id, limit)

        query = """
            SELECT t.id, s.name as subject, p.name as predicate, o.name as object,
                   t.confidence, t.source_type, t.source_id, t.session_id, t.created_at, t.metadata_json
            FROM kg_triples t
            JOIN kg_subjects s ON t.subject_id = s.id
            JOIN kg_predicates p ON t.predicate_id = p.id
            JOIN kg_objects o ON t.object_id = o.id
            WHERE 1=1
        """
        params: list[Any] = []

        if subject:
            query += " AND s.name = ?"
            params.append(subject)
        if predicate:
            query += " AND p.name = ?"
            params.append(predicate)
        if object_name:
            query += " AND o.name = ?"
            params.append(object_name)
        if session_id:
            query += " AND t.session_id = ?"
            params.append(session_id)

        query += " ORDER BY t.confidence DESC, t.created_at DESC LIMIT ?"
        params.append(limit)

        async with db.connect() as conn:
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()

        results = []
        for row in rows:
            meta = {}
            try:
                if row["metadata_json"]:
                    meta = json.loads(row["metadata_json"])
            except Exception:
                pass

            results.append(
                {
                    "id": row["id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "confidence": row["confidence"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                    "metadata": meta,
                }
            )

        return results

    # --- Graph Traversal ---

    async def get_neighbors(
        self,
        entity: str,
        direction: str = "both",
        depth: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get neighboring entities in the graph.

        Args:
            entity: Starting entity
            direction: "out" (subject→), "in" (→object), or "both"
            depth: Traversal depth (1 or 2)
            limit: Max results

        Returns:
            List of connected entities with relationships
        """
        if self._neo4j_backend() is not None:
            return await self._get_neighbors_neo4j(entity, direction, limit)

        results = []
        entity_lower = entity.lower()

        async with db.connect() as conn:
            if direction in ("out", "both"):
                # Outgoing: entity -> ? (subject -> object)
                cur = await conn.execute(
                    """SELECT s.name as subject, p.name as predicate, o.name as object, t.confidence
                       FROM kg_triples t
                       JOIN kg_subjects s ON t.subject_id = s.id
                       JOIN kg_predicates p ON t.predicate_id = p.id
                       JOIN kg_objects o ON t.object_id = o.id
                       WHERE LOWER(s.name) = ?
                       ORDER BY t.confidence DESC LIMIT ?""",
                    (entity_lower, limit),
                )
                for row in await cur.fetchall():
                    results.append(
                        {
                            "from": row["subject"],
                            "predicate": row["predicate"],
                            "to": row["object"],
                            "direction": "out",
                            "confidence": row["confidence"],
                        }
                    )

            if direction in ("in", "both"):
                # Incoming: ? -> entity (subject -> object)
                cur = await conn.execute(
                    """SELECT s.name as subject, p.name as predicate, o.name as object, t.confidence
                       FROM kg_triples t
                       JOIN kg_subjects s ON t.subject_id = s.id
                       JOIN kg_predicates p ON t.predicate_id = p.id
                       JOIN kg_objects o ON t.object_id = o.id
                       WHERE LOWER(o.name) = ?
                       ORDER BY t.confidence DESC LIMIT ?""",
                    (entity_lower, limit),
                )
                for row in await cur.fetchall():
                    results.append(
                        {
                            "from": row["subject"],
                            "predicate": row["predicate"],
                            "to": row["object"],
                            "direction": "in",
                            "confidence": row["confidence"],
                        }
                    )

        return results

    async def find_path(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int = 3,
    ) -> Optional[GraphPath]:
        """Find a path between two entities (BFS).

        Args:
            from_entity: Starting entity
            to_entity: Target entity
            max_depth: Maximum search depth

        Returns:
            GraphPath if found, None otherwise
        """
        # Simple BFS implementation
        visited: set[str] = set()
        queue: list[tuple[str, list[str], list[str]]] = [(from_entity.lower(), [], [])]

        while queue:
            current, nodes, edges = queue.pop(0)

            if current in visited:
                continue
            visited.add(current)

            if current == to_entity.lower():
                return GraphPath(
                    nodes=nodes + [current],
                    edges=edges,
                    length=len(edges),
                    confidence=1.0 / max(1, len(edges)),
                )

            if len(edges) >= max_depth:
                continue

            # Get neighbors
            neighbors = await self.get_neighbors(current, limit=20)
            for n in neighbors:
                if n["direction"] == "out":
                    next_entity = n["to"].lower()
                    queue.append(
                        (
                            next_entity,
                            nodes + [current],
                            edges + [n["predicate"]],
                        )
                    )
                elif n["direction"] == "in":
                    next_entity = n["from"].lower()
                    queue.append(
                        (
                            next_entity,
                            nodes + [current],
                            edges + [f"<-{n['predicate']}-"],
                        )
                    )

        return None

    # --- Inference ---

    async def infer_transitive(
        self,
        subject: str,
        predicate: str,
        max_depth: int = 3,
    ) -> list[str]:
        """Infer transitive relations (e.g., A works_for B, B part_of C => A works_for C).

        Args:
            subject: Starting entity
            predicate: Predicate to follow
            max_depth: Maximum inference depth

        Returns:
            List of inferred related entities
        """
        visited = set()
        results = []
        queue = [(subject.lower(), 0)]

        while queue:
            current, depth = queue.pop(0)

            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            # Get direct relations
            triples = await self.get_triples(subject=current, predicate=predicate, limit=20)

            for t in triples:
                obj = t["object"].lower()
                if obj not in visited:
                    results.append(t["object"])
                    queue.append((obj, depth + 1))

        return results

    # --- Entity Operations ---

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search for entities in the knowledge graph.

        Args:
            query: Search query
            entity_type: Filter by entity type (optional)
            limit: Max results

        Returns:
            List of matching entities
        """
        if self._neo4j_backend() is not None:
            return await self._search_entities_neo4j(query, entity_type, limit)

        q = (query or "").strip()
        if not q:
            return []
        tokens = query_tokens(q, max_terms=10)
        if not tokens:
            tokens = [q.casefold()]

        clause, params = build_like_clause(["name"], tokens, require_all_tokens=False)
        if not clause:
            return []

        pool = max(int(limit) * 6, int(limit), 20)
        results = []

        async with db.connect() as conn:
            # Search subjects
            if entity_type:
                cur = await conn.execute(
                    """SELECT name, entity_type, mention_count FROM kg_subjects 
                       WHERE """
                    + clause
                    + """
                       AND entity_type = ?
                       ORDER BY mention_count DESC LIMIT ?""",
                    (*params, entity_type, pool),
                )
            else:
                cur = await conn.execute(
                    """SELECT name, entity_type, mention_count FROM kg_subjects 
                       WHERE """
                    + clause
                    + """
                       ORDER BY mention_count DESC LIMIT ?""",
                    (*params, pool),
                )

            for row in await cur.fetchall():
                results.append(
                    {
                        "name": row["name"],
                        "type": row["entity_type"],
                        "mention_count": row["mention_count"],
                        "role": "subject",
                    }
                )

            # Search objects
            if entity_type:
                cur = await conn.execute(
                    """SELECT name, entity_type, mention_count FROM kg_objects 
                       WHERE """
                    + clause
                    + """
                       AND entity_type = ?
                       ORDER BY mention_count DESC LIMIT ?""",
                    (*params, entity_type, pool),
                )
            else:
                cur = await conn.execute(
                    """SELECT name, entity_type, mention_count FROM kg_objects 
                       WHERE """
                    + clause
                    + """
                       ORDER BY mention_count DESC LIMIT ?""",
                    (*params, pool),
                )

            for row in await cur.fetchall():
                results.append(
                    {
                        "name": row["name"],
                        "type": row["entity_type"],
                        "mention_count": row["mention_count"],
                        "role": "object",
                    }
                )

        # Dedupe by name, then rank by lexical score + mention_count.
        seen = set()
        unique_results = []
        for r in results:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique_results.append(r)

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in unique_results:
            score = score_fields(
                q, [str(item.get("name") or ""), str(item.get("type") or "")], tokens=tokens
            )
            score += float(item.get("mention_count") or 0) * 0.01
            if score <= 0.0:
                continue
            it2 = dict(item)
            it2["search_score"] = float(score)
            scored.append((score, it2))

        scored.sort(key=lambda it: it[0], reverse=True)
        return [item for _, item in scored[: int(limit)]]

    async def get_entity_facts(
        self,
        entity: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get all facts about an entity.

        Args:
            entity: Entity name
            limit: Max results

        Returns:
            List of facts (as natural language)
        """
        triples = await self.get_triples(subject=entity, limit=limit)

        facts = []
        for t in triples:
            facts.append(
                {
                    "fact": f"{t['subject']} {t['predicate']} {t['object']}",
                    "subject": t["subject"],
                    "predicate": t["predicate"],
                    "object": t["object"],
                    "confidence": t["confidence"],
                }
            )

        return facts

    # --- Statistics ---

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        if self._neo4j_backend() is not None:
            return await self._get_stats_neo4j()

        async with db.connect() as conn:
            cur = await conn.execute("SELECT COUNT(*) as c FROM kg_subjects")
            row = await cur.fetchone()
            subjects = int(row["c"]) if row else 0

            cur = await conn.execute("SELECT COUNT(*) as c FROM kg_predicates")
            row = await cur.fetchone()
            predicates = int(row["c"]) if row else 0

            cur = await conn.execute("SELECT COUNT(*) as c FROM kg_objects")
            row = await cur.fetchone()
            objects = int(row["c"]) if row else 0

            cur = await conn.execute("SELECT COUNT(*) as c FROM kg_triples")
            row = await cur.fetchone()
            triples = int(row["c"]) if row else 0

        return {
            "subjects": subjects,
            "predicates": predicates,
            "objects": objects,
            "triples": triples,
        }

    async def clear(self, session_id: Optional[str] = None) -> dict[str, Any]:
        """Clear knowledge graph data.

        Args:
            session_id: Only clear for specific session (optional)
        """
        if self._neo4j_backend() is not None:
            return await self._clear_neo4j(session_id)

        async with db.connect() as conn:
            if session_id:
                cur = await conn.execute(
                    "DELETE FROM kg_triples WHERE session_id = ?", (session_id,)
                )
                deleted = cur.rowcount
            else:
                cur = await conn.execute("DELETE FROM kg_triples")
                deleted = cur.rowcount
                await conn.execute("DELETE FROM kg_subjects")
                await conn.execute("DELETE FROM kg_predicates")
                await conn.execute("DELETE FROM kg_objects")

            await conn.commit()

        return {"deleted": deleted}

    # --- Helpers ---

    def _classify_entity(self, name: str) -> str:
        """Classify entity type based on name."""
        name_lower = name.lower()

        for entity_type, keywords in ENTITY_TYPES.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return entity_type

        return "entity"


# Singleton
_knowledge_graph: Optional[KnowledgeGraph] = None


def knowledge_graph() -> KnowledgeGraph:
    """Get knowledge graph singleton."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph
