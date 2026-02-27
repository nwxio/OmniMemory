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

TEMPORAL_RELATION_POLICY: dict[str, str] = {
    # Single-active: at most one active object per (subject, predicate).
    "works_for": "single_active",
    "belongs_to": "single_active",
    "prefers": "single_active",
}

DEFAULT_SINGLE_ACTIVE_PREDICATES = ",".join(TEMPORAL_RELATION_POLICY.keys())


def _normalize_ts(value: Optional[str], *, default_now: bool = False) -> Optional[str]:
    if value is None:
        if default_now:
            return _utc_now()
        return None
    s = str(value).strip()
    if not s:
        if default_now:
            return _utc_now()
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return _utc_now() if default_now else None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _temporal_policy(predicate: str) -> str:
    configured = str(
        getattr(settings, "kg_temporal_single_active_predicates", DEFAULT_SINGLE_ACTIVE_PREDICATES)
        or ""
    )
    single_active = {
        token.strip().lower() for token in configured.replace(";", ",").split(",") if token.strip()
    }
    key = str(predicate or "").strip().lower()
    return "single_active" if key in single_active else "multi_active"


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
                r.updated_at = $now,
                r.valid_from = $now,
                r.valid_to = NULL,
                r.is_active = true,
                r.version = 1,
                r.last_event_type = 'assert',
                r.metadata_json = $metadata_json
            ON MATCH SET
                r.confidence = $confidence,
                r.source_type = $source_type,
                r.source_id = $source_id,
                r.session_id = $session_id,
                r.updated_at = $now,
                r.valid_to = NULL,
                r.is_active = true,
                r.version = coalesce(r.version, 1) + 1,
                r.last_event_type = 'assert',
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

    async def _insert_temporal_event_neo4j(self, payload: dict[str, Any]) -> str:
        neo4j = self._neo4j_backend()
        if neo4j is None:
            raise RuntimeError("neo4j backend is not active")

        event_id = new_id("kgev")
        await neo4j.query(
            """
            CREATE (e:KGTemporalEvent {
                id: $event_id,
                triple_id: $triple_id,
                triple_key: $triple_key,
                subject: $subject,
                subject_id: $subject_id,
                predicate: $predicate,
                object: $object,
                object_id: $object_id,
                action: $action,
                observed_at: $observed_at,
                valid_from: $valid_from,
                valid_to: $valid_to,
                state_active: $state_active,
                state_version: $state_version,
                confidence: $confidence,
                source_type: $source_type,
                source_id: $source_id,
                session_id: $session_id,
                metadata_json: $metadata_json,
                created_at: $created_at
            })
            """,
            {**payload, "event_id": event_id},
        )
        return event_id

    async def _upsert_fact_neo4j(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        *,
        action: str,
        confidence: float,
        source_type: str,
        source_id: Optional[str],
        session_id: Optional[str],
        metadata: Optional[dict],
        observed_at: str,
        valid_from: str,
        valid_to: Optional[str],
        policy: str,
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

        if action == "assert":
            closed_count = 0
            if policy == "single_active":
                rows_to_close = await neo4j.query(
                    """
                    MATCH (s:Entity {name: $subject})-[r:RELATES_TO]->(o:Entity)
                    WHERE r.predicate = $predicate
                      AND coalesce(r.is_active, true) = true
                      AND o.name <> $object_name
                    RETURN
                        r.id AS triple_id,
                        r.triple_key AS triple_key,
                        o.name AS object_name,
                        o.id AS object_id,
                        coalesce(r.version, 1) AS version,
                        coalesce(r.confidence, 1.0) AS confidence,
                        r.source_type AS source_type,
                        r.source_id AS source_id,
                        r.session_id AS session_id,
                        r.metadata_json AS metadata_json,
                        r.valid_from AS valid_from
                    """,
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object_name": object_name,
                    },
                )
                for row in rows_to_close:
                    next_version = int(row.get("version") or 1) + 1
                    await neo4j.query(
                        """
                        MATCH (:Entity)-[r:RELATES_TO {id: $triple_id}]->(:Entity)
                        SET r.is_active = false,
                            r.valid_to = $observed_at,
                            r.updated_at = $now,
                            r.version = $next_version,
                            r.last_event_type = 'close_replaced'
                        RETURN r.id AS triple_id
                        """,
                        {
                            "triple_id": row.get("triple_id"),
                            "observed_at": observed_at,
                            "now": now,
                            "next_version": next_version,
                        },
                    )
                    await self._insert_temporal_event_neo4j(
                        {
                            "triple_id": row.get("triple_id"),
                            "triple_key": row.get("triple_key"),
                            "subject": subject,
                            "subject_id": subj_id,
                            "predicate": predicate,
                            "object": row.get("object_name"),
                            "object_id": row.get("object_id"),
                            "action": "close_replaced",
                            "observed_at": observed_at,
                            "valid_from": row.get("valid_from"),
                            "valid_to": observed_at,
                            "state_active": False,
                            "state_version": next_version,
                            "confidence": float(row.get("confidence") or 1.0),
                            "source_type": row.get("source_type"),
                            "source_id": row.get("source_id"),
                            "session_id": row.get("session_id"),
                            "metadata_json": str(row.get("metadata_json") or "{}"),
                            "created_at": now,
                        }
                    )
                    closed_count += 1

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
                    r.created_at = $observed_at,
                    r.updated_at = $now,
                    r.valid_from = $valid_from,
                    r.valid_to = NULL,
                    r.is_active = true,
                    r.version = 1,
                    r.last_event_type = 'assert',
                    r.metadata_json = $metadata_json
                ON MATCH SET
                    r.confidence = $confidence,
                    r.source_type = $source_type,
                    r.source_id = $source_id,
                    r.session_id = $session_id,
                    r.updated_at = $now,
                    r.valid_from = CASE
                        WHEN coalesce(r.is_active, true) THEN coalesce(r.valid_from, $valid_from)
                        ELSE $valid_from
                    END,
                    r.valid_to = NULL,
                    r.is_active = true,
                    r.version = coalesce(r.version, 1) + 1,
                    r.last_event_type = 'assert',
                    r.metadata_json = $metadata_json
                RETURN r.id AS triple_id, coalesce(r.version, 1) AS version, r.valid_from AS valid_from
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
                    "observed_at": observed_at,
                    "valid_from": valid_from,
                    "metadata_json": meta_json,
                    "now": now,
                },
            )
            if rows and rows[0].get("triple_id"):
                triple_id = str(rows[0]["triple_id"])
            version = int(rows[0].get("version") or 1) if rows else 1
            persisted_valid_from = (
                str(rows[0].get("valid_from") or valid_from) if rows else valid_from
            )

            await self._insert_temporal_event_neo4j(
                {
                    "triple_id": triple_id,
                    "triple_key": triple_key,
                    "subject": subject,
                    "subject_id": subj_id,
                    "predicate": predicate,
                    "object": object_name,
                    "object_id": obj_id,
                    "action": "assert",
                    "observed_at": observed_at,
                    "valid_from": persisted_valid_from,
                    "valid_to": None,
                    "state_active": True,
                    "state_version": version,
                    "confidence": confidence,
                    "source_type": source_type,
                    "source_id": source_id,
                    "session_id": session_id,
                    "metadata_json": meta_json,
                    "created_at": now,
                }
            )

            return {
                "ok": True,
                "status": "asserted",
                "policy": policy,
                "closed_count": closed_count,
                "triple_id": triple_id,
                "subject": subject,
                "predicate": predicate,
                "object": object_name,
                "valid_from": persisted_valid_from,
                "valid_to": None,
                "version": version,
            }

        rows = await neo4j.query(
            """
            MATCH (s:Entity {name: $subject})-[r:RELATES_TO {triple_key: $triple_key}]->(o:Entity {name: $object_name})
            RETURN
                r.id AS triple_id,
                r.triple_key AS triple_key,
                s.id AS subject_id,
                o.id AS object_id,
                coalesce(r.version, 1) AS version,
                coalesce(r.is_active, true) AS is_active,
                r.valid_from AS valid_from,
                r.valid_to AS valid_to,
                coalesce(r.confidence, 1.0) AS confidence,
                r.source_type AS source_type,
                r.source_id AS source_id,
                r.session_id AS session_id,
                r.metadata_json AS metadata_json
            """,
            {
                "subject": subject,
                "object_name": object_name,
                "triple_key": triple_key,
            },
        )
        if not rows:
            return {
                "ok": True,
                "status": "not_found",
                "policy": policy,
                "subject": subject,
                "predicate": predicate,
                "object": object_name,
            }

        row = rows[0]
        triple_id = str(row.get("triple_id") or triple_id)
        is_active = bool(row.get("is_active", True))
        current_version = int(row.get("version") or 1)
        close_ts = valid_to or observed_at
        if is_active:
            next_version = current_version + 1
            await neo4j.query(
                """
                MATCH (:Entity)-[r:RELATES_TO {id: $triple_id}]->(:Entity)
                SET r.is_active = false,
                    r.valid_to = $close_ts,
                    r.updated_at = $now,
                    r.version = $next_version,
                    r.last_event_type = 'retract'
                RETURN r.id AS triple_id
                """,
                {
                    "triple_id": triple_id,
                    "close_ts": close_ts,
                    "now": now,
                    "next_version": next_version,
                },
            )
            status = "retracted"
        else:
            next_version = current_version
            status = "already_inactive"

        await self._insert_temporal_event_neo4j(
            {
                "triple_id": triple_id,
                "triple_key": row.get("triple_key") or triple_key,
                "subject": subject,
                "subject_id": row.get("subject_id") or subj_id,
                "predicate": predicate,
                "object": object_name,
                "object_id": row.get("object_id") or obj_id,
                "action": "retract",
                "observed_at": observed_at,
                "valid_from": row.get("valid_from"),
                "valid_to": close_ts if is_active else row.get("valid_to"),
                "state_active": False,
                "state_version": next_version,
                "confidence": float(row.get("confidence") or 1.0),
                "source_type": row.get("source_type"),
                "source_id": row.get("source_id"),
                "session_id": row.get("session_id"),
                "metadata_json": str(row.get("metadata_json") or "{}"),
                "created_at": now,
            }
        )

        return {
            "ok": True,
            "status": status,
            "policy": policy,
            "triple_id": triple_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_name,
            "valid_to": close_ts if is_active else row.get("valid_to"),
            "version": next_version,
        }

    async def _get_triples_as_of_neo4j(
        self,
        *,
        as_of: str,
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
            MATCH (e:KGTemporalEvent)
            WHERE e.observed_at <= $as_of
              AND ($subject IS NULL OR toLower(e.subject) = toLower($subject))
              AND ($predicate IS NULL OR toLower(e.predicate) = toLower($predicate))
              AND ($object_name IS NULL OR toLower(e.object) = toLower($object_name))
              AND ($session_id IS NULL OR e.session_id = $session_id)
            WITH e.triple_key AS triple_key, e
            ORDER BY e.observed_at DESC, e.created_at DESC
            WITH triple_key, collect(e)[0] AS latest
            WHERE coalesce(latest.state_active, true) = true
              AND (latest.valid_from IS NULL OR latest.valid_from <= $as_of)
              AND (latest.valid_to IS NULL OR latest.valid_to > $as_of)
            RETURN
                latest.triple_id AS id,
                latest.subject AS subject,
                latest.predicate AS predicate,
                latest.object AS object,
                latest.confidence AS confidence,
                latest.source_type AS source_type,
                latest.source_id AS source_id,
                latest.session_id AS session_id,
                latest.observed_at AS created_at,
                latest.metadata_json AS metadata_json,
                latest.valid_from AS valid_from,
                latest.valid_to AS valid_to,
                latest.state_active AS is_active,
                latest.state_version AS version,
                latest.action AS last_event_type
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {
                "as_of": as_of,
                "subject": subject,
                "predicate": predicate,
                "object_name": object_name,
                "session_id": session_id,
                "limit": max(1, int(limit)),
            },
        )

        out: list[dict[str, Any]] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            raw_meta = row.get("metadata_json")
            try:
                if isinstance(raw_meta, str) and raw_meta:
                    loaded = json.loads(raw_meta)
                    if isinstance(loaded, dict):
                        metadata = loaded
            except Exception:
                pass
            out.append(
                {
                    "id": row.get("id"),
                    "subject": row.get("subject"),
                    "predicate": row.get("predicate"),
                    "object": row.get("object"),
                    "confidence": row.get("confidence"),
                    "source_type": row.get("source_type"),
                    "source_id": row.get("source_id"),
                    "session_id": row.get("session_id"),
                    "created_at": row.get("created_at"),
                    "metadata": metadata,
                    "valid_from": row.get("valid_from"),
                    "valid_to": row.get("valid_to"),
                    "is_active": bool(row.get("is_active", True)),
                    "version": int(row.get("version") or 1),
                    "last_event_type": row.get("last_event_type"),
                    "as_of": as_of,
                }
            )
        return out

    async def _get_fact_history_neo4j(
        self,
        *,
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
            MATCH (e:KGTemporalEvent)
            WHERE ($subject IS NULL OR toLower(e.subject) = toLower($subject))
              AND ($predicate IS NULL OR toLower(e.predicate) = toLower($predicate))
              AND ($object_name IS NULL OR toLower(e.object) = toLower($object_name))
              AND ($session_id IS NULL OR e.session_id = $session_id)
            RETURN
                e.id AS event_id,
                e.triple_id AS triple_id,
                e.subject AS subject,
                e.predicate AS predicate,
                e.object AS object,
                e.action AS action,
                e.observed_at AS observed_at,
                e.valid_from AS valid_from,
                e.valid_to AS valid_to,
                e.state_active AS state_active,
                e.state_version AS state_version,
                e.confidence AS confidence,
                e.source_type AS source_type,
                e.source_id AS source_id,
                e.session_id AS session_id,
                e.metadata_json AS metadata_json,
                e.created_at AS created_at
            ORDER BY observed_at DESC, created_at DESC
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
            metadata: dict[str, Any] = {}
            raw_meta = row.get("metadata_json")
            try:
                if isinstance(raw_meta, str) and raw_meta:
                    loaded = json.loads(raw_meta)
                    if isinstance(loaded, dict):
                        metadata = loaded
            except Exception:
                pass
            out.append(
                {
                    "event_id": row.get("event_id"),
                    "triple_id": row.get("triple_id"),
                    "subject": row.get("subject"),
                    "predicate": row.get("predicate"),
                    "object": row.get("object"),
                    "action": row.get("action"),
                    "observed_at": row.get("observed_at"),
                    "valid_from": row.get("valid_from"),
                    "valid_to": row.get("valid_to"),
                    "state_active": bool(row.get("state_active", False)),
                    "state_version": int(row.get("state_version") or 1),
                    "confidence": row.get("confidence"),
                    "source_type": row.get("source_type"),
                    "source_id": row.get("source_id"),
                    "session_id": row.get("session_id"),
                    "metadata": metadata,
                    "created_at": row.get("created_at"),
                }
            )
        return out

    def _build_entity_timeline_summary(
        self,
        *,
        entity: str,
        events: list[dict[str, Any]],
        predicate: Optional[str],
        session_id: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        entity_l = str(entity or "").strip().lower()
        timeline: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {}
        latest_by_triple: dict[str, dict[str, Any]] = {}
        first_observed: Optional[str] = None
        last_observed: Optional[str] = None

        for ev in events:
            subj = str(ev.get("subject") or "")
            obj = str(ev.get("object") or "")
            direction = (
                "subject"
                if subj.lower() == entity_l
                else "object"
                if obj.lower() == entity_l
                else "other"
            )
            if direction == "other":
                continue

            action = str(ev.get("action") or "unknown")
            action_counts[action] = int(action_counts.get(action, 0)) + 1

            observed_at = str(ev.get("observed_at") or "")
            if observed_at:
                if first_observed is None or observed_at < first_observed:
                    first_observed = observed_at
                if last_observed is None or observed_at > last_observed:
                    last_observed = observed_at

            triple_id = str(ev.get("triple_id") or "")
            if triple_id and triple_id not in latest_by_triple:
                latest_by_triple[triple_id] = ev

            timeline.append(
                {
                    "event_id": ev.get("event_id"),
                    "triple_id": triple_id,
                    "direction": direction,
                    "subject": subj,
                    "predicate": ev.get("predicate"),
                    "object": obj,
                    "action": action,
                    "observed_at": ev.get("observed_at"),
                    "valid_from": ev.get("valid_from"),
                    "valid_to": ev.get("valid_to"),
                    "state_active": bool(ev.get("state_active", False)),
                    "state_version": int(ev.get("state_version") or 1),
                }
            )

        active_relationships: list[dict[str, Any]] = []
        for triple_id, ev in latest_by_triple.items():
            if not bool(ev.get("state_active", False)):
                continue
            subj = str(ev.get("subject") or "")
            obj = str(ev.get("object") or "")
            direction = "subject" if subj.lower() == entity_l else "object"
            counterpart = obj if direction == "subject" else subj
            active_relationships.append(
                {
                    "triple_id": triple_id,
                    "direction": direction,
                    "counterpart": counterpart,
                    "predicate": ev.get("predicate"),
                    "valid_from": ev.get("valid_from"),
                    "valid_to": ev.get("valid_to"),
                    "state_version": int(ev.get("state_version") or 1),
                    "last_action": ev.get("action"),
                }
            )

        active_relationships.sort(key=lambda it: str(it.get("valid_from") or ""), reverse=True)

        return {
            "entity": entity,
            "predicate_filter": predicate,
            "session_id": session_id,
            "total_events": len(timeline),
            "action_counts": action_counts,
            "first_observed_at": first_observed,
            "last_observed_at": last_observed,
            "active_relationships": active_relationships,
            "timeline": timeline[: max(1, int(limit))],
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
              AND coalesce(r.is_active, true) = true
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
                r.valid_from AS valid_from,
                r.valid_to AS valid_to,
                coalesce(r.is_active, true) AS is_active,
                coalesce(r.version, 1) AS version,
                r.last_event_type AS last_event_type,
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
                    "valid_from": row.get("valid_from"),
                    "valid_to": row.get("valid_to"),
                    "is_active": bool(row.get("is_active", True)),
                    "version": int(row.get("version") or 1),
                    "last_event_type": row.get("last_event_type"),
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
                WHERE coalesce(r.is_active, true) = true
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
                WHERE coalesce(r.is_active, true) = true
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
            WHERE coalesce(r.is_active, true) = true
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
                MATCH (e:KGTemporalEvent)
                WHERE e.session_id = $session_id
                DELETE e
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

    async def _ensure_entities_for_assert(
        self,
        conn: Any,
        subject: str,
        predicate: str,
        object_name: str,
        now: str,
    ) -> tuple[str, str, str]:
        subj_id = _hash_name(subject)
        pred_id = _hash_name(predicate)
        obj_id = _hash_name(object_name)

        await conn.execute(
            """INSERT INTO kg_subjects(id, name, entity_type, created_at, mention_count)
               VALUES (?, ?, ?, ?, 1)
               ON CONFLICT(name) DO UPDATE SET mention_count = kg_subjects.mention_count + 1""",
            (subj_id, subject, self._classify_entity(subject), now),
        )
        await conn.execute(
            """INSERT INTO kg_predicates(id, name, created_at)
               VALUES (?, ?, ?)
               ON CONFLICT(name) DO NOTHING""",
            (pred_id, predicate, now),
        )
        await conn.execute(
            """INSERT INTO kg_objects(id, name, entity_type, created_at, mention_count)
               VALUES (?, ?, ?, ?, 1)
               ON CONFLICT(name) DO UPDATE SET mention_count = kg_objects.mention_count + 1""",
            (obj_id, object_name, self._classify_entity(object_name), now),
        )
        return subj_id, pred_id, obj_id

    async def _resolve_existing_ids(
        self,
        conn: Any,
        subject: str,
        predicate: str,
        object_name: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        cur = await conn.execute("SELECT id FROM kg_subjects WHERE name = ?", (subject,))
        subj_row = await cur.fetchone()
        await cur.close()
        cur = await conn.execute("SELECT id FROM kg_predicates WHERE name = ?", (predicate,))
        pred_row = await cur.fetchone()
        await cur.close()
        cur = await conn.execute("SELECT id FROM kg_objects WHERE name = ?", (object_name,))
        obj_row = await cur.fetchone()
        await cur.close()

        subj_id = str(subj_row["id"]) if subj_row else None
        pred_id = str(pred_row["id"]) if pred_row else None
        obj_id = str(obj_row["id"]) if obj_row else None
        return subj_id, pred_id, obj_id

    async def _insert_temporal_event(
        self,
        conn: Any,
        *,
        triple_id: Optional[str],
        subject_id: str,
        predicate_id: str,
        object_id: str,
        action: str,
        observed_at: str,
        valid_from: Optional[str],
        valid_to: Optional[str],
        state_active: int,
        state_version: int,
        confidence: Optional[float],
        source_type: Optional[str],
        source_id: Optional[str],
        session_id: Optional[str],
        metadata_json: str,
        created_at: str,
    ) -> str:
        event_id = new_id("kgev")
        await conn.execute(
            """
            INSERT INTO kg_triple_events(
                id, triple_id, subject_id, predicate_id, object_id, action,
                observed_at, valid_from, valid_to, state_active, state_version,
                confidence, source_type, source_id, session_id, metadata_json, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id,
                triple_id,
                subject_id,
                predicate_id,
                object_id,
                action,
                observed_at,
                valid_from,
                valid_to,
                int(state_active),
                int(state_version),
                confidence,
                source_type,
                source_id,
                session_id,
                metadata_json,
                created_at,
            ),
        )
        return event_id

    async def upsert_fact(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        *,
        action: str = "assert",
        confidence: float = 1.0,
        source_type: str = "text",
        source_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        observed_at: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> dict[str, Any]:
        act = str(action or "assert").strip().lower()
        if act not in {"assert", "retract"}:
            raise ValueError("action must be 'assert' or 'retract'")

        now = _utc_now()
        observed = _normalize_ts(observed_at, default_now=True) or now
        effective_from = _normalize_ts(valid_from) or observed
        effective_to = _normalize_ts(valid_to)
        if act == "assert" and effective_to is not None and effective_to <= effective_from:
            raise ValueError("valid_to must be greater than valid_from for assert action")

        policy = _temporal_policy(predicate)

        if self._neo4j_backend() is not None:
            return await self._upsert_fact_neo4j(
                subject=subject,
                predicate=predicate,
                object_name=object_name,
                action=act,
                confidence=confidence,
                source_type=source_type,
                source_id=source_id,
                session_id=session_id,
                metadata=metadata,
                observed_at=observed,
                valid_from=effective_from,
                valid_to=effective_to,
                policy=policy,
            )

        meta_json = json.dumps(metadata or {})

        async with db.connect() as conn:
            if act == "assert":
                subj_id, pred_id, obj_id = await self._ensure_entities_for_assert(
                    conn, subject, predicate, object_name, now
                )

                closed_count = 0
                if policy == "single_active":
                    cur = await conn.execute(
                        """
                        SELECT id, object_id, confidence, source_type, source_id, session_id,
                               metadata_json, valid_from, version
                        FROM kg_triples
                        WHERE subject_id = ?
                          AND predicate_id = ?
                          AND object_id <> ?
                          AND COALESCE(is_active, 1) = 1
                        """,
                        (subj_id, pred_id, obj_id),
                    )
                    rows_to_close = await cur.fetchall()
                    await cur.close()
                    for row in rows_to_close:
                        next_version = int(row["version"] or 1) + 1
                        await conn.execute(
                            """
                            UPDATE kg_triples
                            SET is_active = 0,
                                valid_to = ?,
                                updated_at = ?,
                                version = ?,
                                last_event_type = 'close_replaced'
                            WHERE id = ?
                            """,
                            (observed, now, next_version, row["id"]),
                        )
                        await self._insert_temporal_event(
                            conn,
                            triple_id=str(row["id"]),
                            subject_id=subj_id,
                            predicate_id=pred_id,
                            object_id=str(row["object_id"]),
                            action="close_replaced",
                            observed_at=observed,
                            valid_from=row["valid_from"],
                            valid_to=observed,
                            state_active=0,
                            state_version=next_version,
                            confidence=float(row["confidence"] or 1.0),
                            source_type=row["source_type"],
                            source_id=row["source_id"],
                            session_id=row["session_id"],
                            metadata_json=str(row["metadata_json"] or "{}"),
                            created_at=now,
                        )
                        closed_count += 1

                cur = await conn.execute(
                    """
                    SELECT id, created_at, valid_from, valid_to, is_active, version
                    FROM kg_triples
                    WHERE subject_id = ? AND predicate_id = ? AND object_id = ?
                    """,
                    (subj_id, pred_id, obj_id),
                )
                row = await cur.fetchone()
                await cur.close()

                if row:
                    triple_id = str(row["id"])
                    current_version = int(row["version"] or 1)
                    next_version = current_version + 1
                    prev_active = int(row["is_active"] or 0)
                    next_valid_from = row["valid_from"] if prev_active else effective_from
                    await conn.execute(
                        """
                        UPDATE kg_triples
                        SET confidence = ?,
                            source_type = ?,
                            source_id = ?,
                            session_id = ?,
                            metadata_json = ?,
                            updated_at = ?,
                            valid_from = ?,
                            valid_to = NULL,
                            is_active = 1,
                            version = ?,
                            last_event_type = 'assert'
                        WHERE id = ?
                        """,
                        (
                            confidence,
                            source_type,
                            source_id,
                            session_id,
                            meta_json,
                            now,
                            next_valid_from,
                            next_version,
                            triple_id,
                        ),
                    )
                else:
                    triple_id = new_id("triple")
                    next_version = 1
                    await conn.execute(
                        """
                        INSERT INTO kg_triples(
                            id, subject_id, predicate_id, object_id, confidence,
                            source_type, source_id, session_id, created_at, updated_at,
                            valid_from, valid_to, is_active, version, last_event_type, metadata_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            triple_id,
                            subj_id,
                            pred_id,
                            obj_id,
                            confidence,
                            source_type,
                            source_id,
                            session_id,
                            observed,
                            now,
                            effective_from,
                            None,
                            1,
                            next_version,
                            "assert",
                            meta_json,
                        ),
                    )

                await self._insert_temporal_event(
                    conn,
                    triple_id=triple_id,
                    subject_id=subj_id,
                    predicate_id=pred_id,
                    object_id=obj_id,
                    action="assert",
                    observed_at=observed,
                    valid_from=effective_from,
                    valid_to=None,
                    state_active=1,
                    state_version=next_version,
                    confidence=confidence,
                    source_type=source_type,
                    source_id=source_id,
                    session_id=session_id,
                    metadata_json=meta_json,
                    created_at=now,
                )

                await conn.commit()
                return {
                    "ok": True,
                    "status": "asserted",
                    "policy": policy,
                    "closed_count": closed_count,
                    "triple_id": triple_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_name,
                    "valid_from": effective_from,
                    "valid_to": None,
                    "version": next_version,
                }

            subj_id, pred_id, obj_id = await self._resolve_existing_ids(
                conn, subject, predicate, object_name
            )
            if not subj_id or not pred_id or not obj_id:
                return {
                    "ok": True,
                    "status": "not_found",
                    "policy": policy,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_name,
                }

            cur = await conn.execute(
                """
                SELECT id, confidence, source_type, source_id, session_id,
                       metadata_json, valid_from, valid_to, is_active, version
                FROM kg_triples
                WHERE subject_id = ? AND predicate_id = ? AND object_id = ?
                """,
                (subj_id, pred_id, obj_id),
            )
            row = await cur.fetchone()
            await cur.close()
            if not row:
                return {
                    "ok": True,
                    "status": "not_found",
                    "policy": policy,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_name,
                }

            triple_id = str(row["id"])
            current_version = int(row["version"] or 1)
            if int(row["is_active"] or 0) == 1:
                next_version = current_version + 1
                close_ts = effective_to or observed
                await conn.execute(
                    """
                    UPDATE kg_triples
                    SET is_active = 0,
                        valid_to = ?,
                        updated_at = ?,
                        version = ?,
                        last_event_type = 'retract'
                    WHERE id = ?
                    """,
                    (close_ts, now, next_version, triple_id),
                )
                state_active = 0
                state_valid_to = close_ts
                status = "retracted"
            else:
                next_version = current_version
                state_active = 0
                state_valid_to = row["valid_to"]
                status = "already_inactive"

            await self._insert_temporal_event(
                conn,
                triple_id=triple_id,
                subject_id=subj_id,
                predicate_id=pred_id,
                object_id=obj_id,
                action="retract",
                observed_at=observed,
                valid_from=row["valid_from"],
                valid_to=state_valid_to,
                state_active=state_active,
                state_version=next_version,
                confidence=float(row["confidence"] or 1.0),
                source_type=row["source_type"],
                source_id=row["source_id"],
                session_id=row["session_id"],
                metadata_json=str(row["metadata_json"] or "{}"),
                created_at=now,
            )

            await conn.commit()
            return {
                "ok": True,
                "status": status,
                "policy": policy,
                "triple_id": triple_id,
                "subject": subject,
                "predicate": predicate,
                "object": object_name,
                "valid_to": state_valid_to,
                "version": next_version,
            }

    async def get_triples_as_of(
        self,
        as_of: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        at = _normalize_ts(as_of, default_now=True) or _utc_now()
        if self._neo4j_backend() is not None:
            return await self._get_triples_as_of_neo4j(
                as_of=at,
                subject=subject,
                predicate=predicate,
                object_name=object_name,
                session_id=session_id,
                limit=limit,
            )

        query = """
            WITH latest AS (
                SELECT
                    e.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.triple_id
                        ORDER BY e.observed_at DESC, e.created_at DESC
                    ) AS rn
                FROM kg_triple_events e
                JOIN kg_subjects s ON e.subject_id = s.id
                JOIN kg_predicates p ON e.predicate_id = p.id
                JOIN kg_objects o ON e.object_id = o.id
                WHERE e.observed_at <= ?
                  AND (? IS NULL OR LOWER(s.name) = LOWER(?))
                  AND (? IS NULL OR LOWER(p.name) = LOWER(?))
                  AND (? IS NULL OR LOWER(o.name) = LOWER(?))
                  AND (? IS NULL OR e.session_id = ?)
            )
            SELECT
                l.triple_id AS id,
                s.name AS subject,
                p.name AS predicate,
                o.name AS object,
                l.confidence,
                l.source_type,
                l.source_id,
                l.session_id,
                l.observed_at AS created_at,
                l.metadata_json,
                l.valid_from,
                l.valid_to,
                l.state_active AS is_active,
                l.state_version AS version,
                l.action AS last_event_type
            FROM latest l
            JOIN kg_subjects s ON l.subject_id = s.id
            JOIN kg_predicates p ON l.predicate_id = p.id
            JOIN kg_objects o ON l.object_id = o.id
            WHERE l.rn = 1
              AND l.state_active = 1
              AND (l.valid_from IS NULL OR l.valid_from <= ?)
              AND (l.valid_to IS NULL OR l.valid_to > ?)
            ORDER BY l.observed_at DESC
            LIMIT ?
        """

        async with db.connect() as conn:
            cur = await conn.execute(
                query,
                (
                    at,
                    subject,
                    subject,
                    predicate,
                    predicate,
                    object_name,
                    object_name,
                    session_id,
                    session_id,
                    at,
                    at,
                    max(1, int(limit)),
                ),
            )
            rows = await cur.fetchall()
            await cur.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            try:
                raw_meta = row["metadata_json"]
                if raw_meta:
                    loaded = json.loads(raw_meta)
                    if isinstance(loaded, dict):
                        metadata = loaded
            except Exception:
                pass
            out.append(
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
                    "metadata": metadata,
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "is_active": bool(row["is_active"]),
                    "version": int(row["version"] or 1),
                    "last_event_type": row["last_event_type"],
                    "as_of": at,
                }
            )
        return out

    async def get_fact_history(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._neo4j_backend() is not None:
            return await self._get_fact_history_neo4j(
                subject=subject,
                predicate=predicate,
                object_name=object_name,
                session_id=session_id,
                limit=limit,
            )

        query = """
            SELECT
                e.id,
                e.triple_id,
                s.name AS subject,
                p.name AS predicate,
                o.name AS object,
                e.action,
                e.observed_at,
                e.valid_from,
                e.valid_to,
                e.state_active,
                e.state_version,
                e.confidence,
                e.source_type,
                e.source_id,
                e.session_id,
                e.metadata_json,
                e.created_at
            FROM kg_triple_events e
            JOIN kg_subjects s ON e.subject_id = s.id
            JOIN kg_predicates p ON e.predicate_id = p.id
            JOIN kg_objects o ON e.object_id = o.id
            WHERE (? IS NULL OR LOWER(s.name) = LOWER(?))
              AND (? IS NULL OR LOWER(p.name) = LOWER(?))
              AND (? IS NULL OR LOWER(o.name) = LOWER(?))
              AND (? IS NULL OR e.session_id = ?)
            ORDER BY e.observed_at DESC, e.created_at DESC
            LIMIT ?
        """

        async with db.connect() as conn:
            cur = await conn.execute(
                query,
                (
                    subject,
                    subject,
                    predicate,
                    predicate,
                    object_name,
                    object_name,
                    session_id,
                    session_id,
                    max(1, int(limit)),
                ),
            )
            rows = await cur.fetchall()
            await cur.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            try:
                raw_meta = row["metadata_json"]
                if raw_meta:
                    loaded = json.loads(raw_meta)
                    if isinstance(loaded, dict):
                        metadata = loaded
            except Exception:
                pass
            out.append(
                {
                    "event_id": row["id"],
                    "triple_id": row["triple_id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "action": row["action"],
                    "observed_at": row["observed_at"],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "state_active": bool(row["state_active"]),
                    "state_version": int(row["state_version"] or 1),
                    "confidence": row["confidence"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "session_id": row["session_id"],
                    "metadata": metadata,
                    "created_at": row["created_at"],
                }
            )
        return out

    async def get_entity_timeline_summary(
        self,
        entity: str,
        predicate: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        ent = str(entity or "").strip()
        if not ent:
            return {
                "entity": entity,
                "predicate_filter": predicate,
                "session_id": session_id,
                "total_events": 0,
                "action_counts": {},
                "first_observed_at": None,
                "last_observed_at": None,
                "active_relationships": [],
                "timeline": [],
            }

        if self._neo4j_backend() is not None:
            events = await self._get_fact_history_neo4j(
                subject=None,
                predicate=predicate,
                object_name=None,
                session_id=session_id,
                limit=max(int(limit) * 10, int(limit), 100),
            )
            return self._build_entity_timeline_summary(
                entity=ent,
                events=events,
                predicate=predicate,
                session_id=session_id,
                limit=limit,
            )

        query = """
            SELECT
                e.id,
                e.triple_id,
                s.name AS subject,
                p.name AS predicate,
                o.name AS object,
                e.action,
                e.observed_at,
                e.valid_from,
                e.valid_to,
                e.state_active,
                e.state_version,
                e.confidence,
                e.source_type,
                e.source_id,
                e.session_id,
                e.metadata_json,
                e.created_at
            FROM kg_triple_events e
            JOIN kg_subjects s ON e.subject_id = s.id
            JOIN kg_predicates p ON e.predicate_id = p.id
            JOIN kg_objects o ON e.object_id = o.id
            WHERE (LOWER(s.name) = LOWER(?) OR LOWER(o.name) = LOWER(?))
              AND (? IS NULL OR LOWER(p.name) = LOWER(?))
              AND (? IS NULL OR e.session_id = ?)
            ORDER BY e.observed_at DESC, e.created_at DESC
            LIMIT ?
        """
        async with db.connect() as conn:
            cur = await conn.execute(
                query,
                (
                    ent,
                    ent,
                    predicate,
                    predicate,
                    session_id,
                    session_id,
                    max(int(limit) * 10, int(limit), 100),
                ),
            )
            rows = await cur.fetchall()
            await cur.close()

        events: list[dict[str, Any]] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            try:
                raw_meta = row["metadata_json"]
                if raw_meta:
                    loaded = json.loads(raw_meta)
                    if isinstance(loaded, dict):
                        metadata = loaded
            except Exception:
                pass
            events.append(
                {
                    "event_id": row["id"],
                    "triple_id": row["triple_id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "action": row["action"],
                    "observed_at": row["observed_at"],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "state_active": bool(row["state_active"]),
                    "state_version": int(row["state_version"] or 1),
                    "confidence": row["confidence"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "session_id": row["session_id"],
                    "metadata": metadata,
                    "created_at": row["created_at"],
                }
            )
        return self._build_entity_timeline_summary(
            entity=ent,
            events=events,
            predicate=predicate,
            session_id=session_id,
            limit=limit,
        )

    async def find_path_as_of(
        self,
        from_entity: str,
        to_entity: str,
        as_of: Optional[str] = None,
        max_depth: int = 3,
    ) -> Optional[GraphPath]:
        at = _normalize_ts(as_of, default_now=True) or _utc_now()
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

            out_neighbors = await self.get_triples_as_of(
                as_of=at,
                subject=current,
                limit=50,
            )
            for n in out_neighbors:
                nxt = str(n.get("object") or "").lower()
                if not nxt:
                    continue
                queue.append((nxt, nodes + [current], edges + [str(n.get("predicate") or "")]))

            in_neighbors = await self.get_triples_as_of(
                as_of=at,
                object_name=current,
                limit=50,
            )
            for n in in_neighbors:
                nxt = str(n.get("subject") or "").lower()
                if not nxt:
                    continue
                queue.append(
                    (
                        nxt,
                        nodes + [current],
                        edges + [f"<-{str(n.get('predicate') or '')}-"],
                    )
                )

        return None

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
        res = await self.upsert_fact(
            subject=subject,
            predicate=predicate,
            object_name=object_name,
            action="assert",
            confidence=confidence,
            source_type=source_type,
            source_id=source_id,
            session_id=session_id,
            metadata=metadata,
        )
        return {
            "triple_id": res.get("triple_id"),
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
                   t.confidence, t.source_type, t.source_id, t.session_id, t.created_at, t.metadata_json,
                   t.valid_from, t.valid_to, t.is_active, t.version, t.last_event_type
            FROM kg_triples t
            JOIN kg_subjects s ON t.subject_id = s.id
            JOIN kg_predicates p ON t.predicate_id = p.id
            JOIN kg_objects o ON t.object_id = o.id
            WHERE COALESCE(t.is_active, 1) = 1
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
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "is_active": bool(row["is_active"]),
                    "version": int(row["version"] or 1),
                    "last_event_type": row["last_event_type"],
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
                         AND COALESCE(t.is_active, 1) = 1
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
                         AND COALESCE(t.is_active, 1) = 1
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

            cur = await conn.execute(
                "SELECT COUNT(*) as c FROM kg_triples WHERE COALESCE(is_active, 1) = 1"
            )
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
                await conn.execute(
                    "DELETE FROM kg_triple_events WHERE session_id = ?", (session_id,)
                )
            else:
                cur = await conn.execute("DELETE FROM kg_triples")
                deleted = cur.rowcount
                await conn.execute("DELETE FROM kg_triple_events")
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
