from __future__ import annotations

import asyncio

from core.knowledge_graph import KnowledgeGraph


class _FakeNeo4jTemporalBackend:
    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.relations: dict[str, dict] = {}
        self.events: list[dict] = []

    def is_available(self) -> bool:
        return True

    async def query(self, cypher: str, parameters=None):
        params = parameters or {}

        if "CREATE (e:KGTemporalEvent" in cypher:
            self.events.append(dict(params))
            return []

        if (
            "MATCH (s:Entity {name: $subject})-[r:RELATES_TO]->(o:Entity)" in cypher
            and "o.name <> $object_name" in cypher
        ):
            out = []
            for rel in self.relations.values():
                if (
                    rel.get("subject") == params.get("subject")
                    and rel.get("predicate") == params.get("predicate")
                    and bool(rel.get("is_active", True))
                    and rel.get("object") != params.get("object_name")
                ):
                    out.append(
                        {
                            "triple_id": rel.get("id"),
                            "triple_key": rel.get("triple_key"),
                            "object_name": rel.get("object"),
                            "object_id": rel.get("object_id"),
                            "version": rel.get("version", 1),
                            "confidence": rel.get("confidence", 1.0),
                            "source_type": rel.get("source_type"),
                            "source_id": rel.get("source_id"),
                            "session_id": rel.get("session_id"),
                            "metadata_json": rel.get("metadata_json", "{}"),
                            "valid_from": rel.get("valid_from"),
                        }
                    )
            return out

        if (
            "MATCH (:Entity)-[r:RELATES_TO {id: $triple_id}]->(:Entity)" in cypher
            and "close_replaced" in cypher
        ):
            triple_id = params.get("triple_id")
            for rel in self.relations.values():
                if rel.get("id") == triple_id:
                    rel["is_active"] = False
                    rel["valid_to"] = params.get("observed_at")
                    rel["updated_at"] = params.get("now")
                    rel["version"] = int(params.get("next_version") or rel.get("version", 1))
                    rel["last_event_type"] = "close_replaced"
                    return [{"triple_id": triple_id}]
            return []

        if (
            "MATCH (:Entity)-[r:RELATES_TO {id: $triple_id}]->(:Entity)" in cypher
            and "last_event_type = 'retract'" in cypher
        ):
            triple_id = params.get("triple_id")
            for rel in self.relations.values():
                if rel.get("id") == triple_id:
                    rel["is_active"] = False
                    rel["valid_to"] = params.get("close_ts")
                    rel["updated_at"] = params.get("now")
                    rel["version"] = int(params.get("next_version") or rel.get("version", 1))
                    rel["last_event_type"] = "retract"
                    return [{"triple_id": triple_id}]
            return []

        if (
            "MERGE (s:Entity {name: $subject})" in cypher
            and "MERGE (s)-[r:RELATES_TO {triple_key: $triple_key}]->(o)" in cypher
        ):
            subject = str(params.get("subject"))
            object_name = str(params.get("object_name"))
            self.entities.setdefault(subject, {"id": params.get("subj_id")})
            self.entities.setdefault(object_name, {"id": params.get("obj_id")})

            triple_key = str(params.get("triple_key"))
            rel = self.relations.get(triple_key)
            if rel is None:
                rel = {
                    "id": params.get("triple_id"),
                    "triple_key": triple_key,
                    "subject": subject,
                    "subject_id": params.get("subj_id"),
                    "predicate": params.get("predicate"),
                    "object": object_name,
                    "object_id": params.get("obj_id"),
                    "confidence": params.get("confidence", 1.0),
                    "source_type": params.get("source_type"),
                    "source_id": params.get("source_id"),
                    "session_id": params.get("session_id"),
                    "created_at": params.get("observed_at"),
                    "updated_at": params.get("now"),
                    "valid_from": params.get("valid_from"),
                    "valid_to": None,
                    "is_active": True,
                    "version": 1,
                    "last_event_type": "assert",
                    "metadata_json": params.get("metadata_json", "{}"),
                }
                self.relations[triple_key] = rel
            else:
                previous_active = bool(rel.get("is_active", True))
                rel.update(
                    {
                        "confidence": params.get("confidence", 1.0),
                        "source_type": params.get("source_type"),
                        "source_id": params.get("source_id"),
                        "session_id": params.get("session_id"),
                        "updated_at": params.get("now"),
                        "valid_from": rel.get("valid_from")
                        if previous_active
                        else params.get("valid_from"),
                        "valid_to": None,
                        "is_active": True,
                        "version": int(rel.get("version", 1)) + 1,
                        "last_event_type": "assert",
                        "metadata_json": params.get("metadata_json", "{}"),
                    }
                )
            return [
                {
                    "triple_id": rel.get("id"),
                    "version": rel.get("version", 1),
                    "valid_from": rel.get("valid_from"),
                }
            ]

        if (
            "MATCH (s:Entity {name: $subject})-[r:RELATES_TO {triple_key: $triple_key}]->(o:Entity {name: $object_name})"
            in cypher
        ):
            triple_key = str(params.get("triple_key"))
            rel = self.relations.get(triple_key)
            if rel is None:
                return []
            return [
                {
                    "triple_id": rel.get("id"),
                    "triple_key": rel.get("triple_key"),
                    "subject_id": rel.get("subject_id"),
                    "object_id": rel.get("object_id"),
                    "version": rel.get("version", 1),
                    "is_active": rel.get("is_active", True),
                    "valid_from": rel.get("valid_from"),
                    "valid_to": rel.get("valid_to"),
                    "confidence": rel.get("confidence", 1.0),
                    "source_type": rel.get("source_type"),
                    "source_id": rel.get("source_id"),
                    "session_id": rel.get("session_id"),
                    "metadata_json": rel.get("metadata_json", "{}"),
                }
            ]

        if "MATCH (e:KGTemporalEvent)" in cypher and "WITH e.triple_key AS triple_key" in cypher:
            as_of = str(params.get("as_of") or "")
            subject = (params.get("subject") or "").lower() if params.get("subject") else None
            predicate = (params.get("predicate") or "").lower() if params.get("predicate") else None
            object_name = (
                (params.get("object_name") or "").lower() if params.get("object_name") else None
            )
            session_id = params.get("session_id")
            filtered = []
            for ev in self.events:
                if str(ev.get("observed_at") or "") > as_of:
                    continue
                if subject and str(ev.get("subject") or "").lower() != subject:
                    continue
                if predicate and str(ev.get("predicate") or "").lower() != predicate:
                    continue
                if object_name and str(ev.get("object") or "").lower() != object_name:
                    continue
                if session_id is not None and ev.get("session_id") != session_id:
                    continue
                filtered.append(ev)

            filtered.sort(
                key=lambda it: (str(it.get("observed_at") or ""), str(it.get("created_at") or "")),
                reverse=True,
            )
            latest_by_key: dict[str, dict] = {}
            for ev in filtered:
                key = str(ev.get("triple_key") or "")
                if key and key not in latest_by_key:
                    latest_by_key[key] = ev

            out = []
            for ev in latest_by_key.values():
                if not bool(ev.get("state_active", True)):
                    continue
                valid_from = ev.get("valid_from")
                valid_to = ev.get("valid_to")
                if valid_from and str(valid_from) > as_of:
                    continue
                if valid_to and str(valid_to) <= as_of:
                    continue
                out.append(
                    {
                        "id": ev.get("triple_id"),
                        "subject": ev.get("subject"),
                        "predicate": ev.get("predicate"),
                        "object": ev.get("object"),
                        "confidence": ev.get("confidence"),
                        "source_type": ev.get("source_type"),
                        "source_id": ev.get("source_id"),
                        "session_id": ev.get("session_id"),
                        "created_at": ev.get("observed_at"),
                        "metadata_json": ev.get("metadata_json", "{}"),
                        "valid_from": ev.get("valid_from"),
                        "valid_to": ev.get("valid_to"),
                        "is_active": ev.get("state_active", True),
                        "version": ev.get("state_version", 1),
                        "last_event_type": ev.get("action"),
                    }
                )
            out.sort(key=lambda it: str(it.get("created_at") or ""), reverse=True)
            return out[: int(params.get("limit") or 50)]

        if "MATCH (e:KGTemporalEvent)" in cypher and "RETURN" in cypher and "event_id" in cypher:
            subject = (params.get("subject") or "").lower() if params.get("subject") else None
            predicate = (params.get("predicate") or "").lower() if params.get("predicate") else None
            object_name = (
                (params.get("object_name") or "").lower() if params.get("object_name") else None
            )
            session_id = params.get("session_id")
            out = []
            for ev in self.events:
                if subject and str(ev.get("subject") or "").lower() != subject:
                    continue
                if predicate and str(ev.get("predicate") or "").lower() != predicate:
                    continue
                if object_name and str(ev.get("object") or "").lower() != object_name:
                    continue
                if session_id is not None and ev.get("session_id") != session_id:
                    continue
                out.append(
                    {
                        "event_id": ev.get("event_id"),
                        "triple_id": ev.get("triple_id"),
                        "subject": ev.get("subject"),
                        "predicate": ev.get("predicate"),
                        "object": ev.get("object"),
                        "action": ev.get("action"),
                        "observed_at": ev.get("observed_at"),
                        "valid_from": ev.get("valid_from"),
                        "valid_to": ev.get("valid_to"),
                        "state_active": ev.get("state_active"),
                        "state_version": ev.get("state_version"),
                        "confidence": ev.get("confidence"),
                        "source_type": ev.get("source_type"),
                        "source_id": ev.get("source_id"),
                        "session_id": ev.get("session_id"),
                        "metadata_json": ev.get("metadata_json", "{}"),
                        "created_at": ev.get("created_at"),
                    }
                )
            out.sort(
                key=lambda it: (str(it.get("observed_at") or ""), str(it.get("created_at") or "")),
                reverse=True,
            )
            return out[: int(params.get("limit") or 50)]

        if (
            "MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)" in cypher
            and "RETURN" in cypher
            and "r.id AS id" in cypher
        ):
            out = []
            for rel in self.relations.values():
                if params.get("subject") and rel.get("subject") != params.get("subject"):
                    continue
                if params.get("predicate") and rel.get("predicate") != params.get("predicate"):
                    continue
                if params.get("object_name") and rel.get("object") != params.get("object_name"):
                    continue
                if params.get("session_id") and rel.get("session_id") != params.get("session_id"):
                    continue
                if not bool(rel.get("is_active", True)):
                    continue
                out.append(
                    {
                        "id": rel.get("id"),
                        "subject": rel.get("subject"),
                        "predicate": rel.get("predicate"),
                        "object": rel.get("object"),
                        "confidence": rel.get("confidence", 1.0),
                        "source_type": rel.get("source_type"),
                        "source_id": rel.get("source_id"),
                        "session_id": rel.get("session_id"),
                        "created_at": rel.get("created_at"),
                        "valid_from": rel.get("valid_from"),
                        "valid_to": rel.get("valid_to"),
                        "is_active": rel.get("is_active", True),
                        "version": rel.get("version", 1),
                        "last_event_type": rel.get("last_event_type"),
                        "metadata_json": rel.get("metadata_json", "{}"),
                    }
                )
            out.sort(key=lambda it: str(it.get("created_at") or ""), reverse=True)
            return out[: int(params.get("limit") or 50)]

        return []

    async def clear(self):
        self.entities.clear()
        self.relations.clear()
        self.events.clear()


def _run(coro):
    return asyncio.run(coro)


def test_neo4j_temporal_upsert_as_of_history_and_summary() -> None:
    kg = KnowledgeGraph()
    fake = _FakeNeo4jTemporalBackend()
    kg._use_neo4j = True
    kg._neo4j = fake

    _run(
        kg.upsert_fact(
            subject="Alice",
            predicate="works_for",
            object_name="Acme",
            action="assert",
            observed_at="2026-01-01T10:00:00+00:00",
        )
    )
    _run(
        kg.upsert_fact(
            subject="Alice",
            predicate="works_for",
            object_name="Contoso",
            action="assert",
            observed_at="2026-01-02T10:00:00+00:00",
        )
    )

    old = _run(
        kg.get_triples_as_of(
            as_of="2026-01-01T12:00:00+00:00",
            subject="alice",
            predicate="works_for",
            limit=10,
        )
    )
    assert len(old) == 1
    assert old[0]["object"] == "Acme"

    now = _run(
        kg.get_triples_as_of(
            as_of="2026-01-03T12:00:00+00:00",
            subject="Alice",
            predicate="works_for",
            limit=10,
        )
    )
    assert len(now) == 1
    assert now[0]["object"] == "Contoso"

    hist = _run(kg.get_fact_history(subject="Alice", predicate="works_for", limit=20))
    actions = {str(it.get("action")) for it in hist}
    assert "assert" in actions
    assert "close_replaced" in actions

    summary = _run(kg.get_entity_timeline_summary(entity="Alice", predicate="works_for", limit=20))
    assert int(summary.get("total_events") or 0) >= 3
    assert int((summary.get("action_counts") or {}).get("assert", 0)) >= 2
