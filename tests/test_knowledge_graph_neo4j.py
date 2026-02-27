import asyncio
import re

from core.knowledge_graph import KnowledgeGraph


class _FakeNeo4jBackend:
    def __init__(self):
        self.queries = []
        self.clear_calls = 0

    def is_available(self) -> bool:
        return True

    async def query(self, cypher: str, parameters=None):
        params = parameters or {}
        self.queries.append({"cypher": cypher, "params": params})

        if "RETURN r.id AS triple_id" in cypher:
            return [{"triple_id": "triple-neo-1"}]
        if "r.id AS id" in cypher and "MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)" in cypher:
            return [
                {
                    "id": "triple-neo-1",
                    "subject": "Alice",
                    "predicate": "works_for",
                    "object": "Acme",
                    "confidence": 0.9,
                    "source_type": "text",
                    "source_id": None,
                    "session_id": "s-1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "metadata_json": '{"x": 1}',
                }
            ]
        if "MATCH (s:Entity {name: $entity})-[r:RELATES_TO]->(o:Entity)" in cypher:
            return [
                {
                    "from_name": "Alice",
                    "predicate": "works_for",
                    "to_name": "Acme",
                    "confidence": 0.9,
                }
            ]
        if "MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity {name: $entity})" in cypher:
            return [
                {
                    "from_name": "Contoso",
                    "predicate": "owns",
                    "to_name": "Alice",
                    "confidence": 0.8,
                }
            ]
        if "RETURN n.name AS name" in cypher:
            return [{"name": "Alice", "type": "person", "mention_count": 3}]
        if "RETURN count(n) AS entities" in cypher:
            return [{"entities": 2}]
        if "count(r) AS triples" in cypher:
            return [{"triples": 1, "predicates": 1}]
        if "WHERE r.session_id = $session_id" in cypher and "RETURN count(r) AS deleted" in cypher:
            return [{"deleted": 4}]
        if "RETURN count(r) AS deleted" in cypher:
            return [{"deleted": 7}]
        return []

    async def clear(self):
        self.clear_calls += 1


def _run(coro):
    return asyncio.run(coro)


def test_knowledge_graph_neo4j_add_and_read_triples():
    kg = KnowledgeGraph()
    fake = _FakeNeo4jBackend()
    kg._use_neo4j = True
    kg._neo4j = fake

    added = _run(
        kg.add_triple(
            subject="Alice",
            predicate="works_for",
            object_name="Acme",
            confidence=0.9,
            source_type="text",
            session_id="s-1",
            metadata={"x": 1},
        )
    )
    assert added["triple_id"] == "triple-neo-1"

    triples = _run(
        kg.get_triples(
            subject="Alice",
            predicate="works_for",
            object_name="Acme",
            session_id="s-1",
            limit=10,
        )
    )
    assert len(triples) == 1
    assert triples[0]["id"] == "triple-neo-1"
    assert triples[0]["metadata"] == {"x": 1}

    triple_key_params = next(
        q["params"]
        for q in fake.queries
        if isinstance(q.get("params"), dict) and "triple_key" in q["params"]
    )
    assert re.match(r"^[0-9a-f]{16}\|works_for\|[0-9a-f]{16}$", triple_key_params["triple_key"])


def test_knowledge_graph_neo4j_neighbors_search_stats_and_clear():
    kg = KnowledgeGraph()
    fake = _FakeNeo4jBackend()
    kg._use_neo4j = True
    kg._neo4j = fake

    neighbors = _run(kg.get_neighbors("Alice", direction="both", limit=5))
    assert len(neighbors) == 2
    assert neighbors[0]["direction"] == "out"
    assert neighbors[1]["direction"] == "in"

    entities = _run(kg.search_entities("ali", entity_type="person", limit=5))
    assert len(entities) == 1
    assert entities[0]["name"] == "Alice"
    assert entities[0]["type"] == "person"
    assert entities[0]["mention_count"] == 3
    assert entities[0]["role"] == "entity"
    assert float(entities[0].get("search_score", 0.0)) > 0.0

    stats = _run(kg.get_stats())
    assert stats == {"subjects": 2, "predicates": 1, "objects": 2, "triples": 1}

    deleted_session = _run(kg.clear(session_id="s-1"))
    assert deleted_session == {"deleted": 4}
    assert fake.clear_calls == 0

    deleted_all = _run(kg.clear())
    assert deleted_all == {"deleted": 7}
    assert fake.clear_calls == 1
