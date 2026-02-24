import asyncio

import pytest

from core.graph_db.neo4j_backend import Neo4jConfig, Neo4jGraphDB


class _FakeCounters:
    def __init__(self, nodes_deleted: int = 0):
        self.nodes_deleted = nodes_deleted


class _FakeSummary:
    def __init__(self, nodes_deleted: int = 0):
        self.counters = _FakeCounters(nodes_deleted=nodes_deleted)


class _FakeResult:
    def __init__(self, records=None, single_record=None, nodes_deleted: int = 0):
        self._records = records or []
        self._single_record = single_record
        self._nodes_deleted = nodes_deleted

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._single_record

    def consume(self):
        return _FakeSummary(nodes_deleted=self._nodes_deleted)


class _FakeSession:
    def __init__(self, run_func):
        self._run_func = run_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher, params=None):
        return self._run_func(cypher, params or {})


class _FakeDriver:
    def __init__(self, run_handler):
        self.calls = []
        self._run_handler = run_handler

    def session(self, database=None):
        def _run(cypher, params):
            self.calls.append({"database": database, "cypher": cypher, "params": params})
            return self._run_handler(cypher, params)

        return _FakeSession(_run)


def _run(coro):
    return asyncio.run(coro)


def test_add_node_builds_valid_cypher():
    def _handler(cypher, params):
        assert "CREATE (n:Entity" in cypher
        assert "name: $name" in cypher
        assert params["name"] == "Alice"
        return _FakeResult(single_record={"id": "kg-node-1"})

    backend = Neo4jGraphDB(Neo4jConfig(uri="bolt://fake", user="neo4j", password="pw"))
    backend._driver = _FakeDriver(_handler)

    node_id = _run(backend.add_node("Entity", {"name": "Alice"}))
    assert node_id == "kg-node-1"


def test_add_edge_builds_valid_cypher_and_params():
    def _handler(cypher, params):
        assert "CREATE (s)-[r:WORKS_FOR" in cypher
        assert "id: $id" in cypher
        assert "created_at: $created_at" in cypher
        assert "weight: $weight" in cypher
        assert params["subject_id"] == "s-1"
        assert params["object_id"] == "o-1"
        assert params["weight"] == 3
        return _FakeResult(single_record={"id": "edge-1"})

    backend = Neo4jGraphDB(Neo4jConfig(uri="bolt://fake", user="neo4j", password="pw"))
    backend._driver = _FakeDriver(_handler)

    edge_id = _run(backend.add_edge("s-1", "WORKS_FOR", "o-1", {"weight": 3}))
    assert edge_id == "edge-1"


def test_get_neighbors_with_relation_uses_safe_query():
    records = [
        {
            "m": {"id": "n-2", "name": "Bob"},
            "r": {"id": "edge-2", "predicate": "KNOWS"},
        }
    ]

    def _handler(cypher, params):
        assert "MATCH (n {id: $node_id})-[r:KNOWS]->(m)" in cypher
        assert params["node_id"] == "n-1"
        assert params["limit"] == 5
        return _FakeResult(records=records)

    backend = Neo4jGraphDB(Neo4jConfig(uri="bolt://fake", user="neo4j", password="pw"))
    backend._driver = _FakeDriver(_handler)

    out = _run(backend.get_neighbors("n-1", relation="KNOWS", limit=5))
    assert out == [
        {
            "node": {"id": "n-2", "name": "Bob"},
            "relation": {"id": "edge-2", "predicate": "KNOWS"},
        }
    ]


def test_invalid_identifiers_are_rejected():
    backend = Neo4jGraphDB(Neo4jConfig(uri="bolt://fake", user="neo4j", password="pw"))
    backend._driver = _FakeDriver(lambda _cypher, _params: _FakeResult())

    with pytest.raises(ValueError):
        _run(backend.search_nodes("alice", node_type="bad-type", limit=5))

    with pytest.raises(ValueError):
        _run(backend.get_neighbors("n-1", relation="DROP TABLE", limit=5))
