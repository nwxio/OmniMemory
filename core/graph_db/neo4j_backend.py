"""Neo4j graph database backend for knowledge graph."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class BaseGraphDB(ABC):
    """Abstract base class for graph database backends."""

    @abstractmethod
    async def add_node(
        self,
        node_type: str,
        properties: Dict[str, Any],
    ) -> str:
        """Add a node and return its ID."""
        pass

    @abstractmethod
    async def add_edge(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an edge between two nodes."""
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        pass

    @abstractmethod
    async def query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        pass

    @abstractmethod
    async def search_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search nodes by text."""
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all data."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    dimension: int = 1536


class Neo4jGraphDB(BaseGraphDB):
    """Neo4j graph database backend."""

    def __init__(self, config: Optional[Neo4jConfig] = None):
        self._config = config or self._load_config()
        self._driver = None

    def _load_config(self) -> Neo4jConfig:
        """Load config from environment."""
        return Neo4jConfig(
            uri=os.getenv("OMNIMIND_NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("OMNIMIND_NEO4J_USER", "neo4j"),
            password=os.getenv("OMNIMIND_NEO4J_PASSWORD", ""),
            database=os.getenv("OMNIMIND_NEO4J_DATABASE", "neo4j"),
            dimension=int(os.getenv("OMNIMIND_EMBEDDINGS_DIMENSION", "1536")),
        )

    def _get_driver(self):
        """Lazy-load Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self._config.uri,
                    auth=(self._config.user, self._config.password),
                )
            except ImportError:
                raise RuntimeError("neo4j package not installed: pip install neo4j")
        return self._driver

    @staticmethod
    def _safe_identifier(value: str, kind: str) -> str:
        """Validate Neo4j label/relation identifiers."""
        candidate = (value or "").strip()
        if not candidate or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", candidate):
            raise ValueError(f"invalid {kind}: {value!r}")
        return candidate

    def is_available(self) -> bool:
        """Check if Neo4j is available."""
        try:
            driver = self._get_driver()
            with driver.session(database=self._config.database) as session:
                result = session.run("RETURN 1 AS n")
                result.consume()
            return True
        except Exception:
            return False

    async def add_node(
        self,
        node_type: str,
        properties: Dict[str, Any],
    ) -> str:
        """Add a node to Neo4j."""
        from core.ids import new_id

        node_id = new_id("kg")
        properties = dict(properties)
        properties["id"] = node_id
        properties["created_at"] = datetime.now(timezone.utc).isoformat()
        label = self._safe_identifier(node_type, "node_type")

        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            cypher = f"CREATE (n:{label} {{ {props_str} }}) RETURN n.id AS id"

            props = {k: self._serialize_value(v) for k, v in properties.items()}
            result = session.run(cypher, props)
            record = result.single()
            return record["id"] if record else node_id

    async def add_edge(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an edge between two nodes."""
        from core.ids import new_id

        edge_id = new_id("edge")
        props = {"id": edge_id, "created_at": datetime.now(timezone.utc).isoformat()}
        if properties:
            props.update(properties)
        relation = self._safe_identifier(predicate, "predicate")

        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
            cypher = f"""
            MATCH (s {{id: $subject_id}})
            MATCH (o {{id: $object_id}})
            CREATE (s)-[r:{relation} {{ {props_str} }}]->(o)
            RETURN r.id AS id
            """
            params = {
                "subject_id": subject_id,
                "object_id": object_id,
                **{k: self._serialize_value(v) for k, v in props.items()},
            }
            result = session.run(cypher, params)
            record = result.single()
            return record["id"] if record else edge_id

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            cypher = "MATCH (n) WHERE n.id = $id RETURN n"
            result = session.run(cypher, {"id": node_id})
            record = result.single()
            if record:
                return dict(record["n"])
            return None

    async def query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    async def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            if relation:
                safe_relation = self._safe_identifier(relation, "relation")
                cypher = f"""
                MATCH (n {{id: $node_id}})-[r:{safe_relation}]->(m)
                RETURN n, r, m LIMIT $limit
                """
            else:
                cypher = """
                MATCH (n {id: $node_id})-[r]->(m)
                RETURN n, r, m LIMIT $limit
                """
            result = session.run(cypher, {"node_id": node_id, "limit": limit})
            neighbors = []
            for record in result:
                neighbors.append(
                    {
                        "node": dict(record["m"]),
                        "relation": dict(record["r"]),
                    }
                )
            return neighbors

    async def search_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search nodes by text (case-insensitive contains)."""
        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            if node_type:
                safe_node_type = self._safe_identifier(node_type, "node_type")
                cypher = f"""
                MATCH (n:{safe_node_type})
                WHERE ANY(key IN keys(n) WHERE toString(n[key]) CONTAINS $query)
                RETURN n LIMIT $limit
                """
            else:
                cypher = """
                MATCH (n)
                WHERE ANY(key IN keys(n) WHERE toString(n[key]) CONTAINS $query)
                RETURN n LIMIT $limit
                """
            result = session.run(cypher, {"query": query, "limit": limit})
            return [dict(record["n"]) for record in result]

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            cypher = "MATCH (n {id: $id}) DETACH DELETE n"
            result = session.run(cypher, {"id": node_id})
            return result.consume().counters.nodes_deleted > 0

    async def clear(self) -> None:
        """Clear all data."""
        driver = self._get_driver()
        with driver.session(database=self._config.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for Neo4j."""
        if isinstance(value, (dict, list)):
            import json

            return json.dumps(value)
        return value


def get_neo4j_backend() -> Optional[Neo4jGraphDB]:
    """Get Neo4j backend instance if available."""
    enabled = os.getenv("OMNIMIND_NEO4J_ENABLED", "").lower()
    if enabled not in ("1", "true", "yes"):
        return None

    try:
        backend = Neo4jGraphDB()
        if backend.is_available():
            return backend
    except Exception:
        pass
    return None
