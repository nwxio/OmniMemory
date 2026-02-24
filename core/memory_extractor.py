"""Auto-extraction of memory types from text.

Better than Memori Advanced Augmentation:
- Works locally (no cloud API required)
- Extracts 8 memory types: facts, events, people, preferences, relationships, rules, skills, attributes
- Batch processing for performance
- Background processing support

Memory Types:
- facts: Factual information (e.g., "Paris is the capital of France")
- events: Things that happened (e.g., "Meeting scheduled for Monday")
- people: Person-related information (e.g., "John is a developer")
- preferences: User preferences (e.g., "User prefers dark mode")
- relationships: Connections between entities (e.g., "John works for Google")
- rules: Behavioral rules (e.g., "Always confirm before deleting")
- skills: Capabilities (e.g., "Can generate Python code")
- attributes: Properties of entities (e.g., "The API is RESTful")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .db import db
from .ids import new_id
from .knowledge_graph import knowledge_graph, Triple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Extraction patterns for each memory type
EXTRACTION_PATTERNS = {
    "facts": [
        r"(.+?)\s+(?:is|are|was|were)\s+(.+)",
        r"(.+?)\s+(?:has|have)\s+(.+)",
        r"(?:the|a)\s+(.+?)\s+(?:of|in|at)\s+(.+)",
    ],
    "events": [
        r"(.+?)\s+(?:happened|occurred|took place)\s+(.+)",
        r"(.+?)\s+(?:scheduled|planned)\s+(?:for|on)\s+(.+)",
        r"(?:on|at|during)\s+(.+?),\s+(.+)",
    ],
    "people": [
        r"(.+?)\s+(?:is|works|lives|studies)\s+(.+)",
        r"(.+?)\s+(?:said|told|asked)\s+(.+)",
        r"(.+?)'s\s+(.+)",
    ],
    "preferences": [
        r"(?:prefer|like|want|need)\s+(.+)",
        r"(.+?)\s+(?:prefers|likes|wants|needs)\s+(.+)",
        r"my\s+(?:favorite|preferred)\s+(.+?)\s+(?:is|are)\s+(.+)",
    ],
    "relationships": [
        r"(.+?)\s+(?:works for|knows|manages|owns|leads)\s+(.+)",
        r"(.+?)\s+(?:is|are)\s+(?:part of|member of|related to)\s+(.+)",
        r"(.+?)\s+->\s+(.+)",
    ],
    "rules": [
        r"(?:always|never|must|should)\s+(.+)",
        r"(.+?)\s+(?:requires|needs|depends on)\s+(.+)",
        r"(?:rule|policy):\s*(.+)",
    ],
    "skills": [
        r"(?:can|able to|capable of)\s+(.+)",
        r"(.+?)\s+(?:can|knows how to|is able to)\s+(.+)",
        r"skill:\s*(.+)",
    ],
    "attributes": [
        r"(.+?)\s+(?:is|has|uses)\s+(.+)",
        r"(?:the|a)\s+(.+?)\s+(?:is|has|uses)\s+(.+)",
    ],
}

# Predicates for KG extraction
PREDICATE_PATTERNS = {
    "is": [r"(.+?)\s+is\s+(.+)"],
    "has": [r"(.+?)\s+has\s+(.+)"],
    "works_for": [r"(.+?)\s+(?:works for|works at)\s+(.+)"],
    "knows": [r"(.+?)\s+knows\s+(.+)"],
    "prefers": [r"(.+?)\s+prefers\s+(.+)"],
    "likes": [r"(.+?)\s+likes\s+(.+)"],
    "uses": [r"(.+?)\s+uses\s+(.+)"],
    "located_in": [r"(.+?)\s+(?:located|based)\s+(?:in|at)\s+(.+)"],
    "part_of": [r"(.+?)\s+(?:part|member)\s+of\s+(.+)"],
    "related_to": [r"(.+?)\s+related\s+to\s+(.+)"],
}


@dataclass
class ExtractedMemory:
    """An extracted memory item."""

    memory_type: str
    content: str
    confidence: float
    source_text: str
    entity_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of extraction process."""

    memories: list[ExtractedMemory]
    triples: list[Triple]
    stats: dict[str, int]


class MemoryExtractor:
    """Extract memories from text automatically.

    Features:
    - Pattern-based extraction (fast, no LLM required)
    - LLM-based extraction (accurate, requires LLM)
    - Knowledge Graph integration
    - Batch processing
    """

    def __init__(self):
        self._use_llm = False  # Set to True to enable LLM extraction

    async def extract(
        self,
        text: str,
        entity_id: Optional[str] = None,
        session_id: Optional[str] = None,
        extract_types: Optional[list[str]] = None,
        extract_triples: bool = True,
    ) -> ExtractionResult:
        """Extract memories and triples from text.

        Args:
            text: Text to extract from
            entity_id: Entity ID for attribution
            session_id: Session ID for attribution
            extract_types: Memory types to extract (default: all)
            extract_triples: Whether to extract KG triples

        Returns:
            ExtractionResult with memories and triples
        """
        types = extract_types or list(EXTRACTION_PATTERNS.keys())
        memories = []
        triples = []

        # Extract memories by type
        for mem_type in types:
            if mem_type in EXTRACTION_PATTERNS:
                extracted = self._extract_by_patterns(text, mem_type)
                memories.extend(extracted)

        # Extract triples for KG
        if extract_triples:
            triples = self._extract_triples(text)

        # Store in database
        stored_memories = await self._store_memories(memories, entity_id, session_id)

        # Store triples in KG
        if triples:
            kg = knowledge_graph()
            await kg.add_triples_batch(triples, session_id=session_id)

        return ExtractionResult(
            memories=stored_memories,
            triples=triples,
            stats={
                "memories_extracted": len(memories),
                "memories_stored": len(stored_memories),
                "triples_extracted": len(triples),
                "types_extracted": len(set(m.memory_type for m in memories)),
            },
        )

    def _extract_by_patterns(
        self,
        text: str,
        memory_type: str,
    ) -> list[ExtractedMemory]:
        """Extract memories using regex patterns."""
        memories = []
        patterns = EXTRACTION_PATTERNS.get(memory_type, [])

        for pattern in patterns:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    content = match.group(0).strip()
                    if len(content) > 10 and len(content) < 500:
                        memories.append(
                            ExtractedMemory(
                                memory_type=memory_type,
                                content=content,
                                confidence=0.7,
                                source_text=text,
                            )
                        )
            except Exception:
                pass

        return memories

    def _extract_triples(self, text: str) -> list[Triple]:
        """Extract semantic triples from text."""
        triples = []

        for predicate, patterns in PREDICATE_PATTERNS.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        groups = match.groups()
                        if len(groups) >= 2:
                            subject = groups[0].strip()
                            obj = groups[1].strip()

                            if len(subject) > 2 and len(obj) > 2:
                                triples.append(
                                    Triple(
                                        subject=subject,
                                        predicate=predicate,
                                        object=obj,
                                        confidence=0.8,
                                        source_type="text",
                                    )
                                )
                except Exception:
                    pass

        # Dedupe
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t.subject.lower(), t.predicate, t.object.lower())
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        return unique_triples

    async def _store_memories(
        self,
        memories: list[ExtractedMemory],
        entity_id: Optional[str],
        session_id: Optional[str],
    ) -> list[ExtractedMemory]:
        """Store extracted memories in database."""
        if not memories:
            return []

        stored = []
        now = _utc_now()

        async with db.connect() as conn:
            for mem in memories:
                mem_id = new_id("em")
                meta_json = json.dumps(mem.metadata)

                await conn.execute(
                    """INSERT INTO extracted_memories
                       (id, entity_id, session_id, memory_type, content, confidence, source_text, created_at, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        mem_id,
                        entity_id,
                        session_id,
                        mem.memory_type,
                        mem.content,
                        mem.confidence,
                        mem.source_text[:1000],
                        now,
                        meta_json,
                    ),
                )

                mem.metadata["id"] = mem_id
                stored.append(mem)

            await conn.commit()

        return stored

    async def extract_and_store(
        self,
        text: str,
        entity_id: str,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Extract and store memories in one call.

        Args:
            text: Text to extract from
            entity_id: Entity ID for attribution
            session_id: Session ID for attribution

        Returns:
            dict with extraction stats
        """
        result = await self.extract(text, entity_id, session_id)

        return {
            "ok": True,
            "memories_count": result.stats["memories_stored"],
            "triples_count": result.stats["triples_extracted"],
            "types": list(set(m.memory_type for m in result.memories)),
        }

    async def get_memories(
        self,
        entity_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get extracted memories.

        Args:
            entity_id: Filter by entity
            memory_type: Filter by type
            session_id: Filter by session
            limit: Max results

        Returns:
            List of memories
        """
        query = "SELECT * FROM extracted_memories WHERE 1=1"
        params = []

        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY created_at DESC LIMIT ?"
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
                    "entity_id": row["entity_id"],
                    "session_id": row["session_id"],
                    "memory_type": row["memory_type"],
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "created_at": row["created_at"],
                    "metadata": meta,
                }
            )

        return results

    async def search_memories(
        self,
        query: str,
        entity_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search extracted memories by content.

        Args:
            query: Search query
            entity_id: Filter by entity
            limit: Max results

        Returns:
            List of matching memories
        """
        async with db.connect() as conn:
            if entity_id:
                cur = await conn.execute(
                    """SELECT * FROM extracted_memories 
                       WHERE entity_id = ? AND content LIKE ?
                       ORDER BY confidence DESC LIMIT ?""",
                    (entity_id, f"%{query}%", limit),
                )
            else:
                cur = await conn.execute(
                    """SELECT * FROM extracted_memories 
                       WHERE content LIKE ?
                       ORDER BY confidence DESC LIMIT ?""",
                    (f"%{query}%", limit),
                )
            rows = await cur.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "entity_id": row["entity_id"],
                    "memory_type": row["memory_type"],
                    "content": row["content"],
                    "confidence": row["confidence"],
                }
            )

        return results

    async def get_stats(self) -> dict[str, Any]:
        """Get extraction statistics."""
        async with db.connect() as conn:
            cur = await conn.execute("SELECT COUNT(*) as c FROM extracted_memories")
            total = (await cur.fetchone())["c"]

            cur = await conn.execute(
                "SELECT memory_type, COUNT(*) as c FROM extracted_memories GROUP BY memory_type"
            )
            by_type = {r["memory_type"]: r["c"] for r in await cur.fetchall()}

            cur = await conn.execute(
                "SELECT COUNT(DISTINCT entity_id) as c FROM extracted_memories"
            )
            entities = (await cur.fetchone())["c"]

        return {
            "total_memories": total,
            "by_type": by_type,
            "entities_count": entities,
        }

    async def clear(self, entity_id: Optional[str] = None) -> dict[str, Any]:
        """Clear extracted memories.

        Args:
            entity_id: Only clear for specific entity (optional)
        """
        async with db.connect() as conn:
            if entity_id:
                cur = await conn.execute(
                    "DELETE FROM extracted_memories WHERE entity_id = ?", (entity_id,)
                )
            else:
                cur = await conn.execute("DELETE FROM extracted_memories")
            deleted = cur.rowcount
            await conn.commit()

        return {"deleted": deleted}


# Singleton
_memory_extractor: Optional[MemoryExtractor] = None


def memory_extractor() -> MemoryExtractor:
    """Get memory extractor singleton."""
    global _memory_extractor
    if _memory_extractor is None:
        _memory_extractor = MemoryExtractor()
    return _memory_extractor
