"""Knowledge Base - document storage and retrieval.

Provides storage for parsed documents that can be used for RAG.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from .db import db
from .ids import new_id
from .doc_parser import document_parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeBase:
    """Knowledge base for document storage."""

    async def add_document(
        self,
        title: str,
        content: str,
        source_type: str,
        source_url: Optional[str] = None,
        source_path: Optional[str] = None,
        format: str = "markdown",
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Add a document to knowledge base.

        Args:
            title: Document title
            content: Document content
            source_type: "file", "url", "text"
            source_url: Source URL (for url type)
            source_path: Source file path (for file type)
            format: Format type (markdown, text, html, pdf, docx)
            session_id: Optional session ID
            metadata: Additional metadata

        Returns:
            dict with document id
        """
        now = _utc_now()
        doc_id = new_id("kb")

        meta_json = json.dumps(metadata or {})

        async with db.connect() as conn:
            await conn.execute(
                """INSERT INTO knowledge_base(id, session_id, title, content, source_type, source_url, source_path, format, created_at, metadata_json, indexed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    session_id,
                    title,
                    content,
                    source_type,
                    source_url,
                    source_path,
                    format,
                    now,
                    meta_json,
                    0,
                ),
            )
            await conn.commit()

        return {"id": doc_id, "created_at": now}

    async def add_document_from_file(
        self,
        file_path: str,
        source_type: str = "file",
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a document by parsing a file.

        Args:
            file_path: Path to the file
            source_type: Source type (file, url)
            session_id: Optional session ID

        Returns:
            dict with document id and parsed metadata
        """
        parser = document_parser()
        parsed = await parser.parse_file(file_path)

        source_path = file_path if source_type == "file" else None
        source_url = None

        return await self.add_document(
            title=parsed.title,
            content=parsed.content,
            source_type=source_type,
            source_url=source_url,
            source_path=source_path,
            format=parsed.metadata.get("format", "text"),
            session_id=session_id,
            metadata=parsed.metadata,
        )

    async def add_document_from_url(
        self,
        url: str,
        content: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a document from URL.

        Args:
            url: Source URL
            content: Optional pre-fetched content (will fetch if not provided)
            session_id: Optional session ID

        Returns:
            dict with document id
        """
        if content is None:
            # Try to fetch
            try:
                import httpx

                response = httpx.get(url, timeout=30)
                content = response.text
                content_type = response.headers.get("content-type", "")

                # Parse based on content type
                parser = document_parser()
                if "html" in content_type:
                    parsed = await parser.parse_content(content, "html")
                else:
                    parsed = await parser.parse_content(content, "markdown")

                return await self.add_document(
                    title=parsed.title,
                    content=parsed.content,
                    source_type="url",
                    source_url=url,
                    format=parsed.metadata.get("format", "text"),
                    session_id=session_id,
                    metadata=parsed.metadata,
                )
            except Exception as e:
                return {"ok": False, "error": f"Failed to fetch URL: {str(e)}"}

        # Use provided content
        parser = document_parser()
        parsed = await parser.parse_content(content)

        return await self.add_document(
            title=parsed.title,
            content=parsed.content,
            source_type="url",
            source_url=url,
            format=parsed.metadata.get("format", "text"),
            session_id=session_id,
            metadata=parsed.metadata,
        )

    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Get a document by ID."""
        async with db.connect() as conn:
            cur = await conn.execute("SELECT * FROM knowledge_base WHERE id = ?", (doc_id,))
            row = await cur.fetchone()

        if not row:
            return None

        meta = {}
        try:
            if row["metadata_json"]:
                meta = json.loads(row["metadata_json"])
        except Exception:
            pass

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "title": row["title"],
            "content": row["content"],
            "source_type": row["source_type"],
            "source_url": row["source_url"],
            "source_path": row["source_path"],
            "format": row["format"],
            "created_at": row["created_at"],
            "metadata": meta,
            "indexed": bool(row["indexed"]),
        }

    async def list_documents(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List documents.

        Args:
            session_id: Optional session filter
            limit: Max results
            offset: Offset

        Returns:
            List of documents
        """
        if session_id:
            async with db.connect() as conn:
                cur = await conn.execute(
                    """SELECT id, title, source_type, source_url, source_path, format, created_at, indexed
                       FROM knowledge_base WHERE session_id = ?
                       ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                    (session_id, limit, offset),
                )
                rows = await cur.fetchall()
        else:
            async with db.connect() as conn:
                cur = await conn.execute(
                    """SELECT id, title, source_type, source_url, source_path, format, created_at, indexed
                       FROM knowledge_base
                       ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                    (limit, offset),
                )
                rows = await cur.fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source_type": row["source_type"],
                "source_url": row["source_url"],
                "source_path": row["source_path"],
                "format": row["format"],
                "created_at": row["created_at"],
                "indexed": bool(row["indexed"]),
            }
            for row in rows
        ]

    async def search_documents(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents by content.

        Args:
            query: Search query
            session_id: Optional session filter
            limit: Max results

        Returns:
            List of matching documents
        """
        if session_id:
            async with db.connect() as conn:
                cur = await conn.execute(
                    """SELECT id, title, source_type, source_url, content
                       FROM knowledge_base 
                       WHERE session_id = ? AND content LIKE ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, f"%{query}%", limit),
                )
                rows = await cur.fetchall()
        else:
            async with db.connect() as conn:
                cur = await conn.execute(
                    """SELECT id, title, source_type, source_url, content
                       FROM knowledge_base 
                       WHERE content LIKE ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (f"%{query}%", limit),
                )
                rows = await cur.fetchall()

        results = []
        for row in rows:
            # Find snippet
            content = row["content"]
            idx = content.lower().find(query.lower())
            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 50)
                snippet = content[start:end]
            else:
                snippet = content[:100]

            results.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "source_type": row["source_type"],
                    "source_url": row["source_url"],
                    "snippet": snippet,
                }
            )

        return results

    async def delete_document(self, doc_id: str) -> dict[str, Any]:
        """Delete a document."""
        async with db.connect() as conn:
            cur = await conn.execute("DELETE FROM knowledge_base WHERE id = ?", (doc_id,))
            await conn.commit()
            deleted = cur.rowcount if cur.rowcount else 0

        return {"deleted": deleted}

    async def mark_indexed(self, doc_id: str) -> None:
        """Mark document as indexed."""
        async with db.connect() as conn:
            await conn.execute("UPDATE knowledge_base SET indexed = 1 WHERE id = ?", (doc_id,))
            await conn.commit()

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        async with db.connect() as conn:
            # Total
            cur = await conn.execute("SELECT COUNT(*) as c FROM knowledge_base")
            row = await cur.fetchone()
            total = int(row["c"]) if row else 0

            # By source type
            cur = await conn.execute(
                "SELECT source_type, COUNT(*) as c FROM knowledge_base GROUP BY source_type"
            )
            by_type = {r["source_type"]: r["c"] for r in await cur.fetchall()}

            # Indexed
            cur = await conn.execute("SELECT COUNT(*) as c FROM knowledge_base WHERE indexed = 1")
            row = await cur.fetchone()
            indexed = int(row["c"]) if row else 0

            # Sessions
            cur = await conn.execute(
                "SELECT COUNT(DISTINCT session_id) as c FROM knowledge_base WHERE session_id IS NOT NULL"
            )
            row = await cur.fetchone()
            sessions = int(row["c"]) if row else 0

        return {
            "total_documents": total,
            "by_source_type": by_type,
            "indexed": indexed,
            "sessions_with_docs": sessions,
        }


_knowledge_base: Optional[KnowledgeBase] = None


def knowledge_base() -> KnowledgeBase:
    """Get knowledge base singleton."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base
