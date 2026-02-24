from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .db import db
from .ids import new_id
from .config import settings
from .redact import redact_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    """A single message in a conversation."""

    id: str
    session_id: str
    created_at: str
    role: str  # "user", "assistant", "system"
    content: str
    model: Optional[str] = None
    tokens: Optional[int] = None
    metadata: Optional[dict] = None


class ConversationStore:
    """Store and retrieve conversations."""

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: Optional[str] = None,
        tokens: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Add a message to the conversation history.

        Args:
            session_id: Session ID
            role: "user", "assistant", or "system"
            content: Message content
            model: Optional model name
            tokens: Optional token count
            metadata: Optional additional metadata

        Returns:
            dict with message id and created_at
        """
        now = _utc_now()
        msg_id = new_id("msg")

        # Redact if needed
        safe_content = content
        if bool(getattr(settings, "conversation_redact_secrets", True)):
            safe_content = redact_text(content)

        meta_json = json.dumps(metadata or {})

        async with db.connect() as conn:
            # Ensure session exists (auto-create)
            await conn.execute(
                """INSERT OR IGNORE INTO sessions(id, created_at, role, capabilities_json, llm_settings_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, now, "user", "{}", "{}"),
            )

            await conn.execute(
                """INSERT INTO conversations(id, session_id, created_at, role, content, model, tokens, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, session_id, now, role, safe_content, model, tokens, meta_json),
            )
            await conn.commit()

        return {"id": msg_id, "created_at": now}

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get conversation messages for a session.

        Args:
            session_id: Session ID
            limit: Max messages to return
            offset: Number of messages to skip

        Returns:
            List of message dicts
        """
        async with db.connect() as conn:
            cur = await conn.execute(
                """SELECT id, session_id, created_at, role, content, model, tokens, metadata_json
                   FROM conversations 
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (session_id, limit, offset),
            )
            rows = await cur.fetchall()

        messages = []
        for row in rows:
            meta = {}
            try:
                if row["metadata_json"]:
                    meta = json.loads(row["metadata_json"])
            except Exception:
                pass

            messages.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                    "role": row["role"],
                    "content": row["content"],
                    "model": row["model"],
                    "tokens": row["tokens"],
                    "metadata": meta,
                }
            )

        return messages

    async def get_messages_asc(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get conversation messages in chronological order (oldest first).

        Args:
            session_id: Session ID
            limit: Max messages to return

        Returns:
            List of message dicts (oldest first)
        """
        async with db.connect() as conn:
            cur = await conn.execute(
                """SELECT id, session_id, created_at, role, content, model, tokens, metadata_json
                   FROM conversations 
                   WHERE session_id = ?
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (session_id, limit),
            )
            rows = await cur.fetchall()

        messages = []
        for row in rows:
            meta = {}
            try:
                if row["metadata_json"]:
                    meta = json.loads(row["metadata_json"])
            except Exception:
                pass

            messages.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                    "role": row["role"],
                    "content": row["content"],
                    "model": row["model"],
                    "tokens": row["tokens"],
                    "metadata": meta,
                }
            )

        return messages

    async def search_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search messages by content.

        Args:
            session_id: Session ID
            query: Search query
            limit: Max results

        Returns:
            List of matching messages
        """
        async with db.connect() as conn:
            cur = await conn.execute(
                """SELECT id, session_id, created_at, role, content, model, tokens, metadata_json
                   FROM conversations 
                   WHERE session_id = ? AND content LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, f"%{query}%", limit),
            )
            rows = await cur.fetchall()

        messages = []
        for row in rows:
            meta = {}
            try:
                if row["metadata_json"]:
                    meta = json.loads(row["metadata_json"])
            except Exception:
                pass

            messages.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                    "role": row["role"],
                    "content": row["content"],
                    "model": row["model"],
                    "tokens": row["tokens"],
                    "metadata": meta,
                }
            )

        return messages

    async def get_message_count(self, session_id: str) -> int:
        """Get total message count for a session."""
        async with db.connect() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) as c FROM conversations WHERE session_id = ?", (session_id,)
            )
            row = await cur.fetchone()
        return int(row["c"]) if row else 0

    async def delete_session_messages(self, session_id: str) -> dict[str, Any]:
        """Delete all messages for a session.

        Returns:
            dict with deleted count
        """
        async with db.connect() as conn:
            cur = await conn.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            await conn.commit()
            deleted = cur.rowcount if cur.rowcount else 0

        return {"deleted": deleted}

    async def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        async with db.connect() as conn:
            # Total messages
            cur = await conn.execute("SELECT COUNT(*) as c FROM conversations")
            row = await cur.fetchone()
            total = int(row["c"]) if row else 0

            # Messages by role
            cur = await conn.execute("SELECT role, COUNT(*) as c FROM conversations GROUP BY role")
            by_role = {r["role"]: r["c"] for r in await cur.fetchall()}

            # Total sessions with conversations
            cur = await conn.execute("SELECT COUNT(DISTINCT session_id) as c FROM conversations")
            row = await cur.fetchone()
            sessions = int(row["c"]) if row else 0

        return {
            "total_messages": total,
            "by_role": by_role,
            "sessions_with_conversations": sessions,
        }


_conversation_store: Optional[ConversationStore] = None


def conversation_store() -> ConversationStore:
    """Get or create conversation store singleton."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store
