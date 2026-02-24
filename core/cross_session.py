"""Cross-session memory management.

Provides persistent memory across sessions - agents can recall context,
decisions, and learnings from previous sessions automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from .db import db
from .config import settings
from .redact import redact_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> float:
    return datetime.now(timezone.utc).timestamp()


SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_COMPLETED = "completed"
SESSION_STATUS_ENDED = "ended"

SESSION_TIMEOUT_MINUTES = 20
SESSION_END_MARKERS = [
    "сессия закончена",
    "сессия окончена",
]


@dataclass
class ContextBundle:
    """Token-budgeted context from previous sessions."""

    content: str
    tokens: int
    entries_count: int
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FinalizationReport:
    """Report after session finalization."""

    session_id: str
    entries_stored: int
    observations_count: int
    summary: str


class CrossSessionManager:
    """Manages cross-session memory lifecycle.

    Flow:
    1. start_session() - begins new session, injects context from previous
    2. record_message() / record_tool_use() - capture events
    3. stop_session() - extract observations, generate summary
    4. end_session() - cleanup, mark as ended
    """

    def __init__(self):
        self._max_context_tokens = int(getattr(settings, "cross_session_max_context_tokens", 2000))
        self._max_context_chars = self._max_context_tokens * 4  # rough estimate

    async def start_session(
        self,
        session_id: str,
        user_prompt: str = "",
    ) -> dict[str, Any]:
        """Start a new session with context injection from previous sessions.

        Args:
            session_id: The session ID to start
            user_prompt: Optional user prompt for context-aware retrieval

        Returns:
            dict with memory_session_id, context (injected context), tokens
        """
        now = _utc_now()

        # Create session if not exists - need all required fields
        async with db.connect() as conn:
            await conn.execute(
                """INSERT OR IGNORE INTO sessions(id, created_at, role, capabilities_json, llm_settings_json, session_status, started_at, user_prompt)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    now,
                    "user",  # default role
                    "{}",  # capabilities_json - required NOT NULL
                    "{}",  # llm_settings_json
                    SESSION_STATUS_ACTIVE,
                    now,
                    user_prompt[:500] if user_prompt else "",
                ),
            )
            # Then update
            await conn.execute(
                """UPDATE sessions SET session_status = ?, started_at = ?, user_prompt = ?
                   WHERE id = ?""",
                (SESSION_STATUS_ACTIVE, now, user_prompt[:500] if user_prompt else "", session_id),
            )
            await conn.commit()

        # Get context from previous sessions
        context_bundle = await self.get_context_for_prompt(user_prompt)

        return {
            "session_id": session_id,
            "memory_session_id": f"mem_{session_id}",
            "context": context_bundle.content,
            "tokens": context_bundle.tokens,
            "entries_count": context_bundle.entries_count,
            "sources": context_bundle.sources,
        }

    def _detect_end_marker(self, content: str) -> bool:
        """Check if content contains session end marker."""
        content_lower = content.lower()
        return any(marker in content_lower for marker in SESSION_END_MARKERS)

    async def _update_last_activity(self, session_id: str) -> None:
        """Update last_activity timestamp for session."""
        now = _utc_now()
        async with db.connect() as conn:
            await conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE id = ?",
                (now, session_id),
            )
            await conn.commit()

    async def check_session_timeout(self, session_id: str) -> bool:
        """Check if session has exceeded inactivity timeout.

        Returns:
            True if session should be finalized due to timeout
        """
        async with db.connect() as conn:
            cursor = await conn.execute(
                """SELECT last_activity, session_status FROM sessions WHERE id = ?""",
                (session_id,),
            )
            row = await cursor.fetchone()

        if not row:
            return False

        status = row["session_status"] if hasattr(row, "__getitem__") else row[1]
        if status != SESSION_STATUS_ACTIVE:
            return False

        last_activity = row["last_activity"] if hasattr(row, "__getitem__") else row[0]
        if not last_activity:
            return False

        try:
            last_dt = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            elapsed_minutes = (now_dt - last_dt).total_seconds() / 60
            return elapsed_minutes >= SESSION_TIMEOUT_MINUTES
        except (ValueError, TypeError):
            return False

    async def record_message(
        self,
        session_id: str,
        content: str,
        role: str = "user",
    ) -> dict[str, Any]:
        """Record a chat message event.

        Args:
            session_id: Session ID
            content: Message content
            role: 'user' or 'assistant'
        """
        now = _utc_now()

        safe_content = content
        if bool(getattr(settings, "cross_session_redact_secrets", True)):
            safe_content = redact_text(content)

        should_finalize = self._detect_end_marker(safe_content)

        async with db.connect() as conn:
            await conn.execute(
                """INSERT INTO episodes(session_id, title, summary, tags_json, data_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    f"Message: {role}",
                    safe_content[:1000],
                    json.dumps(["cross_session", "message", role]),
                    json.dumps({"role": role, "content": safe_content}),
                    now,
                ),
            )
            await conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE id = ?",
                (now, session_id),
            )
            await conn.commit()

        result = {"ok": True, "session_id": session_id, "finalized": False}

        if should_finalize:
            await self.stop_session(session_id)
            result["finalized"] = True
            result["reason"] = "end_marker_detected"

        return result

    async def record_tool_use(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
    ) -> dict[str, Any]:
        """Record a tool invocation event.

        Args:
            session_id: Session ID
            tool_name: Name of the tool
            tool_input: Tool input (will be redacted)
            tool_output: Tool output (will be redacted)
        """
        now = _utc_now()

        # Truncate for storage
        input_str = str(tool_input)[:2000] if tool_input else ""
        output_str = str(tool_output)[:2000] if tool_output else ""

        # Redact
        if bool(getattr(settings, "cross_session_redact_secrets", True)):
            input_str = redact_text(input_str)
            output_str = redact_text(output_str)

        async with db.connect() as conn:
            await conn.execute(
                """INSERT INTO episodes(session_id, title, summary, tags_json, data_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    f"Tool: {tool_name}",
                    f"{tool_name}: {input_str[:200]}...",
                    json.dumps(["cross_session", "tool_use", tool_name]),
                    json.dumps({"tool_name": tool_name, "input": input_str, "output": output_str}),
                    now,
                ),
            )
            await conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE id = ?",
                (now, session_id),
            )
            await conn.commit()

        return {"ok": True, "session_id": session_id}

    async def stop_session(self, session_id: str) -> FinalizationReport:
        """Finalize session: extract observations and generate summary.

        Args:
            session_id: Session to finalize

        Returns:
            FinalizationReport with stats
        """
        now = _utc_now()

        # Get all episodes for this session
        async with db.connect() as conn:
            cursor = await conn.execute(
                "SELECT id, title, summary, data_json FROM episodes WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            )
            rows = await cursor.fetchall()

        # Extract observations (simple heuristic: extract key info)
        observations = []
        for row in rows:
            data = json.loads(row["data_json"]) if row["data_json"] else {}

            if row["title"].startswith("Message:"):
                role = data.get("role", "user")
                content = data.get("content", "")[:500]
                if role == "user" and len(content) > 50:
                    observations.append(f"User mentioned: {content[:200]}")
            elif row["title"].startswith("Tool:"):
                tool = data.get("tool_name", "unknown")
                observations.append(f"Used tool: {tool}")

        observations_json = json.dumps(observations[:20])  # Limit

        # Generate summary (from episode titles)
        summary_parts = [row["title"] for row in rows[:10]]
        summary = " | ".join(summary_parts)

        # Update session - mark as completed AND ended
        now = _utc_now()
        async with db.connect() as conn:
            await conn.execute(
                """UPDATE sessions SET session_status = ?, summary = ?, observations_json = ?, ended_at = ?
                   WHERE id = ?""",
                (SESSION_STATUS_COMPLETED, summary[:2000], observations_json, now, session_id),
            )
            await conn.commit()

        return FinalizationReport(
            session_id=session_id,
            entries_stored=len(rows),
            observations_count=len(observations),
            summary=summary[:500],
        )

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End session and cleanup.

        Args:
            session_id: Session to end
        """
        now = _utc_now()

        async with db.connect() as conn:
            await conn.execute(
                """UPDATE sessions SET session_status = ?, ended_at = ? 
                   WHERE id = ?""",
                (SESSION_STATUS_ENDED, now, session_id),
            )
            await conn.commit()

        return {"ok": True, "session_id": session_id}

    async def get_context_for_prompt(
        self,
        user_prompt: str = "",
        max_tokens: Optional[int] = None,
    ) -> ContextBundle:
        """Build token-budgeted context from previous sessions.

        Args:
            user_prompt: Current user prompt for relevance scoring
            max_tokens: Max tokens (default from settings)

        Returns:
            ContextBundle with context text
        """
        max_tokens = max_tokens or self._max_context_tokens

        # Get recent sessions (excluding current)
        async with db.connect() as conn:
            cursor = await conn.execute(
                """SELECT id, title, summary, observations_json, ended_at, started_at
                   FROM sessions 
                   WHERE session_status IN (?, ?)
                     AND ended_at IS NOT NULL
                   ORDER BY ended_at DESC
                   LIMIT 20""",
                (SESSION_STATUS_COMPLETED, SESSION_STATUS_ENDED),
            )
            rows = await cursor.fetchall()

        if not rows:
            return ContextBundle(content="", tokens=0, entries_count=0)

        # Build context from sessions
        context_parts = []
        total_chars = 0
        entries_count = 0
        sources = []

        for row in rows:
            # Check token budget
            if total_chars >= self._max_context_chars:
                break

            session_parts = []

            # Add summary
            if row["summary"]:
                session_parts.append(f"Session {row['id'][:8]}: {row['summary']}")

            # Add observations
            if row["observations_json"]:
                try:
                    obs = json.loads(row["observations_json"])
                    if obs:
                        session_parts.append(f"Observations: {'; '.join(obs[:5])}")
                except Exception:
                    pass

            if session_parts:
                part = "\n".join(session_parts)
                if total_chars + len(part) <= self._max_context_chars:
                    context_parts.append(part)
                    total_chars += len(part)
                    entries_count += 1
                    sources.append({"session_id": row["id"], "type": "session_summary"})

        content = "\n\n".join(context_parts)

        # Estimate tokens
        tokens = total_chars // 4

        return ContextBundle(
            content=content,
            tokens=tokens,
            entries_count=entries_count,
            sources=sources,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search across all session memories.

        Uses simple text search in summaries and observations.
        """
        async with db.connect() as conn:
            # Search in summaries and observations
            cursor = await conn.execute(
                """SELECT id, title, summary, observations_json, ended_at, created_at
                   FROM sessions 
                   WHERE session_status IN (?, ?)
                     AND (summary LIKE ? OR observations_json LIKE ?)
                   ORDER BY ended_at DESC
                   LIMIT ?""",
                (SESSION_STATUS_COMPLETED, SESSION_STATUS_ENDED, f"%{query}%", f"%{query}%", limit),
            )
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "session_id": row["id"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "observations": json.loads(row["observations_json"])
                    if row["observations_json"]
                    else [],
                    "ended_at": row["ended_at"],
                    "created_at": row["created_at"],
                }
            )

        return results

    async def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        async with db.connect() as conn:
            # Total sessions
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE session_status IN (?, ?)",
                (SESSION_STATUS_COMPLETED, SESSION_STATUS_ENDED),
            )
            row = await cursor.fetchone()
            total_sessions = row["cnt"] if row else 0

            # Sessions with observations
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE observations_json != '[]' AND observations_json IS NOT NULL"
            )
            row = await cursor.fetchone()
            sessions_with_obs = row["cnt"] if row else 0

            # Total episodes in cross-session
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM episodes WHERE session_id IN (SELECT id FROM sessions WHERE session_status IN (?, ?))",
                (SESSION_STATUS_COMPLETED, SESSION_STATUS_ENDED),
            )
            row = await cursor.fetchone()
            total_episodes = row["cnt"] if row else 0

        return {
            "total_sessions": total_sessions,
            "sessions_with_observations": sessions_with_obs,
            "total_episodes": total_episodes,
        }


_cross_session_manager: Optional[CrossSessionManager] = None


def cross_session_manager() -> CrossSessionManager:
    """Get or create cross-session manager singleton."""
    global _cross_session_manager
    if _cross_session_manager is None:
        _cross_session_manager = CrossSessionManager()
    return _cross_session_manager
