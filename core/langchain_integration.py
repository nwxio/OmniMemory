"""LangChain integration for memory.

Provides LangChain-compatible memory and retriever implementations.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .memory import memory
from .knowledge_base import knowledge_base


class OmnimindMemory:
    """LangChain-compatible memory implementation.

    Compatible with LangChain's BaseMemory interface.
    """

    def __init__(
        self,
        session_id: str,
        limit: int = 10,
    ):
        """Initialize memory.

        Args:
            session_id: Session ID
            limit: Number of memories to retrieve
        """
        self.session_id = session_id
        self.limit = limit

    def _get_langchain_format(self) -> List[dict]:
        """Get memories in LangChain format (messages)."""
        # This is a simplified version - returns conversation as messages
        # In production, you'd want to format properly for LangChain
        return []

    async def aload_memory_variables(self) -> dict[str, Any]:
        """Load memory variables (async)."""
        # Get conversation messages
        messages = await memory.get_conversation_messages_asc(self.session_id, limit=self.limit)

        # Get lessons (global)
        lessons = await memory.list_lessons(limit=5)

        # Get preferences
        prefs = await memory.list_preferences(scope="session", session_id=self.session_id, limit=10)

        return {
            "messages": messages,
            "lessons": lessons,
            "preferences": prefs,
        }

    def load_memory_variables(self) -> dict[str, Any]:
        """Load memory variables (sync - calls async)."""
        # LangChain expects sync, but we need async
        # This is a limitation - use aload_memory_variables in async contexts
        return {}

    async def astore_context(self, messages: List[Any], actions: List[Any]) -> None:
        """Store context after agent run."""
        # Could extract and store learnings here
        pass

    def store_context(self, messages: List[Any], actions: List[Any]) -> None:
        """Store context (sync)."""
        pass

    async def aclear(self) -> None:
        """Clear memory."""
        # Could clear session-specific data
        pass

    def clear(self) -> None:
        """Clear memory (sync)."""
        pass


class OmnimindRetriever:
    """LangChain-compatible retriever for memory.

    Can be used as a retriever in LangChain chains.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        source: str = "all",  # "lessons", "preferences", "knowledge", "all"
        limit: int = 5,
    ):
        """Initialize retriever.

        Args:
            session_id: Optional session ID filter
            source: Source to search ("lessons", "preferences", "knowledge", "all")
            limit: Number of results
        """
        self.session_id = session_id
        self.source = source
        self.limit = limit

    async def _aget_relevant_documents(self, query: str) -> List[dict]:
        """Get relevant documents (async)."""
        results = []

        if self.source in ("lessons", "all"):
            lessons = await memory.search_lessons(query, limit=self.limit)
            for lesson in lessons:
                results.append(
                    {
                        "page_content": lesson.get("lesson", ""),
                        "metadata": {
                            "type": "lesson",
                            "key": lesson.get("key"),
                            "source": "memory",
                        },
                    }
                )

        if self.source in ("preferences", "all"):
            prefs = await memory.list_preferences(
                scope="session" if self.session_id else "global",
                session_id=self.session_id,
                prefix="",
                limit=self.limit,
            )
            for pref in prefs:
                results.append(
                    {
                        "page_content": f"{pref.get('key')}: {pref.get('value')}",
                        "metadata": {
                            "type": "preference",
                            "key": pref.get("key"),
                            "source": "memory",
                        },
                    }
                )

        if self.source in ("knowledge", "all"):
            kb_docs = await knowledge_base().search_documents(
                query, session_id=self.session_id, limit=self.limit
            )
            for doc in kb_docs:
                results.append(
                    {
                        "page_content": doc.get("snippet", ""),
                        "metadata": {
                            "type": "knowledge",
                            "title": doc.get("title"),
                            "source": "knowledge_base",
                        },
                    }
                )

        return results

    def get_relevant_documents(self, query: str) -> List[dict]:
        """Get relevant documents (sync)."""
        return []

    async def ainvoke(self, query: str) -> List[dict]:
        """Async invoke (LangChain 0.3+ interface)."""
        return await self._aget_relevant_documents(query)

    def invoke(self, query: str) -> List[dict]:
        """Invoke (sync)."""
        return []


def get_memory(session_id: str, limit: int = 10) -> OmnimindMemory:
    """Get LangChain-compatible memory for a session.

    Args:
        session_id: Session ID
        limit: Number of memories to retrieve

    Returns:
        OmnimindMemory instance
    """
    return OmnimindMemory(session_id=session_id, limit=limit)


def get_retriever(
    session_id: Optional[str] = None,
    source: str = "all",
    limit: int = 5,
) -> OmnimindRetriever:
    """Get LangChain-compatible retriever.

    Args:
        session_id: Optional session ID filter
        source: Source to search ("lessons", "preferences", "knowledge", "all")
        limit: Number of results

    Returns:
        OmnimindRetriever instance
    """
    return OmnimindRetriever(
        session_id=session_id,
        source=source,
        limit=limit,
    )
