import aiosqlite
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

from ..llm_config import llm_settings


def _hash_content(prompt: str, system: str = "") -> str:
    """Generate hash for cache key."""
    content = f"{system}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


class LLMCache:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or llm_settings.cache_path
        self._db: Optional[aiosqlite.Connection] = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._init_db()
        return self._db

    async def _init_db(self) -> None:
        db = self._db
        await db.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash TEXT PRIMARY KEY,
                system_hash TEXT,
                response TEXT,
                model TEXT,
                created_at TEXT,
                expires_at TEXT
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON llm_cache(expires_at)
        """)
        await db.commit()

    async def get(self, prompt: str, system: str = "") -> Optional[str]:
        """Get cached response if exists and not expired."""
        if not llm_settings.cache_enabled:
            return None

        prompt_hash = _hash_content(prompt, system)
        db = await self._get_db()

        async with db.execute(
            "SELECT response, expires_at FROM llm_cache WHERE prompt_hash = ?",
            (prompt_hash,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        response, expires_at = row
        if expires_at:
            expires_dt = datetime.fromisoformat(expires_at)
            if expires_dt < datetime.now(timezone.utc):
                await self._delete(prompt_hash)
                return None

        return response

    async def set(
        self,
        prompt: str,
        system: str = "",
        response: str = "",
        ttl_hours: Optional[int] = None,
    ) -> None:
        """Cache a response with TTL."""
        if not llm_settings.cache_enabled:
            return

        prompt_hash = _hash_content(prompt, system)
        system_hash = hashlib.sha256(system.encode()).hexdigest() if system else ""

        ttl = ttl_hours or llm_settings.cache_ttl_hours
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl)

        db = await self._get_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO llm_cache 
            (prompt_hash, system_hash, response, model, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                prompt_hash,
                system_hash,
                response,
                llm_settings.model,
                datetime.now(timezone.utc).isoformat(),
                expires_at.isoformat(),
            ),
        )
        await db.commit()

    async def _delete(self, prompt_hash: str) -> None:
        """Delete a cache entry."""
        db = await self._get_db()
        await db.execute("DELETE FROM llm_cache WHERE prompt_hash = ?", (prompt_hash,))
        await db.commit()

    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of deleted."""
        if not llm_settings.cache_enabled:
            return 0

        db = await self._get_db()
        now = datetime.now(timezone.utc).isoformat()

        cursor = await db.execute(
            "DELETE FROM llm_cache WHERE expires_at < ?",
            (now,),
        )
        await db.commit()
        return cursor.rowcount

    async def clear(self) -> None:
        """Clear all cache entries."""
        db = await self._get_db()
        await db.execute("DELETE FROM llm_cache")
        await db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None


llm_cache = LLMCache()
