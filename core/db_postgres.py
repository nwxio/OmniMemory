from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from .integrations_config import db_settings


class PostgresError(RuntimeError):
    pass


class PostgresDB:
    """Minimal async PostgreSQL helper used by optional integrations.

    This module is intentionally lightweight and side-effect free.
    It is safe to import even when asyncpg is not installed.
    """

    def __init__(self) -> None:
        self._pool = None

    @property
    def dsn(self) -> str:
        password = db_settings.postgres_password or ""
        auth = f"{db_settings.postgres_user}:{password}" if password else db_settings.postgres_user
        return f"postgresql://{auth}@{db_settings.postgres_host}:{db_settings.postgres_port}/{db_settings.postgres_db}"

    async def init(self) -> None:
        try:
            import asyncpg  # type: ignore
        except Exception as e:  # pragma: no cover
            raise PostgresError("asyncpg is not installed") from e

        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Any]:
        if self._pool is None:
            await self.init()
        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, sql: str, *args: Any) -> str:
        async with self.connection() as conn:
            return await conn.execute(sql, *args)

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        async with self.connection() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(r) for r in rows]

    async def fetchrow(self, sql: str, *args: Any) -> Optional[dict[str, Any]]:
        async with self.connection() as conn:
            row = await conn.fetchrow(sql, *args)
            return dict(row) if row else None


postgres_db = PostgresDB()
