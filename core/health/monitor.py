from __future__ import annotations

from typing import Any

from ..cache import redis_cache
from ..db import db_backend_info
from ..llm.client import llm_client


class HealthMonitor:
    """Best-effort runtime health monitor for external dependencies."""

    async def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": True,
            "db_backend": db_backend_info(),
        }

        try:
            out["llm"] = await llm_client.check_availability()
        except Exception as e:
            out["llm"] = {"provider_configured": False, "error": str(e)}
            out["ok"] = False

        try:
            out["redis"] = {
                "enabled": bool(redis_cache.is_enabled()),
            }
        except Exception as e:
            out["redis"] = {"enabled": False, "error": str(e)}

        return out


health_monitor = HealthMonitor()
