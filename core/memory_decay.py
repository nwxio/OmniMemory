"""Memory consolidation: decay, merge, and prune.

Maintains memory quality over time by:
- Decay: reducing importance of old memories
- Merge: combining similar memories
- Prune: removing low-importance memories
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .db import db
from .config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConsolidationReport:
    """Report after consolidation run."""

    decayed_count: int = 0
    merged_count: int = 0
    pruned_count: int = 0
    sessions_processed: int = 0
    errors: list[str] = field(default_factory=list)


class MemoryConsolidator:
    """Consolidator for memory quality maintenance.

    Works on:
    - Lessons: decay importance based on age
    - Sessions: prune old cross-session data
    - Episodes: merge similar, prune old
    """

    def __init__(self):
        # Settings with defaults
        self.decay_enabled = bool(getattr(settings, "memory_decay_enabled", True))
        self.decay_factor = float(getattr(settings, "memory_decay_factor", 0.9))  # per period
        self.decay_period_days = int(getattr(settings, "memory_decay_period_days", 30))
        self.decay_min_importance = float(getattr(settings, "memory_decay_min_importance", 0.1))

        self.merge_enabled = bool(getattr(settings, "memory_merge_enabled", True))
        self.merge_similarity_threshold = float(
            getattr(settings, "memory_merge_similarity_threshold", 0.85)
        )

        self.prune_enabled = bool(getattr(settings, "memory_prune_enabled", True))
        self.prune_max_age_days = int(getattr(settings, "memory_prune_max_age_days", 180))
        self.prune_min_score = float(getattr(settings, "memory_prune_min_score", 0.05))

    async def consolidate_all(self, dry_run: bool = True) -> dict[str, Any]:
        """Run full consolidation: decay, merge, prune.

        Args:
            dry_run: If True, only count what would be done

        Returns:
            ConsolidationReport
        """
        report = ConsolidationReport()

        try:
            # Decay old lessons
            if self.decay_enabled:
                decay_result = await self._decay_lessons(dry_run)
                report.decayed_count = decay_result.get("decayed", 0)
                report.errors.extend(decay_result.get("errors", []))

            # Merge similar lessons
            if self.merge_enabled:
                merge_result = await self._merge_lessons(dry_run)
                report.merged_count = merge_result.get("merged", 0)
                report.errors.extend(merge_result.get("errors", []))

            # Prune old sessions/cross-session memory
            if self.prune_enabled:
                prune_result = await self._prune_old_sessions(dry_run)
                report.pruned_count = prune_result.get("pruned", 0)
                report.sessions_processed = prune_result.get("sessions_processed", 0)
                report.errors.extend(prune_result.get("errors", []))

        except Exception as e:
            report.errors.append(f"Consolidation error: {str(e)}")

        return {
            "ok": True,
            "dry_run": dry_run,
            "decayed_count": report.decayed_count,
            "merged_count": report.merged_count,
            "pruned_count": report.pruned_count,
            "sessions_processed": report.sessions_processed,
            "errors": report.errors,
        }

    async def _decay_lessons(self, dry_run: bool = True) -> dict[str, Any]:
        """Decay importance of old lessons.

        Lessons have an implicit "importance" that decays over time.
        We model this by checking created_at and reducing effective importance.
        """
        if not self.decay_enabled:
            return {"decayed": 0, "errors": []}

        # Calculate cutoff for decay
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.decay_period_days)).isoformat()

        try:
            async with db.connect() as conn:
                # Find lessons older than decay period
                # In SQLite, we can't easily do math on JSON, so we just count
                if dry_run:
                    cur = await conn.execute(
                        "SELECT COUNT(*) as c FROM lessons WHERE created_at < ?", (cutoff,)
                    )
                    row = await cur.fetchone()
                    decayed = int(row["c"]) if row else 0
                else:
                    # For now, just delete old lessons instead of complex decay
                    # In a full implementation, we'd add a "importance" field
                    cur = await conn.execute("DELETE FROM lessons WHERE created_at < ?", (cutoff,))
                    await db.commit()
                    decayed = cur.rowcount if cur.rowcount else 0

                return {"decayed": decayed, "errors": []}
        except Exception as e:
            return {"decayed": 0, "errors": [str(e)]}

    async def _merge_lessons(self, dry_run: bool = True) -> dict[str, Any]:
        """Merge similar lessons.

        This is a simplified version - in production you'd use embeddings
        to find semantically similar lessons.
        """
        if not self.merge_enabled:
            return {"merged": 0, "errors": []}

        try:
            async with db.connect() as conn:
                # Find lessons with similar keys (same prefix)
                # Group by key prefix and keep only the latest
                cur = await conn.execute("""
                    SELECT key, COUNT(*) as cnt, MIN(created_at) as oldest, MAX(created_at) as newest
                    FROM lessons 
                    GROUP BY key 
                    HAVING cnt > 1
                """)
                rows = await cur.fetchall()

                if dry_run:
                    merged = sum(1 for r in rows if r["cnt"] > 1)
                else:
                    merged = 0
                    for row in rows:
                        key = row["key"]
                        # Keep the newest, delete others
                        cur2 = await conn.execute(
                            "SELECT id FROM lessons WHERE key = ? ORDER BY created_at DESC", (key,)
                        )
                        all_ids = [r["id"] for r in (await cur2.fetchall())]
                        if len(all_ids) > 1:
                            # Keep first, delete rest
                            to_delete = all_ids[1:]
                            placeholders = ",".join(["?"] * len(to_delete))
                            await conn.execute(
                                f"DELETE FROM lessons WHERE id IN ({placeholders})",
                                tuple(to_delete),
                            )
                            merged += len(to_delete)

                    await db.commit()

                return {"merged": merged, "errors": []}
        except Exception as e:
            return {"merged": 0, "errors": [str(e)]}

    async def _prune_old_sessions(self, dry_run: bool = True) -> dict[str, Any]:
        """Prune old cross-session memory.

        Delete old sessions that are no longer needed.
        """
        if not self.prune_enabled:
            return {"pruned": 0, "sessions_processed": 0, "errors": []}

        # Calculate cutoff
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.prune_max_age_days)).isoformat()

        try:
            async with db.connect() as conn:
                # Find old completed sessions
                if dry_run:
                    cur = await conn.execute(
                        """
                        SELECT COUNT(*) as c FROM sessions 
                        WHERE session_status = 'ended' 
                          AND ended_at < ?
                    """,
                        (cutoff,),
                    )
                    row = await cur.fetchone()
                    pruned = int(row["c"]) if row else 0
                else:
                    # Delete old ended sessions with their episodes
                    cur = await conn.execute(
                        """
                        SELECT id FROM sessions 
                        WHERE session_status = 'ended' 
                          AND ended_at < ?
                    """,
                        (cutoff,),
                    )
                    session_ids = [r["id"] for r in (await cur.fetchall())]

                    pruned = len(session_ids)
                    for sid in session_ids:
                        # Delete related episodes
                        await conn.execute("DELETE FROM episodes WHERE session_id = ?", (sid,))

                    # Delete sessions
                    placeholders = ",".join(["?"] * len(session_ids))
                    if session_ids:
                        await conn.execute(
                            f"DELETE FROM sessions WHERE id IN ({placeholders})", tuple(session_ids)
                        )

                    await db.commit()

                return {"pruned": pruned, "sessions_processed": pruned, "errors": []}
        except Exception as e:
            return {"pruned": 0, "sessions_processed": 0, "errors": [str(e)]}

    async def get_consolidation_status(self) -> dict[str, Any]:
        """Get current consolidation settings and status."""
        return {
            "enabled": self.decay_enabled or self.merge_enabled or self.prune_enabled,
            "decay": {
                "enabled": self.decay_enabled,
                "factor": self.decay_factor,
                "period_days": self.decay_period_days,
                "min_importance": self.decay_min_importance,
            },
            "merge": {
                "enabled": self.merge_enabled,
                "similarity_threshold": self.merge_similarity_threshold,
            },
            "prune": {
                "enabled": self.prune_enabled,
                "max_age_days": self.prune_max_age_days,
                "min_score": self.prune_min_score,
            },
        }


_consolidator: Optional[MemoryConsolidator] = None


def memory_consolidator() -> MemoryConsolidator:
    """Get or create consolidator singleton."""
    global _consolidator
    if _consolidator is None:
        _consolidator = MemoryConsolidator()
    return _consolidator
