from __future__ import annotations

import asyncio
import json
import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings
from .db import db, dumps, fetch_one, fetch_all
from .ids import new_id
from .redact import redact_text, redact_dict
from .search_match import build_like_clause, query_tokens, score_fields


def _row_get(row: Any, name: str, default: Any = None) -> Any:
    """Safely read a column from aiosqlite.Row (or similar).

    We use this to keep forward/backward compatibility during lightweight
    migrations where some columns may not exist yet.
    """
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(name, default)
    try:
        keys = row.keys()  # sqlite3.Row / aiosqlite.Row
        if name in keys:
            return row[name]
    except Exception:
        pass
    try:
        return row[name]
    except Exception:
        return default


def _sanitize_fts_query(q: str) -> str:
    """Sanitize user text for SQLite FTS5 MATCH queries.

    FTS5's MATCH parser is picky: punctuation like "!", ".", quotes, etc. can trigger
    syntax errors. We intentionally *strip almost all punctuation* and collapse
    whitespace so casual prompts like "hello !!" or "ping 1.1.1.1" won't break
    retrieval.

    This is not meant to be perfect linguistics — just robust and safe.
    """
    q = (q or "").strip()
    if not q:
        return ""
    # Keep unicode word characters + spaces only.
    q = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


_DEFAULT_DENY = [
    # Secrets
    ".env",
    "**/.env",
    ".env.local",
    ".env.development",
    ".env.test",
    ".env.production",
    ".ssh",
    "**/.ssh/**",
    "**/*.key",
    "**/*.pem",
    # VCS / deps / build artifacts
    ".git",
    "**/.git/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/.venv/**",
    "**/venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    # Large/binary-ish blobs and archives
    "**/*.db",
    "**/*.sqlite",
    "**/*.sqlite3",
    "**/*.zip",
    "**/*.tar",
    "**/*.gz",
    "**/*.7z",
    "**/*.bin",
    "**/*.so",
    "**/*.dylib",
    "**/*.dll",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_probably_binary(b: bytes) -> bool:
    if not b:
        return False
    if b"\x00" in b:
        return True
    # Fast path: if the bytes are valid UTF-8, treat as text.
    # This prevents misclassifying non-ASCII languages (e.g. Cyrillic) as "binary".
    try:
        b.decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass

    # Fallback heuristic: too many non-text bytes.
    # Count high-bit bytes as textish as well (common in non-ASCII encodings).
    textish = sum((c in (9, 10, 13) or 32 <= c <= 126 or c >= 128) for c in b)
    return (textish / max(1, len(b))) < 0.7


def _denied(rel_path: str, deny_patterns: List[str]) -> bool:
    import fnmatch

    rel = rel_path.lstrip("/")
    for pat in deny_patterns:
        p = pat.lstrip("/")
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch("/" + rel, p):
            return True
    return False


def _sha256_bytes(b: bytes) -> str:
    try:
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return ""


def _episode_fingerprint(title: str, summary: str) -> str:
    """Stable fingerprint for an episode.

    Used to deduplicate imported episodes and prevent unbounded growth when
    users repeatedly import the same snapshot.

    We intentionally hash a normalized (title + summary) string; this keeps it
    deterministic even if episode IDs differ across exports.
    """
    t = (title or "").strip().lower()
    s = (summary or "").strip().lower()
    base = (t + "\n" + s).encode("utf-8", errors="ignore")
    return hashlib.sha256(base).hexdigest()


def _mono() -> float:
    try:
        return time.monotonic()
    except Exception:
        return 0.0


@dataclass
class MemorySearchHit:
    path: str
    snippet: str
    rank: float


class MemorySQL:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # In-process throttle for housekeeping to avoid doing expensive work
        # on every single task completion.
        self._last_housekeep_mono: float = 0.0

    async def save_session_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        now = _utc_now()

        # Session snapshots may contain tool params/results. Redact obvious secrets
        # before persisting, so the UI and retrieval do not accidentally expose them.
        if bool(getattr(settings, "snapshot_redact_secrets", True)):
            try:
                if isinstance(snapshot, dict):
                    snapshot = redact_dict(snapshot)
            except Exception:
                pass

        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO session_memory(session_id, snapshot_json, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET snapshot_json=excluded.snapshot_json, updated_at=excluded.updated_at",
                (session_id, dumps(snapshot), now),
            )
            await conn.commit()

    async def get_session_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        async with db.connect() as conn:
            row = await fetch_one(
                conn, "SELECT snapshot_json FROM session_memory WHERE session_id=?", (session_id,)
            )
        if not row:
            return None
        return json.loads(row["snapshot_json"])

    async def set_working_memory(self, session_id: str, content: str) -> None:
        """Replace the whole working memory blob for a session."""
        now = _utc_now()
        content = str(content or "")
        if bool(getattr(settings, "wm_redact_secrets", True)):
            try:
                content = redact_text(content)
            except Exception:
                pass
        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO working_memory(session_id, content, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                (session_id, content, now),
            )
            await conn.commit()

    async def get_working_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return {content, updated_at} or None."""
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT content, updated_at FROM working_memory WHERE session_id=?",
                (session_id,),
            )
        if not row:
            return None
        return {"content": row["content"], "updated_at": row["updated_at"]}

    async def clear_working_memory(self, session_id: str) -> None:
        async with db.connect() as conn:
            await conn.execute("DELETE FROM working_memory WHERE session_id=?", (session_id,))
            await conn.commit()

    async def append_working_memory(
        self, session_id: str, text: str, *, max_chars: int = 12000
    ) -> Dict[str, Any]:
        """Append text to WM, keeping it bounded."""
        add = str(text or "").strip()
        if not add:
            cur = await self.get_working_memory(session_id)
            return {
                "content": (cur or {}).get("content", ""),
                "updated_at": (cur or {}).get("updated_at"),
            }

        cur = await self.get_working_memory(session_id)
        base = (cur or {}).get("content", "") or ""

        if base and not base.endswith("\n"):
            base += "\n"
        merged = base + add

        max_chars = int(max(0, max_chars))
        if max_chars and len(merged) > max_chars:
            merged = merged[-max_chars:]

        await self.set_working_memory(session_id, merged)
        cur2 = await self.get_working_memory(session_id)
        return {
            "content": (cur2 or {}).get("content", merged),
            "updated_at": (cur2 or {}).get("updated_at"),
        }

    async def add_lesson(
        self, key: str, lesson: str, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        now = _utc_now()
        if bool(getattr(settings, "lessons_redact_secrets", True)):
            try:
                lesson = redact_text(str(lesson or ""))
            except Exception:
                lesson = str(lesson or "")

        # TTL: calculate expires_at
        ttl_days = int(getattr(settings, "memory_lessons_ttl_days", 90))
        expires_at = None
        if ttl_days > 0:
            expires_dt = datetime.now(timezone.utc) + timedelta(days=ttl_days)
            expires_at = expires_dt.isoformat()

        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO lessons(key, lesson, meta_json, created_at, expires_at, updated_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET lesson=excluded.lesson, meta_json=excluded.meta_json, created_at=excluded.created_at, expires_at=excluded.expires_at, updated_at=excluded.updated_at",
                (key, lesson, dumps(meta or {}), now, expires_at, now),
            )
            await conn.commit()

    async def list_lessons(self, limit: int = 50) -> List[Dict[str, Any]]:
        now = _utc_now()
        async with db.connect() as conn:
            # Filter out expired lessons
            rows = await fetch_all(
                conn,
                "SELECT key, lesson, meta_json, created_at, expires_at, updated_at FROM lessons "
                "WHERE expires_at IS NULL OR expires_at > ? ORDER BY updated_at DESC LIMIT ?",
                (now, limit),
            )
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "key": r["key"],
                    "lesson": r["lesson"],
                    "meta": json.loads(r["meta_json"]),
                    "created_at": r["created_at"],
                    "expires_at": r["expires_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return out

    async def get_lesson(self, key: str) -> Optional[Dict[str, Any]]:
        k = (key or "").strip()
        if not k:
            return None
        now = _utc_now()
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT key, lesson, meta_json, created_at, expires_at, updated_at FROM lessons "
                "WHERE key=? AND (expires_at IS NULL OR expires_at > ?)",
                (k, now),
            )
        if not row:
            return None
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}
        return {
            "key": row["key"],
            "lesson": row["lesson"],
            "meta": meta,
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "updated_at": row["updated_at"],
        }

    async def delete_lesson(self, key: str) -> Dict[str, Any]:
        """Delete a lesson by key."""
        k = (key or "").strip()
        if not k:
            return {"ok": False, "error": "missing key"}
        async with db.connect() as conn:
            cur = await conn.execute("DELETE FROM lessons WHERE key=?", (k,))
            await conn.commit()
        return {"ok": True, "deleted": int(cur.rowcount or 0)}

    async def export_lessons(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """Export lessons for backup/migration.

        This intentionally does not include large workspace indexes.
        """
        limit = int(max(1, min(20000, int(limit or 10000))))
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                "SELECT key, lesson, meta_json, updated_at FROM lessons ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                meta = json.loads(r["meta_json"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "key": r["key"],
                    "lesson": r["lesson"],
                    "meta": meta,
                    "updated_at": r["updated_at"],
                }
            )
        return out

    async def export_preferences(
        self,
        *,
        scope: str,
        session_id: Optional[str],
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        scope = (scope or "").strip().lower() or "global"
        if scope not in ("global", "session"):
            return []
        if scope == "session" and not session_id:
            return []
        limit = int(max(1, min(20000, int(limit or 10000))))

        q = "SELECT scope, session_id, key, value_json, source, is_locked, created_at, updated_by, updated_at FROM preferences WHERE scope=? AND session_id IS ? ORDER BY updated_at DESC LIMIT ?"
        async with db.connect() as conn:
            rows = await fetch_all(conn, q, (scope, session_id, limit))

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                v = json.loads(r["value_json"])
            except Exception:
                v = r["value_json"]
            out.append(
                {
                    "scope": r["scope"],
                    "session_id": r["session_id"],
                    "key": r["key"],
                    "value": v,
                    "source": _row_get(r, "source"),
                    "is_locked": bool(int(_row_get(r, "is_locked", 0) or 0)),
                    "created_at": _row_get(r, "created_at"),
                    "updated_by": _row_get(r, "updated_by"),
                    "updated_at": _row_get(r, "updated_at"),
                }
            )
        return out

    async def export_episodes(self, session_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        limit = int(max(1, min(20000, int(limit or 5000))))
        async with db.connect() as conn:
            try:
                rows = await fetch_all(
                    conn,
                    "SELECT id, session_id, task_id, created_at, title, summary, tags_json, data_json, fingerprint FROM episodes WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                )
            except Exception:
                rows = await fetch_all(
                    conn,
                    "SELECT id, session_id, task_id, created_at, title, summary, tags_json, data_json FROM episodes WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                )

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                tags = json.loads(r["tags_json"] or "[]")
            except Exception:
                tags = []
            try:
                data = json.loads(r["data_json"] or "{}")
            except Exception:
                data = {}
            out.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "task_id": r["task_id"],
                    "created_at": r["created_at"],
                    "title": r["title"],
                    "summary": r["summary"],
                    "tags": tags,
                    "data": data,
                    "fingerprint": _row_get(r, "fingerprint"),
                }
            )
        return out

    async def search_lessons(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search lessons using SQLite FTS5.

        Returns a list of dicts with lesson text + a compact snippet for UI.
        This is best-effort and must never raise.
        """
        q = (query or "").strip()
        if not q:
            return []
        q_s = _sanitize_fts_query(q)
        if not q_s:
            return []

        async with db.connect() as conn:
            try:
                rows = await fetch_all(
                    conn,
                    """
                    SELECT l.key,
                           l.lesson,
                           l.meta_json,
                           l.updated_at,
                           snippet(lessons_fts, 1, '[', ']', '…', 16) AS snippet,
                           bm25(lessons_fts) AS score
                    FROM lessons_fts
                    JOIN lessons l ON lessons_fts.rowid = l.rowid
                    WHERE lessons_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (q_s, int(limit)),
                )
            except Exception:
                return []

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                out.append(
                    {
                        "key": str(r["key"]),
                        "lesson": str(r["lesson"]),
                        "meta": json.loads(r["meta_json"] or "{}"),
                        "updated_at": r["updated_at"],
                        "snippet": str(r["snippet"] or ""),
                        "rank": float(r["score"]) if r["score"] is not None else 0.0,
                    }
                )
            except Exception:
                continue
        return out

    async def lessons_maintenance(
        self, *, dry_run: bool = True, mode: str = "strict"
    ) -> Dict[str, Any]:
        """Perform lightweight maintenance on lessons.

        Goals:
        - Remove exact duplicate lessons that differ only by key.
        - Keep the most recent version and record alias keys in meta.

        Safety:
        - Only de-duplicates by exact *normalized* text match.
        - Never raises; returns best-effort stats.

        mode:
          - strict: lower+strip+collapse whitespace
          - loose: strict + remove punctuation
        """
        try:
            mode = (mode or "strict").strip().lower()
            if mode not in ("strict", "loose"):
                mode = "strict"

            async with db.connect() as conn:
                rows = await fetch_all(
                    conn, "SELECT key, lesson, meta_json, updated_at FROM lessons"
                )

            def _norm(s: str) -> str:
                import re

                t = (s or "").strip().lower()
                t = re.sub(r"\s+", " ", t)
                if mode == "loose":
                    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
                    t = re.sub(r"\s+", " ", t).strip()
                return t

            buckets: dict[str, list[dict]] = {}
            for r in rows:
                key = str(r["key"])
                lesson = str(r["lesson"] or "")
                norm = _norm(lesson)
                if not norm:
                    # Skip empty lessons.
                    continue
                h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                buckets.setdefault(h, []).append(
                    {
                        "key": key,
                        "lesson": lesson,
                        "meta_json": str(r["meta_json"] or "{}"),
                        "updated_at": str(r["updated_at"] or ""),
                    }
                )

            duplicates: list[dict] = []
            removed = 0
            updated = 0

            # Plan actions first.
            for h, items in buckets.items():
                if len(items) <= 1:
                    continue
                # Winner = newest updated_at; tie-breaker = shortest key.
                items_sorted = sorted(
                    items, key=lambda x: (x.get("updated_at", ""), -len(x.get("key", "")))
                )
                winner = items_sorted[-1]
                losers = [x for x in items if x["key"] != winner["key"]]

                duplicates.append(
                    {
                        "hash": h,
                        "winner_key": winner["key"],
                        "winner_updated_at": winner.get("updated_at"),
                        "loser_keys": [x["key"] for x in losers],
                    }
                )

                if dry_run:
                    continue

                # Update winner meta with alias keys.
                try:
                    meta = json.loads(winner.get("meta_json") or "{}")
                    if not isinstance(meta, dict):
                        meta = {}
                except Exception:
                    meta = {}
                alias = meta.get("dedup_alias_keys")
                if not isinstance(alias, list):
                    alias = []
                for lk in [x["key"] for x in losers]:
                    if lk not in alias:
                        alias.append(lk)
                meta["dedup_alias_keys"] = alias

                async with db.connect() as conn:
                    await conn.execute(
                        "UPDATE lessons SET meta_json=?, updated_at=? WHERE key=?",
                        (dumps(meta), _utc_now(), winner["key"]),
                    )
                    # Delete losers.
                    for lo in losers:
                        await conn.execute("DELETE FROM lessons WHERE key=?", (lo["key"],))
                        removed += 1
                    await conn.commit()
                updated += 1

            return {
                "ok": True,
                "dry_run": bool(dry_run),
                "mode": mode,
                "lesson_count": len(rows),
                "duplicate_groups": len(duplicates),
                "planned": duplicates if dry_run else [],
                "removed": removed,
                "updated": updated,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def lessons_stats(self) -> Dict[str, Any]:
        """Return basic counts about lessons (best-effort)."""
        now = _utc_now()
        try:
            async with db.connect() as conn:
                # Count total and expired
                total_row = await fetch_one(conn, "SELECT COUNT(*) AS c FROM lessons")
                expired_row = await fetch_one(
                    conn,
                    "SELECT COUNT(*) AS c FROM lessons WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
            return {
                "ok": True,
                "count": int(total_row["c"] if total_row else 0),
                "expired": int(expired_row["c"] if expired_row else 0),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def ttl_cleanup(self, *, dry_run: bool = True) -> Dict[str, Any]:
        """Cleanup expired lessons and episodes based on TTL.

        This is called automatically on startup or can be triggered manually.
        """
        now = _utc_now()
        actions: Dict[str, int] = {"lessons_deleted": 0, "episodes_deleted": 0}

        try:
            async with db.connect() as conn:
                # Delete expired lessons
                if not dry_run:
                    cur = await conn.execute(
                        "DELETE FROM lessons WHERE expires_at IS NOT NULL AND expires_at <= ?",
                        (now,),
                    )
                    actions["lessons_deleted"] = cur.rowcount if cur.rowcount else 0

                    # Delete expired episodes
                    cur = await conn.execute(
                        "DELETE FROM episodes WHERE expires_at IS NOT NULL AND expires_at <= ?",
                        (now,),
                    )
                    actions["episodes_deleted"] = cur.rowcount if cur.rowcount else 0

                    await conn.commit()
                else:
                    # Just count for dry run
                    row = await fetch_one(
                        conn,
                        "SELECT COUNT(*) AS c FROM lessons WHERE expires_at IS NOT NULL AND expires_at <= ?",
                        (now,),
                    )
                    actions["lessons_deleted"] = int(row["c"] if row else 0)

                    row = await fetch_one(
                        conn,
                        "SELECT COUNT(*) AS c FROM episodes WHERE expires_at IS NOT NULL AND expires_at <= ?",
                        (now,),
                    )
                    actions["episodes_deleted"] = int(row["c"] if row else 0)

            return {"ok": True, "dry_run": dry_run, "actions": actions}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def memory_stats(self) -> Dict[str, Any]:
        """Return lightweight counts for key memory tables.

        This is intended for health/debug endpoints. It should be fast and best-effort.
        """
        out: Dict[str, Any] = {"ok": True}
        try:
            async with db.connect() as conn:
                rows = await fetch_all(
                    conn,
                    """
                    SELECT 'lessons' AS name, COUNT(1) AS c FROM lessons
                    UNION ALL SELECT 'episodes', COUNT(1) FROM episodes
                    UNION ALL SELECT 'preferences', COUNT(1) FROM preferences
                    UNION ALL SELECT 'working_memory', COUNT(1) FROM working_memory
                    UNION ALL SELECT 'session_memory', COUNT(1) FROM session_memory
                    UNION ALL SELECT 'memory_docs', COUNT(1) FROM memory_docs
                    UNION ALL SELECT 'vector_chunks', COUNT(1) FROM vector_chunks
                    UNION ALL SELECT 'vector_files_meta', COUNT(1) FROM vector_files_meta
                    """,
                )
            counts: Dict[str, int] = {}
            for r in rows or []:
                try:
                    name = str(r[0]) if not isinstance(r, dict) else str(r.get("name"))
                    c = r[1] if not isinstance(r, dict) else r.get("c")
                    counts[name] = int(c or 0)
                except Exception:
                    continue
            out["counts"] = counts
        except Exception as e:
            out["ok"] = False
            out["error"] = str(e)
        return out

    # --- Maintenance state (internal key/value) ---

    async def set_maintenance_state(self, key: str, value: dict) -> None:
        """Persist a small internal state blob.

        This is used for background job bookkeeping (e.g. last consolidation run).
        Best-effort: failures should not break the app.
        """
        k = str(key or "").strip()
        if not k:
            return
        now = _utc_now()
        try:
            payload = dumps(value if isinstance(value, dict) else {"value": value})
        except Exception:
            payload = dumps({"value": str(value)})
        try:
            async with db.connect() as conn:
                await conn.execute(
                    "INSERT INTO maintenance_state(key, value_json, updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                    (k, payload, now),
                )
                await conn.commit()
        except Exception:
            return

    async def get_maintenance_state(self, key: str) -> dict | None:
        """Return internal state blob by key (or None)."""
        k = str(key or "").strip()
        if not k:
            return None
        try:
            async with db.connect() as conn:
                row = await fetch_one(
                    conn, "SELECT value_json, updated_at FROM maintenance_state WHERE key=?", (k,)
                )
            if not row:
                return None
            try:
                v = json.loads(row["value_json"])
            except Exception:
                v = {"raw": row["value_json"]}
            if isinstance(v, dict):
                v.setdefault("updated_at", _row_get(row, "updated_at"))
            return (
                v
                if isinstance(v, dict)
                else {"value": v, "updated_at": _row_get(row, "updated_at")}
            )
        except Exception:
            return None

    async def list_maintenance_state(self, prefix: str = "", limit: int = 50) -> list[dict]:
        """List maintenance_state entries by key prefix (newest first).

        Used for UI/debug status pages. Best-effort: returns [] on errors.
        """
        pref = str(prefix or "").strip()
        try:
            limit_i = max(1, int(limit))
        except Exception:
            limit_i = 50
        try:
            async with db.connect() as conn:
                if pref:
                    like = pref + "%"
                    rows = await conn.execute_fetchall(
                        "SELECT key, value_json, updated_at FROM maintenance_state WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
                        (like, limit_i),
                    )
                else:
                    rows = await conn.execute_fetchall(
                        "SELECT key, value_json, updated_at FROM maintenance_state ORDER BY updated_at DESC LIMIT ?",
                        (limit_i,),
                    )
            out: list[dict] = []
            for r in rows or []:
                try:
                    k = r["key"] if isinstance(r, dict) else r[0]
                    vj = r["value_json"] if isinstance(r, dict) else r[1]
                    ua = r["updated_at"] if isinstance(r, dict) else r[2]
                    try:
                        v = json.loads(vj)
                    except Exception:
                        v = {"raw": vj}
                    if isinstance(v, dict):
                        v.setdefault("updated_at", ua)
                    out.append({"key": k, "value": v, "updated_at": ua})
                except Exception:
                    continue
            return out
        except Exception:
            return []

    # --- Preferences (durable key/value) ---

    async def set_preference(
        self,
        *,
        scope: str,
        session_id: Optional[str],
        key: str,
        value: Any,
        source: str = "auto",
        is_locked: Optional[bool] = None,
        updated_by: str = "auto",
    ) -> Dict[str, Any]:
        """Set or update a preference.

        scope:
          - global: applies to all sessions
          - session: applies only to the given session_id

        value is stored as JSON.
        """
        scope = (scope or "").strip().lower() or "global"
        if scope not in ("global", "session"):
            return {"ok": False, "error": "invalid scope"}
        if scope == "session" and not session_id:
            return {"ok": False, "error": "missing session_id for session scope"}
        k = (key or "").strip()
        if not k:
            return {"ok": False, "error": "missing key"}

        src = (source or "").strip().lower() or "auto"
        if src not in ("auto", "manual", "system"):
            src = "auto"

        # Defaults: manual/system writes are locked by default; auto writes are not.
        if is_locked is None:
            locked = True if src in ("manual", "system") else False
        else:
            locked = bool(is_locked)

        pref_id = new_id("pref")
        now = _utc_now()

        # Optional redaction for preference values (opt-in).
        if bool(getattr(settings, "preferences_redact_secrets", False)):
            try:
                if isinstance(value, str):
                    value = redact_text(value)
                elif isinstance(value, (dict, list)):
                    value = redact_dict(value)  # handles nested dict/list
            except Exception:
                pass

        async with db.connect() as conn:
            # If a user locked this preference, auto-capture must not stomp it.
            if src == "auto":
                existing = await fetch_one(
                    conn,
                    "SELECT is_locked, source, updated_at FROM preferences WHERE scope=? AND session_id IS ? AND key=?",
                    (scope, session_id, k),
                )
                if existing and int(existing["is_locked"] or 0) == 1:
                    return {
                        "ok": True,
                        "scope": scope,
                        "session_id": session_id,
                        "key": k,
                        "updated_at": existing["updated_at"],
                        "skipped": True,
                        "reason": "locked",
                    }

            await conn.execute(
                "INSERT INTO preferences(id, scope, session_id, key, value_json, source, is_locked, created_at, updated_by, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(scope, session_id, key) DO UPDATE SET "
                "value_json=excluded.value_json, "
                "source=excluded.source, "
                "is_locked=excluded.is_locked, "
                "updated_by=excluded.updated_by, "
                "updated_at=excluded.updated_at",
                (
                    pref_id,
                    scope,
                    session_id,
                    k,
                    dumps(value),
                    src,
                    1 if locked else 0,
                    now,
                    updated_by,
                    now,
                ),
            )
            await conn.commit()

        return {
            "ok": True,
            "scope": scope,
            "session_id": session_id,
            "key": k,
            "source": src,
            "is_locked": locked,
            "updated_by": updated_by,
            "updated_at": now,
        }

    async def get_preference(
        self, *, scope: str, session_id: Optional[str], key: str
    ) -> Optional[Dict[str, Any]]:
        scope = (scope or "").strip().lower() or "global"
        if scope not in ("global", "session"):
            return None
        if scope == "session" and not session_id:
            return None
        k = (key or "").strip()
        if not k:
            return None
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT scope, session_id, key, value_json, source, is_locked, created_at, updated_by, updated_at "
                "FROM preferences WHERE scope=? AND session_id IS ? AND key=?",
                (scope, session_id, k),
            )
        if not row:
            return None
        try:
            value = json.loads(row["value_json"])
        except Exception:
            value = row["value_json"]
        return {
            "scope": row["scope"],
            "session_id": row["session_id"],
            "key": row["key"],
            "value": value,
            "source": _row_get(row, "source"),
            "is_locked": bool(int(_row_get(row, "is_locked", 0) or 0)),
            "created_at": _row_get(row, "created_at"),
            "updated_by": _row_get(row, "updated_by"),
            "updated_at": _row_get(row, "updated_at"),
        }

    async def list_preferences(
        self,
        *,
        scope: str,
        session_id: Optional[str],
        prefix: str = "",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        scope = (scope or "").strip().lower() or "global"
        if scope not in ("global", "session"):
            return []
        if scope == "session" and not session_id:
            return []
        prefix = (prefix or "").strip()
        limit = int(max(1, min(1000, int(limit))))

        if prefix:
            like = prefix + "%"
            q = "SELECT scope, session_id, key, value_json, source, is_locked, created_at, updated_by, updated_at FROM preferences WHERE scope=? AND session_id IS ? AND key LIKE ? ORDER BY updated_at DESC LIMIT ?"
            args = (scope, session_id, like, limit)
        else:
            q = "SELECT scope, session_id, key, value_json, source, is_locked, created_at, updated_by, updated_at FROM preferences WHERE scope=? AND session_id IS ? ORDER BY updated_at DESC LIMIT ?"
            args = (scope, session_id, limit)

        async with db.connect() as conn:
            rows = await fetch_all(conn, q, args)

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                v = json.loads(r["value_json"])
            except Exception:
                v = r["value_json"]
            out.append(
                {
                    "scope": r["scope"],
                    "session_id": r["session_id"],
                    "key": r["key"],
                    "value": v,
                    "source": _row_get(r, "source"),
                    "is_locked": bool(int(_row_get(r, "is_locked", 0) or 0)),
                    "created_at": _row_get(r, "created_at"),
                    "updated_by": _row_get(r, "updated_by"),
                    "updated_at": _row_get(r, "updated_at"),
                }
            )
        return out

    async def delete_preference(
        self, *, scope: str, session_id: Optional[str], key: str
    ) -> Dict[str, Any]:
        scope = (scope or "").strip().lower() or "global"
        if scope not in ("global", "session"):
            return {"ok": False, "error": "invalid scope"}
        if scope == "session" and not session_id:
            return {"ok": False, "error": "missing session_id for session scope"}
        k = (key or "").strip()
        if not k:
            return {"ok": False, "error": "missing key"}
        async with db.connect() as conn:
            cur = await conn.execute(
                "DELETE FROM preferences WHERE scope=? AND session_id IS ? AND key=?",
                (scope, session_id, k),
            )
            await conn.commit()
        return {"ok": True, "deleted": int(cur.rowcount or 0)}

    async def add_episode(
        self,
        *,
        session_id: str,
        task_id: Optional[str],
        title: str,
        summary: str,
        tags: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
        index_into_fts: bool = True,
    ) -> Dict[str, Any]:
        """Store an episodic memory entry.

        Episodes are durable, chronological "what happened" notes.
        Optionally, we also index a markdown snapshot into memory_docs so
        existing FTS search can retrieve episodes without extra schema.
        TTL: Auto-expires based on episodes_ttl_days setting.
        """
        if not session_id:
            return {"ok": False, "error": "missing session_id"}
        episode_id = new_id("ep")
        now = _utc_now()
        t = (title or "").strip() or "Episode"
        s = (summary or "").strip()
        tag_list = tags if isinstance(tags, list) else []
        payload = data if isinstance(data, dict) else {}

        # TTL: calculate expires_at
        ttl_days = int(getattr(settings, "memory_episodes_ttl_days", 60))
        expires_at = None
        if ttl_days > 0:
            expires_dt = datetime.now(timezone.utc) + timedelta(days=ttl_days)
            expires_at = expires_dt.isoformat()

        fp = _episode_fingerprint(t, s)

        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO episodes(id, session_id, task_id, created_at, expires_at, title, summary, tags_json, data_json, fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    episode_id,
                    session_id,
                    task_id,
                    now,
                    expires_at,
                    t,
                    s,
                    dumps(tag_list),
                    dumps(payload),
                    fp,
                ),
            )
            await conn.commit()

        if index_into_fts:
            # Keep it compact; the detailed payload lives in episodes.data_json.
            md = "# " + t + "\n\n" + s + "\n\n" + "Tags: " + ", ".join(tag_list)
            await self.upsert_doc(path=f"episodes/{session_id}/{episode_id}.md", content=md)

        return {"ok": True, "episode_id": episode_id, "created_at": now}

    async def episode_exists_by_fingerprint(self, session_id: str, fingerprint: str) -> bool:
        if not session_id or not fingerprint:
            return False
        async with db.connect() as conn:
            try:
                row = await fetch_one(
                    conn,
                    "SELECT 1 AS one FROM episodes WHERE session_id=? AND fingerprint=? LIMIT 1",
                    (session_id, fingerprint),
                )
                return bool(row)
            except Exception:
                return False

    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        if not episode_id:
            return None
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT id, session_id, task_id, created_at, title, summary, tags_json, data_json FROM episodes WHERE id=?",
                (episode_id,),
            )
        if not row:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "task_id": row["task_id"],
            "created_at": row["created_at"],
            "title": row["title"],
            "summary": row["summary"],
            "tags": json.loads(row["tags_json"]),
            "data": json.loads(row["data_json"]),
        }

    async def list_episodes(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                "SELECT id, task_id, created_at, title, summary, tags_json FROM episodes WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                (session_id, int(limit)),
            )
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "task_id": r["task_id"],
                    "created_at": r["created_at"],
                    "title": r["title"],
                    "summary": r["summary"],
                    "tags": json.loads(r["tags_json"]),
                }
            )
        return out

    async def search_episodes(
        self, session_id: str, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Lightweight search over episodes (title/summary) using LIKE.

        We intentionally keep this simple: episodes are also indexed into memory_docs
        so FTS search can find them. This endpoint is just for UI convenience.
        """
        q = (query or "").strip()
        if not session_id or not q:
            return []

        tokens = query_tokens(q, max_terms=10)
        if not tokens:
            tokens = [q.casefold()]
        clause, params = build_like_clause(["title", "summary"], tokens, require_all_tokens=False)
        if not clause:
            return []

        pool = max(int(limit) * 6, int(limit), 20)
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                """
                SELECT id, task_id, created_at, title, summary, tags_json
                FROM episodes
                WHERE session_id=? AND
                """
                + clause
                + """
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, *params, int(pool)),
            )
        scored: List[tuple[float, Dict[str, Any]]] = []
        for r in rows:
            score = score_fields(
                q,
                [str(r["title"] or ""), str(r["summary"] or "")],
                tokens=tokens,
            )
            if score <= 0.0:
                continue
            item = {
                "id": r["id"],
                "task_id": r["task_id"],
                "created_at": r["created_at"],
                "title": r["title"],
                "summary": r["summary"],
                "tags": json.loads(r["tags_json"]),
                "search_score": float(score),
            }
            scored.append((score, item))

        scored.sort(key=lambda it: (it[0], str(it[1].get("created_at") or "")), reverse=True)
        return [item for _, item in scored[: int(limit)]]

    async def get_index_state(self, name: str) -> Dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {}
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT name, last_scan_at, stats_json FROM memory_index_state WHERE name=?",
                (name,),
            )
        if not row:
            return {}
        stats = {}
        try:
            stats = json.loads(row["stats_json"] or "{}")
        except Exception:
            stats = {}
        return {
            "name": row["name"],
            "last_scan_at": row["last_scan_at"],
            "stats": stats,
        }

    async def set_index_state(
        self, name: str, *, last_scan_at: Optional[str], stats: Optional[Dict[str, Any]] = None
    ) -> None:
        name = (name or "").strip()
        if not name:
            return
        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO memory_index_state(name, last_scan_at, stats_json) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET last_scan_at=excluded.last_scan_at, stats_json=excluded.stats_json",
                (name, last_scan_at, dumps(stats or {})),
            )
            await conn.commit()

    async def workspace_status(self) -> Dict[str, Any]:
        async with db.connect() as conn:
            row_docs = await fetch_one(conn, "SELECT COUNT(1) AS c FROM memory_docs")
            row_meta = await fetch_one(conn, "SELECT COUNT(1) AS c FROM workspace_files_meta")
        st = await self.get_index_state("workspace_fts")
        return {
            "doc_count": int(row_docs["c"]) if row_docs and row_docs["c"] is not None else 0,
            "meta_count": int(row_meta["c"]) if row_meta and row_meta["c"] is not None else 0,
            "index_state": st,
        }

    async def clear_workspace_index(self) -> Dict[str, Any]:
        async with self._lock:
            async with db.connect() as conn:
                await conn.execute("DELETE FROM memory_docs")
                await conn.execute("DELETE FROM workspace_files_meta")
                await conn.execute(
                    "DELETE FROM memory_index_state WHERE name=?", ("workspace_fts",)
                )
                await conn.commit()
        return {"ok": True}

    async def index_workspace_incremental(
        self,
        *,
        root: Optional[str] = None,
        deny_patterns: Optional[List[str]] = None,
        max_file_bytes: Optional[int] = None,
        max_files: Optional[int] = None,
        max_seconds: Optional[float] = None,
        min_interval_seconds: Optional[float] = None,
        force: bool = False,
        prune_missing: bool = False,
    ) -> Dict[str, Any]:
        """Incrementally index workspace files into memory_docs/memory_fts.

        Uses workspace_files_meta (path, mtime, size, sha256) to skip unchanged files.
        Throttled via memory_index_state(name='workspace_fts') to avoid scanning on every request.

        If force=True, ignores meta and reindexes all eligible files (still uses deny/limits).
        If prune_missing=True and the scan completes without hitting limits, removes docs/meta
        for files that disappeared from the workspace.
        """
        ws = Path(root or settings.workspace).resolve()
        deny = deny_patterns or _DEFAULT_DENY

        max_file_bytes = int(
            max_file_bytes
            if max_file_bytes is not None
            else getattr(settings, "workspace_fts_max_file_bytes", 512_000)
        )
        max_files = int(
            max_files
            if max_files is not None
            else getattr(settings, "workspace_fts_max_files", 6000)
        )
        max_seconds = float(
            max_seconds
            if max_seconds is not None
            else getattr(settings, "workspace_fts_max_seconds", 2.0)
        )
        min_interval_seconds = float(
            min_interval_seconds
            if min_interval_seconds is not None
            else getattr(settings, "workspace_fts_min_interval_seconds", 120.0)
        )

        indexed = 0
        skipped = 0
        unchanged = 0
        errors = 0
        started_at = _utc_now()
        t0 = _mono()
        hit_limits = False
        seen: set[str] = set()

        # Throttle background scanning unless forced.
        if not force and min_interval_seconds > 0:
            st = await self.get_index_state("workspace_fts")
            last = (st.get("stats") or {}).get("last_scan_mono")
            try:
                last_mono = float(last)
            except Exception:
                last_mono = 0.0
            if last_mono > 0 and (_mono() - last_mono) < float(min_interval_seconds):
                return {
                    "root": str(ws),
                    "started_at": started_at,
                    "throttled": True,
                    "indexed": 0,
                    "skipped": 0,
                    "unchanged": 0,
                    "errors": 0,
                    "hit_limits": False,
                }

        async with self._lock:
            async with db.connect() as conn:
                ops = 0
                for p in ws.rglob("*"):
                    if p.is_dir():
                        continue

                    rel = str(p.relative_to(ws))
                    if _denied(rel, deny):
                        skipped += 1
                        continue

                    # Limits
                    if (indexed + skipped + unchanged + errors) >= max_files:
                        hit_limits = True
                        break
                    if max_seconds > 0 and (_mono() - t0) >= max_seconds:
                        hit_limits = True
                        break

                    try:
                        st = p.stat()
                    except Exception:
                        errors += 1
                        continue

                    if st.st_size <= 0 or st.st_size > max_file_bytes:
                        skipped += 1
                        continue

                    seen.add(rel)

                    # Check meta for unchanged files
                    if not force:
                        row = await fetch_one(
                            conn,
                            "SELECT mtime, size FROM workspace_files_meta WHERE path=?",
                            (rel,),
                        )
                        if row:
                            try:
                                mtime = float(row["mtime"]) if row["mtime"] is not None else 0.0
                                size = int(row["size"]) if row["size"] is not None else -1
                            except Exception:
                                mtime = 0.0
                                size = -1
                            if abs(float(st.st_mtime) - mtime) < 1e-6 and int(st.st_size) == size:
                                unchanged += 1
                                continue

                    # Read + decode
                    try:
                        b = p.read_bytes()
                        if _is_probably_binary(b):
                            skipped += 1
                            continue
                        content = b.decode("utf-8", errors="replace")
                        if getattr(settings, "workspace_redact_secrets", True):
                            content = redact_text(content)
                    except Exception:
                        errors += 1
                        continue

                    doc_id = new_id("doc")
                    now = _utc_now()
                    sha = _sha256_bytes(b)

                    await conn.execute(
                        "INSERT INTO memory_docs(id, path, content, updated_at) VALUES(?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                        (doc_id, rel, content, now),
                    )
                    await conn.execute(
                        "INSERT INTO workspace_files_meta(path, mtime, size, sha256, updated_at) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, sha256=excluded.sha256, updated_at=excluded.updated_at",
                        (rel, float(st.st_mtime), int(st.st_size), sha, now),
                    )
                    indexed += 1
                    ops += 1

                    # Commit in batches for performance
                    if ops >= 100:
                        await conn.commit()
                        ops = 0

                # Optional prune (only safe when scan completed)
                if prune_missing and not hit_limits:
                    # Delete rows for paths not present on disk.
                    rows_meta = await fetch_all(conn, "SELECT path FROM workspace_files_meta")
                    stale = [str(r[0]) for r in rows_meta if str(r[0]) not in seen]
                    if stale:
                        q = ",".join(["?"] * len(stale))
                        await conn.execute(
                            f"DELETE FROM workspace_files_meta WHERE path IN ({q})", tuple(stale)
                        )
                        await conn.execute(
                            f"DELETE FROM memory_docs WHERE path IN ({q})", tuple(stale)
                        )

                await conn.commit()

        # Save index state
        stats = {
            "last_scan_mono": _mono(),
            "indexed": indexed,
            "unchanged": unchanged,
            "skipped": skipped,
            "errors": errors,
            "hit_limits": hit_limits,
            "max_files": max_files,
            "max_seconds": max_seconds,
            "force": bool(force),
            "prune_missing": bool(prune_missing),
        }
        await self.set_index_state("workspace_fts", last_scan_at=_utc_now(), stats=stats)

        return {
            "root": str(ws),
            "started_at": started_at,
            "throttled": False,
            "indexed": indexed,
            "unchanged": unchanged,
            "skipped": skipped,
            "errors": errors,
            "hit_limits": hit_limits,
        }

    async def index_workspace(
        self,
        *,
        root: Optional[str] = None,
        deny_patterns: Optional[List[str]] = None,
        max_file_bytes: int = 512_000,
        max_files: int = 6000,
    ) -> Dict[str, Any]:
        # Backwards-compatible full scan: force reindex and (when scan completes)
        # prune missing files so the index doesn't keep stale content forever.
        return await self.index_workspace_incremental(
            root=root,
            deny_patterns=deny_patterns,
            max_file_bytes=max_file_bytes,
            max_files=max_files,
            max_seconds=getattr(settings, "workspace_fts_fullscan_max_seconds", 0) or 0,
            min_interval_seconds=0,
            force=True,
            prune_missing=True,
        )

    async def index_workspace_paths(
        self,
        *,
        paths: List[str],
        root: Optional[str] = None,
        deny_patterns: Optional[List[str]] = None,
        max_file_bytes: Optional[int] = None,
        max_files: Optional[int] = None,
        max_seconds: Optional[float] = None,
        force: bool = True,
    ) -> Dict[str, Any]:
        """Index a *specific* list of workspace-relative paths into FTS.

        This is used to keep the index fresh right after writes/commits,
        without paying for a full workspace scan.
        """
        ws = Path(root or settings.workspace).expanduser().resolve()
        if not ws.exists():
            return {"status": "failed", "error": f"workspace does not exist: {ws}"}

        deny = deny_patterns or _DEFAULT_DENY
        max_file_bytes_i = int(
            max_file_bytes if max_file_bytes is not None else settings.workspace_fts_max_file_bytes
        )
        max_files_i = int(
            max_files
            if max_files is not None
            else getattr(settings, "workspace_index_on_write_max_files", 25)
        )
        max_seconds_f = float(
            max_seconds
            if max_seconds is not None
            else getattr(settings, "workspace_index_on_write_max_seconds", 2.0)
        )

        started_at = _utc_now()
        t0 = _mono()
        indexed = 0
        skipped = 0
        unchanged = 0
        errors = 0
        hit_limits = False

        # Normalize + de-dupe while preserving order.
        seen: set[str] = set()
        norm_paths: List[str] = []
        for p in paths or []:
            rel = str(p or "").lstrip("/").replace("\\", "/")
            if not rel or rel in seen:
                continue
            seen.add(rel)
            norm_paths.append(rel)

        async with self._lock:
            async with db.connect() as conn:
                ops = 0
                for rel in norm_paths:
                    if indexed + skipped + unchanged + errors >= max_files_i:
                        hit_limits = True
                        break
                    if max_seconds_f > 0 and (_mono() - t0) >= max_seconds_f:
                        hit_limits = True
                        break

                    if _denied(rel, deny):
                        skipped += 1
                        continue

                    abs_p = (ws / rel).resolve()
                    # Prevent path escape.
                    if not str(abs_p).startswith(str(ws)):
                        skipped += 1
                        continue
                    if not abs_p.exists() or not abs_p.is_file():
                        skipped += 1
                        continue

                    try:
                        st = abs_p.stat()
                    except Exception:
                        errors += 1
                        continue

                    if st.st_size <= 0 or st.st_size > max_file_bytes_i:
                        skipped += 1
                        continue

                    # Check meta for unchanged (unless forced).
                    if not force:
                        row = await fetch_one(
                            conn,
                            "SELECT mtime, size FROM workspace_files_meta WHERE path=?",
                            (rel,),
                        )
                        if row:
                            try:
                                mtime = float(row["mtime"]) if row["mtime"] is not None else 0.0
                                size = int(row["size"]) if row["size"] is not None else -1
                            except Exception:
                                mtime = 0.0
                                size = -1
                            if abs(float(st.st_mtime) - mtime) < 1e-6 and int(st.st_size) == size:
                                unchanged += 1
                                continue

                    try:
                        b = abs_p.read_bytes()
                        if _is_probably_binary(b):
                            skipped += 1
                            continue
                        text = b.decode("utf-8", errors="replace")
                        if getattr(settings, "workspace_redact_secrets", True):
                            text = redact_text(text)
                    except Exception:
                        errors += 1
                        continue

                    if not text.strip():
                        skipped += 1
                        continue

                    try:
                        await self._upsert_doc_and_meta(
                            conn,
                            rel,
                            text,
                            mtime=float(st.st_mtime),
                            size=int(st.st_size),
                            sha256=_sha256_bytes(b),
                        )
                        indexed += 1
                        ops += 1
                    except Exception:
                        errors += 1

                    if ops >= 25:
                        await conn.commit()
                        ops = 0

                await conn.commit()

        return {
            "status": "ok",
            "root": str(ws),
            "started_at": started_at,
            "indexed": indexed,
            "unchanged": unchanged,
            "skipped": skipped,
            "errors": errors,
            "hit_limits": hit_limits,
        }

    async def upsert_doc(self, path: str, content: str) -> Dict[str, Any]:
        """Upsert a single document into memory_docs (and FTS triggers)."""
        if not path:
            return {"ok": False, "error": "missing path"}
        now = _utc_now()
        doc_id = new_id("doc")
        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO memory_docs(id, path, content, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                (doc_id, path, content or "", now),
            )
            await conn.commit()
        return {"ok": True, "path": path, "chars": len(content or ""), "updated_at": now}

    async def get_doc(self, path: str) -> Optional[Dict[str, Any]]:
        """Fetch a single document from memory_docs by path.

        This is used by web_research_tool to retrieve cached web pages
        and by other tools that need to read indexed content without
        touching the filesystem.
        """
        if not path:
            return None
        async with db.connect() as conn:
            row = await fetch_one(
                conn, "SELECT path, content, updated_at FROM memory_docs WHERE path=?", (path,)
            )
        if not row:
            return None
        return {"path": row["path"], "content": row["content"], "updated_at": row["updated_at"]}

    async def get_docs(self, paths: List[str], limit_chars: int = 12000) -> List[Dict[str, Any]]:
        """Fetch multiple documents by paths, preserving order.

        The overall returned content is truncated to limit_chars to avoid
        huge prompts. Missing docs are silently skipped.
        """
        out: List[Dict[str, Any]] = []
        if not paths:
            return out
        remaining = max(0, int(limit_chars))
        async with db.connect() as conn:
            for path in paths:
                if remaining <= 0:
                    break
                if not path:
                    continue
                row = await fetch_one(
                    conn, "SELECT path, content, updated_at FROM memory_docs WHERE path=?", (path,)
                )
                if not row:
                    continue
                content = row["content"] or ""
                if remaining and len(content) > remaining:
                    content = content[: max(0, remaining - 1)] + ("…" if remaining > 1 else "")
                remaining -= len(content)
                out.append(
                    {"path": row["path"], "content": content, "updated_at": row["updated_at"]}
                )
        return out

    async def get_workspace_file_mtimes(self, paths: List[str]) -> Dict[str, float]:
        """Return workspace file mtimes for given relative paths.

        Uses workspace_files_meta table populated by workspace indexing.
        Missing paths are simply omitted.
        """
        out: Dict[str, float] = {}
        if not paths:
            return out

        # Deduplicate while preserving order (stable behavior for callers).
        seen: set[str] = set()
        uniq: List[str] = []
        for p in paths:
            p2 = (p or "").strip()
            if not p2 or p2 in seen:
                continue
            seen.add(p2)
            uniq.append(p2)

        if not uniq:
            return out

        # SQLite has practical limits on bound params; chunk queries.
        CHUNK = 400
        async with db.connect() as conn:
            for i in range(0, len(uniq), CHUNK):
                chunk = uniq[i : i + CHUNK]
                q = ",".join(["?"] * len(chunk))
                rows = await fetch_all(
                    conn,
                    f"SELECT path, mtime FROM workspace_files_meta WHERE path IN ({q})",
                    tuple(chunk),
                )
                for r in rows or []:
                    try:
                        out[str(r["path"])] = float(r["mtime"] or 0.0)
                    except Exception:
                        continue

        return out

    async def search(self, query: str, limit: int = 8) -> List[MemorySearchHit]:
        """Search the FTS index (path + content) and return best hits.

        This is intentionally simple: BM25 + snippets is enough for v1, and keeps
        the implementation portable (no extra deps).
        """
        if not query.strip():
            return []
        query_s = _sanitize_fts_query(query)
        if not query_s:
            return []

        async with self._lock:
            async with db.connect() as conn:
                rows = await fetch_all(
                    conn,
                    """
                    SELECT path, snippet(memory_fts, 1, '[', ']', '…', 16) AS snippet,
                           bm25(memory_fts) AS score
                    FROM memory_fts
                    WHERE memory_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (query_s, int(limit)),
                )
        hits: List[MemorySearchHit] = []
        for r in rows:
            hits.append(
                MemorySearchHit(
                    path=str(r[0]),
                    snippet=str(r[1]),
                    rank=float(r[2]) if r[2] is not None else 0.0,
                )
            )
        return hits

    async def doc_count(self) -> int:
        async with self._lock:
            async with db.connect() as conn:
                row = await fetch_one(conn, "SELECT COUNT(1) AS c FROM memory_docs")
                return int(row["c"]) if row and row["c"] is not None else 0

    async def _episode_count(self, session_id: str) -> int:
        if not session_id:
            return 0
        async with db.connect() as conn:
            row = await fetch_one(
                conn, "SELECT COUNT(1) AS c FROM episodes WHERE session_id=?", (session_id,)
            )
        return int(row["c"]) if row and row["c"] is not None else 0

    async def prune_episodes(
        self,
        *,
        session_id: str,
        max_per_session: int,
        retention_days: int,
    ) -> Dict[str, Any]:
        """Prune episodic memory for a session.

        We delete:
          - episodes older than retention_days (if > 0)
          - episodes beyond max_per_session newest (if > 0)

        We also delete the corresponding FTS-indexed markdown docs under
        `episodes/{session_id}/{episode_id}.md` to keep memory_fts clean.
        """
        out: Dict[str, Any] = {"deleted_by_age": 0, "deleted_by_count": 0, "kept": None}
        if not session_id:
            out["error"] = "missing session_id"
            return out

        # Time-based pruning
        ids_deleted: List[str] = []
        if int(retention_days) > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=int(retention_days))).isoformat()
            async with db.connect() as conn:
                rows = await fetch_all(
                    conn,
                    "SELECT id FROM episodes WHERE session_id=? AND created_at < ?",
                    (session_id, cutoff),
                )
                old_ids = [str(r[0]) for r in rows]
                if old_ids:
                    # Delete episodes
                    q = ",".join(["?"] * len(old_ids))
                    await conn.execute(
                        f"DELETE FROM episodes WHERE session_id=? AND id IN ({q})",
                        (session_id, *old_ids),
                    )
                    # Delete associated memory_docs
                    paths = [f"episodes/{session_id}/{eid}.md" for eid in old_ids]
                    qp = ",".join(["?"] * len(paths))
                    await conn.execute(
                        f"DELETE FROM memory_docs WHERE path IN ({qp})", tuple(paths)
                    )
                    await conn.commit()
                    out["deleted_by_age"] = len(old_ids)
                    ids_deleted.extend(old_ids)

        # Count-based pruning
        max_per_session = int(max(0, max_per_session))
        if max_per_session > 0:
            cnt = await self._episode_count(session_id)
            if cnt > max_per_session:
                async with db.connect() as conn:
                    rows2 = await fetch_all(
                        conn,
                        "SELECT id FROM episodes WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                        (session_id, max_per_session),
                    )
                    keep_ids = [str(r[0]) for r in rows2]
                    out["kept"] = len(keep_ids)
                    if keep_ids:
                        qk = ",".join(["?"] * len(keep_ids))
                        # Find ids to delete for report
                        rows_del = await fetch_all(
                            conn,
                            f"SELECT id FROM episodes WHERE session_id=? AND id NOT IN ({qk})",
                            (session_id, *keep_ids),
                        )
                        del_ids = [str(r[0]) for r in rows_del]
                        if del_ids:
                            qd = ",".join(["?"] * len(del_ids))
                            await conn.execute(
                                f"DELETE FROM episodes WHERE session_id=? AND id IN ({qd})",
                                (session_id, *del_ids),
                            )
                            paths2 = [f"episodes/{session_id}/{eid}.md" for eid in del_ids]
                            qp2 = ",".join(["?"] * len(paths2))
                            await conn.execute(
                                f"DELETE FROM memory_docs WHERE path IN ({qp2})", tuple(paths2)
                            )
                            await conn.commit()
                            out["deleted_by_count"] = len(del_ids)
                            ids_deleted.extend(del_ids)

        out["deleted_total"] = len(set(ids_deleted))
        return out

    async def prune_audit_events(self, *, retention_days: int, max_rows: int) -> Dict[str, Any]:
        """Prune audit_events table by age and hard cap (best-effort)."""
        out: Dict[str, Any] = {"deleted_by_age": 0, "deleted_by_cap": 0}
        retention_days = int(retention_days)
        max_rows = int(max(0, max_rows))

        if retention_days > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
            async with db.connect() as conn:
                cur = await conn.execute("DELETE FROM audit_events WHERE ts < ?", (cutoff,))
                await conn.commit()
                try:
                    out["deleted_by_age"] = int(cur.rowcount or 0)
                except Exception:
                    out["deleted_by_age"] = 0

        if max_rows > 0:
            async with db.connect() as conn:
                row = await fetch_one(conn, "SELECT COUNT(1) AS c FROM audit_events")
                cnt = int(row["c"]) if row and row["c"] is not None else 0
                if cnt > max_rows:
                    to_delete = cnt - max_rows
                    # Delete oldest rows first.
                    rows2 = await fetch_all(
                        conn, "SELECT id FROM audit_events ORDER BY id ASC LIMIT ?", (to_delete,)
                    )
                    ids = [int(r[0]) for r in rows2]
                    if ids:
                        q = ",".join(["?"] * len(ids))
                        cur2 = await conn.execute(
                            f"DELETE FROM audit_events WHERE id IN ({q})", tuple(ids)
                        )
                        await conn.commit()
                        try:
                            out["deleted_by_cap"] = int(cur2.rowcount or 0)
                        except Exception:
                            out["deleted_by_cap"] = 0

        out["ok"] = True
        return out

    async def prune_autonomy_actions(
        self,
        *,
        session_id: Optional[str] = None,
        retention_days: int = 14,
        max_rows: int = 50_000,
        max_rows_per_session: int = 10_000,
    ) -> Dict[str, Any]:
        """Prune autonomy_actions table by age and caps (best-effort).

        autonomy_actions logs tool executions/fingerprints for autonomy guardrails and UI history.
        It can grow quickly on long-running deployments (especially with A3/A4), so we keep it bounded.
        """
        import time

        out: Dict[str, Any] = {"deleted_by_age": 0, "deleted_by_cap": 0}
        retention_days = int(retention_days)
        max_rows = int(max(0, max_rows))
        max_rows_per_session = int(max(0, max_rows_per_session))

        # Age-based pruning
        if retention_days > 0:
            cutoff_ts = float(time.time() - (retention_days * 86400))
            async with db.connect() as conn:
                if session_id:
                    cur = await conn.execute(
                        "DELETE FROM autonomy_actions WHERE session_id=? AND ts < ?",
                        (session_id, cutoff_ts),
                    )
                else:
                    cur = await conn.execute(
                        "DELETE FROM autonomy_actions WHERE ts < ?", (cutoff_ts,)
                    )
                await conn.commit()
                try:
                    out["deleted_by_age"] = int(cur.rowcount or 0)
                except Exception:
                    out["deleted_by_age"] = 0

        # Cap pruning (delete oldest)
        if session_id and max_rows_per_session > 0:
            async with db.connect() as conn:
                row = await fetch_one(
                    conn,
                    "SELECT COUNT(1) AS c FROM autonomy_actions WHERE session_id=?",
                    (session_id,),
                )
                cnt = int(row["c"]) if row and _row_get(row, "c") is not None else 0
                if cnt > max_rows_per_session:
                    to_delete = cnt - max_rows_per_session
                    rows2 = await fetch_all(
                        conn,
                        "SELECT id FROM autonomy_actions WHERE session_id=? ORDER BY ts ASC LIMIT ?",
                        (session_id, to_delete),
                    )
                    ids = [int(r[0]) for r in rows2]
                    if ids:
                        q = ",".join(["?"] * len(ids))
                        cur2 = await conn.execute(
                            f"DELETE FROM autonomy_actions WHERE id IN ({q})", tuple(ids)
                        )
                        await conn.commit()
                        try:
                            out["deleted_by_cap"] += int(cur2.rowcount or 0)
                        except Exception:
                            pass

        if max_rows > 0:
            async with db.connect() as conn:
                row = await fetch_one(conn, "SELECT COUNT(1) AS c FROM autonomy_actions")
                cnt = int(row["c"]) if row and _row_get(row, "c") is not None else 0
                if cnt > max_rows:
                    to_delete = cnt - max_rows
                    rows2 = await fetch_all(
                        conn,
                        "SELECT id FROM autonomy_actions ORDER BY ts ASC LIMIT ?",
                        (to_delete,),
                    )
                    ids = [int(r[0]) for r in rows2]
                    if ids:
                        q = ",".join(["?"] * len(ids))
                        cur2 = await conn.execute(
                            f"DELETE FROM autonomy_actions WHERE id IN ({q})", tuple(ids)
                        )
                        await conn.commit()
                        try:
                            out["deleted_by_cap"] += int(cur2.rowcount or 0)
                        except Exception:
                            pass

        out["ok"] = True
        return out

    async def prune_monitor_results(
        self,
        *,
        session_id: Optional[str] = None,
        retention_days: int = 14,
        max_rows: int = 50_000,
        max_rows_per_session: int = 10_000,
    ) -> Dict[str, Any]:
        """Prune monitor_results table by age and caps (best-effort).

        monitor_results can grow quickly on frequent checks, so we keep it bounded.
        """
        out: Dict[str, Any] = {"deleted_by_age": 0, "deleted_by_cap": 0}
        retention_days = int(retention_days)
        max_rows = int(max(0, max_rows))
        max_rows_per_session = int(max(0, max_rows_per_session))

        # Age-based pruning
        if retention_days > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
            async with db.connect() as conn:
                if session_id:
                    cur = await conn.execute(
                        "DELETE FROM monitor_results WHERE session_id=? AND checked_at < ?",
                        (session_id, cutoff),
                    )
                else:
                    cur = await conn.execute(
                        "DELETE FROM monitor_results WHERE checked_at < ?", (cutoff,)
                    )
                await conn.commit()
                try:
                    out["deleted_by_age"] = int(cur.rowcount or 0)
                except Exception:
                    out["deleted_by_age"] = 0

        # Cap pruning (delete oldest)
        if session_id and max_rows_per_session > 0:
            async with db.connect() as conn:
                row = await fetch_one(
                    conn,
                    "SELECT COUNT(1) AS c FROM monitor_results WHERE session_id=?",
                    (session_id,),
                )
                cnt = int(row["c"]) if row and _row_get(row, "c") is not None else 0
                if cnt > max_rows_per_session:
                    to_delete = cnt - max_rows_per_session
                    rows2 = await fetch_all(
                        conn,
                        "SELECT id FROM monitor_results WHERE session_id=? ORDER BY checked_at ASC LIMIT ?",
                        (session_id, to_delete),
                    )
                    ids = [str(r[0]) for r in rows2]
                    if ids:
                        q = ",".join(["?"] * len(ids))
                        cur2 = await conn.execute(
                            f"DELETE FROM monitor_results WHERE id IN ({q})", tuple(ids)
                        )
                        await conn.commit()
                        try:
                            out["deleted_by_cap"] += int(cur2.rowcount or 0)
                        except Exception:
                            pass

        if max_rows > 0:
            async with db.connect() as conn:
                row = await fetch_one(conn, "SELECT COUNT(1) AS c FROM monitor_results")
                cnt = int(row["c"]) if row and _row_get(row, "c") is not None else 0
                if cnt > max_rows:
                    to_delete = cnt - max_rows
                    rows2 = await fetch_all(
                        conn,
                        "SELECT id FROM monitor_results ORDER BY checked_at ASC LIMIT ?",
                        (to_delete,),
                    )
                    ids = [str(r[0]) for r in rows2]
                    if ids:
                        q = ",".join(["?"] * len(ids))
                        cur2 = await conn.execute(
                            f"DELETE FROM monitor_results WHERE id IN ({q})", tuple(ids)
                        )
                        await conn.commit()
                        try:
                            out["deleted_by_cap"] += int(cur2.rowcount or 0)
                        except Exception:
                            pass

        out["ok"] = True
        return out

    async def housekeep(self, *, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Best-effort housekeeping to keep the DB bounded.

        This runs in-process with a throttle; it should never block main flows.
        """
        import time

        min_interval = int(getattr(settings, "housekeeping_min_interval_s", 30) or 30)
        now_mono = time.monotonic()
        if min_interval > 0 and (now_mono - self._last_housekeep_mono) < float(min_interval):
            return {"skipped": True, "reason": "throttled"}
        self._last_housekeep_mono = now_mono

        out: Dict[str, Any] = {"ok": True}
        try:
            if session_id:
                out["episodes"] = await self.prune_episodes(
                    session_id=session_id,
                    max_per_session=int(getattr(settings, "episode_max_per_session", 500) or 500),
                    retention_days=int(getattr(settings, "episode_retention_days", 180) or 180),
                )
        except Exception as e:
            out["episodes_error"] = str(e)

        try:
            out["audit"] = await self.prune_audit_events(
                retention_days=int(getattr(settings, "audit_retention_days", 60) or 60),
                max_rows=int(getattr(settings, "audit_max_rows", 100_000) or 100_000),
            )
        except Exception as e:
            out["audit_error"] = str(e)

        try:
            out["monitor_results"] = await self.prune_monitor_results(
                session_id=session_id,
                retention_days=int(getattr(settings, "monitor_results_retention_days", 14) or 14),
                max_rows=int(getattr(settings, "monitor_results_max_rows", 50_000) or 50_000),
                max_rows_per_session=int(
                    getattr(settings, "monitor_results_max_rows_per_session", 10_000) or 10_000
                ),
            )
        except Exception as e:
            out["monitor_results_error"] = str(e)

        try:
            out["autonomy_actions"] = await self.prune_autonomy_actions(
                session_id=session_id,
                retention_days=int(getattr(settings, "autonomy_actions_retention_days", 14) or 14),
                max_rows=int(getattr(settings, "autonomy_actions_max_rows", 50_000) or 50_000),
                max_rows_per_session=int(
                    getattr(settings, "autonomy_actions_max_rows_per_session", 10_000) or 10_000
                ),
            )
        except Exception as e:
            out["autonomy_actions_error"] = str(e)

        try:
            if bool(getattr(settings, "db_wal_checkpoint_on_housekeep", True)):
                mode = (
                    str(getattr(settings, "db_wal_checkpoint_mode", "TRUNCATE") or "TRUNCATE")
                    .upper()
                    .strip()
                )
                if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
                    mode = "TRUNCATE"
                async with db.connect() as conn:
                    # Keep WAL from growing without bound on long-running deployments
                    cur = await conn.execute(f"PRAGMA wal_checkpoint({mode});")
                    try:
                        rows = await cur.fetchall()
                        out["wal_checkpoint"] = [list(r) for r in rows]
                    except Exception:
                        out["wal_checkpoint"] = True
                    try:
                        await cur.close()
                    except Exception:
                        pass
                    # Also let SQLite optimize internal statistics (best-effort)
                    try:
                        cur2 = await conn.execute("PRAGMA optimize;")
                        try:
                            await cur2.close()
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception as e:
            out["wal_checkpoint_error"] = str(e)

        # Persist last-run summary for UI/debug (best-effort).
        try:
            import time as _time

            now_epoch = float(_time.time())
            summary = dict(out)
            summary.setdefault("run_at", _utc_now())
            summary.setdefault("run_epoch", now_epoch)
            if session_id:
                summary.setdefault("session_id", str(session_id))
            await self.set_maintenance_state("housekeep:last", summary)
            if session_id:
                await self.set_maintenance_state(f"housekeep:last:{session_id}", summary)
        except Exception:
            pass

        return out

    # ========== Procedural Memory ==========
    async def add_procedure(
        self,
        key: str,
        title: str,
        steps: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a procedure (how-to)."""
        now = _utc_now()
        proc_id = new_id("proc")
        async with db.connect() as conn:
            await conn.execute(
                """INSERT INTO procedural_memory(id, key, title, steps, metadata, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                   title=excluded.title, steps=excluded.steps, metadata=excluded.metadata,
                   updated_at=excluded.updated_at""",
                (proc_id, key, title, dumps(steps), dumps(metadata or {}), now, now),
            )
            await conn.commit()

    async def get_procedure(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a procedure by key."""
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT key, title, steps, metadata, created_at, updated_at FROM procedural_memory WHERE key = ?",
                (key,),
            )
        if not row:
            return None
        return {
            "key": row["key"],
            "title": row["title"],
            "steps": json.loads(row["steps"]),
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def search_procedures(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search procedures by title or content."""
        q = (query or "").strip()
        if not q:
            return []

        tokens = query_tokens(q, max_terms=10)
        if not tokens:
            tokens = [q.casefold()]
        clause, params = build_like_clause(["title", "steps"], tokens, require_all_tokens=False)
        if not clause:
            return []

        pool = max(int(limit) * 6, int(limit), 20)
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                """SELECT key, title, steps, metadata, created_at, updated_at FROM procedural_memory
                   WHERE """
                + clause
                + """ LIMIT ?""",
                (*params, pool),
            )
        scored: List[tuple[float, Dict[str, Any]]] = []
        for r in rows:
            score = score_fields(
                q,
                [str(r["title"] or ""), str(r["steps"] or "")],
                tokens=tokens,
            )
            if score <= 0.0:
                continue
            item = {
                "key": r["key"],
                "title": r["title"],
                "steps": json.loads(r["steps"]),
                "metadata": json.loads(r["metadata"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "search_score": float(score),
            }
            scored.append((score, item))

        scored.sort(key=lambda it: (it[0], str(it[1].get("updated_at") or "")), reverse=True)
        return [item for _, item in scored[: int(limit)]]

    async def list_procedures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all procedures."""
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                "SELECT key, title, steps, metadata, created_at, updated_at FROM procedural_memory ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        out = []
        for r in rows:
            out.append(
                {
                    "key": r["key"],
                    "title": r["title"],
                    "steps": json.loads(r["steps"]),
                    "metadata": json.loads(r["metadata"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return out

    async def delete_procedure(self, key: str) -> Dict[str, Any]:
        """Delete a procedure."""
        async with db.connect() as conn:
            await conn.execute("DELETE FROM procedural_memory WHERE key = ?", (key,))
            await conn.commit()
        return {"ok": True, "deleted": key}

    # ========== Semantic Memory ==========
    async def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add or update an entity."""
        now = _utc_now()
        entity_id = new_id("ent")
        async with db.connect() as conn:
            await conn.execute(
                """INSERT INTO semantic_entities(id, name, entity_type, properties, created_at, updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name, entity_type=excluded.entity_type, properties=excluded.properties,
                   updated_at=excluded.updated_at""",
                (entity_id, name, entity_type, dumps(properties or {}), now, now),
            )
            await conn.commit()
        return entity_id

    async def get_entity(self, id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT id, name, entity_type, properties, created_at, updated_at FROM semantic_entities WHERE id = ?",
                (id,),
            )
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "properties": json.loads(row["properties"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def search_entities(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search entities by name or type."""
        q = (query or "").strip()
        if not q:
            return []

        tokens = query_tokens(q, max_terms=10)
        if not tokens:
            tokens = [q.casefold()]
        clause, params = build_like_clause(
            ["name", "entity_type", "properties"],
            tokens,
            require_all_tokens=False,
        )
        if not clause:
            return []

        pool = max(int(limit) * 6, int(limit), 20)
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                """SELECT id, name, entity_type, properties, created_at, updated_at FROM semantic_entities
                   WHERE """
                + clause
                + """ LIMIT ?""",
                (*params, pool),
            )
        scored: List[tuple[float, Dict[str, Any]]] = []
        for r in rows:
            props = json.loads(r["properties"])
            score = score_fields(
                q,
                [
                    str(r["name"] or ""),
                    str(r["entity_type"] or ""),
                    json.dumps(props, ensure_ascii=False),
                ],
                tokens=tokens,
            )
            if score <= 0.0:
                continue
            item = {
                "id": r["id"],
                "name": r["name"],
                "entity_type": r["entity_type"],
                "properties": props,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "search_score": float(score),
            }
            scored.append((score, item))

        scored.sort(key=lambda it: (it[0], str(it[1].get("updated_at") or "")), reverse=True)
        return [item for _, item in scored[: int(limit)]]

    async def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a relation between entities."""
        now = _utc_now()
        rel_id = new_id("rel")
        async with db.connect() as conn:
            await conn.execute(
                """INSERT INTO semantic_relations(id, subject_id, predicate, object_id, properties, created_at)
                   VALUES(?,?,?,?,?,?)""",
                (rel_id, subject_id, predicate, object_id, dumps(properties or {}), now),
            )
            await conn.commit()
        return rel_id

    async def get_relations(self, entity_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all relations for an entity (as subject or object)."""
        async with db.connect() as conn:
            rows = await fetch_all(
                conn,
                """SELECT r.id, r.subject_id, r.predicate, r.object_id, r.properties, r.created_at,
                          s.name as subject_name, o.name as object_name
                   FROM semantic_relations r
                   LEFT JOIN semantic_entities s ON r.subject_id = s.id
                   LEFT JOIN semantic_entities o ON r.object_id = o.id
                   WHERE r.subject_id = ? OR r.object_id = ?
                   ORDER BY r.created_at DESC LIMIT ?""",
                (entity_id, entity_id, limit),
            )
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "subject_id": r["subject_id"],
                    "subject_name": r["subject_name"],
                    "predicate": r["predicate"],
                    "object_id": r["object_id"],
                    "object_name": r["object_name"],
                    "properties": json.loads(r["properties"]),
                    "created_at": r["created_at"],
                }
            )
        return out


memory_sql = MemorySQL()
