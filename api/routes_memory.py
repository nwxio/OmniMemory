from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from core.request_context import request_id_ctx
from core.db import db, fetch_one, row_get, loads, dumps
from core.memory import memory
from api.deps import ApiKeyDep

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/health", response_model=dict, dependencies=[ApiKeyDep])
async def memory_health(session_id: str) -> dict:
    """Lightweight snapshot of memory/index health.

    Useful for UI diagnostics and debugging.
    """
    # session_id currently unused; kept for symmetry.
    return await memory.health(session_id=session_id)


@router.post("/index", response_model=dict, dependencies=[ApiKeyDep])
async def index(session_id: str, root_path: str = "/workspace") -> dict:
    # session_id currently unused; kept for symmetry and future per-session policy.
    return await memory.index_project(root_path)


@router.post("/index_incremental", response_model=dict, dependencies=[ApiKeyDep])
async def index_incremental(
    session_id: str, root_path: str = "/workspace", force: bool = False
) -> dict:
    # Fast incremental refresh; used by UI and on-demand retrieval.
    return await memory.index_project_incremental(root_path, force=force)


@router.get("/index_status", response_model=dict, dependencies=[ApiKeyDep])
async def index_status(session_id: str) -> dict:
    st = await memory.workspace_status()
    return {"ok": True, "workspace": st}


@router.delete("/index_clear", response_model=dict, dependencies=[ApiKeyDep])
async def index_clear(session_id: str) -> dict:
    return await memory.clear_workspace_index()


@router.post("/index_vectors", response_model=dict, dependencies=[ApiKeyDep])
async def index_vectors(session_id: str, root_path: str = "/workspace") -> dict:
    # Index into vector store for semantic search.
    try:
        return await memory.index_project_vectors(root_path)
    except Exception as e:
        # Vectors may be disabled or embeddings unavailable; do not break UI with HTTP 500.
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
        }


@router.post("/index_vectors_incremental", response_model=dict, dependencies=[ApiKeyDep])
async def index_vectors_incremental(
    session_id: str, root_path: str = "/workspace", force: bool = False
) -> dict:
    try:
        return await memory.index_project_vectors_incremental(root_path, force=force)
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
        }


@router.get("/index_vectors_status", response_model=dict, dependencies=[ApiKeyDep])
async def index_vectors_status(session_id: str) -> dict:
    try:
        st = await memory.vectors_status()
        return {"ok": True, "vectors": st}
    except Exception as e:
        # Some deployments do not have vector tables yet; return a soft error.
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
        }


@router.delete("/index_vectors_clear", response_model=dict, dependencies=[ApiKeyDep])
async def index_vectors_clear(session_id: str) -> dict:
    try:
        return await memory.clear_vectors_index()
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
        }


@router.get("/semantic_search", response_model=dict, dependencies=[ApiKeyDep])
async def semantic_search(session_id: str, q: str, limit: int = 8) -> dict:
    try:
        hits = await memory.semantic_search(q, limit=limit)
        return {"ok": True, "hits": [h.__dict__ for h in hits]}
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
            "hits": [],
        }


@router.get("/search", response_model=dict, dependencies=[ApiKeyDep])
async def search(session_id: str, q: str, limit: int = 8) -> dict:
    hits = await memory.search(q, limit=limit)
    return {"hits": [h.__dict__ for h in hits]}


@router.get("/hybrid_search", response_model=dict, dependencies=[ApiKeyDep])
async def hybrid_search(
    session_id: str,
    q: str,
    limit: int = 8,
    fts_limit: int = 24,
    vec_limit: int = 24,
    per_file_cap: int = 2,
    fts_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> dict:
    """Hybrid search over workspace (FTS + vectors), merged and re-ranked.

    The UI can use this instead of running two separate searches.
    """
    hits = await memory.hybrid_search(
        q,
        limit=limit,
        fts_limit=fts_limit,
        vec_limit=vec_limit,
        per_file_cap=per_file_cap,
        fts_weight=fts_weight,
        vec_weight=vec_weight,
    )
    return {"hits": [h.__dict__ for h in hits]}


@router.post("/lesson", response_model=dict, dependencies=[ApiKeyDep])
async def add_lesson(session_id: str, key: str, lesson: str) -> dict:
    await memory.add_lesson(key, lesson, meta={"session_id": session_id})
    return {"ok": True}


@router.get("/lessons", response_model=dict, dependencies=[ApiKeyDep])
async def lessons(session_id: str, limit: int = 50) -> dict:
    return {"lessons": await memory.list_lessons(limit=limit)}


@router.get("/lessons/search", response_model=dict, dependencies=[ApiKeyDep])
async def lessons_search(session_id: str, q: str, limit: int = 20) -> dict:
    """Search lessons by query (FTS).

    Useful for UI and for building more relevant RAG context.
    """
    hits = await memory.search_lessons(q, limit=limit)
    # order by rank (bm25: smaller is better)
    hits.sort(key=lambda x: float(x.get("rank") or 0.0))
    return {"ok": True, "hits": hits}


@router.get("/lessons/stats", response_model=dict, dependencies=[ApiKeyDep])
async def lessons_stats(session_id: str) -> dict:
    st = await memory.lessons_stats()
    return st


@router.post("/lessons/maintenance", response_model=dict, dependencies=[ApiKeyDep])
async def lessons_maintenance(session_id: str, body: dict | None = None) -> dict:
    # body: {"dry_run": true, "mode": "strict"|"loose"}
    payload = body if isinstance(body, dict) else {}
    dry_run = bool(payload.get("dry_run", True))
    mode = str(payload.get("mode", "strict") or "strict")
    res = await memory.lessons_maintenance(dry_run=dry_run, mode=mode)
    # Attach session_id for UI debugging.
    if isinstance(res, dict):
        res.setdefault("session_id", session_id)
    return res


# --- Preferences (durable key/value) ---


@router.get("/preferences", response_model=dict, dependencies=[ApiKeyDep])
async def list_preferences(
    session_id: str, scope: str = "global", prefix: str = "", limit: int = 100
) -> dict:
    sid = session_id if (scope or "").strip().lower() == "session" else None
    prefs = await memory.list_preferences(scope=scope, session_id=sid, prefix=prefix, limit=limit)
    return {"ok": True, "preferences": prefs}


@router.get("/preference", response_model=dict, dependencies=[ApiKeyDep])
async def get_preference(session_id: str, scope: str, key: str) -> dict:
    sid = session_id if (scope or "").strip().lower() == "session" else None
    p = await memory.get_preference(scope=scope, session_id=sid, key=key)
    return {"ok": True, "preference": p}


@router.post("/preference", response_model=dict, dependencies=[ApiKeyDep])
async def set_preference(session_id: str, scope: str, key: str, body: dict) -> dict:
    # body: {"value": ...}
    value = body.get("value") if isinstance(body, dict) else None
    is_locked = None
    if isinstance(body, dict) and "is_locked" in body:
        try:
            is_locked = bool(body.get("is_locked"))
        except Exception:
            is_locked = None
    sid = session_id if (scope or "").strip().lower() == "session" else None
    # API/UI writes are treated as manual and locked by default.
    return await memory.set_preference(
        scope=scope,
        session_id=sid,
        key=key,
        value=value,
        source="manual",
        is_locked=is_locked,
        updated_by="api",
    )


@router.delete("/preference", response_model=dict, dependencies=[ApiKeyDep])
async def delete_preference(session_id: str, scope: str, key: str) -> dict:
    sid = session_id if (scope or "").strip().lower() == "session" else None
    return await memory.delete_preference(scope=scope, session_id=sid, key=key)


# --- Session snapshot (STM) ---


@router.get("/snapshot", response_model=dict, dependencies=[ApiKeyDep])
async def get_snapshot(session_id: str) -> dict:
    snap = await memory.get_session_snapshot(session_id)
    return {"ok": True, "snapshot": snap or {}}


@router.post("/snapshot", response_model=dict, dependencies=[ApiKeyDep])
async def set_snapshot(session_id: str, snapshot: dict) -> dict:
    # Explicit override is allowed for power-users. Most of the time snapshots are managed by TaskManager.
    await memory.save_session_snapshot(session_id, snapshot)
    return {"ok": True}


@router.post("/episode", response_model=dict, dependencies=[ApiKeyDep])
async def add_episode(session_id: str, task_id: str | None, title: str, summary: str) -> dict:
    res = await memory.add_episode(
        session_id=session_id,
        task_id=task_id,
        title=title,
        summary=summary,
        tags=["manual"],
        data={"session_id": session_id, "task_id": task_id},
    )
    return res


@router.get("/episodes", response_model=dict, dependencies=[ApiKeyDep])
async def list_episodes(session_id: str, limit: int = 50) -> dict:
    return {"episodes": await memory.list_episodes(session_id, limit=limit)}


@router.get("/episode/{episode_id}", response_model=dict, dependencies=[ApiKeyDep])
async def get_episode(session_id: str, episode_id: str) -> dict:
    ep = await memory.get_episode(episode_id)
    if not ep:
        return {"ok": False, "error": "not found"}
    # Basic guard: don't leak episodes across sessions accidentally.
    if ep.get("session_id") != session_id:
        return {"ok": False, "error": "forbidden"}
    return {"ok": True, "episode": ep}


@router.get("/episodes_search", response_model=dict, dependencies=[ApiKeyDep])
async def search_episodes(session_id: str, q: str, limit: int = 20) -> dict:
    return {"episodes": await memory.search_episodes(session_id, q, limit=limit)}


@router.post("/consolidate", response_model=dict, dependencies=[ApiKeyDep])
async def consolidate(session_id: str, body: dict | None = None) -> dict:
    """Consolidate recent episodes into durable Lessons/Preferences.

    Body:
      {
        "dry_run": true,
        "episode_limit": 50,
        "max_lessons": 10,
        "include_preferences": true,
        "preferences_scope": "global"|"session"
      }
    """
    payload = body if isinstance(body, dict) else {}
    res = await memory.consolidate(
        session_id=session_id,
        dry_run=bool(payload.get("dry_run", True)),
        episode_limit=int(payload.get("episode_limit", 50) or 50),
        max_lessons=int(payload.get("max_lessons", 10) or 10),
        include_preferences=bool(payload.get("include_preferences", True)),
        preferences_scope=str(payload.get("preferences_scope", "global") or "global"),
    )
    return res


# --- Working memory (mutable scratchpad) ---


@router.get("/working", response_model=dict, dependencies=[ApiKeyDep])
async def get_working_memory(session_id: str) -> dict:
    wm = await memory.get_working_memory(session_id)
    return {"ok": True, "working_memory": wm or {"content": "", "updated_at": None}}


@router.post("/working", response_model=dict, dependencies=[ApiKeyDep])
async def set_working_memory(session_id: str, content: str) -> dict:
    await memory.set_working_memory(session_id, content)
    return {"ok": True}


@router.post("/working/append", response_model=dict, dependencies=[ApiKeyDep])
async def append_working_memory(session_id: str, text: str, max_chars: int = 12000) -> dict:
    res = await memory.append_working_memory(session_id, text, max_chars=max_chars)
    return {"ok": True, "working_memory": res}


@router.delete("/working", response_model=dict, dependencies=[ApiKeyDep])
async def clear_working_memory(session_id: str) -> dict:
    await memory.clear_working_memory(session_id)
    return {"ok": True}


# --- Export / Import (backup / migration) ---


@router.get("/export", response_model=dict, dependencies=[ApiKeyDep])
async def export_memory(
    session_id: str,
    include_global_preferences: bool = True,
    include_session_preferences: bool = True,
    include_lessons: bool = True,
    include_working_memory: bool = True,
    include_snapshot: bool = True,
    include_episodes: bool = True,
    redact_secrets: bool = True,
    limit_preferences: int = 10000,
    limit_lessons: int = 10000,
    limit_episodes: int = 5000,
) -> dict:
    exp = await memory.export_session_memory(
        session_id=session_id,
        include_global_preferences=include_global_preferences,
        include_session_preferences=include_session_preferences,
        include_lessons=include_lessons,
        include_working_memory=include_working_memory,
        include_snapshot=include_snapshot,
        include_episodes=include_episodes,
        redact_secrets=redact_secrets,
        limit_preferences=limit_preferences,
        limit_lessons=limit_lessons,
        limit_episodes=limit_episodes,
    )
    return {"ok": True, "export": exp}


@router.post("/import", response_model=dict, dependencies=[ApiKeyDep])
async def import_memory(
    session_id: str,
    body: dict | None = None,
    dry_run: bool = True,
    mode: str = "merge",
    allow_override_locked: bool = False,
    redact_secrets: bool = True,
) -> dict:
    payload = body if isinstance(body, dict) else {}
    export = payload.get("export") if isinstance(payload.get("export"), dict) else payload
    res = await memory.import_session_memory(
        session_id=session_id,
        export=export,
        dry_run=bool(dry_run),
        mode=str(mode or "merge"),
        allow_override_locked=bool(allow_override_locked),
        redact_secrets=bool(redact_secrets),
    )
    return res


# --- Retrieval trace (durable, last) ---


@router.get("/retrieval_trace_last", response_model=dict, dependencies=[ApiKeyDep])
async def retrieval_trace_last(session_id: str) -> dict:
    """Return the most recent retrieval trace for the session (if any).

    This is used by UI so the 'Retrieval Trace (last)' panel survives page reloads.
    """
    try:
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT ts, task_id, trace_json FROM retrieval_traces WHERE session_id=? ORDER BY ts DESC LIMIT 1",
                (session_id,),
            )
            if not row:
                return {"ok": True, "trace": None}
            trace = loads(str(row_get(row, "trace_json", "{}") or "{}"))
            return {
                "ok": True,
                "trace": trace,
                "ts": row_get(row, "ts"),
                "task_id": row_get(row, "task_id"),
            }
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
            "trace": None,
        }


@router.get("/retrieval_trace/last", response_model=dict, dependencies=[ApiKeyDep])
async def retrieval_trace_last_alias(session_id: str) -> dict:
    # Backward/alternate path alias
    return await retrieval_trace_last(session_id=session_id)


# --- Retrieval eval runs (durable history) ---


@router.post("/retrieval_eval", response_model=dict, dependencies=[ApiKeyDep])
async def save_retrieval_eval(session_id: str, body: dict | None = None) -> dict:
    """Persist a retrieval evaluation run for later comparison.

    Body (flexible):
      - query/q: str
      - top_n/topN/limit: int
      - modes: list[str]
      - metrics: dict
      - results: dict (compact hits recommended)
      - app_version: str (optional)
    """
    payload = body if isinstance(body, dict) else {}
    q = str(payload.get("query") or payload.get("q") or "").strip()
    if not q:
        return {"ok": False, "error": "bad_request", "detail": "query is required"}

    def _to_int(v, default=0):
        try:
            return int(v)
        except Exception:
            return default

    top_n = _to_int(payload.get("top_n") or payload.get("topN") or payload.get("limit") or 8, 8)
    top_n = max(1, min(30, top_n))

    modes = payload.get("modes")
    if not isinstance(modes, list):
        # infer from results keys
        res = payload.get("results")
        if isinstance(res, dict):
            modes = list(res.keys())
        else:
            modes = ["fts", "vector", "hybrid"]

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    app_version = payload.get("app_version")
    if app_version is not None:
        app_version = str(app_version)

    import datetime

    ts = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

    try:
        async with db.connect() as conn:
            cur = await conn.execute(
                "INSERT INTO retrieval_eval_runs(ts, session_id, query, top_n, modes_json, metrics_json, results_json, app_version) VALUES(?,?,?,?,?,?,?,?)",
                (
                    ts,
                    session_id,
                    q,
                    top_n,
                    dumps(modes),
                    dumps(metrics),
                    dumps(results),
                    app_version,
                ),
            )
            await conn.commit()
            new_id = getattr(cur, "lastrowid", None)
        return {"ok": True, "id": new_id, "ts": ts}
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
        }


@router.get("/retrieval_evals", response_model=dict, dependencies=[ApiKeyDep])
async def list_retrieval_evals(session_id: str, limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(200, int(limit)))
    offset = max(0, int(offset))
    try:
        async with db.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, ts, query, top_n, metrics_json, app_version FROM retrieval_eval_runs WHERE session_id=? ORDER BY ts DESC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            )
        items = []
        for r in rows or []:
            items.append(
                {
                    "id": row_get(r, "id"),
                    "ts": row_get(r, "ts"),
                    "query": row_get(r, "query"),
                    "top_n": row_get(r, "top_n"),
                    "metrics": loads(str(row_get(r, "metrics_json", "{}") or "{}")),
                    "app_version": row_get(r, "app_version"),
                }
            )
        return {"ok": True, "items": items, "limit": limit, "offset": offset}
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
            "items": [],
        }


@router.get("/retrieval_eval/{eval_id}", response_model=dict, dependencies=[ApiKeyDep])
async def get_retrieval_eval(session_id: str, eval_id: int) -> dict:
    try:
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT id, ts, session_id, query, top_n, modes_json, metrics_json, results_json, app_version FROM retrieval_eval_runs WHERE id=?",
                (int(eval_id),),
            )
            if not row:
                return {"ok": False, "error": "not_found"}
            if str(row_get(row, "session_id")) != str(session_id):
                return {"ok": False, "error": "forbidden"}
            item = {
                "id": row_get(row, "id"),
                "ts": row_get(row, "ts"),
                "query": row_get(row, "query"),
                "top_n": row_get(row, "top_n"),
                "modes": loads(str(row_get(row, "modes_json", "[]") or "[]")),
                "metrics": loads(str(row_get(row, "metrics_json", "{}") or "{}")),
                "results": loads(str(row_get(row, "results_json", "{}") or "{}")),
                "app_version": row_get(row, "app_version"),
            }
            return {"ok": True, "eval": item}
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
        }


@router.delete("/retrieval_eval/{eval_id}", response_model=dict, dependencies=[ApiKeyDep])
async def delete_retrieval_eval(session_id: str, eval_id: int) -> dict:
    try:
        async with db.connect() as conn:
            row = await fetch_one(
                conn, "SELECT session_id FROM retrieval_eval_runs WHERE id=?", (int(eval_id),)
            )
            if not row:
                return {"ok": False, "error": "not_found"}
            if str(row_get(row, "session_id")) != str(session_id):
                return {"ok": False, "error": "forbidden"}
            await conn.execute("DELETE FROM retrieval_eval_runs WHERE id=?", (int(eval_id),))
            await conn.commit()
        return {"ok": True}
    except Exception as e:
        return {
            "ok": False,
            "error": "internal_error",
            "detail": str(e),
            "request_id": request_id_ctx.get(),
        }


@router.post("/correct", response_model=dict, dependencies=[ApiKeyDep])
async def memory_correct(
    session_id: str,
    key: str,
    value: Any,
    memory_type: str = "preference",
    scope: str = "global",
) -> dict:
    """Direct correction of a memory entry by key."""
    return await memory.memory_correct(
        key=key,
        value=value,
        memory_type=memory_type,
        scope=scope,
        session_id=session_id,
    )


@router.post("/feedback", response_model=dict, dependencies=[ApiKeyDep])
async def memory_feedback(
    session_id: str,
    feedback: str,
    use_llm: bool = True,
) -> dict:
    """Process natural language feedback to correct/update memory."""
    return await memory.memory_feedback(
        feedback=feedback,
        session_id=session_id,
        use_llm=use_llm,
    )
