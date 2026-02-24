from __future__ import annotations

from typing import Any, Dict

from plugins.base import ToolBase, ToolManifest, ToolError
from core.events import Event
from core.memory import memory


class MemoryTool(ToolBase):
    manifest = ToolManifest(
        name="memory",
        version="1.5.0",
        description="Hybrid memory: working memory + session snapshot + lessons + episodes + SQLite FTS (keyword) + Vector (semantic).",
        permissions={"read": True, "write": True},
        side_effects={"storage": "sqlite+vector"},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    async def run(self, session_id: str, task_id: str, action: str, **kwargs: Any) -> Dict[str, Any]:
        action = (action or "").strip()

        if action == "set_snapshot":
            snapshot = kwargs.get("snapshot", {}) or {}
            if not isinstance(snapshot, dict):
                raise ToolError("snapshot must be an object")
            await memory.set_snapshot(session_id, snapshot)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.set_snapshot"}))
            return {"ok": True}

        if action == "get_snapshot":
            snap = await memory.get_snapshot(session_id)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.get_snapshot"}))
            return {"snapshot": snap}

        if action == "get_working_memory":
            wm = await memory.get_working_memory(session_id)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.get_working_memory"}))
            return {"working_memory": wm or {"content": "", "updated_at": None}}

        if action == "set_working_memory":
            content = str(kwargs.get("content", "") or "")
            await memory.set_working_memory(session_id, content)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.set_working_memory"}))
            return {"ok": True}

        if action == "append_working_memory":
            text = str(kwargs.get("text", "") or "")
            max_chars = int(kwargs.get("max_chars", 12000))
            res = await memory.append_working_memory(session_id, text, max_chars=max_chars)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.append_working_memory", "chars": len(text)}))
            return {"working_memory": res}

        if action == "clear_working_memory":
            await memory.clear_working_memory(session_id)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.clear_working_memory"}))
            return {"ok": True}

        if action == "add_lesson":
            key = str(kwargs.get("key", "") or "").strip()
            lesson = str(kwargs.get("lesson", "") or "").strip()
            meta = kwargs.get("meta")
            if not key or not lesson:
                raise ToolError("key and lesson are required")
            await memory.add_lesson(key, lesson, meta=meta if isinstance(meta, dict) else None)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.add_lesson", "key": key}))
            return {"ok": True}

        if action == "list_lessons":
            limit = int(kwargs.get("limit", 50))
            lessons = await memory.list_lessons(limit=limit)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.list_lessons", "limit": limit, "count": len(lessons)}))
            return {"lessons": lessons}
        if action == "set_preference":
            scope = str(kwargs.get("scope", "global") or "global").strip().lower()
            key = str(kwargs.get("key", "") or "").strip()
            value = kwargs.get("value")
            if not key:
                raise ToolError("key is required")
            src = str(kwargs.get("source", "auto") or "auto").strip().lower()
            if src not in ("auto", "manual", "system"):
                src = "auto"
            is_locked = kwargs.get("is_locked") if "is_locked" in kwargs else None
            updated_by = str(kwargs.get("updated_by", "tool:memory") or "tool:memory")

            res = await memory.set_preference(
                scope=scope,
                session_id=session_id if scope == "session" else None,
                key=key,
                value=value,
                source=src,
                is_locked=is_locked,
                updated_by=updated_by,
            )
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.set_preference", "scope": scope, "key": key}))
            return res

        if action == "get_preference":
            scope = str(kwargs.get("scope", "global") or "global").strip().lower()
            key = str(kwargs.get("key", "") or "").strip()
            pref = await memory.get_preference(scope=scope, session_id=session_id if scope == "session" else None, key=key)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.get_preference", "scope": scope, "key": key, "ok": bool(pref)}))
            return {"preference": pref}

        if action == "list_preferences":
            scope = str(kwargs.get("scope", "global") or "global").strip().lower()
            prefix = str(kwargs.get("prefix", "") or "").strip()
            limit = int(kwargs.get("limit", 100))
            prefs = await memory.list_preferences(scope=scope, session_id=session_id if scope == "session" else None, prefix=prefix, limit=limit)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.list_preferences", "scope": scope, "prefix": prefix, "limit": limit, "count": len(prefs)}))
            return {"preferences": prefs}

        if action == "delete_preference":
            scope = str(kwargs.get("scope", "global") or "global").strip().lower()
            key = str(kwargs.get("key", "") or "").strip()
            res = await memory.delete_preference(scope=scope, session_id=session_id if scope == "session" else None, key=key)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.delete_preference", "scope": scope, "key": key, "deleted": res.get("deleted")}))
            return res


        if action == "add_episode":
            title = str(kwargs.get("title", "") or "").strip()
            summary = str(kwargs.get("summary", "") or "").strip()
            tags = kwargs.get("tags")
            data = kwargs.get("data")
            if not title or not summary:
                raise ToolError("title and summary are required")
            res = await memory.add_episode(
                session_id=session_id,
                task_id=task_id,
                title=title,
                summary=summary,
                tags=tags if isinstance(tags, list) else None,
                data=data if isinstance(data, dict) else None,
            )
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.add_episode", "episode_id": res.get("episode_id")}))
            return res

        if action == "list_episodes":
            limit = int(kwargs.get("limit", 50))
            eps = await memory.list_episodes(session_id, limit=limit)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.list_episodes", "limit": limit, "count": len(eps)}))
            return {"episodes": eps}

        if action == "get_episode":
            episode_id = str(kwargs.get("episode_id", "") or "").strip()
            ep = await memory.get_episode(episode_id)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.get_episode", "episode_id": episode_id, "ok": bool(ep)}))
            # Caller must validate session if needed
            return {"episode": ep}

        if action == "search_episodes":
            query = str(kwargs.get("query", "") or "").strip()
            limit = int(kwargs.get("limit", 20))
            eps = await memory.search_episodes(session_id, query, limit=limit)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.search_episodes", "query": query, "limit": limit, "count": len(eps)}))
            return {"episodes": eps}

        if action == "index_project":
            root_path = str(kwargs.get("root_path", "/workspace") or "/workspace")
            res = await memory.index_project(root_path)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.index_project", "root": root_path, **res}))
            return res

        if action == "search":
            query = str(kwargs.get("query", "") or "")
            limit = int(kwargs.get("limit", 5))
            hits = await memory.search(query, limit=limit)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.search", "query": query, "limit": limit, "hits": len(hits)}))
            return {"hits": [h.__dict__ for h in hits]}

        if action == "index_project_vectors":
            root_path = str(kwargs.get("root_path", "/workspace") or "/workspace")
            res = await memory.index_project_vectors(root_path)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.index_project_vectors", "root": root_path, **res}))
            return res

        if action == "semantic_search":
            query = str(kwargs.get("query", "") or "")
            limit = int(kwargs.get("limit", 5))
            hits = await memory.semantic_search(query, limit=limit)
            await self.event_bus.publish(Event(type="trace_event", session_id=session_id, task_id=task_id, data={"tool": "memory.semantic_search", "query": query, "limit": limit, "hits": len(hits)}))
            return {"hits": [h.__dict__ for h in hits]}

        raise ToolError(f"Unknown action: {action}")
