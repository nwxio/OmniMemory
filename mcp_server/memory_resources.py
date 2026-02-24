"""MCP memory resources."""

from mcp_server.memory_tools import mcp, _ensure_db_ready


@mcp.resource("memory://lessons")
async def lessons_resource() -> str:
    """List all lessons as a resource."""
    await _ensure_db_ready()
    from core.memory import memory

    lessons = await memory.list_lessons(limit=100)
    lines = [f"# Lessons ({len(lessons)} total)\n"]
    for lesson in lessons:
        key = lesson.get("key", "")
        summary = lesson.get("lesson", "")[:100]
        lines.append(f"- **{key}**: {summary}...")
    return "\n".join(lines)


@mcp.resource("memory://preferences")
async def preferences_resource() -> str:
    """List all preferences as a resource."""
    await _ensure_db_ready()
    from core.memory import memory

    prefs = await memory.list_preferences(scope="global", session_id=None, limit=100)
    lines = [f"# Preferences ({len(prefs)} total)\n"]
    for pref in prefs:
        key = pref.get("key", "")
        value = str(pref.get("value", ""))[:80]
        locked = pref.get("is_locked", False)
        lines.append(f"- **{key}**: {value} {'🔒' if locked else ''}")
    return "\n".join(lines)


@mcp.resource("memory://health")
async def health_resource() -> str:
    """Get memory system health status."""
    await _ensure_db_ready()
    from core.memory import memory

    health = await memory.health()
    lines = ["# Memory Health\n"]
    lines.append(f"- Status: {'✅ OK' if health.get('ok') else '❌ Error'}\n")
    if "workspace" in health:
        ws = health.get("workspace", {})
        lines.append(f"- Workspace docs: {ws.get('doc_count', 0)}\n")
    if "vectors" in health:
        vec = health.get("vectors", {})
        lines.append(f"- Vectors: {vec.get('chunk_count', 0)}\n")
    if "lessons" in health:
        ls = health.get("lessons", {})
        lines.append(f"- Lessons: {ls.get('count', 0)}\n")
    return "\n".join(lines)
