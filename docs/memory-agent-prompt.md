## Memory (Memory MCP) - Critical

### Required Behavior

Memory must be used by the agent automatically and proactively:

- Detect when memory context is relevant to the current task
- Retrieve only what is needed (minimum sufficient context)
- Decide what to save (only long-lived, reusable facts/rules)
- Update memory when the user changes a rule or preference
- Use memory as part of the standard workflow, not only on request

Additionally (required):

- Always apply source priority: `current user request > locked preferences > recent verified lessons > older lessons`
- If memory conflicts with the current user request, follow the current request and update memory
- Do not pollute memory: do not store one-off steps, temporary paths, raw logs, secrets, tokens, passwords
- Check for duplicates by `key` and meaning before writing (use upsert, do not create near-duplicate entries)
- If memory fails, do not block task execution: continue work, keep pending lessons, write them after recovery

### Required Checks

| When | Action |
|------|--------|
| Session start | `memory_memory_health()` (on error: 2 retries with backoff, then degraded mode) |
| Greeting | `memory_memory_search_preferences(query="greeting")` |
| Before task | `memory_memory_search_lessons(query="keywords")` + if needed `memory_memory_search_preferences(query="topic")` |
| User changes rule | Immediately `memory_memory_upsert(type="preference")` with the same key (upsert) |
| After task | `memory_memory_upsert(type="lesson")` only for confirmed long-lived findings |
| Session end | `memory_memory_consolidate(session_id=..., dry_run=false)` |
| Periodic maintenance | `memory_memory_ttl_cleanup(dry_run=false)` |

### Main Rule

**All relevant memory must be applied.**

If there is a conflict, use this order:

1. Current explicit user instruction
2. Locked preference
3. Recent verified lesson
4. Older lesson

When conflicts happen, update the outdated memory entry.

### Memory Tools

| Tool | Purpose |
|------|---------|
| `memory_memory_search` | Search workspace files (FTS + vectors) |
| `memory_memory_search_lessons` | Search lessons by keywords |
| `memory_memory_search_preferences` | Search preferences by keywords |
| `memory_memory_search_all` | Search both lessons and preferences |
| `memory_memory_list` | List all lessons or preferences |
| `memory_memory_upsert` | Save a lesson or preference |
| `memory_memory_get` | Get one entry by key |
| `memory_memory_delete` | Delete an entry |
| `memory_memory_consolidate` | Consolidate episodes into lessons |
| `memory_memory_index_workspace` | Index workspace files |
| `memory_memory_health` | Memory system health status |
| `memory_memory_ttl_cleanup` | Cleanup expired entries |

### Data Types

| Type | Description | TTL |
|------|-------------|-----|
| `lesson` | Lessons, findings, patterns | 90 days |
| `preference` | User preferences | 180 days |
| `episode` | Session episodes | 60 days |

### What to Save / What Not to Save

Save:

- Stable user preferences (style, format, constraints)
- Reusable technical lessons (fixes, runbooks, root cause)
- Verified project agreements (contracts, compatibility rules)

Do not save:

- Temporary artifacts and one-off actions
- Unverified hypotheses
- Secrets, tokens, keys, personal data unless explicitly required

### Write Quality Standards

- `key`: stable, snake_case, no date prefix, scoped by domain
- `value`: 1-3 sentences, explicit when/what/why
- `meta`: minimum `{"context": "...", "confidence": "high|medium|low", "source": "user|system|test"}`
- Per completed task: usually 0-2 lessons (no more than 3 unless necessary)

### Save Example

```python
memory_memory_upsert(
    key="unique_key",
    value="Lesson or preference text",
    type="lesson",  # or "preference"
    meta={"context": "..."}
)
```
