# Memory-MCP

Memory system with MCP server - Better than Memori and Mem0.

## Features

- **Hybrid Search**: FTS (keyword) + Vector (semantic) + MMR deduplication + LLM reranking
- **TTL Support**: Auto-expiry for lessons and episodes
- **Local Embeddings**: FastEmbed (no external API required)
- **Knowledge Graph**: Semantic triples with graph traversal and inference
- **Auto-Extraction**: 8 memory types extracted automatically (facts, events, people, preferences, relationships, rules, skills, attributes)
- **MCP Server**: Exposes memory as MCP tools and resources
- **Cross-Session Memory**: Persistent context across sessions with auto-injection
- **Memory Consolidation**: Decay, merge, and prune old memories
- **Conversations**: Chat history storage with search
- **Knowledge Base**: Document parsing (MD, TXT, PDF, DOCX) and storage
- **LangChain Integration**: Memory and Retriever compatible with LangChain
- **Async Scheduler**: Background task scheduling
- **100% Self-Hosted**: No cloud API required (unlike Memori)

## Quick Start

### Вариант 1: SQLite (по умолчанию)

```bash
# Install dependencies
pip install -e .

# Run MCP server (stdio)
python -m mcp_server.server

# Or with HTTP transport
python -m mcp_server.server --transport http --port 8080
```

### Вариант 2: PostgreSQL + Redis (Docker)

```bash
# Запустить PostgreSQL + Redis
./docker-compose.sh start

# Настроить окружение
cp .env.docker .env

# Запустить MCP server
python -m mcp_server.server
```

См. [`docker/README.md`](docker/README.md) для деталей.

## Environment Variables

```bash
# Database
OMNIMIND_DB_PATH=./memory.db

# TTL (days)
OMNIMIND_MEMORY_LESSONS_TTL_DAYS=90
OMNIMIND_MEMORY_EPISODES_TTL_DAYS=60
OMNIMIND_MEMORY_PREFERENCES_TTL_DAYS=180

# Cross-session memory
OMNIMIND_CROSS_SESSION_ENABLED=true
OMNIMIND_CROSS_SESSION_MAX_CONTEXT_TOKENS=2000

# Memory consolidation (decay/merge/prune)
OMNIMIND_MEMORY_DECAY_ENABLED=true
OMNIMIND_MEMORY_DECAY_FACTOR=0.9
OMNIMIND_MEMORY_DECAY_PERIOD_DAYS=30
OMNIMIND_MEMORY_MERGE_ENABLED=true
OMNIMIND_MEMORY_PRUNE_ENABLED=true
OMNIMIND_MEMORY_PRUNE_MAX_AGE_DAYS=180

# Vector settings (multilingual: RU, EN, DE, FR + 50+)
OMNIMIND_VECTOR_MEMORY_ENABLED=true
OMNIMIND_VECTOR_MEMORY_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Graph settings
OMNIMIND_GRAPH_MEMORY_ENABLED=true

# Embeddings provider: fastembed (default), ollama, openai
OMNIMIND_EMBEDDINGS_PROVIDER=fastembed

# Reranking (uses LLM)
OMNIMIND_SEARCH_RERANK_ENABLED=true
```

## MCP Tools

### Core Memory
- `memory_search` - Hybrid search across memory
- `memory_upsert` - Save lesson/preference
- `memory_get` - Get memory entry
- `memory_list` - List lessons/preferences
- `memory_delete` - Delete entry
- `memory_consolidate` - Consolidate episodes into lessons
- `memory_consolidate_decay` - Run decay/merge/prune consolidation
- `memory_consolidation_status` - Get consolidation settings
- `memory_index_workspace` - Index workspace files
- `memory_health` - Get system health

### Cross-Session Memory
- `cross_session_start` - Start session with context injection
- `cross_session_message` - Record message
- `cross_session_tool_use` - Record tool use
- `cross_session_stop` - Finalize session
- `cross_session_end` - End session
- `cross_session_context` - Get context for prompt
- `cross_session_search` - Search across sessions
- `cross_session_stats` - Get statistics

### Conversations
- `conversation_add_message` - Add chat message
- `conversation_get_messages` - Get messages (newest first)
- `conversation_get_messages_asc` - Get messages (oldest first)
- `conversation_search` - Search messages
- `conversation_stats` - Get statistics

### Knowledge Base
- `kb_add_document` - Add document
- `kb_add_document_from_file` - Parse and add file
- `kb_add_document_from_url` - Fetch and add URL
- `kb_get_document` - Get document
- `kb_list_documents` - List documents
- `kb_search_documents` - Search documents
- `kb_delete_document` - Delete document
- `kb_stats` - Get statistics

### Procedural / Skill Memory
- `memory_add_procedure` - Save procedural memory (how-to)
- `memory_get_procedure` - Get a procedural memory entry
- `memory_search_procedures` - Search procedural entries
- `memory_list_procedures` - List all procedures
- `memory_delete_procedure` - Delete procedure

### Semantic Memory
- `memory_add_entity` - Add semantic entity
- `memory_search_entities` - Search semantic entities
- `memory_add_relation` - Add semantic relation
- `memory_get_relations` - Get semantic relations for entity

### System
- `memory_metrics` - Runtime metrics snapshot

### Knowledge Graph
- `kg_add_triple` - Add semantic triple (subject, predicate, object)
- `kg_get_triples` - Query triples from knowledge graph
- `kg_get_neighbors` - Get neighboring entities
- `kg_find_path` - Find path between two entities
- `kg_search_entities` - Search entities in knowledge graph
- `kg_get_entity_facts` - Get all facts about an entity
- `kg_stats` - Get knowledge graph statistics

### Memory Extraction (Auto-Augmentation)
- `extract_memories` - Extract 8 memory types from text automatically
- `get_extracted_memories` - Get extracted memories
- `search_extracted_memories` - Search extracted memories
- `extraction_stats` - Get extraction statistics

## Reliability and Safety

- **LLM Circuit Breaker**: automatic cooldown on repeated provider failures
- **Fallback Mode**: deterministic summarization/consolidation when LLM fails
- **Rate Limiting**: per-tool request throttling (local + Redis fallback)
- **Audit + GDPR**: audit log table and export/delete/anonymize workflows
- **Health Snapshot**: `memory_health` includes runtime dependency status

## Quality Gates (CI)

The repository includes a CI workflow (`.github/workflows/quality.yml`) with:

- Ruff lint for changed/core modules
- Focused mypy type gate for critical runtime paths
- Full test suite (`pytest tests -v`)

## Example: Procedural + Semantic Memory

```python
import asyncio
from mcp_server.memory_tools import (
    memory_add_procedure,
    memory_get_procedure,
    memory_add_entity,
    memory_add_relation,
    memory_get_relations,
)


async def demo() -> None:
    await memory_add_procedure(
        key="deploy.web",
        title="Deploy web service",
        steps=["Build image", "Run migrations", "Restart service"],
        metadata={"owner": "devops"},
    )

    p = await memory_get_procedure("deploy.web")
    print(p)

    a = await memory_add_entity("web-api", "service", {"lang": "python"})
    b = await memory_add_entity("postgres", "database", {"engine": "postgres"})
    await memory_add_relation(a["id"], "uses", b["id"], {"critical": True})

    rels = await memory_get_relations(a["id"])
    print(rels)


asyncio.run(demo())
```

## MCP Resources

- `memory://lessons` - List all lessons
- `memory://lessons/{key}` - Get specific lesson
- `memory://preferences` - List preferences
- `memory://preferences/{key}` - Get specific preference
- `memory://health` - System health
- `memory://session/{session_id}/episodes` - Session episodes
- `memory://search/{query}` - Search results

## Architecture

```
memory/
├── core/              # Memory core (from OmniMind)
│   ├── memory.py      # Main API
│   ├── memory_sqlite.py
│   ├── vector_memory.py
│   ├── graph_memory.py
│   ├── embeddings.py  # FastEmbed/Ollama/OpenAI
│   ├── cross_session.py  # Cross-session memory
│   ├── memory_decay.py   # Memory consolidation
│   ├── conversations.py  # Chat history
│   ├── knowledge_base.py # Document storage
│   ├── doc_parser.py    # Document parsing
│   ├── langchain_integration.py # LangChain compatibility
│   └── async_scheduler.py # Background tasks
├── plugins/           # AI tool interface
├── api/              # HTTP endpoints
└── mcp_server/       # MCP server
    ├── server.py
    ├── memory_tools.py
    └── memory_resources.py
```

## LangChain Integration

```python
from core.langchain_integration import get_memory, get_retriever

# Get LangChain-compatible memory
memory = get_memory(session_id="my-session", limit=10)
vars = await memory.aload_memory_variables()

# Get LangChain-compatible retriever
retriever = get_retriever(
    session_id="my-session",
    source="all",  # lessons, preferences, knowledge, all
    limit=5
)
docs = await retriever.ainvoke("search query")
```

## Async Scheduler

```python
from core.async_scheduler import get_scheduler, TaskPriority

scheduler = get_scheduler()

# Schedule background task
task_id = await scheduler.schedule(
    name="index_documents",
    func=my_indexing_func,
    priority=TaskPriority.NORMAL
)

# Check status
task = await scheduler.get_task(task_id)
```
