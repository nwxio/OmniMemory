# Memory-MCP

A production-friendly memory platform with an MCP server interface.

`memory-mcp` combines structured memory, semantic retrieval, knowledge graph operations,
cross-session context, and safety controls in a self-hosted package.

## Why this project

- Works as an MCP backend for coding agents and assistants.
- Supports durable memory primitives (lessons, preferences, procedures, entities, relations).
- Includes search, extraction, consolidation, and quality/safety checks out of the box.
- Can run fully local (SQLite + local embeddings) or with PostgreSQL/Redis.

## Key capabilities

- Hybrid memory search: keyword + semantic retrieval.
- Cross-session memory and automatic context injection.
- Knowledge graph with triples, neighbor traversal, and path discovery.
- Auto-extraction pipeline for facts/events/preferences/relations/rules/skills.
- Document knowledge base (file/url/text ingestion + content search).
- Conversation history storage and retrieval.
- Procedural memory (how-to steps) and semantic entity graph.
- Memory lifecycle controls: TTL cleanup, decay/merge/prune consolidation.
- Reliability controls: circuit breaker, fallback mode, rate limiting, health endpoint.
- Multilingual heuristics for `ru/uk/en` with universal Unicode-safe token handling.

## Architecture overview

Core components:

- `core/memory.py`: high-level memory orchestration.
- `core/memory_sqlite.py`: storage layer and memory operations.
- `core/search/*`: BM25/hybrid retrieval, expansion, reranking.
- `core/knowledge_graph.py` + `core/graph_db/neo4j_backend.py`: graph operations.
- `core/knowledge_base.py`: KB documents and search.
- `core/cross_session.py`: cross-session lifecycle and context bundles.
- `mcp_server/memory_tools.py`: MCP tool surface.
- `mcp_server/memory_resources.py`: MCP resources.

## Installation

Requirements:

- Python `>=3.11`

Install:

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Quick start

### Option 1: Local mode (SQLite, default)

```bash
cp .env.local .env
python -m mcp_server.server
```

### Option 2: Docker infra (PostgreSQL + Redis)

```bash
./docker-compose.sh start
cp .env.docker .env
python -m mcp_server.server
```

Docker details: `docker/README.md`

Environment presets: `ENV_CONFIGS.md`

## Configuration highlights

Common environment values:

```bash
# Database
OMNIMIND_DB_TYPE=sqlite
OMNIMIND_DB_PATH=./memory.db

# Optional postgres/redis mode
OMNIMIND_DB_TYPE=postgres
OMNIMIND_POSTGRES_HOST=localhost
OMNIMIND_POSTGRES_PORT=5442
OMNIMIND_POSTGRES_DB=memory
OMNIMIND_POSTGRES_USER=memory_user
OMNIMIND_POSTGRES_PASSWORD=***
OMNIMIND_REDIS_ENABLED=true

# Embeddings
OMNIMIND_EMBEDDINGS_PROVIDER=fastembed
OMNIMIND_EMBEDDINGS_FASTEMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Optional Neo4j backend for knowledge graph
OMNIMIND_NEO4J_ENABLED=true
OMNIMIND_NEO4J_URI=bolt://localhost:7687
OMNIMIND_NEO4J_USER=neo4j
OMNIMIND_NEO4J_PASSWORD=***
OMNIMIND_NEO4J_DATABASE=neo4j
```

## MCP tools

The server exposes the following MCP tools.

### Memory search and storage

- `memory_search`
- `memory_search_lessons`
- `memory_search_preferences`
- `memory_search_all`
- `memory_upsert`
- `memory_get`
- `memory_list`
- `memory_delete`
- `memory_index_workspace`
- `memory_health`
- `memory_ttl_cleanup`
- `memory_metrics`

### Memory consolidation and correction

- `memory_consolidate`
- `memory_consolidate_decay`
- `memory_consolidation_status`
- `memory_correct`
- `memory_feedback`

### Procedural and semantic memory

- `memory_add_procedure`
- `memory_get_procedure`
- `memory_search_procedures`
- `memory_add_entity`
- `memory_search_entities`
- `memory_add_relation`
- `memory_get_relations`

### Cross-session memory

- `cross_session_start`
- `cross_session_message`
- `cross_session_tool_use`
- `cross_session_stop`
- `cross_session_end`
- `cross_session_context`
- `cross_session_search`
- `cross_session_stats`
- `cross_session_check_timeout`

### Conversation memory

- `conversation_add_message`
- `conversation_get_messages`
- `conversation_get_messages_asc`
- `conversation_search`
- `conversation_stats`

### Knowledge base

- `kb_add_document`
- `kb_add_document_from_file`
- `kb_add_document_from_url`
- `kb_get_document`
- `kb_list_documents`
- `kb_search_documents`
- `kb_delete_document`
- `kb_stats`

### Knowledge graph

- `kg_add_triple`
- `kg_get_triples`
- `kg_get_neighbors`
- `kg_find_path`
- `kg_search_entities`
- `kg_get_entity_facts`
- `kg_stats`

### Extraction pipeline

- `extract_memories`
- `get_extracted_memories`
- `search_extracted_memories`
- `extraction_stats`

## MCP resources

- `memory://lessons`
- `memory://preferences`
- `memory://health`

## Usage examples

### Example: procedural + semantic memory

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

    procedure = await memory_get_procedure("deploy.web")
    print(procedure)

    service = await memory_add_entity("web-api", "service", {"lang": "python"})
    database = await memory_add_entity("postgres", "database", {"engine": "postgres"})
    await memory_add_relation(service["id"], "uses", database["id"], {"critical": True})

    relations = await memory_get_relations(service["id"])
    print(relations)


asyncio.run(demo())
```

### Example: knowledge graph operations

```python
import asyncio
from mcp_server.memory_tools import kg_add_triple, kg_get_neighbors, kg_find_path


async def demo_kg() -> None:
    await kg_add_triple("Alice", "works_for", "Acme", confidence=0.95, source_type="text")
    await kg_add_triple("Acme", "located_in", "Kyiv", confidence=0.9, source_type="text")

    neighbors = await kg_get_neighbors("Alice", direction="both", limit=20)
    print("neighbors:", neighbors)

    path = await kg_find_path("Alice", "Kyiv", max_depth=3)
    print("path:", path)


asyncio.run(demo_kg())
```

## Reliability and safety

- Per-tool rate limiting.
- LLM circuit breaker with fallback behavior.
- Health snapshots with dependency status.
- Security/audit helpers in `core/security`.
- CI quality gates for lint, focused typing checks, and tests.

## CI quality gates

Workflow: `.github/workflows/quality.yml`

- Ruff checks for critical modules.
- Focused mypy gate on runtime-critical paths.
- Full test suite with coverage threshold.
- Postgres fallback behavior check.

## Development workflow

```bash
# Lint
ruff check .

# Tests
python -m pytest tests -q

# Focused mypy gate (same as CI)
python -m mypy core/security/audit.py core/security/gdpr.py core/search/bm25.py core/search/hybrid.py core/llm/client.py core/health/monitor.py core/knowledge_graph.py core/graph_db/neo4j_backend.py mcp_server/memory_tools.py --ignore-missing-imports --follow-imports=skip
```

## Related docs

- Environment presets: `ENV_CONFIGS.md`
- Docker deployment: `docker/README.md`
- Install notes: `INSTALL.md`

## License

Use according to your repository license policy.
