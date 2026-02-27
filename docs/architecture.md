# Architecture notes

The architecture diagram is generated in two formats:

- `docs/architecture.png` (high-resolution raster)
- `docs/architecture.svg` (vector)

It covers the full end-to-end flow:

- client entry points and MCP surface
- tool wrappers and orchestration
- retrieval / KB / KG / extraction subsystems
- safety and reliability controls
- storage and optional infrastructure backends
- provider and quality layers

Detailed data relationships and memory domain map:

- `docs/memory-data-model.md`

## Generate or refresh diagram

Install optional diagram libraries:

```bash
python3 -m pip install matplotlib networkx
```

Regenerate files:

```bash
python3 scripts/generate_architecture_diagram.py
```

## Logical flow (text)

1. MCP-compatible clients call tools/resources in `mcp_server/memory_tools.py` and `mcp_server/memory_resources.py`.
2. Tool wrappers enforce DB readiness, rate limits, and request metrics.
3. `core/memory.py` orchestrates retrieval, KB, KG, extraction, session, and correction workflows.
4. Subsystems persist data to SQLite by default, with optional PostgreSQL/Redis/Neo4j paths.
5. Reliability and safety layers provide circuit breaking, fallback behavior, and health visibility.
6. Temporal KG operations persist current state (`kg_triples`) and event history (`kg_triple_events` / `KGTemporalEvent`) for as-of reasoning.
