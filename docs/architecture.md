# Architecture notes

`docs/architecture.png` is a high-level visual map of the project.

It highlights:

- MCP entry points (`mcp_server/*`)
- core orchestration (`core/memory.py` and related modules)
- retrieval/knowledge graph/knowledge base subsystems
- storage and optional infrastructure backends
- reliability and safety controls

The image is intentionally simplified for public README use.

## Logical flow (text)

1. MCP-compatible clients call tools/resources exposed by `mcp_server/memory_tools.py` and `mcp_server/memory_resources.py`.
2. Requests are validated, rate-limited, and routed into core memory services.
3. Retrieval/KG/KB subsystems process and persist data.
4. Data is stored in SQLite by default, with optional external systems.
5. Reliability and safety wrappers provide fallback behavior and health visibility.
