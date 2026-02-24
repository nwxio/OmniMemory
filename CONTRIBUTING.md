# Contributing

Thanks for considering a contribution to Memory-MCP.

## Development setup

```bash
git clone git@github.com:nwxio/OmniMemory.git
cd OmniMemory
pip install -e .[dev]
```

## Local checks before opening a PR

```bash
ruff check .
python -m pytest tests -q
python -m mypy core/security/audit.py core/security/gdpr.py core/search/bm25.py core/search/hybrid.py core/llm/client.py core/health/monitor.py core/knowledge_graph.py core/graph_db/neo4j_backend.py mcp_server/memory_tools.py --ignore-missing-imports --follow-imports=skip
```

## Pull request guidelines

- Keep changes focused and scoped.
- Include tests for behavior changes.
- Preserve backward-compatible behavior unless explicitly discussed.
- Update documentation (`README.md`, `ENV_CONFIGS.md`, `docker/README.md`) when relevant.
- Use clear commit messages explaining why the change is needed.

## Code style

- Python: Ruff + project defaults.
- Prefer explicit, readable code over clever shortcuts.
- Keep comments concise and in English.

## Reporting issues

When opening an issue, include:

- what you expected
- what actually happened
- reproducible steps
- environment details (`python version`, DB mode, relevant env vars)
