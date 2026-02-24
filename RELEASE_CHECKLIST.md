# Release checklist

Use this checklist before creating a tagged release.

## 1) Scope and versioning

- [ ] Confirm release scope (feature, fix, patch level).
- [ ] Update version in `pyproject.toml`.
- [ ] Review changelog/release notes content.

## 2) Quality gates

- [ ] `ruff check .`
- [ ] `python -m pytest tests -q`
- [ ] Focused mypy gate:

```bash
python -m mypy core/security/audit.py core/security/gdpr.py core/search/bm25.py core/search/hybrid.py core/llm/client.py core/health/monitor.py core/knowledge_graph.py core/graph_db/neo4j_backend.py mcp_server/memory_tools.py --ignore-missing-imports --follow-imports=skip
```

- [ ] CI workflow green on target branch.

## 3) Documentation

- [ ] `README.md` reflects current features and MCP tool surface.
- [ ] `ENV_CONFIGS.md` reflects current env presets.
- [ ] `docker/README.md` matches current compose stack.
- [ ] New features include usage example(s).

## 4) Security and operational checks

- [ ] No secrets committed (`.env`, keys, credentials).
- [ ] Default credentials reviewed for docs and examples.
- [ ] `SECURITY.md` contact/reporting instructions are valid.
- [ ] Backup/restore paths validated for current deployment mode.

## 5) Packaging and release metadata

- [ ] License file present (`LICENSE`).
- [ ] Contribution guide present (`CONTRIBUTING.md`).
- [ ] Release notes summarize: what changed, migration notes, risks.
- [ ] Tag created using semantic versioning (for example `v0.2.0`).

## 6) Post-release

- [ ] Verify installation from clean environment.
- [ ] Verify MCP server startup command: `python -m mcp_server.server`.
- [ ] Smoke test core tools (`memory_health`, `memory_search`, `kg_stats`).
- [ ] Announce release with links to docs and changelog.
