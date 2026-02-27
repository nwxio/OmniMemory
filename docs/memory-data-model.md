# Memory data model and relationships

This document describes what is persisted, how components depend on each other,
and how temporal graph reasoning works.

## 1) Storage layers

- Relational core (`SQLite` by default, optional `PostgreSQL` parity).
- Optional cache/rate-limit layer (`Redis`).
- Optional graph backend (`Neo4j`) for knowledge graph operations.

## 2) Core memory domains

### Durable memory

- `lessons`: validated long-lived findings (key/value + metadata + timestamps).
- `preferences`: stable user preferences/constraints, optionally scoped and locked.

### Session and operational memory

- `episodes`: session events used by consolidation pipelines.
- working/session memory tables: short-lived context for active workflows.
- `conversation_messages`: ordered conversation stream with role/model/tokens.

### Retrieval and knowledge

- `memory_docs`: workspace/document content slices for keyword retrieval.
- vector chunk/meta tables: semantic embedding index for hybrid retrieval.
- knowledge base document tables (`kb_*` domain).

### Knowledge graph

- `kg_subjects`, `kg_predicates`, `kg_objects`.
- `kg_triples`: current fact state.
- `kg_triple_events`: append-only temporal event stream.

## 3) Key relationships between domains

- Conversation -> Episodes: user/assistant/tool activity is captured into episodes.
- Episodes -> Lessons/Preferences: consolidation extracts durable knowledge.
- Memory docs + vectors -> Hybrid search: keyword and semantic results are fused.
- Knowledge graph current state (`kg_triples`) is derived from event history (`kg_triple_events`).
- Cross-session context combines durable memory + session traces under token budget.

## 4) Temporal knowledge graph model

## Current fact state

Each row in `kg_triples` represents the latest state of a fact:

- identifiers: subject/predicate/object IDs
- temporal state: `valid_from`, `valid_to`, `is_active`
- versioning: `version`, `last_event_type`
- provenance: `source_type`, `source_id`, `session_id`, metadata

## Event stream

Each change is recorded in `kg_triple_events`:

- `action`: `assert`, `retract`, `close_replaced`
- `observed_at`: event timestamp
- state snapshot fields: active flag, version, validity interval
- provenance and metadata copied for auditability

This allows:

- as-of queries (state at timestamp T)
- history browsing
- temporal path reasoning based on facts valid at T

## Policy C (default)

- `single_active` predicates: one active object per `(subject, predicate)`.
- `multi_active` predicates: multiple active objects allowed.
- Default single-active set is configured via:
  - `OMNIMIND_KG_TEMPORAL_SINGLE_ACTIVE_PREDICATES`
  - default value: `works_for,belongs_to,prefers`

## 5) MCP tool mapping

### Core memory

- `memory_search*`, `memory_upsert/get/list/delete`
- consolidation and correction tools

### Graph and temporal graph

- classic graph: `kg_add_triple`, `kg_get_triples`, `kg_get_neighbors`, `kg_find_path`
- temporal graph: `kg_upsert_fact`, `kg_get_triples_as_of`, `kg_get_fact_history`, `kg_find_path_as_of`
- aggregation: `kg_get_entity_timeline_summary`

### KB and extraction

- `kb_*` tools for document memory
- `extract_memories` and extracted-memory search/stat tools

## 6) Runtime behavior and compatibility

- Existing classic KG tools remain available and backward-compatible.
- `kg_add_triple` maps to temporal `assert` behavior under the hood.
- If Neo4j is enabled and available, temporal operations use Neo4j event nodes
  (`KGTemporalEvent`) for parity with relational temporal reasoning.
