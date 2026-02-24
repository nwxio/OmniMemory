import asyncio
from uuid import uuid4


def _run(coro):
    return asyncio.run(coro)


def test_audit_logger_roundtrip():
    from core.db import init_db
    from core.security.audit import audit_logger

    async def _case():
        await init_db()
        await audit_logger.log(action="test_action", resource_type="unit", resource_id="r1")
        rows = await audit_logger.get_logs(action="test_action", limit=5)
        assert isinstance(rows, list)
        assert any(r.get("action") == "test_action" for r in rows)

    _run(_case())


def test_gdpr_export_works():
    from core.db import init_db
    from core.security.gdpr import gdpr

    async def _case():
        await init_db()
        data = await gdpr.export_user_data("session-does-not-exist")
        assert isinstance(data, dict)
        assert "sessions" in data
        assert "episodes" in data
        assert "preferences" in data

    _run(_case())


def test_procedural_memory_sets_id():
    from core.db import init_db, db
    from core.memory_sqlite import memory_sql

    async def _case():
        await init_db()
        await memory_sql.add_procedure("proc_quality", "Quality", ["step1"])
        async with db.connect() as conn:
            cur = await conn.execute(
                "SELECT id FROM procedural_memory WHERE key = ?", ("proc_quality",)
            )
            row = await cur.fetchone()
            await cur.close()
        assert row is not None
        assert bool(row["id"])

    _run(_case())


def test_new_hybrid_search_bm25_scores_present():
    from core.search.hybrid import hybrid_search

    async def _case():
        fts = [
            {"path": "a.py", "text": "python async code", "score": 0.5, "meta": {}},
            {"path": "b.py", "text": "banana apple", "score": 0.4, "meta": {}},
        ]
        out = await hybrid_search.search("python", fts_results=fts, vector_results=[], limit=5)
        assert len(out) >= 1
        assert any(float(r.get("bm25_score", 0.0)) > 0 for r in out)

    _run(_case())


def test_llm_client_exported_from_package():
    from core.llm import llm_client

    assert llm_client is not None


def test_health_monitor_snapshot_contains_runtime_dependencies():
    from core.health import health_monitor

    async def _case():
        snap = await health_monitor.snapshot()
        assert isinstance(snap, dict)
        assert "db_backend" in snap
        assert "llm" in snap
        assert "redis" in snap

    _run(_case())


def test_llm_circuit_breaker_opens_on_repeated_failures():
    from core.llm.client import llm_client

    class _FailingProvider:
        def is_available(self):
            return True

        async def complete(self, *args, **kwargs):
            raise RuntimeError("forced failure")

    async def _case():
        original_provider = llm_client._provider
        original_init = llm_client._provider_initialized
        llm_client._provider = _FailingProvider()
        llm_client._provider_initialized = True
        llm_client._breaker.failures = 0
        llm_client._breaker.open_until_ts = 0.0

        # Trigger failures (fallback is enabled, so complete() won't raise).
        for _ in range(3):
            _ = await llm_client.complete("breaker test", use_cache=False)

        assert llm_client._breaker.open_until_ts > 0.0

        llm_client._provider = original_provider
        llm_client._provider_initialized = original_init

    _run(_case())


def test_knowledge_graph_duplicate_triple_reuses_id_and_mentions():
    from core.db import init_db, db
    from core.knowledge_graph import KnowledgeGraph

    async def _case():
        await init_db()
        kg = KnowledgeGraph()
        suffix = uuid4().hex[:10]

        subject = f"kg-subject-{suffix}"
        predicate = "related_to"
        object_name = f"kg-object-{suffix}"

        first = await kg.add_triple(subject, predicate, object_name)
        second = await kg.add_triple(subject, predicate, object_name)

        assert first["triple_id"] == second["triple_id"]

        triples = await kg.get_triples(
            subject=subject,
            predicate=predicate,
            object_name=object_name,
            limit=5,
        )
        assert len(triples) == 1

        async with db.connect() as conn:
            cur = await conn.execute(
                "SELECT mention_count FROM kg_subjects WHERE name = ?",
                (subject,),
            )
            subject_row = await cur.fetchone()
            await cur.close()

            cur = await conn.execute(
                "SELECT mention_count FROM kg_objects WHERE name = ?",
                (object_name,),
            )
            object_row = await cur.fetchone()
            await cur.close()

        assert subject_row is not None
        assert object_row is not None
        assert int(subject_row["mention_count"]) == 2
        assert int(object_row["mention_count"]) == 2

    _run(_case())
