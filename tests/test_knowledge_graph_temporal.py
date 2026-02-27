from __future__ import annotations

import asyncio
from uuid import uuid4


def _run(coro):
    return asyncio.run(coro)


def test_temporal_policy_single_active_works_for() -> None:
    from core.db import init_db
    from core.knowledge_graph import KnowledgeGraph

    async def _case() -> None:
        await init_db()
        kg = KnowledgeGraph()

        suffix = uuid4().hex[:10]
        subject = f"Alice-{suffix}"
        object_a = f"Acme-{suffix}"
        object_b = f"Contoso-{suffix}"

        await kg.upsert_fact(
            subject=subject,
            predicate="works_for",
            object_name=object_a,
            action="assert",
            observed_at="2026-01-01T10:00:00+00:00",
        )
        await kg.upsert_fact(
            subject=subject,
            predicate="works_for",
            object_name=object_b,
            action="assert",
            observed_at="2026-01-02T10:00:00+00:00",
        )

        active = await kg.get_triples(subject=subject, predicate="works_for", limit=10)
        assert len(active) == 1
        assert active[0]["object"] == object_b

        as_of_old = await kg.get_triples_as_of(
            as_of="2026-01-01T12:00:00+00:00",
            subject=subject,
            predicate="works_for",
            limit=10,
        )
        assert len(as_of_old) == 1
        assert as_of_old[0]["object"] == object_a

        history = await kg.get_fact_history(subject=subject, predicate="works_for", limit=20)
        actions = {str(item.get("action")) for item in history}
        assert "assert" in actions
        assert "close_replaced" in actions

    _run(_case())


def test_temporal_policy_multi_active_likes() -> None:
    from core.db import init_db
    from core.knowledge_graph import KnowledgeGraph

    async def _case() -> None:
        await init_db()
        kg = KnowledgeGraph()

        suffix = uuid4().hex[:10]
        subject = f"Bob-{suffix}"
        tea = f"Tea-{suffix}"
        coffee = f"Coffee-{suffix}"

        await kg.upsert_fact(
            subject=subject,
            predicate="likes",
            object_name=tea,
            action="assert",
            observed_at="2026-01-03T10:00:00+00:00",
        )
        await kg.upsert_fact(
            subject=subject,
            predicate="likes",
            object_name=coffee,
            action="assert",
            observed_at="2026-01-04T10:00:00+00:00",
        )

        active = await kg.get_triples(subject=subject, predicate="likes", limit=10)
        objects = {str(item.get("object")) for item in active}
        assert tea in objects
        assert coffee in objects
        assert len(active) == 2

    _run(_case())


def test_temporal_path_as_of_respects_retract() -> None:
    from core.db import init_db
    from core.knowledge_graph import KnowledgeGraph

    async def _case() -> None:
        await init_db()
        kg = KnowledgeGraph()

        suffix = uuid4().hex[:10]
        a = f"A-{suffix}"
        b = f"B-{suffix}"
        c = f"C-{suffix}"

        await kg.upsert_fact(
            subject=a,
            predicate="works_for",
            object_name=b,
            action="assert",
            observed_at="2026-01-05T10:00:00+00:00",
        )
        await kg.upsert_fact(
            subject=b,
            predicate="part_of",
            object_name=c,
            action="assert",
            observed_at="2026-01-05T10:10:00+00:00",
        )

        before_retract = await kg.find_path_as_of(
            from_entity=a,
            to_entity=c,
            as_of="2026-01-05T11:00:00+00:00",
            max_depth=3,
        )
        assert before_retract is not None
        assert before_retract.length >= 2

        await kg.upsert_fact(
            subject=b,
            predicate="part_of",
            object_name=c,
            action="retract",
            observed_at="2026-01-06T10:00:00+00:00",
        )

        after_retract = await kg.find_path_as_of(
            from_entity=a,
            to_entity=c,
            as_of="2026-01-06T12:00:00+00:00",
            max_depth=3,
        )
        assert after_retract is None

    _run(_case())
