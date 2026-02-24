import asyncio

from mcp_server.memory_tools import (
    memory_add_entity,
    memory_add_procedure,
    memory_add_relation,
    memory_get_procedure,
    memory_get_relations,
    memory_metrics,
    memory_search,
)


async def main() -> None:
    # Procedural memory (how-to)
    await memory_add_procedure(
        key="release.backend",
        title="Release backend service",
        steps=[
            "Run test suite",
            "Build container image",
            "Deploy to staging",
            "Run smoke checks",
            "Promote to production",
        ],
        metadata={"team": "platform", "risk": "medium"},
    )

    procedure = await memory_get_procedure("release.backend")
    print("Procedure:", procedure)

    # Semantic memory (entities + relations)
    service = await memory_add_entity("backend-api", "service", {"language": "python"})
    db = await memory_add_entity("postgres", "database", {"version": "15"})
    await memory_add_relation(service["id"], "uses", db["id"], {"critical": True})

    relations = await memory_get_relations(service["id"])
    print("Relations:", relations)

    # Retrieval
    hits = await memory_search("release backend production", limit=5)
    print("Search hits:", hits)

    # Runtime metrics
    stats = await memory_metrics()
    print("Metrics:", stats)


if __name__ == "__main__":
    asyncio.run(main())
