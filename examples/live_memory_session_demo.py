import argparse
import asyncio
import json
from uuid import uuid4

from mcp_server.memory_tools import (
    conversation_add_message,
    conversation_search,
    cross_session_context,
    cross_session_message,
    cross_session_start,
    cross_session_stop,
    extract_memories,
    get_extracted_memories,
    kb_add_document,
    kb_search_documents,
    kg_add_triple,
    kg_find_path,
    kg_get_neighbors,
    memory_add_entity,
    memory_add_procedure,
    memory_add_relation,
    memory_feedback,
    memory_get_procedure,
    memory_get_relations,
    memory_health,
    memory_list,
    memory_metrics,
    memory_search_all,
    memory_upsert,
)


def show_step(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def run_short(session_id: str) -> None:
    show_step("1) Store durable memory (preference + lesson)")
    pref = await memory_upsert(
        key="style.answer_format",
        value="Use concise bullet points and practical examples.",
        type="preference",
    )
    lesson = await memory_upsert(
        key="release.tests_first",
        value="Run full tests before release and verify health endpoints.",
        type="lesson",
        meta={"source": "live_demo", "priority": "high"},
    )
    show_json({"preference": pref, "lesson": lesson})

    show_step("2) Search across lessons and preferences")
    combined = await memory_search_all("tests release concise", limit=5)
    show_json(combined)

    show_step("3) Apply natural-language feedback")
    feedback = await memory_feedback(
        feedback="I don't like long answers, I prefer concise responses.",
        session_id=session_id,
        use_llm=False,
    )
    show_json(feedback)

    show_step("4) Show current preferences snapshot")
    prefs = await memory_list(type="preferences", limit=10)
    show_json(prefs)

    show_step("5) Health + runtime metrics")
    health = await memory_health()
    metrics = await memory_metrics()
    show_json({"health": health, "metrics": metrics})


async def run_extended(session_id: str) -> None:
    show_step("1) Store durable memory (preference + lesson)")
    pref = await memory_upsert(
        key="style.answer_format",
        value="Use concise bullet points and practical examples.",
        type="preference",
    )
    lesson = await memory_upsert(
        key="release.tests_first",
        value="Run full tests before release and verify health endpoints.",
        type="lesson",
        meta={"source": "live_demo", "priority": "high"},
    )
    show_json({"preference": pref, "lesson": lesson})

    show_step("2) Search across lessons and preferences")
    combined = await memory_search_all("tests release concise", limit=5)
    show_json(combined)

    show_step("3) Apply natural-language feedback")
    feedback = await memory_feedback(
        feedback="I don't like long answers, I prefer concise responses.",
        session_id=session_id,
        use_llm=False,
    )
    show_json(feedback)

    show_step("4) Show current preferences snapshot")
    prefs = await memory_list(type="preferences", limit=10)
    show_json(prefs)

    show_step("5) Add searchable knowledge-base content")
    kb_doc = await kb_add_document(
        title="Deployment checklist",
        content=(
            "Release checklist: run tests, build image, deploy staging, "
            "run smoke tests, verify health, promote to production."
        ),
        source_type="text",
        session_id=session_id,
    )
    kb_hits = await kb_search_documents(
        "deploy staging smoke tests", session_id=session_id, limit=5
    )
    show_json({"kb_add_document": kb_doc, "kb_search_documents": kb_hits})

    show_step("6) Work with procedural + semantic memory")
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

    service = await memory_add_entity("backend-api", "service", {"language": "python"})
    database = await memory_add_entity("postgres", "database", {"version": "15"})
    await memory_add_relation(service["id"], "uses", database["id"], {"critical": True})
    relations = await memory_get_relations(service["id"], limit=10)
    show_json({"procedure": procedure, "relations": relations})

    show_step("7) Work with knowledge graph")
    triple_1 = await kg_add_triple(
        subject="Memory-MCP",
        predicate="integrates_with",
        object_name="OpenCode",
        confidence=0.99,
        source_type="text",
        session_id=session_id,
    )
    triple_2 = await kg_add_triple(
        subject="OpenCode",
        predicate="uses",
        object_name="MCP",
        confidence=0.95,
        source_type="text",
        session_id=session_id,
    )
    neighbors = await kg_get_neighbors("OpenCode", direction="both", limit=20)
    path = await kg_find_path("Memory-MCP", "MCP", max_depth=3)
    show_json({"triples": [triple_1, triple_2], "neighbors": neighbors, "path": path})

    show_step("8) Auto extraction and conversation memory")
    extracted = await extract_memories(
        text="OpenCode uses Memory-MCP for durable context. Release flow includes tests and smoke checks.",
        session_id=session_id,
    )
    extracted_hits = await get_extracted_memories(session_id=session_id, limit=10)

    await conversation_add_message(
        session_id=session_id,
        role="user",
        content="Please remember that production release requires smoke checks.",
    )
    conv_hits = await conversation_search(session_id=session_id, query="smoke checks", limit=5)
    show_json(
        {
            "extract_memories": extracted,
            "get_extracted_memories": extracted_hits,
            "conversation_search": conv_hits,
        }
    )

    show_step("9) Cross-session context injection")
    started = await cross_session_start(
        session_id=session_id,
        user_prompt="We are preparing release notes and need prior context.",
    )
    await cross_session_message(
        session_id=session_id,
        content="Remember that production releases must include smoke tests.",
        role="user",
    )
    context = await cross_session_context(
        user_prompt="Give me release advice with the remembered rules.",
        max_tokens=1200,
    )
    stopped = await cross_session_stop(session_id=session_id)
    show_json({"start": started, "context": context, "stop": stopped})

    show_step("10) Health + runtime metrics")
    health = await memory_health()
    metrics = await memory_metrics()
    show_json({"health": health, "metrics": metrics})


async def main(mode: str) -> None:
    session_id = f"recording-demo-{mode}-{uuid4().hex[:8]}"

    show_step(f"OpenCode-style memory session ({mode})")
    print(f"Session ID: {session_id}")
    print("Tip: keep this terminal visible while recording.")

    if mode == "short":
        await run_short(session_id)
    else:
        await run_extended(session_id)

    print("\nDemo completed.")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live Memory-MCP demo session")
    parser.add_argument(
        "--mode",
        choices=["short", "extended"],
        default="extended",
        help="Demo mode for recording",
    )
    args = parser.parse_args()
    asyncio.run(main(args.mode))
