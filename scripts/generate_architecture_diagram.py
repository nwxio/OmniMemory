from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


@dataclass(frozen=True)
class Node:
    key: str
    title: str
    body: str
    x: float
    y: float
    w: float = 0.19
    h: float = 0.09
    fill: str = "#FFFFFF"


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    label: str = ""


NODES = [
    Node(
        key="clients",
        title="Clients",
        body="OpenCode CLI/Desktop/IDE\nMCP-compatible hosts\nAutomation scripts",
        x=0.05,
        y=0.86,
        fill="#E8F0FE",
    ),
    Node(
        key="mcp_server",
        title="MCP Server",
        body="mcp_server/server.py\nFastMCP tool surface\nResource endpoints",
        x=0.29,
        y=0.86,
        fill="#E8F0FE",
    ),
    Node(
        key="tool_wrappers",
        title="Tool Wrappers",
        body="db-ready guard\nrate limiting\nrequest metrics",
        x=0.53,
        y=0.86,
        fill="#E8F0FE",
    ),
    Node(
        key="core_orchestrator",
        title="Core Orchestrator",
        body="core/memory.py\nunified memory API\nworkflow routing",
        x=0.77,
        y=0.86,
        fill="#E8F0FE",
    ),
    Node(
        key="cross_session",
        title="Cross Session",
        body="context injection\nsession lifecycle\ntimeout + finalization",
        x=0.05,
        y=0.67,
        fill="#EAFBF3",
    ),
    Node(
        key="conversations",
        title="Conversations",
        body="message history\nsearch and stats\noldest/newest retrieval",
        x=0.29,
        y=0.67,
        fill="#EAFBF3",
    ),
    Node(
        key="retrieval",
        title="Retrieval Stack",
        body="hybrid search\nBM25 + vector\nquery expansion + rerank",
        x=0.53,
        y=0.67,
        fill="#EAFBF3",
    ),
    Node(
        key="consolidation",
        title="Consolidation",
        body="memory_consolidation\ndecay/merge/prune\nTTL cleanup",
        x=0.77,
        y=0.67,
        fill="#EAFBF3",
    ),
    Node(
        key="knowledge_base",
        title="Knowledge Base",
        body="document ingest\nfile/url/text parsers\nKB search",
        x=0.05,
        y=0.48,
        fill="#FFF6E9",
    ),
    Node(
        key="extraction",
        title="Extraction Pipeline",
        body="facts/events/people\npreferences/relations\nrules/skills/attributes",
        x=0.29,
        y=0.48,
        fill="#FFF6E9",
    ),
    Node(
        key="knowledge_graph",
        title="Knowledge Graph",
        body="triples\nneighbors + paths\nentity facts",
        x=0.53,
        y=0.48,
        fill="#FFF6E9",
    ),
    Node(
        key="safety",
        title="Safety + Reliability",
        body="circuit breaker\nfallback heuristics\naudit + security hooks",
        x=0.77,
        y=0.48,
        fill="#FFF6E9",
    ),
    Node(
        key="sqlite",
        title="Primary Storage",
        body="SQLite\nlessons/preferences/episodes\nKG + KB + vectors",
        x=0.05,
        y=0.29,
        fill="#F3EEFF",
    ),
    Node(
        key="postgres",
        title="Optional PostgreSQL",
        body="requested backend\nfallback-aware path\nfuture parity target",
        x=0.29,
        y=0.29,
        fill="#F3EEFF",
    ),
    Node(
        key="redis",
        title="Optional Redis",
        body="distributed rate limit\ncache primitives\noperational support",
        x=0.53,
        y=0.29,
        fill="#F3EEFF",
    ),
    Node(
        key="neo4j",
        title="Optional Neo4j",
        body="graph backend\nentity/relation traversal\nCypher query path",
        x=0.77,
        y=0.29,
        fill="#F3EEFF",
    ),
    Node(
        key="embeddings",
        title="Embeddings",
        body="FastEmbed (default)\nOpenAI-compatible\nmultilingual vectors",
        x=0.17,
        y=0.10,
        fill="#F5F5F5",
    ),
    Node(
        key="llm",
        title="LLM Providers",
        body="Ollama\nOpenAI/DeepSeek compatible\nprovider failover",
        x=0.43,
        y=0.10,
        fill="#F5F5F5",
    ),
    Node(
        key="quality",
        title="Quality Gates",
        body="Ruff + mypy + pytest\nquality workflow\ncoverage threshold",
        x=0.69,
        y=0.10,
        fill="#F5F5F5",
    ),
]


EDGES = [
    Edge("clients", "mcp_server", "MCP calls"),
    Edge("mcp_server", "tool_wrappers", "tool dispatch"),
    Edge("tool_wrappers", "core_orchestrator", "validated requests"),
    Edge("core_orchestrator", "cross_session"),
    Edge("core_orchestrator", "conversations"),
    Edge("core_orchestrator", "retrieval"),
    Edge("core_orchestrator", "consolidation"),
    Edge("core_orchestrator", "knowledge_base"),
    Edge("core_orchestrator", "extraction"),
    Edge("core_orchestrator", "knowledge_graph"),
    Edge("core_orchestrator", "safety"),
    Edge("retrieval", "sqlite"),
    Edge("conversations", "sqlite"),
    Edge("knowledge_base", "sqlite"),
    Edge("knowledge_graph", "sqlite"),
    Edge("knowledge_graph", "neo4j", "optional"),
    Edge("tool_wrappers", "redis", "optional"),
    Edge("core_orchestrator", "postgres", "optional"),
    Edge("retrieval", "embeddings"),
    Edge("consolidation", "llm"),
    Edge("safety", "quality"),
    Edge("core_orchestrator", "quality", "CI + local checks"),
]


def _node_map() -> dict[str, Node]:
    return {node.key: node for node in NODES}


def _draw_layer(ax, x: float, y: float, w: float, h: float, title: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=1.4,
        edgecolor="#CBD5E1",
        facecolor=fill,
        alpha=0.85,
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=11, color="#334155", weight="bold")


def _draw_node(ax, node: Node) -> None:
    patch = FancyBboxPatch(
        (node.x, node.y),
        node.w,
        node.h,
        boxstyle="round,pad=0.012,rounding_size=0.01",
        linewidth=1.2,
        edgecolor="#64748B",
        facecolor=node.fill,
    )
    ax.add_patch(patch)
    ax.text(
        node.x + 0.01,
        node.y + node.h - 0.023,
        node.title,
        fontsize=9.6,
        weight="bold",
        color="#0F172A",
    )
    ax.text(
        node.x + 0.01, node.y + node.h - 0.047, node.body, fontsize=8.2, color="#1E293B", va="top"
    )


def _draw_edge(ax, source: Node, target: Node, label: str = "") -> None:
    sx = source.x + source.w / 2.0
    sy = source.y
    tx = target.x + target.w / 2.0
    ty = target.y + target.h

    if abs(source.y - target.y) < 0.02:
        sy = source.y + source.h / 2.0
        ty = target.y + target.h / 2.0
        if source.x < target.x:
            sx = source.x + source.w
            tx = target.x
        else:
            sx = source.x
            tx = target.x + target.w

    arrow = FancyArrowPatch(
        (sx, sy),
        (tx, ty),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color="#0EA5E9",
        alpha=0.95,
        connectionstyle="arc3,rad=0.03",
    )
    ax.add_patch(arrow)

    if label:
        lx = (sx + tx) / 2.0
        ly = (sy + ty) / 2.0 + 0.01
        ax.text(lx, ly, label, fontsize=7.6, color="#0369A1", ha="center")


def generate(out_png: str, out_svg: str) -> None:
    fig, ax = plt.subplots(figsize=(18, 11), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(
        0.03, 0.965, "Memory-MCP Detailed Architecture", fontsize=19, weight="bold", color="#0F172A"
    )
    ax.text(
        0.03,
        0.942,
        "MCP interface, orchestration, retrieval, KB/KG, safety, and storage topology",
        fontsize=10.5,
        color="#475569",
    )

    _draw_layer(ax, 0.03, 0.82, 0.94, 0.15, "Interface Layer", "#F0F9FF")
    _draw_layer(ax, 0.03, 0.63, 0.94, 0.17, "Orchestration Layer", "#F0FDF4")
    _draw_layer(ax, 0.03, 0.44, 0.94, 0.17, "Domain Services Layer", "#FFF7ED")
    _draw_layer(ax, 0.03, 0.25, 0.94, 0.17, "Persistence & Infrastructure Layer", "#FAF5FF")
    _draw_layer(ax, 0.03, 0.06, 0.94, 0.16, "Provider & Quality Layer", "#F8FAFC")

    node_lookup = _node_map()
    for node in NODES:
        _draw_node(ax, node)

    for edge in EDGES:
        _draw_edge(ax, node_lookup[edge.source], node_lookup[edge.target], edge.label)

    ax.text(
        0.03,
        0.02,
        "Generated by scripts/generate_architecture_diagram.py",
        fontsize=8.5,
        color="#64748B",
    )

    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate("docs/architecture.png", "docs/architecture.svg")
