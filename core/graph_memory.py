from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .db import db

logger = logging.getLogger("GraphMemory")

# Optional ast-scope for enhanced graph analysis
_ast_scope_available = False
try:
    import ast_scope
    _ast_scope_available = True
except ImportError:
    logger.info("ast-scope not installed. Using basic AST parsing for graph memory.")


def _node_id(node_type: str, label: str) -> str:
    h = hashlib.sha256(f"{node_type}:{label}".encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"gn_{h}"


def _edge_key(source_id: str, target_id: str, relation: str) -> str:
    return f"{source_id}|{relation}|{target_id}"


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    meta: Optional[dict[str, Any]] = None


class GraphMemory:
    """Lightweight associative memory over workspace artifacts.

    Design goals:
    - Deterministic IDs (so re-indexing is idempotent)
    - Cheap ingestion on writes (post-commit)
    - Helpful retrieval summaries ("this file imports X", "symbol Y defined here")

    This is intentionally *not* a full code graph / LSP. It's a pragmatic
    connection layer that improves navigation and recall.
    """

    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = str(workspace_root or "/workspace")

    def _abs(self, rel: str) -> Path:
        rel = rel.lstrip("/")
        return Path(self.workspace_root) / rel

    async def upsert_node(
        self,
        *,
        label: str,
        node_type: str,
        content: str | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        label = str(label or "").strip()
        node_type = str(node_type or "concept").strip().lower() or "concept"
        nid = _node_id(node_type, label)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        async with db.connect() as conn:
            await conn.execute(
                """
                INSERT INTO graph_nodes(id, label, type, content, metadata_json, created_at, last_accessed)
                VALUES(?,?,?,?,?,datetime('now'),datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                  label=excluded.label,
                  type=excluded.type,
                  content=excluded.content,
                  metadata_json=excluded.metadata_json,
                  last_accessed=datetime('now')
                """,
                (nid, label, node_type, content or None, meta_json),
            )
            await conn.commit()
        return nid

    async def upsert_edge(self, edge: GraphEdge) -> None:
        meta_json = json.dumps(edge.meta or {}, ensure_ascii=False)
        async with db.connect() as conn:
            await conn.execute(
                """
                INSERT INTO graph_edges(source_id, target_id, relation, weight, metadata_json, created_at)
                VALUES(?,?,?,?,?,datetime('now'))
                ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                  weight=excluded.weight,
                  metadata_json=excluded.metadata_json
                """,
                (edge.source, edge.target, edge.relation, float(edge.weight), meta_json),
            )
            await conn.commit()

    async def ingest_paths(
        self,
        paths: Iterable[str],
        *,
        root: str | None = None,
        deny_patterns: Optional[list[str]] = None,
        deny_globs: Optional[list[str]] = None,
        max_files: int = 200,
        max_bytes_per_file: int = 300_000,
        max_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Ingest a list of workspace-relative paths into the graph.

        Back/forward compatible API:
        - Older callers pass `deny_globs` (Path.match-style)
        - Newer callers pass `deny_patterns` (fnmatch-style, same as workspace FTS deny list)
        - `max_seconds` allows cheap best-effort ingestion during post-write hooks.

        This function must *never* hard-fail tool workflows; it returns {ok: True/False, ...}.
        """
        import fnmatch
        import time

        base = Path(root or self.workspace_root).expanduser().resolve()

        # Defaults: keep it conservative.
        deny_globs = deny_globs or [
            '**/.git/**',
            '**/node_modules/**',
            '**/.venv/**',
            '**/__pycache__/**',
            '**/.idea/**',
        ]
        deny_patterns = deny_patterns or []

        def _denied(rel: str) -> bool:
            rel2 = rel.lstrip('/').replace('\\', '/')
            # First: fnmatch deny patterns (workspace-style)
            for pat in deny_patterns:
                try:
                    p0 = (pat or '').lstrip('/')
                    if not p0:
                        continue
                    if fnmatch.fnmatch(rel2, p0) or fnmatch.fnmatch('/' + rel2, p0):
                        return True
                except Exception:
                    continue
            # Second: Path.match globs
            try:
                pp = Path(rel2)
                for g in deny_globs or []:
                    try:
                        if pp.match(g):
                            return True
                    except Exception:
                        continue
            except Exception:
                pass
            return False

        t0 = time.monotonic()
        count = 0
        edges = 0
        nodes = 0
        notes: list[str] = []

        for rel in paths or []:
            if max_files and count >= max_files:
                notes.append(f'limit_reached(max_files={max_files})')
                break
            if max_seconds is not None and max_seconds > 0 and (time.monotonic() - t0) >= float(max_seconds):
                notes.append(f'limit_reached(max_seconds={max_seconds})')
                break

            rel = str(rel or '').lstrip('/').replace('\\', '/')
            if not rel:
                continue
            if _denied(rel):
                continue

            p_abs = (base / rel).resolve()
            # Prevent path escape.
            if not str(p_abs).startswith(str(base)):
                continue

            try:
                if not p_abs.exists() or not p_abs.is_file():
                    continue
                size = p_abs.stat().st_size
                if size <= 0 or size > max_bytes_per_file:
                    continue

                try:
                    raw = p_abs.read_text(encoding='utf-8', errors='replace')
                except Exception:
                    continue

                file_node = await self.upsert_node(label=rel, node_type='file', content=None, metadata={'size': size})
                nodes += 1

                file_edges, file_nodes = await self._ingest_file_content(rel, raw, file_node)
                edges += file_edges
                nodes += file_nodes
                count += 1
            except Exception as e:
                notes.append(f'{rel}: {e}')
                continue

        return {
            'ok': True,
            'files': count,
            'nodes': nodes,
            'edges': edges,
            'notes': notes[:20],
        }

    async def _ingest_file_content(self, rel: str, text: str, file_node_id: str) -> tuple[int, int]:
        ext = Path(rel).suffix.lower()
        edges = 0
        nodes = 0

        # common concepts: directory/package
        pkg = str(Path(rel).parent).replace("\\", "/")
        if pkg and pkg != ".":
            pkg_id = await self.upsert_node(label=pkg, node_type="folder")
            nodes += 1
            await self.upsert_edge(GraphEdge(source=file_node_id, target=pkg_id, relation="in_folder", weight=1.0))
            edges += 1

        if ext == ".py":
            e, n = await self._ingest_python(rel, text, file_node_id)
            edges += e
            nodes += n
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            e, n = await self._ingest_js(rel, text, file_node_id)
            edges += e
            nodes += n
        else:
            # cheap keyword concepts for non-code
            for kw in self._top_keywords(text, limit=6):
                kid = await self.upsert_node(label=kw, node_type="keyword")
                nodes += 1
                await self.upsert_edge(GraphEdge(source=file_node_id, target=kid, relation="mentions", weight=0.5))
                edges += 1

        return edges, nodes

    async def _ingest_python(self, rel: str, text: str, file_node_id: str) -> tuple[int, int]:
        edges = 0
        nodes = 0
        try:
            tree = ast.parse(text)
        except Exception:
            return 0, 0

        defined: list[str] = []
        imports: list[str] = []
        function_calls: list[str] = []
        
        # Use ast-scope for enhanced analysis if available
        if _ast_scope_available:
            try:
                scope_info = ast_scope.annotate(tree)
                
                # Get static dependency graph (imports, defines, references)
                graph = scope_info.static_dependency_graph()
                
                # Extract nodes from graph
                for node in graph.nodes:
                    if hasattr(node, 'name'):
                        defined.append(node.name)
                    if hasattr(node, 'module'):
                        mod = getattr(node, 'module', None)
                        if mod:
                            imports.append(mod)
                
                # Get references (function calls, variable uses)
                for node in graph.nodes:
                    if hasattr(node, 'references'):
                        for ref in node.references:
                            if hasattr(ref, 'name'):
                                function_calls.append(ref.name)
            except Exception as e:
                logger.debug(f"ast-scope analysis failed: {e}")
        
        # Fallback to basic AST walking
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name not in defined:
                    defined.append(node.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name and a.name not in imports:
                        imports.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level and node.level > 0:
                    mod = "." * node.level + mod
                if mod and mod not in imports:
                    imports.append(mod)
            # Track function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                    if call_name not in function_calls:
                        function_calls.append(call_name)

        # Store defined symbols (functions, classes)
        for name in sorted(set(defined))[:40]:
            sid = await self.upsert_node(label=name, node_type="symbol")
            nodes += 1
            await self.upsert_edge(GraphEdge(source=file_node_id, target=sid, relation="defines", weight=1.0))
            edges += 1

        # Store imports
        for mod in sorted(set(imports))[:60]:
            mid = await self.upsert_node(label=mod, node_type="module")
            nodes += 1
            await self.upsert_edge(GraphEdge(source=file_node_id, target=mid, relation="imports", weight=1.0))
            edges += 1

            # Map module -> file in workspace
            target_rel = self._module_to_relpath(rel, mod)
            if target_rel:
                fid = await self.upsert_node(label=target_rel, node_type="file")
                nodes += 1
                await self.upsert_edge(GraphEdge(source=file_node_id, target=fid, relation="imports_file", weight=0.9, meta={"module": mod}))
                edges += 1
        
        # Store function calls (references)
        for call in sorted(set(function_calls))[:30]:
            if call not in defined:  # Don't create self-reference edges
                cid = await self.upsert_node(label=call, node_type="function_call")
                nodes += 1
                await self.upsert_edge(GraphEdge(source=file_node_id, target=cid, relation="calls", weight=0.7))
                edges += 1

        return edges, nodes

    def _module_to_relpath(self, current_rel: str, mod: str) -> Optional[str]:
        # Only try for absolute module names without leading dots.
        if not mod or mod.startswith("."):
            return None
        root = Path(self.workspace_root)
        cand = Path(mod.replace(".", "/") + ".py")
        if (root / cand).exists():
            return str(cand).replace("\\", "/")
        # package __init__.py
        cand2 = Path(mod.replace(".", "/") + "/__init__.py")
        if (root / cand2).exists():
            return str(cand2).replace("\\", "/")
        return None

    async def _ingest_js(self, rel: str, text: str, file_node_id: str) -> tuple[int, int]:
        edges = 0
        nodes = 0

        # import ... from 'x' | require('x')
        mods = set()
        for m in re.finditer(r"\bfrom\s+['\"]([^'\"]+)['\"]", text):
            mods.add(m.group(1))
        for m in re.finditer(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)", text):
            mods.add(m.group(1))

        # exported symbols (very rough)
        syms = set()
        for m in re.finditer(r"\bexport\s+(?:default\s+)?(?:function|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            syms.add(m.group(1))

        for s in sorted(syms)[:40]:
            sid = await self.upsert_node(label=s, node_type="symbol")
            nodes += 1
            await self.upsert_edge(GraphEdge(source=file_node_id, target=sid, relation="defines", weight=0.9))
            edges += 1

        for mname in sorted(mods)[:60]:
            mid = await self.upsert_node(label=mname, node_type="module")
            nodes += 1
            await self.upsert_edge(GraphEdge(source=file_node_id, target=mid, relation="imports", weight=0.8))
            edges += 1

        return edges, nodes

    def _top_keywords(self, text: str, *, limit: int = 8) -> list[str]:
        # extremely simple keywords: identifiers & headings, stopword-lite
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", text)
        stop = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "return",
            "import",
            "const",
            "function",
            "class",
            "async",
            "await",
            "true",
            "false",
            "none",
            "null",
        }
        freq: dict[str, int] = {}
        for w in words:
            lw = w.lower()
            if lw in stop:
                continue
            freq[lw] = freq.get(lw, 0) + 1
        items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return [k for k, _ in items[: max(0, int(limit))]]

    async def query_summary(self, query: str, *, limit: int = 20) -> str:
        q = (query or "").strip().lower()
        if not q:
            return ""
        toks = [t for t in re.findall(r"[A-Za-z0-9_\-\.]{3,}", q)][:8]
        if not toks:
            return ""

        # Find candidate nodes by label LIKE any token.
        like = " OR ".join(["lower(label) LIKE ?" for _ in toks])
        params = [f"%{t.lower()}%" for t in toks]
        async with db.connect() as conn:
            rows = await conn.execute_fetchall(
                f"SELECT id,label,type FROM graph_nodes WHERE {like} ORDER BY last_accessed DESC LIMIT ?",
                (*params, int(limit)),
            )
            nodes = [(r[0], r[1], r[2]) for r in rows]

            lines: list[str] = []
            seen = set()
            for nid, label, ntype in nodes:
                if len(lines) >= limit:
                    break
                key = f"{ntype}:{label}"
                if key in seen:
                    continue
                seen.add(key)

                # neighbors (outgoing)
                erows = await conn.execute_fetchall(
                    """
                    SELECT e.relation, n2.type, n2.label, e.weight
                    FROM graph_edges e
                    JOIN graph_nodes n2 ON n2.id = e.target_id
                    WHERE e.source_id=?
                    ORDER BY e.weight DESC, n2.type ASC
                    LIMIT 8
                    """,
                    (nid,),
                )
                if not erows:
                    continue

                parts = []
                for rel, ttype, tlabel, w in erows:
                    parts.append(f"{rel}->{ttype}:{tlabel}")
                lines.append(f"- {ntype}:{label} :: " + ", ".join(parts))

        return "\n".join(lines)


_graph: Optional[GraphMemory] = None


def graph_memory() -> GraphMemory:
    global _graph
    if _graph is None:
        # late import to avoid settings cycles
        from .config import settings

        _graph = GraphMemory(settings.workspace)
    return _graph
