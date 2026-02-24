from __future__ import annotations

import json
import re

from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .memory import memory, MemoryHit
from .vector_memory import vector_memory
from .graph_memory import graph_memory


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _dedup_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate hits while preserving order.

    We dedup by (source, path, normalized text preview).

    Why the preview? Some hit types carry long expanded excerpts; without
    clamping the dedupe key, we can keep near-duplicates that differ only
    in tail text. Conversely, clamping prevents huge keys and keeps the
    operation cheap.
    """

    key_chars = int(getattr(settings, "retrieval_dedup_key_chars", 800) or 800)

    def _norm_preview(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        # Normalize whitespace so minor formatting differences don't bypass dedupe.
        s = " ".join(s.split())
        if key_chars > 0 and len(s) > key_chars:
            s = s[:key_chars]
        return s

    seen = set()
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = str(h.get("source") or "")
        path = str(h.get("path") or "")
        text = _norm_preview(str(h.get("text") or ""))
        key = (src, path, text)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_Ѐ-ӿ]{2,}", re.UNICODE)


def _query_tokens(q: str, *, limit: int = 12) -> List[str]:
    """Extract a small set of query tokens for best-effort matching in docs.

    This is intentionally simple and robust across languages (Latin/Cyrillic).
    """
    q = (q or "").strip()
    if not q:
        return []
    toks = [m.group(0) for m in _WORD_RE.finditer(q)]
    # Prefer longer tokens first; keep order stable for ties.
    toks2 = sorted(enumerate(toks), key=lambda it: (-len(it[1]), it[0]))
    uniq: List[str] = []
    seen = set()
    for _, t in toks2:
        tl = t.casefold()
        if tl in seen:
            continue
        seen.add(tl)
        uniq.append(t)
        if len(uniq) >= int(limit):
            break
    return uniq


def _rewrite_workspace_query(q: str) -> Tuple[str, Optional[str]]:
    """Best-effort rewrite of the query for workspace retrieval.

    Goal: make FTS/hybrid retrieval more stable by focusing on the high-signal tokens
    (works for Cyrillic/Latin) and dropping filler words/punctuation.

    Returns: (query_for_search, note)
    """
    original = (q or "").strip()
    if not original:
        return ("", None)

    if not bool(getattr(settings, "retrieval_workspace_query_rewrite", True)):
        return (original, None)

    min_toks = int(getattr(settings, "retrieval_workspace_query_rewrite_min_tokens", 3) or 3)
    max_toks = int(getattr(settings, "retrieval_workspace_query_rewrite_max_tokens", 12) or 12)
    toks = _query_tokens(original, limit=max_toks)
    if len(toks) < min_toks:
        return (original, None)

    rewritten = " ".join(toks).strip()
    if not rewritten:
        return (original, None)

    if rewritten.casefold() == original.casefold():
        return (original, None)

    # Only accept rewrite if it does not become ridiculously short.
    if len(rewritten) < 6:
        return (original, None)

    return (rewritten, f"workspace_query_rewrite: {rewritten}")


def _looks_like_code_path(path: str) -> bool:
    p = (path or "").lower()
    return any(
        p.endswith(s)
        for s in (
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
            ".sh",
            ".bash",
            ".zsh",
            ".md",
            ".css",
            ".html",
        )
    )


def _best_match_index(text: str, tokens: List[str]) -> int:
    """Return best-effort match index for any token in text, else 0."""
    if not text:
        return 0
    if not tokens:
        return 0
    t = text.casefold()
    best = None
    for tok in tokens:
        i = t.find(tok.casefold())
        if i < 0:
            continue
        if best is None or i < best:
            best = i
    return int(best or 0)


def _clip_window(text: str, center: int, window_chars: int) -> Tuple[int, int]:
    n = len(text)
    if n <= 0:
        return (0, 0)
    wc = max(200, int(window_chars))
    half = wc // 2
    start = max(0, int(center) - half)
    end = min(n, start + wc)
    # If we clipped at the end, shift start back to keep window size.
    start = max(0, end - wc)
    return (start, end)


def _format_excerpt(
    text: str,
    *,
    path: str,
    start: int,
    end: int,
    with_line_numbers: bool,
) -> str:
    """Format excerpt from [start:end], aligned to line boundaries."""
    if not text:
        return ""
    start = max(0, int(start))
    end = min(len(text), int(end))
    if end <= start:
        return ""

    # Align to line boundaries (best effort)
    ls = text.rfind("\n", 0, start)
    if ls >= 0:
        start = ls + 1
    le = text.find("\n", end)
    if le >= 0:
        end = le

    chunk = text[start:end]
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    if not with_line_numbers:
        # Keep it compact
        return chunk.strip()

    # Compute line numbers
    # Note: splitting huge text is expensive; keep it bounded.
    prefix = text[:start].replace("\r\n", "\n").replace("\r", "\n")
    start_line = prefix.count("\n") + 1
    lines = chunk.split("\n")

    # Clamp extremely long excerpts
    max_lines = int(getattr(settings, "retrieval_workspace_excerpt_max_lines", 80) or 80)
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("…")

    width = len(str(start_line + max(0, len(lines) - 1)))
    out_lines: List[str] = []
    for i, ln in enumerate(lines):
        num = str(start_line + i).rjust(width)
        out_lines.append(f"{num}: {ln}")
    return "\n".join(out_lines).rstrip()


async def _expand_fts_hit(query: str, path: str) -> Optional[Dict[str, Any]]:
    """Expand an FTS hit into a more useful excerpt with surrounding context."""
    try:
        doc = await memory.get_doc(path)
    except Exception:
        doc = None
    if not doc:
        return None

    content = str(doc.get("content") or "")
    if not content.strip():
        return None

    # Hard clamp: memory_docs may contain big files; don't blow prompts.
    hard = int(getattr(settings, "retrieval_workspace_hard_max_chars", 350_000) or 350_000)
    if hard > 0 and len(content) > hard:
        content = content[:hard]

    toks = _query_tokens(query, limit=12)
    idx = _best_match_index(content, toks)
    win = int(getattr(settings, "retrieval_workspace_excerpt_chars", 1200) or 1200)
    start, end = _clip_window(content, idx, win)
    with_ln = bool(
        getattr(settings, "retrieval_workspace_line_numbers", True)
    ) and _looks_like_code_path(path)
    excerpt = _format_excerpt(content, path=path, start=start, end=end, with_line_numbers=with_ln)
    if not excerpt:
        return None
    return {
        "path": path,
        "excerpt": excerpt,
        "lines": None,  # kept for future; excerpt already embeds line numbers when enabled
    }


async def _expand_vector_hit(path: str, chunk_index: Optional[int]) -> Optional[Dict[str, Any]]:
    """Expand a vector hit using neighboring chunks (no filesystem reads)."""
    try:
        ci = int(chunk_index) if chunk_index is not None else None
    except Exception:
        ci = None
    if ci is None:
        return None
    radius = int(getattr(settings, "retrieval_vector_neighbor_radius", 1) or 1)
    max_chars = int(getattr(settings, "retrieval_vector_excerpt_chars", 1400) or 1400)
    try:
        bundle = await vector_memory.get_neighbor_text(
            path=path, chunk_index=ci, radius=radius, max_chars=max_chars
        )
    except Exception:
        return None
    if not bundle or not str(bundle.get("text") or "").strip():
        return None
    return bundle


def format_context(hits: List[Dict[str, Any]], max_total_chars: Optional[int] = None) -> str:
    """Pack retrieval hits into a compact context string.

    Features:
    - Optional round-robin packing by *path* to avoid one file dominating the prompt.
    - Optional per-path and per-source caps (count of blocks), to keep diversity.
    - Respects the existing split budgets (memory vs workspace) when enabled.

    Deterministic and cheap: it only reshuffles already-selected hits.
    """

    if max_total_chars is None:
        max_total_chars = int(getattr(settings, "retrieval_context_max_chars", 4000) or 4000)
    max_total_chars = max(200, int(max_total_chars))

    split = bool(getattr(settings, "retrieval_context_split_budgets", True))
    mem_budget = int(getattr(settings, "retrieval_context_memory_chars", 0) or 0)
    ws_budget = int(getattr(settings, "retrieval_context_workspace_chars", 0) or 0)

    if split:
        if mem_budget <= 0 or ws_budget <= 0:
            mem_budget = int(max_total_chars * 0.35)
            ws_budget = max_total_chars - mem_budget
        else:
            total = mem_budget + ws_budget
            if total < max_total_chars:
                ws_budget += max_total_chars - total
            elif total > max_total_chars:
                scale = float(max_total_chars) / float(total)
                mem_budget = max(100, int(mem_budget * scale))
                ws_budget = max(100, max_total_chars - mem_budget)
    else:
        mem_budget = max_total_chars
        ws_budget = max_total_chars

    memory_sources = {"prefs", "wm", "stm", "lessons", "episodes"}
    workspace_sources = {"fts", "vector", "hybrid"}

    mem_hits = [h for h in hits if str(h.get("source") or "") in memory_sources]
    ws_hits = [h for h in hits if str(h.get("source") or "") in workspace_sources]

    # Future-proof: any other sources go with the memory bucket.
    other_hits = [h for h in hits if h not in mem_hits and h not in ws_hits]
    mem_hits = mem_hits + other_hits

    rr_enabled = bool(getattr(settings, "retrieval_pack_round_robin", True))
    per_path_cap = int(getattr(settings, "retrieval_pack_per_path_cap", 2) or 2)
    per_source_cap = int(getattr(settings, "retrieval_pack_per_source_cap", 0) or 0)

    # Normalize caps: 0 means unlimited.
    if per_path_cap < 0:
        per_path_cap = 0
    if per_source_cap < 0:
        per_source_cap = 0

    def _block_for_hit(h: Dict[str, Any]) -> str:
        path = str(h.get("path", ""))
        source = str(h.get("source", ""))
        score = h.get("score")
        txt = str(h.get("text", ""))
        return f"[{source} score={score} path={path}]\n{txt}\n"

    def _pack_sequential(bucket: List[Dict[str, Any]], budget: int) -> tuple[str, int]:
        parts: List[str] = []
        total = 0
        per_path_used: Dict[str, int] = {}
        per_source_used: Dict[str, int] = {}

        for h in bucket:
            path = str(h.get("path") or "")
            source = str(h.get("source") or "")

            if per_path_cap and path and per_path_used.get(path, 0) >= per_path_cap:
                continue
            if per_source_cap and source and per_source_used.get(source, 0) >= per_source_cap:
                continue

            block = _block_for_hit(h)
            if total + len(block) > budget:
                remaining = max(0, budget - total)
                if remaining > 80:
                    parts.append(_truncate(block, remaining))
                    total += len(parts[-1])
                break

            parts.append(block)
            total += len(block)

            if per_path_cap and path:
                per_path_used[path] = per_path_used.get(path, 0) + 1
            if per_source_cap and source:
                per_source_used[source] = per_source_used.get(source, 0) + 1

        return ("\n".join(parts).strip(), total)

    def _pack_round_robin(bucket: List[Dict[str, Any]], budget: int) -> tuple[str, int]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        path_order: List[str] = []
        for h in bucket:
            path = str(h.get("path") or "") or "(no-path)"
            if path not in groups:
                groups[path] = []
                path_order.append(path)
            groups[path].append(h)

        parts: List[str] = []
        total = 0
        per_path_used: Dict[str, int] = {}
        per_source_used: Dict[str, int] = {}

        progressed = True
        while progressed:
            progressed = False
            for path in path_order:
                if not groups.get(path):
                    continue
                if per_path_cap and per_path_used.get(path, 0) >= per_path_cap:
                    # Drop remaining items from this path to avoid endless loops.
                    groups[path].clear()
                    continue

                # Pop next allowed hit for this path.
                while groups[path]:
                    h = groups[path].pop(0)
                    source = str(h.get("source") or "")
                    real_path = str(h.get("path") or "")

                    if (
                        per_source_cap
                        and source
                        and per_source_used.get(source, 0) >= per_source_cap
                    ):
                        continue

                    block = _block_for_hit(h)
                    if total + len(block) > budget:
                        remaining = max(0, budget - total)
                        if remaining > 80:
                            parts.append(_truncate(block, remaining))
                            total += len(parts[-1])
                            progressed = True
                        return ("\n".join(parts).strip(), total)

                    parts.append(block)
                    total += len(block)
                    progressed = True

                    per_path_used[path] = per_path_used.get(path, 0) + 1
                    if per_source_cap and source:
                        per_source_used[source] = per_source_used.get(source, 0) + 1

                    # If the hit had an empty path, we still count it under (no-path)
                    # but keep the original path in the header.
                    _ = real_path
                    break

        return ("\n".join(parts).strip(), total)

    def _pack(bucket: List[Dict[str, Any]], budget: int) -> tuple[str, int]:
        if not rr_enabled:
            return _pack_sequential(bucket, budget)
        return _pack_round_robin(bucket, budget)

    if not split:
        txt, _ = _pack(hits, max_total_chars)
        return txt

    mem_txt, mem_used = _pack(mem_hits, mem_budget)
    spill = max(0, mem_budget - mem_used)
    ws_txt, _ = _pack(ws_hits, ws_budget + spill)

    if mem_txt and ws_txt:
        return (mem_txt + "\n\n" + ws_txt).strip()

    return (mem_txt or ws_txt or "").strip()


async def _maybe_autobuild_vectors(root: str, out: Dict[str, Any]) -> None:
    """Best-effort incremental vector indexing.

    We keep this light because it can run during retrieval:
    - Uses file meta to skip unchanged files
    - Throttled via DB state
    - Budgeted by settings.vector_ws_autorefresh_*

    The first time you use semantic search, this will gradually build the index
    without blocking the app for a full workspace scan.
    """
    if not bool(getattr(settings, "vector_autobuild_on_first_use", True)):
        return

    # If embeddings are disabled, indexing will fail; don't spam.
    provider = (getattr(settings, "embeddings_provider", "none") or "none").strip().lower()
    if provider in ("none", "", "disabled", "off"):
        out["notes"].append("vector_autobuild_skipped: embeddings provider disabled")
        return

    try:
        res = await vector_memory.index_workspace_incremental(
            root_path=root,
            max_files=int(getattr(settings, "vector_ws_autorefresh_max_files", 60)),
            max_seconds=float(getattr(settings, "vector_ws_autorefresh_max_seconds", 1.0)),
            min_interval_seconds=float(getattr(settings, "vector_ws_min_interval_seconds", 300.0)),
            force=False,
            prune_missing=False,
        )
        out["vector_index"] = res
    except Exception as e:
        out["notes"].append(f"vector_autobuild_failed: {e}")


async def gather_context(
    query: str,
    *,
    root_path: Optional[str] = None,
    fts_limit: int = 6,
    vec_limit: int = 6,
    session_id: Optional[str] = None,
    include_memory: bool = True,
) -> Dict[str, Any]:
    root = root_path or settings.workspace
    out: Dict[str, Any] = {"query": query, "hits": [], "notes": []}

    # Ensure FTS is indexed at least once.
    try:
        ensure_res = await memory.ensure_indexed(root)
        out["fts_index"] = ensure_res
    except Exception as e:
        out["notes"].append(f"fts_index_failed: {e}")

    # Optional: auto-build vector index on first use (budgeted)
    await _maybe_autobuild_vectors(root, out)

    # Session-aware memory snippets (WM/STM/episodes/lessons)
    if include_memory and session_id:
        # Preferences (global + session-scoped)
        try:
            prefs_global = await memory.list_preferences(scope="global", session_id=None, limit=30)
            prefs_session = await memory.list_preferences(
                scope="session", session_id=session_id, limit=30
            )
            lines: List[str] = []
            for it in prefs_global:
                lines.append(f"- [global] {it.get('key')}: {it.get('value')}")
            for it in prefs_session:
                lines.append(f"- [session] {it.get('key')}: {it.get('value')}")
            if lines:
                out["hits"].append(
                    {
                        "source": "prefs",
                        "score": 1100.0,
                        "path": "prefs://",
                        "text": _truncate("\n".join(lines), 900),
                        "why": "durable preferences",
                    }
                )
        except Exception as e:
            out["notes"].append(f"preferences_failed: {e}")

        try:
            wm = await memory.get_working_memory(session_id)
            if wm and str(wm.get("content", "")).strip():
                out["hits"].append(
                    {
                        "source": "wm",
                        "score": 1000.0,
                        "path": f"wm://{session_id}",
                        "text": _truncate(str(wm.get("content", "")), 1000),
                        "why": "session working memory",
                    }
                )
        except Exception as e:
            out["notes"].append(f"working_memory_failed: {e}")

        try:
            snap = await memory.get_snapshot(session_id)
            if snap:
                # Keep snapshot compact: pretty-print but truncate.
                txt = json.dumps(snap, ensure_ascii=False, indent=2)
                out["hits"].append(
                    {
                        "source": "stm",
                        "score": 900.0,
                        "path": f"stm://{session_id}",
                        "text": _truncate(txt, 900),
                        "why": "session snapshot",
                    }
                )
        except Exception as e:
            out["notes"].append(f"snapshot_failed: {e}")

        try:
            use_search = bool(getattr(settings, "retrieval_lessons_use_search", True))
            limit = int(getattr(settings, "retrieval_lessons_limit", 8) or 8)
            limit = max(0, limit)

            lessons: list[dict] = []
            if use_search and (query or "").strip() and limit:
                # Prefer query-relevant lessons; fall back to recent if nothing matches.
                try:
                    lessons = await memory.search_lessons(query, limit=max(10, limit * 3))
                    lessons.sort(key=lambda x: float(x.get("rank") or 0.0))
                except Exception:
                    lessons = []

            if not lessons and limit:
                lessons = await memory.list_lessons(limit=max(10, limit * 2))

            if lessons:
                lines: List[str] = []
                for lesson_item in lessons:
                    meta = lesson_item.get("meta") or {}
                    if (
                        isinstance(meta, dict)
                        and meta.get("session_id")
                        and str(meta.get("session_id")) != str(session_id)
                    ):
                        continue
                    key = str(lesson_item.get("key", ""))
                    body = str(lesson_item.get("lesson", ""))
                    if not key and not body:
                        continue
                    if key:
                        lines.append(f"- {key}: {body}")
                    else:
                        lines.append(f"- {body}")
                    if limit and len(lines) >= limit:
                        break
                if lines:
                    out["hits"].append(
                        {
                            "source": "lessons",
                            "score": 800.0,
                            "path": "lessons://search" if use_search else "lessons://recent",
                            "text": _truncate("\n".join(lines), 900),
                            "why": "relevant lessons" if use_search else "recent lessons",
                        }
                    )
        except Exception as e:
            out["notes"].append(f"lessons_failed: {e}")

        try:
            eps = await memory.list_episodes(session_id, limit=5)
            if eps:
                lines2: List[str] = []
                for ep in eps:
                    ts = str(ep.get("created_at", ""))
                    title = str(ep.get("title", ""))
                    summary = str(ep.get("summary", ""))
                    lines2.append(f"- [{ts}] {title}: {summary}")
                out["hits"].append(
                    {
                        "source": "episodes",
                        "score": 700.0,
                        "path": f"episodes://{session_id}",
                        "text": _truncate("\n".join(lines2), 900),
                        "why": "recent episodes",
                    }
                )
        except Exception as e:
            out["notes"].append(f"episodes_failed: {e}")

    # Graph memory: lightweight associative links between files/symbols.
    try:
        if bool(getattr(settings, "graph_memory_enabled", True)) and bool(
            getattr(settings, "graph_memory_retrieval", True)
        ):
            gm = graph_memory()
            limit = int(getattr(settings, "graph_memory_retrieval_limit", 6) or 6)
            limit = max(0, limit)
            if limit and (query or "").strip():
                gtxt = await gm.query_summary(query, limit=limit)
                if gtxt:
                    out["hits"].append(
                        {
                            "source": "graph",
                            "score": 650.0,
                            "path": "graph://summary",
                            "text": _truncate(gtxt, 900),
                            "why": "graph memory links",
                        }
                    )
    except Exception as e:
        out["notes"].append(f"graph_failed: {e}")

    # Rewrite query for workspace retrieval (FTS/hybrid) to improve stability.
    ws_query, note = _rewrite_workspace_query(query)
    if note:
        out["notes"].append(note)
    out["workspace_query"] = ws_query or query

    # Workspace hits (project/workspace)
    # Prefer hybrid search (FTS + vectors) when enabled, as it yields
    # more relevant and diverse context for the agent.
    use_hybrid = bool(getattr(settings, "retrieval_use_hybrid_workspace", True))
    hybrid_used = False
    if use_hybrid:
        try:
            limit = int(getattr(settings, "retrieval_hybrid_limit", 0) or 0)
            if limit <= 0:
                limit = max(int(fts_limit), int(vec_limit))
            fts_pool = int(getattr(settings, "retrieval_hybrid_fts_pool", 0) or 0)
            vec_pool = int(getattr(settings, "retrieval_hybrid_vec_pool", 0) or 0)
            if fts_pool <= 0:
                fts_pool = max(24, int(fts_limit) * 4)
            if vec_pool <= 0:
                vec_pool = max(24, int(vec_limit) * 4)
            per_file_cap = int(getattr(settings, "retrieval_hybrid_per_file_cap", 2) or 2)
            fts_w = float(getattr(settings, "retrieval_hybrid_fts_weight", 1.0) or 1.0)
            vec_w = float(getattr(settings, "retrieval_hybrid_vec_weight", 1.0) or 1.0)

            hyb_hits: List[MemoryHit] = await memory.hybrid_search(
                ws_query,
                limit=limit,
                fts_limit=fts_pool,
                vec_limit=vec_pool,
                per_file_cap=per_file_cap,
                fts_weight=fts_w,
                vec_weight=vec_w,
            )
            for h in hyb_hits:
                meta = dict(h.meta or {})
                out["hits"].append(
                    {
                        "source": h.source,
                        "score": float(h.score),
                        "path": meta.get("path"),
                        "chunk_index": meta.get("chunk_index"),
                        "text": _truncate(h.text, 600),
                        "meta": {
                            "src": meta.get("src"),
                            "fts_norm": meta.get("fts_norm"),
                            "vec_norm": meta.get("vec_norm"),
                            "bonus": meta.get("bonus"),
                            "recency_bonus": meta.get("recency_bonus"),
                            "age_days": meta.get("age_days"),
                        },
                        "why": "workspace hybrid match",
                    }
                )
            hybrid_used = True
        except Exception as e:
            out["notes"].append(f"hybrid_search_failed: {e}")
            use_hybrid = False

    if not use_hybrid:
        # Fallback: separate keyword + semantic searches.
        try:
            fts_hits: List[MemoryHit] = await memory.search(ws_query, limit=fts_limit)
            for h in fts_hits:
                out["hits"].append(
                    {
                        "source": h.source,
                        "score": h.score,
                        "path": h.meta.get("path"),
                        "text": _truncate(h.text, 600),
                        "why": "workspace keyword match",
                    }
                )
        except Exception as e:
            out["notes"].append(f"fts_search_failed: {e}")

        try:
            vec_hits: List[MemoryHit] = await memory.semantic_search(ws_query, limit=vec_limit)
            for h in vec_hits:
                out["hits"].append(
                    {
                        "source": h.source,
                        "score": h.score,
                        "path": h.meta.get("path"),
                        "chunk_index": h.meta.get("chunk_index"),
                        "text": _truncate(h.text, 600),
                        "why": "workspace semantic match",
                    }
                )
        except Exception as e:
            out["notes"].append(f"vector_search_failed: {e}")

    # Optional: expand workspace hits into larger, more useful excerpts.
    expanded = 0
    if bool(getattr(settings, "retrieval_expand_workspace_hits", True)):
        try:
            max_files = int(getattr(settings, "retrieval_workspace_max_files", 4) or 4)
            max_files = max(0, max_files)
            seen_paths = set()
            expanded = 0

            for h in out["hits"]:
                src = str(h.get("source") or "")
                if src not in ("fts", "vector", "hybrid"):
                    continue
                path = str(h.get("path") or "").strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                if max_files and expanded >= max_files:
                    break

                new_text = None

                base_src = src
                if src == "hybrid":
                    try:
                        base_src = str((h.get("meta") or {}).get("src") or "")
                    except Exception:
                        base_src = ""
                    if base_src not in ("fts", "vector"):
                        base_src = "fts"

                if base_src == "vector":
                    bundle = await _expand_vector_hit(path, h.get("chunk_index"))
                    if bundle:
                        new_text = str(bundle.get("text") or "").strip()

                if not new_text:
                    ex = await _expand_fts_hit(query, path)
                    if ex:
                        new_text = str(ex.get("excerpt") or "").strip()

                if new_text:
                    h["text"] = _truncate(
                        new_text,
                        int(
                            getattr(settings, "retrieval_workspace_excerpt_truncate", 1600) or 1600
                        ),
                    )
                    h["why"] = str(h.get("why") or "").rstrip() + " (expanded)"
                    expanded += 1
        except Exception as e:
            out["notes"].append(f"workspace_expand_failed: {e}")

    # Order: session context first, then workspace.
    # This makes the agent less "amnesic" within a session.
    order = {"prefs": 0, "wm": 1, "stm": 2, "lessons": 3, "episodes": 4}
    out["hits"] = sorted(
        out["hits"],
        key=lambda h: (order.get(str(h.get("source")), 10), -(float(h.get("score") or 0.0))),
    )
    hits_before_dedup = len(out["hits"])
    deduped = _dedup_hits(out["hits"])
    out["hits"] = deduped
    hits_after_dedup = len(deduped)
    out["stats"] = {
        "hits_before_dedup": hits_before_dedup,
        "hits_after_dedup": hits_after_dedup,
        "hits_deduped": max(0, hits_before_dedup - hits_after_dedup),
        "workspace_hybrid_used": bool(hybrid_used),
        "workspace_expanded_files": int(expanded or 0),
    }

    return out
