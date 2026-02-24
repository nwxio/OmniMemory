from __future__ import annotations

import re
from typing import Iterable, Sequence


_WORD_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)


def query_tokens(query: str, *, max_terms: int = 12) -> list[str]:
    """Extract normalized query tokens for tolerant matching.

    Language-agnostic (Unicode aware), deterministic, and cheap.
    """
    q = (query or "").strip().casefold()
    if not q:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for m in _WORD_RE.finditer(q):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= int(max_terms):
            break
    return out


def build_like_clause(
    columns: Sequence[str],
    tokens: Sequence[str],
    *,
    require_all_tokens: bool = False,
) -> tuple[str, list[str]]:
    """Build tokenized SQL LIKE clause for SQLite.

    Returns SQL fragment and params. Tokens are matched case-insensitively via LOWER(...).
    """
    cols = [c.strip() for c in columns if str(c or "").strip()]
    toks = [t.strip().casefold() for t in tokens if str(t or "").strip()]
    if not cols or not toks:
        return ("", [])

    groups: list[str] = []
    params: list[str] = []
    for tok in toks:
        sub = []
        for col in cols:
            sub.append(f"LOWER({col}) LIKE ?")
            params.append(f"%{tok}%")
        groups.append("(" + " OR ".join(sub) + ")")

    joiner = " AND " if require_all_tokens else " OR "
    return ("(" + joiner.join(groups) + ")", params)


def score_text_match(query: str, text: str, *, tokens: Sequence[str] | None = None) -> float:
    """Compute a lightweight lexical relevance score.

    The score favors:
    - full query phrase match
    - number of matched tokens
    - matched tokens close to the beginning
    """
    hay = (text or "").casefold()
    if not hay:
        return 0.0

    q = (query or "").strip().casefold()
    toks = list(tokens) if tokens is not None else query_tokens(q)

    score = 0.0
    if q and q in hay:
        score += 3.0

    hit_count = 0
    early_hits = 0
    for tok in toks:
        idx = hay.find(tok)
        if idx < 0:
            continue
        hit_count += 1
        if idx <= 32:
            early_hits += 1

    score += float(hit_count)
    score += float(early_hits) * 0.15
    if toks:
        score += (float(hit_count) / float(len(toks))) * 0.25
    return score


def score_fields(
    query: str,
    fields: Iterable[str],
    *,
    tokens: Sequence[str] | None = None,
) -> float:
    """Score combined text fields against query."""
    text = "\n".join(str(f or "") for f in fields)
    return score_text_match(query, text, tokens=tokens)
