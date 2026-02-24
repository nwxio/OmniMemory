from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .config import settings
from .prefs_infer import infer_preferences_from_text
from .redact import redact_dict, redact_text


_FLAGS = re.IGNORECASE | re.UNICODE


@dataclass
class ConsolidationSuggestion:
    kind: str  # "lesson" | "preference"
    key: str
    value: Any
    meta: Dict[str, Any]
    score: float = 0.0


_MARKER_RE = re.compile(
    r"^\s*(?:lesson|rule|note|fix|decision|takeaway|важно|вывод|правило|заметка|решение)\s*[:\-—]+\s*",
    _FLAGS,
)


def _norm_text(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _hash_key(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    p = (prefix or "auto").strip() or "auto"
    return f"{p}:{h}"


def _redact_any(v: Any) -> Any:
    """Best-effort redaction for consolidation values.

    We keep this local to avoid importing a bunch of helpers in hot paths.
    """
    if v is None:
        return v
    if isinstance(v, str):
        return redact_text(v)
    if isinstance(v, dict):
        return redact_dict(v)
    if isinstance(v, list):
        out: list[Any] = []
        for x in v:
            out.append(_redact_any(x))
        return out
    return v


def _extract_lesson_candidates(text: str) -> List[Tuple[str, float]]:
    """Extract high-signal short lessons from raw episode text.

    This is intentionally heuristic and conservative; we prefer missing a lesson
    to adding junk.
    """
    t = (text or "").strip()
    if not t:
        return []

    min_chars = int(getattr(settings, "consolidate_lesson_min_chars", 18) or 18)
    max_chars = int(getattr(settings, "consolidate_lesson_max_chars", 320) or 320)

    out: List[Tuple[str, float]] = []
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue

        score = 0.0

        # Marker-based lines.
        if _MARKER_RE.search(line):
            score += 0.8
            line = _MARKER_RE.sub("", line).strip()

        # Bullet lines often contain actionable info.
        if re.match(r"^[\-*•]\s+", line):
            score += 0.4
            line = re.sub(r"^[\-*•]\s+", "", line).strip()

        # High-signal tokens.
        if re.search(r"\bnever\b|\bmust\b|\bshould\b|\bdo not\b", line, _FLAGS):
            score += 0.25
        if re.search(r"\bне\s+надо\b|\bнельзя\b|\bвсегда\b|\bникогда\b", line, _FLAGS):
            score += 0.25
        if re.search(r"\bbug\b|\bfix\b|\bошибк\w*\b|\bпочин\w*\b", line, _FLAGS):
            score += 0.15

        line = _norm_text(line)
        if len(line) < min_chars:
            continue
        if len(line) > max_chars:
            # Keep it short and readable.
            line = line[:max_chars].rstrip() + "…"

        # Avoid obvious noise.
        if re.fullmatch(r"[\W_\d]+", line):
            continue
        if line.lower() in ("ok", "okay", "done"):
            continue

        out.append((line, score))

    # Prefer stronger candidates.
    out.sort(key=lambda x: float(x[1]), reverse=True)
    return out


def propose_from_episodes(
    episodes: Iterable[Dict[str, Any]],
    *,
    session_id: str,
    episode_limit: int = 50,
    max_lessons: int = 10,
    include_preferences: bool = True,
) -> Dict[str, Any]:
    """Propose lesson/preference suggestions from recent episodes.

    Returns a dict suitable for API output.
    """
    sid = (session_id or "").strip()
    ep_lim = int(max(1, min(500, int(episode_limit))))
    max_less = int(max(0, min(100, int(max_lessons))))

    # Deterministic order: assume caller provides newest-first.
    eps: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes):
        if i >= ep_lim:
            break
        if isinstance(ep, dict):
            eps.append(ep)

    lesson_dedup: Dict[str, Dict[str, Any]] = {}
    pref_dedup: Dict[str, Dict[str, Any]] = {}

    for ep in eps:
        ep_id = str(ep.get("id") or "")
        title = str(ep.get("title") or "")
        summary = str(ep.get("summary") or "")
        tags = ep.get("tags") if isinstance(ep.get("tags"), list) else []

        # Lessons: mine from title/summary.
        combined = "\n".join([title, summary]).strip()
        for line, score in _extract_lesson_candidates(combined):
            # Key derived from normalized text to avoid duplicates.
            k = _hash_key("lesson", _norm_text(line).lower())
            meta = {
                "session_id": sid,
                "source_episode_id": ep_id,
                "source_tags": tags,
            }
            cur = lesson_dedup.get(k)
            if not cur or float(score) > float(cur.get("score") or 0.0):
                lesson_dedup[k] = {
                    "kind": "lesson",
                    "key": k,
                    "value": line,
                    "meta": meta,
                    "score": float(score),
                }

        # Preferences: mine from summary (conservative allow-list).
        if include_preferences and summary:
            try:
                pairs = infer_preferences_from_text(summary)
            except Exception:
                pairs = []
            for pk, pv in pairs:
                if not pk:
                    continue
                k2 = str(pk)
                cur2 = pref_dedup.get(k2)
                meta2 = {
                    "session_id": sid,
                    "source_episode_id": ep_id,
                    "source_tags": tags,
                }
                # If multiple episodes mention the same pref, keep the latest.
                pref_dedup[k2] = {
                    "kind": "preference",
                    "key": k2,
                    "value": pv,
                    "meta": meta2,
                    "score": float((cur2 or {}).get("score") or 0.0) + 0.2,
                }

    lessons = list(lesson_dedup.values())
    lessons.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    if max_less:
        lessons = lessons[:max_less]

    prefs = list(pref_dedup.values())
    prefs.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    # Safety: redact secret-like patterns in the final proposals.
    if bool(getattr(settings, "consolidate_redact_secrets", True)):
        for it in lessons:
            it["value"] = _redact_any(it.get("value"))
        for it in prefs:
            it["value"] = _redact_any(it.get("value"))

    return {
        "ok": True,
        "session_id": sid,
        "episode_limit": ep_lim,
        "max_lessons": max_less,
        "proposals": {
            "lessons": lessons,
            "preferences": prefs,
        },
    }
