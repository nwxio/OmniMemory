from __future__ import annotations

import re
from typing import Any, List, Tuple


_FLAGS = re.IGNORECASE | re.UNICODE


def infer_preferences_from_text(text: str) -> List[Tuple[str, Any]]:
    """Infer durable preferences from a single user message.

    Design goals:
      - Conservative: extract only a small allow-list of high-signal preferences.
      - Deterministic: no LLM needed.
      - Safe: avoid capturing personal/sensitive data.

    Returned list items are (key, value).
    """

    t = (text or "").strip()
    if not t:
        return []

    out: List[Tuple[str, Any]] = []

    # ---- Language / output style ----
    if re.search(r"\bна\s+русском\b|\bрусском\s+языке\b", t, _FLAGS):
        out.append(("style.language", "ru"))

    # ---- Code conventions ----
    if re.search(r"комментари[йи]\w*\s+в\s+коде\s+.*английск", t, _FLAGS):
        out.append(("style.code_comments_language", "en"))

    if re.search(r"\bкод\b\s+.*\bанглийск", t, _FLAGS):
        # This means the programming-language tokens should be English.
        out.append(("style.code_language", "en"))

    # ---- Web/source restrictions ----
    if re.search(r"не\s+предлагать\s+ресурс\w*\s+из\s+доменов?\s+\*\.ru", t, _FLAGS) or re.search(
        r"никогда\s+не\s+предлагать\s+.*\*\.ru", t, _FLAGS
    ):
        out.append(("web.block_tlds", ["ru", "рф", "xn--p1ai"]))

    # ---- Currency preference ----
    if re.search(r"\bв\s+гривн", t, _FLAGS) or re.search(r"\bUAH\b", t, _FLAGS):
        out.append(("finance.currency", "UAH"))

    # Dedupe by key (last write wins)
    dedup = {}
    for k, v in out:
        dedup[k] = v
    return list(dedup.items())
