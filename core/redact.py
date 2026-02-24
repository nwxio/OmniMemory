from __future__ import annotations

import re
from typing import Iterable

# Basic redaction. Extend with your own org-specific patterns.
_PATTERNS: list[re.Pattern[str]] = [
    # Common key/value pairs
    re.compile(r"(?i)(api[_-]?key|token|secret|password|passphrase)\s*[:=]\s*([^\s'\"<>]+)", re.MULTILINE),
    # Authorization headers / bearer tokens
    re.compile(r"(?i)(authorization\s*:\s*bearer)\s+([^\s'\"<>]+)", re.MULTILINE),
    # Popular token formats (best-effort)
    re.compile(r"\b(sk-[A-Za-z0-9]{16,})\b"),
    re.compile(r"\b(ghp_[A-Za-z0-9]{30,})\b"),
    re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{10,})\b"),
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    # Private keys
    re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|DSA|ED25519) PRIVATE KEY-----.*?-----END .*?-----", re.DOTALL),
    re.compile(r"-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----", re.DOTALL),
    # Long base64-ish blobs (often tokens/certs). Keep it conservative.
    re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/=])"),
]

_REPLACEMENT = r"\1=<REDACTED>"


def redact_text(text: str) -> str:
    if not text:
        return text
    out = text
    for p in _PATTERNS:
        pat = p.pattern
        if "PRIVATE KEY" in pat:
            out = p.sub("<REDACTED_PRIVATE_KEY>", out)
        elif "authorization" in pat.lower():
            out = p.sub(r"\1 <REDACTED>", out)
        elif "[A-Za-z0-9+/]{80" in pat:
            out = p.sub("<REDACTED_BLOB>", out)
        elif "sk-" in pat or "ghp_" in pat or "xox" in pat or "AKIA" in pat:
            # Token formats: replace whole match
            out = p.sub("<REDACTED_TOKEN>", out)
        else:
            out = p.sub(_REPLACEMENT, out)
    return out


def redact_dict(d: dict) -> dict:
    def scrub(v):
        if isinstance(v, str):
            return redact_text(v)
        if isinstance(v, dict):
            return {k: scrub(val) for k, val in v.items()}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        return v
    return scrub(d)


def redact_lines(lines: Iterable[str]) -> list[str]:
    return [redact_text(x) for x in lines]
