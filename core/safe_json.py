from __future__ import annotations

import json
import ast
import math
from typing import Any

from starlette.responses import JSONResponse


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize values for strict JSON.

    Python floats can represent NaN/Infinity, but JSON cannot.
    If such values leak into API responses, browsers will fail with
    `Unexpected token N` (NaN) or similar errors.

    We convert non-finite floats to None, while keeping the overall
    structure intact.
    """

    # Fast-path common primitives
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # Mapping types
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # Sequence types
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    # Sets are not JSON-serializable; coerce to list.
    if isinstance(obj, set):
        return [sanitize_for_json(v) for v in obj]

    # Best-effort: for objects with __dict__ (rare in API output)
    try:
        d = vars(obj)
    except Exception:
        return obj
    return sanitize_for_json(d)


class SafeJSONResponse(JSONResponse):
    """JSONResponse that guarantees strict JSON (no NaN/Infinity).

    We sanitize the content first, then serialize with allow_nan=False.
    """

    def render(self, content: Any) -> bytes:
        content = sanitize_for_json(content)
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")

def safe_json_loads(raw: Any) -> Any:
    """Best-effort JSON loader.

    Handles common sqlite edge-cases:
    - NULL/empty strings
    - bytes/memoryview values
    - legacy python repr strings

    Returns decoded Python object, or None.
    """
    if raw is None:
        return None

    # Normalize bytes-like
    try:
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode('utf-8', errors='replace')
    except Exception:
        pass

    # Already an object
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw

    s = raw if isinstance(raw, str) else None
    if s is None:
        try:
            s = str(raw)
        except Exception:
            return None

    if not s.strip():
        return None

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def safe_json_loads_dict(raw: Any) -> dict:
    """Load JSON-ish payload, always returning a dict."""
    obj = safe_json_loads(raw)
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return {'_payload': obj}
