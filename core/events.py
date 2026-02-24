from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    type: str
    session_id: str | None = None
    task_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
