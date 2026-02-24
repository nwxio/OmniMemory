from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    pass


@dataclass
class ToolManifest:
    name: str
    version: str
    description: str
    permissions: dict[str, Any] = field(default_factory=dict)
    side_effects: dict[str, Any] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


class _NoopEventBus:
    async def publish(self, event: Any) -> None:
        return None


class ToolBase:
    manifest: ToolManifest

    def __init__(self, event_bus: Any | None = None) -> None:
        self.event_bus = event_bus or _NoopEventBus()
