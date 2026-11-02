"""Provider adapters. Callers always speak the Anthropic Messages API; adapters translate."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Result:
    status: int
    body: dict[str, Any]
    usage: dict[str, Any]


class Adapter(Protocol):
    async def create(self, body: dict[str, Any], headers: dict[str, str]) -> Result: ...

    def stream(
        self, body: dict[str, Any], headers: dict[str, str], usage: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Server-sent events, passed through; fills `usage` as events arrive."""
        ...
