"""Anthropic: requests pass through unchanged, so prompt caching (cache_control), tools, thinking
and streaming behave exactly as with the API. Only the credential is swapped."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..policy import Provider
from . import Result

FORWARD_HEADERS = ("anthropic-version", "anthropic-beta")


class AnthropicAdapter:
    def __init__(self, provider: Provider, client: httpx.AsyncClient) -> None:
        self.provider = provider
        self.client = client

    def _headers(self, incoming: dict[str, str]) -> dict[str, str]:
        out = {
            "x-api-key": self.provider.api_key(),
            "content-type": "application/json",
            "anthropic-version": incoming.get("anthropic-version", "2023-06-01"),
        }
        if incoming.get("anthropic-beta"):
            out["anthropic-beta"] = incoming["anthropic-beta"]
        return out

    async def create(self, body: dict[str, Any], headers: dict[str, str]) -> Result:
        res = await self.client.post(
            f"{self.provider.base_url}/v1/messages", json=body, headers=self._headers(headers), timeout=600
        )
        data = res.json()
        return Result(status=res.status_code, body=data, usage=data.get("usage") or {})

    async def stream(
        self, body: dict[str, Any], headers: dict[str, str], usage: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        async with self.client.stream(
            "POST",
            f"{self.provider.base_url}/v1/messages",
            json=body,
            headers=self._headers(headers),
            timeout=600,
        ) as res:
            buffer = b""
            async for chunk in res.aiter_bytes():
                yield chunk
                buffer += chunk
                while b"\n\n" in buffer:  # read usage from message_start and message_delta events
                    event, buffer = buffer.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if line.startswith(b"data:"):
                            try:
                                data = json.loads(line[5:])
                            except ValueError:
                                continue
                            if data.get("type") == "message_start":
                                usage.update(data.get("message", {}).get("usage") or {})
                            elif data.get("type") == "message_delta":
                                usage.update(
                                    {k: v for k, v in (data.get("usage") or {}).items() if v is not None}
                                )
