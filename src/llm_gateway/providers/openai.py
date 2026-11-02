"""OpenAI-compatible chat completions (OpenAI, Azure OpenAI, vLLM, and others that speak it).
Translates the common subset: system, text, tool definitions, tool calls and tool results.
Streaming is not translated; agents that need it use an Anthropic-type provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..policy import Provider
from . import Result

STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def to_openai(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": _text(body["system"])})
    for m in body.get("messages", []):
        content = m["content"]
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        calls = [b for b in content if b.get("type") == "tool_use"]
        results = [b for b in content if b.get("type") == "tool_result"]
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": _text(content) or None}
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c.get("input") or {})},
                    }
                    for c in calls
                ]
            messages.append(msg)
        else:
            for r in results:
                messages.append(
                    {"role": "tool", "tool_call_id": r["tool_use_id"], "content": _text(r.get("content"))}
                )
            if _text(content):
                messages.append({"role": "user", "content": _text(content)})
    out: dict[str, Any] = {"model": body["model"], "messages": messages, "max_tokens": body.get("max_tokens")}
    if body.get("tools"):
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {},
                },
            }
            for t in body["tools"]
        ]
    for k in ("temperature", "top_p"):
        if k in body:
            out[k] = body[k]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    return out


def from_openai(data: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        content.append(
            {
                "type": "tool_use",
                "id": c["id"],
                "name": c["function"]["name"],
                "input": json.loads(c["function"].get("arguments") or "{}"),
            }
        )
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": STOP.get(choice.get("finish_reason", "stop"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": (u.get("prompt_tokens") or 0) - cached,
            "output_tokens": u.get("completion_tokens") or 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        },
    }


class OpenAIAdapter:
    def __init__(self, provider: Provider, client: httpx.AsyncClient) -> None:
        self.provider = provider
        self.client = client

    async def create(self, body: dict[str, Any], headers: dict[str, str]) -> Result:
        res = await self.client.post(
            f"{self.provider.base_url}/v1/chat/completions",
            json=to_openai(body),
            headers={"authorization": f"Bearer {self.provider.api_key()}"},
            timeout=600,
        )
        data = res.json()
        if res.status_code >= 400:
            message = (data.get("error") or {}).get("message", "upstream error")
            return Result(
                status=res.status_code,
                usage={},
                body={"type": "error", "error": {"type": "api_error", "message": message}},
            )
        out = from_openai(data, body["model"])
        return Result(status=200, body=out, usage=out["usage"])

    async def stream(
        self, body: dict[str, Any], headers: dict[str, str], usage: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("streaming is only available from Anthropic-type providers")
        yield b""  # pragma: no cover
