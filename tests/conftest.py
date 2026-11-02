import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.policy import Policy

KEY = "agent-key-123"


def policy(data_class="phi", **over):
    raw = {
        "product": "emr",
        "data_class": data_class,
        "providers": {
            "anthropic": {
                "type": "anthropic",
                "base_url": "https://anthropic.test",
                "api_key_env": "T_ANTHROPIC",
                "baa": True,
                "models": ["claude-*"],
            },
            "openai": {
                "type": "openai",
                "base_url": "https://openai.test",
                "api_key_env": "T_OPENAI",
                "baa": False,
                "models": ["gpt-*"],
            },
        },
        "data_classes": {
            "phi": {"require_baa": True, "redaction": "off"},
            "payments": {"redaction": "block", "detectors": ["card_number"]},
            "general": {"redaction": "redact", "detectors": ["secret", "card_number", "us_ssn"]},
        },
        "callers": {
            "intake-agent": {
                "key_sha256": hashlib.sha256(KEY.encode()).hexdigest(),
                "budget_usd_per_day": 1.0,
                "models": ["claude-sonnet-*", "gpt-*"],
            }
        },
        "pricing": {"claude-sonnet-*": {"input": 3.0, "output": 15.0}},
    }
    raw.update(over)
    return Policy.model_validate(raw)


class Upstream:
    """Fake providers: records requests, answers like the real APIs."""

    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((str(request.url), dict(request.headers), body))
        if request.url.host == "anthropic.test":
            if body.get("stream"):
                events = [
                    {
                        "type": "message_start",
                        "message": {
                            "usage": {"input_tokens": 100, "output_tokens": 1, "cache_read_input_tokens": 900}
                        },
                    },
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
                    {"type": "message_delta", "usage": {"output_tokens": 20}},
                    {"type": "message_stop"},
                ]
                sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
                return httpx.Response(
                    200, content=sse.encode(), headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": "Hello"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 200,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 10},
                },
            },
        )


@pytest.fixture
def upstream():
    return Upstream()


@pytest.fixture
def make_client(upstream, monkeypatch):
    monkeypatch.setenv("T_ANTHROPIC", "sk-real")
    monkeypatch.setenv("T_OPENAI", "sk-openai")

    def make(pol=None):
        app = create_app(pol or policy(), transport=httpx.MockTransport(upstream))
        return TestClient(app)

    return make


def call(client, body=None, key=KEY, **headers):
    body = body or {
        "model": "claude-sonnet-5-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    return client.post(
        "/v1/messages", json=body, headers={"x-api-key": key, "anthropic-version": "2023-06-01", **headers}
    )
