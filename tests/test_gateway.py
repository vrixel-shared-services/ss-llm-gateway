import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.ledger import cost_usd
from llm_gateway.policy import Policy
from llm_gateway.providers.openai import from_openai, to_openai
from llm_gateway.redaction import Findings, scrub, scrub_request

from .conftest import call, policy


def test_passes_through_to_anthropic_with_the_platform_key(make_client, upstream):
    with make_client() as c:
        res = call(c, **{"anthropic-beta": "prompt-caching-2024-07-31", "x-vrixel-feature": "EMR-FEAT-1"})
    assert res.status_code == 200 and res.json()["content"][0]["text"] == "Hello"
    url, headers, body = upstream.requests[0]
    assert url == "https://anthropic.test/v1/messages"
    assert headers["x-api-key"] == "sk-real"  # the agent's gateway key never leaves the gateway
    assert headers["anthropic-beta"] == "prompt-caching-2024-07-31"
    assert body["messages"][0]["content"] == "Hello"
    assert res.headers["request-id"]


def test_rejects_unknown_keys_and_models_outside_the_callers_list(make_client):
    with make_client() as c:
        assert call(c, key="nope").status_code == 401
        res = call(c, {"model": "claude-opus-5-5", "max_tokens": 10, "messages": []})
        assert res.status_code == 403 and res.json()["error"]["type"] == "permission_error"


def test_phi_never_reaches_a_provider_without_a_baa(make_client, upstream):
    with make_client() as c:
        res = call(c, {"model": "gpt-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 403 and "phi" in res.json()["error"]["message"]
    assert upstream.requests == []


def test_payments_block_card_numbers(make_client, upstream):
    with make_client(policy("payments")) as c:
        res = call(
            c,
            {
                "model": "claude-sonnet-5-5",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Refund card 4111 1111 1111 1111 please"}],
            },
        )
    assert res.status_code == 400 and "card_number" in res.json()["error"]["message"]
    assert upstream.requests == []


def test_general_products_redact_before_sending(make_client, upstream):
    with make_client(policy("general")) as c:
        res = call(
            c,
            {
                "model": "claude-sonnet-5-5",
                "max_tokens": 10,
                "system": "Key AKIAABCDEFGHIJKLMNOP",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "SSN 123-45-6789"},
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [{"type": "text", "text": "card 4111-1111-1111-1111"}],
                            },
                        ],
                    }
                ],
            },
        )
    assert res.status_code == 200
    sent = upstream.requests[0][2]
    assert sent["system"] == "Key [SECRET REDACTED]"
    assert sent["messages"][0]["content"][0]["text"] == "SSN [US_SSN REDACTED]"
    assert sent["messages"][0]["content"][1]["content"][0]["text"] == "card [CARD_NUMBER REDACTED]"
    assert sent["messages"][0]["content"][1]["tool_use_id"] == "t1"


def test_budget_is_enforced_per_caller_per_day(make_client):
    # Each call costs 1000 input x $3 + 200 output x $15 per million tokens = $0.006.
    caller = policy().callers["intake-agent"].model_dump() | {"budget_usd_per_day": 0.006}
    with make_client(policy(callers={"intake-agent": caller})) as c:
        assert call(c).status_code == 200
        res = call(c)
        assert res.status_code == 429 and res.json()["error"]["type"] == "rate_limit_error"


def test_max_tokens_is_capped(make_client):
    with make_client() as c:
        res = call(c, {"model": "claude-sonnet-5-5", "max_tokens": 100000, "messages": []})
    assert res.status_code == 400


def test_streams_and_records_usage_from_events(make_client, caplog):
    with make_client() as c, caplog.at_level("INFO", logger="llm_gateway.usage"):
        res = call(
            c,
            {
                "model": "claude-sonnet-5-5",
                "max_tokens": 50,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert res.status_code == 200 and b"message_stop" in res.content
    record = json.loads(next(r.message for r in caplog.records if "llm_usage" in r.message))
    assert (record["input_tokens"], record["output_tokens"], record["cache_read_input_tokens"]) == (
        100,
        20,
        900,
    )
    assert record["caller"] == "intake-agent" and record["product"] == "emr"


def test_translates_to_openai_compatible_providers(make_client, upstream):
    pol = policy("general")
    with make_client(pol) as c:
        res = call(
            c,
            {
                "model": "gpt-5",
                "max_tokens": 50,
                "system": "Be brief",
                "tools": [{"name": "lookup", "description": "Look up", "input_schema": {"type": "object"}}],
                "messages": [{"role": "user", "content": "find x"}],
            },
        )
    assert res.status_code == 200
    out = res.json()
    assert out["stop_reason"] == "tool_use" and out["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {"q": "x"},
    }
    assert out["usage"] == {
        "input_tokens": 40,
        "output_tokens": 5,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 0,
    }
    url, headers, sent = upstream.requests[0]
    assert url == "https://openai.test/v1/chat/completions" and headers["authorization"] == "Bearer sk-openai"
    assert sent["messages"][0] == {"role": "system", "content": "Be brief"}
    with make_client(pol) as c:
        streamed = call(c, {"model": "gpt-5", "max_tokens": 5, "stream": True, "messages": []})
    assert streamed.status_code == 400


def test_tool_round_trip_translation():
    body = {
        "model": "gpt-5",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "42"}]},
        ],
    }
    msgs = to_openai(body)["messages"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a": 1}'}
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "42"}
    assert (
        from_openai({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]}, "m")[
            "stop_reason"
        ]
        == "max_tokens"
    )


def test_readiness_needs_provider_keys(make_client, monkeypatch):
    with make_client() as c:
        assert c.get("/readyz").json()["data_class"] == "phi"
        monkeypatch.delenv("T_OPENAI")
        res = c.get("/readyz")
        assert res.status_code == 503 and res.json()["missing_keys"] == ["openai"]
        assert c.get("/healthz").status_code == 200
        assert b"llm_gateway_requests_total" in c.get("/metrics").content


@pytest.mark.parametrize(
    "text,detector,hits",
    [
        ("4111 1111 1111 1111", "card_number", 1),
        ("4111 1111 1111 1112", "card_number", 0),  # fails the Luhn check
        ("call 555-123-4567", "phone", 1),
        ("jane@example.com", "email", 1),
        ("000-12-3456", "us_ssn", 0),  # never issued
        ("-----BEGIN RSA PRIVATE KEY-----", "secret", 1),
    ],
)
def test_detectors(text, detector, hits):
    f = Findings()
    scrub(text, [detector], f)
    assert f.counts.get(detector, 0) == hits


def test_cost_includes_cache_reads_and_writes():
    pol: Policy = policy()
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 0,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
    }
    assert cost_usd(pol, "claude-sonnet-5-5", usage) == pytest.approx(3.0 + 0.3 + 3.75)
    assert cost_usd(pol, "unknown-model", usage) == 0.0


def test_scrub_request_leaves_structure_alone():
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "image", "source": {"data": "4111111111111111"}}]}
        ],
    }
    out, findings = scrub_request(body, ["card_number"])
    assert out == body and findings.total == 0


def test_every_request_is_audited_before_it_reaches_a_model(upstream, monkeypatch):
    from llm_gateway.audit import Auditor, MemorySink

    monkeypatch.setenv("T_ANTHROPIC", "sk-real")
    sink = MemorySink()
    pol = policy("payments")
    app = create_app(
        pol,
        transport=httpx.MockTransport(upstream),
        auditor=Auditor(sink, service="llm-gateway", product="emr"),
    )
    with TestClient(app) as client:
        assert call(client, **{"x-vrixel-run": "run-1"}).status_code == 200
        card = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "4111 1111 1111 1111"}],
        }
        assert call(client, card).status_code == 400
        assert call(client, key="nope").status_code == 401
        data = [e["data"] for e in sink.events]
        assert [(d["action"], d["actor"]["id"], d["outcome"]) for d in data] == [
            ("llm.request", "intake-agent", "success"),
            ("llm.request", "intake-agent", "denied"),
            ("llm.request", "unknown", "denied"),
        ]
        assert data[0]["resource"] == {"type": "model", "ids": ["claude-sonnet-5-5"]}
        assert data[0]["details"]["provider"] == "anthropic" and data[0]["details"]["labels"] == {
            "run": "run-1"
        }
        assert data[1]["details"]["blocked"] == {"card_number": 1}
        assert "4111" not in str(sink.events) and "Hello" not in str(sink.events)

        sink.fail = True
        sent = len(upstream.requests)
        r = call(client)
        assert r.status_code == 503 and "audit trail" in r.json()["error"]["message"]
        assert len(upstream.requests) == sent  # nothing reached the model unrecorded


def test_audit_settings_come_from_the_environment():
    from llm_gateway.app import make_auditor
    from llm_gateway.audit import KafkaSink, LogSink

    a = make_auditor(policy(), {"GATEWAY_ENVIRONMENT": "staging", "GATEWAY_AUDIT_REQUIRED": "false"})
    assert (
        isinstance(a.sink, LogSink)
        and not a.required
        and a.environment == "staging"
        and a.data_class == "phi"
    )
    k = make_auditor(
        policy(),
        {
            "GATEWAY_AUDIT_SINK": "kafka",
            "GATEWAY_AUDIT_KAFKA_BOOTSTRAP": "127.0.0.1:9",
            "GATEWAY_AUDIT_TOPIC": "emr.audit.v1",
            "GATEWAY_AUDIT_KAFKA_CONFIG": '{"client.id": "gw"}',
        },
    )
    assert isinstance(k.sink, KafkaSink) and k.sink.topic == "emr.audit.v1" and k.required
