"""The gateway's HTTP API. Callers use the Anthropic SDK with base_url pointed here and their
gateway key as the API key; nothing else changes in agent code.

  POST /v1/messages     Anthropic Messages API (streaming included for Anthropic providers)
  GET  /healthz         liveness;  GET /readyz  readiness (policy loaded, provider keys present)
  GET  /metrics         Prometheus

Every request is audited (audit.py) before it goes to a provider: which caller, which model and
provider, which kinds of data were redacted or blocked, and requests that were refused. Spend is
in the usage ledger. With GATEWAY_AUDIT_REQUIRED (the default) a request whose audit record
can't be published is refused, so nothing reaches a model unrecorded. Requests are traced with
OpenTelemetry (telemetry.py); the call to the provider is a child span.

Audit settings come from the environment: GATEWAY_AUDIT_SINK (kafka | log),
GATEWAY_AUDIT_KAFKA_BOOTSTRAP, GATEWAY_AUDIT_TOPIC (default audit.v1), GATEWAY_AUDIT_KAFKA_CONFIG
(JSON of extra client settings), GATEWAY_AUDIT_REQUIRED (default true), GATEWAY_ENVIRONMENT.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import telemetry
from .audit import Actor, Auditor, AuditUnavailable, Outcome, make_sink
from .ledger import REDACTIONS, REQUESTS, Ledger
from .policy import Policy, load_policy
from .providers import Adapter
from .providers.anthropic import AnthropicAdapter
from .providers.openai import OpenAIAdapter
from .redaction import scrub_request


def error(status: int, kind: str, message: str) -> JSONResponse:
    """Errors in the Anthropic API's shape, so SDKs raise the usual exception types."""
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


def make_auditor(pol: Policy, env: Any = os.environ) -> Auditor:
    sink = make_sink(
        env.get("GATEWAY_AUDIT_SINK", "log"),
        bootstrap=env.get("GATEWAY_AUDIT_KAFKA_BOOTSTRAP", ""),
        topic=env.get("GATEWAY_AUDIT_TOPIC", "audit.v1"),
        config=json.loads(env.get("GATEWAY_AUDIT_KAFKA_CONFIG") or "{}"),
    )
    return Auditor(
        sink,
        service="llm-gateway",
        product=pol.product,
        environment=env.get("GATEWAY_ENVIRONMENT", ""),
        data_class=pol.data_class,
        required=env.get("GATEWAY_AUDIT_REQUIRED", "true").lower() != "false",
    )


def annotate(
    span: Any,
    *,
    caller: str,
    model: str,
    provider: str,
    usage: dict[str, Any],
    cost: float,
    labels: dict[str, str],
) -> None:
    """The model call on the request's span, in OpenTelemetry's GenAI terms: which model and
    provider, the tokens and the cost, and the caller's labels (agent, variant, run). Never the
    prompt or the reply: the gateway's traces are structure only, whatever the data class."""
    if span is None or not span.is_recording():
        return
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model)
    for key, attr in (
        ("input_tokens", "gen_ai.usage.input_tokens"),
        ("output_tokens", "gen_ai.usage.output_tokens"),
        ("cache_read_input_tokens", "gen_ai.usage.cache_read_input_tokens"),
        ("cache_creation_input_tokens", "gen_ai.usage.cache_creation_input_tokens"),
    ):
        if usage.get(key) is not None:
            span.set_attribute(attr, int(usage.get(key) or 0))
    span.set_attribute("vrixel.cost_usd", round(float(cost), 6))
    span.set_attribute("vrixel.caller", caller)
    for k, v in labels.items():
        if k in ("agent", "variant", "run", "thread", "feature", "tenant"):
            span.set_attribute(f"vrixel.{k}", str(v)[:200])


def create_app(
    policy: Policy | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    auditor: Auditor | None = None,
) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["policy"] = policy or load_policy(
            os.environ.get("GATEWAY_POLICY", "/etc/llm-gateway/policy.yaml")
        )
        state["client"] = httpx.AsyncClient(transport=transport)
        state["ledger"] = Ledger(state["policy"])
        state["audit"] = auditor or make_auditor(state["policy"])
        yield
        await state["client"].aclose()

    app = FastAPI(title="Vrixel LLM gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    telemetry.setup("llm-gateway", app=app)

    def audited(caller: str, model: str, outcome: Outcome, reason: str = "", **details: Any) -> None:
        state["audit"].record(
            "llm.request",
            actor=Actor(id=caller, type="agent" if caller != "unknown" else "service"),
            resource=("model", [model] if model else []),
            outcome=outcome,
            reason=reason,
            details={k: v for k, v in details.items() if v},
        )

    def adapter(provider_name: str) -> Adapter:
        p = state["policy"].providers[provider_name]
        cls = AnthropicAdapter if p.type == "anthropic" else OpenAIAdapter
        return cls(p, state["client"])

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> Response:
        pol: Policy = state["policy"]
        missing = [n for n, p in pol.providers.items() if not p.api_key()]
        if missing:
            return JSONResponse({"status": "not ready", "missing_keys": missing}, status_code=503)
        return JSONResponse({"status": "ready", "product": pol.product, "data_class": pol.data_class})

    @app.get("/metrics")
    def metrics() -> Response:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        pol: Policy = state["policy"]
        ledger: Ledger = state["ledger"]
        key = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix(
            "Bearer "
        )
        found = pol.caller_for(key) if key else None
        if found is None:
            REQUESTS.labels(pol.product, "unknown", "unauthenticated").inc()
            with contextlib.suppress(AuditUnavailable):  # refused either way
                audited("unknown", "", "denied", "invalid gateway key")
            return error(401, "authentication_error", "invalid gateway key")
        caller_name, caller = found
        try:
            body = await request.json()
        except ValueError:
            return error(400, "invalid_request_error", "body must be JSON")
        model = str(body.get("model", ""))
        labels = {k[len("x-vrixel-") :]: v for k, v in request.headers.items() if k.startswith("x-vrixel-")}

        def refuse(status: int, kind: str, message: str, metric: str, **details: Any) -> Response:
            if metric:
                REQUESTS.labels(pol.product, caller_name, metric).inc()
            with contextlib.suppress(AuditUnavailable):  # refused either way
                audited(caller_name, model, "denied", message, labels=labels, **details)
            return error(status, kind, message)

        if not caller.may_use(model):
            return refuse(403, "permission_error", f"{caller_name} may not use {model}", "model_denied")
        route = pol.provider_for(model)
        if route is None:
            return refuse(
                403,
                "permission_error",
                f"no provider approved for {pol.data_class} data serves {model}",
                "no_provider",
            )
        provider_name, _ = route
        if int(body.get("max_tokens") or 0) > caller.max_tokens:
            return refuse(
                400,
                "invalid_request_error",
                f"max_tokens is capped at {caller.max_tokens} for {caller_name}",
                "",
            )
        if ledger.spent_today(caller_name) >= caller.budget_usd_per_day:
            return refuse(429, "rate_limit_error", f"{caller_name} has used its daily budget", "over_budget")

        rule = pol.rule()
        redacted: dict[str, int] = {}
        if rule.redaction != "off":
            scrubbed, findings = scrub_request(body, rule.detectors)
            for kind, n in findings.counts.items():
                REDACTIONS.labels(pol.product, caller_name, kind).inc(n)
            if findings.total and rule.redaction == "block":
                return refuse(
                    400,
                    "invalid_request_error",
                    "request contains data this product may not send to a model: "
                    + ", ".join(sorted(findings.counts)),
                    "blocked",
                    blocked=dict(findings.counts),
                )
            redacted = dict(findings.counts)
            body = scrubbed

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        # Nothing reaches a model unrecorded.
        try:
            audited(
                caller_name,
                model,
                "success",
                provider=provider_name,
                redacted=redacted,
                request_id=request_id,
                labels=labels,
                stream=bool(body.get("stream")),
            )
        except AuditUnavailable:
            REQUESTS.labels(pol.product, caller_name, "audit_unavailable").inc()
            return error(503, "api_error", "the audit trail is unavailable; try again shortly")
        headers = {k: v for k, v in request.headers.items()}
        ad = adapter(provider_name)

        if body.get("stream"):
            usage: dict[str, Any] = {}

            span = trace.get_current_span()

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async for chunk in ad.stream(body, headers, usage):
                        yield chunk
                finally:
                    cost = ledger.record(
                        caller=caller_name,
                        model=model,
                        provider=provider_name,
                        usage=usage,
                        request_id=request_id,
                        labels=labels,
                    )
                    annotate(
                        span,
                        caller=caller_name,
                        model=model,
                        provider=provider_name,
                        usage=usage,
                        cost=cost,
                        labels=labels,
                    )
                    REQUESTS.labels(pol.product, caller_name, "ok").inc()

            if not isinstance(ad, AnthropicAdapter):
                return error(400, "invalid_request_error", f"streaming is not available for {provider_name}")
            return StreamingResponse(
                relay(), media_type="text/event-stream", headers={"request-id": request_id}
            )

        try:
            res = await ad.create(body, headers)
        except httpx.HTTPError as exc:
            REQUESTS.labels(pol.product, caller_name, "upstream_error").inc()
            return error(502, "api_error", f"provider unreachable: {type(exc).__name__}")
        if res.status < 400:
            cost = ledger.record(
                caller=caller_name,
                model=model,
                provider=provider_name,
                usage=res.usage,
                request_id=request_id,
                labels=labels,
            )
            annotate(
                trace.get_current_span(),
                caller=caller_name,
                model=model,
                provider=provider_name,
                usage=res.usage,
                cost=cost,
                labels=labels,
            )
        REQUESTS.labels(
            pol.product, caller_name, "ok" if res.status < 400 else f"upstream_{res.status}"
        ).inc()
        return JSONResponse(res.body, status_code=res.status, headers={"request-id": request_id})

    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    telemetry.setup_logging("llm-gateway")
    uvicorn.run(
        create_app(),
        host="0.0.0.0",  # noqa: S104 - a container listens on all interfaces
        port=int(os.environ.get("PORT", "8080")),
        proxy_headers=True,
        log_config=None,
    )
