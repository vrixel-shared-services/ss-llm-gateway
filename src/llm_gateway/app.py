"""The gateway's HTTP API. Callers use the Anthropic SDK with base_url pointed here and their
gateway key as the API key; nothing else changes in agent code.

  POST /v1/messages     Anthropic Messages API (streaming included for Anthropic providers)
  GET  /healthz         liveness;  GET /readyz  readiness (policy loaded, provider keys present)
  GET  /metrics         Prometheus
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .ledger import REDACTIONS, REQUESTS, Ledger
from .policy import Policy, load_policy
from .providers import Adapter
from .providers.anthropic import AnthropicAdapter
from .providers.openai import OpenAIAdapter
from .redaction import scrub_request

logging.basicConfig(level=logging.INFO, format="%(message)s")


def error(status: int, kind: str, message: str) -> JSONResponse:
    """Errors in the Anthropic API's shape, so SDKs raise the usual exception types."""
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


def create_app(policy: Policy | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["policy"] = policy or load_policy(
            os.environ.get("GATEWAY_POLICY", "/etc/llm-gateway/policy.yaml")
        )
        state["client"] = httpx.AsyncClient(transport=transport)
        state["ledger"] = Ledger(state["policy"])
        yield
        await state["client"].aclose()

    app = FastAPI(title="Vrixel LLM gateway", lifespan=lifespan, docs_url=None, redoc_url=None)

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
            return error(401, "authentication_error", "invalid gateway key")
        caller_name, caller = found
        try:
            body = await request.json()
        except ValueError:
            return error(400, "invalid_request_error", "body must be JSON")
        model = str(body.get("model", ""))
        if not caller.may_use(model):
            REQUESTS.labels(pol.product, caller_name, "model_denied").inc()
            return error(403, "permission_error", f"{caller_name} may not use {model}")
        route = pol.provider_for(model)
        if route is None:
            REQUESTS.labels(pol.product, caller_name, "no_provider").inc()
            return error(
                403, "permission_error", f"no provider approved for {pol.data_class} data serves {model}"
            )
        provider_name, _ = route
        if int(body.get("max_tokens") or 0) > caller.max_tokens:
            return error(
                400, "invalid_request_error", f"max_tokens is capped at {caller.max_tokens} for {caller_name}"
            )
        if ledger.spent_today(caller_name) >= caller.budget_usd_per_day:
            REQUESTS.labels(pol.product, caller_name, "over_budget").inc()
            return error(429, "rate_limit_error", f"{caller_name} has used its daily budget")

        rule = pol.rule()
        if rule.redaction != "off":
            scrubbed, findings = scrub_request(body, rule.detectors)
            for kind, n in findings.counts.items():
                REDACTIONS.labels(pol.product, caller_name, kind).inc(n)
            if findings.total and rule.redaction == "block":
                REQUESTS.labels(pol.product, caller_name, "blocked").inc()
                return error(
                    400,
                    "invalid_request_error",
                    "request contains data this product may not send to a model: "
                    + ", ".join(sorted(findings.counts)),
                )
            body = scrubbed

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        labels = {k[len("x-vrixel-") :]: v for k, v in request.headers.items() if k.startswith("x-vrixel-")}
        headers = {k: v for k, v in request.headers.items()}
        ad = adapter(provider_name)

        if body.get("stream"):
            usage: dict[str, Any] = {}

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async for chunk in ad.stream(body, headers, usage):
                        yield chunk
                finally:
                    ledger.record(
                        caller=caller_name,
                        model=model,
                        provider=provider_name,
                        usage=usage,
                        request_id=request_id,
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
            ledger.record(
                caller=caller_name,
                model=model,
                provider=provider_name,
                usage=res.usage,
                request_id=request_id,
                labels=labels,
            )
        REQUESTS.labels(
            pol.product, caller_name, "ok" if res.status < 400 else f"upstream_{res.status}"
        ).inc()
        return JSONResponse(res.body, status_code=res.status, headers={"request-id": request_id})

    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        create_app(),
        host="0.0.0.0",  # noqa: S104 - a container listens on all interfaces
        port=int(os.environ.get("PORT", "8080")),
        proxy_headers=True,
        log_config=None,
    )
