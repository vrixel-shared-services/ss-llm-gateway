# Shared LLM gateway

One way for every Vrixel agent to reach a model. Agents use the standard Anthropic SDK with
`base_url` pointed at the gateway and their own gateway key; the gateway holds the provider keys
and applies the product's rules. Shared services ship no runtime, so **each product runs its own
gateway** in its OpenChoreo project, with its own policy, keys and budgets. PHI and spend stay
inside the product.

```python
client = anthropic.Anthropic(base_url=os.environ["LLM_GATEWAY_URL"], api_key=os.environ["LLM_GATEWAY_KEY"])
```

## What it enforces

| Rule | How |
| --- | --- |
| Approved providers only | A model is served only by a provider listed in the policy; PHI products only by providers under a BAA (`require_baa`) |
| Who may call what | Each agent has its own key (only its SHA-256 is in the policy), its own model list and a `max_tokens` cap |
| Data that must not reach a model | Payments: card numbers (Luhn-checked) block the request. General: secrets, card numbers and SSNs are redacted. PHI: provider-level control instead, since the BAA covers the content |
| Spend | A daily budget per agent; the request is refused with `rate_limit_error` once it is used |
| Accountability | One JSON log line per call (product, agent, model, tokens, cache reads and writes, cost, request id, any `x-vrixel-*` labels such as feature id) and Prometheus metrics |
| Audit | Every request is audited before it reaches a provider (CloudEvents `vrixel.audit.v1` on the product's Kafka audit topic): caller, model, provider, kinds of data redacted, and every refusal with its reason. Never content. If the record can't be published the request is refused (`GATEWAY_AUDIT_REQUIRED`, default true) |
| Tracing | OpenTelemetry (`OTEL_EXPORTER_OTLP_ENDPOINT`): each request joins the agent's trace, with the provider call as a child span; logs are JSON with `trace.id` |

It passes Anthropic requests through unchanged, so prompt caching (`cache_control`), tools,
extended thinking and streaming behave exactly as with the API. OpenAI-compatible providers
(OpenAI, Azure OpenAI, vLLM) are translated for text and tool use, without streaming.

## Run it

```bash
uv sync
GATEWAY_POLICY=config/policy.example.yaml ANTHROPIC_API_KEY=... uv run llm-gateway
uv run pytest                 # tests, coverage at least 90%
uv run ruff check . && uv run mypy
```

Deploy it in each product's OpenChoreo project with:
- `GATEWAY_POLICY` pointing at the product's policy (a config map); `data_class` must match the
  product's data class in the platform registry.
- Provider keys from secret references (`ANTHROPIC_API_KEY`, ...). Never in the policy or the repo.
- One replica, or move the budget ledger to Redis or Postgres first (it is in memory).
- No ingress from outside the project: only the product's agents call it.

## Audit settings

| Variable | Default | |
| --- | --- | --- |
| `GATEWAY_AUDIT_SINK` | `log` | `kafka` in deployed environments |
| `GATEWAY_AUDIT_KAFKA_BOOTSTRAP`, `GATEWAY_AUDIT_TOPIC` | `audit.v1` | The product's Kafka and audit topic |
| `GATEWAY_AUDIT_KAFKA_CONFIG` | `{}` | JSON of extra client settings (`security.protocol`, `sasl.*`) |
| `GATEWAY_AUDIT_REQUIRED` | `true` | Refuse requests whose audit record isn't acknowledged |
| `GATEWAY_ENVIRONMENT` | | Stamped on audit records |

## Endpoints

| Path | Purpose |
| --- | --- |
| `POST /v1/messages` | Anthropic Messages API |
| `GET /healthz`, `GET /readyz` | Liveness; readiness checks the policy and provider keys |
| `GET /metrics` | `llm_gateway_tokens_total`, `llm_gateway_cost_usd_total`, `llm_gateway_requests_total`, `llm_gateway_redactions_total` |
