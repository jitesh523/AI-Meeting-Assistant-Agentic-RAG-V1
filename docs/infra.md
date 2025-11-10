# Infrastructure & Observability

This doc summarizes how to run the stack in prod-like mode, how secrets/configs are wired, and what metrics/health endpoints exist across services.

## Services
- ingestion (FastAPI)
- asr (FastAPI)
- nlu (FastAPI)
- rag (FastAPI)
- agent (FastAPI)
- integrations (FastAPI)
- postgres (pgvector)
- redis (cache, pub/sub, streams)
- ui (Next.js)

## Secrets
Docker secrets are mounted to /run/secrets by Compose. We keep small one-line files:
- ./secrets/database_url
- ./secrets/redis_url
- ./secrets/openai_api_key (optional; only needed if using OpenAI)

Each Python service reads DATABASE_URL and REDIS_URL via env file entries in docker-compose and/or secrets. The agent/nlu/rag can also use OPENAI_API_KEY if present.

Create files (example):
```
# Postgres (pgvector)
postgresql://postgres:postgres@postgres:5432/app

# Redis
redis://redis:6379/0

# OpenAI API Key (optional)
sk-...
```

## Environment variables
Core (set in env.example and consumed via each service's settings):
- DATABASE_URL
- REDIS_URL
- NEXT_PUBLIC_API_URL (ui)
- OTEL_EXPORTER_OTLP_ENDPOINT (optional; enables OpenTelemetry)
- CORS allow list via settings.cors_allow_origins (default deny)

Guards (have safe defaults but can be overridden):
- MAX_REQUEST_BYTES (default 1048576)
- REQUEST_TIMEOUT_SECONDS (default 15 for most services, 20 for ASR/RAG)

### Security & Access Control
- AUTH_ENABLED (0/1): enable API key auth middleware across services
- SERVICE_API_KEY: shared bearer or X-API-Key value expected by services (per-env)
- CORS_ALLOW_ORIGINS: comma-separated allow-list (e.g. `https://app.example.com,https://admin.example.com`)

### Idempotency
- IDEMPOTENCY_TTL_SECONDS (default 600): TTL window for duplicate detection
- Storage: Redis-backed `SET NX EX` with per-service namespace, falls back to in-memory if Redis unavailable

### Integrations Provider Resilience
- Circuit breaker window/max failures:
  - CB_WINDOW_SECONDS (default 60)
  - CB_MAX_FAILURES (default 5)
- Retries:
  - RETRY_MAX_ATTEMPTS (default 3)
  - RETRY_BASE_SECONDS (default 0.5)
- Timeouts/budgets:
  - PROVIDER_TIMEOUT_DEFAULT_SECONDS (default 5) per-attempt timeout
  - PROVIDER_TOTAL_BUDGET_SECONDS (default 10) overall call budget
  - TIMEOUT_<PROVIDER>_SECONDS optional per-provider override (e.g., TIMEOUT_SLACK_SECONDS)

Metrics exposed in Integrations:
- integrations_provider_retries_total{service}
- integrations_provider_cb_open_total{service}
- integrations_provider_latency_seconds{service}

## Health endpoints
- GET /health on each FastAPI service.
- Returns healthy or degraded with dependency probes for Redis and Postgres.
- Prometheus gauge per service: <service>_dependency_up{component="redis|postgres"}

## Metrics
- GET /metrics returns Prometheus format.
- Counters/Histograms:
  - <service>_http_requests_total{method,path,status}
  - <service>_http_request_duration_seconds
  - <service>_http_errors_total{type="validation|unhandled|timeout"}

## Tracing (optional)
- If OTEL_EXPORTER_OTLP_ENDPOINT is set, OpenTelemetry is initialized with FastAPI, ASGI, asyncpg, and redis instrumentation.
- Service name is set per service (agent, nlu, rag, asr, ingestion, integrations).

## Rate limiting
- SlowAPI is enabled globally.
- Per-route limits are in code (e.g., SSE stream, posting/processing endpoints).

## Security headers
- Added via middleware: X-Content-Type-Options, X-Frame-Options, Referrer-Policy, COOP/CORP, HSTS.

## Request IDs & structured logging
- Each request receives an X-Request-ID; a context var stores the request ID.
- Logs redact emails and bearer tokens.

## Startup/shutdown
- Redis and Postgres connections use exponential backoff on startup.
- Shutdown closes connections cleanly.

## Production Compose
File: docker-compose.prod.yml
- Resource limits (mem_limit, cpus) per service.
- Healthchecks with start_period to avoid flapping.
- Deploy resources (Swarm-compatible) present for some services (postgres, redis, ingestion, nlu, agent, ui). If you plan to run in Swarm, mirror these blocks for any service missing them.

Example override (compose.override.yml):
```
services:
  nlu:
    deploy:
      resources:
        reservations:
          cpus: '0.50'
          memory: 512M
        limits:
          cpus: '1.0'
          memory: 1G
```

## Local run (prod-like)
```
docker compose -f docker-compose.prod.yml --env-file env.example up --build
```

Then visit:
- http://localhost:8001/health (ingestion)
- http://localhost:8002/health (asr)
- http://localhost:8003/health (nlu)
- http://localhost:8004/health (rag)
- http://localhost:8005/health (agent)
- http://localhost:8006/health (integrations)
- http://localhost:3000 (ui)

## Notes
- OpenAI is optional; services fail open with a degraded path if missing.
- For PDF or heavy docs in RAG, consider increasing REQUEST_TIMEOUT_SECONDS and service CPU/memory limits.
