"""
Agent Service - Agentic orchestrator for tool use and decision making
"""
import asyncio
import contextvars
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import asyncpg
import openai
import redis.asyncio as redis
from fastapi import Body, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import StreamingResponse
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import settings

# Configure logging with PII redaction
logging.basicConfig(level=logging.INFO)
class _RedactFilter(logging.Filter):
    _email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    _bearer = re.compile(r"Bearer\s+[A-Za-z0-9\-_.=:+/]{10,}", re.IGNORECASE)
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        msg = self._email.sub("<redacted_email>", msg)
        msg = self._bearer.sub("Bearer <redacted>", msg)
        record.msg = msg
        record.args = ()
        return True

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)
logger.addFilter(_RedactFilter())

app = FastAPI(title="Agent Service", version="1.0.0")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.state.idem_store = {}
app.state.cb_failures = []  # (timestamp) rolling window for circuit breaker

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: OpenTelemetry tracing

if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.asgi import ASGIInstrumentor
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name": "agent",
        })
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        ASGIInstrumentor().instrument()
        AsyncPGInstrumentor().instrument()
        RedisInstrumentor().instrument()
        logger.info("OpenTelemetry tracing enabled for agent")
    except Exception as _otel_err:
        logger.warning(f"Failed to initialize OpenTelemetry: {_otel_err}")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "agent_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "agent_http_request_duration_seconds",
    "HTTP request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
ERROR_COUNT = Counter(
    "agent_http_errors_total",
    "Total HTTP errors",
    ["type"],
)
HEALTH_GAUGE = Gauge(
    "agent_dependency_up",
    "Health of dependencies (1 up, 0 down)",
    ["component"],
)


@app.middleware("http")
async def size_limit_and_timeout(request: Request, call_next):
    max_bytes = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > max_bytes:
        return JSONResponse(status_code=413, content={"error": "Request too large"})
    timeout_s = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    try:
        return await asyncio.wait_for(call_next(request), timeout=timeout_s)
    except asyncio.TimeoutError:
        ERROR_COUNT.labels("timeout").inc()
        return JSONResponse(status_code=504, content={"error": "Request timed out"})


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    if os.getenv("AUTH_ENABLED", "0") == "1":
        path = request.url.path
        if path not in {"/health", "/metrics", "/docs", "/openapi.json"}:
            hdr = request.headers.get("authorization") or request.headers.get("x-api-key")
            if hdr and hdr.lower().startswith("bearer "):
                hdr = hdr.split(" ", 1)[1]
            expected = os.getenv("SERVICE_API_KEY")
            if not expected or hdr != expected:
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)


@app.middleware("http")
async def idempotency_guard(request: Request, call_next):
    # Enforce idempotency for mutating methods when client provides Idempotency-Key
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        key = request.headers.get("Idempotency-Key")
        if key:
            ttl = int(float(os.getenv("IDEMPOTENCY_TTL_SECONDS", "600")))
            rc = globals().get("redis_client")
            if rc:
                try:
                    rkey = f"idemp:agent:{key}"
                    ok = await rc.set(rkey, "1", ex=ttl, nx=True)
                    if not ok:
                        return JSONResponse(status_code=409, content={"error": "Duplicate request"})
                except Exception:
                    now = time.time()
                    store = app.state.idem_store
                    expired = [k for k, v in store.items() if now - v > ttl]
                    for kx in expired:
                        store.pop(kx, None)
                    if key in store:
                        return JSONResponse(status_code=409, content={"error": "Duplicate request"})
                    store[key] = now
            else:
                now = time.time()
                store = app.state.idem_store
                expired = [k for k, v in store.items() if now - v > ttl]
                for kx in expired:
                    store.pop(kx, None)
                if key in store:
                    return JSONResponse(status_code=409, content={"error": "Duplicate request"})
                store[key] = now
    return await call_next(request)


@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(request_id)
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start
    route_path = getattr(request.scope.get("route"), "path", request.url.path)
    REQUEST_COUNT.labels(request.method, route_path, str(response.status_code)).inc()
    REQUEST_LATENCY.observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-site"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    ERROR_COUNT.labels("validation").inc()
    return JSONResponse(status_code=422, content={"error": "Validation error", "details": exc.errors(), "request_id": request_id_var.get()})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    ERROR_COUNT.labels("unhandled").inc()
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "request_id": request_id_var.get()})

# Global variables
redis_client = None
db_pool = None
openai_client = None

class Suggestion(BaseModel):
    id: str
    meeting_id: str
    kind: str  # "ask", "fact", "task", "email", "calendar"
    text: str
    payload: Dict[str, Any]
    confidence: float
    reasons: List[str]
    citations: List[str]
    status: str = "pending"  # "pending", "approved", "rejected"
    approved_by: Optional[str] = None

class Action(BaseModel):
    id: str
    meeting_id: str
    tool: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    status: str = "pending"  # "pending", "executing", "completed", "failed"
    approved_by: Optional[str] = None
    error_message: Optional[str] = None

class NLUResult(BaseModel):
    meeting_id: str
    speaker: str
    text: str
    timestamp: float
    intent: str
    entities: List[Dict[str, str]]
    sentiment: str
    confidence: float
    topics: List[str]
    is_decision: bool
    is_question: bool

class RAGResult(BaseModel):
    meeting_id: str
    query: str
    context: str
    confidence: float
    documents: List[Dict[str, Any]]

class TextInput(BaseModel):
    speaker: str = "User"
    text: str
    timestamp: float = 0.0

# Available tools
AVAILABLE_TOOLS = {
    "search_docs": {
        "description": "Search for relevant documents",
        "parameters": ["query", "max_results"]
    },
    "compose_email": {
        "description": "Draft an email",
        "parameters": ["to", "subject", "body", "priority"]
    },
    "create_ticket": {
        "description": "Create a task or ticket",
        "parameters": ["title", "description", "assignee", "priority", "due_date"]
    },
    "schedule_event": {
        "description": "Schedule a calendar event",
        "parameters": ["title", "start_time", "end_time", "attendees", "description"]
    },
    "log_crm_note": {
        "description": "Log a note to CRM",
        "parameters": ["contact_id", "note", "type"]
    },
    "summarize": {
        "description": "Create a summary",
        "parameters": ["content", "type", "length"]
    }
}

@app.on_event("startup")
async def startup():
    global redis_client, db_pool, openai_client
    max_attempts = 5
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            break
        except Exception:
            if attempt == max_attempts:
                if os.getenv("ALLOW_DEGRADED_STARTUP") == "1":
                    logger.warning("Redis unavailable after retries; starting Agent in degraded mode")
                    redis_client = None
                    break
                raise
            await asyncio.sleep(delay)
            delay *= 2
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            db_pool = await asyncpg.create_pool(settings.database_url, min_size=5, max_size=20)
            break
        except Exception:
            if attempt == max_attempts:
                if os.getenv("ALLOW_DEGRADED_STARTUP") == "1":
                    logger.warning("Postgres unavailable after retries; starting Agent in degraded mode")
                    db_pool = None
                    break
                raise
            await asyncio.sleep(delay)
            delay *= 2
    if not settings.openai_api_key:
        if settings.require_openai:
            raise RuntimeError("OPENAI_API_KEY is required but not set")
        logger.warning("OPENAI_API_KEY not set; features depending on LLM may be limited")
        openai_client = None
    else:
        openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    logger.info("Agent service started")

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

async def process_agent_stream():
    """Process agent tasks from Redis"""
    while True:
        try:
            # Subscribe to agent processing channels
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("agent_process", "agent_rag_result")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    if message["channel"] == "agent_process":
                        data = json.loads(message["data"])
                        await process_nlu_result(NLUResult(**data))
                    elif message["channel"] == "agent_rag_result":
                        data = json.loads(message["data"])
                        await process_rag_result(RAGResult(**data))

        except Exception as e:
            logger.error(f"Error processing agent stream: {e}")
            await asyncio.sleep(1)

# --- New: Generate suggestions directly from raw text (no audio path) ---
@app.post("/agent/meetings/{meeting_id}/suggest-from-text")
@limiter.limit("60/minute")
async def suggest_from_text(meeting_id: str, inp: TextInput = Body(...)):
    try:
        nlu = NLUResult(
            meeting_id=meeting_id,
            speaker=inp.speaker,
            text=inp.text,
            timestamp=inp.timestamp,
            intent="question" if inp.text.strip().endswith("?") else "statement",
            entities=[],
            sentiment="neutral",
            confidence=0.8,
            topics=[],
            is_decision=False,
            is_question=inp.text.strip().endswith("?"),
        )
        suggestions = await analyze_and_suggest(nlu)
        for s in suggestions:
            await store_suggestion(s)
        await send_to_ui(suggestions, meeting_id)
        return {"status": "success", "generated": [s.dict() for s in suggestions]}
    except Exception as e:
        logger.error(f"Error in suggest_from_text: {e}")
        return {"status": "error", "message": str(e)}

# --- New: Summarize meeting and fetch summary ---
@app.post("/agent/meetings/{meeting_id}/summarize")
@limiter.limit("10/minute")
async def summarize_meeting(meeting_id: str):
    try:
        # Fetch last N utterances
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT speaker, text, start_ms
                FROM utterances
                WHERE meeting_id = $1
                ORDER BY id DESC
                LIMIT 50
                """,
                meeting_id,
            )
        transcript = "\n".join([f"{r['speaker']}: {r['text']}" for r in reversed(rows)])

        summary_text = None
        try:
            # Circuit breaker: if too many failures in window, skip remote call
            now = time.time()
            window = float(os.getenv("CB_WINDOW_SECONDS", "60"))
            max_fail = int(os.getenv("CB_MAX_FAILURES", "5"))
            app.state.cb_failures = [t for t in app.state.cb_failures if now - t <= window]
            if len(app.state.cb_failures) >= max_fail:
                raise RuntimeError("circuit_open")

            # Use OpenAI if configured with a real key
            if openai_client and getattr(openai_client, "api_key", "your-api-key-here") != "your-api-key-here":
                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), retry=retry_if_exception_type(Exception))
                def _call_openai():
                    return openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Summarize the meeting into Objectives, Key Points, Decisions, and Action Items (with owners). Be concise."},
                            {"role": "user", "content": transcript or "No transcript"},
                        ],
                        max_tokens=300,
                        temperature=0.3,
                    )
                resp = _call_openai()
                summary_text = resp.choices[0].message.content.strip()
        except RetryError as e:
            logger.warning(f"OpenAI summarization retry exhausted: {e}")
            app.state.cb_failures.append(time.time())
            summary_text = None
        except Exception as e:
            if str(e) == "circuit_open":
                logger.warning("Circuit open for OpenAI summarization; skipping remote call")
            else:
                logger.warning(f"OpenAI summarization failed, using heuristic: {e}")
                app.state.cb_failures.append(time.time())
            summary_text = None

        if not summary_text:
            # Heuristic fallback
            bullets = []
            bullets.append("Objectives: Discuss goals and next steps.")
            if transcript:
                lines = [line for line in transcript.splitlines() if line.strip()]
                bullets.append(f"Key Points: {min(len(lines),5)} key exchanges.")
            bullets.append("Decisions: None recorded.")
            bullets.append("Action Items: Capture tasks in follow-up.")
            summary_text = "\n".join([f"- {b}" for b in bullets])

        # Store in meeting_analytics
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO meeting_analytics (id, meeting_id, metric_name, metric_value, calculated_at)
                VALUES (gen_random_uuid(), $1, 'summary', $2::jsonb, now())
                """,
                meeting_id,
                json.dumps({"text": summary_text}),
            )

        return {"status": "success", "summary": summary_text}
    except Exception as e:
        logger.error(f"Error in summarize_meeting: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/agent/meetings/{meeting_id}/summary")
@limiter.limit("60/minute")
async def get_summary(meeting_id: str):
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT metric_value, calculated_at
                FROM meeting_analytics
                WHERE meeting_id = $1 AND metric_name = 'summary'
                ORDER BY calculated_at DESC
                LIMIT 1
                """,
                meeting_id,
            )
        if not row:
            return {"summary": None}
        val = row["metric_value"]
        text = val.get("text") if isinstance(val, dict) else json.loads(val).get("text")
        return {"summary": text, "timestamp": row["calculated_at"].isoformat()}
    except Exception as e:
        logger.error(f"Error in get_summary: {e}")
        return {"status": "error", "message": str(e)}

async def process_nlu_result(nlu_result: NLUResult):
    """Process NLU result and generate suggestions"""
    try:
        # Analyze the NLU result
        suggestions = await analyze_and_suggest(nlu_result)
        
        # Store suggestions
        for suggestion in suggestions:
            await store_suggestion(suggestion)
            
        # Send to UI
        await send_to_ui(suggestions, nlu_result.meeting_id)
        
        logger.debug(f"Processed NLU result for meeting {nlu_result.meeting_id}")
        
    except Exception as e:
        logger.error(f"Error processing NLU result: {e}")

async def process_rag_result(rag_result: RAGResult):
    """Process RAG result and generate context-aware suggestions"""
    try:
        # Generate context-aware suggestions based on RAG result
        suggestions = await generate_context_suggestions(rag_result)
        
        # Store suggestions
        for suggestion in suggestions:
            await store_suggestion(suggestion)
            
        # Send to UI
        await send_to_ui(suggestions, rag_result.meeting_id)
        
        logger.debug(f"Processed RAG result for meeting {rag_result.meeting_id}")
        
    except Exception as e:
        logger.error(f"Error processing RAG result: {e}")

async def analyze_and_suggest(nlu_result: NLUResult) -> List[Suggestion]:
    """Analyze NLU result and generate suggestions"""
    suggestions = []
    
    # Generate suggestions based on intent
    if nlu_result.intent == "question":
        suggestion = Suggestion(
            id=f"suggestion_{nlu_result.meeting_id}_{nlu_result.timestamp}",
            meeting_id=nlu_result.meeting_id,
            kind="ask",
            text=f"Would you like me to search for information about: {nlu_result.text}?",
            payload={"query": nlu_result.text, "topics": nlu_result.topics},
            confidence=nlu_result.confidence,
            reasons=["Question detected", "Relevant topics identified"],
            citations=[]
        )
        suggestions.append(suggestion)
    
    elif nlu_result.intent == "action_item":
        suggestion = Suggestion(
            id=f"suggestion_{nlu_result.meeting_id}_{nlu_result.timestamp}",
            meeting_id=nlu_result.meeting_id,
            kind="task",
            text=f"Create task: {nlu_result.text}",
            payload={"title": nlu_result.text, "topics": nlu_result.topics},
            confidence=nlu_result.confidence,
            reasons=["Action item detected", "Task creation suggested"],
            citations=[]
        )
        suggestions.append(suggestion)
    
    elif nlu_result.is_decision:
        suggestion = Suggestion(
            id=f"suggestion_{nlu_result.meeting_id}_{nlu_result.timestamp}",
            meeting_id=nlu_result.meeting_id,
            kind="fact",
            text=f"Decision made: {nlu_result.text}",
            payload={"decision": nlu_result.text, "topics": nlu_result.topics},
            confidence=nlu_result.confidence,
            reasons=["Decision detected", "Important for meeting record"],
            citations=[]
        )
        suggestions.append(suggestion)
    
    return suggestions

async def generate_context_suggestions(rag_result: RAGResult) -> List[Suggestion]:
    """Generate context-aware suggestions based on RAG result"""
    suggestions = []
    
    if rag_result.confidence > 0.7:  # High confidence context
        suggestion = Suggestion(
            id=f"rag_suggestion_{rag_result.meeting_id}",
            meeting_id=rag_result.meeting_id,
            kind="fact",
            text=f"Relevant context found: {rag_result.context[:200]}...",
            payload={"context": rag_result.context, "documents": rag_result.documents},
            confidence=rag_result.confidence,
            reasons=["High-confidence context retrieved", "Relevant documents found"],
            citations=[doc.get("source", "Unknown") for doc in rag_result.documents]
        )
        suggestions.append(suggestion)
    
    return suggestions

async def store_suggestion(suggestion: Suggestion):
    """Store suggestion in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO suggestions (
                    id, meeting_id, kind, text, payload, confidence, 
                    reasons, citations, status, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
            """, 
                suggestion.id, suggestion.meeting_id, suggestion.kind,
                suggestion.text, json.dumps(suggestion.payload), suggestion.confidence,
                json.dumps(suggestion.reasons), json.dumps(suggestion.citations),
                suggestion.status
            )
            
    except Exception as e:
        logger.error(f"Error storing suggestion: {e}")

async def send_to_ui(suggestions: List[Suggestion], meeting_id: str):
    """Send suggestions to UI"""
    try:
        ui_data = {
            "meeting_id": meeting_id,
            "suggestions": [suggestion.dict() for suggestion in suggestions]
        }
        await redis_client.publish("ui_suggestions", json.dumps(ui_data))
        
    except Exception as e:
        logger.error(f"Error sending to UI: {e}")

@app.get("/agent/meetings/{meeting_id}/suggestions/stream")
@limiter.limit("120/minute")
async def suggestions_stream(meeting_id: str):
    async def event_generator():
        try:
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("ui_suggestions")
            try:
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
                    if message and message.get("type") == "message":
                        data = json.loads(message["data"]) if isinstance(message["data"], str) else json.loads(message["data"].decode("utf-8"))
                        if data.get("meeting_id") == meeting_id:
                            payload = json.dumps({"suggestions": data.get("suggestions", [])})
                            yield f"data: {payload}\n\n"
            finally:
                await pubsub.unsubscribe("ui_suggestions")
                await pubsub.close()
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

@app.post("/agent/suggestions/{suggestion_id}/approve")
@limiter.limit("30/minute")
async def approve_suggestion(suggestion_id: str, approved_by: str):
    """Approve a suggestion"""
    try:
        async with db_pool.acquire() as conn:
            # Update suggestion status
            await conn.execute("""
                UPDATE suggestions 
                SET status = 'approved', approved_by = $1 
                WHERE id = $2
            """, approved_by, suggestion_id)
            
            # Get suggestion details
            row = await conn.fetchrow("""
                SELECT meeting_id, kind, payload FROM suggestions WHERE id = $1
            """, suggestion_id)
            
            if row:
                # Create action based on suggestion
                action = Action(
                    id=f"action_{suggestion_id}",
                    meeting_id=row["meeting_id"],
                    tool=row["kind"],
                    input_data=json.loads(row["payload"]),
                    output_data={},
                    status="pending",
                    approved_by=approved_by
                )
                
                # Store action
                await store_action(action)
                
                # Execute action
                await execute_action(action)
        
        return {"status": "success", "suggestion_id": suggestion_id}
        
    except Exception as e:
        logger.error(f"Error approving suggestion: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/agent/suggestions/{suggestion_id}/reject")
@limiter.limit("30/minute")
async def reject_suggestion(suggestion_id: str):
    """Reject a suggestion"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE suggestions 
                SET status = 'rejected' 
                WHERE id = $1
            """, suggestion_id)
        
        return {"status": "success", "suggestion_id": suggestion_id}
        
    except Exception as e:
        logger.error(f"Error rejecting suggestion: {e}")
        return {"status": "error", "message": str(e)}

async def store_action(action: Action):
    """Store action in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO actions (
                    id, meeting_id, tool, input_data, output_data, 
                    status, approved_by, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, now())
            """, 
                action.id, action.meeting_id, action.tool,
                json.dumps(action.input_data), json.dumps(action.output_data),
                action.status, action.approved_by
            )
            
    except Exception as e:
        logger.error(f"Error storing action: {e}")

async def execute_action(action: Action):
    """Execute an action using the appropriate tool"""
    try:
        # Update action status
        action.status = "executing"
        await update_action_status(action)
        
        result = {}
        
        # Execute based on tool type
        if action.tool == "search_docs":
            result = await search_documents(action.input_data)
        elif action.tool == "compose_email":
            result = await publish_integration_task("gmail", "draft_email", action.input_data, action.approved_by)
        elif action.tool == "create_ticket":
            # Assuming 'create_ticket' maps to Notion for now, or we could add a 'service' field to payload
            result = await publish_integration_task("notion", "create_page", action.input_data, action.approved_by)
        elif action.tool == "schedule_event":
            result = await publish_integration_task("calendar", "create_event", action.input_data, action.approved_by)
        elif action.tool == "log_crm_note":
            # Mapping CRM note to Notion for now
            result = await publish_integration_task("notion", "create_page", action.input_data, action.approved_by)
        elif action.tool == "summarize":
            result = await create_summary(action.input_data)
        else:
            result = {"error": f"Unknown tool: {action.tool}"}
        
        # Update action with result
        action.output_data = result
        action.status = "completed" if "error" not in result else "failed"
        action.error_message = result.get("error")
        
        await update_action_status(action)
        
    except Exception as e:
        logger.error(f"Error executing action: {e}")
        action.status = "failed"
        action.error_message = str(e)
        await update_action_status(action)

async def publish_integration_task(service: str, action: str, data: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
    """Publish task to integrations service via Redis"""
    try:
        if not redis_client:
            return {"error": "Redis client not available"}
            
        task_data = {
            "service": service,
            "action": action,
            "data": {
                **data,
                "user_id": user_id or "default_user"  # Fallback if no approver
            }
        }
        
        await redis_client.publish("integration_task", json.dumps(task_data))
        return {"status": "queued", "service": service, "action": action}
        
    except Exception as e:
        logger.error(f"Error publishing integration task: {e}")
        return {"error": str(e)}

async def update_action_status(action: Action):
    """Update action status in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE actions 
                SET status = $1, output_data = $2, error_message = $3
                WHERE id = $4
            """, action.status, json.dumps(action.output_data), 
                action.error_message, action.id)
            
    except Exception as e:
        logger.error(f"Error updating action status: {e}")

# Tool implementations (stubs)
async def search_documents(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Search for documents"""
    # TODO: Implement document search
    return {"status": "success", "results": []}

# Stubs removed, replaced by publish_integration_task
# async def compose_email...
# async def create_ticket...
# async def schedule_event...
# async def log_crm_note...

async def create_summary(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary"""
    # TODO: Implement summary creation
    return {"status": "success", "summary_id": "summary_123"}

@app.get("/agent/meetings/{meeting_id}/suggestions")
@limiter.limit("60/minute")
async def get_suggestions(meeting_id: str):
    """Get suggestions for a meeting"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, kind, text, payload, confidence, reasons, 
                       citations, status, approved_by, created_at
                FROM suggestions
                WHERE meeting_id = $1
                ORDER BY created_at DESC
            """, meeting_id)
            
            suggestions = []
            for row in rows:
                suggestions.append({
                    "id": row["id"],
                    "kind": row["kind"],
                    "text": row["text"],
                    "payload": json.loads(row["payload"]),
                    "confidence": row["confidence"],
                    "reasons": json.loads(row["reasons"]),
                    "citations": json.loads(row["citations"]),
                    "status": row["status"],
                    "approved_by": row["approved_by"],
                    "created_at": row["created_at"].isoformat()
                })
            
            return {"meeting_id": meeting_id, "suggestions": suggestions}
            
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    ok_redis = 0
    ok_db = 0
    try:
        if redis_client:
            pong = await redis_client.ping()
            ok_redis = 1 if pong else 0
    except Exception:
        ok_redis = 0
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                row = await conn.fetchval("SELECT 1")
                ok_db = 1 if row == 1 else 0
    except Exception:
        ok_db = 0
    HEALTH_GAUGE.labels("redis").set(ok_redis)
    HEALTH_GAUGE.labels("postgres").set(ok_db)
    overall = ok_redis and ok_db
    return {"status": "healthy" if overall else "degraded", "service": "agent", "dependencies": {"redis": bool(ok_redis), "postgres": bool(ok_db)}}

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    if redis_client:
        asyncio.create_task(process_agent_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
