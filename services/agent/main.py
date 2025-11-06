"""
Agent Service - Agentic orchestrator for tool use and decision making
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, BackgroundTasks
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel
import openai
from .config import settings
import time
import uuid
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.responses import StreamingResponse

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

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
logger = logging.getLogger(__name__)
logger.addFilter(_RedactFilter())

app = FastAPI(title="Agent Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: OpenTelemetry tracing
import os
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.asgi import ASGIInstrumentor
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor

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


@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start
    # Label path as route path if available
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
    """Initialize connections"""
    global redis_client, db_pool, openai_client
    
    # Redis connection
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    
    # Database connection pool
    db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=5,
        max_size=20
    )
    
    # Initialize OpenAI client
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
            # Use OpenAI if configured with a real key
            if openai_client and getattr(openai_client, "api_key", "your-api-key-here") != "your-api-key-here":
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Summarize the meeting into Objectives, Key Points, Decisions, and Action Items (with owners). Be concise."},
                        {"role": "user", "content": transcript or "No transcript"},
                    ],
                    max_tokens=300,
                    temperature=0.3,
                )
                summary_text = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI summarization failed, using heuristic: {e}")
            summary_text = None

        if not summary_text:
            # Heuristic fallback
            bullets = []
            bullets.append("Objectives: Discuss goals and next steps.")
            if transcript:
                lines = [l for l in transcript.splitlines() if l.strip()]
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
        
        # Execute based on tool type
        if action.tool == "search_docs":
            result = await search_documents(action.input_data)
        elif action.tool == "compose_email":
            result = await compose_email(action.input_data)
        elif action.tool == "create_ticket":
            result = await create_ticket(action.input_data)
        elif action.tool == "schedule_event":
            result = await schedule_event(action.input_data)
        elif action.tool == "log_crm_note":
            result = await log_crm_note(action.input_data)
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

async def compose_email(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compose an email"""
    # TODO: Implement email composition
    return {"status": "success", "draft_id": "draft_123"}

async def create_ticket(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a ticket"""
    # TODO: Implement ticket creation
    return {"status": "success", "ticket_id": "ticket_123"}

async def schedule_event(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule an event"""
    # TODO: Implement event scheduling
    return {"status": "success", "event_id": "event_123"}

async def log_crm_note(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Log a CRM note"""
    # TODO: Implement CRM logging
    return {"status": "success", "note_id": "note_123"}

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
    """Health check endpoint"""
    return {"status": "healthy", "service": "agent"}

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_agent_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
