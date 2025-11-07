"""
Ingestion Service - WebSocket audio ingestion and real-time processing
"""
import asyncio
import json
import logging
import re
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel
import os

# Configure logging
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
logger = logging.getLogger(__name__)
logger.addFilter(_RedactFilter())

app = FastAPI(title="Ingestion Service", version="1.0.0")

# Config and observability
from .config import settings
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import contextvars
import time
import uuid

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
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.asgi import ASGIInstrumentor
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        resource = Resource.create({
            "service.name": "ingestion",
        })
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Instrument frameworks
        FastAPIInstrumentor.instrument_app(app)
        ASGIInstrumentor().instrument()
        AsyncPGInstrumentor().instrument()
        RedisInstrumentor().instrument()
        logger.info("OpenTelemetry tracing enabled for ingestion")
    except Exception as _otel_err:
        logger.warning(f"Failed to initialize OpenTelemetry: {_otel_err}")

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "ingestion_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "ingestion_http_request_duration_seconds",
    "HTTP request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
ERROR_COUNT = Counter(
    "ingestion_http_errors_total",
    "Total HTTP errors",
    ["type"],
)
HEALTH_GAUGE = Gauge(
    "ingestion_dependency_up",
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

# Global variables for connections
redis_client = None
db_pool = None

class AudioChunk(BaseModel):
    meeting_id: str
    audio_data: bytes
    timestamp: float
    sample_rate: int
    channels: int

class MeetingMetadata(BaseModel):
    meeting_id: str
    title: str
    platform: str
    start_time: float
    privacy_mode: str
    participants: list[str]

class TextUtterance(BaseModel):
    speaker: str = "User"
    text: str
    timestamp: float = 0.0

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.on_event("startup")
async def startup():
    """Initialize database and Redis connections with retries"""
    global redis_client, db_pool
    max_attempts = 5
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            break
        except Exception:
            if attempt == max_attempts:
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
                raise
            await asyncio.sleep(delay)
            delay *= 2
    logger.info("Ingestion service started")

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

class ConnectionManager:
    """Manages WebSocket connections for each meeting"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, meeting_id: str):
        await websocket.accept()
        self.active_connections[meeting_id] = websocket
        logger.info(f"WebSocket connected for meeting {meeting_id}")
    
    def disconnect(self, meeting_id: str):
        if meeting_id in self.active_connections:
            del self.active_connections[meeting_id]
            logger.info(f"WebSocket disconnected for meeting {meeting_id}")
    
    async def send_to_meeting(self, meeting_id: str, message: dict):
        if meeting_id in self.active_connections:
            try:
                await self.active_connections[meeting_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to meeting {meeting_id}: {e}")
                self.disconnect(meeting_id)

manager = ConnectionManager()

@app.websocket("/ws/audio/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    """WebSocket endpoint for audio streaming"""
    await manager.connect(websocket, meeting_id)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process audio chunk
            await process_audio_chunk(meeting_id, data)
            
    except WebSocketDisconnect:
        manager.disconnect(meeting_id)
        logger.info(f"WebSocket disconnected for meeting {meeting_id}")

async def process_audio_chunk(meeting_id: str, audio_data: bytes):
    """Process incoming audio chunk"""
    try:
        # Store audio chunk in Redis for ASR processing
        chunk_data = {
            "meeting_id": meeting_id,
            "audio_data": audio_data.hex(),  # Convert to hex for JSON storage
            "timestamp": asyncio.get_event_loop().time(),
            "sample_rate": 16000,  # Default sample rate
            "channels": 1
        }
        
        # Push to Redis stream for ASR processing
        await redis_client.xadd(
            "audio_stream",
            chunk_data,
            maxlen=1000  # Keep last 1000 chunks
        )
        
        # Notify ASR service
        await redis_client.publish("audio_ready", meeting_id)
        
        logger.debug(f"Processed audio chunk for meeting {meeting_id}")
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")

@app.post("/meetings/start")
@limiter.limit("5/minute")
async def start_meeting(metadata: MeetingMetadata):
    """Start a new meeting session"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO meetings (id, title, platform, start_ts, privacy_mode, created_at)
                VALUES ($1, $2, $3, to_timestamp($4), $5, now())
            """, metadata.meeting_id, metadata.title, metadata.platform, 
                metadata.start_time, metadata.privacy_mode)
        
        # Store meeting metadata in Redis
        await redis_client.hset(
            f"meeting:{metadata.meeting_id}",
            mapping={
                "title": metadata.title,
                "platform": metadata.platform,
                "privacy_mode": metadata.privacy_mode,
                "status": "active"
            }
        )
        
        logger.info(f"Started meeting {metadata.meeting_id}")
        return {"status": "success", "meeting_id": metadata.meeting_id}
        
    except Exception as e:
        logger.error(f"Error starting meeting: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/meetings/{meeting_id}/end")
@limiter.limit("10/minute")
async def end_meeting(meeting_id: str):
    """End a meeting session"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE meetings 
                SET end_ts = now() 
                WHERE id = $1
            """, meeting_id)
        
        # Update meeting status in Redis
        await redis_client.hset(f"meeting:{meeting_id}", "status", "ended")
        
        # Disconnect WebSocket if active
        manager.disconnect(meeting_id)
        
        logger.info(f"Ended meeting {meeting_id}")
        return {"status": "success", "meeting_id": meeting_id}
        
    except Exception as e:
        logger.error(f"Error ending meeting: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint with dependency probes"""
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
    return {"status": "healthy" if overall else "degraded", "service": "ingestion", "dependencies": {"redis": bool(ok_redis), "postgres": bool(ok_db)}}

# --- New: Accept text utterances without audio ---
@app.post("/meetings/{meeting_id}/utterances")
@limiter.limit("60/minute")
async def post_utterance(meeting_id: str, utterance: TextUtterance = Body(...)):
    """Accept a text utterance (fallback when no audio UI yet).
    - Stores in DB (utterances)
    - Publishes a minimal NLU-like event to agent via Redis ('agent_process')
    """
    try:
        # Store utterance in DB
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO utterances (meeting_id, speaker, start_ms, end_ms, text, conf)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                meeting_id, utterance.speaker, int(utterance.timestamp * 1000), int(utterance.timestamp * 1000) + 1000,
                utterance.text, 0.99,
            )

        # Very simple intent heuristics
        text_l = utterance.text.lower()
        intent = "question" if text_l.strip().endswith("?") else (
            "action_item" if any(k in text_l for k in ["todo", "action", "task", "assign"]) else "statement"
        )
        is_decision = any(k in text_l for k in ["decide", "approved", "agreed", "decision"])  # naive
        is_question = intent == "question"

        nlu_event = {
            "meeting_id": meeting_id,
            "speaker": utterance.speaker,
            "text": utterance.text,
            "timestamp": utterance.timestamp,
            "intent": intent,
            "entities": [],
            "sentiment": "neutral",
            "confidence": 0.8,
            "topics": [],
            "is_decision": is_decision,
            "is_question": is_question,
        }

        # Publish to agent channel consumed by Agent service
        await redis_client.publish("agent_process", json.dumps(nlu_event))

        return {"status": "success", "published": True}
    except Exception as e:
        logger.error(f"Error posting utterance: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
