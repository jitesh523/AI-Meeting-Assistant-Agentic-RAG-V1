"""
ASR Service - OpenAI Whisper for offline audio-to-text processing
Uses Redis Streams for audio ingestion and publishes NLU events.
"""
import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import redis.asyncio as redis
import asyncpg
import whisper
import os
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            "service.name": "asr",
        })
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        ASGIInstrumentor().instrument()
        AsyncPGInstrumentor().instrument()
        RedisInstrumentor().instrument()
        logger.info("OpenTelemetry tracing enabled for asr")
    except Exception as _otel_err:
        logger.warning(f"Failed to initialize OpenTelemetry: {_otel_err}")


app = FastAPI(title="ASR Service", version="1.0.0")

# CORS middleware
from .config import settings
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import uuid
import contextvars

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "asr_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "asr_http_request_duration_seconds",
    "HTTP request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
ERROR_COUNT = Counter(
    "asr_http_errors_total",
    "Total HTTP errors",
    ["type"],
)
HEALTH_GAUGE = Gauge(
    "asr_dependency_up",
    "Health of dependencies (1 up, 0 down)",
    ["component"],
)


@app.middleware("http")
async def size_limit_and_timeout(request: Request, call_next):
    max_bytes = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > max_bytes:
        return JSONResponse(status_code=413, content={"error": "Request too large"})
    timeout_s = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
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

# Global variables
redis_client = None
db_pool = None
whisper_model = None  # openai-whisper model
faster_model = None   # faster-whisper model

class Utterance(BaseModel):
    meeting_id: str
    speaker: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    timestamp: float

class AudioChunk(BaseModel):
    meeting_id: str
    audio_data: bytes
    timestamp: float
    sample_rate: int
    channels: int


async def transcribe_audio(audio_array: np.ndarray) -> tuple[str, float]:
    """Transcribe using the selected backend. Returns (text, confidence)."""
    loop = asyncio.get_running_loop()

    # Prefer faster-whisper if available
    if faster_model is not None:
        def _fw() -> tuple[str, float]:
            # Use vad_filter to improve quality a bit; reduce beam_size for speed
            segments, info = faster_model.transcribe(
                audio_array,
                beam_size=1,
                vad_filter=True,
            )
            text_parts = []
            for seg in segments:
                text_parts.append(seg.text)
            text = " ".join(t.strip() for t in text_parts).strip()
            # faster-whisper doesn't expose per-utterance confidence; use language_probability as a proxy if present
            conf = float(getattr(info, "language_probability", 0.9) or 0.9)
            return text, conf

        return await loop.run_in_executor(None, _fw)

    # Fallback to openai-whisper
    if whisper_model is not None:
        def _ow() -> tuple[str, float]:
            result = whisper_model.transcribe(audio_array, fp16=False)
            text = result.get("text", "").strip()
            conf = float(result.get("confidence", 0.8))
            return text, conf

        return await loop.run_in_executor(None, _ow)

    # No backend available
    return "", 0.0

@app.on_event("startup")
async def startup():
    """Initialize models and connections"""
    global redis_client, db_pool, whisper_model

    # Redis/DB with retries
    max_attempts = 5
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            break
        except Exception:
            if attempt == max_attempts:
                if os.getenv("ALLOW_DEGRADED_STARTUP") == "1":
                    logger.warning("Redis unavailable after retries; starting ASR in degraded mode")
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
                    logger.warning("Postgres unavailable after retries; starting ASR in degraded mode")
                    db_pool = None
                    break
                raise
            await asyncio.sleep(delay)
            delay *= 2

    # Load ASR model based on config
    try:
        if settings.asr_impl.lower() == "faster":
            from faster_whisper import WhisperModel  # type: ignore
            device = settings.asr_device
            compute_type = settings.asr_compute_type
            model_size = settings.asr_model_size
            logger.info(f"Loading faster-whisper model: size={model_size}, device={device}, compute_type={compute_type}")
            # Faster-whisper loads quickly and supports CPU/GPU
            global faster_model
            faster_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("faster-whisper model loaded successfully")
        else:
            model_size = settings.asr_model_size
            logger.info(f"Loading openai-whisper model: size={model_size}")
            global whisper_model
            whisper_model = whisper.load_model(model_size)
            logger.info("openai-whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        whisper_model = None
        faster_model = None

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

async def process_audio_stream():
    """Process audio chunks from Redis Stream 'audio_stream' using Whisper"""
    last_id = "0-0"
    while True:
        try:
            streams = await redis_client.xread({"audio_stream": last_id}, block=1000, count=10)
            if not streams:
                continue
            for stream_key, messages in streams:
                for message_id, fields in messages:
                    last_id = message_id
                    try:
                        meeting_id = fields.get("meeting_id")
                        audio_hex = fields.get("audio_data")
                        timestamp = float(fields.get("timestamp", 0))
                        # Convert hex to PCM16 -> float32 [-1,1]
                        audio_bytes = bytes.fromhex(audio_hex) if audio_hex else b""
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        if audio_array.size > 0:
                            text, confidence = await transcribe_audio(audio_array)

                            # Store transcription in database
                            async with db_pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    INSERT INTO utterances (meeting_id, speaker, start_ms, end_ms, text, conf)
                                    VALUES ($1, $2, $3, $4, $5, $6)
                                    """,
                                    meeting_id,
                                    "unknown",
                                    int(timestamp * 1000),
                                    int(timestamp * 1000) + max(1, int(len(audio_array) / 16)),
                                    text,
                                    confidence,
                                )

                            # Publish to NLU service
                            await redis_client.publish(
                                "nlu_process",
                                json.dumps(
                                    {
                                        "meeting_id": meeting_id,
                                        "speaker": "unknown",
                                        "text": text,
                                        "timestamp": timestamp,
                                        "confidence": confidence,
                                    }
                                ),
                            )

                            logger.info(
                                f"Processed audio chunk for meeting {meeting_id}: {text[:50]}..."
                            )
                        else:
                            logger.warning("Whisper model not available or empty audio chunk")
                    except Exception as inner_e:
                        logger.error(f"Error processing message {message_id}: {inner_e}")
        except Exception as e:
            logger.error(f"Error reading from audio_stream: {e}")
            await asyncio.sleep(1)

@app.post("/asr/process")
@limiter.limit("60/minute")
async def process_audio(audio_chunk: AudioChunk, background_tasks: BackgroundTasks):
    """Process audio chunk directly"""
    try:
        # Validate any ASR backend available
        if not (whisper_model or faster_model):
            return {"status": "error", "message": "ASR model not available"}
        
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_chunk.audio_data, dtype=np.int16)
        
        # Reshape if stereo
        if audio_chunk.channels == 2:
            audio_array = audio_array.reshape(-1, 2)
            audio_array = audio_array.mean(axis=1)  # Convert to mono
        
        # Normalize audio
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Transcribe
        text, conf = await transcribe_audio(audio_array)
        
        # Store utterance
        utterance = Utterance(
            meeting_id=audio_chunk.meeting_id,
            speaker="unknown",
            start_ms=0,
            end_ms=len(audio_array) * 1000 // audio_chunk.sample_rate,
            text=text,
            confidence=float(conf),
            timestamp=audio_chunk.timestamp
        )
        
        # Store in database
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO utterances (meeting_id, speaker, start_ms, end_ms, text, conf) VALUES ($1, $2, $3, $4, $5, $6)",
                utterance.meeting_id,
                utterance.speaker,
                utterance.start_ms,
                utterance.end_ms,
                utterance.text,
                utterance.confidence,
            )
        
        # Send to NLU service
        await redis_client.publish(
            "nlu_process",
            json.dumps(
                {
                    "meeting_id": utterance.meeting_id,
                    "speaker": utterance.speaker,
                    "text": utterance.text,
                    "timestamp": utterance.timestamp,
                    "confidence": utterance.confidence,
                }
            ),
        )
        
        return {"status": "success", "transcript": utterance.text}
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/asr/meetings/{meeting_id}/transcript")
async def get_transcript(meeting_id: str):
    """Get transcript for a meeting"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT speaker, start_ms, end_ms, text, conf, created_at
                FROM utterances
                WHERE meeting_id = $1
                ORDER BY start_ms
                """,
                meeting_id,
            )

        transcript = []
        for row in rows:
            transcript.append(
                {
                    "speaker": row["speaker"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "text": row["text"],
                    "confidence": row["conf"],
                    "timestamp": row["created_at"].isoformat() if row["created_at"] else None,
                }
            )

        return {"meeting_id": meeting_id, "transcript": transcript}
            
    except Exception as e:
        logger.error(f"Error getting transcript: {e}")
        return {"error": str(e)}

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
    return {
        "status": "healthy" if overall else "degraded",
        "service": "asr",
        "impl": ("faster" if faster_model else ("openai" if whisper_model else "none")),
        "whisper_available": whisper_model is not None,
        "faster_available": faster_model is not None,
        "database_available": bool(ok_db),
        "dependencies": {"redis": bool(ok_redis), "postgres": bool(ok_db)}
    }

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_audio_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)