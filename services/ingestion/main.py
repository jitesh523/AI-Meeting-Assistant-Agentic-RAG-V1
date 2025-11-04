"""
Ingestion Service - WebSocket audio ingestion and real-time processing
"""
import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ingestion Service", version="1.0.0")

# Config and observability
from .config import settings
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
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


@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
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

@app.on_event("startup")
async def startup():
    """Initialize database and Redis connections"""
    global redis_client, db_pool
    
    # Redis connection
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    
    # Database connection pool
    db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=5,
        max_size=20
    )
    
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
    """Health check endpoint"""
    return {"status": "healthy", "service": "ingestion"}

# --- New: Accept text utterances without audio ---
@app.post("/meetings/{meeting_id}/utterances")
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
