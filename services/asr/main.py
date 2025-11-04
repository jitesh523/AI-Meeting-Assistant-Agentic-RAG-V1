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
import redis.asyncio as redis
import asyncpg
import whisper
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ASR Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
redis_client = None
db_pool = None
whisper_model = None

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

from .config import settings


@app.on_event("startup")
async def startup():
    """Initialize models and connections"""
    global redis_client, db_pool, whisper_model

    # Redis connection
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)

    # Database connection pool
    db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=5,
        max_size=20,
    )

    # Load Whisper model (small for speed; use 'base' or 'large' for accuracy)
    try:
        whisper_model = whisper.load_model("small")  # Downloads ~475MB on first run
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        whisper_model = None

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

                        if whisper_model and audio_array.size > 0:
                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                None, lambda: whisper_model.transcribe(audio_array, fp16=False)
                            )

                            text = result.get("text", "").strip()
                            confidence = float(result.get("confidence", 0.8))

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
async def process_audio(audio_chunk: AudioChunk, background_tasks: BackgroundTasks):
    """Process audio chunk directly"""
    try:
        if not whisper_model:
            return {"status": "error", "message": "Whisper model not available"}
        
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_chunk.audio_data, dtype=np.int16)
        
        # Reshape if stereo
        if audio_chunk.channels == 2:
            audio_array = audio_array.reshape(-1, 2)
            audio_array = audio_array.mean(axis=1)  # Convert to mono
        
        # Normalize audio
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Transcribe with Whisper
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: whisper_model.transcribe(audio_array, fp16=False))
        
        # Store utterance
        utterance = Utterance(
            meeting_id=audio_chunk.meeting_id,
            speaker="unknown",
            start_ms=0,
            end_ms=len(audio_array) * 1000 // audio_chunk.sample_rate,
            text=result["text"].strip(),
            confidence=float(result.get("confidence", 0.8)),
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
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "asr",
        "whisper_available": whisper_model is not None,
        "database_available": db_pool is not None
    }

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_audio_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)