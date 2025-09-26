"""
ASR Service - OpenAI Whisper for offline audio-to-text processing
Simplified version with Redis integration
"""
import asyncio
import json
import logging
import os
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis
import psycopg2
import whisper
from pydantic import BaseModel
from datetime import datetime

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
db_conn = None
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

@app.on_event("startup")
async def startup():
    """Initialize models and connections"""
    global redis_client, db_conn, whisper_model
    
    # Redis connection
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"), 
        port=6379, 
        decode_responses=True
    )
    
    # Database connection
    try:
        db_conn = psycopg2.connect(
            dbname="meetings",
            user="admin",
            password="secret",
            host=os.getenv("POSTGRES_HOST", "db")
        )
        logger.info("Connected to PostgreSQL database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        # Create in-memory fallback
        db_conn = None
    
    # Load Whisper model (small for speed; use 'base' or 'large' for accuracy)
    try:
        whisper_model = whisper.load_model("small")  # Downloads ~475MB on first run
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        whisper_model = None

    # Create utterances table if not exists
    if db_conn:
        with db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS utterances (
                    id BIGSERIAL PRIMARY KEY,
                    meeting_id UUID,
                    speaker TEXT,
                    start_ms INT,
                    end_ms INT,
                    text TEXT,
                    conf REAL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            db_conn.commit()

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if db_conn:
        db_conn.close()

async def process_audio_chunks():
    """Process audio chunks from Redis using Whisper"""
    while True:
        try:
            # Check for audio chunks in Redis
            chunk = redis_client.brpop(f"meeting:*:audio", timeout=1)
            if chunk:
                _, data = chunk
                data = json.loads(data)
                
                # Convert hex back to bytes, then to numpy array for Whisper (expects float32 [-1,1])
                audio_bytes = bytes.fromhex(data["data"])
                audio_array = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0  # Normalize PCM16
                
                # Transcribe with Whisper
                if whisper_model:
                    result = whisper_model.transcribe(audio_array, fp16=False)  # fp16=False for CPU
                    
                    # Store transcription in database
                    if db_conn:
                        with db_conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO utterances (meeting_id, speaker, start_ms, text, conf) VALUES (%s, %s, %s, %s, %s)",
                                (data["meeting_id"], "unknown", data["timestamp"], result["text"], result.get("confidence", 0.8))
                            )
                            db_conn.commit()
                    
                    # Publish to next service (e.g., NLU via Redis)
                    redis_client.publish("nlu_process", json.dumps({
                        "meeting_id": data["meeting_id"], 
                        "text": result["text"],
                        "timestamp": data["timestamp"],
                        "confidence": result.get("confidence", 0.8)
                    }))
                    
                    logger.info(f"Processed audio chunk for meeting {data['meeting_id']}: {result['text'][:50]}...")
                else:
                    logger.warning("Whisper model not available, skipping audio processing")
                    
        except Exception as e:
            logger.error(f"Error processing audio chunks: {e}")
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
        result = whisper_model.transcribe(audio_array, fp16=False)
        
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
        if db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO utterances (meeting_id, speaker, start_ms, end_ms, text, conf) VALUES (%s, %s, %s, %s, %s, %s)",
                    (utterance.meeting_id, utterance.speaker, utterance.start_ms, utterance.end_ms, utterance.text, utterance.confidence)
                )
                db_conn.commit()
        
        # Send to NLU service
        redis_client.publish("nlu_process", json.dumps({
            "meeting_id": utterance.meeting_id,
            "text": utterance.text,
            "timestamp": utterance.timestamp,
            "confidence": utterance.confidence
        }))
        
        return {"status": "success", "transcript": utterance.text}
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/asr/meetings/{meeting_id}/transcript")
async def get_transcript(meeting_id: str):
    """Get transcript for a meeting"""
    try:
        if not db_conn:
            return {"error": "Database not available"}
            
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT speaker, start_ms, end_ms, text, conf, created_at
                FROM utterances
                WHERE meeting_id = %s
                ORDER BY start_ms
            """, (meeting_id,))
            
            rows = cur.fetchall()
            transcript = []
            for row in rows:
                transcript.append({
                    "speaker": row[0],
                    "start_ms": row[1],
                    "end_ms": row[2],
                    "text": row[3],
                    "confidence": row[4],
                    "timestamp": row[5].isoformat() if row[5] else None
                })
            
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
        "database_available": db_conn is not None
    }

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_audio_chunks())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)