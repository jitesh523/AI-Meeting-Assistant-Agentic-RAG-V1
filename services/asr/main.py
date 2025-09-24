"""
ASR Service - WhisperX + diarization for real-time speech recognition
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
from pydantic import BaseModel
import whisperx
import torch
from pyannote.audio import Pipeline

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
diarization_pipeline = None

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
    global redis_client, db_pool, whisper_model, diarization_pipeline
    
    # Redis connection
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    
    # Database connection pool
    db_pool = await asyncpg.create_pool(
        "postgresql://postgres:postgres@postgres:5432/meeting_assistant",
        min_size=5,
        max_size=20
    )
    
    # Load WhisperX model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        whisper_model = whisperx.load_model(
            "large-v3", 
            device=device, 
            compute_type=compute_type,
            language="en"
        )
        
        # Load diarization pipeline
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="hf_token_here"  # Replace with actual token
        )
        
        logger.info(f"ASR service started with device: {device}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Fallback to CPU-only mode
        whisper_model = whisperx.load_model("base", device="cpu", compute_type="int8")
        diarization_pipeline = None
        logger.warning("Using fallback CPU-only mode")

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

async def process_audio_stream():
    """Process audio stream from Redis"""
    while True:
        try:
            # Read from Redis stream
            messages = await redis_client.xread(
                {"audio_stream": "$"},
                count=1,
                block=1000  # 1 second timeout
            )
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    await process_audio_chunk(fields)
                    
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
            await asyncio.sleep(1)

async def process_audio_chunk(fields: Dict[str, str]):
    """Process a single audio chunk"""
    try:
        meeting_id = fields["meeting_id"]
        audio_data = bytes.fromhex(fields["audio_data"])
        timestamp = float(fields["timestamp"])
        sample_rate = int(fields["sample_rate"])
        channels = int(fields["channels"])
        
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Reshape if stereo
        if channels == 2:
            audio_array = audio_array.reshape(-1, 2)
            audio_array = audio_array.mean(axis=1)  # Convert to mono
        
        # Normalize audio
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Transcribe with WhisperX
        result = whisper_model.transcribe(
            audio_array,
            batch_size=1,
            language="en"
        )
        
        # Process segments
        for segment in result["segments"]:
            utterance = Utterance(
                meeting_id=meeting_id,
                speaker="unknown",  # Will be updated by diarization
                start_ms=int(segment["start"] * 1000),
                end_ms=int(segment["end"] * 1000),
                text=segment["text"].strip(),
                confidence=float(segment.get("confidence", 0.0)),
                timestamp=timestamp
            )
            
            # Store utterance
            await store_utterance(utterance)
            
            # Send to NLU service
            await send_to_nlu(utterance)
            
        logger.debug(f"Processed audio chunk for meeting {meeting_id}")
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")

async def store_utterance(utterance: Utterance):
    """Store utterance in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO utterances (meeting_id, speaker, start_ms, end_ms, text, conf, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
            """, utterance.meeting_id, utterance.speaker, utterance.start_ms,
                utterance.end_ms, utterance.text, utterance.confidence)
                
    except Exception as e:
        logger.error(f"Error storing utterance: {e}")

async def send_to_nlu(utterance: Utterance):
    """Send utterance to NLU service for processing"""
    try:
        nlu_data = {
            "meeting_id": utterance.meeting_id,
            "speaker": utterance.speaker,
            "text": utterance.text,
            "timestamp": utterance.timestamp,
            "confidence": utterance.confidence
        }
        
        await redis_client.publish("nlu_process", json.dumps(nlu_data))
        
    except Exception as e:
        logger.error(f"Error sending to NLU: {e}")

async def perform_diarization(meeting_id: str, audio_data: np.ndarray):
    """Perform speaker diarization on audio"""
    if diarization_pipeline is None:
        return {}
    
    try:
        # Run diarization
        diarization = diarization_pipeline(audio_data)
        
        # Convert to speaker mapping
        speaker_map = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            speaker_map[(start_ms, end_ms)] = speaker
            
        return speaker_map
        
    except Exception as e:
        logger.error(f"Error performing diarization: {e}")
        return {}

@app.post("/asr/process")
async def process_audio(audio_chunk: AudioChunk, background_tasks: BackgroundTasks):
    """Process audio chunk directly"""
    background_tasks.add_task(process_audio_chunk, audio_chunk.dict())
    return {"status": "processing"}

@app.get("/asr/meetings/{meeting_id}/transcript")
async def get_transcript(meeting_id: str):
    """Get transcript for a meeting"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT speaker, start_ms, end_ms, text, conf, created_at
                FROM utterances
                WHERE meeting_id = $1
                ORDER BY start_ms
            """, meeting_id)
            
            transcript = []
            for row in rows:
                transcript.append({
                    "speaker": row["speaker"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "text": row["text"],
                    "confidence": row["conf"],
                    "timestamp": row["created_at"].isoformat()
                })
            
            return {"meeting_id": meeting_id, "transcript": transcript}
            
    except Exception as e:
        logger.error(f"Error getting transcript: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "asr"}

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_audio_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
