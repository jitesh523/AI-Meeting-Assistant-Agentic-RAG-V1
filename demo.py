#!/usr/bin/env python3
"""
AI Meeting Assistant Demo - Co-pilot Nexus UI Integration
This demonstrates the core functionality with the beautiful co-pilot-nexus UI
"""

import asyncio
import json
import time
import os
import uuid
import hashlib
import aiofiles
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
import tempfile
import io
import json as pyjson
import math
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except Exception:
    AIOSQLITE_AVAILABLE = False
try:
    from faster_whisper import WhisperModel  # optional
    FAST_WHISPER_AVAILABLE = True
except Exception:
    FAST_WHISPER_AVAILABLE = False
try:
    # Optional OpenAI whisper fallback
    from openai import OpenAI as OpenAIClient  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
import sqlite3

# Configure logging
import logging
class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return pyjson.dumps(payload)

handler = logging.StreamHandler()
handler.setFormatter(JsonLogFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Meeting Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simple API key auth + naive rate limiting ---
DEMO_API_KEY = os.getenv("DEMO_API_KEY")  # if set, require X-API-Key header
RATE_LIMIT_RPM = int(os.getenv("DEMO_RATE_LIMIT_RPM", "120"))
_rate_buckets: Dict[str, list] = {}
DEMO_JWT_SECRET = os.getenv("DEMO_JWT_SECRET")
TOKEN_BUCKET_RPS = float(os.getenv("DEMO_RATE_LIMIT_RPS", "3"))
TOKEN_BUCKET_BURST = int(os.getenv("DEMO_RATE_LIMIT_BURST", "10"))
try:
    import jwt  # PyJWT optional
    JWT_AVAILABLE = True
except Exception:
    JWT_AVAILABLE = False

# Metrics (simple counters)
METRICS = {
    "http_requests_total": 0,
    "search_requests_total": 0,
    "semantic_search_requests_total": 0,
    "ws_transcripts_total": 0,
    "asr_chunks_transcribed_total": 0,
}

@app.middleware("http")
async def auth_and_rate_limit(request, call_next):
    path = request.url.path
    # Skip for health and UI root
    if path in ("/", "/health"):
        return await call_next(request)
    # Auth
    if DEMO_JWT_SECRET and JWT_AVAILABLE and request.headers.get("Authorization", "").startswith("Bearer "):
        token = request.headers.get("Authorization")[7:]
        try:
            jwt.decode(token, DEMO_JWT_SECRET, algorithms=["HS256"])  # payload unused
        except Exception:
            return HTMLResponse(status_code=401, content="Unauthorized")
    elif DEMO_API_KEY:
        if request.headers.get("X-API-Key") != DEMO_API_KEY:
            return HTMLResponse(status_code=401, content="Unauthorized")
    # Rate limit per IP
    try:
        ip = request.client.host if request.client else "unknown"
        b = _rate_buckets.setdefault(ip, [TOKEN_BUCKET_BURST, time.time()])  # [tokens, last_ts]
        tokens, last_ts = b
        now = time.time()
        # refill
        tokens = min(TOKEN_BUCKET_BURST, tokens + (now - last_ts) * TOKEN_BUCKET_RPS)
        if tokens < 1:
            _rate_buckets[ip] = [tokens, now]
            return HTMLResponse(status_code=429, content="Rate limit exceeded")
        tokens -= 1
        _rate_buckets[ip] = [tokens, now]
    except Exception:
        pass
    METRICS["http_requests_total"] += 1
    return await call_next(request)

# Data models
class MeetingData(BaseModel):
    meeting_id: str
    title: str
    platform: str
    start_time: float
    privacy_mode: str
    participants: List[str]

class Utterance(BaseModel):
    speaker: str
    text: str
    timestamp: str
    confidence: float

class Suggestion(BaseModel):
    id: str
    kind: str
    text: str
    confidence: float
    reasons: List[str]
    status: str = "pending"
    source: str = "meeting"  # "meeting" or "document"

class UploadedFile(BaseModel):
    id: str
    filename: str
    size: int
    upload_time: str
    status: str = "uploaded"  # "uploaded", "processing", "ready", "error"
    content_type: str
    analysis: Optional[Dict[str, Any]] = None
    file_hash: Optional[str] = None

class FileAction(BaseModel):
    file_id: str
    action: str  # "summarize", "extract_actions", "ask_ai"
    query: Optional[str] = None

# In-memory storage (for demo purposes)
meetings = {}
transcripts = {}
suggestions = {}
uploaded_files = {}
summaries = {}

# --- SQLite persistence (optional, best-effort) ---
DB_PATH = os.getenv("DEMO_DB_PATH", "demo.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # performance/concurrency pragmas
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id TEXT PRIMARY KEY,
                title TEXT,
                platform TEXT,
                start_time REAL,
                privacy_mode TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT,
                speaker TEXT,
                text TEXT,
                timestamp TEXT,
                confidence REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                filename TEXT,
                size INTEGER,
                upload_time TEXT,
                status TEXT,
                content_type TEXT,
                file_hash TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                meeting_id TEXT PRIMARY KEY,
                summary TEXT,
                key_points TEXT,
                action_items TEXT
            )
            """
        )
        # Optional FTS5 (best-effort)
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
                    meeting_id, speaker, text, timestamp
                )
                """
            )
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                    id, filename, summary
                )
                """
            )
        except Exception as _:
            pass
        # Embeddings table (JSON vector for simplicity)
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    kind TEXT,
                    text TEXT,
                    vector TEXT
                )
                """
            )
        except Exception:
            pass
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"DB init failed: {e}")

def db_execute(query: str, params: tuple = ()):  # best-effort write
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"DB write skipped: {e}")

def db_query_all(query: str, params: tuple = ()):  # safe read helper
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.debug(f"DB read skipped: {e}")
        return []

# Async DB helpers (used when available)
async def adb_execute(query: str, params: tuple = ()):  # best-effort async write
    if not AIOSQLITE_AVAILABLE:
        return db_execute(query, params)
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute(query, params)
            await conn.commit()
    except Exception as e:
        logger.debug(f"ADB write skipped: {e}")

async def adb_query_all(query: str, params: tuple = ()):  # async read helper
    if not AIOSQLITE_AVAILABLE:
        return db_query_all(query, params)
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            async with conn.execute(query, params) as cur:
                rows = await cur.fetchall()
                return rows
    except Exception as e:
        logger.debug(f"ADB read skipped: {e}")
        return []

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Groq with better error handling
groq_client = None
groq_available = False

def initialize_groq():
    global groq_client, groq_available
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            logger.warning("No valid Groq API key found. Using mock responses.")
            return False

        groq_client = Groq(api_key=api_key)

        # Try different model names
        model_names = ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768']
        for model_name in model_names:
            try:
                # Test the model with a simple request
                groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                logger.info(f"Groq model '{model_name}' initialized successfully")
                groq_available = True
                return True
            except Exception as model_error:
                logger.warning(f"Failed to initialize model '{model_name}': {model_error}")
                continue

        logger.warning("All Groq models failed to initialize. Using mock responses.")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize Groq: {e}. Using mock responses.")
        return False

# Initialize Groq
initialize_groq()
active_connections = {}
init_db()
audio_buffers: Dict[str, bytearray] = {}
audio_last_flush: Dict[str, float] = {}

# Simulated AI responses
AI_RESPONSES = [
    "That's an interesting point. Would you like me to search for more information about this topic?",
    "I notice you mentioned a deadline. Should I create a task to track this?",
    "This sounds like an important decision. Would you like me to document this for the meeting summary?",
    "I can help you draft an email about this discussion. Would that be useful?",
    "I found some relevant context about this topic. Would you like me to share it?",
    "This seems like a good action item. Should I add it to the task list?",
    "I can help you schedule a follow-up meeting about this. Would that be helpful?",
    "I notice some key points here. Should I highlight these in the meeting summary?"
]

class ConnectionManager:
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

# Simulate AI processing
async def simulate_ai_processing(meeting_id: str, utterance: Utterance):
    """Process with Groq AI and generate suggestions"""
    try:
        if groq_available and groq_client:
            try:
                # Use Groq for real AI processing
                prompt = f"""You are an AI meeting assistant. Based on this meeting transcript:
            Speaker: {utterance.speaker}
            Text: {utterance.text}
            Timestamp: {utterance.timestamp}
            
            Generate a helpful suggestion for the meeting. Consider:
            - If it's a question, suggest follow-up questions
            - If it's a decision point, suggest next steps
            - If it's a problem, suggest solutions
            - If it's a task, suggest action items
            
Respond with a concise suggestion (max 150 words) that would be helpful for this meeting context."""
                
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7
                )
                
                # Parse Groq response
                ai_text = response.choices[0].message.content.strip()
                suggestion_data = {
                    "kind": "ask",
                    "text": ai_text[:200] + "..." if len(ai_text) > 200 else ai_text,
                    "confidence": 0.85,
                    "reasons": ["AI analysis", "Contextual understanding"]
                }
                logger.info("Successfully generated Groq AI suggestion")
            except Exception as groq_error:
                logger.warning(f"Groq API error during processing: {groq_error}")
                # Fallback to mock response
                import random
                suggestion_data = {
                    "kind": random.choice(["ask", "task", "fact", "email"]),
                    "text": random.choice(AI_RESPONSES),
                    "confidence": random.uniform(0.7, 0.95),
                    "reasons": ["AI detected relevant context", "Meeting pattern analysis"]
                }
        else:
            # Fallback to mock response
            import random
            suggestion_data = {
                "kind": random.choice(["ask", "task", "fact", "email"]),
                "text": random.choice(AI_RESPONSES),
                "confidence": random.uniform(0.7, 0.95),
                "reasons": ["AI detected relevant context", "Meeting pattern analysis"]
            }
        
        # Create suggestion
        suggestion = Suggestion(
            id=f"suggestion_{int(time.time())}",
            kind=suggestion_data["kind"],
            text=suggestion_data["text"],
            confidence=suggestion_data["confidence"],
            reasons=suggestion_data["reasons"],
            status="pending"
        )
        
        # Store suggestion (in-memory)
        suggestions[meeting_id] = suggestions.get(meeting_id, [])
        suggestions[meeting_id].append(suggestion)
        
        # Send suggestion to UI
        await manager.send_to_meeting(meeting_id, {
            "type": "suggestion",
            "suggestion": suggestion.model_dump()
        })
        
    except Exception as e:
        logger.error(f"Error in AI processing: {e}")
        # Fallback to simple response
        import random
        suggestion = Suggestion(
            id=f"suggestion_{int(time.time())}",
            kind="ask",
            text="I'm processing your input. Please continue the discussion.",
            confidence=0.5,
            reasons=["Processing in progress"],
            status="pending"
        )
        
        suggestions[meeting_id] = suggestions.get(meeting_id, [])
        suggestions[meeting_id].append(suggestion)
        await manager.send_to_meeting(meeting_id, {
            "type": "suggestion",
            "suggestion": suggestion.model_dump()
        })

# WebSocket endpoint
@app.websocket("/ws/audio/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    await manager.connect(websocket, meeting_id)
    
    try:
        while True:
            # Check if WebSocket is still connected
            if websocket.client_state.name != "CONNECTED":
                logger.info(f"WebSocket disconnected for meeting {meeting_id}")
                break
                
            try:
                # Try to receive text data first (for chat messages) with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                logger.info(f"Received text data: {data}")
                
                # Check if it's a JSON message
                try:
                    message = json.loads(data)
                    if message.get("type") == "chat":
                        # Handle chat message
                        utterance = Utterance(
                            speaker="User",
                            text=message.get("text", ""),
                            timestamp=datetime.now().isoformat(),
                            confidence=1.0
                        )
                        logger.info(f"Processed chat message: {utterance.text}")
                    else:
                        # Handle other text messages
                        utterance = Utterance(
                            speaker="Speaker 1",
                            text=data,
                            timestamp=datetime.now().isoformat(),
                            confidence=0.9
                        )
                except json.JSONDecodeError:
                    # Handle plain text messages
                    utterance = Utterance(
                        speaker="Speaker 1",
                        text=data,
                        timestamp=datetime.now().isoformat(),
                        confidence=0.9
                    )
                
            except asyncio.TimeoutError:
                logger.info(f"No data received for 30 seconds, checking connection...")
                # Check if connection is still alive
                if websocket.client_state.name != "CONNECTED":
                    logger.info(f"WebSocket disconnected for meeting {meeting_id}")
                    break
                continue
            except Exception as e:
                logger.warning(f"Failed to receive text, trying binary: {e}")
                # Try to receive binary data (for audio)
                try:
                    audio_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
                    # Transcribe server-side if possible
                    text = await transcribe_audio_bytes(audio_data)
                    if not text:
                        text = f"[Audio data received: {len(audio_data)} bytes]"
                    utterance = Utterance(
                        speaker="Speaker 1",
                        text=text,
                        timestamp=datetime.now().isoformat(),
                        confidence=0.9 if text and not text.startswith("[") else 0.8,
                    )
                    logger.info(f"Audio bytes: {len(audio_data)}; transcript: {text[:60]}...")
                except Exception as e2:
                    logger.warning(f"Failed to receive binary data: {e2}")
                    # If both fail, break the loop to prevent infinite retries
                    break
            
            # Store transcript
            if meeting_id not in transcripts:
                transcripts[meeting_id] = []
            transcripts[meeting_id].append(utterance)
            # persist transcript row
            try:
                await adb_execute(
                    "INSERT INTO transcripts (meeting_id, speaker, text, timestamp, confidence) VALUES (?,?,?,?,?)",
                    (meeting_id, utterance.speaker, utterance.text, utterance.timestamp, utterance.confidence)
                )
                # index into FTS (best-effort)
                await adb_execute(
                    "INSERT INTO transcripts_fts (meeting_id, speaker, text, timestamp) VALUES (?,?,?,?)",
                    (meeting_id, utterance.speaker, utterance.text, utterance.timestamp)
                )
                # upsert embedding
                emb = compute_embedding(utterance.text)
                await adb_execute(
                    "INSERT OR REPLACE INTO embeddings (id, kind, text, vector) VALUES (?,?,?,?)",
                    (f"T:{meeting_id}:{utterance.timestamp}", "transcript", utterance.text, pyjson.dumps(emb))
                )
                # fallback to simple text search if FTS fails
                try:
                    await adb_execute(
                        "INSERT INTO transcripts_fts (meeting_id, speaker, text, timestamp) VALUES (?,?,?,?)",
                        (meeting_id, utterance.speaker, utterance.text, utterance.timestamp)
                    )
                except Exception:
                    pass
                
            except Exception as _:
                pass
            
            # Send transcript to UI
            await manager.send_to_meeting(meeting_id, {
                "type": "transcript",
                "utterance": utterance.model_dump()
            })
            
            # Simulate AI processing
            await simulate_ai_processing(meeting_id, utterance)
    except WebSocketDisconnect:
        manager.disconnect(meeting_id)

# API endpoints
@app.post("/meetings/start")
async def start_meeting(metadata: MeetingData):
    meetings[metadata.meeting_id] = metadata
    logger.info(f"Started meeting {metadata.meeting_id}")
    # persist
    await adb_execute(
        "INSERT OR REPLACE INTO meetings (meeting_id, title, platform, start_time, privacy_mode) VALUES (?,?,?,?,?)",
        (
            metadata.meeting_id,
            metadata.title,
            metadata.platform,
            metadata.start_time,
            metadata.privacy_mode,
        ),
    )
    return {"status": "success", "meeting_id": metadata.meeting_id}

@app.post("/meetings/{meeting_id}/end")
async def end_meeting(meeting_id: str):
    if meeting_id in meetings:
        del meetings[meeting_id]
    manager.disconnect(meeting_id)
    logger.info(f"Ended meeting {meeting_id}")
    return {"status": "success", "meeting_id": meeting_id}

@app.get("/meetings/{meeting_id}/transcript")
async def get_transcript(meeting_id: str):
    items = [u.model_dump() for u in transcripts.get(meeting_id, [])]
    # also include from DB if any
    rows = await adb_query_all(
        "SELECT speaker, text, timestamp, confidence FROM transcripts WHERE meeting_id=? ORDER BY id ASC",
        (meeting_id,),
    )
    for speaker, text, ts, conf in rows:
        items.append({
            "speaker": speaker,
            "text": text,
            "timestamp": ts,
            "confidence": conf,
        })
    return {"meeting_id": meeting_id, "transcript": items}

# Simple search across transcript and uploaded file metadata/analysis
@app.get("/search")
async def search(query: str, meeting_id: Optional[str] = None, k: int = 10):
    q = (query or "").lower()
    hits = []
    # Try FTS5 first (best-effort)
    try:
        if meeting_id:
            rows = await adb_query_all(
                "SELECT speaker, text, timestamp FROM transcripts_fts WHERE text MATCH ? LIMIT ?",
                (query, k),
            )
            for speaker, text, ts in rows:
                hits.append({
                    "source": "transcript",
                    "speaker": speaker,
                    "text": text,
                    "timestamp": ts
                })
        rowsf = await adb_query_all(
            "SELECT id, filename, summary FROM files_fts WHERE files_fts MATCH ? LIMIT ?",
            (query, k),
        )
        for fid, fname, summ in rowsf:
            hits.append({
                "source": "document",
                "file_id": fid,
                "filename": fname,
                "snippet": (summ or "")[:280]
            })
    except Exception:
        pass
    # Search transcript
    if meeting_id and meeting_id in transcripts:
        for u in transcripts[meeting_id]:
            if q in u.text.lower():
                hits.append({
                    "source": "transcript",
                    "speaker": u.speaker,
                    "text": u.text,
                    "timestamp": u.timestamp
                })
    # Search uploaded files (filename and analysis summary)
    for f in uploaded_files.values():
        text_blob = f"{f.filename} { (f.analysis or {}).get('summary','') }"
        if q in text_blob.lower():
            hits.append({
                "source": "document",
                "file_id": f.id,
                "filename": f.filename,
                "snippet": (f.analysis or {}).get('summary', '')[:280]
            })
    # Also search DB files by filename
    db_files = await adb_query_all("SELECT id, filename FROM files WHERE filename LIKE ?", (f"%{query}%",))
    for fid, fname in db_files:
        hits.append({
            "source": "document",
            "file_id": fid,
            "filename": fname,
            "snippet": "",
        })
    result = {"query": query, "count": len(hits), "hits": hits[:k]}
    # If Groq available, add a structured answer with citations
    if groq_available and hits:
        try:
            # Build labeled snippets for citations
            labeled = []
            for idx, h in enumerate(hits[:k]):
                if h.get("source") == "transcript":
                    label = f"T{idx}"
                    labeled.append({"label": label, "type": "transcript", "timestamp": h.get("timestamp"), "speaker": h.get("speaker"), "text": h.get("text", "")})
                else:
                    label = f"D{idx}"
                    labeled.append({"label": label, "type": "document", "file_id": h.get("file_id"), "filename": h.get("filename"), "text": h.get("snippet", "")})
            prompt_context = "\n".join([f"[{x['label']}] ({x['type']}) {x.get('speaker','')}{'@'+x.get('timestamp','') if x['type']=='transcript' else ''}: {x['text']}" for x in labeled])
            prompt = (
                "You are a helpful meeting assistant. Based on the user's query and the labeled snippets, "
                "produce a brief answer as JSON with an 'answers' array. Each item has: text (string), sources (array of objects).\n"
                "Each source is either {type:'transcript', label:'T#', timestamp:'...'} or {type:'document', label:'D#', file_id:'...'}\n"
                "Return strictly valid JSON with only the 'answers' key.\n\n"
                f"Query: {query}\nSnippets:\n{prompt_context}"
            )
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": "You output only JSON."}, {"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            try:
                parsed = pyjson.loads(content)
                if isinstance(parsed, dict) and "answers" in parsed:
                    result["answers"] = parsed["answers"]
                else:
                    result["answer"] = content
            except Exception:
                result["answer"] = content
        except Exception as e:
            logger.debug(f"Groq search answer failed: {e}")
    return result

# Semantic search using embeddings
@app.get("/semantic_search")
async def semantic_search(query: str, page: int = 1, per_page: int = 10):
    METRICS["semantic_search_requests_total"] += 1
    qv = compute_embedding(query)
    # Pull candidates (limit for demo)
    rows = await adb_query_all("SELECT id, kind, text, vector FROM embeddings LIMIT 1000", ())
    scored = []
    for _id, kind, text, vec_json in rows:
        try:
            v = pyjson.loads(vec_json)
            score = cosine(qv, v)
            scored.append((_id, kind, text, float(score)))
        except Exception:
            continue
    scored.sort(key=lambda x: x[3], reverse=True)
    total = len(scored)
    if page < 1:
        page = 1
    start = (page - 1) * max(1, per_page)
    end = start + max(1, per_page)
    slice_items = scored[start:end]
    hits = []
    for _id, kind, text, score in slice_items:
        entry = {"id": _id, "kind": kind, "text": text, "score": score}
        if kind == "transcript" and _id.startswith("T:"):
            try:
                _, meeting_id, ts = _id.split(":", 2)
                entry.update({"meeting_id": meeting_id, "timestamp": ts})
            except Exception:
                pass
        if kind == "document" and _id.startswith("D:"):
            entry["file_id"] = _id[2:]
        if kind == "document" and _id.startswith("DPG:"):
            # DPG:{file_id}:{page_idx}:{chunk_idx}
            try:
                _, fid, pidx, _c = _id.split(":", 3)
                entry["file_id"] = fid
                entry["page_idx"] = int(pidx)
            except Exception:
                pass
        hits.append(entry)
    return {"query": query, "page": page, "per_page": per_page, "total": total, "count": len(hits), "hits": hits}

# Hybrid search combining keyword (FTS) and semantic (embeddings)
@app.get("/hybrid_search")
async def hybrid_search(query: str, meeting_id: Optional[str] = None, alpha: float = 0.5, k: int = 10):
    alpha = max(0.0, min(1.0, float(alpha)))
    # Keyword candidates from FTS tables
    tokens = [t for t in (query or "").lower().split() if t]
    kw_items = []
    try:
        if meeting_id:
            rows = await adb_query_all("SELECT meeting_id, speaker, text, timestamp FROM transcripts_fts WHERE text MATCH ? LIMIT 200", (query,))
        else:
            rows = await adb_query_all("SELECT meeting_id, speaker, text, timestamp FROM transcripts_fts LIMIT 200", ())
        for m, sp, txt, ts in rows:
            score = sum(txt.lower().count(t) for t in tokens)
            kid = f"FTS:T:{m}:{ts}"
            kw_items.append((kid, "transcript", txt, score, {"meeting_id": m, "timestamp": ts}))
    except Exception:
        pass
    try:
        frows = await adb_query_all("SELECT id, filename, summary FROM files_fts LIMIT 200", ())
        for fid, fname, summ in frows:
            score = sum((summ or "").lower().count(t) for t in tokens)
            kid = f"FTS:D:{fid}"
            kw_items.append((kid, "document", summ or "", score, {"file_id": fid}))
    except Exception:
        pass
    max_kw = max((s for _i,_k,_t,s,_m in kw_items), default=1.0) or 1.0

    # Semantic candidates from embeddings
    qv = compute_embedding(query)
    erows = await adb_query_all("SELECT id, kind, text, vector FROM embeddings LIMIT 1000", ())
    sem_items = []
    for _id, kind, text, vec_json in erows:
        try:
            v = pyjson.loads(vec_json)
            sem = cosine(qv, v)
            meta = {}
            if kind == "transcript" and _id.startswith("T:"):
                try:
                    _, m, ts = _id.split(":", 2)
                    meta = {"meeting_id": m, "timestamp": ts}
                except Exception:
                    pass
            if kind == "document" and _id.startswith("D:"):
                meta = {"file_id": _id[2:]}
            if kind == "document" and _id.startswith("DPG:"):
                try:
                    _, fid, pidx, _c = _id.split(":", 3)
                    meta = {"file_id": fid, "page_idx": int(pidx)}
                except Exception:
                    meta = {"file_id": _id}
            sem_items.append((_id, kind, text, sem, meta))
        except Exception:
            continue

    # Merge by id preference; if an id appears only in one list, missing score is 0
    merged = {}
    for kid, kind, text, s, meta in kw_items:
        merged[kid] = {"id": kid, "kind": kind, "text": text, "kw": (s/max_kw), "sem": 0.0, **meta}
    for _id, kind, text, sem, meta in sem_items:
        e = merged.get(_id)
        if e:
            e["sem"] = sem
        else:
            merged[_id] = {"id": _id, "kind": kind, "text": text, "kw": 0.0, "sem": sem, **meta}

    scored = []
    for v in merged.values():
        score = alpha * float(v["sem"]) + (1.0 - alpha) * float(v["kw"]) 
        v["score"] = score
        scored.append(v)
    scored.sort(key=lambda x: x["score"], reverse=True)
    hits = scored[:k]
    return {"query": query, "alpha": alpha, "k": k, "count": len(hits), "hits": hits}

@app.get("/meetings/{meeting_id}/suggestions")
async def get_suggestions(meeting_id: str):
    return {
        "meeting_id": meeting_id,
        "suggestions": [s.model_dump() for s in suggestions.get(meeting_id, [])]
    }

# Generate and cache a simple meeting summary
@app.post("/meetings/{meeting_id}/summarize")
async def summarize_meeting(meeting_id: str):
    uts = transcripts.get(meeting_id, [])
    all_text = "\n".join([f"{u.speaker}: {u.text}" for u in uts[-100:]])
    files = [f.filename for f in uploaded_files.values()]
    # Default summary
    summary = {
        "summary": "This meeting covered several topics.",
        "key_points": [],
        "action_items": [],
    }
    if groq_available and all_text:
        try:
            prompt = (
                "Summarize the following meeting transcript. Provide:\n"
                "1) A concise 2-3 sentence summary\n"
                "2) 3-5 bullet key points\n"
                "3) 2-3 actionable next steps\n"
                f"Transcript (truncated):\n{all_text}\n"
                f"Files referenced: {', '.join(files) if files else 'none'}"
            )
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": "You output only JSON."}, {"role": "user", "content": (
                    "Return a compact JSON object with keys: summary (string), key_points (array of 3-5 short strings), action_items (array of 2-4 short strings).\n" +
                    "Make it strictly valid JSON and nothing else.\n\n" +
                    prompt
                )}],
                max_tokens=250,
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            # Try strict JSON parse first
            parsed = None
            try:
                parsed = pyjson.loads(content)
            except Exception:
                # fallback to naive parsing
                summary_text, key_points, action_items = parse_summary_text(content)
                parsed = {"summary": summary_text, "key_points": key_points, "action_items": action_items}
            summary = {
                "summary": parsed.get("summary") or "",
                "key_points": parsed.get("key_points") or [],
                "action_items": parsed.get("action_items") or [],
            }
        except Exception as e:
            logger.debug(f"Groq summarize failed, falling back: {e}")
            last_texts = [u.text for u in uts[-10:]]
            summary = {
                "summary": (
                    "This meeting discussed: " + ("; ".join(last_texts[:3]) or "general topics") +
                    (". Referenced files: " + ", ".join(files) if files else ".")
                ),
                "key_points": last_texts[:5],
                "action_items": ["Review notes and finalize next steps", "Follow up with stakeholders"],
            }
    else:
        last_texts = [u.text for u in uts[-10:]]
        summary = {
            "summary": (
                "This meeting discussed: " + ("; ".join(last_texts[:3]) or "general topics") +
                (". Referenced files: " + ", ".join(files) if files else ".")
            ),
            "key_points": last_texts[:5],
            "action_items": ["Review notes and finalize next steps", "Follow up with stakeholders"],
        }
    summaries[meeting_id] = summary
    # persist
    await adb_execute(
        "INSERT OR REPLACE INTO summaries (meeting_id, summary, key_points, action_items) VALUES (?,?,?,?)",
        (meeting_id, summary["summary"], "\n".join(summary["key_points"]), "\n".join(summary["action_items"]))
    )
    # Also broadcast as a suggestion card
    await manager.send_to_meeting(meeting_id, {
        "type": "suggestion",
        "suggestion": Suggestion(
            id=f"summary_{int(time.time())}",
            kind="summary",
            text=summary["summary"],
            confidence=0.9,
            reasons=["Auto-generated from transcript"],
        ).model_dump()
    })
    return {"status": "ok", **summary}

@app.get("/meetings/{meeting_id}/summary")
async def get_summary(meeting_id: str):
    s = summaries.get(meeting_id)
    if s:
        return s
    rows = db_query_all("SELECT summary, key_points, action_items FROM summaries WHERE meeting_id=?", (meeting_id,))
    if rows:
        summ, kp, ai = rows[0]
        return {"summary": summ, "key_points": (kp.split("\n") if kp else []), "action_items": (ai.split("\n") if ai else [])}
    return {"summary": "No summary yet", "key_points": [], "action_items": []}

# --- Helpers ---
def parse_summary_text(content: str):
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    summary_lines, keys, acts = [], [], []
    mode = "summary"
    for l in lines:
        low = l.lower()
        if any(h in low for h in ["key points", "key bullets", "bullets", "points:"]):
            mode = "keys"; continue
        if any(h in low for h in ["action items", "next steps", "actions:"]):
            mode = "acts"; continue
        if mode == "summary":
            summary_lines.append(l.strip("-• "))
        elif mode == "keys":
            keys.append(l.strip("-• "))
        else:
            acts.append(l.strip("-• "))
    summary_text = " ".join(summary_lines) or (lines[0] if lines else content)
    return summary_text, keys[:5], acts[:5]

# Embedding helpers
try:
    from sentence_transformers import SentenceTransformer  # optional heavy dep
    _st_model = None
    SENTENCE_TRANS_AVAILABLE = True
except Exception:
    SENTENCE_TRANS_AVAILABLE = False

def lazy_st_model():
    global _st_model
    if _st_model is None and SENTENCE_TRANS_AVAILABLE:
        name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _st_model = SentenceTransformer(name)
    return _st_model

def hash_embed(text: str, dim: int = 256) -> List[float]:
    # Simple hashing trick embedding as fallback
    vec = [0.0] * dim
    for token in (text or "").lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec]

def compute_embedding(text: str) -> List[float]:
    try:
        model = lazy_st_model()
        if model is not None:
            v = model.encode([text])[0]
            # Normalize
            norm = float((v**2).sum())**0.5 if hasattr(v, "sum") else math.sqrt(sum(float(x)*float(x) for x in v))
            return [float(x)/ (norm or 1.0) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    except Exception as e:
        logger.debug(f"ST embedding failed: {e}")
    return hash_embed(text)

def cosine(a: List[float], b: List[float]) -> float:
    s = 0.0
    for i in range(min(len(a), len(b))):
        s += a[i]*b[i]
    return s

async def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using faster-whisper or OpenAI whisper if available. Returns empty string on failure."""
    # Try faster-whisper
    if FAST_WHISPER_AVAILABLE:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                # Lazy load model to avoid startup hit
                model = WhisperModel(os.getenv("WHISPER_MODEL", "tiny"), device="cpu", compute_type="int8")
                segments, _ = model.transcribe(tmp.name, beam_size=1)
                text = " ".join([seg.text.strip() for seg in segments])
                return text.strip()
        except Exception as e:
            logger.debug(f"faster-whisper failed: {e}")
    # Try OpenAI Whisper
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAIClient()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                with open(tmp.name, "rb") as fh:
                    resp = client.audio.transcriptions.create(model="whisper-1", file=fh)
                # resp.text depending on SDK; fallback to str
                text = getattr(resp, "text", None) or str(resp)
                return text.strip()
        except Exception as e:
            logger.debug(f"OpenAI whisper failed: {e}")
    return ""

@app.post("/suggestions/{suggestion_id}/approve")
async def approve_suggestion(suggestion_id: str):
    for meeting_id, meeting_suggestions in suggestions.items():
        for suggestion in meeting_suggestions:
            if suggestion.id == suggestion_id:
                suggestion.status = "approved"
                return {"status": "success", "suggestion_id": suggestion_id}
    return {"status": "error", "message": "Suggestion not found"}

@app.post("/suggestions/{suggestion_id}/reject")
async def reject_suggestion(suggestion_id: str):
    for meeting_id, meeting_suggestions in suggestions.items():
        for suggestion in meeting_suggestions:
            if suggestion.id == suggestion_id:
                suggestion.status = "rejected"
                return {"status": "success", "suggestion_id": suggestion_id}
    return {"status": "error", "message": "Suggestion not found"}

# File upload endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), meeting_id: str = Form(...)):
    """Upload a file and process it for RAG analysis"""
    try:
        # Read file content first to check for duplicates
        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()
        
        # Check for duplicate files
        for existing_file in uploaded_files.values():
            if hasattr(existing_file, 'file_hash') and existing_file.file_hash == file_hash:
                logger.info(f"Duplicate file detected: {file.filename}")
                return {"status": "duplicate", "message": "File already uploaded", "file_id": existing_file.id}
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save file to uploads directory
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Create file record
        uploaded_file = UploadedFile(
            id=file_id,
            filename=file.filename,
            size=len(content),
            upload_time=datetime.now().isoformat(),
            status="processing",
            content_type=file.content_type,
            file_hash=file_hash
        )
        
        uploaded_files[file_id] = uploaded_file
        # persist file metadata (initial status: processing)
        await adb_execute(
            "INSERT OR REPLACE INTO files (id, filename, size, upload_time, status, content_type, file_hash) VALUES (?,?,?,?,?,?,?)",
            (
                file_id,
                uploaded_file.filename,
                uploaded_file.size,
                uploaded_file.upload_time,
                uploaded_file.status,
                uploaded_file.content_type,
                uploaded_file.file_hash,
            ),
        )
        
        # Simulate file processing
        await asyncio.sleep(2)  # Simulate processing time
        
        # Update file status
        # Optional text extraction for search quality
        extracted_text = None
        extracted_pages = None  # for PDFs
        try:
            fname = (file.filename or "").lower()
            if file.content_type == "application/pdf" or fname.endswith(".pdf"):
                try:
                    import pypdf  # type: ignore
                    reader = pypdf.PdfReader(io.BytesIO(content))
                    extracted_pages = [page.extract_text() or "" for page in reader.pages]
                    extracted_text = "\n".join(extracted_pages)
                except Exception:
                    extracted_text = None
            elif file.content_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",) or fname.endswith(".docx"):
                try:
                    import docx  # type: ignore
                    doc = docx.Document(io.BytesIO(content))
                    extracted_text = "\n".join(p.text for p in doc.paragraphs)
                except Exception:
                    extracted_text = None
            elif file.content_type.startswith("text/") or fname.endswith((".txt", ".md")):
                try:
                    extracted_text = content.decode("utf-8", errors="replace")
                except Exception:
                    extracted_text = None
        except Exception:
            extracted_text = None

        uploaded_file.status = "ready"
        summary_blob = extracted_text[:1000] + "..." if extracted_text and len(extracted_text) > 1000 else (extracted_text or f"Document '{file.filename}' has been processed and indexed for analysis.")
        uploaded_file.analysis = {
            "summary": summary_blob,
            "key_topics": ["strategy", "planning", "objectives"],
            "action_items": ["Review quarterly goals", "Update project timeline"],
            "word_count": len(extracted_text.split()) if extracted_text else (len(content.split()) if file.content_type.startswith('text/') else "N/A")
        }
        # update DB status to ready
        await adb_execute(
            "UPDATE files SET status=? WHERE id=?",
            (uploaded_file.status, file_id),
        )
        # index into FTS (best-effort)
        try:
            await adb_execute(
                "INSERT INTO files_fts (id, filename, summary) VALUES (?,?,?)",
                (file_id, uploaded_file.filename, uploaded_file.analysis.get("summary", ""))
            )
        except Exception as _:
            pass
        # upsert embedding for file summary
        try:
            emb = compute_embedding(uploaded_file.analysis.get("summary", ""))
            await adb_execute(
                "INSERT OR REPLACE INTO embeddings (id, kind, text, vector) VALUES (?,?,?,?)",
                (f"D:{file_id}", "document", uploaded_file.analysis.get("summary", ""), pyjson.dumps(emb))
            )
        except Exception:
            pass
        # upsert embeddings for full document text in chunks if available
        try:
            if extracted_text:
                # If we have page-level text (PDF), embed per page; otherwise chunk by text length
                if extracted_pages:
                    for p_idx, p_txt in enumerate(extracted_pages):
                        txt = (p_txt or "").strip()
                        if not txt:
                            continue
                        # further chunk long pages ~1000 chars
                        chunk_size = 1000
                        for c_idx in range(0, len(txt), chunk_size):
                            chunk = txt[c_idx:c_idx+chunk_size]
                            if not chunk.strip():
                                continue
                            emb_page = compute_embedding(chunk)
                            await adb_execute(
                                "INSERT OR REPLACE INTO embeddings (id, kind, text, vector) VALUES (?,?,?,?)",
                                (f"DPG:{file_id}:{p_idx}:{c_idx//chunk_size}", "document", chunk, pyjson.dumps(emb_page))
                            )
                else:
                    chunk_size = 1000
                    max_chars = 8000
                    text = extracted_text[:max_chars]
                    for idx in range(0, len(text), chunk_size):
                        chunk = text[idx: idx+chunk_size]
                        if not chunk.strip():
                            continue
                        emb_full = compute_embedding(chunk)
                        await adb_execute(
                            "INSERT OR REPLACE INTO embeddings (id, kind, text, vector) VALUES (?,?,?,?)",
                            (f"DTXT:{file_id}:{idx//chunk_size}", "document", chunk, pyjson.dumps(emb_full))
                        )
        except Exception:
            pass
        
        # Send file upload event to transcript
        if meeting_id in manager.active_connections:
            await manager.send_to_meeting(meeting_id, {
                "type": "file_uploaded",
                "file": uploaded_file.model_dump(),
                "transcript_entry": {
                    "speaker": "System",
                    "text": f"📄 {file.filename} uploaded and indexed. AI will use it for contextual answers.",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 1.0
                }
            })
        
        # Generate document-based suggestions
        doc_suggestion = Suggestion(
            id=f"doc_suggestion_{int(time.time())}",
            kind="fact",
            text=f"Based on {file.filename}, I can help answer questions about the content and provide contextual insights.",
            confidence=0.9,
            reasons=["Document analysis", "Content indexing"],
            source="document"
        )
        
        suggestions[meeting_id] = suggestions.get(meeting_id, [])
        suggestions[meeting_id].append(doc_suggestion)
        
        # Send document-based suggestion
        if meeting_id in manager.active_connections:
            await manager.send_to_meeting(meeting_id, {
                "type": "suggestion",
                "suggestion": doc_suggestion.model_dump()
            })
        
        return {"status": "success", "file_id": file_id, "file": uploaded_file.model_dump()}
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/files/{meeting_id}")
async def get_uploaded_files(meeting_id: str):
    """Get all uploaded files for a meeting"""
    meeting_files = [f for f in uploaded_files.values() if f.status == "ready"]
    return {"meeting_id": meeting_id, "files": [f.model_dump() for f in meeting_files]}

@app.post("/files/{file_id}/action")
async def perform_file_action(file_id: str, action: FileAction):
    """Perform an action on an uploaded file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file = uploaded_files[file_id]
    
    if action.action == "summarize":
        result = {
            "action": "summarize",
            "file_id": file_id,
            "summary": f"Summary of {file.filename}: This document contains strategic planning information with key objectives and action items for Q4 planning.",
            "key_points": ["Strategic objectives", "Timeline considerations", "Resource requirements"]
        }
    elif action.action == "extract_actions":
        result = {
            "action": "extract_actions",
            "file_id": file_id,
            "action_items": [
                "Review and approve Q4 budget allocation",
                "Schedule follow-up meeting with stakeholders",
                "Update project timeline based on new requirements"
            ]
        }
    elif action.action == "ask_ai":
        result = {
            "action": "ask_ai",
            "file_id": file_id,
            "query": action.query,
            "answer": f"Based on {file.filename}, here's what I found regarding '{action.query}': [AI-generated response based on document content]"
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    return result

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file = uploaded_files[file_id]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        del uploaded_files[file_id]
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ai-meeting-assistant",
        "groq_available": groq_available,
        "groq_client": "Groq Client" if groq_client else None,
        "asr_server": (
            "faster-whisper" if FAST_WHISPER_AVAILABLE else ("openai" if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY") else "none")
        ),
        "embeddings": "sentence-transformers" if SENTENCE_TRANS_AVAILABLE else "hashing",
    }

@app.get("/metrics")
async def metrics():
    # Simple JSON metrics for demo
    return METRICS

# Serve the demo HTML with Co-pilot Nexus UI
@app.get("/")
async def get_demo():
    # Serve embedded HTML UI
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Meeting Assistant</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            :root {
                --background: 224 15% 8%;
                --foreground: 210 40% 98%;
                --card: 224 20% 12%;
                --card-foreground: 210 40% 98%;
                --popover: 224 20% 12%;
                --popover-foreground: 210 40% 98%;
                --primary: 261 70% 58%;
                --primary-foreground: 210 40% 98%;
                --primary-glow: 261 70% 68%;
                --secondary: 224 15% 18%;
                --secondary-foreground: 210 40% 98%;
                --success: 142 72% 45%;
                --success-foreground: 210 40% 98%;
                --danger: 0 84% 60%;
                --danger-foreground: 210 40% 98%;
                --muted: 224 15% 15%;
                --muted-foreground: 215 20% 65%;
                --accent: 224 15% 20%;
                --accent-foreground: 210 40% 98%;
                --destructive: 0 84% 60%;
                --destructive-foreground: 210 40% 98%;
                --border: 224 15% 20%;
                --input: 224 15% 15%;
                --ring: 261 70% 58%;
                --radius: 0.75rem;
                --gradient-primary: linear-gradient(135deg, hsl(261 70% 58%), hsl(261 70% 68%));
                --gradient-card: linear-gradient(135deg, hsl(224 20% 12%), hsl(224 15% 15%));
                --gradient-subtle: linear-gradient(180deg, hsl(224 15% 10%), hsl(224 15% 8%));
                --shadow-glow: 0 0 40px hsl(261 70% 58% / 0.15);
                --shadow-card: 0 8px 32px hsl(224 50% 2% / 0.4);
                --shadow-elevated: 0 16px 64px hsl(224 50% 2% / 0.6);
                --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                --transition-spring: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            }
            
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Inter', sans-serif;
                background: hsl(var(--background));
                color: hsl(var(--foreground));
                line-height: 1.6;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                transition: all 0.3s ease;
            }
            
            .status-connected {
                background: hsl(var(--success));
                box-shadow: 0 0 12px hsl(var(--success) / 0.6);
            }
            
            .status-disconnected {
                background: hsl(var(--danger));
                box-shadow: 0 0 12px hsl(var(--danger) / 0.6);
            }

            .ai-card {
                background: var(--gradient-card);
                border: 1px solid hsl(var(--border));
                border-radius: 12px;
                padding: 24px;
                backdrop-filter: blur(8px);
                box-shadow: var(--shadow-card);
                transition: var(--transition-smooth);
            }

            .ai-card:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-elevated);
            }

            .ai-button-primary {
                background: var(--gradient-primary);
                color: hsl(var(--primary-foreground));
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 500;
                box-shadow: var(--shadow-glow);
                transition: var(--transition-smooth);
                cursor: pointer;
            }

            .ai-button-primary:hover {
                transform: translateY(-1px);
                box-shadow: 0 0 50px hsl(261 70% 58% / 0.25);
            }

            .ai-button-primary:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }

            .ai-input {
                background: hsl(var(--input));
                border: 1px solid hsl(var(--border));
                border-radius: 8px;
                padding: 12px 16px;
                color: hsl(var(--foreground));
                transition: var(--transition-smooth);
                width: 100%;
            }

            .ai-input:focus {
                outline: none;
                border-color: hsl(var(--primary));
                box-shadow: 0 0 0 2px hsl(var(--primary) / 0.2);
            }

            .ai-input:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .suggestion-card {
                background: var(--gradient-card);
                border: 1px solid hsl(var(--border));
                border-radius: 8px;
                padding: 16px;
                transition: var(--transition-smooth);
                animation: slideInRight 0.4s ease-out;
            }

            .suggestion-card:hover {
                transform: translateX(4px);
                box-shadow: var(--shadow-card);
            }

            .transcript-entry {
                animation: slideInLeft 0.4s ease-out;
            }

            @keyframes slideInLeft {
                from { opacity: 0; transform: translateX(-20px); }
                to { opacity: 1; transform: translateX(0); }
            }

            @keyframes slideInRight {
                from { opacity: 0; transform: translateX(20px); }
                to { opacity: 1; transform: translateX(0); }
            }

            .gradient-text {
                background: var(--gradient-primary);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: 500;
                border: 1px solid;
            }

            .badge-ask { background: hsl(220 70% 50% / 0.1); color: hsl(220 70% 70%); border-color: hsl(220 70% 50% / 0.2); }
            .badge-email { background: hsl(142 70% 50% / 0.1); color: hsl(142 70% 70%); border-color: hsl(142 70% 50% / 0.2); }
            .badge-task { background: hsl(30 70% 50% / 0.1); color: hsl(30 70% 70%); border-color: hsl(30 70% 50% / 0.2); }
            .badge-fact { background: hsl(280 70% 50% / 0.1); color: hsl(280 70% 70%); border-color: hsl(280 70% 50% / 0.2); }
            .badge-document { background: hsl(200 70% 50% / 0.1); color: hsl(200 70% 70%); border-color: hsl(200 70% 50% / 0.2); }

            .file-item {
                background: hsl(var(--muted));
                border: 1px solid hsl(var(--border));
                border-radius: 8px;
                padding: 12px;
                transition: var(--transition-smooth);
            }

            .file-item:hover {
                background: hsl(var(--accent));
                transform: translateX(2px);
            }

            .file-actions {
                display: flex;
                gap: 4px;
                margin-top: 8px;
            }

            .file-action-btn {
                background: hsl(var(--secondary));
                border: 1px solid hsl(var(--border));
                color: hsl(var(--secondary-foreground));
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                cursor: pointer;
                transition: var(--transition-smooth);
            }

            .file-action-btn:hover {
                background: hsl(var(--primary));
                color: hsl(var(--primary-foreground));
            }

            .upload-area {
                transition: var(--transition-smooth);
            }

            .upload-area.dragover {
                border-color: hsl(var(--primary));
                background: hsl(var(--primary) / 0.05);
            }

            .suggestion-tab {
                background: hsl(var(--muted));
                border: 1px solid hsl(var(--border));
                color: hsl(var(--muted-foreground));
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 12px;
                cursor: pointer;
                transition: var(--transition-smooth);
            }

            .suggestion-tab.active {
                background: hsl(var(--primary));
                color: hsl(var(--primary-foreground));
                border-color: hsl(var(--primary));
            }

            .suggestion-tab:hover:not(.active) {
                background: hsl(var(--accent));
                color: hsl(var(--accent-foreground));
            }
        </style>
    </head>
    <body class="min-h-screen">
        <!-- Header -->
        <header class="border-b border-gray-700 bg-gray-800/50 backdrop-blur-sm sticky top-0 z-10">
            <div class="container mx-auto px-6 py-6">
                <div class="flex items-center gap-4 mb-2">
                    <div class="p-3 rounded-xl bg-purple-500/10 backdrop-blur-sm">
                        <i data-lucide="bot" class="h-8 w-8 text-purple-400"></i>
                    </div>
                    <div>
                        <h1 class="text-3xl font-bold gradient-text">AI Meeting Assistant</h1>
                        <p class="text-gray-400 text-lg">Your intelligent co-pilot for productive meetings</p>
                    </div>
                    <span id="asrEngineBadge" class="ml-auto badge bg-gray-700 text-gray-300">ASR: unknown</span>
                </div>
                <div class="flex items-center gap-2 text-sm text-gray-400">
                    <i data-lucide="sparkles" class="h-4 w-4"></i>
                    <span>Real-time transcription • Smart suggestions • Seamless collaboration</span>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="container mx-auto px-6 py-8">
            <!-- Meeting Header -->
            <div class="ai-card mb-8">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h2 class="text-2xl font-bold mb-2">Q4 Strategic Planning Meeting</h2>
                        <p class="text-gray-400">Duration: <span id="duration" class="text-white font-mono">00:00</span> | 5 participants</p>
                    </div>
                    <div class="flex items-center gap-3">
                        <button class="ai-button-primary px-4 py-2" onclick="document.getElementById('fileInput').click()">
                            <i data-lucide="upload" class="h-4 w-4 mr-2"></i>
                            Upload
                        </button>
                        <button class="ai-button-secondary px-4 py-2" onclick="toggleSettings()">
                            <i data-lucide="settings" class="h-4 w-4"></i>
                        </button>
                        <div class="status-indicator" id="statusIndicator"></div>
                        <span id="statusText" class="text-sm font-medium">Disconnected</span>
                    </div>
                </div>
                <div class="flex gap-2">
                    <span class="badge bg-gray-700 text-gray-300">Alex Thompson</span>
                    <span class="badge bg-gray-700 text-gray-300">Sarah Chen</span>
                    <span class="badge bg-gray-700 text-gray-300">Marcus Johnson</span>
                    <span class="badge bg-gray-700 text-gray-300">Emily Rodriguez</span>
                    <span class="badge bg-gray-700 text-gray-300">David Kim</span>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                <!-- Meeting Controls -->
                <div class="ai-card">
                    <div class="space-y-6">
                        <div class="flex items-center gap-3">
                            <div class="status-indicator" id="connectionIndicator"></div>
                            <div>
                                <h3 class="text-lg font-semibold">Meeting Status</h3>
                                <p class="text-sm text-gray-400" id="connectionStatus">Not connected</p>
                            </div>
                        </div>

                        <div class="flex gap-3">
                            <button id="connectBtn" class="ai-button-primary flex-1" onclick="toggleConnection()">
                                <i data-lucide="play" class="h-4 w-4 mr-2"></i>
                                <span id="connectText">Start Meeting</span>
                            </button>
                            <button id="recordBtn" class="ai-button-primary" onclick="toggleRecording()" disabled>
                                <i data-lucide="mic" class="h-4 w-4 mr-2"></i>
                                <span id="recordText">Record</span>
                            </button>
                        </div>

                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Chat Input</h3>
                            <div class="flex gap-2">
                                <input type="text" id="messageInput" class="ai-input flex-1" 
                                       placeholder="Type your message or question..." disabled>
                                <button id="sendBtn" class="ai-button-primary px-4" onclick="sendMessage()" disabled>
                                    <i data-lucide="send" class="h-4 w-4"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- AI Suggestions -->
                <div class="ai-card">
                    <div class="space-y-6">
                        <div class="flex items-center gap-3">
                            <div class="p-2 rounded-lg bg-purple-500/10">
                                <i data-lucide="sparkles" class="h-5 w-5 text-purple-400"></i>
                            </div>
                            <div>
                                <h3 class="text-lg font-semibold">AI Suggestions</h3>
                                <p class="text-sm text-gray-400">Smart recommendations for your meeting</p>
                            </div>
                        </div>

                        <!-- Suggestion Tabs -->
                        <div class="flex gap-2 mb-4">
                            <button class="suggestion-tab active" onclick="switchSuggestionTab('all')" id="tab-all">
                                All
                            </button>
                            <button class="suggestion-tab" onclick="switchSuggestionTab('questions')" id="tab-questions">
                                Questions
                            </button>
                            <button class="suggestion-tab" onclick="switchSuggestionTab('tasks')" id="tab-tasks">
                                Tasks
                            </button>
                            <button class="suggestion-tab" onclick="switchSuggestionTab('insights')" id="tab-insights">
                                Insights
                            </button>
                        </div>

                        <div id="suggestions" class="space-y-4 max-h-96 overflow-y-auto">
                            <div class="suggestion-card" data-category="questions">
                                <div class="flex items-start gap-3 mb-3">
                                    <div class="p-1.5 rounded bg-purple-500/10">
                                        <i data-lucide="message-square" class="h-4 w-4 text-purple-400"></i>
                                    </div>
                                    <div class="flex-1">
                                        <div class="flex items-center gap-2 mb-2">
                                            <span class="badge badge-ask">ASK</span>
                                            <span class="text-xs text-gray-500">Confidence: 85%</span>
                                        </div>
                                        <p class="text-sm text-gray-200 leading-relaxed">
                                            Ask about the project timeline and key milestones for Q4.
                                        </p>
                                    </div>
                                </div>
                                <div class="flex gap-2">
                                    <button class="flex-1 bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded text-sm font-medium" 
                                            onclick="approveSuggestion('suggestion_1')">
                                        <i data-lucide="check" class="h-3 w-3 mr-1 inline"></i>
                                        Approve
                                    </button>
                                    <button class="flex-1 border border-red-500 text-red-400 hover:bg-red-500/10 px-3 py-2 rounded text-sm font-medium" 
                                            onclick="rejectSuggestion('suggestion_1')">
                                        <i data-lucide="x" class="h-3 w-3 mr-1 inline"></i>
                                        Reject
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Document Upload -->
                <div class="ai-card">
                    <div class="space-y-6">
                        <div class="flex items-center gap-3">
                            <div class="p-2 rounded-lg bg-blue-500/10">
                                <i data-lucide="upload" class="h-5 w-5 text-blue-400"></i>
                            </div>
                            <div>
                                <h3 class="text-lg font-semibold">Document Upload</h3>
                                <p class="text-sm text-gray-400">Upload files for AI analysis</p>
                            </div>
                        </div>

                        <!-- File Upload Area -->
                        <div id="fileUploadArea" class="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-purple-500 transition-colors cursor-pointer" 
                             onclick="document.getElementById('fileInput').click()">
                            <i data-lucide="upload-cloud" class="h-12 w-12 text-gray-400 mx-auto mb-4"></i>
                            <p class="text-gray-400 mb-2">Drag & drop files here or click to upload</p>
                            <p class="text-xs text-gray-500">PDF, DOC, TXT, MD files supported</p>
                        </div>
                        
                        <input type="file" id="fileInput" class="hidden" multiple accept=".pdf,.doc,.docx,.txt,.md" onchange="handleFileUpload(event)">

                        <!-- Uploaded Files List -->
                        <div id="uploadedFiles" class="space-y-3 max-h-64 overflow-y-auto">
                            <!-- Files will be added here dynamically -->
                        </div>

                        <!-- Search Controls -->
                        <div class="space-y-2">
                            <h4 class="text-sm font-medium text-gray-300">Search Documents & Transcript</h4>
                            <div class="flex gap-2 items-center">
                                <input type="text" id="searchInput" class="ai-input flex-1" placeholder="Search across all content...">
                                <button id="searchBtn" class="ai-button-primary px-4" onclick="performSearch()"><i data-lucide="search" class="h-4 w-4"></i></button>
                                <div id="semanticPager" class="flex items-center gap-2 text-xs">
                                    <button class="ai-button-secondary px-2" onclick="semanticPrev()">Prev</button>
                                    <span id="semanticPageLabel" class="text-gray-400">Page 1</span>
                                    <button class="ai-button-secondary px-2" onclick="semanticNext()">Next</button>
                                </div>
                            </div>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-3" id="searchPanels">
                                <div class="p-3 rounded bg-gray-800/40 border border-gray-700">
                                    <div class="text-sm text-gray-300 mb-2">Keyword Results</div>
                                    <div id="searchResultsKeyword" class="space-y-2 text-sm text-gray-200"></div>
                                </div>
                                <div class="p-3 rounded bg-gray-800/40 border border-gray-700">
                                    <div class="flex items-center justify-between mb-2">
                                        <div class="text-sm text-gray-300">Semantic Results</div>
                                        <div class="text-xs text-gray-500" id="semanticTotal"></div>
                                    </div>
                                    <div id="searchResultsSemantic" class="space-y-2 text-sm text-gray-200"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Live Transcript -->
            <div class="ai-card">
                <div class="space-y-4">
                    <div class="flex items-center justify-between">
                        <h3 class="text-lg font-semibold flex items-center gap-2">
                            <i data-lucide="file-text" class="h-5 w-5"></i>
                            Live Transcript
                        </h3>
                        <div class="flex gap-2">
                            <button class="ai-button-secondary px-3 py-1 text-sm" onclick="exportTranscript()">
                                <i data-lucide="download" class="h-4 w-4 mr-1"></i>
                                Export
                            </button>
                            <button class="ai-button-primary px-3 py-1 text-sm" onclick="generateSummary()">
                                <i data-lucide="file-text" class="h-4 w-4 mr-1"></i>
                                Summary
                            </button>
                        </div>
                    </div>
                    <div id="transcript" class="space-y-3 max-h-96 overflow-y-auto">
                        <div class="transcript-entry bg-gray-800/50 p-4 rounded-lg border-l-4 border-blue-500">
                            <div class="flex justify-between items-start mb-2">
                                <span class="font-semibold text-blue-400">System</span>
                                <span class="text-xs text-gray-500">Just now</span>
                            </div>
                            <p class="text-gray-200">AI Meeting Assistant initialized. Ready to help you with your meeting.</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Post-Meeting Summary -->
            <div class="ai-card" id="summarySection" style="display: none;">
                <div class="space-y-4">
                    <h3 class="text-lg font-semibold flex items-center gap-2">
                        <i data-lucide="clipboard-list" class="h-5 w-5"></i>
                        Meeting Summary
                    </h3>
                    <div id="meetingSummary" class="space-y-4">
                        <!-- Summary content will be generated here -->
                    </div>
                </div>
            </div>
        </main>

        <script>
            let ws = null;
            let isConnected = false;
            let isRecording = false;
            let meetingId = 'meeting_' + Date.now();
            let startTime = Date.now();
            
            // Initialize Lucide icons
            lucide.createIcons();
            
            function updateDuration() {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                document.getElementById('duration').textContent = 
                    minutes.toString().padStart(2, '0') + ':' + seconds.toString().padStart(2, '0');
            }
            
            function updateStatus(connected) {
                isConnected = connected;
                const indicator = document.getElementById('statusIndicator');
                const text = document.getElementById('statusText');
                const connectionIndicator = document.getElementById('connectionIndicator');
                const connectionStatus = document.getElementById('connectionStatus');
                const connectBtn = document.getElementById('connectBtn');
                const connectText = document.getElementById('connectText');
                const recordBtn = document.getElementById('recordBtn');
                const messageInput = document.getElementById('messageInput');
                const sendBtn = document.getElementById('sendBtn');
                
                if (connected) {
                    indicator.className = 'status-indicator status-connected';
                    text.textContent = 'Connected';
                    connectionIndicator.className = 'status-indicator status-connected';
                    connectionStatus.textContent = 'Connected to AI assistant';
                    connectText.innerHTML = '<i data-lucide="square" class="h-4 w-4 mr-2"></i>End Meeting';
                    recordBtn.disabled = false;
                    messageInput.disabled = false;
                    sendBtn.disabled = false;
                } else {
                    indicator.className = 'status-indicator status-disconnected';
                    text.textContent = 'Disconnected';
                    connectionIndicator.className = 'status-indicator status-disconnected';
                    connectionStatus.textContent = 'Not connected';
                    connectText.innerHTML = '<i data-lucide="play" class="h-4 w-4 mr-2"></i>Start Meeting';
                    recordBtn.disabled = true;
                    messageInput.disabled = true;
                    sendBtn.disabled = true;
                }
                lucide.createIcons();
            }
            
            function toggleConnection() {
                if (isConnected) {
                    disconnect();
                } else {
                    connect();
                }
            }
            
            let mediaRecorder = null;
            let audioChunks = [];
            let recognition = null; // Web Speech API

            function hasWebSpeech() {
                return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
            }

            function startASR() {
                try {
                    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                    if (!SR) return false;
                    recognition = new SR();
                    recognition.continuous = true;
                    recognition.interimResults = true;
                    recognition.lang = 'en-US';
                    recognition.onresult = (event) => {
                        let finalText = '';
                        for (let i = event.resultIndex; i < event.results.length; i++) {
                            const res = event.results[i];
                            const text = res[0].transcript.trim();
                            if (!text) continue;
                            // Stream partials/finals as chat messages to backend
                            if (ws) {
                                ws.send(JSON.stringify({ type: 'chat', text }));
                            }
                            if (res.isFinal) finalText += text + ' ';
                        }
                    };
                    recognition.onerror = (e) => { console.warn('ASR error', e); };
                    recognition.onend = () => { /* restart if still recording */ if (isRecording && recognition) recognition.start(); };
                    recognition.start();
                    isRecording = true;
                    const recordBtn = document.getElementById('recordBtn');
                    const recordText = document.getElementById('recordText');
                    recordBtn.className = 'ai-button-primary bg-red-600 hover:bg-red-700';
                    recordText.innerHTML = '<i data-lucide="square" class="h-4 w-4 mr-2"></i>Stop';
                    lucide.createIcons();
                    return true;
                } catch (e) {
                    console.warn('ASR init failed', e);
                    return false;
                }
            }

            function stopASR() {
                try { if (recognition) { recognition.onend = null; recognition.stop(); } } catch(e) {}
                recognition = null;
            }

            function toggleRecording() {
                if (!isConnected) return;
                
                if (isRecording) {
                    stopASR();
                    stopRecording();
                } else {
                    // Prefer Web Speech API streaming; fallback to MediaRecorder blob
                    const started = hasWebSpeech() ? startASR() : false;
                    if (!started) startRecording();
                }
            }

            let audioCtx = null, srcNode = null, analyser = null, vadRaf = 0;
            let vadSpeaking = false, vadLastVoice = 0;

            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    // Setup WebAudio for simple VAD
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    srcNode = audioCtx.createMediaStreamSource(stream);
                    analyser = audioCtx.createAnalyser();
                    analyser.fftSize = 2048;
                    srcNode.connect(analyser);
                    const dataArr = new Uint8Array(analyser.fftSize);
                    const VAD_RMS_TH = 0.04; // tune as needed
                    const VAD_SIL_MS = 600;  // ms of silence before gating off
                    const postVoiceMs = 400; // grace period after speech
                    function loopVAD(){
                        if (!analyser) return;
                        analyser.getByteTimeDomainData(dataArr);
                        let sum=0; for(let i=0;i<dataArr.length;i++){ const v=(dataArr[i]-128)/128; sum+=v*v; }
                        const rms = Math.sqrt(sum/dataArr.length);
                        const now = performance.now();
                        if (rms > VAD_RMS_TH){ vadSpeaking = true; vadLastVoice = now; }
                        else if (now - vadLastVoice > VAD_SIL_MS){ vadSpeaking = false; }
                        vadRaf = requestAnimationFrame(loopVAD);
                    }
                    vadRaf = requestAnimationFrame(loopVAD);

                    mediaRecorder.ondataavailable = function(event) {
                        if (event.data.size > 0) {
                            const withinGrace = (performance.now() - vadLastVoice) < 400;
                            if (vadSpeaking || withinGrace) {
                                audioChunks.push(event.data);
                                if (ws && ws.readyState === 1) {
                                    try { ws.send(event.data); } catch (e) { /* ignore */ }
                                }
                            }
                        }
                    };

                    mediaRecorder.onstop = function() {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        if (ws && ws.readyState === 1) {
                            try { ws.send(audioBlob); } catch(e) { /* ignore */ }
                        }
                        stream.getTracks().forEach(track => track.stop());
                    };

                    // emit chunks every 500ms for steadier server-side ASR
                    mediaRecorder.start(500);
                    isRecording = true;
                    
                    const recordBtn = document.getElementById('recordBtn');
                    const recordText = document.getElementById('recordText');
                    recordBtn.className = 'ai-button-primary bg-red-600 hover:bg-red-700';
                    recordText.innerHTML = '<i data-lucide="square" class="h-4 w-4 mr-2"></i>Stop';
                    lucide.createIcons();
                    
                } catch (error) {
                    console.error('Error accessing microphone:', error);
                    alert('Could not access microphone. Please check permissions.');
                }
            }

            function stopRecording() {
                if (mediaRecorder && isRecording) {
                    try { mediaRecorder.stop(); } catch(_) {}
                    isRecording = false;
                }
                // VAD cleanup
                try { if (vadRaf) cancelAnimationFrame(vadRaf); } catch(_) {}
                vadRaf = 0; vadSpeaking = false; vadLastVoice = 0;
                try { if (srcNode) srcNode.disconnect(); } catch(_) {}
                try { if (analyser) analyser.disconnect(); } catch(_) {}
                srcNode = null; analyser = null;
                try { if (audioCtx) audioCtx.close(); } catch(_) {}
                audioCtx = null;
                
                const recordBtn = document.getElementById('recordBtn');
                const recordText = document.getElementById('recordText');
                recordBtn.className = 'ai-button-primary';
                recordText.innerHTML = '<i data-lucide="mic" class="h-4 w-4 mr-2"></i>Record';
                lucide.createIcons();
            }
            
            async function connect() {
                if (ws) return;
                
                // Start meeting
                try {
                    const response = await fetch('/meetings/start', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            meeting_id: meetingId,
                            title: "Q4 Strategic Planning Meeting",
                            platform: "AI Meeting Assistant",
                            start_time: Date.now() / 1000,
                            privacy_mode: "private",
                            participants: ["Alex Thompson", "Sarah Chen", "Marcus Johnson", "Emily Rodriguez", "David Kim"]
                        })
                    });
                    
                    if (response.ok) {
                        console.log('Meeting started successfully');
                    }
                } catch (error) {
                    console.error('Error starting meeting:', error);
                }
                
                ws = new WebSocket(`ws://localhost:8002/ws/audio/${meetingId}`);
                
                ws.onopen = function() {
                    updateStatus(true);
                    console.log('Connected to WebSocket');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'transcript') {
                        addTranscriptEntry(data.utterance);
                    } else if (data.type === 'suggestion') {
                        addSuggestion(data.suggestion);
                    } else if (data.type === 'file_uploaded') {
                        addTranscriptEntry(data.transcript_entry);
                        addFileToList(data.file);
                    }
                };
                
                ws.onclose = function() {
                    updateStatus(false);
                    ws = null;
                    console.log('Disconnected from WebSocket');
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    updateStatus(false);
                    alert('WebSocket connection failed. Please try clicking "Start Meeting" again.');
                };
            }
            
            async function disconnect() {
                if (ws) {
                    ws.close();
                }
                
                // End meeting
                try {
                    const response = await fetch(`/meetings/${meetingId}/end`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        console.log('Meeting ended successfully');
                    }
                } catch (error) {
                    console.error('Error ending meeting:', error);
                }
            }
            
            function addTranscriptEntry(utterance) {
                const transcript = document.getElementById('transcript');
                const entry = document.createElement('div');
                entry.className = 'transcript-entry bg-gray-800/50 p-4 rounded-lg border-l-4 border-green-500';
                entry.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <span class="font-semibold text-green-400">${utterance.speaker}</span>
                        <span class="text-xs text-gray-500">${new Date(utterance.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <p class="text-gray-200">${utterance.text}</p>
                `;
                transcript.appendChild(entry);
                transcript.scrollTop = transcript.scrollHeight;
            }
            
            function addSuggestion(suggestion) {
                const suggestions = document.getElementById('suggestions');
                const suggestionDiv = document.createElement('div');
                suggestionDiv.className = 'suggestion-card';
                suggestionDiv.id = suggestion.id;
                
                const iconMap = {
                    'ask': 'message-square',
                    'email': 'mail',
                    'task': 'check-square',
                    'fact': 'file-text',
                    'summary': 'file-text'
                };
                
                const badgeClass = {
                    'ask': 'badge-ask',
                    'email': 'badge-email',
                    'task': 'badge-task',
                    'fact': 'badge-fact',
                    'summary': 'badge-summary'
                };
                
                suggestionDiv.innerHTML = `
                    <div class="flex items-start gap-3 mb-3">
                        <div class="p-1.5 rounded bg-purple-500/10">
                            <i data-lucide="${iconMap[suggestion.kind] || 'message-square'}" class="h-4 w-4 text-purple-400"></i>
                        </div>
                        <div class="flex-1">
                            <div class="flex items-center gap-2 mb-2">
                                <span class="badge ${badgeClass[suggestion.kind] || 'badge-ask'}">${suggestion.kind.toUpperCase()}</span>
                                ${suggestion.source === 'document' ? '<span class="badge badge-document">FROM DOC</span>' : ''}
                            </div>
                            <p class="text-sm text-gray-200 leading-relaxed">${suggestion.text}</p>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <button class="flex-1 bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded text-sm font-medium" 
                                onclick="approveSuggestion('${suggestion.id}')">
                            <i data-lucide="check" class="h-3 w-3 mr-1 inline"></i>
                            Approve
                        </button>
                        <button class="flex-1 border border-red-500 text-red-400 hover:bg-red-500/10 px-3 py-2 rounded text-sm font-medium" 
                                onclick="rejectSuggestion('${suggestion.id}')">
                            <i data-lucide="x" class="h-3 w-3 mr-1 inline"></i>
                            Reject
                        </button>
                    </div>
                `;
                suggestions.appendChild(suggestionDiv);
                lucide.createIcons();
            }
            
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (message && ws) {
                    // Send as JSON message
                    ws.send(JSON.stringify({
                        type: "chat",
                        text: message
                    }));
                    input.value = '';
                }
            }
            
            function approveSuggestion(suggestionId) {
                fetch(`/suggestions/${suggestionId}/approve`, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        const suggestion = document.getElementById(suggestionId);
                        if (suggestion) {
                            suggestion.querySelector('.flex.gap-2').innerHTML = 
                                '<span class="flex-1 text-center text-green-400 font-medium">✓ Approved</span>';
                        }
                    });
            }
            
            function rejectSuggestion(suggestionId) {
                fetch(`/suggestions/${suggestionId}/reject`, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        const suggestion = document.getElementById(suggestionId);
                        if (suggestion) {
                            suggestion.querySelector('.flex.gap-2').innerHTML = 
                                '<span class="flex-1 text-center text-red-400 font-medium">✗ Rejected</span>';
                        }
                    });
            }
            
            // Auto-connect on page load with error handling
            setTimeout(() => {
                try {
            connect();
                } catch (error) {
                    console.error('Auto-connect failed:', error);
                    // Enable manual connection
                    document.getElementById('connectBtn').disabled = false;
                }
            }, 1000);
            
            // Update duration every second
            setInterval(updateDuration, 1000);
            
            // File upload functions
            function handleFileUpload(event) {
                const files = event.target.files;
                for (let file of files) {
                    uploadFile(file);
                }
            }

            async function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('meeting_id', meetingId);

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    if (result.status === 'success') {
                        addFileToList(result.file);
                        console.log('File uploaded successfully:', result.file.filename);
                    } else {
                        console.error('Upload failed:', result.message);
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                }
            }

            function addFileToList(file) {
                const uploadedFiles = document.getElementById('uploadedFiles');
                const fileDiv = document.createElement('div');
                fileDiv.className = 'file-item';
                fileDiv.id = `file_${file.id}`;
                
                const fileSize = (file.size / 1024).toFixed(1) + ' KB';
                const uploadTime = new Date(file.upload_time).toLocaleTimeString();
                
                fileDiv.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <i data-lucide="file" class="h-4 w-4 text-blue-400"></i>
                            <span class="font-medium text-sm">${file.filename}</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-xs text-gray-500">${fileSize}</span>
                            <button onclick="deleteFile('${file.id}')" class="text-red-400 hover:text-red-300">
                                <i data-lucide="x" class="h-3 w-3"></i>
                            </button>
                        </div>
                    </div>
                    <div class="text-xs text-gray-400 mb-2">Uploaded at ${uploadTime}</div>
                    <div class="file-actions">
                        <button class="file-action-btn" onclick="performFileAction('${file.id}', 'summarize')">
                            <i data-lucide="file-text" class="h-3 w-3 mr-1"></i>
                            Summarize
                        </button>
                        <button class="file-action-btn" onclick="performFileAction('${file.id}', 'extract_actions')">
                            <i data-lucide="list" class="h-3 w-3 mr-1"></i>
                            Extract Actions
                        </button>
                        <button class="file-action-btn" onclick="askAIAboutFile('${file.id}')">
                            <i data-lucide="message-circle" class="h-3 w-3 mr-1"></i>
                            Ask AI
                        </button>
                    </div>
                `;
                
                uploadedFiles.appendChild(fileDiv);
                lucide.createIcons();
            }

            async function performFileAction(fileId, action) {
                try {
                    const response = await fetch(`/files/${fileId}/action`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            file_id: fileId,
                            action: action
                        })
                    });
                    
                    const result = await response.json();
                    if (result.action === 'summarize') {
                        showFileResult('Summary', result.summary, result.key_points);
                    } else if (result.action === 'extract_actions') {
                        showFileResult('Action Items', result.action_items.join('\\n'), null);
                    }
                } catch (error) {
                    console.error('File action error:', error);
                }
            }

            function askAIAboutFile(fileId) {
                const query = prompt('What would you like to know about this file?');
                if (query) {
                    performFileAction(fileId, 'ask_ai', query);
                }
            }

            function showFileResult(title, content, additional) {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'ai-card mt-4';
                resultDiv.innerHTML = `
                    <h4 class="text-lg font-semibold mb-3">${title}</h4>
                    <div class="text-sm text-gray-200 whitespace-pre-line">${content}</div>
                    ${additional ? `<div class="mt-3"><strong>Key Points:</strong><ul class="list-disc list-inside mt-2">${additional.map(point => `<li>${point}</li>`).join('')}</ul></div>` : ''}
                `;
                
                document.getElementById('uploadedFiles').appendChild(resultDiv);
            }

            async function deleteFile(fileId) {
                try {
                    const response = await fetch(`/files/${fileId}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        const fileElement = document.getElementById(`file_${fileId}`);
                        if (fileElement) {
                            fileElement.remove();
                        }
                    }
                } catch (error) {
                    console.error('Delete file error:', error);
                }
            }

            function performSearch() {
                const query = document.getElementById('searchInput').value.trim();
                if (!query) return;
                const results = document.getElementById('searchResultsKeyword');
                results.innerHTML = '<div class="text-sm text-gray-400">Searching…</div>';

                fetch(`/search?query=${encodeURIComponent(query)}&meeting_id=${encodeURIComponent(meetingId)}&k=20`)
                    .then(r => r.json())
                    .then(data => {
                        // Render synthesized answers with citations if present
                        let answerHtml = '';
                        if (Array.isArray(data.answers)) {
                            const items = data.answers.map(a => {
                                const sources = (a.sources||[]).map(s => {
                                    if (s.type === 'transcript') return `<span class="text-xs text-gray-400">[${s.label} @ ${s.timestamp||''}]</span>`;
                                    if (s.type === 'document') return `<span class="text-xs text-gray-400">[${s.label} ${s.file_id||''}]</span>`;
                                    return '';
                                }).join(' ');
                                return `<div class="p-3 rounded bg-gray-800/60 border border-gray-700">
                                    <div class="text-sm text-gray-200">${escapeHtml(a.text||'')}</div>
                                    <div class="mt-1">${sources}</div>
                                </div>`;
                            }).join('');
                            answerHtml = `<div class="mb-2"><div class="text-sm text-purple-300 mb-1">Answer</div>${items}</div>`;
                        } else if (data.answer) {
                            answerHtml = `<div class="mb-2"><div class="text-sm text-purple-300 mb-1">Answer</div><div class="text-sm text-gray-200">${escapeHtml(data.answer)}</div></div>`;
                        }
                        if (!data.hits || data.hits.length === 0) {
                            results.innerHTML = answerHtml || '<div class="text-sm text-gray-400">No results.</div>';
                            return;
                        }
                        // Simple highlighter
                        const hi = (t)=> (t||'').replace(new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'ig'), m=>`<mark class="bg-yellow-600/40 px-1 rounded">${m}</mark>`);
                        const items = data.hits.map(h => {
                            if (h.source === 'transcript') {
                                return `<div class="p-3 rounded bg-gray-800/60 border border-gray-700">
                                    <div class="text-xs text-gray-500 mb-1">Transcript • ${new Date(h.timestamp).toLocaleTimeString()}</div>
                                    <div class="text-sm text-gray-200"><strong>${escapeHtml(h.speaker)}:</strong> ${hi(escapeHtml(h.text))}</div>
                                </div>`;
                            } else {
                                return `<div class="p-3 rounded bg-gray-800/60 border border-gray-700">
                                    <div class="text-xs text-gray-500 mb-1">Document • ${escapeHtml(h.filename || '')}</div>
                                    <div class="text-sm text-gray-200">${hi(escapeHtml(h.snippet || ''))}</div>
                                </div>`;
                            }
                        }).join('');
                        // Filters UI
                        const filters = `<div class="flex gap-2 mb-2 text-xs"><button id="filterAll" class="px-2 py-1 rounded bg-gray-700">All</button><button id="filterTranscript" class="px-2 py-1 rounded bg-gray-700">Transcript</button><button id="filterDocs" class="px-2 py-1 rounded bg-gray-700">Docs</button></div>`;
                        results.innerHTML = `${answerHtml}<div class="text-sm text-gray-400 mb-2">${data.count} results</div>${filters}<div id="searchList">${items}</div>`;
                        // Filter behavior
                        const listEl = document.getElementById('searchList');
                        document.getElementById('filterAll').onclick = ()=>{ Array.from(listEl.children).forEach(c=>c.style.display='block'); };
                        document.getElementById('filterTranscript').onclick = ()=>{ Array.from(listEl.children).forEach(c=>{ c.querySelector('.text-xs')?.textContent.includes('Transcript') ? c.style.display='block' : c.style.display='none'; }); };
                        document.getElementById('filterDocs').onclick = ()=>{ Array.from(listEl.children).forEach(c=>{ c.querySelector('.text-xs')?.textContent.includes('Document') ? c.style.display='block' : c.style.display='none'; }); };
                    })
                    .catch(e => {
                        console.error('Search error', e);
                        results.innerHTML = '<div class="text-sm text-red-400">Search failed.</div>';
                    });
            }

            let semPage = 1;
            function semanticPrev(){ if (semPage>1){ semPage--; performSemanticSearch(); } }
            function semanticNext(){ semPage++; performSemanticSearch(); }

            function performSemanticSearch() {
                const query = document.getElementById('searchInput').value.trim();
                if (!query) return;
                const results = document.getElementById('searchResultsSemantic');
                const totalEl = document.getElementById('semanticTotal');
                results.innerHTML = '<div class="text-sm text-gray-400">Semantic searching…</div>';
                fetch(`/semantic_search?query=${encodeURIComponent(query)}&page=${semPage}&per_page=10`)
                    .then(r => r.json())
                    .then(data => {
                        if (!data.hits || data.hits.length === 0) {
                            results.innerHTML = '<div class="text-sm text-gray-400">No semantic matches.</div>';
                            return;
                        }
                        const items = data.hits.map(h => {
                            const page = (typeof h.page_idx === 'number') ? ` • p. ${h.page_idx+1}` : '';
                            const meta = h.kind === 'transcript' ? `Transcript • ${h.timestamp||''}` : `Document • ${h.file_id||''}${page}`;
                            return `<div class=\"p-3 rounded bg-gray-800/60 border border-gray-700\">\n`
                                + `  <div class=\"text-xs text-gray-500 mb-1\">${meta} • score ${Number(h.score||0).toFixed(3)}</div>\n`
                                + `  <div class=\"text-sm text-gray-200\">${escapeHtml(h.text||'')}</div>\n`
                                + `</div>`;
                        }).join('');
                        const total = data.total || data.count || 0;
                        document.getElementById('semanticPageLabel').textContent = `Page ${data.page||semPage}`;
                        totalEl.textContent = `${total} results`;
                        results.innerHTML = items;
                    })
                    .catch(e => {
                        console.error('Semantic search error', e);
                        results.innerHTML = '<div class="text-sm text-red-400">Semantic search failed.</div>';
                    });
            }

            // Basic HTML escaper
            function escapeHtml(s){ return (s||'').replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]||c)); }

            // Drag and drop functionality
            const uploadArea = document.getElementById('fileUploadArea');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                for (let file of files) {
                    uploadFile(file);
                }
            });

            // Handle Enter key in input
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            // Handle Enter key in search input
            document.getElementById('searchInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    performSearch();
                }
            });
            
            // Handle Enter key in chat input
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            // Suggestion tab switching
            function switchSuggestionTab(category) {
                // Update tab buttons
                document.querySelectorAll('.suggestion-tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.getElementById(`tab-${category}`).classList.add('active');

                // Filter suggestions
                const suggestions = document.querySelectorAll('.suggestion-card');
                suggestions.forEach(suggestion => {
                    if (category === 'all' || suggestion.dataset.category === category) {
                        suggestion.style.display = 'block';
                    } else {
                        suggestion.style.display = 'none';
                    }
                });
            }

            // Settings modal
            function toggleSettings() {
                // Simple settings toggle - could be expanded to a full modal
                alert('Settings panel will be implemented in the next iteration!\n\nPlanned features:\n• Privacy controls\n• Language settings\n• Notification preferences\n• Theme customization');
            }

            // Export transcript
            function exportTranscript() {
                const transcript = document.getElementById('transcript');
                const text = Array.from(transcript.children).map(entry => {
                    const speaker = entry.querySelector('.font-semibold').textContent;
                    const content = entry.querySelector('p').textContent;
                    const time = entry.querySelector('.text-xs').textContent;
                    return `[${time}] ${speaker}: ${content}`;
                }).join('\n\n');

                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `meeting-transcript-${new Date().toISOString().split('T')[0]}.txt`;
                a.click();
                URL.revokeObjectURL(url);
            }

            // Generate meeting summary (calls backend)
            function generateSummary() {
                const summarySection = document.getElementById('summarySection');
                const summaryContent = document.getElementById('meetingSummary');
                summaryContent.innerHTML = '<div class="text-sm text-gray-400">Generating summary…</div>';
                fetch(`/meetings/${encodeURIComponent(meetingId)}/summarize`, { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        const kp = (data.key_points || []).map(p => `<li>• ${p}</li>`).join('');
                        const ai = (data.action_items || []).map(p => `<li>• ${p}</li>`).join('');
                        summaryContent.innerHTML = `
                            <div class="space-y-4">
                                <div class="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                                    <h4 class="font-semibold text-blue-400 mb-2">Summary</h4>
                                    <p class="text-sm text-gray-300">${data.summary || 'No summary'}</p>
                                </div>
                                ${kp ? `<div class=\"bg-green-500/10 border border-green-500/20 rounded-lg p-4\">
                                    <h4 class=\"font-semibold text-green-400 mb-2\">Key Points</h4>
                                    <ul class=\"text-sm text-gray-300 space-y-1\">${kp}</ul>
                                </div>` : ''}
                                ${ai ? `<div class=\"bg-purple-500/10 border border-purple-500/20 rounded-lg p-4\">
                                    <h4 class=\"font-semibold text-purple-400 mb-2\">Action Items</h4>
                                    <ul class=\"text-sm text-gray-300 space-y-1\">${ai}</ul>
                                </div>` : ''}
                            </div>`;
                        summarySection.style.display = 'block';
                        summarySection.scrollIntoView({ behavior: 'smooth' });
                    })
                    .catch(e => {
                        console.error('Summarize error', e);
                        summaryContent.innerHTML = '<div class="text-sm text-red-400">Failed to generate summary.</div>';
                        summarySection.style.display = 'block';
                    });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("Starting AI Meeting Assistant with Co-pilot Nexus UI (API-only root)...")
    port = int(os.getenv("PORT", "8002"))
    print(f"Open your browser to: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)