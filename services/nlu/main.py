"""
NLU Service - Intent detection, entity extraction, and sentiment analysis
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel
import spacy
from transformers import pipeline
import openai
from .config import settings
import time
import uuid
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NLU Service", version="1.0.0")

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
    "nlu_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "nlu_http_request_duration_seconds",
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

# Global variables
redis_client = None
db_pool = None
nlp = None
sentiment_analyzer = None
openai_client = None

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

class Utterance(BaseModel):
    meeting_id: str
    speaker: str
    text: str
    timestamp: float
    confidence: float

# Intent patterns
INTENT_PATTERNS = {
    "decision": [
        r"let's decide",
        r"we should",
        r"i think we",
        r"agreed",
        r"consensus",
        r"final decision"
    ],
    "question": [
        r"\?",
        r"what do you think",
        r"how about",
        r"can you",
        r"could you",
        r"would you"
    ],
    "action_item": [
        r"action item",
        r"todo",
        r"task",
        r"follow up",
        r"next steps",
        r"deadline"
    ],
    "meeting_control": [
        r"let's start",
        r"wrap up",
        r"end meeting",
        r"next meeting",
        r"schedule"
    ]
}

# Entity patterns
ENTITY_PATTERNS = {
    "person": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "date": r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b",
    "time": r"\b(?:1[0-2]|[1-9])(?::[0-5][0-9])?\s*(?:am|pm)\b",
    "number": r"\b\d+\b",
    "currency": r"\$\d+(?:\.\d{2})?\b",
    "url": r"https?://[^\s]+"
}

@app.on_event("startup")
async def startup():
    """Initialize models and connections"""
    global redis_client, db_pool, nlp, sentiment_analyzer, openai_client
    
    # Redis connection
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    
    # Database connection pool
    db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=5,
        max_size=20
    )
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model not found, installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # Load sentiment analyzer
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    # Initialize OpenAI client
    if not settings.openai_api_key and settings.require_openai:
        raise RuntimeError("OPENAI_API_KEY is required but not set")
    openai_client = openai.OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    
    logger.info("NLU service started")

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

async def process_nlu_stream():
    """Process NLU stream from Redis"""
    while True:
        try:
            # Subscribe to NLU processing channel
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("nlu_process")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await process_utterance(Utterance(**data))
                    
        except Exception as e:
            logger.error(f"Error processing NLU stream: {e}")
            await asyncio.sleep(1)

async def process_utterance(utterance: Utterance):
    """Process a single utterance for NLU"""
    try:
        # Detect intent
        intent = detect_intent(utterance.text)
        
        # Extract entities
        entities = extract_entities(utterance.text)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(utterance.text)
        
        # Extract topics
        topics = extract_topics(utterance.text)
        
        # Check if it's a decision or question
        is_decision = intent == "decision"
        is_question = intent == "question" or "?" in utterance.text
        
        # Create NLU result
        nlu_result = NLUResult(
            meeting_id=utterance.meeting_id,
            speaker=utterance.speaker,
            text=utterance.text,
            timestamp=utterance.timestamp,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            confidence=utterance.confidence,
            topics=topics,
            is_decision=is_decision,
            is_question=is_question
        )
        
        # Store NLU result
        await store_nlu_result(nlu_result)
        
        # Send to agent service
        await send_to_agent(nlu_result)
        
        logger.debug(f"Processed NLU for meeting {utterance.meeting_id}")
        
    except Exception as e:
        logger.error(f"Error processing utterance: {e}")

def detect_intent(text: str) -> str:
    """Detect intent from text"""
    text_lower = text.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    
    return "other"

def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract entities from text"""
    entities = []
    
    # Use spaCy for NER
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    # Use regex patterns for additional entities
    for entity_type, pattern in ENTITY_PATTERNS.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                "text": match.group(),
                "label": entity_type,
                "start": match.start(),
                "end": match.end()
            })
    
    return entities

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text"""
    try:
        result = sentiment_analyzer(text)[0]
        return result["label"].lower()
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return "neutral"

def extract_topics(text: str) -> List[str]:
    """Extract topics from text using keywords"""
    # Simple keyword-based topic extraction
    topic_keywords = {
        "budget": ["budget", "cost", "price", "money", "financial"],
        "timeline": ["deadline", "schedule", "timeline", "due date", "when"],
        "project": ["project", "task", "work", "development", "build"],
        "meeting": ["meeting", "call", "discussion", "review"],
        "client": ["client", "customer", "user", "stakeholder"],
        "technical": ["technical", "code", "system", "architecture", "bug"]
    }
    
    topics = []
    text_lower = text.lower()
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics

async def store_nlu_result(nlu_result: NLUResult):
    """Store NLU result in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO nlu_results (
                    meeting_id, speaker, text, timestamp, intent, 
                    entities, sentiment, confidence, topics, 
                    is_decision, is_question, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
            """, 
                nlu_result.meeting_id, nlu_result.speaker, nlu_result.text,
                nlu_result.timestamp, nlu_result.intent, 
                json.dumps(nlu_result.entities), nlu_result.sentiment,
                nlu_result.confidence, json.dumps(nlu_result.topics),
                nlu_result.is_decision, nlu_result.is_question
            )
            
    except Exception as e:
        logger.error(f"Error storing NLU result: {e}")

async def send_to_agent(nlu_result: NLUResult):
    """Send NLU result to agent service"""
    try:
        agent_data = nlu_result.dict()
        await redis_client.publish("agent_process", json.dumps(agent_data))
        
    except Exception as e:
        logger.error(f"Error sending to agent: {e}")

@app.post("/nlu/process")
async def process_text(utterance: Utterance, background_tasks: BackgroundTasks):
    """Process text directly"""
    background_tasks.add_task(process_utterance, utterance)
    return {"status": "processing"}

@app.get("/nlu/meetings/{meeting_id}/results")
async def get_nlu_results(meeting_id: str):
    """Get NLU results for a meeting"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT speaker, text, timestamp, intent, entities, 
                       sentiment, confidence, topics, is_decision, is_question
                FROM nlu_results
                WHERE meeting_id = $1
                ORDER BY timestamp
            """, meeting_id)
            
            results = []
            for row in rows:
                results.append({
                    "speaker": row["speaker"],
                    "text": row["text"],
                    "timestamp": row["timestamp"],
                    "intent": row["intent"],
                    "entities": json.loads(row["entities"]),
                    "sentiment": row["sentiment"],
                    "confidence": row["confidence"],
                    "topics": json.loads(row["topics"]),
                    "is_decision": row["is_decision"],
                    "is_question": row["is_question"]
                })
            
            return {"meeting_id": meeting_id, "results": results}
            
    except Exception as e:
        logger.error(f"Error getting NLU results: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "nlu"}

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_nlu_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
