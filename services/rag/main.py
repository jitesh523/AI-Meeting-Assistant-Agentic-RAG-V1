"""
RAG Service - Retrieval-Augmented Generation for context-aware responses
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from .config import settings
from sklearn.metrics.pairwise import cosine_similarity
import time
import uuid
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import contextvars

# Configure logging
logging.basicConfig(level=logging.INFO)

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
            "service.name": "rag",
        })
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        ASGIInstrumentor().instrument()
        AsyncPGInstrumentor().instrument()
        RedisInstrumentor().instrument()
        logger.info("OpenTelemetry tracing enabled for rag")
    except Exception as _otel_err:
        logger.warning(f"Failed to initialize OpenTelemetry: {_otel_err}")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Service", version="1.0.0")

# CORS middleware
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
app.state.idem_store = {}

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
ERROR_COUNT = Counter(
    "rag_http_errors_total",
    "Total HTTP errors",
    ["type"],
)
HEALTH_GAUGE = Gauge(
    "rag_dependency_up",
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
async def idempotency_guard(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        key = request.headers.get("Idempotency-Key")
        if key:
            now = time.perf_counter()
            ttl = float(os.getenv("IDEMPOTENCY_TTL_SECONDS", "600"))
            store = app.state.idem_store
            expired = [k for k, v in store.items() if (now - v) > ttl]
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
    logger = logging.getLogger(__name__)
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
embedding_model = None
openai_client = None

class Document(BaseModel):
    doc_id: str
    tenant_id: str
    source: str
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]

class QueryResult(BaseModel):
    query: str
    documents: List[Document]
    context: str
    confidence: float
    # Optional: per-document scores for transparency
    similarity_scores: Optional[List[Dict[str, Any]]] = None

class RAGQuery(BaseModel):
    query: str
    meeting_id: str
    user_id: str
    tenant_id: str
    max_docs: int = 5

@app.on_event("startup")
async def startup():
    """Initialize models and connections with retries"""
    global redis_client, db_pool, embedding_model, openai_client
    max_attempts = 5
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            break
        except Exception:
            if attempt == max_attempts:
                if os.getenv("ALLOW_DEGRADED_STARTUP") == "1":
                    logger.warning("Redis unavailable after retries; starting RAG in degraded mode")
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
                    logger.warning("Postgres unavailable after retries; starting RAG in degraded mode")
                    db_pool = None
                    break
                raise
            await asyncio.sleep(delay)
            delay *= 2
    
    # Load embedding model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded embedding model")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        embedding_model = None
    
    # Initialize OpenAI client
    if not settings.openai_api_key and settings.require_openai:
        raise RuntimeError("OPENAI_API_KEY is required but not set")
    openai_client = openai.OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    
    logger.info("RAG service started")

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

async def process_rag_stream():
    """Process RAG queries from Redis"""
    while True:
        try:
            # Subscribe to RAG processing channel
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("rag_query")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await process_rag_query(RAGQuery(**data))
                    
        except Exception as e:
            logger.error(f"Error processing RAG stream: {e}")
            await asyncio.sleep(1)

async def process_rag_query(rag_query: RAGQuery):
    """Process a RAG query"""
    try:
        # Retrieve relevant documents and scores
        documents, sim_scores = await retrieve_documents(rag_query)
        
        # Generate context
        context = generate_context(documents)
        
        # Create query result
        result = QueryResult(
            query=rag_query.query,
            documents=documents,
            context=context,
            confidence=calculate_confidence(documents),
            similarity_scores=sim_scores,
        )
        
        # Store result
        await store_query_result(result, rag_query.meeting_id)
        
        # Send to agent service
        await send_to_agent(result, rag_query.meeting_id)
        
        logger.debug(f"Processed RAG query for meeting {rag_query.meeting_id}")
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")

async def retrieve_documents(rag_query: RAGQuery) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """Retrieve relevant documents using hybrid similarity (vector + lexical)."""
    try:
        # Generate query embedding
        query_embedding = generate_embedding(rag_query.query)

        async with db_pool.acquire() as conn:
            # Vector results (cosine distance: smaller is better). Convert to similarity = 1 - distance.
            vrows = await conn.fetch(
                """
                SELECT doc_id, tenant_id, source, text, metadata, embedding,
                       (embedding <-> $2) AS dist
                FROM documents
                WHERE tenant_id = $1
                ORDER BY dist
                LIMIT $3
                """,
                rag_query.tenant_id,
                query_embedding,
                rag_query.max_docs,
            )

            # Lexical results using pg_trgm; use the % operator to leverage GIN index
            lrows = await conn.fetch(
                """
                SELECT doc_id, tenant_id, source, text, metadata,
                       similarity(text, $2) AS lex
                FROM documents
                WHERE tenant_id = $1 AND text % $2
                ORDER BY lex DESC
                LIMIT $3
                """,
                rag_query.tenant_id,
                rag_query.query,
                rag_query.max_docs,
            )

        # Normalize and merge by doc_id
        vector_scores: Dict[str, float] = {}
        for r in vrows:
            dist = float(r["dist"]) if r["dist"] is not None else 1.0
            sim = max(0.0, 1.0 - dist)
            vector_scores[str(r["doc_id"])] = sim

        lexical_scores: Dict[str, float] = {}
        for r in lrows:
            lex = float(r["lex"]) if r["lex"] is not None else 0.0
            lexical_scores[str(r["doc_id"])] = max(0.0, min(1.0, lex))

        # Combine with weights
        alpha = 0.7  # weight for vector
        all_ids = set(vector_scores.keys()) | set(lexical_scores.keys())
        combined: List[Tuple[str, float]] = []
        for did in all_ids:
            vs = vector_scores.get(did, 0.0)
            ls = lexical_scores.get(did, 0.0)
            combined.append((did, alpha * vs + (1 - alpha) * ls))
        # Sort by combined score desc and take top max_docs
        combined.sort(key=lambda x: x[1], reverse=True)
        top_ids = [did for did, _ in combined[: rag_query.max_docs]]

        # Build documents list preserving order
        row_by_id: Dict[str, Any] = {str(r["doc_id"]): r for r in list(vrows) + list(lrows)}
        documents: List[Document] = []
        sim_scores: List[Dict[str, Any]] = []
        for did in top_ids:
            r = row_by_id.get(did)
            if not r:
                continue
            documents.append(
                Document(
                    doc_id=str(r["doc_id"]),
                    tenant_id=str(r["tenant_id"]),
                    source=r["source"],
                    text=r["text"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                    embedding=r.get("embedding", []),
                )
            )
            sim_scores.append(
                {
                    "doc_id": did,
                    "vector": vector_scores.get(did, 0.0),
                    "lexical": lexical_scores.get(did, 0.0),
                    "hybrid": next((score for _id, score in combined if _id == did), 0.0),
                }
            )

        return documents, sim_scores
            
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    if embedding_model:
        return embedding_model.encode(text).tolist()
    else:
        # Fallback to OpenAI embeddings
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 384  # Default dimension

def generate_context(documents: List[Document]) -> str:
    """Generate context from retrieved documents"""
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"Source {i} ({doc.source}): {doc.text[:500]}...")
    
    return "\n\n".join(context_parts)

def calculate_confidence(documents: List[Document]) -> float:
    """Calculate confidence score for retrieved documents"""
    if not documents:
        return 0.0
    
    # Simple confidence based on number of documents and their relevance
    base_confidence = min(len(documents) / 5.0, 1.0)
    
    # TODO: Add more sophisticated confidence calculation
    # based on similarity scores, document quality, etc.
    
    return base_confidence

async def store_query_result(result: QueryResult, meeting_id: str):
    """Store query result in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO rag_results (
                    meeting_id, query, context, confidence,
                    document_ids, similarity_scores, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, now())
                """,
                meeting_id,
                result.query,
                result.context,
                result.confidence,
                json.dumps([doc.doc_id for doc in result.documents]),
                json.dumps(result.similarity_scores or []),
            )
            
    except Exception as e:
        logger.error(f"Error storing query result: {e}")

async def send_to_agent(result: QueryResult, meeting_id: str):
    """Send RAG result to agent service"""
    try:
        agent_data = {
            "meeting_id": meeting_id,
            "query": result.query,
            "context": result.context,
            "confidence": result.confidence,
            "documents": [doc.dict() for doc in result.documents]
        }
        await redis_client.publish("agent_rag_result", json.dumps(agent_data))
        
    except Exception as e:
        logger.error(f"Error sending to agent: {e}")

@app.post("/rag/query")
@limiter.limit("60/minute")
async def query_rag(rag_query: RAGQuery, background_tasks: BackgroundTasks):
    """Query RAG system directly"""
    background_tasks.add_task(process_rag_query, rag_query)
    return {"status": "processing"}

@app.get("/rag/search")
@limiter.limit("120/minute")
async def search_rag(q: str, tenant_id: str, k: int = 5):
    """Simple search endpoint returning top-k documents for a query"""
    try:
        query_embedding = generate_embedding(q)
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT doc_id, tenant_id, source, text, metadata, embedding
                FROM documents
                WHERE tenant_id = $1
                ORDER BY embedding <-> $2
                LIMIT $3
                """,
                tenant_id, query_embedding, k,
            )

        hits = []
        for row in rows:
            hits.append({
                "doc_id": str(row["doc_id"]),
                "tenant_id": str(row["tenant_id"]),
                "source": row["source"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            })
        return {"query": q, "tenant_id": tenant_id, "k": k, "hits": hits}
    except Exception as e:
        logger.error(f"Error in /rag/search: {e}")
        return {"error": str(e)}

@app.post("/rag/upload")
@limiter.limit("10/minute")
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    source: str = Form("upload")
):
    """Upload a small text file and index it. For PDFs, add parsing later."""
    try:
        content_bytes = await file.read()
        # Basic size guard (e.g., 1 MB)
        if len(content_bytes) > 1_000_000:
            return {"status": "error", "message": "File too large (max 1MB)"}

        # Assume UTF-8 text for MVP
        text = content_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            return {"status": "error", "message": "Empty or invalid text content"}

        # Generate embedding and store
        embedding = generate_embedding(text)
        doc_id = str(asyncpg.uuid.uuid4()) if hasattr(asyncpg, "uuid") else None

        async with db_pool.acquire() as conn:
            # Generate doc_id in SQL if not generated in Python
            if not doc_id:
                row = await conn.fetchrow("SELECT gen_random_uuid() AS id")
                doc_id = str(row["id"])

            await conn.execute(
                """
                INSERT INTO documents (doc_id, tenant_id, source, text, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (doc_id) DO UPDATE SET text = EXCLUDED.text,
                                                metadata = EXCLUDED.metadata,
                                                embedding = EXCLUDED.embedding
                """,
                doc_id, tenant_id, source, text, json.dumps({"filename": file.filename}), embedding,
            )

        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Error in /rag/upload: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/rag/documents")
@limiter.limit("30/minute")
async def add_document(document: Document):
    """Add a document to the vector database"""
    try:
        # Generate embedding
        embedding = generate_embedding(document.text)
        
        # Store in database
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO documents (
                    doc_id, tenant_id, source, text, metadata, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (doc_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
            """, 
                document.doc_id, document.tenant_id, document.source,
                document.text, json.dumps(document.metadata), embedding
            )
        
        return {"status": "success", "doc_id": document.doc_id}
        
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/rag/meetings/{meeting_id}/context")
async def get_meeting_context(meeting_id: str):
    """Get context for a meeting"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT query, context, confidence, created_at
                FROM rag_results
                WHERE meeting_id = $1
                ORDER BY created_at DESC
                LIMIT 10
            """, meeting_id)
            
            contexts = []
            for row in rows:
                contexts.append({
                    "query": row["query"],
                    "context": row["context"],
                    "confidence": row["confidence"],
                    "timestamp": row["created_at"].isoformat()
                })
            
            return {"meeting_id": meeting_id, "contexts": contexts}
            
    except Exception as e:
        logger.error(f"Error getting meeting context: {e}")
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
    return {"status": "healthy" if overall else "degraded", "service": "rag", "dependencies": {"redis": bool(ok_redis), "postgres": bool(ok_db)}}

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_rag_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
