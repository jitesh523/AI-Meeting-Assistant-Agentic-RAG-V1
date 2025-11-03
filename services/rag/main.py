"""
RAG Service - Retrieval-Augmented Generation for context-aware responses
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from .config import settings
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
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

class RAGQuery(BaseModel):
    query: str
    meeting_id: str
    user_id: str
    tenant_id: str
    max_docs: int = 5

@app.on_event("startup")
async def startup():
    """Initialize models and connections"""
    global redis_client, db_pool, embedding_model, openai_client
    
    # Redis connection
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    
    # Database connection pool
    db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=5,
        max_size=20
    )
    
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
        # Retrieve relevant documents
        documents = await retrieve_documents(rag_query)
        
        # Generate context
        context = generate_context(documents)
        
        # Create query result
        result = QueryResult(
            query=rag_query.query,
            documents=documents,
            context=context,
            confidence=calculate_confidence(documents)
        )
        
        # Store result
        await store_query_result(result, rag_query.meeting_id)
        
        # Send to agent service
        await send_to_agent(result, rag_query.meeting_id)
        
        logger.debug(f"Processed RAG query for meeting {rag_query.meeting_id}")
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")

async def retrieve_documents(rag_query: RAGQuery) -> List[Document]:
    """Retrieve relevant documents using vector similarity"""
    try:
        # Generate query embedding
        query_embedding = generate_embedding(rag_query.query)
        
        # Search vector database
        async with db_pool.acquire() as conn:
            # Use pgvector for similarity search
            rows = await conn.fetch("""
                SELECT doc_id, tenant_id, source, text, metadata, embedding
                FROM documents
                WHERE tenant_id = $1
                ORDER BY embedding <-> $2
                LIMIT $3
            """, rag_query.tenant_id, query_embedding, rag_query.max_docs)
            
            documents = []
            for row in rows:
                doc = Document(
                    doc_id=row["doc_id"],
                    tenant_id=row["tenant_id"],
                    source=row["source"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    embedding=row["embedding"]
                )
                documents.append(doc)
            
            return documents
            
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
            await conn.execute("""
                INSERT INTO rag_results (
                    meeting_id, query, context, confidence, 
                    document_ids, created_at
                ) VALUES ($1, $2, $3, $4, $5, now())
            """, 
                meeting_id, result.query, result.context, 
                result.confidence, json.dumps([doc.doc_id for doc in result.documents])
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
async def query_rag(rag_query: RAGQuery, background_tasks: BackgroundTasks):
    """Query RAG system directly"""
    background_tasks.add_task(process_rag_query, rag_query)
    return {"status": "processing"}

@app.get("/rag/search")
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
    """Health check endpoint"""
    return {"status": "healthy", "service": "rag"}

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_rag_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
