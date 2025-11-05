"""
Integrations Service - Connectors for external services (Gmail, Notion, Slack, etc.)
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel
import httpx
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import slack_sdk
from notion_client import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Integrations Service", version="1.0.0")

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

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "integrations_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "integrations_http_request_duration_seconds",
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

class IntegrationConfig(BaseModel):
    service: str
    user_id: str
    tenant_id: str
    credentials: Dict[str, Any]
    settings: Dict[str, Any]

class EmailDraft(BaseModel):
    to: List[str]
    subject: str
    body: str
    priority: str = "normal"
    attachments: List[str] = []

class Task(BaseModel):
    title: str
    description: str
    assignee: Optional[str] = None
    priority: str = "medium"
    due_date: Optional[str] = None
    labels: List[str] = []

class CalendarEvent(BaseModel):
    title: str
    start_time: str
    end_time: str
    attendees: List[str] = []
    description: str = ""
    location: str = ""

class SlackMessage(BaseModel):
    channel: str
    text: str
    blocks: Optional[List[Dict[str, Any]]] = None

@app.on_event("startup")
async def startup():
    """Initialize connections"""
    global redis_client, db_pool
    
    # Redis connection
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    
    # Database connection pool
    db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=5,
        max_size=20
    )
    
    logger.info("Integrations service started")

@app.on_event("shutdown")
async def shutdown():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

async def process_integration_stream():
    """Process integration tasks from Redis"""
    while True:
        try:
            # Subscribe to integration processing channel
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("integration_task")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await process_integration_task(data)
                    
        except Exception as e:
            logger.error(f"Error processing integration stream: {e}")
            await asyncio.sleep(1)

async def process_integration_task(task_data: Dict[str, Any]):
    """Process an integration task"""
    try:
        service = task_data.get("service")
        action = task_data.get("action")
        data = task_data.get("data", {})
        
        if service == "gmail":
            await handle_gmail_action(action, data)
        elif service == "slack":
            await handle_slack_action(action, data)
        elif service == "notion":
            await handle_notion_action(action, data)
        elif service == "calendar":
            await handle_calendar_action(action, data)
        else:
            logger.warning(f"Unknown service: {service}")
        
        logger.debug(f"Processed integration task: {service}.{action}")
        
    except Exception as e:
        logger.error(f"Error processing integration task: {e}")

# Gmail Integration
async def handle_gmail_action(action: str, data: Dict[str, Any]):
    """Handle Gmail actions"""
    try:
        if action == "draft_email":
            await draft_gmail_email(data)
        elif action == "search_emails":
            await search_gmail_emails(data)
        else:
            logger.warning(f"Unknown Gmail action: {action}")
    except Exception as e:
        logger.error(f"Error handling Gmail action: {e}")

async def draft_gmail_email(data: Dict[str, Any]):
    """Draft an email in Gmail"""
    try:
        # Get user credentials
        credentials = await get_user_credentials(data["user_id"], "gmail")
        
        # Build Gmail service
        service = build('gmail', 'v1', credentials=credentials)
        
        # Create email message
        message = {
            'raw': create_email_message(
                to=data["to"],
                subject=data["subject"],
                body=data["body"]
            )
        }
        
        # Create draft
        draft = service.users().drafts().create(
            userId='me',
            body={'message': message}
        ).execute()
        
        logger.info(f"Created Gmail draft: {draft['id']}")
        
    except Exception as e:
        logger.error(f"Error drafting Gmail email: {e}")

async def search_gmail_emails(data: Dict[str, Any]):
    """Search emails in Gmail"""
    try:
        # Get user credentials
        credentials = await get_user_credentials(data["user_id"], "gmail")
        
        # Build Gmail service
        service = build('gmail', 'v1', credentials=credentials)
        
        # Search emails
        results = service.users().messages().list(
            userId='me',
            q=data["query"],
            maxResults=data.get("max_results", 10)
        ).execute()
        
        messages = results.get('messages', [])
        logger.info(f"Found {len(messages)} Gmail messages")
        
    except Exception as e:
        logger.error(f"Error searching Gmail emails: {e}")

# Slack Integration
async def handle_slack_action(action: str, data: Dict[str, Any]):
    """Handle Slack actions"""
    try:
        if action == "send_message":
            await send_slack_message(data)
        elif action == "create_channel":
            await create_slack_channel(data)
        else:
            logger.warning(f"Unknown Slack action: {action}")
    except Exception as e:
        logger.error(f"Error handling Slack action: {e}")

async def send_slack_message(data: Dict[str, Any]):
    """Send a message to Slack"""
    try:
        # Get user credentials
        token = await get_user_credentials(data["user_id"], "slack")
        
        # Create Slack client
        client = slack_sdk.WebClient(token=token)
        
        # Send message
        response = client.chat_postMessage(
            channel=data["channel"],
            text=data["text"],
            blocks=data.get("blocks")
        )
        
        logger.info(f"Sent Slack message: {response['ts']}")
        
    except Exception as e:
        logger.error(f"Error sending Slack message: {e}")

# Notion Integration
async def handle_notion_action(action: str, data: Dict[str, Any]):
    """Handle Notion actions"""
    try:
        if action == "create_page":
            await create_notion_page(data)
        elif action == "search_pages":
            await search_notion_pages(data)
        else:
            logger.warning(f"Unknown Notion action: {action}")
    except Exception as e:
        logger.error(f"Error handling Notion action: {e}")

async def create_notion_page(data: Dict[str, Any]):
    """Create a page in Notion"""
    try:
        # Get user credentials
        token = await get_user_credentials(data["user_id"], "notion")
        
        # Create Notion client
        notion = Client(auth=token)
        
        # Create page
        response = notion.pages.create(
            parent={"database_id": data["database_id"]},
            properties={
                "title": {"title": [{"text": {"content": data["title"]}}]},
                "content": {"rich_text": [{"text": {"content": data["content"]}}]}
            }
        )
        
        logger.info(f"Created Notion page: {response['id']}")
        
    except Exception as e:
        logger.error(f"Error creating Notion page: {e}")

# Calendar Integration
async def handle_calendar_action(action: str, data: Dict[str, Any]):
    """Handle Calendar actions"""
    try:
        if action == "create_event":
            await create_calendar_event(data)
        elif action == "list_events":
            await list_calendar_events(data)
        else:
            logger.warning(f"Unknown Calendar action: {action}")
    except Exception as e:
        logger.error(f"Error handling Calendar action: {e}")

async def create_calendar_event(data: Dict[str, Any]):
    """Create a calendar event"""
    try:
        # Get user credentials
        credentials = await get_user_credentials(data["user_id"], "calendar")
        
        # Build Calendar service
        service = build('calendar', 'v3', credentials=credentials)
        
        # Create event
        event = {
            'summary': data["title"],
            'start': {'dateTime': data["start_time"]},
            'end': {'dateTime': data["end_time"]},
            'attendees': [{'email': email} for email in data.get("attendees", [])],
            'description': data.get("description", "")
        }
        
        response = service.events().insert(
            calendarId='primary',
            body=event
        ).execute()
        
        logger.info(f"Created calendar event: {response['id']}")
        
    except Exception as e:
        logger.error(f"Error creating calendar event: {e}")

# Helper functions
async def get_user_credentials(user_id: str, service: str) -> Dict[str, Any]:
    """Get user credentials for a service"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT credentials FROM integration_configs
                WHERE user_id = $1 AND service = $2
            """, user_id, service)
            
            if row:
                return json.loads(row["credentials"])
            else:
                raise HTTPException(status_code=404, detail="Credentials not found")
                
    except Exception as e:
        logger.error(f"Error getting user credentials: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving credentials")

def create_email_message(to: List[str], subject: str, body: str) -> str:
    """Create email message in Gmail format"""
    import base64
    from email.mime.text import MIMEText
    
    message = MIMEText(body)
    message['to'] = ', '.join(to)
    message['subject'] = subject
    
    return base64.urlsafe_b64encode(message.as_bytes()).decode()

# API Endpoints
@app.post("/integrations/gmail/draft")
@limiter.limit("30/minute")
async def draft_email(email_draft: EmailDraft, user_id: str):
    """Draft an email"""
    try:
        task_data = {
            "service": "gmail",
            "action": "draft_email",
            "data": {
                "user_id": user_id,
                "to": email_draft.to,
                "subject": email_draft.subject,
                "body": email_draft.body,
                "priority": email_draft.priority
            }
        }
        
        await redis_client.publish("integration_task", json.dumps(task_data))
        return {"status": "processing"}
        
    except Exception as e:
        logger.error(f"Error drafting email: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/integrations/slack/message")
@limiter.limit("60/minute")
async def send_message(slack_message: SlackMessage, user_id: str):
    """Send a Slack message"""
    try:
        task_data = {
            "service": "slack",
            "action": "send_message",
            "data": {
                "user_id": user_id,
                "channel": slack_message.channel,
                "text": slack_message.text,
                "blocks": slack_message.blocks
            }
        }
        
        await redis_client.publish("integration_task", json.dumps(task_data))
        return {"status": "processing"}
        
    except Exception as e:
        logger.error(f"Error sending Slack message: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/integrations/calendar/event")
@limiter.limit("30/minute")
async def create_event(calendar_event: CalendarEvent, user_id: str):
    """Create a calendar event"""
    try:
        task_data = {
            "service": "calendar",
            "action": "create_event",
            "data": {
                "user_id": user_id,
                "title": calendar_event.title,
                "start_time": calendar_event.start_time,
                "end_time": calendar_event.end_time,
                "attendees": calendar_event.attendees,
                "description": calendar_event.description
            }
        }
        
        await redis_client.publish("integration_task", json.dumps(task_data))
        return {"status": "processing"}
        
    except Exception as e:
        logger.error(f"Error creating calendar event: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "integrations"}

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(process_integration_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
