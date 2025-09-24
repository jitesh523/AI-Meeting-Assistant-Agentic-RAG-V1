#!/usr/bin/env python3
"""
AI Meeting Assistant Demo - Co-pilot Nexus UI Integration
This demonstrates the core functionality with the beautiful co-pilot-nexus UI
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
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

# In-memory storage (for demo purposes)
meetings = {}
transcripts = {}
suggestions = {}
active_connections = {}

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
    """Simulate AI processing and generate suggestions"""
    await asyncio.sleep(1)  # Simulate processing time
    
    # Generate a random suggestion
    import random
    suggestion = Suggestion(
        id=f"suggestion_{int(time.time())}",
        kind=random.choice(["ask", "task", "fact", "email"]),
        text=random.choice(AI_RESPONSES),
        confidence=random.uniform(0.7, 0.95),
        reasons=["AI detected relevant context", "Meeting pattern analysis"]
    )
    
    suggestions[meeting_id] = suggestions.get(meeting_id, [])
    suggestions[meeting_id].append(suggestion)
    
    # Send suggestion to UI
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
            # Simulate receiving audio data
            data = await websocket.receive_text()
            
            # Simulate processing
            utterance = Utterance(
                speaker="Speaker 1",
                text=data,
                timestamp=datetime.now().isoformat(),
                confidence=0.9
            )
            
            # Store transcript
            if meeting_id not in transcripts:
                transcripts[meeting_id] = []
            transcripts[meeting_id].append(utterance)
            
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
    return {
        "meeting_id": meeting_id,
        "transcript": [u.model_dump() for u in transcripts.get(meeting_id, [])]
    }

@app.get("/meetings/{meeting_id}/suggestions")
async def get_suggestions(meeting_id: str):
    return {
        "meeting_id": meeting_id,
        "suggestions": [s.model_dump() for s in suggestions.get(meeting_id, [])]
    }

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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-meeting-assistant"}

# Serve the demo HTML with Co-pilot Nexus UI
@app.get("/")
async def get_demo():
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

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
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

                        <div id="suggestions" class="space-y-4 max-h-96 overflow-y-auto">
                            <div class="suggestion-card">
                                <div class="flex items-start gap-3 mb-3">
                                    <div class="p-1.5 rounded bg-purple-500/10">
                                        <i data-lucide="message-square" class="h-4 w-4 text-purple-400"></i>
                                    </div>
                                    <div class="flex-1">
                                        <span class="badge badge-ask mb-2">ASK</span>
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
            </div>

            <!-- Live Transcript -->
            <div class="ai-card">
                <div class="space-y-4">
                    <h3 class="text-lg font-semibold flex items-center gap-2">
                        <i data-lucide="file-text" class="h-5 w-5"></i>
                        Live Transcript
                    </h3>
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
            
            function toggleRecording() {
                if (!isConnected) return;
                isRecording = !isRecording;
                const recordBtn = document.getElementById('recordBtn');
                const recordText = document.getElementById('recordText');
                
                if (isRecording) {
                    recordBtn.className = 'ai-button-primary bg-red-600 hover:bg-red-700';
                    recordText.innerHTML = '<i data-lucide="square" class="h-4 w-4 mr-2"></i>Stop';
                } else {
                    recordBtn.className = 'ai-button-primary';
                    recordText.innerHTML = '<i data-lucide="mic" class="h-4 w-4 mr-2"></i>Record';
                }
                lucide.createIcons();
            }
            
            function connect() {
                if (ws) return;
                
                ws = new WebSocket(`ws://localhost:8001/ws/audio/${meetingId}`);
                
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
                    }
                };
                
                ws.onclose = function() {
                    updateStatus(false);
                    ws = null;
                    console.log('Disconnected from WebSocket');
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
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
                    'fact': 'file-text'
                };
                
                const badgeClass = {
                    'ask': 'badge-ask',
                    'email': 'badge-email',
                    'task': 'badge-task',
                    'fact': 'badge-fact'
                };
                
                suggestionDiv.innerHTML = `
                    <div class="flex items-start gap-3 mb-3">
                        <div class="p-1.5 rounded bg-purple-500/10">
                            <i data-lucide="${iconMap[suggestion.kind] || 'message-square'}" class="h-4 w-4 text-purple-400"></i>
                        </div>
                        <div class="flex-1">
                            <span class="badge ${badgeClass[suggestion.kind] || 'badge-ask'} mb-2">${suggestion.kind.toUpperCase()}</span>
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
                    ws.send(message);
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
            
            // Auto-connect on page load
            connect();
            
            // Update duration every second
            setInterval(updateDuration, 1000);
            
            // Handle Enter key in input
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("🚀 Starting AI Meeting Assistant with Co-pilot Nexus UI...")
    print("📱 Open your browser to: http://localhost:8001")
    print("🔧 Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)