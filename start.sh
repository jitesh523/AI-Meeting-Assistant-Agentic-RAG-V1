#!/bin/bash

# AI Meeting Assistant Startup Script

echo "🚀 Starting AI Meeting Assistant..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cat > .env << EOF
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/meeting_assistant
REDIS_URL=redis://localhost:6379

# OpenAI API
OPENAI_API_KEY=your-openai-api-key-here

# Hugging Face (for diarization)
HUGGING_FACE_TOKEN=your-hf-token-here

# Integration APIs
GMAIL_CLIENT_ID=your-gmail-client-id
GMAIL_CLIENT_SECRET=your-gmail-client-secret
SLACK_BOT_TOKEN=your-slack-bot-token
NOTION_API_KEY=your-notion-api-key
EOF
    echo "📝 Please edit .env file with your API keys and run again."
    exit 1
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Initialize database
echo "🗄️  Initializing database..."
docker exec -i ai-meeting-assistant-agentic-rag-v1-_postgres_1 psql -U postgres -d meeting_assistant < init.sql

# Start all services
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Check service health
echo "🔍 Checking service health..."
services=("ingestion:8001" "asr:8002" "nlu:8003" "rag:8004" "agent:8005" "integrations:8006" "ui:3000")

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s http://localhost:$port/health > /dev/null; then
        echo "✅ $name service is healthy"
    else
        echo "❌ $name service is not responding"
    fi
done

echo ""
echo "🎉 AI Meeting Assistant is ready!"
echo ""
echo "📱 Access the application:"
echo "   UI: http://localhost:3000"
echo "   API Docs: http://localhost:8001/docs"
echo ""
echo "🔧 To stop all services:"
echo "   docker-compose down"
echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f"
