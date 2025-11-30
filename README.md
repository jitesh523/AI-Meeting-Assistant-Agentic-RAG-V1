# AI Meeting Assistant - Agentic RAG V1

![CI](https://github.com/jitesh523/AI-Meeting-Assistant-Agentic-RAG-V1/actions/workflows/ci.yml/badge.svg)
![Release SBOM](https://github.com/jitesh523/AI-Meeting-Assistant-Agentic-RAG-V1/actions/workflows/release.yml/badge.svg)

A proactive, Jarvis-like co-pilot for Zoom/Teams/Meet that listens, understands, retrieves context, and takes action with strong privacy controls.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd AI-Meeting-Assistant-Agentic-RAG-V1-
```

### Makefile helper

Common tasks are available via Makefile:

```bash
make build   # build prod compose images
make up      # start prod compose
make down    # stop and remove prod compose
make logs    # tail logs
make unit    # run unit tests locally (RUN_UNIT=1)
make smoke   # run local smoke against /health and /metrics
make k6      # run a short k6 smoke (requires Docker)
```

### Load testing (k6)

A tiny smoke script is provided at `scripts/load/k6-smoke.js`.

```bash
# Ensure services are running locally on default ports
make up
make k6
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
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
```

### 3. Start the Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Or start individual services for development
docker-compose up postgres redis
```

### 4. Initialize Database

```bash
# Connect to PostgreSQL and run the init script
docker exec -i ai-meeting-assistant-agentic-rag-v1-_postgres_1 psql -U postgres -d meeting_assistant < init.sql
```

### 5. Access the Application

- **UI**: http://localhost:3000
- **API Documentation**: 
  - Ingestion: http://localhost:8001/docs
  - ASR: http://localhost:8002/docs
  - NLU: http://localhost:8003/docs
  - RAG: http://localhost:8004/docs
  - Agent: http://localhost:8005/docs
  - Integrations: http://localhost:8006/docs

## 🏗️ Architecture

### Services

1. **Ingestion Service** (Port 8001)
   - WebSocket audio ingestion
   - Real-time audio processing
   - Meeting session management

2. **ASR Service** (Port 8002)
   - WhisperX speech recognition
   - Speaker diarization with PyAnnote
   - Real-time transcript generation

3. **NLU Service** (Port 8003)
   - Intent detection
   - Entity extraction
   - Sentiment analysis
   - Topic classification

4. **RAG Service** (Port 8004)
   - Document retrieval
   - Context generation
   - Vector similarity search

5. **Agent Service** (Port 8005)
   - Suggestion generation
   - Action planning
   - Tool orchestration

6. **Integrations Service** (Port 8006)
   - Gmail integration
   - Slack integration
   - Notion integration
   - Calendar integration

7. **UI Service** (Port 3000)
   - Next.js companion interface
   - Real-time transcript display
   - Suggestion management

### Data Flow

```
Audio → Ingestion → ASR → NLU → Agent → UI
                ↓
            RAG ← Integrations
```

## 🔧 Development

### Local Development Setup

1. **Start Infrastructure**:
   ```bash
   docker-compose up postgres redis -d
   ```

2. **Run Services Locally**:
   ```bash
   # Terminal 1 - Ingestion
   cd services/ingestion
   pip install -r requirements.txt
   python main.py

   # Terminal 2 - ASR
   cd services/asr
   pip install -r requirements.txt
   python main.py

   # Terminal 3 - NLU
   cd services/nlu
   pip install -r requirements.txt
   python main.py

   # Terminal 4 - RAG
   cd services/rag
   pip install -r requirements.txt
   python main.py

   # Terminal 5 - Agent
   cd services/agent
   pip install -r requirements.txt
   python main.py

   # Terminal 6 - Integrations
   cd services/integrations
   pip install -r requirements.txt
   python main.py

   # Terminal 7 - UI
   cd services/ui
   npm install
   npm run dev
   ```

### Testing

```bash
# Run tests for each service
cd services/ingestion && python -m pytest
cd services/asr && python -m pytest
cd services/nlu && python -m pytest
cd services/rag && python -m pytest
cd services/agent && python -m pytest
cd services/integrations && python -m pytest
```

## 📊 Features

### Current (MVP)
- ✅ Real-time audio transcription (WebSocket Stream)
- ✅ Speaker diarization
- ✅ Intent detection and entity extraction
- ✅ Basic RAG for context retrieval
- ✅ AI suggestion generation
- ✅ Web-based companion interface
- ✅ Privacy mode controls
- ✅ Post-meeting summaries (Enterprise)

### Planned (V1)
- 🔄 Gmail/Outlook integration
- 🔄 Slack integration
- 🔄 Notion integration
- 🔄 Calendar integration
- 🔄 Action item tracking
- 🔄 Advanced privacy controls

## 🔒 Privacy & Security

### Privacy Modes
- **Notes-Only**: AI takes notes but doesn't store audio
- **Transcript+Notes**: Full transcription with note-taking
- **Off-the-Record**: Ephemeral processing, no storage

### Security Features
- End-to-end encryption for sensitive data
- Granular consent management
- Audit logging for all actions
- Data retention policies

### Access Control & Reliability (Ops quick refs)
- Auth (API key): set `AUTH_ENABLED=1` and `SERVICE_API_KEY=<secret>` per service. Health/metrics/docs are exempt by default.
- CORS: default deny; allow-list via `CORS_ALLOW_ORIGINS="https://app.example.com,https://admin.example.com"`.
- Idempotency: Redis-backed keys with TTL (`IDEMPOTENCY_TTL_SECONDS`, default 600) and in-memory fallback.
- Integrations resilience (retry/circuit):
  - Circuit: `CB_WINDOW_SECONDS` (60), `CB_MAX_FAILURES` (5)
  - Retries: `RETRY_MAX_ATTEMPTS` (3), `RETRY_BASE_SECONDS` (0.5)
  - Timeouts: `PROVIDER_TIMEOUT_DEFAULT_SECONDS` (5), `PROVIDER_TOTAL_BUDGET_SECONDS` (10), overrides: `TIMEOUT_<PROVIDER>_SECONDS`

See `docs/infra.md` for full details.

## 📈 Monitoring

### Health Checks
- All services expose `/health` endpoints
- Docker health checks configured
- Redis and PostgreSQL health monitoring

### Logging
- Structured logging across all services
- Centralized log aggregation
- Error tracking and alerting

### Prometheus and Grafana

- Prometheus alert rules: `monitoring/prometheus/alerts.yml`
- Grafana dashboard template: `monitoring/grafana/dashboard.json`

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**:
   ```bash
   # Set production environment variables
   export NODE_ENV=production
   export DATABASE_URL=postgresql://user:pass@prod-db:5432/meeting_assistant
   export REDIS_URL=redis://prod-redis:6379
   ```

2. **Build and Deploy**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Database Migration**:
   ```bash
   docker exec -i <postgres-container> psql -U postgres -d meeting_assistant < init.sql
   ```

#### Prod-like compose and infra docs

For a prod-like run with observability enabled and healthchecks tuned:

```
docker compose -f docker-compose.prod.yml --env-file env.example up --build
```

See docs/infra.md for details on:

- Secrets and environment variables
- Health and metrics endpoints per service
- Request size/time guards, request IDs, and startup retries
- Optional OpenTelemetry tracing

#### Example compose.override.yml (resource reservations)

If you use Docker Swarm or want to define deploy resource reservations/limits, you can add an override file:

```yaml
services:
  nlu:
    deploy:
      resources:
        reservations:
          cpus: '0.50'
          memory: 512M
        limits:
          cpus: '1.0'
          memory: 1G
```

### Scaling

- **Horizontal Scaling**: Add more ASR workers
- **Database Scaling**: Read replicas for queries
- **Caching**: Redis clustering for high availability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: [Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)

## 🗺️ Roadmap

### Phase 1 (Current)
- [x] Core infrastructure setup
- [x] Basic ASR and NLU
- [x] Simple RAG implementation
- [x] Web UI prototype

### Phase 2 (Next 4-6 weeks)
- [ ] Integration connectors
- [ ] Advanced agent capabilities
- [ ] Post-meeting workflows
- [ ] Privacy controls

### Phase 3 (Next 8-10 weeks)
- [ ] Enterprise features
- [ ] Mobile app
- [ ] Advanced analytics
- [ ] Multi-language support

---

**Built with ❤️ for better meetings**
