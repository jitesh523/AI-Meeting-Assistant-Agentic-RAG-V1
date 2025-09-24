-- AI Meeting Assistant Database Schema
-- This file initializes the PostgreSQL database with all required tables

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Meetings table
CREATE TABLE meetings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID,
    platform TEXT NOT NULL,
    title TEXT,
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ,
    privacy_mode TEXT DEFAULT 'transcript+notes',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Utterances table (transcript data)
CREATE TABLE utterances (
    id BIGSERIAL PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    speaker TEXT,
    start_ms INTEGER,
    end_ms INTEGER,
    text TEXT,
    conf REAL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- NLU results table
CREATE TABLE nlu_results (
    id BIGSERIAL PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    speaker TEXT,
    text TEXT,
    timestamp REAL,
    intent TEXT,
    entities JSONB,
    sentiment TEXT,
    confidence REAL,
    topics JSONB,
    is_decision BOOLEAN DEFAULT FALSE,
    is_question BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Suggestions table (AI suggestions)
CREATE TABLE suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    payload JSONB,
    confidence REAL,
    reasons JSONB,
    citations JSONB,
    status TEXT DEFAULT 'pending',
    approved_by UUID,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Actions table (tool executions)
CREATE TABLE actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    tool TEXT NOT NULL,
    input_data JSONB,
    output_data JSONB,
    status TEXT DEFAULT 'pending',
    approved_by UUID,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- RAG results table
CREATE TABLE rag_results (
    id BIGSERIAL PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    context TEXT,
    confidence REAL,
    document_ids JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Documents table (for RAG)
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(384), -- Adjust dimension based on embedding model
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Integration configurations table
CREATE TABLE integration_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    service TEXT NOT NULL,
    credentials JSONB NOT NULL,
    settings JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, service)
);

-- Privacy consents table
CREATE TABLE privacy_consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    participant_email TEXT,
    consent_type TEXT, -- 'transcript', 'storage', 'actions'
    granted_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);

-- Meeting analytics table
CREATE TABLE meeting_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    metric_name TEXT,
    metric_value JSONB,
    calculated_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX idx_utterances_meeting_id ON utterances(meeting_id);
CREATE INDEX idx_utterances_created_at ON utterances(created_at);
CREATE INDEX idx_nlu_results_meeting_id ON nlu_results(meeting_id);
CREATE INDEX idx_nlu_results_intent ON nlu_results(intent);
CREATE INDEX idx_suggestions_meeting_id ON suggestions(meeting_id);
CREATE INDEX idx_suggestions_status ON suggestions(status);
CREATE INDEX idx_actions_meeting_id ON actions(meeting_id);
CREATE INDEX idx_actions_status ON actions(status);
CREATE INDEX idx_rag_results_meeting_id ON rag_results(meeting_id);
CREATE INDEX idx_documents_tenant_id ON documents(tenant_id);
CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for integration_configs
CREATE TRIGGER update_integration_configs_updated_at 
    BEFORE UPDATE ON integration_configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some sample data for testing
INSERT INTO meetings (id, platform, title, start_ts, privacy_mode) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'web', 'AI Meeting Assistant Demo', now(), 'transcript+notes');

-- Insert sample utterances
INSERT INTO utterances (meeting_id, speaker, start_ms, end_ms, text, conf) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'Speaker 1', 0, 3000, 'Hello everyone, welcome to our meeting.', 0.95),
('550e8400-e29b-41d4-a716-446655440000', 'Speaker 2', 3000, 6000, 'Thanks for having me. I am excited to discuss the project.', 0.88),
('550e8400-e29b-41d4-a716-446655440000', 'Speaker 1', 6000, 9000, 'Let us start with the agenda items.', 0.92);

-- Insert sample NLU results
INSERT INTO nlu_results (meeting_id, speaker, text, timestamp, intent, entities, sentiment, confidence, topics, is_decision, is_question) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'Speaker 1', 'Hello everyone, welcome to our meeting.', 0, 'greeting', '[]', 'positive', 0.95, '["meeting"]', false, false),
('550e8400-e29b-41d4-a716-446655440000', 'Speaker 2', 'Thanks for having me. I am excited to discuss the project.', 3, 'greeting', '[]', 'positive', 0.88, '["project"]', false, false),
('550e8400-e29b-41d4-a716-446655440000', 'Speaker 1', 'Let us start with the agenda items.', 6, 'meeting_control', '[]', 'neutral', 0.92, '["meeting", "agenda"]', false, false);

-- Insert sample suggestions
INSERT INTO suggestions (meeting_id, kind, text, payload, confidence, reasons, citations, status) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'fact', 'Meeting started with agenda discussion', '{"type": "meeting_start", "participants": 2}', 0.9, '["Meeting control detected", "Agenda mentioned"]', '[]', 'pending'),
('550e8400-e29b-41d4-a716-446655440000', 'ask', 'Would you like me to search for project information?', '{"query": "project information", "topics": ["project"]}', 0.85, '["Project mentioned", "Context available"]', '[]', 'pending');

-- Insert sample documents for RAG
INSERT INTO documents (tenant_id, source, text, metadata, embedding) VALUES
('tenant-123', 'gmail', 'Project Alpha is our main initiative for Q4. We need to complete the MVP by December 15th.', '{"subject": "Project Alpha Update", "date": "2024-01-15"}', '[0.1, 0.2, 0.3]'::vector),
('tenant-123', 'notion', 'The project timeline includes three phases: planning, development, and testing.', '{"page": "Project Timeline", "updated": "2024-01-10"}', '[0.4, 0.5, 0.6]'::vector),
('tenant-123', 'slack', 'Team standup: We are on track for the sprint goals. No blockers identified.', '{"channel": "general", "timestamp": "2024-01-16T10:00:00Z"}', '[0.7, 0.8, 0.9]'::vector);
