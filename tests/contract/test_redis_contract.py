import json
import pytest

# Gate by RUN_UNIT to keep tests lightweight
import os
pytestmark = pytest.mark.skipif(os.getenv("RUN_UNIT") != "1", reason="RUN_UNIT not set")


class FakeRedis:
    def __init__(self):
        self.published = []

    async def publish(self, channel, data):
        # Record channel and data (as string)
        if not isinstance(data, str):
            data = json.dumps(data)
        self.published.append((channel, data))
        return 1


@pytest.mark.asyncio
async def test_ingestion_publishes_agent_process(monkeypatch):
    from services.ingestion import main as ingestion

    fake = FakeRedis()
    ingestion.redis_client = fake

    # Build minimal utterance payload
    Utterance = getattr(ingestion, "TextUtterance")
    payload = Utterance(speaker="u1", text="What is the status?", timestamp=0.0)

    resp = await ingestion.post_utterance("m1", payload)
    assert resp.get("status") == "success"
    assert any(ch == "agent_process" for ch, _ in fake.published)


@pytest.mark.asyncio
async def test_nlu_send_to_agent_channel(monkeypatch):
    from services.nlu import main as nlu

    fake = FakeRedis()
    nlu.redis_client = fake

    NLUResult = getattr(nlu, "NLUResult")
    obj = NLUResult(
        meeting_id="m1",
        speaker="u1",
        text="Schedule a meeting tomorrow",
        timestamp=0.0,
        intent="schedule",
        entities=[],
        sentiment="neutral",
        confidence=0.9,
        topics=[],
        is_decision=False,
        is_question=False,
    )

    await nlu.send_to_agent(obj)
    assert any(ch == "agent_process" for ch, _ in fake.published)


@pytest.mark.asyncio
async def test_rag_send_to_agent_channel(monkeypatch):
    from services.rag import main as rag

    fake = FakeRedis()
    rag.redis_client = fake

    QueryResult = getattr(rag, "QueryResult")
    obj = QueryResult(
        meeting_id="m1",
        query="project plan",
        context="context",
        confidence=0.7,
        documents=[],
    )

    await rag.send_to_agent(obj, meeting_id="m1")
    assert any(ch == "agent_rag_result" for ch, _ in fake.published)


@pytest.mark.asyncio
async def test_integrations_publish_integration_task(monkeypatch):
    from services.integrations import main as integ

    fake = FakeRedis()
    integ.redis_client = fake

    EmailDraft = getattr(integ, "EmailDraft")
    email = EmailDraft(to=["a@example.com"], subject="Sub", body="Body")
    resp1 = await integ.draft_email(email, user_id="u1")

    SlackMessage = getattr(integ, "SlackMessage")
    slack = SlackMessage(channel="C123", text="Hello")
    resp2 = await integ.send_message(slack, user_id="u1")

    CalendarEvent = getattr(integ, "CalendarEvent")
    cal = CalendarEvent(title="Meet", start_time="2024-01-01T10:00:00Z", end_time="2024-01-01T11:00:00Z")
    resp3 = await integ.create_event(cal, user_id="u1")

    assert resp1.get("status") == "processing"
    assert resp2.get("status") == "processing"
    assert resp3.get("status") == "processing"
    channels = [ch for ch, _ in fake.published]
    # All should publish to the same task channel
    assert all(ch == "integration_task" for ch in channels)


@pytest.mark.asyncio
async def test_agent_send_to_ui_channel(monkeypatch):
    from services.agent import main as agent

    fake = FakeRedis()
    agent.redis_client = fake

    Suggestion = getattr(agent, "Suggestion")
    s = Suggestion(
        id="s1",
        meeting_id="m1",
        kind="fact",
        text="A fact",
        payload={},
        confidence=0.9,
        reasons=[],
        citations=[],
    )

    await agent.send_to_ui([s], meeting_id="m1")
    assert any(ch == "ui_suggestions" for ch, _ in fake.published)
