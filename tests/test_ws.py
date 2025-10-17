import os
import json
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "")

from demo import app  # noqa: E402

client = TestClient(app)


def test_websocket_transcript_roundtrip():
    meeting_id = "ws_test_meeting"
    with client.websocket_connect(f"/ws/audio/{meeting_id}") as ws:
        # send a chat text message
        ws.send_text(json.dumps({"type": "chat", "text": "hello world"}))
        # receive back a transcript event
        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["type"] == "transcript"
        assert "utterance" in data
        assert data["utterance"]["text"]
