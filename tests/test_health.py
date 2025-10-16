import os
from fastapi.testclient import TestClient

# Ensure optional integrations don't break tests without keys
os.environ.setdefault("GROQ_API_KEY", "")

from demo import app  # noqa: E402

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "asr_server" in data
    assert "embeddings" in data
