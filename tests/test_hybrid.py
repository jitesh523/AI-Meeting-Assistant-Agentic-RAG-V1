import os
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "")

from demo import app  # noqa: E402

client = TestClient(app)

def test_hybrid_simple():
    r = client.get("/hybrid_search", params={"query": "quick brown fox", "alpha": 0.6, "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    assert isinstance(data["hits"], list)
