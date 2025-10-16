import os
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "")

from demo import app  # noqa: E402

client = TestClient(app)

def test_search_empty():
    r = client.get("/search", params={"query": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    assert isinstance(data["hits"], list)

def test_semantic_empty():
    r = client.get("/semantic_search", params={"query": "strategy"})
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    assert isinstance(data["hits"], list)
