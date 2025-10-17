import os
import sqlite3
import json as pyjson
import time
from fastapi.testclient import TestClient

# Ensure no external API is required
os.environ.setdefault("GROQ_API_KEY", "")

from demo import app, DB_PATH, compute_embedding  # noqa: E402

client = TestClient(app)


def seed_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Clean
    try:
        cur.execute("DELETE FROM transcripts_fts")
    except Exception:
        pass
    try:
        cur.execute("DELETE FROM embeddings")
    except Exception:
        pass
    # Seed FTS: include exact keyword match and a distractor
    cur.execute(
        "INSERT INTO transcripts_fts (meeting_id, speaker, text, timestamp) VALUES (?,?,?,?)",
        ("m1", "A", "the quick brown fox jumps over the lazy dog", str(time.time()))
    )
    cur.execute(
        "INSERT INTO transcripts_fts (meeting_id, speaker, text, timestamp) VALUES (?,?,?,?)",
        ("m1", "B", "unrelated sentence about weather and sunshine", str(time.time()))
    )
    # Seed embeddings: one semantically similar to query, one not
    q_sim = "fast brown animal leaps"
    v_sim = compute_embedding(q_sim)
    v_far = compute_embedding("invoice balance sheet and accounting numbers")
    cur.execute(
        "INSERT OR REPLACE INTO embeddings (id, kind, text, vector) VALUES (?,?,?,?)",
        ("T:m1:sim1", "transcript", q_sim, pyjson.dumps(v_sim))
    )
    cur.execute(
        "INSERT OR REPLACE INTO embeddings (id, kind, text, vector) VALUES (?,?,?,?)",
        ("T:m1:far1", "transcript", "finance text", pyjson.dumps(v_far))
    )
    conn.commit()
    conn.close()


def test_ranking_keyword_vs_semantic():
    seed_db()
    # FTS should surface exact keyword result first
    r1 = client.get("/search", params={"query": "lazy dog", "meeting_id": "m1", "k": 5})
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1.get("hits"), "expected hits from FTS"
    top = data1["hits"][0]
    assert top["source"] == "transcript"
    assert "lazy dog" in top["text"].lower()

    # Semantic should rank the semantically similar text higher
    r2 = client.get("/semantic_search", params={"query": "quick brown fox", "page": 1, "per_page": 5})
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2.get("hits"), "expected semantic hits"
    texts = [h["text"].lower() for h in data2["hits"]]
    assert texts[0].startswith("fast brown") or "brown" in texts[0]
