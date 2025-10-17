import os
import io
import time
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "")

from demo import app  # noqa: E402

client = TestClient(app)


def make_pdf(pages):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except Exception:  # pragma: no cover
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    for text in pages:
        c.setFont("Helvetica", 12)
        c.drawString(72, height - 72, text)
        c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def test_pdf_semantic_page_idx_ranking():
    pdf_bytes = make_pdf([
        "This is a generic introduction page.",
        "Unique zebra alpha content lives on this page for testing."
    ])
    if pdf_bytes is None:
        # reportlab not installed; skip
        return

    meeting_id = "pdf_sem_meeting"
    # Start meeting (best-effort)
    client.post("/meetings/start", json={
        "meeting_id": meeting_id,
        "title": "T",
        "platform": "X",
        "start_time": time.time(),
        "privacy_mode": "private",
        "participants": ["t"]
    })

    files = {
        "file": ("sample.pdf", pdf_bytes, "application/pdf")
    }
    data = {"meeting_id": meeting_id}
    r = client.post("/upload", files=files, data=data)
    assert r.status_code == 200
    resp = r.json()
    assert resp.get("status") == "success"
    file_id = resp["file_id"]

    # Query semantic for distinctive terms from page 2
    r2 = client.get("/semantic_search", params={"query": "zebra alpha", "page": 1, "per_page": 5})
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2.get("hits"), "expected semantic hits for uploaded PDF"
    top = data2["hits"][0]
    # Should include page_idx and match our file_id for page-aware chunks
    assert "page_idx" in top
    assert top.get("file_id") == file_id
    # We placed the unique string on page index 1 (0-based), so expect 1
    assert top["page_idx"] == 1
