import os
import sys
import pytest

RUN_UNIT = os.getenv("RUN_UNIT", "0") == "1"

try:
    from fastapi.testclient import TestClient  # type: ignore
except Exception:
    TestClient = None  # type: ignore


def _import_app(service_name: str):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    service_path = os.path.join(root, "services", service_name)
    sys.path.insert(0, service_path)
    mod = __import__("main")
    return getattr(mod, "app", None)


@pytest.mark.skipif(not RUN_UNIT, reason="RUN_UNIT not set; skipping unit test")
@pytest.mark.skipif(TestClient is None, reason="fastapi.testclient not available")
def test_rag_metrics():
    app = _import_app("rag")
    assert app is not None, "Failed to import rag app"
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/plain")
