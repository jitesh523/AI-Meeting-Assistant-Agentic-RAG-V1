import os
import sys
import pytest

RUN_UNIT = os.getenv("RUN_UNIT", "0") == "1"

try:
    from fastapi.testclient import TestClient  # type: ignore
except Exception:
    TestClient = None  # type: ignore

SEC_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
}


def _import_app(service_name: str):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    service_path = os.path.join(root, "services", service_name)
    sys.path.insert(0, service_path)
    mod = __import__("main")
    return getattr(mod, "app", None)


@pytest.mark.skipif(not RUN_UNIT, reason="RUN_UNIT not set; skipping unit test")
@pytest.mark.skipif(TestClient is None, reason="fastapi.testclient not available")
@pytest.mark.parametrize("service", ["ingestion", "agent", "asr", "nlu", "rag", "integrations"]) 
def test_security_headers_on_health(service):
    app = _import_app(service)
    assert app is not None, f"Failed to import {service} app"
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    for k, v in SEC_HEADERS.items():
        assert r.headers.get(k) == v
