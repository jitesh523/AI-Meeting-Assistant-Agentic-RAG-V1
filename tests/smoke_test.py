import os
import time
import uuid
import requests

BASES = {
    "ingestion": os.getenv("INGESTION_URL", "http://localhost:8001"),
    "nlu": os.getenv("NLU_URL", "http://localhost:8003"),
    "agent": os.getenv("AGENT_URL", "http://localhost:8005"),
}

RUN_SMOKE = os.getenv("RUN_SMOKE", "0") == "1"
TIMEOUT = int(os.getenv("SMOKE_TIMEOUT_SEC", "20"))


def wait_health(name: str, url: str, timeout: int = TIMEOUT):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.ok:
                return True
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise AssertionError(f"{name} not healthy at {url}/health: {last_err}")


def test_health_endpoints():
    if not RUN_SMOKE:
        import pytest
        pytest.skip("RUN_SMOKE not set; skipping health checks")
    for name, base in BASES.items():
        wait_health(name, base)


def test_end_to_end_suggestions():
    if not RUN_SMOKE:
        # Skip heavy E2E unless explicitly requested
        import pytest
        pytest.skip("RUN_SMOKE not set; skipping end-to-end smoke test")

    meeting_id = str(uuid.uuid4())

    # 1) Start meeting (optional)
    r = requests.post(
        f"{BASES['ingestion']}/meetings/start",
        json={
            "meeting_id": meeting_id,
            "title": "Smoke Test",
            "platform": "web",
            "start_time": time.time(),
            "privacy_mode": "transcript+notes",
            "participants": ["User"],
        },
        timeout=5,
    )
    assert r.ok, r.text

    # 2) Post a text utterance (bypasses audio path)
    r = requests.post(
        f"{BASES['ingestion']}/meetings/{meeting_id}/utterances",
        json={
            "speaker": "User",
            "text": "Can you create a task to follow up?",
            "timestamp": time.time(),
        },
        timeout=5,
    )
    assert r.ok, r.text

    # 3) Poll agent suggestions
    deadline = time.time() + 15
    suggestions = []
    while time.time() < deadline:
        rs = requests.get(
            f"{BASES['agent']}/agent/meetings/{meeting_id}/suggestions",
            timeout=5,
        )
        if rs.ok:
            data = rs.json()
            suggestions = data.get("suggestions", [])
            if suggestions:
                break
        time.sleep(1)

    assert suggestions, "Expected at least one suggestion from agent"
