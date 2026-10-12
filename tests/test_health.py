"""Tests for the liveness and readiness probes."""

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import health as health_module
from app.core.config import Settings
from app.core.metrics import record_backend_result, reset_backend_health
from app.main import app


@pytest.fixture(autouse=True)
def _clean_backend_health():
    reset_backend_health()
    yield
    reset_backend_health()


def test_liveness_is_always_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_reports_not_ready_without_graph(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "get_settings", lambda: Settings(openai_api_key="k"))
    with TestClient(app) as client:
        app.state.graph = None
        app.state.graph_error = "missing credentials"
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readiness_reports_ready_when_graph_initialized(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "get_settings", lambda: Settings(openai_api_key="k"))
    with TestClient(app) as client:
        app.state.graph = object()
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_reports_ok_when_no_backend_failures() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "backends": {}, "failing_backends": []}


def test_health_reports_degraded_after_backend_failure() -> None:
    record_backend_result("neo4j", ok=False)
    record_backend_result("chroma", ok=True)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["backends"] == {"neo4j": False, "chroma": True}
    assert body["failing_backends"] == ["neo4j"]


def test_health_recovers_to_ok_after_backend_succeeds() -> None:
    record_backend_result("neo4j", ok=False)
    record_backend_result("neo4j", ok=True)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
