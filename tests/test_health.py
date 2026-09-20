"""Tests for the liveness and readiness probes."""

from fastapi.testclient import TestClient

from app.api.v1.routes import health as health_module
from app.core.config import Settings
from app.main import app


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
