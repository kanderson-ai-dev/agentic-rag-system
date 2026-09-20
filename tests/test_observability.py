"""Tests for observability: readiness probes, request ID and secret redaction."""

from fastapi.testclient import TestClient

from app.api.v1.routes import health as health_module
from app.core.config import Settings
from app.core.logging import _redact_secrets
from app.main import app


def test_liveness_is_always_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_reports_missing_openai_key(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "get_settings", lambda: Settings(openai_api_key=None))
    with TestClient(app) as client:
        app.state.graph = object()
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert any("OPENAI_API_KEY" in issue for issue in response.json()["issues"])


def test_readiness_reports_not_ready_without_graph(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "get_settings", lambda: Settings(openai_api_key="k"))
    with TestClient(app) as client:
        app.state.graph = None
        app.state.graph_error = "missing credentials"
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert any("graph" in issue for issue in response.json()["issues"])


def test_request_id_is_added_to_response() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert "X-Request-ID" in response.headers


def test_request_id_is_propagated() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


class TestSecretRedaction:
    def test_redacts_sensitive_fields(self) -> None:
        event = {
            "openai_api_key": "secret",
            "auth_token": "secret",
            "password": "secret",
            "authorization": "Bearer secret",
            "normal_field": "ok",
        }

        redacted = _redact_secrets(None, "info", event)

        assert redacted["openai_api_key"] == "***"
        assert redacted["auth_token"] == "***"
        assert redacted["password"] == "***"
        assert redacted["authorization"] == "***"
        assert redacted["normal_field"] == "ok"
