"""Tests that security headers are applied to every response."""

from fastapi.testclient import TestClient

from app.main import app


def test_security_headers_are_present() -> None:
    client = TestClient(app)
    response = client.get("/health/live")
    client.close()

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_security_headers_on_frontend() -> None:
    client = TestClient(app)
    response = client.get("/")
    client.close()

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
