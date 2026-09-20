"""Tests that the frontend static files are served without shadowing API routes."""

from fastapi.testclient import TestClient

from app.main import app


def test_serves_index_html() -> None:
    client = TestClient(app)
    response = client.get("/")
    client.close()

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_serves_static_assets() -> None:
    client = TestClient(app)
    css = client.get("/styles.css")
    js = client.get("/app.js")
    client.close()

    assert css.status_code == 200
    assert js.status_code == 200


def test_api_routes_are_not_shadowed() -> None:
    client = TestClient(app)
    health = client.get("/health/live")
    openapi = client.get("/openapi.json")
    client.close()

    assert health.status_code == 200
    assert openapi.status_code == 200
