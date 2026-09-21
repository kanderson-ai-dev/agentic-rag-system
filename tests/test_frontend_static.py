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


def test_design_system_phase1_theme_toggle_and_tokens() -> None:
    """Phase 1: base visual system is present and wired.

    Asserts the theme toggle exists in the markup, both color themes are
    defined via CSS custom properties (tokens), and the JS persists the
    choice to localStorage rather than leaving the theme hard-coded.
    """
    client = TestClient(app)
    html = client.get("/").text
    css = client.get("/styles.css").text
    js = client.get("/app.js").text
    client.close()

    # Theme toggle button present in the header, exposed to assistive tech.
    assert 'id="theme-toggle"' in html
    assert 'aria-label="Toggle color theme"' in html

    # Both themes are declared as design-token blocks (light and dark).
    assert '[data-theme="light"]' in css
    assert ':root[data-theme="dark"]' in css or '[data-theme="dark"]' in css

    # Theme persistence uses localStorage, not a session-only or hard-coded value.
    assert 'localStorage.setItem("theme", theme)' in js
    assert 'localStorage.getItem("theme")' in html
