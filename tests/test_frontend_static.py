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


def test_layout_phase2_responsive_grid_and_breakpoints() -> None:
    """Phase 2: mobile-first grid layout with explicit breakpoints.

    Asserts the main region uses a CSS Grid that cannot force horizontal
    overflow (`minmax(0, 1fr)`), the header is sticky, explicit tablet/desktop
    `min-width` breakpoints exist, and the wide "recent requests" table is
    wrapped in a scroll container rather than overflowing the page at 320px.
    """
    client = TestClient(app)
    html = client.get("/").text
    css = client.get("/styles.css").text
    client.close()

    # Main content is a single-column, overflow-safe grid.
    assert ".layout" in css
    assert "display: grid" in css
    assert "grid-template-columns: minmax(0, 1fr)" in css

    # The header stays fixed to the top while the content scrolls beneath it.
    assert "position: sticky" in css

    # Explicit mobile-first breakpoints for tablet and desktop (min-width,
    # not max-width, so base rules target the smallest screen).
    assert "--bp-tablet: 640px" in css
    assert "--bp-desktop: 1024px" in css
    assert "@media (min-width: 640px)" in css
    assert "@media (min-width: 1024px)" in css

    # The dashboard table is wrapped so it scrolls inside its own container
    # and never forces a horizontal page scroll on narrow viewports.
    assert 'class="table-wrap"' in html
    assert ".table-wrap" in css
    assert "overflow-x: auto" in css
