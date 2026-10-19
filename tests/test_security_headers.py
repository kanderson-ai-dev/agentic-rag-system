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


_STRICT_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.tailwindcss.com; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:"
)


def _directive(csp: str, name: str) -> str:
    for part in csp.split(";"):
        part = part.strip()
        if part.startswith(name):
            return part
    return ""


class TestContentSecurityPolicy:
    def test_strict_csp_unchanged_on_landing_console_api_and_openapi(self) -> None:
        client = TestClient(app)
        for path in ("/", "/console/", "/api/v1/auth/status", "/openapi.json"):
            response = client.get(path)
            assert response.headers["Content-Security-Policy"] == _STRICT_CSP, path
        client.close()

    def test_docs_pages_get_scoped_relaxed_csp(self) -> None:
        client = TestClient(app)
        for path in ("/docs", "/redoc"):
            response = client.get(path)
            csp = response.headers["Content-Security-Policy"]

            script_src = _directive(csp, "script-src")
            assert "'unsafe-inline'" in script_src, path
            assert "https://cdn.jsdelivr.net" in script_src, path
            assert "cdn.tailwindcss.com" not in script_src, path

            style_src = _directive(csp, "style-src")
            assert "https://cdn.jsdelivr.net" in style_src, path

            img_src = _directive(csp, "img-src")
            assert "https://fastapi.tiangolo.com" in img_src, path

            assert "default-src 'self'" in csp, path
            assert response.headers["X-Frame-Options"] == "DENY", path
        client.close()
