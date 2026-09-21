"""FastAPI middleware."""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or propagate an `X-Request-ID` and bind it to structured logs."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply security headers to every response (OWASP A05).

    The CSP keeps ``script-src`` to ``'self'`` plus a single hash for the
    inline anti-FOUC theme bootstrap in ``frontend/index.html`` (the only
    inline script; everything else is an external ES module under ``/js/``).
    This forbids arbitrary inline/eval'd scripts while still letting the
    theme apply before first paint. ``img-src`` additionally allows ``data:``
    so the inline SVG favicon loads.
    """

    # SHA-256 (base64) of the inline theme-bootstrap script. Regenerate with:
    #   python -c "import hashlib,base64; s=open('frontend/index.html').read(); \
    #     m=__import__('re').search(r'<script>\\s*(.*?)\\s*</script>', s, __import__('re').S); \
    #     print(base64.b64encode(hashlib.sha256(m.group(1).encode()).digest()).decode())"
    _THEME_SCRIPT_HASH = "R3wic9bWfHJN9coXPA5E2AnFTEGN74be6rfe7Vtk8Jk="

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'sha256-" + self._THEME_SCRIPT_HASH + "'; "
            "img-src 'self' data:"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
