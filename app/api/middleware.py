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

    ``script-src`` is ``'self'`` plus a single narrow exception: the Tailwind
    Play CDN origin used by the public landing (``frontend/landing/``) — the
    landing has no build step, so the utility CSS is compiled in the browser.
    All other scripts are external files under ``/js/`` or ``/console/js/``
    (including the console's anti-FOUC theme bootstrap, kept external so no
    fragile content hash is needed). Arbitrary inline/eval'd scripts stay
    forbidden.

    ``style-src`` allows ``'unsafe-inline'`` because the Tailwind CDN injects
    its generated stylesheet as a ``<style>`` element at runtime; style
    injection cannot execute script, so the risk stays contained. ``img-src``
    additionally allows ``data:`` so the inline SVG favicon loads.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
