"""Shared rate limiter instance (slowapi).

A single `Limiter` is shared across the auth routes and wired into the FastAPI
app in `app/main.py`, so the `/auth/login` endpoint can be rate-limited without
duplicating state.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings

limiter = Limiter(key_func=get_remote_address)


def request_rate_limit() -> str:
    """Per-IP limit for the request-path endpoints, resolved per request.

    Passed to ``@limiter.limit`` as a callable so the value always reflects
    the current ``RATE_LIMIT_PER_MINUTE`` setting instead of the value the
    decorator happened to see at import time.
    """
    return f"{get_settings().rate_limit_per_minute}/minute"
