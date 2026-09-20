"""Shared rate limiter instance (slowapi).

A single `Limiter` is shared across the auth routes and wired into the FastAPI
app in `app/main.py`, so the `/auth/login` endpoint can be rate-limited without
duplicating state.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
