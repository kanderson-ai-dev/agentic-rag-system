"""JWT and password helpers for single-user authentication."""

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.core.config import Settings


def create_access_token(settings: Settings, username: str) -> str:
    """Create a short-lived HS256 JWT for the given username."""
    now = datetime.now(UTC)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(settings: Settings, token: str) -> dict[str, Any]:
    """Decode and validate a JWT, raising `jwt.PyJWTError` on failure."""
    return jwt.decode(
        token,
        settings.jwt_secret_key_value(),
        algorithms=[settings.jwt_algorithm],
    )


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
