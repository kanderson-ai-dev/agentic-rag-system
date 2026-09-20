"""Authentication endpoints (single-user JWT login)."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.v1.schemas import LoginRequest, LoginResponse
from app.core.config import Settings, get_settings
from app.core.rate_limit import limiter
from app.core.security import create_access_token, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
def login(
    request: Request,
    payload: LoginRequest,
    settings: Settings = Depends(get_settings),
) -> LoginResponse:
    """Authenticate a user and return a short-lived JWT.

    Returns a generic 401 for any invalid credential (without revealing whether
    the username exists) and 503 when authentication is not configured.
    """
    if not settings.jwt_secret_key or not settings.auth_username or not settings.auth_password_hash:
        raise HTTPException(status_code=503, detail="Authentication is not configured")

    if payload.username != settings.auth_username:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(payload.password, settings.auth_password_hash_value() or ""):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(settings, settings.auth_username)
    return LoginResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
    )
