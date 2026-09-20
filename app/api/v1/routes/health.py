"""Liveness and readiness probes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health/live")
def liveness() -> dict[str, str]:
    """Always returns 200 once the process is up and serving requests."""
    return {"status": "alive"}


@router.get("/health/ready")
def readiness(request: Request) -> JSONResponse:
    """Return 200 only when the service is ready, otherwise 503 with issues."""
    settings = get_settings()
    issues: list[str] = []

    if not settings.openai_api_key:
        issues.append("OPENAI_API_KEY is not configured")

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        error = getattr(request.app.state, "graph_error", "graph not initialized")
        issues.append(f"graph not initialized: {error}")

    if settings.neo4j_uri and not settings.neo4j_password:
        issues.append("NEO4J_URI is configured but NEO4J_PASSWORD is missing")

    if issues:
        return JSONResponse(status_code=503, content={"status": "not_ready", "issues": issues})
    return JSONResponse(status_code=200, content={"status": "ready"})
