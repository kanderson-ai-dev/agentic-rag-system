"""Liveness and readiness probes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health/live")
def liveness() -> dict[str, str]:
    """Always returns 200 once the process is up and serving requests."""
    return {"status": "alive"}


@router.get("/health/ready")
def readiness(request: Request) -> JSONResponse:
    """Returns 200 only when the Self-RAG graph has been initialized."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        error = getattr(request.app.state, "graph_error", "graph not initialized")
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": error})
    return JSONResponse(status_code=200, content={"status": "ready"})
