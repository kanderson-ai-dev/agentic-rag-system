"""Dashboard endpoints feeding usage aggregates to the frontend."""

from fastapi import APIRouter, Request

from app.services.usage_store import UsageStore

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def _get_usage_store(request: Request) -> UsageStore:
    return request.app.state.usage_store


@router.get("/summary")
def summary(request: Request) -> dict:
    """Return aggregate usage metrics (total cost, latency, blocked, reviews)."""
    return _get_usage_store(request).summary()


@router.get("/recent")
def recent(request: Request, limit: int = 20) -> list[dict]:
    """Return the most recent requests' usage metadata."""
    return _get_usage_store(request).recent(limit=limit)
