"""FastAPI dependency providers.

The compiled graph and its checkpointer are expensive to build (they open a
SQLite connection and, in production, an OpenAI client) so they are created
once during the application lifespan and exposed here as a thin dependency
that reads them back from `app.state`.
"""

from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.graph.state import CompiledStateGraph

from app.core.config import Settings, get_settings
from app.core.security import decode_access_token
from app.services.ab_test_jobs import AbTestJobManager, AbTestRunner, default_ab_test_runner
from app.services.prompt_registry import PromptRegistry

_bearer = HTTPBearer(auto_error=False)

_ab_test_manager: AbTestJobManager | None = None


def get_session_id(x_session_id: str | None = Header(default=None)) -> str | None:
    """Return the caller-supplied `X-Session-Id`, or `None` if absent.

    The frontend generates a random id per browser session (sessionStorage)
    so the dashboard can be scoped to "what happened in this visit" instead
    of the service's lifetime totals. Callers that don't send the header
    (curl, other API clients) are simply not scoped to any session.
    """
    return x_session_id


def get_graph(request: Request) -> CompiledStateGraph[Any, None, Any, Any]:
    """Return the compiled Self-RAG graph stored on the application state.

    Raises a 503 if the graph could not be initialized (e.g. missing
    `OPENAI_API_KEY`), instead of letting requests fail with an
    unhandled `AttributeError`.
    """
    graph: CompiledStateGraph[Any, None, Any, Any] | None = getattr(
        request.app.state, "graph", None
    )
    if graph is None:
        error = getattr(request.app.state, "graph_error", "graph not initialized")
        raise HTTPException(status_code=503, detail=f"Service not ready: {error}")
    return graph


def verify_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any] | None:
    """Require a valid Bearer JWT when authentication is configured.

    When `JWT_SECRET_KEY` is not configured the service remains open for local
    quickstart (returns `None` without checking). Otherwise a missing or invalid
    token raises 401.
    """
    if not settings.jwt_secret_key:
        return None
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return decode_access_token(settings, credentials.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


def require_prompt_write(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> str:
    """Actor allowed to mutate the prompt registry, else 401/403.

    Registry writes are a prompt-tampering surface (OWASP LLM01/LLM03/LLM07):
    a valid JWT is required whenever auth is configured, and without auth
    (open quickstart) writes are refused unless the operator explicitly sets
    `PROMPT_REGISTRY_WRITE_ENABLED=true`. Returns the actor recorded in the
    audit chain — the JWT subject, or "local" in the open+enabled case.
    """
    if settings.jwt_secret_key:
        if credentials is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            payload = decode_access_token(settings, credentials.credentials)
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=401, detail="Invalid or expired token"
            ) from exc
        return str(payload.get("sub", "unknown"))
    if not settings.prompt_registry_write_enabled:
        raise HTTPException(
            status_code=403, detail="Prompt registry writes are disabled"
        )
    return "local"


def get_prompt_registry() -> PromptRegistry:
    """Return the process-wide prompt registry (see `app.graph.prompts`)."""
    from app.graph.prompts import get_registry

    return get_registry()


def get_ab_test_manager() -> AbTestJobManager:
    """Return the process-wide single-flight A/B job manager."""
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = AbTestJobManager()
    return _ab_test_manager


def get_ab_test_runner() -> AbTestRunner:
    """Return the evaluation engine the A/B job delegates to."""
    return default_ab_test_runner
