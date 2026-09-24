"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from app.api.v1.router import api_v1_router
from app.api.v1.routes.health import router as health_router
from app.core.config import get_settings
from app.core.cost_tracking import TokenUsageCallbackHandler
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import limiter
from app.graph.graph import build_default_graph
from app.services.usage_store import UsageStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.environment)
    settings.configure_langsmith_env()

    checkpoint_path = Path(settings.checkpoint_db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    app.state.graph = None
    app.state.graph_error = None
    app.state.usage_store = UsageStore(settings.usage_db_path)
    app.state.token_handler = TokenUsageCallbackHandler()

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        try:
            app.state.graph = build_default_graph(settings, checkpointer, app.state.token_handler)
            logger.info("Self-RAG graph initialized successfully")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't crash startup
            app.state.graph_error = str(exc)
            logger.error("Failed to initialize Self-RAG graph", error=str(exc))

        yield

    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Agentic RAG & Knowledge Systems - a Self-RAG microservice.",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    Instrumentator().instrument(app).expose(app)
    app.include_router(health_router)
    app.include_router(api_v1_router)

    @app.get("/console", include_in_schema=False)
    def console_redirect() -> RedirectResponse:
        """Send `/console` to `/console/` so the console index is served."""
        return RedirectResponse(url="/console/", status_code=308)

    # Two static surfaces, no build step:
    #   `/`        — minimalist public landing (frontend/landing/)
    #   `/console` — full operator console: chat + dashboard + HITL review
    #   `/js/*`    — ES modules shared by both surfaces (frontend/js/)
    # The catch-all `/` mount is registered last so the more specific prefixes
    # win. `/console` redirects to `/console/` (StaticFiles html=True).
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    app.mount(
        "/console",
        StaticFiles(directory=str(frontend_dir / "console"), html=True),
        name="console",
    )
    app.mount("/js", StaticFiles(directory=str(frontend_dir / "js")), name="shared-js")
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dir / "landing"), html=True),
        name="landing",
    )
    return app


app = create_app()
