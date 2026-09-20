"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.sqlite import SqliteSaver
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_v1_router
from app.api.v1.routes.health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import limiter
from app.graph.graph import build_default_graph

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.configure_langsmith_env()

    checkpoint_path = Path(settings.checkpoint_db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    app.state.graph = None
    app.state.graph_error = None

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        try:
            app.state.graph = build_default_graph(settings, checkpointer)
            logger.info("Self-RAG graph initialized successfully")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't crash startup
            app.state.graph_error = str(exc)
            logger.error("Failed to initialize Self-RAG graph: %s", exc)

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
    app.include_router(health_router)
    app.include_router(api_v1_router)
    return app


app = create_app()
