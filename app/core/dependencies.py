"""FastAPI dependency providers.

The compiled graph and its checkpointer are expensive to build (they open a
SQLite connection and, in production, an OpenAI client) so they are created
once during the application lifespan and exposed here as a thin dependency
that reads them back from `app.state`.
"""

from fastapi import HTTPException, Request
from langgraph.graph.state import CompiledStateGraph


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the compiled Self-RAG graph stored on the application state.

    Raises a 503 if the graph could not be initialized (e.g. missing
    `OPENAI_API_KEY`), instead of letting requests fail with an
    unhandled `AttributeError`.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        error = getattr(request.app.state, "graph_error", "graph not initialized")
        raise HTTPException(status_code=503, detail=f"Service not ready: {error}")
    return graph
