"""FastAPI dependency providers.

The compiled graph and its checkpointer are expensive to build (they open a
SQLite connection and, in production, an OpenAI client) so they are created
once during the application lifespan and exposed here as a thin dependency
that reads them back from `app.state`.
"""

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the compiled Self-RAG graph stored on the application state."""
    return request.app.state.graph
