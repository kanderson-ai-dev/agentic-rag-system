"""Self-RAG query endpoint."""

import uuid

from fastapi import APIRouter, Depends
from langgraph.graph.state import CompiledStateGraph

from app.api.v1.schemas import QueryRequest, QueryResponse
from app.core.dependencies import get_graph
from app.graph.graph import initial_state

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> QueryResponse:
    """Invoke the compiled Self-RAG graph for a question.

    Returns `status="completed"` with the final answer, or
    `status="interrupted"` when the correction loop escalated to a human
    reviewer; resolve it via `POST /api/v1/rag/query/{thread_id}/review`.
    """
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state(payload.question), config=config)
    return QueryResponse.from_graph_result(result, thread_id)
