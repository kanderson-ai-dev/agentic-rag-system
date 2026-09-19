"""Human-in-the-loop resolution endpoint.

Resumes a graph execution that was paused by `human_review_node` when the
Self-RAG correction loop exhausted its retries.
"""

from fastapi import APIRouter, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.api.v1.schemas import HumanReviewRequest, QueryResponse
from app.core.dependencies import get_graph

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query/{thread_id}/review", response_model=QueryResponse)
def review(
    thread_id: str,
    payload: HumanReviewRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> QueryResponse:
    """Resume an interrupted graph run with a human's decision."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail=f"No pending human review found for thread_id '{thread_id}'.",
        )

    result = graph.invoke(Command(resume=payload.model_dump()), config=config)
    return QueryResponse.from_graph_result(result, thread_id)
