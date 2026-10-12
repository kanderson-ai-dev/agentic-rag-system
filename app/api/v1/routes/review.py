"""Human-in-the-loop resolution endpoint.

Resumes a graph execution that was paused by `human_review_node` when the
Self-RAG correction loop exhausted its retries.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.api.v1.schemas import HumanReviewRequest, QueryResponse
from app.core import tracing
from app.core.config import get_settings
from app.core.dependencies import get_graph, verify_jwt

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query/{thread_id}/review", response_model=QueryResponse)
def review(
    thread_id: str,
    payload: HumanReviewRequest,
    response: Response,
    graph: CompiledStateGraph[Any, None, Any, Any] = Depends(get_graph),
    _: dict[str, Any] | None = Depends(verify_jwt),
) -> QueryResponse:
    """Resume an interrupted graph run with a human's decision."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail=f"No pending human review found for thread_id '{thread_id}'.",
        )

    settings = get_settings()
    with tracing.trace_run(
        settings,
        name="rag.review",
        metadata={"thread_id": thread_id},
        tags=["api", "hitl"],
    ) as trace_id:
        result = graph.invoke(
            Command(resume=payload.model_dump()),
            config={
                "configurable": {"thread_id": thread_id},
                "callbacks": tracing.build_callbacks(settings),
            },
        )
    if trace_id:
        response.headers["X-Trace-Id"] = trace_id
    return QueryResponse.from_graph_result(result, thread_id)
