"""Human-in-the-loop resolution endpoint.

Resumes a graph execution that was paused by `human_review_node` when the
Self-RAG correction loop exhausted its retries.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.api.v1.schemas import HumanReviewRequest, QueryResponse
from app.core import tracing
from app.core.config import get_settings
from app.core.dependencies import enforce_daily_budget, get_graph, verify_jwt
from app.core.metrics import agent_output_flagged_total
from app.core.rate_limit import limiter, request_rate_limit

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query/{thread_id}/review", response_model=QueryResponse)
@limiter.limit(request_rate_limit)
def review(
    thread_id: str,
    payload: HumanReviewRequest,
    request: Request,
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

    # The resume re-enters the graph past the guardrail/budget_gate pair, so
    # the daily-budget check runs here instead — still before any LLM call,
    # and only for a review that would actually resume (not a 409).
    enforce_daily_budget(request)

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
    if result.get("output_flagged"):
        agent_output_flagged_total.inc()
    return QueryResponse.from_graph_result(result, thread_id)
