"""Human-in-the-loop resolution endpoint.

Resumes a graph execution that was paused by `human_review_node` when the
Self-RAG correction loop exhausted its retries.
"""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.api.v1.schemas import HumanReviewRequest, QueryResponse
from app.api.v1.usage import record_request_usage
from app.core import tracing
from app.core.config import get_settings
from app.core.cost_tracking import track_request_usage
from app.core.dependencies import (
    enforce_daily_budget,
    get_graph,
    get_session_id,
    verify_jwt,
)
from app.core.rate_limit import limiter, request_rate_limit
from app.graph.guardrails import detect_prompt_injection, sanitize_input

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query/{thread_id}/review", response_model=QueryResponse)
@limiter.limit(request_rate_limit)
def review(
    thread_id: Annotated[
        str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    ],
    payload: HumanReviewRequest,
    request: Request,
    response: Response,
    graph: CompiledStateGraph[Any, None, Any, Any] = Depends(get_graph),
    _: dict[str, Any] | None = Depends(verify_jwt),
    session_id: str | None = Depends(get_session_id),
) -> QueryResponse:
    """Resume an interrupted graph run with a human's decision."""
    # The `retry` decision re-enters the graph past the input guardrail, so
    # the revised question is screened here instead — before any graph call.
    resume = payload.model_dump()
    if payload.decision == "retry":
        revised = sanitize_input(payload.revised_question or "")
        if not revised or detect_prompt_injection(revised) is not None:
            raise HTTPException(status_code=422, detail="Revised question rejected.")
        resume["revised_question"] = revised

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
    start = time.perf_counter()
    with (
        track_request_usage() as token_handler,
        tracing.trace_run(
            settings,
            name="rag.review",
            metadata={"thread_id": thread_id},
            tags=["api", "hitl"],
        ) as trace_id,
    ):
        result = graph.invoke(
            Command(resume=resume),
            config={
                "configurable": {"thread_id": thread_id},
                "callbacks": tracing.build_callbacks(settings),
            },
        )
    if trace_id:
        response.headers["X-Trace-Id"] = trace_id
    latency_ms = int((time.perf_counter() - start) * 1000)
    record_request_usage(
        request,
        thread_id=thread_id,
        result=result,
        latency_ms=latency_ms,
        session_id=session_id,
        token_handler=token_handler,
    )
    return QueryResponse.from_graph_result(result, thread_id)
