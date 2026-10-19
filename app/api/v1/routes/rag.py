"""Self-RAG query endpoint."""

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from langgraph.graph.state import CompiledStateGraph

from app.api.v1.schemas import QueryRequest, QueryResponse
from app.api.v1.usage import record_request_usage, reset_token_handler
from app.core import tracing
from app.core.config import get_settings
from app.core.dependencies import (
    BUDGET_EXCEEDED_DETAIL,
    get_graph,
    get_session_id,
    verify_jwt,
)
from app.core.rate_limit import limiter, request_rate_limit
from app.graph.graph import initial_state

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit(request_rate_limit)
def query(
    payload: QueryRequest,
    request: Request,
    response: Response,
    graph: CompiledStateGraph[Any, None, Any, Any] = Depends(get_graph),
    _: dict[str, Any] | None = Depends(verify_jwt),
    session_id: str | None = Depends(get_session_id),
) -> QueryResponse:
    """Invoke the compiled Self-RAG graph for a question.

    Returns `status="completed"` with the final answer, or
    `status="interrupted"` when the correction loop escalated to a human
    reviewer; resolve it via `POST /api/v1/rag/query/{thread_id}/review`.
    """
    start = time.perf_counter()  # whole request, not just the graph call
    thread_id = payload.thread_id or str(uuid.uuid4())
    settings = get_settings()

    reset_token_handler(request)

    with tracing.trace_run(
        settings,
        name="rag.query",
        session_id=session_id,
        metadata={"thread_id": thread_id},
        tags=["api", "rag"],
    ) as trace_id:
        result = graph.invoke(
            initial_state(payload.question),
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
    )

    if result.get("budget_exhausted"):
        raise HTTPException(status_code=429, detail=BUDGET_EXCEEDED_DETAIL)

    return QueryResponse.from_graph_result(result, thread_id)
