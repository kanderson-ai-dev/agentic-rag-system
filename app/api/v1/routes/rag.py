"""Self-RAG query endpoint."""

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from langgraph.graph.state import CompiledStateGraph

from app.api.v1.schemas import QueryRequest, QueryResponse
from app.core.config import get_settings
from app.core.cost_tracking import calculate_cost_usd
from app.core.dependencies import get_graph, get_session_id, verify_jwt
from app.core.metrics import (
    agent_blocked_requests_total,
    agent_human_review_total,
    agent_llm_cost_usd_total,
)
from app.graph.graph import initial_state

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    request: Request,
    graph: CompiledStateGraph[Any, None, Any, Any] = Depends(get_graph),
    _: dict[str, Any] | None = Depends(verify_jwt),
    session_id: str | None = Depends(get_session_id),
) -> QueryResponse:
    """Invoke the compiled Self-RAG graph for a question.

    Returns `status="completed"` with the final answer, or
    `status="interrupted"` when the correction loop escalated to a human
    reviewer; resolve it via `POST /api/v1/rag/query/{thread_id}/review`.
    """
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    token_handler = getattr(request.app.state, "token_handler", None)
    if token_handler is not None:
        token_handler.prompt_tokens = 0
        token_handler.completion_tokens = 0

    start = time.perf_counter()
    result = graph.invoke(initial_state(payload.question), config=config)
    latency_ms = int((time.perf_counter() - start) * 1000)

    settings = get_settings()
    prompt_tokens = token_handler.prompt_tokens if token_handler else 0
    completion_tokens = token_handler.completion_tokens if token_handler else 0
    cost_usd = calculate_cost_usd(
        prompt_tokens,
        completion_tokens,
        settings.cost_input_price_per_1m,
        settings.cost_output_price_per_1m,
    )

    usage_store = getattr(request.app.state, "usage_store", None)
    if usage_store is not None:
        usage_store.record(
            thread_id=thread_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            blocked=bool(result.get("blocked", False)),
            escalated=bool(result.get("__interrupt__")),
            retry_count=result.get("retry_count", 0),
            session_id=session_id,
        )

    if result.get("blocked"):
        agent_blocked_requests_total.inc()
    if result.get("__interrupt__"):
        agent_human_review_total.inc()
    if cost_usd > 0:
        agent_llm_cost_usd_total.inc(cost_usd)

    return QueryResponse.from_graph_result(result, thread_id)
