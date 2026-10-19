"""Shared per-request usage accounting for the rag routes.

The query and human-review endpoints emit identical telemetry; keeping the
logic here means the resume path cannot drift from the query path again.
"""

from typing import Any

from fastapi import Request

from app.core.config import get_settings
from app.core.cost_tracking import TokenUsageCallbackHandler, calculate_cost_usd
from app.core.metrics import (
    agent_blocked_requests_total,
    agent_human_review_total,
    agent_llm_cost_usd_total,
    agent_output_flagged_total,
)


def record_request_usage(
    request: Request,
    *,
    thread_id: str,
    result: dict[str, Any],
    latency_ms: int,
    session_id: str | None,
    token_handler: TokenUsageCallbackHandler,
) -> float:
    """Persist usage metadata for a finished graph run and bump counters.

    Returns the request's cost in USD.
    """
    settings = get_settings()
    prompt_tokens = token_handler.prompt_tokens
    completion_tokens = token_handler.completion_tokens
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
            prompt_versions=result.get("prompt_versions") or None,
            node_latencies=result.get("node_latencies") or None,
        )

    if result.get("blocked"):
        agent_blocked_requests_total.inc()
    if result.get("output_flagged"):
        agent_output_flagged_total.inc()
    if result.get("__interrupt__"):
        agent_human_review_total.inc()
    if cost_usd > 0:
        agent_llm_cost_usd_total.inc(cost_usd)
    return cost_usd
