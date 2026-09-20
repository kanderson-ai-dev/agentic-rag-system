"""Prometheus metrics for node latency, cost and request outcomes."""

import functools
import time
from collections.abc import Callable
from typing import Any

from prometheus_client import Counter, Histogram

agent_node_latency_seconds = Histogram(
    "agent_node_latency_seconds",
    "Latency of each graph node in seconds.",
    labelnames=["node"],
)
agent_llm_cost_usd_total = Counter(
    "agent_llm_cost_usd_total",
    "Total LLM cost in USD across all requests.",
)
agent_blocked_requests_total = Counter(
    "agent_blocked_requests_total",
    "Total requests blocked by the input guardrail.",
)
agent_human_review_total = Counter(
    "agent_human_review_total",
    "Total requests escalated to human review.",
)


def timed_node(node_name: str) -> Callable:
    """Decorate a graph node to record its latency in the histogram."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            start = time.perf_counter()
            try:
                return func(state)
            finally:
                agent_node_latency_seconds.labels(node=node_name).observe(
                    time.perf_counter() - start
                )

        return wrapper

    return decorator
