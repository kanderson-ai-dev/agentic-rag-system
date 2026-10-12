"""Prometheus metrics for node latency, cost and request outcomes."""

import functools
import threading
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from prometheus_client import Counter, Gauge, Histogram

P = ParamSpec("P")
R = TypeVar("R")

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
agent_backend_errors_total = Counter(
    "agent_backend_errors_total",
    "Total retrieval backend failures (timeouts and errors) by backend.",
    labelnames=["backend"],
)
agent_prompt_active_version = Gauge(
    "agent_prompt_active_version",
    "Currently promoted prompt version by prompt name.",
    labelnames=["prompt"],
)

# Last observed outcome per retrieval backend, written from the request path
# only. Read by `GET /health` to report `degraded` without probing backends
# (a probe against a paused Neo4j Aura instance would stall the probe itself).
_backend_ok: dict[str, bool] = {}
_backend_ok_lock = threading.Lock()


def record_backend_result(backend: str, ok: bool) -> None:
    """Record the latest request-path outcome for a retrieval backend."""
    with _backend_ok_lock:
        _backend_ok[backend] = ok


def backend_health() -> dict[str, bool]:
    """Return a snapshot of the last known outcome per backend (`True` = ok)."""
    with _backend_ok_lock:
        return dict(_backend_ok)


def reset_backend_health() -> None:
    """Clear all recorded backend outcomes (used by tests)."""
    with _backend_ok_lock:
        _backend_ok.clear()


def timed_node(node_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a graph node to record its latency in the histogram and in
    the graph state's ``node_latencies`` (persisted by the usage store so the
    dashboard can break end-to-end latency down per node)."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                agent_node_latency_seconds.labels(node=node_name).observe(
                    time.perf_counter() - start
                )
            if isinstance(result, dict):
                merged = dict(result.get("node_latencies") or {})
                merged[node_name] = (time.perf_counter() - start) * 1000.0
                return cast(R, {**result, "node_latencies": merged})
            return result

        return wrapper

    return decorator
