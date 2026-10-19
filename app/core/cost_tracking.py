"""Token usage tracking and cost calculation."""

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def calculate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_price: float = 0.15,
    output_price: float = 0.60,
) -> float:
    """Calculate the USD cost of a request given per-1M-token prices.

    The defaults match `gpt-4o-mini` ($0.15/1M input, $0.60/1M output).
    """
    return (prompt_tokens / 1_000_000 * input_price) + (
        completion_tokens / 1_000_000 * output_price
    )


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Accumulate prompt/completion tokens across all LLM calls in a request.

    One instance is registered on the shared chat model, so its own counters
    would mix concurrent requests. ``on_llm_end`` therefore records into the
    accumulator named by :func:`track_request_usage` when a request scope is
    active, and falls back to ``self`` everywhere else — e.g. evaluation
    harnesses that intentionally hold one handler across a whole run.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_llm_end(self, response: Any, **_: Any) -> None:
        """Capture token usage from an LLM response's ``llm_output``."""
        usage = getattr(response, "llm_output", None) or {}
        token_usage = usage.get("token_usage", {}) if isinstance(usage, dict) else {}
        target = _active_usage.get() or self
        target.prompt_tokens += int(token_usage.get("prompt_tokens", 0))
        target.completion_tokens += int(token_usage.get("completion_tokens", 0))


_active_usage: ContextVar[TokenUsageCallbackHandler | None] = ContextVar(
    "active_token_usage", default=None
)


@contextlib.contextmanager
def track_request_usage() -> Iterator[TokenUsageCallbackHandler]:
    """Per-request token accumulator, isolated from concurrent requests.

    The calling thread's context is copied into the nodes' thread pools (see
    ``nodes.py``) so LLM calls on worker threads land on this request's
    counters too.
    """
    handler = TokenUsageCallbackHandler()
    token = _active_usage.set(handler)
    try:
        yield handler
    finally:
        _active_usage.reset(token)
