"""Token usage tracking and cost calculation."""

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
    """Accumulate prompt/completion tokens across all LLM calls in a request."""

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
        self.prompt_tokens += int(token_usage.get("prompt_tokens", 0))
        self.completion_tokens += int(token_usage.get("completion_tokens", 0))
