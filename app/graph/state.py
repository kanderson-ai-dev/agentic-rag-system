"""Shared state for the Self-RAG LangGraph workflow."""

from typing import Annotated

from langchain_core.documents import Document
from typing_extensions import TypedDict


def merge_node_latencies(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> dict[str, float]:
    """Accumulate per-node latency measurements across the graph run.

    Each node contributes one ``{node_name: milliseconds}`` update; the
    reducer unions them so the final state carries every node's timing —
    persisted by the usage store for the per-node performance breakdown.
    """
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict):
    """State threaded through every node of the Self-RAG graph.

    Attributes:
        question: The current user question (may be rewritten by
            `transform_query_node` across correction loop iterations).
        generation: The final generated answer, populated by `generate_node`
            or directly by a human reviewer via the `override` decision.
        documents: The documents currently held as retrieval context.
        web_search_needed: Set by `grade_documents_node` when the retrieved
            documents are not relevant enough to answer the question. Despite
            the name (kept for compatibility with the classic Self-RAG/CRAG
            terminology), in this implementation it drives the
            transform-query / human-review escalation path rather than an
            actual web search.
        retry_count: Number of times the query has been rewritten and
            re-retrieved. Bounded by `Settings.max_retries` to guarantee the
            correction loop always terminates.
        human_decision: The decision returned by a human reviewer
            ("approve", "retry" or "override") after a human-in-the-loop
            escalation triggered by `human_review_node`. `None` while no
            escalation has occurred yet.
        blocked: Set by the input guardrail node when the question is flagged
            as a prompt injection (or sanitizes to empty). The graph then
            short-circuits to `error_output` without touching retrieval/LLM.
        output_flagged: Set by the output guardrail node when the generated
            answer leaks the system prompt or reflects injected instructions.
        speculative_generation: Answer generated in parallel with document
            grading, using the full unfiltered retrieved context. Reused by
            `generate_node` only when `speculative_context` and
            `speculative_question` match what generation would use now —
            i.e. when the grader kept every retrieved document.
        speculative_context: The serialized context the speculative answer
            was generated from.
        speculative_question: The question the speculative answer was
            generated for.
        prompt_versions: The prompt versions (``{prompt_name: version}``) the
            thread started with, pinned by the guardrail node on first pass.
            LLM calls resolve those exact versions, so a thread paused for
            human review keeps its original prompts even if a promotion lands
            while it waits.
        node_latencies: Per-node wall-clock milliseconds, accumulated by the
            ``timed_node`` decorator via ``merge_node_latencies``. Feeds the
            per-node p50/p95 breakdown on the performance dashboard.
        budget_exhausted: Set by the budget-gate node when the recorded
            UTC-day spend has reached the configured daily cap. The request
            then short-circuits to END before any LLM call and the API
            translates the flag into a friendly HTTP 429. Runs *after* the
            input guardrail, so injection attempts still short-circuit for
            free even when the budget is spent.
    """

    question: str
    generation: str
    documents: list[Document]
    web_search_needed: bool
    retry_count: int
    human_decision: str | None
    blocked: bool
    output_flagged: bool
    speculative_generation: str
    speculative_context: str
    speculative_question: str
    prompt_versions: dict[str, int]
    node_latencies: Annotated[dict[str, float], merge_node_latencies]
    budget_exhausted: bool
