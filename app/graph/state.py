"""Shared state for the Self-RAG LangGraph workflow."""

from langchain_core.documents import Document
from typing_extensions import TypedDict


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
    """

    question: str
    generation: str
    documents: list[Document]
    web_search_needed: bool
    retry_count: int
    human_decision: str | None
