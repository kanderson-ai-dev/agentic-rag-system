"""Node factories for the Self-RAG graph.

Every `make_*_node` factory closes over its dependencies (a retriever or an
LLM chain) and returns a plain function of `GraphState -> dict`. Dependencies
only need to expose an `.invoke()` method, so unit tests can pass in simple
stub objects instead of real LangChain runnables or network calls.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, cast

from langchain_core.documents import Document
from langgraph.types import interrupt

from app.core.metrics import timed_node
from app.graph.guardrails import detect_prompt_injection, sanitize_input, screen_output
from app.graph.state import GraphState


class Invokable(Protocol):
    def invoke(self, input: Any) -> Any: ...  # noqa: A002


NodeFn = Callable[[GraphState], dict[str, Any]]
"""Signature shared by every compiled graph node produced here."""


def _binary_score(grading_result: Any) -> str:
    """Extract the 'yes'/'no' score from a grader chain result.

    Accepts either a pydantic model with a `binary_score` attribute (the
    real structured-output chain) or a plain dict (convenient for tests).
    """
    if hasattr(grading_result, "binary_score"):
        return cast(str, grading_result.binary_score)
    return cast(str, grading_result["binary_score"])


def make_hybrid_retrieve_node(
    retriever: Invokable, graph_search_fn: Callable[[str], list[Document]]
) -> NodeFn:
    """Retrieve documents from both the vector store and the graph store.

    ``graph_search_fn`` is a callable `(term: str) -> list[Document]` provided
    by the graph store service (Neo4j or NetworkX). Combined results are merged
    into ``documents`` with per-document ``source``/``backend`` metadata.

    The vector and graph queries are independent I/O calls to different
    backends, so they run concurrently in a thread pool instead of back to
    back — this roughly halves retrieval latency (dominated by network round
    trips, not CPU), which matters most when the graph backend is a Neo4j
    Aura free-tier instance waking up from sleep.
    """

    @timed_node("retrieve")
    def retrieve_node(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(retriever.invoke, question)
            graph_future = executor.submit(graph_search_fn, question)
            documents: list[Document] = list(vector_future.result())
            documents.extend(graph_future.result())
        return {"documents": documents}

    return retrieve_node


def make_grade_documents_node(grader_chain: Invokable) -> NodeFn:
    """Filter out documents that are not relevant to the question.

    Each document is graded with its own LLM call; grading them concurrently
    (instead of one-by-one) turns N sequential round trips into roughly the
    latency of a single call, which is the main lever on end-to-end latency
    for questions that retrieve several documents.
    """

    @timed_node("grade_documents")
    def grade_documents_node(state: GraphState) -> dict[str, Any]:
        documents = state["documents"]
        question = state["question"]
        if not documents:
            return {"documents": [], "web_search_needed": True}

        def _grade(document: Document) -> str:
            grading_input = {"question": question, "document": document.page_content}
            return _binary_score(grader_chain.invoke(grading_input))

        with ThreadPoolExecutor(max_workers=min(len(documents), 8)) as executor:
            scores = list(executor.map(_grade, documents))

        relevant_documents = [
            doc for doc, score in zip(documents, scores, strict=True) if score == "yes"
        ]
        return {
            "documents": relevant_documents,
            "web_search_needed": len(relevant_documents) == 0,
        }

    return grade_documents_node


def make_generate_node(generation_chain: Invokable) -> NodeFn:
    """Generate the final answer from the currently relevant documents."""

    @timed_node("generate")
    def generate_node(state: GraphState) -> dict[str, Any]:
        context = "\n\n".join(document.page_content for document in state["documents"])
        answer = generation_chain.invoke({"context": context, "question": state["question"]})
        return {"generation": answer}

    return generate_node


def make_transform_query_node(rewriter_chain: Invokable) -> NodeFn:
    """Rewrite the question to improve retrieval and count the attempt."""

    @timed_node("transform_query")
    def transform_query_node(state: GraphState) -> dict[str, Any]:
        rewritten_question = rewriter_chain.invoke({"question": state["question"]})
        return {
            "question": rewritten_question,
            "retry_count": state["retry_count"] + 1,
        }

    return transform_query_node


def make_human_review_node() -> NodeFn:
    """Escalate to a human reviewer when the correction loop is exhausted.

    Pauses graph execution via `interrupt()` and surfaces the current
    question, retry count and best-effort documents. The graph resumes when
    the caller invokes it again with `Command(resume=decision)`, where
    `decision` is a dict of the shape::

        {
            "decision": "approve" | "retry" | "override",
            "revised_question": str | None,   # required for "retry"
            "override_answer": str | None,    # required for "override"
        }
    """

    @timed_node("human_review")
    def human_review_node(state: GraphState) -> dict[str, Any]:
        decision_payload = interrupt(
            {
                "reason": "max_retries_exhausted",
                "question": state["question"],
                "retry_count": state["retry_count"],
                "best_documents": [document.page_content for document in state["documents"]],
            }
        )

        decision = decision_payload.get("decision", "approve")
        updates: dict[str, Any] = {"human_decision": decision}

        if decision == "override":
            updates["generation"] = decision_payload.get("override_answer", "")
        elif decision == "retry":
            updates["question"] = decision_payload.get("revised_question") or state["question"]
            updates["retry_count"] = 0
            updates["web_search_needed"] = False

        return updates

    return human_review_node


def make_guardrail_node() -> NodeFn:
    """Sanitize the input question and block it if a prompt injection is detected.

    This runs first in the graph so malicious input never reaches the retriever
    or the LLM (guardrail-first, OWASP LLM01/LLM04).
    """

    @timed_node("guardrail")
    def guardrail_node(state: GraphState) -> dict[str, Any]:
        question = sanitize_input(state["question"])
        blocked = not question or detect_prompt_injection(question) is not None
        return {"question": question, "blocked": blocked}

    return guardrail_node


def make_output_guardrail_node() -> NodeFn:
    """Screen the final generation for system-prompt leakage or reflected injection."""

    @timed_node("output_guardrail")
    def output_guardrail_node(state: GraphState) -> dict[str, Any]:
        flagged = screen_output(state["generation"]) is not None
        return {"output_flagged": flagged}

    return output_guardrail_node


def make_error_output_node() -> NodeFn:
    """Produce a generic refusal without touching retrieval or the LLM."""

    @timed_node("error_output")
    def error_output_node(state: GraphState) -> dict[str, Any]:
        return {"generation": "I'm sorry, but I can't help with that request."}

    return error_output_node
