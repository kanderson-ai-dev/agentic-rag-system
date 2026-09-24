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

# Per-backend ceiling on retrieval time. A backend that exceeds it
# contributes an empty result set instead of stalling the request.
_BACKEND_TIMEOUT_SECONDS = 8.0


def _scores(grading_result: Any) -> list[str]:
    """Extract the ordered 'yes'/'no' verdicts from a batched grader result.

    Accepts either a pydantic model with a `scores` attribute (the real
    structured-output chain) or a plain dict (convenient for tests).
    """
    if hasattr(grading_result, "scores"):
        return list(cast(list[str], grading_result.scores))
    return list(cast(list[str], grading_result["scores"]))


def _pad_scores(scores: list[str], expected: int) -> list[str]:
    """Align grader verdicts to the document count.

    Structured output can occasionally return fewer verdicts than documents;
    missing ones default to "yes" — keeping an extra document only adds
    context noise, while dropping a relevant one can trigger a needless
    correction loop.
    """
    return (scores + ["yes"] * expected)[:expected]


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    """Drop duplicate contents, keeping first occurrence (vector before graph).

    The two retrieval backends can surface the same underlying text; grading
    a duplicate twice wastes an LLM verdict and inflates the context.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for document in documents:
        if document.page_content not in seen:
            seen.add(document.page_content)
            unique.append(document)
    return unique


def _result_or_empty_on_timeout(future: Any) -> list[Document]:
    """Return a backend's results, or [] when it exceeds the timeout.

    A slow backend (e.g. a sleeping Neo4j Aura instance) degrades to an empty
    contribution instead of stalling the whole pipeline; genuine errors still
    propagate so a real outage surfaces as a request error, not silently
    empty context.
    """
    try:
        return list(future.result(timeout=_BACKEND_TIMEOUT_SECONDS))
    except TimeoutError:
        return []


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
        # Not a `with` block: on timeout the loser keeps running in the
        # background while the winner's results proceed (shutdown(wait=False)).
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            vector_future = executor.submit(retriever.invoke, question)
            graph_future = executor.submit(graph_search_fn, question)
            documents = _result_or_empty_on_timeout(vector_future)
            documents.extend(_result_or_empty_on_timeout(graph_future))
        finally:
            executor.shutdown(wait=False)
        return {"documents": _dedupe_documents(documents)}

    return retrieve_node


def make_grade_documents_node(
    grader_chain: Invokable,
    speculative_generation_chain: Invokable | None = None,
) -> NodeFn:
    """Filter out documents that are not relevant to the question.

    All documents are graded in a single batched LLM call — one round trip
    regardless of document count, which is the main lever on end-to-end
    latency for questions that retrieve several documents.

    When ``speculative_generation_chain`` is provided, a candidate answer is
    generated concurrently with grading over the full (unfiltered) context.
    If grading keeps every document, that answer is identical to what
    ``generate_node`` would produce, so it is reused — saving a full LLM
    round trip on the happy path. If grading drops documents, the candidate
    is discarded and generation reruns on the filtered set (a small wasted
    generation cost).
    """

    @timed_node("grade_documents")
    def grade_documents_node(state: GraphState) -> dict[str, Any]:
        documents = state["documents"]
        question = state["question"]
        if not documents:
            return {
                "documents": [],
                "web_search_needed": True,
                "speculative_generation": "",
                "speculative_context": "",
                "speculative_question": "",
            }

        serialized = "\n\n".join(
            f"[{index}] {document.page_content}"
            for index, document in enumerate(documents, start=1)
        )
        context = "\n\n".join(document.page_content for document in documents)

        updates: dict[str, Any] = {}
        if speculative_generation_chain is None:
            scores = _scores(
                grader_chain.invoke({"question": question, "documents": serialized})
            )
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                scores_future = executor.submit(
                    grader_chain.invoke, {"question": question, "documents": serialized}
                )
                generation_future = executor.submit(
                    speculative_generation_chain.invoke,
                    {"context": context, "question": question},
                )
                scores = _scores(scores_future.result())
                updates["speculative_generation"] = generation_future.result()
                updates["speculative_context"] = context
                updates["speculative_question"] = question

        relevant_documents = [
            doc
            for doc, score in zip(documents, _pad_scores(scores, len(documents)), strict=True)
            if score == "yes"
        ]
        updates["documents"] = relevant_documents
        updates["web_search_needed"] = len(relevant_documents) == 0
        return updates

    return grade_documents_node


def make_generate_node(generation_chain: Invokable) -> NodeFn:
    """Generate the final answer from the currently relevant documents.

    Reuses the speculative answer produced during grading when the surviving
    context and the current question are identical to what it was generated
    from — i.e. when the grader kept every retrieved document (the common
    case). Otherwise it regenerates from the filtered set.
    """

    @timed_node("generate")
    def generate_node(state: GraphState) -> dict[str, Any]:
        context = "\n\n".join(document.page_content for document in state["documents"])
        speculative = state.get("speculative_generation", "")
        if (
            speculative
            and state.get("speculative_context") == context
            and state.get("speculative_question") == state["question"]
        ):
            return {"generation": speculative}
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
