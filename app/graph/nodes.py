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

from app.core.config import DEFAULT_MAX_QUESTION_CHARS
from app.core.logging import get_logger
from app.core.metrics import (
    agent_backend_errors_total,
    record_backend_result,
    timed_node,
)
from app.graph.guardrails import detect_prompt_injection, sanitize_input, screen_output
from app.graph.prompts import active_versions, prompt_hash
from app.graph.state import GraphState

logger = get_logger(__name__)


class Invokable(Protocol):
    def invoke(self, input: Any, config: Any | None = None) -> Any: ...  # noqa: A002


NodeFn = Callable[[GraphState], dict[str, Any]]
"""Signature shared by every compiled graph node produced here."""


def _llm_config(prompt_name: str, versions: dict[str, int]) -> dict[str, Any]:
    """Runnable config stamping prompt attribution on every LLM call.

    Propagates as LangSmith run metadata/tags (which prompt produced the
    output is always known) without changing the prompt variables.
    """
    metadata: dict[str, Any] = {
        "prompt_versions": versions,
        "prompt_version": versions.get(prompt_name),
        "stage": prompt_name,
    }
    content_hash = prompt_hash(prompt_name, versions.get(prompt_name))
    if content_hash:
        metadata["prompt_hash"] = content_hash
    return {
        "metadata": metadata,
        "tags": [f"prompt:{prompt_name}@{versions.get(prompt_name) or 'promoted'}"],
    }

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


def _result_or_empty_on_timeout(future: Any, backend: str) -> list[Document]:
    """Return a backend's results, or [] when it times out or fails.

    A slow backend (e.g. a sleeping Neo4j Aura instance) or an unreachable one
    degrades to an empty contribution instead of stalling or crashing the whole
    pipeline: retrieval backend failures must never surface as a request error.
    The failure is logged once per call, counted in
    `agent_backend_errors_total{backend}` and recorded for `GET /health`.
    """
    try:
        results = list(future.result(timeout=_BACKEND_TIMEOUT_SECONDS))
    except Exception as exc:  # noqa: BLE001 - a backend outage degrades, never 500s
        logger.warning(
            "retrieval_backend_failed",
            backend=backend,
            exception=type(exc).__name__,
        )
        agent_backend_errors_total.labels(backend=backend).inc()
        record_backend_result(backend, ok=False)
        return []
    record_backend_result(backend, ok=True)
    return results


def make_hybrid_retrieve_node(
    retriever: Invokable,
    graph_search_fn: Callable[[str], list[Document]],
    *,
    retrieval_mode: str = "hybrid",
    vector_backend: str = "vector",
    graph_backend: str = "graph",
) -> NodeFn:
    """Retrieve documents from the configured backend combination.

    ``graph_search_fn`` is a callable `(term: str) -> list[Document]` provided
    by the graph store service (Neo4j or NetworkX). Combined results are merged
    into ``documents`` with per-document ``source``/``backend`` metadata.

    ``retrieval_mode`` selects the arms: ``hybrid`` (both), ``vector_only`` or
    ``graph_only``. ``no_retrieval`` returns an empty set — the evaluation
    baseline drives that mode through a bare-LLM chain, so it should never
    reach this node in production.

    The vector and graph queries are independent I/O calls to different
    backends, so they run concurrently in a thread pool instead of back to
    back — this roughly halves retrieval latency (dominated by network round
    trips, not CPU), which matters most when the graph backend is a Neo4j
    Aura free-tier instance waking up from sleep.
    """
    use_vector = retrieval_mode in ("hybrid", "vector_only")
    use_graph = retrieval_mode in ("hybrid", "graph_only")

    @timed_node("retrieve")
    def retrieve_node(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        if not use_vector and not use_graph:
            return {"documents": []}
        # Not a `with` block: on timeout the loser keeps running in the
        # background while the winner's results proceed (shutdown(wait=False)).
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            # Submit both arms first so they run concurrently; only then wait.
            vector_future = (
                executor.submit(retriever.invoke, question) if use_vector else None
            )
            graph_future = (
                executor.submit(graph_search_fn, question) if use_graph else None
            )
            documents: list[Document] = []
            if vector_future is not None:
                documents = _result_or_empty_on_timeout(vector_future, vector_backend)
            if graph_future is not None:
                documents.extend(
                    _result_or_empty_on_timeout(graph_future, graph_backend)
                )
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

        versions = state.get("prompt_versions") or {}
        updates: dict[str, Any] = {}
        if speculative_generation_chain is None:
            scores = _scores(
                grader_chain.invoke(
                    {
                        "question": question,
                        "documents": serialized,
                        "prompt_version": versions.get("grade_documents"),
                    },
                    config=_llm_config("grade_documents", versions),
                )
            )
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                scores_future = executor.submit(
                    grader_chain.invoke,
                    {
                        "question": question,
                        "documents": serialized,
                        "prompt_version": versions.get("grade_documents"),
                    },
                    _llm_config("grade_documents", versions),
                )
                generation_future = executor.submit(
                    speculative_generation_chain.invoke,
                    {
                        "context": context,
                        "question": question,
                        "prompt_version": versions.get("generate_answer"),
                    },
                    _llm_config("generate_answer", versions),
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
        versions = state.get("prompt_versions") or {}
        answer = generation_chain.invoke(
            {
                "context": context,
                "question": state["question"],
                "prompt_version": versions.get("generate_answer"),
            },
            config=_llm_config("generate_answer", versions),
        )
        return {"generation": answer}

    return generate_node


def make_transform_query_node(rewriter_chain: Invokable) -> NodeFn:
    """Rewrite the question to improve retrieval and count the attempt."""

    @timed_node("transform_query")
    def transform_query_node(state: GraphState) -> dict[str, Any]:
        versions = state.get("prompt_versions") or {}
        rewritten_question = rewriter_chain.invoke(
            {
                "question": state["question"],
                "prompt_version": versions.get("transform_query"),
            },
            config=_llm_config("transform_query", versions),
        )
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


def make_guardrail_node(max_length: int = DEFAULT_MAX_QUESTION_CHARS) -> NodeFn:
    """Sanitize the input question and block it if a prompt injection is detected.

    This runs first in the graph so malicious input never reaches the retriever
    or the LLM (guardrail-first, OWASP LLM01/LLM04). Questions longer than
    ``max_length`` are blocked, matching the API's 422 behaviour so evaluation
    scripts that bypass the HTTP layer see identical treatment.
    """

    @timed_node("guardrail")
    def guardrail_node(state: GraphState) -> dict[str, Any]:
        raw_question = state["question"]
        question = sanitize_input(raw_question)
        blocked = (
            not question
            or len(raw_question) > max_length
            or detect_prompt_injection(question) is not None
        )
        # Pin the promoted prompt set for this thread: every LLM call resolves
        # these exact versions, so a paused HITL thread keeps the prompts it
        # started with even if a promotion lands while it waits.
        return {
            "question": question,
            "blocked": blocked,
            "prompt_versions": active_versions(),
        }

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
