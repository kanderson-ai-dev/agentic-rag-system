"""Node factories for the Self-RAG graph.

Every `make_*_node` factory closes over its dependencies (a retriever or an
LLM chain) and returns a plain function of `GraphState -> dict`. Dependencies
only need to expose an `.invoke()` method, so unit tests can pass in simple
stub objects instead of real LangChain runnables or network calls.
"""

from typing import Any, Protocol

from langchain_core.documents import Document
from langgraph.types import interrupt

from app.graph.state import GraphState


class Invokable(Protocol):
    def invoke(self, input: Any) -> Any: ...  # noqa: A002


def _binary_score(grading_result: Any) -> str:
    """Extract the 'yes'/'no' score from a grader chain result.

    Accepts either a pydantic model with a `binary_score` attribute (the
    real structured-output chain) or a plain dict (convenient for tests).
    """
    if hasattr(grading_result, "binary_score"):
        return grading_result.binary_score
    return grading_result["binary_score"]


def make_retrieve_node(retriever: Invokable):
    """Retrieve documents for the current question."""

    def retrieve_node(state: GraphState) -> dict:
        documents: list[Document] = retriever.invoke(state["question"])
        return {"documents": documents}

    return retrieve_node


def make_grade_documents_node(grader_chain: Invokable):
    """Filter out documents that are not relevant to the question."""

    def grade_documents_node(state: GraphState) -> dict:
        relevant_documents = []
        for document in state["documents"]:
            grading_input = {"question": state["question"], "document": document.page_content}
            if _binary_score(grader_chain.invoke(grading_input)) == "yes":
                relevant_documents.append(document)
        return {
            "documents": relevant_documents,
            "web_search_needed": len(relevant_documents) == 0,
        }

    return grade_documents_node


def make_generate_node(generation_chain: Invokable):
    """Generate the final answer from the currently relevant documents."""

    def generate_node(state: GraphState) -> dict:
        context = "\n\n".join(document.page_content for document in state["documents"])
        answer = generation_chain.invoke({"context": context, "question": state["question"]})
        return {"generation": answer}

    return generate_node


def make_transform_query_node(rewriter_chain: Invokable):
    """Rewrite the question to improve retrieval and count the attempt."""

    def transform_query_node(state: GraphState) -> dict:
        rewritten_question = rewriter_chain.invoke({"question": state["question"]})
        return {
            "question": rewritten_question,
            "retry_count": state["retry_count"] + 1,
        }

    return transform_query_node


def make_human_review_node():
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

    def human_review_node(state: GraphState) -> dict:
        decision_payload = interrupt(
            {
                "reason": "max_retries_exhausted",
                "question": state["question"],
                "retry_count": state["retry_count"],
                "best_documents": [document.page_content for document in state["documents"]],
            }
        )

        decision = decision_payload.get("decision", "approve")
        updates: dict = {"human_decision": decision}

        if decision == "override":
            updates["generation"] = decision_payload.get("override_answer", "")
        elif decision == "retry":
            updates["question"] = decision_payload.get("revised_question") or state["question"]
            updates["retry_count"] = 0
            updates["web_search_needed"] = False

        return updates

    return human_review_node
