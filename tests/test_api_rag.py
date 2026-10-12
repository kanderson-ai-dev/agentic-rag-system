"""API-level tests for the rag query and human review endpoints.

The compiled graph dependency is overridden with a fully-stubbed graph, so
these tests never require real credentials.
"""

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.v1.schemas import QueryResponse
from app.core.config import get_settings
from app.core.dependencies import get_graph
from app.graph.prompts import REFUSAL_TEXT
from app.main import app
from tests.conftest import IRRELEVANT_DOC, RELEVANT_DOC, build_test_graph


def _client_with_graph(graph) -> TestClient:
    app.dependency_overrides[get_graph] = lambda: graph
    return TestClient(app)


def test_query_returns_completed_answer_for_relevant_documents() -> None:
    graph = build_test_graph([RELEVANT_DOC])
    with _client_with_graph(graph) as client:
        response = client.post("/api/v1/rag/query", json={"question": "what is langgraph?"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["answer"].startswith("Answer based on:")
    assert body["sources"] == ["langgraph-overview"]
    assert body["thread_id"]


def test_query_returns_interrupted_when_retries_are_exhausted() -> None:
    graph = build_test_graph([IRRELEVANT_DOC], max_retries=1)
    with _client_with_graph(graph) as client:
        response = client.post("/api/v1/rag/query", json={"question": "what is langgraph?"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "interrupted"
    assert body["interrupt"]["reason"] == "max_retries_exhausted"


def test_review_resumes_an_interrupted_thread_with_override() -> None:
    graph = build_test_graph([IRRELEVANT_DOC], max_retries=1)
    with _client_with_graph(graph) as client:
        query_response = client.post(
            "/api/v1/rag/query", json={"question": "what is langgraph?"}
        ).json()
        thread_id = query_response["thread_id"]

        review_response = client.post(
            f"/api/v1/rag/query/{thread_id}/review",
            json={"decision": "override", "override_answer": "manual answer"},
        )

    app.dependency_overrides.clear()
    assert review_response.status_code == 200
    body = review_response.json()
    assert body["status"] == "completed"
    assert body["answer"] == "manual answer"


def test_review_rejects_unknown_thread_id() -> None:
    graph = build_test_graph([RELEVANT_DOC])
    with _client_with_graph(graph) as client:
        response = client.post(
            "/api/v1/rag/query/does-not-exist/review",
            json={"decision": "approve"},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 409


def test_query_returns_service_unavailable_when_graph_not_initialized() -> None:
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        app.state.graph = None
        app.state.graph_error = "missing credentials"
        response = client.post("/api/v1/rag/query", json={"question": "hi"})

    assert response.status_code == 503


def test_query_returns_200_when_retrieval_backend_raises() -> None:
    """A dead retrieval backend must degrade to the HITL path, never a 500."""

    class _UnavailableBackend:
        def invoke(self, question: str) -> list:
            from neo4j.exceptions import ServiceUnavailable

            raise ServiceUnavailable("Failed to DNS resolve address")

    graph = build_test_graph(retriever=_UnavailableBackend(), max_retries=0)
    with _client_with_graph(graph) as client:
        response = client.post("/api/v1/rag/query", json={"question": "what is langgraph?"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["status"] == "interrupted"


def test_query_rejects_over_length_question_with_422() -> None:
    max_chars = get_settings().max_question_chars
    graph = build_test_graph([RELEVANT_DOC])
    with _client_with_graph(graph) as client:
        response = client.post(
            "/api/v1/rag/query", json={"question": "x" * (max_chars + 1)}
        )

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_query_accepts_question_at_length_cap() -> None:
    max_chars = get_settings().max_question_chars
    graph = build_test_graph([RELEVANT_DOC])
    with _client_with_graph(graph) as client:
        response = client.post("/api/v1/rag/query", json={"question": "x" * max_chars})

    app.dependency_overrides.clear()
    assert response.status_code == 200


def _result(answer: str, sources: list[str]) -> dict:
    return {
        "generation": answer,
        "documents": [
            Document(page_content=f"chunk {index}", metadata={"source": source})
            for index, source in enumerate(sources)
        ],
    }


def test_response_dedupes_sources_preserving_order() -> None:
    result = _result("an answer", ["a-doc", "a-doc", "graph", "b-doc", "graph"])

    response = QueryResponse.from_graph_result(result, "t")

    assert response.sources == ["a-doc", "graph", "b-doc"]


def test_response_returns_no_sources_for_refusal_answer() -> None:
    result = _result(REFUSAL_TEXT, ["self-rag-pattern", "graph"])

    response = QueryResponse.from_graph_result(result, "t")

    assert response.answer == REFUSAL_TEXT
    assert response.sources == []


def test_response_returns_no_sources_when_no_document_graded_relevant() -> None:
    response = QueryResponse.from_graph_result(_result("an answer", []), "t")

    assert response.sources == []
