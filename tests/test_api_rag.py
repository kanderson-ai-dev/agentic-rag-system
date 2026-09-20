"""API-level tests for the rag query and human review endpoints.

The compiled graph dependency is overridden with a fully-stubbed graph, so
these tests never require real credentials.
"""

from fastapi.testclient import TestClient

from app.core.dependencies import get_graph
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
