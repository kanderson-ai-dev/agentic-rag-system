"""Tests that the human-in-the-loop review endpoint respects JWT auth."""

import bcrypt
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.dependencies import get_graph
from app.core.rate_limit import limiter
from app.main import app
from tests.conftest import IRRELEVANT_DOC, build_test_graph

PASSWORD = "correct-horse-battery"
PASSWORD_HASH = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt()).decode()


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    limiter.reset()
    app.dependency_overrides.clear()


def _settings(**overrides) -> Settings:
    base = {
        "jwt_secret_key": "test-secret-key-for-local-tests-only",
        "auth_username": "admin",
        "auth_password_hash": PASSWORD_HASH,
    }
    base.update(overrides)
    return Settings(**base)


def _client(settings: Settings) -> TestClient:
    graph = build_test_graph([IRRELEVANT_DOC], max_retries=1)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_graph] = lambda: graph
    return TestClient(app)


def _login(client: TestClient) -> str:
    return client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": PASSWORD},
    ).json()["access_token"]


def test_review_requires_token_when_configured() -> None:
    with _client(_settings()) as client:
        response = client.post(
            "/api/v1/rag/query/any-thread/review",
            json={"decision": "approve"},
        )

    assert response.status_code == 401


def test_review_accepts_valid_token() -> None:
    with _client(_settings()) as client:
        token = _login(client)
        query = client.post(
            "/api/v1/rag/query",
            json={"question": "what is langgraph?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert query.json()["status"] == "interrupted"
        thread_id = query.json()["thread_id"]

        response = client.post(
            f"/api/v1/rag/query/{thread_id}/review",
            json={"decision": "override", "override_answer": "manual answer"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "manual answer"


def test_review_is_open_when_auth_not_configured() -> None:
    with _client(_settings(jwt_secret_key=None)) as client:
        query = client.post("/api/v1/rag/query", json={"question": "what is langgraph?"})
        thread_id = query.json()["thread_id"]

        response = client.post(
            f"/api/v1/rag/query/{thread_id}/review",
            json={"decision": "override", "override_answer": "manual answer"},
        )

    assert response.status_code == 200
