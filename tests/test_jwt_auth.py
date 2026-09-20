"""Tests for JWT authentication and the login endpoint."""

import bcrypt
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.dependencies import get_graph
from app.core.rate_limit import limiter
from app.main import app
from tests.conftest import RELEVANT_DOC, build_test_graph

PASSWORD = "correct-horse-battery"
PASSWORD_HASH = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt()).decode()


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()
    app.dependency_overrides.clear()


def _settings(**overrides) -> Settings:
    base = {
        "jwt_secret_key": "test-secret-key-with-at-least-32-bytes-of-entropy-123456",
        "jwt_algorithm": "HS256",
        "jwt_expire_minutes": 30,
        "auth_username": "admin",
        "auth_password_hash": PASSWORD_HASH,
    }
    base.update(overrides)
    return Settings(**base)


def _client(settings: Settings) -> TestClient:
    graph = build_test_graph([RELEVANT_DOC])
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_graph] = lambda: graph
    return TestClient(app)


class TestLogin:
    def test_login_returns_token(self) -> None:
        with _client(_settings()) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": PASSWORD},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["access_token"]
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 1800

    def test_login_wrong_password_returns_generic_401(self) -> None:
        with _client(_settings()) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "wrong"},
            )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_unknown_username_returns_same_generic_401(self) -> None:
        with _client(_settings()) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"username": "nobody", "password": PASSWORD},
            )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_returns_503_when_not_configured(self) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": PASSWORD},
            )

        assert response.status_code == 503

    def test_login_is_rate_limited(self) -> None:
        with _client(_settings()) as client:
            statuses = [
                client.post(
                    "/api/v1/auth/login",
                    json={"username": "admin", "password": PASSWORD},
                ).status_code
                for _ in range(6)
            ]

        assert statuses[:5] == [200] * 5
        assert statuses[5] == 429


class TestProtectedEndpoints:
    def test_rag_query_requires_token_when_configured(self) -> None:
        with _client(_settings()) as client:
            response = client.post("/api/v1/rag/query", json={"question": "hi"})

        assert response.status_code == 401

    def test_rag_query_accepts_valid_token(self) -> None:
        with _client(_settings()) as client:
            token = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": PASSWORD},
            ).json()["access_token"]
            response = client.post(
                "/api/v1/rag/query",
                json={"question": "what is langgraph?"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200

    def test_rag_query_rejects_invalid_token(self) -> None:
        with _client(_settings()) as client:
            response = client.post(
                "/api/v1/rag/query",
                json={"question": "hi"},
                headers={"Authorization": "Bearer invalid-token"},
            )

        assert response.status_code == 401

    def test_rag_query_is_open_when_auth_not_configured(self) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.post("/api/v1/rag/query", json={"question": "hi"})

        assert response.status_code == 200
