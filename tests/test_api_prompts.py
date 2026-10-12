"""Tests for the prompt registry API (reads open, mutations JWT-gated)."""

import shutil
import time
from pathlib import Path
from typing import Any

import bcrypt
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.dependencies import get_ab_test_manager, get_ab_test_runner
from app.core.rate_limit import limiter
from app.graph.prompts import set_registry
from app.main import app
from app.services.ab_test_jobs import AbTestJobManager
from app.services.prompt_registry import PromptRegistry

_REPO_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

PASSWORD = "correct-horse-battery"
PASSWORD_HASH = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt()).decode()

_GOOD_SCORECARD = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.91,
    "context_precision": 0.80,
    "context_recall": 0.82,
    "llm_as_judge": 4.5,
}
_REGRESSED_SCORECARD = {**_GOOD_SCORECARD, "faithfulness": 0.50}

# Must declare exactly {context} + {question} — the generate_answer contract.
_NEW_GENERATE_SYSTEM = "You are a strict assistant. Context: {context}."


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()
    app.dependency_overrides.clear()


@pytest.fixture
def registry(tmp_path: Path) -> PromptRegistry:
    """Writable tmp-dir registry swapped in as the process singleton."""
    shutil.copytree(_REPO_PROMPTS, tmp_path / "prompts")
    instance = PromptRegistry(tmp_path / "prompts")
    set_registry(instance)
    return instance


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "jwt_secret_key": "test-secret-key-for-local-tests-only",
        "jwt_algorithm": "HS256",
        "jwt_expire_minutes": 30,
        "auth_username": "admin",
        "auth_password_hash": PASSWORD_HASH,
    }
    base.update(overrides)
    return Settings(**base)


def _client(
    settings: Settings,
    *,
    tmp_path: Path | None = None,
    runner: Any | None = None,
) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    if tmp_path is not None:
        manager = AbTestJobManager(tmp_path / "ab_results")
        app.dependency_overrides[get_ab_test_manager] = lambda: manager
    if runner is not None:
        app.dependency_overrides[get_ab_test_runner] = lambda: runner
    return TestClient(app)


def _token(client: TestClient) -> str:
    return client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": PASSWORD},
    ).json()["access_token"]


def _candidate_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "system": _NEW_GENERATE_SYSTEM,
        "human": "Question: {question}",
        "author": "tester",
        "description": "candidate for tests",
        "changelog": "test candidate",
    }
    body.update(overrides)
    return body


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/prompts/generate_answer/ab-test/{job_id}").json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"A/B job {job_id} did not finish in {timeout}s")


class TestReadEndpoints:
    def test_list_prompts_reports_active_versions_and_chain(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.get("/api/v1/prompts/")

        assert response.status_code == 200
        body = response.json()
        assert body["audit_chain_verified"] is True
        assert {p["name"] for p in body["prompts"]} == {
            "grade_documents",
            "generate_answer",
            "transform_query",
            "llm_judge",
        }
        assert all(p["active_version"] == 1 for p in body["prompts"])

    def test_list_versions_hides_bodies(self, registry: PromptRegistry) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.get("/api/v1/prompts/generate_answer")

        assert response.status_code == 200
        body = response.json()
        assert body["promoted_version"] == 1
        assert body["versions"][0]["status"] == "promoted"
        assert "system" not in body["versions"][0]
        assert "human" not in body["versions"][0]

    def test_get_version_returns_full_content(self, registry: PromptRegistry) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.get("/api/v1/prompts/generate_answer/1")

        assert response.status_code == 200
        body = response.json()
        assert body["version"] == 1
        assert "{context}" in body["human"]

    def test_get_missing_version_404(self, registry: PromptRegistry) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.get("/api/v1/prompts/generate_answer/99")

        assert response.status_code == 404

    def test_diff_between_versions(self, registry: PromptRegistry) -> None:
        registry.create_candidate(
            "generate_answer",
            **_candidate_body(),
            actor="tester",
        )
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.get("/api/v1/prompts/generate_answer/diff?a=1&b=2")

        assert response.status_code == 200
        ops = {h["op"] for h in response.json()["hunks"]}
        assert {"add", "del"} <= ops

    def test_unknown_prompt_name_is_422(self, registry: PromptRegistry) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            assert client.get("/api/v1/prompts/not_a_prompt").status_code == 422
            assert client.get("/api/v1/prompts/not_a_prompt/1").status_code == 422

    def test_traversal_name_is_rejected(self, registry: PromptRegistry) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.get("/api/v1/prompts/..%2F..%2Fetc%2Fpasswd/1")

        assert response.status_code in (404, 422)


class TestWriteProtection:
    def test_mutation_requires_jwt_when_auth_configured(
        self, registry: PromptRegistry
    ) -> None:
        with _client(_settings()) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/candidates", json=_candidate_body()
            )

        assert response.status_code == 401

    def test_mutation_rejects_invalid_token(self, registry: PromptRegistry) -> None:
        with _client(_settings()) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/candidates",
                json=_candidate_body(),
                headers={"Authorization": "Bearer not-a-token"},
            )

        assert response.status_code == 401

    def test_mutation_rejected_when_writes_disabled_and_open(
        self, registry: PromptRegistry
    ) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=False)
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/candidates", json=_candidate_body()
            )

        assert response.status_code == 403

    def test_mutation_allowed_when_writes_enabled_and_open(
        self, registry: PromptRegistry
    ) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/candidates", json=_candidate_body()
            )

        assert response.status_code == 201
        assert response.json()["version"] == 2
        assert response.json()["status"] == "candidate"
        entries = registry.audit_entries()
        assert entries[-1]["action"] == "create_candidate"
        assert entries[-1]["actor"] == "local"

    def test_mutation_with_jwt_records_subject_as_actor(
        self, registry: PromptRegistry
    ) -> None:
        with _client(_settings()) as client:
            token = _token(client)
            response = client.post(
                "/api/v1/prompts/generate_answer/candidates",
                json=_candidate_body(),
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 201
        assert registry.audit_entries()[-1]["actor"] == "admin"

    def test_mutations_are_rate_limited(self, registry: PromptRegistry) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings) as client:
            statuses = [
                client.post(
                    "/api/v1/prompts/generate_answer/candidates",
                    json=_candidate_body(),
                ).status_code
                for _ in range(11)
            ]

        assert statuses[:10] == [201] * 10
        assert statuses[10] == 429

    def test_extra_fields_rejected(self, registry: PromptRegistry) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/candidates",
                json=_candidate_body(unexpected="field"),
            )

        assert response.status_code == 422

    def test_invalid_prompt_content_returns_422(
        self, registry: PromptRegistry
    ) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/candidates",
                json=_candidate_body(system="no variables at all"),
            )

        assert response.status_code == 422


class TestAbTestEndpoints:
    def test_ab_test_lifecycle_with_stub_runner(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        runner = lambda name, version: {"scorecard": _GOOD_SCORECARD}  # noqa: E731
        registry.create_candidate("generate_answer", **_candidate_body())
        with _client(settings, tmp_path=tmp_path, runner=runner) as client:
            started = client.post(
                "/api/v1/prompts/generate_answer/ab-test",
                json={"candidate_version": 2},
            )
            assert started.status_code == 202
            job_id = started.json()["job_id"]

            job = _wait_for_job(client, job_id)

        assert job["status"] == "succeeded"
        assert job["result"]["scorecard"] == _GOOD_SCORECARD
        assert Path(job["result_path"]).exists()

    def test_ab_test_rejects_promoted_version(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings, tmp_path=tmp_path, runner=lambda n, v: {}) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/ab-test",
                json={"candidate_version": 1},
            )

        assert response.status_code == 409

    def test_ab_test_unknown_job_404(self, registry: PromptRegistry) -> None:
        with _client(_settings(jwt_secret_key=None)) as client:
            response = client.get("/api/v1/prompts/generate_answer/ab-test/nope")

        assert response.status_code == 404

    def test_promote_resolves_scorecard_from_ab_test_id(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        """The real API flow: run A/B -> promote by job id without resending
        the scorecard."""
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        runner = lambda name, version: {  # noqa: E731
            "candidate_scorecard": _GOOD_SCORECARD,
            "meta": {"dataset_version": "abc123"},
        }
        registry.create_candidate("generate_answer", **_candidate_body())
        with _client(settings, tmp_path=tmp_path, runner=runner) as client:
            job_id = client.post(
                "/api/v1/prompts/generate_answer/ab-test",
                json={"candidate_version": 2},
            ).json()["job_id"]
            assert _wait_for_job(client, job_id)["status"] == "succeeded"

            promoted = client.post(
                "/api/v1/prompts/generate_answer/promote",
                json={"version": 2, "evaluation": {"ab_test_id": job_id}},
            )

        assert promoted.status_code == 200
        evaluation = registry.promoted("generate_answer").evaluation
        assert evaluation is not None
        assert evaluation.ab_test_id == job_id
        assert evaluation.scorecard["faithfulness"] == pytest.approx(0.90)


class TestPromoteRollback:
    def _create_candidate(self, registry: PromptRegistry) -> int:
        return registry.create_candidate(
            "generate_answer", **_candidate_body(), actor="tester"
        ).version

    def test_happy_path_create_promote_rollback_audited(
        self, registry: PromptRegistry
    ) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings) as client:
            created = client.post(
                "/api/v1/prompts/generate_answer/candidates", json=_candidate_body()
            )
            assert created.status_code == 201

            promoted = client.post(
                "/api/v1/prompts/generate_answer/promote",
                json={
                    "version": 2,
                    "evaluation": {
                        "dataset_version": "v1",
                        "scorecard": _GOOD_SCORECARD,
                        "ab_test_id": "job123",
                    },
                },
            )
            assert promoted.status_code == 200
            assert promoted.json()["status"] == "promoted"
            assert registry.promoted("generate_answer").version == 2
            assert (
                registry.promoted("generate_answer").evaluation.ab_test_id == "job123"
            )

            rolled_back = client.post("/api/v1/prompts/generate_answer/rollback")
            assert rolled_back.status_code == 200
            assert registry.promoted("generate_answer").version == 1

        actions = [e["action"] for e in registry.audit_entries()]
        assert actions[-3:] == ["create_candidate", "promote", "rollback"]
        assert registry.verify_chain() is True

    def test_promote_without_evidence_409(self, registry: PromptRegistry) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        self._create_candidate(registry)
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/promote", json={"version": 2}
            )

        assert response.status_code == 409
        assert "evaluation" in str(response.json()["detail"])

    def test_promote_threshold_regression_409(
        self, registry: PromptRegistry
    ) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        self._create_candidate(registry)
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/promote",
                json={
                    "version": 2,
                    "evaluation": {"scorecard": _REGRESSED_SCORECARD},
                },
            )

        assert response.status_code == 409
        failures = response.json()["detail"]["failures"]
        assert any("faithfulness" in f for f in failures)

    def test_promote_baseline_regression_409(
        self, registry: PromptRegistry
    ) -> None:
        """A candidate that meets every threshold but regresses the promoted
        baseline beyond the tolerance is still rejected."""
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        v2 = self._create_candidate(registry)
        registry.promote(
            "generate_answer",
            v2,
            actor="tester",
            evaluation={"scorecard": _GOOD_SCORECARD},
        )
        v3 = self._create_candidate(registry)
        # faithfulness 0.90 -> 0.87 drops 0.03, beyond the 0.02 tolerance.
        card = {**_GOOD_SCORECARD, "faithfulness": 0.87}
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/promote",
                json={"version": v3, "evaluation": {"scorecard": card}},
            )

        assert response.status_code == 409
        failures = response.json()["detail"]["failures"]
        assert any("faithfulness" in f for f in failures)

    def test_promote_non_candidate_409(self, registry: PromptRegistry) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings) as client:
            response = client.post(
                "/api/v1/prompts/generate_answer/promote",
                json={"version": 1, "evaluation": {"scorecard": _GOOD_SCORECARD}},
            )

        assert response.status_code == 409

    def test_rollback_without_superseded_409(self, registry: PromptRegistry) -> None:
        settings = _settings(jwt_secret_key=None, prompt_registry_write_enabled=True)
        with _client(settings) as client:
            response = client.post("/api/v1/prompts/generate_answer/rollback")

        assert response.status_code == 409
