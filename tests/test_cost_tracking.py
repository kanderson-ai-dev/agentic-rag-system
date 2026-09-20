"""Tests for cost tracking, usage storage, node timing and dashboard endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.core.cost_tracking import TokenUsageCallbackHandler, calculate_cost_usd
from app.core.metrics import timed_node
from app.main import app
from app.services.usage_store import UsageStore


class TestCalculateCost:
    def test_full_million_tokens(self) -> None:
        assert calculate_cost_usd(1_000_000, 1_000_000) == pytest.approx(0.75)

    def test_input_only(self) -> None:
        assert calculate_cost_usd(1_000, 0) == pytest.approx(0.00015)

    def test_custom_prices(self) -> None:
        assert calculate_cost_usd(1_000_000, 0, input_price=1.0) == pytest.approx(1.0)


class _FakeResponse:
    llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}


class TestTokenUsageCallbackHandler:
    def test_captures_tokens(self) -> None:
        handler = TokenUsageCallbackHandler()
        handler.on_llm_end(_FakeResponse())
        assert handler.prompt_tokens == 100
        assert handler.completion_tokens == 50
        assert handler.total_tokens == 150


class TestUsageStore:
    def test_record_summary_and_recent(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        store.record(
            thread_id="t1",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            latency_ms=250,
            blocked=False,
            escalated=False,
            retry_count=1,
        )
        store.record(
            thread_id="t2",
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_ms=10,
            blocked=True,
            escalated=True,
            retry_count=0,
        )

        summary = store.summary()

        assert summary["total_requests"] == 2
        assert summary["total_prompt_tokens"] == 100
        assert summary["blocked_requests"] == 1
        assert summary["escalated_requests"] == 1
        assert summary["total_cost_usd"] == pytest.approx(0.001)

        recent = store.recent(limit=10)
        assert len(recent) == 2
        assert recent[0]["thread_id"] == "t2"  # most recent first


class TestTimedNode:
    def test_wraps_and_returns_result(self) -> None:
        @timed_node("test_node")
        def node(state):
            return {"result": "ok"}

        assert node({"x": 1}) == {"result": "ok"}


class TestDashboardEndpoints:
    def test_summary_and_recent(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        store.record(
            thread_id="t1",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.001,
            latency_ms=100,
            blocked=False,
            escalated=False,
            retry_count=0,
        )
        app.state.usage_store = store
        try:
            client = TestClient(app)
            summary = client.get("/api/v1/dashboard/summary")
            recent = client.get("/api/v1/dashboard/recent")
            client.close()
        finally:
            del app.state.usage_store

        assert summary.status_code == 200
        assert summary.json()["total_requests"] == 1
        assert summary.json()["total_cost_usd"] == pytest.approx(0.001)
        assert recent.status_code == 200
        assert len(recent.json()) == 1
