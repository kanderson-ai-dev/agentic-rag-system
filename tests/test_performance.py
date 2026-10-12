"""Tests for the performance dashboard endpoint and its telemetry math.

Covers the percentile helper, the session-scoped ``UsageStore.performance``
aggregation, per-prompt-version rollups, the Human-vs-Agent comparison and
the ``GET /api/v1/dashboard/performance`` endpoint — all offline.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import dashboard as dashboard_routes
from app.core.config import Settings
from app.main import app
from app.services.usage_store import UsageStore, percentile


def _record(store: UsageStore, **overrides) -> None:
    base = {
        "thread_id": "t",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cost_usd": 0.001,
        "latency_ms": 1000,
        "blocked": False,
        "escalated": False,
        "retry_count": 0,
    }
    base.update(overrides)
    store.record(**base)


class TestPercentile:
    def test_empty_returns_zero(self) -> None:
        assert percentile([], 50) == 0.0
        assert percentile([], 95) == 0.0

    def test_single_item(self) -> None:
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 95) == 42.0

    def test_nearest_rank(self) -> None:
        assert percentile([10, 20, 30, 40], 50) == 20
        assert percentile(list(range(1, 21)), 95) == 19

    def test_ties_and_unsorted_input(self) -> None:
        assert percentile([30, 10, 30, 10], 50) == 10
        assert percentile([30, 10, 30, 10], 95) == 30


class TestPerformanceAggregation:
    def test_aggregates_end_to_end_and_per_node(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        for i, latency in enumerate([100, 200, 300, 400]):
            _record(
                store,
                thread_id=f"t{i}",
                latency_ms=latency,
                node_latencies={"retrieve": 50 + i * 10, "generate": 200},
            )

        perf = store.performance()

        assert perf["requests"] == 4
        assert perf["latency"]["p50_ms"] == 200
        assert perf["latency"]["p95_ms"] == 400
        assert perf["latency"]["target_p50_ms"] == 3000
        assert perf["cost"]["avg_per_query_usd"] == pytest.approx(0.001)
        assert perf["cost"]["total_usd"] == pytest.approx(0.004)
        assert perf["tokens"]["avg_per_query"] == pytest.approx(150)

        nodes = {n["node"]: n for n in perf["per_node"]}
        assert nodes["retrieve"]["samples"] == 4
        assert nodes["retrieve"]["p50_ms"] == 60
        assert nodes["retrieve"]["p95_ms"] == 80
        assert nodes["generate"]["p95_ms"] == 200

    def test_rates_and_daily_trend(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        _record(store, thread_id="a", blocked=True)
        _record(store, thread_id="b", escalated=True)

        perf = store.performance()
        assert perf["rates"]["blocked"] == pytest.approx(0.5)
        assert perf["rates"]["escalated"] == pytest.approx(0.5)
        assert len(perf["daily_cost_trend"]) == 1
        assert perf["daily_cost_trend"][0]["requests"] == 2

    def test_empty_store_returns_zeros(self, tmp_path) -> None:
        perf = UsageStore(str(tmp_path / "usage.sqlite")).performance()
        assert perf["requests"] == 0
        assert perf["latency"]["p50_ms"] == 0.0
        assert perf["cost"]["avg_per_query_usd"] == 0.0
        assert perf["per_node"] == []
        assert perf["daily_cost_trend"] == []

    def test_scopes_to_session(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        _record(store, thread_id="a", session_id="s1", cost_usd=0.01)
        _record(store, thread_id="b", session_id="s2", cost_usd=0.02)

        s1 = store.performance(session_id="s1")
        assert s1["requests"] == 1
        assert s1["cost"]["total_usd"] == pytest.approx(0.01)
        # A session that never made a request gets the empty payload.
        assert store.performance(session_id="s3")["requests"] == 0

    def test_prompt_version_aggregation_scopes_to_session(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        _record(
            store,
            thread_id="a",
            session_id="s1",
            prompt_versions={"generate_answer": 1},
        )
        _record(
            store,
            thread_id="b",
            session_id="s2",
            prompt_versions={"generate_answer": 2},
        )

        by_version = store.summary_by_prompt_version()
        assert {v["generate_answer_version"] for v in by_version} == {1, 2}
        scoped = store.summary_by_prompt_version(session_id="s1")
        assert len(scoped) == 1
        assert scoped[0]["generate_answer_version"] == 1


class TestHumanVsAgent:
    def test_math_and_assumptions(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        _record(store, thread_id="a", cost_usd=0.001, latency_ms=2000)
        _record(store, thread_id="b", cost_usd=0.001, latency_ms=3000)

        perf = store.performance()
        settings = Settings(human_minutes_per_query=6.0, human_hourly_usd=30.0)
        hva = dashboard_routes._human_vs_agent(perf, settings)

        assert hva["label"] == "estimate"
        assert hva["assumptions"]["human_minutes_per_query"] == 6.0
        assert hva["assumptions"]["human_hourly_usd"] == 30.0
        # 6 min at $30/h -> $3.00 per query for the human estimate.
        assert hva["per_query"]["human_cost_usd"] == pytest.approx(3.0)
        assert hva["per_query"]["human_seconds"] == pytest.approx(360.0)
        assert hva["per_query"]["agent_cost_usd"] == pytest.approx(0.001)
        assert hva["per_query"]["agent_seconds"] == pytest.approx(2.0)
        assert hva["totals"]["human_cost_usd"] == pytest.approx(6.0)
        assert hva["totals"]["agent_cost_usd"] == pytest.approx(0.002)
        assert hva["cost_savings_pct"] == pytest.approx(
            (1 - 0.002 / 6.0) * 100, abs=0.01
        )

    def test_empty_traffic_yields_zero_savings(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        hva = dashboard_routes._human_vs_agent(store.performance(), Settings())
        assert hva["totals"]["requests"] == 0
        assert hva["cost_savings_pct"] == 0.0

    def test_settings_defaults(self) -> None:
        assert (
            Settings.model_fields["human_minutes_per_query"].default == 6.0
        )
        assert Settings.model_fields["human_hourly_usd"].default == 30.0


class TestPerformanceEndpoint:
    def test_returns_session_scoped_payload(self, tmp_path) -> None:
        store = UsageStore(str(tmp_path / "usage.sqlite"))
        _record(
            store,
            thread_id="a",
            session_id="mine",
            cost_usd=0.004,
            latency_ms=1500,
            node_latencies={"retrieve": 300.0},
            prompt_versions={"generate_answer": 2},
        )
        _record(store, thread_id="b", session_id="other", cost_usd=0.02)
        app.state.usage_store = store
        try:
            client = TestClient(app)
            response = client.get(
                "/api/v1/dashboard/performance",
                headers={"X-Session-Id": "mine"},
            )
            client.close()
        finally:
            del app.state.usage_store

        assert response.status_code == 200
        body = response.json()
        assert body["requests"] == 1
        assert body["latency"]["p50_ms"] == 1500
        assert body["per_node"][0]["node"] == "retrieve"
        assert body["per_prompt_version"][0]["generate_answer_version"] == 2
        assert body["human_vs_agent"]["label"] == "estimate"
        assert "budget" in body

    def test_empty_session_returns_zeroed_payload(self, tmp_path) -> None:
        app.state.usage_store = UsageStore(str(tmp_path / "usage.sqlite"))
        try:
            client = TestClient(app)
            response = client.get(
                "/api/v1/dashboard/performance",
                headers={"X-Session-Id": "never-seen"},
            )
            client.close()
        finally:
            del app.state.usage_store

        assert response.status_code == 200
        body = response.json()
        assert body["requests"] == 0
        assert body["human_vs_agent"]["totals"]["requests"] == 0
        assert body["per_node"] == []


class TestPerformanceFrontend:
    def test_console_serves_performance_section(self) -> None:
        client = TestClient(app)
        try:
            html = client.get("/console").text
            js = client.get("/console/js/dashboard.js").text
        finally:
            client.close()
        for element_id in (
            'id="perf-cards"',
            'id="perf-nodes"',
            'id="human-vs-agent"',
            'id="perf-versions"',
            'id="perf-empty"',
        ):
            assert element_id in html
        assert "/api/v1/dashboard/performance" in js
        assert "renderPerformance" in js
