"""Tests for the promotion gate truth table and the A/B engine (offline)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from app.services.promotion_gate import (
    EDD_THRESHOLDS,
    MAX_COST_PER_QUERY_USD,
    evaluate_promotion,
)
from app.services.prompt_registry import PromptRegistry
from evaluation.ab_test import AbTestError, run_ab_test
from evaluation.dataset import build_dataset

_REPO_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _good_card(**overrides: Any) -> dict[str, Any]:
    card: dict[str, Any] = {
        "faithfulness": 0.90,
        "answer_relevancy": 0.90,
        "context_precision": 0.80,
        "context_recall": 0.80,
        "llm_as_judge": 4.5,
        "injection_block_rate": 1.0,
        "ood_refusal_rate": 1.0,
        "cost_per_query_usd": 0.005,
        "latency_p95_ms": 3000,
    }
    card.update(overrides)
    return card


class TestGateTruthTable:
    def test_allows_good_candidate(self) -> None:
        result = evaluate_promotion(_good_card(), _good_card())
        assert result.allowed is True
        assert result.failures == []

    def test_allows_first_promotion_without_baseline(self) -> None:
        assert evaluate_promotion(_good_card(), None).allowed is True

    @pytest.mark.parametrize("metric", list(EDD_THRESHOLDS))
    def test_blocks_each_threshold_miss(self, metric: str) -> None:
        card = _good_card(**{metric: EDD_THRESHOLDS[metric] - 0.05})
        result = evaluate_promotion(card, None)
        assert result.allowed is False
        assert any(metric in f for f in result.failures)

    def test_blocks_missing_metric(self) -> None:
        card = _good_card()
        del card["faithfulness"]
        result = evaluate_promotion(card, None)
        assert result.allowed is False
        assert any("faithfulness" in f for f in result.failures)

    def test_blocks_baseline_regression_beyond_tolerance(self) -> None:
        result = evaluate_promotion(
            _good_card(faithfulness=0.87), _good_card()
        )
        assert result.allowed is False
        assert any("faithfulness" in f for f in result.failures)

    def test_allows_regression_within_tolerance(self) -> None:
        assert evaluate_promotion(
            _good_card(faithfulness=0.89), _good_card()
        ).allowed is True

    def test_blocks_cost_cap(self) -> None:
        result = evaluate_promotion(
            _good_card(cost_per_query_usd=MAX_COST_PER_QUERY_USD + 0.001), None
        )
        assert result.allowed is False
        assert any("cost_per_query_usd" in f for f in result.failures)

    def test_blocks_p95_regression_over_20pct(self) -> None:
        result = evaluate_promotion(
            _good_card(latency_p95_ms=3700), _good_card(latency_p95_ms=3000)
        )
        assert result.allowed is False
        assert any("latency_p95_ms" in f for f in result.failures)

    def test_allows_p95_regression_under_20pct(self) -> None:
        assert evaluate_promotion(
            _good_card(latency_p95_ms=3500), _good_card(latency_p95_ms=3000)
        ).allowed is True

    def test_latency_improvement_is_not_a_regression(self) -> None:
        """Lower-is-better metrics improving vs. baseline must not trip the
        generic regression check (a Phase-7 real run exposed this)."""
        result = evaluate_promotion(
            _good_card(latency_p50_ms=1500, latency_p95_ms=2000),
            _good_card(latency_p50_ms=3000, latency_p95_ms=3000),
        )
        assert result.allowed is True
        assert result.failures == []

    def test_blocks_p50_regression_over_20pct(self) -> None:
        result = evaluate_promotion(
            _good_card(latency_p50_ms=3700), _good_card(latency_p50_ms=3000)
        )
        assert result.allowed is False
        assert any("latency_p50_ms" in f for f in result.failures)

    def test_blocks_ood_refusal_drop(self) -> None:
        result = evaluate_promotion(_good_card(ood_refusal_rate=0.5), None)
        assert result.allowed is False
        assert any("ood_refusal_rate" in f for f in result.failures)

    def test_blocks_injection_block_drop(self) -> None:
        result = evaluate_promotion(_good_card(injection_block_rate=0.5), None)
        assert result.allowed is False
        assert any("injection_block_rate" in f for f in result.failures)

    def test_reports_multiple_failures(self) -> None:
        result = evaluate_promotion(
            _good_card(faithfulness=0.1, injection_block_rate=0.0), None
        )
        assert len(result.failures) >= 2


class _FakeGraph:
    """Arm-dependent fake: the answer differs per arm and the result stamps
    `active_versions()` — the same field the real nodes write — so tests prove
    which prompt version each case actually resolved."""

    def __init__(self, baseline_version: int) -> None:
        self._baseline_version = baseline_version

    def invoke(self, state: dict, config: dict | None = None) -> dict[str, Any]:
        from app.graph.prompts import active_versions

        versions = active_versions()
        question = state["question"]
        if "ignore all previous" in question.lower() or "do anything now" in question.lower():
            return {
                **state,
                "blocked": True,
                "generation": "",
                "documents": [],
                "prompt_versions": versions,
            }
        if "capital of france" in question.lower() or "sourdough" in question.lower():
            return {
                **state,
                "generation": "I don't know.",
                "documents": [],
                "prompt_versions": versions,
            }
        arm = (
            "candidate"
            if versions.get("generate_answer") != self._baseline_version
            else "baseline"
        )
        return {
            **state,
            "blocked": False,
            "generation": f"{arm} answer",
            "documents": [
                Document(page_content="ctx", metadata={"source": "doc"})
            ],
            "prompt_versions": versions,
        }


def _arm_scorers() -> dict[str, Any]:
    """Scorers that reward the candidate arm (score reads the answer text)."""

    def scored(question="", answer="", context="", base=0.8):
        return base + (0.1 if "candidate" in answer else 0.0)

    return {
        "faithfulness": lambda **kw: scored(**kw, base=0.9),
        "answer_relevancy": lambda **kw: scored(**kw, base=0.9),
        "context_precision": lambda **kw: scored(**kw, base=0.8),
        "context_recall": lambda **kw: scored(**kw, base=0.8),
        "llm_as_judge": lambda **kw: scored(**kw, base=4.2),
    }


@pytest.fixture
def registry(tmp_path: Path) -> PromptRegistry:
    shutil.copytree(_REPO_PROMPTS, tmp_path / "prompts")
    instance = PromptRegistry(tmp_path / "prompts")
    from app.graph.prompts import set_registry

    set_registry(instance)
    return instance


def _make_candidate(registry: PromptRegistry) -> int:
    return registry.create_candidate(
        "generate_answer",
        system="Candidate system. {context}",
        human="Q: {question}",
        author="tester",
        description="d",
        changelog="c",
    ).version


def _baseline(registry: PromptRegistry) -> _FakeGraph:
    return _FakeGraph(registry.promoted("generate_answer").version)


class TestAbEngine:
    def test_run_ab_test_scores_both_arms(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        v2 = _make_candidate(registry)
        result = run_ab_test(
            "generate_answer",
            v2,
            registry=registry,
            graph=_baseline(registry),
            scorers=_arm_scorers(),
            results_dir=tmp_path / "ab",
        )

        assert result["baseline_version"] == registry.promoted(
            "generate_answer"
        ).version
        assert result["candidate_version"] == v2
        assert result["candidate_scorecard"]["faithfulness"] > (
            result["baseline_scorecard"]["faithfulness"]
        )
        assert result["deltas"]["faithfulness"] == pytest.approx(0.1, abs=0.01)
        assert result["gate"]["allowed"] is True
        persisted = json.loads(Path(result["result_path"]).read_text())
        assert persisted["ab_test_id"] == result["ab_test_id"]

    def test_candidate_cases_record_pinned_version(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        """Per-case evidence: the candidate arm's cases record v2 as the
        resolved generate_answer version (pins reach the graph)."""
        v2 = _make_candidate(registry)
        result = run_ab_test(
            "generate_answer",
            v2,
            registry=registry,
            graph=_baseline(registry),
            scorers=_arm_scorers(),
            results_dir=tmp_path / "ab",
        )
        cand_cases = result["candidate_detail"]["cases"]
        base_cases = result["baseline_detail"]["cases"]
        assert all(
            c["prompt_versions"]["generate_answer"] == v2 for c in cand_cases
        )
        promoted = registry.promoted("generate_answer").version
        assert all(
            c["prompt_versions"]["generate_answer"] == promoted
            for c in base_cases
        )

    def test_rejects_same_version_both_arms(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        with pytest.raises(AbTestError):
            run_ab_test(
                "generate_answer",
                registry.promoted("generate_answer").version,
                registry=registry,
                graph=_baseline(registry),
                scorers=_arm_scorers(),
                results_dir=tmp_path / "ab",
            )

    def test_unknown_candidate_errors(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        from app.services.prompt_registry import PromptNotFoundError

        with pytest.raises(PromptNotFoundError):
            run_ab_test(
                "generate_answer",
                99,
                registry=registry,
                graph=_baseline(registry),
                scorers=_arm_scorers(),
                results_dir=tmp_path / "ab",
            )

    def test_explicit_baseline_version(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        v2 = _make_candidate(registry)
        v3 = _make_candidate(registry)
        result = run_ab_test(
            "generate_answer",
            v3,
            baseline_version=v2,
            registry=registry,
            graph=_FakeGraph(v2),
            scorers=_arm_scorers(),
            results_dir=tmp_path / "ab",
        )
        assert result["baseline_version"] == v2

    def test_gate_preview_blocks_weak_candidate(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        v2 = _make_candidate(registry)
        weak = {name: (lambda **kw: 0.1) for name in _arm_scorers()}
        result = run_ab_test(
            "generate_answer",
            v2,
            registry=registry,
            graph=_baseline(registry),
            scorers=weak,
            results_dir=tmp_path / "ab",
        )
        assert result["gate"]["allowed"] is False
        assert result["gate"]["failures"]

    def test_cost_cap_stops_run(
        self,
        registry: PromptRegistry,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.core.cost_tracking import TokenUsageCallbackHandler
        from evaluation import ab_test as ab_module
        from evaluation.harness import CostCapExceeded

        v2 = _make_candidate(registry)

        def _spent_handler() -> TokenUsageCallbackHandler:
            handler = TokenUsageCallbackHandler()
            handler.prompt_tokens = 10_000_000  # already over any sane cap
            return handler

        monkeypatch.setattr(
            ab_module, "TokenUsageCallbackHandler", _spent_handler
        )
        with pytest.raises(CostCapExceeded):
            run_ab_test(
                "generate_answer",
                v2,
                registry=registry,
                max_cost_usd=0.50,
                graph=_baseline(registry),
                scorers=_arm_scorers(),
                results_dir=tmp_path / "ab",
            )

    def test_dataset_has_cases_per_arm(
        self, registry: PromptRegistry, tmp_path: Path
    ) -> None:
        v2 = _make_candidate(registry)
        result = run_ab_test(
            "generate_answer",
            v2,
            registry=registry,
            graph=_baseline(registry),
            scorers=_arm_scorers(),
            results_dir=tmp_path / "ab",
        )
        n = len(build_dataset())
        assert result["candidate_detail"]["cases_count"] == n
        assert result["baseline_detail"]["cases_count"] == n
